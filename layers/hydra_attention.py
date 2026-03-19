"""
Hydra Attention v3.2e — Mean pairwise correlation proxy as gate signal.

Changes from v3.2d:
  - data_net input changed from chan_var [B, d_model] to corr_proxy [B, 1].
  - data_net first layer: Linear(1, d_model//4) instead of Linear(d_model, d_model//4).

  Problem with chan_var (v3.2c/d):
    Channel variance is ambiguous as a gating signal. High variance can mean:
      (a) independent channels (Exchange — correctly suppress mixing)
      (b) correlated-but-heteroscedastic channels (Solar — incorrectly suppressed)
    This caused Solar regression in v3.2c and partial recovery in v3.2d.

  New signal — mean pairwise correlation proxy:
    Measures "how much do channels point in the same direction" using O(B*C*D)
    computation — no expensive C×C matrix (which would be O(B*C²*D), ~292B
    flops/batch for Traffic C=862).

    Algorithm:
      1. Center each channel's D-dim vector: x_centered = x - mean(x, dim=D)
      2. L2-normalize per channel: x_norm = x_centered / ||x_centered||
      3. Centroid direction: mean_channel = mean(x_norm, dim=C)  [B, D]
      4. Each channel's cosine similarity to centroid: dot product [B, C]
      5. corr_proxy = mean over C → [B, 1] scalar

    Semantics:
      - Perfectly correlated channels → all align with centroid → proxy → 1
      - Independent channels → random directions cancel → proxy → ~0
      - Heteroscedastic-but-correlated (Solar) → still aligns with centroid → proxy > 0
      - Genuinely independent (Exchange) → centroid near zero → proxy ~0

    This correctly distinguishes Solar (correlated, should mix) from Exchange
    (independent, should not mix), whereas chan_var could not.

  Computational cost per batch:
    O(B*P * C * D) — e.g. Traffic: 768 * 862 * 512 ≈ 339M float ops (fast).
    Full C×C correlation would be 768 * 862² * 512 ≈ 292B ops (prohibitive).

  data_net is now smaller (input dim 1 vs d_model), reducing parameter count
  slightly.

Gate types:
  'hybrid':   log(C) primary + ε·corr_proxy correction (RECOMMENDED)
  'channel':  log(C) only (ablation baseline)
  'adaptive': log(C) + variance + mean_abs fully mixed (v3.1 design)
  'scalar':   Single learned bias (simplest baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HydraAttention(nn.Module):
    """Core Hydra: O(Nd) linear attention via cosine similarity."""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q = F.normalize(self.W_q(x), p=2, dim=-1)
        K = F.normalize(self.W_k(x), p=2, dim=-1)
        V = self.W_v(x)
        global_feat = (K * V).sum(dim=1, keepdim=True)
        return self.dropout(Q * global_feat)


class AdaptiveGate(nn.Module):
    """
    Hybrid gate: channel-primary with learned correlation-based bonus.

    gate = sigmoid( f(logC) + ε_eff · g(corr_proxy) )

    f(logC):      MLP: normalized log(C) → per-feature logits
    g(corr_proxy): MLP: mean-pairwise-correlation scalar → per-feature correction
    ε_raw:        Unconstrained learnable scalar, init=0
    ε_eff:        softplus(ε_raw) — always >= 0, variance only opens gate

    corr_proxy is the mean cosine similarity between each channel and the
    channel centroid, computed in O(B*C*D). See module docstring for details.
    """
    def __init__(self, d_model, n_channels=1, gate_type='hybrid', init_bias=-5.0, gate_temp=1.0):
        super().__init__()
        self.gate_type = gate_type
        self.d_model = d_model
        self.gate_temp = gate_temp

        self.register_buffer(
            'log_c', torch.tensor(math.log(max(n_channels, 2)) / math.log(1000))
        )

        if gate_type == 'scalar':
            self.bias = nn.Parameter(torch.tensor(init_bias))

        elif gate_type == 'channel':
            self.channel_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.channel_net[2].weight)
            nn.init.constant_(self.channel_net[2].bias, init_bias)

        elif gate_type == 'hybrid':
            # Primary: f(logC) → per-feature logits
            self.channel_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.channel_net[2].weight)
            nn.init.constant_(self.channel_net[2].bias, init_bias)

            # Data bonus: g(corr_proxy) → per-feature correction.
            # Input dim is 1 (scalar correlation proxy), not d_model.
            # Weight zero-init: "pure channel at start" guarantee.
            # Bias default init: gives ε real gradient from epoch 1.
            self.data_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.data_net[2].weight)
            # data_net[2].bias left at default Kaiming uniform init

            # ε_eff = softplus(ε_raw) >= 0: correlation only opens gate.
            self._epsilon_raw = nn.Parameter(torch.tensor(0.0))

        elif gate_type == 'adaptive':
            gate_input_dim = 2 * d_model + 1
            hidden = max(d_model // 4, 8)
            self.gate_net = nn.Sequential(
                nn.Linear(gate_input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
            nn.init.zeros_(self.gate_net[2].weight)
            nn.init.constant_(self.gate_net[2].bias, init_bias)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

    def _epsilon_eff(self):
        return F.softplus(self._epsilon_raw)

    @staticmethod
    def _corr_proxy(x):
        """
        Mean pairwise correlation proxy via centroid cosine similarity.

        Args:
            x: [B, C, D]
        Returns:
            corr_proxy: [B, 1] scalar in approximately [0, 1]

        O(B * C * D) — safe for large C (Traffic C=862, Electricity C=321).
        """
        # Center each channel's D-dim representation
        x_centered = x - x.mean(dim=-1, keepdim=True)          # [B, C, D]
        # L2-normalize per channel (handle near-zero channels safely)
        x_norm = F.normalize(x_centered, dim=-1, eps=1e-6)      # [B, C, D]
        # Centroid direction in unit-sphere space
        mean_channel = x_norm.mean(dim=1)                        # [B, D]
        # Each channel's cosine similarity to the centroid
        cosine_to_mean = (x_norm * mean_channel.unsqueeze(1)).sum(dim=-1)  # [B, C]
        # Mean over channels → scalar proxy
        return cosine_to_mean.mean(dim=1, keepdim=True)          # [B, 1]

    def forward(self, x_input, mixed):
        """
        Args:
            x_input: [B, C, D] original input
            mixed:   [B, C, D] Hydra output
        Returns:
            gate: values in (0, 1), broadcastable to [B, C, D]
        """
        if self.gate_type == 'scalar':
            return torch.sigmoid(self.gate_temp * self.bias)

        elif self.gate_type == 'channel':
            log_c_input = self.log_c.view(1, 1)
            logits = self.channel_net(log_c_input)
            return torch.sigmoid(self.gate_temp * logits).unsqueeze(1)

        elif self.gate_type == 'hybrid':
            # Primary: channel-count signal
            log_c_input = self.log_c.view(1, 1)
            channel_logits = self.channel_net(log_c_input)          # [1, D]

            # Data bonus: mean pairwise correlation proxy
            corr = self._corr_proxy(x_input)                        # [B, 1]
            data_correction = self.data_net(corr)                    # [B, D]

            # ε_eff >= 0: correlation only opens the gate, never closes it
            logits = channel_logits + self._epsilon_eff() * data_correction
            return torch.sigmoid(self.gate_temp * logits).unsqueeze(1)

        elif self.gate_type == 'adaptive':
            B, C, D = x_input.shape
            chan_var = x_input.var(dim=1)
            chan_mean_abs = x_input.abs().mean(dim=1)
            log_c_expanded = self.log_c.expand(B, 1)
            gate_input = torch.cat([chan_var, chan_mean_abs, log_c_expanded], dim=1)
            gate_logits = self.gate_net(gate_input)
            return torch.sigmoid(self.gate_temp * gate_logits).unsqueeze(1)

    def get_gate_info(self):
        with torch.no_grad():
            log_c_val = self.log_c.item()
            t = self.gate_temp
            t_str = f", τ={t:.1f}" if t != 1.0 else ""
            if self.gate_type == 'scalar':
                return f"scalar: {torch.sigmoid(t * self.bias).item():.4f}{t_str}"
            elif self.gate_type == 'channel':
                log_c_input = self.log_c.view(1, 1)
                vals = torch.sigmoid(t * self.channel_net(log_c_input))
                return f"channel (logC={log_c_val:.3f}): mean_gate={vals.mean().item():.4f}{t_str}"
            elif self.gate_type == 'hybrid':
                log_c_input = self.log_c.view(1, 1)
                base = torch.sigmoid(t * self.channel_net(log_c_input))
                eps_raw = self._epsilon_raw.item()
                eps_eff = self._epsilon_eff().item()
                return (f"hybrid (logC={log_c_val:.3f}): "
                        f"base_gate={base.mean().item():.4f}, "
                        f"ε_raw={eps_raw:.4f}, ε_eff={eps_eff:.4f}{t_str}")
            else:
                return f"adaptive (logC={log_c_val:.3f}): input-dependent{t_str}"


class HydraChannelMixer(nn.Module):
    """
    Cross-variable mixing via Hydra with hybrid gating.
    """
    def __init__(self, d_model, variant='hydra_gated', rank=32,
                 n_channels=7, dropout=0.0,
                 gate_type='hybrid', gate_init=-5.0, gate_temp=1.0):
        super().__init__()
        self.variant = variant

        if variant == 'hydra':
            self.attn = HydraAttention(d_model, dropout=dropout)
        elif variant == 'hydra_bottleneck':
            self.proj_down = nn.Linear(d_model, rank)
            self.attn = HydraAttention(rank, dropout=dropout)
            self.proj_up = nn.Linear(rank, d_model)
        elif variant == 'hydra_gated':
            self.proj_down = nn.Linear(d_model, rank)
            self.attn = HydraAttention(rank, dropout=dropout)
            self.content_gate = nn.Sequential(
                nn.Linear(rank, rank),
                nn.Sigmoid()
            )
            self.proj_up = nn.Linear(rank, d_model)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.norm = nn.LayerNorm(d_model)

        self.gate = AdaptiveGate(
            d_model, n_channels=n_channels,
            gate_type=gate_type, init_bias=gate_init,
            gate_temp=gate_temp
        )

    def _mix(self, h_norm):
        if self.variant == 'hydra':
            return self.attn(h_norm)
        elif self.variant == 'hydra_bottleneck':
            h_low = self.proj_down(h_norm)
            h_mixed = self.attn(h_low)
            return self.proj_up(h_mixed)
        elif self.variant == 'hydra_gated':
            h_low = self.proj_down(h_norm)
            h_attn = self.attn(h_low)
            gate = self.content_gate(h_low)
            return self.proj_up(h_attn * gate)

    def forward(self, x, B, C):
        """
        Args:
            x: [B*C, P, D]
            B: Batch size
            C: Number of channels
        Returns:
            [B*C, P, D]
        """
        BC, P, D = x.shape

        h = x.reshape(B, C, P, D).permute(0, 2, 1, 3).reshape(B * P, C, D)

        h_norm = self.norm(h)
        mixed = self._mix(h_norm)
        gate_val = self.gate(h, mixed)
        out = h + gate_val * mixed

        with torch.no_grad():
            self._last_gate_mean = gate_val.mean().item()
            self._last_gate_min = gate_val.min().item()
            self._last_gate_max = gate_val.max().item()
            self._last_mixed_norm = mixed.norm(dim=-1).mean().item()
            self._last_input_norm = h.norm(dim=-1).mean().item()
            self._last_contribution = (gate_val * mixed).norm(dim=-1).mean().item()
            # Log corr_proxy mean for monitoring
            if self.gate.gate_type == 'hybrid':
                with torch.no_grad():
                    self._last_corr_proxy = self.gate._corr_proxy(h).mean().item()

        out = out.reshape(B, P, C, D).permute(0, 2, 1, 3).reshape(BC, P, D)
        return out

    def get_gate_info(self):
        return self.gate.get_gate_info()

    def get_gate_stats(self):
        if not hasattr(self, '_last_gate_mean'):
            return "no forward pass yet"
        eps_str = ""
        corr_str = ""
        if hasattr(self.gate, '_epsilon_raw'):
            eps_eff = self.gate._epsilon_eff().item()
            eps_raw = self.gate._epsilon_raw.item()
            eps_str = f", ε_raw={eps_raw:.6f}, ε_eff={eps_eff:.6f}"
        if hasattr(self, '_last_corr_proxy'):
            corr_str = f", corr_proxy={self._last_corr_proxy:.4f}"
        return (f"gate=[{self._last_gate_min:.6f}, {self._last_gate_mean:.6f}, {self._last_gate_max:.6f}]"
                f"{eps_str}"
                f"{corr_str}"
                f", |mixed|={self._last_mixed_norm:.4f}"
                f", |input|={self._last_input_norm:.4f}"
                f", |gate*mixed|={self._last_contribution:.4f}")
