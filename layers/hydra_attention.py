"""
Hydra Attention v3.2d — Epsilon constrained non-negative via softplus.

Changes from v3.2c:
  - ε is now constrained to be >= 0 via softplus reparameterization.
    In v3.2c, ε went strongly negative on Solar (−0.407) and ETTm1 (−0.328),
    suppressing mixing on Solar which actually benefits from cross-channel
    mixing (v3.2 showed −4.1% vs GLPatch). The variance signal is ambiguous:
    high inter-channel variance can mean independent channels (Exchange —
    correctly suppressed) OR correlated-but-heteroscedastic channels (Solar —
    incorrectly suppressed).

    Fix: replace learnable scalar ε with softplus-reparameterized version:
        ε_raw:  unconstrained learnable scalar, init=0
        ε_eff:  softplus(ε_raw) — always >= 0
        gate logit = channel_logits + ε_eff * data_correction

    This means variance can only OPEN the gate further, never suppress it.
    Suppression of low-C datasets is already handled by channel_net's
    logC-based signal (init_bias=-5.0 keeps gate near-closed for small C).

    softplus(0) ≈ 0.693, so ε_eff starts slightly above zero — still small
    relative to the −5.0 channel_net bias, so early behaviour is still
    dominated by the channel gate.

  - get_gate_stats and get_gate_info now report ε_eff = softplus(ε_raw)
    for interpretability, alongside ε_raw in get_gate_info.

Gate types:
  'hybrid':   log(C) primary + ε·variance correction (RECOMMENDED)
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
        # Sum pooling: preserves high-C gate dynamics.
        global_feat = (K * V).sum(dim=1, keepdim=True)
        return self.dropout(Q * global_feat)


class AdaptiveGate(nn.Module):
    """
    Hybrid gate: channel-primary with learned data bonus.

    gate = sigmoid( f(logC) + ε_eff · g(variance) )

    where ε_eff = softplus(ε_raw) >= 0 always.

    f(logC):      MLP that maps normalized log(C) → per-feature logits
    g(variance):  MLP that maps channel variance → per-feature correction
    ε_raw:        Unconstrained learnable scalar, init=0
    ε_eff:        softplus(ε_raw) — non-negative effective epsilon
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

            # Data bonus: g(variance) → per-feature correction.
            # Weight zero-init preserves "pure channel at start" guarantee.
            # Bias default init gives ε_eff real gradient signal from epoch 1.
            self.data_net = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.data_net[2].weight)
            # data_net[2].bias left at default Kaiming uniform init

            # Unconstrained raw parameter; effective epsilon = softplus(ε_raw) >= 0.
            # Variance can only open the gate further, never suppress it.
            # Suppression of low-C is handled by channel_net's logC signal.
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
        """Effective epsilon: softplus(ε_raw), always >= 0."""
        return F.softplus(self._epsilon_raw)

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
            log_c_input = self.log_c.view(1, 1)
            channel_logits = self.channel_net(log_c_input)     # [1, D]

            chan_var = x_input.var(dim=1)                       # [B, D]
            data_correction = self.data_net(chan_var)           # [B, D]

            # ε_eff >= 0: variance only opens the gate, never closes it
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

        out = out.reshape(B, P, C, D).permute(0, 2, 1, 3).reshape(BC, P, D)
        return out

    def get_gate_info(self):
        return self.gate.get_gate_info()

    def get_gate_stats(self):
        if not hasattr(self, '_last_gate_mean'):
            return "no forward pass yet"
        eps_str = ""
        if hasattr(self.gate, '_epsilon_raw'):
            eps_eff = self.gate._epsilon_eff().item()
            eps_raw = self.gate._epsilon_raw.item()
            eps_str = f", ε_raw={eps_raw:.6f}, ε_eff={eps_eff:.6f}"
        return (f"gate=[{self._last_gate_min:.6f}, {self._last_gate_mean:.6f}, {self._last_gate_max:.6f}]"
                f"{eps_str}"
                f", |mixed|={self._last_mixed_norm:.4f}"
                f", |input|={self._last_input_norm:.4f}"
                f", |gate*mixed|={self._last_contribution:.4f}")
