"""
Hydra Attention v3.2g — Mean pooling in HydraAttention (normalize by C).

Changes from v3.2d:
  - Single line change in HydraAttention.forward():
      sum(dim=1)  →  sum(dim=1) / C   (i.e., mean pooling)

  Why this is safe now when v3.2 mean_pool failed:
    v3.2 mean_pool regressed Traffic badly because the gate opened to 0.922
    (too fast, too wide). That experiment had ε=0 throughout — the ε gradient
    was killed by the zero-bias init bug, so the gate had no data signal at all
    and the channel_net's logC signal alone couldn't compensate.

    v3.2d has all the machinery that was missing then:
      - ε gradient flow fixed (Kaiming uniform bias init on data_net final layer)
      - softplus constraint: ε_eff = softplus(ε_raw) >= 0
      - gate_init=-5.0: channel_net starts near-closed for all C
      - variance-based data_net provides per-batch correction signal

    The norm explosion diagnosis:
      global_feat = (K*V).sum(dim=1) accumulates C terms.
      At C=862 (Traffic), |mixed| reaches 150-235 vs |input|~5.
      At C=7 (ETTm1), |mixed| is 1-60 — manageable.
      Dividing by C makes the expected magnitude O(1) regardless of dataset,
      removing the structural instability that v3.2d's gate fights against.

  Expected changes in probe stats:
    Traffic C=862:  |mixed| drops from ~150-230 to ~0.17-0.27
    Electricity C=321: |mixed| drops from ~10-57 to ~0.03-0.18
    Solar C=137:    |mixed| drops from ~115-200 to ~0.84-1.46
    ETTm1 C=7:     |mixed| drops from ~1-90 to ~0.14-13
    ETTh1 C=7:     similar to ETTm1
    Exchange C=8:  |mixed| already small (~1-8), drops to ~0.12-1

    The gate*mixed contribution will drop proportionally, but the gate itself
    should open further since the gating mechanism now needs less suppression
    to keep contributions bounded. The net effect on the residual stream
    depends on whether the gate opens enough to compensate — which is exactly
    what the probe will tell us.

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
    """Core Hydra: O(Nd) linear attention via cosine similarity.

    v3.2g: mean pooling — global_feat divided by C so magnitude is O(1)
    regardless of channel count. Eliminates the structural norm explosion
    that scaled linearly with C under sum pooling.
    """
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
        # Mean pooling: divide by C to make magnitude O(1) across all datasets.
        # Key difference from v3.2d sum pooling.
        C = x.shape[1]
        global_feat = (K * V).sum(dim=1, keepdim=True) / C
        return self.dropout(Q * global_feat)


class AdaptiveGate(nn.Module):
    """
    Hybrid gate: channel-primary with learned data bonus.
    Identical to v3.2d.

    gate = sigmoid( f(logC) + ε_eff · g(variance) )

    where ε_eff = softplus(ε_raw) >= 0 always.
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
            self.channel_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.channel_net[2].weight)
            nn.init.constant_(self.channel_net[2].bias, init_bias)

            self.data_net = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.data_net[2].weight)
            # data_net[2].bias left at default Kaiming uniform init

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

    def forward(self, x_input, mixed):
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
    Identical to v3.2d except HydraAttention now uses mean pooling.
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
