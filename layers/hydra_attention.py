"""
Hydra Attention v3.2h — Sigmoid-bounded epsilon on top of v3.2g mean pooling.

Changes from v3.2g:
  - ε reparameterization changed from softplus to sigmoid-bounded:
      v3.2d/g:  ε_eff = softplus(ε_raw)          range: [0, +∞)
      v3.2h:    ε_eff = ε_max · sigmoid(ε_raw)   range: [0, ε_max]

  Why softplus failed on Traffic (v3.2g probe diagnosis):
    ε_raw grew monotonically to 2.77 by epoch 100, still climbing.
    ε_eff ≈ 2.78 — data_net correction amplified ~4× vs channel_net logit.
    Gate became bimodal (min≈0, max≈1) with mean oscillating ~0.35–0.39.
    Traffic has C=862 and high inter-channel variance (road sensor
    heterogeneity), so data_net sees a large signal every batch and
    softplus gives ε no incentive to stop growing.

    On Electricity (C=321, more homogeneous): ε_raw stabilized at ~1.37
    → ε_eff ≈ 1.37, gate_mean settled gracefully at 0.45–0.53. MSE best ever.

  Why sigmoid-bounded fixes this:
    sigmoid saturates at both ends. For ε_max=2.0:
      - Traffic at ε_raw=2.77: ε_eff = 2.0 * sigmoid(2.77) ≈ 2.0 * 0.94 = 1.88
        (vs 2.78 with softplus — reduction of ~0.9)
      - Electricity at ε_raw=1.37: ε_eff = 2.0 * sigmoid(1.37) ≈ 2.0 * 0.80 = 1.60
        (slightly above the 1.37 that was already working — negligible change)

    The key property: both datasets are now in the sigmoid's saturation zone,
    so the gap between them compresses from ~2× to ~1.17×. Traffic's ε can
    no longer escape to ∞ by gradient descent — it hits diminishing returns.

  Why ε_max=2.0:
    - Electricity worked well at ε_eff≈1.37 (softplus). 2.0 gives headroom.
    - Solar peaked at ε_eff≈0.70 under softplus (v3.2d) — well within range.
    - Traffic's effective cap at ~1.88 is a 32% reduction vs what it reached.
    - ETTh1/ETTm1 had ε_raw<0.2 in v3.2d — sigmoid(0.2)≈0.55, so
      ε_eff≈1.1 vs 0.77 before. Slightly more open but gate_init=-5.0 still
      dominates early training.

  Init behaviour:
    ε_raw=0 → ε_eff = 2.0 * sigmoid(0) = 2.0 * 0.5 = 1.0
    But data_net[2].weight is zero-initialized, so data_correction=0 at epoch 0.
    The gate therefore starts at sigmoid(channel_logits + 1.0 * 0) = sigmoid(-5.0)
    — identical to all prior versions. The 1.0 initial ε_eff has no effect until
    data_net starts producing nonzero corrections.

  Both changes from v3.2 baseline are retained:
    1. Mean pooling in HydraAttention (from v3.2g) — fixes O(C) norm explosion
    2. Sigmoid-bounded ε (new in v3.2h) — caps runaway ε on high-C datasets

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

# Maximum value of ε_eff. Chosen so that:
#   - ε can open the gate strongly on high-C correlated datasets (Electricity, Traffic)
#   - ε cannot grow without bound (Traffic runaway prevention)
EPSILON_MAX = 2.0


class HydraAttention(nn.Module):
    """Core Hydra: O(Nd) linear attention via cosine similarity.

    Mean pooling (from v3.2g): global_feat divided by C so magnitude is O(1).
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
        C = x.shape[1]
        global_feat = (K * V).sum(dim=1, keepdim=True) / C   # mean pooling
        return self.dropout(Q * global_feat)


class AdaptiveGate(nn.Module):
    """
    Hybrid gate: channel-primary with sigmoid-bounded data bonus.

    gate = sigmoid( f(logC) + ε_eff · g(variance) )

    ε_eff = ε_max · sigmoid(ε_raw)   in range [0, ε_max]

    Both ends are soft-bounded:
      - ε_eff >= 0 always (variance only opens gate, never closes)
      - ε_eff <= ε_max (prevents runaway amplification on high-C datasets)
    """
    def __init__(self, d_model, n_channels=1, gate_type='hybrid',
                 init_bias=-5.0, gate_temp=1.0, epsilon_max=EPSILON_MAX):
        super().__init__()
        self.gate_type = gate_type
        self.d_model = d_model
        self.gate_temp = gate_temp
        self.epsilon_max = epsilon_max

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

            # Unconstrained raw parameter.
            # ε_eff = epsilon_max * sigmoid(ε_raw) ∈ [0, epsilon_max]
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
        """Effective epsilon: epsilon_max * sigmoid(ε_raw), in [0, epsilon_max]."""
        return self.epsilon_max * torch.sigmoid(self._epsilon_raw)

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

            # ε_eff ∈ [0, epsilon_max]: bounded on both sides
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
                        f"ε_raw={eps_raw:.4f}, ε_eff={eps_eff:.4f} "
                        f"[max={self.epsilon_max}]{t_str}")
            else:
                return f"adaptive (logC={log_c_val:.3f}): input-dependent{t_str}"


class HydraChannelMixer(nn.Module):
    """
    Cross-variable mixing via Hydra with sigmoid-bounded hybrid gating.
    """
    def __init__(self, d_model, variant='hydra_gated', rank=32,
                 n_channels=7, dropout=0.0,
                 gate_type='hybrid', gate_init=-5.0, gate_temp=1.0,
                 epsilon_max=EPSILON_MAX):
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
            gate_temp=gate_temp, epsilon_max=epsilon_max
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
