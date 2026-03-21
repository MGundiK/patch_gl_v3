"""
Hydra Attention v3.3b — PCA-fraction gate signal, fully detached.

Changes from v3.3a:
  - _pca_fraction() now runs entirely under torch.no_grad() with h_c.detach().
    The Rayleigh quotient that was on the gradient tape in v3.3a has been moved
    inside the no_grad block, so NO gradient flows from pca_frac into proj_down.

  WHY v3.3a failed on Traffic:
    v3.3a's pca_frac at Traffic ep1 was 0.58–0.68 — identical to Electricity's
    0.54–0.68. The signal failed to discriminate the two datasets from the start.

    Two possible explanations:
      (A) RANDOM INIT: At C=862, even random proj_down weights create spurious
          low-dimensional correlations. When 862 independent channels are projected
          to rank=32, the birthday-problem-like accumulation of random inner products
          produces pca_frac ~ 0.1–0.4 from random init alone — much higher than the
          true 1/rank = 0.03 floor for uniform independent data.

      (B) GRADIENT INFLATION: The Rayleigh quotient in v3.3a was on the gradient
          tape (h_c, not h_c.detach()). This allowed proj_down to learn to maximise
          pca_frac as a side effect of training — it is rewarded for making all
          channels align in the rank-32 space because that opens the gate and allows
          mixing, which the model can exploit for Electricity. The same reward signal
          applies to Traffic, inflating its pca_frac to Electricity levels even though
          the original channels are independent.

    v3.3b tests hypothesis (B). By computing pca_frac entirely under no_grad:
      - If Traffic pca_frac ep1 drops clearly below Electricity (e.g. Traffic ~0.1–0.2,
        Electricity ~0.5+), gradient inflation was the cause. ✓ v3.3b fixes it.
      - If Traffic pca_frac ep1 is still 0.5+ (same as v3.3a), the problem is random
        init (hypothesis A). → Need input-space signal instead (v3.4a).

  ARCHITECTURE CHANGES vs v3.3a:
    _pca_fraction(): entire function now runs under torch.no_grad() with h_c.detach().
    The return value is a detached scalar tensor — no gradient path into proj_down.
    All other code identical to v3.3a.

  EXPECTED BEHAVIOUR:
    If (B) is correct:
      Traffic:      pca_frac ep1 drops to ~0.05–0.15 → gate stays closed → MSE improves
      Electricity:  pca_frac ep1 stays high ~0.5+    → gate opens correctly → MSE neutral
      Solar:        pca_frac ep1 moderate ~0.3–0.5   → partial gate opening → TBD

    If (A) is correct (both at ~0.5+ at ep1 again):
      Next step is Option A: input-space random-pair cosine similarity signal (v3.4a).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core Hydra linear attention — identical to v3.2g / v3.3a (mean pooling)
# ---------------------------------------------------------------------------

class HydraAttention(nn.Module):
    """
    O(Nd) linear attention via cosine similarity with mean pooling.
    Identical to v3.2g / v3.3a: global_feat = (K*V).sum(dim=1) / C
    so magnitude is O(1) regardless of channel count.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*P, C, d_model]
        Q = F.normalize(self.W_q(x), p=2, dim=-1)
        K = F.normalize(self.W_k(x), p=2, dim=-1)
        V = self.W_v(x)
        C = x.shape[1]
        global_feat = (K * V).sum(dim=1, keepdim=True) / C
        return self.dropout(Q * global_feat)


# ---------------------------------------------------------------------------
# Adaptive gate — v3.3b: pca_frac fully detached from gradient tape
# ---------------------------------------------------------------------------

class AdaptiveGate(nn.Module):
    """
    Hybrid gate: logC channel prior + ε * PCA-fraction data correction.

    gate = sigmoid( channel_net(logC) + ε_eff * data_net(pca_frac) )

    v3.3b change: pca_frac computed entirely under torch.no_grad() using
    h_c.detach(), so NO gradient flows from the gate signal into proj_down.
    This tests whether gradient inflation caused Traffic's pca_frac to be
    artificially high in v3.3a.
    """

    def __init__(
        self,
        d_model: int,
        n_channels: int = 1,
        gate_type: str = 'hybrid',
        init_bias: float = -5.0,
        gate_temp: float = 1.0,
        n_power_iter: int = 3,
    ):
        super().__init__()
        self.gate_type = gate_type
        self.d_model = d_model
        self.gate_temp = gate_temp
        self.n_power_iter = n_power_iter

        self.register_buffer(
            'log_c',
            torch.tensor(math.log(max(n_channels, 2)) / math.log(1000))
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
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.data_net[2].weight)
            # data_net[2].bias: Kaiming uniform (default) — same as v3.3a

            self._epsilon_raw = nn.Parameter(torch.tensor(0.0))
            self._last_pca_frac_mean: float = float('nan')

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
            raise ValueError(f"Unknown gate_type: {gate_type!r}")

    def _epsilon_eff(self) -> torch.Tensor:
        return F.softplus(self._epsilon_raw)

    def _pca_fraction(self, h_low: torch.Tensor) -> torch.Tensor:
        """
        Estimate the fraction of inter-channel variance explained by the
        top principal component of h_low.

        v3.3b: FULLY DETACHED — runs entirely under torch.no_grad() using
        h_c.detach(). No gradient flows from pca_frac into proj_down.

        Args:
            h_low: [B*P, C, rank]  post-proj_down representations

        Returns:
            pca_frac: [B*P, 1]  in [0, 1], detached tensor

        Algorithm:
            1. Mean-centre h_low over the C dimension → h_c
            2. Run n_power_iter steps of power iteration on cov = h_c.T @ h_c / (C-1)
            3. Compute Rayleigh quotient: λ_max ≈ ||h_c @ v||² / (C-1)
            4. Normalise by trace(cov) = ||h_c||²_F / (C-1)

        v3.3a had step 3 on the gradient tape (h_c, not detached).
        v3.3b moves everything including step 3 inside no_grad.
        """
        with torch.no_grad():
            BP, C, rank = h_low.shape
            h_c = h_low.detach() - h_low.detach().mean(dim=1, keepdim=True)

            # Power iteration to find top eigenvector of cov = h_c.T @ h_c / (C-1)
            v = F.normalize(
                torch.randn(BP, rank, device=h_low.device, dtype=h_low.dtype),
                dim=-1,
            )

            for _ in range(self.n_power_iter):
                w = (h_c @ v.unsqueeze(-1)).squeeze(-1)          # [B*P, C]
                v_new = (h_c.transpose(-2, -1) @ w.unsqueeze(-1)).squeeze(-1)  # [B*P, rank]
                v = F.normalize(v_new, dim=-1)

            # Rayleigh quotient (fully detached in v3.3b)
            w_final = (h_c @ v.unsqueeze(-1)).squeeze(-1)         # [B*P, C]
            lambda_max = (w_final ** 2).sum(dim=-1) / max(C - 1, 1)  # [B*P]
            total_var = (h_c ** 2).sum(dim=(1, 2)) / max(C - 1, 1)   # [B*P]
            pca_frac = lambda_max / (total_var + 1e-8)                # [B*P]

        return pca_frac.unsqueeze(-1)   # [B*P, 1], detached

    def forward(
        self,
        x_input: torch.Tensor,
        mixed: torch.Tensor,
        h_low: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.gate_type == 'scalar':
            return torch.sigmoid(self.gate_temp * self.bias)

        elif self.gate_type == 'channel':
            log_c_input = self.log_c.view(1, 1)
            logits = self.channel_net(log_c_input)
            return torch.sigmoid(self.gate_temp * logits).unsqueeze(1)

        elif self.gate_type == 'hybrid':
            log_c_input = self.log_c.view(1, 1)
            channel_logits = self.channel_net(log_c_input)    # [1, D]

            if h_low is not None:
                pca_frac = self._pca_fraction(h_low)          # [B*P, 1], detached
                data_correction = self.data_net(pca_frac)     # [B*P, D]
                self._last_pca_frac_mean = pca_frac.mean().item()
            else:
                data_correction = torch.zeros(
                    x_input.shape[0], self.d_model,
                    device=x_input.device, dtype=x_input.dtype,
                )
                self._last_pca_frac_mean = float('nan')

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

    def get_gate_info(self) -> str:
        with torch.no_grad():
            log_c_val = self.log_c.item()
            t = self.gate_temp
            t_str = f", τ={t:.1f}" if t != 1.0 else ""
            if self.gate_type == 'scalar':
                return f"scalar: {torch.sigmoid(t * self.bias).item():.4f}{t_str}"
            elif self.gate_type == 'channel':
                log_c_input = self.log_c.view(1, 1)
                vals = torch.sigmoid(t * self.channel_net(log_c_input))
                return (f"channel (logC={log_c_val:.3f}): "
                        f"mean_gate={vals.mean().item():.4f}{t_str}")
            elif self.gate_type == 'hybrid':
                log_c_input = self.log_c.view(1, 1)
                base = torch.sigmoid(t * self.channel_net(log_c_input))
                eps_raw = self._epsilon_raw.item()
                eps_eff = self._epsilon_eff().item()
                return (f"hybrid-pca-detach (logC={log_c_val:.3f}): "
                        f"base_gate={base.mean().item():.4f}, "
                        f"ε_raw={eps_raw:.4f}, ε_eff={eps_eff:.4f}{t_str}")
            else:
                return f"adaptive (logC={log_c_val:.3f}): input-dependent{t_str}"


# ---------------------------------------------------------------------------
# HydraChannelMixer — identical to v3.3a except version label
# ---------------------------------------------------------------------------

class HydraChannelMixer(nn.Module):
    """
    Cross-variable mixing via Hydra with adaptive gating.
    Identical to v3.3a except AdaptiveGate now uses fully-detached pca_frac.
    """

    def __init__(
        self,
        d_model: int,
        variant: str = 'hydra_gated',
        rank: int = 32,
        n_channels: int = 7,
        dropout: float = 0.0,
        gate_type: str = 'hybrid',
        gate_init: float = -5.0,
        gate_temp: float = 1.0,
        n_power_iter: int = 3,
    ):
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
                nn.Sigmoid(),
            )
            self.proj_up = nn.Linear(rank, d_model)

        else:
            raise ValueError(f"Unknown variant: {variant!r}")

        self.norm = nn.LayerNorm(d_model)

        self.gate = AdaptiveGate(
            d_model,
            n_channels=n_channels,
            gate_type=gate_type,
            init_bias=gate_init,
            gate_temp=gate_temp,
            n_power_iter=n_power_iter,
        )

    def _mix(self, h_norm: torch.Tensor):
        """
        Returns:
            mixed: [B*P, C, D]
            h_low: [B*P, C, rank] or None
        """
        if self.variant == 'hydra':
            return self.attn(h_norm), None

        elif self.variant == 'hydra_bottleneck':
            h_low = self.proj_down(h_norm)
            h_mixed = self.attn(h_low)
            return self.proj_up(h_mixed), h_low

        elif self.variant == 'hydra_gated':
            h_low = self.proj_down(h_norm)
            h_attn = self.attn(h_low)
            content_g = self.content_gate(h_low)
            return self.proj_up(h_attn * content_g), h_low

    def forward(self, x: torch.Tensor, B: int, C: int) -> torch.Tensor:
        BC, P, D = x.shape

        h = x.reshape(B, C, P, D).permute(0, 2, 1, 3).reshape(B * P, C, D)

        h_norm = self.norm(h)
        mixed, h_low = self._mix(h_norm)
        gate_val = self.gate(h, mixed, h_low=h_low)
        out = h + gate_val * mixed

        with torch.no_grad():
            self._last_gate_mean        = gate_val.mean().item()
            self._last_gate_min         = gate_val.min().item()
            self._last_gate_max         = gate_val.max().item()
            self._last_mixed_norm       = mixed.norm(dim=-1).mean().item()
            self._last_input_norm       = h.norm(dim=-1).mean().item()
            self._last_contribution     = (gate_val * mixed).norm(dim=-1).mean().item()

        out = out.reshape(B, P, C, D).permute(0, 2, 1, 3).reshape(BC, P, D)
        return out

    def get_gate_info(self) -> str:
        return self.gate.get_gate_info()

    def get_gate_stats(self) -> str:
        if not hasattr(self, '_last_gate_mean'):
            return "no forward pass yet"

        eps_str = ""
        if hasattr(self.gate, '_epsilon_raw'):
            eps_eff = self.gate._epsilon_eff().item()
            eps_raw = self.gate._epsilon_raw.item()
            eps_str = f", ε_raw={eps_raw:.6f}, ε_eff={eps_eff:.6f}"

        pca_str = ""
        pca_mean = getattr(self.gate, '_last_pca_frac_mean', float('nan'))
        if not math.isnan(pca_mean):
            pca_str = f", pca_frac={pca_mean:.4f}"

        return (
            f"gate=[{self._last_gate_min:.6f}, "
            f"{self._last_gate_mean:.6f}, "
            f"{self._last_gate_max:.6f}]"
            f"{eps_str}"
            f", |mixed|={self._last_mixed_norm:.4f}"
            f", |input|={self._last_input_norm:.4f}"
            f", |gate*mixed|={self._last_contribution:.4f}"
            f"{pca_str}"
        )
