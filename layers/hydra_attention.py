"""
Hydra Attention v3.3a — PCA-fraction gate signal (Low-rank Mahalanobis direction).

Changes from v3.2g:
  - AdaptiveGate 'hybrid' mode: replaces marginal variance signal (chan_var)
    with PCA fraction: the fraction of total inter-channel variance explained
    by the top principal component, estimated via power iteration on h_low.

  WHY chan_var failed:
    chan_var = x.var(dim=1) [B*P, D] measures MARGINAL variance across channels.
    It cannot distinguish:
      (a) High variance from genuine inter-channel CORRELATION  → gate should OPEN   (Electricity)
      (b) High variance from heteroscedastic INDEPENDENCE       → gate should CLOSE  (Traffic, Solar)
    Both cases produce high chan_var, so Traffic and Electricity trigger the same
    gate response. v3.2g regressed Traffic and Solar precisely because of this.

  WHY pca_frac works:
    pca_frac = λ_max(cov) / trace(cov) where cov is the rank×rank covariance matrix
    of the C channel representations in the proj_down space [B*P, C, rank].

    - If channels share a common direction (genuinely correlated): λ_max ≈ trace,
      so pca_frac → 1. Gate should open.  ✓  (Electricity: homogeneous load meters)

    - If channels are heteroscedastic but independent: variance is spread across all
      rank directions, λ_max ≈ trace/rank, so pca_frac → 1/rank ≈ 0.03.
      Gate should stay closed.  ✓  (Traffic: road sensors, Solar: semi-independent)

  IMPLEMENTATION — power iteration (O(B×P×C×rank) per step):
    Computing the full covariance matrix [B*P, rank, rank] costs O(B*P*C*rank²).
    We only need λ_max, so 3 steps of power iteration suffice at O(B*P*C*rank):

        h_c = h_low - mean(h_low, dim=C)          # center over channels
        v ← random unit vector in R^rank
        for k in 1..n_power_iter:
            w ← h_c @ v       # [B*P, C]
            v ← h_c.T @ w     # [B*P, rank]  (= cov @ v, unnormalized)
            v ← normalize(v)
        λ_max ≈ ||h_c @ v||² / (C-1)             # Rayleigh quotient
        trace = ||h_c||²_F / (C-1)
        pca_frac = λ_max / (trace + ε)

    Power iterations run under torch.no_grad() with h_c.detach() for efficiency.
    The final Rayleigh quotient uses h_c on the gradient tape (v treated as constant),
    giving valid gradients via the envelope theorem. This means the gate signal is
    differentiable: the model can learn to produce more (or less) correlated channel
    representations in response to the training signal.

  ARCHITECTURE CHANGES vs v3.2g:
    data_net input dim:  d_model → 1  (scalar pca_frac, not per-feature variance)
    _mix() now returns:  (mixed, h_low) tuple instead of just mixed
    gate.forward() now:  accepts h_low kwarg and passes it to _pca_fraction()
    new gate stat:       pca_frac logged alongside existing GATE_STATS fields

  Everything else is identical to v3.2g (mean pooling in HydraAttention,
  softplus ε constraint, gate_init=-5.0, channel_net logC prior, etc.).

  GATE_STATS log format (extended):
    gate=[min, mean, max], ε_raw=X, ε_eff=Y, |mixed|=Z, |input|=W,
    |gate*mixed|=V, pca_frac=P

  EXPECTED BEHAVIOUR vs v3.2g:
    Electricity (C=321, correlated):    pca_frac high → ε grows → gate opens  ✓
    Traffic (C=862, independent):       pca_frac low  → ε stays small → gate closed  ✓
    Solar (C=137, semi-independent):    pca_frac moderate → partial gate opening  (TBD)
    ETTs (C=7, small):                  pca_frac variable per batch, gate slow to open  ✓
    Weather (C=21, correlated):         pca_frac high → aggressive gate opening  ✓
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core Hydra linear attention — identical to v3.2g (mean pooling)
# ---------------------------------------------------------------------------

class HydraAttention(nn.Module):
    """
    O(Nd) linear attention via cosine similarity with mean pooling.
    Identical to v3.2g: global_feat = (K*V).sum(dim=1) / C
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
        # Mean pooling: magnitude O(1) across all datasets (v3.2g fix)
        global_feat = (K * V).sum(dim=1, keepdim=True) / C
        return self.dropout(Q * global_feat)


# ---------------------------------------------------------------------------
# Adaptive gate — v3.3a: PCA-fraction replaces marginal variance in hybrid mode
# ---------------------------------------------------------------------------

class AdaptiveGate(nn.Module):
    """
    Hybrid gate: logC channel prior + ε * PCA-fraction data correction.

    gate = sigmoid( channel_net(logC) + ε_eff * data_net(pca_frac) )

    where:
      channel_net: small MLP mapping normalised log(C) → per-feature logits
                   initialised with bias=gate_init so gate starts near-closed
      data_net:    small MLP mapping scalar pca_frac [0,1] → per-feature correction
                   weight zero-init; bias Kaiming (same approach as v3.2g data_net)
      ε_eff:       softplus(ε_raw) ≥ 0; ε_raw starts at 0 so data signal starts silent
      pca_frac:    λ_max / trace of inter-channel covariance in proj_down space

    Gate types other than 'hybrid' are carried over from v3.2g unchanged.
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
            # Channel prior — unchanged from v3.2g
            self.channel_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.channel_net[2].weight)
            nn.init.constant_(self.channel_net[2].bias, init_bias)

            # Data correction — v3.3a: input is scalar pca_frac (dim 1)
            # instead of per-feature channel variance (dim d_model).
            self.data_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.data_net[2].weight)
            # data_net[2].bias: Kaiming uniform (PyTorch default) — same as v3.2g

            # Unconstrained learnable scalar; ε_eff = softplus(ε_raw) ≥ 0
            self._epsilon_raw = nn.Parameter(torch.tensor(0.0))

            # Diagnostic cache (populated during forward, read in get_gate_stats)
            self._last_pca_frac_mean: float = float('nan')

        elif gate_type == 'adaptive':
            # Unchanged from v3.2g — uses chan_var and mean_abs of x_input
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

    # ------------------------------------------------------------------
    # ε helpers
    # ------------------------------------------------------------------

    def _epsilon_eff(self) -> torch.Tensor:
        return F.softplus(self._epsilon_raw)

    # ------------------------------------------------------------------
    # PCA fraction (v3.3a core signal)
    # ------------------------------------------------------------------

    def _pca_fraction(self, h_low: torch.Tensor) -> torch.Tensor:
        """
        Estimate the fraction of inter-channel variance explained by the
        top principal component of h_low.

        Args:
            h_low: [B*P, C, rank]  post-proj_down channel representations

        Returns:
            pca_frac: [B*P, 1]  in [0, 1]
                → ≈ 1     if channels are perfectly correlated (all aligned)
                → ≈ 1/rank if channels are spread uniformly (independent)

        Algorithm:
            1. Mean-centre h_low over the C dimension → h_c
            2. Run n_power_iter steps of power iteration on cov = h_c.T @ h_c / (C-1)
               using h_c.detach() (no gradients through iterations)
            3. Compute Rayleigh quotient WITH h_c on the gradient tape:
               λ_max ≈ ||h_c @ v||² / (C-1)
            4. Normalise by trace(cov) = ||h_c||²_F / (C-1)

        Cost: O(B*P * C * rank) per iteration.
        Gradient: flows through the final Rayleigh quotient into proj_down weights.
        """
        BP, C, rank = h_low.shape

        # Centre over channel dimension
        h_c = h_low - h_low.mean(dim=1, keepdim=True)   # [B*P, C, rank]

        # ---- power iteration (no gradients) --------------------------------
        with torch.no_grad():
            h_c_det = h_c.detach()                        # [B*P, C, rank]

            # Random initialisation, normalised
            v = F.normalize(
                torch.randn(BP, rank, device=h_low.device, dtype=h_low.dtype),
                dim=-1,
            )                                             # [B*P, rank]

            for _ in range(self.n_power_iter):
                # w = h_c @ v   →  [B*P, C]
                w = (h_c_det @ v.unsqueeze(-1)).squeeze(-1)
                # v_new = h_c.T @ w   →  [B*P, rank]  (= cov @ v, unnormalised)
                v_new = (h_c_det.transpose(-2, -1) @ w.unsqueeze(-1)).squeeze(-1)
                v = F.normalize(v_new, dim=-1)
            # v: [B*P, rank]  ≈ top eigenvector of cov (treated as constant below)

        # ---- Rayleigh quotient ON gradient tape ----------------------------
        # λ_max ≈ v^T cov v  =  ||h_c @ v||² / (C-1)
        w_on = (h_c @ v.unsqueeze(-1)).squeeze(-1)        # [B*P, C]  — with grad
        lambda_max = (w_on ** 2).sum(dim=-1) / max(C - 1, 1)  # [B*P]

        # trace(cov) = ||h_c||²_F / (C-1)  — total inter-channel variance
        total_var = (h_c ** 2).sum(dim=(1, 2)) / max(C - 1, 1)  # [B*P]

        pca_frac = lambda_max / (total_var + 1e-8)         # [B*P]  in [0, 1]
        return pca_frac.unsqueeze(-1)                      # [B*P, 1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_input: torch.Tensor,
        mixed: torch.Tensor,
        h_low: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x_input:  [B*P, C, D]    raw channel features (pre-norm)
            mixed:    [B*P, C, D]    HydraAttention output
            h_low:    [B*P, C, rank] proj_down representations (hybrid only)

        Returns:
            gate_val: scalar or [B*P, 1, D] gate values in (0, 1)
        """
        if self.gate_type == 'scalar':
            return torch.sigmoid(self.gate_temp * self.bias)

        elif self.gate_type == 'channel':
            log_c_input = self.log_c.view(1, 1)
            logits = self.channel_net(log_c_input)              # [1, D]
            return torch.sigmoid(self.gate_temp * logits).unsqueeze(1)

        elif self.gate_type == 'hybrid':
            # --- channel prior (unchanged from v3.2g) ---
            log_c_input = self.log_c.view(1, 1)
            channel_logits = self.channel_net(log_c_input)      # [1, D]

            # --- data correction (v3.3a: PCA fraction) ---
            if h_low is not None:
                pca_frac = self._pca_fraction(h_low)            # [B*P, 1]
                data_correction = self.data_net(pca_frac)       # [B*P, D]
                # Cache for GATE_STATS logging (no grad needed)
                self._last_pca_frac_mean = pca_frac.detach().mean().item()
            else:
                # Fallback for 'hydra' variant which has no proj_down
                data_correction = torch.zeros(
                    x_input.shape[0], self.d_model,
                    device=x_input.device, dtype=x_input.dtype,
                )
                self._last_pca_frac_mean = float('nan')

            logits = channel_logits + self._epsilon_eff() * data_correction
            return torch.sigmoid(self.gate_temp * logits).unsqueeze(1)

        elif self.gate_type == 'adaptive':
            # Unchanged from v3.2g — uses x_input statistics, ignores h_low
            B, C, D = x_input.shape
            chan_var = x_input.var(dim=1)
            chan_mean_abs = x_input.abs().mean(dim=1)
            log_c_expanded = self.log_c.expand(B, 1)
            gate_input = torch.cat([chan_var, chan_mean_abs, log_c_expanded], dim=1)
            gate_logits = self.gate_net(gate_input)
            return torch.sigmoid(self.gate_temp * gate_logits).unsqueeze(1)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

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
                return (f"hybrid-pca (logC={log_c_val:.3f}): "
                        f"base_gate={base.mean().item():.4f}, "
                        f"ε_raw={eps_raw:.4f}, ε_eff={eps_eff:.4f}{t_str}")
            else:
                return f"adaptive (logC={log_c_val:.3f}): input-dependent{t_str}"


# ---------------------------------------------------------------------------
# HydraChannelMixer — v3.3a: _mix() returns (mixed, h_low) tuple
# ---------------------------------------------------------------------------

class HydraChannelMixer(nn.Module):
    """
    Cross-variable mixing via Hydra with adaptive gating.

    v3.3a changes vs v3.2g:
      - _mix() returns (mixed, h_low) instead of just mixed
      - forward() unpacks the tuple and passes h_low to the gate
      - GATE_STATS now includes pca_frac field
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

    # ------------------------------------------------------------------
    # Internal mixing — now returns (mixed, h_low) tuple
    # ------------------------------------------------------------------

    def _mix(self, h_norm: torch.Tensor):
        """
        Returns:
            mixed: [B*P, C, D]    — mixed channel output
            h_low: [B*P, C, rank] — proj_down representations, or None
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, B: int, C: int) -> torch.Tensor:
        BC, P, D = x.shape

        # Reshape to [B*P, C, D] for channel-mixing
        h = x.reshape(B, C, P, D).permute(0, 2, 1, 3).reshape(B * P, C, D)

        h_norm = self.norm(h)

        # v3.3a: unpack (mixed, h_low) from _mix
        mixed, h_low = self._mix(h_norm)

        # v3.3a: pass h_low to gate for PCA-fraction computation
        gate_val = self.gate(h, mixed, h_low=h_low)

        out = h + gate_val * mixed

        # ---- diagnostic probe (no gradients) ----
        with torch.no_grad():
            self._last_gate_mean = gate_val.mean().item()
            self._last_gate_min  = gate_val.min().item()
            self._last_gate_max  = gate_val.max().item()
            self._last_mixed_norm = mixed.norm(dim=-1).mean().item()
            self._last_input_norm = h.norm(dim=-1).mean().item()
            self._last_contribution = (gate_val * mixed).norm(dim=-1).mean().item()
            # pca_frac_mean is already cached in gate.forward() above

        out = out.reshape(B, P, C, D).permute(0, 2, 1, 3).reshape(BC, P, D)
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_gate_info(self) -> str:
        return self.gate.get_gate_info()

    def get_gate_stats(self) -> str:
        if not hasattr(self, '_last_gate_mean'):
            return "no forward pass yet"

        # ε stats (hybrid only)
        eps_str = ""
        if hasattr(self.gate, '_epsilon_raw'):
            eps_eff = self.gate._epsilon_eff().item()
            eps_raw = self.gate._epsilon_raw.item()
            eps_str = f", ε_raw={eps_raw:.6f}, ε_eff={eps_eff:.6f}"

        # pca_frac (hybrid only, v3.3a)
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
