"""
Hydra Attention v3.4a — Input-space random-pair cosine similarity gate signal.

Changes from v3.3b:
  - Gate signal replaced: pca_frac (proj_down space) → mean_cos (input space).
  - mean_cos = mean pairwise cosine similarity over K random channel pairs,
    computed on the raw [B*P, C, D] input BEFORE any learned transformation.

  WHY pca_frac failed (v3.3a and v3.3b):
    At C=862, random proj_down weights project 862 independent channels to
    rank=32 vectors that are spuriously correlated by concentration-of-measure.
    pca_frac ep1 was ~0.62 for both Traffic (independent) and Electricity
    (correlated) — identical, so the signal cannot discriminate them.
    v3.3b confirmed the cause is random initialisation, not gradient inflation:
    fully detaching the Rayleigh quotient made no difference to ep1 pca_frac.

  WHY mean_cos works:
    Before proj_down, the input h is [B*P, C, D] where each channel vector
    has been normalised via LayerNorm. For genuinely correlated channels
    (Electricity, Weather), these vectors point in similar directions → high
    mean cosine similarity. For heteroscedastic-independent channels (Traffic,
    Solar partially), vectors are spread → near-zero or slightly negative mean
    cosine similarity.

    This signal is purely geometric on the pre-projection representations.
    It cannot be corrupted by what proj_down learns.

  ALGORITHM — random-pair cosine similarity (K pairs):
    h_norm:   [B*P, C, D]  (LayerNorm-normalised input, already computed)
    1. Sample K random index pairs (i, j), i ≠ j, same for the whole batch.
    2. u = F.normalize(h_norm[:, idx_i, :], dim=-1)   # [B*P, K, D]
       v = F.normalize(h_norm[:, idx_j, :], dim=-1)   # [B*P, K, D]
    3. cos_ij = (u * v).sum(dim=-1)                   # [B*P, K]  in [-1, 1]
    4. mean_cos = cos_ij.mean(dim=-1, keepdim=True)   # [B*P, 1]  in [-1, 1]

    Cost: O(B×P×K×D) — independent of C.
    At K=64, D=96: ~393K FLOPs per forward, cheaper than power iteration.

    The pair indices are resampled each forward pass (no_grad, from a fixed
    per-module RNG for reproducibility within a single forward, but fresh
    across calls). This prevents the gate from over-fitting to fixed pairs.

  DATA_NET input:
    Same as v3.3a/b: scalar [B*P, 1] fed into the 1→d_model MLP.
    mean_cos is already in [-1, 1], well-conditioned for a small MLP.
    No normalisation needed.

  ARCHITECTURE CHANGES vs v3.3b:
    _pca_fraction() removed entirely.
    _mean_cos_similarity() added — runs under torch.no_grad().
    AdaptiveGate.forward(): h_low replaced by h_norm as the signal input.
    HydraChannelMixer.forward(): passes h_norm (not h_low) to gate.
    GATE_STATS: pca_frac field replaced by mean_cos field.
    get_gate_info(): label updated to 'hybrid-cos'.

  EXPECTED BEHAVIOUR:
    Traffic (C=862, independent road sensors):
      mean_cos ep1 ≈ near zero or small negative → gate stays closed → MSE improves  ✓
    Electricity (C=321, homogeneous load meters):
      mean_cos ep1 ≈ moderate positive → gate opens appropriately → MSE neutral/+     ✓
    Solar (C=137, semi-independent panels):
      mean_cos ep1 ≈ small positive (lower than Electricity) → partial gate opening   ✓
    ETTs (C=7, small):
      mean_cos ep1 variable → gate behaviour similar to v3.2g                          ✓
    Weather (C=21, correlated):
      mean_cos ep1 ≈ moderate-high → gate opens                                        ✓

  IF mean_cos IS ALSO INDISTINGUISHABLE for Traffic vs Electricity:
    The LayerNorm before the signal computation is the confound — it normalises
    each channel independently, potentially destroying scale information that
    would otherwise separate the datasets. Next step would be to compute
    mean_cos on the pre-LayerNorm input (x, not h_norm).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core Hydra linear attention — identical to v3.2g/v3.3a/v3.3b (mean pooling)
# ---------------------------------------------------------------------------

class HydraAttention(nn.Module):
    """
    O(Nd) linear attention via cosine similarity with mean pooling.
    Identical to v3.2g/v3.3x: global_feat = (K*V).sum(dim=1) / C
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
# Adaptive gate — v3.4a: input-space random-pair cosine similarity
# ---------------------------------------------------------------------------

class AdaptiveGate(nn.Module):
    """
    Hybrid gate: logC channel prior + ε * mean_cos data correction.

    gate = sigmoid( channel_net(logC) + ε_eff * data_net(mean_cos) )

    mean_cos: mean pairwise cosine similarity of K random channel pairs,
    computed on the LayerNorm-normalised input before any learned transform.
    Runs fully under torch.no_grad() — no gradient path into the signal.
    """

    def __init__(
        self,
        d_model: int,
        n_channels: int = 1,
        gate_type: str = 'hybrid',
        init_bias: float = -5.0,
        gate_temp: float = 1.0,
        n_cos_pairs: int = 64,
    ):
        super().__init__()
        self.gate_type = gate_type
        self.d_model = d_model
        self.gate_temp = gate_temp
        self.n_cos_pairs = n_cos_pairs
        self._n_channels = n_channels   # stored for pair sampling

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
            # Channel prior — unchanged from v3.2g/v3.3x
            self.channel_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.channel_net[2].weight)
            nn.init.constant_(self.channel_net[2].bias, init_bias)

            # Data correction — input dim 1 (scalar mean_cos), same as v3.3x
            self.data_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.data_net[2].weight)
            # data_net[2].bias: Kaiming uniform (default)

            self._epsilon_raw = nn.Parameter(torch.tensor(0.0))
            self._last_mean_cos: float = float('nan')

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

    def _mean_cos_similarity(self, h_norm: torch.Tensor) -> torch.Tensor:
        """
        Estimate mean pairwise cosine similarity from K random channel pairs.

        Args:
            h_norm: [B*P, C, D]  LayerNorm-normalised channel representations

        Returns:
            mean_cos: [B*P, 1]  in [-1, 1], fully detached

        For genuinely correlated channels (Electricity): mean_cos > 0, moderate.
        For heteroscedastic-independent channels (Traffic): mean_cos ≈ 0.
        For anti-correlated channels: mean_cos < 0.

        Pair indices are sampled fresh each call (no_grad, CPU, fast).
        When C <= 1, returns zeros (degenerate case).
        When K >= C*(C-1)/2 (few channels), use all pairs instead.
        """
        with torch.no_grad():
            BP, C, D = h_norm.shape

            if C <= 1:
                return torch.zeros(BP, 1, device=h_norm.device, dtype=h_norm.dtype)

            # Build random pair indices — done on CPU then moved to device
            # Use all pairs if C is small enough (C=7 → 21 pairs < K=64)
            max_pairs = C * (C - 1) // 2
            K = min(self.n_cos_pairs, max_pairs)

            if max_pairs <= self.n_cos_pairs:
                # Enumerate all pairs (small C case, e.g. ETTs C=7)
                idx_i, idx_j = zip(*[
                    (i, j) for i in range(C) for j in range(i + 1, C)
                ])
                idx_i = torch.tensor(idx_i, device=h_norm.device)
                idx_j = torch.tensor(idx_j, device=h_norm.device)
            else:
                # Sample K random distinct pairs
                # Rejection-sample until we have K distinct pairs
                # (at high C this succeeds in one shot with near-certainty)
                pairs = set()
                while len(pairs) < K:
                    batch_i = torch.randint(0, C, (K * 2,)).tolist()
                    batch_j = torch.randint(0, C, (K * 2,)).tolist()
                    for a, b in zip(batch_i, batch_j):
                        if a != b:
                            pairs.add((min(a, b), max(a, b)))
                        if len(pairs) >= K:
                            break
                pairs = list(pairs)[:K]
                idx_i = torch.tensor([p[0] for p in pairs], device=h_norm.device)
                idx_j = torch.tensor([p[1] for p in pairs], device=h_norm.device)

            # Extract channel vectors for sampled pairs
            h_det = h_norm.detach()                          # [B*P, C, D]
            u = h_det[:, idx_i, :]                           # [B*P, K, D]
            v = h_det[:, idx_j, :]                           # [B*P, K, D]

            # Normalise to unit vectors (LayerNorm ensures non-zero but add eps)
            u = F.normalize(u, p=2, dim=-1, eps=1e-8)
            v = F.normalize(v, p=2, dim=-1, eps=1e-8)

            # Cosine similarity per pair, mean over pairs
            cos_ij = (u * v).sum(dim=-1)                     # [B*P, K]
            mean_cos = cos_ij.mean(dim=-1, keepdim=True)     # [B*P, 1]

        return mean_cos   # detached

    def forward(
        self,
        x_input: torch.Tensor,
        mixed: torch.Tensor,
        h_norm: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x_input:  [B*P, C, D]    raw channel features (pre-norm)
            mixed:    [B*P, C, D]    HydraAttention output
            h_norm:   [B*P, C, D]    LayerNorm output — signal computed here

        Returns:
            gate_val: [B*P, 1, D] or scalar, in (0, 1)
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

            if h_norm is not None:
                mean_cos = self._mean_cos_similarity(h_norm)   # [B*P, 1], detached
                data_correction = self.data_net(mean_cos)      # [B*P, D]
                self._last_mean_cos = mean_cos.mean().item()
            else:
                data_correction = torch.zeros(
                    x_input.shape[0], self.d_model,
                    device=x_input.device, dtype=x_input.dtype,
                )
                self._last_mean_cos = float('nan')

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
                return (f"hybrid-cos (logC={log_c_val:.3f}): "
                        f"base_gate={base.mean().item():.4f}, "
                        f"ε_raw={eps_raw:.4f}, ε_eff={eps_eff:.4f}{t_str}")
            else:
                return f"adaptive (logC={log_c_val:.3f}): input-dependent{t_str}"


# ---------------------------------------------------------------------------
# HydraChannelMixer — v3.4a: passes h_norm to gate instead of h_low
# ---------------------------------------------------------------------------

class HydraChannelMixer(nn.Module):
    """
    Cross-variable mixing via Hydra with adaptive gating.

    v3.4a change vs v3.3b:
      - gate.forward() receives h_norm (LayerNorm output) as signal input,
        replacing h_low (proj_down output) used in v3.3a/b.
      - GATE_STATS field: pca_frac → mean_cos
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
        n_cos_pairs: int = 64,
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
            n_cos_pairs=n_cos_pairs,
        )

    def _mix(self, h_norm: torch.Tensor):
        """
        Returns:
            mixed: [B*P, C, D]
        """
        if self.variant == 'hydra':
            return self.attn(h_norm)

        elif self.variant == 'hydra_bottleneck':
            h_low = self.proj_down(h_norm)
            return self.proj_up(self.attn(h_low))

        elif self.variant == 'hydra_gated':
            h_low = self.proj_down(h_norm)
            h_attn = self.attn(h_low)
            content_g = self.content_gate(h_low)
            return self.proj_up(h_attn * content_g)

    def forward(self, x: torch.Tensor, B: int, C: int) -> torch.Tensor:
        BC, P, D = x.shape

        h = x.reshape(B, C, P, D).permute(0, 2, 1, 3).reshape(B * P, C, D)

        h_norm = self.norm(h)
        mixed = self._mix(h_norm)

        # v3.4a: pass h_norm as the signal input (not h_low)
        gate_val = self.gate(h, mixed, h_norm=h_norm)

        out = h + gate_val * mixed

        with torch.no_grad():
            self._last_gate_mean    = gate_val.mean().item()
            self._last_gate_min     = gate_val.min().item()
            self._last_gate_max     = gate_val.max().item()
            self._last_mixed_norm   = mixed.norm(dim=-1).mean().item()
            self._last_input_norm   = h.norm(dim=-1).mean().item()
            self._last_contribution = (gate_val * mixed).norm(dim=-1).mean().item()

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

        # v3.4a: mean_cos replaces pca_frac in GATE_STATS output
        cos_str = ""
        mean_cos = getattr(self.gate, '_last_mean_cos', float('nan'))
        if not math.isnan(mean_cos):
            cos_str = f", mean_cos={mean_cos:.4f}"

        return (
            f"gate=[{self._last_gate_min:.6f}, "
            f"{self._last_gate_mean:.6f}, "
            f"{self._last_gate_max:.6f}]"
            f"{eps_str}"
            f", |mixed|={self._last_mixed_norm:.4f}"
            f", |input|={self._last_input_norm:.4f}"
            f", |gate*mixed|={self._last_contribution:.4f}"
            f"{cos_str}"
        )
