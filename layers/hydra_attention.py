"""
Hydra Attention for Cross-Variable Mixing in Time Series Forecasting.

Adapted from: "Hydra Attention: Efficient Attention with Many Heads"
(Bolya et al., ECCV 2022 Workshops)

Key idea: #heads = feature_dim → each head operates on a scalar.
Uses L2-normalized cosine similarity instead of softmax.
Complexity: O(C * D) — linear in both channels and features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HydraAttention(nn.Module):
    """
    Core Hydra Attention operation.
    
    Standard attention: softmax(QK^T / sqrt(d)) V  → O(N²d)
    Hydra attention:    Q ⊙ Σ_n(K_n ⊙ V_n)        → O(Nd)
    
    Uses L2-normalization on Q, K so the operation computes
    cosine-similarity-weighted aggregation.
    
    Args:
        d_model: Feature dimension per token.
        dropout: Dropout on output.
    """
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, N, D]  (N = tokens/channels, D = features)
        Returns:
            [B, N, D]  with cross-token information mixed
        """
        Q = F.normalize(self.W_q(x), p=2, dim=-1)   # [B, N, D]
        K = F.normalize(self.W_k(x), p=2, dim=-1)   # [B, N, D]
        V = self.W_v(x)                               # [B, N, D]

        # Hydra trick: element-wise K*V, sum across tokens → global vector
        global_feat = (K * V).sum(dim=1, keepdim=True)  # [B, 1, D]

        # Each token queries the global feature via element-wise product
        out = Q * global_feat   # [B, N, D]
        return self.dropout(out)


class HydraChannelMixer(nn.Module):
    """
    Cross-variable mixing via Hydra Attention.
    
    Designed to slot into a channel-independent pipeline where the
    channel dim is merged with batch (B*C). Call with the original
    B and C so it can temporarily un-merge, apply Hydra across C,
    and re-merge.
    
    Operates per-patch-position: reshapes [B, C, P, D] → [B*P, C, D],
    applies Hydra across C tokens, reshapes back.
    
    Variants:
        'hydra':            Full-dim Hydra on patch embeddings
        'hydra_bottleneck': Project D→rank, Hydra, project rank→D
        'hydra_gated':      Hydra + learned sigmoid gate (filters noise)
    
    Args:
        d_model:        Patch embedding dimension (D = patch_len²).
        variant:        One of 'hydra', 'hydra_bottleneck', 'hydra_gated'.
        rank:           Bottleneck rank (only for 'hydra_bottleneck').
        dropout:        Dropout rate.
        residual_scale: Initial α for residual blend (small = safe start).
    """
    def __init__(self, d_model, variant='hydra', rank=32,
                 dropout=0.0, residual_scale=0.1):
        super().__init__()
        self.variant = variant

        if variant == 'hydra':
            # Full-dim Hydra — high param count, use only for small d_model
            self.attn = HydraAttention(d_model, dropout=dropout)

        elif variant == 'hydra_bottleneck':
            # Project D→rank, Hydra on rank, project back
            self.proj_down = nn.Linear(d_model, rank)
            self.attn = HydraAttention(rank, dropout=dropout)
            self.proj_up = nn.Linear(rank, d_model)

        elif variant == 'hydra_gated':
            # Bottleneck + sigmoid gating — RECOMMENDED default
            # Gating filters noisy cross-variable signals;
            # bottleneck keeps params low (~20K vs ~263K for full-dim)
            self.proj_down = nn.Linear(d_model, rank)
            self.attn = HydraAttention(rank, dropout=dropout)
            self.gate = nn.Sequential(
                nn.Linear(rank, rank),
                nn.Sigmoid()
            )
            self.proj_up = nn.Linear(rank, d_model)

        else:
            raise ValueError(f"Unknown Hydra variant: {variant}")

        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(residual_scale))

    def forward(self, x, B, C):
        """
        Args:
            x: [B*C, P, D]  — merged batch-channel tensor from the CI pipeline
            B: int           — original batch size
            C: int           — number of channels/variables
        Returns:
            [B*C, P, D]     — same shape, with cross-variable info mixed in
        """
        P = x.shape[1]
        D = x.shape[2]

        # Un-merge channels: [B*C, P, D] → [B, C, P, D]
        h = x.reshape(B, C, P, D)

        # Treat each patch position independently: [B, C, P, D] → [B*P, C, D]
        h = h.permute(0, 2, 1, 3).reshape(B * P, C, D)

        # Pre-norm
        h_norm = self.norm(h)

        # Apply variant
        if self.variant == 'hydra':
            mixed = self.attn(h_norm)

        elif self.variant == 'hydra_bottleneck':
            h_low = self.proj_down(h_norm)      # [B*P, C, rank]
            h_mixed = self.attn(h_low)           # [B*P, C, rank]
            mixed = self.proj_up(h_mixed)        # [B*P, C, D]

        elif self.variant == 'hydra_gated':
            h_low = self.proj_down(h_norm)       # [B*P, C, rank]
            h_attn = self.attn(h_low)            # [B*P, C, rank]
            gate = self.gate(h_low)              # [B*P, C, rank] ∈ (0,1)
            mixed = self.proj_up(h_attn * gate)  # [B*P, C, D]

        # Residual with learnable scale
        out = h + self.alpha * mixed

        # Re-merge: [B*P, C, D] → [B, P, C, D] → [B, C, P, D] → [B*C, P, D]
        out = out.reshape(B, P, C, D).permute(0, 2, 1, 3).reshape(B * C, P, D)

        return out
