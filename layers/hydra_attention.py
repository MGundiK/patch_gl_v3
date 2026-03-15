"""
Hydra Attention for Cross-Variable Mixing — v3 with Adaptive Gating.

Key change: replaces scalar α with a sigmoid gate initialized cold (near-zero).
The model must actively learn to open the gate if mixing helps.

Gate designs (controlled by gate_type):
  'scalar':   Single sigmoid(bias), shared across all dims. Cheapest.
  'vector':   Per-feature sigmoid(bias), different strength per dimension.
  'adaptive': Gate depends on input cross-channel statistics.
              Computes channel variance → linear → sigmoid.
              If channels are correlated/diverse → gate opens.
              If channels are uniform → gate stays closed.

All gates initialize at ~0.5-1% mixing strength (bias=-5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Learned gate that controls mixing strength.
    
    Starts near-zero so the model defaults to channel-independent.
    Must learn to open the gate if cross-variable mixing helps.
    
    Args:
        d_model:   Feature dimension
        gate_type: 'scalar', 'vector', or 'adaptive'
        init_bias: Initial bias for sigmoid. -5 → ~0.7% mixing.
    """
    def __init__(self, d_model, gate_type='adaptive', init_bias=-5.0):
        super().__init__()
        self.gate_type = gate_type
        
        if gate_type == 'scalar':
            # Single shared gate value
            self.bias = nn.Parameter(torch.tensor(init_bias))
            
        elif gate_type == 'vector':
            # Per-feature gate
            self.bias = nn.Parameter(torch.full((d_model,), init_bias))
            
        elif gate_type == 'adaptive':
            # Gate depends on cross-channel statistics
            # Input: per-batch channel variance (how diverse are channels?)
            # High variance → channels carry different info → worth mixing
            # Low variance → channels are similar → mixing is noise
            self.gate_net = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            # Initialize output layer to produce values near init_bias
            nn.init.zeros_(self.gate_net[2].weight)
            nn.init.constant_(self.gate_net[2].bias, init_bias)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
    
    def forward(self, x_input, mixed):
        """
        Args:
            x_input: Original input [B, C, D] (before mixing)
            mixed:   Hydra output [B, C, D] (after mixing)
        Returns:
            gate:    [scalar, D, or B×1×D] values in (0, 1)
        """
        if self.gate_type == 'scalar':
            return torch.sigmoid(self.bias)
            
        elif self.gate_type == 'vector':
            return torch.sigmoid(self.bias)
            
        elif self.gate_type == 'adaptive':
            # Compute cross-channel variance as signal of diversity
            # x_input: [B, C, D]
            chan_var = x_input.var(dim=1)     # [B, D] — variance across channels
            gate_logits = self.gate_net(chan_var)  # [B, D]
            return torch.sigmoid(gate_logits).unsqueeze(1)  # [B, 1, D] — broadcast over C


class HydraChannelMixer(nn.Module):
    """
    Cross-variable mixing via Hydra Attention with adaptive gating.
    
    v3 changes from v2:
      - Replaces scalar α with AdaptiveGate
      - Gate starts near-zero (~0.7% mixing)
      - Model learns to open gate only if mixing helps
      - Works universally: auto-disables on low-C datasets
    
    Args:
        d_model:        Feature dim (D)
        variant:        'hydra', 'hydra_bottleneck', 'hydra_gated'
        rank:           Bottleneck rank
        dropout:        Dropout rate
        gate_type:      'scalar', 'vector', 'adaptive'
        gate_init:      Initial sigmoid bias (-5 ≈ 0.7%, -3 ≈ 5%, 0 = 50%)
    """
    def __init__(self, d_model, variant='hydra_gated', rank=32,
                 dropout=0.0, gate_type='adaptive', gate_init=-5.0):
        super().__init__()
        self.variant = variant
        
        # Choose attention architecture based on variant
        if variant == 'hydra':
            self.attn = HydraAttention(d_model, dropout=dropout)
            attn_dim = d_model
            
        elif variant == 'hydra_bottleneck':
            self.proj_down = nn.Linear(d_model, rank)
            self.attn = HydraAttention(rank, dropout=dropout)
            self.proj_up = nn.Linear(rank, d_model)
            attn_dim = d_model
            
        elif variant == 'hydra_gated':
            self.proj_down = nn.Linear(d_model, rank)
            self.attn = HydraAttention(rank, dropout=dropout)
            self.content_gate = nn.Sequential(
                nn.Linear(rank, rank),
                nn.Sigmoid()
            )
            self.proj_up = nn.Linear(rank, d_model)
            attn_dim = d_model
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Pre-norm
        self.norm = nn.LayerNorm(d_model)
        
        # Adaptive residual gate (replaces scalar α)
        self.gate = AdaptiveGate(d_model, gate_type=gate_type, init_bias=gate_init)
    
    def _mix(self, h_norm):
        """Apply the mixing operation (without residual)."""
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
            h_gated = h_attn * gate
            return self.proj_up(h_gated)

    def forward(self, x, B, C):
        """
        Args:
            x: [B*C, P, D] — merged batch-channel tensor
            B: Batch size
            C: Number of channels
        Returns:
            [B*C, P, D] — same shape, with adaptive cross-channel mixing
        """
        BC, P, D = x.shape
        
        # Reshape: [B*C, P, D] → [B, C, P, D] → [B*P, C, D]
        h = x.reshape(B, C, P, D).permute(0, 2, 1, 3).reshape(B * P, C, D)
        
        # Pre-norm + mix
        h_norm = self.norm(h)
        mixed = self._mix(h_norm)
        
        # Adaptive gate: decides how much mixing to apply
        gate_val = self.gate(h, mixed)  # scalar, [D], or [B*P, 1, D]
        
        # Gated residual
        out = h + gate_val * mixed
        
        # Reshape back: [B*P, C, D] → [B, C, P, D] → [B*C, P, D]
        out = out.reshape(B, P, C, D).permute(0, 2, 1, 3).reshape(BC, P, D)
        return out
    
    def get_gate_values(self):
        """Return current gate values for monitoring."""
        with torch.no_grad():
            if self.gate.gate_type == 'scalar':
                return torch.sigmoid(self.gate.bias).item()
            elif self.gate.gate_type == 'vector':
                return torch.sigmoid(self.gate.bias).mean().item()
            else:
                return "adaptive (input-dependent)"
