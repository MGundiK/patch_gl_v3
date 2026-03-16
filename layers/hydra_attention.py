"""
Hydra Attention for Cross-Variable Mixing — v3.1 with Channel-Aware Gating.

Key change from v3: The adaptive gate receives log(C) as an explicit input,
giving it a strong prior about when mixing is useful.

v3 problem: channel variance was too noisy a signal — Exchange (C=8) has
high variance, Traffic (C=862) can have low variance. The gate couldn't
distinguish "few diverse channels" from "many correlated channels."

v3.1 fix: Concatenate normalized log(C) to the gate input. Now the gate
network sees BOTH the data statistics (variance) AND the structural prior
(how many channels exist). This lets it learn rules like:
  - C=7 + any variance → stay closed
  - C=321 + high variance → open wide
  - C=862 + low variance → still open (many sensors = worth mixing)

Gate types:
  'scalar':    Single sigmoid(bias). No data dependence.
  'channel':   Gate depends on log(C) only. Learned C-threshold.
  'adaptive':  Gate depends on log(C) + channel variance. Full flexibility.
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
    Channel-aware adaptive gate.
    
    Receives:
      - Cross-channel variance (data signal)
      - log(C) / log(1000) (structural prior, normalized to ~0-1)
    
    Concatenates them and produces per-feature gate values.
    
    Args:
        d_model:     Feature dimension
        n_channels:  Number of variables (C) — used for log(C) injection
        gate_type:   'scalar', 'channel', 'adaptive'
        init_bias:   Initial bias for sigmoid (-5 ≈ 0.7%, -3 ≈ 5%)
    """
    def __init__(self, d_model, n_channels=1, gate_type='adaptive', init_bias=-5.0):
        super().__init__()
        self.gate_type = gate_type
        self.d_model = d_model
        
        # Precompute normalized log(C) as a constant
        # log(7)/log(1000) ≈ 0.28, log(21)/log(1000) ≈ 0.44,
        # log(137)/log(1000) ≈ 0.71, log(321)/log(1000) ≈ 0.84,
        # log(862)/log(1000) ≈ 0.98
        self.register_buffer(
            'log_c', torch.tensor(math.log(max(n_channels, 2)) / math.log(1000))
        )
        
        if gate_type == 'scalar':
            self.bias = nn.Parameter(torch.tensor(init_bias))
            
        elif gate_type == 'channel':
            # Gate depends ONLY on log(C) — learns a channel threshold
            # Input: [1] (just log_c) → output: [d_model] gate values
            self.gate_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.gate_net[2].weight)
            nn.init.constant_(self.gate_net[2].bias, init_bias)
            
        elif gate_type == 'adaptive':
            # Gate depends on log(C) + channel variance + channel mean_abs
            # Input: [d_model + d_model + 1] = [2*d_model + 1]
            #   - d_model: per-feature channel variance
            #   - d_model: per-feature mean absolute value (scale signal)
            #   - 1: log(C) normalized
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
    
    def forward(self, x_input, mixed):
        """
        Args:
            x_input: [B, C, D] original input (before mixing)
            mixed:   [B, C, D] Hydra output
        Returns:
            gate: values in (0, 1), broadcastable to [B, C, D]
        """
        if self.gate_type == 'scalar':
            return torch.sigmoid(self.bias)
            
        elif self.gate_type == 'channel':
            # log_c is constant per dataset — gate is same for all batches
            log_c_input = self.log_c.unsqueeze(0)  # [1, 1]
            gate_logits = self.gate_net(log_c_input)  # [1, D]
            return torch.sigmoid(gate_logits).unsqueeze(1)  # [1, 1, D]
            
        elif self.gate_type == 'adaptive':
            B, C, D = x_input.shape
            
            # Channel variance: how diverse are the channels?
            chan_var = x_input.var(dim=1)        # [B, D]
            
            # Mean absolute: what's the overall scale?
            chan_mean_abs = x_input.abs().mean(dim=1)  # [B, D]
            
            # log(C): structural prior (same for all batches, broadcast)
            log_c_expanded = self.log_c.expand(B, 1)  # [B, 1]
            
            # Concatenate: [B, 2*D + 1]
            gate_input = torch.cat([chan_var, chan_mean_abs, log_c_expanded], dim=1)
            
            gate_logits = self.gate_net(gate_input)  # [B, D]
            return torch.sigmoid(gate_logits).unsqueeze(1)  # [B, 1, D]


class HydraChannelMixer(nn.Module):
    """
    Cross-variable mixing via Hydra with channel-aware adaptive gating.
    
    Args:
        d_model:        Feature dim (D)
        variant:        'hydra', 'hydra_bottleneck', 'hydra_gated'
        rank:           Bottleneck rank
        n_channels:     Number of variables (C) — passed to gate
        dropout:        Dropout rate
        gate_type:      'scalar', 'channel', 'adaptive'
        gate_init:      Initial sigmoid bias
    """
    def __init__(self, d_model, variant='hydra_gated', rank=32,
                 n_channels=7, dropout=0.0,
                 gate_type='adaptive', gate_init=-5.0):
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
        
        # Channel-aware adaptive gate
        self.gate = AdaptiveGate(
            d_model, n_channels=n_channels,
            gate_type=gate_type, init_bias=gate_init
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
        
        # Reshape: [B*C, P, D] → [B*P, C, D]
        h = x.reshape(B, C, P, D).permute(0, 2, 1, 3).reshape(B * P, C, D)
        
        h_norm = self.norm(h)
        mixed = self._mix(h_norm)
        
        # Adaptive gate with log(C) prior
        gate_val = self.gate(h, mixed)
        
        out = h + gate_val * mixed
        
        # Reshape back: [B*P, C, D] → [B*C, P, D]
        out = out.reshape(B, P, C, D).permute(0, 2, 1, 3).reshape(BC, P, D)
        return out
    
    def get_gate_values(self):
        """Return current gate info for monitoring."""
        with torch.no_grad():
            if self.gate.gate_type == 'scalar':
                return f"scalar: {torch.sigmoid(self.gate.bias).item():.4f}"
            elif self.gate.gate_type == 'channel':
                log_c = self.gate.log_c.unsqueeze(0)
                vals = torch.sigmoid(self.gate.gate_net(log_c))
                return f"channel (logC={self.gate.log_c.item():.3f}): mean={vals.mean().item():.4f}"
            else:
                return f"adaptive (logC={self.gate.log_c.item():.3f}): input-dependent"
