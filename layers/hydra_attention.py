"""
Hydra Attention v3.2 — Hybrid Gate: Channel-Primary + Learned Data Bonus.

Design: gate = sigmoid( f(logC) + ε · g(data_stats) )

- f(logC): Channel-based gate (works like v3.1 'channel'). Primary signal.
- g(data_stats): Data-dependent correction from channel variance.
- ε: Learnable scalar, initialized at 0. Model must learn to use data signal.

Why this works:
- At init, ε=0 → pure channel gate → matches 'channel' performance exactly
- During training, if data stats help (e.g., Exchange), ε grows
- If data stats hurt (e.g., Solar), ε stays near zero
- Best of both worlds: channel's reliability + adaptive's flexibility

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
        global_feat = (K * V).sum(dim=1, keepdim=True)
        return self.dropout(Q * global_feat)


class AdaptiveGate(nn.Module):
    """
    Hybrid gate: channel-primary with learned data bonus.
    
    gate = sigmoid( f(logC) + ε · g(variance) )
    
    f(logC):      MLP that maps normalized log(C) → per-feature logits
    g(variance):  MLP that maps channel variance → per-feature correction
    ε:            Learnable scalar, init=0 (pure channel at start)
    
    Args:
        d_model:     Feature dimension
        n_channels:  Number of variables (C)
        gate_type:   'hybrid', 'channel', 'adaptive', 'scalar'
        init_bias:   Initial bias for sigmoid
    """
    def __init__(self, d_model, n_channels=1, gate_type='hybrid', init_bias=-5.0):
        super().__init__()
        self.gate_type = gate_type
        self.d_model = d_model
        
        # Precompute normalized log(C)
        self.register_buffer(
            'log_c', torch.tensor(math.log(max(n_channels, 2)) / math.log(1000))
        )
        
        if gate_type == 'scalar':
            self.bias = nn.Parameter(torch.tensor(init_bias))
            
        elif gate_type == 'channel':
            # Pure log(C) gate
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
            
            # Data bonus: g(variance) → per-feature correction
            # Uses channel variance only (mean_abs hurt in v3.1)
            self.data_net = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
            )
            nn.init.zeros_(self.data_net[2].weight)
            nn.init.zeros_(self.data_net[2].bias)  # g starts at zero output
            
            # ε: learnable mixing weight, init=0 → pure channel at start
            self.epsilon = nn.Parameter(torch.tensor(0.0))
            
        elif gate_type == 'adaptive':
            # Full adaptive (v3.1 design, kept for comparison)
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
            x_input: [B, C, D] original input
            mixed:   [B, C, D] Hydra output
        Returns:
            gate: values in (0, 1), broadcastable to [B, C, D]
        """
        if self.gate_type == 'scalar':
            return torch.sigmoid(self.bias)
            
        elif self.gate_type == 'channel':
            log_c_input = self.log_c.view(1, 1)         # [1, 1]
            logits = self.channel_net(log_c_input)        # [1, D]
            return torch.sigmoid(logits).unsqueeze(1)     # [1, 1, D]
            
        elif self.gate_type == 'hybrid':
            # Primary signal: f(logC)
            log_c_input = self.log_c.view(1, 1)           # [1, 1]
            channel_logits = self.channel_net(log_c_input) # [1, D]
            
            # Data bonus: ε · g(variance)
            chan_var = x_input.var(dim=1)                   # [B, D]
            data_correction = self.data_net(chan_var)        # [B, D]
            
            # Combine: channel primary + scaled data correction
            # channel_logits broadcasts over batch dim
            logits = channel_logits + self.epsilon * data_correction  # [B, D]
            
            return torch.sigmoid(logits).unsqueeze(1)      # [B, 1, D]
            
        elif self.gate_type == 'adaptive':
            B, C, D = x_input.shape
            chan_var = x_input.var(dim=1)
            chan_mean_abs = x_input.abs().mean(dim=1)
            log_c_expanded = self.log_c.expand(B, 1)
            gate_input = torch.cat([chan_var, chan_mean_abs, log_c_expanded], dim=1)
            gate_logits = self.gate_net(gate_input)
            return torch.sigmoid(gate_logits).unsqueeze(1)
    
    def get_gate_info(self):
        """Return gate state for monitoring."""
        with torch.no_grad():
            log_c_val = self.log_c.item()
            if self.gate_type == 'scalar':
                return f"scalar: {torch.sigmoid(self.bias).item():.4f}"
            elif self.gate_type == 'channel':
                log_c_input = self.log_c.view(1, 1)
                vals = torch.sigmoid(self.channel_net(log_c_input))
                return f"channel (logC={log_c_val:.3f}): mean_gate={vals.mean().item():.4f}"
            elif self.gate_type == 'hybrid':
                log_c_input = self.log_c.view(1, 1)
                base = torch.sigmoid(self.channel_net(log_c_input))
                eps = self.epsilon.item()
                return (f"hybrid (logC={log_c_val:.3f}): "
                        f"base_gate={base.mean().item():.4f}, ε={eps:.4f}")
            else:
                return f"adaptive (logC={log_c_val:.3f}): input-dependent"


class HydraChannelMixer(nn.Module):
    """
    Cross-variable mixing via Hydra with hybrid gating.
    """
    def __init__(self, d_model, variant='hydra_gated', rank=32,
                 n_channels=7, dropout=0.0,
                 gate_type='hybrid', gate_init=-5.0):
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
        
        h = x.reshape(B, C, P, D).permute(0, 2, 1, 3).reshape(B * P, C, D)
        
        h_norm = self.norm(h)
        mixed = self._mix(h_norm)
        gate_val = self.gate(h, mixed)
        out = h + gate_val * mixed
        
        # Store actual gate stats for logging (detached, no grad impact)
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
        """Return actual runtime gate statistics from last forward pass."""
        if not hasattr(self, '_last_gate_mean'):
            return "no forward pass yet"
        eps_str = ""
        if hasattr(self.gate, 'epsilon'):
            eps_str = f", ε={self.gate.epsilon.item():.6f}"
        return (f"gate=[{self._last_gate_min:.6f}, {self._last_gate_mean:.6f}, {self._last_gate_max:.6f}]"
                f"{eps_str}"
                f", |mixed|={self._last_mixed_norm:.4f}"
                f", |input|={self._last_input_norm:.4f}"
                f", |gate*mixed|={self._last_contribution:.4f}")
