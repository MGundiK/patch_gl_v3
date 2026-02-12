import torch
from torch import nn


class InterPatchGating(nn.Module):
    """
    GLCN-inspired inter-patch gating module.
    
    Captures global dynamics by learning patch importance weights:
    GlobalAvgPool (over features) → MLP → Sigmoid → element-wise scaling.
    """
    def __init__(self, patch_num, reduction=4):
        super(InterPatchGating, self).__init__()
        hidden = max(patch_num // reduction, 2)
        self.mlp = nn.Sequential(
            nn.Linear(patch_num, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, patch_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B*C, patch_num, patch_len]
        w = x.mean(dim=2)          # GAP over feature dim → [B*C, patch_num]
        w = self.mlp(w)            # Importance weights   → [B*C, patch_num]
        w = w.unsqueeze(2)         # Broadcast shape      → [B*C, patch_num, 1]
        return x * w


class GLPatchNetwork(nn.Module):
    """
    GLPatch v6 dual-stream network.
    
    v6 fix over v5:
    - CONSTRAINED fusion gate to [0.2, 0.8] — prevents degenerate collapse
      where one stream is fully suppressed (caused ETTh1-720 catastrophe)
    - Gate biases initialized to 0 → sigmoid(0)=0.5 → starts at equal
      weighting, matching xPatch's balanced combination
    - Gate weights initialized with small std for stability
    
    Architecture unchanged from v5:
    1. Pre-pointwise inter-patch gating (alpha=0.05)
    2. Adaptive stream fusion with constrained gate
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(GLPatchNetwork, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        # ================================================================
        # Non-linear Stream (Seasonality)
        # ================================================================

        # Patching (overlapping, from xPatch)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding (from xPatch)
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise (from xPatch)
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream (from xPatch)
        self.fc2 = nn.Linear(self.dim, patch_len)

        # [GLCN] Pre-pointwise gating with conservative alpha
        self.inter_patch_gate = InterPatchGating(self.patch_num, reduction=4)
        self.res_alpha = nn.Parameter(torch.tensor(0.05))

        # CNN Pointwise (from xPatch)
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head (from xPatch)
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # ================================================================
        # Linear Stream (Trend) — identical to xPatch
        # ================================================================
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # ================================================================
        # Constrained Adaptive Stream Fusion (v6)
        # ================================================================
        # Gate projects from both streams to produce per-timestep weights
        self.gate_s = nn.Linear(pred_len, pred_len)
        self.gate_t = nn.Linear(pred_len, pred_len)

        # Initialize for stability:
        # - Small weights so gate inputs start near 0
        # - Zero bias so sigmoid(0)=0.5 → equal weighting at init
        nn.init.normal_(self.gate_s.weight, std=0.01)
        nn.init.normal_(self.gate_t.weight, std=0.01)
        nn.init.zeros_(self.gate_s.bias)
        nn.init.zeros_(self.gate_t.bias)

        # Final projection
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: seasonality [Batch, Input, Channel]
        # t: trend       [Batch, Input, Channel]

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]

        # Channel split for channel independence
        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        s = torch.reshape(s, (B * C, I))  # [B*C, I]
        t = torch.reshape(t, (B * C, I))  # [B*C, I]

        # ---- Non-linear Stream ----

        # Patching (overlapping, from xPatch)
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [B*C, patch_num, patch_len]

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # [GLCN] Pre-pointwise gating with learnable blend
        s_base = s
        s_gated = self.inter_patch_gate(s)
        s = s_base + self.res_alpha * (s_gated - s_base)

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)
        # s: [B*C, pred_len]

        # ---- Linear Stream (identical to xPatch) ----

        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)
        # t: [B*C, pred_len]

        # ---- Constrained Adaptive Stream Fusion ----
        # Compute input-dependent gate, then clamp to [0.2, 0.8]
        # to prevent full suppression of either stream
        gate = torch.sigmoid(self.gate_s(s) + self.gate_t(t))  # [B*C, pred_len]
        gate = gate * 0.6 + 0.2  # remap from [0,1] to [0.2, 0.8]

        # Dynamically blend: gate→0.8 favors seasonality, gate→0.2 favors trend
        x = gate * s + (1 - gate) * t

        # Refine the combined output
        x = self.fc8(x)

        # Channel concatenation
        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x
