import torch
from torch import nn


class InterPatchGating(nn.Module):
    """
    GLCN-inspired inter-patch gating module.
    
    Captures global dynamics by learning patch importance weights:
    GlobalAvgPool (over features) → MLP → Sigmoid → element-wise scaling.
    
    This explicitly models which temporal patches carry more predictive
    information — something xPatch's pointwise conv only does implicitly.
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


class MultiscaleLocalConv(nn.Module):
    """
    GLCN-inspired multiscale local feature extraction.
    
    Parallel depthwise convolutions with kernels {1, 3, 5, 7} capture local
    temporal patterns at multiple resolutions within each patch. This is richer
    than xPatch's single-scale depthwise conv.
    
    Each conv is grouped (depthwise) to maintain per-patch independence.
    Outputs are averaged (not summed) for magnitude stability.
    """
    def __init__(self, patch_num, kernels=(1, 3, 5, 7)):
        super(MultiscaleLocalConv, self).__init__()
        self.n_scales = len(kernels)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=patch_num,
                out_channels=patch_num,
                kernel_size=k,
                padding=k // 2,
                groups=patch_num
            )
            for k in kernels
        ])
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(patch_num)

    def forward(self, x):
        # x: [B*C, patch_num, patch_len]
        out = sum(conv(x) for conv in self.convs) / self.n_scales
        out = self.gelu(out)
        out = self.bn(out)
        return out


class GLPatchNetwork(nn.Module):
    """
    GLPatch dual-stream network combining xPatch and GLCN innovations.
    
    Seasonality (non-linear) stream:
        1. Aggregate Conv1D (GLCN)     — smooths patch boundary transitions
        2. Overlapping patching (xPatch) — richer temporal context
        3. Patch embedding (xPatch)     — project to higher-dim space
        4. Depthwise conv + residual (xPatch) — per-patch feature extraction
        5. Inter-patch gating (GLCN)    — global patch importance weighting
        6. Multiscale local conv (GLCN) — multi-resolution local features
        7. Pointwise conv (xPatch)      — inter-patch feature aggregation
        8. Flatten MLP head (xPatch)    — project to prediction length
    
    Trend (linear) stream:
        Identical to xPatch — pure linear MLP with AvgPool + LayerNorm.
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
        # Non-linear Stream (Seasonality) — enhanced with GLCN modules
        # ================================================================

        # [NEW] Aggregate Conv1D (GLCN): sliding aggregation preserves
        # edge information between adjacent patches. Initialized as
        # uniform moving average for stable training start.
        self.agg_conv = nn.Conv1d(1, 1, kernel_size=patch_len,
                                  padding='same', bias=False)
        nn.init.constant_(self.agg_conv.weight, 1.0 / patch_len)

        # Patching
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding (from xPatch)
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise (from xPatch): reduces dim from patch_len² → patch_len
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream (from xPatch)
        self.fc2 = nn.Linear(self.dim, patch_len)

        # [NEW] Inter-patch Gating (GLCN): learns global patch importance
        self.inter_patch_gate = InterPatchGating(self.patch_num, reduction=4)

        # [NEW] Multiscale Local Conv (GLCN): multi-resolution local features
        self.multiscale_conv = MultiscaleLocalConv(self.patch_num,
                                                   kernels=(1, 3, 5, 7))

        # CNN Pointwise (from xPatch): inter-patch feature aggregation
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
        # Pure linear MLP with AvgPool + LayerNorm, no activations.
        # Emphasizes linear features in the trend component.
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # ================================================================
        # Streams Concatenation
        # ================================================================
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

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

        # [NEW] Aggregate Conv1D: smooth patch boundary transitions
        s = s.unsqueeze(1)          # [B*C, 1, I]
        s = self.agg_conv(s)        # [B*C, 1, I]
        s = s.squeeze(1)            # [B*C, I]

        # Patching (overlapping, from xPatch)
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [B*C, patch_num, patch_len]

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)
        # s: [B*C, patch_num, dim]

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)
        # s: [B*C, patch_num, patch_len]

        # Residual Stream
        res = self.fc2(res)         # [B*C, patch_num, patch_len]
        s = s + res

        # [NEW] Inter-patch Gating: weight patches by global importance
        s_pre = s                   # save for residual around new modules
        s = self.inter_patch_gate(s)

        # [NEW] Multiscale Local Conv: multi-resolution feature extraction
        s = self.multiscale_conv(s)
        s = s + s_pre               # residual connection for gradient flow

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

        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)
        # t: [B*C, pred_len]

        # ---- Streams Concatenation ----
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # Channel concatenation
        x = torch.reshape(x, (B, C, self.pred_len))  # [B, C, Output]
        x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return x
