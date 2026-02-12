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
    GLPatch v4 dual-stream network.
    
    Key insight from v2 vs v3: gating BEFORE pointwise conv helped ETTh2
    long horizons; gating AFTER pointwise conv helped ETTm1 and Exchange.
    The optimal position is dataset-dependent.
    
    v4 solution: dual gating at BOTH positions, each with its own learnable
    alpha. The model discovers which position matters per dataset.
    
    v4 changes vs v3:
    1. DUAL GATING — gate_pre (before pointwise) + gate_post (after pointwise)
    2. INDEPENDENT alphas — alpha_pre and alpha_post, both init 0.1
    3. Minimal param increase — two small MLPs (patch_num → patch_num/4 → patch_num)
       plus two scalars. Still very close to xPatch in size.
    
    Architecture (seasonality stream):
        Patching → Embedding → Depthwise+Residual →
        [Pre-gating with alpha_pre] → Pointwise →
        [Post-gating with alpha_post] → Flatten Head
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

        # [GLCN] Pre-pointwise gating — gates on per-patch features
        # Helps ETTh2 long horizons (v2 evidence)
        self.gate_pre = InterPatchGating(self.patch_num, reduction=4)
        self.alpha_pre = nn.Parameter(torch.tensor(0.1))

        # CNN Pointwise (from xPatch)
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # [GLCN] Post-pointwise gating — gates on aggregated features
        # Helps ETTm1, Exchange (v3 evidence)
        self.gate_post = InterPatchGating(self.patch_num, reduction=4)
        self.alpha_post = nn.Parameter(torch.tensor(0.1))

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
        # s: [B*C, patch_num, patch_len]

        # [GLCN] Pre-pointwise gating with learnable blend
        s_pre = s
        s_gated_pre = self.gate_pre(s)
        s = s_pre + self.alpha_pre * (s_gated_pre - s_pre)

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)
        # s: [B*C, patch_num, patch_len]

        # [GLCN] Post-pointwise gating with learnable blend
        s_post = s
        s_gated_post = self.gate_post(s)
        s = s_post + self.alpha_post * (s_gated_post - s_post)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # ---- Linear Stream (identical to xPatch) ----

        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # ---- Streams Concatenation ----
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # Channel concatenation
        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x
