"""
GLPatch v8 network WITH Hydra cross-variable mixing.

Identical to network_glpatch.py except:
  1. __init__ accepts cv_mixing, cv_rank, n_channels args
  2. Builds a HydraChannelMixer (if cv_mixing != 'none')
  3. forward() inserts one reshape → Hydra → reshape after patch embedding

All line numbers and logic outside the marked [HYDRA] blocks are unchanged.
"""

import torch
from torch import nn

from layers.hydra_attention import HydraChannelMixer


class InterPatchGating(nn.Module):
    """
    GLCN-inspired inter-patch gating module.
    Unchanged from network_glpatch.py.
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
        w = x.mean(dim=2)
        w = self.mlp(w)
        w = w.unsqueeze(2)
        return x * w


class GLPatchHydraNetwork(nn.Module):
    """
    GLPatch v8 dual-stream network + Hydra cross-variable mixing.
    
    New args vs GLPatchNetwork:
        cv_mixing:  'none' | 'hydra' | 'hydra_bottleneck' | 'hydra_gated'
        cv_rank:    Bottleneck rank (used by 'hydra_bottleneck'). Default 32.
        n_channels: Number of input variables (enc_in). Needed for shape checks.
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 cv_mixing='none', cv_rank=32, n_channels=7):
        super(GLPatchHydraNetwork, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        # ================================================================
        # Non-linear Stream (Seasonality) — identical to GLPatch v8
        # ================================================================

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # ================================================================
        # [HYDRA] Cross-Variable Mixing — inserted after patch embedding
        # ================================================================
        self.cv_mixing = cv_mixing
        if cv_mixing != 'none':
            self.hydra = HydraChannelMixer(
                d_model=self.dim,           # patch_len² = 256 for patch_len=16
                variant=cv_mixing,
                rank=cv_rank,
                dropout=0.0,
                residual_scale=0.1,
            )
            print(f"[GLPatch_Hydra] CV mixing enabled: {cv_mixing}, "
                  f"rank={cv_rank}, d_model={self.dim}, "
                  f"n_channels={n_channels}")
        else:
            self.hydra = None
        # ================================================================

        # CNN Depthwise (from xPatch)
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream (from xPatch)
        self.fc2 = nn.Linear(self.dim, patch_len)

        # [GLCN] Pre-pointwise gating
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
        # Linear Stream (Trend) — identical to xPatch / GLPatch v8
        # ================================================================
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # ================================================================
        # Bottleneck Adaptive Stream Fusion (v8) — unchanged
        # ================================================================
        gate_hidden = min(32, pred_len)

        self.gate_compress_s = nn.Linear(pred_len, gate_hidden)
        self.gate_compress_t = nn.Linear(pred_len, gate_hidden)
        self.gate_expand = nn.Linear(gate_hidden, pred_len)

        nn.init.normal_(self.gate_compress_s.weight, std=0.01)
        nn.init.normal_(self.gate_compress_t.weight, std=0.01)
        nn.init.normal_(self.gate_expand.weight, std=0.01)
        nn.init.zeros_(self.gate_compress_s.bias)
        nn.init.zeros_(self.gate_compress_t.bias)
        nn.init.zeros_(self.gate_expand.bias)

        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: seasonality [Batch, Input, Channel]
        # t: trend       [Batch, Input, Channel]

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        s = torch.reshape(s, (B * C, I))  # [B*C, I]
        t = torch.reshape(t, (B * C, I))

        # ---- Non-linear Stream ----

        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [B*C, patch_num, patch_len]

        # Patch Embedding
        s = self.fc1(s)    # [B*C, patch_num, dim]
        s = self.gelu1(s)
        s = self.bn1(s)
        # s: [B*C, patch_num, dim]  (e.g. [B*C, 12, 256])

        # ==============================================================
        # [HYDRA] Cross-variable mixing on patch embeddings
        # Temporarily un-merges B*C → B, C for attention across channels,
        # then re-merges. Only active when cv_mixing != 'none'.
        # ==============================================================
        if self.hydra is not None:
            s = self.hydra(s, B, C)
        # ==============================================================

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # [GLCN] Pre-pointwise gating
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

        # ---- Linear Stream (identical to xPatch) ----

        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # ---- Bottleneck Adaptive Stream Fusion ----
        gate = torch.sigmoid(
            self.gate_expand(
                self.gate_compress_s(s) + self.gate_compress_t(t)
            )
        )
        gate = gate * 0.8 + 0.1  # constrain to [0.1, 0.9]

        x = gate * s + (1 - gate) * t
        x = self.fc8(x)

        # Channel concatenation
        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x
