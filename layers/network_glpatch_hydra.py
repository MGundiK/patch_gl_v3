"""
GLPatch v8 network WITH Hydra cross-variable mixing — v2.

v2 changes from v1:
  - Multiple placement options via cv_placement argument
  - 'post_embed':   After patch embedding (v1 location — too early, kept for reference)
  - 'post_pw':      After pointwise conv — RECOMMENDED, matches AdaPatch's successful ppw
  - 'post_stream':  After seasonal flatten head, before fusion
  - 'post_fusion':  After stream fusion, before final projection
  - Auto-sets d_model based on placement (no user error possible)
  - Auto-clips rank to not exceed feature dim
"""

import torch
from torch import nn

from layers.hydra_attention import HydraChannelMixer


class InterPatchGating(nn.Module):
    """GLCN-inspired inter-patch gating. Unchanged from GLPatch v8."""
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
    GLPatch v8 dual-stream + Hydra cross-variable mixing (v2).

    Args (new vs GLPatchNetwork):
        cv_mixing:    'none' | 'hydra' | 'hydra_bottleneck' | 'hydra_gated'
        cv_rank:      Bottleneck rank. Auto-clipped to not exceed feature dim.
        cv_placement: Where to insert cross-variable mixing:
                      'post_embed'  — after patch embedding (dim=256, v1 default)
                      'post_pw'     — after pointwise conv (dim=16) ← RECOMMENDED
                      'post_stream' — after seasonal head (dim=pred_len)
                      'post_fusion' — after stream fusion (dim=pred_len)
        n_channels:   Number of input variables (enc_in).
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 cv_mixing='none', cv_rank=32, cv_placement='post_pw',
                 n_channels=7):
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

        # ================================================================
        # [HYDRA v2] Cross-Variable Mixing — placement-aware
        # ================================================================
        self.cv_mixing = cv_mixing
        self.cv_placement = cv_placement
        # is_3d: True for placements where tensor is [B*C, P, D]
        #        False for placements where tensor is [B*C, pred_len] (1D per channel)
        self.hydra_is_3d = cv_placement in ('post_embed', 'post_pw')

        if cv_mixing != 'none':
            # Determine feature dimension based on placement
            if cv_placement == 'post_embed':
                hydra_dim = self.dim           # patch_len² = 256
            elif cv_placement == 'post_pw':
                hydra_dim = patch_len          # 16
            elif cv_placement in ('post_stream', 'post_fusion'):
                hydra_dim = pred_len           # 96/192/336/720
            else:
                raise ValueError(f"Unknown cv_placement: {cv_placement}")

            # Auto-clip rank: can't exceed feature dim, use half at most
            #effective_rank = min(cv_rank, max(hydra_dim // 2, 4))
            # removed clipping
            effective_rank = min(cv_rank, hydra_dim)


            self.hydra = HydraChannelMixer(
                d_model=hydra_dim,
                variant=cv_mixing,
                rank=effective_rank,
                dropout=0.0,
                residual_scale=0.1,
            )

            print(f"[GLPatch_Hydra v2] {cv_mixing} @ {cv_placement}, "
                  f"d_model={hydra_dim}, rank={effective_rank}, C={n_channels}")
        else:
            self.hydra = None

    def _apply_hydra_3d(self, x, B, C):
        """Apply Hydra on 3D tensor [B*C, P, D] — for post_embed and post_pw."""
        return self.hydra(x, B, C)

    def _apply_hydra_1d(self, x, B, C):
        """Apply Hydra on 1D tensor [B*C, pred_len] — for post_stream/post_fusion.
        Reshapes to [B, C, pred_len] so Hydra treats each channel as a token
        with pred_len features, then reshapes back."""
        # [B*C, pred_len] → [B, C, 1, pred_len] — fake patch dim of 1
        h = x.reshape(B, C, 1, -1)
        # Flatten back for the HydraChannelMixer interface: [B*C, 1, pred_len]
        h_flat = h.reshape(B * C, 1, -1)
        h_out = self.hydra(h_flat, B, C)
        return h_out.reshape(B * C, -1)

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
        s = self.fc1(s)    # [B*C, patch_num, dim=256]
        s = self.gelu1(s)
        s = self.bn1(s)

        # [HYDRA] post_embed placement (v1 — kept for comparison)
        if self.hydra is not None and self.cv_placement == 'post_embed':
            s = self._apply_hydra_3d(s, B, C)

        res = s

        # CNN Depthwise
        s = self.conv1(s)   # [B*C, patch_num, patch_len=16]
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
        s = self.conv2(s)   # [B*C, patch_num, patch_len=16]
        s = self.gelu3(s)
        s = self.bn3(s)

        # [HYDRA] post_pw placement — RECOMMENDED
        if self.hydra is not None and self.cv_placement == 'post_pw':
            s = self._apply_hydra_3d(s, B, C)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)    # [B*C, pred_len]

        # [HYDRA] post_stream placement
        if self.hydra is not None and self.cv_placement == 'post_stream':
            s = self._apply_hydra_1d(s, B, C)

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

        # [HYDRA] post_fusion placement
        if self.hydra is not None and self.cv_placement == 'post_fusion':
            x = self._apply_hydra_1d(x, B, C)

        x = self.fc8(x)

        # Channel concatenation
        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x
