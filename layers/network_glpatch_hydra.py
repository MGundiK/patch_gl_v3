"""
GLPatch v3.2 network — Hydra with Hybrid Gate (channel-primary + data bonus).
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
    GLPatch v8 dual-stream + always-on Hydra with adaptive gating.

    The model decides FOR ITSELF how much cross-variable mixing to use.
    On ETTm1 (C=7), the gate learns to stay near zero.
    On Electricity (C=321), the gate opens to mix aggressively.
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 cv_rank=32, gate_type='adaptive', gate_init=-5.0,
                 gate_temp=1.0, n_channels=7):
        super(GLPatchHydraNetwork, self).__init__()

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

        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        self.fc2 = nn.Linear(self.dim, patch_len)

        self.inter_patch_gate = InterPatchGating(self.patch_num, reduction=4)
        self.res_alpha = nn.Parameter(torch.tensor(0.05))

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

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
        # [HYDRA v3] Always-on with adaptive gating — post-fusion
        # ================================================================
        # Effective rank: min(cv_rank, pred_len)
        effective_rank = min(cv_rank, pred_len)

        self.hydra = HydraChannelMixer(
            d_model=pred_len,
            variant='hydra_gated',
            rank=effective_rank,
            n_channels=n_channels,
            dropout=0.0,
            gate_type=gate_type,
            gate_init=gate_init,
            gate_temp=gate_temp,
        )

        temp_str = f", τ={gate_temp}" if gate_temp != 1.0 else ""
        print(f"[GLPatch_Hydra v3.2] ALWAYS-ON hydra_gated @ post_fusion, "
              f"d_model={pred_len}, rank={effective_rank}, C={n_channels}, "
              f"gate={gate_type} (init={gate_init}{temp_str}), logC={__import__('math').log(max(n_channels,2))/__import__('math').log(1000):.3f}")

    def forward(self, s, t):
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        s = torch.reshape(s, (B * C, I))
        t = torch.reshape(t, (B * C, I))

        # ---- Non-linear Stream ----

        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        res = self.fc2(res)
        s = s + res

        s_base = s
        s_gated = self.inter_patch_gate(s)
        s = s_base + self.res_alpha * (s_gated - s_base)

        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # ---- Linear Stream ----

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
        gate = gate * 0.8 + 0.1

        x = gate * s + (1 - gate) * t

        # ---- [HYDRA v3] Always-on post-fusion mixing ----
        # Reshape to [B*C, 1, pred_len] for HydraChannelMixer interface
        x_3d = x.unsqueeze(1)  # [B*C, 1, pred_len]
        x_3d = self.hydra(x_3d, B, C)
        x = x_3d.squeeze(1)    # [B*C, pred_len]

        x = self.fc8(x)

        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x
