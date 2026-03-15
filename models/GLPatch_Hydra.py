"""
GLPatch_Hydra v3: Universal architecture with adaptive gating.

Hydra is always on. The gate learns whether to use it.
No more cv_mixing='none' vs 'hydra_gated' decision.

Args:
  --cv_rank:      Bottleneck rank (default 32)
  --gate_type:    'scalar', 'vector', 'adaptive' (default 'adaptive')
  --gate_init:    Initial sigmoid bias (default -5.0)
"""

import torch
import torch.nn as nn

from layers.decomp import DECOMP
from layers.network_glpatch_hydra import GLPatchHydraNetwork
from layers.revin import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        self.ma_type = configs.ma_type
        alpha = configs.alpha
        beta = configs.beta

        self.decomp = DECOMP(self.ma_type, alpha, beta)

        # Hydra config (universal — no on/off switch)
        cv_rank = getattr(configs, 'cv_rank', 32)
        gate_type = getattr(configs, 'gate_type', 'adaptive')
        gate_init = getattr(configs, 'gate_init', -5.0)

        self.net = GLPatchHydraNetwork(
            seq_len, pred_len, patch_len, stride, padding_patch,
            cv_rank=cv_rank,
            gate_type=gate_type,
            gate_init=gate_init,
            n_channels=c_in,
        )

    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
