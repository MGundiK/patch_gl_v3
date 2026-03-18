"""
GLPatch_Hydra v3.2: Hybrid gate (channel-primary + learned data bonus).

Args:
  --cv_rank:      Bottleneck rank (default 32)
  --gate_type:    'hybrid', 'channel', 'adaptive', 'scalar' (default 'hybrid')
  --gate_init:    Initial sigmoid bias (default -5.0)
  --gate_temp:    Sigmoid temperature τ — higher = sharper gate (default 1.0)
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
        gate_type = getattr(configs, 'gate_type', 'hybrid')
        gate_init = getattr(configs, 'gate_init', -5.0)
        gate_temp = getattr(configs, 'gate_temp', 1.0)

        self.net = GLPatchHydraNetwork(
            seq_len, pred_len, patch_len, stride, padding_patch,
            cv_rank=cv_rank,
            gate_type=gate_type,
            gate_init=gate_init,
            gate_temp=gate_temp,
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
