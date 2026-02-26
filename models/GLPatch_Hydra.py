"""
GLPatch_Hydra: GLPatch with Hydra cross-variable mixing (v2).

Drop-in replacement for GLPatch. New args:
  --cv_mixing:    none | hydra | hydra_bottleneck | hydra_gated
  --cv_rank:      Bottleneck rank (default 32, auto-clipped per placement)
  --cv_placement: post_embed | post_pw | post_stream | post_fusion

Recommended: --cv_mixing hydra_gated --cv_placement post_pw
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

        # Hydra config (safe defaults — 'none' = plain GLPatch)
        cv_mixing = getattr(configs, 'cv_mixing', 'none')
        cv_rank = getattr(configs, 'cv_rank', 32)
        cv_placement = getattr(configs, 'cv_placement', 'post_pw')

        self.net = GLPatchHydraNetwork(
            seq_len, pred_len, patch_len, stride, padding_patch,
            cv_mixing=cv_mixing,
            cv_rank=cv_rank,
            cv_placement=cv_placement,
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
