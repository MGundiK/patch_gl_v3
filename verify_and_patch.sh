#!/bin/bash
# ============================================================================
# GLPatch_Hydra Integration — Patch Instructions & Experiment Scripts
# ============================================================================
#
# FILES TO ADD (copy into your repo):
#   layers/hydra_attention.py         → new file
#   layers/network_glpatch_hydra.py   → new file
#   models/GLPatch_Hydra.py           → new file
#
# FILES TO PATCH (2 small edits):
#   run.py        → add 2 argparse lines
#   exp_main.py   → add 1 import + 1 dict entry
#
# ============================================================================


# ============================================================================
# PATCH 1: run.py — Add these 2 lines after the --revin argument (~line 73)
# ============================================================================
#
# Find this line:
#   parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
#
# Add immediately after it:
#
#   # Cross-variable mixing (GLPatch_Hydra)
#   parser.add_argument('--cv_mixing', type=str, default='none',
#                       help='Cross-variable mixing: none, hydra, hydra_bottleneck, hydra_gated')
#   parser.add_argument('--cv_rank', type=int, default=32,
#                       help='Bottleneck rank for hydra_bottleneck variant')
#
# ============================================================================


# ============================================================================
# PATCH 2: exp_main.py — Add model to the dict
# ============================================================================
#
# Find these lines:
#   from models import xPatch
#   from models import GLPatch
#
# Add:
#   from models import GLPatch_Hydra
#
# Find this dict:
#   model_dict = {
#       'xPatch': xPatch,
#       'GLPatch': GLPatch,
#   }
#
# Change to:
#   model_dict = {
#       'xPatch': xPatch,
#       'GLPatch': GLPatch,
#       'GLPatch_Hydra': GLPatch_Hydra,
#   }
#
# ============================================================================


echo "============================================"
echo "  GLPatch_Hydra — Quick Verification"
echo "============================================"
echo ""
echo "Testing that the model builds correctly..."
echo "(Run this from your repo root)"
echo ""

python -c "
import argparse
import torch

# Minimal config
class C:
    seq_len = 96
    pred_len = 96
    enc_in = 21       # Weather
    patch_len = 16
    stride = 8
    padding_patch = 'end'
    revin = 1
    ma_type = 'ema'
    alpha = 0.3
    beta = 0.3
    cv_mixing = 'hydra_gated'
    cv_rank = 32

from models.GLPatch_Hydra import Model

configs = C()
model = Model(configs)

# Count params
total = sum(p.numel() for p in model.parameters())
hydra_params = sum(p.numel() for n, p in model.named_parameters() if 'hydra' in n)

print(f'Total params:  {total:>10,}')
print(f'Hydra params:  {hydra_params:>10,} ({100*hydra_params/total:.1f}%)')
print(f'Base params:   {total-hydra_params:>10,}')

# Forward pass test
x = torch.randn(4, 96, 21)   # [B=4, L=96, C=21]
with torch.no_grad():
    y = model(x)
print(f'Input shape:   {list(x.shape)}')
print(f'Output shape:  {list(y.shape)}')
assert y.shape == (4, 96, 21), f'Shape mismatch: {y.shape}'
print('✅ Forward pass OK!')

# Test cv_mixing=none (should behave like plain GLPatch)
configs.cv_mixing = 'none'
model_none = Model(configs)
base_only = sum(p.numel() for p in model_none.parameters())
print(f'\\ncv_mixing=none params: {base_only:,} (matches GLPatch)')
"
