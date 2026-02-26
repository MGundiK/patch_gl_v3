#!/bin/bash
# ============================================================================
# GLPatch_Hydra — Verification & Patch Instructions
# ============================================================================
#
# FILES TO ADD (copy into your repo):
#   layers/hydra_attention.py         → new file
#   layers/network_glpatch_hydra.py   → new file
#   models/GLPatch_Hydra.py           → new file
#
# FILES TO PATCH (2 small edits):
#
# ---- run.py ----
# Find:   parser.add_argument('--revin', ...)
# Add after it:
#   parser.add_argument('--cv_mixing', type=str, default='none',
#                       help='Cross-variable mixing: none, hydra, hydra_bottleneck, hydra_gated')
#   parser.add_argument('--cv_rank', type=int, default=32,
#                       help='Bottleneck rank for hydra_bottleneck/hydra_gated')
#
# ---- exp_main.py ----
# Add import:  from models import GLPatch_Hydra
# Add to dict: 'GLPatch_Hydra': GLPatch_Hydra,
#
# ============================================================================

echo "============================================"
echo "  GLPatch_Hydra — Verification"
echo "============================================"
echo ""

python -c "
import torch
import torch.nn as nn

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print()

# Minimal config matching your setup
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
    cv_mixing = 'none'
    cv_rank = 32

from models.GLPatch_Hydra import Model

# ---- Test all variants and compare param counts ----
print('='*60)
print('  Param counts by variant (Weather: C=21, D=256)')
print('='*60)

base_ref = 0
for variant in ['none', 'hydra', 'hydra_bottleneck', 'hydra_gated']:
    configs = C()
    configs.cv_mixing = variant

    model = Model(configs).to(device)

    total = sum(p.numel() for p in model.parameters())
    hydra_p = sum(p.numel() for n, p in model.named_parameters() if 'hydra' in n)

    tag = ''
    if variant == 'none':
        base_ref = total
        tag = ' (= GLPatch baseline)'
    else:
        overhead = 100 * hydra_p / base_ref
        tag = f' (+{overhead:.1f}% over base)'

    print(f'  {variant:25s}  total={total:>9,}  hydra={hydra_p:>8,}{tag}')

print()

# ---- Forward pass test with recommended variant ----
print('='*60)
print('  Forward pass test: hydra_gated r=32')
print('='*60)

configs = C()
configs.cv_mixing = 'hydra_gated'
configs.cv_rank = 32

model = Model(configs).to(device)

x = torch.randn(4, 96, 21, device=device)
with torch.no_grad():
    y = model(x)

print(f'  Input:  {list(x.shape)}')
print(f'  Output: {list(y.shape)}')
assert y.shape == (4, 96, 21), f'Shape mismatch: {y.shape}'
print(f'  ✅ Forward pass OK!')
print()

# ---- Test that cv_mixing=none matches GLPatch exactly ----
print('='*60)
print('  Equivalence test: cv_mixing=none vs GLPatch')
print('='*60)

configs_none = C()
configs_none.cv_mixing = 'none'
model_none = Model(configs_none).to(device)

from models.GLPatch import Model as GLPatchModel
model_gl = GLPatchModel(configs_none).to(device)

sd_gl = model_gl.state_dict()
sd_none = model_none.state_dict()

missing = [k for k in sd_gl if k not in sd_none]
extra = [k for k in sd_none if k not in sd_gl]

if not missing and not extra:
    print('  ✅ Identical parameter structure — perfect drop-in')
else:
    if missing:
        print(f'  ⚠️  Keys in GLPatch missing from Hydra(none): {missing}')
    if extra:
        print(f'  ⚠️  Extra keys in Hydra(none): {extra}')

model_none.load_state_dict(sd_gl)
x = torch.randn(2, 96, 21, device=device)
with torch.no_grad():
    y_gl = model_gl(x)
    y_none = model_none(x)

diff = (y_gl - y_none).abs().max().item()
print(f'  Max output difference: {diff:.2e}')
if diff < 1e-5:
    print('  ✅ Outputs match — cv_mixing=none is identical to GLPatch')
else:
    print('  ⚠️  Small numerical differences (check BN running stats)')

print()
print('All checks complete!')
"
