#!/bin/bash
# ============================================================================
# GLPatch_Hydra v2 — Verification
# ============================================================================
# PATCH run.py — add after --revin line:
#
#   parser.add_argument('--cv_mixing', type=str, default='none',
#       help='none, hydra, hydra_bottleneck, hydra_gated')
#   parser.add_argument('--cv_rank', type=int, default=32,
#       help='Bottleneck rank (auto-clipped per placement)')
#   parser.add_argument('--cv_placement', type=str, default='post_pw',
#       help='post_embed, post_pw, post_stream, post_fusion')
# ============================================================================

echo "============================================"
echo "  GLPatch_Hydra v2 — Verification"
echo "============================================"

python -c "
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

class C:
    seq_len = 96; pred_len = 96; enc_in = 21
    patch_len = 16; stride = 8; padding_patch = 'end'
    revin = 1; ma_type = 'ema'; alpha = 0.3; beta = 0.3
    cv_mixing = 'none'; cv_rank = 32; cv_placement = 'post_pw'

from models.GLPatch_Hydra import Model

# ---- Param counts for all placements ----
print('='*70)
print(f'  Param counts: hydra_gated r=32, Weather C=21, pred_len=96')
print('='*70)
fmt = '  {:<20s}  dim={:<4d}  eff_rank={:<3d}  total={:>9,}  hydra={:>7,}  +{:.1f}%'

configs = C()
configs.cv_mixing = 'none'
base_model = Model(configs).to(device)
base_params = sum(p.numel() for p in base_model.parameters())
print(f'  {\"none (GLPatch)\":<20s}  {\"\":>30s}  total={base_params:>9,}')

for place in ['post_embed', 'post_pw', 'post_stream', 'post_fusion']:
    configs = C()
    configs.cv_mixing = 'hydra_gated'
    configs.cv_rank = 32
    configs.cv_placement = place
    model = Model(configs).to(device)
    total = sum(p.numel() for p in model.parameters())
    hydra_p = sum(p.numel() for n, p in model.named_parameters() if 'hydra' in n)
    
    # Figure out effective dim and rank
    if place == 'post_embed': dim = 256
    elif place == 'post_pw': dim = 16
    else: dim = 96  # pred_len
    eff_rank = min(32, max(dim // 2, 4))
    
    pct = 100 * hydra_p / base_params
    print(fmt.format(place, dim, eff_rank, total, hydra_p, pct))

print()

# ---- Forward pass for each placement ----
print('='*70)
print('  Forward pass tests')
print('='*70)

x = torch.randn(4, 96, 21, device=device)

for place in ['post_pw', 'post_stream', 'post_fusion']:
    configs = C()
    configs.cv_mixing = 'hydra_gated'
    configs.cv_placement = place
    model = Model(configs).to(device)
    with torch.no_grad():
        y = model(x)
    ok = '✅' if y.shape == (4, 96, 21) else '❌'
    print(f'  {place:20s}  {list(x.shape)} → {list(y.shape)}  {ok}')

print()
print('All checks passed!')
"
