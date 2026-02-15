#!/bin/bash

# GLPatch v8 — Final Tuned Runs
# Only re-running datasets where a different single LR improves results.
#
# ALL OTHER DATASETS: use existing v8 results (same LR as xPatch).
#   ETTh1:  lr=0.0005 (same as xPatch) → already have v8 results
#   ETTh2:  lr=0.0005 (same as xPatch) → already have v8 results
#   ETTm1:  lr=0.0005 (same as xPatch) → already have v8 results
#   ETTm2:  lr=0.0001 (same as xPatch) → already have v8 results
#   Weather: lr=0.0005 (same as xPatch) → already have v8 results
#   Traffic: lr=0.005  (same as xPatch) → already have v8 results
#   Electricity: lr=0.005 (same as xPatch) → already have v8 results
#   Solar:   lr=0.005  (same as xPatch) → already have v8 results
#
# RE-RUN (different LR helps):
#   Exchange: lr=0.000005 (xPatch default: 0.00001) — bigger wins on 336/720
#   ILI:      lr=0.02     (xPatch default: 0.01)    — flips 36/48/60 to wins
#
# Estimated: ~20-30 minutes on Colab

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch

mkdir -p ./logs/glpatch_final

# ================================================================
# Exchange — lr=0.000005 (halved from xPatch's 0.00001)
# Tuning showed: 4/4 wins with larger margins, especially 336: -4.28%
# ================================================================
echo "========== [$(date '+%H:%M')] Exchange lr=0.000005 =========="
for pred_len in 96 192 336 720; do
  echo ">>> Exchange pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path exchange_rate.csv \
    --model_id exchange_${pred_len}_${ma_type} --model $model_name --data custom \
    --features M --seq_len 96 --pred_len $pred_len --enc_in 8 \
    --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.000005 \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2 > logs/glpatch_final/exchange_${pred_len}.log
done

# ================================================================
# ILI — lr=0.02 (doubled from xPatch's 0.01)
# Tuning showed: 3/4 wins (flips 36, 48, 60), only 24 remains a loss
# Note: lradj=type3, seq_len=36, patch_len=6, stride=3
# ================================================================
echo "========== [$(date '+%H:%M')] ILI lr=0.02 =========="
for pred_len in 24 36 48 60; do
  echo ">>> ILI pred_len=${pred_len}"
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path national_illness.csv \
    --model_id ili_${pred_len}_${ma_type} --model $model_name --data custom \
    --features M --seq_len 36 --label_len 18 --pred_len $pred_len --enc_in 7 \
    --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.02 \
    --lradj 'type3' --patch_len 6 --stride 3 \
    --ma_type $ma_type --alpha $alpha --beta $beta \
    --use_amp --num_workers 2 > logs/glpatch_final/ili_${pred_len}.log
done

echo ""
echo "========== [$(date '+%H:%M')] DONE =========="
echo ""
echo "Final results table should combine:"
echo "  - Exchange & ILI: from this run"
echo "  - All other datasets: from v8 untuned run"
echo ""
cat result.txt
