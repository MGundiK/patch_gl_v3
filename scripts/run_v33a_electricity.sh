#!/bin/bash
# run_v33a_electricity.sh
# v3.3a Electricity (C=321) — 4 pred_lens
# Logs → logs/glpatch_hydra_v33a/Electricity/

set -e
mkdir -p logs/glpatch_hydra_v33a/Electricity

for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id electricity_${pred_len}_v33a \
  --model GLPatch_Hydra \
  --data custom \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 321 \
  --patch_len 16 \
  --stride 8 \
  --padding_patch end \
  --ma_type ema \
  --alpha 0.3 \
  --revin 1 \
  --cv_rank 32 \
  --gate_type hybrid \
  --gate_init -5.0 \
  --gate_temp 1.0 \
  --batch_size 256 \
  --learning_rate 0.005 \
  --lradj sigmoid \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  2>&1 | tee logs/glpatch_hydra_v33a/Electricity/electricity_${pred_len}.log
done

echo "=== v3.3a Electricity complete ==="
