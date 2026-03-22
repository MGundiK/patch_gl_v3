#!/bin/bash
# run_v32g_small.sh
# v3.2g small datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, ILI
# Flat dataset layout: ./dataset/{file}
# Hyperparameters from handoff table (corrected values)
#
# Requires: layers/hydra_attention.py = hydra_attention_v32g.py
# Logs → logs/glpatch_hydra_v32g/{dataset}/

set -e
mkdir -p logs/glpatch_hydra_v32g

# ── ETTh1 ──────────────────────────────────────────────────────────────────
mkdir -p logs/glpatch_hydra_v32g/ETTh1
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id ETTh1_${pred_len}_v32g \
  --model GLPatch_Hydra \
  --data ETTh1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
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
  --batch_size 2048 \
  --learning_rate 0.0005 \
  --lradj sigmoid \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  2>&1 | tee logs/glpatch_hydra_v32g/ETTh1/ETTh1_${pred_len}.log
done

# ── ETTh2 ──────────────────────────────────────────────────────────────────
mkdir -p logs/glpatch_hydra_v32g/ETTh2
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id ETTh2_${pred_len}_v32g \
  --model GLPatch_Hydra \
  --data ETTh2 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
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
  --batch_size 2048 \
  --learning_rate 0.0005 \
  --lradj sigmoid \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  2>&1 | tee logs/glpatch_hydra_v32g/ETTh2/ETTh2_${pred_len}.log
done

# ── ETTm1 ──────────────────────────────────────────────────────────────────
mkdir -p logs/glpatch_hydra_v32g/ETTm1
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id ETTm1_${pred_len}_v32g \
  --model GLPatch_Hydra \
  --data ETTm1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
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
  --batch_size 2048 \
  --learning_rate 0.0005 \
  --lradj sigmoid \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  2>&1 | tee logs/glpatch_hydra_v32g/ETTm1/ETTm1_${pred_len}.log
done

# ── ETTm2 ──────────────────────────────────────────────────────────────────
mkdir -p logs/glpatch_hydra_v32g/ETTm2
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id ETTm2_${pred_len}_v32g \
  --model GLPatch_Hydra \
  --data ETTm2 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
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
  --batch_size 2048 \
  --learning_rate 0.0001 \
  --lradj sigmoid \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  2>&1 | tee logs/glpatch_hydra_v32g/ETTm2/ETTm2_${pred_len}.log
done

# ── Weather ────────────────────────────────────────────────────────────────
mkdir -p logs/glpatch_hydra_v32g/Weather
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id Weather_${pred_len}_v32g \
  --model GLPatch_Hydra \
  --data custom \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 21 \
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
  --batch_size 2048 \
  --learning_rate 0.0005 \
  --lradj sigmoid \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  2>&1 | tee logs/glpatch_hydra_v32g/Weather/Weather_${pred_len}.log
done

# ── Exchange ───────────────────────────────────────────────────────────────
mkdir -p logs/glpatch_hydra_v32g/Exchange
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id Exchange_${pred_len}_v32g \
  --model GLPatch_Hydra \
  --data custom \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 8 \
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
  --batch_size 32 \
  --learning_rate 0.00001 \
  --lradj sigmoid \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  2>&1 | tee logs/glpatch_hydra_v32g/Exchange/Exchange_${pred_len}.log
done

# ── ILI ────────────────────────────────────────────────────────────────────
mkdir -p logs/glpatch_hydra_v32g/ILI
for pred_len in 24 36 48 60; do
python -u run.py \
  --is_training 1 \
  --model_id ILI_${pred_len}_v32g \
  --model GLPatch_Hydra \
  --data custom \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len $pred_len \
  --enc_in 7 \
  --patch_len 6 \
  --stride 3 \
  --padding_patch end \
  --ma_type ema \
  --alpha 0.3 \
  --revin 1 \
  --cv_rank 32 \
  --gate_type hybrid \
  --gate_init -5.0 \
  --gate_temp 1.0 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  2>&1 | tee logs/glpatch_hydra_v32g/ILI/ILI_${pred_len}.log
done

echo "=== v3.2g small datasets complete ==="
