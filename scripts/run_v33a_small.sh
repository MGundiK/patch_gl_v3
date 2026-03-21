#!/bin/bash
# run_v33a_small.sh
# v3.3a small datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, ILI
# Logs → logs/glpatch_hydra_v33a/{dataset}/
# Hyperparameters from handoff Section 8 (corrected values)

set -e
mkdir -p logs/glpatch_hydra_v33a

for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id ETTh1_${pred_len}_v33a \
  --model GLPatch_Hydra \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
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
  2>&1 | tee logs/glpatch_hydra_v33a/ETTh1_${pred_len}.log
done

for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id ETTh2_${pred_len}_v33a \
  --model GLPatch_Hydra \
  --data ETTh2 \
  --root_path ./dataset/ETT-small/ \
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
  2>&1 | tee logs/glpatch_hydra_v33a/ETTh2_${pred_len}.log
done

for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id ETTm1_${pred_len}_v33a \
  --model GLPatch_Hydra \
  --data ETTm1 \
  --root_path ./dataset/ETT-small/ \
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
  2>&1 | tee logs/glpatch_hydra_v33a/ETTm1_${pred_len}.log
done

for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id ETTm2_${pred_len}_v33a \
  --model GLPatch_Hydra \
  --data ETTm2 \
  --root_path ./dataset/ETT-small/ \
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
  2>&1 | tee logs/glpatch_hydra_v33a/ETTm2_${pred_len}.log
done

for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id Weather_${pred_len}_v33a \
  --model GLPatch_Hydra \
  --data custom \
  --root_path ./dataset/weather/ \
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
  2>&1 | tee logs/glpatch_hydra_v33a/Weather_${pred_len}.log
done

for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id Exchange_${pred_len}_v33a \
  --model GLPatch_Hydra \
  --data custom \
  --root_path ./dataset/exchange_rate/ \
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
  2>&1 | tee logs/glpatch_hydra_v33a/Exchange_${pred_len}.log
done

# ILI uses different seq_len/label_len/patch settings
for pred_len in 24 36 48 60; do
python -u run.py \
  --is_training 1 \
  --model_id ILI_${pred_len}_v33a \
  --model GLPatch_Hydra \
  --data custom \
  --root_path ./dataset/illness/ \
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
  2>&1 | tee logs/glpatch_hydra_v33a/ILI_${pred_len}.log
done

echo "=== v3.3a small datasets complete ==="
