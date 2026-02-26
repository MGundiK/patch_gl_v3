#!/bin/bash

# GLPatch v8 — LR Tuning Round 2 (Large Datasets)
# Traffic, Electricity, Solar — never tuned before (used xPatch default 0.005)
#
# Strategy: 3-point grid centered on default, spaced by ~2x
#   All three: 0.002, 0.005 (default), 0.01
#
# These are expensive so we keep the grid minimal.
# Estimated: ~6-8 hours on Colab with AMP

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
seq_len=96

mkdir -p ./logs/tuning_r2

# ================================================================
# Electricity — default: 0.005
# Currently 3/4 (losing 720 by ~0.2%)
# ================================================================
for lr in 0.002 0.005 0.01
do
  tag="electricity_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path electricity.csv \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 321 \
      --des 'Exp' --itr 1 --batch_size 256 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# Traffic — default: 0.005
# Currently 3/4 (losing 96 in MSE)
# ================================================================
for lr in 0.002 0.005 0.01
do
  tag="traffic_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path traffic.csv \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 862 \
      --des 'Exp' --itr 1 --batch_size 96 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# Solar — default: 0.005
# Currently 2/4 (losing 192, 720)
# ================================================================
for lr in 0.002 0.005 0.01
do
  tag="solar_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path solar.txt \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data Solar \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 137 \
      --des 'Exp' --itr 1 --batch_size 512 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

echo ""
echo "========== [$(date '+%H:%M')] LARGE DATASET TUNING COMPLETE =========="
echo "3 datasets × 3 LRs × 4 horizons = 36 runs"
