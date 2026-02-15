#!/bin/bash

# GLPatch v8 — Hyperparameter Tuning (LR grid search)
# Only small/fast datasets. Traffic, Electricity, Solar keep v8 defaults.
# Tests 3 learning rates per dataset: lower, current (xPatch default), higher
#
# Rationale for LR ranges:
# - The adaptive fusion gate adds learnable parameters that may converge
#   better at different LR than xPatch's vanilla architecture
# - ETTh1/h2 at long horizons overfit → try lower LR
# - ILI is noisy/tiny → try both directions
#
# Estimated time: ~2-3 hours on Colab A100 with AMP

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
seq_len=96

mkdir -p ./logs/tuning

# ================================================================
# ETTh1 — current: 0.0005, test: 0.0003, 0.0005, 0.0008
# Main target: fix 720 (+6.1%)
# ================================================================
for lr in 0.0003 0.0005 0.0008
do
  tag="ETTh1_lr${lr}"
  mkdir -p ./logs/tuning/${tag}
  for pred_len in 96 192 336 720
  do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ${tag}_${pred_len}_${ma_type} \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 2048 \
      --learning_rate $lr \
      --lradj 'sigmoid' \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --use_amp \
      --num_workers 2 > logs/tuning/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# ETTh2 — current: 0.0005, test: 0.0002, 0.0005, 0.001
# Main target: fix 336 (+1.4%), improve 192
# ================================================================
for lr in 0.0002 0.0005 0.001
do
  tag="ETTh2_lr${lr}"
  mkdir -p ./logs/tuning/${tag}
  for pred_len in 96 192 336 720
  do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --model_id ${tag}_${pred_len}_${ma_type} \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 2048 \
      --learning_rate $lr \
      --lradj 'sigmoid' \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --use_amp \
      --num_workers 2 > logs/tuning/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# ETTm1 — current: 0.0005, test: 0.0003, 0.0005, 0.001
# Already 4/4, widening margins
# ================================================================
for lr in 0.0003 0.0005 0.001
do
  tag="ETTm1_lr${lr}"
  mkdir -p ./logs/tuning/${tag}
  for pred_len in 96 192 336 720
  do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ${tag}_${pred_len}_${ma_type} \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 2048 \
      --learning_rate $lr \
      --lradj 'sigmoid' \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --use_amp \
      --num_workers 2 > logs/tuning/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# ETTm2 — current: 0.0001, test: 0.00005, 0.0001, 0.0003
# Already 4/4, widening margins
# ================================================================
for lr in 0.00005 0.0001 0.0003
do
  tag="ETTm2_lr${lr}"
  mkdir -p ./logs/tuning/${tag}
  for pred_len in 96 192 336 720
  do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ${tag}_${pred_len}_${ma_type} \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 2048 \
      --learning_rate $lr \
      --lradj 'sigmoid' \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --use_amp \
      --num_workers 2 > logs/tuning/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# Weather — current: 0.0005, test: 0.0003, 0.0005, 0.001
# Already 4/4, widening margins
# ================================================================
for lr in 0.0003 0.0005 0.001
do
  tag="weather_lr${lr}"
  mkdir -p ./logs/tuning/${tag}
  for pred_len in 96 192 336 720
  do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path weather.csv \
      --model_id ${tag}_${pred_len}_${ma_type} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 2048 \
      --learning_rate $lr \
      --lradj 'sigmoid' \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --use_amp \
      --num_workers 2 > logs/tuning/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# Exchange — current: 0.00001, test: 0.000005, 0.00001, 0.00003
# Already 4/4, widening margins
# ================================================================
for lr in 0.000005 0.00001 0.00003
do
  tag="exchange_lr${lr}"
  mkdir -p ./logs/tuning/${tag}
  for pred_len in 96 192 336 720
  do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path exchange_rate.csv \
      --model_id ${tag}_${pred_len}_${ma_type} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 32 \
      --learning_rate $lr \
      --lradj 'sigmoid' \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --use_amp \
      --num_workers 2 > logs/tuning/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# ILI — current: 0.01/type3, test: 0.005, 0.01, 0.02
# Main target: fix 24 (+16%), 36 (+8%)
# Note: ILI uses lradj=type3 and different patch config
# ================================================================
for lr in 0.005 0.01 0.02
do
  tag="ili_lr${lr}"
  mkdir -p ./logs/tuning/${tag}
  for pred_len in 24 36 48 60
  do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path national_illness.csv \
      --model_id ${tag}_${pred_len}_${ma_type} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 36 \
      --label_len 18 \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 32 \
      --learning_rate $lr \
      --lradj 'type3' \
      --patch_len 6 \
      --stride 3 \
      --ma_type $ma_type \
      --alpha $alpha \
      --beta $beta \
      --use_amp \
      --num_workers 2 > logs/tuning/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

echo ""
echo "========== [$(date '+%H:%M')] TUNING COMPLETE =========="
echo "========== All results in result.txt =========="
echo ""
echo "To extract best LR per dataset, grep result.txt for each tag."
echo "Example: grep 'ETTh1_lr' result.txt"
