#!/bin/bash

# GLPatch v8 — LR Tuning Round 2 (Small/Fast Datasets)
# Wider grid, informed by Round 1 results.
#
# Strategy per dataset:
#   ETTh1:    Extend lower (0.0002) to help 720. Add 0.0004 intermediate.
#   ETTh2:    Light touch — default already optimal, just check 0.0003 and 0.0007
#   ETTm1:    Add 0.0007 compromise between short-optimal 0.001 and long-optimal 0.0005
#   ETTm2:    Light touch — default already optimal, skip or minimal
#   Weather:  Light touch — default already optimal, just check 0.0003 and 0.0007
#   Exchange: Extend lower — 5e-6 was at grid edge. Try 2e-6, 3e-6
#   ILI:      Extend higher — 0.02 was at grid edge. Try 0.03, 0.05
#
# NOTE: Only test NEW values not already tested in Round 1.
#       Round 1 results can be reused for existing LR values.
#
# Estimated: ~2-3 hours on Colab with AMP

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch
seq_len=96

mkdir -p ./logs/tuning_r2

# ================================================================
# ETTh1 — Round 1: 0.0003, 0.0005, 0.0008
# NEW: 0.0002 (lower), 0.0004 (intermediate)
# Goal: find LR that helps 720 without destroying 96/192
# ================================================================
for lr in 0.0002 0.0004
do
  tag="ETTh1_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data ETTh1 \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# ETTh2 — Round 1: 0.0002, 0.0005, 0.001
# NEW: 0.0003, 0.0007 (intermediate values)
# Goal: marginally improve on default 0.0005
# ================================================================
for lr in 0.0003 0.0007
do
  tag="ETTh2_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTh2.csv \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data ETTh2 \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# ETTm1 — Round 1: 0.0003, 0.0005, 0.001
# NEW: 0.0007 (compromise), 0.002 (extend higher)
# Goal: single LR that works for both short and long horizons
# ================================================================
for lr in 0.0007 0.002
do
  tag="ETTm1_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTm1.csv \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data ETTm1 \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# ETTm2 — Round 1: 0.00005, 0.0001, 0.0003
# NEW: 0.00007, 0.00015 (fill gaps around optimal 0.0001)
# Goal: marginal improvement, low priority
# ================================================================
for lr in 0.00007 0.00015
do
  tag="ETTm2_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTm2.csv \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data ETTm2 \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# Weather — Round 1: 0.0003, 0.0005, 0.001
# NEW: 0.0007, 0.002 (extend higher)
# Goal: check if higher LR helps — probably marginal
# ================================================================
for lr in 0.0007 0.002
do
  tag="weather_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path weather.csv \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 21 \
      --des 'Exp' --itr 1 --batch_size 2048 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# Exchange — Round 1: 5e-6, 1e-5, 3e-5
# NEW: 2e-6, 3e-6, 7e-6 (extend lower + fill gap)
# Best was at lower edge — worth exploring further
# ================================================================
for lr in 0.000002 0.000003 0.000007
do
  tag="exchange_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 96 192 336 720; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path exchange_rate.csv \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len $seq_len --pred_len $pred_len --enc_in 8 \
      --des 'Exp' --itr 1 --batch_size 32 --learning_rate $lr \
      --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

# ================================================================
# ILI — Round 1: 0.005, 0.01, 0.02
# NEW: 0.03, 0.05 (extend higher — 0.02 was at edge)
# Goal: maybe fix ILI-24, which was the stubborn loss
# ================================================================
for lr in 0.03 0.05
do
  tag="ili_lr${lr}"
  mkdir -p ./logs/tuning_r2/${tag}
  for pred_len in 24 36 48 60; do
    echo ">>> [$(date '+%H:%M')] ${tag} pred_len=${pred_len}"
    python -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path national_illness.csv \
      --model_id ${tag}_${pred_len}_${ma_type} --model $model_name --data custom \
      --features M --seq_len 36 --label_len 18 --pred_len $pred_len --enc_in 7 \
      --des 'Exp' --itr 1 --batch_size 32 --learning_rate $lr \
      --lradj 'type3' --patch_len 6 --stride 3 \
      --ma_type $ma_type --alpha $alpha --beta $beta \
      --use_amp --num_workers 2 > logs/tuning_r2/${tag}/${pred_len}.log
  done
  echo "=== ${tag} complete ==="
  grep "mse:" result.txt | tail -4
  echo ""
done

echo ""
echo "========== [$(date '+%H:%M')] TUNING ROUND 2 COMPLETE =========="
echo ""
echo "Combine with Round 1 results for full picture."
echo "Total NEW LRs tested: 2+2+2+2+2+3+2 = 15 values × 4 horizons = 60 runs"
