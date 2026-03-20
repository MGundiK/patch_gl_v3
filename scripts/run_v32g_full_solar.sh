#!/bin/bash

# GLPatch_Hydra v3.2g — Full 10-dataset eval (seq_len=96)
# Hyperparameters match probe run script used throughout development.
# Gate config: hybrid, gate_init=-5.0, cv_rank=32, placement=post_fusion
# Mean pooling in HydraAttention (fixes O(C) norm explosion).
# Softplus epsilon: ε_eff = softplus(ε_raw) ∈ [0, +∞)

ma_type=ema
alpha=0.3
beta=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/glpatch_hydra_v32g" ]; then
    mkdir ./logs/glpatch_hydra_v32g
fi

model_name=GLPatch_Hydra
seq_len=96

# Shared Hydra gate config
gate_type=hybrid
gate_init=-5.0
cv_rank=32
placement=post_fusion

for pred_len in 96 192 336 720
do


  echo "========== Solar pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path solar.txt \
    --model_id solar_${pred_len}_v32g \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 137 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 512 \
    --learning_rate 0.005 \
    --lradj 'sigmoid' \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta \
    --gate_type $gate_type \
    --gate_init $gate_init \
    --cv_rank $cv_rank \
    --placement $placement > logs/glpatch_hydra_v32g/${model_name}_solar_${seq_len}_${pred_len}.log
done

