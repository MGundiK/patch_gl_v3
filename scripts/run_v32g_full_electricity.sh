#!/bin/bash

# GLPatch_Hydra v3.2g — Electricity

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch_Hydra
gate_type=hybrid
gate_init=-5.0
cv_rank=32
seq_len=96

mkdir -p ./logs/glpatch_hydra_v32g/Electricity

for pred_len in 96 192 336 720
do
  echo "========== Electricity pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path electricity.csv \
    --model_id electricity_${pred_len}_v32g --model $model_name --data custom \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 321 \
    --des 'Exp' --itr 1 --batch_size 256 --learning_rate 0.005 \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --gate_type $gate_type --gate_init $gate_init --cv_rank $cv_rank \
    > logs/glpatch_hydra_v32g/Electricity/${model_name}_${seq_len}_${pred_len}.log
done

echo "========== v3.2g Electricity complete =========="
