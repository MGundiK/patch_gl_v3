#!/bin/bash

# GLPatch_Hydra v3.2g — Traffic

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch_Hydra
gate_type=hybrid
gate_init=-5.0
cv_rank=32
placement=post_fusion
seq_len=96

mkdir -p ./logs/glpatch_hydra_v32g

for pred_len in 96 192 336 720
do
  echo "========== Traffic pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path traffic.csv \
    --model_id traffic_${pred_len}_v32g --model $model_name --data custom \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 862 \
    --des 'Exp' --itr 1 --batch_size 96 --learning_rate 0.005 \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --gate_type $gate_type --gate_init $gate_init --cv_rank $cv_rank \
    --placement $placement \
    > logs/glpatch_hydra_v32g/${model_name}_traffic_${seq_len}_${pred_len}.log
done

echo "========== v3.2g Traffic complete =========="
