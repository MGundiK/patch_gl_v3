#!/bin/bash

# GLPatch_Hydra v3.2g — Solar

ma_type=ema
alpha=0.3
beta=0.3
model_name=GLPatch_Hydra
gate_type=hybrid
gate_init=-5.0
cv_rank=32
seq_len=96

mkdir -p ./logs/glpatch_hydra_v32g/Solar

for pred_len in 96 192 336 720
do
  echo "========== Solar pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 --root_path ./dataset/ --data_path solar.txt \
    --model_id solar_${pred_len}_v32g --model $model_name --data Solar \
    --features M --seq_len $seq_len --pred_len $pred_len --enc_in 137 \
    --des 'Exp' --itr 1 --batch_size 512 --learning_rate 0.005 \
    --lradj 'sigmoid' --ma_type $ma_type --alpha $alpha --beta $beta \
    --gate_type $gate_type --gate_init $gate_init --cv_rank $cv_rank \
    > logs/glpatch_hydra_v32g/Solar/${model_name}_${seq_len}_${pred_len}.log
done

echo "========== v3.2g Solar complete =========="
