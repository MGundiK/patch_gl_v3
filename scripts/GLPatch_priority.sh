#!/bin/bash

# GLPatch Quick Test — Priority datasets only
# ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange
# Run this first to validate before full suite.

ma_type=ema
alpha=0.3
beta=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/glpatch_${ma_type}" ]; then
    mkdir ./logs/glpatch_${ma_type}
fi

model_name=GLPatch
seq_len=96

for pred_len in 96 192 336 720
do
  echo "========== ETTh1 pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${pred_len}_${ma_type} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/glpatch_${ma_type}/${model_name}_ETTh1_${seq_len}_${pred_len}.log

  echo "========== ETTh2 pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_${pred_len}_${ma_type} \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/glpatch_${ma_type}/${model_name}_ETTh2_${seq_len}_${pred_len}.log

  echo "========== ETTm1 pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_${pred_len}_${ma_type} \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/glpatch_${ma_type}/${model_name}_ETTm1_${seq_len}_${pred_len}.log

  echo "========== ETTm2 pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_${pred_len}_${ma_type} \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/glpatch_${ma_type}/${model_name}_ETTm2_${seq_len}_${pred_len}.log

  echo "========== Weather pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_${pred_len}_${ma_type} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/glpatch_${ma_type}/${model_name}_weather_${seq_len}_${pred_len}.log

  echo "========== Exchange pred_len=${pred_len} =========="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id exchange_${pred_len}_${ma_type} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 8 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/glpatch_${ma_type}/${model_name}_exchange_${seq_len}_${pred_len}.log
done

echo "========== Priority experiments complete =========="
