#!/bin/bash
# ============================================================================
# GLPatch_Hydra Experiments — Phase 1: Weather (21v)
# ============================================================================
# Weather is the #1 priority: AdaPatch's MLP mixing gets −5.5% vs GLPatch here.
# If Hydra can match or beat that, it validates the approach.
#
# Uses SAME hyperparameters as your GLPatch Weather runs.
# Only varies: --model, --cv_mixing, --cv_rank
# ============================================================================

MODEL="GLPatch_Hydra"
DATA="custom"
ROOT="./dataset"
DATA_PATH="weather.csv"
ENC_IN=21
SEQ_LEN=96
LABEL_LEN=48
FEATURES="M"
PATCH_LEN=16
STRIDE=8

# Hyperparameters matching your reference Weather settings
BATCH_SIZE=2048
LR=0.0005
LRADJ="sigmoid"
EPOCHS=100
PATIENCE=10

COMMON_ARGS="--is_training 1 \
  --data $DATA --root_path $ROOT --data_path $DATA_PATH \
  --features $FEATURES --enc_in $ENC_IN \
  --seq_len $SEQ_LEN --label_len $LABEL_LEN \
  --patch_len $PATCH_LEN --stride $STRIDE --padding_patch end \
  --ma_type ema --alpha 0.3 --beta 0.3 \
  --batch_size $BATCH_SIZE --learning_rate $LR --lradj $LRADJ \
  --train_epochs $EPOCHS --patience $PATIENCE \
  --revin 1 --use_amp"


echo "============================================================"
echo " Phase 1: Weather — Hydra variant comparison"
echo "============================================================"

for PRED_LEN in 96 192 336 720; do
  echo ""
  echo "=== Weather T=${PRED_LEN} ==="

  # Experiment 1: GLPatch_Hydra with cv_mixing=none (should match GLPatch exactly)
  echo "--- Baseline: cv_mixing=none ---"
  python run.py \
    --model $MODEL --model_id "Weather_${PRED_LEN}_hydra_none" \
    --pred_len $PRED_LEN \
    --cv_mixing none \
    --des "hydra_none" \
    $COMMON_ARGS

  # Experiment 2: hydra (full-dim, simplest)
  echo "--- hydra (full-dim) ---"
  python run.py \
    --model $MODEL --model_id "Weather_${PRED_LEN}_hydra_full" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra \
    --des "hydra_full" \
    $COMMON_ARGS

  # Experiment 3: hydra_bottleneck r=32 (closest to AdaPatch MLP r=32)
  echo "--- hydra_bottleneck r=32 ---"
  python run.py \
    --model $MODEL --model_id "Weather_${PRED_LEN}_hydra_bn32" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_bottleneck --cv_rank 32 \
    --des "hydra_bn32" \
    $COMMON_ARGS

  # Experiment 4: hydra_gated r=32 (recommended — gating filters noise)
  echo "--- hydra_gated ---"
  python run.py \
    --model $MODEL --model_id "Weather_${PRED_LEN}_hydra_gated" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_gated \
    --des "hydra_gated" \
    $COMMON_ARGS

done

echo ""
echo "============================================================"
echo " Weather experiments complete. Check result.txt for MSE/MAE."
echo "============================================================"
echo ""
echo "Expected comparisons (Weather avg MSE):"
echo "  GLPatch (CI):          0.229"
echo "  AdaPatch (MLP r=32):   0.217  (−5.5%)"
echo "  Hydra target:         <0.217  (beat AdaPatch)"
echo ""
