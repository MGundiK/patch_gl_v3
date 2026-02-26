#!/bin/bash
# ============================================================================
# GLPatch_Hydra Experiments — Phase 2: Traffic (862v)
# ============================================================================
# Traffic is where the gap is largest: iTransformer (full attention across 862
# vars) beats GLPatch by ~14%. Hydra's O(C*D) attention could close this gap
# without iTransformer's O(C²*D) cost.
#
# At C=862: std attention = 862²×256 = 190M FLOPs/patch
#           Hydra         = 862×256  = 221K FLOPs/patch  (860x cheaper!)
#
# Uses your Traffic reference hyperparameters.
# ============================================================================

MODEL="GLPatch_Hydra"
DATA="custom"
ROOT="./dataset"
DATA_PATH="traffic.csv"
ENC_IN=862
SEQ_LEN=96
LABEL_LEN=48
FEATURES="M"
PATCH_LEN=16
STRIDE=8

# Traffic reference hparams
BATCH_SIZE=96
LR=0.005
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
echo " Phase 2: Traffic (862v) — Hydra at scale"
echo "============================================================"

for PRED_LEN in 96 192 336 720; do
  echo ""
  echo "=== Traffic T=${PRED_LEN} ==="

  # Baseline: none (should match GLPatch)
  echo "--- Baseline: cv_mixing=none ---"
  python run.py \
    --model $MODEL --model_id "Traffic_${PRED_LEN}_hydra_none" \
    --pred_len $PRED_LEN \
    --cv_mixing none \
    --des "hydra_none" \
    $COMMON_ARGS

  # hydra_bottleneck r=32 (compressed — safe for 862 channels)
  echo "--- hydra_bottleneck r=32 ---"
  python run.py \
    --model $MODEL --model_id "Traffic_${PRED_LEN}_hydra_bn32" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_bottleneck --cv_rank 32 \
    --des "hydra_bn32" \
    $COMMON_ARGS

  # hydra_bottleneck r=64 (more capacity for 862 channels)
  echo "--- hydra_bottleneck r=64 ---"
  python run.py \
    --model $MODEL --model_id "Traffic_${PRED_LEN}_hydra_bn64" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_bottleneck --cv_rank 64 \
    --des "hydra_bn64" \
    $COMMON_ARGS

  # hydra_gated (gating may be essential at 862 channels to filter noise)
  echo "--- hydra_gated ---"
  python run.py \
    --model $MODEL --model_id "Traffic_${PRED_LEN}_hydra_gated" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_gated \
    --des "hydra_gated" \
    $COMMON_ARGS

done

echo ""
echo "============================================================"
echo " Traffic experiments complete."
echo "============================================================"
echo ""
echo "Expected comparisons (Traffic avg MSE):"
echo "  iTransformer (SOTA):   0.428  (full O(C²d) attention)"
echo "  GLPatch (CI):          0.499"
echo "  AdaPatch (MLP r=32):   0.492  (−1.4% vs GL)"
echo "  Hydra target:         <0.480  (meaningful gain over MLP mixing)"
echo "  Stretch goal:         <0.450  (approaching iTransformer)"
echo ""
