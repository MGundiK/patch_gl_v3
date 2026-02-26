#!/bin/bash
# ============================================================================
# GLPatch_Hydra v2 — Weather Placement Experiment
# ============================================================================
# v1 showed Hydra at post_embed (raw patch embeddings) barely helps.
# v2 tests 3 later placements where representations are richer:
#
#   post_pw:      After pointwise conv — dim=16, patches carry extracted features
#   post_stream:  After seasonal head  — dim=pred_len, full temporal prediction
#   post_fusion:  After stream fusion  — dim=pred_len, combined s+t prediction
#
# Uses hydra_gated only (best variant from v1). Rank auto-clipped per dim.
# Same hyperparameters as GLPatch Weather runs.
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
echo " GLPatch_Hydra v2 — Weather Placement Comparison"
echo " Using hydra_gated, rank=32 (auto-clipped per placement)"
echo "============================================================"

for PRED_LEN in 96 192 336 720; do
  echo ""
  echo "=== Weather T=${PRED_LEN} ==="

  # Baseline: no mixing (= GLPatch)
  echo "--- none (baseline) ---"
  python run.py \
    --model $MODEL --model_id "W${PRED_LEN}_v2_none" \
    --pred_len $PRED_LEN \
    --cv_mixing none \
    --des "v2_none" \
    $COMMON_ARGS

  # post_pw: After pointwise conv, dim=16, rank auto-clipped to 8
  echo "--- hydra_gated @ post_pw (dim=16) ---"
  python run.py \
    --model $MODEL --model_id "W${PRED_LEN}_v2_postpw" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_gated --cv_rank 32 --cv_placement post_pw \
    --des "v2_postpw" \
    $COMMON_ARGS

  # post_stream: After seasonal head, dim=pred_len
  echo "--- hydra_gated @ post_stream (dim=${PRED_LEN}) ---"
  python run.py \
    --model $MODEL --model_id "W${PRED_LEN}_v2_poststream" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_gated --cv_rank 32 --cv_placement post_stream \
    --des "v2_poststream" \
    $COMMON_ARGS

  # post_fusion: After gate fusion, dim=pred_len
  echo "--- hydra_gated @ post_fusion (dim=${PRED_LEN}) ---"
  python run.py \
    --model $MODEL --model_id "W${PRED_LEN}_v2_postfusion" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_gated --cv_rank 32 --cv_placement post_fusion \
    --des "v2_postfusion" \
    $COMMON_ARGS

done

echo ""
echo "============================================================"
echo " Done. Compare with:"
echo "   grep 'v2_' result.txt"
echo ""
echo " Expected dims & effective ranks:"
echo "   post_pw:     dim=16,  rank=8  (auto-clipped from 32)"
echo "   post_stream: dim=T,   rank=32 (or T//2 if T<64)"
echo "   post_fusion: dim=T,   rank=32 (or T//2 if T<64)"
echo ""
echo " GLPatch baselines (from your runs):"
echo "   T=96:  0.1658    T=192: 0.2092"
echo "   T=336: 0.2349    T=720: 0.3079"
echo "============================================================"
