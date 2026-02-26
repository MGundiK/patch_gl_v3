#!/bin/bash
# ============================================================================
# GLPatch_Hydra v2 — Traffic (862v) with post_fusion
# ============================================================================
# Weather showed post_fusion is the clear winner (-3.8% at T=96).
# Traffic (C=862) is the big prize: iTransformer beats GLPatch by ~14%
# using full O(C²d) attention. Hydra's O(Cd) could close this gap.
#
# At C=862, post_fusion operates on dim=pred_len:
#   T=96:  dim=96,  rank=32  → Hydra mixes 862 channels in 96-dim space
#   T=720: dim=720, rank=32  → larger feature space, same rank
#
# Tests: hydra_gated (best from Weather) + rank sweep (32/64/128)
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

# Traffic reference hparams (from GLPatch runs)
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
echo " GLPatch_Hydra v2 — Traffic (862v) post_fusion"
echo " hydra_gated with rank sweep: 32, 64, 128"
echo "============================================================"

for PRED_LEN in 96 192 336 720; do
  echo ""
  echo "=== Traffic T=${PRED_LEN} ==="

  # Baseline: no mixing (= GLPatch)
  echo "--- none (baseline) ---"
  python run.py \
    --model $MODEL --model_id "T${PRED_LEN}_v2_none" \
    --pred_len $PRED_LEN \
    --cv_mixing none \
    --des "v2_none" \
    $COMMON_ARGS

  # hydra_gated @ post_fusion, rank=32
  echo "--- hydra_gated @ post_fusion r=32 ---"
  python run.py \
    --model $MODEL --model_id "T${PRED_LEN}_v2_pf_r32" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_gated --cv_rank 32 --cv_placement post_fusion \
    --des "v2_pf_r32" \
    $COMMON_ARGS

  # hydra_gated @ post_fusion, rank=64
  echo "--- hydra_gated @ post_fusion r=64 ---"
  python run.py \
    --model $MODEL --model_id "T${PRED_LEN}_v2_pf_r64" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_gated --cv_rank 64 --cv_placement post_fusion \
    --des "v2_pf_r64" \
    $COMMON_ARGS

  # hydra_gated @ post_fusion, rank=128
  echo "--- hydra_gated @ post_fusion r=128 ---"
  python run.py \
    --model $MODEL --model_id "T${PRED_LEN}_v2_pf_r128" \
    --pred_len $PRED_LEN \
    --cv_mixing hydra_gated --cv_rank 128 --cv_placement post_fusion \
    --des "v2_pf_r128" \
    $COMMON_ARGS

done

echo ""
echo "============================================================"
echo " Done. Extract results:"
echo "   grep 'v2_' result.txt"
echo ""
echo " GLPatch baselines (Traffic):"
echo "   T=96:  0.479    T=192: 0.476"
echo "   T=336: 0.498    T=720: 0.544"
echo ""
echo " iTransformer targets:"
echo "   T=96:  0.395    T=192: 0.417"
echo "   T=336: 0.433    T=720: 0.467"
echo ""
echo " Weather post_fusion got -3.8% at T=96."
echo " With 862 channels (vs 21), expect stronger gains."
echo "============================================================"
