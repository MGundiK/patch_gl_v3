#!/bin/bash
# ============================================================================
# Gate Value Logging — What is the gate ACTUALLY doing?
# ============================================================================
#
# Before trying more architectural changes, we need to KNOW:
#   1. What is the gate value on C=7 datasets? (0.001? 0.05? 0.15?)
#   2. What is the gate value on C=321 datasets?
#   3. How big is |gate * mixed| relative to |input|?
#   4. What is epsilon learning to be?
#
# This runs a SUBSET of datasets at T=96 only, with gate logging enabled.
# ============================================================================

MODEL="GLPatch_Hydra"
LOGDIR="./logs/gate_probe"

echo ""
echo "========== [$(date '+%H:%M:%S')] Gate Value Probe =========="
echo ""

# Short runs: just T=96, the 4 key datasets
DATASETS=(
  "ETTm1:ETTm1:ETTm1.csv:7:2048:0.0005:96:48:96:"
  "ETTh1:ETTh1:ETTh1.csv:7:2048:0.0005:96:48:96:"
  "Electricity:custom:electricity.csv:321:256:0.005:96:48:96:"
  "Traffic:custom:traffic.csv:862:96:0.005:96:48:96:"
  "Solar:Solar:solar.txt:137:512:0.005:96:48:96:"
  "Exchange:custom:exchange_rate.csv:8:32:0.00001:96:48:96:"
)

for DS_INFO in "${DATASETS[@]}"; do
  IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
  IFS=',' read -ra PLS <<< "$PRED_LENS"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [$(date '+%H:%M:%S')] ${DS} (C=${ENC_IN})"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  LRADJ_ARG="--lradj sigmoid"

  for PL in "${PLS[@]}"; do
    sdir="${LOGDIR}/${DS}"; mkdir -p ${sdir}
    echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL}"
    python -u run.py --model $MODEL \
      --model_id "probe_${DS}_${PL}" \
      --pred_len $PL --des "gate_probe" \
      --cv_rank 32 --gate_type hybrid --gate_init -5.0 \
      --is_training 1 --root_path ./dataset/ --data_path ${DATA_PATH} \
      --data ${DATA_FLAG} --features M --enc_in ${ENC_IN} \
      --seq_len ${SL} --label_len ${LL} \
      --patch_len 16 --stride 8 --padding_patch end \
      --ma_type ema --alpha 0.3 --beta 0.3 \
      --batch_size ${BS} --learning_rate ${LR} ${LRADJ_ARG} \
      --train_epochs 100 --patience 10 --revin 1 --use_amp \
      ${EXTRA} \
      2>&1 | tee ${sdir}/${PL}.log
  done
done

echo ""
echo "========== [$(date '+%H:%M:%S')] Probe complete =========="
echo ""
echo "Extract gate stats with:"
echo "  grep 'GATE_STATS' logs/gate_probe/*/*.log"
echo ""
