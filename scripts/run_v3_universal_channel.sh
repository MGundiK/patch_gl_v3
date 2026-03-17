#!/bin/bash
# ============================================================================
# GLPatch_Hydra v3.1 — Channel-Aware Gating
# ============================================================================
# Tests two gate types:
#   adaptive: log(C) + variance + mean_abs → gate
#   channel:  log(C) only → gate (ablation: is C alone sufficient?)
#
# Both use gate_init=-5.0 (cold start).
# Universal config: cv_rank=32 for all datasets.
# ============================================================================

MODEL="GLPatch_Hydra"
LOGDIR="./logs/hydra_v3.1"

echo ""
echo "========== [$(date '+%H:%M:%S')] GLPatch_Hydra v3.1 — Channel-Aware Gating =========="
echo ""

DATASETS=(
  "ETTh1:ETTh1:ETTh1.csv:7:2048:0.0005:96:48:96,192,336,720:"
  "ETTh2:ETTh2:ETTh2.csv:7:2048:0.0005:96:48:96,192,336,720:"
  "ETTm1:ETTm1:ETTm1.csv:7:2048:0.0005:96:48:96,192,336,720:"
  "ETTm2:ETTm2:ETTm2.csv:7:2048:0.0001:96:48:96,192,336,720:"
  "Weather:custom:weather.csv:21:2048:0.0005:96:48:96,192,336,720:"
  "Exchange:custom:exchange_rate.csv:8:32:0.00001:96:48:96,192,336,720:"
  "Solar:Solar:solar.txt:137:512:0.005:96:48:96,192,336,720:"
  "Electricity:custom:electricity.csv:321:256:0.005:96:48:96,192,336,720:"
  "Traffic:custom:traffic.csv:862:96:0.005:96:48:96,192,336,720:"
  "ILI:custom:national_illness.csv:7:32:0.01:36:18:24,36,48,60:--patch_len 6 --stride 3 --lradj type3"
)

GATE_TYPES=("adaptive" "channel")

for GATE in "${GATE_TYPES[@]}"; do
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  Gate type: ${GATE} (with log(C) injection)"
  echo "══════════════════════════════════════════════════════════════"

  for DS_INFO in "${DATASETS[@]}"; do
    IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
    IFS=',' read -ra PLS <<< "$PRED_LENS"

    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${DS} (C=${ENC_IN}, logC_norm=$(python3 -c "import math; print(f'{math.log(max(${ENC_IN},2))/math.log(1000):.3f}')"))"

    LRADJ_ARG=""
    if [[ "$EXTRA" != *"lradj"* ]]; then
      LRADJ_ARG="--lradj sigmoid"
    fi

    COMMON="--is_training 1 --root_path ./dataset/ --data_path ${DATA_PATH} \
      --data ${DATA_FLAG} --features M --enc_in ${ENC_IN} \
      --seq_len ${SL} --label_len ${LL} \
      --patch_len 16 --stride 8 --padding_patch end \
      --ma_type ema --alpha 0.3 --beta 0.3 \
      --batch_size ${BS} --learning_rate ${LR} ${LRADJ_ARG} \
      --train_epochs 100 --patience 10 --revin 1 --use_amp \
      --cv_rank 32 --gate_type ${GATE} --gate_init -5.0 \
      ${EXTRA}"

    for PL in "${PLS[@]}"; do
      sdir="${LOGDIR}/${GATE}/${DS}"; mkdir -p ${sdir}
      echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} gate=${GATE}"
      python -u run.py --model $MODEL \
        --model_id "v31_${DS}_${PL}_${GATE}" \
        --pred_len $PL --des "v31_${GATE}" \
        $COMMON \
        2>&1 | tee ${sdir}/${PL}.log
    done

    printf "  %-12s|" "${DS}"
    for PL in "${PLS[@]}"; do
      logfile="${LOGDIR}/${GATE}/${DS}/${PL}.log"
      if [ -f "$logfile" ]; then
        mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
        [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
      else
        printf "     N/A"
      fi
    done
    echo ""
  done
done

# ============================================================
# FINAL SUMMARY
# ============================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] FINAL SUMMARY =========="
echo ""

for GATE in "${GATE_TYPES[@]}"; do
  echo "Gate: ${GATE}"
  echo "─────────────────────────────────────────────────────────────"
  printf "  %-12s %4s logC  |" "Dataset" "C"
  echo "     T1       T2       T3       T4"
  echo "  ────────────────────────+────────────────────────────────────────"
  
  for DS_INFO in "${DATASETS[@]}"; do
    IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
    IFS=',' read -ra PLS <<< "$PRED_LENS"
    LOG_C=$(python3 -c "import math; print(f'{math.log(max(${ENC_IN},2))/math.log(1000):.2f}')")
    
    printf "  %-12s %4d %4s  |" "$DS" "$ENC_IN" "$LOG_C"
    for PL in "${PLS[@]}"; do
      logfile="${LOGDIR}/${GATE}/${DS}/${PL}.log"
      if [ -f "$logfile" ]; then
        mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
        [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
      else
        printf "     N/A"
      fi
    done
    echo ""
  done
  echo ""
done

echo "Log files: ${LOGDIR}/<gate_type>/<dataset>/<pred_len>.log"
echo ""
echo "Expected log(C) values:"
echo "  ETT/ILI (C=7):    0.28"
echo "  Exchange (C=8):    0.30"
echo "  Weather (C=21):    0.44"
echo "  Solar (C=137):     0.71"
echo "  Electricity (C=321): 0.84"
echo "  Traffic (C=862):   0.98"
