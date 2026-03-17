#!/bin/bash
# ============================================================================
# GLPatch_Hydra v3.2 — Hybrid Gate: channel-primary + data bonus
# ============================================================================
# gate = sigmoid( f(logC) + ε·g(variance) )
#   - f(logC): channel-based gate (primary signal)
#   - g(variance): data-dependent correction
#   - ε: learnable scalar, init=0 (starts as pure channel gate)
#
# Universal config: cv_rank=32, gate_type=hybrid, gate_init=-5.0
# ============================================================================

MODEL="GLPatch_Hydra"
LOGDIR="./logs/hydra_v3.2"

echo ""
echo "========== [$(date '+%H:%M:%S')] GLPatch_Hydra v3.2 — Hybrid Gate =========="
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

for DS_INFO in "${DATASETS[@]}"; do
  IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
  IFS=',' read -ra PLS <<< "$PRED_LENS"

  LOG_C=$(python3 -c "import math; print(f'{math.log(max(${ENC_IN},2))/math.log(1000):.3f}')")
  echo ""
  echo ">>> [$(date '+%H:%M:%S')] ${DS} (C=${ENC_IN}, logC=${LOG_C})"

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
    --cv_rank 32 --gate_type hybrid --gate_init -5.0 \
    ${EXTRA}"

  for PL in "${PLS[@]}"; do
    sdir="${LOGDIR}/${DS}"; mkdir -p ${sdir}
    echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL}"
    python -u run.py --model $MODEL \
      --model_id "v32_${DS}_${PL}" \
      --pred_len $PL --des "v32_hybrid" \
      $COMMON \
      2>&1 | tee ${sdir}/${PL}.log
  done

  printf "  %-12s|" "${DS}"
  for PL in "${PLS[@]}"; do
    logfile="${LOGDIR}/${DS}/${PL}.log"
    if [ -f "$logfile" ]; then
      mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
      [ -n "$mse" ] && printf "  %7s" $(printf "%.4f" $mse) || printf "     N/A"
    else
      printf "     N/A"
    fi
  done
  echo ""
done

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========== [$(date '+%H:%M:%S')] v3.2 HYBRID GATE SUMMARY =========="
echo ""
echo "Config: cv_rank=32, gate_type=hybrid, gate_init=-5.0"
echo "Design: gate = sigmoid( f(logC) + ε·g(variance) ), ε init=0"
echo ""
printf "  %-12s %4s %5s  |" "Dataset" "C" "logC"
echo "     T1       T2       T3       T4"
echo "  ────────────────────────+────────────────────────────────────────"

for DS_INFO in "${DATASETS[@]}"; do
  IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
  IFS=',' read -ra PLS <<< "$PRED_LENS"
  LOG_C=$(python3 -c "import math; print(f'{math.log(max(${ENC_IN},2))/math.log(1000):.2f}')")

  printf "  %-12s %4d %5s  |" "$DS" "$ENC_IN" "$LOG_C"
  for PL in "${PLS[@]}"; do
    logfile="${LOGDIR}/${DS}/${PL}.log"
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
echo "Log files: ${LOGDIR}/<dataset>/<pred_len>.log"
echo ""
echo "Check epsilon values in logs:"
echo "  grep 'ε=' ${LOGDIR}/*/*.log"
echo ""
echo "GLPatch baselines for comparison:"
echo "  ETTh1:  0.376  0.417  0.449  0.499"
echo "  ETTh2:  0.233  0.291  0.349  0.401"
echo "  ETTm1:  0.309  0.347  0.384  0.458"
echo "  ETTm2:  0.163  0.228  0.291  0.379"
echo "  Weather: 0.166  0.209  0.235  0.308"
echo "  Exchange: 0.082  0.175  0.334  0.877"
echo "  Solar:  0.198  0.236  0.259  0.262"
echo "  Elec:   0.157  0.158  0.179  0.216"
echo "  Traffic: 0.479  0.477  0.498  0.544"
