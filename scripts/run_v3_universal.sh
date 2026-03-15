#!/bin/bash
# ============================================================================
# GLPatch_Hydra v3 — UNIVERSAL CONFIG TEST
# ============================================================================
# ONE config for ALL datasets. No dataset-dependent switches.
# Hydra is always on. The adaptive gate decides mixing strength.
#
# Config: cv_rank=32, gate_type=adaptive, gate_init=-5.0
#
# Hypothesis: gate learns to open on high-C (Electricity, Traffic, Solar)
#             and stay closed on low-C (ETTh1, ETTm1, Exchange)
#
# Also tests gate_type=scalar and gate_type=vector for ablation.
# ============================================================================

MODEL="GLPatch_Hydra"
LOGDIR="./logs/hydra_v3"

echo ""
echo "========== [$(date '+%H:%M:%S')] GLPatch_Hydra v3 — Universal Config =========="
echo ""

# ── Dataset definitions ──────────────────────────────────────────
# Format: "label:data_flag:data_path:enc_in:batch_size:lr:seq_len:label_len:pred_lens:extra_args"

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

# ── Gate types to test ───────────────────────────────────────────
# Start with adaptive only. Add scalar/vector for ablation later.
GATE_TYPES=("adaptive")

for GATE in "${GATE_TYPES[@]}"; do
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  Gate type: ${GATE}"
  echo "══════════════════════════════════════════════════════════════"

  for DS_INFO in "${DATASETS[@]}"; do
    IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
    
    IFS=',' read -ra PLS <<< "$PRED_LENS"
    
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] ${DS} (C=${ENC_IN})"

    # Default args (no lradj override unless in EXTRA)
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
      echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL}"
      python -u run.py --model $MODEL \
        --model_id "v3_${DS}_${PL}_${GATE}" \
        --pred_len $PL --des "v3_${GATE}" \
        $COMMON \
        2>&1 | tee ${sdir}/${PL}.log
    done

    # Print results for this dataset
    echo "  === ${DS} (C=${ENC_IN}) ==="
    printf "  %-10s|" "${GATE}"
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
echo "Universal config: cv_rank=32, gate_init=-5.0"
echo ""

for GATE in "${GATE_TYPES[@]}"; do
  echo "Gate type: ${GATE}"
  echo "─────────────────────────────────────────────────────────────"
  printf "  %-12s C    |" "Dataset"
  echo "     T1       T2       T3       T4"
  echo "  ──────────────────+────────────────────────────────────────"
  
  for DS_INFO in "${DATASETS[@]}"; do
    IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
    IFS=',' read -ra PLS <<< "$PRED_LENS"
    
    printf "  %-12s %3d  |" "$DS" "$ENC_IN"
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

echo ""
echo "Compare with GLPatch baselines to see if gate auto-disables on low-C"
echo "Log files: ${LOGDIR}/<gate_type>/<dataset>/<pred_len>.log"
