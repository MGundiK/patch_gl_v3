#!/bin/bash
# ============================================================================
# Cold Gate Init Ablation — gate_init = {-7.0, -9.0} vs baseline -5.0
# ============================================================================
#
# Problem: On C≤8 datasets the hybrid gate doesn't fully close.
#   gate_init=-5.0 → sigmoid ≈ 0.67%  initial mixing
#   gate_init=-7.0 → sigmoid ≈ 0.09%  initial mixing
#   gate_init=-9.0 → sigmoid ≈ 0.01%  initial mixing
#
# Test on:
#   ETTm1     (C=7)   — worst offender at +0.7% vs GLPatch across all versions
#   ETTh1     (C=7)   — +0.9% vs xPatch avg
#   Exchange  (C=8)   — +0.8% vs GLPatch
#   Electricity (C=321) — sanity check: must NOT lose the −5.2% gain
#
# Each dataset runs:  none (baseline), -5.0 (current), -7.0 (new), -9.0 (new)
# ============================================================================

MODEL="GLPatch_Hydra"
LOGDIR="./logs/cold_init"

echo ""
echo "========== [$(date '+%H:%M:%S')] Cold Gate Init Ablation =========="
echo "gate_init = {none, -5.0, -7.0, -9.0}"
echo ""

# Format: NAME:DATA_FLAG:DATA_PATH:ENC_IN:BS:LR:SL:LL:PRED_LENS:EXTRA
DATASETS=(
  "ETTm1:ETTm1:ETTm1.csv:7:2048:0.0005:96:48:96,192,336,720:"
  "ETTh1:ETTh1:ETTh1.csv:7:2048:0.0005:96:48:96,192,336,720:"
  "Exchange:custom:exchange_rate.csv:8:32:0.00001:96:48:96,192,336,720:"
  "Electricity:custom:electricity.csv:321:256:0.005:96:48:96,192,336,720:"
)

GATE_INITS=("none" "-5.0" "-7.0" "-9.0")

for DS_INFO in "${DATASETS[@]}"; do
  IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
  IFS=',' read -ra PLS <<< "$PRED_LENS"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [$(date '+%H:%M:%S')] ${DS} (C=${ENC_IN})"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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
    ${EXTRA}"

  for GINIT in "${GATE_INITS[@]}"; do
    if [ "$GINIT" == "none" ]; then
      # Base GLPatch model — no Hydra at all
      RUN_MODEL="GLPatch"
      CV_ARGS=""
      TAG="none"
    else
      # GLPatch_Hydra with hybrid gate at specified init
      RUN_MODEL="GLPatch_Hydra"
      CV_ARGS="--cv_rank 32 --gate_type hybrid --gate_init ${GINIT}"
      TAG="gi${GINIT}"
    fi

    echo ""
    echo "  --- model=${RUN_MODEL}, gate_init=${GINIT} ---"

    for PL in "${PLS[@]}"; do
      sdir="${LOGDIR}/${DS}/${TAG}"; mkdir -p ${sdir}
      echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} gate_init=${GINIT}"
      python -u run.py --model $RUN_MODEL \
        --model_id "cold_${DS}_${PL}_${TAG}" \
        --pred_len $PL --des "cold_${TAG}" \
        ${CV_ARGS} \
        $COMMON \
        2>&1 | tee ${sdir}/${PL}.log
    done
  done
done

# ============================================================
# RESULTS SUMMARY
# ============================================================
echo ""
echo ""
echo "========== [$(date '+%H:%M:%S')] COLD INIT ABLATION RESULTS =========="
echo ""

printf "  %-12s %-8s |" "Dataset" "gate_init"
echo "     T1       T2       T3       T4      Avg"
echo "  ──────────────────────+───────────────────────────────────────────────"

for DS_INFO in "${DATASETS[@]}"; do
  IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
  IFS=',' read -ra PLS <<< "$PRED_LENS"

  for GINIT in "${GATE_INITS[@]}"; do
    if [ "$GINIT" == "none" ]; then
      TAG="none"
    else
      TAG="gi${GINIT}"
    fi

    printf "  %-12s %-8s |" "$DS" "$GINIT"
    total=0
    count=0
    for PL in "${PLS[@]}"; do
      logfile="${LOGDIR}/${DS}/${TAG}/${PL}.log"
      if [ -f "$logfile" ]; then
        mse=$(grep "mse:" "$logfile" | tail -1 | sed -n 's/.*mse:\([0-9.]*\).*/\1/p')
        if [ -n "$mse" ]; then
          printf "  %7s" $(printf "%.4f" $mse)
          total=$(python3 -c "print(${total}+${mse})")
          count=$((count+1))
        else
          printf "     N/A"
        fi
      else
        printf "     N/A"
      fi
    done
    if [ $count -gt 0 ]; then
      avg=$(python3 -c "print(f'{${total}/${count}:.4f}')")
      printf "   %7s" "$avg"
    fi
    echo ""
  done
  echo "  ──────────────────────+───────────────────────────────────────────────"
done

echo ""
echo "Expected behavior:"
echo "  • ETTm1/ETTh1/Exchange: colder init → closer to 'none' baseline"
echo "  • Electricity: should remain close to -5.0 results (gate opens via logC)"
echo ""
echo "GLPatch reference:"
echo "  ETTm1:  0.309  0.347  0.384  0.458  (avg 0.3745)"
echo "  ETTh1:  0.376  0.417  0.449  0.499  (avg 0.4353)"
echo "  Exchange: 0.082  0.175  0.334  0.877  (avg 0.3670)"
echo "  Elec:   0.157  0.158  0.179  0.216  (avg 0.1775)"
echo ""
echo "v3.2 (gate_init=-5.0) reference:"
echo "  ETTm1:  0.3085  0.3496  0.3879  0.4636  (avg 0.3774, +0.8% vs GL)"
echo "  ETTh1:  0.3763  0.4213  0.4511  0.4787  (avg 0.4319, -0.8% vs GL)"
echo "  Exchange: 0.0817  0.1762  0.3410  0.8845  (avg 0.3709, +1.0% vs GL)"
echo "  Elec:   0.1441  0.1524  0.1758  0.2001  (avg 0.1681, -5.3% vs GL)"
echo ""
echo "Target: -7.0 or -9.0 should bring ETTm1/Exchange to ≤ GLPatch"
echo "        while Electricity stays at ≈ -5% vs GL"
