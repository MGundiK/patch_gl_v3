#!/bin/bash
# ============================================================================
# Sharp Gate Ablation — gate_temp τ = {3, 5}
# ============================================================================
#
# Cold init showed gate converges to same values regardless of init.
# Hypothesis: making the gate sharper (more binary) will force it to
# commit to 0 or 1, eliminating the small noise leak on low-C datasets.
#
#   τ=1 (current): sigmoid(-5) = 0.67%  — soft, allows partial mixing
#   τ=3:           sigmoid(3×-5) = sigmoid(-15) ≈ 0.00003% — very sharp
#   τ=5:           sigmoid(5×-5) = sigmoid(-25) ≈ 0% — essentially binary
#
# But the network learns the logits, not fixed at init. The real effect:
#   τ=1: gate can sit comfortably at 5% mixing (logits ≈ -3)
#   τ=3: same 5% requires logits ≈ -1, but 1% requires logits ≈ -1.5
#        → sharper transition, harder to stay in the "leak" zone
#   τ=5: even sharper — gate is essentially 0 or 1
#
# Test datasets:
#   ETTm1 (C=7)        — worst offender +0.8%
#   ETTh1 (C=7)        — mild gains, check we don't lose them
#   Exchange (C=8)      — already near-neutral
#   Electricity (C=321) — sanity: must keep −5% gains
#
# We already have τ=1 results, so only run τ=3 and τ=5.
# ============================================================================

MODEL="GLPatch_Hydra"
LOGDIR="./logs/sharp_gate"

echo ""
echo "========== [$(date '+%H:%M:%S')] Sharp Gate Ablation (τ=3, τ=5) =========="
echo ""

# Format: NAME:DATA_FLAG:DATA_PATH:ENC_IN:BS:LR:SL:LL:PRED_LENS:EXTRA
DATASETS=(
  "ETTm1:ETTm1:ETTm1.csv:7:2048:0.0005:96:48:96,192,336,720:"
  "ETTh1:ETTh1:ETTh1.csv:7:2048:0.0005:96:48:96,192,336,720:"
  "Exchange:custom:exchange_rate.csv:8:32:0.00001:96:48:96,192,336,720:"
  "Electricity:custom:electricity.csv:321:256:0.005:96:48:96,192,336,720:"
)

TEMPS=("3.0" "5.0")

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

  for TAU in "${TEMPS[@]}"; do
    TAG="t${TAU}"

    echo ""
    echo "  --- τ=${TAU} ---"

    for PL in "${PLS[@]}"; do
      sdir="${LOGDIR}/${DS}/${TAG}"; mkdir -p ${sdir}
      echo "  [$(date '+%H:%M:%S')] ${DS} pl=${PL} τ=${TAU}"
      python -u run.py --model $MODEL \
        --model_id "sharp_${DS}_${PL}_${TAG}" \
        --pred_len $PL --des "sharp_${TAG}" \
        --cv_rank 32 --gate_type hybrid --gate_init -5.0 \
        --gate_temp ${TAU} \
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
echo "========== [$(date '+%H:%M:%S')] SHARP GATE RESULTS =========="
echo ""
echo "All use gate_type=hybrid, gate_init=-5.0, cv_rank=32"
echo ""

printf "  %-12s %4s %-6s |" "Dataset" "C" "τ"
echo "     T1       T2       T3       T4      Avg"
echo "  ──────────────────────────+───────────────────────────────────────────────"

for DS_INFO in "${DATASETS[@]}"; do
  IFS=':' read -r DS DATA_FLAG DATA_PATH ENC_IN BS LR SL LL PRED_LENS EXTRA <<< "$DS_INFO"
  IFS=',' read -ra PLS <<< "$PRED_LENS"

  for TAU in "1.0" "${TEMPS[@]}"; do
    if [ "$TAU" == "1.0" ]; then
      TAG="gi-5.0"
      SRCDIR="./logs/cold_init/${DS}/${TAG}"
    else
      TAG="t${TAU}"
      SRCDIR="${LOGDIR}/${DS}/${TAG}"
    fi

    printf "  %-12s %4s %-6s |" "$DS" "$ENC_IN" "$TAU"
    total=0
    count=0
    for PL in "${PLS[@]}"; do
      logfile="${SRCDIR}/${PL}.log"
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
  echo "  ──────────────────────────+───────────────────────────────────────────────"
done

echo ""
echo "Reference (τ=1.0, gate_init=-5.0):"
echo "  ETTm1:      0.3085  0.3496  0.3879  0.4636  (avg 0.3774, +0.8% vs GL)"
echo "  ETTh1:      0.3763  0.4213  0.4511  0.4787  (avg 0.4319, -0.7% vs GL)"
echo "  Exchange:   0.0817  0.1762  0.3410  0.8845  (avg 0.3709, -0.2% vs GL)"
echo "  Electricity: 0.1441  0.1524  0.1758  0.2001  (avg 0.1681, -5.3% vs GL)"
echo ""
echo "GLPatch baseline:"
echo "  ETTm1:      0.309   0.347   0.384   0.458   (avg 0.3745)"
echo "  ETTh1:      0.376   0.417   0.449   0.499   (avg 0.4353)"
echo "  Exchange:   0.082   0.174   0.345   0.886   (avg 0.3715)"
echo "  Electricity: 0.157   0.158   0.179   0.216   (avg 0.1775)"
echo ""
echo "Target: τ>1 should push ETTm1 closer to GLPatch baseline"
echo "        while Electricity stays at ≈ -5% vs GL"
