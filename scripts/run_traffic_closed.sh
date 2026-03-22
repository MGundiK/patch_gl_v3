#!/bin/bash
# run_traffic_closed.sh
# Traffic (C=862) with gate_init=-10.0 — gate forced near-closed.
#
# Rationale: Traffic channels share temporal macro-patterns (rush hours,
# weekends) which produce high mean_cos (~0.71) and pca_frac (~0.62),
# indistinguishable from Electricity. But mixing is contraindicated because
# the residual cross-channel signal is sensor-heterogeneous noise, not
# generalisable structure. gate_init=-10.0 sets sigmoid(-10)=0.005% —
# effectively zero mixing regardless of what the data signal does.
#
# Requires: layers/hydra_attention.py = hydra_attention_v32g.py
# Logs → logs/glpatch_hydra_v32g_traffic_closed/

set -e
mkdir -p logs/glpatch_hydra_v32g_traffic_closed

for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 \
  --model_id traffic_${pred_len}_v32g_closed \
  --model GLPatch_Hydra \
  --data custom \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 862 \
  --patch_len 16 \
  --stride 8 \
  --padding_patch end \
  --ma_type ema \
  --alpha 0.3 \
  --revin 1 \
  --cv_rank 32 \
  --gate_type hybrid \
  --gate_init -10.0 \
  --gate_temp 1.0 \
  --batch_size 96 \
  --learning_rate 0.005 \
  --lradj sigmoid \
  --train_epochs 100 \
  --patience 10 \
  --des Exp \
  2>&1 | tee logs/glpatch_hydra_v32g_traffic_closed/traffic_${pred_len}.log
done

echo "=== Traffic closed-gate complete ==="
