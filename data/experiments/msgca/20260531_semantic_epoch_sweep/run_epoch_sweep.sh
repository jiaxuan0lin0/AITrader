#!/usr/bin/env bash
set -euo pipefail

cd /data/jiaxuanLin/AItrader/code

for E in 5; do
  /home/sutai/home/envs/aitrader/bin/python -m model.msgca.run_systematic_ablations \
    --base-config model/msgca/config.yaml \
    --matrix gpt_final_soft \
    --only gpt_final_upgrade_h48_proto2_topk005_softgate_train20260520 \
    --epochs "$E" \
    --job-id "20260531_semantic_final_softgate_e${E}" \
    --run-root /data/jiaxuanLin/AItrader/data/experiments/msgca/20260531_semantic_epoch_sweep/runs \
    --config-root /data/jiaxuanLin/AItrader/data/experiments/msgca/20260531_semantic_epoch_sweep/configs \
    --force
done
