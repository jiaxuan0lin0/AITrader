#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jiaxuanLin/AItrader
PY=/home/sutai/home/envs/aitrader/bin/python
EXP="$ROOT/data/experiments/msgca/20260601_r020_top10_strong_ablation"
CFG_DIR="$EXP/configs"
RUN_DIR="$EXP/runs"
QUEUE_LOG="$EXP/followup_queue.log"

run_one() {
  local name="$1"
  local cfg="$CFG_DIR/$name.yaml"
  local run="$RUN_DIR/$name"
  local log="$run/tmux_train.log"
  local ckpt="$run/checkpoints/msgca_latest.pt"
  mkdir -p "$run"
  date +"[%F %T] run_start $name"
  if [ -f "$ckpt" ]; then
    PYTHONPATH=code "$PY" -m model.msgca.train --config "$cfg" --device cuda --resume-checkpoint "$ckpt" 2>&1 | tee -a "$log"
  else
    PYTHONPATH=code "$PY" -m model.msgca.train --config "$cfg" --device cuda 2>&1 | tee -a "$log"
  fi
  date +"[%F %T] run_done $name"
}

{
  date +"[%F %T] followup_queue_start"
  for name in \
    r020_top10_topk010_vwap50 \
    r020_top10_topk010_tw2025 \
    r020_top10_topk010_twrecent
  do
    run_one "$name"
  done

  date +"[%F %T] followup_rerank_start"
  PYTHONPATH=code "$PY" -m model.msgca.rerank_loss_ablation \
    --experiment-root "$EXP" \
    --run r020_top10_topk010_vwap50 \
    --run r020_top10_topk010_tw2025 \
    --run r020_top10_topk010_twrecent \
    --output-root "$EXP/strategy_recheck/followup_weighted_top10_k3" \
    --device cuda 2>&1 | tee -a "$EXP/followup_strategy_recheck.log"
  date +"[%F %T] followup_rerank_done"
  date +"[%F %T] followup_queue_done"
} >> "$QUEUE_LOG" 2>&1
