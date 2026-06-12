#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jiaxuanLin/AItrader
EXP="$ROOT/data/experiments/msgca/20260601_r020_top10_strong_ablation"
LOG="$EXP/watch_queue.log"
SESSION=msgca_r020_top10_strong
RESTART_COUNT_FILE="$EXP/watch_restart_count.txt"
MAX_RESTARTS=1

while true; do
  {
    date +"[%F %T] watch_tick"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "session=running"
    else
      echo "session=missing"
    fi
    ps -eo pid,ppid,stat,pcpu,pmem,cmd | rg 'model.msgca.train|rerank_loss_ablation|20260601_r020_top10_strong_ablation/run_queue' || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    tail -n 5 "$EXP/queue.log" 2>/dev/null || true
    if ! tmux has-session -t "$SESSION" 2>/dev/null && ! tail -n 80 "$EXP/queue.log" 2>/dev/null | rg -q 'queue_done'; then
      echo "ALERT queue session missing before queue_done"
      restarts=0
      if [ -f "$RESTART_COUNT_FILE" ]; then
        restarts="$(cat "$RESTART_COUNT_FILE" 2>/dev/null || echo 0)"
      fi
      if [ "${restarts:-0}" -lt "$MAX_RESTARTS" ]; then
        next=$((restarts + 1))
        echo "$next" > "$RESTART_COUNT_FILE"
        echo "action=restart_queue attempt=$next"
        tmux new-session -d -s "$SESSION" "cd $ROOT && bash $EXP/run_queue.sh"
      else
        echo "action=restart_skipped max_restarts=$MAX_RESTARTS"
      fi
    fi
    echo
  } >> "$LOG" 2>&1
  sleep 60
done
