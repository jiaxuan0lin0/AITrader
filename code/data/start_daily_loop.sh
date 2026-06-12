#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=load_a_share_env.sh
source "$SCRIPT_DIR/load_a_share_env.sh"

PID_FILE="${A_SHARE_LOOP_PID_FILE:-$AITRADER_RUNTIME_DIR/a_share_loop.pid}"
LOG_FILE="${A_SHARE_LOOP_LOG_FILE:-$AITRADER_LOG_DIR/a_share_loop.log}"

mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
cd "$ROOT_DIR"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Daily loop already running with PID $pid"
    exit 0
  fi
fi

nohup bash data/run_daily_loop.sh >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Started daily loop with PID $(cat "$PID_FILE")"
