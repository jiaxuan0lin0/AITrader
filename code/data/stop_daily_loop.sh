#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=load_a_share_env.sh
source "$SCRIPT_DIR/load_a_share_env.sh"

PID_FILE="${A_SHARE_LOOP_PID_FILE:-$AITRADER_RUNTIME_DIR/a_share_loop.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found."
  exit 0
fi

pid="$(cat "$PID_FILE")"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped daily loop PID $pid"
else
  echo "Process not running."
fi

rm -f "$PID_FILE"
