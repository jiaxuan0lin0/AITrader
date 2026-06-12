#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=load_a_share_env.sh
source "$SCRIPT_DIR/load_a_share_env.sh"

RUN_AT="${A_SHARE_DAILY_TIME:-08:00}"
STATE_FILE="${A_SHARE_STATE_FILE:-$AITRADER_RUNTIME_DIR/a_share_last_run.txt}"
LOG_FILE="${A_SHARE_LOOP_LOG_FILE:-$AITRADER_LOG_DIR/a_share_loop.log}"
SLEEP_SECONDS="${A_SHARE_LOOP_SLEEP_SECONDS:-30}"

mkdir -p "$(dirname "$STATE_FILE")" "$(dirname "$LOG_FILE")"
cd "$ROOT_DIR"

log() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

log "Daily loop started. Run time=$RUN_AT"

while true; do
  today="$(date '+%F')"
  now_hm="$(date '+%H:%M')"
  last_run="$(cat "$STATE_FILE" 2>/dev/null || true)"

  if [[ "$now_hm" == "$RUN_AT" && "$last_run" != "$today" ]]; then
    log "Triggering daily update."
    if bash data/daily_update_a_share.sh >> "$LOG_FILE" 2>&1; then
      printf '%s\n' "$today" > "$STATE_FILE"
      log "Daily update completed."
    else
      log "Daily update failed. Will retry on next loop within the same minute until success."
    fi
  fi

  sleep "$SLEEP_SECONDS"
done
