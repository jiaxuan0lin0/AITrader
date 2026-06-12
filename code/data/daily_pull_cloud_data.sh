#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=load_a_share_env.sh
source "$SCRIPT_DIR/load_a_share_env.sh"

LOG_FILE="${A_SHARE_SYNC_LOG_FILE:-${DATASET_CLOUD_LOG_FILE:-$AITRADER_LOG_DIR/cloud_data_pull.log}}"
LOCK_DIR="${A_SHARE_SYNC_LOCK_DIR:-${DATASET_CLOUD_LOCK_DIR:-$AITRADER_RUNTIME_DIR/cloud_data_pull.lock}}"
PYTHON_BIN="${AITRADER_PYTHON_BIN:-python3}"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

cleanup() {
  rm -rf "$LOCK_DIR"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  log "Another cloud data pull is already running."
  exit 10
fi
trap cleanup EXIT

cd "$ROOT_DIR"

log "Cloud data pull started."
log "Target dir: ${A_SHARE_SYNC_TARGET:-${USTC_WEBDAV_TARGET:-${DATASET_CLOUD_TARGET:-$AITRADER_RAW_DATA_DIR}}}"
log "Date range: start=${A_SHARE_SYNC_START_DATE:-${DATASET_CLOUD_START_DATE:-auto}} end=${A_SHARE_SYNC_END_DATE:-${DATASET_CLOUD_END_DATE:-latest-remote}}"

"$PYTHON_BIN" data/sync_ustc_webdav.py --auto-start "$@" 2>&1 | tee -a "$LOG_FILE"

log "Cloud data pull finished successfully."
