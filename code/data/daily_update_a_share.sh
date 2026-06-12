#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=load_a_share_env.sh
source "$SCRIPT_DIR/load_a_share_env.sh"

LOG_FILE="${A_SHARE_LOG_FILE:-$AITRADER_LOG_DIR/a_share_pipeline_daily.log}"
LOCK_DIR="${A_SHARE_LOCK_DIR:-$AITRADER_RUNTIME_DIR/a_share_daily_update.lock}"
RAW_DIR_DEFAULT="$AITRADER_RAW_DATA_DIR"
OUTPUT_DIR_DEFAULT="$AITRADER_DATASETS_ROOT"
SOURCE_URL_DEFAULT="https://pan.ustc.edu.cn/seafdav/"
PYTHON_BIN="${AITRADER_PYTHON_BIN:-python3}"

mkdir -p "$(dirname "$LOG_FILE")"

RAW_DIR="${USTC_WEBDAV_TARGET:-$RAW_DIR_DEFAULT}"
OUTPUT_DIR="${A_SHARE_OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"
SOURCE_URL="${USTC_WEBDAV_URL:-$SOURCE_URL_DEFAULT}"

log() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

cleanup() {
  rm -rf "$LOCK_DIR"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  log "Another daily update process is already running."
  exit 10
fi
trap cleanup EXIT

cd "$ROOT_DIR"

log "Daily update started."
log "WebDAV source: $SOURCE_URL"
log "Raw dir: $RAW_DIR"
log "Output dir: $OUTPUT_DIR"
if [[ -n "${A_SHARE_SYNC_START_DATE:-}" || -n "${A_SHARE_SYNC_END_DATE:-}" || -n "${A_SHARE_SYNC_AUTO_START:-}" ]]; then
  log "Sync date range: start=${A_SHARE_SYNC_START_DATE:-auto} end=${A_SHARE_SYNC_END_DATE:-latest-remote} auto_start=${A_SHARE_SYNC_AUTO_START:-0}"
fi

"$PYTHON_BIN" data/sync_ustc_webdav.py \
  --source-url "$SOURCE_URL" \
  --target-dir "$RAW_DIR" \
  --log-level "${A_SHARE_SYNC_LOG_LEVEL:-INFO}" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

"$PYTHON_BIN" data/a_share_pipeline.py \
  --raw-dir "$RAW_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --log-level "${A_SHARE_PIPELINE_LOG_LEVEL:-INFO}" 2>&1 | tee -a "$LOG_FILE"

log "Daily update finished successfully."
