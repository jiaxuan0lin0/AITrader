#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
AITRADER_ROOT="${AITRADER_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)}"
AITRADER_DATA_ROOT="${AITRADER_DATA_ROOT:-$AITRADER_ROOT/data}"
AITRADER_LOG_DIR="${AITRADER_LOG_DIR:-$AITRADER_DATA_ROOT/logs}"
PYTHON_BIN="${AITRADER_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
CHECKPOINT_SIZE="${CHECKPOINT_SIZE:-1000}"
CONCURRENCY="${CONCURRENCY:-20}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
USE_GUIDED_JSON="${USE_GUIDED_JSON:-0}"
LOG_FILE="${LOG_FILE:-$AITRADER_LOG_DIR/news_score_all.log}"
SINCE="${SINCE:-}"
UNTIL="${UNTIL:-}"
LIMIT="${LIMIT:-}"

mkdir -p "$(dirname "${LOG_FILE}")"
cd "${PROJECT_DIR}"

args=(
  --checkpoint-size "${CHECKPOINT_SIZE}"
  --concurrency "${CONCURRENCY}"
  --request-batch-size "${REQUEST_BATCH_SIZE}"
  --max-tokens "${MAX_TOKENS}"
)
if [[ "${USE_GUIDED_JSON}" == "1" || "${USE_GUIDED_JSON}" == "true" || "${USE_GUIDED_JSON}" == "TRUE" ]]; then
  args+=(--use-guided-json)
fi
if [[ -n "${SINCE}" ]]; then
  args+=(--since "${SINCE}")
fi
if [[ -n "${UNTIL}" ]]; then
  args+=(--until "${UNTIL}")
fi
if [[ -n "${LIMIT}" ]]; then
  args+=(--limit "${LIMIT}")
fi
args+=("$@")

{
  echo "news scoring started at $(date -Is)"
  echo "project_dir=${PROJECT_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "checkpoint_size=${CHECKPOINT_SIZE}"
  echo "concurrency=${CONCURRENCY}"
  echo "request_batch_size=${REQUEST_BATCH_SIZE}"
  echo "max_tokens=${MAX_TOKENS}"
  echo "use_guided_json=${USE_GUIDED_JSON}"
  if [[ -n "${SINCE}" ]]; then
    echo "since=${SINCE}"
  fi
  if [[ -n "${UNTIL}" ]]; then
    echo "until=${UNTIL}"
  fi
  if [[ -n "${LIMIT}" ]]; then
    echo "limit=${LIMIT}"
  fi
  echo "args=${args[*]}"
} | tee -a "${LOG_FILE}"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m FactorMiner.news_scoring.score_news_items \
  "${args[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
