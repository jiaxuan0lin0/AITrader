#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B-AWQ}"
AITRADER_ROOT="${AITRADER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
AITRADER_DATA_ROOT="${AITRADER_DATA_ROOT:-$AITRADER_ROOT/data}"
AITRADER_MODELS_ROOT="${AITRADER_MODELS_ROOT:-$AITRADER_DATA_ROOT/models}"
AITRADER_LOG_DIR="${AITRADER_LOG_DIR:-$AITRADER_DATA_ROOT/logs}"
MODEL_DIR="${MODEL_DIR:-$AITRADER_MODELS_ROOT/Qwen3-32B-AWQ}"
PYTHON_BIN="${AITRADER_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://mirrors.aliyun.com/pypi/simple}"
MODELSCOPE_ENDPOINT="${MODELSCOPE_ENDPOINT:-https://www.modelscope.cn}"
MAX_WORKERS="${MAX_WORKERS:-8}"
LOG_FILE="${LOG_FILE:-$AITRADER_LOG_DIR/qwen3_32b_awq_modelscope_download.log}"

"$PYTHON_BIN" -m pip install -U modelscope \
  -i "${PIP_INDEX_URL}" \
  --trusted-host mirrors.aliyun.com

mkdir -p "${MODEL_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"
modelscope download \
  --model "${MODEL_ID}" \
  --local_dir "${MODEL_DIR}" \
  --endpoint "${MODELSCOPE_ENDPOINT}" \
  --max-workers "${MAX_WORKERS}" \
  2>&1 | tee -a "${LOG_FILE}"
