#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B-AWQ}"
AITRADER_ROOT="${AITRADER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
AITRADER_DATA_ROOT="${AITRADER_DATA_ROOT:-$AITRADER_ROOT/data}"
AITRADER_MODELS_ROOT="${AITRADER_MODELS_ROOT:-$AITRADER_DATA_ROOT/models}"
MODEL_DIR="${MODEL_DIR:-$AITRADER_MODELS_ROOT/Qwen3-32B-AWQ}"
PYTHON_BIN="${AITRADER_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://mirrors.aliyun.com/pypi/simple}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

"$PYTHON_BIN" -m pip install -U huggingface_hub hf_transfer \
  -i "${PIP_INDEX_URL}" \
  --trusted-host mirrors.aliyun.com

export HF_ENDPOINT
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p "${MODEL_DIR}"
hf download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}" \
  --max-workers 8
