#!/usr/bin/env bash
set -euo pipefail

AITRADER_ROOT="${AITRADER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
AITRADER_DATA_ROOT="${AITRADER_DATA_ROOT:-$AITRADER_ROOT/data}"
AITRADER_MODELS_ROOT="${AITRADER_MODELS_ROOT:-$AITRADER_DATA_ROOT/models}"
AITRADER_PYDEPS="${AITRADER_PYDEPS:-$AITRADER_ROOT/pydeps}"
MODEL_DIR="${MODEL_DIR:-$AITRADER_MODELS_ROOT/Qwen3-32B-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-news}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
QUANTIZATION="${QUANTIZATION:-awq}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -d "$AITRADER_PYDEPS" ]]; then
  export PYTHONPATH="$AITRADER_PYDEPS${PYTHONPATH:+:$PYTHONPATH}"
  export PATH="$AITRADER_PYDEPS/bin:$PATH"
fi

if command -v vllm >/dev/null 2>&1; then
  VLLM_CMD=(vllm serve)
else
  VLLM_CMD=(python3 -m vllm.entrypoints.openai.api_server --model)
fi

"${VLLM_CMD[@]}" "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --quantization "${QUANTIZATION}" \
  --dtype half \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-prefix-caching
