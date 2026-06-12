#!/usr/bin/env bash
set -euo pipefail

ROOT="${AITRADER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
AITRADER_ROOT="${AITRADER_ROOT:-$(cd "$ROOT/.." && pwd)}"
AITRADER_DATA_ROOT="${AITRADER_DATA_ROOT:-$AITRADER_ROOT/data}"
AITRADER_DATASETS_ROOT="${AITRADER_DATASETS_ROOT:-$AITRADER_DATA_ROOT/datasets}"
AITRADER_LOG_DIR="${AITRADER_LOG_DIR:-$AITRADER_DATA_ROOT/logs}"
PYTHON_BIN="${AITRADER_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-aitrader}"

SELECT_SINCE="${SELECT_SINCE:-2016-01-05}"
SELECT_UNTIL="${SELECT_UNTIL:-2025-09-30}"
CORR_ROW_LIMIT="${CORR_ROW_LIMIT:-0}"
BLOCKS="${BLOCKS:-all}"
SOURCE_EVALUATION_DIR="${SOURCE_EVALUATION_DIR:-$AITRADER_DATASETS_ROOT/factors/evaluation/final}"
EVALUATION_DIR="${EVALUATION_DIR:-$AITRADER_DATASETS_ROOT/factors/evaluation/experiment/select_${SELECT_SINCE//-/}_${SELECT_UNTIL//-/}_slice}"
LOG_DIR="${LOG_DIR:-$AITRADER_LOG_DIR}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/factorminer_select_${SELECT_SINCE//-/}_${SELECT_UNTIL//-/}_slice.log}"
PREPARE_REVIEW="${PREPARE_REVIEW:-1}"
REVIEW_PROFILE="${REVIEW_PROFILE:-research}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/root/miniconda3/etc/profile.d/conda.sh
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

cmd=(
  "$PYTHON_BIN" -m FactorMiner.run_factor_workflow
  --mode select
  --select-engine slice
  --source-evaluation-dir "$SOURCE_EVALUATION_DIR"
  --evaluation-dir "$EVALUATION_DIR"
  --select-since "$SELECT_SINCE"
  --select-until "$SELECT_UNTIL"
  --blocks "$BLOCKS"
  --corr-row-limit "$CORR_ROW_LIMIT"
  --review-profile "$REVIEW_PROFILE"
)

if [[ "$PREPARE_REVIEW" == "1" ]]; then
  cmd+=(--prepare-review)
fi

cmd+=("$@")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\nLog: %s\n' "$LOG_PATH"
"${cmd[@]}" 2>&1 | tee "$LOG_PATH"
