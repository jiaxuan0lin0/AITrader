#!/usr/bin/env bash
set -euo pipefail

export SELECT_SINCE="${SELECT_SINCE:-2016-01-05}"
export SELECT_UNTIL="${SELECT_UNTIL:-2026-05-20}"
export REVIEW_PROFILE="${REVIEW_PROFILE:-competition}"
ROOT="${AITRADER_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
AITRADER_ROOT="${AITRADER_ROOT:-$(cd "$ROOT/.." && pwd)}"
AITRADER_DATA_ROOT="${AITRADER_DATA_ROOT:-$AITRADER_ROOT/data}"
AITRADER_DATASETS_ROOT="${AITRADER_DATASETS_ROOT:-$AITRADER_DATA_ROOT/datasets}"
AITRADER_LOG_DIR="${AITRADER_LOG_DIR:-$AITRADER_DATA_ROOT/logs}"
export EVALUATION_DIR="${EVALUATION_DIR:-$AITRADER_DATASETS_ROOT/factors/evaluation/final}"
export LOG_PATH="${LOG_PATH:-$AITRADER_LOG_DIR/factorminer_select_${SELECT_SINCE//-/}_${SELECT_UNTIL//-/}_competition.log}"

exec "$(dirname "$0")/run_train_selection_slice.sh" "$@"
