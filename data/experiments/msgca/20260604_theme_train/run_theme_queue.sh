#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jiaxuanLin/AItrader/data/experiments/msgca/20260604_theme_train
CODE=/data/jiaxuanLin/AItrader/code
PY=/home/sutai/home/envs/aitrader/bin/python
CTX=/data/jiaxuanLin/AItrader/data/datasets/features/blocks/sample/strict_context_theme_sample.parquet
export PYTHONPATH="$CODE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "$ROOT/logs" "$ROOT/eval"

log() {
  echo "[$(date -Is)] $*" | tee -a "$ROOT/logs/queue.log" >&2
}

safe_variant() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

context_cache_ready() {
  "$PY" - <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq

path = Path("/data/jiaxuanLin/AItrader/data/datasets/features/blocks/sample/strict_context_theme_sample.parquet")
samples = Path("/data/jiaxuanLin/AItrader/data/datasets/processed/samples.parquet")
required = {"sample_id", "strict_theme_strength", "strict_theme_peer_ret20", "strict_theme_peer_ret60", "strict_theme_hp"}
if not path.exists():
    sys.exit(1)
pf = pq.ParquetFile(path)
sample_rows = pq.ParquetFile(samples).metadata.num_rows
if pf.metadata.num_rows != sample_rows:
    sys.exit(2)
if not required.issubset(set(pf.schema.names)):
    sys.exit(3)
sys.exit(0)
PY
}

build_context_cache() {
  if context_cache_ready; then
    log "theme context cache ready: $CTX"
    return
  fi
  log "build theme context cache: $CTX"
  stdbuf -oL -eL "$PY" -m model.msgca.build_strict_context_cache \
    --config "$ROOT/runs/theme_ctx002_scratch_seed2031/config.yaml" \
    --output "$CTX" > "$ROOT/logs/build_theme_context.log" 2>&1
  context_cache_ready
  log "theme context cache built: $CTX"
}

train_run() {
  local run_id="$1"
  local cfg="$ROOT/runs/$run_id/config.yaml"
  local run_dir="$ROOT/runs/$run_id"
  local log_path="$ROOT/logs/${run_id}.train.log"
  if [[ -f "$run_dir/validation_predictions.parquet" && -f "$run_dir/checkpoints/msgca_best.pt" ]]; then
    log "skip completed train $run_id"
    return
  fi
  log "train $run_id"
  stdbuf -oL -eL "$PY" -m model.msgca.train --config "$cfg" --device cuda > "$log_path" 2>&1
  log "train done $run_id"
}

ensure_predictions() {
  local run_id="$1"
  local split="$2"
  local cfg="$ROOT/runs/$run_id/config.yaml"
  local run_dir="$ROOT/runs/$run_id"
  local pred_dir="$ROOT/eval/$run_id/predictions"
  local pred_path="$pred_dir/${split}_predictions.parquet"
  mkdir -p "$pred_dir"
  if [[ "$split" == "validation" && -f "$run_dir/validation_predictions.parquet" ]]; then
    echo "$run_dir/validation_predictions.parquet"
    return
  fi
  if [[ ! -f "$pred_path" ]]; then
    log "checkpoint predictions $run_id $split"
    stdbuf -oL -eL "$PY" -m model.msgca.backtest \
      --config "$cfg" \
      --checkpoint "$run_dir/checkpoints/msgca_best.pt" \
      --split "$split" \
      --output-root "$pred_dir" \
      --output-prefix "$split" \
      --score-variant final_score \
      --top-n 20 \
      --daily-replace-k 5 \
      --exclude-st \
      --exclude-bj \
      --save-predictions > "$pred_dir/${split}.predict.log" 2>&1
  fi
  echo "$pred_path"
}

run_metrics() {
  local run_id="$1"
  local split="$2"
  local pred_path="$3"
  local variant="$4"
  local variant_safe
  variant_safe="$(safe_variant "$variant")"
  local cfg="$ROOT/runs/$run_id/config.yaml"
  local out_dir="$ROOT/eval/$run_id/$split/$variant_safe"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/competition_metrics_summary.csv" ]]; then
    return
  fi
  log "metrics $run_id $split $variant"
  stdbuf -oL -eL "$PY" -m model.msgca.competition_metrics \
    --config "$cfg" \
    --predictions-path "$pred_path" \
    --name "${run_id}_${split}_${variant}" \
    --output-root "$out_dir" \
    --score-variant "$variant" \
    --top-n 20 \
    --daily-replace-k 5 \
    --recent-window-count 10 \
    --exclude-st \
    --exclude-bj > "$out_dir/metrics.log" 2>&1
}

eval_run() {
  local run_id="$1"
  local variants=(direct_multihead direct_theme_soft direct_theme_medium context_theme_s2)
  for split in validation holdout; do
    local pred_path
    pred_path="$(ensure_predictions "$run_id" "$split")"
    for variant in "${variants[@]}"; do
      run_metrics "$run_id" "$split" "$pred_path" "$variant"
    done
  done
}

summarize() {
  "$PY" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/data/jiaxuanLin/AItrader/data/experiments/msgca/20260604_theme_train")
rows = []
for path in sorted((root / "eval").glob("*/*/*/competition_metrics_summary.csv")):
    frame = pd.read_csv(path)
    rel = path.relative_to(root / "eval")
    frame.insert(0, "experiment_id", rel.parts[0])
    frame.insert(1, "eval_split", rel.parts[1])
    frame.insert(2, "eval_variant_dir", rel.parts[2])
    rows.append(frame)
if rows:
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["eval_split", "competition_score"], ascending=[True, False])
    summary = root / "eval" / "combined_competition_summary.csv"
    out.to_csv(summary, index=False)
    cols = [
        "eval_split",
        "experiment_id",
        "score_variant",
        "period_return",
        "period_excess_equal",
        "rolling_return_mean",
        "rolling_excess_equal_mean",
        "latest_window_return",
        "max_drawdown",
        "competition_score",
    ]
    print(f"summary={summary}")
    print(out[cols].to_string(index=False))
else:
    print("summary=no_metrics_yet")
PY
}

build_context_cache
for run_id in theme_ctx002_scratch_seed2031 theme_ctx005_scratch_seed2031 theme_ctx008_scratch_seed2031; do
  train_run "$run_id"
  eval_run "$run_id"
  summarize | tee -a "$ROOT/logs/queue.log"
done
log "queue done"
