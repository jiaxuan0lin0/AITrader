#!/usr/bin/env bash
set -euo pipefail

ROOT=data/experiments/msgca/20260605_cluster_train
CODE=code
PY=python3
export PYTHONPATH="$CODE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "$ROOT/logs" "$ROOT/eval" "$ROOT/ensemble"

log() {
  echo "[$(date -Is)] $*" | tee -a "$ROOT/logs/multiseed_queue.log" >&2
}

safe_variant() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
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
  local variants=(direct_theme_soft direct_theme_medium)
  for split in validation holdout; do
    local pred_path
    pred_path="$(ensure_predictions "$run_id" "$split")"
    for variant in "${variants[@]}"; do
      run_metrics "$run_id" "$split" "$pred_path" "$variant"
    done
  done
}

ensemble_split() {
  local split="$1"
  local out_dir="$ROOT/ensemble/cluster_inrank020_seed2031_2041_2051"
  local out_path="$out_dir/${split}_predictions.parquet"
  mkdir -p "$out_dir"
  if [[ ! -f "$out_path" ]]; then
    log "ensemble predictions $split"
    if [[ "$split" == "validation" ]]; then
      stdbuf -oL -eL "$PY" -m model.msgca.ensemble_predictions \
        --predictions-path "$ROOT/runs/cluster_inrank020_scratch_seed2031/validation_predictions.parquet" \
        --predictions-path "$ROOT/runs/cluster_inrank020_scratch_seed2041/validation_predictions.parquet" \
        --predictions-path "$ROOT/runs/cluster_inrank020_scratch_seed2051/validation_predictions.parquet" \
        --output-path "$out_path" > "$out_dir/${split}.ensemble.log" 2>&1
    else
      stdbuf -oL -eL "$PY" -m model.msgca.ensemble_predictions \
        --predictions-path "$ROOT/eval/cluster_inrank020_scratch_seed2031/predictions/holdout_predictions.parquet" \
        --predictions-path "$ROOT/eval/cluster_inrank020_scratch_seed2041/predictions/holdout_predictions.parquet" \
        --predictions-path "$ROOT/eval/cluster_inrank020_scratch_seed2051/predictions/holdout_predictions.parquet" \
        --output-path "$out_path" > "$out_dir/${split}.ensemble.log" 2>&1
    fi
  fi
  echo "$out_path"
}

eval_ensemble() {
  local run_id="cluster_inrank020_ensemble_seed2031_2041_2051"
  local cfg="$ROOT/runs/cluster_inrank020_scratch_seed2031/config.yaml"
  for split in validation holdout; do
    local pred_path
    pred_path="$(ensemble_split "$split")"
    local out_dir="$ROOT/eval/$run_id/$split/direct_theme_soft"
    mkdir -p "$out_dir"
    if [[ ! -f "$out_dir/competition_metrics_summary.csv" ]]; then
      log "metrics $run_id $split direct_theme_soft"
      stdbuf -oL -eL "$PY" -m model.msgca.competition_metrics \
        --config "$cfg" \
        --predictions-path "$pred_path" \
        --name "${run_id}_${split}_direct_theme_soft" \
        --output-root "$out_dir" \
        --score-variant direct_theme_soft \
        --top-n 20 \
        --daily-replace-k 5 \
        --recent-window-count 10 \
        --exclude-st \
        --exclude-bj > "$out_dir/metrics.log" 2>&1
    fi
  done
}

summarize() {
  "$PY" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("data/experiments/msgca/20260605_cluster_train")
rows = []
for path in sorted((root / "eval").glob("cluster_inrank020*/*/*/competition_metrics_summary.csv")):
    frame = pd.read_csv(path)
    rel = path.relative_to(root / "eval")
    frame.insert(0, "experiment_id", rel.parts[0])
    frame.insert(1, "eval_split", rel.parts[1])
    rows.append(frame)
if rows:
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["eval_split", "competition_score"], ascending=[True, False])
    summary = root / "eval" / "multiseed_competition_summary.csv"
    out.to_csv(summary, index=False)
    cols = [
        "eval_split",
        "experiment_id",
        "score_variant",
        "period_return",
        "period_excess_equal",
        "rolling_return_mean",
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

for run_id in cluster_inrank020_scratch_seed2041 cluster_inrank020_scratch_seed2051; do
  train_run "$run_id"
  eval_run "$run_id"
  summarize | tee -a "$ROOT/logs/multiseed_queue.log"
done

eval_run cluster_inrank020_scratch_seed2031
eval_ensemble
summarize | tee -a "$ROOT/logs/multiseed_queue.log"
log "multiseed queue done"
