#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/jiaxuanLin/AItrader
PY=/home/sutai/home/envs/aitrader/bin/python
CFG_DIR="$ROOT/data/experiments/msgca/20260601_loss_ablation/configs"
RUN_DIR="$ROOT/data/experiments/msgca/20260601_loss_ablation/runs"
QUEUE_LOG="$ROOT/data/experiments/msgca/20260601_loss_ablation/queue.log"

run_one() {
  local name="$1"
  local cfg="$CFG_DIR/$name.yaml"
  local run="$RUN_DIR/$name"
  local log="$run/tmux_train.log"
  local ckpt="$run/checkpoints/msgca_latest.pt"

  mkdir -p "$run"

  date +"[%F %T] run_start $name"
  if [ -f "$ckpt" ]; then
    PYTHONPATH=code "$PY" -m model.msgca.train \
      --config "$cfg" \
      --device cuda \
      --resume-checkpoint "$ckpt" 2>&1 | tee -a "$log"
  else
    PYTHONPATH=code "$PY" -m model.msgca.train \
      --config "$cfg" \
      --device cuda 2>&1 | tee -a "$log"
  fi
  date +"[%F %T] run_done $name"
}

select_best_r() {
  "$PY" - "$RUN_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
names = ["r020_topk005_open", "r030_topk005_open", "r050_topk005_open"]
best_name = None
best_value = None
for name in names:
    path = run_dir / name / "checkpoints" / "msgca_best.json"
    if not path.exists():
        raise SystemExit(f"missing best checkpoint metadata: {path}")
    value = float(json.loads(path.read_text())["value"])
    if best_value is None or value > best_value:
        best_name = name
        best_value = value

match = re.match(r"^(r\d+)_", best_name or "")
if match is None:
    raise SystemExit(f"cannot parse best r from {best_name}")
print(match.group(1))
PY
}

make_variant_configs() {
  local best_r="$1"
  "$PY" - "$CFG_DIR" "$RUN_DIR" "$best_r" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_dir = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
best_r = sys.argv[3]
base_name = f"{best_r}_topk005_open"
base_path = cfg_dir / f"{base_name}.yaml"
if not base_path.exists():
    raise SystemExit(f"missing base config: {base_path}")

with base_path.open("r", encoding="utf-8") as fh:
    base = yaml.safe_load(fh)

variants = {
    f"{best_r}_topk010_open": {
        "train": {
            "topk_return_loss_weight": 0.10,
            "return_secondary_weight": 0.0,
            "topk_secondary_weight": 0.0,
        },
    },
    f"{best_r}_topk005_blend50": {
        "train": {
            "topk_return_loss_weight": 0.05,
            "return_secondary_weight": 0.5,
            "topk_secondary_weight": 0.5,
        },
    },
    f"{best_r}_topk005_vwap": {
        "train": {
            "topk_return_loss_weight": 0.05,
            "return_secondary_weight": 1.0,
            "topk_secondary_weight": 1.0,
        },
    },
}

for name, updates in variants.items():
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["paths"]["model_root"] = str(run_dir / name)
    for section, values in updates.items():
        cfg.setdefault(section, {}).update(values)
    path = cfg_dir / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=False)
    print(name)
PY
}

{
  date +"[%F %T] queue_start"

  while tmux has-session -t msgca_r010_e20 2>/dev/null; do
    date +"[%F %T] waiting_for_msgca_r010_e20"
    sleep 60
  done

  for name in r020_topk005_open r030_topk005_open; do
    run_one "$name"
  done

  date +"[%F %T] rerank_start best_weighted_top10_k3"
  PYTHONPATH=code "$PY" -m model.msgca.rerank_loss_ablation \
    --experiment-root "$ROOT/data/experiments/msgca/20260601_loss_ablation" \
    --run r010_topk005_open \
    --run r020_topk005_open \
    --run r030_topk005_open \
    --device cuda 2>&1 | tee -a "$ROOT/data/experiments/msgca/20260601_loss_ablation/strategy_recheck.log"
  date +"[%F %T] rerank_done best_weighted_top10_k3"

  if [ "${RUN_R050:-0}" = "1" ]; then
    run_one r050_topk005_open
  else
    date +"[%F %T] r050_skipped_pending_strategy_recheck"
  fi

  if [ "${RUN_VARIANTS:-0}" != "1" ]; then
    date +"[%F %T] variants_skipped_pending_r_selection"
    date +"[%F %T] queue_done"
    exit 0
  fi

  best_r="$(select_best_r)"
  date +"[%F %T] best_r_selected $best_r"

  while read -r name; do
    run_one "$name"
  done < <(make_variant_configs "$best_r")

  if [ "$best_r" != "r020" ]; then
    date +"[%F %T] run_reference_variant r020_topk010_open"
    "$PY" - "$CFG_DIR" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_dir = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
with (cfg_dir / "r020_topk005_open.yaml").open("r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
cfg["paths"]["model_root"] = str(run_dir / "r020_topk010_open")
cfg["train"]["topk_return_loss_weight"] = 0.10
cfg["train"]["return_secondary_weight"] = 0.0
cfg["train"]["topk_secondary_weight"] = 0.0
with (cfg_dir / "r020_topk010_open.yaml").open("w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=False)
PY
    run_one r020_topk010_open
  else
    date +"[%F %T] reference_variant_skipped best_r_is_r020"
  fi

  date +"[%F %T] queue_done"
} >> "$QUEUE_LOG" 2>&1
