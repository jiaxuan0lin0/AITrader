#!/usr/bin/env bash
set -euo pipefail

cd /data/jiaxuanLin/AItrader
PY=/home/sutai/home/envs/aitrader/bin/python
CFG=/data/jiaxuanLin/AItrader/data/experiments/msgca/20260602_strategy_window/configs/stagewise_top20_h48_strategy_window.yaml
RUN_ROOT=/data/jiaxuanLin/AItrader/data/experiments/msgca/20260602_strategy_window/runs/stagewise_top20_h48_swloss_v3
LOG_DIR=/data/jiaxuanLin/AItrader/data/experiments/msgca/20260602_strategy_window/logs

mkdir -p "$LOG_DIR" "$RUN_ROOT"
PYTHONPATH=code "$PY" -m model.msgca.train --config "$CFG" --device cuda 2>&1 | tee -a "$LOG_DIR/stagewise_top20_h48_strategy_window_v3.log"
