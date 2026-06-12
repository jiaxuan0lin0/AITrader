#!/usr/bin/env bash
set -euo pipefail
cd /data/jiaxuanLin/AItrader/code
export PYTHONUNBUFFERED=1
export MSGCA_RUN_DIR="/data/jiaxuanLin/AItrader/data/experiments/msgca/20260525_noleak_baseline/runs/20260525_174109_noleak_selected_e5"
export MSGCA_CONFIG="/data/jiaxuanLin/AItrader/data/experiments/msgca/20260525_noleak_baseline/configs/20260525_174109_noleak_selected_e5.yaml"
export MSGCA_CHECKPOINT="$MSGCA_RUN_DIR/checkpoints/msgca_latest.pt"

echo "[$(date -Is)] MSGCA no-leak baseline started"
echo "run_dir=$MSGCA_RUN_DIR"
echo "config=$MSGCA_CONFIG"
python3 - <<'PY_PREFLIGHT'
from model.msgca.config import load_config
from model.msgca.feature_set import load_selected_features
cfg = load_config('/data/jiaxuanLin/AItrader/data/experiments/msgca/20260525_noleak_baseline/configs/20260525_174109_noleak_selected_e5.yaml')
sel = load_selected_features(cfg.paths.evaluation_dir, allow_smoke_fallback=False)
print('preflight_evaluation_dir=', cfg.paths.evaluation_dir)
print('preflight_selected_path=', sel.source_path)
print('preflight_selected_mode=', sel.mode)
print('preflight_selected_count=', len(sel.selected_features))
print('preflight_train=', cfg.data.train_start, cfg.data.train_end)
print('preflight_validation=', cfg.data.validation_start, cfg.data.validation_end)
print('preflight_holdout_start=', cfg.data.holdout_start)
PY_PREFLIGHT

echo "[$(date -Is)] training_start"
python3 -m model.msgca.train --config "$MSGCA_CONFIG"
echo "[$(date -Is)] training_done"

echo "[$(date -Is)] holdout_evaluate_start"
python3 -m model.msgca.evaluate \
  --config "$MSGCA_CONFIG" \
  --checkpoint "$MSGCA_CHECKPOINT" \
  --split holdout \
  --output-prefix holdout
echo "[$(date -Is)] holdout_evaluate_done"

echo "[$(date -Is)] backtest_and_sanity_start"
python3 - <<'PY_BACKTEST'
from pathlib import Path
import json
import pandas as pd
from model.msgca.config import load_config
from model.msgca.backtest import write_backtest_outputs
from model.msgca.strategy import StrategyParams

run_dir = Path('/data/jiaxuanLin/AItrader/data/experiments/msgca/20260525_noleak_baseline/runs/20260525_174109_noleak_selected_e5')
cfg = load_config('/data/jiaxuanLin/AItrader/data/experiments/msgca/20260525_noleak_baseline/configs/20260525_174109_noleak_selected_e5.yaml')
params = StrategyParams(
    initial_cash=cfg.strategy.initial_cash,
    top_n=cfg.strategy.top_n,
    daily_replace_k=cfg.strategy.daily_replace_k,
    fee_rate=cfg.strategy.fee_rate,
    slippage_rate=cfg.strategy.slippage_rate,
    full_investment=cfg.strategy.full_investment,
)
checks = {}
for split, pred_name, prefix in [
    ('validation', 'validation_predictions.parquet', 'validation_backtest'),
    ('holdout', 'holdout_predictions.parquet', 'holdout_backtest'),
]:
    pred_path = run_dir / pred_name
    preds = pd.read_parquet(pred_path)
    numeric_cols = ['y_score', 'return_pred', 'direction_prob', 'g_price', 'g_text', 'g_fundamental', 'label_next_open_return']
    nan_counts = {col: int(pd.to_numeric(preds[col], errors='coerce').isna().sum()) for col in numeric_cols if col in preds.columns}
    checks[split] = {
        'prediction_path': str(pred_path),
        'rows': int(len(preds)),
        'days': int(pd.to_datetime(preds['target_trade_date']).nunique()),
        'nan_counts': nan_counts,
        'gate_mean': {
            'g_price': float(pd.to_numeric(preds['g_price'], errors='coerce').mean()),
            'g_text': float(pd.to_numeric(preds['g_text'], errors='coerce').mean()),
            'g_fundamental': float(pd.to_numeric(preds['g_fundamental'], errors='coerce').mean()),
        },
    }
    if any(v > 0 for v in nan_counts.values()):
        raise RuntimeError(f'NaN found in {split} predictions: {nan_counts}')
    write_backtest_outputs(preds, run_dir, params, prefix=prefix)

validation_metrics_path = run_dir / 'validation_metrics.json'
summary = {
    'run_dir': str(run_dir),
    'config': '/data/jiaxuanLin/AItrader/data/experiments/msgca/20260525_noleak_baseline/configs/20260525_174109_noleak_selected_e5.yaml',
    'checkpoint': str(run_dir / 'checkpoints' / 'msgca_latest.pt'),
    'validation_metrics': json.loads(validation_metrics_path.read_text()),
    'holdout_metrics': json.loads((run_dir / 'holdout_validation_metrics.json').read_text()),
    'validation_backtest': json.loads((run_dir / 'validation_backtest_metrics.json').read_text()),
    'holdout_backtest': json.loads((run_dir / 'holdout_backtest_metrics.json').read_text()),
    'sanity': checks,
}
(run_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY_BACKTEST

echo "[$(date -Is)] backtest_and_sanity_done"
echo "[$(date -Is)] MSGCA no-leak baseline finished"
