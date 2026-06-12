#!/usr/bin/env bash
set -euo pipefail
cd /data/jiaxuanLin/AItrader/code
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "[$(date -Is)] pipeline_start job_id=20260526_093446_factor_aware_and_strong_mlp"
echo "factor_run=/data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/runs/20260526_093446_factor_aware_e5"
echo "strong_mlp_run=/data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/runs/20260526_093446_strong_factor_mlp_e5"
echo "factor_config=/data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/configs/20260526_093446_factor_aware_e5.yaml"
echo "strong_mlp_config=/data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/configs/20260526_093446_strong_factor_mlp_e5.yaml"

while ps -eo comm,args | awk '$1 ~ /^python/ && $0 ~ /FactorMiner[.]run_factor_workflow/ {found=1} END {exit !found}'; do
  echo "[$(date -Is)] waiting_for_factor_select_full"
  sleep 300
done

echo "[$(date -Is)] factor_aware_train_start"
python3 -m model.msgca.train --config /data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/configs/20260526_093446_factor_aware_e5.yaml --final-validate
echo "[$(date -Is)] factor_aware_train_done"

echo "[$(date -Is)] factor_aware_holdout_start"
python3 -m model.msgca.evaluate --config /data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/configs/20260526_093446_factor_aware_e5.yaml --checkpoint /data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/runs/20260526_093446_factor_aware_e5/checkpoints/msgca_latest.pt --split holdout --output-prefix holdout
echo "[$(date -Is)] factor_aware_holdout_done"

echo "[$(date -Is)] factor_aware_backtest_summary_start"
python3 - <<'PYFA'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from model.msgca.config import load_config
from model.msgca.backtest import write_backtest_outputs
from model.msgca.strategy import StrategyParams

run_dir = Path('/data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/runs/20260526_093446_factor_aware_e5')
cfg = load_config('/data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/configs/20260526_093446_factor_aware_e5.yaml')
params = StrategyParams(
    initial_cash=cfg.strategy.initial_cash,
    top_n=cfg.strategy.top_n,
    daily_replace_k=cfg.strategy.daily_replace_k,
    fee_rate=cfg.strategy.fee_rate,
    slippage_rate=cfg.strategy.slippage_rate,
    full_investment=cfg.strategy.full_investment,
)

def market_baseline(preds):
    frame = preds.copy()
    frame['target_trade_date'] = pd.to_datetime(frame['target_trade_date']).dt.normalize()
    daily = frame.groupby('target_trade_date')['label_next_open_return'].mean().dropna()
    total = float((1.0 + daily).prod() - 1.0) if len(daily) else float('nan')
    annual = float((1.0 + total) ** (252.0 / len(daily)) - 1.0) if len(daily) else float('nan')
    vol = float(daily.std(ddof=0)) if len(daily) else float('nan')
    sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if vol and np.isfinite(vol) and vol > 0 else float('nan')
    return {'total_return': total, 'annual_return': annual, 'sharpe': sharpe, 'day_count': int(len(daily))}

checks = {}
market = {}
for split, pred_name, prefix in [
    ('validation', 'validation_predictions.parquet', 'validation_backtest'),
    ('holdout', 'holdout_predictions.parquet', 'holdout_backtest'),
]:
    preds = pd.read_parquet(run_dir / pred_name)
    write_backtest_outputs(preds, run_dir, params, prefix=prefix)
    market[split] = market_baseline(preds)
    numeric_cols = ['y_score', 'return_pred', 'direction_prob', 'g_price', 'g_text', 'g_fundamental', 'label_next_open_return']
    nan_counts = {col: int(pd.to_numeric(preds[col], errors='coerce').isna().sum()) for col in numeric_cols if col in preds.columns}
    if any(value > 0 for value in nan_counts.values()):
        raise RuntimeError(f'NaN found in {split} predictions: {nan_counts}')
    checks[split] = {
        'rows': int(len(preds)),
        'days': int(pd.to_datetime(preds['target_trade_date']).nunique()),
        'date_min': str(pd.to_datetime(preds['target_trade_date']).min().date()),
        'date_max': str(pd.to_datetime(preds['target_trade_date']).max().date()),
        'nan_counts': nan_counts,
        'gate_mean': {
            'g_price': float(pd.to_numeric(preds['g_price'], errors='coerce').mean()),
            'g_text': float(pd.to_numeric(preds['g_text'], errors='coerce').mean()),
            'g_fundamental': float(pd.to_numeric(preds['g_fundamental'], errors='coerce').mean()),
        },
    }
summary = {
    'run_dir': str(run_dir),
    'config': '/data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/configs/20260526_093446_factor_aware_e5.yaml',
    'checkpoint': str(run_dir / 'checkpoints' / 'msgca_latest.pt'),
    'train_log': str(run_dir / 'train_log.csv'),
    'validation_metrics': json.loads((run_dir / 'validation_validation_metrics.json').read_text()),
    'holdout_metrics': json.loads((run_dir / 'holdout_validation_metrics.json').read_text()),
    'validation_backtest': json.loads((run_dir / 'validation_backtest_metrics.json').read_text()),
    'holdout_backtest': json.loads((run_dir / 'holdout_backtest_metrics.json').read_text()),
    'validation_market_equal_weight': market['validation'],
    'holdout_market_equal_weight': market['holdout'],
    'sanity': checks,
}
(run_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PYFA
echo "[$(date -Is)] factor_aware_backtest_summary_done"

echo "[$(date -Is)] strong_factor_mlp_start"
python3 /data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/scripts/20260526_093446_factor_aware_and_strong_mlp_strong_factor_mlp.py
echo "[$(date -Is)] strong_factor_mlp_done"

echo "[$(date -Is)] pipeline_done"
