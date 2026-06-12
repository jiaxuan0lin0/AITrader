from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.msgca.backtest import write_backtest_outputs
from model.msgca.config import load_config, write_resolved_config
from model.msgca.dataset import DayBatchSampler, build_datasets, build_train_validation_datasets, collate_msgca_batch
from model.msgca.losses import LossWeights, msgca_loss, set_torch_seed
from model.msgca.metrics import write_evaluation_outputs
from model.msgca.modules import StrongFactorMLP
from model.msgca.strategy import StrategyParams

CONFIG_PATH = Path('/data/jiaxuanLin/AItrader/data/experiments/msgca/20260526_factor_aware_vs_mlp/configs/20260526_093446_strong_factor_mlp_e5.yaml')


def log(message: str) -> None:
    print(f"[{pd.Timestamp.now(tz='UTC').isoformat()}] strong_mlp_{message}", flush=True)


def strategy_params(config):
    return StrategyParams(
        initial_cash=config.strategy.initial_cash,
        top_n=config.strategy.top_n,
        daily_replace_k=config.strategy.daily_replace_k,
        fee_rate=config.strategy.fee_rate,
        slippage_rate=config.strategy.slippage_rate,
        full_investment=config.strategy.full_investment,
    )


def train_model(config, train_dataset, layout, device):
    model = StrongFactorMLP(
        text_features=len(layout.text_columns),
        fundamental_features=len(layout.fundamental_columns),
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=config.train.amp and device.type == 'cuda')
    weights = LossWeights(
        rank=config.train.rank_loss_weight,
        return_mse=config.train.return_loss_weight,
        direction_bce=config.train.direction_loss_weight,
        topk_return=config.train.topk_return_loss_weight,
        topk_temperature=config.train.topk_temperature,
        gate_entropy=config.train.gate_entropy_weight,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=DayBatchSampler(train_dataset.samples, batch_days=config.train.batch_days, shuffle=True, seed=config.train.seed),
        collate_fn=collate_msgca_batch,
    )
    logs = []
    for epoch in range(1, config.train.epochs + 1):
        started = time.monotonic()
        log(f'epoch_start epoch={epoch}')
        model.train()
        losses = []
        for batch in loader:
            tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=config.train.amp and device.type == 'cuda'):
                output = model(
                    tensors['text_features'],
                    tensors['text_mask'],
                    tensors['fundamental_features'],
                    tensors['fundamental_mask'],
                )
                loss, _ = msgca_loss(
                    output,
                    tensors['label_next_open_return'],
                    tensors['label_next_vwap_return'],
                    tensors['label_direction'],
                    batch['target_trade_date'],
                    weights,
                    config.train.max_pairs_per_day,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        train_loss = sum(losses) / max(len(losses), 1)
        logs.append({'epoch': epoch, 'train_loss': train_loss})
        log(f'epoch_done epoch={epoch} elapsed_sec={time.monotonic() - started:.1f} train_loss={train_loss}')
    return model, logs


@torch.no_grad()
def predict_model(model, dataset, config, device):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_sampler=DayBatchSampler(dataset.samples, batch_days=config.train.batch_days, shuffle=False),
        collate_fn=collate_msgca_batch,
    )
    frames = []
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        output = model(
            tensors['text_features'],
            tensors['text_mask'],
            tensors['fundamental_features'],
            tensors['fundamental_mask'],
        )
        frame = pd.DataFrame({
            'sample_id': batch['sample_id'],
            'stock_code': batch['stock_code'],
            'stock_name': batch.get('stock_name', [''] * len(output.y_score)),
            'industry': batch.get('industry', [''] * len(output.y_score)),
            'target_trade_date': pd.to_datetime(batch['target_trade_date']),
            'y_score': output.y_score.detach().cpu().numpy(),
            'return_pred': output.return_pred.detach().cpu().numpy(),
            'direction_prob': torch.sigmoid(output.direction_logit).detach().cpu().numpy(),
            'g_price': output.g_price.detach().cpu().numpy(),
            'g_text': output.g_text.detach().cpu().numpy(),
            'g_fundamental': output.g_fundamental.detach().cpu().numpy(),
            'label_next_open_return': batch['label_next_open_return'].detach().cpu().numpy(),
            'label_next_vwap_return': batch['label_next_vwap_return'].detach().cpu().numpy(),
        })
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not predictions.empty:
        predictions['rank'] = predictions.groupby('target_trade_date')['y_score'].rank(method='first', ascending=False)
    return predictions


def market_baseline(predictions):
    frame = predictions.copy()
    frame['target_trade_date'] = pd.to_datetime(frame['target_trade_date']).dt.normalize()
    daily = frame.groupby('target_trade_date')['label_next_open_return'].mean().dropna()
    total = float((1.0 + daily).prod() - 1.0) if len(daily) else float('nan')
    annual = float((1.0 + total) ** (252.0 / len(daily)) - 1.0) if len(daily) else float('nan')
    vol = float(daily.std(ddof=0)) if len(daily) else float('nan')
    sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if vol and np.isfinite(vol) and vol > 0 else float('nan')
    return {'total_return': total, 'annual_return': annual, 'sharpe': sharpe, 'day_count': int(len(daily))}


def write_summary(run_dir, validation_predictions, holdout_predictions):
    summary = {
        'run_dir': str(run_dir),
        'config': str(CONFIG_PATH),
        'checkpoint': str(run_dir / 'checkpoints' / 'strong_factor_mlp_latest.pt'),
        'validation_metrics': json.loads((run_dir / 'validation_validation_metrics.json').read_text()),
        'holdout_metrics': json.loads((run_dir / 'holdout_validation_metrics.json').read_text()),
        'validation_backtest': json.loads((run_dir / 'validation_backtest_metrics.json').read_text()),
        'holdout_backtest': json.loads((run_dir / 'holdout_backtest_metrics.json').read_text()),
        'validation_market_equal_weight': market_baseline(validation_predictions),
        'holdout_market_equal_weight': market_baseline(holdout_predictions),
        'sanity': {},
    }
    for split, preds in [('validation', validation_predictions), ('holdout', holdout_predictions)]:
        numeric_cols = ['y_score', 'return_pred', 'direction_prob', 'g_price', 'g_text', 'g_fundamental', 'label_next_open_return']
        nan_counts = {col: int(pd.to_numeric(preds[col], errors='coerce').isna().sum()) for col in numeric_cols if col in preds.columns}
        if any(value > 0 for value in nan_counts.values()):
            raise RuntimeError(f'NaN found in {split} predictions: {nan_counts}')
        summary['sanity'][split] = {
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
    (run_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    started = time.monotonic()
    config = load_config(CONFIG_PATH)
    set_torch_seed(config.train.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = config.paths.output_root
    config.ensure_output_dirs()
    write_resolved_config(config, run_dir / 'config.resolved.yaml')
    log(f'run_start output_root={run_dir} device={device}')
    log('build_train_validation_start')
    train_dataset, valid_dataset, layout, _ = build_train_validation_datasets(config, allow_smoke_fallback=False)
    log(f'build_train_validation_done elapsed_sec={time.monotonic() - started:.1f} train_rows={len(train_dataset)} valid_rows={len(valid_dataset)} text={len(layout.text_columns)} fundamental={len(layout.fundamental_columns)}')
    model, logs = train_model(config, train_dataset, layout, device)
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / 'checkpoints' / 'strong_factor_mlp_latest.pt'
    torch.save({'model': model.state_dict(), 'config': config.to_dict(), 'layout': layout.__dict__}, checkpoint)
    pd.DataFrame(logs).to_csv(run_dir / 'train_log.csv', index=False)
    log('validation_predict_start')
    validation_predictions = predict_model(model, valid_dataset, config, device)
    write_evaluation_outputs(validation_predictions, run_dir, prefix='validation')
    write_backtest_outputs(validation_predictions, run_dir, strategy_params(config), prefix='validation_backtest')
    log('holdout_build_start')
    holdout_dataset, _, _ = build_datasets(config, split='holdout', allow_smoke_fallback=False)
    log(f'holdout_build_done rows={len(holdout_dataset)} elapsed_sec={time.monotonic() - started:.1f}')
    holdout_predictions = predict_model(model, holdout_dataset, config, device)
    write_evaluation_outputs(holdout_predictions, run_dir, prefix='holdout')
    write_backtest_outputs(holdout_predictions, run_dir, strategy_params(config), prefix='holdout_backtest')
    write_summary(run_dir, validation_predictions, holdout_predictions)
    log(f'run_done elapsed_sec={time.monotonic() - started:.1f}')


if __name__ == '__main__':
    main()
