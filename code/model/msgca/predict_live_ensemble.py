from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from model.msgca.backtest import strategy_params_from_config
from model.msgca.config import MSGCAConfig, load_config
from model.msgca.ensemble_predictions import ensemble_predictions
from model.msgca.inference import evaluate_checkpoint
from model.msgca.strategy import prepare_strategy_predictions, write_competition_signals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate live MSGCA competition signals from a multi-seed ensemble.")
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--target-date", required=True, help="YYYY-MM-DD target_trade_date to predict.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--positions-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--daily-replace-k", type=int, default=None)
    parser.add_argument("--score-variant", default=None)
    parser.add_argument("--score-weight-y", type=float, default=None)
    parser.add_argument("--score-weight-return", type=float, default=None)
    parser.add_argument("--score-weight-direction", type=float, default=None)
    parser.add_argument("--score-weight-cap", type=float, default=None)
    parser.add_argument("--cap-min-pct", type=float, default=None)
    parser.add_argument("--cap-bonus", type=float, default=None)
    parser.add_argument("--exclude-st", action="store_true")
    parser.add_argument("--exclude-bj", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.config) != len(args.checkpoint):
        raise ValueError("--config and --checkpoint counts must match")

    configs = [load_config(path) for path in args.config]
    target = pd.Timestamp(args.target_date).normalize()
    frames = [
        _load_target_predictions(config, checkpoint, target, limit=args.limit)
        for config, checkpoint in zip(configs, args.checkpoint, strict=True)
    ]
    ensemble = ensemble_predictions(frames)
    base_config = configs[0]
    output_root = Path(args.output_root or Path(base_config.paths.model_root) / "competition_signals_ensemble")
    output_root.mkdir(parents=True, exist_ok=True)

    raw_path = output_root / f"ensemble_raw_predictions_{target.strftime('%Y%m%d')}.parquet"
    ensemble.to_parquet(raw_path, index=False)

    params = strategy_params_from_config(
        base_config.strategy,
        top_n=args.top_n,
        daily_replace_k=args.daily_replace_k,
        score_variant=args.score_variant,
        score_weight_y=args.score_weight_y,
        score_weight_return=args.score_weight_return,
        score_weight_direction=args.score_weight_direction,
        score_weight_cap=args.score_weight_cap,
        cap_min_pct=args.cap_min_pct,
        cap_bonus=args.cap_bonus,
        exclude_st=True if args.exclude_st else None,
        exclude_bj=True if args.exclude_bj else None,
    )
    scored = prepare_strategy_predictions(
        ensemble,
        params,
        samples_path=base_config.paths.samples_path,
        metric_path=base_config.paths.metric_path,
        price_path=base_config.paths.price_path,
        feature_registry_path=base_config.paths.feature_registry_path,
        news_path=base_config.paths.news_path,
        news_scores_path=base_config.paths.news_scores_path,
        context_cache_path=getattr(base_config.train, "context_cache_path", None),
        news_cache_path=getattr(base_config.train, "context_news_cache_path", None),
    )
    scored_path = output_root / f"ensemble_scored_predictions_{target.strftime('%Y%m%d')}.parquet"
    scored.to_parquet(scored_path, index=False)

    positions = pd.read_csv(args.positions_path) if args.positions_path and args.positions_path.exists() else None
    signals_path = write_competition_signals(
        scored,
        output_root,
        current_positions=positions,
        top_n=params.top_n,
        daily_replace_k=params.daily_replace_k,
    )
    buy_list_path = write_buy_list(signals_path)
    print(f"ensemble_raw_predictions={raw_path}")
    print(f"ensemble_scored_predictions={scored_path}")
    print(f"signals={signals_path}")
    print(f"buy_list={buy_list_path}")
    return 0


def _load_target_predictions(
    config: MSGCAConfig,
    checkpoint_path: str | Path,
    target: pd.Timestamp,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    predictions = evaluate_checkpoint(config, checkpoint_path, split="holdout", limit=limit)
    if predictions.empty:
        predictions = evaluate_checkpoint(config, checkpoint_path, split="all", limit=limit)
    return filter_target_date(predictions, target)


def filter_target_date(predictions: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    if "target_trade_date" not in predictions.columns:
        raise KeyError("Missing prediction columns: ['target_trade_date']")
    target = pd.Timestamp(target).normalize()
    dates = pd.to_datetime(predictions["target_trade_date"], errors="coerce").dt.normalize()
    out = predictions.loc[dates.eq(target)].copy()
    if out.empty:
        raise ValueError(f"No predictions available for target date: {target.date()}")
    return out


def write_buy_list(signals_path: str | Path) -> Path:
    signals_path = Path(signals_path)
    signals = pd.read_csv(signals_path)
    if "suggested_action" not in signals.columns:
        raise KeyError("Missing signal columns: ['suggested_action']")
    target = pd.to_datetime(signals["target_trade_date"], errors="coerce").max().strftime("%Y%m%d")
    buy_list = signals.loc[signals["suggested_action"].astype(str).eq("buy")].copy()
    buy_list_path = signals_path.with_name(f"buy_list_{target}.csv")
    buy_list.to_csv(buy_list_path, index=False)
    return buy_list_path


if __name__ == "__main__":
    raise SystemExit(main())
