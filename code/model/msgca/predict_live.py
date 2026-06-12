from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from model.msgca.backtest import strategy_params_from_config
from model.msgca.config import load_config
from model.msgca.inference import evaluate_checkpoint
from model.msgca.strategy import prepare_strategy_predictions, write_competition_signals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate latest MSGCA competition signals.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--target-date", default=None, help="YYYY-MM-DD. Defaults to latest available target_trade_date.")
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
    config = load_config(args.config)
    config.ensure_output_dirs()
    params = strategy_params_from_config(
        config.strategy,
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
    predictions = evaluate_checkpoint(config, args.checkpoint, split="holdout", limit=args.limit)
    if predictions.empty:
        predictions = evaluate_checkpoint(config, args.checkpoint, split="all", limit=args.limit)
    if args.target_date:
        target = pd.Timestamp(args.target_date).normalize()
        predictions = predictions.loc[pd.to_datetime(predictions["target_trade_date"]).dt.normalize().eq(target)].copy()
        if predictions.empty:
            raise ValueError(f"No predictions available for target date: {args.target_date}")
    else:
        latest = pd.to_datetime(predictions["target_trade_date"]).max()
        predictions = predictions.loc[pd.to_datetime(predictions["target_trade_date"]).eq(latest)].copy()
    predictions = prepare_strategy_predictions(
        predictions,
        params,
        samples_path=config.paths.samples_path,
        metric_path=config.paths.metric_path,
        price_path=config.paths.price_path,
        feature_registry_path=config.paths.feature_registry_path,
        news_path=config.paths.news_path,
        news_scores_path=config.paths.news_scores_path,
        context_cache_path=getattr(config.train, "context_cache_path", None),
        news_cache_path=getattr(config.train, "context_news_cache_path", None),
    )
    scored_path = Path(config.paths.model_root) / "competition_signals" / "latest_scored_predictions.parquet"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(scored_path, index=False)
    positions = pd.read_csv(args.positions_path) if args.positions_path and args.positions_path.exists() else None
    output_path = write_competition_signals(
        predictions,
        Path(config.paths.model_root) / "competition_signals",
        current_positions=positions,
        top_n=params.top_n,
        daily_replace_k=params.daily_replace_k,
    )
    print(f"scored_predictions={scored_path}")
    print(f"signals={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
