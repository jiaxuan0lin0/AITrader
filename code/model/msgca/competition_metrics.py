from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from model.msgca.backtest import (
    read_predictions,
    run_rolling_topk_backtest,
    run_topk_backtest,
    strategy_params_from_config,
)
from model.msgca.config import load_config
from model.msgca.strategy import StrategyParams, prepare_strategy_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score MSGCA predictions with competition-oriented 10-day metrics.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--predictions-path", action="append", type=Path, required=True)
    parser.add_argument("--name", action="append", default=None, help="Optional name for each --predictions-path.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label-col", default="label_next_open_return")
    parser.add_argument("--window-days", type=int, default=10)
    parser.add_argument("--recent-window-count", type=int, default=5)
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
    names = _names(args.predictions_path, args.name)
    rows: list[dict[str, object]] = []
    for name, path in zip(names, args.predictions_path, strict=True):
        predictions = read_predictions(path)
        row = score_competition_predictions(
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
            label_col=args.label_col,
            window_days=args.window_days,
            recent_window_count=args.recent_window_count,
        )
        rows.append({"name": name, "predictions_path": str(path), **row})
    out = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "competition_metrics_summary.csv"
    json_path = args.output_root / "competition_metrics_summary.json"
    out.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")
    print(out.to_string(index=False))
    return 0


def score_competition_predictions(
    predictions: pd.DataFrame,
    params: StrategyParams,
    *,
    samples_path: str | Path | None = None,
    metric_path: str | Path | None = None,
    price_path: str | Path | None = None,
    feature_registry_path: str | Path | None = None,
    news_path: str | Path | None = None,
    news_scores_path: str | Path | None = None,
    context_cache_path: str | Path | None = None,
    news_cache_path: str | Path | None = None,
    label_col: str = "label_next_open_return",
    window_days: int = 10,
    recent_window_count: int = 5,
) -> dict[str, object]:
    scored = prepare_strategy_predictions(
        predictions,
        params,
        samples_path=samples_path,
        metric_path=metric_path,
        price_path=price_path,
        feature_registry_path=feature_registry_path,
        news_path=news_path,
        news_scores_path=news_scores_path,
        context_cache_path=context_cache_path,
        news_cache_path=news_cache_path,
    )
    if scored.empty:
        return _empty_row()
    backtest_params = StrategyParams(
        initial_cash=params.initial_cash,
        top_n=params.top_n,
        daily_replace_k=params.daily_replace_k,
        fee_rate=params.fee_rate,
        slippage_rate=params.slippage_rate,
        full_investment=params.full_investment,
    )
    _, _, _, period_metrics = run_topk_backtest(scored, backtest_params, label_col=label_col)
    _, _, _, windows, rolling_metrics = run_rolling_topk_backtest(
        scored,
        backtest_params,
        window_days=window_days,
        step_days=1,
        label_col=label_col,
    )
    market_period = equal_weight_period_return(scored, label_col=label_col)
    market_rolling = rolling_equal_weight_metrics(scored, label_col=label_col, window_days=window_days)
    recent = _recent_window_metrics(windows, recent_window_count)
    competition_score = (
        0.35 * rolling_metrics["return_mean"]
        + 0.25 * recent["recent_return_mean"]
        + 0.20 * (rolling_metrics["return_mean"] - market_rolling["return_mean"])
        + 0.10 * (period_metrics["period_return"] - market_period)
        + 0.01 * (rolling_metrics["win_rate"] - market_rolling["win_rate"])
        + 0.10 * period_metrics["max_drawdown"]
    )
    return {
        "start_date": pd.to_datetime(scored["target_trade_date"]).min().date().isoformat(),
        "end_date": pd.to_datetime(scored["target_trade_date"]).max().date().isoformat(),
        "day_count": int(pd.to_datetime(scored["target_trade_date"]).nunique()),
        "top_n": params.top_n,
        "daily_replace_k": params.daily_replace_k,
        "score_variant": params.score_variant,
        "score_weight_y": params.score_weight_y,
        "score_weight_return": params.score_weight_return,
        "score_weight_direction": params.score_weight_direction,
        "score_weight_cap": params.score_weight_cap,
        "cap_min_pct": params.cap_min_pct,
        "cap_bonus": params.cap_bonus,
        "exclude_st": params.exclude_st,
        "exclude_bj": params.exclude_bj,
        "period_return": period_metrics["period_return"],
        "period_excess_equal": period_metrics["period_return"] - market_period,
        "market_equal_period_return": market_period,
        "sharpe": period_metrics["sharpe"],
        "max_drawdown": period_metrics["max_drawdown"],
        "turnover_mean": period_metrics["turnover_mean"],
        "rolling_return_mean": rolling_metrics["return_mean"],
        "rolling_excess_equal_mean": rolling_metrics["return_mean"] - market_rolling["return_mean"],
        "rolling_return_median": rolling_metrics["return_median"],
        "rolling_win_rate": rolling_metrics["win_rate"],
        "rolling_return_min": rolling_metrics["return_min"],
        "market_equal_rolling_mean": market_rolling["return_mean"],
        "market_equal_rolling_win_rate": market_rolling["win_rate"],
        "competition_score": competition_score,
        **recent,
    }


def equal_weight_period_return(predictions: pd.DataFrame, *, label_col: str) -> float:
    values = pd.to_numeric(predictions[label_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    daily = values.groupby(predictions["target_trade_date"]).mean().sort_index()
    return float((1.0 + daily.fillna(0.0)).prod() - 1.0)


def rolling_equal_weight_metrics(predictions: pd.DataFrame, *, label_col: str, window_days: int) -> dict[str, float]:
    values = pd.to_numeric(predictions[label_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    daily = values.groupby(predictions["target_trade_date"]).mean().sort_index()
    returns = [
        float((1.0 + daily.iloc[start : start + window_days].fillna(0.0)).prod() - 1.0)
        for start in range(0, max(len(daily) - window_days + 1, 0))
    ]
    values = pd.Series(returns, dtype="float64")
    if values.empty:
        return {"return_mean": float("nan"), "return_median": float("nan"), "win_rate": float("nan"), "return_min": float("nan")}
    return {
        "return_mean": float(values.mean()),
        "return_median": float(values.median()),
        "win_rate": float(values.gt(0).mean()),
        "return_min": float(values.min()),
    }


def _recent_window_metrics(windows: pd.DataFrame, recent_window_count: int) -> dict[str, float]:
    base = {
        "recent_window_count": 0,
        "latest_window_return": float("nan"),
        "recent_return_mean": float("nan"),
        "recent_return_min": float("nan"),
        "recent_win_rate": float("nan"),
    }
    if windows.empty or "period_return" not in windows.columns:
        return base
    recent = windows.sort_values("end_date").tail(max(int(recent_window_count), 1))
    returns = pd.to_numeric(recent["period_return"], errors="coerce").dropna()
    if returns.empty:
        return base
    return {
        "recent_window_count": int(len(returns)),
        "latest_window_return": float(returns.iloc[-1]),
        "recent_return_mean": float(returns.mean()),
        "recent_return_min": float(returns.min()),
        "recent_win_rate": float(returns.gt(0).mean()),
    }


def _empty_row() -> dict[str, object]:
    return {
        "start_date": "",
        "end_date": "",
        "day_count": 0,
        "period_return": float("nan"),
        "period_excess_equal": float("nan"),
        "rolling_return_mean": float("nan"),
        "rolling_excess_equal_mean": float("nan"),
    }


def _names(paths: Sequence[Path], names: Sequence[str] | None) -> list[str]:
    if names is None:
        return [path.stem for path in paths]
    if len(names) != len(paths):
        raise ValueError("--name must be provided the same number of times as --predictions-path")
    return list(names)


if __name__ == "__main__":
    raise SystemExit(main())
