from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from model.msgca.config import load_config
from model.msgca.inference import evaluate_checkpoint
from model.msgca.strategy import StrategyParams, prepare_strategy_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MSGCA TopN backtest from predictions or a checkpoint.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint. Used when --predictions-path is omitted.")
    parser.add_argument("--predictions-path", type=Path, default=None, help="Existing predictions parquet/csv file to backtest.")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "valid", "val", "holdout", "all"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None, help="Backtest only the first N target_trade_date values after sorting.")
    parser.add_argument("--rolling-window-days", type=int, default=None, help="Run rolling backtests over N-trading-day windows.")
    parser.add_argument("--rolling-step-days", type=int, default=1, help="Trading-day step between rolling windows.")
    parser.add_argument("--rolling-max-windows", type=int, default=None, help="Optional cap on rolling windows.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-prefix", default="backtest")
    parser.add_argument("--label-col", default="label_next_open_return")
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--daily-replace-k", type=int, default=None)
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--slippage-rate", type=float, default=None)
    parser.add_argument("--no-full-investment", action="store_true")
    parser.add_argument("--score-variant", default=None)
    parser.add_argument("--score-weight-y", type=float, default=None)
    parser.add_argument("--score-weight-return", type=float, default=None)
    parser.add_argument("--score-weight-direction", type=float, default=None)
    parser.add_argument("--score-weight-cap", type=float, default=None)
    parser.add_argument("--cap-min-pct", type=float, default=None)
    parser.add_argument("--cap-bonus", type=float, default=None)
    parser.add_argument("--exclude-st", action="store_true")
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--save-predictions", action="store_true", help="Write generated checkpoint predictions next to backtest outputs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config.ensure_output_dirs()
    output_root = args.output_root or config.paths.output_root
    params = strategy_params_from_config(
        config.strategy,
        initial_cash=args.initial_cash,
        top_n=args.top_n,
        daily_replace_k=args.daily_replace_k,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        full_investment=False if args.no_full_investment else None,
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
    if args.predictions_path is not None:
        predictions = read_predictions(args.predictions_path)
    else:
        predictions = evaluate_checkpoint(config, args.checkpoint, split=args.split, limit=args.limit)
        if args.save_predictions:
            prediction_path = Path(output_root) / f"{args.output_prefix}_predictions.parquet"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(prediction_path, index=False)
            print(f"predictions={prediction_path}")

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

    if args.max_days is not None:
        predictions = filter_first_trade_days(predictions, args.max_days)

    if args.rolling_window_days is not None:
        paths = write_rolling_backtest_outputs(
            predictions,
            output_root,
            params,
            window_days=args.rolling_window_days,
            step_days=args.rolling_step_days,
            max_windows=args.rolling_max_windows,
            prefix=args.output_prefix,
            label_col=args.label_col,
        )
    else:
        paths = write_backtest_outputs(
            predictions,
            output_root,
            params,
            prefix=args.output_prefix,
            label_col=args.label_col,
        )
    for name, path in paths.items():
        print(f"{name}={path}")
    if "metrics" in paths:
        print_backtest_summary(paths["metrics"])
    if "rolling_metrics" in paths:
        print_rolling_backtest_summary(paths["rolling_metrics"])
    return 0


def print_backtest_summary(metrics_path: str | Path) -> None:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    initial_cash = float(metrics.get("initial_cash", float("nan")))
    final_nav = float(metrics.get("final_nav", float("nan")))
    day_count = int(metrics.get("day_count", 0) or 0)
    period_return = float(metrics.get("period_return", metrics.get("total_return", float("nan"))))
    print(f"initial_cash={initial_cash:.2f}")
    print(f"final_nav={final_nav:.2f}")
    print(f"day_count={day_count}")
    print(f"period_return={period_return:.4%}")
    ten_day_return = metrics.get("ten_day_return")
    if day_count == 10 and ten_day_return is not None:
        print(f"ten_day_return={float(ten_day_return):.4%}")


def print_rolling_backtest_summary(metrics_path: str | Path) -> None:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    print(f"rolling_window_days={int(metrics.get('window_days', 0) or 0)}")
    print(f"rolling_step_days={int(metrics.get('step_days', 0) or 0)}")
    print(f"rolling_window_count={int(metrics.get('window_count', 0) or 0)}")
    print(f"rolling_return_mean={float(metrics.get('return_mean', float('nan'))):.4%}")
    print(f"rolling_return_median={float(metrics.get('return_median', float('nan'))):.4%}")
    print(f"rolling_win_rate={float(metrics.get('win_rate', float('nan'))):.2%}")
    if int(metrics.get("window_days", 0) or 0) == 10:
        print(f"rolling_ten_day_return_mean={float(metrics.get('ten_day_return_mean', float('nan'))):.4%}")


def strategy_params_from_config(
    strategy,
    *,
    initial_cash: float | None = None,
    top_n: int | None = None,
    daily_replace_k: int | None = None,
    fee_rate: float | None = None,
    slippage_rate: float | None = None,
    full_investment: bool | None = None,
    score_variant: str | None = None,
    score_weight_y: float | None = None,
    score_weight_return: float | None = None,
    score_weight_direction: float | None = None,
    score_weight_cap: float | None = None,
    cap_min_pct: float | None = None,
    cap_bonus: float | None = None,
    exclude_st: bool | None = None,
    exclude_bj: bool | None = None,
) -> StrategyParams:
    return StrategyParams(
        initial_cash=float(strategy.initial_cash if initial_cash is None else initial_cash),
        top_n=int(strategy.top_n if top_n is None else top_n),
        daily_replace_k=int(strategy.daily_replace_k if daily_replace_k is None else daily_replace_k),
        fee_rate=float(strategy.fee_rate if fee_rate is None else fee_rate),
        slippage_rate=float(strategy.slippage_rate if slippage_rate is None else slippage_rate),
        full_investment=bool(strategy.full_investment if full_investment is None else full_investment),
        score_variant=str(strategy.score_variant if score_variant is None else score_variant),
        score_weight_y=float(strategy.score_weight_y if score_weight_y is None else score_weight_y),
        score_weight_return=float(strategy.score_weight_return if score_weight_return is None else score_weight_return),
        score_weight_direction=float(strategy.score_weight_direction if score_weight_direction is None else score_weight_direction),
        score_weight_cap=float(strategy.score_weight_cap if score_weight_cap is None else score_weight_cap),
        cap_min_pct=float(strategy.cap_min_pct if cap_min_pct is None else cap_min_pct),
        cap_bonus=float(strategy.cap_bonus if cap_bonus is None else cap_bonus),
        exclude_st=bool(strategy.exclude_st if exclude_st is None else exclude_st),
        exclude_bj=bool(strategy.exclude_bj if exclude_bj is None else exclude_bj),
    )


def filter_first_trade_days(predictions: pd.DataFrame, max_days: int) -> pd.DataFrame:
    if max_days <= 0:
        raise ValueError("--max-days must be positive")
    if "target_trade_date" not in predictions.columns:
        raise KeyError("Missing prediction columns: ['target_trade_date']")
    work = predictions.copy()
    normalized_dates = pd.to_datetime(work["target_trade_date"]).dt.normalize()
    keep_dates = sorted(normalized_dates.dropna().unique())[:max_days]
    return work.loc[normalized_dates.isin(keep_dates)].copy()


def filter_trade_days(predictions: pd.DataFrame, trade_dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    if "target_trade_date" not in predictions.columns:
        raise KeyError("Missing prediction columns: ['target_trade_date']")
    keep_dates = {pd.Timestamp(date).normalize() for date in trade_dates}
    normalized_dates = pd.to_datetime(predictions["target_trade_date"]).dt.normalize()
    return predictions.loc[normalized_dates.isin(keep_dates)].copy()


def rolling_trade_date_windows(
    predictions: pd.DataFrame,
    window_days: int,
    step_days: int = 1,
    max_windows: int | None = None,
) -> list[list[pd.Timestamp]]:
    if window_days <= 0:
        raise ValueError("--rolling-window-days must be positive")
    if step_days <= 0:
        raise ValueError("--rolling-step-days must be positive")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("--rolling-max-windows must be positive")
    if "target_trade_date" not in predictions.columns:
        raise KeyError("Missing prediction columns: ['target_trade_date']")
    dates = sorted(pd.to_datetime(predictions["target_trade_date"]).dt.normalize().dropna().unique())
    windows: list[list[pd.Timestamp]] = []
    for start in range(0, max(len(dates) - window_days + 1, 0), step_days):
        window = [pd.Timestamp(date) for date in dates[start : start + window_days]]
        windows.append(window)
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def read_predictions(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing predictions file: {source}")
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported predictions file extension: {source.suffix}")


def run_topk_backtest(
    predictions: pd.DataFrame,
    params: StrategyParams,
    label_col: str = "label_next_open_return",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Daily open-to-open TopN backtest with a simple T+1 sell restriction."""
    required = {"target_trade_date", "stock_code", "y_score", label_col}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    work = predictions.copy()
    work["target_trade_date"] = pd.to_datetime(work["target_trade_date"]).dt.normalize()
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work = work.sort_values(["target_trade_date", "y_score"], ascending=[True, False])

    cash = float(params.initial_cash)
    nav = cash
    positions: dict[str, pd.Timestamp] = {}
    nav_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []

    for date, day in work.groupby("target_trade_date", sort=True):
        score_by_code = day.set_index("stock_code")["y_score"].to_dict()
        tradable = day["stock_code"].astype(str).tolist()
        held = set(positions)
        sellable = {code for code, entry_date in positions.items() if pd.Timestamp(entry_date) < pd.Timestamp(date)}

        if not positions:
            target = set(tradable[: params.top_n])
            sell = set()
            buy = target
        else:
            held_scores = sorted(
                [(code, score_by_code.get(code, -np.inf)) for code in held if code in sellable],
                key=lambda item: item[1],
            )
            sell = {code for code, _ in held_scores[: params.daily_replace_k]}
            target = held - sell
            buy: set[str] = set()
            for code in tradable:
                if code not in target and code not in sell:
                    buy.add(code)
                if len(buy) >= params.daily_replace_k:
                    break
            target |= buy
            if params.full_investment and len(target) < params.top_n:
                for code in tradable:
                    if code not in target and code not in sell:
                        target.add(code)
                    if len(target) >= params.top_n:
                        break

        for code in sell:
            positions.pop(code, None)
            trade_rows.append({"target_trade_date": date, "stock_code": code, "action": "sell", "weight": 0.0})
        for code in buy:
            positions[code] = pd.Timestamp(date)
            trade_rows.append({"target_trade_date": date, "stock_code": code, "action": "buy", "weight": 1.0 / max(len(target), 1)})

        selected = day.loc[day["stock_code"].astype(str).isin(positions)]
        gross_return = float(selected[label_col].mean()) if not selected.empty else 0.0
        turnover = (len(sell) + len(buy)) / max(params.top_n, 1)
        cost = turnover * (params.fee_rate + params.slippage_rate)
        daily_return = gross_return - cost
        nav *= 1.0 + daily_return
        cash = 0.0 if params.full_investment and positions else nav
        nav_rows.append(
            {
                "target_trade_date": date,
                "nav": nav,
                "daily_return": daily_return,
                "gross_return": gross_return,
                "cost": cost,
                "cash": cash,
                "holding_count": len(positions),
                "turnover": turnover,
            }
        )
        weight = 1.0 / max(len(positions), 1) if positions else 0.0
        for code in sorted(positions):
            position_rows.append(
                {
                    "target_trade_date": date,
                    "stock_code": code,
                    "weight": weight,
                    "score": score_by_code.get(code, np.nan),
                    "entry_date": positions[code],
                }
            )

    nav_frame = pd.DataFrame(nav_rows)
    trades = pd.DataFrame(trade_rows)
    positions_frame = pd.DataFrame(position_rows)
    metrics = summarize_backtest(nav_frame, params.initial_cash)
    return nav_frame, trades, positions_frame, metrics


def summarize_backtest(nav: pd.DataFrame, initial_cash: float) -> dict[str, float]:
    if nav.empty:
        return {
            "initial_cash": float(initial_cash),
            "final_nav": float("nan"),
            "day_count": 0.0,
            "period_return": float("nan"),
            "ten_day_return": float("nan"),
            "annual_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "total_return": float("nan"),
            "turnover_mean": float("nan"),
        }
    returns = pd.to_numeric(nav["daily_return"], errors="coerce").fillna(0.0)
    day_count = len(nav)
    final_nav = float(nav["nav"].iloc[-1])
    total_return = float(final_nav / initial_cash - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / max(day_count, 1)) - 1.0)
    vol = float(returns.std(ddof=0))
    sharpe = float(returns.mean() / vol * np.sqrt(252.0)) if vol > 0 else float("nan")
    cummax = nav["nav"].cummax()
    drawdown = nav["nav"] / cummax - 1.0
    return {
        "initial_cash": float(initial_cash),
        "final_nav": final_nav,
        "day_count": float(day_count),
        "period_return": total_return,
        "ten_day_return": total_return if day_count == 10 else float("nan"),
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "total_return": total_return,
        "turnover_mean": float(pd.to_numeric(nav["turnover"], errors="coerce").mean()),
    }


def run_rolling_topk_backtest(
    predictions: pd.DataFrame,
    params: StrategyParams,
    window_days: int = 10,
    step_days: int = 1,
    max_windows: int | None = None,
    label_col: str = "label_next_open_return",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    windows = rolling_trade_date_windows(predictions, window_days, step_days=step_days, max_windows=max_windows)
    nav_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []

    for window_id, dates in enumerate(windows, start=1):
        window_predictions = filter_trade_days(predictions, dates)
        nav, trades, positions, metrics = run_topk_backtest(window_predictions, params, label_col=label_col)
        start_date = pd.Timestamp(dates[0]).date().isoformat()
        end_date = pd.Timestamp(dates[-1]).date().isoformat()
        row = {
            "window_id": window_id,
            "start_date": start_date,
            "end_date": end_date,
            **metrics,
        }
        window_rows.append(row)
        if not nav.empty:
            nav = nav.copy()
            nav.insert(0, "window_id", window_id)
            nav.insert(1, "window_start_date", start_date)
            nav.insert(2, "window_end_date", end_date)
            nav_frames.append(nav)
        if not trades.empty:
            trades = trades.copy()
            trades.insert(0, "window_id", window_id)
            trades.insert(1, "window_start_date", start_date)
            trades.insert(2, "window_end_date", end_date)
            trade_frames.append(trades)
        if not positions.empty:
            positions = positions.copy()
            positions.insert(0, "window_id", window_id)
            positions.insert(1, "window_start_date", start_date)
            positions.insert(2, "window_end_date", end_date)
            position_frames.append(positions)

    nav_frame = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame()
    trades_frame = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    positions_frame = pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame()
    windows_frame = pd.DataFrame(window_rows)
    metrics = summarize_rolling_backtest(windows_frame, window_days=window_days, step_days=step_days)
    return nav_frame, trades_frame, positions_frame, windows_frame, metrics


def summarize_rolling_backtest(windows: pd.DataFrame, window_days: int, step_days: int) -> dict[str, float]:
    base = {
        "window_days": float(window_days),
        "step_days": float(step_days),
        "window_count": float(len(windows)),
        "return_mean": float("nan"),
        "return_median": float("nan"),
        "return_std": float("nan"),
        "return_min": float("nan"),
        "return_max": float("nan"),
        "win_rate": float("nan"),
        "positive_count": 0.0,
        "negative_count": 0.0,
        "max_drawdown_min": float("nan"),
        "turnover_mean": float("nan"),
        "ten_day_return_mean": float("nan"),
        "ten_day_return_median": float("nan"),
    }
    if windows.empty or "period_return" not in windows.columns:
        return base
    returns = pd.to_numeric(windows["period_return"], errors="coerce").dropna()
    if returns.empty:
        return base
    base.update(
        {
            "return_mean": float(returns.mean()),
            "return_median": float(returns.median()),
            "return_std": float(returns.std(ddof=0)),
            "return_min": float(returns.min()),
            "return_max": float(returns.max()),
            "win_rate": float(returns.gt(0).mean()),
            "positive_count": float(returns.gt(0).sum()),
            "negative_count": float(returns.lt(0).sum()),
            "max_drawdown_min": float(pd.to_numeric(windows["max_drawdown"], errors="coerce").min())
            if "max_drawdown" in windows.columns
            else float("nan"),
            "turnover_mean": float(pd.to_numeric(windows["turnover_mean"], errors="coerce").mean())
            if "turnover_mean" in windows.columns
            else float("nan"),
        }
    )
    if window_days == 10:
        base["ten_day_return_mean"] = base["return_mean"]
        base["ten_day_return_median"] = base["return_median"]
    return base


def daily_orders_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "target_trade_date",
        "buy_count",
        "sell_count",
        "buy_stock_codes",
        "sell_stock_codes",
        "buy_weights",
    ]
    if trades.empty:
        return pd.DataFrame(columns=base_columns)
    required = {"target_trade_date", "stock_code", "action"}
    missing = required - set(trades.columns)
    if missing:
        raise KeyError(f"Missing trade columns: {sorted(missing)}")

    work = trades.copy()
    work["target_trade_date"] = pd.to_datetime(work["target_trade_date"]).dt.normalize()
    window_columns = [column for column in ("window_id", "window_start_date", "window_end_date") if column in work.columns]
    group_columns = [*window_columns, "target_trade_date"]
    rows: list[dict[str, object]] = []
    for keys, group in work.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        buy = group.loc[group["action"].eq("buy")].copy()
        sell = group.loc[group["action"].eq("sell")].copy()
        buy_codes = buy["stock_code"].astype(str).sort_values().tolist()
        sell_codes = sell["stock_code"].astype(str).sort_values().tolist()
        if "weight" in buy.columns and not buy.empty:
            buy_weights = [
                f"{code}:{float(weight):.6f}"
                for code, weight in buy.sort_values("stock_code")[["stock_code", "weight"]].itertuples(index=False)
            ]
        else:
            buy_weights = []
        row.update(
            {
                "buy_count": len(buy_codes),
                "sell_count": len(sell_codes),
                "buy_stock_codes": ",".join(buy_codes),
                "sell_stock_codes": ",".join(sell_codes),
                "buy_weights": ",".join(buy_weights),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=[*window_columns, *base_columns])


def write_backtest_outputs(
    predictions: pd.DataFrame,
    output_root: str | Path,
    params: StrategyParams,
    prefix: str = "backtest",
    label_col: str = "label_next_open_return",
) -> dict[str, Path]:
    nav, trades, positions, metrics = run_topk_backtest(predictions, params, label_col=label_col)
    daily_orders = daily_orders_from_trades(trades)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "nav": root / f"{prefix}_nav.csv",
        "trades": root / f"{prefix}_trades.csv",
        "positions": root / f"{prefix}_positions.csv",
        "daily_orders": root / f"{prefix}_daily_orders.csv",
        "metrics": root / f"{prefix}_metrics.json",
    }
    nav.to_csv(paths["nav"], index=False)
    trades.to_csv(paths["trades"], index=False)
    positions.to_csv(paths["positions"], index=False)
    daily_orders.to_csv(paths["daily_orders"], index=False)
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def write_rolling_backtest_outputs(
    predictions: pd.DataFrame,
    output_root: str | Path,
    params: StrategyParams,
    window_days: int = 10,
    step_days: int = 1,
    max_windows: int | None = None,
    prefix: str = "backtest",
    label_col: str = "label_next_open_return",
) -> dict[str, Path]:
    nav, trades, positions, windows, metrics = run_rolling_topk_backtest(
        predictions,
        params,
        window_days=window_days,
        step_days=step_days,
        max_windows=max_windows,
        label_col=label_col,
    )
    daily_orders = daily_orders_from_trades(trades)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "rolling_nav": root / f"{prefix}_rolling_nav.csv",
        "rolling_trades": root / f"{prefix}_rolling_trades.csv",
        "rolling_positions": root / f"{prefix}_rolling_positions.csv",
        "rolling_daily_orders": root / f"{prefix}_rolling_daily_orders.csv",
        "rolling_windows": root / f"{prefix}_rolling_windows.csv",
        "rolling_metrics": root / f"{prefix}_rolling_metrics.json",
    }
    nav.to_csv(paths["rolling_nav"], index=False)
    trades.to_csv(paths["rolling_trades"], index=False)
    positions.to_csv(paths["rolling_positions"], index=False)
    daily_orders.to_csv(paths["rolling_daily_orders"], index=False)
    windows.to_csv(paths["rolling_windows"], index=False)
    paths["rolling_metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
