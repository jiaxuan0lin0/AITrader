from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
import pandas as pd

from aitrader_paths import DATASETS_ROOT


DEFAULT_SOURCE_DIR = DATASETS_ROOT / "factors" / "evaluation" / "final"
DEFAULT_OUTPUT_DIR = DATASETS_ROOT / "factors" / "evaluation" / "experiment" / "train_slice"
KEY_COLUMNS = ["block", "factor_name", "label"]
KEY_DATE_COLUMNS = [*KEY_COLUMNS, "target_trade_date"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild factor_summary.csv from existing daily IC reports for a date window.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--since", required=True, help="Inclusive target_trade_date lower bound.")
    parser.add_argument("--until", required=True, help="Inclusive target_trade_date upper bound.")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--no-copy-quality", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_slice_summary(args)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


def run_slice_summary(args: argparse.Namespace) -> dict[str, Any]:
    since = _parse_required_date(args.since, "since")
    until = _parse_required_date(args.until, "until")
    if since > until:
        raise ValueError("--since cannot be later than --until")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    source_dir = args.source_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(source_dir / "factor_summary.csv")
    ic = _aggregate_daily_metric(source_dir / "factor_ic.csv", "ic", since, until, args.chunksize, "ic")
    rankic = _aggregate_daily_metric(source_dir / "factor_rankic.csv", "rank_ic", since, until, args.chunksize, "rank_ic")
    spread = _aggregate_group_spread(source_dir / "group_return.csv", since, until, args.chunksize)
    summary = _build_summary(metadata, ic, rankic, spread)

    summary_path = output_dir / "factor_summary.csv"
    summary.to_csv(summary_path, index=False)
    copied_quality = _copy_quality_reports(source_dir, output_dir) if not args.no_copy_quality else []
    result = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "since": since.date().isoformat(),
        "until": until.date().isoformat(),
        "summary_path": str(summary_path),
        "summary_row_count": int(len(summary)),
        "factor_count": int(summary[["block", "factor_name"]].drop_duplicates().shape[0]) if not summary.empty else 0,
        "copied_quality_files": copied_quality,
    }
    metadata_path = output_dir / "single_factor_summary.json"
    metadata_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    result["metadata_path"] = str(metadata_path)
    return result


def _load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing source factor_summary.csv: {path}")
    frame = pd.read_csv(path)
    columns = ["block", "factor_name", "source", "category", "availability", "window", "lookback", "label"]
    _require_columns(frame, ("block", "factor_name", "label"), "factor_summary")
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, columns].drop_duplicates(["block", "factor_name", "label"]).reset_index(drop=True)


def _aggregate_daily_metric(
    path: Path,
    value_column: str,
    since: pd.Timestamp,
    until: pd.Timestamp,
    chunksize: int,
    prefix: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing daily metric csv: {path}")
    state: dict[tuple[str, str, str], dict[str, float]] = {}
    columns = [*KEY_DATE_COLUMNS, "pair_count", "coverage", value_column]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        chunk = _filter_dates(chunk, since, until)
        if chunk.empty:
            continue
        chunk["pair_count"] = pd.to_numeric(chunk["pair_count"], errors="coerce")
        chunk["coverage"] = pd.to_numeric(chunk["coverage"], errors="coerce")
        chunk[value_column] = pd.to_numeric(chunk[value_column], errors="coerce")
        chunk["__value_sq"] = chunk[value_column] * chunk[value_column]
        chunk["__value_pos"] = chunk[value_column].gt(0) & chunk[value_column].notna()
        grouped = chunk.groupby(KEY_COLUMNS, sort=False, dropna=False).agg(
            target_day_count=("target_trade_date", "count"),
            pair_count_sum=("pair_count", "sum"),
            coverage_sum=("coverage", "sum"),
            value_count=(value_column, "count"),
            value_sum=(value_column, "sum"),
            value_sumsq=("__value_sq", "sum"),
            value_positive_count=("__value_pos", "sum"),
        )
        for key, row in grouped.iterrows():
            record = state.setdefault(tuple(map(str, key)), _metric_state())
            record["target_day_count"] += float(row["target_day_count"])
            record["pair_count_sum"] += float(row["pair_count_sum"])
            record["coverage_sum"] += float(row["coverage_sum"])
            record["value_count"] += float(row["value_count"])
            record["value_sum"] += float(row["value_sum"])
            record["value_sumsq"] += float(row["value_sumsq"])
            record["value_positive_count"] += float(row["value_positive_count"])

    records: list[dict[str, Any]] = []
    for key, record in state.items():
        count = int(record["value_count"])
        target_days = int(record["target_day_count"])
        records.append(
            {
                "block": key[0],
                "factor_name": key[1],
                "label": key[2],
                f"{prefix}_day_count": count,
                f"{prefix}_mean": _safe_mean(record["value_sum"], count),
                f"{prefix}_std": _safe_std(record["value_sum"], record["value_sumsq"], count),
                f"{prefix}_ir": _safe_ir(record["value_sum"], record["value_sumsq"], count),
                f"{prefix}_positive_rate": _safe_mean(record["value_positive_count"], count),
                "target_day_count": target_days,
                "pair_count_mean": _safe_mean(record["pair_count_sum"], target_days),
                "coverage_mean": _safe_mean(record["coverage_sum"], target_days),
            }
        )
    return pd.DataFrame(records)


def _aggregate_group_spread(path: Path, since: pd.Timestamp, until: pd.Timestamp, chunksize: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[*KEY_COLUMNS, "group_spread_day_count", "group_spread_mean", "group_spread_positive_rate"])
    state: dict[tuple[str, str, str], dict[str, float]] = {}
    carry = pd.DataFrame()
    columns = [*KEY_DATE_COLUMNS, "group", "mean_return"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        chunk = _filter_dates(chunk, since, until)
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame()
        if chunk.empty:
            continue
        last_key = tuple(chunk.iloc[-1][KEY_DATE_COLUMNS].tolist())
        is_last = (chunk[KEY_DATE_COLUMNS] == pd.Series(last_key, index=KEY_DATE_COLUMNS)).all(axis=1)
        process = chunk.loc[~is_last].copy()
        carry = chunk.loc[is_last].copy()
        _accumulate_spreads(state, process)
    _accumulate_spreads(state, carry)

    records: list[dict[str, Any]] = []
    for key, record in state.items():
        count = int(record["count"])
        records.append(
            {
                "block": key[0],
                "factor_name": key[1],
                "label": key[2],
                "group_spread_day_count": count,
                "group_spread_mean": _safe_mean(record["sum"], count),
                "group_spread_positive_rate": _safe_mean(record["positive_count"], count),
            }
        )
    return pd.DataFrame(records)


def _accumulate_spreads(state: dict[tuple[str, str, str], dict[str, float]], frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    work = frame.copy()
    work["group"] = pd.to_numeric(work["group"], errors="coerce")
    work["mean_return"] = pd.to_numeric(work["mean_return"], errors="coerce")
    work = work.dropna(subset=["group", "mean_return"])
    if work.empty:
        return
    grouped = work.groupby(KEY_DATE_COLUMNS, sort=False, dropna=False)
    min_idx = grouped["group"].idxmin()
    max_idx = grouped["group"].idxmax()
    low = work.loc[min_idx, KEY_DATE_COLUMNS + ["mean_return"]].rename(columns={"mean_return": "low_return"})
    high = work.loc[max_idx, KEY_DATE_COLUMNS + ["mean_return"]].rename(columns={"mean_return": "high_return"})
    spread = low.merge(high, on=KEY_DATE_COLUMNS, how="inner")
    spread["group_spread"] = spread["high_return"] - spread["low_return"]
    by_factor = spread.groupby(KEY_COLUMNS, sort=False, dropna=False).agg(
        count=("group_spread", "count"),
        sum=("group_spread", "sum"),
        positive_count=("group_spread", lambda value: value.gt(0).sum()),
    )
    for key, row in by_factor.iterrows():
        record = state.setdefault(tuple(map(str, key)), {"count": 0.0, "sum": 0.0, "positive_count": 0.0})
        record["count"] += float(row["count"])
        record["sum"] += float(row["sum"])
        record["positive_count"] += float(row["positive_count"])


def _build_summary(metadata: pd.DataFrame, ic: pd.DataFrame, rankic: pd.DataFrame, spread: pd.DataFrame) -> pd.DataFrame:
    work = metadata.merge(ic, on=KEY_COLUMNS, how="left")
    rank_columns = [
        *KEY_COLUMNS,
        "rank_ic_day_count",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_ic_ir",
        "rank_ic_positive_rate",
    ]
    work = work.merge(rankic.loc[:, [column for column in rank_columns if column in rankic.columns]], on=KEY_COLUMNS, how="left")
    work = work.merge(spread, on=KEY_COLUMNS, how="left")
    numeric_defaults = {
        "target_day_count": 0,
        "ic_day_count": 0,
        "rank_ic_day_count": 0,
        "pair_count_mean": np.nan,
        "coverage_mean": np.nan,
        "ic_mean": np.nan,
        "ic_std": np.nan,
        "ic_ir": np.nan,
        "ic_positive_rate": np.nan,
        "rank_ic_mean": np.nan,
        "rank_ic_std": np.nan,
        "rank_ic_ir": np.nan,
        "rank_ic_positive_rate": np.nan,
        "group_spread_mean": np.nan,
        "group_spread_positive_rate": np.nan,
    }
    for column, default in numeric_defaults.items():
        if column not in work.columns:
            work[column] = default
        else:
            work[column] = work[column].fillna(default)
    output_columns = [
        "block",
        "factor_name",
        "source",
        "category",
        "availability",
        "window",
        "lookback",
        "label",
        "target_day_count",
        "ic_day_count",
        "rank_ic_day_count",
        "pair_count_mean",
        "coverage_mean",
        "ic_mean",
        "ic_std",
        "ic_ir",
        "ic_positive_rate",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_ic_ir",
        "rank_ic_positive_rate",
        "group_spread_mean",
        "group_spread_positive_rate",
    ]
    return work.loc[:, output_columns].copy()


def _copy_quality_reports(source_dir: Path, output_dir: Path) -> list[str]:
    names = ["sample_feature_quality.csv", "sample_feature_block_quality.csv", "sample_feature_quality_summary.json"]
    copied: list[str] = []
    for name in names:
        source = source_dir / name
        target = output_dir / name
        if source.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
            copied.append(str(target))
    return copied


def _filter_dates(frame: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    work = frame.copy()
    work["target_trade_date"] = pd.to_datetime(work["target_trade_date"], errors="coerce").dt.normalize()
    return work.loc[work["target_trade_date"].ge(since) & work["target_trade_date"].le(until)].copy()


def _metric_state() -> dict[str, float]:
    return {
        "target_day_count": 0.0,
        "pair_count_sum": 0.0,
        "coverage_sum": 0.0,
        "value_count": 0.0,
        "value_sum": 0.0,
        "value_sumsq": 0.0,
        "value_positive_count": 0.0,
    }


def _safe_mean(total: float, count: int | float) -> float:
    return float(total) / float(count) if count else np.nan


def _safe_std(total: float, total_sq: float, count: int | float) -> float:
    if count < 2:
        return np.nan
    variance = (float(total_sq) - (float(total) ** 2 / float(count))) / (float(count) - 1.0)
    return float(np.sqrt(max(variance, 0.0)))


def _safe_ir(total: float, total_sq: float, count: int | float) -> float:
    std = _safe_std(total, total_sq, count)
    if pd.isna(std) or std == 0:
        return np.nan
    return _safe_mean(total, count) / std


def _parse_required_date(value: str, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --{name} date: {value}")
    return parsed.normalize()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
