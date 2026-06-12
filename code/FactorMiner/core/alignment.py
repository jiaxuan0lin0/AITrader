from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


SAMPLE_KEY_COLUMNS = ("sample_id",)
SAMPLE_DAILY_JOIN_COLUMNS = ("stock_code", "feature_asof_date")
DAILY_KEY_COLUMNS = ("stock_code", "trade_date")
_ROW_ORDER_COLUMN = "__factor_miner_row_order__"


def align_daily_factors_to_samples(
    samples: pd.DataFrame,
    daily_factors: pd.DataFrame,
    factor_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join daily factors to samples using feature_asof_date, never target_trade_date."""
    _require_columns(samples, (*SAMPLE_KEY_COLUMNS, *SAMPLE_DAILY_JOIN_COLUMNS), "samples")
    _require_columns(daily_factors, DAILY_KEY_COLUMNS, "daily_factors")

    selected_factors = list(factor_columns) if factor_columns is not None else [
        column for column in daily_factors.columns if column not in DAILY_KEY_COLUMNS
    ]
    _require_columns(daily_factors, selected_factors, "daily_factors")
    overlapping = sorted(set(samples.columns) & set(selected_factors))
    if overlapping:
        raise ValueError(f"Daily factor columns already exist in samples: {overlapping}")

    sample_work = samples.copy()
    sample_work[_ROW_ORDER_COLUMN] = range(len(sample_work))
    sample_work["stock_code"] = sample_work["stock_code"].astype(str)
    sample_work["feature_asof_date"] = pd.to_datetime(sample_work["feature_asof_date"], errors="coerce").dt.normalize()

    daily_work = daily_factors[[*DAILY_KEY_COLUMNS, *selected_factors]].copy()
    daily_work["stock_code"] = daily_work["stock_code"].astype(str)
    daily_work["trade_date"] = pd.to_datetime(daily_work["trade_date"], errors="coerce").dt.normalize()
    daily_work = daily_work.dropna(subset=list(DAILY_KEY_COLUMNS)).reset_index(drop=True)
    duplicated = daily_work.duplicated(list(DAILY_KEY_COLUMNS))
    if duplicated.any():
        examples = daily_work.loc[duplicated, list(DAILY_KEY_COLUMNS)].head(5).to_dict("records")
        raise ValueError(f"Daily factor keys must be unique before sample alignment: {examples}")

    daily_work = daily_work.rename(columns={"trade_date": "feature_asof_date"})
    aligned = sample_work.merge(daily_work, on=list(SAMPLE_DAILY_JOIN_COLUMNS), how="left")
    aligned = aligned.sort_values(_ROW_ORDER_COLUMN, kind="mergesort").drop(columns=[_ROW_ORDER_COLUMN])
    return aligned.reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")
