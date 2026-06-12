from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SCORE_COLUMNS = ("y_score", "final_score", "return_pred")
PROB_COLUMNS = ("direction_prob",)
GATE_COLUMNS = ("g_price", "g_text", "g_fundamental")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Average aligned MSGCA prediction files across seeds.")
    parser.add_argument("--predictions-path", action="append", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frames = [read_predictions(path) for path in args.predictions_path]
    ensemble = ensemble_predictions(frames)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_parquet(args.output_path, index=False)
    print(f"ensemble_predictions={args.output_path}")
    print(f"rows={len(ensemble)} seeds={len(frames)}")
    return 0


def read_predictions(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported predictions file extension: {source.suffix}")


def ensemble_predictions(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one prediction frame is required")
    prepared = [_prepare_frame(frame) for frame in frames]
    sample_ids = set(prepared[0]["sample_id"].astype(str))
    for frame in prepared[1:]:
        sample_ids &= set(frame["sample_id"].astype(str))
    if not sample_ids:
        raise ValueError("Prediction files have no overlapping sample_id values")

    base = prepared[0].loc[prepared[0]["sample_id"].isin(sample_ids)].copy()
    base = base.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    ordered_ids = base["sample_id"].astype(str).tolist()
    aligned = [_align_frame(frame, ordered_ids) for frame in prepared]

    out = base.copy()
    for column in SCORE_COLUMNS:
        if all(column in frame.columns for frame in aligned):
            values = [_daily_zscore(frame, column) for frame in aligned]
            out[column] = np.nanmean(np.vstack([item.to_numpy(dtype="float64") for item in values]), axis=0)
    for column in PROB_COLUMNS:
        if all(column in frame.columns for frame in aligned):
            values = [pd.to_numeric(frame[column], errors="coerce").clip(1e-6, 1.0 - 1e-6) for frame in aligned]
            out[column] = np.nanmean(np.vstack([item.to_numpy(dtype="float64") for item in values]), axis=0)
    for column in GATE_COLUMNS:
        if all(column in frame.columns for frame in aligned):
            values = [pd.to_numeric(frame[column], errors="coerce") for frame in aligned]
            out[column] = np.nanmean(np.vstack([item.to_numpy(dtype="float64") for item in values]), axis=0)
    out["rank"] = out.groupby(pd.to_datetime(out["target_trade_date"]).dt.normalize())["final_score"].rank(
        method="first",
        ascending=False,
    )
    return out


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"sample_id", "target_trade_date"} - set(frame.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    out = frame.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    out["target_trade_date"] = pd.to_datetime(out["target_trade_date"]).dt.normalize()
    return out


def _align_frame(frame: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    aligned = frame.set_index("sample_id").reindex(sample_ids).reset_index()
    if aligned["target_trade_date"].isna().any():
        raise ValueError("Prediction alignment introduced missing target_trade_date values")
    return aligned


def _daily_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    dates = pd.to_datetime(frame["target_trade_date"]).dt.normalize()
    mean = values.groupby(dates).transform("mean")
    std = values.groupby(dates).transform("std").replace(0.0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


if __name__ == "__main__":
    raise SystemExit(main())
