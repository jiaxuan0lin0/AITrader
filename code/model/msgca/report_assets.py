from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from model.msgca.metrics import daily_rank_ic, summarize_predictions, topk_returns
from aitrader_paths import EXPERIMENTS_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect MSGCA report-ready assets.")
    parser.add_argument("--model-root", type=Path, default=EXPERIMENTS_ROOT / "msgca" / "ad_hoc")
    parser.add_argument("--predictions-path", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictions_path = args.predictions_path or args.model_root / "predictions.parquet"
    paths = build_report_assets(args.model_root, predictions_path)
    print(f"report_summary={paths['summary']}")
    return 0


def build_report_assets(model_root: str | Path, predictions_path: str | Path) -> dict[str, Path]:
    root = Path(model_root)
    output_dir = root / "report_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(predictions_path)
    rankic = daily_rank_ic(predictions)
    topk = topk_returns(predictions)
    summary = summarize_predictions(predictions).to_dict()
    gates = _gate_attribution(predictions)
    paths = {
        "daily_rankic": output_dir / "daily_rankic.csv",
        "topk_returns": output_dir / "topk_returns.csv",
        "gate_attribution": output_dir / "gate_attribution.csv",
        "summary": output_dir / "report_summary.json",
    }
    rankic.to_csv(paths["daily_rankic"], index=False)
    topk.to_csv(paths["topk_returns"], index=False)
    gates.to_csv(paths["gate_attribution"], index=False)
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def _gate_attribution(predictions: pd.DataFrame) -> pd.DataFrame:
    gate_cols = [column for column in ("g_price", "g_text", "g_fundamental") if column in predictions.columns]
    if not gate_cols:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["year"] = pd.to_datetime(frame["target_trade_date"]).dt.year
    by_year = frame.groupby("year", sort=True)[gate_cols].mean().reset_index()
    overall = pd.DataFrame([{**{"year": "overall"}, **frame[gate_cols].mean(numeric_only=True).to_dict()}])
    return pd.concat([by_year, overall], ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
