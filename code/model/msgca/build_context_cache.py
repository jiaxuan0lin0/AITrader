from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from model.msgca.config import load_config
from model.msgca.context_features import CONTEXT_COLUMNS, attach_context_features


SAMPLE_COLUMNS = (
    "sample_id",
    "stock_code",
    "industry",
    "feature_asof_date",
    "target_trade_date",
    "decision_ts",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize strict MSGCA context variables in sample row order.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start-date", default=None, help="Optional target_trade_date lower bound for validation only.")
    parser.add_argument("--end-date", default=None, help="Optional target_trade_date upper bound for validation only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output = args.output or Path(config.train.context_cache_path or "")
    if not output:
        raise ValueError("--output or train.context_cache_path is required")
    samples = pd.read_parquet(config.paths.samples_path, columns=list(SAMPLE_COLUMNS))
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["__row_pos"] = np.arange(len(samples), dtype=np.int64)
    if args.start_date or args.end_date:
        dates = pd.to_datetime(samples["target_trade_date"], errors="coerce").dt.normalize()
        mask = pd.Series(True, index=samples.index)
        if args.start_date:
            mask &= dates.ge(pd.Timestamp(args.start_date))
        if args.end_date:
            mask &= dates.le(pd.Timestamp(args.end_date))
        check = samples.loc[mask].copy()
        print(f"validation_subset_rows={len(check)}")
    else:
        check = samples
    print(f"samples_rows={len(samples)} output={output}")
    context = attach_context_features(
        samples,
        samples_path=config.paths.samples_path,
        price_path=config.paths.price_path,
        metric_path=config.paths.metric_path,
        feature_registry_path=config.paths.feature_registry_path,
        news_path=config.paths.news_path,
        news_scores_path=config.paths.news_scores_path,
        news_cache_path=config.train.context_news_cache_path,
        context_columns=CONTEXT_COLUMNS,
        strict=True,
    )
    out = context[["sample_id", *CONTEXT_COLUMNS]].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    print(f"written={output} rows={len(out)} columns={len(out.columns)}")
    if check is not samples:
        subset = attach_context_features(
            check,
            samples_path=config.paths.samples_path,
            price_path=config.paths.price_path,
            metric_path=config.paths.metric_path,
            feature_registry_path=config.paths.feature_registry_path,
            news_path=config.paths.news_path,
            news_scores_path=config.paths.news_scores_path,
            context_cache_path=output,
            news_cache_path=config.train.context_news_cache_path,
            context_columns=CONTEXT_COLUMNS,
            strict=True,
        )
        print(f"validated_subset_rows={len(subset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
