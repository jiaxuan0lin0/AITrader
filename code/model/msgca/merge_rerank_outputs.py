from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from model.msgca.rerank_loss_ablation import parse_windows, write_outputs
from model.msgca.strategy import StrategyParams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge MSGCA rerank output directories.")
    parser.add_argument("--input-root", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window", action="append", default=None)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--daily-replace-k", type=int, default=3)
    parser.add_argument("--fee-rate", type=float, default=0.0003)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--score-variant", default="weighted")
    parser.add_argument("--score-weight-y", type=float, default=0.25)
    parser.add_argument("--score-weight-return", type=float, default=1.5)
    parser.add_argument("--score-weight-direction", type=float, default=0.5)
    parser.add_argument("--score-weight-cap", type=float, default=0.75)
    parser.add_argument("--cap-min-pct", type=float, default=0.0)
    parser.add_argument("--cap-bonus", type=float, default=0.0)
    parser.add_argument("--exclude-st", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-bj", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, object]] = []
    for root in args.input_root:
        path = root / "checkpoint_strategy_recheck_summary.csv"
        if not path.exists():
            print(f"skip_missing={path}")
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        rows.extend(frame.to_dict(orient="records"))

    params = StrategyParams(
        initial_cash=args.initial_cash,
        top_n=args.top_n,
        daily_replace_k=args.daily_replace_k,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        full_investment=True,
        score_variant=args.score_variant,
        score_weight_y=args.score_weight_y,
        score_weight_return=args.score_weight_return,
        score_weight_direction=args.score_weight_direction,
        score_weight_cap=args.score_weight_cap,
        cap_min_pct=args.cap_min_pct,
        cap_bonus=args.cap_bonus,
        exclude_st=args.exclude_st,
        exclude_bj=args.exclude_bj,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_outputs(rows, args.output_root, params, parse_windows(args.window))
    print(f"rows={len(rows)}")
    print(f"summary={args.output_root / 'checkpoint_strategy_recheck_summary.csv'}")
    print(f"best_by_run={args.output_root / 'best_by_run.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
