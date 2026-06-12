from __future__ import annotations

import argparse
import copy
import gc
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from model.msgca.competition_metrics import score_competition_predictions
from model.msgca.config import MSGCAConfig, load_config
from model.msgca.dataset import ScalerState, build_datasets
from model.msgca.inference import build_model_from_layout, predict_dataset
from model.msgca.strategy import StrategyParams


@dataclass(frozen=True)
class EvalWindow:
    name: str
    start: str
    end: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-rank MSGCA loss-ablation checkpoints under one strategy.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="Run name under <experiment-root>/runs.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        help="Evaluation window as name:start:end. Defaults to validation, holdout, and two 2025 reference windows.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label-col", default="label_next_open_return")
    parser.add_argument("--window-days", type=int, default=10)
    parser.add_argument("--recent-window-count", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
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
    experiment_root = args.experiment_root
    output_root = args.output_root or experiment_root / "strategy_recheck" / "best_weighted_top10_k3"
    output_root.mkdir(parents=True, exist_ok=True)

    windows = parse_windows(args.window)
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

    rows = load_existing_rows(output_root)
    completed = completed_keys(rows)
    if rows:
        print(f"resume_existing_rows={len(rows)}", flush=True)
    for run_name in args.run:
        evaluate_run(
            experiment_root=experiment_root,
            run_name=run_name,
            windows=windows,
            params=params,
            label_col=args.label_col,
            window_days=args.window_days,
            recent_window_count=args.recent_window_count,
            device=args.device,
            output_root=output_root,
            max_epochs=args.max_epochs,
            save_predictions=args.save_predictions,
            rows=rows,
            completed=completed,
        )
        write_outputs(rows, output_root, params, windows)

    write_outputs(rows, output_root, params, windows)
    print(f"rows={len(rows)}")
    print(f"summary={output_root / 'checkpoint_strategy_recheck_summary.csv'}")
    print(f"best_by_run={output_root / 'best_by_run.csv'}")
    return 0


def evaluate_run(
    *,
    experiment_root: Path,
    run_name: str,
    windows: Sequence[EvalWindow],
    params: StrategyParams,
    label_col: str,
    window_days: int,
    recent_window_count: int,
    device: str,
    output_root: Path,
    max_epochs: int | None,
    save_predictions: bool,
    rows: list[dict[str, object]],
    completed: set[tuple[str, int, str]],
) -> None:
    config_path = experiment_root / "configs" / f"{run_name}.yaml"
    run_root = experiment_root / "runs" / run_name
    checkpoints = list_epoch_checkpoints(run_root / "checkpoints", max_epochs=max_epochs)
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found for {run_name}: {run_root / 'checkpoints'}")

    config = load_config(config_path)
    first_state = load_checkpoint(checkpoints[0][1])
    scaler = scaler_from_checkpoint(first_state)
    for window in windows:
        window_config = config_for_window(config, window)
        print(f"[{run_name}] build_window name={window.name} start={window.start} end={window.end}", flush=True)
        dataset, layout, _ = build_datasets(window_config, split="validation", feature_scaler=scaler)
        if len(dataset) == 0:
            print(f"[{run_name}] skip_empty_window name={window.name}", flush=True)
            continue

        for epoch, checkpoint_path in checkpoints:
            key = (run_name, epoch, window.name)
            if key in completed:
                print(f"[{run_name}] skip_completed epoch={epoch} window={window.name}", flush=True)
                continue
            print(f"[{run_name}] eval epoch={epoch} window={window.name}", flush=True)
            state = load_checkpoint(checkpoint_path)
            validate_layout(state, layout)
            model = build_model_from_layout(window_config, layout)
            model.load_state_dict(state["model"])
            predictions = predict_dataset(model, dataset, batch_days=window_config.train.batch_days, device=device)
            metrics = score_competition_predictions(
                predictions,
                params,
                samples_path=window_config.paths.samples_path,
                metric_path=window_config.paths.metric_path,
                price_path=window_config.paths.price_path,
                feature_registry_path=window_config.paths.feature_registry_path,
                news_path=window_config.paths.news_path,
                news_scores_path=window_config.paths.news_scores_path,
                context_cache_path=getattr(window_config.train, "context_cache_path", None),
                news_cache_path=getattr(window_config.train, "context_news_cache_path", None),
                label_col=label_col,
                window_days=window_days,
                recent_window_count=recent_window_count,
            )
            row = {
                "run": run_name,
                "r_value": parse_r_value(run_name),
                "epoch": epoch,
                "checkpoint": str(checkpoint_path),
                "window": window.name,
                "window_start": window.start,
                "window_end": window.end,
                **metrics,
            }
            rows.append(row)
            completed.add(key)
            write_outputs(rows, output_root, params, windows)
            if save_predictions:
                pred_dir = output_root / "predictions" / run_name / f"epoch_{epoch:02d}"
                pred_dir.mkdir(parents=True, exist_ok=True)
                predictions.to_parquet(pred_dir / f"{window.name}.parquet", index=False)
            del model, predictions, state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del dataset
        gc.collect()
    return None


def list_epoch_checkpoints(checkpoint_dir: Path, *, max_epochs: int | None) -> list[tuple[int, Path]]:
    pattern = re.compile(r"msgca_epoch_(\d+)\.pt$")
    items: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("msgca_epoch_*.pt"):
        match = pattern.search(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if max_epochs is not None and epoch > max_epochs:
            continue
        items.append((epoch, path))
    return sorted(items, key=lambda item: item[0])


def load_checkpoint(path: Path) -> dict[str, object]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Invalid checkpoint: {path}")
    return state


def scaler_from_checkpoint(state: dict[str, object]) -> ScalerState:
    payload = state.get("feature_scaler")
    if payload is None:
        raise ValueError("Checkpoint is missing feature_scaler; reranking would not match live inference.")
    return ScalerState.from_dict(payload)


def validate_layout(state: dict[str, object], layout) -> None:
    saved = state.get("layout")
    if not isinstance(saved, dict):
        return
    for key in ("price_columns", "text_columns", "fundamental_columns", "text_group_ids", "fundamental_group_ids"):
        previous = list(saved.get(key, []))
        current = list(getattr(layout, key, []))
        if previous != current:
            raise ValueError(f"Checkpoint layout mismatch for {key}: {len(previous)} saved vs {len(current)} current")


def config_for_window(config: MSGCAConfig, window: EvalWindow) -> MSGCAConfig:
    out = copy.deepcopy(config)
    out.data.validation_start = window.start
    out.data.validation_end = window.end
    return out


def parse_windows(values: Sequence[str] | None) -> list[EvalWindow]:
    if not values:
        return [
            EvalWindow("val2026_strict", "2026-01-01", "2026-05-20"),
            EvalWindow("hold2026_recent", "2026-05-21", "2026-05-31"),
            EvalWindow("same_calendar_2025", "2025-05-21", "2025-05-31"),
            EvalWindow("same_comp_2025", "2025-06-02", "2025-06-13"),
        ]
    windows: list[EvalWindow] = []
    for raw in values:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"Window must be name:start:end, got: {raw}")
        windows.append(EvalWindow(parts[0], parts[1], parts[2]))
    return windows


def parse_r_value(run_name: str) -> float:
    match = re.search(r"r(\d+)", run_name)
    if match is None:
        return float("nan")
    return int(match.group(1)) / 100.0


def load_existing_rows(output_root: Path) -> list[dict[str, object]]:
    path = output_root / "checkpoint_strategy_recheck_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def completed_keys(rows: Sequence[dict[str, object]]) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    for row in rows:
        run = row.get("run")
        epoch = row.get("epoch")
        window = row.get("window")
        if run is None or epoch is None or window is None:
            continue
        try:
            keys.add((str(run), int(epoch), str(window)))
        except (TypeError, ValueError):
            continue
    return keys


def write_outputs(rows: list[dict[str, object]], output_root: Path, params: StrategyParams, windows: Sequence[EvalWindow]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows).drop_duplicates(subset=["run", "epoch", "window"], keep="last")
    summary_path = output_root / "checkpoint_strategy_recheck_summary.csv"
    frame.to_csv(summary_path, index=False)
    (output_root / "checkpoint_strategy_recheck_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    aggregate = aggregate_selection(frame)
    aggregate.to_csv(output_root / "checkpoint_selection_score.csv", index=False)
    best_by_run = (
        aggregate.sort_values(["run", "selection_score"], ascending=[True, False])
        .groupby("run", as_index=False)
        .head(1)
        .sort_values("selection_score", ascending=False)
    )
    best_by_run.to_csv(output_root / "best_by_run.csv", index=False)
    manifest = {
        "strategy": asdict(params),
        "windows": [asdict(window) for window in windows],
        "summary": str(summary_path),
        "selection": str(output_root / "checkpoint_selection_score.csv"),
        "best_by_run": str(output_root / "best_by_run.csv"),
    }
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_root, params, windows, best_by_run)


def aggregate_selection(frame: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["run", "r_value", "epoch", "checkpoint"]
    metrics = [
        "competition_score",
        "period_return",
        "period_excess_equal",
        "rolling_return_mean",
        "rolling_excess_equal_mean",
        "rolling_win_rate",
        "max_drawdown",
        "turnover_mean",
        "latest_window_return",
        "recent_return_mean",
        "recent_return_min",
        "market_equal_period_return",
    ]
    available = [col for col in metrics if col in frame.columns]
    wide = frame.pivot_table(index=index_cols, columns="window", values=available, aggfunc="first")
    wide.columns = [f"{window}_{metric}" for metric, window in wide.columns]
    out = wide.reset_index()
    score = pd.Series(0.0, index=out.index)
    weights = {
        "val2026_strict": 0.45,
        "hold2026_recent": 0.30,
        "same_calendar_2025": 0.125,
        "same_comp_2025": 0.125,
    }
    used_weight = pd.Series(0.0, index=out.index)
    for window, weight in weights.items():
        values = selection_component(out, window)
        out[f"{window}_selection_component"] = values
        keep = values.notna()
        score.loc[keep] += weight * values.loc[keep]
        used_weight.loc[keep] += weight
    out["selection_score"] = score.where(used_weight.eq(0), score / used_weight.replace(0.0, pd.NA))
    out["selection_score_weight_sum"] = used_weight
    return out.sort_values("selection_score", ascending=False)


def selection_component(frame: pd.DataFrame, window: str) -> pd.Series:
    for metric in ("competition_score", "period_excess_equal", "period_return"):
        col = f"{window}_{metric}"
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            return values
    return pd.Series(float("nan"), index=frame.index)


def write_readme(output_root: Path, params: StrategyParams, windows: Sequence[EvalWindow], best_by_run: pd.DataFrame) -> None:
    lines = [
        "# Loss Ablation Strategy Recheck",
        "",
        "This directory re-ranks saved loss-ablation epoch checkpoints under one fixed strategy.",
        "",
        "## Strategy",
        "",
        "```json",
        json.dumps(asdict(params), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Windows",
        "",
    ]
    for window in windows:
        lines.append(f"- {window.name}: {window.start} to {window.end}")
    lines.extend(
        [
            "",
            "## Selection Score",
            "",
            "- val2026_strict weight: 0.45",
            "- hold2026_recent weight: 0.30",
            "- same_calendar_2025 weight: 0.125",
            "- same_comp_2025 weight: 0.125",
            "- Each window uses competition_score when available, then period_excess_equal, then period_return.",
            "",
            "## Current Best By Run",
            "",
        ]
    )
    if best_by_run.empty:
        lines.append("No completed rows yet.")
    else:
        cols = [
            col
            for col in [
                "run",
                "epoch",
                "selection_score",
                "selection_score_weight_sum",
                "val2026_context_selection_component",
                "hold2026_recent_selection_component",
                "same_calendar_2025_selection_component",
                "same_comp_2025_selection_component",
            ]
            if col in best_by_run.columns
        ]
        lines.append("```text")
        lines.append(best_by_run[cols].to_string(index=False))
        lines.append("```")
    output_root.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
