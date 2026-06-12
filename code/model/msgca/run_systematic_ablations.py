from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Sequence

import numpy as np
import pandas as pd

from model.msgca.backtest import write_backtest_outputs
from model.msgca.config import MSGCAConfig, load_config, write_resolved_config
from model.msgca.strategy import StrategyParams
from aitrader_paths import CODE_ROOT, DATASETS_ROOT, EXPERIMENTS_ROOT


DEFAULT_MSGCA_ROOT = EXPERIMENTS_ROOT / "msgca" / "generated_systematic"
DEFAULT_EVALUATION_ROOT = DATASETS_ROOT / "factors" / "evaluation"
DEFAULT_RUN_ROOT = DEFAULT_MSGCA_ROOT / "runs"
DEFAULT_CONFIG_ROOT = DEFAULT_MSGCA_ROOT / "configs"
PYTHON_BIN = os.environ.get("AITRADER_PYTHON_BIN") or sys.executable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sequential MSGCA systematic ablation experiments.")
    parser.add_argument("--base-config", type=Path, default=Path("model/msgca/config.yaml"))
    parser.add_argument("--job-id", default=None)
    parser.add_argument(
        "--matrix",
        default="first_wave",
        choices=(
            "first_wave",
            "extended",
            "factor",
            "factor_clean",
            "gpt_final",
            "gpt_final_soft",
            "topk",
            "time",
            "interaction",
            "scale",
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--only", nargs="*", default=None, help="Optional list of variant names to run.")
    parser.add_argument("--start-at", default=None, help="Skip variants until this name is reached.")
    parser.add_argument("--force", action="store_true", help="Re-run a variant even if run_summary.json exists.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failed variants and continue with later variants.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    job_id = args.job_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_systematic_ablation")
    config_dir = args.config_root / job_id
    config_dir.mkdir(parents=True, exist_ok=True)
    variants = select_variants(args.matrix)
    if args.only:
        keep = set(args.only)
        variants = [variant for variant in variants if variant["name"] in keep]
    if args.start_at:
        names = [variant["name"] for variant in variants]
        if args.start_at not in names:
            raise ValueError(f"--start-at={args.start_at} is not in matrix {args.matrix}: {names}")
        variants = variants[names.index(args.start_at) :]
    if not variants:
        raise ValueError("No variants selected")

    print(f"job_id={job_id}", flush=True)
    print(f"matrix={args.matrix} variant_count={len(variants)} epochs={args.epochs}", flush=True)
    summary_rows: list[dict[str, Any]] = []
    for order, variant in enumerate(variants, start=1):
        run_dir = args.run_root / f"{job_id}_{variant['name']}"
        config_path = config_dir / f"{variant['name']}.yaml"
        config = build_variant_config(args.base_config, variant, run_dir, args.epochs)
        write_resolved_config(config, config_path)
        config_path.with_suffix(".meta.json").write_text(json.dumps(variant, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{now()}] variant_start order={order}/{len(variants)} name={variant['name']}", flush=True)
        print(f"run_dir={run_dir}", flush=True)
        print(f"config={config_path}", flush=True)
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists() and not args.force:
            print(f"[{now()}] variant_skip_existing name={variant['name']} summary={summary_path}", flush=True)
        else:
            try:
                run_variant(config_path, run_dir)
            except Exception as exc:
                error_path = write_error_record(run_dir, variant["name"], exc)
                print(f"[{now()}] variant_failed name={variant['name']} error={error_path}", flush=True)
                if not args.continue_on_error:
                    raise
        if summary_path.exists():
            row = flatten_summary(variant["name"], json.loads(summary_path.read_text(encoding="utf-8")))
            row["status"] = "ok"
            summary_rows.append(row)
            write_matrix_summary(args.run_root / f"{job_id}_ablation_summary.csv", summary_rows)
        else:
            error_path = run_dir / "run_error.json"
            if error_path.exists():
                row = flatten_error(variant["name"], run_dir, config_path, json.loads(error_path.read_text(encoding="utf-8")))
                summary_rows.append(row)
                write_matrix_summary(args.run_root / f"{job_id}_ablation_summary.csv", summary_rows)
        print(f"[{now()}] variant_done name={variant['name']}", flush=True)
    if summary_rows:
        summary_csv = args.run_root / f"{job_id}_ablation_summary.csv"
        write_matrix_summary(summary_csv, summary_rows)
        print(f"ablation_summary={summary_csv}", flush=True)
    print(f"[{now()}] all_done job_id={job_id}", flush=True)
    return 0


def build_variant_config(base_config: Path, variant: dict[str, Any], run_dir: Path, epochs: int) -> MSGCAConfig:
    config = load_config(base_config)
    config.paths.model_root = str(run_dir)
    config.train.epochs = int(epochs)
    config.train.final_validate = True
    config.train.validate_each_epoch = False
    config.train.validation_interval = 0
    config.paths.evaluation_dir = str(DEFAULT_EVALUATION_ROOT / "experiment" / "select_20160105_20250930_slice")
    config.data.train_start = "2019-01-01"
    config.data.train_end = "2025-09-30"
    config.data.validation_start = "2025-10-01"
    config.data.validation_end = "2025-12-31"
    config.data.holdout_start = "2026-01-01"
    config.model.factor_encoder = "factor_aware"
    config.model.hidden_dim = 64
    config.model.n_heads = 4
    config.model.price_layers = 2
    config.model.factor_layers = 2
    config.model.factor_group_layers = 1
    config.model.factor_group_prototypes = 1
    config.model.use_factor_gate = True
    config.train.topk_return_loss_weight = 0.02
    config.train.topk_temperature = 0.1
    config.train.time_weight_bins = []
    config.train.normalize_time_weights = False
    for key, value in variant.get("overrides", {}).items():
        set_nested(config, key, value)
    return config


def run_variant(config_path: Path, run_dir: Path) -> None:
    variant = json.loads((config_path.with_suffix(".meta.json")).read_text(encoding="utf-8"))
    subprocess.run(train_command(config_path, variant), check=True, cwd=str(CODE_ROOT))
    checkpoint = run_dir / "checkpoints" / "msgca_latest.pt"
    subprocess.run(evaluate_command(config_path, run_dir), check=True, cwd=str(CODE_ROOT))
    write_run_summary(run_dir, config_path, checkpoint)


def train_command(config_path: Path, variant: dict[str, Any] | None = None) -> list[str]:
    command = [PYTHON_BIN, "-m", "model.msgca.train", "--config", str(config_path), "--final-validate"]
    return command


def evaluate_command(config_path: Path, run_dir: Path) -> list[str]:
    return [
        PYTHON_BIN,
        "-m",
        "model.msgca.evaluate",
        "--config",
        str(config_path),
        "--checkpoint",
        str(run_dir / "checkpoints" / "msgca_latest.pt"),
        "--split",
        "holdout",
        "--output-prefix",
        "holdout",
    ]


def write_run_summary(run_dir: Path, config_path: Path, checkpoint: Path) -> None:
    config = load_config(config_path)
    params = StrategyParams(
        initial_cash=config.strategy.initial_cash,
        top_n=config.strategy.top_n,
        daily_replace_k=config.strategy.daily_replace_k,
        fee_rate=config.strategy.fee_rate,
        slippage_rate=config.strategy.slippage_rate,
        full_investment=config.strategy.full_investment,
    )
    sanity: dict[str, Any] = {}
    market: dict[str, Any] = {}
    for split, pred_name, prefix in [
        ("validation", "validation_predictions.parquet", "validation_backtest"),
        ("holdout", "holdout_predictions.parquet", "holdout_backtest"),
    ]:
        predictions = pd.read_parquet(run_dir / pred_name)
        write_backtest_outputs(predictions, run_dir, params, prefix=prefix)
        market[split] = market_baseline(predictions)
        sanity[split] = sanity_checks(predictions)
    summary = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "train_log": str(run_dir / "train_log.csv"),
        "validation_metrics": read_json(run_dir / "validation_validation_metrics.json"),
        "holdout_metrics": read_json(run_dir / "holdout_validation_metrics.json"),
        "validation_backtest": read_json(run_dir / "validation_backtest_metrics.json"),
        "holdout_backtest": read_json(run_dir / "holdout_backtest_metrics.json"),
        "validation_market_equal_weight": market["validation"],
        "holdout_market_equal_weight": market["holdout"],
        "sanity": sanity,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def write_error_record(run_dir: Path, variant_name: str, exc: Exception) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    error = {
        "variant": variant_name,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "failed_at": now(),
    }
    path = run_dir / "run_error.json"
    path.write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def sanity_checks(predictions: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = ["y_score", "return_pred", "direction_prob", "g_price", "g_text", "g_fundamental", "label_next_open_return"]
    nan_counts = {col: int(pd.to_numeric(predictions[col], errors="coerce").isna().sum()) for col in numeric_cols if col in predictions.columns}
    if any(value > 0 for value in nan_counts.values()):
        raise RuntimeError(f"NaN found in predictions: {nan_counts}")
    return {
        "rows": int(len(predictions)),
        "days": int(pd.to_datetime(predictions["target_trade_date"]).nunique()),
        "date_min": str(pd.to_datetime(predictions["target_trade_date"]).min().date()),
        "date_max": str(pd.to_datetime(predictions["target_trade_date"]).max().date()),
        "nan_counts": nan_counts,
        "gate_mean": {
            "g_price": float(pd.to_numeric(predictions["g_price"], errors="coerce").mean()),
            "g_text": float(pd.to_numeric(predictions["g_text"], errors="coerce").mean()),
            "g_fundamental": float(pd.to_numeric(predictions["g_fundamental"], errors="coerce").mean()),
        },
    }


def market_baseline(predictions: pd.DataFrame) -> dict[str, Any]:
    frame = predictions.copy()
    frame["target_trade_date"] = pd.to_datetime(frame["target_trade_date"]).dt.normalize()
    daily = frame.groupby("target_trade_date")["label_next_open_return"].mean().dropna()
    total = float((1.0 + daily).prod() - 1.0) if len(daily) else float("nan")
    annual = float((1.0 + total) ** (252.0 / len(daily)) - 1.0) if len(daily) else float("nan")
    vol = float(daily.std(ddof=0)) if len(daily) else float("nan")
    sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if vol and np.isfinite(vol) and vol > 0 else float("nan")
    return {"total_return": total, "annual_return": annual, "sharpe": sharpe, "day_count": int(len(daily))}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_summary(variant: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": variant, "run_dir": summary.get("run_dir"), "config": summary.get("config")}
    for split in ("validation", "holdout"):
        metrics = summary.get(f"{split}_metrics", {})
        backtest = summary.get(f"{split}_backtest", {})
        market = summary.get(f"{split}_market_equal_weight", {})
        row[f"{split}_rank_ic"] = metrics.get("rank_ic_mean")
        row[f"{split}_icir"] = metrics.get("rank_ic_ir")
        row[f"{split}_topk_return"] = metrics.get("topk_return_mean")
        row[f"{split}_direction_accuracy"] = metrics.get("direction_accuracy")
        row[f"{split}_total_return"] = backtest.get("total_return", backtest.get("period_return"))
        row[f"{split}_annual_return"] = backtest.get("annual_return")
        row[f"{split}_sharpe"] = backtest.get("sharpe")
        row[f"{split}_max_drawdown"] = backtest.get("max_drawdown")
        market_total = market.get("total_return")
        model_total = row[f"{split}_total_return"]
        row[f"{split}_excess_equal_weight"] = (
            float(model_total) - float(market_total) if model_total is not None and market_total is not None else np.nan
        )
    return row


def flatten_error(variant: str, run_dir: Path, config_path: Path, error: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": variant,
        "status": "failed",
        "run_dir": str(run_dir),
        "config": str(config_path),
        "error_type": error.get("error_type"),
        "error": error.get("error"),
        "failed_at": error.get("failed_at"),
    }


def write_matrix_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def set_nested(config: MSGCAConfig, dotted_key: str, value: Any) -> None:
    target: Any = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def select_variants(matrix: str) -> list[dict[str, Any]]:
    variants = variant_definitions()
    groups = {
        "first_wave": [
            "price_only_no_factors_h64",
            "selected_factors_no_price_h64",
            "selected_factors_no_news_h64",
            "fa_base_topk002_h64",
            "fa_topk000_h64",
            "fa_topk005_h64",
            "fa_topk010_h64",
            "fa_recent_mild_topk005_h64",
            "fa_recent_strong_topk005_h64",
            "fa_proto2_topk005_h64",
            "fa_no_factor_gate_proto2_topk005_h64",
            "fa_scale_small_proto2_topk005_h48",
            "fa_scale_large_proto2_topk005_h96",
        ],
        "factor": [
            "price_only_no_factors_h64",
            "selected_factors_no_price_h64",
            "selected_factors_no_news_h64",
            "fa_base_topk002_h64",
        ],
        "factor_clean": [
            "clean_price_only_no_factors_h64",
            "clean_selected_manual_only_no_price_h64",
            "clean_selected_all_no_price_h64",
            "clean_price_plus_selected_manual_h64",
            "clean_price_plus_news_only_h64",
            "clean_price_plus_all_selected_h64",
        ],
        "gpt_final": [
            "gpt_final_upgrade_h48_proto2_topk005_sparsegate_train20260520",
        ],
        "gpt_final_soft": [
            "gpt_final_upgrade_h48_proto2_topk005_softgate_train20260520",
            "gpt_final_upgrade_h48_proto2_topk005_softgate_recent_train20260520",
        ],
        "topk": ["fa_topk000_h64", "fa_base_topk002_h64", "fa_topk005_h64", "fa_topk010_h64"],
        "time": ["fa_base_topk002_h64", "fa_recent_mild_topk005_h64", "fa_recent_strong_topk005_h64"],
        "interaction": [
            "fa_base_topk002_h64",
            "fa_proto2_topk005_h64",
            "fa_proto3_topk005_h64",
            "fa_no_factor_gate_proto2_topk005_h64",
        ],
        "scale": [
            "fa_scale_small_proto2_topk005_h48",
            "fa_proto2_topk005_h64",
            "fa_scale_large_proto2_topk005_h96",
        ],
        "extended": list(variants),
    }
    selected = groups[matrix]
    lookup = {variant["name"]: variant for variant in variants}
    return [deepcopy(lookup[name]) for name in selected]


def variant_definitions() -> list[dict[str, Any]]:
    mild_bins = [
        {"start": "2019-01-01", "end": "2022-12-31", "weight": 0.7},
        {"start": "2023-01-01", "end": "2023-12-31", "weight": 0.9},
        {"start": "2024-01-01", "end": "2024-12-31", "weight": 1.2},
        {"start": "2025-01-01", "end": "2025-09-30", "weight": 1.6},
    ]
    strong_bins = [
        {"start": "2019-01-01", "end": "2022-12-31", "weight": 0.5},
        {"start": "2023-01-01", "end": "2023-12-31", "weight": 0.8},
        {"start": "2024-01-01", "end": "2024-12-31", "weight": 1.5},
        {"start": "2025-01-01", "end": "2025-09-30", "weight": 2.0},
    ]
    final_recent_bins = [
        {"start": "2019-01-01", "end": "2022-12-31", "weight": 0.5},
        {"start": "2023-01-01", "end": "2023-12-31", "weight": 0.8},
        {"start": "2024-01-01", "end": "2024-12-31", "weight": 1.2},
        {"start": "2025-01-01", "end": "2025-12-31", "weight": 1.6},
        {"start": "2026-01-01", "end": "2026-05-20", "weight": 2.0},
    ]
    return [
        {
            "name": "price_only_no_factors_h64",
            "overrides": {
                "paths.evaluation_dir": str(DEFAULT_EVALUATION_ROOT / "__no_selected_features__"),
                "data.use_polars": False,
                "model.factor_encoder": "simple",
                "model.enable_news": False,
                "model.enable_fundamental": False,
                "train.topk_return_loss_weight": 0.02,
            },
        },
        {
            "name": "clean_price_only_no_factors_h64",
            "overrides": {
                "paths.evaluation_dir": str(DEFAULT_EVALUATION_ROOT / "__no_selected_features__"),
                "data.use_polars": False,
                "model.factor_encoder": "simple",
                "model.enable_news": False,
                "model.enable_fundamental": False,
                "train.topk_return_loss_weight": 0.02,
            },
        },
        {
            "name": "clean_selected_manual_only_no_price_h64",
            "overrides": {
                "model.enable_price": False,
                "model.enable_news": False,
                "model.enable_fundamental": True,
                "train.topk_return_loss_weight": 0.02,
            },
        },
        {
            "name": "clean_selected_all_no_price_h64",
            "overrides": {
                "model.enable_price": False,
                "model.enable_news": True,
                "model.enable_fundamental": True,
                "train.topk_return_loss_weight": 0.02,
            },
        },
        {
            "name": "clean_price_plus_selected_manual_h64",
            "overrides": {
                "model.enable_price": True,
                "model.enable_news": False,
                "model.enable_fundamental": True,
                "train.topk_return_loss_weight": 0.02,
            },
        },
        {
            "name": "clean_price_plus_news_only_h64",
            "overrides": {
                "model.enable_price": True,
                "model.enable_news": True,
                "model.enable_fundamental": False,
                "train.topk_return_loss_weight": 0.02,
            },
        },
        {
            "name": "clean_price_plus_all_selected_h64",
            "overrides": {
                "model.enable_price": True,
                "model.enable_news": True,
                "model.enable_fundamental": True,
                "train.topk_return_loss_weight": 0.02,
            },
        },
        {
            "name": "gpt_final_upgrade_h48_proto2_topk005_sparsegate_train20260520",
            "overrides": {
                "paths.evaluation_dir": str(DEFAULT_EVALUATION_ROOT / "final"),
                "data.train_start": "2019-01-01",
                "data.train_end": "2026-05-20",
                "data.validation_start": "2026-05-01",
                "data.validation_end": "2026-05-20",
                "data.holdout_start": "2026-05-01",
                "model.hidden_dim": 48,
                "model.n_heads": 4,
                "model.price_layers": 1,
                "model.factor_layers": 1,
                "model.factor_group_layers": 1,
                "model.factor_group_prototypes": 2,
                "model.factor_encoder": "factor_aware",
                "model.factor_gate_activation": "sparsemax",
                "model.enable_price": True,
                "model.enable_news": True,
                "model.enable_fundamental": True,
                "model.use_factor_gate": True,
                "train.topk_return_loss_weight": 0.05,
                "train.topk_temperature": 0.1,
            },
        },
        {
            "name": "gpt_final_upgrade_h48_proto2_topk005_softgate_train20260520",
            "overrides": {
                "paths.evaluation_dir": str(DEFAULT_EVALUATION_ROOT / "final"),
                "data.train_start": "2019-01-01",
                "data.train_end": "2026-05-20",
                "data.validation_start": "2026-05-01",
                "data.validation_end": "2026-05-20",
                "data.holdout_start": "2026-05-01",
                "model.hidden_dim": 48,
                "model.n_heads": 4,
                "model.price_layers": 1,
                "model.factor_layers": 1,
                "model.factor_group_layers": 1,
                "model.factor_group_prototypes": 2,
                "model.factor_encoder": "factor_aware",
                "model.factor_gate_activation": "softmax",
                "model.enable_price": True,
                "model.enable_news": True,
                "model.enable_fundamental": True,
                "model.use_factor_gate": True,
                "train.topk_return_loss_weight": 0.05,
                "train.topk_temperature": 0.1,
            },
        },
        {
            "name": "gpt_final_upgrade_h48_proto2_topk005_softgate_recent_train20260520",
            "overrides": {
                "paths.evaluation_dir": str(DEFAULT_EVALUATION_ROOT / "final"),
                "data.train_start": "2019-01-01",
                "data.train_end": "2026-05-20",
                "data.validation_start": "2026-05-01",
                "data.validation_end": "2026-05-20",
                "data.holdout_start": "2026-05-01",
                "model.hidden_dim": 48,
                "model.n_heads": 4,
                "model.price_layers": 1,
                "model.factor_layers": 1,
                "model.factor_group_layers": 1,
                "model.factor_group_prototypes": 2,
                "model.factor_encoder": "factor_aware",
                "model.factor_gate_activation": "softmax",
                "model.enable_price": True,
                "model.enable_news": True,
                "model.enable_fundamental": True,
                "model.use_factor_gate": True,
                "train.topk_return_loss_weight": 0.05,
                "train.topk_temperature": 0.1,
                "train.time_weight_bins": final_recent_bins,
                "train.normalize_time_weights": False,
            },
        },
        {
            "name": "selected_factors_no_price_h64",
            "overrides": {
                "model.enable_price": False,
                "train.topk_return_loss_weight": 0.02,
            },
        },
        {
            "name": "selected_factors_no_news_h64",
            "overrides": {
                "model.enable_news": False,
                "train.topk_return_loss_weight": 0.02,
            },
        },
        {"name": "fa_base_topk002_h64", "overrides": {"train.topk_return_loss_weight": 0.02}},
        {"name": "fa_topk000_h64", "overrides": {"train.topk_return_loss_weight": 0.0}},
        {"name": "fa_topk005_h64", "overrides": {"train.topk_return_loss_weight": 0.05}},
        {"name": "fa_topk010_h64", "overrides": {"train.topk_return_loss_weight": 0.10}},
        {
            "name": "fa_recent_mild_topk005_h64",
            "overrides": {
                "train.topk_return_loss_weight": 0.05,
                "train.time_weight_bins": mild_bins,
            },
        },
        {
            "name": "fa_recent_strong_topk005_h64",
            "overrides": {
                "train.topk_return_loss_weight": 0.05,
                "train.time_weight_bins": strong_bins,
            },
        },
        {
            "name": "fa_proto2_topk005_h64",
            "overrides": {
                "train.topk_return_loss_weight": 0.05,
                "model.factor_group_prototypes": 2,
            },
        },
        {
            "name": "fa_proto3_topk005_h64",
            "overrides": {
                "train.topk_return_loss_weight": 0.05,
                "model.factor_group_prototypes": 3,
            },
        },
        {
            "name": "fa_no_factor_gate_proto2_topk005_h64",
            "overrides": {
                "train.topk_return_loss_weight": 0.05,
                "model.factor_group_prototypes": 2,
                "model.use_factor_gate": False,
            },
        },
        {
            "name": "fa_scale_small_proto2_topk005_h48",
            "overrides": {
                "train.topk_return_loss_weight": 0.05,
                "model.hidden_dim": 48,
                "model.n_heads": 4,
                "model.price_layers": 1,
                "model.factor_layers": 1,
                "model.factor_group_layers": 1,
                "model.factor_group_prototypes": 2,
            },
        },
        {
            "name": "fa_scale_large_proto2_topk005_h96",
            "overrides": {
                "train.topk_return_loss_weight": 0.05,
                "model.hidden_dim": 96,
                "model.n_heads": 4,
                "model.price_layers": 2,
                "model.factor_layers": 2,
                "model.factor_group_layers": 2,
                "model.factor_group_prototypes": 2,
            },
        },
    ]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
