from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from FactorMiner.evaluation import quality as quality_eval
from FactorMiner.evaluation import review_selection as review_selection_eval
from FactorMiner.evaluation import selection as selection_eval
from FactorMiner.evaluation import single_factor as single_factor_eval
from FactorMiner.evaluation import slice_summary as slice_summary_eval
from model.msgca.feature_set import load_sample_feature_panel, load_selected_features
from aitrader_paths import DATASETS_ROOT


DEFAULT_DATASETS_ROOT = DATASETS_ROOT
DEFAULT_PROCESSED_DIR = DEFAULT_DATASETS_ROOT / "processed"
DEFAULT_FACTOR_ROOT = DEFAULT_DATASETS_ROOT / "factors"
DEFAULT_FEATURE_ROOT = DEFAULT_DATASETS_ROOT / "features"
DEFAULT_EVALUATION_ROOT = DEFAULT_FACTOR_ROOT / "evaluation"
DEFAULT_FINAL_EVALUATION_DIR = DEFAULT_EVALUATION_ROOT / "final"
DEFAULT_EXPERIMENT_EVALUATION_ROOT = DEFAULT_EVALUATION_ROOT / "experiment"
DEFAULT_FEATURE_REGISTRY_PATH = DEFAULT_FEATURE_ROOT / "feature_registry.json"
DEFAULT_MODEL_FEATURE_ROOT = DEFAULT_DATASETS_ROOT / "model_features"
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run date-safe factor selection or assemble inference feature panels.")
    parser.add_argument("--mode", choices=("select", "inference"), required=True)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTOR_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument("--feature-registry-path", type=Path, default=None)
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--select-since", default=None, help="Inclusive train target_trade_date lower bound.")
    parser.add_argument("--select-until", default=None, help="Inclusive train target_trade_date upper bound.")
    parser.add_argument("--select-engine", choices=("full", "slice"), default="full")
    parser.add_argument("--source-evaluation-dir", type=Path, default=DEFAULT_FINAL_EVALUATION_DIR)
    parser.add_argument("--blocks", default="all", help="Feature blocks used by select mode.")
    parser.add_argument("--labels", default=",".join(single_factor_eval.DEFAULT_LABELS))
    parser.add_argument("--quality-max-missing-rate", type=float, default=0.98)
    parser.add_argument("--quality-min-non-missing", type=int, default=100)
    parser.add_argument("--quality-min-year-coverage", type=float, default=0.01)
    parser.add_argument("--quality-workers", type=int, default=1, help="Parallel workers for quality block checks.")
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--single-factor-workers", type=int, default=1, help="Parallel workers for full single-factor evaluation.")
    parser.add_argument("--primary-label", default=selection_eval.DEFAULT_PRIMARY_LABEL)
    parser.add_argument("--secondary-label", default="label_next_vwap_return")
    parser.add_argument("--min-rank-ic-days", type=int, default=60)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--min-abs-rank-ic", type=float, default=0.01)
    parser.add_argument("--min-abs-rank-ic-ir", type=float, default=0.0)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--min-corr-pairs", type=int, default=10_000)
    parser.add_argument("--corr-method", choices=("spearman", "pearson"), default="spearman")
    parser.add_argument("--corr-row-limit", type=int, default=0, help="0 means use all rows for correlation de-duplication.")
    parser.add_argument("--max-selected", type=int, default=0)
    parser.add_argument("--prepare-review", action="store_true")
    parser.add_argument("--review-profile", choices=("research", "competition"), default="research")
    parser.add_argument("--skip-registry-validate", action="store_true")
    parser.add_argument("--full-registry-validate", action="store_true")

    parser.add_argument("--selected-features-path", type=Path, default=None)
    parser.add_argument("--target-date", default=None, help="Inference target_trade_date, YYYY-MM-DD.")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--include-labels", action="store_true")
    parser.add_argument("--no-strict-features", dest="strict_features", action="store_false")
    parser.set_defaults(strict_features=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_paths(args)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.mode == "select":
        result = run_select(args)
    else:
        result = run_inference(args)
    print(" ".join(f"{key}={value}" for key, value in _printable_result(result).items()))
    return 0


def run_select(args: argparse.Namespace) -> dict[str, Any]:
    since = _parse_required_date(args.select_since, "select-since")
    until = _parse_required_date(args.select_until, "select-until")
    if since > until:
        raise ValueError("--select-since cannot be later than --select-until")
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    args.summary_path = args.summary_path or args.evaluation_dir / "factor_workflow_summary.json"

    summary = _initial_summary(
        args,
        {
            "select_since": since.date().isoformat(),
            "select_until": until.date().isoformat(),
            "select_engine": args.select_engine,
            "source_evaluation_dir": str(args.source_evaluation_dir),
            "review_profile": args.review_profile,
            "corr_row_limit": args.corr_row_limit,
            "quality_workers": args.quality_workers,
            "single_factor_workers": args.single_factor_workers,
        },
    )
    _write_json(args.summary_path, summary)
    stages: list[tuple[str, Callable[[], dict[str, Any]]]]
    if args.select_engine == "slice":
        stages = [
            ("quality", lambda: _run_quality(args, since, until)),
            ("slice_summary", lambda: _run_slice_summary(args, since, until)),
            ("selection", lambda: _run_selection(args)),
        ]
    else:
        stages = [
            ("quality", lambda: _run_quality(args, since, until)),
            ("single_factor", lambda: _run_single_factor(args, since, until)),
            ("selection", lambda: _run_selection(args)),
        ]
    if args.prepare_review:
        stages.append(("review_prepare", lambda: _run_review_prepare(args)))
    for name, action in stages:
        _run_stage(name, action, args.summary_path, summary)
    summary["status"] = "ok"
    summary["finished_at"] = _utc_now()
    _write_json(args.summary_path, summary)
    return {
        "mode": "select",
        "evaluation_dir": str(args.evaluation_dir),
        "summary_path": str(args.summary_path),
        "selected_features_path": str(args.evaluation_dir / "selected_features.json"),
    }


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    target_date = _parse_required_date(args.target_date, "target-date")
    selected = load_selected_features(args.evaluation_dir, explicit_path=args.selected_features_path)
    samples = _load_target_samples(args.samples_path, target_date, args.include_labels)
    if samples.empty:
        raise ValueError(f"No samples found for target_trade_date={target_date.date().isoformat()}")

    LOGGER.info("inference_feature_load_start target_date=%s samples=%s selected=%s", target_date.date().isoformat(), len(samples), len(selected.selected_features))
    panel = load_sample_feature_panel(
        args.feature_registry_path,
        selected.selected_features,
        sample_ids=samples["sample_id"],
        strict=args.strict_features,
    )
    LOGGER.info("inference_feature_load_done rows=%s columns=%s", len(panel), len(panel.columns))

    output = samples.merge(panel, on="sample_id", how="left")
    missing = [feature for feature in selected.selected_features if feature not in output.columns]
    if missing:
        raise KeyError(f"Selected features missing from inference panel: {missing[:10]}")
    ordered_columns = [*samples.columns.tolist(), *selected.selected_features]
    output = output.loc[:, list(dict.fromkeys(ordered_columns))]

    output_path = args.output_path or _default_inference_output_path(target_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    metadata = {
        "mode": "inference",
        "target_date": target_date.date().isoformat(),
        "samples_path": str(args.samples_path),
        "feature_registry_path": str(args.feature_registry_path),
        "selected_features_path": str(selected.source_path),
        "selected_mode": selected.mode,
        "row_count": int(len(output)),
        "feature_count": int(len(selected.selected_features)),
        "output_path": str(output_path),
        "created_at": _utc_now(),
    }
    metadata_path = output_path.with_suffix(".json")
    _write_json(metadata_path, metadata)
    return {**metadata, "metadata_path": str(metadata_path)}


def _run_quality(args: argparse.Namespace, since: pd.Timestamp, until: pd.Timestamp) -> dict[str, Any]:
    quality_args = argparse.Namespace(
        samples_path=args.samples_path,
        feature_registry_path=args.feature_registry_path,
        output_dir=args.evaluation_dir,
        blocks=args.blocks,
        since=since.date().isoformat(),
        until=until.date().isoformat(),
        max_missing_rate=args.quality_max_missing_rate,
        min_non_missing=args.quality_min_non_missing,
        min_year_coverage=args.quality_min_year_coverage,
        workers=args.quality_workers,
        skip_registry_validate=args.skip_registry_validate,
        full_registry_validate=args.full_registry_validate,
    )
    return quality_eval.run_quality(quality_args)


def _run_single_factor(args: argparse.Namespace, since: pd.Timestamp, until: pd.Timestamp) -> dict[str, Any]:
    single_args = argparse.Namespace(
        samples_path=args.samples_path,
        feature_registry_path=args.feature_registry_path,
        output_dir=args.evaluation_dir,
        blocks=args.blocks,
        labels=args.labels,
        since=since.date().isoformat(),
        until=until.date().isoformat(),
        min_pairs=args.min_pairs,
        groups=args.groups,
        workers=args.single_factor_workers,
        skip_registry_validate=args.skip_registry_validate,
        full_registry_validate=args.full_registry_validate,
    )
    return single_factor_eval.run_single_factor(single_args)


def _run_slice_summary(args: argparse.Namespace, since: pd.Timestamp, until: pd.Timestamp) -> dict[str, Any]:
    slice_args = argparse.Namespace(
        source_dir=args.source_evaluation_dir,
        output_dir=args.evaluation_dir,
        since=since.date().isoformat(),
        until=until.date().isoformat(),
        chunksize=500_000,
        no_copy_quality=True,
    )
    return slice_summary_eval.run_slice_summary(slice_args)


def _run_selection(args: argparse.Namespace) -> dict[str, Any]:
    selection_args = argparse.Namespace(
        quality_path=args.evaluation_dir / "sample_feature_quality.csv",
        factor_summary_path=args.evaluation_dir / "factor_summary.csv",
        feature_registry_path=args.feature_registry_path,
        samples_path=args.samples_path,
        output_dir=args.evaluation_dir,
        primary_label=args.primary_label,
        secondary_label=args.secondary_label,
        since=args.select_since,
        until=args.select_until,
        min_rank_ic_days=args.min_rank_ic_days,
        min_coverage=args.min_coverage,
        min_abs_rank_ic=args.min_abs_rank_ic,
        min_abs_rank_ic_ir=args.min_abs_rank_ic_ir,
        corr_threshold=args.corr_threshold,
        min_corr_pairs=args.min_corr_pairs,
        corr_method=args.corr_method,
        corr_row_limit=args.corr_row_limit,
        max_selected=args.max_selected,
        skip_registry_validate=args.skip_registry_validate,
        full_registry_validate=args.full_registry_validate,
    )
    return selection_eval.run_selection(selection_args)


def _run_review_prepare(args: argparse.Namespace) -> dict[str, Any]:
    review_args = argparse.Namespace(
        selected_features_path=args.evaluation_dir / "selected_features.json",
        candidate_features_path=args.evaluation_dir / "candidate_features.csv",
        review_packet_path=args.evaluation_dir / "review_packet.json",
        correlation_clusters_path=args.evaluation_dir / "correlation_clusters.csv",
        correlation_conflicts_path=args.evaluation_dir / "correlation_conflicts.csv",
        response_path=None,
        output_dir=args.evaluation_dir,
        prompt_path=args.evaluation_dir / "review_prompt.md",
        response_template_path=args.evaluation_dir / "review_response_template.json",
        review_inputs_path=args.evaluation_dir / "review_inputs.txt",
        reviewed_json_path=args.evaluation_dir / "selected_features_reviewed.json",
        reviewed_csv_path=args.evaluation_dir / "selected_features_reviewed.csv",
        audit_path=args.evaluation_dir / "selection_review_audit.csv",
        report_path=args.evaluation_dir / "selection_review_report.md",
        review_profile=args.review_profile,
        prepare=True,
        apply=False,
        max_selected_preview=250,
        max_rejected_preview=150,
        max_borderline_preview=120,
        max_cluster_preview=80,
        allow_quality_failed_add_back=False,
    )
    return review_selection_eval.prepare_review(review_args)


def _load_target_samples(path: Path, target_date: pd.Timestamp, include_labels: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing samples parquet: {path}")
    available = set(pq.ParquetFile(path).schema.names)
    columns = [
        column
        for column in (
            "sample_id",
            "stock_code",
            "stock_name",
            "industry",
            "target_trade_date",
            "feature_asof_date",
            "decision_ts",
        )
        if column in available
    ]
    if "sample_id" not in columns or "target_trade_date" not in columns:
        raise KeyError("samples must include sample_id and target_trade_date")
    if include_labels:
        columns.extend(sorted(column for column in available if column.startswith("label_") and column not in columns))
    samples = pd.read_parquet(path, columns=columns)
    samples = samples.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["target_trade_date"] = pd.to_datetime(samples["target_trade_date"], errors="coerce").dt.normalize()
    samples = samples.loc[samples["target_trade_date"].eq(target_date)].copy()
    for column in ("feature_asof_date", "decision_ts"):
        if column in samples.columns:
            samples[column] = pd.to_datetime(samples[column], errors="coerce")
    if samples["sample_id"].duplicated().any():
        examples = samples.loc[samples["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"samples.sample_id must be unique for target date. Duplicate examples: {examples}")
    return samples.reset_index(drop=True)


def _resolve_paths(args: argparse.Namespace) -> None:
    args.samples_path = args.samples_path or args.processed_dir / "samples.parquet"
    args.feature_registry_path = args.feature_registry_path or args.feature_root / "feature_registry.json"
    if args.mode == "select":
        if args.evaluation_dir is None:
            since = _date_label(args.select_since, "start")
            until = _date_label(args.select_until, "end")
            suffix = "_slice" if args.select_engine == "slice" else ""
            args.evaluation_dir = args.factor_root / "evaluation" / "experiment" / f"select_{since}_{until}{suffix}"
    else:
        args.evaluation_dir = args.evaluation_dir or args.factor_root / "evaluation" / "final"


def _run_stage(
    name: str,
    action: Callable[[], dict[str, Any]],
    summary_path: Path,
    summary: dict[str, Any],
) -> None:
    record: dict[str, Any] = {"stage": name, "started_at": _utc_now()}
    summary["stages"].append(record)
    _write_json(summary_path, summary)
    LOGGER.info("stage_start stage=%s", name)
    start = time.perf_counter()
    try:
        result = action()
    except Exception as exc:
        record.update({"status": "failed", "finished_at": _utc_now(), "duration_sec": round(time.perf_counter() - start, 3), "error": f"{type(exc).__name__}: {exc}"})
        summary["status"] = "failed"
        summary["finished_at"] = _utc_now()
        _write_json(summary_path, summary)
        LOGGER.exception("stage_failed stage=%s", name)
        raise
    record.update({"status": "ok", "finished_at": _utc_now(), "duration_sec": round(time.perf_counter() - start, 3), "result": _jsonable(result)})
    _write_json(summary_path, summary)
    LOGGER.info("stage_done stage=%s duration_sec=%s", name, record["duration_sec"])


def _initial_summary(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "mode": args.mode,
        "paths": {
            "samples_path": str(args.samples_path),
            "feature_registry_path": str(args.feature_registry_path),
            "evaluation_dir": str(args.evaluation_dir),
        },
        "config": config,
        "stages": [],
    }


def _parse_required_date(value: str | None, name: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"--{name} is required")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --{name} date: {value}")
    return parsed.normalize()


def _date_label(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    return _parse_required_date(value, "date").strftime("%Y%m%d")


def _default_inference_output_path(target_date: pd.Timestamp) -> Path:
    date_label = target_date.strftime("%Y%m%d")
    return DEFAULT_MODEL_FEATURE_ROOT / "inference" / f"target_date={date_label}" / "features.parquet"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
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


def _printable_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("mode", "evaluation_dir", "summary_path", "selected_features_path", "target_date", "row_count", "feature_count", "output_path", "metadata_path")
    return {key: result[key] for key in keys if key in result}


if __name__ == "__main__":
    raise SystemExit(main())
