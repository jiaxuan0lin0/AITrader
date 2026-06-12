from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import signal
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from FactorMiner.build import daily as daily_build
from FactorMiner.build import news_sample as news_sample_build
from FactorMiner.build import sample_features as sample_features_build
from FactorMiner.core.registry import validate_registry
from FactorMiner.evaluation import quality as quality_eval
from FactorMiner.evaluation import review_selection as review_selection_eval
from FactorMiner.evaluation import selection as selection_eval
from FactorMiner.evaluation import single_factor as single_factor_eval
from aitrader_paths import DATASETS_ROOT


DEFAULT_DATASETS_ROOT = DATASETS_ROOT
DEFAULT_PROCESSED_DIR = DEFAULT_DATASETS_ROOT / "processed"
DEFAULT_FACTOR_ROOT = DEFAULT_DATASETS_ROOT / "factors"
DEFAULT_FEATURE_ROOT = DEFAULT_DATASETS_ROOT / "features"
DEFAULT_EVALUATION_DIR = DEFAULT_FACTOR_ROOT / "evaluation" / "experiment" / "pipeline"
DEFAULT_FACTOR_REGISTRY_PATH = DEFAULT_FACTOR_ROOT / "factor_registry.json"
DEFAULT_FEATURE_REGISTRY_PATH = DEFAULT_FEATURE_ROOT / "feature_registry.json"
DEFAULT_NEWS_SCORES_PATH = DEFAULT_FACTOR_ROOT / "news_llm_scores.parquet"
DEFAULT_DAILY_BLOCKS = ("metric", "moneyflow", "alpha158")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the FactorMiner pipeline from processed data and cached news LLM scores "
            "through feature construction, evaluation, and automatic selection."
        )
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTOR_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument("--factor-registry-path", type=Path, default=None)
    parser.add_argument("--feature-registry-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)

    parser.add_argument("--price-path", type=Path, default=None)
    parser.add_argument("--metric-path", type=Path, default=None)
    parser.add_argument("--moneyflow-path", type=Path, default=None)
    parser.add_argument("--basic-path", type=Path, default=None)
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument("--news-path", type=Path, default=None)
    parser.add_argument("--news-scores-path", type=Path, default=None)

    parser.add_argument(
        "--daily-blocks",
        default=",".join(DEFAULT_DAILY_BLOCKS),
        help=(
            "Comma-separated daily blocks, or all. Default is metric,moneyflow,alpha158."
        ),
    )
    parser.add_argument(
        "--daily-block",
        choices=(*daily_build.DAILY_BLOCKS, "all"),
        default=None,
        help="Deprecated alias for selecting one daily block.",
    )
    parser.add_argument("--sample-blocks", default="all")
    parser.add_argument("--evaluation-blocks", default="all")
    parser.add_argument("--news-windows", default="1,3,5,10")
    parser.add_argument(
        "--news-scope",
        choices=("split", "all", "market", "stock"),
        default="split",
        help="Build news sample factors as split market/stock blocks by default, or one legacy all block.",
    )
    parser.add_argument("--since", default=None, help="Optional lower date bound for partial runs.")
    parser.add_argument("--until", default=None, help="Optional upper date bound for partial runs.")
    parser.add_argument("--stock-limit", type=int, default=None, help="Optional stock count limit. Requires non-default --factor-root.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample/news row limit. Requires non-default output roots.")

    parser.add_argument("--disable-neutral", action="store_true")
    parser.add_argument("--alpha-workers", type=int, default=6, help="Parallel workers for split alpha158 daily blocks.")
    parser.add_argument(
        "--alpha-layout",
        choices=("split", "single"),
        default="split",
        help="Materialize alpha158 as restartable sub-blocks or one legacy block.",
    )
    parser.add_argument("--max-industry-missing-rate", type=float, default=0.20)
    parser.add_argument("--sample-feature-workers", type=int, default=4, help="Parallel workers for sample feature alignment.")

    parser.add_argument("--quality-max-missing-rate", type=float, default=0.98)
    parser.add_argument("--quality-min-non-missing", type=int, default=100)
    parser.add_argument("--quality-min-year-coverage", type=float, default=0.01)
    parser.add_argument("--quality-workers", type=int, default=1, help="Parallel workers for quality block checks.")

    parser.add_argument("--labels", default=",".join(single_factor_eval.DEFAULT_LABELS))
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--single-factor-workers", type=int, default=1, help="Parallel workers for single-factor block evaluation.")

    parser.add_argument("--primary-label", default=selection_eval.DEFAULT_PRIMARY_LABEL)
    parser.add_argument("--secondary-label", default="label_next_vwap_return")
    parser.add_argument("--min-rank-ic-days", type=int, default=60)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--min-abs-rank-ic", type=float, default=0.01)
    parser.add_argument("--min-abs-rank-ic-ir", type=float, default=0.0)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--min-corr-pairs", type=int, default=10_000)
    parser.add_argument("--corr-method", choices=("spearman", "pearson"), default="spearman")
    parser.add_argument("--corr-row-limit", type=int, default=0)
    parser.add_argument("--max-selected", type=int, default=0)

    parser.add_argument("--prepare-review", action="store_true", help="Generate review_prompt.md after automatic selection.")
    parser.add_argument("--review-profile", choices=("research", "competition"), default="research")
    parser.add_argument("--skip-precheck", action="store_true")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-daily-validate", action="store_true")
    parser.add_argument("--skip-sample-features", action="store_true")
    parser.add_argument("--skip-news-sample", action="store_true")
    parser.add_argument("--skip-feature-validate", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-single-factor", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--skip-registry-validate", action="store_true", help="Skip registry validation inside evaluation stages.")
    parser.add_argument(
        "--full-registry-validate",
        action="store_true",
        help="Use full parquet reads for pipeline registry validation. Default uses parquet metadata to avoid OOM.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_paths(args)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    summary = _initial_summary(args)
    _write_summary(args.summary_path, summary)
    _install_signal_handlers(args, summary)

    stages = _build_stages(args)
    for name, action in stages:
        _run_stage(name, action, args, summary)

    summary["finished_at"] = _utc_now()
    summary["status"] = "ok"
    _write_summary(args.summary_path, summary)
    logging.info("pipeline_done summary=%s", args.summary_path)
    return 0


def _resolve_paths(args: argparse.Namespace) -> None:
    args.evaluation_dir = args.evaluation_dir or args.factor_root / "evaluation" / "experiment" / "pipeline"
    args.factor_registry_path = args.factor_registry_path or args.factor_root / "factor_registry.json"
    args.feature_registry_path = args.feature_registry_path or args.feature_root / "feature_registry.json"
    args.summary_path = args.summary_path or args.evaluation_dir / "pipeline_summary.json"

    args.price_path = args.price_path or args.processed_dir / "price.parquet"
    args.metric_path = args.metric_path or args.processed_dir / "metric.parquet"
    args.moneyflow_path = args.moneyflow_path or args.processed_dir / "moneyflow.parquet"
    args.basic_path = args.basic_path or args.processed_dir / "basic.parquet"
    args.samples_path = args.samples_path or args.processed_dir / "samples.parquet"
    args.news_path = args.news_path or args.processed_dir / "news.parquet"
    args.news_scores_path = args.news_scores_path or args.factor_root / "news_llm_scores.parquet"
    args.daily_blocks_tuple = _parse_daily_blocks(args.daily_block, args.daily_blocks)


def _build_stages(args: argparse.Namespace) -> list[tuple[str, Callable[[], dict[str, Any]]]]:
    stages: list[tuple[str, Callable[[], dict[str, Any]]]] = []
    if not args.skip_precheck:
        stages.append(("precheck", lambda: _precheck_inputs(args)))
    if not args.skip_daily:
        for block_key in args.daily_blocks_tuple:
            stages.append((f"daily_{block_key}", lambda block_key=block_key: _run_daily(args, block_key)))
    if not args.skip_daily and not args.skip_daily_validate:
        stages.append(("daily_validate", lambda: _validate(args.factor_registry_path, args)))
    if not args.skip_sample_features:
        stages.append(("sample_features", lambda: _run_sample_features(args)))
    if not args.skip_news_sample:
        stages.append(("news_sample", lambda: _run_news_sample(args)))
    if not args.skip_feature_validate:
        stages.append(("feature_validate", lambda: _validate(args.feature_registry_path, args)))
    if not args.skip_quality:
        stages.append(("quality", lambda: _run_quality(args)))
    if not args.skip_single_factor:
        stages.append(("single_factor", lambda: _run_single_factor(args)))
    if not args.skip_selection:
        stages.append(("selection", lambda: _run_selection(args)))
    if args.prepare_review:
        stages.append(("review_prepare", lambda: _run_review_prepare(args)))
    return stages


def _run_stage(
    name: str,
    action: Callable[[], dict[str, Any]],
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> None:
    record: dict[str, Any] = {"stage": name, "started_at": _utc_now()}
    summary["stages"].append(record)
    _write_summary(args.summary_path, summary)
    logging.info("stage_start stage=%s", name)
    start = time.perf_counter()
    try:
        result = action()
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "duration_sec": round(time.perf_counter() - start, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        summary["status"] = "failed"
        summary["finished_at"] = _utc_now()
        _write_summary(args.summary_path, summary)
        logging.exception("stage_failed stage=%s", name)
        raise

    record.update(
        {
            "status": "ok",
            "finished_at": _utc_now(),
            "duration_sec": round(time.perf_counter() - start, 3),
            "result": _jsonable(result),
        }
    )
    _write_summary(args.summary_path, summary)
    logging.info("stage_done stage=%s duration_sec=%s", name, record["duration_sec"])


def _precheck_inputs(args: argparse.Namespace) -> dict[str, Any]:
    required: list[Path] = []
    if not args.skip_daily:
        if "alpha158" in args.daily_blocks_tuple or "moneyflow" in args.daily_blocks_tuple:
            required.append(args.price_path)
        if "metric" in args.daily_blocks_tuple:
            required.append(args.metric_path)
        if "moneyflow" in args.daily_blocks_tuple:
            required.append(args.moneyflow_path)
        required.append(args.basic_path)
    if not args.skip_sample_features or not args.skip_news_sample or not args.skip_quality or not args.skip_single_factor:
        required.append(args.samples_path)
    if not args.skip_news_sample:
        required.extend([args.news_path, args.news_scores_path])
    if args.skip_daily and not args.skip_sample_features:
        required.append(args.factor_registry_path)
    if args.skip_sample_features and args.skip_news_sample and (
        not args.skip_quality or not args.skip_single_factor or not args.skip_selection
    ):
        required.append(args.feature_registry_path)

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")
    return {"checked_files": [str(path) for path in required], "missing_count": 0}


def _run_daily(args: argparse.Namespace, block_key: str) -> dict[str, Any]:
    daily_args = argparse.Namespace(
        price_path=args.price_path,
        metric_path=args.metric_path,
        moneyflow_path=args.moneyflow_path,
        basic_path=args.basic_path,
        output_root=args.factor_root,
        registry_path=args.factor_registry_path,
        block=block_key,
        since=args.since,
        until=args.until,
        stock_limit=args.stock_limit,
        disable_neutral=args.disable_neutral,
        alpha_workers=args.alpha_workers,
        alpha_layout=args.alpha_layout,
        max_industry_missing_rate=args.max_industry_missing_rate,
        validate_only=False,
    )
    blocks = daily_build.build_daily_blocks(daily_args)
    return {"blocks": [block.to_record() for block in blocks], "registry_path": str(args.factor_registry_path)}


def _run_sample_features(args: argparse.Namespace) -> dict[str, Any]:
    sample_args = argparse.Namespace(
        samples_path=args.samples_path,
        source_registry_path=args.factor_registry_path,
        output_root=args.feature_root,
        feature_registry_path=args.feature_registry_path,
        blocks=args.sample_blocks,
        since=args.since,
        until=args.until,
        limit=args.limit,
        workers=args.sample_feature_workers,
        validate_source=not args.skip_registry_validate,
        full_registry_validate=args.full_registry_validate,
        overwrite=False,
        validate_only=False,
    )
    blocks = sample_features_build.build_sample_feature_blocks(sample_args)
    return {"blocks": [block.to_record() for block in blocks], "feature_registry_path": str(args.feature_registry_path)}


def _run_news_sample(args: argparse.Namespace) -> dict[str, Any]:
    news_args = argparse.Namespace(
        samples_path=args.samples_path,
        news_path=args.news_path,
        scores_path=args.news_scores_path,
        output_root=args.feature_root,
        feature_registry_path=args.feature_registry_path,
        block_name=news_sample_build.DEFAULT_BLOCK_NAME,
        output_path=None,
        manifest_path=None,
        windows=args.news_windows,
        scope=args.news_scope,
        since=args.since,
        until=args.until,
        limit=args.limit,
        skip_registry=False,
        full_registry_validate=args.full_registry_validate,
        validate_only=False,
    )
    blocks = news_sample_build.build_news_sample_blocks(news_args)
    return {"blocks": [block.to_record() for block in blocks], "feature_registry_path": str(args.feature_registry_path)}


def _run_quality(args: argparse.Namespace) -> dict[str, Any]:
    quality_args = argparse.Namespace(
        samples_path=args.samples_path,
        feature_registry_path=args.feature_registry_path,
        output_dir=args.evaluation_dir,
        blocks=args.evaluation_blocks,
        since=args.since,
        until=args.until,
        max_missing_rate=args.quality_max_missing_rate,
        min_non_missing=args.quality_min_non_missing,
        min_year_coverage=args.quality_min_year_coverage,
        workers=args.quality_workers,
        skip_registry_validate=args.skip_registry_validate,
        full_registry_validate=args.full_registry_validate,
    )
    return quality_eval.run_quality(quality_args)


def _run_single_factor(args: argparse.Namespace) -> dict[str, Any]:
    single_factor_args = argparse.Namespace(
        samples_path=args.samples_path,
        feature_registry_path=args.feature_registry_path,
        output_dir=args.evaluation_dir,
        blocks=args.evaluation_blocks,
        labels=args.labels,
        since=args.since,
        until=args.until,
        min_pairs=args.min_pairs,
        groups=args.groups,
        workers=args.single_factor_workers,
        skip_registry_validate=args.skip_registry_validate,
        full_registry_validate=args.full_registry_validate,
    )
    return single_factor_eval.run_single_factor(single_factor_args)


def _run_selection(args: argparse.Namespace) -> dict[str, Any]:
    selection_args = argparse.Namespace(
        quality_path=args.evaluation_dir / "sample_feature_quality.csv",
        factor_summary_path=args.evaluation_dir / "factor_summary.csv",
        feature_registry_path=args.feature_registry_path,
        samples_path=args.samples_path,
        output_dir=args.evaluation_dir,
        primary_label=args.primary_label,
        secondary_label=args.secondary_label,
        since=args.since,
        until=args.until,
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


def _validate(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    metadata_only = not args.full_registry_validate
    validate_registry(path, metadata_only=metadata_only)
    return {"registry_path": str(path), "valid": True, "metadata_only": metadata_only}


def _initial_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "paths": {
            "processed_dir": str(args.processed_dir),
            "factor_root": str(args.factor_root),
            "feature_root": str(args.feature_root),
            "evaluation_dir": str(args.evaluation_dir),
            "factor_registry_path": str(args.factor_registry_path),
            "feature_registry_path": str(args.feature_registry_path),
            "news_scores_path": str(args.news_scores_path),
        },
        "config": _config_summary(args),
        "stages": [],
    }


def _config_summary(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "daily_blocks",
        "sample_blocks",
        "evaluation_blocks",
        "news_windows",
        "news_scope",
        "since",
        "until",
        "disable_neutral",
        "alpha_workers",
        "alpha_layout",
        "max_industry_missing_rate",
        "sample_feature_workers",
        "full_registry_validate",
        "quality_workers",
        "labels",
        "single_factor_workers",
        "primary_label",
        "secondary_label",
        "min_rank_ic_days",
        "min_coverage",
        "min_abs_rank_ic",
        "corr_threshold",
        "corr_method",
        "corr_row_limit",
        "max_selected",
        "prepare_review",
    )
    summary = {key: getattr(args, key) for key in keys}
    summary["daily_blocks_resolved"] = list(args.daily_blocks_tuple)
    return summary


def _parse_daily_blocks(single_block: str | None, blocks_arg: str) -> tuple[str, ...]:
    if single_block:
        return _expand_daily_block_value(single_block)
    requested = [item.strip() for item in blocks_arg.split(",") if item.strip()]
    if not requested:
        raise ValueError("--daily-blocks must include at least one block")
    if "all" in requested:
        return ("metric", "moneyflow", "alpha158")
    allowed = set(daily_build.DAILY_BLOCKS)
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(f"Unknown daily blocks: {invalid}. Allowed: {sorted(allowed)} or all")
    ordered: list[str] = []
    for item in requested:
        if item not in ordered:
            ordered.append(item)
    return tuple(ordered)


def _expand_daily_block_value(value: str) -> tuple[str, ...]:
    if value == "all":
        return ("metric", "moneyflow", "alpha158")
    return (value,)


def _install_signal_handlers(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    def _handle_signal(signum: int, _frame: object) -> None:
        now = _utc_now()
        if summary.get("stages"):
            current = summary["stages"][-1]
            if current.get("status") is None:
                current["status"] = "terminated"
                current["finished_at"] = now
                current["error"] = f"Received signal {signum}"
        summary["status"] = "terminated"
        summary["finished_at"] = now
        _write_summary(args.summary_path, summary)
        logging.warning("pipeline_terminated signal=%s summary=%s", signum, args.summary_path)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")


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


if __name__ == "__main__":
    raise SystemExit(main())
