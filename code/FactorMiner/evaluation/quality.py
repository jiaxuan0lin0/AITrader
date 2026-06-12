from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from FactorMiner.core.factor_block import FactorBlock
from FactorMiner.core.registry import FactorRegistry, resolve_registry_path, validate_registry
from aitrader_paths import DATASETS_ROOT


DEFAULT_SAMPLES_PATH = DATASETS_ROOT / "processed" / "samples.parquet"
DEFAULT_FEATURE_REGISTRY_PATH = DATASETS_ROOT / "features" / "feature_registry.json"
DEFAULT_OUTPUT_DIR = DATASETS_ROOT / "factors" / "evaluation" / "experiment" / "ad_hoc"
LOGGER = logging.getLogger(__name__)
_WORKER_SAMPLES: pd.DataFrame | None = None
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_BASE_DIR: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run quality checks for sample-level feature blocks.")
    parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--feature-registry-path", type=Path, default=DEFAULT_FEATURE_REGISTRY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blocks", default="all", help="Comma-separated feature block names, or all.")
    parser.add_argument("--since", default=None, help="Inclusive target_trade_date lower bound.")
    parser.add_argument("--until", default=None, help="Inclusive target_trade_date upper bound.")
    parser.add_argument("--max-missing-rate", type=float, default=0.98)
    parser.add_argument("--min-non-missing", type=int, default=100)
    parser.add_argument("--min-year-coverage", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=1, help="Number of sample feature blocks to check in parallel.")
    parser.add_argument("--skip-registry-validate", action="store_true")
    parser.add_argument("--full-registry-validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_quality(args)
    print(
        " ".join(
            [
                f"blocks={result['block_count']}",
                f"factors={result['factor_count']}",
                f"quality_pass={result['quality_pass_count']}",
                f"quality_report={result['quality_report_path']}",
                f"block_report={result['block_report_path']}",
            ]
        )
    )
    return 0


def run_quality(args: argparse.Namespace) -> dict[str, Any]:
    since = _parse_optional_date(getattr(args, "since", None), "since")
    until = _parse_optional_date(getattr(args, "until", None), "until")
    _validate_thresholds(args, since, until)
    if not args.skip_registry_validate:
        validate_registry(args.feature_registry_path, metadata_only=not args.full_registry_validate)

    samples = _load_samples(args.samples_path, since, until)
    registry = FactorRegistry.load(args.feature_registry_path)
    blocks = _select_sample_blocks(registry, args.blocks)
    if not blocks:
        raise ValueError("No sample feature blocks selected for quality checks")

    base_dir = args.feature_registry_path.parent
    workers = int(getattr(args, "workers", 1) or 1)
    if workers > 1:
        factor_rows, block_rows = _evaluate_blocks_parallel(blocks, base_dir, samples, args, workers)
    else:
        factor_rows, block_rows = _evaluate_blocks_serial(blocks, base_dir, samples, args)

    quality = pd.DataFrame(factor_rows)
    block_quality = pd.DataFrame(block_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_report_path = args.output_dir / "sample_feature_quality.csv"
    block_report_path = args.output_dir / "sample_feature_block_quality.csv"
    summary_path = args.output_dir / "sample_feature_quality_summary.json"
    quality.to_csv(quality_report_path, index=False)
    block_quality.to_csv(block_report_path, index=False)

    summary = {
        "samples_path": str(args.samples_path),
        "feature_registry_path": str(args.feature_registry_path),
        "since": str(since.date()) if since is not None else None,
        "until": str(until.date()) if until is not None else None,
        "sample_count": int(len(samples)),
        "block_count": int(len(blocks)),
        "workers": workers,
        "factor_count": int(len(quality)),
        "quality_pass_count": int(quality["quality_pass"].sum()) if not quality.empty else 0,
        "quality_report_path": str(quality_report_path),
        "block_report_path": str(block_report_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _evaluate_blocks_serial(
    blocks: list[FactorBlock],
    base_dir: Path,
    samples: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    factor_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for block in blocks:
        LOGGER.info("quality_block_start block=%s factors=%s rows=%s", block.name, block.factor_count, block.row_count)
        block_factor_rows, block_row = _evaluate_block(block, base_dir, samples, args)
        factor_rows.extend(block_factor_rows)
        block_rows.append(block_row)
        LOGGER.info("quality_block_done block=%s factors=%s", block.name, len(block_factor_rows))
    return factor_rows, block_rows


def _evaluate_blocks_parallel(
    blocks: list[FactorBlock],
    base_dir: Path,
    samples: pd.DataFrame,
    args: argparse.Namespace,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        context = mp.get_context("fork")
    except ValueError:
        LOGGER.warning("quality_parallel_unavailable reason=no_fork_context")
        return _evaluate_blocks_serial(blocks, base_dir, samples, args)

    max_workers = min(workers, len(blocks))
    LOGGER.info("quality_parallel_start workers=%s blocks=%s", max_workers, len(blocks))
    part_records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(samples, args, base_dir),
    ) as executor:
        futures = {
            executor.submit(_evaluate_block_part, index, block): block
            for index, block in enumerate(blocks)
        }
        for future in as_completed(futures):
            block = futures[future]
            record = future.result()
            part_records.append(record)
            LOGGER.info("quality_block_done block=%s factors=%s", block.name, len(record["factor_rows"]))

    factor_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for record in sorted(part_records, key=lambda item: int(item["order"])):
        factor_rows.extend(record["factor_rows"])
        block_rows.append(record["block_row"])
    LOGGER.info("quality_parallel_done workers=%s factors=%s", max_workers, len(factor_rows))
    return factor_rows, block_rows


def _init_worker(samples: pd.DataFrame, args: argparse.Namespace, base_dir: Path) -> None:
    global _WORKER_SAMPLES, _WORKER_ARGS, _WORKER_BASE_DIR
    _WORKER_SAMPLES = samples
    _WORKER_ARGS = args
    _WORKER_BASE_DIR = base_dir


def _evaluate_block_part(order: int, block: FactorBlock) -> dict[str, Any]:
    if _WORKER_SAMPLES is None or _WORKER_ARGS is None or _WORKER_BASE_DIR is None:
        raise RuntimeError("quality worker state is not initialized")
    LOGGER.info("quality_block_start block=%s factors=%s rows=%s", block.name, block.factor_count, block.row_count)
    factor_rows, block_row = _evaluate_block(block, _WORKER_BASE_DIR, _WORKER_SAMPLES, _WORKER_ARGS)
    return {"order": order, "factor_rows": factor_rows, "block_row": block_row}


def _load_samples(path: Path, since: pd.Timestamp | None, until: pd.Timestamp | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing samples parquet: {path}")
    samples = pd.read_parquet(path, columns=["sample_id", "target_trade_date"])
    _require_columns(samples, ("sample_id", "target_trade_date"), "samples")
    work = samples[["sample_id", "target_trade_date"]].copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["target_trade_date"] = pd.to_datetime(work["target_trade_date"], errors="coerce").dt.normalize()
    if work["sample_id"].duplicated().any():
        examples = work.loc[work["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"samples.sample_id must be unique. Duplicate examples: {examples}")
    if work["target_trade_date"].isna().any():
        missing = int(work["target_trade_date"].isna().sum())
        raise ValueError(f"samples.target_trade_date contains missing or invalid dates: {missing}")
    if since is not None:
        work = work.loc[work["target_trade_date"].ge(since)].copy()
    if until is not None:
        work = work.loc[work["target_trade_date"].le(until)].copy()
    work["target_year"] = work["target_trade_date"].dt.year.astype("int64")
    return work.reset_index(drop=True)


def _select_sample_blocks(registry: FactorRegistry, blocks_arg: str) -> list[FactorBlock]:
    sample_blocks = [block for block in registry.blocks if block.granularity == "sample"]
    if blocks_arg == "all":
        return sample_blocks
    requested = [item.strip() for item in blocks_arg.split(",") if item.strip()]
    by_name = {block.name: block for block in sample_blocks}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise KeyError(f"Unknown sample feature blocks: {missing}")
    return [by_name[name] for name in requested]


def _evaluate_block(
    block: FactorBlock,
    base_dir: Path,
    samples: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    factor_path = resolve_registry_path(block.factor_path, base_dir)
    manifest_path = resolve_registry_path(block.manifest_path, base_dir)
    manifest_records = _load_manifest_records(manifest_path)
    factor_names = [str(record["name"]) for record in manifest_records]
    frame = pd.read_parquet(factor_path, columns=[*block.key_columns, *factor_names])
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)

    sample_index = pd.Index(samples["sample_id"])
    frame = frame.loc[frame["sample_id"].isin(sample_index)].copy()
    block_index = pd.Index(frame["sample_id"].unique())
    matched_mask = block_index.isin(sample_index)
    matched_sample_count = int(matched_mask.sum())
    missing_sample_count = int(len(samples) - matched_sample_count)
    extra_sample_count = int((~matched_mask).sum())
    block_row = {
        "block": block.name,
        "row_count": int(len(frame)),
        "factor_count": int(len(factor_names)),
        "sample_count": int(len(samples)),
        "matched_sample_count": int(matched_sample_count),
        "missing_sample_count": int(missing_sample_count),
        "extra_sample_count": int(extra_sample_count),
        "sample_match_rate": _safe_rate(matched_sample_count, len(samples)),
        "factor_path": str(factor_path),
        "manifest_path": str(manifest_path),
    }

    spec_by_name = {str(record["name"]): record for record in manifest_records}
    sample_year_by_id = samples.set_index("sample_id")["target_year"]
    frame["target_year"] = frame["sample_id"].map(sample_year_by_id)
    rows: list[dict[str, Any]] = []
    for factor_name in factor_names:
        spec = spec_by_name[factor_name]
        rows.append(_evaluate_factor(block, frame, factor_name, spec, frame["target_year"], block_row, args))
    return rows, block_row


def _evaluate_factor(
    block: FactorBlock,
    frame: pd.DataFrame,
    factor_name: str,
    spec: dict[str, Any],
    target_years: pd.Series,
    block_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw = frame[factor_name]
    numeric = pd.to_numeric(raw, errors="coerce")
    raw_missing = raw.isna()
    non_numeric = raw.notna() & numeric.isna()
    finite = np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
    valid = pd.Series(finite, index=frame.index)
    row_count = len(frame)
    non_missing_count = int(valid.sum())
    missing_count = int(row_count - non_missing_count)
    inf_count = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
    non_numeric_count = int(non_numeric.sum())
    values = numeric.loc[valid]

    year_stats = _year_coverage(target_years, valid)
    flags = _quality_flags(
        values,
        row_count,
        non_missing_count,
        inf_count,
        non_numeric_count,
        year_stats,
        args,
    )
    return {
        "block": block.name,
        "factor_name": factor_name,
        "source": spec.get("source", ""),
        "category": spec.get("category", ""),
        "availability": spec.get("availability", ""),
        "window": spec.get("window"),
        "lookback": spec.get("lookback"),
        "row_count": int(row_count),
        "sample_count": int(block_row["sample_count"]),
        "block_sample_match_rate": block_row["sample_match_rate"],
        "non_missing_count": non_missing_count,
        "missing_count": missing_count,
        "raw_nan_count": int(raw_missing.sum()),
        "inf_count": inf_count,
        "non_numeric_count": non_numeric_count,
        "missing_rate": _safe_rate(missing_count, row_count),
        "non_missing_rate": _safe_rate(non_missing_count, row_count),
        "zero_rate": _zero_rate(values),
        "unique_count": int(values.nunique(dropna=True)),
        "constant_flag": bool(values.nunique(dropna=True) <= 1),
        "mean": _stat(values, "mean"),
        "std": _stat(values, "std"),
        "min": _stat(values, "min"),
        "p01": _quantile(values, 0.01),
        "p05": _quantile(values, 0.05),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": _stat(values, "max"),
        **year_stats,
        "quality_pass": not flags,
        "quality_flags": ";".join(flags),
    }


def _year_coverage(target_years: pd.Series, valid: pd.Series) -> dict[str, Any]:
    work = pd.DataFrame({"target_year": target_years, "__valid": valid.to_numpy(dtype=bool)})
    work = work.dropna(subset=["target_year"])
    if work.empty:
        return {
            "year_count": 0,
            "year_coverage_min": np.nan,
            "year_coverage_max": np.nan,
            "year_coverage_std": np.nan,
            "worst_year": pd.NA,
        }
    coverage = work.groupby("target_year")["__valid"].mean().sort_index()
    worst_year = int(coverage.idxmin()) if not coverage.empty else pd.NA
    return {
        "year_count": int(len(coverage)),
        "year_coverage_min": float(coverage.min()) if not coverage.empty else np.nan,
        "year_coverage_max": float(coverage.max()) if not coverage.empty else np.nan,
        "year_coverage_std": float(coverage.std(ddof=0)) if len(coverage) > 1 else 0.0,
        "worst_year": worst_year,
    }


def _quality_flags(
    values: pd.Series,
    row_count: int,
    non_missing_count: int,
    inf_count: int,
    non_numeric_count: int,
    year_stats: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    flags: list[str] = []
    missing_rate = _safe_rate(row_count - non_missing_count, row_count)
    if non_numeric_count > 0:
        flags.append("non_numeric_values")
    if inf_count > 0:
        flags.append("has_inf")
    if missing_rate > args.max_missing_rate:
        flags.append("missing_rate_high")
    if non_missing_count < args.min_non_missing:
        flags.append("non_missing_too_low")
    if values.nunique(dropna=True) <= 1:
        flags.append("constant")
    year_min = year_stats.get("year_coverage_min")
    if pd.notna(year_min) and float(year_min) < args.min_year_coverage:
        flags.append("year_coverage_low")
    return flags


def _load_manifest_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Factor manifest must contain a JSON list: {path}")
    records: list[dict[str, Any]] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record must be an object at index {index}: {path}")
        if not record.get("name"):
            raise ValueError(f"Manifest record missing FactorSpec.name at index {index}: {path}")
        records.append(record)
    return records


def _parse_optional_date(value: str | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --{name} date: {value}")
    return parsed.normalize()


def _validate_thresholds(args: argparse.Namespace, since: pd.Timestamp | None, until: pd.Timestamp | None) -> None:
    if not 0 <= args.max_missing_rate <= 1:
        raise ValueError("--max-missing-rate must be between 0 and 1")
    if args.min_non_missing < 0:
        raise ValueError("--min-non-missing cannot be negative")
    if not 0 <= args.min_year_coverage <= 1:
        raise ValueError("--min-year-coverage must be between 0 and 1")
    if int(getattr(args, "workers", 1) or 1) <= 0:
        raise ValueError("--workers must be positive")
    if since is not None and until is not None and since > until:
        raise ValueError("--since cannot be later than --until")


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _zero_rate(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    return float(values.eq(0).mean())


def _stat(values: pd.Series, name: str) -> float:
    if values.empty:
        return np.nan
    return float(getattr(values, name)())


def _quantile(values: pd.Series, q: float) -> float:
    if values.empty:
        return np.nan
    return float(values.quantile(q))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
