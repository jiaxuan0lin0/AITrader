from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from dataclasses import replace
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq

from FactorMiner.core.alignment import align_daily_factors_to_samples
from FactorMiner.core.factor_block import FactorBlock, write_factor_block
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.core.registry import FactorRegistry, resolve_registry_path, upsert_block, validate_registry
from aitrader_paths import DATASETS_ROOT


DEFAULT_SAMPLES_PATH = DATASETS_ROOT / "processed" / "samples.parquet"
DEFAULT_SOURCE_REGISTRY_PATH = DATASETS_ROOT / "factors" / "factor_registry.json"
DEFAULT_FEATURE_ROOT = DATASETS_ROOT / "features"
DEFAULT_FEATURE_REGISTRY_PATH = DEFAULT_FEATURE_ROOT / "feature_registry.json"
LOGGER = logging.getLogger(__name__)
_SAMPLE_WORKER_SAMPLES: pd.DataFrame | None = None
_SAMPLE_WORKER_SOURCE_BASE_DIR: Path | None = None
_SAMPLE_WORKER_OUTPUT_ROOT: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align daily factor blocks to sample-level feature blocks.")
    parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--source-registry-path", type=Path, default=DEFAULT_SOURCE_REGISTRY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--feature-registry-path", type=Path, default=DEFAULT_FEATURE_REGISTRY_PATH)
    parser.add_argument("--blocks", default="all", help="Comma-separated source block names, or all.")
    parser.add_argument("--since", default=None, help="Inclusive target_trade_date lower bound for sample filtering.")
    parser.add_argument("--until", default=None, help="Inclusive target_trade_date upper bound for sample filtering.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample row limit.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for sample feature block alignment.")
    parser.add_argument("--validate-source", action="store_true", help="Validate source factor registry before alignment.")
    parser.add_argument(
        "--full-registry-validate",
        action="store_true",
        help="Use full parquet reads during registry validation. Default uses parquet metadata to avoid OOM.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild sample feature blocks even if completed outputs exist.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate the feature registry.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        validate_registry(args.feature_registry_path, metadata_only=not args.full_registry_validate)
        print(f"feature_registry_valid={args.feature_registry_path}")
        return 0

    blocks = build_sample_feature_blocks(args)
    for block in blocks:
        print(
            " ".join(
                [
                    f"block={block.name}",
                    f"rows={block.row_count}",
                    f"factors={block.factor_count}",
                    f"factor_path={block.factor_path}",
                    f"manifest_path={block.manifest_path}",
                ]
            )
        )
    print(f"feature_registry={args.feature_registry_path}")
    return 0


def build_sample_feature_blocks(args: argparse.Namespace) -> list[FactorBlock]:
    since = _parse_optional_date(args.since, "since")
    until = _parse_optional_date(args.until, "until")
    if since is not None and until is not None and since > until:
        raise ValueError("--since cannot be later than --until")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    workers = int(getattr(args, "workers", 4) or 4)
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if args.limit is not None and _same_path(args.output_root, DEFAULT_FEATURE_ROOT):
        raise ValueError("--limit requires a non-default --output-root")
    if args.validate_source:
        validate_registry(args.source_registry_path, metadata_only=not args.full_registry_validate)

    samples = _load_samples(args.samples_path, since, until, args.limit)
    source_registry = FactorRegistry.load(args.source_registry_path)
    source_blocks = _select_source_blocks(source_registry, args.blocks)
    if not source_blocks:
        raise ValueError("No daily source blocks selected for sample feature alignment")

    written_by_order: dict[int, FactorBlock] = {}
    pending: list[tuple[int, FactorBlock]] = []
    source_base_dir = args.source_registry_path.parent
    for order, source_block in enumerate(source_blocks):
        existing = None if args.overwrite else _completed_feature_block(source_block, samples, args)
        if existing is not None:
            LOGGER.info("sample_feature_block_skip block=%s rows=%s factors=%s", existing.name, existing.row_count, existing.factor_count)
            written_by_order[order] = existing
            continue
        pending.append((order, source_block))

    if pending and workers > 1:
        for order, feature_block in _align_blocks_parallel(pending, samples, source_base_dir, args.output_root, workers):
            upsert_block(args.feature_registry_path, feature_block)
            written_by_order[order] = feature_block

    if pending and workers <= 1:
        for order, source_block in pending:
            LOGGER.info("sample_feature_block_start source_block=%s factors=%s", source_block.name, source_block.factor_count)
            result = _align_one_block(samples, source_block, source_base_dir)
            feature_block = _write_sample_feature_block(source_block, result, args.output_root)
            upsert_block(args.feature_registry_path, feature_block)
            written_by_order[order] = feature_block
            del result
            gc.collect()
            LOGGER.info("sample_feature_block_done block=%s rows=%s factors=%s", feature_block.name, feature_block.row_count, feature_block.factor_count)

    validate_registry(args.feature_registry_path, metadata_only=not args.full_registry_validate)
    written = [written_by_order[index] for index in sorted(written_by_order)]
    return written


def _align_blocks_parallel(
    pending: list[tuple[int, FactorBlock]],
    samples: pd.DataFrame,
    source_base_dir: Path,
    output_root: Path,
    workers: int,
) -> list[tuple[int, FactorBlock]]:
    try:
        context = mp.get_context("fork")
    except ValueError:
        LOGGER.warning("sample_feature_parallel_unavailable reason=no_fork_context")
        return _align_blocks_serial(pending, samples, source_base_dir, output_root)

    max_workers = min(workers, len(pending))
    LOGGER.info("sample_feature_parallel_start workers=%s blocks=%s", max_workers, len(pending))
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_init_sample_worker,
        initargs=(samples, source_base_dir, output_root),
    ) as executor:
        futures = {
            executor.submit(_align_sample_block_part, order, source_block): source_block
            for order, source_block in pending
        }
        for future in as_completed(futures):
            source_block = futures[future]
            record = future.result()
            records.append(record)
            block_record = record["block"]
            row_count = block_record["row_count"] if isinstance(block_record, dict) else "unknown"
            LOGGER.info("sample_feature_block_done block=%s_sample rows=%s", source_block.name, row_count)

    blocks: list[tuple[int, FactorBlock]] = []
    for record in sorted(records, key=lambda item: int(item["order"])):
        blocks.append((int(record["order"]), FactorBlock.from_record(record["block"])))  # type: ignore[arg-type]
    LOGGER.info("sample_feature_parallel_done workers=%s blocks=%s", max_workers, len(blocks))
    return blocks


def _align_blocks_serial(
    pending: list[tuple[int, FactorBlock]],
    samples: pd.DataFrame,
    source_base_dir: Path,
    output_root: Path,
) -> list[tuple[int, FactorBlock]]:
    blocks: list[tuple[int, FactorBlock]] = []
    for order, source_block in pending:
        LOGGER.info("sample_feature_block_start source_block=%s factors=%s", source_block.name, source_block.factor_count)
        result = _align_one_block(samples, source_block, source_base_dir)
        feature_block = _write_sample_feature_block(source_block, result, output_root)
        blocks.append((order, feature_block))
        del result
        gc.collect()
        LOGGER.info("sample_feature_block_done block=%s rows=%s factors=%s", feature_block.name, feature_block.row_count, feature_block.factor_count)
    return blocks


def _init_sample_worker(samples: pd.DataFrame, source_base_dir: Path, output_root: Path) -> None:
    global _SAMPLE_WORKER_SAMPLES, _SAMPLE_WORKER_SOURCE_BASE_DIR, _SAMPLE_WORKER_OUTPUT_ROOT
    _SAMPLE_WORKER_SAMPLES = samples
    _SAMPLE_WORKER_SOURCE_BASE_DIR = source_base_dir
    _SAMPLE_WORKER_OUTPUT_ROOT = output_root


def _align_sample_block_part(order: int, source_block: FactorBlock) -> dict[str, object]:
    if (
        _SAMPLE_WORKER_SAMPLES is None
        or _SAMPLE_WORKER_SOURCE_BASE_DIR is None
        or _SAMPLE_WORKER_OUTPUT_ROOT is None
    ):
        raise RuntimeError("sample feature worker state is not initialized")
    LOGGER.info("sample_feature_block_start source_block=%s factors=%s", source_block.name, source_block.factor_count)
    result = _align_one_block(_SAMPLE_WORKER_SAMPLES, source_block, _SAMPLE_WORKER_SOURCE_BASE_DIR)
    feature_block = _write_sample_feature_block(source_block, result, _SAMPLE_WORKER_OUTPUT_ROOT)
    return {"order": order, "block": feature_block.to_record()}


def _load_samples(
    path: Path,
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
    limit: int | None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing samples parquet: {path}")
    columns = ["sample_id", "stock_code", "feature_asof_date"]
    available = set(pq.ParquetFile(path).schema.names)
    if "target_trade_date" in available:
        columns.append("target_trade_date")
    samples = pd.read_parquet(path, columns=columns)
    _require_columns(samples, ("sample_id", "stock_code", "feature_asof_date"), "samples")
    samples = samples.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["stock_code"] = samples["stock_code"].astype(str)
    samples["feature_asof_date"] = pd.to_datetime(samples["feature_asof_date"], errors="coerce").dt.normalize()
    date_column = "target_trade_date" if "target_trade_date" in samples.columns else "feature_asof_date"
    dates = pd.to_datetime(samples[date_column], errors="coerce").dt.normalize()
    if since is not None:
        samples = samples.loc[dates.ge(since)].copy()
        dates = dates.loc[samples.index]
    if until is not None:
        samples = samples.loc[dates.le(until)].copy()
    if limit is not None:
        samples = samples.head(limit).copy()
    samples = samples.reset_index(drop=True)
    duplicated = samples.duplicated(["sample_id"])
    if duplicated.any():
        examples = samples.loc[duplicated, ["sample_id"]].head(5).to_dict("records")
        raise ValueError(f"samples.sample_id must be unique: {examples}")
    return samples


def _completed_feature_block(
    source_block: FactorBlock,
    samples: pd.DataFrame,
    args: argparse.Namespace,
) -> FactorBlock | None:
    block_name = f"{source_block.name}_sample"
    registry = FactorRegistry.load(args.feature_registry_path)
    block = next((item for item in registry.blocks if item.name == block_name), None)
    if block is None:
        return None
    factor_path = resolve_registry_path(block.factor_path, args.feature_registry_path.parent)
    manifest_path = resolve_registry_path(block.manifest_path, args.feature_registry_path.parent)
    if not factor_path.exists() or not manifest_path.exists():
        return None
    if block.factor_count != source_block.factor_count or block.row_count != len(samples):
        return None
    try:
        parquet = pq.ParquetFile(factor_path)
    except Exception:
        return None
    if parquet.metadata.num_rows != len(samples):
        return None
    columns = set(parquet.schema.names)
    if "sample_id" not in columns:
        return None
    specs = _load_specs(manifest_path)
    factor_names = [spec.name for spec in specs]
    if len(factor_names) != source_block.factor_count:
        return None
    if any(name not in columns for name in factor_names):
        return None
    return block


def _select_source_blocks(registry: FactorRegistry, blocks_arg: str) -> list[FactorBlock]:
    daily_blocks = [block for block in registry.blocks if block.granularity == "daily"]
    if blocks_arg == "all":
        return daily_blocks
    requested = [item.strip() for item in blocks_arg.split(",") if item.strip()]
    by_name = {block.name: block for block in daily_blocks}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise KeyError(f"Unknown daily source blocks: {missing}")
    return [by_name[name] for name in requested]


def _align_one_block(samples: pd.DataFrame, source_block: FactorBlock, source_base_dir: Path) -> FactorResult:
    factor_path = resolve_registry_path(source_block.factor_path, source_base_dir)
    manifest_path = resolve_registry_path(source_block.manifest_path, source_base_dir)
    if not factor_path.exists():
        raise FileNotFoundError(f"Missing source factor block parquet: {factor_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source factor manifest: {manifest_path}")

    specs = _load_specs(manifest_path)
    factor_names = [spec.name for spec in specs]
    daily = pd.read_parquet(factor_path, columns=[*source_block.key_columns, *factor_names])
    aligned = align_daily_factors_to_samples(samples, daily, factor_names)
    features = aligned[["sample_id", *factor_names]].copy()
    result = FactorResult(features, specs, key_columns=("sample_id",))
    result.validate()
    return result


def _write_sample_feature_block(source_block: FactorBlock, result: FactorResult, output_root: Path) -> FactorBlock:
    block_name = f"{source_block.name}_sample"
    factor_path = output_root / "blocks" / "sample" / f"{block_name}.parquet"
    manifest_path = output_root / "manifests" / f"{block_name}.json"
    block = write_factor_block(
        result,
        block_name,
        "sample",
        factor_path,
        manifest_path,
        description=f"Sample-aligned features from daily block {source_block.name}.",
    )
    return replace(
        block,
        factor_path=str(_relative_to(factor_path, output_root)),
        manifest_path=str(_relative_to(manifest_path, output_root)),
    )


def _load_specs(path: Path) -> list[FactorSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Factor manifest must contain a JSON list: {path}")
    specs: list[FactorSpec] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record must be an object at index {index}: {path}")
        item = dict(record)
        item["inputs"] = tuple(item.get("inputs", ()))
        specs.append(FactorSpec(**item))
    return specs


def _parse_optional_date(value: str | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --{name} date: {value}")
    return parsed.normalize()


def _relative_to(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
