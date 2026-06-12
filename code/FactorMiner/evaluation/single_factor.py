from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
import pandas as pd

from FactorMiner.core.factor_block import FactorBlock
from FactorMiner.core.registry import FactorRegistry, resolve_registry_path, validate_registry
from aitrader_paths import DATASETS_ROOT


DEFAULT_SAMPLES_PATH = DATASETS_ROOT / "processed" / "samples.parquet"
DEFAULT_FEATURE_REGISTRY_PATH = DATASETS_ROOT / "features" / "feature_registry.json"
DEFAULT_OUTPUT_DIR = DATASETS_ROOT / "factors" / "evaluation" / "experiment" / "ad_hoc"
DEFAULT_LABELS = ("label_next_open_return", "label_next_vwap_return")
LOGGER = logging.getLogger(__name__)
_WORKER_SAMPLES: pd.DataFrame | None = None
_WORKER_LABELS: tuple[str, ...] | None = None
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_BASE_DIR: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate sample-level features with single-factor IC metrics.")
    parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--feature-registry-path", type=Path, default=DEFAULT_FEATURE_REGISTRY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blocks", default="all", help="Comma-separated feature block names, or all.")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--since", default=None, help="Inclusive target_trade_date lower bound.")
    parser.add_argument("--until", default=None, help="Inclusive target_trade_date upper bound.")
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="Number of sample feature blocks to evaluate in parallel.")
    parser.add_argument("--skip-registry-validate", action="store_true")
    parser.add_argument("--full-registry-validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_single_factor(args)
    print(
        " ".join(
            [
                f"blocks={result['block_count']}",
                f"factors={result['factor_count']}",
                f"labels={result['label_count']}",
                f"daily_rows={result['daily_row_count']}",
                f"summary_rows={result['summary_row_count']}",
                f"summary={result['summary_path']}",
            ]
        )
    )
    return 0


def run_single_factor(args: argparse.Namespace) -> dict[str, Any]:
    labels = _parse_labels(args.labels)
    since = _parse_optional_date(args.since, "since")
    until = _parse_optional_date(args.until, "until")
    _validate_args(args, since, until)
    if not args.skip_registry_validate:
        validate_registry(args.feature_registry_path, metadata_only=not args.full_registry_validate)

    samples = _load_samples(args.samples_path, labels, since, until)
    registry = FactorRegistry.load(args.feature_registry_path)
    blocks = _select_sample_blocks(registry, args.blocks)
    if not blocks:
        raise ValueError("No sample feature blocks selected for single-factor evaluation")

    base_dir = args.feature_registry_path.parent
    workers = int(getattr(args, "workers", 1) or 1)
    if workers > 1:
        daily, group_part_paths, factor_meta, group_row_count, parts_dir = _evaluate_blocks_parallel(
            blocks,
            base_dir,
            samples,
            labels,
            args,
            workers,
        )
        groups: pd.DataFrame | None = None
    else:
        daily, groups, factor_meta = _evaluate_blocks_serial(blocks, base_dir, samples, labels, args)
        group_part_paths = []
        group_row_count = int(len(groups))
        parts_dir = None
    summary = _build_summary(daily, factor_meta, labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ic_path = args.output_dir / "factor_ic.csv"
    rankic_path = args.output_dir / "factor_rankic.csv"
    group_path = args.output_dir / "group_return.csv"
    summary_path = args.output_dir / "factor_summary.csv"
    metadata_path = args.output_dir / "single_factor_summary.json"

    _write_csv(_factor_ic_frame(daily), ic_path)
    _write_csv(_factor_rankic_frame(daily), rankic_path)
    if groups is None:
        _concat_csv_parts(group_part_paths, group_path, _group_return_columns())
        if parts_dir is not None:
            shutil.rmtree(parts_dir, ignore_errors=True)
    else:
        _write_csv(groups, group_path)
    _write_csv(summary, summary_path)

    result = {
        "samples_path": str(args.samples_path),
        "feature_registry_path": str(args.feature_registry_path),
        "sample_count": int(len(samples)),
        "block_count": int(len(blocks)),
        "factor_count": int(len(factor_meta)),
        "label_count": int(len(labels)),
        "workers": workers,
        "daily_row_count": int(len(daily)),
        "group_row_count": int(group_row_count),
        "summary_row_count": int(len(summary)),
        "ic_path": str(ic_path),
        "rankic_path": str(rankic_path),
        "group_return_path": str(group_path),
        "summary_path": str(summary_path),
    }
    metadata_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metadata_path"] = str(metadata_path)
    return result


def _evaluate_blocks_serial(
    blocks: list[FactorBlock],
    base_dir: Path,
    samples: pd.DataFrame,
    labels: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    daily_records: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    factor_meta: list[dict[str, Any]] = []
    for block in blocks:
        LOGGER.info("single_factor_block_start block=%s factors=%s rows=%s", block.name, block.factor_count, block.row_count)
        block_daily, block_groups, block_meta = _evaluate_block(block, base_dir, samples, labels, args)
        daily_records.extend(block_daily)
        group_records.extend(block_groups)
        factor_meta.extend(block_meta)
        LOGGER.info(
            "single_factor_block_done block=%s factors=%s daily_rows=%s group_rows=%s",
            block.name,
            len(block_meta),
            len(block_daily),
            len(block_groups),
        )
    return pd.DataFrame(daily_records), pd.DataFrame(group_records), factor_meta


def _evaluate_blocks_parallel(
    blocks: list[FactorBlock],
    base_dir: Path,
    samples: pd.DataFrame,
    labels: tuple[str, ...],
    args: argparse.Namespace,
    workers: int,
) -> tuple[pd.DataFrame, list[Path], list[dict[str, Any]], int, Path]:
    try:
        context = mp.get_context("fork")
    except ValueError:
        LOGGER.warning("single_factor_parallel_unavailable reason=no_fork_context")
        daily, groups, factor_meta = _evaluate_blocks_serial(blocks, base_dir, samples, labels, args)
        parts_dir = args.output_dir / "_single_factor_parts"
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        parts_dir.mkdir(parents=True, exist_ok=True)
        group_path = parts_dir / "0000_serial_group.csv"
        _write_csv(groups, group_path)
        return daily, [group_path], factor_meta, int(len(groups)), parts_dir

    max_workers = min(workers, len(blocks))
    parts_dir = args.output_dir / "_single_factor_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("single_factor_parallel_start workers=%s blocks=%s parts_dir=%s", max_workers, len(blocks), parts_dir)

    part_records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(samples, labels, args, base_dir),
    ) as executor:
        futures = {
            executor.submit(_evaluate_block_part, index, block, parts_dir): block
            for index, block in enumerate(blocks)
        }
        for future in as_completed(futures):
            block = futures[future]
            record = future.result()
            part_records.append(record)
            LOGGER.info(
                "single_factor_block_done block=%s factors=%s daily_rows=%s group_rows=%s",
                block.name,
                record["factor_count"],
                record["daily_row_count"],
                record["group_row_count"],
            )

    ordered = sorted(part_records, key=lambda record: int(record["order"]))
    daily_frames = [pd.read_parquet(record["daily_path"]) for record in ordered if int(record["daily_row_count"]) > 0]
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    factor_meta: list[dict[str, Any]] = []
    for record in ordered:
        factor_meta.extend(json.loads(Path(record["meta_path"]).read_text(encoding="utf-8")))
    group_paths = [Path(record["group_path"]) for record in ordered]
    group_row_count = int(sum(int(record["group_row_count"]) for record in ordered))
    LOGGER.info("single_factor_parallel_done workers=%s daily_rows=%s group_rows=%s", max_workers, len(daily), group_row_count)
    return daily, group_paths, factor_meta, group_row_count, parts_dir


def _init_worker(
    samples: pd.DataFrame,
    labels: tuple[str, ...],
    args: argparse.Namespace,
    base_dir: Path,
) -> None:
    global _WORKER_SAMPLES, _WORKER_LABELS, _WORKER_ARGS, _WORKER_BASE_DIR
    _WORKER_SAMPLES = samples
    _WORKER_LABELS = labels
    _WORKER_ARGS = args
    _WORKER_BASE_DIR = base_dir


def _evaluate_block_part(order: int, block: FactorBlock, parts_dir: Path) -> dict[str, Any]:
    if _WORKER_SAMPLES is None or _WORKER_LABELS is None or _WORKER_ARGS is None or _WORKER_BASE_DIR is None:
        raise RuntimeError("single_factor worker state is not initialized")
    LOGGER.info("single_factor_block_start block=%s factors=%s rows=%s", block.name, block.factor_count, block.row_count)
    block_daily, block_groups, block_meta = _evaluate_block(block, _WORKER_BASE_DIR, _WORKER_SAMPLES, _WORKER_LABELS, _WORKER_ARGS)
    prefix = f"{order:04d}_{block.name}"
    daily_path = parts_dir / f"{prefix}_daily.parquet"
    group_path = parts_dir / f"{prefix}_group.csv"
    meta_path = parts_dir / f"{prefix}_meta.json"
    pd.DataFrame(block_daily).to_parquet(daily_path, index=False)
    pd.DataFrame(block_groups, columns=_group_return_columns()).to_csv(group_path, index=False)
    meta_path.write_text(json.dumps(block_meta, ensure_ascii=False), encoding="utf-8")
    return {
        "order": order,
        "block": block.name,
        "factor_count": int(len(block_meta)),
        "daily_row_count": int(len(block_daily)),
        "group_row_count": int(len(block_groups)),
        "daily_path": str(daily_path),
        "group_path": str(group_path),
        "meta_path": str(meta_path),
    }


def _load_samples(
    path: Path,
    labels: tuple[str, ...],
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing samples parquet: {path}")
    columns = ["sample_id", "target_trade_date", *labels]
    samples = pd.read_parquet(path, columns=columns)
    _require_columns(samples, columns, "samples")
    work = samples.loc[:, columns].copy()
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
    for label in labels:
        work[label] = pd.to_numeric(work[label], errors="coerce")
    return work.reset_index(drop=True)


def _evaluate_block(
    block: FactorBlock,
    base_dir: Path,
    samples: pd.DataFrame,
    labels: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    factor_path = resolve_registry_path(block.factor_path, base_dir)
    manifest_path = resolve_registry_path(block.manifest_path, base_dir)
    manifest_records = _load_manifest_records(manifest_path)
    factor_names = [str(record["name"]) for record in manifest_records]
    frame = pd.read_parquet(factor_path, columns=[*block.key_columns, *factor_names])
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    data = samples.merge(frame, on="sample_id", how="inner")
    if data.empty:
        raise ValueError(f"No matching samples for feature block {block.name}")

    meta_by_name = {str(record["name"]): record for record in manifest_records}
    daily_records: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    factor_meta: list[dict[str, Any]] = []
    for factor_name in factor_names:
        spec = meta_by_name[factor_name]
        factor_meta.append(
            {
                "block": block.name,
                "factor_name": factor_name,
                "source": spec.get("source", ""),
                "category": spec.get("category", ""),
                "availability": spec.get("availability", ""),
                "window": spec.get("window"),
                "lookback": spec.get("lookback"),
            }
        )
        for label in labels:
            factor_daily, factor_groups = _evaluate_factor_label(
                data,
                block.name,
                factor_name,
                spec,
                label,
                args.min_pairs,
                args.groups,
            )
            daily_records.extend(factor_daily)
            group_records.extend(factor_groups)
    return daily_records, group_records, factor_meta


def _evaluate_factor_label(
    data: pd.DataFrame,
    block_name: str,
    factor_name: str,
    spec: dict[str, Any],
    label: str,
    min_pairs: int,
    group_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    daily_records: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    for target_date, group in data.groupby("target_trade_date", sort=True):
        x = pd.to_numeric(group[factor_name], errors="coerce")
        y = pd.to_numeric(group[label], errors="coerce")
        finite = np.isfinite(x.to_numpy(dtype="float64", na_value=np.nan)) & np.isfinite(y.to_numpy(dtype="float64", na_value=np.nan))
        valid = pd.DataFrame({"factor": x[finite], "label": y[finite]})
        pair_count = int(len(valid))
        row_count = int(len(group))
        ic = np.nan
        rank_ic = np.nan
        spread = np.nan
        if pair_count >= min_pairs and valid["factor"].nunique(dropna=True) > 1 and valid["label"].nunique(dropna=True) > 1:
            ic = _corr(valid["factor"], valid["label"])
            rank_ic = _rank_corr(valid["factor"], valid["label"])
            rows, spread = _group_returns(valid, group_count)
            for row in rows:
                group_records.append(
                    {
                        "block": block_name,
                        "factor_name": factor_name,
                        "label": label,
                        "target_trade_date": target_date,
                        **row,
                    }
                )
        daily_records.append(
            {
                "block": block_name,
                "factor_name": factor_name,
                "source": spec.get("source", ""),
                "category": spec.get("category", ""),
                "label": label,
                "target_trade_date": target_date,
                "row_count": row_count,
                "pair_count": pair_count,
                "coverage": _safe_rate(pair_count, row_count),
                "ic": ic,
                "rank_ic": rank_ic,
                "group_spread": spread,
            }
        )
    return daily_records, group_records


def _group_returns(valid: pd.DataFrame, group_count: int) -> tuple[list[dict[str, Any]], float]:
    if len(valid) < group_count or valid["factor"].nunique(dropna=True) <= 1:
        return [], np.nan
    ranks = valid["factor"].rank(method="first")
    groups = pd.qcut(ranks, q=group_count, labels=list(range(1, group_count + 1)))
    work = valid.assign(group=groups.astype("int64"))
    grouped = work.groupby("group", sort=True)["label"].agg(["count", "mean"]).reset_index()
    rows = [
        {
            "group": int(row["group"]),
            "count": int(row["count"]),
            "mean_return": float(row["mean"]),
        }
        for _, row in grouped.iterrows()
    ]
    means = dict(zip(grouped["group"], grouped["mean"], strict=True))
    spread = float(means.get(group_count, np.nan) - means.get(1, np.nan))
    return rows, spread


def _build_summary(daily: pd.DataFrame, factor_meta: list[dict[str, Any]], labels: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if daily.empty:
        return pd.DataFrame()
    for meta in factor_meta:
        for label in labels:
            subset = daily.loc[(daily["block"].eq(meta["block"])) & (daily["factor_name"].eq(meta["factor_name"])) & (daily["label"].eq(label))]
            ic = subset["ic"].dropna()
            rank_ic = subset["rank_ic"].dropna()
            spread = subset["group_spread"].dropna()
            rows.append(
                {
                    **meta,
                    "label": label,
                    "target_day_count": int(len(subset)),
                    "ic_day_count": int(len(ic)),
                    "rank_ic_day_count": int(len(rank_ic)),
                    "pair_count_mean": _series_mean(subset["pair_count"]),
                    "coverage_mean": _series_mean(subset["coverage"]),
                    "ic_mean": _series_mean(ic),
                    "ic_std": _series_std(ic),
                    "ic_ir": _ir(ic),
                    "ic_positive_rate": _positive_rate(ic),
                    "rank_ic_mean": _series_mean(rank_ic),
                    "rank_ic_std": _series_std(rank_ic),
                    "rank_ic_ir": _ir(rank_ic),
                    "rank_ic_positive_rate": _positive_rate(rank_ic),
                    "group_spread_mean": _series_mean(spread),
                    "group_spread_positive_rate": _positive_rate(spread),
                }
            )
    return pd.DataFrame(rows)


def _factor_ic_frame(daily: pd.DataFrame) -> pd.DataFrame:
    columns = ["block", "factor_name", "source", "category", "label", "target_trade_date", "row_count", "pair_count", "coverage", "ic"]
    return _select_columns(daily, columns)


def _factor_rankic_frame(daily: pd.DataFrame) -> pd.DataFrame:
    columns = ["block", "factor_name", "source", "category", "label", "target_trade_date", "row_count", "pair_count", "coverage", "rank_ic"]
    return _select_columns(daily, columns)


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


def _parse_labels(value: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in value.split(",") if item.strip())
    if not labels:
        raise ValueError("--labels must include at least one label column")
    return labels


def _parse_optional_date(value: str | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --{name} date: {value}")
    return parsed.normalize()


def _validate_args(args: argparse.Namespace, since: pd.Timestamp | None, until: pd.Timestamp | None) -> None:
    if args.min_pairs <= 1:
        raise ValueError("--min-pairs must be greater than 1")
    if args.groups <= 1:
        raise ValueError("--groups must be greater than 1")
    if int(getattr(args, "workers", 1) or 1) <= 0:
        raise ValueError("--workers must be positive")
    if since is not None and until is not None and since > until:
        raise ValueError("--since cannot be later than --until")


def _corr(left: pd.Series, right: pd.Series) -> float:
    value = left.corr(right, method="pearson")
    return float(value) if pd.notna(value) else np.nan


def _rank_corr(left: pd.Series, right: pd.Series) -> float:
    value = left.rank(method="average").corr(right.rank(method="average"), method="pearson")
    return float(value) if pd.notna(value) else np.nan


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _series_mean(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.mean())


def _series_std(series: pd.Series) -> float:
    if len(series) < 2:
        return np.nan
    return float(series.std(ddof=1))


def _ir(series: pd.Series) -> float:
    std = _series_std(series)
    if pd.isna(std) or std == 0:
        return np.nan
    return _series_mean(series) / std


def _positive_rate(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.gt(0).mean())


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.loc[:, columns].copy()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _concat_csv_parts(paths: list[Path], output_path: Path, columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with output_path.open("w", encoding="utf-8", newline="") as output:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as source:
                header = source.readline()
                if not header:
                    continue
                if not wrote_header:
                    output.write(header)
                    wrote_header = True
                shutil.copyfileobj(source, output)
        if not wrote_header:
            output.write(",".join(columns) + "\n")


def _group_return_columns() -> list[str]:
    return ["block", "factor_name", "label", "target_trade_date", "group", "count", "mean_return"]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
