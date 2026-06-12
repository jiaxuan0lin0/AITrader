from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from dataclasses import replace
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from FactorMiner.core.factor_block import FactorBlock, write_factor_block
from FactorMiner.core.factor_spec import FactorResult
from FactorMiner.pools.alpha158 import Alpha158Config, build_alpha158_factors
from FactorMiner.pools.metric import MetricConfig, build_metric_factors
from FactorMiner.pools.moneyflow import MoneyflowConfig, build_moneyflow_factors
from FactorMiner.core.registry import FactorRegistry, upsert_block, validate_registry
from aitrader_paths import DATASETS_ROOT


LOGGER = logging.getLogger(__name__)
_ALPHA_WORKER_PRICE: pd.DataFrame | None = None
_ALPHA_WORKER_OUTPUT_ROOT: Path | None = None
_ALPHA_WORKER_DISABLE_NEUTRAL = False
_ALPHA_WORKER_SINCE: pd.Timestamp | None = None
_ALPHA_WORKER_UNTIL: pd.Timestamp | None = None

DEFAULT_PROCESSED_DIR = DATASETS_ROOT / "processed"
DEFAULT_FACTOR_ROOT = DATASETS_ROOT / "factors"
DEFAULT_PRICE_PATH = DEFAULT_PROCESSED_DIR / "price.parquet"
DEFAULT_METRIC_PATH = DEFAULT_PROCESSED_DIR / "metric.parquet"
DEFAULT_MONEYFLOW_PATH = DEFAULT_PROCESSED_DIR / "moneyflow.parquet"
DEFAULT_BASIC_PATH = DEFAULT_PROCESSED_DIR / "basic.parquet"
DEFAULT_REGISTRY_PATH = DEFAULT_FACTOR_ROOT / "factor_registry.json"

DAILY_BLOCKS = ("alpha158", "metric", "moneyflow")
BLOCK_NAMES = {
    "alpha158": "manual_alpha158",
    "alpha158_kbar": "manual_alpha158_kbar",
    "alpha158_price": "manual_alpha158_price",
    "alpha158_return": "manual_alpha158_return",
    "alpha158_rolling3": "manual_alpha158_rolling3",
    "alpha158_rolling5": "manual_alpha158_rolling5",
    "alpha158_rolling10": "manual_alpha158_rolling10",
    "alpha158_rolling20": "manual_alpha158_rolling20",
    "alpha158_rolling60": "manual_alpha158_rolling60",
    "metric": "manual_metric",
    "moneyflow": "manual_moneyflow",
}
BLOCK_DESCRIPTIONS = {
    "alpha158": "Manual Alpha158-style daily price-volume factors.",
    "alpha158_kbar": "Manual Alpha158-style candlestick shape factors.",
    "alpha158_price": "Manual Alpha158-style relative price factors.",
    "alpha158_return": "Manual Alpha158-style return lookback factors.",
    "alpha158_rolling3": "Manual Alpha158-style rolling factors with 3-day windows.",
    "alpha158_rolling5": "Manual Alpha158-style rolling factors with 5-day windows.",
    "alpha158_rolling10": "Manual Alpha158-style rolling factors with 10-day windows.",
    "alpha158_rolling20": "Manual Alpha158-style rolling factors with 20-day windows.",
    "alpha158_rolling60": "Manual Alpha158-style rolling factors with 60-day windows.",
    "metric": "Manual daily valuation, market-cap, and turnover factors.",
    "moneyflow": "Manual daily moneyflow ratio, momentum, and confirmation factors.",
}
ALPHA158_SPLIT_COMPONENTS = (
    "alpha158_kbar",
    "alpha158_price",
    "alpha158_return",
    "alpha158_rolling3",
    "alpha158_rolling5",
    "alpha158_rolling10",
    "alpha158_rolling20",
    "alpha158_rolling60",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build daily FactorMiner factor blocks.")
    parser.add_argument("--price-path", type=Path, default=DEFAULT_PRICE_PATH)
    parser.add_argument("--metric-path", type=Path, default=DEFAULT_METRIC_PATH)
    parser.add_argument("--moneyflow-path", type=Path, default=DEFAULT_MONEYFLOW_PATH)
    parser.add_argument("--basic-path", type=Path, default=DEFAULT_BASIC_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FACTOR_ROOT)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--block", choices=(*DAILY_BLOCKS, "all"), default="all")
    parser.add_argument("--since", default=None, help="Inclusive output trade_date lower bound.")
    parser.add_argument("--until", default=None, help="Inclusive input/output trade_date upper bound.")
    parser.add_argument("--stock-limit", type=int, default=None, help="Optional stock count limit.")
    parser.add_argument("--disable-neutral", action="store_true", help="Do not append cross-sectional or industry-neutral factors.")
    parser.add_argument("--alpha-workers", type=int, default=6, help="Parallel workers for split alpha158 component blocks.")
    parser.add_argument(
        "--alpha-layout",
        choices=("split", "single"),
        default="split",
        help="Materialize alpha158 as restartable sub-blocks or one legacy block.",
    )
    parser.add_argument(
        "--max-industry-missing-rate",
        type=float,
        default=0.20,
        help="Maximum allowed missing industry rate when neutral factors are enabled.",
    )
    parser.add_argument("--validate-only", action="store_true", help="Only validate the existing registry.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        validate_registry(args.registry_path)
        print(f"registry_valid={args.registry_path}")
        return 0

    blocks = build_daily_blocks(args)
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
    print(f"registry={args.registry_path}")
    return 0


def build_daily_blocks(args: argparse.Namespace) -> list[FactorBlock]:
    selected_blocks = _selected_blocks(args.block)
    since = _parse_optional_date(args.since, "since")
    until = _parse_optional_date(args.until, "until")
    if since is not None and until is not None and since > until:
        raise ValueError("--since cannot be later than --until")
    if args.stock_limit is not None and args.stock_limit <= 0:
        raise ValueError("--stock-limit must be positive when provided")
    alpha_workers = int(getattr(args, "alpha_workers", 6) or 6)
    if alpha_workers <= 0:
        raise ValueError("--alpha-workers must be positive")
    if args.stock_limit is not None and _same_path(args.output_root, DEFAULT_FACTOR_ROOT):
        raise ValueError("--stock-limit requires a non-default output root; pass a non-default --output-root")
    if not 0 <= args.max_industry_missing_rate <= 1:
        raise ValueError("--max-industry-missing-rate must be between 0 and 1")
    alpha_layout = getattr(args, "alpha_layout", "split")
    if alpha_layout not in {"split", "single"}:
        raise ValueError("--alpha-layout must be split or single")

    frames = _load_required_inputs(selected_blocks, args, until)
    basic = _load_basic(args.basic_path)
    if basic is not None:
        frames = {name: _fill_industry_from_basic(frame, basic) for name, frame in frames.items()}

    if args.stock_limit is not None:
        frames = _apply_stock_limit(frames, args.stock_limit)

    if not args.disable_neutral:
        _validate_industry_inputs(selected_blocks, frames, args.max_industry_missing_rate)

    written_blocks: list[FactorBlock] = []
    for block_key in selected_blocks:
        if block_key == "alpha158":
            _remove_registered_blocks(args.registry_path, _alpha158_registered_block_names())
        if block_key == "alpha158" and alpha_layout == "split" and alpha_workers > 1:
            blocks = _build_alpha158_split_parallel(frames, args, since, until, alpha_workers)
            written_blocks.extend(blocks)
            continue
        for materialized_key, result in _build_results(block_key, frames, args.disable_neutral, alpha_layout):
            LOGGER.info("daily_block_materialize_start block=%s", materialized_key)
            block = _materialize_daily_result(materialized_key, result, args.output_root, since, until)
            upsert_block(args.registry_path, block)
            written_blocks.append(block)
            del result
            gc.collect()
            LOGGER.info(
                "daily_block_materialize_done block=%s rows=%s factors=%s",
                block.name,
                block.row_count,
                block.factor_count,
            )

    validate_registry(args.registry_path)
    return written_blocks


def _build_alpha158_split_parallel(
    frames: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
    workers: int,
) -> list[FactorBlock]:
    try:
        context = mp.get_context("fork")
    except ValueError:
        LOGGER.warning("daily_alpha_parallel_unavailable reason=no_fork_context")
        return _build_alpha158_split_serial(frames, args, since, until)

    max_workers = min(workers, len(ALPHA158_SPLIT_COMPONENTS))
    LOGGER.info("daily_alpha_parallel_start workers=%s components=%s", max_workers, len(ALPHA158_SPLIT_COMPONENTS))
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_init_alpha_worker,
        initargs=(frames["price"], args.output_root, args.disable_neutral, since, until),
    ) as executor:
        futures = {
            executor.submit(_build_alpha158_component_part, order, component): component
            for order, component in enumerate(ALPHA158_SPLIT_COMPONENTS)
        }
        for future in as_completed(futures):
            component = futures[future]
            record = future.result()
            records.append(record)
            block_record = record["block"]
            factor_count = block_record["factor_count"] if isinstance(block_record, dict) else "unknown"
            LOGGER.info("daily_alpha_component_done block=%s factors=%s", component, factor_count)

    blocks: list[FactorBlock] = []
    for record in sorted(records, key=lambda item: int(item["order"])):
        block = FactorBlock.from_record(record["block"])  # type: ignore[arg-type]
        upsert_block(args.registry_path, block)
        blocks.append(block)
    LOGGER.info("daily_alpha_parallel_done workers=%s blocks=%s", max_workers, len(blocks))
    return blocks


def _build_alpha158_split_serial(
    frames: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
) -> list[FactorBlock]:
    blocks: list[FactorBlock] = []
    for component in ALPHA158_SPLIT_COMPONENTS:
        LOGGER.info("daily_block_compute_start block=%s", component)
        result = build_alpha158_factors(frames["price"], _alpha158_component_config(component, args.disable_neutral))
        block = _materialize_daily_result(component, result, args.output_root, since, until)
        upsert_block(args.registry_path, block)
        blocks.append(block)
        del result
        gc.collect()
    return blocks


def _init_alpha_worker(
    price: pd.DataFrame,
    output_root: Path,
    disable_neutral: bool,
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
) -> None:
    global _ALPHA_WORKER_PRICE, _ALPHA_WORKER_OUTPUT_ROOT, _ALPHA_WORKER_DISABLE_NEUTRAL
    global _ALPHA_WORKER_SINCE, _ALPHA_WORKER_UNTIL
    _ALPHA_WORKER_PRICE = price
    _ALPHA_WORKER_OUTPUT_ROOT = output_root
    _ALPHA_WORKER_DISABLE_NEUTRAL = disable_neutral
    _ALPHA_WORKER_SINCE = since
    _ALPHA_WORKER_UNTIL = until


def _build_alpha158_component_part(order: int, component: str) -> dict[str, object]:
    if _ALPHA_WORKER_PRICE is None or _ALPHA_WORKER_OUTPUT_ROOT is None:
        raise RuntimeError("alpha worker state is not initialized")
    LOGGER.info("daily_block_compute_start block=%s", component)
    result = build_alpha158_factors(
        _ALPHA_WORKER_PRICE,
        _alpha158_component_config(component, _ALPHA_WORKER_DISABLE_NEUTRAL),
    )
    block = _materialize_daily_result(
        component,
        result,
        _ALPHA_WORKER_OUTPUT_ROOT,
        _ALPHA_WORKER_SINCE,
        _ALPHA_WORKER_UNTIL,
    )
    return {"order": order, "block": block.to_record()}


def _load_required_inputs(
    selected_blocks: tuple[str, ...],
    args: argparse.Namespace,
    until: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if "alpha158" in selected_blocks or "moneyflow" in selected_blocks:
        frames["price"] = _load_daily_frame(args.price_path, "price", until=until)
    if "metric" in selected_blocks:
        frames["metric"] = _load_daily_frame(args.metric_path, "metric", until=until)
    if "moneyflow" in selected_blocks:
        frames["moneyflow"] = _load_daily_frame(args.moneyflow_path, "moneyflow", until=until)
    return frames


def _load_daily_frame(path: Path, name: str, until: pd.Timestamp | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} input parquet: {path}")
    frame = pd.read_parquet(path)
    _require_columns(frame, ("stock_code", "trade_date"), name)
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["stock_code", "trade_date"]).reset_index(drop=True)
    if until is not None:
        frame = frame.loc[frame["trade_date"].le(until)].reset_index(drop=True)
    duplicated = frame.duplicated(["stock_code", "trade_date"])
    if duplicated.any():
        examples = frame.loc[duplicated, ["stock_code", "trade_date"]].head(5).to_dict("records")
        raise ValueError(f"{name} keys must be unique. Duplicate examples: {examples}")
    return frame.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def _load_basic(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    basic = pd.read_parquet(path)
    if "stock_code" not in basic.columns or "industry" not in basic.columns:
        return None
    result = basic[["stock_code", "industry"]].copy()
    result["stock_code"] = result["stock_code"].astype(str)
    return result.drop_duplicates("stock_code", keep="last")


def _fill_industry_from_basic(frame: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "stock_code" not in frame.columns:
        return frame
    if "industry" in frame.columns and not _industry_missing_mask(frame["industry"]).any():
        return frame
    work = frame.merge(basic.rename(columns={"industry": "industry_basic"}), on="stock_code", how="left")
    if "industry" in work.columns:
        work["industry"] = work["industry"].where(~_industry_missing_mask(work["industry"]), work["industry_basic"])
    else:
        work["industry"] = work["industry_basic"]
    return work.drop(columns=["industry_basic"])


def _apply_stock_limit(frames: dict[str, pd.DataFrame], stock_limit: int) -> dict[str, pd.DataFrame]:
    stock_sets = [set(frame["stock_code"].dropna().astype(str)) for frame in frames.values() if "stock_code" in frame.columns]
    if not stock_sets:
        return frames
    common_stocks = set.intersection(*stock_sets) if len(stock_sets) > 1 else stock_sets[0]
    selected = set(sorted(common_stocks)[:stock_limit])
    return {
        name: frame.loc[frame["stock_code"].astype(str).isin(selected)].reset_index(drop=True)
        for name, frame in frames.items()
    }


def _validate_industry_inputs(
    selected_blocks: tuple[str, ...],
    frames: dict[str, pd.DataFrame],
    max_missing_rate: float,
) -> None:
    if "alpha158" in selected_blocks:
        _validate_industry_frame(frames["price"], "price", max_missing_rate)
    if "metric" in selected_blocks:
        _validate_industry_frame(frames["metric"], "metric", max_missing_rate)
    if "moneyflow" in selected_blocks:
        if "industry" in frames["moneyflow"].columns:
            _validate_industry_frame(frames["moneyflow"], "moneyflow", max_missing_rate)
        else:
            _validate_industry_frame(frames["price"], "price", max_missing_rate)


def _validate_industry_frame(frame: pd.DataFrame, name: str, max_missing_rate: float) -> None:
    if "industry" not in frame.columns:
        raise KeyError(f"Missing industry column for neutral daily factors: {name}.industry")
    missing = _industry_missing_mask(frame["industry"])
    missing_rate = float(missing.mean()) if len(missing) else 1.0
    if missing_rate > max_missing_rate:
        raise ValueError(
            f"{name}.industry missing rate {missing_rate:.4f} exceeds max {max_missing_rate:.4f}"
        )
    coverage = frame.loc[~missing].groupby("trade_date")["industry"].size()
    if coverage.empty:
        raise ValueError(f"{name}.industry has no non-missing values")


def _industry_missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().fillna("").eq("")


def _build_results(
    block_key: str,
    frames: dict[str, pd.DataFrame],
    disable_neutral: bool,
    alpha_layout: str,
) -> Iterator[tuple[str, FactorResult]]:
    if block_key == "alpha158":
        if alpha_layout == "single":
            config = _alpha158_config(disable_neutral)
            LOGGER.info("daily_block_compute_start block=alpha158")
            yield "alpha158", build_alpha158_factors(frames["price"], config)
            return
        for component in ALPHA158_SPLIT_COMPONENTS:
            LOGGER.info("daily_block_compute_start block=%s", component)
            yield component, build_alpha158_factors(frames["price"], _alpha158_component_config(component, disable_neutral))
        return
    if block_key == "metric":
        config = MetricConfig(neutral=None) if disable_neutral else MetricConfig()
        LOGGER.info("daily_block_compute_start block=metric")
        yield "metric", build_metric_factors(frames["metric"], config=config)
        return
    if block_key == "moneyflow":
        config = MoneyflowConfig(neutral=None) if disable_neutral else MoneyflowConfig()
        LOGGER.info("daily_block_compute_start block=moneyflow")
        yield "moneyflow", build_moneyflow_factors(frames["moneyflow"], frames["price"], config)
        return
    raise ValueError(f"Unsupported daily block: {block_key}")


def _alpha158_config(disable_neutral: bool, **kwargs) -> Alpha158Config:
    if disable_neutral:
        kwargs["neutral"] = None
    return Alpha158Config(**kwargs)


def _alpha158_component_config(component: str, disable_neutral: bool) -> Alpha158Config:
    if component == "alpha158_kbar":
        return _alpha158_config(
            disable_neutral,
            include_price=False,
            include_return=False,
            include_rolling=False,
            return_windows=(),
            rolling_windows=(),
        )
    if component == "alpha158_price":
        return _alpha158_config(
            disable_neutral,
            include_kbar=False,
            include_return=False,
            include_rolling=False,
            return_windows=(),
            rolling_windows=(),
        )
    if component == "alpha158_return":
        return _alpha158_config(
            disable_neutral,
            include_kbar=False,
            include_price=False,
            include_rolling=False,
            rolling_windows=(),
        )
    if component.startswith("alpha158_rolling"):
        window = int(component.removeprefix("alpha158_rolling"))
        return _alpha158_config(
            disable_neutral,
            include_kbar=False,
            include_price=False,
            include_return=False,
            rolling_windows=(window,),
            return_windows=(),
        )
    raise ValueError(f"Unsupported alpha158 component: {component}")


def _filter_result_dates(
    result: FactorResult,
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
) -> FactorResult:
    if since is None and until is None:
        return result
    factors = result.factors.copy()
    dates = pd.to_datetime(factors["trade_date"], errors="coerce").dt.normalize()
    mask = pd.Series(True, index=factors.index)
    if since is not None:
        mask &= dates.ge(since)
    if until is not None:
        mask &= dates.le(until)
    filtered = factors.loc[mask].reset_index(drop=True)
    return FactorResult(factors=filtered, specs=result.specs, key_columns=result.key_columns)


def _materialize_daily_result(
    block_key: str,
    result: FactorResult,
    output_root: Path,
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
) -> FactorBlock:
    result = _filter_result_dates(result, since, until)
    result.validate()
    return _write_daily_block(block_key, result, output_root)


def _write_daily_block(block_key: str, result: FactorResult, output_root: Path) -> FactorBlock:
    block_name = BLOCK_NAMES[block_key]
    factor_path = output_root / "blocks" / "daily" / f"{block_name}.parquet"
    manifest_path = output_root / "manifests" / f"{block_name}.json"
    block = write_factor_block(
        result,
        block_name,
        "daily",
        factor_path,
        manifest_path,
        description=BLOCK_DESCRIPTIONS[block_key],
    )
    return replace(
        block,
        factor_path=str(_relative_to(factor_path, output_root)),
        manifest_path=str(_relative_to(manifest_path, output_root)),
    )


def _remove_registered_blocks(registry_path: Path, block_names: set[str]) -> None:
    registry = FactorRegistry.load(registry_path)
    remaining = [block for block in registry.blocks if block.name not in block_names]
    if len(remaining) == len(registry.blocks):
        return
    registry.blocks = remaining
    registry.save(registry_path)


def _alpha158_registered_block_names() -> set[str]:
    return {BLOCK_NAMES["alpha158"], *(BLOCK_NAMES[component] for component in ALPHA158_SPLIT_COMPONENTS)}


def _selected_blocks(block: str) -> tuple[str, ...]:
    if block == "all":
        return DAILY_BLOCKS
    if block not in DAILY_BLOCKS:
        raise ValueError(f"Unsupported daily block: {block}")
    return (block,)


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


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
