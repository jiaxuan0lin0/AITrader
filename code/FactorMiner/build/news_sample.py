from __future__ import annotations

import argparse
import gc
from dataclasses import replace
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq

from FactorMiner.core.factor_block import FactorBlock, write_factor_block
from FactorMiner.core.registry import remove_blocks, upsert_block, validate_registry
from FactorMiner.pools.news_llm import prepare_news_items
from FactorMiner.pools.news_sample import NewsSampleConfig, build_news_sample_factors
from aitrader_paths import DATASETS_ROOT


DEFAULT_SAMPLES_PATH = DATASETS_ROOT / "processed" / "samples.parquet"
DEFAULT_NEWS_PATH = DATASETS_ROOT / "processed" / "news.parquet"
DEFAULT_SCORES_PATH = DATASETS_ROOT / "factors" / "news_llm_scores.parquet"
DEFAULT_FEATURE_ROOT = DATASETS_ROOT / "features"
DEFAULT_FEATURE_REGISTRY_PATH = DEFAULT_FEATURE_ROOT / "feature_registry.json"
DEFAULT_BLOCK_NAME = "news_llm_sample"
DEFAULT_MARKET_BLOCK_NAME = "news_llm_market_sample"
DEFAULT_STOCK_BLOCK_NAME = "news_llm_stock_sample"
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build sample-level LLM news factors.")
    parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--news-path", type=Path, default=DEFAULT_NEWS_PATH)
    parser.add_argument("--scores-path", type=Path, default=DEFAULT_SCORES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--feature-registry-path", type=Path, default=DEFAULT_FEATURE_REGISTRY_PATH)
    parser.add_argument("--block-name", default=DEFAULT_BLOCK_NAME)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--windows", default="1,3,5,10", help="Natural-day windows, e.g. 1,3,5,10")
    parser.add_argument(
        "--scope",
        choices=("split", "all", "market", "stock"),
        default="split",
        help="Build split market/stock blocks by default, or one legacy all block.",
    )
    parser.add_argument("--since", default=None, help="Optional inclusive target_trade_date lower bound.")
    parser.add_argument("--until", default=None, help="Optional inclusive target_trade_date upper bound.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-registry", action="store_true", help="Write parquet/manifest without updating feature_registry.json.")
    parser.add_argument(
        "--full-registry-validate",
        action="store_true",
        help="Use full parquet reads during registry validation. Default uses parquet metadata to avoid OOM.",
    )
    parser.add_argument("--validate-only", action="store_true", help="Only validate the feature registry.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        validate_registry(args.feature_registry_path, metadata_only=not args.full_registry_validate)
        print(f"feature_registry_valid={args.feature_registry_path}")
        return 0

    blocks = build_news_sample_blocks(args)
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
    if not args.skip_registry:
        print(f"feature_registry={args.feature_registry_path}")
    return 0


def build_news_sample_block(args: argparse.Namespace) -> FactorBlock:
    blocks = build_news_sample_blocks(args)
    if len(blocks) != 1:
        raise ValueError("build_news_sample_block requires --scope all, market, or stock. Use build_news_sample_blocks for split output.")
    return blocks[0]


def build_news_sample_blocks(args: argparse.Namespace) -> list[FactorBlock]:
    windows = tuple(int(value) for value in args.windows.split(",") if value.strip())
    if not windows:
        raise ValueError("--windows must include at least one positive integer")
    if any(window <= 0 for window in windows):
        raise ValueError("--windows values must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")

    scopes = _resolve_scopes(args.scope)
    if args.scope == "split" and (args.output_path is not None or args.manifest_path is not None):
        raise ValueError("--output-path/--manifest-path are ambiguous with --scope split. Use --output-root or a single --scope.")
    output_paths = [_resolve_output_paths(args, _block_name_for_scope(args.block_name, scope)) for scope in scopes]
    for output_path, manifest_path in output_paths:
        _guard_limited_outputs(args, output_path, manifest_path)
    config = NewsSampleConfig(windows=windows)

    LOGGER.info("news_sample_load_samples_start path=%s", args.samples_path)
    samples = _load_samples(args.samples_path, args.since, args.until, args.limit)
    LOGGER.info("news_sample_load_samples_done rows=%s", len(samples))
    LOGGER.info("news_sample_load_news_start path=%s", args.news_path)
    news = _load_news(args.news_path)
    LOGGER.info("news_sample_load_news_done rows=%s columns=%s", len(news), len(news.columns))
    LOGGER.info("news_sample_load_scores_start path=%s", args.scores_path)
    scores = _load_scores(args.scores_path)
    LOGGER.info("news_sample_load_scores_done rows=%s columns=%s", len(scores), len(scores.columns))
    LOGGER.info("news_sample_prepare_news_start")
    news_prepared = prepare_news_items(news)
    LOGGER.info(
        "news_sample_prepare_news_done news_items=%s stock_map=%s",
        len(news_prepared.news_items),
        len(news_prepared.news_stock_map),
    )

    registry_blocks: list[FactorBlock] = []
    if not args.skip_registry:
        _drop_superseded_news_blocks(args, scopes)
    for scope, (output_path, manifest_path) in zip(scopes, output_paths):
        block_name = _block_name_for_scope(args.block_name, scope)
        LOGGER.info(
            "news_sample_factor_build_start block=%s scope=%s windows=%s",
            block_name,
            scope,
            ",".join(str(item) for item in windows),
        )
        result = build_news_sample_factors(
            samples,
            news_prepared.news_items,
            news_prepared.news_stock_map,
            scores,
            config,
            scope=scope,
        )
        LOGGER.info("news_sample_factor_build_done block=%s rows=%s factors=%s", block_name, len(result.factors), len(result.specs))

        LOGGER.info("news_sample_write_start block=%s", block_name)
        block = write_factor_block(
            result,
            block_name,
            "sample",
            output_path,
            manifest_path,
            description=_description_for_scope(scope),
        )
        registry_block = replace(
            block,
            factor_path=str(_relative_to(output_path, args.feature_registry_path.parent)),
            manifest_path=str(_relative_to(manifest_path, args.feature_registry_path.parent)),
        )
        registry_blocks.append(registry_block)
        if not args.skip_registry:
            upsert_block(args.feature_registry_path, registry_block)
        LOGGER.info(
            "news_sample_write_done block=%s rows=%s factors=%s",
            registry_block.name,
            registry_block.row_count,
            registry_block.factor_count,
        )
        del result
        gc.collect()

    if not args.skip_registry:
        validate_registry(args.feature_registry_path, metadata_only=not args.full_registry_validate)
    return registry_blocks


def _load_samples(path: Path, since: str | None, until: str | None, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing samples parquet: {path}")
    columns = ["sample_id", "stock_code", "decision_ts"]
    available = set(pq.ParquetFile(path).schema.names)
    if since or until:
        if "target_trade_date" not in available:
            raise KeyError("samples must contain target_trade_date when --since or --until is used")
        columns.append("target_trade_date")
    samples = pd.read_parquet(path, columns=columns)
    if since or until:
        if "target_trade_date" not in samples.columns:
            raise KeyError("samples must contain target_trade_date when --since or --until is used")
        target_dates = pd.to_datetime(samples["target_trade_date"], errors="coerce").dt.normalize()
    if since:
        samples = samples.loc[target_dates.ge(_parse_date(since, "since"))].copy()
        target_dates = target_dates.loc[samples.index]
    if until:
        samples = samples.loc[target_dates.le(_parse_date(until, "until"))].copy()
    if limit is not None:
        samples = samples.head(limit).copy()
    return samples.reset_index(drop=True)


def _load_news(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing news parquet: {path}")
    available = set(pq.ParquetFile(path).schema.names)
    columns = [
        column
        for column in (
            "stock_code",
            "matched_stock_codes",
            "matched_stock_count",
            "trade_date",
            "publish_time",
            "news_text",
            "__source_file",
        )
        if column in available
    ]
    return pd.read_parquet(path, columns=columns)


def _load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing news score parquet: {path}")
    columns = [
        "news_text_hash",
        "sentiment_score",
        "impact_score",
        "risk_score",
        "relevance_score",
        "novelty_score",
        "event_type",
    ]
    return pd.read_parquet(path, columns=columns)


def _resolve_output_paths(args: argparse.Namespace, block_name: str) -> tuple[Path, Path]:
    output_path = args.output_path or args.output_root / "blocks" / "sample" / f"{block_name}.parquet"
    if args.manifest_path is not None:
        manifest_path = args.manifest_path
    elif args.output_path is not None:
        manifest_path = args.output_path.with_suffix(".json")
    else:
        manifest_path = args.output_root / "manifests" / f"{block_name}.json"
    return output_path, manifest_path


def _resolve_scopes(scope: str) -> tuple[str, ...]:
    if scope == "split":
        return ("market", "stock")
    if scope in {"all", "market", "stock"}:
        return (scope,)
    raise ValueError("scope must be one of: split, all, market, stock")


def _block_name_for_scope(base_name: str, scope: str) -> str:
    if scope == "all":
        return base_name
    if base_name == DEFAULT_BLOCK_NAME:
        return DEFAULT_MARKET_BLOCK_NAME if scope == "market" else DEFAULT_STOCK_BLOCK_NAME
    return f"{base_name}_{scope}"


def _description_for_scope(scope: str) -> str:
    if scope == "market":
        return "Sample-level market news factors aggregated from cached LLM news scores."
    if scope == "stock":
        return "Sample-level stock news factors aggregated from cached LLM news scores."
    return "Sample-level news factors aggregated from cached LLM news scores."


def _drop_superseded_news_blocks(args: argparse.Namespace, scopes: tuple[str, ...]) -> None:
    if scopes == ("market", "stock"):
        remove_blocks(args.feature_registry_path, (args.block_name,))
    elif scopes == ("all",):
        remove_blocks(args.feature_registry_path, (DEFAULT_MARKET_BLOCK_NAME, DEFAULT_STOCK_BLOCK_NAME))


def _guard_limited_outputs(args: argparse.Namespace, output_path: Path, manifest_path: Path) -> None:
    if args.limit is None:
        return
    writes_default_files = _is_relative_to(output_path, DEFAULT_FEATURE_ROOT) or _is_relative_to(manifest_path, DEFAULT_FEATURE_ROOT)
    writes_default_registry = not args.skip_registry and _same_path(args.feature_registry_path, DEFAULT_FEATURE_REGISTRY_PATH)
    if writes_default_files or writes_default_registry:
        raise ValueError(
            "--limit requires non-default --output-root/--output-path "
            "and --feature-registry-path, or use --skip-registry"
        )


def _parse_date(value: str, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --{name} date: {value}")
    return parsed.normalize()


def _relative_to(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(base.expanduser().resolve())
        return True
    except ValueError:
        return False


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


if __name__ == "__main__":
    raise SystemExit(main())
