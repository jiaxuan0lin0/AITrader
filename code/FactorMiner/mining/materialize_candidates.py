from __future__ import annotations

import argparse
import ast
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from FactorMiner.core.factor_block import write_factor_block
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.core.registry import upsert_block, validate_registry
from model.msgca.feature_set import load_feature_blocks
from aitrader_paths import DATASETS_ROOT


DEFAULT_DATASETS_ROOT = DATASETS_ROOT
DEFAULT_PROCESSED_DIR = DEFAULT_DATASETS_ROOT / "processed"
DEFAULT_FEATURE_ROOT = DEFAULT_DATASETS_ROOT / "features"
DEFAULT_FEATURE_REGISTRY_PATH = DEFAULT_FEATURE_ROOT / "feature_registry.json"
DEFAULT_GPT_MINING_ROOT = DEFAULT_DATASETS_ROOT / "factors" / "gpt_mining"
DEFAULT_OUTPUT_ROOT = DEFAULT_GPT_MINING_ROOT / "experiment"
LOGGER = logging.getLogger(__name__)
RAW_TABLE_PRIORITY = {
    "open": "price",
    "high": "price",
    "low": "price",
    "close": "price",
    "preclose": "price",
    "change": "price",
    "pct_chg": "price",
    "volume": "price",
    "amount": "price",
    "vwap": "price",
    "turnover_rate": "metric",
    "turnover_rate_f": "metric",
    "volume_ratio": "metric",
    "pe": "metric",
    "pe_ttm": "metric",
    "pb": "metric",
    "ps": "metric",
    "ps_ttm": "metric",
    "dv_ratio": "metric",
    "dv_ttm": "metric",
    "total_mv": "metric",
    "circ_mv": "metric",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize validated GPT candidate formulas into sample feature blocks.")
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--validated-path", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument("--price-path", type=Path, default=None)
    parser.add_argument("--metric-path", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--feature-registry-path", type=Path, default=DEFAULT_FEATURE_REGISTRY_PATH)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--categories", default="all", help="Comma-separated candidate categories, or all.")
    parser.add_argument("--block-prefix", default="gpt_mined")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit from samples.")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-registry-validate", action="store_true")
    parser.add_argument("--full-registry-validate", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_paths(args)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = materialize_candidates(args)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


def _resolve_paths(args: argparse.Namespace) -> None:
    args.validated_path = args.validated_path or args.round_dir / "validated" / "candidates_validated.json"
    args.samples_path = args.samples_path or args.processed_dir / "samples.parquet"
    args.price_path = args.price_path or args.processed_dir / "price.parquet"
    args.metric_path = args.metric_path or args.processed_dir / "metric.parquet"
    args.summary_path = args.summary_path or args.round_dir / "materialized" / "materialization_summary.json"


def materialize_candidates(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _load_candidates(args.validated_path)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    categories = _select_categories(candidates, args.categories)
    selected_candidates = [item for item in candidates if item["category"] in categories]
    if not selected_candidates:
        raise ValueError("No candidates selected for materialization")
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "gpt_candidate_materialization_v2",
        "started_at": _utc_now(),
        "round_dir": str(args.round_dir),
        "validated_path": str(args.validated_path),
        "samples_path": str(args.samples_path),
        "feature_registry_path": str(args.feature_registry_path),
        "feature_root": str(args.feature_root),
        "categories": sorted(categories),
        "limit": args.limit,
        "blocks": [],
        "status": "running",
    }
    _write_json(args.summary_path, summary)

    LOGGER.info("materialize_base_panel_load_start candidates=%s categories=%s", len(selected_candidates), len(categories))
    base_panel = _load_base_panel(args.samples_path, args.limit)
    LOGGER.info(
        "materialize_base_panel_load_done rows=%s columns=%s rss_mb=%.1f",
        len(base_panel),
        len(base_panel.columns),
        _rss_mb(),
    )

    written_blocks = []
    for category in sorted(categories):
        category_candidates = [item for item in selected_candidates if item["category"] == category]
        block_name = f"{args.block_prefix}_{category}_sample"
        factor_path = args.feature_root / "blocks" / "sample" / f"{block_name}.parquet"
        manifest_path = args.feature_root / "manifests" / f"{block_name}.json"
        if factor_path.exists() and manifest_path.exists() and not args.overwrite:
            LOGGER.info("materialize_skip_existing block=%s", block_name)
            written_blocks.append(
                {
                    "block": block_name,
                    "category": category,
                    "status": "skipped_existing",
                    "factor_path": str(factor_path),
                    "manifest_path": str(manifest_path),
                    "factor_count": len(category_candidates),
                }
            )
            continue
        LOGGER.info("materialize_category_start category=%s candidates=%s", category, len(category_candidates))
        LOGGER.info("materialize_category_panel_load_start category=%s", category)
        panel = _load_materialization_panel(category_candidates, args, base_panel=base_panel)
        LOGGER.info(
            "materialize_category_panel_load_done category=%s rows=%s columns=%s rss_mb=%.1f",
            category,
            len(panel),
            len(panel.columns),
            _rss_mb(),
        )
        result = _materialize_category(category_candidates, panel)
        block = write_factor_block(
            result,
            block_name,
            "sample",
            factor_path,
            manifest_path,
            description=f"GPT-mined candidate sample features for category={category}.",
        )
        block = replace(
            block,
            factor_path=str(_relative_to(factor_path, args.feature_root)),
            manifest_path=str(_relative_to(manifest_path, args.feature_root)),
        )
        upsert_block(args.feature_registry_path, block)
        record = {
            "block": block.name,
            "category": category,
            "status": "written",
            "row_count": block.row_count,
            "factor_count": block.factor_count,
            "factor_path": str(factor_path),
            "manifest_path": str(manifest_path),
            "factors": result.factor_names(),
        }
        written_blocks.append(record)
        summary["blocks"] = written_blocks
        _write_json(args.summary_path, summary)
        del result, panel
        gc.collect()
        LOGGER.info("materialize_category_done block=%s factors=%s rss_mb=%.1f", block.name, block.factor_count, _rss_mb())

    if not args.skip_registry_validate:
        validate_registry(args.feature_registry_path, metadata_only=not args.full_registry_validate)
    summary["blocks"] = written_blocks
    summary["status"] = "ok"
    summary["finished_at"] = _utc_now()
    _write_json(args.summary_path, summary)
    return {
        "round_dir": str(args.round_dir),
        "summary": str(args.summary_path),
        "blocks": len(written_blocks),
        "status": "ok",
    }


def _load_materialization_panel(
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    base_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    raw_inputs, feature_inputs = _split_inputs(candidates)
    panel = _load_base_panel(args.samples_path, args.limit) if base_panel is None else base_panel.copy(deep=False)
    if raw_inputs:
        panel = _attach_raw_inputs(panel, raw_inputs, args)
    if feature_inputs:
        panel = _attach_feature_inputs(panel, feature_inputs, args)
    missing_inputs = sorted((raw_inputs | feature_inputs) - set(panel.columns))
    if missing_inputs:
        raise KeyError(f"Materialization panel missing inputs: {missing_inputs}")
    return panel


def _materialize_category(candidates: list[dict[str, Any]], panel: pd.DataFrame) -> FactorResult:
    evaluator = FormulaEvaluator(panel)
    output_data: dict[str, Any] = {"sample_id": panel["sample_id"].astype(str).to_numpy(copy=False)}
    specs: list[FactorSpec] = []
    for candidate in candidates:
        name = str(candidate["factor_name"])
        LOGGER.info("materialize_factor_start factor=%s", name)
        series = evaluator.evaluate(candidate["formula"])
        values = pd.to_numeric(series, errors="coerce").astype("float32")
        output_data[name] = values.to_numpy(dtype="float32", copy=False)
        windows = [int(item) for item in candidate.get("windows", []) if isinstance(item, int)]
        specs.append(
            FactorSpec(
                name=name,
                source="gpt_mining",
                category=f"gpt_mining.{candidate['category']}",
                inputs=tuple(map(str, candidate.get("inputs", []))),
                expression=str(candidate["formula"]),
                window=max(windows) if windows else None,
                lookback=max(windows) if windows else 0,
                availability="feature_asof_date",
                description=str(candidate.get("regime_link") or candidate.get("hypothesis") or ""),
            )
        )
        LOGGER.info("materialize_factor_done factor=%s non_missing=%s", name, int(values.notna().sum()))
        del series, values
    outputs = pd.DataFrame(output_data)
    return FactorResult(outputs, specs, key_columns=("sample_id",))


class FormulaEvaluator:
    def __init__(self, panel: pd.DataFrame) -> None:
        self.panel = panel
        self.cache: dict[str, pd.Series] = {}
        self._prepare_time_order()

    def evaluate(self, formula: str) -> pd.Series:
        normalized = re.sub(r"\breturn\s*\(", "ts_return(", formula)
        tree = ast.parse(normalized, mode="eval")
        return self._to_series(self._eval_node(tree.body))

    def _prepare_time_order(self) -> None:
        work = self.panel[["stock_code", "trade_date"]].copy()
        work["__pos"] = np.arange(len(work), dtype=np.int64)
        ordered = work.sort_values(["stock_code", "trade_date", "__pos"], kind="mergesort").reset_index(drop=True)
        self.order = ordered["__pos"].to_numpy(dtype=np.int64)
        self.ordered_groups = ordered["stock_code"].astype(str).to_numpy()
        self.n = len(work)

    def _eval_node(self, node: ast.AST) -> pd.Series | float:
        key = ast.dump(node)
        if key in self.cache:
            return self.cache[key]
        if isinstance(node, ast.Name):
            if node.id not in self.panel.columns:
                raise KeyError(f"Unknown formula field: {node.id}")
            result: pd.Series | float = pd.to_numeric(self.panel[node.id], errors="coerce")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                result = float(node.value)
            else:
                raise ValueError(f"Unsupported formula constant: {node.value!r}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                result = -operand
            elif isinstance(node.op, ast.UAdd):
                result = operand
            else:
                raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            result = self._eval_binop(left, right, node.op)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported in formulas")
            result = self._eval_call(node.func.id, node.args)
        else:
            raise ValueError(f"Unsupported formula node: {ast.dump(node)}")
        if isinstance(result, pd.Series):
            self.cache[key] = result
        return result

    def _eval_binop(self, left: pd.Series | float, right: pd.Series | float, op_node: ast.operator) -> pd.Series | float:
        if isinstance(op_node, ast.Add):
            return left + right
        if isinstance(op_node, ast.Sub):
            return left - right
        if isinstance(op_node, ast.Mult):
            return left * right
        if isinstance(op_node, ast.Div):
            return _safe_div(self._to_series(left), self._to_series(right))
        raise ValueError(f"Unsupported binary operator: {ast.dump(op_node)}")

    def _eval_call(self, name: str, args: list[ast.AST]) -> pd.Series:
        if name == "rank_cs":
            self._check_arg_count(name, args, 1)
            return self._rank_cs(self._to_series(self._eval_node(args[0])))
        if name == "zscore_cs":
            self._check_arg_count(name, args, 1)
            return self._zscore_cs(self._to_series(self._eval_node(args[0])))
        if name == "industry_neutralize":
            self._check_arg_count(name, args, 1)
            return self._industry_neutralize(self._to_series(self._eval_node(args[0])))
        if name == "ts_return":
            self._check_arg_count(name, args, 2)
            return self._ts_return(self._to_series(self._eval_node(args[0])), self._window(args[1]))
        if name == "delta":
            self._check_arg_count(name, args, 2)
            return self._delta(self._to_series(self._eval_node(args[0])), self._window(args[1]))
        if name == "rolling_mean":
            self._check_arg_count(name, args, 2)
            return self._rolling(self._to_series(self._eval_node(args[0])), self._window(args[1]), "mean")
        if name == "rolling_sum":
            self._check_arg_count(name, args, 2)
            return self._rolling(self._to_series(self._eval_node(args[0])), self._window(args[1]), "sum")
        if name == "rolling_std":
            self._check_arg_count(name, args, 2)
            return self._rolling(self._to_series(self._eval_node(args[0])), self._window(args[1]), "std")
        if name == "rolling_min":
            self._check_arg_count(name, args, 2)
            return self._rolling(self._to_series(self._eval_node(args[0])), self._window(args[1]), "min")
        if name == "rolling_max":
            self._check_arg_count(name, args, 2)
            return self._rolling(self._to_series(self._eval_node(args[0])), self._window(args[1]), "max")
        if name == "safe_div":
            self._check_arg_count(name, args, 2)
            return _safe_div(self._to_series(self._eval_node(args[0])), self._to_series(self._eval_node(args[1])))
        if name == "log1p":
            self._check_arg_count(name, args, 1)
            return pd.Series(np.log1p(self._to_series(self._eval_node(args[0])).where(lambda item: item > -1)), index=self.panel.index)
        if name == "abs":
            self._check_arg_count(name, args, 1)
            return self._to_series(self._eval_node(args[0])).abs()
        if name == "sign":
            self._check_arg_count(name, args, 1)
            return pd.Series(np.sign(self._to_series(self._eval_node(args[0]))), index=self.panel.index)
        if name == "interaction":
            if len(args) < 2:
                raise ValueError("interaction requires at least two arguments")
            result = self._to_series(self._eval_node(args[0]))
            for arg in args[1:]:
                result = result * self._to_series(self._eval_node(arg))
            return result
        if name == "winsorize":
            self._check_arg_count(name, args, 1)
            return self._winsorize(self._to_series(self._eval_node(args[0])))
        raise ValueError(f"Unsupported formula function: {name}")

    def _check_arg_count(self, name: str, args: list[ast.AST], expected: int) -> None:
        if len(args) != expected:
            raise ValueError(f"{name} expects {expected} args, got {len(args)}")

    def _window(self, node: ast.AST) -> int:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, int) or node.value <= 0:
            raise ValueError(f"Window must be a positive integer: {ast.dump(node)}")
        return int(node.value)

    def _to_series(self, value: pd.Series | float) -> pd.Series:
        if isinstance(value, pd.Series):
            return pd.to_numeric(value, errors="coerce")
        return pd.Series(float(value), index=self.panel.index, dtype="float64")

    def _ordered_series(self, series: pd.Series) -> pd.Series:
        return pd.Series(pd.to_numeric(series, errors="coerce").to_numpy()[self.order], dtype="float64")

    def _restore_order(self, ordered_values: pd.Series) -> pd.Series:
        restored = np.full(self.n, np.nan, dtype="float64")
        restored[self.order] = pd.to_numeric(ordered_values, errors="coerce").to_numpy()
        return pd.Series(restored, index=self.panel.index)

    def _delta(self, series: pd.Series, window: int) -> pd.Series:
        ordered = self._ordered_series(series)
        shifted = ordered.groupby(self.ordered_groups, sort=False).shift(window)
        return _replace_inf(self._restore_order(ordered - shifted))

    def _ts_return(self, series: pd.Series, window: int) -> pd.Series:
        ordered = self._ordered_series(series)
        shifted = ordered.groupby(self.ordered_groups, sort=False).shift(window)
        return _replace_inf(self._restore_order(_safe_div(ordered, shifted) - 1))

    def _rolling(self, series: pd.Series, window: int, method: str) -> pd.Series:
        ordered = self._ordered_series(series)
        result = (
            ordered.groupby(self.ordered_groups, sort=False)
            .rolling(window, min_periods=window)
            .agg(method)
            .reset_index(level=0, drop=True)
        )
        return _replace_inf(self._restore_order(result))

    def _rank_cs(self, series: pd.Series) -> pd.Series:
        result = series.groupby(self.panel["trade_date"], dropna=False).rank(method="average", pct=True, na_option="keep")
        return _replace_inf(result)

    def _zscore_cs(self, series: pd.Series) -> pd.Series:
        grouped = series.groupby(self.panel["trade_date"], dropna=False)
        mean = grouped.transform("mean")
        std = grouped.transform(lambda item: item.std(ddof=0))
        return _safe_div(series - mean, std)

    def _industry_neutralize(self, series: pd.Series) -> pd.Series:
        industry = self.panel["industry"].astype("string")
        valid = industry.notna() & industry.str.strip().ne("")
        result = pd.Series(np.nan, index=self.panel.index, dtype="float64")
        if valid.any():
            mean = series.loc[valid].groupby([self.panel.loc[valid, "trade_date"], industry.loc[valid]], dropna=True).transform("mean")
            result.loc[valid] = series.loc[valid] - mean
        return _replace_inf(result)

    def _winsorize(self, series: pd.Series) -> pd.Series:
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        return _replace_inf(series.clip(lower=lower, upper=upper))


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing validated candidates: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Validated candidates must be a JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def _select_categories(candidates: list[dict[str, Any]], categories_arg: str) -> set[str]:
    available = {str(item["category"]) for item in candidates}
    if categories_arg == "all":
        return available
    requested = {item.strip() for item in categories_arg.split(",") if item.strip()}
    missing = sorted(requested - available)
    if missing:
        raise KeyError(f"Unknown candidate categories: {missing}")
    return requested


def _split_inputs(candidates: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    raw_inputs: set[str] = set()
    feature_inputs: set[str] = set()
    for candidate in candidates:
        input_sources = candidate.get("input_sources", {})
        for name in candidate.get("inputs", []):
            source = str(input_sources.get(name, ""))
            if "feature:" in source:
                feature_inputs.add(str(name))
            elif "processed:" in source:
                raw_inputs.add(str(name))
            else:
                raw_inputs.add(str(name))
    return raw_inputs, feature_inputs


def _load_base_panel(samples_path: Path, limit: int | None) -> pd.DataFrame:
    columns = ["sample_id", "stock_code", "feature_asof_date", "industry"]
    samples = pd.read_parquet(samples_path, columns=columns)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        samples = samples.head(limit).copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["stock_code"] = samples["stock_code"].astype(str)
    samples["trade_date"] = pd.to_datetime(samples["feature_asof_date"], errors="coerce").dt.normalize()
    samples["industry"] = samples["industry"].astype("string")
    samples = samples.drop(columns=["feature_asof_date"])
    if samples["sample_id"].duplicated().any():
        raise ValueError("samples.sample_id must be unique")
    return samples


def _attach_raw_inputs(panel: pd.DataFrame, raw_inputs: set[str], args: argparse.Namespace) -> pd.DataFrame:
    table_fields: dict[str, list[str]] = {}
    for field in sorted(raw_inputs):
        table = RAW_TABLE_PRIORITY.get(field)
        if table is None:
            raise KeyError(f"No raw table mapping for input field: {field}")
        table_fields.setdefault(table, []).append(field)
    result = panel
    for table, fields in table_fields.items():
        path = {"price": args.price_path, "metric": args.metric_path}[table]
        daily = pd.read_parquet(path, columns=["stock_code", "trade_date", *fields])
        daily["stock_code"] = daily["stock_code"].astype(str)
        daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.normalize()
        for field in fields:
            daily[field] = pd.to_numeric(daily[field], errors="coerce").astype("float32")
        daily = daily.drop_duplicates(["stock_code", "trade_date"], keep="last")
        result = result.merge(daily, on=["stock_code", "trade_date"], how="left")
        del daily
    return result


def _attach_feature_inputs(panel: pd.DataFrame, feature_inputs: set[str], args: argparse.Namespace) -> pd.DataFrame:
    blocks = load_feature_blocks(args.feature_registry_path)
    remaining = set(feature_inputs)
    panel_ids = panel["sample_id"].astype(str).reset_index(drop=True)
    feature_frames: list[pd.DataFrame] = []
    for block in blocks:
        columns = sorted(remaining & set(block.factors))
        if not columns:
            continue
        LOGGER.info("materialize_feature_block_load_start block=%s columns=%s", block.name, len(columns))
        frame = pd.read_parquet(block.factor_path, columns=["sample_id", *columns])
        frame["sample_id"] = frame["sample_id"].astype(str)
        if args.limit is not None and len(frame) >= len(panel_ids):
            frame = frame.head(len(panel_ids)).copy()
        if len(frame) == len(panel_ids) and frame["sample_id"].reset_index(drop=True).equals(panel_ids):
            values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").astype("float32")
            values = values.reset_index(drop=True)
            feature_frames.append(values)
        else:
            frame = frame.loc[:, ["sample_id", *columns]].drop_duplicates("sample_id", keep="last")
            aligned = panel[["sample_id"]].merge(frame, on="sample_id", how="left")
            values = aligned.loc[:, columns].apply(pd.to_numeric, errors="coerce").astype("float32")
            feature_frames.append(values)
            del aligned
        remaining -= set(columns)
        del frame, values
        LOGGER.info("materialize_feature_block_load_done block=%s remaining=%s", block.name, len(remaining))
    if remaining:
        raise KeyError(f"Feature inputs missing from feature registry: {sorted(remaining)}")
    if feature_frames:
        panel = pd.concat([panel.reset_index(drop=True), *feature_frames], axis=1)
    return panel


def _safe_div(left: pd.Series, right: pd.Series) -> pd.Series:
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    if not left.index.equals(right.index):
        left, right = left.align(right, join="outer")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = left / right
    result = result.where(right != 0)
    return _replace_inf(result)


def _replace_inf(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _relative_to(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rss_mb() -> float:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
