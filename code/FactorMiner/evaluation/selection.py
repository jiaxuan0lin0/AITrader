from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd

from FactorMiner.core.factor_block import FactorBlock
from FactorMiner.core.registry import FactorRegistry, resolve_registry_path, validate_registry
from aitrader_paths import DATASETS_ROOT


DEFAULT_FEATURE_REGISTRY_PATH = DATASETS_ROOT / "features" / "feature_registry.json"
DEFAULT_EVALUATION_DIR = DATASETS_ROOT / "factors" / "evaluation" / "experiment" / "ad_hoc"
DEFAULT_SAMPLES_PATH = DATASETS_ROOT / "processed" / "samples.parquet"
DEFAULT_PRIMARY_LABEL = "label_next_open_return"
EXACT_CORR_REFINE_MAX_ROWS = 1_000_000
LOGGER = logging.getLogger(__name__)


@dataclass
class UnionFind:
    parent: dict[str, str]

    @classmethod
    def from_values(cls, values: Sequence[str]) -> UnionFind:
        return cls({value: value for value in values})

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select usable factors from quality and single-factor reports.")
    parser.add_argument("--quality-path", type=Path, default=DEFAULT_EVALUATION_DIR / "sample_feature_quality.csv")
    parser.add_argument("--factor-summary-path", type=Path, default=DEFAULT_EVALUATION_DIR / "factor_summary.csv")
    parser.add_argument("--feature-registry-path", type=Path, default=DEFAULT_FEATURE_REGISTRY_PATH)
    parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--primary-label", default=DEFAULT_PRIMARY_LABEL)
    parser.add_argument("--secondary-label", default="label_next_vwap_return")
    parser.add_argument("--since", default=None, help="Inclusive target_trade_date lower bound for correlation sampling.")
    parser.add_argument("--until", default=None, help="Inclusive target_trade_date upper bound for correlation sampling.")
    parser.add_argument("--min-rank-ic-days", type=int, default=60)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--min-abs-rank-ic", type=float, default=0.01)
    parser.add_argument("--min-abs-rank-ic-ir", type=float, default=0.0)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--min-corr-pairs", type=int, default=10_000)
    parser.add_argument("--corr-method", choices=("spearman", "pearson"), default="spearman")
    parser.add_argument("--corr-row-limit", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--max-selected", type=int, default=0, help="0 means no cap after redundancy removal.")
    parser.add_argument("--skip-registry-validate", action="store_true")
    parser.add_argument("--full-registry-validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_selection(args)
    print(
        " ".join(
            [
                f"candidates={result['candidate_count']}",
                f"selected={result['selected_count']}",
                f"clusters={result['cluster_count']}",
                f"conflicts={result['conflict_count']}",
                f"selected_json={result['selected_json_path']}",
            ]
        )
    )
    return 0


def run_selection(args: argparse.Namespace) -> dict[str, Any]:
    since = _parse_optional_date(getattr(args, "since", None), "since")
    until = _parse_optional_date(getattr(args, "until", None), "until")
    _validate_args(args, since, until)
    if not args.skip_registry_validate:
        validate_registry(args.feature_registry_path, metadata_only=not args.full_registry_validate)

    quality = _load_csv(args.quality_path, "quality")
    summary = _load_csv(args.factor_summary_path, "factor_summary")
    candidates = _build_candidate_table(quality, summary, args)
    candidate_pool = candidates.loc[candidates["selection_status"].eq("candidate")].copy()

    conflicts = pd.DataFrame()
    clusters = _empty_clusters()
    if not candidate_pool.empty:
        LOGGER.info("selection_feature_panel_load_start candidates=%s row_limit=%s", len(candidate_pool), args.corr_row_limit)
        sample_ids = _load_allowed_sample_ids(getattr(args, "samples_path", DEFAULT_SAMPLES_PATH), since, until, args.corr_row_limit)
        feature_panel = _load_feature_panel(args.feature_registry_path, candidate_pool, args.corr_row_limit, sample_ids)
        LOGGER.info("selection_feature_panel_load_done rows=%s columns=%s", len(feature_panel), len(feature_panel.columns))
        LOGGER.info("selection_correlation_start factors=%s", len(candidate_pool))
        conflicts = _compute_correlation_conflicts(feature_panel, candidate_pool["factor_name"].tolist(), args)
        LOGGER.info("selection_correlation_done conflicts=%s", len(conflicts))
        clusters = _build_correlation_clusters(candidate_pool, conflicts)
        LOGGER.info("selection_clusters_done clusters=%s", clusters["cluster_id"].nunique() if not clusters.empty else 0)
        candidates = _apply_cluster_selection(candidates, clusters, conflicts, args)
    else:
        candidates["cluster_id"] = pd.NA
        candidates["cluster_size"] = 0
        candidates["selected"] = False
        candidates["final_reject_reason"] = candidates["reject_reason"]
        candidates["high_corr_with"] = pd.NA
        candidates["corr_with_selected"] = np.nan

    selected = candidates.loc[candidates["selected"]].copy()
    rejected = candidates.loc[~candidates["selected"]].copy()
    output_paths = _write_outputs(args, candidates, selected, rejected, conflicts, clusters)
    result = {
        "quality_path": str(args.quality_path),
        "factor_summary_path": str(args.factor_summary_path),
        "feature_registry_path": str(args.feature_registry_path),
        "primary_label": args.primary_label,
        "candidate_count": int(candidates["selection_status"].eq("candidate").sum()),
        "borderline_count": int(candidates["selection_status"].eq("borderline").sum()),
        "rejected_count": int((~candidates["selected"]).sum()),
        "selected_count": int(len(selected)),
        "cluster_count": int(clusters["cluster_id"].nunique()) if not clusters.empty else 0,
        "conflict_count": int(len(conflicts)),
        **output_paths,
    }
    metadata_path = args.output_dir / "selection_summary.json"
    metadata_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    result["metadata_path"] = str(metadata_path)
    return result


def _build_candidate_table(quality: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    _require_columns(quality, ("block", "factor_name", "quality_pass", "missing_rate", "constant_flag", "quality_flags"), "quality")
    _require_columns(
        summary,
        (
            "block",
            "factor_name",
            "label",
            "rank_ic_day_count",
            "rank_ic_mean",
            "rank_ic_ir",
            "rank_ic_positive_rate",
            "coverage_mean",
            "group_spread_mean",
            "group_spread_positive_rate",
        ),
        "factor_summary",
    )
    primary = summary.loc[summary["label"].eq(args.primary_label)].copy()
    if primary.empty:
        raise ValueError(f"Primary label not found in factor_summary: {args.primary_label}")
    secondary = summary.loc[summary["label"].eq(args.secondary_label)].copy()

    merged = quality.merge(primary, on=["block", "factor_name"], how="inner", suffixes=("_quality", ""))
    if merged.empty:
        raise ValueError("No factors overlap between quality and factor_summary for primary label")
    merged = _merge_secondary_metrics(merged, secondary)
    for column in (
        "rank_ic_day_count",
        "rank_ic_mean",
        "rank_ic_ir",
        "rank_ic_positive_rate",
        "coverage_mean",
        "group_spread_mean",
        "group_spread_positive_rate",
        "missing_rate",
        "non_missing_count",
    ):
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["quality_pass"] = _as_bool(merged["quality_pass"])
    merged["constant_flag"] = _as_bool(merged["constant_flag"])
    statuses = merged.apply(lambda row: _selection_status(row, args), axis=1, result_type="expand")
    statuses.columns = ["selection_status", "reject_reason"]
    merged = pd.concat([merged, statuses], axis=1)
    rank_ic_mean = pd.to_numeric(merged["rank_ic_mean"], errors="coerce")
    merged["direction"] = np.where(rank_ic_mean.gt(0), 1, np.where(rank_ic_mean.lt(0), -1, 0)).astype("int64")
    merged["abs_rank_ic_mean"] = pd.to_numeric(merged["rank_ic_mean"], errors="coerce").abs()
    merged["abs_rank_ic_ir"] = pd.to_numeric(merged["rank_ic_ir"], errors="coerce").abs()
    merged["directional_rank_ic_hit_rate"] = merged.apply(_directional_hit_rate, axis=1)
    merged["secondary_direction_consistent"] = merged.apply(_secondary_direction_consistent, axis=1)
    merged["selection_score"] = merged.apply(_selection_score, axis=1)
    merged["selected"] = False
    merged["cluster_id"] = pd.NA
    merged["cluster_size"] = 0
    merged["final_reject_reason"] = merged["reject_reason"]
    merged["high_corr_with"] = pd.NA
    merged["corr_with_selected"] = np.nan
    return merged.sort_values(["selection_score", "factor_name"], ascending=[False, True]).reset_index(drop=True)


def _selection_status(row: pd.Series, args: argparse.Namespace) -> tuple[str, str]:
    reasons: list[str] = []
    if not bool(row.get("quality_pass", False)):
        reasons.append("quality_failed")
    if bool(row.get("constant_flag", False)):
        reasons.append("constant")
    rank_days = _num(row.get("rank_ic_day_count"))
    coverage = _num(row.get("coverage_mean"))
    abs_rank_ic = abs(_num(row.get("rank_ic_mean")))
    abs_ir = abs(_num(row.get("rank_ic_ir")))
    if pd.isna(rank_days) or rank_days < args.min_rank_ic_days:
        reasons.append("rank_ic_days_too_low")
    if pd.isna(coverage) or coverage < args.min_coverage:
        reasons.append("coverage_too_low")
    if pd.isna(abs_rank_ic) or abs_rank_ic < args.min_abs_rank_ic:
        reasons.append("rank_ic_too_weak")
    if args.min_abs_rank_ic_ir > 0 and (pd.isna(abs_ir) or abs_ir < args.min_abs_rank_ic_ir):
        reasons.append("rank_ic_ir_too_weak")

    if not reasons:
        return "candidate", ""
    source = str(row.get("source", ""))
    factor_name = str(row.get("factor_name", ""))
    if source == "news_llm" and "quality_failed" not in reasons and "constant" not in reasons:
        if abs_rank_ic >= args.min_abs_rank_ic and ("coverage_too_low" in reasons or "rank_ic_days_too_low" in reasons):
            return "borderline", ";".join(reasons)
    if factor_name.startswith("news_") and "quality_failed" not in reasons and "constant" not in reasons:
        if abs_rank_ic >= args.min_abs_rank_ic and ("coverage_too_low" in reasons or "rank_ic_days_too_low" in reasons):
            return "borderline", ";".join(reasons)
    return "rejected", ";".join(reasons)


def _load_feature_panel(
    feature_registry_path: Path,
    candidates: pd.DataFrame,
    row_limit: int,
    allowed_sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    registry = FactorRegistry.load(feature_registry_path)
    base_dir = feature_registry_path.parent
    by_block = candidates.groupby("block")["factor_name"].apply(list).to_dict()
    sample_ids = allowed_sample_ids if allowed_sample_ids is not None else _sample_feature_ids(registry, base_dir, by_block, row_limit)
    sample_id_set = set(sample_ids) if sample_ids is not None else None
    panel: pd.DataFrame | None = pd.DataFrame({"sample_id": sample_ids}) if sample_ids is not None else None
    for block in registry.blocks:
        if block.name not in by_block:
            continue
        columns = ["sample_id", *by_block[block.name]]
        factor_path = resolve_registry_path(block.factor_path, base_dir)
        LOGGER.info("selection_feature_block_load_start block=%s columns=%s", block.name, len(columns))
        frame = pd.read_parquet(factor_path, columns=columns)
        frame = frame.copy()
        frame["sample_id"] = frame["sample_id"].astype(str)
        _downcast_factor_columns(frame, by_block[block.name])
        if sample_id_set is not None:
            frame = frame.loc[frame["sample_id"].isin(sample_id_set)].copy()
            frame = pd.DataFrame({"sample_id": sample_ids}).merge(frame, on="sample_id", how="left")
        LOGGER.info("selection_feature_block_load_done block=%s rows=%s columns=%s", block.name, len(frame), len(frame.columns))
        if panel is None:
            panel = frame
        elif len(panel) == len(frame) and panel["sample_id"].equals(frame["sample_id"]):
            panel = pd.concat([panel, frame.drop(columns=["sample_id"])], axis=1)
        else:
            panel = panel.merge(frame, on="sample_id", how="left" if sample_id_set is not None else "outer")
        del frame
        gc.collect()
    if panel is None:
        return pd.DataFrame(columns=["sample_id", *candidates["factor_name"].tolist()])
    return panel


def _load_allowed_sample_ids(
    samples_path: Path,
    since: pd.Timestamp | None,
    until: pd.Timestamp | None,
    row_limit: int,
) -> list[str] | None:
    if since is None and until is None:
        return None
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples parquet for date-filtered selection: {samples_path}")
    samples = pd.read_parquet(samples_path, columns=["sample_id", "target_trade_date"])
    _require_columns(samples, ("sample_id", "target_trade_date"), "samples")
    work = samples.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["target_trade_date"] = pd.to_datetime(work["target_trade_date"], errors="coerce").dt.normalize()
    if work["target_trade_date"].isna().any():
        missing = int(work["target_trade_date"].isna().sum())
        raise ValueError(f"samples.target_trade_date contains missing or invalid dates: {missing}")
    if since is not None:
        work = work.loc[work["target_trade_date"].ge(since)]
    if until is not None:
        work = work.loc[work["target_trade_date"].le(until)]
    work = work.drop_duplicates("sample_id")
    if row_limit > 0 and len(work) > row_limit:
        work = work.sample(n=row_limit, random_state=20260520)
    ids = work["sample_id"].tolist()
    LOGGER.info("selection_date_sample_filter samples=%s since=%s until=%s", len(ids), since, until)
    return ids


def _downcast_factor_columns(frame: pd.DataFrame, factor_names: Sequence[str]) -> None:
    for factor_name in factor_names:
        if factor_name in frame.columns:
            frame[factor_name] = pd.to_numeric(frame[factor_name], errors="coerce").astype("float32")


def _compute_correlation_conflicts(panel: pd.DataFrame, factor_names: list[str], args: argparse.Namespace) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    available = [name for name in factor_names if name in panel.columns]
    if len(available) < 2:
        return pd.DataFrame(records)

    numeric = panel.loc[:, available]
    LOGGER.info("selection_correlation_matrix_start rows=%s factors=%s method=%s", len(numeric), len(available), args.corr_method)
    values, non_missing = _fast_correlation_matrix(numeric, args.corr_method, work_dir=getattr(args, "output_dir", None))
    LOGGER.info("selection_correlation_matrix_done factors=%s", len(available))
    min_counts = np.minimum.outer(non_missing, non_missing)
    upper = np.triu(np.ones(values.shape, dtype=bool), k=1)
    mask = upper & np.isfinite(values) & (min_counts >= args.min_corr_pairs) & (np.abs(values) >= args.corr_threshold)
    left_indexes, right_indexes = np.where(mask)
    LOGGER.info("selection_correlation_refine_start approximate_conflicts=%s", len(left_indexes))
    exact_refine = len(numeric) <= EXACT_CORR_REFINE_MAX_ROWS
    if not exact_refine:
        LOGGER.info(
            "selection_correlation_refine_mode mode=matrix rows=%s max_exact_rows=%s",
            len(numeric),
            EXACT_CORR_REFINE_MAX_ROWS,
        )
    for left_index, right_index in zip(left_indexes, right_indexes):
        left = available[left_index]
        right = available[right_index]
        if exact_refine:
            corr, overlap_count = _exact_pair_corr(numeric[left], numeric[right], args.corr_method)
        else:
            corr = float(values[left_index, right_index])
            overlap_count = int(min_counts[left_index, right_index])
        if overlap_count < args.min_corr_pairs:
            continue
        if pd.isna(corr):
            continue
        corr = float(corr)
        if abs(corr) < args.corr_threshold:
            continue
        records.append(
            {
                "factor_a": str(left),
                "factor_b": str(right),
                "corr": corr,
                "abs_corr": abs(corr),
                "overlap_count": overlap_count,
                "corr_method": args.corr_method,
            }
        )
    LOGGER.info("selection_correlation_refine_done conflicts=%s", len(records))
    return pd.DataFrame(records)


def _fast_correlation_matrix(frame: pd.DataFrame, method: str, work_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = frame.shape
    work_root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="selection_corr_zscores_", suffix=".dat", dir=work_root, delete=False) as handle:
        zscore_path = Path(handle.name)
    zscores: np.memmap | None = None
    try:
        zscores = np.memmap(zscore_path, dtype="float32", mode="w+", shape=(rows, cols), order="F")
        non_missing = np.zeros(cols, dtype="int64")
        for column_index, column_name in enumerate(frame.columns):
            if column_index and column_index % 25 == 0:
                LOGGER.info("selection_correlation_standardize_progress columns=%s/%s", column_index, cols)
            zscores[:, column_index], non_missing[column_index] = _standardized_column(frame[column_name], method)
        zscores.flush()
        LOGGER.info("selection_correlation_standardize_done columns=%s work_path=%s", cols, zscore_path)

        corr = np.full((cols, cols), np.nan, dtype="float32")
        block_size = 64
        for start in range(0, cols, block_size):
            end = min(start + block_size, cols)
            LOGGER.info("selection_correlation_dot_progress columns=%s:%s/%s", start, end, cols)
            numerator = zscores[:, start:end].T @ zscores
            denominator = np.minimum(non_missing[start:end, None], non_missing[None, :]).astype("float32") - 1.0
            np.divide(
                numerator,
                denominator,
                out=numerator,
                where=denominator > 0,
            )
            corr[start:end, :] = numerator
        np.fill_diagonal(corr, 1.0)
        return corr.astype("float64", copy=False), non_missing
    finally:
        if zscores is not None:
            del zscores
        try:
            zscore_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("selection_correlation_workfile_cleanup_failed path=%s", zscore_path)


def _standardized_column(series: pd.Series, method: str) -> tuple[np.ndarray, int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    values[~np.isfinite(values)] = np.nan
    if method == "spearman":
        values = pd.Series(values, copy=False).rank(method="average", na_option="keep").to_numpy(dtype="float64", na_value=np.nan)
    valid = np.isfinite(values)
    count = int(valid.sum())
    output = np.zeros(len(values), dtype="float32")
    if count <= 1:
        return output, count
    valid_values = values[valid]
    centered = valid_values - valid_values.mean(dtype="float64")
    variance = float(np.dot(centered, centered))
    if variance <= 0 or not np.isfinite(variance):
        return output, count
    output[valid] = (centered / np.sqrt(variance / (count - 1))).astype("float32", copy=False)
    return output, count


def _exact_pair_corr(left: pd.Series, right: pd.Series, method: str) -> tuple[float, int]:
    left_values = pd.to_numeric(left, errors="coerce")
    right_values = pd.to_numeric(right, errors="coerce")
    finite = np.isfinite(left_values.to_numpy(dtype="float64", na_value=np.nan)) & np.isfinite(
        right_values.to_numpy(dtype="float64", na_value=np.nan)
    )
    if not finite.any():
        return np.nan, 0
    overlap_count = int(finite.sum())
    left_valid = left_values.loc[finite]
    right_valid = right_values.loc[finite]
    if left_valid.nunique(dropna=True) <= 1 or right_valid.nunique(dropna=True) <= 1:
        return np.nan, overlap_count
    if method == "spearman":
        left_valid = left_valid.rank(method="average")
        right_valid = right_valid.rank(method="average")
    left_array = left_valid.to_numpy(dtype="float64", copy=False)
    right_array = right_valid.to_numpy(dtype="float64", copy=False)
    left_array = left_array - left_array.mean()
    right_array = right_array - right_array.mean()
    denominator = np.sqrt(np.dot(left_array, left_array) * np.dot(right_array, right_array))
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan, overlap_count
    return float(np.dot(left_array, right_array) / denominator), overlap_count


def _sample_feature_ids(
    registry: FactorRegistry,
    base_dir: Path,
    by_block: dict[str, list[str]],
    row_limit: int,
) -> list[str] | None:
    if row_limit <= 0:
        return None
    for block in registry.blocks:
        if block.name not in by_block:
            continue
        factor_path = resolve_registry_path(block.factor_path, base_dir)
        ids = pd.read_parquet(factor_path, columns=["sample_id"])
        ids["sample_id"] = ids["sample_id"].astype(str)
        if len(ids) > row_limit:
            ids = ids.sample(n=row_limit, random_state=20260520)
        ids = ids.drop_duplicates("sample_id")
        return ids["sample_id"].tolist()
    return []


def _build_correlation_clusters(candidates: pd.DataFrame, conflicts: pd.DataFrame) -> pd.DataFrame:
    factors = candidates["factor_name"].tolist()
    uf = UnionFind.from_values(factors)
    if not conflicts.empty:
        for _, row in conflicts.iterrows():
            uf.union(str(row["factor_a"]), str(row["factor_b"]))
    groups: dict[str, list[str]] = defaultdict(list)
    for factor in factors:
        groups[uf.find(factor)].append(factor)

    candidate_by_factor = candidates.set_index("factor_name")
    records: list[dict[str, Any]] = []
    for cluster_number, members in enumerate(sorted(groups.values(), key=lambda items: (-len(items), sorted(items)[0])), start=1):
        ranked = sorted(
            members,
            key=lambda name: (
                -_num(candidate_by_factor.loc[name, "selection_score"]),
                -_num(candidate_by_factor.loc[name, "coverage_mean"]),
                name,
            ),
        )
        representative = ranked[0]
        cluster_id = f"cluster_{cluster_number:04d}"
        for member in ranked:
            row = candidate_by_factor.loc[member]
            records.append(
                {
                    "cluster_id": cluster_id,
                    "factor_name": member,
                    "representative": representative,
                    "is_representative": member == representative,
                    "cluster_size": len(members),
                    "score": _num(row["selection_score"]),
                    "rank_ic_mean": _num(row["rank_ic_mean"]),
                    "rank_ic_ir": _num(row["rank_ic_ir"]),
                    "coverage_mean": _num(row["coverage_mean"]),
                    "block": row["block"],
                    "source": row.get("source", ""),
                    "category": row.get("category", ""),
                }
            )
    return pd.DataFrame(records)


def _apply_cluster_selection(
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    conflicts: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    work = candidates.copy()
    cluster_info = clusters.loc[:, ["factor_name", "cluster_id", "representative", "is_representative", "cluster_size"]]
    work = work.merge(cluster_info, on="factor_name", how="left", suffixes=("", "_cluster"))
    work["cluster_id"] = work["cluster_id_cluster"].combine_first(work["cluster_id"])
    work["cluster_size"] = pd.to_numeric(work["cluster_size_cluster"], errors="coerce").fillna(0).astype("int64")
    work = work.drop(columns=[column for column in ("cluster_id_cluster", "cluster_size_cluster") if column in work.columns])
    work["selected"] = work["selection_status"].eq("candidate") & work["is_representative"].fillna(False).astype(bool)

    if args.max_selected > 0:
        selected_names = (
            work.loc[work["selected"]]
            .sort_values(["selection_score", "factor_name"], ascending=[False, True])
            .head(args.max_selected)["factor_name"]
            .tolist()
        )
        work["selected"] = work["factor_name"].isin(selected_names)

    conflict_lookup = _conflict_lookup(conflicts)
    for index, row in work.iterrows():
        if row["selected"]:
            work.at[index, "final_reject_reason"] = ""
            continue
        if row["selection_status"] == "candidate":
            representative = row.get("representative")
            if pd.notna(representative) and representative != row["factor_name"]:
                work.at[index, "high_corr_with"] = representative
                work.at[index, "corr_with_selected"] = conflict_lookup.get((row["factor_name"], representative), np.nan)
                work.at[index, "final_reject_reason"] = f"high_corr_with:{representative}"
            elif args.max_selected > 0:
                work.at[index, "final_reject_reason"] = "max_selected_cap"
            else:
                work.at[index, "final_reject_reason"] = row.get("reject_reason", "")
        else:
            work.at[index, "final_reject_reason"] = row.get("reject_reason", "")
    return work.sort_values(["selected", "selection_score", "factor_name"], ascending=[False, False, True]).reset_index(drop=True)


def _write_outputs(
    args: argparse.Namespace,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    rejected: pd.DataFrame,
    conflicts: pd.DataFrame,
    clusters: pd.DataFrame,
) -> dict[str, str]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_features_path": args.output_dir / "candidate_features.csv",
        "rejected_features_path": args.output_dir / "rejected_features.csv",
        "selected_features_csv_path": args.output_dir / "selected_features.csv",
        "selected_json_path": args.output_dir / "selected_features.json",
        "correlation_conflicts_path": args.output_dir / "correlation_conflicts.csv",
        "correlation_clusters_path": args.output_dir / "correlation_clusters.csv",
        "review_packet_path": args.output_dir / "review_packet.json",
    }
    candidates.to_csv(paths["candidate_features_path"], index=False)
    rejected.to_csv(paths["rejected_features_path"], index=False)
    selected.to_csv(paths["selected_features_csv_path"], index=False)
    conflicts.to_csv(paths["correlation_conflicts_path"], index=False)
    clusters.to_csv(paths["correlation_clusters_path"], index=False)
    selected_json = _selected_features_json(args, selected)
    paths["selected_json_path"].write_text(json.dumps(_jsonable(selected_json), ensure_ascii=False, indent=2), encoding="utf-8")
    review_packet = _review_packet(args, candidates, selected, rejected, conflicts, clusters)
    paths["review_packet_path"].write_text(json.dumps(_jsonable(review_packet), ensure_ascii=False, indent=2), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _selected_features_json(args: argparse.Namespace, selected: pd.DataFrame) -> dict[str, Any]:
    blocks: dict[str, list[str]] = {}
    for block, group in selected.groupby("block", sort=True):
        blocks[str(block)] = group.sort_values("factor_name")["factor_name"].tolist()
    selected_features = selected.sort_values(["block", "factor_name"])["factor_name"].tolist()
    directions = {row["factor_name"]: int(row["direction"]) for _, row in selected.iterrows()}
    return {
        "version": f"auto_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "selection_mode": "auto",
        "primary_label": args.primary_label,
        "selected_features": selected_features,
        "directions": directions,
        "blocks": blocks,
        "config": _selection_config(args),
    }


def _review_packet(
    args: argparse.Namespace,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    rejected: pd.DataFrame,
    conflicts: pd.DataFrame,
    clusters: pd.DataFrame,
) -> dict[str, Any]:
    rejected_counts = Counter(str(value) for value in rejected["final_reject_reason"].fillna(""))
    borderline = candidates.loc[candidates["selection_status"].eq("borderline")].sort_values("selection_score", ascending=False)
    cluster_records: list[dict[str, Any]] = []
    if not clusters.empty:
        for cluster_id, group in clusters.groupby("cluster_id", sort=True):
            if len(group) <= 1:
                continue
            cluster_records.append(
                {
                    "cluster_id": cluster_id,
                    "representative": group.loc[group["is_representative"], "factor_name"].iloc[0],
                    "members": group.sort_values("score", ascending=False)[
                        ["factor_name", "score", "rank_ic_mean", "rank_ic_ir", "coverage_mean", "source", "category"]
                    ].to_dict("records"),
                }
            )
    return {
        "selection_config": _selection_config(args),
        "summary_counts": {
            "total_features": int(len(candidates)),
            "candidate_count": int(candidates["selection_status"].eq("candidate").sum()),
            "borderline_count": int(candidates["selection_status"].eq("borderline").sum()),
            "selected_count": int(len(selected)),
            "rejected_count": int(len(rejected)),
            "conflict_count": int(len(conflicts)),
            "correlation_cluster_count": int(len(cluster_records)),
        },
        "selected_features": selected.sort_values("selection_score", ascending=False).head(500).to_dict("records"),
        "borderline_cases": borderline.head(200).to_dict("records"),
        "rejected_reason_counts": dict(rejected_counts),
        "correlation_clusters": cluster_records[:200],
        "top_rejected_by_score": rejected.sort_values("selection_score", ascending=False).head(100).to_dict("records"),
    }


def _merge_secondary_metrics(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    if secondary.empty:
        primary["secondary_rank_ic_mean"] = np.nan
        primary["secondary_rank_ic_ir"] = np.nan
        primary["secondary_rank_ic_day_count"] = np.nan
        return primary
    columns = ["block", "factor_name", "rank_ic_mean", "rank_ic_ir", "rank_ic_day_count"]
    renamed = secondary.loc[:, columns].rename(
        columns={
            "rank_ic_mean": "secondary_rank_ic_mean",
            "rank_ic_ir": "secondary_rank_ic_ir",
            "rank_ic_day_count": "secondary_rank_ic_day_count",
        }
    )
    return primary.merge(renamed, on=["block", "factor_name"], how="left")


def _selection_score(row: pd.Series) -> float:
    if row.get("selection_status") not in {"candidate", "borderline"}:
        return 0.0
    abs_rank_ic = _num(row.get("abs_rank_ic_mean"))
    day_count = max(_num(row.get("rank_ic_day_count")), 1.0)
    coverage = max(_num(row.get("coverage_mean")), 1e-6)
    ir = min(_num(row.get("abs_rank_ic_ir")), 3.0)
    hit_rate = _num(row.get("directional_rank_ic_hit_rate"))
    if pd.isna(hit_rate):
        hit_rate = 0.5
    return float(abs_rank_ic * np.sqrt(day_count) * np.sqrt(coverage) * (1.0 + ir) * max(hit_rate, 0.1))


def _directional_hit_rate(row: pd.Series) -> float:
    positive_rate = _num(row.get("rank_ic_positive_rate"))
    direction = int(row.get("direction", 0)) if pd.notna(row.get("direction", np.nan)) else 0
    if pd.isna(positive_rate) or direction == 0:
        return np.nan
    return float(positive_rate if direction > 0 else 1.0 - positive_rate)


def _secondary_direction_consistent(row: pd.Series) -> bool | float:
    primary = _num(row.get("rank_ic_mean"))
    secondary = _num(row.get("secondary_rank_ic_mean"))
    if pd.isna(primary) or pd.isna(secondary) or primary == 0 or secondary == 0:
        return np.nan
    return bool(np.sign(primary) == np.sign(secondary))


def _conflict_lookup(conflicts: pd.DataFrame) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    if conflicts.empty:
        return lookup
    for _, row in conflicts.iterrows():
        left = str(row["factor_a"])
        right = str(row["factor_b"])
        corr = float(row["corr"])
        lookup[(left, right)] = corr
        lookup[(right, left)] = corr
    return lookup


def _empty_clusters() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cluster_id",
            "factor_name",
            "representative",
            "is_representative",
            "cluster_size",
            "score",
            "rank_ic_mean",
            "rank_ic_ir",
            "coverage_mean",
            "block",
            "source",
            "category",
        ]
    )


def _selection_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "primary_label": args.primary_label,
        "secondary_label": args.secondary_label,
        "samples_path": str(getattr(args, "samples_path", "")),
        "since": getattr(args, "since", None),
        "until": getattr(args, "until", None),
        "min_rank_ic_days": args.min_rank_ic_days,
        "min_coverage": args.min_coverage,
        "min_abs_rank_ic": args.min_abs_rank_ic,
        "min_abs_rank_ic_ir": args.min_abs_rank_ic_ir,
        "corr_threshold": args.corr_threshold,
        "min_corr_pairs": args.min_corr_pairs,
        "corr_method": args.corr_method,
        "corr_row_limit": args.corr_row_limit,
        "max_selected": args.max_selected,
    }


def _load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} csv: {path}")
    return pd.read_csv(path)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def _num(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return np.nan
    return parsed


def _parse_optional_date(value: str | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --{name} date: {value}")
    return parsed.normalize()


def _validate_args(args: argparse.Namespace, since: pd.Timestamp | None, until: pd.Timestamp | None) -> None:
    if args.min_rank_ic_days < 0:
        raise ValueError("--min-rank-ic-days cannot be negative")
    if not 0 <= args.min_coverage <= 1:
        raise ValueError("--min-coverage must be between 0 and 1")
    if args.min_abs_rank_ic < 0:
        raise ValueError("--min-abs-rank-ic cannot be negative")
    if args.min_abs_rank_ic_ir < 0:
        raise ValueError("--min-abs-rank-ic-ir cannot be negative")
    if not 0 <= args.corr_threshold <= 1:
        raise ValueError("--corr-threshold must be between 0 and 1")
    if args.min_corr_pairs < 2:
        raise ValueError("--min-corr-pairs must be at least 2")
    if args.corr_row_limit < 0:
        raise ValueError("--corr-row-limit cannot be negative")
    if args.max_selected < 0:
        raise ValueError("--max-selected cannot be negative")
    if since is not None and until is not None and since > until:
        raise ValueError("--since cannot be later than --until")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
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
