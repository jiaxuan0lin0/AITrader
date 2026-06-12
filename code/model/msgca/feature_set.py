from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REVIEWED_FILE = "selected_features_reviewed.json"
AUTO_FILE = "selected_features.json"


@dataclass(frozen=True)
class SelectedFeatureSet:
    selected_features: list[str]
    blocks: dict[str, list[str]]
    source_path: Path | None
    mode: str


@dataclass(frozen=True)
class FeatureBlockInfo:
    name: str
    granularity: str
    factor_path: Path
    manifest_path: Path
    factors: list[str]


@dataclass
class SampleFeatureMatrix:
    columns: list[str]
    values: np.ndarray


def load_selected_features(
    evaluation_dir: str | Path,
    explicit_path: str | Path | None = None,
) -> SelectedFeatureSet:
    """Load selected features with reviewed > auto priority."""
    candidates: list[Path]
    if explicit_path is not None:
        candidates = [Path(explicit_path)]
    else:
        root = Path(evaluation_dir)
        candidates = [root / REVIEWED_FILE, root / AUTO_FILE]

    for path in candidates:
        if path.exists():
            return _parse_selected_feature_file(path)

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No selected feature file found. Searched: {searched}")


def load_feature_blocks(feature_registry_path: str | Path) -> list[FeatureBlockInfo]:
    registry_path = Path(feature_registry_path)
    if not registry_path.exists():
        return []
    records = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"feature_registry.json must contain a list: {registry_path}")
    base_dir = registry_path.parent
    blocks: list[FeatureBlockInfo] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid registry record in {registry_path}: {record!r}")
        granularity = str(record.get("granularity", ""))
        if granularity != "sample":
            continue
        factor_path = _resolve_path(record["factor_path"], base_dir)
        manifest_path = _resolve_path(record["manifest_path"], base_dir)
        factors = _load_manifest_factor_names(manifest_path) if manifest_path.exists() else []
        blocks.append(
            FeatureBlockInfo(
                name=str(record["name"]),
                granularity=granularity,
                factor_path=factor_path,
                manifest_path=manifest_path,
                factors=factors,
            )
        )
    return blocks


def load_sample_feature_panel(
    feature_registry_path: str | Path,
    selected_features: Sequence[str],
    sample_ids: Iterable[str] | None = None,
    strict: bool = False,
    use_polars: bool = False,
) -> pd.DataFrame:
    """Load selected sample-level feature columns from registered blocks."""
    selected = list(dict.fromkeys(str(item) for item in selected_features))
    if not selected:
        return pd.DataFrame({"sample_id": _materialize_ids(sample_ids)})

    blocks = load_feature_blocks(feature_registry_path)
    feature_to_block: dict[str, FeatureBlockInfo] = {}
    for block in blocks:
        for factor in block.factors:
            feature_to_block[factor] = block

    missing = [feature for feature in selected if feature not in feature_to_block]
    if missing and strict:
        preview = ", ".join(missing[:10])
        raise KeyError(f"Selected features missing from feature registry: {preview}")

    requested_ids = None if sample_ids is None else list(dict.fromkeys(map(str, sample_ids)))
    if use_polars:
        return _load_sample_feature_panel_polars(blocks, selected, requested_ids)

    requested_id_set = None if requested_ids is None else set(requested_ids)
    frames: list[pd.DataFrame] = []
    for block in blocks:
        columns = [feature for feature in selected if feature in set(block.factors)]
        if not columns:
            continue
        if not block.factor_path.exists():
            if strict:
                raise FileNotFoundError(f"Missing feature block parquet: {block.factor_path}")
            continue
        frame = _read_factor_columns(block.factor_path, columns, requested_id_set)
        frame["sample_id"] = frame["sample_id"].astype(str)
        frames.append(frame)

    if not frames:
        return pd.DataFrame({"sample_id": _materialize_ids(sample_ids)})

    panel = frames[0]
    for frame in frames[1:]:
        overlap = sorted((set(panel.columns) & set(frame.columns)) - {"sample_id"})
        if overlap:
            raise ValueError(f"Duplicate feature columns across sample blocks: {overlap}")
        panel = panel.merge(frame, on="sample_id", how="outer")
    return panel


def load_sample_feature_matrix(
    feature_registry_path: str | Path,
    selected_features: Sequence[str],
    sample_ids: Iterable[str],
    strict: bool = False,
) -> SampleFeatureMatrix:
    """Load selected sample-level features as an ordered float32 matrix."""
    selected = list(dict.fromkeys(str(item) for item in selected_features))
    requested_ids = list(dict.fromkeys(map(str, sample_ids)))
    if not selected:
        return SampleFeatureMatrix([], np.zeros((len(requested_ids), 0), dtype="float32"))

    blocks = load_feature_blocks(feature_registry_path)
    feature_to_block: dict[str, FeatureBlockInfo] = {}
    for block in blocks:
        for factor in block.factors:
            feature_to_block[factor] = block

    missing = [feature for feature in selected if feature not in feature_to_block]
    if missing and strict:
        preview = ", ".join(missing[:10])
        raise KeyError(f"Selected features missing from feature registry: {preview}")

    values = np.full((len(requested_ids), len(selected)), np.nan, dtype="float32")
    row_by_id = {sample_id: index for index, sample_id in enumerate(requested_ids)}
    selected_index = {feature: index for index, feature in enumerate(selected)}
    loaded_columns: set[str] = set()
    for block in blocks:
        columns = [feature for feature in selected if feature in set(block.factors)]
        if not columns:
            continue
        overlap = sorted(loaded_columns & set(columns))
        if overlap:
            raise ValueError(f"Duplicate feature columns across sample blocks: {overlap}")
        loaded_columns.update(columns)
        if not block.factor_path.exists():
            if strict:
                raise FileNotFoundError(f"Missing feature block parquet: {block.factor_path}")
            continue
        target_columns = np.array([selected_index[column] for column in columns], dtype=np.int64)
        _fill_feature_matrix_from_block(block.factor_path, columns, row_by_id, target_columns, values)
    return SampleFeatureMatrix(selected, values)


def load_sample_feature_matrix_by_row_positions(
    feature_registry_path: str | Path,
    selected_features: Sequence[str],
    row_positions: Sequence[int],
    strict: bool = False,
) -> SampleFeatureMatrix:
    """Load selected sample-level features by parquet row positions."""
    selected = list(dict.fromkeys(str(item) for item in selected_features))
    positions = np.asarray(row_positions, dtype=np.int64)
    if not selected:
        return SampleFeatureMatrix([], np.zeros((len(positions), 0), dtype="float32"))
    if positions.size and np.any(positions[1:] < positions[:-1]):
        raise ValueError("row_positions must be sorted ascending")

    blocks = load_feature_blocks(feature_registry_path)
    feature_to_block: dict[str, FeatureBlockInfo] = {}
    for block in blocks:
        for factor in block.factors:
            feature_to_block[factor] = block

    missing = [feature for feature in selected if feature not in feature_to_block]
    if missing and strict:
        preview = ", ".join(missing[:10])
        raise KeyError(f"Selected features missing from feature registry: {preview}")

    values = np.full((len(positions), len(selected)), np.nan, dtype="float32")
    selected_index = {feature: index for index, feature in enumerate(selected)}
    loaded_columns: set[str] = set()
    for block in blocks:
        columns = [feature for feature in selected if feature in set(block.factors)]
        if not columns:
            continue
        overlap = sorted(loaded_columns & set(columns))
        if overlap:
            raise ValueError(f"Duplicate feature columns across sample blocks: {overlap}")
        loaded_columns.update(columns)
        if not block.factor_path.exists():
            if strict:
                raise FileNotFoundError(f"Missing feature block parquet: {block.factor_path}")
            continue
        target_columns = np.array([selected_index[column] for column in columns], dtype=np.int64)
        _fill_feature_matrix_from_block_positions(block.factor_path, columns, positions, target_columns, values)
    return SampleFeatureMatrix(selected, values)


def split_features_by_modality(
    features: Sequence[str],
    text_prefixes: Sequence[str],
    fundamental_prefixes: Sequence[str],
    selected_blocks: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str]]:
    feature_to_block: dict[str, str] = {}
    for block_name, block_features in (selected_blocks or {}).items():
        for feature in block_features:
            feature_to_block[str(feature)] = str(block_name)
    text: list[str] = []
    fundamental: list[str] = []
    for feature in features:
        block_name = feature_to_block.get(str(feature), "")
        if any(feature.startswith(prefix) for prefix in text_prefixes) or _is_text_feature_block(block_name):
            text.append(feature)
        elif any(feature.startswith(prefix) for prefix in fundamental_prefixes):
            fundamental.append(feature)
        else:
            fundamental.append(feature)
    return text, fundamental


def _is_text_feature_block(block_name: str) -> bool:
    return "news" in block_name.lower()


def _parse_selected_feature_file(path: Path) -> SelectedFeatureSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_features = payload.get("selected_features", [])
    features: list[str] = []
    for item in raw_features:
        if isinstance(item, str):
            features.append(item)
        elif isinstance(item, dict) and "factor_name" in item:
            features.append(str(item["factor_name"]))
        else:
            raise ValueError(f"Invalid selected feature entry in {path}: {item!r}")
    blocks = {
        str(block): [str(feature) for feature in values]
        for block, values in dict(payload.get("blocks", {})).items()
    }
    mode = "reviewed" if path.name == REVIEWED_FILE else "auto"
    return SelectedFeatureSet(list(dict.fromkeys(features)), blocks, path, mode)


def _load_manifest_factor_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Manifest must contain a list: {path}")
    names: list[str] = []
    for record in data:
        if not isinstance(record, dict) or not record.get("name"):
            raise ValueError(f"Invalid manifest record in {path}: {record!r}")
        names.append(str(record["name"]))
    return names


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _materialize_ids(sample_ids: Iterable[str] | None) -> list[str]:
    if sample_ids is None:
        return []
    return [str(item) for item in sample_ids]


def _load_sample_feature_panel_polars(
    blocks: list[FeatureBlockInfo],
    selected: list[str],
    requested_ids: list[str] | None,
) -> pd.DataFrame:
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Polars is required when data.use_polars=true. Install it with `pip install polars`.") from exc

    panel = _build_polars_feature_panel(blocks, selected, requested_ids, pl)
    if panel is None:
        return pd.DataFrame({"sample_id": _materialize_ids(requested_ids)})
    return panel.collect().to_pandas()


def _build_polars_feature_panel(blocks, selected, requested_ids, pl):
    selected_by_block = {
        block.name: [feature for feature in selected if feature in set(block.factors)]
        for block in blocks
    }
    if requested_ids is None:
        panel = None
    else:
        panel = pl.DataFrame({"sample_id": requested_ids}).lazy()

    loaded_columns: set[str] = set()
    for block in blocks:
        columns = selected_by_block[block.name]
        if not columns:
            continue
        overlap = sorted(loaded_columns & set(columns))
        if overlap:
            raise ValueError(f"Duplicate feature columns across sample blocks: {overlap}")
        loaded_columns.update(columns)
        expressions = [pl.col("sample_id").cast(pl.Utf8)]
        expressions.extend(pl.col(column).cast(pl.Float32, strict=False).alias(column) for column in columns)
        block_frame = pl.scan_parquet(str(block.factor_path)).select(expressions)
        if panel is None:
            panel = block_frame
        else:
            panel = panel.join(block_frame, on="sample_id", how="left")
    return panel


def _read_factor_columns(path: Path, columns: Sequence[str], requested_ids: set[str] | None) -> pd.DataFrame:
    read_columns = ["sample_id", *columns]
    if requested_ids is None:
        return pd.read_parquet(path, columns=read_columns)
    if not requested_ids:
        return pd.DataFrame(columns=read_columns)

    frames: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=200_000, columns=read_columns):
        frame = batch.to_pandas()
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame = frame.loc[frame["sample_id"].isin(requested_ids)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=read_columns)
    return pd.concat(frames, ignore_index=True)


def _fill_feature_matrix_from_block(
    path: Path,
    columns: Sequence[str],
    row_by_id: dict[str, int],
    target_columns: np.ndarray,
    values: np.ndarray,
) -> None:
    read_columns = ["sample_id", *columns]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=read_columns):
        frame = batch.to_pandas()
        row_positions = frame["sample_id"].astype(str).map(row_by_id)
        valid = row_positions.notna().to_numpy()
        if not valid.any():
            continue
        target_rows = row_positions.loc[valid].to_numpy(dtype=np.int64)
        block_values = frame.loc[valid, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32", copy=False)
        values[target_rows[:, None], target_columns] = block_values
        del frame, row_positions, valid, target_rows, block_values
    gc.collect()
    pa.default_memory_pool().release_unused()


def _fill_feature_matrix_from_block_positions(
    path: Path,
    columns: Sequence[str],
    source_positions: np.ndarray,
    target_columns: np.ndarray,
    values: np.ndarray,
) -> None:
    parquet = pq.ParquetFile(path)
    offset = 0
    for batch in parquet.iter_batches(batch_size=200_000, columns=list(columns)):
        batch_start = offset
        batch_end = offset + batch.num_rows
        offset = batch_end
        left = int(np.searchsorted(source_positions, batch_start, side="left"))
        right = int(np.searchsorted(source_positions, batch_end, side="left"))
        if left == right:
            continue
        local_rows = source_positions[left:right] - batch_start
        frame = batch.to_pandas()
        block_values = frame.iloc[local_rows].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32", copy=False)
        values[np.arange(left, right)[:, None], target_columns] = block_values
        del frame, block_values, local_rows
    gc.collect()
    pa.default_memory_pool().release_unused()
