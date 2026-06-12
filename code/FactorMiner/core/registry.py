from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from FactorMiner.core.factor_block import FactorBlock


@dataclass
class FactorRegistry:
    """Collection of registered factor blocks."""

    blocks: list[FactorBlock] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> FactorRegistry:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("factor_registry.json must contain a JSON list")
        return cls([FactorBlock.from_record(record) for record in data])

    def save(self, path: str | Path) -> None:
        self._validate_unique_block_names()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([block.to_record() for block in self.blocks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert(self, block: FactorBlock) -> None:
        self.blocks = [existing for existing in self.blocks if existing.name != block.name]
        self.blocks.append(block)
        self._validate_unique_block_names()

    def validate(self, base_dir: str | Path, metadata_only: bool = False) -> None:
        base_dir = Path(base_dir)
        self._validate_unique_block_names()
        seen_factors: dict[str, str] = {}
        for block in self.blocks:
            factor_path = resolve_registry_path(block.factor_path, base_dir)
            manifest_path = resolve_registry_path(block.manifest_path, base_dir)
            if not factor_path.exists():
                raise FileNotFoundError(f"Missing factor block parquet: {factor_path}")
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing factor block manifest: {manifest_path}")

            factor_names = _load_manifest_factor_names(manifest_path)
            if metadata_only:
                _validate_block_metadata(block, factor_path, factor_names)
            else:
                frame = pd.read_parquet(factor_path)
                _validate_block_frame(block, frame)
                _validate_block_manifest(block, frame.columns, factor_names)

            for factor_name in factor_names:
                if factor_name in seen_factors:
                    raise ValueError(
                        f"Duplicate factor column across blocks: {factor_name} "
                        f"appears in {seen_factors[factor_name]} and {block.name}"
                    )
                seen_factors[factor_name] = block.name

    def _validate_unique_block_names(self) -> None:
        duplicated = _duplicates(block.name for block in self.blocks)
        if duplicated:
            raise ValueError(f"FactorBlock names must be unique: {duplicated}")


def load_registry(path: str | Path) -> FactorRegistry:
    return FactorRegistry.load(path)


def save_registry(registry: FactorRegistry, path: str | Path) -> None:
    registry.save(path)


def upsert_block(registry_path: str | Path, block: FactorBlock) -> FactorRegistry:
    registry_path = Path(registry_path)
    registry = FactorRegistry.load(registry_path)
    registry.upsert(block)
    registry.save(registry_path)
    return registry


def remove_blocks(registry_path: str | Path, names: Iterable[str]) -> FactorRegistry:
    registry_path = Path(registry_path)
    registry = FactorRegistry.load(registry_path)
    remove_set = set(names)
    if not remove_set:
        return registry
    registry.blocks = [block for block in registry.blocks if block.name not in remove_set]
    registry.save(registry_path)
    return registry


def validate_registry(registry_path: str | Path, metadata_only: bool = False) -> None:
    registry_path = Path(registry_path)
    registry = FactorRegistry.load(registry_path)
    registry.validate(registry_path.parent, metadata_only=metadata_only)


def resolve_registry_path(path: str | Path, base_dir: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _validate_block_frame(block: FactorBlock, frame: pd.DataFrame) -> None:
    missing = [column for column in block.key_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing key columns in block {block.name}: {missing}")
    duplicated = frame.duplicated(list(block.key_columns))
    if duplicated.any():
        sample = frame.loc[duplicated, list(block.key_columns)].head(5).to_dict("records")
        raise ValueError(f"Factor block keys must be unique: {block.name}. Duplicate examples: {sample}")
    if block.row_count != len(frame):
        raise ValueError(f"FactorBlock.row_count mismatch for {block.name}: expected {block.row_count}, got {len(frame)}")


def _validate_block_manifest(block: FactorBlock, columns: Iterable[str], factor_names: list[str]) -> None:
    duplicated = _duplicates(factor_names)
    if duplicated:
        raise ValueError(f"Manifest factor names must be unique for {block.name}: {duplicated}")
    if block.factor_count != len(factor_names):
        raise ValueError(
            f"FactorBlock.factor_count mismatch for {block.name}: expected {block.factor_count}, got {len(factor_names)}"
        )
    available = set(columns)
    missing = [name for name in factor_names if name not in available]
    if missing:
        raise KeyError(f"Manifest factors missing from parquet for {block.name}: {missing}")


def _validate_block_metadata(block: FactorBlock, factor_path: Path, factor_names: list[str]) -> None:
    parquet = pq.ParquetFile(factor_path)
    columns = set(parquet.schema.names)
    missing_keys = [column for column in block.key_columns if column not in columns]
    if missing_keys:
        raise KeyError(f"Missing key columns in block {block.name}: {missing_keys}")
    row_count = parquet.metadata.num_rows
    if block.row_count != row_count:
        raise ValueError(f"FactorBlock.row_count mismatch for {block.name}: expected {block.row_count}, got {row_count}")
    _validate_block_manifest(block, columns, factor_names)


def _load_manifest_factor_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Factor manifest must contain a JSON list: {path}")
    factor_names: list[str] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record must be an object at index {index}: {path}")
        name = record.get("name")
        if not name:
            raise ValueError(f"Manifest record missing FactorSpec.name at index {index}: {path}")
        factor_names.append(str(name))
    return factor_names


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicated: list[str] = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    return duplicated
