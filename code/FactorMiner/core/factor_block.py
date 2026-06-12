from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

from FactorMiner.core.factor_spec import DAILY_KEY_COLUMNS, FactorResult


SAMPLE_KEY_COLUMNS = ("sample_id",)
GRANULARITY_KEY_COLUMNS = {
    "daily": DAILY_KEY_COLUMNS,
    "sample": SAMPLE_KEY_COLUMNS,
}


@dataclass(frozen=True)
class FactorBlock:
    """Metadata for one materialized factor block."""

    name: str
    granularity: str
    key_columns: tuple[str, ...]
    factor_path: str
    manifest_path: str
    factor_count: int
    row_count: int
    created_at: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FactorBlock.name cannot be empty")
        expected_keys = key_columns_for_granularity(self.granularity)
        object.__setattr__(self, "key_columns", tuple(self.key_columns))
        if self.key_columns != expected_keys:
            raise ValueError(
                f"FactorBlock key_columns must be {expected_keys} for granularity={self.granularity}"
            )
        if not self.factor_path:
            raise ValueError(f"FactorBlock.factor_path cannot be empty: {self.name}")
        if not self.manifest_path:
            raise ValueError(f"FactorBlock.manifest_path cannot be empty: {self.name}")
        if self.factor_count < 0:
            raise ValueError(f"FactorBlock.factor_count cannot be negative: {self.name}")
        if self.row_count < 0:
            raise ValueError(f"FactorBlock.row_count cannot be negative: {self.name}")

    @classmethod
    def from_result(
        cls,
        name: str,
        granularity: str,
        result: FactorResult,
        factor_path: str | Path,
        manifest_path: str | Path,
        description: str = "",
        created_at: str | None = None,
    ) -> FactorBlock:
        result.validate()
        expected_keys = key_columns_for_granularity(granularity)
        if result.key_columns != expected_keys:
            raise ValueError(f"FactorResult key_columns must be {expected_keys} for granularity={granularity}")
        return cls(
            name=name,
            granularity=granularity,
            key_columns=expected_keys,
            factor_path=str(factor_path),
            manifest_path=str(manifest_path),
            factor_count=len(result.specs),
            row_count=len(result.factors),
            created_at=created_at or utc_now_iso(),
            description=description,
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> FactorBlock:
        data = dict(record)
        data["key_columns"] = tuple(data.get("key_columns", ()))
        return cls(**data)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["key_columns"] = list(self.key_columns)
        return record


def write_factor_block(
    result: FactorResult,
    name: str,
    granularity: str,
    factor_path: str | Path,
    manifest_path: str | Path,
    description: str = "",
) -> FactorBlock:
    """Write a FactorResult to parquet and manifest JSON, then return block metadata."""
    block = FactorBlock.from_result(
        name=name,
        granularity=granularity,
        result=result,
        factor_path=factor_path,
        manifest_path=manifest_path,
        description=description,
    )
    factor_path = Path(factor_path)
    manifest_path = Path(manifest_path)
    factor_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(result.select_output_columns(), factor_path)
    _write_text_atomic(json.dumps(result.manifest_records(), ensure_ascii=False, indent=2), manifest_path)
    return block


def _write_parquet_atomic(frame, path: Path) -> None:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        frame.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise


def _write_text_atomic(text: str, path: Path) -> None:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
        tmp_path.replace(path)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise


def key_columns_for_granularity(granularity: str) -> tuple[str, ...]:
    try:
        return GRANULARITY_KEY_COLUMNS[granularity]
    except KeyError as exc:
        raise ValueError(f"Unsupported factor granularity: {granularity}") from exc


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
