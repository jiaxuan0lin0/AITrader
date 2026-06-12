from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd


DAILY_KEY_COLUMNS = ("stock_code", "trade_date")


@dataclass(frozen=True)
class FactorSpec:
    """Structured metadata for one computed factor."""

    name: str
    source: str
    category: str
    inputs: tuple[str, ...]
    expression: str
    window: int | None = None
    lookback: int = 0
    availability: str = "feature_asof_date"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FactorSpec.name cannot be empty")
        if not self.source:
            raise ValueError(f"FactorSpec.source cannot be empty: {self.name}")
        if not self.category:
            raise ValueError(f"FactorSpec.category cannot be empty: {self.name}")
        if not self.expression:
            raise ValueError(f"FactorSpec.expression cannot be empty: {self.name}")
        if self.window is not None and self.window <= 0:
            raise ValueError(f"FactorSpec.window must be positive when provided: {self.name}")
        if self.lookback < 0:
            raise ValueError(f"FactorSpec.lookback cannot be negative: {self.name}")
        object.__setattr__(self, "inputs", tuple(self.inputs))

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["inputs"] = list(self.inputs)
        return record


@dataclass
class FactorResult:
    """Computed factor table plus the metadata that defines each factor column."""

    factors: pd.DataFrame
    specs: list[FactorSpec] = field(default_factory=list)
    key_columns: tuple[str, ...] = DAILY_KEY_COLUMNS

    def factor_names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def max_lookback(self) -> int:
        return max((spec.lookback for spec in self.specs), default=0)

    def manifest_records(self) -> list[dict[str, Any]]:
        return [spec.to_record() for spec in self.specs]

    def validate(self) -> None:
        self._validate_key_columns()
        self._validate_unique_keys()
        self._validate_unique_factor_names()
        self._validate_factor_columns_exist()

    def select_output_columns(self) -> pd.DataFrame:
        self.validate()
        columns = [*self.key_columns, *self.factor_names()]
        return self.factors.loc[:, columns].copy()

    def _validate_key_columns(self) -> None:
        missing = [column for column in self.key_columns if column not in self.factors.columns]
        if missing:
            raise KeyError(f"Missing factor key columns: {missing}")

    def _validate_unique_keys(self) -> None:
        duplicated = self.factors.duplicated(list(self.key_columns))
        if duplicated.any():
            sample = self.factors.loc[duplicated, list(self.key_columns)].head(5).to_dict("records")
            raise ValueError(f"Factor keys must be unique. Duplicate examples: {sample}")

    def _validate_unique_factor_names(self) -> None:
        duplicated = _duplicates(self.factor_names())
        if duplicated:
            raise ValueError(f"Factor names must be unique: {duplicated}")

    def _validate_factor_columns_exist(self) -> None:
        missing = [name for name in self.factor_names() if name not in self.factors.columns]
        if missing:
            raise KeyError(f"Missing factor columns: {missing}")


def combine_factor_results(results: Iterable[FactorResult], key_columns: tuple[str, ...] = DAILY_KEY_COLUMNS) -> FactorResult:
    """Merge factor results on daily keys and combine their specs."""
    results = list(results)
    if not results:
        return FactorResult(pd.DataFrame(columns=list(key_columns)), [], key_columns)

    combined: pd.DataFrame | None = None
    specs: list[FactorSpec] = []
    for result in results:
        result.validate()
        if result.key_columns != key_columns:
            raise ValueError(f"Unexpected key columns: {result.key_columns}")
        selected = result.select_output_columns()
        if combined is None:
            combined = selected
        else:
            overlapping = (set(combined.columns) & set(selected.columns)) - set(key_columns)
            if overlapping:
                raise ValueError(f"Duplicate factor columns across results: {sorted(overlapping)}")
            combined = combined.merge(selected, on=list(key_columns), how="outer")
        specs.extend(result.specs)

    merged = FactorResult(combined if combined is not None else pd.DataFrame(columns=list(key_columns)), specs, key_columns)
    merged.validate()
    return merged


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicated: list[str] = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    return duplicated
