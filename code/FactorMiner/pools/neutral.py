from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from FactorMiner import operators as op
from FactorMiner.core.factor_spec import FactorSpec


@dataclass(frozen=True)
class NeutralConfig:
    enabled: bool = True
    add_cs_pct: bool = True
    add_cs_z: bool = True
    add_industry: bool = True
    factor_names: tuple[str, ...] | None = None
    industry_col: str = "industry"
    date_col: str = "trade_date"


def append_neutral_factors(
    factors: pd.DataFrame,
    specs: list[FactorSpec],
    config: NeutralConfig | None,
) -> tuple[pd.DataFrame, list[FactorSpec]]:
    """Append cross-sectional and industry-neutral derived factors."""
    if config is None or not config.enabled:
        return factors, specs

    result = factors.copy()
    derived_columns: dict[str, pd.Series] = {}
    updated_specs = list(specs)
    spec_by_name = {spec.name: spec for spec in specs}
    base_names = list(config.factor_names) if config.factor_names is not None else [spec.name for spec in specs]

    if config.date_col not in result.columns:
        raise KeyError(f"Missing date column for neutral factors: {config.date_col}")
    if config.add_industry and config.industry_col not in result.columns:
        raise KeyError(f"Missing industry column for neutral factors: {config.industry_col}")

    for name in base_names:
        if name not in spec_by_name:
            raise KeyError(f"Cannot neutralize unknown factor: {name}")
        if name not in result.columns:
            raise KeyError(f"Cannot neutralize missing factor column: {name}")
        base_spec = spec_by_name[name]

        if config.add_cs_pct:
            derived_name = f"{name}_cs_pct"
            derived_columns[derived_name] = op.cs_pct_rank(result, name, date_col=config.date_col)
            updated_specs.append(_derive_spec(base_spec, derived_name, "cs_pct", f"cs_pct_rank({name})"))

        if config.add_cs_z:
            derived_name = f"{name}_cs_z"
            derived_columns[derived_name] = op.cs_zscore(result, name, date_col=config.date_col)
            updated_specs.append(_derive_spec(base_spec, derived_name, "cs_z", f"cs_zscore({name})"))

        if config.add_industry:
            derived_name = f"{name}_ind_neu"
            derived_columns[derived_name] = op.industry_neutralize(
                result,
                name,
                industry_col=config.industry_col,
                date_col=config.date_col,
            )
            updated_specs.append(
                _derive_spec(
                    base_spec,
                    derived_name,
                    "ind_neu",
                    f"industry_neutralize({name}, {config.industry_col})",
                    extra_inputs=(config.industry_col,),
                )
            )

    if derived_columns:
        result = pd.concat([result, pd.DataFrame(derived_columns, index=result.index)], axis=1)
    return result, updated_specs


def _derive_spec(
    base_spec: FactorSpec,
    name: str,
    suffix: str,
    expression: str,
    extra_inputs: Iterable[str] = (),
) -> FactorSpec:
    inputs = (*base_spec.inputs, *tuple(extra_inputs))
    return FactorSpec(
        name=name,
        source=base_spec.source,
        category=f"{base_spec.category}.{suffix}",
        inputs=inputs,
        expression=expression,
        window=base_spec.window,
        lookback=base_spec.lookback,
        availability=base_spec.availability,
        description=f"{suffix} derived from {base_spec.name}",
    )
