from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import pandas as pd

from FactorMiner import operators as op
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.pools.neutral import NeutralConfig, append_neutral_factors


@dataclass(frozen=True)
class MetricConfig:
    prefix: str = "metric_"
    raw_fields: tuple[str, ...] = (
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "total_mv",
        "circ_mv",
    )
    missing_flag_fields: tuple[str, ...] = ("pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv")
    delta_windows: tuple[int, ...] = (1, 3, 5, 10)
    rolling_windows: tuple[int, ...] = (3, 5, 10, 20)
    delta_fields: tuple[str, ...] = ("pe_ttm", "pb", "turnover_rate", "volume_ratio")
    rolling_mean_fields: tuple[str, ...] = ("turnover_rate", "turnover_rate_f", "volume_ratio", "pe_ttm", "pb")
    neutral: NeutralConfig | None = field(default_factory=NeutralConfig)
    max_neutral_missing_rate: float | None = 0.25


def build_metric_factors(
    metric: pd.DataFrame,
    industry: pd.DataFrame | None = None,
    config: MetricConfig | None = None,
) -> FactorResult:
    """Build valuation, market-cap, and turnover factors."""
    config = config or MetricConfig()
    if config.max_neutral_missing_rate is not None and not 0 <= config.max_neutral_missing_rate <= 1:
        raise ValueError("MetricConfig.max_neutral_missing_rate must be between 0 and 1")
    _require_columns(metric, ("stock_code", "trade_date"))

    work = _attach_industry(metric, industry)
    factors = work[["stock_code", "trade_date"]].copy()
    if "industry" in work.columns:
        factors["industry"] = work["industry"]

    specs: list[FactorSpec] = []
    factor_columns: dict[str, pd.Series] = {}
    neutral_names: list[str] = []

    for field_name in config.raw_fields:
        if field_name not in work.columns:
            continue
        values = pd.to_numeric(work[field_name], errors="coerce")
        name = f"{config.prefix}{field_name}"
        _add_factor(factor_columns, specs, name, values, "metric.raw", (field_name,), field_name)
        _append_neutral_name(neutral_names, name, values, config)

    for field_name in config.missing_flag_fields:
        if field_name not in work.columns:
            continue
        name = f"{config.prefix}{field_name}_missing"
        values = work[field_name].isna().astype("float64")
        _add_factor(factor_columns, specs, name, values, "metric.missing", (field_name,), f"isna({field_name})")

    derived_definitions = [
        ("earnings_yield", "pe_ttm", op.safe_div(1.0, work["pe_ttm"]) if "pe_ttm" in work.columns else None, "1 / pe_ttm"),
        ("book_to_price", "pb", op.safe_div(1.0, work["pb"]) if "pb" in work.columns else None, "1 / pb"),
        ("sales_to_price", "ps_ttm", op.safe_div(1.0, work["ps_ttm"]) if "ps_ttm" in work.columns else None, "1 / ps_ttm"),
        ("log_total_mv", "total_mv", op.safe_log(work["total_mv"]) if "total_mv" in work.columns else None, "log(total_mv)"),
        ("log_circ_mv", "circ_mv", op.safe_log(work["circ_mv"]) if "circ_mv" in work.columns else None, "log(circ_mv)"),
    ]
    for suffix, input_field, values, expression in derived_definitions:
        if values is None:
            continue
        name = f"{config.prefix}{suffix}"
        _add_factor(factor_columns, specs, name, values, "metric.derived", (input_field,), expression)
        _append_neutral_name(neutral_names, name, values, config)

    for field_name in config.delta_fields:
        if field_name not in work.columns:
            continue
        work[field_name] = pd.to_numeric(work[field_name], errors="coerce")
        for window in config.delta_windows:
            name = f"{config.prefix}{field_name}_delta{window}"
            values = op.delta(work, field_name, window)
            _add_factor(
                factor_columns,
                specs,
                name,
                values,
                "metric.delta",
                (field_name,),
                f"delta({field_name}, {window})",
                window=window,
                lookback=window,
            )
            _append_neutral_name(neutral_names, name, values, config)

    for field_name in config.rolling_mean_fields:
        if field_name not in work.columns:
            continue
        work[field_name] = pd.to_numeric(work[field_name], errors="coerce")
        for window in config.rolling_windows:
            name = f"{config.prefix}{field_name}_ma{window}"
            values = op.ts_mean(work, field_name, window)
            _add_factor(
                factor_columns,
                specs,
                name,
                values,
                "metric.rolling_mean",
                (field_name,),
                f"ts_mean({field_name}, {window})",
                window=window,
                lookback=window,
            )
            _append_neutral_name(neutral_names, name, values, config)

    factors = pd.concat([factors, pd.DataFrame(factor_columns, index=work.index)], axis=1)
    factors, specs = append_neutral_factors(factors, specs, _neutral_config(config.neutral, neutral_names))
    result = FactorResult(factors=factors, specs=specs)
    result.validate()
    return result


def _add_factor(
    factor_columns: dict[str, pd.Series],
    specs: list[FactorSpec],
    name: str,
    values: pd.Series,
    category: str,
    inputs: Iterable[str],
    expression: str,
    window: int | None = None,
    lookback: int = 0,
) -> None:
    factor_columns[name] = op.replace_inf(values)
    specs.append(
        FactorSpec(
            name=name,
            source="metric",
            category=category,
            inputs=tuple(inputs),
            expression=expression,
            window=window,
            lookback=lookback,
            availability="feature_asof_date",
        )
    )


def _attach_industry(metric: pd.DataFrame, industry: pd.DataFrame | None) -> pd.DataFrame:
    work = metric.copy()
    if "industry" in work.columns or industry is None:
        return work
    _require_columns(industry, ("stock_code", "trade_date", "industry"))
    industry_frame = industry[["stock_code", "trade_date", "industry"]].drop_duplicates(["stock_code", "trade_date"])
    return work.merge(industry_frame, on=["stock_code", "trade_date"], how="left")


def _neutral_config(config: NeutralConfig | None, factor_names: list[str]) -> NeutralConfig | None:
    if config is None or not config.enabled or config.factor_names is not None:
        return config
    return replace(config, factor_names=tuple(factor_names))


def _append_neutral_name(neutral_names: list[str], name: str, values: pd.Series, config: MetricConfig) -> None:
    if _eligible_for_neutral(values, config.max_neutral_missing_rate):
        neutral_names.append(name)


def _eligible_for_neutral(values: pd.Series, max_missing_rate: float | None) -> bool:
    if max_missing_rate is None:
        return True
    missing_rate = pd.to_numeric(values, errors="coerce").isna().mean()
    return bool(missing_rate <= max_missing_rate)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing metric input columns: {missing}")
