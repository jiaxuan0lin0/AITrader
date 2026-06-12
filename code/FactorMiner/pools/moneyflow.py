from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
import pandas as pd

from FactorMiner import operators as op
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.pools.neutral import NeutralConfig, append_neutral_factors


AMOUNT_UNIT_MULTIPLIER = 10.0


@dataclass(frozen=True)
class MoneyflowConfig:
    prefix: str = "mf_"
    rolling_windows: tuple[int, ...] = (3, 5, 10, 20)
    corr_windows: tuple[int, ...] = (3, 5, 10, 20)
    positive_windows: tuple[int, ...] = (3, 5, 10)
    neutral: NeutralConfig | None = field(default_factory=NeutralConfig)


def build_moneyflow_factors(
    moneyflow: pd.DataFrame,
    price: pd.DataFrame,
    config: MoneyflowConfig | None = None,
) -> FactorResult:
    """Build moneyflow ratio, momentum, and price-flow confirmation factors."""
    config = config or MoneyflowConfig()
    _require_columns(moneyflow, ("stock_code", "trade_date", *(_moneyflow_amount_columns())))
    _require_columns(price, ("stock_code", "trade_date", "amount", "close"))

    work = _attach_price(moneyflow, price)
    factors = work[["stock_code", "trade_date"]].copy()
    if "industry" in work.columns:
        factors["industry"] = work["industry"]

    for column in _moneyflow_amount_columns() + ("amount", "close"):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    specs: list[FactorSpec] = []
    factor_columns: dict[str, pd.Series] = {}
    neutral_names: list[str] = []

    main_net = work["buy_lg_amount"] + work["buy_elg_amount"] - work["sell_lg_amount"] - work["sell_elg_amount"]
    large_net = work["buy_lg_amount"] - work["sell_lg_amount"]
    elg_net = work["buy_elg_amount"] - work["sell_elg_amount"]
    large_amount = work["buy_lg_amount"] + work["sell_lg_amount"]
    elg_amount = work["buy_elg_amount"] + work["sell_elg_amount"]
    small_pressure = work["sell_sm_amount"] - work["buy_sm_amount"]

    base_definitions = [
        ("net_amount_ratio", work["net_mf_amount"], "net_mf_amount * 10 / amount"),
        ("main_net_amount_ratio", main_net, "(buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount) * 10 / amount"),
        ("lg_net_amount_ratio", large_net, "(buy_lg_amount - sell_lg_amount) * 10 / amount"),
        ("elg_net_amount_ratio", elg_net, "(buy_elg_amount - sell_elg_amount) * 10 / amount"),
        ("lg_amount_ratio", large_amount, "(buy_lg_amount + sell_lg_amount) * 10 / amount"),
        ("elg_amount_ratio", elg_amount, "(buy_elg_amount + sell_elg_amount) * 10 / amount"),
        ("small_order_pressure", small_pressure, "(sell_sm_amount - buy_sm_amount) * 10 / amount"),
    ]
    for suffix, amount_values, expression in base_definitions:
        name = f"{config.prefix}{suffix}"
        values = op.safe_div(amount_values * AMOUNT_UNIT_MULTIPLIER, work["amount"])
        _add_factor(factor_columns, specs, name, values, "moneyflow.ratio", _inputs_for_moneyflow_expression(expression), expression)
        neutral_names.append(name)

    work = pd.concat([work, pd.DataFrame({name: factor_columns[name] for name in neutral_names}, index=work.index)], axis=1)
    close_delay_1 = op.delay(work, "close", 1)
    work["_mf_price_return_1"] = op.safe_div(work["close"], close_delay_1) - 1
    work["_mf_main_positive"] = (work[f"{config.prefix}main_net_amount_ratio"] > 0).astype("float64")
    work["_mf_price_flow_confirm"] = (
        (work["_mf_price_return_1"] > 0) & (work[f"{config.prefix}main_net_amount_ratio"] > 0)
    ).astype("float64")
    missing_flow = work[f"{config.prefix}main_net_amount_ratio"].isna()
    missing_price_or_flow = work["_mf_price_return_1"].isna() | missing_flow
    work.loc[missing_flow, "_mf_main_positive"] = np.nan
    work.loc[missing_price_or_flow, "_mf_price_flow_confirm"] = np.nan

    rolling_base_names = (
        f"{config.prefix}net_amount_ratio",
        f"{config.prefix}main_net_amount_ratio",
        f"{config.prefix}small_order_pressure",
    )
    for base_name in rolling_base_names:
        for window in config.rolling_windows:
            ma_name = f"{base_name}_ma{window}"
            ma_values = op.ts_mean(work, base_name, window)
            _add_factor(
                factor_columns,
                specs,
                ma_name,
                ma_values,
                "moneyflow.rolling_mean",
                (base_name,),
                f"ts_mean({base_name}, {window})",
                window=window,
                lookback=window,
            )
            neutral_names.append(ma_name)

            slope_name = f"{base_name}_slope{window}"
            slope_values = op.ts_slope(work, base_name, window)
            _add_factor(
                factor_columns,
                specs,
                slope_name,
                slope_values,
                "moneyflow.slope",
                (base_name,),
                f"ts_slope({base_name}, {window})",
                window=window,
                lookback=window,
            )
            neutral_names.append(slope_name)

    for window in config.positive_windows:
        name = f"{config.prefix}main_positive_days{window}"
        values = op.ts_sum(work, "_mf_main_positive", window)
        _add_factor(
            factor_columns,
            specs,
            name,
            values,
            "moneyflow.count",
            (f"{config.prefix}main_net_amount_ratio",),
            f"ts_sum(main_net_amount_ratio > 0, {window})",
            window=window,
            lookback=window,
        )
        neutral_names.append(name)

        confirm_name = f"{config.prefix}price_flow_confirm{window}"
        confirm_values = op.ts_mean(work, "_mf_price_flow_confirm", window)
        _add_factor(
            factor_columns,
            specs,
            confirm_name,
            confirm_values,
            "moneyflow.confirm",
            (f"{config.prefix}main_net_amount_ratio", "close"),
            f"ts_mean(price_return_1 > 0 and main_net_amount_ratio > 0, {window})",
            window=window,
            lookback=window + 1,
        )
        neutral_names.append(confirm_name)

    for window in config.corr_windows:
        name = f"{config.prefix}price_flow_corr{window}"
        values = op.ts_corr(work, "_mf_price_return_1", f"{config.prefix}main_net_amount_ratio", window)
        _add_factor(
            factor_columns,
            specs,
            name,
            values,
            "moneyflow.corr",
            ("close", f"{config.prefix}main_net_amount_ratio"),
            f"ts_corr(price_return_1, main_net_amount_ratio, {window})",
            window=window,
            lookback=window + 1,
        )
        neutral_names.append(name)

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
            source="moneyflow",
            category=category,
            inputs=tuple(inputs),
            expression=expression,
            window=window,
            lookback=lookback,
            availability="feature_asof_date",
        )
    )


def _attach_price(moneyflow: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    price_columns = ["stock_code", "trade_date", "amount", "close"]
    price_frame = price[price_columns].drop_duplicates(["stock_code", "trade_date"])
    if "industry" in price.columns:
        price_industry = price[["stock_code", "trade_date", "industry"]].drop_duplicates(["stock_code", "trade_date"])
        price_frame = price_frame.merge(price_industry, on=["stock_code", "trade_date"], how="left")
        price_frame = price_frame.rename(columns={"industry": "industry_price"})

    work = moneyflow.copy().merge(price_frame, on=["stock_code", "trade_date"], how="left")
    if "industry_price" not in work.columns:
        return work
    if "industry" in work.columns:
        work["industry"] = work["industry"].where(work["industry"].notna(), work["industry_price"])
    else:
        work["industry"] = work["industry_price"]
    return work.drop(columns=["industry_price"])


def _moneyflow_amount_columns() -> tuple[str, ...]:
    return (
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    )


def _inputs_for_moneyflow_expression(expression: str) -> tuple[str, ...]:
    return tuple(column for column in _moneyflow_amount_columns() if column in expression) + ("amount",)


def _neutral_config(config: NeutralConfig | None, factor_names: list[str]) -> NeutralConfig | None:
    if config is None or not config.enabled or config.factor_names is not None:
        return config
    return replace(config, factor_names=tuple(factor_names))


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing moneyflow input columns: {missing}")
