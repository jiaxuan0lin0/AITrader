from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from FactorMiner import operators as op
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.pools.neutral import NeutralConfig, append_neutral_factors


EPS = 1e-12


@dataclass(frozen=True)
class Alpha158Config:
    prefix: str = "alpha158_"
    include_kbar: bool = True
    include_price: bool = True
    include_return: bool = True
    include_rolling: bool = True
    return_windows: tuple[int, ...] = (1, 3, 5, 10, 20, 60)
    rolling_windows: tuple[int, ...] = (3, 5, 10, 20, 60)
    price_windows: tuple[int, ...] = (0, 1, 2, 3)
    price_fields: tuple[str, ...] = ("open", "high", "low", "vwap")
    neutral: NeutralConfig | None = field(default_factory=NeutralConfig)


def build_alpha158_factors(price: pd.DataFrame, config: Alpha158Config | None = None) -> FactorResult:
    """Build an Alpha158-style price-volume factor pool with competition windows."""
    config = config or Alpha158Config()
    _require_columns(price, ("stock_code", "trade_date", "open", "high", "low", "close", "vwap", "volume"))

    work = price.copy()
    for column in ("open", "high", "low", "close", "vwap", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    factors = work[["stock_code", "trade_date"]].copy()
    if "industry" in work.columns:
        factors["industry"] = work["industry"]
    specs: list[FactorSpec] = []
    factor_columns: dict[str, pd.Series] = {}

    if config.include_kbar:
        _add_kbar_factors(work, factor_columns, specs, config)
    if config.include_price:
        _add_price_factors(work, factor_columns, specs, config)
    if config.include_return or config.include_rolling:
        _add_rolling_factors(work, factor_columns, specs, config)

    factors = pd.concat([factors, pd.DataFrame(factor_columns, index=work.index)], axis=1)
    factors, specs = append_neutral_factors(factors, specs, config.neutral)
    result = FactorResult(factors=factors, specs=specs)
    result.validate()
    return result


def _add_kbar_factors(work: pd.DataFrame, factor_columns: dict[str, pd.Series], specs: list[FactorSpec], config: Alpha158Config) -> None:
    open_ = work["open"]
    high = work["high"]
    low = work["low"]
    close = work["close"]
    high_low = high - low
    max_open_close = pd.Series(np.maximum(open_, close), index=work.index)
    min_open_close = pd.Series(np.minimum(open_, close), index=work.index)

    definitions = [
        ("KMID", op.safe_div(close - open_, open_), ("open", "close"), "(close - open) / open"),
        ("KLEN", op.safe_div(high - low, open_), ("open", "high", "low"), "(high - low) / open"),
        ("KMID2", op.safe_div(close - open_, high_low + EPS), ("open", "high", "low", "close"), "(close - open) / (high - low + eps)"),
        ("KUP", op.safe_div(high - max_open_close, open_), ("open", "high", "close"), "(high - max(open, close)) / open"),
        ("KUP2", op.safe_div(high - max_open_close, high_low + EPS), ("open", "high", "low", "close"), "(high - max(open, close)) / (high - low + eps)"),
        ("KLOW", op.safe_div(min_open_close - low, open_), ("open", "low", "close"), "(min(open, close) - low) / open"),
        ("KLOW2", op.safe_div(min_open_close - low, high_low + EPS), ("open", "high", "low", "close"), "(min(open, close) - low) / (high - low + eps)"),
        ("KSFT", op.safe_div(2 * close - high - low, open_), ("open", "high", "low", "close"), "(2 * close - high - low) / open"),
        ("KSFT2", op.safe_div(2 * close - high - low, high_low + EPS), ("open", "high", "low", "close"), "(2 * close - high - low) / (high - low + eps)"),
    ]
    for raw_name, values, inputs, expression in definitions:
        _add_factor(factor_columns, specs, config, raw_name, values, "kbar", inputs, expression)


def _add_price_factors(work: pd.DataFrame, factor_columns: dict[str, pd.Series], specs: list[FactorSpec], config: Alpha158Config) -> None:
    for field_name in config.price_fields:
        if field_name not in work.columns:
            raise KeyError(f"Missing Alpha158 price field: {field_name}")
        for window in config.price_windows:
            raw_name = f"{field_name.upper()}{window}"
            if window == 0:
                values = op.safe_div(work[field_name], work["close"])
                expression = f"{field_name} / close"
            else:
                values = op.safe_div(op.delay(work, field_name, window), work["close"])
                expression = f"delay({field_name}, {window}) / close"
            _add_factor(
                factor_columns,
                specs,
                config,
                raw_name,
                values,
                "price",
                (field_name, "close"),
                expression,
                window=window if window > 0 else None,
                lookback=window,
            )


def _add_rolling_factors(work: pd.DataFrame, factor_columns: dict[str, pd.Series], specs: list[FactorSpec], config: Alpha158Config) -> None:
    close = work["close"]
    high = work["high"]
    low = work["low"]
    volume = work["volume"]
    close_delay_1 = op.delay(work, "close", 1)
    volume_delay_1 = op.delay(work, "volume", 1)

    work = work.copy()
    work["_alpha158_log_volume"] = op.safe_log(volume + 1)
    work["_alpha158_close_ratio"] = op.safe_div(close, close_delay_1)
    work["_alpha158_log_volume_ratio"] = op.safe_log(op.safe_div(volume, volume_delay_1) + 1)
    work["_alpha158_close_delta"] = close - close_delay_1
    work["_alpha158_abs_close_delta"] = work["_alpha158_close_delta"].abs()
    work["_alpha158_up_delta"] = work["_alpha158_close_delta"].clip(lower=0)
    work["_alpha158_down_delta"] = (-work["_alpha158_close_delta"]).clip(lower=0)
    work["_alpha158_up_day"] = (close > close_delay_1).astype("float64")
    work["_alpha158_down_day"] = (close < close_delay_1).astype("float64")
    work["_alpha158_volume_delta"] = volume - volume_delay_1
    work["_alpha158_abs_volume_delta"] = work["_alpha158_volume_delta"].abs()
    work["_alpha158_volume_up_delta"] = work["_alpha158_volume_delta"].clip(lower=0)
    work["_alpha158_volume_down_delta"] = (-work["_alpha158_volume_delta"]).clip(lower=0)
    work["_alpha158_abs_ret_volume"] = (op.safe_div(close, close_delay_1) - 1).abs() * volume

    missing_delay = close_delay_1.isna()
    for column in (
        "_alpha158_close_ratio",
        "_alpha158_log_volume_ratio",
        "_alpha158_close_delta",
        "_alpha158_abs_close_delta",
        "_alpha158_up_delta",
        "_alpha158_down_delta",
        "_alpha158_up_day",
        "_alpha158_down_day",
        "_alpha158_abs_ret_volume",
    ):
        work.loc[missing_delay, column] = np.nan
    volume_missing_delay = volume_delay_1.isna()
    for column in (
        "_alpha158_volume_delta",
        "_alpha158_abs_volume_delta",
        "_alpha158_volume_up_delta",
        "_alpha158_volume_down_delta",
    ):
        work.loc[volume_missing_delay, column] = np.nan

    if config.include_return:
        for window in config.return_windows:
            _add_factor(
                factor_columns,
                specs,
                config,
                f"ROC{window}",
                op.safe_div(op.delay(work, "close", window), close),
                "rolling.roc",
                ("close",),
                f"delay(close, {window}) / close",
                window=window,
                lookback=window,
            )

    if not config.include_rolling:
        return

    for window in config.rolling_windows:
        slope, rsquare, residual = op.ts_linear_regression(work, "close", window)
        high_max = op.ts_max(work, "high", window)
        low_min = op.ts_min(work, "low", window)
        up_sum = op.ts_sum(work, "_alpha158_up_delta", window)
        down_sum = op.ts_sum(work, "_alpha158_down_delta", window)
        abs_sum = op.ts_sum(work, "_alpha158_abs_close_delta", window)
        volume_up_sum = op.ts_sum(work, "_alpha158_volume_up_delta", window)
        volume_down_sum = op.ts_sum(work, "_alpha158_volume_down_delta", window)
        volume_abs_sum = op.ts_sum(work, "_alpha158_abs_volume_delta", window)
        abs_ret_volume_std = op.ts_std(work, "_alpha158_abs_ret_volume", window)
        abs_ret_volume_mean = op.ts_mean(work, "_alpha158_abs_ret_volume", window)
        idxmax = op.ts_idxmax(work, "high", window)
        idxmin = op.ts_idxmin(work, "low", window)

        rolling_definitions = [
            ("MA", op.safe_div(op.ts_mean(work, "close", window), close), "rolling.ma", ("close",), f"ts_mean(close, {window}) / close", window),
            ("STD", op.safe_div(op.ts_std(work, "close", window), close), "rolling.std", ("close",), f"ts_std(close, {window}) / close", window),
            ("BETA", op.safe_div(slope, close), "rolling.beta", ("close",), f"ts_slope(close, {window}) / close", window),
            ("RSQR", rsquare, "rolling.rsqr", ("close",), f"ts_rsquare(close, {window})", window),
            ("RESI", op.safe_div(residual, close), "rolling.resi", ("close",), f"ts_residual(close, {window}) / close", window),
            ("MAX", op.safe_div(high_max, close), "rolling.max", ("high", "close"), f"ts_max(high, {window}) / close", window),
            ("MIN", op.safe_div(low_min, close), "rolling.min", ("low", "close"), f"ts_min(low, {window}) / close", window),
            ("QTLU", op.safe_div(op.ts_quantile(work, "close", window, 0.8), close), "rolling.quantile", ("close",), f"ts_quantile(close, {window}, 0.8) / close", window),
            ("QTLD", op.safe_div(op.ts_quantile(work, "close", window, 0.2), close), "rolling.quantile", ("close",), f"ts_quantile(close, {window}, 0.2) / close", window),
            ("RANK", op.ts_rank(work, "close", window), "rolling.rank", ("close",), f"ts_rank(close, {window})", window),
            ("RSV", op.safe_div(close - low_min, high_max - low_min + EPS), "rolling.rsv", ("close", "high", "low"), f"(close - ts_min(low, {window})) / (ts_max(high, {window}) - ts_min(low, {window}) + eps)", window),
            ("IMAX", op.safe_div(idxmax, window), "rolling.idx", ("high",), f"ts_idxmax(high, {window}) / {window}", window),
            ("IMIN", op.safe_div(idxmin, window), "rolling.idx", ("low",), f"ts_idxmin(low, {window}) / {window}", window),
            ("IMXD", op.safe_div(idxmax - idxmin, window), "rolling.idx", ("high", "low"), f"(ts_idxmax(high, {window}) - ts_idxmin(low, {window})) / {window}", window),
            ("CORR", op.ts_corr(work, "close", "_alpha158_log_volume", window), "rolling.corr", ("close", "volume"), f"ts_corr(close, log(volume + 1), {window})", window),
            ("CORD", op.ts_corr(work, "_alpha158_close_ratio", "_alpha158_log_volume_ratio", window), "rolling.corr", ("close", "volume"), f"ts_corr(close / delay(close, 1), log(volume / delay(volume, 1) + 1), {window})", window + 1),
            ("CNTP", op.ts_mean(work, "_alpha158_up_day", window), "rolling.count", ("close",), f"ts_mean(close > delay(close, 1), {window})", window + 1),
            ("CNTN", op.ts_mean(work, "_alpha158_down_day", window), "rolling.count", ("close",), f"ts_mean(close < delay(close, 1), {window})", window + 1),
            ("CNTD", op.ts_mean(work, "_alpha158_up_day", window) - op.ts_mean(work, "_alpha158_down_day", window), "rolling.count", ("close",), f"CNTP{window} - CNTN{window}", window + 1),
            ("SUMP", op.safe_div(up_sum, abs_sum + EPS), "rolling.sum", ("close",), f"sum(max(close - delay(close, 1), 0), {window}) / sum(abs(close - delay(close, 1)), {window})", window + 1),
            ("SUMN", op.safe_div(down_sum, abs_sum + EPS), "rolling.sum", ("close",), f"sum(max(delay(close, 1) - close, 0), {window}) / sum(abs(close - delay(close, 1)), {window})", window + 1),
            ("SUMD", op.safe_div(up_sum - down_sum, abs_sum + EPS), "rolling.sum", ("close",), f"(up_sum{window} - down_sum{window}) / abs_sum{window}", window + 1),
            ("VMA", op.safe_div(op.ts_mean(work, "volume", window), volume + EPS), "rolling.volume", ("volume",), f"ts_mean(volume, {window}) / volume", window),
            ("VSTD", op.safe_div(op.ts_std(work, "volume", window), volume + EPS), "rolling.volume", ("volume",), f"ts_std(volume, {window}) / volume", window),
            ("WVMA", op.safe_div(abs_ret_volume_std, abs_ret_volume_mean + EPS), "rolling.volume", ("close", "volume"), f"ts_std(abs(close / delay(close, 1) - 1) * volume, {window}) / ts_mean(..., {window})", window + 1),
            ("VSUMP", op.safe_div(volume_up_sum, volume_abs_sum + EPS), "rolling.volume", ("volume",), f"sum(max(volume - delay(volume, 1), 0), {window}) / sum(abs(volume - delay(volume, 1)), {window})", window + 1),
            ("VSUMN", op.safe_div(volume_down_sum, volume_abs_sum + EPS), "rolling.volume", ("volume",), f"sum(max(delay(volume, 1) - volume, 0), {window}) / sum(abs(volume - delay(volume, 1)), {window})", window + 1),
            ("VSUMD", op.safe_div(volume_up_sum - volume_down_sum, volume_abs_sum + EPS), "rolling.volume", ("volume",), f"(volume_up_sum{window} - volume_down_sum{window}) / volume_abs_sum{window}", window + 1),
        ]

        for raw_prefix, values, category, inputs, expression, lookback in rolling_definitions:
            _add_factor(
                factor_columns,
                specs,
                config,
                f"{raw_prefix}{window}",
                values,
                category,
                inputs,
                expression,
                window=window,
                lookback=lookback,
            )


def _add_factor(
    factor_columns: dict[str, pd.Series],
    specs: list[FactorSpec],
    config: Alpha158Config,
    raw_name: str,
    values: pd.Series,
    category: str,
    inputs: Iterable[str],
    expression: str,
    window: int | None = None,
    lookback: int = 0,
) -> None:
    name = f"{config.prefix}{raw_name}"
    factor_columns[name] = op.replace_inf(values)
    specs.append(
        FactorSpec(
            name=name,
            source="alpha158",
            category=category,
            inputs=tuple(inputs),
            expression=expression,
            window=window,
            lookback=lookback,
            availability="feature_asof_date",
        )
    )


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing Alpha158 input columns: {missing}")
