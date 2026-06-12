from __future__ import annotations

from numbers import Integral
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_GROUP_COL = "stock_code"
DEFAULT_DATE_COL = "trade_date"
_POSITION_COL = "__factor_miner_position__"


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _validate_window(window: int) -> int:
    if not isinstance(window, Integral) or int(window) <= 0:
        raise ValueError("window must be a positive integer")
    return int(window)


def _resolve_min_periods(window: int, min_periods: int | None) -> int:
    if min_periods is None:
        return window
    if not isinstance(min_periods, Integral) or int(min_periods) <= 0:
        raise ValueError("min_periods must be a positive integer")
    if int(min_periods) > window:
        raise ValueError("min_periods cannot be greater than window")
    return int(min_periods)


def _numeric_series(values: pd.Series | np.ndarray | list[float] | float, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if np.isscalar(values):
        if index is None:
            return pd.Series([values], dtype="float64")
        return pd.Series(values, index=index, dtype="float64")
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _ordered_frame(
    df: pd.DataFrame,
    value_columns: Iterable[str],
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.DataFrame:
    columns = list(dict.fromkeys([group_col, date_col, *value_columns]))
    _require_columns(df, columns)
    ordered = df[columns].copy()
    ordered[_POSITION_COL] = np.arange(len(df))
    return ordered.sort_values([group_col, date_col, _POSITION_COL], kind="mergesort").reset_index(drop=True)


def _restore_order(values: pd.Series, ordered: pd.DataFrame, original_index: pd.Index) -> pd.Series:
    restored = pd.Series(np.nan, index=np.arange(len(original_index)), dtype="float64")
    restored.iloc[ordered[_POSITION_COL].to_numpy()] = pd.to_numeric(values, errors="coerce").to_numpy()
    restored.index = original_index
    return replace_inf(restored)


def replace_inf(values: pd.Series | np.ndarray | list[float] | float) -> pd.Series:
    """Return a numeric Series with positive and negative infinity replaced by NA."""
    series = _numeric_series(values)
    result = series.replace([np.inf, -np.inf], np.nan)
    result.name = None
    return result


def safe_div(
    numerator: pd.Series | np.ndarray | list[float] | float,
    denominator: pd.Series | np.ndarray | list[float] | float,
) -> pd.Series:
    """Divide two numeric arrays and return NA where the denominator is zero or invalid."""
    index = numerator.index if isinstance(numerator, pd.Series) else denominator.index if isinstance(denominator, pd.Series) else None
    left = _numeric_series(numerator, index=index)
    right = _numeric_series(denominator, index=index)
    left, right = left.align(right, join="outer")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = left / right
    result = result.where(right != 0)
    return replace_inf(result)


def safe_log(values: pd.Series | np.ndarray | list[float] | float) -> pd.Series:
    """Take log only for positive values."""
    series = _numeric_series(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(series.where(series > 0))
    return replace_inf(result)


def safe_sqrt(values: pd.Series | np.ndarray | list[float] | float) -> pd.Series:
    """Take square root only for non-negative values."""
    series = _numeric_series(values)
    with np.errstate(invalid="ignore"):
        result = np.sqrt(series.where(series >= 0))
    return replace_inf(result)


def clip_extreme(
    values: pd.Series | np.ndarray | list[float] | float,
    lower: float | None = None,
    upper: float | None = None,
) -> pd.Series:
    """Clip values to explicit lower and upper bounds."""
    series = _numeric_series(values)
    return replace_inf(series.clip(lower=lower, upper=upper))


def winsorize(
    values: pd.Series | np.ndarray | list[float] | float,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """Clip values to empirical quantile bounds."""
    if not 0 <= lower_quantile <= upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower_quantile <= upper_quantile <= 1")
    series = _numeric_series(values)
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return replace_inf(series.clip(lower=lower, upper=upper))


def delay(
    df: pd.DataFrame,
    column: str,
    window: int,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return the value from n prior rows within each stock time series."""
    window = _validate_window(window)
    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])
    result = series.groupby(ordered[group_col], sort=False).shift(window)
    return _restore_order(result, ordered, df.index)


def delta(
    df: pd.DataFrame,
    column: str,
    window: int,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return current value minus the value from n prior rows."""
    window = _validate_window(window)
    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])
    result = series - series.groupby(ordered[group_col], sort=False).shift(window)
    return _restore_order(result, ordered, df.index)


def returns(
    df: pd.DataFrame,
    column: str,
    window: int,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return current value divided by the value from n prior rows minus one."""
    window = _validate_window(window)
    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])
    previous = series.groupby(ordered[group_col], sort=False).shift(window)
    result = safe_div(series, previous) - 1
    return _restore_order(result, ordered, df.index)


def _rolling_unary(
    df: pd.DataFrame,
    column: str,
    window: int,
    method: str,
    min_periods: int | None,
    group_col: str,
    date_col: str,
) -> pd.Series:
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])
    grouped = series.groupby(ordered[group_col], sort=False)
    result = getattr(grouped.rolling(window, min_periods=min_periods), method)().reset_index(level=0, drop=True)
    return _restore_order(result, ordered, df.index)


def _rolling_apply(
    df: pd.DataFrame,
    column: str,
    window: int,
    func,
    min_periods: int | None,
    group_col: str,
    date_col: str,
) -> pd.Series:
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])
    result = (
        series.groupby(ordered[group_col], sort=False)
        .rolling(window, min_periods=min_periods)
        .apply(func, raw=True)
        .reset_index(level=0, drop=True)
    )
    return _restore_order(result, ordered, df.index)


def ts_sum(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    return _rolling_unary(df, column, window, "sum", min_periods, group_col, date_col)


def ts_mean(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    return _rolling_unary(df, column, window, "mean", min_periods, group_col, date_col)


def ts_std(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    return _rolling_unary(df, column, window, "std", min_periods, group_col, date_col)


def ts_min(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    return _rolling_unary(df, column, window, "min", min_periods, group_col, date_col)


def ts_max(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    return _rolling_unary(df, column, window, "max", min_periods, group_col, date_col)


def ts_quantile(
    df: pd.DataFrame,
    column: str,
    window: int,
    quantile: float,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return rolling quantile within each stock time series."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])
    result = (
        series.groupby(ordered[group_col], sort=False)
        .rolling(window, min_periods=min_periods)
        .quantile(quantile)
        .reset_index(level=0, drop=True)
    )
    return _restore_order(result, ordered, df.index)


def ts_idxmax(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return the 1-based position of the maximum value inside the rolling window."""

    def _idxmax(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.argmax(values) + 1)

    return _rolling_apply(df, column, window, _idxmax, min_periods, group_col, date_col)


def ts_idxmin(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return the 1-based position of the minimum value inside the rolling window."""

    def _idxmin(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.argmin(values) + 1)

    return _rolling_apply(df, column, window, _idxmin, min_periods, group_col, date_col)


def ts_corr(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return rolling correlation within each stock time series."""
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    ordered = _ordered_frame(df, [left_column, right_column], group_col, date_col)
    left = _numeric_series(ordered[left_column])
    right = _numeric_series(ordered[right_column])
    result = pd.Series(np.nan, index=ordered.index, dtype="float64")
    for row_positions in ordered.groupby(group_col, sort=False).indices.values():
        positions = np.asarray(row_positions)
        values = left.iloc[positions].rolling(window, min_periods=min_periods).corr(right.iloc[positions])
        result.iloc[positions] = values.to_numpy()
    return _restore_order(result, ordered, df.index)


def ts_rank(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return the percentile rank of the current value inside its rolling window."""
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])

    def _last_pct_rank(values: np.ndarray) -> float:
        last = values[-1]
        if np.isnan(last):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        return float(pd.Series(valid).rank(method="average", pct=True).iloc[-1])

    result = (
        series.groupby(ordered[group_col], sort=False)
        .rolling(window, min_periods=min_periods)
        .apply(_last_pct_rank, raw=True)
        .reset_index(level=0, drop=True)
    )
    return _restore_order(result, ordered, df.index)


def ts_slope(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return the linear trend slope of a rolling window."""
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    if min_periods == window:
        return ts_linear_regression(df, column, window, min_periods, group_col, date_col)[0]
    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        x = np.arange(len(values), dtype="float64")
        x = x - x.mean()
        y = values - values.mean()
        denominator = np.dot(x, x)
        if denominator == 0:
            return np.nan
        return float(np.dot(x, y) / denominator)

    result = (
        series.groupby(ordered[group_col], sort=False)
        .rolling(window, min_periods=min_periods)
        .apply(_slope, raw=True)
        .reset_index(level=0, drop=True)
    )
    return _restore_order(result, ordered, df.index)


def ts_rsquare(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return rolling R-squared of a linear regression on time."""
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    if min_periods == window:
        return ts_linear_regression(df, column, window, min_periods, group_col, date_col)[1]

    def _rsquare(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        x = np.arange(1, len(values) + 1, dtype="float64")
        x_centered = x - x.mean()
        y_centered = values - values.mean()
        x_var = np.dot(x_centered, x_centered)
        y_var = np.dot(y_centered, y_centered)
        if x_var == 0 or y_var == 0:
            return np.nan
        corr = np.dot(x_centered, y_centered) / np.sqrt(x_var * y_var)
        return float(corr * corr)

    return _rolling_apply(df, column, window, _rsquare, min_periods, group_col, date_col)


def ts_residual(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return the latest residual from a rolling linear regression on time."""
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    if min_periods == window:
        return ts_linear_regression(df, column, window, min_periods, group_col, date_col)[2]

    def _residual(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        x = np.arange(1, len(values) + 1, dtype="float64")
        x_centered = x - x.mean()
        denominator = np.dot(x_centered, x_centered)
        if denominator == 0:
            return np.nan
        slope = np.dot(x_centered, values - values.mean()) / denominator
        intercept = values.mean() - slope * x.mean()
        fitted_latest = slope * x[-1] + intercept
        return float(values[-1] - fitted_latest)

    return _rolling_apply(df, column, window, _residual, min_periods, group_col, date_col)


def ts_linear_regression(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return rolling slope, R-squared, and latest residual against time."""
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    if min_periods != window:
        return (
            ts_slope(df, column, window, min_periods, group_col, date_col),
            ts_rsquare(df, column, window, min_periods, group_col, date_col),
            ts_residual(df, column, window, min_periods, group_col, date_col),
        )

    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])
    grouped = series.groupby(ordered[group_col], sort=False)
    position = grouped.cumcount().astype("float64")
    valid = series.notna().astype("float64")

    rolling = grouped.rolling(window, min_periods=window)
    sum_y = rolling.sum().reset_index(level=0, drop=True)
    sum_y2 = (series * series).groupby(ordered[group_col], sort=False).rolling(
        window,
        min_periods=window,
    ).sum().reset_index(level=0, drop=True)
    count_y = valid.groupby(ordered[group_col], sort=False).rolling(
        window,
        min_periods=window,
    ).sum().reset_index(level=0, drop=True)
    sum_pos_y = (series * position).groupby(ordered[group_col], sort=False).rolling(
        window,
        min_periods=window,
    ).sum().reset_index(level=0, drop=True)

    start_position = position - window + 1
    sum_xy = sum_pos_y - start_position * sum_y
    x_mean = (window - 1) / 2.0
    x_var = window * (window * window - 1) / 12.0
    y_mean = sum_y / window
    centered_xy = sum_xy - x_mean * sum_y
    y_var = sum_y2 - (sum_y * sum_y / window)

    slope = centered_xy / x_var
    rsquare = (centered_xy * centered_xy) / (x_var * y_var)
    fitted_latest = y_mean + slope * x_mean
    residual = series - fitted_latest

    complete = count_y.eq(float(window)) & y_var.gt(0)
    slope = slope.where(count_y.eq(float(window)))
    rsquare = rsquare.where(complete)
    residual = residual.where(count_y.eq(float(window)))

    return (
        _restore_order(slope, ordered, df.index),
        _restore_order(rsquare, ordered, df.index),
        _restore_order(residual, ordered, df.index),
    )


def decay_linear(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return a linearly decayed rolling average, with larger weights on recent rows."""
    window = _validate_window(window)
    min_periods = _resolve_min_periods(window, min_periods)
    ordered = _ordered_frame(df, [column], group_col, date_col)
    series = _numeric_series(ordered[column])

    def _weighted_average(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        weights = np.arange(1, len(values) + 1, dtype="float64")
        return float(np.dot(values, weights) / weights.sum())

    result = (
        series.groupby(ordered[group_col], sort=False)
        .rolling(window, min_periods=min_periods)
        .apply(_weighted_average, raw=True)
        .reset_index(level=0, drop=True)
    )
    return _restore_order(result, ordered, df.index)


def cs_rank(
    df: pd.DataFrame,
    column: str,
    date_col: str = DEFAULT_DATE_COL,
    ascending: bool = True,
) -> pd.Series:
    """Return cross-sectional rank within each trade date."""
    _require_columns(df, [date_col, column])
    series = _numeric_series(df[column])
    result = series.groupby(df[date_col], dropna=False).rank(method="average", ascending=ascending, na_option="keep")
    return replace_inf(result)


def cs_pct_rank(
    df: pd.DataFrame,
    column: str,
    date_col: str = DEFAULT_DATE_COL,
    ascending: bool = True,
) -> pd.Series:
    """Return cross-sectional percentile rank within each trade date."""
    _require_columns(df, [date_col, column])
    series = _numeric_series(df[column])
    result = series.groupby(df[date_col], dropna=False).rank(method="average", ascending=ascending, pct=True, na_option="keep")
    return replace_inf(result)


def cs_zscore(
    df: pd.DataFrame,
    column: str,
    date_col: str = DEFAULT_DATE_COL,
    ddof: int = 0,
) -> pd.Series:
    """Return cross-sectional z-score within each trade date."""
    _require_columns(df, [date_col, column])
    series = _numeric_series(df[column])
    grouped = series.groupby(df[date_col], dropna=False)
    mean = grouped.transform("mean")
    std = grouped.transform(lambda item: item.std(ddof=ddof))
    return safe_div(series - mean, std)


def industry_neutralize(
    df: pd.DataFrame,
    column: str,
    industry_col: str = "industry",
    date_col: str = DEFAULT_DATE_COL,
) -> pd.Series:
    """Return residuals after removing same-date industry group means."""
    _require_columns(df, [date_col, industry_col, column])
    series = _numeric_series(df[column])
    valid_industry = df[industry_col].notna() & df[industry_col].astype("string").str.strip().ne("")
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    if valid_industry.any():
        industry_mean = series.loc[valid_industry].groupby(
            [df.loc[valid_industry, date_col], df.loc[valid_industry, industry_col]],
            dropna=True,
        ).transform("mean")
        result.loc[valid_industry] = series.loc[valid_industry] - industry_mean
    return replace_inf(result)
