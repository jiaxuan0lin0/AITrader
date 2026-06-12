import numpy as np
import pandas as pd
import pytest

from FactorMiner import operators as op


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["B", "A", "A", "B", "A", "B"],
            "trade_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "close": [20.0, 13.0, 10.0, 18.0, 11.0, 24.0],
            "volume": [200.0, 130.0, 100.0, 180.0, 110.0, 240.0],
            "industry": ["bank", "bank", "bank", "tech", "bank", "tech"],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def test_delay_preserves_original_index_and_does_not_cross_stocks() -> None:
    df = _sample_frame()

    result = op.delay(df, "close", 1)

    expected = pd.Series(
        [18.0, 11.0, np.nan, np.nan, 10.0, 20.0],
        index=df.index,
        dtype="float64",
    )
    pd.testing.assert_series_equal(result, expected)


def test_returns_uses_past_values_only() -> None:
    df = _sample_frame()

    result = op.returns(df, "close", 2)

    expected = pd.Series(
        [np.nan, 0.3, np.nan, np.nan, np.nan, 24.0 / 18.0 - 1],
        index=df.index,
        dtype="float64",
    )
    pd.testing.assert_series_equal(result, expected)


def test_rolling_mean_requires_full_window_by_default() -> None:
    df = _sample_frame()

    result = op.ts_mean(df, "close", 2)

    expected = pd.Series(
        [19.0, 12.0, np.nan, np.nan, 10.5, 22.0],
        index=df.index,
        dtype="float64",
    )
    pd.testing.assert_series_equal(result, expected)


def test_ts_corr_is_computed_within_each_stock() -> None:
    df = _sample_frame()

    result = op.ts_corr(df, "close", "volume", 3)

    expected = pd.Series(
        [np.nan, 1.0, np.nan, np.nan, np.nan, 1.0],
        index=df.index,
        dtype="float64",
    )
    pd.testing.assert_series_equal(result, expected)


def test_ts_rank_returns_current_value_percentile_inside_window() -> None:
    df = _sample_frame()

    result = op.ts_rank(df, "close", 3)

    expected = pd.Series(
        [np.nan, 1.0, np.nan, np.nan, np.nan, 1.0],
        index=df.index,
        dtype="float64",
    )
    pd.testing.assert_series_equal(result, expected)


def test_alpha158_extra_rolling_operators() -> None:
    df = pd.DataFrame(
        {
            "stock_code": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "trade_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "value": [1.0, 2.0, 4.0, 8.0, 4.0, 3.0, 2.0, 1.0],
        },
        index=[30, 31, 32, 33, 34, 35, 36, 37],
    )

    q80 = op.ts_quantile(df, "value", 3, 0.8)
    idxmax = op.ts_idxmax(df, "value", 3)
    idxmin = op.ts_idxmin(df, "value", 3)
    rsquare = op.ts_rsquare(df, "value", 3)
    residual = op.ts_residual(df, "value", 3)

    expected_q80 = pd.Series([np.nan, np.nan, 3.2, 6.4, np.nan, np.nan, 3.6, 2.6], index=df.index)
    expected_idxmax = pd.Series([np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 1.0, 1.0], index=df.index)
    expected_idxmin = pd.Series([np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 3.0, 3.0], index=df.index)
    expected_rsquare = pd.Series([np.nan, np.nan, 0.9642857142857143, 0.9642857142857143, np.nan, np.nan, 1.0, 1.0], index=df.index)
    expected_residual = pd.Series([np.nan, np.nan, 1 / 6, 1 / 3, np.nan, np.nan, 0.0, 0.0], index=df.index)

    pd.testing.assert_series_equal(q80, expected_q80, check_exact=False)
    pd.testing.assert_series_equal(idxmax, expected_idxmax)
    pd.testing.assert_series_equal(idxmin, expected_idxmin)
    pd.testing.assert_series_equal(rsquare, expected_rsquare, check_exact=False)
    pd.testing.assert_series_equal(residual, expected_residual, check_exact=False)


def test_ts_slope_and_decay_linear() -> None:
    df = _sample_frame()

    slope = op.ts_slope(df, "close", 3)
    decay = op.decay_linear(df, "close", 3)

    expected_slope = pd.Series(
        [np.nan, 1.5, np.nan, np.nan, np.nan, 3.0],
        index=df.index,
        dtype="float64",
    )
    expected_decay = pd.Series(
        [np.nan, (10.0 + 22.0 + 39.0) / 6.0, np.nan, np.nan, np.nan, (18.0 + 40.0 + 72.0) / 6.0],
        index=df.index,
        dtype="float64",
    )
    pd.testing.assert_series_equal(slope, expected_slope)
    pd.testing.assert_series_equal(decay, expected_decay)


def test_cross_sectional_rank_and_zscore_use_trade_date_only() -> None:
    df = _sample_frame()

    rank = op.cs_pct_rank(df, "close")
    zscore = op.cs_zscore(df, "close")

    expected_rank = pd.Series(
        [1.0, 0.5, 0.5, 1.0, 0.5, 1.0],
        index=df.index,
        dtype="float64",
    )
    expected_zscore = pd.Series(
        [1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
        index=df.index,
        dtype="float64",
    )
    pd.testing.assert_series_equal(rank, expected_rank)
    pd.testing.assert_series_equal(zscore, expected_zscore)


def test_industry_neutralize_removes_same_date_industry_mean() -> None:
    df = pd.DataFrame(
        {
            "stock_code": ["A", "B", "C", "D"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03"]),
            "industry": ["bank", "bank", "tech", "bank"],
            "value": [1.0, 3.0, 10.0, 8.0],
        }
    )

    result = op.industry_neutralize(df, "value")

    expected = pd.Series([-1.0, 1.0, 0.0, 0.0], dtype="float64")
    pd.testing.assert_series_equal(result, expected)


def test_industry_neutralize_keeps_missing_industry_as_missing() -> None:
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02"]),
            "industry": ["bank", "bank", None, ""],
            "value": [1.0, 3.0, 10.0, 20.0],
        }
    )

    result = op.industry_neutralize(df, "value")

    expected = pd.Series([-1.0, 1.0, np.nan, np.nan], dtype="float64")
    pd.testing.assert_series_equal(result, expected)


def test_safe_numeric_helpers_do_not_emit_infinity() -> None:
    numerator = pd.Series([1.0, 2.0, 3.0])
    denominator = pd.Series([1.0, 0.0, np.nan])

    divided = op.safe_div(numerator, denominator)
    logged = op.safe_log(pd.Series([1.0, 0.0, -1.0]))
    rooted = op.safe_sqrt(pd.Series([4.0, 0.0, -1.0]))

    assert divided.iloc[0] == 1.0
    assert np.isnan(divided.iloc[1])
    assert np.isnan(divided.iloc[2])
    assert logged.iloc[0] == 0.0
    assert np.isnan(logged.iloc[1])
    assert np.isnan(logged.iloc[2])
    assert rooted.iloc[0] == 2.0
    assert rooted.iloc[1] == 0.0
    assert np.isnan(rooted.iloc[2])


def test_invalid_window_and_missing_columns_raise_clear_errors() -> None:
    df = _sample_frame()

    with pytest.raises(ValueError, match="positive integer"):
        op.ts_mean(df, "close", 0)

    with pytest.raises(KeyError, match="Missing required columns"):
        op.ts_mean(df, "not_a_column", 3)
