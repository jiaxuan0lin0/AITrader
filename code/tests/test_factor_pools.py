import numpy as np
import pandas as pd

from FactorMiner.pools.alpha158 import Alpha158Config, build_alpha158_factors
from FactorMiner.pools.metric import MetricConfig, build_metric_factors
from FactorMiner.pools.moneyflow import MoneyflowConfig, build_moneyflow_factors
from FactorMiner.pools.neutral import NeutralConfig


def test_alpha158_builds_expected_core_formulas() -> None:
    price = pd.DataFrame(
        {
            "stock_code": ["A", "A", "A"],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": [9.0, 11.0, 14.0],
            "high": [11.0, 13.0, 16.0],
            "low": [8.0, 10.0, 13.0],
            "close": [10.0, 12.0, 15.0],
            "vwap": [9.5, 12.0, 14.5],
            "volume": [100.0, 120.0, 150.0],
            "industry": ["bank", "bank", "bank"],
        }
    )
    config = Alpha158Config(
        return_windows=(1,),
        rolling_windows=(3,),
        price_windows=(0,),
        neutral=None,
    )

    result = build_alpha158_factors(price, config)
    result.validate()
    factors = result.factors

    assert "alpha158_KMID" in result.factor_names()
    assert "alpha158_ROC1" in result.factor_names()
    assert "alpha158_MA3" in result.factor_names()
    assert factors.loc[0, "alpha158_KMID"] == (10.0 - 9.0) / 9.0
    assert factors.loc[1, "alpha158_ROC1"] == 10.0 / 12.0
    assert np.isnan(factors.loc[0, "alpha158_ROC1"])
    assert factors.loc[2, "alpha158_MA3"] == (10.0 + 12.0 + 15.0) / 3.0 / 15.0


def test_metric_pool_builds_missing_flags_and_neutral_features() -> None:
    metric = pd.DataFrame(
        {
            "stock_code": ["A", "B", "C"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "pe_ttm": [10.0, 20.0, np.nan],
            "pb": [1.0, 2.0, 3.0],
            "total_mv": [100.0, 200.0, 300.0],
            "turnover_rate": [1.0, 2.0, 3.0],
            "industry": ["bank", "bank", "tech"],
        }
    )
    config = MetricConfig(
        raw_fields=("pe_ttm", "pb", "total_mv", "turnover_rate"),
        missing_flag_fields=("pe_ttm",),
        delta_windows=(),
        rolling_windows=(),
        neutral=NeutralConfig(add_cs_z=False),
        max_neutral_missing_rate=1.0,
    )

    result = build_metric_factors(metric, config=config)
    result.validate()
    factors = result.factors

    assert factors.loc[0, "metric_earnings_yield"] == 0.1
    assert factors.loc[2, "metric_pe_ttm_missing"] == 1.0
    assert "metric_pe_ttm_cs_pct" in result.factor_names()
    assert "metric_pe_ttm_ind_neu" in result.factor_names()
    assert factors.loc[0, "metric_pe_ttm_ind_neu"] == -5.0
    assert factors.loc[1, "metric_pe_ttm_ind_neu"] == 5.0


def test_metric_pool_skips_neutral_features_for_high_missing_factors() -> None:
    metric = pd.DataFrame(
        {
            "stock_code": ["A", "B", "C", "D"],
            "trade_date": pd.to_datetime(["2024-01-02"] * 4),
            "dv_ttm": [1.0, np.nan, np.nan, np.nan],
            "pb": [1.0, 2.0, 3.0, 4.0],
            "industry": ["bank", "bank", "tech", "tech"],
        }
    )
    config = MetricConfig(
        raw_fields=("dv_ttm", "pb"),
        missing_flag_fields=(),
        delta_windows=(),
        rolling_windows=(),
        neutral=NeutralConfig(add_cs_z=False),
        max_neutral_missing_rate=0.25,
    )

    result = build_metric_factors(metric, config=config)

    assert "metric_dv_ttm" in result.factor_names()
    assert "metric_dv_ttm_cs_pct" not in result.factor_names()
    assert "metric_dv_ttm_ind_neu" not in result.factor_names()
    assert "metric_pb_cs_pct" in result.factor_names()


def test_moneyflow_pool_converts_units_and_builds_momentum() -> None:
    moneyflow = pd.DataFrame(
        {
            "stock_code": ["A", "A", "A"],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "buy_sm_amount": [1.0, 1.0, 1.0],
            "sell_sm_amount": [2.0, 2.0, 2.0],
            "buy_lg_amount": [10.0, 12.0, 14.0],
            "sell_lg_amount": [4.0, 5.0, 6.0],
            "buy_elg_amount": [5.0, 6.0, 7.0],
            "sell_elg_amount": [1.0, 2.0, 3.0],
            "net_mf_amount": [8.0, 9.0, 10.0],
        }
    )
    price = pd.DataFrame(
        {
            "stock_code": ["A", "A", "A"],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "amount": [1000.0, 1000.0, 1000.0],
            "close": [10.0, 11.0, 12.0],
            "industry": ["bank", "bank", "bank"],
        }
    )
    config = MoneyflowConfig(
        rolling_windows=(3,),
        corr_windows=(),
        positive_windows=(3,),
        neutral=None,
    )

    result = build_moneyflow_factors(moneyflow, price, config)
    result.validate()
    factors = result.factors

    assert factors.loc[0, "mf_main_net_amount_ratio"] == ((10.0 + 5.0 - 4.0 - 1.0) * 10.0) / 1000.0
    assert factors.loc[0, "mf_small_order_pressure"] == ((2.0 - 1.0) * 10.0) / 1000.0
    assert factors.loc[2, "mf_main_positive_days3"] == 3.0
    assert "mf_main_net_amount_ratio_ma3" in result.factor_names()
