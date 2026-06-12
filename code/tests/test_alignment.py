from __future__ import annotations

import pandas as pd
import pytest

from FactorMiner.core.alignment import align_daily_factors_to_samples


def test_align_daily_factors_uses_feature_asof_date_not_target_trade_date() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["A_20240103"],
            "stock_code": ["A"],
            "feature_asof_date": pd.to_datetime(["2024-01-02"]),
            "target_trade_date": pd.to_datetime(["2024-01-03"]),
        }
    )
    daily = pd.DataFrame(
        {
            "stock_code": ["A", "A"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "factor_a": [1.0, 99.0],
        }
    )

    aligned = align_daily_factors_to_samples(samples, daily)

    assert aligned.loc[0, "factor_a"] == 1.0


def test_align_daily_factors_rejects_duplicate_daily_keys() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["A_20240103"],
            "stock_code": ["A"],
            "feature_asof_date": pd.to_datetime(["2024-01-02"]),
        }
    )
    daily = pd.DataFrame(
        {
            "stock_code": ["A", "A"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "factor_a": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="Daily factor keys must be unique"):
        align_daily_factors_to_samples(samples, daily)
