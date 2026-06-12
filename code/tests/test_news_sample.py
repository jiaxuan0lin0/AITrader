import numpy as np
import pandas as pd
import pytest

from FactorMiner.pools.news_sample import NewsSampleConfig, build_news_sample_factors


def test_news_sample_factors_respect_natural_day_window_and_decision_ts() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["A_20240110", "B_20240110", "A_20240112"],
            "stock_code": ["A", "B", "A"],
            "decision_ts": pd.to_datetime(["2024-01-10 09:25:00", "2024-01-10 09:25:00", "2024-01-12 09:25:00"]),
        }
    )
    news_items = pd.DataFrame(
        {
            "news_id": ["stock_a", "market_recent", "market_boundary", "future_news"],
            "news_text_hash": ["h_stock_a", "h_market_recent", "h_market_boundary", "h_future"],
            "publish_time": pd.to_datetime(
                [
                    "2024-01-10 08:25:00",
                    "2024-01-09 10:00:00",
                    "2024-01-09 09:25:00",
                    "2024-01-10 10:00:00",
                ]
            ),
            "matched_stock_count": [1, 0, 0, 0],
        }
    )
    news_stock_map = pd.DataFrame(
        {
            "news_id": ["stock_a"],
            "stock_code": ["A"],
            "publish_time": pd.to_datetime(["2024-01-10 08:25:00"]),
            "trade_date": pd.to_datetime(["2024-01-10"]),
        }
    )
    scores = pd.DataFrame(
        {
            "news_text_hash": ["h_stock_a", "h_market_recent", "h_market_boundary", "h_future"],
            "sentiment_score": [0.8, -0.6, -1.0, 1.0],
            "impact_score": [0.9, 0.5, 0.9, 0.9],
            "risk_score": [0.2, 0.7, 0.9, 0.0],
            "relevance_score": [0.9, 0.8, 0.8, 0.9],
            "novelty_score": [0.8, 0.6, 0.9, 0.9],
            "event_type": ["earnings", "macro", "geopolitics", "policy"],
        }
    )

    result = build_news_sample_factors(
        samples,
        news_items,
        news_stock_map,
        scores,
        NewsSampleConfig(windows=(1, 3)),
    )

    factors = result.factors.set_index("sample_id")
    assert result.key_columns == ("sample_id",)
    assert "news_market_count_1d" in result.factor_names()
    assert factors.loc["A_20240110", "news_market_count_1d"] == 1.0
    assert factors.loc["B_20240110", "news_market_count_1d"] == 1.0
    assert factors.loc["A_20240110", "news_market_sentiment_mean_1d"] == pytest.approx(-0.6)
    assert factors.loc["A_20240110", "news_market_geopolitics_count_1d"] == 0.0
    assert factors.loc["A_20240110", "news_market_macro_count_1d"] == 1.0
    assert factors.loc["A_20240110", "news_stock_count_1d"] == 1.0
    assert factors.loc["B_20240110", "news_stock_count_1d"] == 0.0
    assert factors.loc["A_20240110", "news_stock_impact_weighted_sentiment_1d"] == 0.8
    assert factors.loc["A_20240110", "news_stock_hours_since_latest_1d"] == 1.0
    assert factors.loc["A_20240112", "news_stock_count_1d"] == 0.0
    assert np.isnan(factors.loc["A_20240112", "news_stock_sentiment_mean_1d"])


def test_news_sample_factors_build_tail_and_event_type_features() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["A_20240110"],
            "stock_code": ["A"],
            "decision_ts": pd.to_datetime(["2024-01-10 09:25:00"]),
        }
    )
    news_items = pd.DataFrame(
        {
            "news_id": ["n1", "n2"],
            "news_text_hash": ["h1", "h2"],
            "publish_time": pd.to_datetime(["2024-01-09 12:00:00", "2024-01-10 08:00:00"]),
            "matched_stock_count": [0, 0],
        }
    )
    news_stock_map = pd.DataFrame(columns=["news_id", "stock_code", "publish_time", "trade_date"])
    scores = pd.DataFrame(
        {
            "news_text_hash": ["h1", "h2"],
            "sentiment_score": [-0.5, 0.4],
            "impact_score": [0.8, 0.2],
            "risk_score": [0.9, 0.1],
            "relevance_score": [0.7, 0.6],
            "novelty_score": [0.5, 0.4],
            "event_type": ["geopolitics", "policy"],
        }
    )

    result = build_news_sample_factors(samples, news_items, news_stock_map, scores, NewsSampleConfig(windows=(1,)))
    row = result.factors.iloc[0]

    assert row["news_market_count_1d"] == 2.0
    assert row["news_market_high_impact_count_1d"] == 1.0
    assert row["news_market_negative_high_impact_count_1d"] == 1.0
    assert row["news_market_max_risk_1d"] == 0.9
    assert row["news_market_min_sentiment_1d"] == -0.5
    assert row["news_market_policy_count_1d"] == 1.0
    assert row["news_market_geopolitics_count_1d"] == 1.0
