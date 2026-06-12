from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from FactorMiner.build.news_sample import main
from FactorMiner.core.registry import validate_registry
from FactorMiner.pools.news_llm import prepare_news_items


def test_build_news_sample_writes_feature_block_and_registry(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.parquet"
    news_path = tmp_path / "news.parquet"
    scores_path = tmp_path / "news_llm_scores.parquet"
    output_root = tmp_path / "features"
    feature_registry_path = output_root / "feature_registry.json"

    pd.DataFrame(
        {
            "sample_id": ["000001_20240110", "000002_20240110"],
            "stock_code": ["000001.SZ", "000002.SZ"],
            "decision_ts": pd.to_datetime(["2024-01-10 09:25:00", "2024-01-10 09:25:00"]),
            "target_trade_date": pd.to_datetime(["2024-01-10", "2024-01-10"]),
        }
    ).to_parquet(samples_path, index=False)
    news = pd.DataFrame(
        {
            "publish_time": pd.to_datetime(["2024-01-10 08:00:00", "2024-01-10 08:30:00"]),
            "trade_date": pd.to_datetime(["2024-01-10", "2024-01-10"]),
            "news_text": ["个股盈利改善", "市场宏观政策更新"],
            "matched_stock_codes": ["000001.SZ", ""],
            "matched_stock_count": [1, 0],
        }
    )
    news.to_parquet(news_path, index=False)
    prepared = prepare_news_items(news)
    pd.DataFrame(
        {
            "news_text_hash": prepared.news_items["news_text_hash"],
            "sentiment_score": [0.7, -0.2],
            "impact_score": [0.8, 0.6],
            "risk_score": [0.1, 0.4],
            "relevance_score": [0.9, 0.8],
            "novelty_score": [0.5, 0.7],
            "event_type": ["earnings", "macro"],
        }
    ).to_parquet(scores_path, index=False)

    exit_code = main(
        [
            "--samples-path",
            str(samples_path),
            "--news-path",
            str(news_path),
            "--scores-path",
            str(scores_path),
            "--output-root",
            str(output_root),
            "--feature-registry-path",
            str(feature_registry_path),
            "--windows",
            "1",
            "--since",
            "2024-01-10",
            "--until",
            "2024-01-10",
        ]
    )

    assert exit_code == 0
    market_path = output_root / "blocks" / "sample" / "news_llm_market_sample.parquet"
    stock_path = output_root / "blocks" / "sample" / "news_llm_stock_sample.parquet"
    market_manifest_path = output_root / "manifests" / "news_llm_market_sample.json"
    stock_manifest_path = output_root / "manifests" / "news_llm_stock_sample.json"
    market_features = pd.read_parquet(market_path)
    stock_features = pd.read_parquet(stock_path)
    market_manifest = json.loads(market_manifest_path.read_text(encoding="utf-8"))
    stock_manifest = json.loads(stock_manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(feature_registry_path.read_text(encoding="utf-8"))

    assert market_features["sample_id"].tolist() == ["000001_20240110", "000002_20240110"]
    assert stock_features["sample_id"].tolist() == ["000001_20240110", "000002_20240110"]
    assert stock_features.loc[0, "news_stock_count_1d"] == 1.0
    assert stock_features.loc[1, "news_stock_count_1d"] == 0.0
    assert market_features["news_market_count_1d"].tolist() == [1.0, 1.0]
    assert market_manifest[0]["name"] == "news_market_count_1d"
    assert stock_manifest[0]["name"] == "news_stock_count_1d"
    assert [record["name"] for record in registry] == ["news_llm_market_sample", "news_llm_stock_sample"]
    assert registry[0]["granularity"] == "sample"
    assert registry[0]["factor_path"] == "blocks/sample/news_llm_market_sample.parquet"
    validate_registry(feature_registry_path)


def test_build_news_sample_rejects_limit_on_default_feature_registry() -> None:
    with pytest.raises(ValueError, match="limit requires non-default --output-root/--output-path"):
        main(["--limit", "10"])
