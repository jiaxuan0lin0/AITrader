import pandas as pd
import pytest

from FactorMiner.pools.news_llm import prepare_news_items


def test_prepare_news_items_deduplicates_news_and_explodes_stock_map() -> None:
    news = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "600000.SH", None, None],
            "matched_stock_codes": ["000001.SZ|000002.SZ", "600000.SH", "", ""],
            "matched_stock_count": [2, 1, 0, 0],
            "trade_date": pd.to_datetime(["2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"]),
            "publish_time": [
                "2024-01-02 18:00:00",
                "2024-01-02 18:00:00",
                "2024-01-03 08:00:00",
                "2024-01-04 08:00:00",
            ],
            "news_text": [
                "同一条新闻",
                "同一条新闻",
                "市场级新闻",
                "",
            ],
            "__source_file": ["a.csv", "b.csv", "c.csv", "d.csv"],
        }
    )

    result = prepare_news_items(news)

    assert len(result.news_items) == 2
    assert len(result.news_stock_map) == 3
    first_item = result.news_items.loc[result.news_items["news_text"].eq("同一条新闻")].iloc[0]
    assert first_item["matched_stock_codes"] == "000001.SZ|000002.SZ|600000.SH"
    assert first_item["matched_stock_count"] == 3
    assert set(result.news_stock_map["stock_code"]) == {"000001.SZ", "000002.SZ", "600000.SH"}
    assert result.news_items.loc[result.news_items["news_text"].eq("市场级新闻"), "matched_stock_count"].iloc[0] == 0


def test_prepare_news_items_generates_stable_ids() -> None:
    news = pd.DataFrame(
        {
            "publish_time": ["2024-01-02 18:00:00", "2024-01-02 18:00:00"],
            "trade_date": ["2024-01-03", "2024-01-03"],
            "news_text": ["  文本\n内容  ", "文本 内容"],
            "matched_stock_codes": ["000001.SZ", "000001.SZ"],
        }
    )

    result = prepare_news_items(news)

    assert len(result.news_items) == 1
    assert result.news_items["news_id"].str.len().iloc[0] == 64
    assert result.news_items["news_text_hash"].str.len().iloc[0] == 64


def test_prepare_news_items_rejects_missing_required_columns() -> None:
    news = pd.DataFrame({"publish_time": ["2024-01-02"]})

    with pytest.raises(KeyError, match="Missing news columns"):
        prepare_news_items(news)


def test_prepare_news_items_handles_empty_valid_input() -> None:
    news = pd.DataFrame(
        {
            "publish_time": ["not-a-date"],
            "news_text": ["市场级新闻"],
        }
    )

    result = prepare_news_items(news)

    assert result.news_items.empty
    assert result.news_stock_map.empty
