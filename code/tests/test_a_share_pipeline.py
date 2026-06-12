from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.a_share_pipeline import main
from FactorMiner.pools.news_llm import prepare_news_items


def test_pipeline_attaches_industry_to_metric_moneyflow_without_breaking_news(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "datasets"
    raw_dir.mkdir()

    price_path = raw_dir / "daily.csv"
    metric_path = raw_dir / "metric.csv"
    moneyflow_path = raw_dir / "moneyflow.csv"
    basic_path = raw_dir / "basic.csv"
    news_path = raw_dir / "news.csv"

    price_path.write_text(
        "\n".join(
            [
                "ts_code,trade_date,stock_name,open,high,low,close,preclose,vol,amount,vwap",
                "000001.SZ,20240102,平安银行,10,11,9,10,9.8,1000,10000,10.0",
                "000001.SZ,20240103,平安银行,10.5,11,10,10.8,10,1100,12000,10.7",
                "000001.SZ,20240104,平安银行,11,12,10.8,11.5,10.8,1200,14000,11.3",
            ]
        ),
        encoding="utf-8",
    )
    metric_path.write_text(
        "\n".join(
            [
                "ts_code,trade_date,pe_ttm,pb,total_mv,circ_mv",
                "000001.SZ,20240102,5,0.5,100000,90000",
                "000001.SZ,20240103,5.1,0.51,101000,91000",
                "000001.SZ,20240104,5.2,0.52,102000,92000",
            ]
        ),
        encoding="utf-8",
    )
    moneyflow_path.write_text(
        "\n".join(
            [
                "ts_code,trade_date,buy_sm_amount,sell_sm_amount,buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount,net_mf_vol,net_mf_amount",
                "000001.SZ,20240102,1,2,10,4,5,1,100,8",
                "000001.SZ,20240103,1,2,11,5,6,2,120,9",
                "000001.SZ,20240104,1,2,12,6,7,3,130,10",
            ]
        ),
        encoding="utf-8",
    )
    basic_path.write_text(
        "ts_code,stock_name,industry,market,area,list_date\n000001.SZ,平安银行,银行,主板,深圳,19910403\n",
        encoding="utf-8",
    )
    news_path.write_text(
        "datetime,title,content\n2024-01-02 16:00:00,平安银行发布经营快报,银行资产质量改善\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
            "--price-file",
            str(price_path),
            "--metric-file",
            str(metric_path),
            "--moneyflow-file",
            str(moneyflow_path),
            "--basic-file",
            str(basic_path),
            "--news-file",
            str(news_path),
        ]
    )

    assert exit_code == 0
    processed = output_dir / "processed"
    metric = pd.read_parquet(processed / "metric.parquet")
    moneyflow = pd.read_parquet(processed / "moneyflow.parquet")
    price = pd.read_parquet(processed / "price.parquet")
    samples = pd.read_parquet(processed / "samples.parquet")
    news = pd.read_parquet(processed / "news.parquet")
    news_stock_daily = pd.read_parquet(processed / "news_stock_daily.parquet")
    news_market_daily = pd.read_parquet(processed / "news_market_daily.parquet")

    assert metric["industry"].dropna().unique().tolist() == ["银行"]
    assert moneyflow["industry"].dropna().unique().tolist() == ["银行"]
    for frame in (price, samples, news):
        assert not any(column.endswith("_x") or column.endswith("_y") for column in frame.columns)
    assert price["stock_name"].dropna().unique().tolist() == ["平安银行"]
    assert samples["industry"].dropna().unique().tolist() == ["银行"]
    assert {"publish_time", "trade_date", "news_text", "matched_stock_codes", "matched_stock_count"}.issubset(news.columns)
    assert news.loc[0, "news_text"] == "平安银行发布经营快报\n银行资产质量改善"
    assert news.loc[0, "matched_stock_codes"] == "000001.SZ"
    assert news.loc[0, "matched_stock_count"] == 1
    assert news_stock_daily.loc[0, "stock_code"] == "000001.SZ"
    assert news_stock_daily.loc[0, "news_count"] == 1
    assert news_market_daily["market_news_count"].sum() == 1

    news_items = prepare_news_items(news)
    assert len(news_items.news_items) == 1
    assert news_items.news_items.loc[0, "news_text"] == "平安银行发布经营快报 银行资产质量改善"
    assert news_items.news_stock_map["stock_code"].tolist() == ["000001.SZ"]
