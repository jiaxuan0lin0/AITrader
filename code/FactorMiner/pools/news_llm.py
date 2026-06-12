from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable

import pandas as pd


NEWS_ITEM_COLUMNS = (
    "news_id",
    "news_text_hash",
    "publish_time",
    "trade_date",
    "news_text",
    "matched_stock_codes",
    "matched_stock_count",
    "source_file",
)
NEWS_STOCK_MAP_COLUMNS = ("news_id", "stock_code", "publish_time", "trade_date")


@dataclass(frozen=True)
class NewsItems:
    """Single-news item table plus an exploded news-to-stock map."""

    news_items: pd.DataFrame
    news_stock_map: pd.DataFrame

    def validate(self) -> None:
        _require_columns(self.news_items, NEWS_ITEM_COLUMNS, "news_items")
        _require_columns(self.news_stock_map, NEWS_STOCK_MAP_COLUMNS, "news_stock_map")
        if self.news_items["news_id"].duplicated().any():
            raise ValueError("news_items.news_id must be unique")
        if self.news_items["publish_time"].isna().any():
            raise ValueError("news_items.publish_time cannot be missing")
        duplicated_map = self.news_stock_map.duplicated(["news_id", "stock_code"])
        if duplicated_map.any():
            examples = self.news_stock_map.loc[duplicated_map, ["news_id", "stock_code"]].head(5).to_dict("records")
            raise ValueError(f"news_stock_map contains duplicate news-stock pairs: {examples}")
        unknown_ids = set(self.news_stock_map["news_id"]) - set(self.news_items["news_id"])
        if unknown_ids:
            raise ValueError(f"news_stock_map references unknown news_id values: {sorted(unknown_ids)[:5]}")


def prepare_news_items(news: pd.DataFrame) -> NewsItems:
    """Prepare one-row-per-news items and an exploded news-to-stock map.

    The input is expected to be the processed `news.parquet` detail table.
    Raw CSV files do not contain the stock matching fields needed here.
    """
    _require_columns(news, ("publish_time", "news_text"), "news")

    work = news.copy()
    work["publish_time"] = pd.to_datetime(work["publish_time"], errors="coerce")
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce") if "trade_date" in work.columns else pd.NaT
    work["news_text"] = work["news_text"].map(_normalize_text)
    work = work.dropna(subset=["publish_time"])
    work = work.loc[work["news_text"] != ""].reset_index(drop=True)

    if work.empty:
        empty_items = pd.DataFrame(columns=NEWS_ITEM_COLUMNS)
        empty_map = pd.DataFrame(columns=NEWS_STOCK_MAP_COLUMNS)
        result = NewsItems(empty_items, empty_map)
        result.validate()
        return result

    work["news_text_hash"] = work["news_text"].map(lambda text: _stable_hash([text]))
    work["news_id"] = [
        _stable_hash([timestamp.isoformat(), text_hash])
        for timestamp, text_hash in zip(work["publish_time"], work["news_text_hash"], strict=True)
    ]
    work["source_file"] = work["__source_file"] if "__source_file" in work.columns else pd.NA

    stock_map = _build_stock_map(work)
    deduped = _deduplicate_news_items(work, stock_map)
    result = NewsItems(deduped.loc[:, NEWS_ITEM_COLUMNS], stock_map.loc[:, NEWS_STOCK_MAP_COLUMNS])
    result.validate()
    return result


def _deduplicate_news_items(work: pd.DataFrame, stock_map: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["news_id", "news_text_hash", "publish_time", "trade_date", "news_text", "source_file"]
    deduped = work.loc[:, base_columns].drop_duplicates("news_id", keep="first").reset_index(drop=True)
    if stock_map.empty:
        deduped["matched_stock_codes"] = ""
        deduped["matched_stock_count"] = 0
        return deduped

    stock_summary = (
        stock_map.groupby("news_id", sort=False)["stock_code"]
        .agg(matched_stock_codes=lambda values: "|".join(dict.fromkeys(values)), matched_stock_count="nunique")
        .reset_index()
    )
    deduped = deduped.merge(stock_summary, on="news_id", how="left")
    deduped["matched_stock_codes"] = deduped["matched_stock_codes"].fillna("")
    deduped["matched_stock_count"] = deduped["matched_stock_count"].fillna(0).astype(int)
    return deduped


def _build_stock_map(work: pd.DataFrame) -> pd.DataFrame:
    matched_codes = work["matched_stock_codes"] if "matched_stock_codes" in work.columns else pd.Series("", index=work.index)
    stock_code = work["stock_code"] if "stock_code" in work.columns else pd.Series("", index=work.index)
    codes_text = matched_codes.fillna("").astype(str) + "|" + stock_code.fillna("").astype(str)
    exploded = work.loc[:, ["news_id", "publish_time", "trade_date"]].copy()
    exploded["stock_code"] = codes_text.str.split(r"[|,，;；\s]+")
    exploded = exploded.explode("stock_code")
    exploded["stock_code"] = exploded["stock_code"].map(_normalize_stock_code)
    exploded = exploded.dropna(subset=["stock_code"])
    if exploded.empty:
        return pd.DataFrame(columns=NEWS_STOCK_MAP_COLUMNS)
    return exploded.loc[:, NEWS_STOCK_MAP_COLUMNS].drop_duplicates(["news_id", "stock_code"]).reset_index(drop=True)


def _normalize_stock_code(value: str) -> str | None:
    text = value.strip().upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return None
    matched = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", text)
    if matched:
        return f"{matched.group(1)}.{matched.group(2)}"
    matched = re.fullmatch(r"(\d{6})(SH|SZ|BJ)", text)
    if matched:
        return f"{matched.group(1)}.{matched.group(2)}"
    matched = re.fullmatch(r"\d{6}", text)
    if matched:
        return text
    return None


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _stable_hash(parts: Iterable[object]) -> str:
    text = "\n".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_columns(df: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")
