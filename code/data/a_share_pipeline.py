#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from aitrader_paths import DATASETS_ROOT, RAW_MARKET_DATA_DIR


LOG = logging.getLogger("a_share_pipeline")
DEFAULT_OUTPUT_DIR = DATASETS_ROOT
DEFAULT_RAW_DIR = RAW_MARKET_DATA_DIR

DATE_CANDIDATES = ("trade_date", "date", "datetime", "publish_time", "发布时间", "日期", "时间")
CODE_CANDIDATES = ("stock_code", "code", "ticker", "symbol", "ts_code", "wind_code", "证券代码", "股票代码")
NAME_CANDIDATES = ("stock_name", "name", "security_name", "证券简称", "股票简称", "简称")
PRICE_HINTS = ("price", "kline", "quote", "daily", "行情")
NEWS_HINTS = ("news", "article", "report", "公告", "新闻")
METRIC_HINTS = ("metric",)
MONEYFLOW_HINTS = ("moneyflow",)
STOCK_ST_HINTS = ("stock_st",)

PRICE_ALIASES = {
    "open": ("open", "open_price", "开盘", "开盘价"),
    "high": ("high", "high_price", "最高", "最高价"),
    "low": ("low", "low_price", "最低", "最低价"),
    "close": ("close", "close_price", "收盘", "收盘价"),
    "preclose": ("preclose", "pre_close", "prev_close", "昨收", "昨收价"),
    "vwap": ("vwap", "avg_price", "均价"),
    "volume": ("volume", "vol", "成交量"),
    "amount": ("amount", "turnover", "成交额"),
}
METRIC_ALIASES = {
    "pe_ttm": ("pe_ttm", "pe", "市盈率"),
    "pb": ("pb", "市净率"),
    "roe": ("roe", "净资产收益率"),
}
STATIC_INFO_COLUMNS = {"stock_name", "industry", "market", "area", "list_date"}


class PipelineError(RuntimeError):
    pass


def norm_label(value: Any) -> str:
    return re.sub(r"[\s_\-()（）/]+", "", str(value or "").strip().lower())


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    reverse = {norm_label(col): col for col in columns}
    for candidate in candidates:
        matched = reverse.get(norm_label(candidate))
        if matched:
            return matched
    return None


def norm_code(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    matched = re.search(r"(\d{6})\.(SH|SZ|BJ)$", text)
    if matched:
        return f"{matched.group(1)}.{matched.group(2)}"
    digits = "".join(re.findall(r"\d", str(value)))
    return digits[-6:] if len(digits) >= 6 else (str(value).strip() or None)


def norm_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", "", str(value).strip())
    return text or None


def to_halfwidth(value: str) -> str:
    return value.translate(str.maketrans("０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))


def build_stock_alias_map(price_df: pd.DataFrame) -> dict[str, str]:
    if price_df.empty or "stock_name" not in price_df.columns:
        return {}
    alias_map: dict[str, str] = {}
    pairs = price_df[["stock_code", "stock_name"]].dropna().drop_duplicates()
    for row in pairs.itertuples(index=False):
        code = str(row.stock_code)
        name = norm_name(row.stock_name)
        if not name:
            continue
        for alias in {name, to_halfwidth(name)}:
            if len(alias) >= 3:
                alias_map.setdefault(alias, code)
    return alias_map


def build_stock_code_lookup(price_df: pd.DataFrame) -> dict[str, str]:
    if price_df.empty or "stock_code" not in price_df.columns:
        return {}
    codes = price_df["stock_code"].dropna().drop_duplicates().astype(str)
    return {code[:6]: code for code in codes if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", code)}


def unique_codes(codes: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for code in codes:
        if pd.isna(code):
            continue
        normalized = norm_code(code)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def auto_rename(df: pd.DataFrame, alias_map: dict[str, Sequence[str]]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for target, aliases in alias_map.items():
        source = find_column(df.columns, aliases)
        if source and source != target:
            rename_map[source] = target
    return df.rename(columns=rename_map)


def read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except pd.errors.EmptyDataError:
        LOG.warning("Skipping empty csv file: %s", path)
        return pd.DataFrame()
    except Exception:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            LOG.warning("Skipping empty csv file: %s", path)
            return pd.DataFrame()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        return read_csv_auto(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return pd.DataFrame()
        if text.startswith("["):
            return pd.DataFrame(json.loads(text))
        if text.startswith("{"):
            data = json.loads(text)
            if isinstance(data, dict):
                for key in ("data", "list", "items", "rows"):
                    if isinstance(data.get(key), list):
                        return pd.DataFrame(data[key])
                return pd.DataFrame([data])
        return pd.read_json(io.StringIO(text), lines=True)
    raise PipelineError(f"Unsupported file type: {path}")


def infer_files(raw_dir: Path, hints: Sequence[str]) -> list[Path]:
    if not raw_dir.exists():
        return []
    files = []
    lowered_hints = [hint.lower() for hint in hints]
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        haystack = str(path.relative_to(raw_dir)).lower()
        if any(hint in haystack for hint in lowered_hints):
            files.append(path)
    return files


def load_many(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        LOG.info("Reading %s", path)
        frame = read_table(path)
        if not frame.empty:
            frame["__source_file"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    normalized = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    compact_datetime = text.str.fullmatch(r"\d{14}").fillna(False)
    if compact_datetime.any():
        normalized.loc[compact_datetime] = pd.to_datetime(text[compact_datetime], format="%Y%m%d%H%M%S", errors="coerce")

    compact_date = text.str.fullmatch(r"\d{8}").fillna(False)
    if compact_date.any():
        normalized.loc[compact_date] = pd.to_datetime(text[compact_date], format="%Y%m%d", errors="coerce")

    remaining = normalized.isna() & text.notna() & (text != "")
    if remaining.any():
        normalized.loc[remaining] = pd.to_datetime(series[remaining], errors="coerce")

    return normalized.dt.tz_localize(None)


def numericize(df: pd.DataFrame, skip: Sequence[str]) -> pd.DataFrame:
    for column in df.columns:
        if column in skip or pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        try:
            df[column] = pd.to_numeric(df[column])
        except (TypeError, ValueError):
            pass
    return df


def build_sample_id(stock_code: pd.Series, trade_date: pd.Series) -> pd.Series:
    code_text = stock_code.astype("string").fillna("").astype(str)
    date_text = pd.to_datetime(trade_date, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return code_text + "_" + date_text


def deduplicate_by_keys(df: pd.DataFrame, keys: Sequence[str], name: str) -> tuple[pd.DataFrame, int]:
    usable_keys = [key for key in keys if key in df.columns]
    if df.empty or not usable_keys:
        return df, 0
    before = len(df)
    sort_cols = usable_keys + [col for col in ("publish_time", "__source_file") if col in df.columns]
    deduped = (
        df.sort_values(sort_cols, kind="stable")
        .drop_duplicates(subset=usable_keys, keep="last")
        .reset_index(drop=True)
    )
    removed = before - len(deduped)
    if removed:
        LOG.warning("%s detected %d duplicate rows on keys=%s; kept the last row.", name, removed, usable_keys)
    return deduped, removed


def filter_bse(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty or "stock_code" not in df.columns:
        return df, 0
    mask = df["stock_code"].astype("string").str.endswith(".BJ").fillna(False)
    if "market" in df.columns:
        mask = mask | df["market"].astype("string").str.contains("北交", na=False)
    removed = int(mask.sum())
    if not removed:
        return df, 0
    return df.loc[~mask].reset_index(drop=True), removed


def st_key_set(st_df: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    if st_df.empty:
        return set()
    return {
        (str(row.stock_code), pd.Timestamp(row.trade_date).normalize())
        for row in st_df[["stock_code", "trade_date"]].dropna().itertuples(index=False)
    }


def filter_st_rows(df: pd.DataFrame, st_keys: set[tuple[str, pd.Timestamp]], date_col: str = "trade_date") -> tuple[pd.DataFrame, int]:
    if df.empty or not st_keys or "stock_code" not in df.columns or date_col not in df.columns:
        return df, 0
    keys = list(zip(df["stock_code"].astype(str), pd.to_datetime(df[date_col], errors="coerce").dt.normalize()))
    mask = pd.Series([key in st_keys for key in keys], index=df.index)
    removed = int(mask.sum())
    if not removed:
        return df, 0
    return df.loc[~mask].reset_index(drop=True), removed


def filter_st_samples(sample_df: pd.DataFrame, st_keys: set[tuple[str, pd.Timestamp]]) -> tuple[pd.DataFrame, int]:
    if sample_df.empty or not st_keys:
        return sample_df, 0
    mask = pd.Series(False, index=sample_df.index)
    for date_col in ("feature_asof_date", "target_trade_date", "label_end_date"):
        if date_col not in sample_df.columns:
            continue
        keys = list(zip(sample_df["stock_code"].astype(str), pd.to_datetime(sample_df[date_col], errors="coerce").dt.normalize()))
        mask = mask | pd.Series([key in st_keys for key in keys], index=sample_df.index)
    removed = int(mask.sum())
    if not removed:
        return sample_df, 0
    return sample_df.loc[~mask].reset_index(drop=True), removed


def prepare_price(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if df.empty:
        raise PipelineError("未找到A股量价数据。")
    df = auto_rename(df.copy(), PRICE_ALIASES)
    date_col = find_column(df.columns, DATE_CANDIDATES)
    code_col = find_column(df.columns, CODE_CANDIDATES)
    name_col = find_column(df.columns, NAME_CANDIDATES)
    if not date_col or not code_col:
        raise PipelineError("量价数据缺少日期列或股票代码列。")
    df = df.rename(columns={date_col: "trade_date", code_col: "stock_code"})
    if name_col:
        df = df.rename(columns={name_col: "stock_name"})
    df["trade_date"] = normalize_dates(df["trade_date"])
    df["stock_code"] = df["stock_code"].map(norm_code)
    if "stock_name" in df.columns:
        df["stock_name"] = df["stock_name"].map(norm_name)
    df = df.dropna(subset=["trade_date", "stock_code"]).reset_index(drop=True)
    df = numericize(df, ["trade_date", "stock_code", "stock_name", "__source_file"])
    df, removed_duplicates = deduplicate_by_keys(df, ["stock_code", "trade_date"], "price_df")
    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    price_cols = [col for col in ("preclose", "close", "open", "high", "low", "vwap") if col in df.columns]
    other_num = [col for col in df.columns if col not in {"trade_date", "stock_code", "stock_name", "__source_file"} and pd.api.types.is_numeric_dtype(df[col])]
    if price_cols:
        df[price_cols] = df.groupby("stock_code")[price_cols].ffill()
    fill_zero = [col for col in ("volume", "amount") if col in df.columns]
    for col in fill_zero:
        df[col] = df[col].fillna(0)
    for col in [col for col in other_num if col not in price_cols and col not in fill_zero]:
        df[col] = df.groupby("stock_code")[col].ffill()
        df[col] = df.groupby("trade_date")[col].transform(lambda s: s.fillna(s.median()))
    if "close" in df.columns:
        for col in ("open", "high", "low", "vwap", "preclose"):
            if col in df.columns:
                df[col] = df[col].fillna(df["close"])
    df["sample_id"] = build_sample_id(df["stock_code"], df["trade_date"])
    return df, df.copy(), removed_duplicates


def prepare_basic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    code_col = find_column(df.columns, ("ts_code", "wind_code", "stock_code", "code", "证券代码", "股票代码"))
    name_col = find_column(df.columns, NAME_CANDIDATES)
    if not code_col:
        return pd.DataFrame()
    keep_cols = [code_col]
    for column in (name_col, "industry", "market", "area", "list_date"):
        if column and column in df.columns and column not in keep_cols:
            keep_cols.append(column)
    basic = df[keep_cols].copy().rename(columns={code_col: "stock_code"})
    if name_col:
        basic = basic.rename(columns={name_col: "stock_name"})
    basic["stock_code"] = basic["stock_code"].map(norm_code)
    if "stock_name" in basic.columns:
        basic["stock_name"] = basic["stock_name"].map(norm_name)
    if "list_date" in basic.columns:
        basic["list_date"] = normalize_dates(basic["list_date"])
    basic = basic.dropna(subset=["stock_code"]).drop_duplicates(subset=["stock_code"], keep="last")
    return basic.reset_index(drop=True)


def prepare_metric(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    df = auto_rename(df.copy(), METRIC_ALIASES)
    date_col = find_column(df.columns, DATE_CANDIDATES)
    code_col = find_column(df.columns, CODE_CANDIDATES)
    if not date_col or not code_col:
        return pd.DataFrame(), 0
    df = df.rename(columns={date_col: "trade_date", code_col: "stock_code"})
    df["trade_date"] = normalize_dates(df["trade_date"])
    df["stock_code"] = df["stock_code"].map(norm_code)
    df = df.dropna(subset=["trade_date", "stock_code"]).reset_index(drop=True)
    df = numericize(df, ["trade_date", "stock_code", "stock_name", "__source_file"])
    df, removed_duplicates = deduplicate_by_keys(df, ["stock_code", "trade_date"], "metric_df")
    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    return df, removed_duplicates


def prepare_moneyflow(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return pd.DataFrame(), 0
    date_col = find_column(df.columns, DATE_CANDIDATES)
    code_col = find_column(df.columns, CODE_CANDIDATES)
    if not date_col or not code_col:
        return pd.DataFrame(), 0
    df = df.copy().rename(columns={date_col: "trade_date", code_col: "stock_code"})
    df["trade_date"] = normalize_dates(df["trade_date"])
    df["stock_code"] = df["stock_code"].map(norm_code)
    df = df.dropna(subset=["trade_date", "stock_code"]).reset_index(drop=True)
    df = numericize(df, ["trade_date", "stock_code", "__source_file"])
    df, removed_duplicates = deduplicate_by_keys(df, ["stock_code", "trade_date"], "moneyflow_df")
    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    return df, removed_duplicates


def attach_basic_fields(df: pd.DataFrame, basic_df: pd.DataFrame, fields: Sequence[str] = ("industry",)) -> pd.DataFrame:
    if df.empty or basic_df.empty or "stock_code" not in df.columns or "stock_code" not in basic_df.columns:
        return df
    available_fields = [field for field in fields if field in basic_df.columns]
    if not available_fields:
        return df
    basic_keep = basic_df[["stock_code", *available_fields]].drop_duplicates("stock_code", keep="last")
    result = df.merge(basic_keep, on="stock_code", how="left", suffixes=("", "_basic"))
    for field in available_fields:
        basic_field = f"{field}_basic"
        if basic_field not in result.columns:
            continue
        if field in df.columns:
            result[field] = result[field].where(result[field].notna(), result[basic_field])
        else:
            result[field] = result[basic_field]
        result = result.drop(columns=[basic_field])
    return result


def prepare_stock_st(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return pd.DataFrame(), 0
    date_col = find_column(df.columns, DATE_CANDIDATES)
    code_col = find_column(df.columns, CODE_CANDIDATES)
    name_col = find_column(df.columns, NAME_CANDIDATES)
    if not date_col or not code_col:
        return pd.DataFrame(), 0
    df = df.copy().rename(columns={date_col: "trade_date", code_col: "stock_code"})
    if name_col:
        df = df.rename(columns={name_col: "stock_name"})
    df["trade_date"] = normalize_dates(df["trade_date"])
    df["stock_code"] = df["stock_code"].map(norm_code)
    if "stock_name" in df.columns:
        df["stock_name"] = df["stock_name"].map(norm_name)
    df = df.dropna(subset=["trade_date", "stock_code"]).reset_index(drop=True)
    keep = [col for col in ("stock_code", "stock_name", "trade_date", "type", "type_name", "__source_file") if col in df.columns]
    df = df[keep].copy()
    df, removed_duplicates = deduplicate_by_keys(df, ["stock_code", "trade_date"], "stock_st_df")
    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    return df, removed_duplicates


def next_trade_day(day: pd.Timestamp, trade_days: list[pd.Timestamp]) -> pd.Timestamp | pd.NaT:
    position = bisect_left(trade_days, day)
    return trade_days[position] if position < len(trade_days) else pd.NaT


def prepare_news(df: pd.DataFrame, price_df: pd.DataFrame, cutoff_hour: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0
    date_col = find_column(df.columns, DATE_CANDIDATES)
    code_col = find_column(df.columns, CODE_CANDIDATES)
    name_col = find_column(df.columns, NAME_CANDIDATES)
    if not date_col:
        raise PipelineError("新闻数据缺少发布时间列。")
    df = df.copy().rename(columns={date_col: "publish_time"})
    if code_col:
        df = df.rename(columns={code_col: "stock_code"})
    if name_col:
        df = df.rename(columns={name_col: "stock_name"})
    df["publish_time"] = normalize_dates(df["publish_time"])
    if "stock_code" not in df.columns:
        df["stock_code"] = None
    df["stock_code"] = df["stock_code"].map(norm_code)
    if "stock_name" in df.columns:
        df["stock_name"] = df["stock_name"].map(norm_name)
    alias = build_stock_alias_map(price_df)
    code_lookup = build_stock_code_lookup(price_df)
    text_keys = {norm_label(x) for x in ("title", "headline", "content", "summary", "text", "标题", "内容", "摘要")}
    text_cols = [col for col in df.columns if norm_label(col) in text_keys]
    if not text_cols:
        text_cols = [col for col in df.columns if df[col].dtype == "object"][:2]
    title_keys = {norm_label(x) for x in ("title", "headline", "标题")}
    title_cols = [col for col in df.columns if norm_label(col) in title_keys]
    df["news_text"] = df[text_cols].fillna("").astype(str).agg("\n".join, axis=1)
    title_text = df[title_cols].fillna("").astype(str).agg("\n".join, axis=1) if title_cols else df["news_text"]
    if "stock_name" in df.columns:
        df["stock_code"] = df["stock_code"].fillna(df["stock_name"].map(alias))
    matched_from_code = pd.Series([[] for _ in range(len(df))], index=df.index)
    if code_lookup:
        matched_from_code = df["news_text"].str.findall(r"(?<!\d)(\d{6})(?:\.(?:SH|SZ|BJ))?(?!\d)").map(
            lambda codes: unique_codes(code_lookup[code] for code in codes if code in code_lookup)
        )
    matched_from_name = pd.Series([[] for _ in range(len(df))], index=df.index)
    if alias:
        pattern = re.compile("|".join(re.escape(name) for name in sorted(alias, key=len, reverse=True)))
        matched_from_name = title_text.str.findall(pattern).map(lambda names: unique_codes(alias[name] for name in names))
    explicit_codes = df["stock_code"].map(lambda code: unique_codes([code]))
    df["_matched_stock_codes"] = [
        unique_codes([*explicit, *code_matched, *name_matched])
        for explicit, code_matched, name_matched in zip(explicit_codes, matched_from_code, matched_from_name)
    ]
    df["stock_code"] = df["_matched_stock_codes"].map(lambda codes: codes[0] if codes else pd.NA)
    df["matched_stock_codes"] = df["_matched_stock_codes"].map(lambda codes: "|".join(codes))
    df["matched_stock_count"] = df["_matched_stock_codes"].map(len)
    trade_days = sorted(price_df["trade_date"].dt.normalize().dropna().unique().tolist())

    def _map_trade_day(ts: pd.Timestamp) -> pd.Timestamp | pd.NaT:
        if pd.isna(ts):
            return pd.NaT
        base = ts.normalize() + pd.Timedelta(days=1) if ts.hour >= cutoff_hour else ts.normalize()
        return next_trade_day(base, trade_days)

    df["trade_date"] = df["publish_time"].map(_map_trade_day)
    df = df.dropna(subset=["trade_date"]).reset_index(drop=True)
    detail_cols = [col for col in ("stock_code", "matched_stock_codes", "matched_stock_count", "trade_date", "publish_time", "news_text", "stock_name", "__source_file") if col in df.columns]
    detail = df[detail_cols].copy()
    detail, detail_removed_duplicates = deduplicate_by_keys(
        detail,
        ["trade_date", "publish_time", "news_text"],
        "news_detail_df",
    )
    market_daily = (
        detail.groupby(["trade_date"], dropna=False)
        .agg(
            market_news_count=("news_text", "size"),
            market_news_text_concat=("news_text", lambda values: "\n\n".join([v for v in values if isinstance(v, str) and v])),
            market_latest_publish_time=("publish_time", "max"),
        )
        .reset_index()
    )
    market_daily = pd.DataFrame({"trade_date": trade_days}).merge(market_daily, on="trade_date", how="left")
    market_daily["market_news_count"] = market_daily["market_news_count"].fillna(0).astype(int)
    market_daily["market_news_text_concat"] = market_daily["market_news_text_concat"].fillna("")

    stock_detail_cols = [col for col in ("_matched_stock_codes", "trade_date", "publish_time", "news_text", "stock_name", "__source_file") if col in df.columns]
    stock_detail = df[df["_matched_stock_codes"].map(bool)][stock_detail_cols].copy()
    if stock_detail.empty:
        return detail, pd.DataFrame(), market_daily, detail_removed_duplicates

    stock_detail = stock_detail.explode("_matched_stock_codes").rename(columns={"_matched_stock_codes": "stock_code"})
    stock_detail, _ = deduplicate_by_keys(stock_detail, ["stock_code", "trade_date", "publish_time", "news_text"], "news_stock_detail_df")
    stock_detail["sample_id"] = build_sample_id(stock_detail["stock_code"], stock_detail["trade_date"])
    stock_daily = (
        stock_detail.groupby(["sample_id", "stock_code", "trade_date"], dropna=False)
        .agg(
            news_count=("news_text", "size"),
            news_text_concat=("news_text", lambda values: "\n\n".join([v for v in values if isinstance(v, str) and v])),
            latest_publish_time=("publish_time", "max"),
        )
        .reset_index()
    )
    stock_daily, _ = deduplicate_by_keys(stock_daily, ["sample_id"], "news_daily_df")
    return detail, stock_daily, market_daily, detail_removed_duplicates


def build_prediction_samples(panel_df: pd.DataFrame) -> pd.DataFrame:
    if panel_df.empty:
        return panel_df
    df = panel_df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True).copy()
    grouped = df.groupby("stock_code")
    open_t = grouped["open"].shift(-1) if "open" in df.columns else pd.Series(pd.NA, index=df.index)
    open_t1 = grouped["open"].shift(-2) if "open" in df.columns else pd.Series(pd.NA, index=df.index)
    vwap_t = grouped["vwap"].shift(-1) if "vwap" in df.columns else pd.Series(pd.NA, index=df.index)
    vwap_t1 = grouped["vwap"].shift(-2) if "vwap" in df.columns else pd.Series(pd.NA, index=df.index)

    df["feature_asof_date"] = df["trade_date"]
    df["target_trade_date"] = grouped["trade_date"].shift(-1)
    df["label_start_date"] = df["target_trade_date"]
    df["label_end_date"] = grouped["trade_date"].shift(-2)
    if "open" in df.columns:
        df["label_next_open_return"] = open_t1 / open_t - 1
    if "vwap" in df.columns:
        df["label_next_vwap_return"] = vwap_t1 / vwap_t - 1
    df = df.dropna(subset=["target_trade_date", "label_end_date"]).reset_index(drop=True)
    df["trade_date"] = df["target_trade_date"]
    df["decision_ts"] = pd.to_datetime(df["target_trade_date"]).dt.normalize() + pd.Timedelta(hours=9, minutes=25)
    df["sample_id"] = build_sample_id(df["stock_code"], df["target_trade_date"])
    return df


def save_outputs(
    basic_df: pd.DataFrame,
    price_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    moneyflow_df: pd.DataFrame,
    stock_st_df: pd.DataFrame,
    news_detail_df: pd.DataFrame,
    news_daily_df: pd.DataFrame,
    market_news_daily_df: pd.DataFrame,
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = output_dir / "processed"
    meta = output_dir / "meta"
    processed.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    for path in processed.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
    for path in meta.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
    if not basic_df.empty:
        basic_df.to_parquet(processed / "basic.parquet", index=False)
    price_df.to_parquet(processed / "price.parquet", index=False)
    panel_df.to_parquet(processed / "panel.parquet", index=False)
    sample_df.to_parquet(processed / "samples.parquet", index=False)
    if not metric_df.empty:
        metric_df.to_parquet(processed / "metric.parquet", index=False)
    if not moneyflow_df.empty:
        moneyflow_df.to_parquet(processed / "moneyflow.parquet", index=False)
    if not stock_st_df.empty:
        stock_st_df.to_parquet(processed / "stock_st.parquet", index=False)
    news_detail_df.to_parquet(processed / "news.parquet", index=False)
    news_daily_df.to_parquet(processed / "news_stock_daily.parquet", index=False)
    market_news_daily_df.to_parquet(processed / "news_market_daily.parquet", index=False)
    sample_df.head(2000).to_csv(processed / "samples_preview.csv", index=False)
    with (meta / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share preprocessing pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--price-file", action="append", default=[])
    parser.add_argument("--news-file", action="append", default=[])
    parser.add_argument("--metric-file", action="append", default=[])
    parser.add_argument("--moneyflow-file", action="append", default=[])
    parser.add_argument("--stock-st-file", action="append", default=[])
    parser.add_argument("--basic-file", action="append", default=[])
    parser.add_argument("--news-cutoff-hour", type=int, default=15)
    parser.add_argument("--log-level", default="INFO")
    return parser


def run(args: argparse.Namespace) -> None:
    price_files = [Path(item) for item in args.price_file] or infer_files(args.raw_dir, PRICE_HINTS)
    news_files = [Path(item) for item in args.news_file] or infer_files(args.raw_dir, NEWS_HINTS)
    metric_files = [Path(item) for item in args.metric_file] or infer_files(args.raw_dir, METRIC_HINTS)
    moneyflow_files = [Path(item) for item in args.moneyflow_file] or infer_files(args.raw_dir, MONEYFLOW_HINTS)
    stock_st_files = [Path(item) for item in args.stock_st_file] or infer_files(args.raw_dir, STOCK_ST_HINTS)
    basic_files = [Path(item) for item in args.basic_file]
    if not basic_files:
        default_basic = args.raw_dir / "A股数据" / "basic.csv"
        basic_files = [default_basic] if default_basic.exists() else []
    if not price_files:
        raise PipelineError(f"没有检测到量价文件，请把原始文件放到 {DEFAULT_RAW_DIR} 或显式传 --price-file。")

    price_df, _, price_duplicate_rows_removed = prepare_price(load_many(price_files))
    basic_df = prepare_basic(load_many(basic_files)) if basic_files else pd.DataFrame()
    if not basic_df.empty:
        price_df = attach_basic_fields(price_df, basic_df, fields=("stock_name", "industry", "market", "area", "list_date"))
    price_df, price_bse_rows_removed = filter_bse(price_df)

    metric_df, metric_duplicate_rows_removed = prepare_metric(load_many(metric_files)) if metric_files else (pd.DataFrame(), 0)
    moneyflow_df, moneyflow_duplicate_rows_removed = prepare_moneyflow(load_many(moneyflow_files)) if moneyflow_files else (pd.DataFrame(), 0)
    stock_st_df, stock_st_duplicate_rows_removed = prepare_stock_st(load_many(stock_st_files)) if stock_st_files else (pd.DataFrame(), 0)
    metric_df = attach_basic_fields(metric_df, basic_df, fields=("industry",))
    moneyflow_df = attach_basic_fields(moneyflow_df, basic_df, fields=("industry",))
    metric_df, metric_bse_rows_removed = filter_bse(metric_df)
    moneyflow_df, moneyflow_bse_rows_removed = filter_bse(moneyflow_df)

    panel_df_raw = price_df.copy()
    if not metric_df.empty:
        keep = [col for col in metric_df.columns if col not in STATIC_INFO_COLUMNS | {"__source_file"}]
        panel_df_raw = panel_df_raw.merge(metric_df[keep], on=["trade_date", "stock_code"], how="left")
    if not moneyflow_df.empty:
        keep = [col for col in moneyflow_df.columns if col not in STATIC_INFO_COLUMNS | {"__source_file"}]
        panel_df_raw = panel_df_raw.merge(moneyflow_df[keep], on=["trade_date", "stock_code"], how="left")

    sample_df = build_prediction_samples(panel_df_raw)
    stock_st_keys = st_key_set(stock_st_df)
    price_df, price_st_rows_removed = filter_st_rows(price_df, stock_st_keys)
    metric_df, metric_st_rows_removed = filter_st_rows(metric_df, stock_st_keys)
    moneyflow_df, moneyflow_st_rows_removed = filter_st_rows(moneyflow_df, stock_st_keys)
    panel_df, panel_st_rows_removed = filter_st_rows(panel_df_raw, stock_st_keys)
    sample_df, sample_st_rows_removed = filter_st_samples(sample_df, stock_st_keys)
    news_detail_df, news_daily_df, market_news_daily_df, news_detail_duplicate_rows_removed = (
        prepare_news(load_many(news_files), price_df, args.news_cutoff_hour) if news_files else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0)
    )
    if not news_daily_df.empty:
        sample_df = sample_df.merge(news_daily_df, on=["sample_id", "stock_code", "trade_date"], how="left")
        sample_df["news_count"] = sample_df["news_count"].fillna(0).astype(int)
    if not market_news_daily_df.empty:
        market_keep = [col for col in market_news_daily_df.columns if col != "market_news_text_concat"]
        sample_df = sample_df.merge(market_news_daily_df[market_keep], on="trade_date", how="left")
        sample_df["market_news_count"] = sample_df["market_news_count"].fillna(0).astype(int)
    sample_df, sample_duplicate_rows_removed = deduplicate_by_keys(sample_df, ["sample_id"], "sample_df")
    sample_df = sample_df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    summary = {
        "price_files": [str(path) for path in price_files],
        "news_files": [str(path) for path in news_files],
        "metric_files": [str(path) for path in metric_files],
        "moneyflow_files": [str(path) for path in moneyflow_files],
        "stock_st_files": [str(path) for path in stock_st_files],
        "basic_files": [str(path) for path in basic_files],
        "basic_rows": int(len(basic_df)),
        "price_rows": int(len(price_df)),
        "panel_rows": int(len(panel_df)),
        "sample_rows": int(len(sample_df)),
        "news_rows": int(len(news_detail_df)),
        "news_daily_rows": int(len(news_daily_df)),
        "market_news_daily_rows": int(len(market_news_daily_df)),
        "metric_rows": int(len(metric_df)),
        "moneyflow_rows": int(len(moneyflow_df)),
        "stock_st_rows": int(len(stock_st_df)),
        "price_duplicate_rows_removed": int(price_duplicate_rows_removed),
        "metric_duplicate_rows_removed": int(metric_duplicate_rows_removed),
        "moneyflow_duplicate_rows_removed": int(moneyflow_duplicate_rows_removed),
        "stock_st_duplicate_rows_removed": int(stock_st_duplicate_rows_removed),
        "price_bse_rows_removed": int(price_bse_rows_removed),
        "metric_bse_rows_removed": int(metric_bse_rows_removed),
        "moneyflow_bse_rows_removed": int(moneyflow_bse_rows_removed),
        "price_st_rows_removed": int(price_st_rows_removed),
        "metric_st_rows_removed": int(metric_st_rows_removed),
        "moneyflow_st_rows_removed": int(moneyflow_st_rows_removed),
        "panel_st_rows_removed": int(panel_st_rows_removed),
        "sample_st_rows_removed": int(sample_st_rows_removed),
        "news_detail_duplicate_rows_removed": int(news_detail_duplicate_rows_removed),
        "sample_duplicate_rows_removed": int(sample_duplicate_rows_removed),
        "output_dir": str(args.output_dir),
    }
    save_outputs(
        basic_df,
        price_df,
        panel_df,
        sample_df,
        metric_df,
        moneyflow_df,
        stock_st_df,
        news_detail_df,
        news_daily_df,
        market_news_daily_df,
        args.output_dir,
        summary,
    )
    LOG.info("All outputs were written to %s", args.output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        run(args)
    except PipelineError as exc:
        LOG.error("%s", exc)
        return 2
    except Exception:
        LOG.exception("Unexpected failure.")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
