from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from FactorMiner.pools.news_llm import prepare_news_items
from model.msgca.feature_set import load_feature_blocks, load_sample_feature_matrix


CONTEXT_COLUMNS = (
    "context_roc3",
    "context_roc5",
    "context_roc20",
    "context_roc60",
    "context_rsv20",
    "context_volume_ratio",
    "context_industry_relative20",
    "context_industry_relative60",
    "context_industry_relative",
    "context_regime_momentum",
    "context_moneyflow_confirm",
    "context_news_exact3",
    "context_news_exact5",
    "context_news_exact",
    "context_log_total_mv",
    "context_broken_ma",
    "context_moderate_pullback",
    "context_tr",
    "context_mf",
    "context_industry",
    "context_regime",
    "context_news",
    "context_cap",
    "context_oh",
    "context_br",
    "context_h",
    "context_hp",
    "context_theme_peer_count",
    "context_theme_peer_ret20",
    "context_theme_peer_ret60",
    "context_theme_peer_pos20",
    "context_theme_peer_mf",
    "context_theme_escape20",
    "context_theme_escape60",
    "context_theme_strength",
    "context_theme_h",
    "context_theme_hp",
    "context_cluster_id",
    "context_cluster_size",
    "context_cluster_strength",
    "context_cluster_mf",
    "context_cluster_hp",
)

REGISTERED_CONTEXT_FEATURES = (
    "mf_price_flow_confirm5",
    "gpt_rg_mom20_turnover_delta5",
    "gpt_rg_mom10_amount_delta5",
    "gpt_rg_indmom20_amount_delta10",
)


def attach_context_features(
    samples: pd.DataFrame,
    *,
    samples_path: str | Path | None = None,
    price_path: str | Path,
    metric_path: str | Path,
    feature_registry_path: str | Path,
    news_path: str | Path | None = None,
    news_scores_path: str | Path | None = None,
    context_cache_path: str | Path | None = None,
    news_cache_path: str | Path | None = None,
    context_columns: Sequence[str] | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Attach deployment context variables for MSGCA loss/score.

    The formulas intentionally use only variables approved in
    report/msgca_strict_variable_audit.md. Missing required inputs fail fast
    when ``strict=True``.
    """
    if samples.empty:
        return _ensure_columns(samples.copy(), context_columns or CONTEXT_COLUMNS)
    requested = list(dict.fromkeys(context_columns or CONTEXT_COLUMNS))
    work = _ensure_sample_metadata(samples.copy(), samples_path=samples_path)
    _require_columns(
        work,
        ("sample_id", "stock_code", "industry", "feature_asof_date", "target_trade_date", "decision_ts"),
        "samples",
    )
    work["sample_id"] = work["sample_id"].astype(str)
    work["stock_code"] = work["stock_code"].astype(str)
    work["industry"] = work["industry"].fillna("").astype(str)
    work["feature_asof_date"] = pd.to_datetime(work["feature_asof_date"], errors="coerce").dt.normalize()
    work["target_trade_date"] = pd.to_datetime(work["target_trade_date"], errors="coerce").dt.normalize()
    work["decision_ts"] = pd.to_datetime(work["decision_ts"], errors="coerce")

    if context_cache_path is not None and Path(context_cache_path).exists():
        _context_log(f"cache_read_start path={Path(context_cache_path).name} rows={len(work)} columns={len(requested)}")
        cached = _read_context_cache(Path(context_cache_path), work, requested, strict=False)
        cached_columns = set(cached.columns)
        missing_cached = [column for column in requested if column not in cached_columns]
        if not missing_cached:
            out = _attach_cached_columns(work, cached, requested)
            if strict:
                _validate_context_coverage(out, requested)
            _context_log(f"cache_read_done path={Path(context_cache_path).name} rows={len(out)} columns={len(requested)}")
            return out
        _context_log(
            "cache_missing_columns_rebuild "
            f"path={Path(context_cache_path).name} missing={','.join(missing_cached[:12])}"
        )

    meta_columns = ["sample_id", "stock_code", "industry", "feature_asof_date", "target_trade_date", "decision_ts"]
    if "__row_pos" in work.columns:
        meta_columns.append("__row_pos")
    context = work[meta_columns].copy()
    _context_log(f"price_context_start rows={len(context)}")
    context = _attach_price_context(context, price_path=price_path, metric_path=metric_path)
    _context_log(f"price_context_done rows={len(context)}")
    _context_log(f"registered_context_start rows={len(context)}")
    context = _attach_registered_context(context, feature_registry_path=feature_registry_path, strict=strict)
    _context_log(f"registered_context_done rows={len(context)}")
    if _needs_news_exact(requested):
        _context_log(f"news_exact_start rows={len(context)}")
        context = _attach_news_exact_context(
            context,
            samples_path=samples_path,
            feature_registry_path=feature_registry_path,
            news_path=news_path,
            news_scores_path=news_scores_path,
            news_cache_path=news_cache_path,
            strict=strict,
        )
        _context_log(f"news_exact_done rows={len(context)}")
    else:
        context["context_news_exact3"] = np.nan
        context["context_news_exact5"] = np.nan

    _context_log(f"derive_start rows={len(context)}")
    context = _derive_context(context)
    _context_log(f"derive_done rows={len(context)}")
    missing = [column for column in requested if column not in context.columns]
    if missing:
        raise KeyError(f"Context builder did not create requested columns: {missing}")
    out = _attach_cached_columns(work, context[["sample_id", *requested]], requested)
    if strict:
        _validate_context_coverage(out, requested)
    return out


def _attach_price_context(samples: pd.DataFrame, *, price_path: str | Path, metric_path: str | Path) -> pd.DataFrame:
    start = samples["feature_asof_date"].min()
    end = samples["feature_asof_date"].max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("feature_asof_date is missing for strict context")
    _context_log(f"price_load_start range={pd.Timestamp(start).date()}:{pd.Timestamp(end).date()}")
    price = _load_price(price_path, start=start - pd.Timedelta(days=160), end=end)
    _context_log(f"price_load_done rows={len(price)}")
    by_stock = price.groupby("stock_code", sort=False)
    for window in (3, 5, 20, 60):
        price[f"context_roc{window}"] = by_stock["close"].transform(lambda s, w=window: s / s.shift(w) - 1.0)
        _context_log(f"price_roc_done window={window}")
    price["context_ma20"] = by_stock["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    price["context_ma60"] = by_stock["close"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    _context_log("price_ma_done")
    high20 = by_stock["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    low20 = by_stock["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    price["context_rsv20"] = (price["close"] - low20) / (high20 - low20).replace(0.0, np.nan)
    price["context_broken_ma"] = (price["close"].lt(price["context_ma20"]) & price["close"].lt(price["context_ma60"])).astype("float64")
    _context_log("price_rsv_broken_done")

    industry_mean = (
        price.groupby(["industry", "trade_date"], sort=True)[["context_roc20", "context_roc60"]]
        .mean()
        .rename(columns={"context_roc20": "__industry_roc20", "context_roc60": "__industry_roc60"})
        .reset_index()
    )
    price = price.merge(industry_mean, on=["industry", "trade_date"], how="left")
    price["context_industry_relative20"] = price["context_roc20"] - price["__industry_roc20"]
    price["context_industry_relative60"] = price["context_roc60"] - price["__industry_roc60"]
    price["context_industry_relative"] = 0.5 * price["context_industry_relative20"] + 0.5 * price["context_industry_relative60"]
    _context_log("price_industry_relative_done")

    keep = [
        "stock_code",
        "trade_date",
        "context_roc3",
        "context_roc5",
        "context_roc20",
        "context_roc60",
        "context_rsv20",
        "context_broken_ma",
        "context_industry_relative20",
        "context_industry_relative60",
        "context_industry_relative",
    ]
    out = samples.merge(
        price[keep],
        left_on=["stock_code", "feature_asof_date"],
        right_on=["stock_code", "trade_date"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    del price
    _context_log("price_sample_merge_done")

    _context_log("metric_load_start")
    metric = _load_metric(metric_path, start=start, end=end)
    _context_log(f"metric_load_done rows={len(metric)}")
    out = out.merge(
        metric,
        left_on=["stock_code", "feature_asof_date"],
        right_on=["stock_code", "trade_date"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    _context_log("metric_sample_merge_done")
    out["context_volume_ratio"] = pd.to_numeric(out.get("volume_ratio"), errors="coerce")
    out["context_log_total_mv"] = np.log(pd.to_numeric(out.get("circ_mv"), errors="coerce").clip(lower=1.0))
    return out.drop(columns=["volume_ratio", "total_mv", "circ_mv"], errors="ignore")


def _read_context_cache(
    path: Path,
    samples: pd.DataFrame,
    requested: Sequence[str],
    *,
    strict: bool,
) -> pd.DataFrame:
    columns = ["sample_id", *list(dict.fromkeys(requested))]
    available = set(pq.ParquetFile(path).schema.names)
    missing = [column for column in columns if column not in available]
    if missing:
        if strict:
            raise KeyError(f"Strict context cache missing columns: {missing}")
        columns = [column for column in columns if column in available]

    positions = samples["__row_pos"].to_numpy(dtype=np.int64, copy=False) if "__row_pos" in samples.columns else None
    if positions is not None and positions.size and np.all(positions[1:] >= positions[:-1]):
        frame = _read_context_cache_by_positions(path, columns, positions)
    else:
        sample_ids = set(samples["sample_id"].astype(str))
        frame = pd.read_parquet(path, columns=columns)
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame = frame.loc[frame["sample_id"].isin(sample_ids)].copy()

    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame


def _attach_cached_columns(samples: pd.DataFrame, cached: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = samples.drop(columns=[column for column in columns if column in samples.columns], errors="ignore").copy()
    if len(cached) == len(out) and np.array_equal(cached["sample_id"].astype(str).to_numpy(), out["sample_id"].astype(str).to_numpy()):
        for column in columns:
            if column in cached.columns:
                out[column] = cached[column].to_numpy()
        return out
    return out.merge(cached[["sample_id", *[column for column in columns if column in cached.columns]]], on="sample_id", how="left")


def _read_context_cache_by_positions(path: Path, columns: Sequence[str], positions: np.ndarray) -> pd.DataFrame:
    row_count = pq.ParquetFile(path).metadata.num_rows
    valid = positions < int(row_count)
    if not bool(valid.any()):
        return pd.DataFrame({column: [np.nan] * len(positions) for column in columns})
    valid_positions = positions[valid]
    start = int(positions[0])
    stop = min(int(valid_positions[-1]) + 1, int(row_count))
    local_positions = valid_positions - start
    try:
        import polars as pl

        frame = pl.scan_parquet(str(path)).select(list(columns)).slice(start, stop - start).collect()
        partial = frame[local_positions].to_pandas()
        data: dict[str, object] = {}
        for column_name in columns:
            values = np.empty(len(positions), dtype=object)
            values[:] = np.nan
            values[valid] = partial[column_name].to_numpy()
            data[column_name] = values
        return pd.DataFrame(data)
    except Exception as exc:
        _context_log(f"cache_polars_fallback path={path.name} reason={type(exc).__name__}:{exc}")
        table = pq.read_table(path, columns=list(columns), use_threads=True, memory_map=True)
        data: dict[str, np.ndarray] = {}
        for column_name in columns:
            column = table[column_name].combine_chunks().to_numpy(zero_copy_only=False)
            values = np.empty(len(positions), dtype=object)
            values[:] = np.nan
            values[valid] = column[valid_positions]
            data[column_name] = values
        return pd.DataFrame(data)


def _attach_registered_context(samples: pd.DataFrame, *, feature_registry_path: str | Path, strict: bool) -> pd.DataFrame:
    if "__row_pos" in samples.columns:
        positions = samples["__row_pos"].to_numpy(dtype=np.int64, copy=False)
        if positions.size and np.all(positions[1:] >= positions[:-1]):
            feature_frame = _load_registered_context_by_positions(
                feature_registry_path,
                REGISTERED_CONTEXT_FEATURES,
                positions,
                samples["sample_id"],
                strict=strict,
            )
        else:
            matrix = load_sample_feature_matrix(
                feature_registry_path,
                REGISTERED_CONTEXT_FEATURES,
                sample_ids=samples["sample_id"],
                strict=strict,
            )
            feature_frame = pd.DataFrame(matrix.values, columns=matrix.columns)
            feature_frame.insert(0, "sample_id", samples["sample_id"].to_numpy())
    else:
        matrix = load_sample_feature_matrix(
            feature_registry_path,
            REGISTERED_CONTEXT_FEATURES,
            sample_ids=samples["sample_id"],
            strict=strict,
        )
        feature_frame = pd.DataFrame(matrix.values, columns=matrix.columns)
        feature_frame.insert(0, "sample_id", samples["sample_id"].to_numpy())
    out = samples.merge(feature_frame, on="sample_id", how="left")
    moneyflow_path = Path(feature_registry_path).parent.parent / "processed" / "moneyflow.parquet"
    out["context_moneyflow_confirm"] = _processed_moneyflow_confirm(out, moneyflow_path)
    if strict and pd.to_numeric(out["context_moneyflow_confirm"], errors="coerce").notna().mean() <= 0.0:
        raise ValueError(f"Strict moneyflow context has no finite coverage: {moneyflow_path}")
    out["context_regime_momentum"] = (
        pd.to_numeric(out["gpt_rg_mom20_turnover_delta5"], errors="coerce")
        + pd.to_numeric(out["gpt_rg_mom10_amount_delta5"], errors="coerce")
        + pd.to_numeric(out["gpt_rg_indmom20_amount_delta10"], errors="coerce")
    )
    if pd.to_numeric(out["context_regime_momentum"], errors="coerce").notna().mean() <= 0.0:
        if strict:
            raise ValueError("Strict regime context has no finite coverage from registered features")
        out["context_regime_momentum"] = np.nan
    return out.drop(columns=list(REGISTERED_CONTEXT_FEATURES), errors="ignore")


def _processed_moneyflow_confirm(samples: pd.DataFrame, moneyflow_path: Path) -> pd.Series:
    result = pd.Series(np.nan, index=samples.index, dtype="float64")
    if not moneyflow_path.exists():
        return result
    columns = [
        "stock_code",
        "trade_date",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]
    available = set(pq.ParquetFile(moneyflow_path).schema.names)
    read_columns = [column for column in columns if column in available]
    if "stock_code" not in read_columns or "trade_date" not in read_columns:
        return result
    start = pd.to_datetime(samples["feature_asof_date"], errors="coerce").min()
    end = pd.to_datetime(samples["feature_asof_date"], errors="coerce").max()
    if pd.isna(start) or pd.isna(end):
        return result
    table = ds.dataset(str(moneyflow_path), format="parquet").to_table(
        columns=read_columns,
        filter=(ds.field("trade_date") >= pd.Timestamp(start)) & (ds.field("trade_date") <= pd.Timestamp(end)),
    )
    flow = table.to_pandas()
    flow["stock_code"] = flow["stock_code"].astype(str)
    flow["trade_date"] = pd.to_datetime(flow["trade_date"], errors="coerce").dt.normalize()
    for column in columns[2:]:
        if column not in flow.columns:
            flow[column] = 0.0
        flow[column] = pd.to_numeric(flow[column], errors="coerce").fillna(0.0)
    large_buy = flow["buy_lg_amount"] + flow["buy_elg_amount"]
    large_sell = flow["sell_lg_amount"] + flow["sell_elg_amount"]
    denom = large_buy.abs() + large_sell.abs() + flow["net_mf_amount"].abs() + 1.0
    flow["context_moneyflow_confirm__raw"] = (large_buy - large_sell + flow["net_mf_amount"]) / denom
    keys = samples[["stock_code", "feature_asof_date"]].copy()
    keys["stock_code"] = keys["stock_code"].astype(str)
    keys["feature_asof_date"] = pd.to_datetime(keys["feature_asof_date"], errors="coerce").dt.normalize()
    merged = keys.merge(
        flow[["stock_code", "trade_date", "context_moneyflow_confirm__raw"]],
        left_on=["stock_code", "feature_asof_date"],
        right_on=["stock_code", "trade_date"],
        how="left",
    )
    return pd.to_numeric(merged["context_moneyflow_confirm__raw"], errors="coerce")


def _load_registered_context_by_positions(
    feature_registry_path: str | Path,
    features: Sequence[str],
    positions: np.ndarray,
    sample_ids: pd.Series,
    *,
    strict: bool,
) -> pd.DataFrame:
    selected = list(dict.fromkeys(str(feature) for feature in features))
    blocks = load_feature_blocks(feature_registry_path)
    feature_to_block = {factor: block for block in blocks for factor in block.factors}
    missing = [feature for feature in selected if feature not in feature_to_block]
    if missing and strict:
        raise KeyError(f"Strict context features missing from feature registry: {missing}")

    values = np.full((len(positions), len(selected)), np.nan, dtype="float32")
    feature_index = {feature: index for index, feature in enumerate(selected)}
    by_block: dict[Path, list[str]] = {}
    for feature in selected:
        block = feature_to_block.get(feature)
        if block is None or not block.factor_path.exists():
            if strict:
                raise FileNotFoundError(f"Missing feature block for strict context feature: {feature}")
            continue
        by_block.setdefault(block.factor_path, []).append(feature)
    for path, columns in by_block.items():
        _context_log(
            "registered_block_start "
            f"path={path.name} columns={len(columns)} rows={len(positions)} "
            f"range={int(positions[0]) if len(positions) else 0}:{int(positions[-1]) if len(positions) else 0}"
        )
        raw_columns = _read_block_columns_by_positions(path, columns, positions)
        for feature in columns:
            values[:, feature_index[feature]] = raw_columns[feature]
        _context_log(f"registered_block_done path={path.name} columns={len(columns)}")
    frame = pd.DataFrame(values, columns=selected)
    frame.insert(0, "sample_id", sample_ids.astype(str).to_numpy())
    return frame


def _read_block_columns_by_positions(path: Path, columns: Sequence[str], positions: np.ndarray) -> dict[str, np.ndarray]:
    if positions.size == 0:
        return {column: np.zeros(0, dtype="float32") for column in columns}
    row_count = pq.ParquetFile(path).metadata.num_rows
    valid = positions < row_count
    if not bool(valid.any()):
        return {column: np.full(len(positions), np.nan, dtype="float32") for column in columns}
    valid_positions = positions[valid]
    start = int(valid_positions[0])
    stop = min(int(valid_positions[-1]) + 1, int(row_count))
    local_positions = valid_positions - start
    try:
        import polars as pl

        frame = (
            pl.scan_parquet(str(path))
            .select([pl.col(column).cast(pl.Float32, strict=False).alias(column) for column in columns])
            .slice(start, stop - start)
            .collect()
        )
        result: dict[str, np.ndarray] = {}
        for column in columns:
            values = np.full(len(positions), np.nan, dtype="float32")
            values[valid] = frame[column].to_numpy().astype("float32", copy=False)[local_positions]
            result[column] = values
        return result
    except Exception as exc:
        _context_log(f"registered_block_polars_fallback path={path.name} reason={type(exc).__name__}:{exc}")
        table = pq.read_table(path, columns=list(columns), use_threads=True, memory_map=True)
        result: dict[str, np.ndarray] = {}
        for column_name in columns:
            column = table[column_name].combine_chunks()
            raw = column.to_numpy(zero_copy_only=False).astype("float32", copy=False)
            values = np.full(len(positions), np.nan, dtype="float32")
            values[valid] = raw[valid_positions]
            result[column_name] = values
        return result


def _context_log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] context_{message}", flush=True)


def _attach_news_exact_context(
    samples: pd.DataFrame,
    *,
    samples_path: str | Path | None,
    feature_registry_path: str | Path | None,
    news_path: str | Path | None,
    news_scores_path: str | Path | None,
    news_cache_path: str | Path | None,
    strict: bool,
) -> pd.DataFrame:
    cache_path = _resolve_news_cache_path(news_cache_path, feature_registry_path=feature_registry_path)
    if cache_path.exists():
        if "__row_pos" in samples.columns:
            positions = samples["__row_pos"].to_numpy(dtype=np.int64, copy=False)
            if positions.size and np.all(positions[1:] >= positions[:-1]):
                news = _read_context_cache_by_positions(
                    cache_path,
                    ["sample_id", "context_news_exact3", "context_news_exact5"],
                    positions,
                )
                if np.array_equal(news["sample_id"].astype(str).to_numpy(), samples["sample_id"].astype(str).to_numpy()):
                    out = samples.copy()
                    out["context_news_exact3"] = pd.to_numeric(news["context_news_exact3"], errors="coerce").to_numpy()
                    out["context_news_exact5"] = pd.to_numeric(news["context_news_exact5"], errors="coerce").to_numpy()
                    return out
        news = _read_news_cache(cache_path, samples["sample_id"])
    else:
        if news_path is None or news_scores_path is None:
            if strict:
                raise ValueError("news_path and news_scores_path are required to build NEWS_exact")
            news = pd.DataFrame({"sample_id": samples["sample_id"], "context_news_exact3": np.nan, "context_news_exact5": np.nan})
        else:
            build_samples = samples
            write_cache = False
            if samples_path is not None and len(samples) > 100_000:
                build_samples = pd.read_parquet(
                    samples_path,
                    columns=["sample_id", "stock_code", "decision_ts"],
                )
                write_cache = True
            news = build_news_exact_sample_block(
                build_samples,
                news_path=news_path,
                news_scores_path=news_scores_path,
            )
            if write_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                news.to_parquet(cache_path, index=False)
            if len(news) != len(samples):
                news = news.loc[news["sample_id"].astype(str).isin(set(samples["sample_id"].astype(str)))]
    news["sample_id"] = news["sample_id"].astype(str)
    return samples.merge(news, on="sample_id", how="left")


def build_news_exact_sample_block(
    samples: pd.DataFrame,
    *,
    news_path: str | Path,
    news_scores_path: str | Path,
) -> pd.DataFrame:
    prepared_samples = samples[["sample_id", "stock_code", "decision_ts"]].copy()
    prepared_samples["sample_id"] = prepared_samples["sample_id"].astype(str)
    prepared_samples["stock_code"] = prepared_samples["stock_code"].astype(str)
    prepared_samples["decision_ts"] = pd.to_datetime(prepared_samples["decision_ts"], errors="coerce")
    prepared_samples = prepared_samples.dropna(subset=["decision_ts"]).reset_index(drop=True)

    news = pd.read_parquet(news_path, columns=["stock_code", "matched_stock_codes", "matched_stock_count", "trade_date", "publish_time", "news_text", "__source_file"])
    items = prepare_news_items(news)
    scores = pd.read_parquet(
        news_scores_path,
        columns=["news_text_hash", "sentiment_score", "impact_score", "risk_score", "relevance_score"],
    )
    scores = scores.drop_duplicates("news_text_hash", keep="last").copy()
    for column in ("sentiment_score", "impact_score", "risk_score", "relevance_score"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores["__news_exact"] = (
        scores["sentiment_score"] * scores["impact_score"] * scores["relevance_score"] - scores["risk_score"]
    )
    scored = items.news_items[["news_id", "news_text_hash", "publish_time"]].merge(
        scores[["news_text_hash", "__news_exact"]],
        on="news_text_hash",
        how="inner",
    )
    stock_events = items.news_stock_map[["news_id", "stock_code"]].merge(scored, on="news_id", how="inner")
    stock_events["stock_code"] = stock_events["stock_code"].astype(str)
    stock_events["publish_time"] = pd.to_datetime(stock_events["publish_time"], errors="coerce")
    stock_events["__news_exact"] = pd.to_numeric(stock_events["__news_exact"], errors="coerce")
    stock_events = stock_events.dropna(subset=["publish_time", "__news_exact"]).sort_values(["stock_code", "publish_time"])

    result = pd.DataFrame(
        {
            "sample_id": prepared_samples["sample_id"].to_numpy(),
            "context_news_exact3": np.nan,
            "context_news_exact5": np.nan,
        }
    ).set_index("sample_id")
    grouped_events = {code: frame for code, frame in stock_events.groupby("stock_code", sort=False)}
    for stock_code, group_samples in prepared_samples.groupby("stock_code", sort=False):
        events = grouped_events.get(str(stock_code))
        if events is None or events.empty:
            continue
        ordered = group_samples.sort_values("decision_ts")
        event_times = events["publish_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        target_times = ordered["decision_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
        values = events["__news_exact"].to_numpy("float64")
        cumulative = np.concatenate(([0.0], np.cumsum(np.nan_to_num(values, nan=0.0))))
        for window in (3, 5):
            window_ns = np.int64(window * 24 * 60 * 60 * 1_000_000_000)
            starts = np.searchsorted(event_times, target_times - window_ns, side="right")
            ends = np.searchsorted(event_times, target_times, side="right")
            counts = ends - starts
            means = np.full(len(ordered), np.nan, dtype="float64")
            mask = counts > 0
            means[mask] = (cumulative[ends[mask]] - cumulative[starts[mask]]) / counts[mask]
            result.loc[ordered["sample_id"].to_numpy(), f"context_news_exact{window}"] = means
    return result.reset_index()


def _derive_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = _derive_context_theme_context(out)
    out["context_news_exact"] = 0.5 * pd.to_numeric(out["context_news_exact3"], errors="coerce") + 0.5 * pd.to_numeric(
        out["context_news_exact5"], errors="coerce"
    )
    out["context_tr"] = _daily_robust_zscore(
        out,
        _weighted_sum(
            out,
            {
                "context_roc20": 0.6,
                "context_roc60": 0.4,
                "context_industry_relative": 0.4,
                "context_regime_momentum": 0.4,
            },
        ),
    )
    out["context_mf"] = _daily_robust_zscore(out, out["context_moneyflow_confirm"])
    out["context_industry"] = _daily_robust_zscore(out, out["context_industry_relative"])
    out["context_regime"] = _daily_robust_zscore(out, out["context_regime_momentum"])
    out["context_news"] = _daily_robust_zscore(out, out["context_news_exact"])
    out["context_cap"] = _daily_robust_zscore(out, out["context_log_total_mv"])
    neg_roc3 = _daily_robust_zscore(out, -pd.to_numeric(out["context_roc3"], errors="coerce")).clip(0.0, 1.5)
    neg_roc5 = _daily_robust_zscore(out, -pd.to_numeric(out["context_roc5"], errors="coerce")).clip(0.0, 1.5)
    roc5 = pd.to_numeric(out["context_roc5"], errors="coerce")
    roc5_floor = roc5.groupby(out["target_trade_date"]).transform(lambda s: s.quantile(0.05))
    out["context_moderate_pullback"] = neg_roc3 * neg_roc5 * roc5.ge(roc5_floor).astype("float64")
    out["context_oh"] = _sigmoid(
        0.8 * _daily_robust_zscore(out, out["context_roc3"])
        + 0.8 * _daily_robust_zscore(out, out["context_roc5"])
        + 0.5 * _daily_robust_zscore(out, out["context_rsv20"])
        + 0.4 * _daily_robust_zscore(out, out["context_volume_ratio"])
        - 0.3 * out["context_mf"]
    )
    out["context_br"] = _sigmoid(
        -0.7 * _daily_robust_zscore(out, out["context_roc20"])
        - 0.5 * _daily_robust_zscore(out, out["context_roc60"])
        - 0.5 * out["context_mf"]
        - 0.4 * out["context_industry"]
        + pd.to_numeric(out["context_broken_ma"], errors="coerce").fillna(0.0)
    )
    trend_t = _sigmoid(
        0.6 * _daily_robust_zscore(out, out["context_roc20"])
        + 0.4 * _daily_robust_zscore(out, out["context_roc60"])
        + 0.4 * out["context_industry"]
        + 0.4 * out["context_regime"]
    )
    out["context_h"] = (
        trend_t
        * pd.to_numeric(out["context_moderate_pullback"], errors="coerce").fillna(0.0)
        * _sigmoid(out["context_mf"])
        * _sigmoid(out["context_industry"])
        * (1.0 - pd.to_numeric(out["context_oh"], errors="coerce").fillna(1.0))
        * (1.0 - pd.to_numeric(out["context_br"], errors="coerce").fillna(1.0))
    )
    out["context_hp"] = _daily_robust_zscore(out, out["context_h"])
    out["context_theme_hp"] = _daily_robust_zscore(out, out["context_theme_h"])
    out = _derive_cluster_context(out)
    return out


def _derive_cluster_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Build same-day dynamic style clusters from visible context.

    The cluster id is intentionally label-free: it uses only as-of price,
    moneyflow, industry-relative, and derived theme/health context. The loss can
    then ask the model to lift strong clusters without leaking future returns
    into the clustering step.
    """
    out = frame.copy()
    dates = pd.to_datetime(out["target_trade_date"], errors="coerce").dt.normalize()
    cluster_raw = pd.DataFrame(
        {
            "__theme": pd.to_numeric(out.get("context_theme_strength"), errors="coerce"),
            "__trend": (
                0.45 * pd.to_numeric(out.get("context_tr"), errors="coerce")
                + 0.25 * pd.to_numeric(out.get("context_industry"), errors="coerce")
                + 0.20 * pd.to_numeric(out.get("context_regime"), errors="coerce")
                + 0.10 * pd.to_numeric(out.get("context_theme_strength"), errors="coerce")
            ),
            "__flow": pd.to_numeric(out.get("context_mf"), errors="coerce"),
            "__risk": (
                0.45 * pd.to_numeric(out.get("context_oh"), errors="coerce")
                + 0.55 * pd.to_numeric(out.get("context_br"), errors="coerce")
                - 0.25 * pd.to_numeric(out.get("context_hp"), errors="coerce")
            ),
        },
        index=out.index,
    )
    theme_bin = _daily_quantile_bin(cluster_raw["__theme"], dates, bins=4)
    trend_bin = _daily_quantile_bin(cluster_raw["__trend"], dates, bins=4)
    flow_bin = _daily_quantile_bin(cluster_raw["__flow"], dates, bins=3)
    risk_bin = _daily_quantile_bin(cluster_raw["__risk"], dates, bins=3)
    valid = dates.notna() & theme_bin.ge(0) & trend_bin.ge(0) & flow_bin.ge(0) & risk_bin.ge(0)
    cluster_id = theme_bin * 36 + trend_bin * 9 + flow_bin * 3 + risk_bin
    out["context_cluster_id"] = cluster_id.where(valid).astype("float64")

    stats_source = pd.DataFrame(
        {
            "__date": dates,
            "__cluster": out["context_cluster_id"],
            "__strength": cluster_raw["__theme"] + cluster_raw["__trend"],
            "__mf": pd.to_numeric(out.get("context_mf"), errors="coerce"),
            "__hp": pd.to_numeric(out.get("context_hp"), errors="coerce"),
        },
        index=out.index,
    ).loc[valid]
    if stats_source.empty:
        out["context_cluster_size"] = np.nan
        out["context_cluster_strength"] = np.nan
        out["context_cluster_mf"] = np.nan
        out["context_cluster_hp"] = np.nan
        return out

    stats = (
        stats_source.groupby(["__date", "__cluster"], sort=False)
        .agg(
            context_cluster_size=("__strength", "size"),
            context_cluster_strength=("__strength", "mean"),
            context_cluster_mf=("__mf", "mean"),
            context_cluster_hp=("__hp", "mean"),
        )
        .reset_index()
    )
    joined = stats_source[["__date", "__cluster"]].merge(stats, on=["__date", "__cluster"], how="left")
    for column in ("context_cluster_size", "context_cluster_strength", "context_cluster_mf", "context_cluster_hp"):
        out[column] = np.nan
        out.loc[stats_source.index, column] = joined[column].to_numpy()
    out["context_cluster_strength"] = _daily_robust_zscore(out, out["context_cluster_strength"])
    out["context_cluster_mf"] = _daily_robust_zscore(out, out["context_cluster_mf"])
    out["context_cluster_hp"] = _daily_robust_zscore(out, out["context_cluster_hp"])
    return out


def _daily_quantile_bin(values: pd.Series, dates: pd.Series, *, bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ranks = numeric.groupby(dates).rank(pct=True)
    raw = np.floor(ranks.clip(lower=0.0, upper=0.999999) * float(bins))
    raw = raw.where(ranks.notna(), -1.0)
    return raw.astype("int16")


def _derive_context_theme_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a data-only dynamic theme proxy from same-day cross-sectional peers.

    This intentionally does not hard-code CPO/semiconductor labels. It groups stocks by
    date, broad industry, and recent 20/60-day momentum quantile buckets, then falls
    back to market-wide momentum buckets when an industry bucket is too thin.
    """
    out = frame.copy()
    dates = pd.to_datetime(out["target_trade_date"], errors="coerce").dt.normalize()
    industry = out.get("industry", pd.Series("", index=out.index)).fillna("").astype(str)
    roc20 = pd.to_numeric(out["context_roc20"], errors="coerce")
    roc60 = pd.to_numeric(out["context_roc60"], errors="coerce")
    mf = pd.to_numeric(out.get("context_moneyflow_confirm"), errors="coerce")

    rank20 = roc20.groupby(dates).rank(pct=True)
    rank60 = roc60.groupby(dates).rank(pct=True)
    bin20 = np.floor(rank20.fillna(-1.0).clip(lower=0.0, upper=0.999999) * 5.0).astype("int16")
    bin60 = np.floor(rank60.fillna(-1.0).clip(lower=0.0, upper=0.999999) * 5.0).astype("int16")
    valid = roc20.notna() & roc60.notna() & dates.notna()

    temp = pd.DataFrame(
        {
            "__idx": np.arange(len(out), dtype=np.int64),
            "__date": dates,
            "__industry": industry,
            "__bin20": bin20,
            "__bin60": bin60,
            "__roc20": roc20,
            "__roc60": roc60,
            "__pos20": roc20.gt(0.0).astype("float64").where(roc20.notna()),
            "__mf": mf,
        },
        index=out.index,
    )
    temp = temp.loc[valid].copy()
    if temp.empty:
        for column in (
            "context_theme_peer_count",
            "context_theme_peer_ret20",
            "context_theme_peer_ret60",
            "context_theme_peer_pos20",
            "context_theme_peer_mf",
            "context_theme_escape20",
            "context_theme_escape60",
            "context_theme_strength",
            "context_theme_h",
        ):
            out[column] = np.nan
        return out

    industry_group = ["__date", "__industry", "__bin20", "__bin60"]
    market_group = ["__date", "__bin20", "__bin60"]
    industry_stats = _theme_group_stats(temp, industry_group, "__industry")
    market_stats = _theme_group_stats(temp, market_group, "__market")
    temp = temp.merge(industry_stats, on=industry_group, how="left")
    temp = temp.merge(market_stats, on=market_group, how="left")
    use_industry = pd.to_numeric(temp["__industry_count"], errors="coerce").ge(5)
    for name in ("count", "ret20", "ret60", "pos20", "mf"):
        temp[f"__theme_{name}"] = temp[f"__industry_{name}"].where(use_industry, temp[f"__market_{name}"])

    out["context_theme_peer_count"] = np.nan
    out["context_theme_peer_ret20"] = np.nan
    out["context_theme_peer_ret60"] = np.nan
    out["context_theme_peer_pos20"] = np.nan
    out["context_theme_peer_mf"] = np.nan
    target_index = temp["__idx"].to_numpy(dtype=np.int64)
    out.iloc[target_index, out.columns.get_loc("context_theme_peer_count")] = temp["__theme_count"].to_numpy()
    out.iloc[target_index, out.columns.get_loc("context_theme_peer_ret20")] = temp["__theme_ret20"].to_numpy()
    out.iloc[target_index, out.columns.get_loc("context_theme_peer_ret60")] = temp["__theme_ret60"].to_numpy()
    out.iloc[target_index, out.columns.get_loc("context_theme_peer_pos20")] = temp["__theme_pos20"].to_numpy()
    out.iloc[target_index, out.columns.get_loc("context_theme_peer_mf")] = temp["__theme_mf"].to_numpy()

    out["context_theme_escape20"] = pd.to_numeric(out["context_roc20"], errors="coerce") - pd.to_numeric(
        out["context_theme_peer_ret20"], errors="coerce"
    )
    out["context_theme_escape60"] = pd.to_numeric(out["context_roc60"], errors="coerce") - pd.to_numeric(
        out["context_theme_peer_ret60"], errors="coerce"
    )
    strength_raw = (
        0.35 * _daily_robust_zscore(out, out["context_theme_peer_ret20"])
        + 0.30 * _daily_robust_zscore(out, out["context_theme_peer_ret60"])
        + 0.15 * _daily_robust_zscore(out, out["context_theme_peer_pos20"])
        + 0.10 * _daily_robust_zscore(out, out["context_theme_peer_mf"])
        + 0.05 * _daily_robust_zscore(out, out["context_industry_relative20"])
        + 0.05 * _daily_robust_zscore(out, out["context_industry_relative60"])
    )
    out["context_theme_strength"] = _daily_robust_zscore(out, strength_raw)
    neg_roc3 = _daily_robust_zscore(out, -pd.to_numeric(out["context_roc3"], errors="coerce")).clip(0.0, 1.5)
    neg_roc5 = _daily_robust_zscore(out, -pd.to_numeric(out["context_roc5"], errors="coerce")).clip(0.0, 1.5)
    roc5 = pd.to_numeric(out["context_roc5"], errors="coerce")
    roc5_floor = roc5.groupby(dates).transform(lambda s: s.quantile(0.05))
    pullback = neg_roc3 * neg_roc5 * roc5.ge(roc5_floor).astype("float64")
    out["context_theme_h"] = (
        _sigmoid(out["context_theme_strength"])
        * pd.to_numeric(pullback, errors="coerce").fillna(0.0)
        * _sigmoid(_daily_robust_zscore(out, out["context_theme_peer_mf"]))
    )
    return out


def _theme_group_stats(frame: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    stats = (
        frame.groupby(keys, sort=False)
        .agg(
            **{
                f"{prefix}_count": ("__roc20", "size"),
                f"{prefix}_ret20": ("__roc20", "mean"),
                f"{prefix}_ret60": ("__roc60", "mean"),
                f"{prefix}_pos20": ("__pos20", "mean"),
                f"{prefix}_mf": ("__mf", "mean"),
            }
        )
        .reset_index()
    )
    return stats


def _ensure_sample_metadata(samples: pd.DataFrame, *, samples_path: str | Path | None) -> pd.DataFrame:
    required = {"sample_id", "stock_code", "industry", "feature_asof_date", "target_trade_date", "decision_ts"}
    missing = sorted(required - set(samples.columns))
    if not missing:
        return samples
    if samples_path is None:
        raise KeyError(f"Missing sample metadata columns and samples_path was not provided: {missing}")
    meta = pd.read_parquet(samples_path, columns=list(required))
    meta["sample_id"] = meta["sample_id"].astype(str)
    out = samples.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    merged = out.merge(meta, on="sample_id", how="left", suffixes=("", "__ctx"))
    for column in required - {"sample_id"}:
        ctx_column = f"{column}__ctx"
        if column not in merged.columns and ctx_column in merged.columns:
            merged[column] = merged[ctx_column]
        elif ctx_column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[ctx_column])
    return merged.drop(columns=[column for column in merged.columns if column.endswith("__ctx")], errors="ignore")


def _load_price(path: str | Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = Path(path)
    requested_columns = [
        "stock_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "change",
        "pct_chg",
        "industry",
    ]
    available = set(pq.ParquetFile(path).schema.names)
    columns = [column for column in requested_columns if column in available]
    table = ds.dataset(str(path), format="parquet").to_table(
        columns=columns,
        filter=(ds.field("trade_date") >= pd.Timestamp(start)) & (ds.field("trade_date") <= pd.Timestamp(end)),
    )
    price = table.to_pandas()
    price["stock_code"] = price["stock_code"].astype(str)
    price["trade_date"] = pd.to_datetime(price["trade_date"], errors="coerce").dt.normalize()
    if "industry" not in price.columns:
        basic_path = path.parent / "basic.parquet"
        if basic_path.exists():
            basic_available = set(pq.ParquetFile(basic_path).schema.names)
            basic_columns = [column for column in ("stock_code", "industry") if column in basic_available]
            if {"stock_code", "industry"}.issubset(basic_columns):
                basic = pd.read_parquet(basic_path, columns=basic_columns)
                basic["stock_code"] = basic["stock_code"].astype(str)
                price = price.merge(
                    basic.drop_duplicates("stock_code", keep="last"),
                    on="stock_code",
                    how="left",
                )
    if "industry" not in price.columns:
        price["industry"] = ""
    price["industry"] = price["industry"].fillna("").astype(str)
    for column in ("open", "high", "low", "close", "preclose", "change", "pct_chg"):
        if column not in price.columns:
            price[column] = np.nan
        price[column] = pd.to_numeric(price[column], errors="coerce")
    close_from_change = price["preclose"] + price["change"]
    close_from_pct = price["preclose"] * (1.0 + price["pct_chg"] / 100.0)
    price["close"] = price["close"].where(price["close"].notna(), close_from_change)
    price["close"] = price["close"].where(price["close"].notna(), close_from_pct)
    price = price.sort_values(["stock_code", "trade_date"])
    close_from_next_preclose = price.groupby("stock_code", sort=False)["preclose"].shift(-1)
    price["close"] = price["close"].where(price["close"].notna(), close_from_next_preclose)
    price["close"] = price["close"].where(price["close"].notna(), price["open"])
    hi_fallback = price[["open", "preclose", "close"]].max(axis=1, skipna=True)
    lo_fallback = price[["open", "preclose", "close"]].min(axis=1, skipna=True)
    price["high"] = price["high"].where(price["high"].notna(), hi_fallback)
    price["low"] = price["low"].where(price["low"].notna(), lo_fallback)
    return price.dropna(subset=["stock_code", "trade_date"]).sort_values(["stock_code", "trade_date"])


def _load_metric(path: str | Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    table = ds.dataset(str(path), format="parquet").to_table(
        columns=["stock_code", "trade_date", "total_mv", "circ_mv", "volume_ratio"],
        filter=(ds.field("trade_date") >= pd.Timestamp(start)) & (ds.field("trade_date") <= pd.Timestamp(end)),
    )
    metric = table.to_pandas()
    metric["stock_code"] = metric["stock_code"].astype(str)
    metric["trade_date"] = pd.to_datetime(metric["trade_date"], errors="coerce").dt.normalize()
    return metric


def _read_news_cache(path: Path, sample_ids: Iterable[object]) -> pd.DataFrame:
    wanted = set(map(str, sample_ids))
    frame = pd.read_parquet(path, columns=["sample_id", "context_news_exact3", "context_news_exact5"])
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame.loc[frame["sample_id"].isin(wanted)].copy()


def _resolve_news_cache_path(news_cache_path: str | Path | None, *, feature_registry_path: str | Path | None) -> Path:
    if news_cache_path is not None:
        return Path(news_cache_path)
    if feature_registry_path is not None:
        return Path(feature_registry_path).parent / "blocks" / "sample" / "context_news_exact_sample.parquet"
    return Path("data/datasets/features/blocks/sample/context_news_exact_sample.parquet")


def _needs_news_exact(columns: Sequence[str]) -> bool:
    return any(column in set(columns) for column in ("context_news_exact3", "context_news_exact5", "context_news_exact", "context_news"))


def _weighted_sum(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    values = pd.Series(0.0, index=frame.index, dtype="float64")
    for column, weight in weights.items():
        values = values + float(weight) * pd.to_numeric(frame[column], errors="coerce")
    return values


def _daily_robust_zscore(frame: pd.DataFrame, values: pd.Series, *, clip: float = 4.0) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    dates = pd.to_datetime(frame["target_trade_date"], errors="coerce").dt.normalize()
    median = values.groupby(dates).transform("median")
    mad = (values - median).abs().groupby(dates).transform("median")
    scale = (1.4826 * mad).replace(0.0, np.nan)
    std = values.groupby(dates).transform("std").replace(0.0, np.nan)
    scale = scale.fillna(std).replace(0.0, np.nan)
    z = ((values - median) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.clip(lower=-clip, upper=clip)


def _sigmoid(values: pd.Series | np.ndarray) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce") if isinstance(values, pd.Series) else pd.Series(values)
    clipped = arr.clip(lower=-30.0, upper=30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _validate_context_coverage(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    required = [column for column in columns if column not in {"context_news_exact3", "context_news_exact5", "context_news_exact", "context_news"}]
    missing = [column for column in required if pd.to_numeric(frame[column], errors="coerce").notna().mean() <= 0.0]
    if missing:
        raise ValueError(f"Strict context columns have no finite coverage: {missing}")


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def _require_columns(df: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")
