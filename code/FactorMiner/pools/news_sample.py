from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from FactorMiner.core.factor_spec import FactorResult, FactorSpec


SAMPLE_KEY_COLUMNS = ("sample_id",)
SCORE_COLUMNS = (
    "sentiment_score",
    "impact_score",
    "risk_score",
    "relevance_score",
    "novelty_score",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsSampleConfig:
    prefix: str = "news_"
    windows: tuple[int, ...] = (1, 3, 5, 10)
    market_event_types: tuple[str, ...] = ("policy", "macro", "rates", "fx", "geopolitics", "commodity")
    stock_event_types: tuple[str, ...] = ("company", "earnings", "litigation", "contract")
    high_impact_threshold: float = 0.7
    negative_high_impact_threshold: float = 0.5


def build_news_sample_factors(
    samples: pd.DataFrame,
    news_items: pd.DataFrame,
    news_stock_map: pd.DataFrame,
    news_scores: pd.DataFrame,
    config: NewsSampleConfig | None = None,
    scope: str = "all",
) -> FactorResult:
    """Build sample-level news factors with natural-day lookback windows.

    News visibility is constrained by each sample's `decision_ts`; an event is
    included only when `decision_ts - window < publish_time <= decision_ts`.
    """
    config = config or NewsSampleConfig()
    if scope not in {"all", "market", "stock"}:
        raise ValueError("scope must be one of: all, market, stock")
    _require_columns(samples, ("sample_id", "stock_code", "decision_ts"), "samples")
    _require_columns(news_items, ("news_id", "news_text_hash", "publish_time", "matched_stock_count"), "news_items")
    _require_columns(news_stock_map, ("news_id", "stock_code"), "news_stock_map")
    _require_columns(news_scores, ("news_text_hash", *SCORE_COLUMNS, "event_type"), "news_scores")

    prepared_samples = _prepare_samples(samples)
    scored_items = _prepare_scored_items(news_items, news_scores)
    market_events = scored_items.loc[scored_items["matched_stock_count"].fillna(0).eq(0)].copy()
    stock_events = _prepare_stock_events(scored_items, news_stock_map)
    LOGGER.info(
        "news_sample_prepare_inputs_done samples=%s scored_items=%s market_events=%s stock_events=%s",
        len(prepared_samples),
        len(scored_items),
        len(market_events),
        len(stock_events),
    )

    factors = prepared_samples[["sample_id"]].copy()
    specs: list[FactorSpec] = []
    if scope in {"all", "market"}:
        LOGGER.info("news_sample_market_aggregate_start")
        market = _aggregate_events(
            prepared_samples[["sample_id", "decision_ts"]],
            market_events,
            prefix=f"{config.prefix}market",
            config=config,
            event_types=config.market_event_types,
            log_windows=True,
        )
        LOGGER.info("news_sample_market_aggregate_done rows=%s columns=%s", len(market), len(market.columns))
        factors = factors.merge(market, on="sample_id", how="left")
        specs.extend(_build_specs(config, f"{config.prefix}market", config.market_event_types))

    if scope in {"all", "stock"}:
        LOGGER.info("news_sample_stock_aggregate_start")
        stock = _aggregate_events(
            prepared_samples[["sample_id", "stock_code", "decision_ts"]],
            stock_events,
            prefix=f"{config.prefix}stock",
            config=config,
            event_types=config.stock_event_types,
            group_col="stock_code",
            log_windows=False,
        )
        LOGGER.info("news_sample_stock_aggregate_done rows=%s columns=%s", len(stock), len(stock.columns))
        factors = factors.merge(stock, on="sample_id", how="left")
        specs.extend(_build_specs(config, f"{config.prefix}stock", config.stock_event_types))

    _fill_count_features(factors)
    result = FactorResult(factors=factors, specs=specs, key_columns=SAMPLE_KEY_COLUMNS)
    result.validate()
    return result


def _prepare_samples(samples: pd.DataFrame) -> pd.DataFrame:
    work = samples[["sample_id", "stock_code", "decision_ts"]].copy()
    work["decision_ts"] = pd.to_datetime(work["decision_ts"], errors="coerce")
    work = work.dropna(subset=["decision_ts"])
    if work["sample_id"].duplicated().any():
        examples = work.loc[work["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"samples.sample_id must be unique. Duplicate examples: {examples}")
    return work.reset_index(drop=True)


def _prepare_scored_items(news_items: pd.DataFrame, news_scores: pd.DataFrame) -> pd.DataFrame:
    items = news_items[["news_id", "news_text_hash", "publish_time", "matched_stock_count"]].copy()
    items["publish_time"] = pd.to_datetime(items["publish_time"], errors="coerce")
    items = items.dropna(subset=["publish_time", "news_text_hash"])

    score_columns = ["news_text_hash", *SCORE_COLUMNS, "event_type"]
    scores = news_scores.loc[:, score_columns].drop_duplicates("news_text_hash", keep="last").copy()
    for column in SCORE_COLUMNS:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores["event_type"] = scores["event_type"].fillna("other").astype(str)

    scored = items.merge(scores, on="news_text_hash", how="inner")
    scored = scored.dropna(subset=list(SCORE_COLUMNS))
    return scored.reset_index(drop=True)


def _prepare_stock_events(scored_items: pd.DataFrame, news_stock_map: pd.DataFrame) -> pd.DataFrame:
    stock_map = news_stock_map[["news_id", "stock_code"]].drop_duplicates(["news_id", "stock_code"])
    events = stock_map.merge(scored_items.drop(columns=["matched_stock_count"]), on="news_id", how="inner")
    return events.reset_index(drop=True)


def _aggregate_events(
    targets: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
    config: NewsSampleConfig,
    event_types: tuple[str, ...],
    group_col: str | None = None,
    log_windows: bool = True,
) -> pd.DataFrame:
    output_columns = _feature_columns(prefix, config.windows, event_types)
    result = _empty_feature_frame(targets["sample_id"], output_columns)
    if targets.empty:
        return result

    prepared_events = _prepare_events(events, group_col)
    if prepared_events.empty:
        return result

    if group_col is None:
        features = _aggregate_single_group(targets[["sample_id", "decision_ts"]], prepared_events, prefix, config, event_types, log_windows)
        return result[["sample_id"]].merge(features, on="sample_id", how="left").pipe(_with_count_defaults)

    result_indexed = result.set_index("sample_id")
    group_count = targets[group_col].nunique(dropna=True)
    for group_index, (_, group_targets) in enumerate(targets.groupby(group_col, sort=False), start=1):
        if group_index == 1 or group_index % 500 == 0 or group_index == group_count:
            LOGGER.info("news_sample_group_aggregate_progress group_col=%s group=%s/%s", group_col, group_index, group_count)
        group_value = group_targets[group_col].iloc[0]
        group_events = prepared_events.loc[prepared_events[group_col].eq(group_value)]
        if group_events.empty:
            continue
        features = _aggregate_single_group(
            group_targets[["sample_id", "decision_ts"]],
            group_events,
            prefix,
            config,
            event_types,
            log_windows,
        ).set_index("sample_id")
        result_indexed.loc[features.index, features.columns] = features
    return _with_count_defaults(result_indexed.reset_index())


def _prepare_events(events: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    if events.empty:
        columns = ["publish_time", *SCORE_COLUMNS, "event_type"]
        if group_col:
            columns.append(group_col)
        return pd.DataFrame(columns=columns)
    columns = ["publish_time", *SCORE_COLUMNS, "event_type"]
    if group_col:
        columns.append(group_col)
    work = events.loc[:, columns].copy()
    work["publish_time"] = pd.to_datetime(work["publish_time"], errors="coerce")
    work = work.dropna(subset=["publish_time"])
    for column in SCORE_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=list(SCORE_COLUMNS))
    work["event_type"] = work["event_type"].fillna("other").astype(str)
    return work.sort_values("publish_time").reset_index(drop=True)


def _aggregate_single_group(
    targets: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
    config: NewsSampleConfig,
    event_types: tuple[str, ...],
    log_windows: bool,
) -> pd.DataFrame:
    ordered_targets = targets.sort_values("decision_ts").reset_index(drop=True)
    out = pd.DataFrame({"sample_id": ordered_targets["sample_id"]})
    event_times = events["publish_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    target_times = ordered_targets["decision_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")

    sentiment = events["sentiment_score"].to_numpy("float64")
    impact = events["impact_score"].to_numpy("float64")
    risk = events["risk_score"].to_numpy("float64")
    relevance = events["relevance_score"].to_numpy("float64")
    novelty = events["novelty_score"].to_numpy("float64")

    cumulative = {
        "sentiment": _cumsum(sentiment),
        "impact": _cumsum(impact),
        "risk": _cumsum(risk),
        "novelty": _cumsum(novelty),
        "sentiment_x_impact": _cumsum(sentiment * impact),
        "sentiment_x_relevance": _cumsum(sentiment * relevance),
        "impact_weight": _cumsum(impact),
        "relevance_weight": _cumsum(relevance),
        "high_impact": _cumsum((impact >= config.high_impact_threshold).astype("float64")),
        "negative": _cumsum((sentiment < 0).astype("float64")),
        "negative_high_impact": _cumsum(((sentiment < 0) & (impact >= config.negative_high_impact_threshold)).astype("float64")),
    }
    for event_type in event_types:
        cumulative[f"event_{event_type}"] = _cumsum(events["event_type"].eq(event_type).to_numpy("float64"))

    for window in config.windows:
        if log_windows:
            LOGGER.info("news_sample_window_aggregate_start prefix=%s window=%s targets=%s events=%s", prefix, window, len(ordered_targets), len(events))
        suffix = f"{window}d"
        window_ns = np.int64(window * 24 * 60 * 60 * 1_000_000_000)
        starts = np.searchsorted(event_times, target_times - window_ns, side="right")
        ends = np.searchsorted(event_times, target_times, side="right")
        counts = (ends - starts).astype("float64")

        out[f"{prefix}_count_{suffix}"] = counts
        out[f"{prefix}_high_impact_count_{suffix}"] = _range_sum(cumulative["high_impact"], starts, ends)
        out[f"{prefix}_negative_count_{suffix}"] = _range_sum(cumulative["negative"], starts, ends)
        out[f"{prefix}_negative_high_impact_count_{suffix}"] = _range_sum(cumulative["negative_high_impact"], starts, ends)
        out[f"{prefix}_sentiment_mean_{suffix}"] = _safe_divide(_range_sum(cumulative["sentiment"], starts, ends), counts)
        out[f"{prefix}_impact_mean_{suffix}"] = _safe_divide(_range_sum(cumulative["impact"], starts, ends), counts)
        out[f"{prefix}_risk_mean_{suffix}"] = _safe_divide(_range_sum(cumulative["risk"], starts, ends), counts)
        out[f"{prefix}_novelty_mean_{suffix}"] = _safe_divide(_range_sum(cumulative["novelty"], starts, ends), counts)
        out[f"{prefix}_impact_weighted_sentiment_{suffix}"] = _safe_divide(
            _range_sum(cumulative["sentiment_x_impact"], starts, ends),
            _range_sum(cumulative["impact_weight"], starts, ends),
        )
        out[f"{prefix}_relevance_weighted_sentiment_{suffix}"] = _safe_divide(
            _range_sum(cumulative["sentiment_x_relevance"], starts, ends),
            _range_sum(cumulative["relevance_weight"], starts, ends),
        )
        out[f"{prefix}_max_impact_{suffix}"] = _range_max(impact, starts, ends)
        out[f"{prefix}_max_risk_{suffix}"] = _range_max(risk, starts, ends)
        out[f"{prefix}_min_sentiment_{suffix}"] = -_range_max(-sentiment, starts, ends)
        out[f"{prefix}_latest_sentiment_{suffix}"] = _latest_value(sentiment, starts, ends)
        out[f"{prefix}_latest_impact_{suffix}"] = _latest_value(impact, starts, ends)
        out[f"{prefix}_latest_risk_{suffix}"] = _latest_value(risk, starts, ends)
        out[f"{prefix}_hours_since_latest_{suffix}"] = _hours_since_latest(event_times, target_times, starts, ends)
        for event_type in event_types:
            out[f"{prefix}_{event_type}_count_{suffix}"] = _range_sum(cumulative[f"event_{event_type}"], starts, ends)
        if log_windows:
            LOGGER.info("news_sample_window_aggregate_done prefix=%s window=%s", prefix, window)

    return out


def _feature_columns(prefix: str, windows: Iterable[int], event_types: Iterable[str]) -> list[str]:
    columns: list[str] = []
    base_names = (
        "count",
        "high_impact_count",
        "negative_count",
        "negative_high_impact_count",
        "sentiment_mean",
        "impact_mean",
        "risk_mean",
        "novelty_mean",
        "impact_weighted_sentiment",
        "relevance_weighted_sentiment",
        "max_impact",
        "max_risk",
        "min_sentiment",
        "latest_sentiment",
        "latest_impact",
        "latest_risk",
        "hours_since_latest",
    )
    for window in windows:
        suffix = f"{window}d"
        columns.extend(f"{prefix}_{name}_{suffix}" for name in base_names)
        columns.extend(f"{prefix}_{event_type}_count_{suffix}" for event_type in event_types)
    return columns


def _empty_feature_frame(sample_ids: pd.Series, columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"sample_id": sample_ids.to_numpy()})
    for column in columns:
        frame[column] = 0.0 if _is_count_column(column) else np.nan
    return frame


def _with_count_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if column != "sample_id" and _is_count_column(column):
            frame[column] = frame[column].fillna(0.0)
    return frame


def _fill_count_features(frame: pd.DataFrame) -> None:
    for column in frame.columns:
        if _is_count_column(column):
            frame[column] = frame[column].fillna(0.0)


def _is_count_column(column: str) -> bool:
    return "_count_" in column or column.endswith("_count")


def _cumsum(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.nan_to_num(values, nan=0.0))))


def _range_sum(cumulative: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return cumulative[ends] - cumulative[starts]


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype="float64")
    mask = denominator > 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def _range_max(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    result = np.full(len(starts), np.nan, dtype="float64")
    lengths = ends - starts
    mask = lengths > 0
    if not mask.any() or len(values) == 0:
        return result

    clean = np.asarray(values, dtype="float64")
    clean = np.where(np.isnan(clean), -np.inf, clean)
    valid_lengths = lengths[mask]
    max_power = int(np.floor(np.log2(valid_lengths.max())))
    table = [clean]
    for power in range(1, max_power + 1):
        half = 1 << (power - 1)
        previous = table[-1]
        table.append(np.maximum(previous[:-half], previous[half:]))

    query_power = np.floor(np.log2(valid_lengths)).astype("int64")
    span = np.left_shift(1, query_power)
    query_starts = starts[mask]
    query_ends = ends[mask]
    values_out = np.empty(len(query_starts), dtype="float64")
    for power in np.unique(query_power):
        power_mask = query_power == power
        block = table[int(power)]
        left = block[query_starts[power_mask]]
        right = block[query_ends[power_mask] - span[power_mask]]
        values_out[power_mask] = np.maximum(left, right)
    values_out[np.isneginf(values_out)] = np.nan
    result[mask] = values_out
    return result


def _latest_value(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    result = np.full(len(starts), np.nan, dtype="float64")
    mask = ends > starts
    result[mask] = values[ends[mask] - 1]
    return result


def _hours_since_latest(event_times: np.ndarray, target_times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    result = np.full(len(starts), np.nan, dtype="float64")
    mask = ends > starts
    latest_times = event_times[ends[mask] - 1]
    result[mask] = (target_times[mask] - latest_times) / (60 * 60 * 1_000_000_000)
    return result


def _build_specs(config: NewsSampleConfig, prefix: str, event_types: tuple[str, ...]) -> list[FactorSpec]:
    specs: list[FactorSpec] = []
    for window in config.windows:
        suffix = f"{window}d"
        common = {
            "source": "news_llm",
            "inputs": ("sentiment_score", "impact_score", "risk_score", "relevance_score", "novelty_score", "event_type"),
            "window": window,
            "lookback": window,
            "availability": "decision_ts",
        }
        definitions = {
            "count": ("news.coverage", "count(events)"),
            "high_impact_count": ("news.coverage", f"count(impact_score >= {config.high_impact_threshold})"),
            "negative_count": ("news.risk", "count(sentiment_score < 0)"),
            "negative_high_impact_count": ("news.risk", f"count(sentiment_score < 0 and impact_score >= {config.negative_high_impact_threshold})"),
            "sentiment_mean": ("news.sentiment", "mean(sentiment_score)"),
            "impact_mean": ("news.impact", "mean(impact_score)"),
            "risk_mean": ("news.risk", "mean(risk_score)"),
            "novelty_mean": ("news.novelty", "mean(novelty_score)"),
            "impact_weighted_sentiment": ("news.sentiment", "sum(sentiment_score * impact_score) / sum(impact_score)"),
            "relevance_weighted_sentiment": ("news.sentiment", "sum(sentiment_score * relevance_score) / sum(relevance_score)"),
            "max_impact": ("news.tail", "max(impact_score)"),
            "max_risk": ("news.tail", "max(risk_score)"),
            "min_sentiment": ("news.tail", "min(sentiment_score)"),
            "latest_sentiment": ("news.latest", "latest(sentiment_score)"),
            "latest_impact": ("news.latest", "latest(impact_score)"),
            "latest_risk": ("news.latest", "latest(risk_score)"),
            "hours_since_latest": ("news.latest", "decision_ts - max(publish_time)"),
        }
        for name, (category, expression) in definitions.items():
            specs.append(
                FactorSpec(
                    name=f"{prefix}_{name}_{suffix}",
                    category=category,
                    expression=f"{expression} over natural {window}d window",
                    **common,
                )
            )
        for event_type in event_types:
            specs.append(
                FactorSpec(
                    name=f"{prefix}_{event_type}_count_{suffix}",
                    category="news.event_type",
                    expression=f"count(event_type == {event_type}) over natural {window}d window",
                    **common,
                )
            )
    return specs


def _require_columns(df: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing {frame_name} columns: {missing}")
