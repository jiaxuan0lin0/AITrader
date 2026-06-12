from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from model.msgca.context_features import CONTEXT_COLUMNS, attach_context_features

TREND_SCORE_VARIANTS: set[str] = {
    "context_a4",
    "context_a4_no_news",
    "exact_s5_soft",
    "direct_theme_soft",
    "direct_theme_medium",
    "context_a4",
    "context_a4_no_news",
    "context_s1",
    "context_s2",
    "context_s2_no_news",
    "context_s3",
    "context_s5",
    "context_s5_rerank",
    "context_s5_soft",
    "context_theme_s2",
    "context_theme_s5_rerank",
}


def _is_trend_score_variant(variant: str) -> bool:
    return variant in TREND_SCORE_VARIANTS or variant.startswith("direct_theme_soft_cluster")


@dataclass(frozen=True)
class StrategyParams:
    initial_cash: float = 1_000_000.0
    top_n: int = 20
    daily_replace_k: int = 3
    fee_rate: float = 0.0003
    slippage_rate: float = 0.0005
    full_investment: bool = True
    score_variant: str = "y_score"
    score_weight_y: float = 1.0
    score_weight_return: float = 0.0
    score_weight_direction: float = 0.0
    score_weight_cap: float = 0.0
    cap_min_pct: float = 0.0
    cap_bonus: float = 0.0
    exclude_st: bool = False
    exclude_bj: bool = False


def prepare_strategy_predictions(
    predictions: pd.DataFrame,
    params: StrategyParams,
    *,
    samples_path: str | Path | None = None,
    metric_path: str | Path | None = None,
    price_path: str | Path | None = None,
    moneyflow_path: str | Path | None = None,
    feature_registry_path: str | Path | None = None,
    news_path: str | Path | None = None,
    news_scores_path: str | Path | None = None,
    context_cache_path: str | Path | None = None,
    news_cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Apply tradable-universe filters and strategy score transforms."""
    work = predictions.copy()
    _require_columns(work, ("target_trade_date", "stock_code", "y_score"), "predictions")
    work["target_trade_date"] = pd.to_datetime(work["target_trade_date"]).dt.normalize()
    work["stock_code"] = work["stock_code"].astype(str)
    if "stock_name" in work.columns:
        work["stock_name"] = work["stock_name"].fillna("").astype(str)
    else:
        work["stock_name"] = ""

    work = filter_tradable_universe(work, exclude_st=params.exclude_st, exclude_bj=params.exclude_bj)
    if work.empty:
        return work

    needs_trend = _is_trend_score_variant(str(params.score_variant))
    needs_cap_size = float(params.cap_bonus) != 0.0 or (
        str(params.score_variant) == "weighted" and float(params.score_weight_cap) != 0.0
    )
    needs_cap_filter = float(params.cap_min_pct) > 0.0
    if needs_trend:
        work = attach_context_features(
            work,
            samples_path=samples_path,
            price_path=price_path,
            metric_path=metric_path,
            feature_registry_path=feature_registry_path,
            news_path=news_path,
            news_scores_path=news_scores_path,
            context_cache_path=context_cache_path,
            news_cache_path=news_cache_path,
            context_columns=CONTEXT_COLUMNS,
            strict=True,
        )
    elif (needs_cap_size and "log_total_mv" not in work.columns) or (
        needs_cap_filter and "cap_pct" not in work.columns and "log_total_mv" not in work.columns
    ):
        work = attach_market_features(work, samples_path=samples_path, metric_path=metric_path)
    if needs_cap_filter and "cap_pct" not in work.columns:
        work["cap_pct"] = work.groupby("target_trade_date")["log_total_mv"].rank(pct=True)

    if "raw_y_score" not in work.columns:
        work["raw_y_score"] = pd.to_numeric(work["y_score"], errors="coerce")
    score = strategy_score(
        work,
        variant=params.score_variant,
        score_weight_y=params.score_weight_y,
        score_weight_return=params.score_weight_return,
        score_weight_direction=params.score_weight_direction,
        score_weight_cap=params.score_weight_cap,
        cap_min_pct=params.cap_min_pct,
        cap_bonus=params.cap_bonus,
    )
    work["strategy_score_variant"] = params.score_variant
    work["strategy_score_weight_y"] = float(params.score_weight_y)
    work["strategy_score_weight_return"] = float(params.score_weight_return)
    work["strategy_score_weight_direction"] = float(params.score_weight_direction)
    work["strategy_score_weight_cap"] = float(params.score_weight_cap)
    work["strategy_cap_min_pct"] = float(params.cap_min_pct)
    work["strategy_cap_bonus"] = float(params.cap_bonus)
    work["y_score"] = score
    work["rank"] = work.groupby("target_trade_date")["y_score"].rank(method="first", ascending=False)
    return work


def filter_tradable_universe(
    predictions: pd.DataFrame,
    *,
    exclude_st: bool = False,
    exclude_bj: bool = False,
) -> pd.DataFrame:
    work = predictions.copy()
    keep = pd.Series(True, index=work.index)
    if exclude_st:
        names = work.get("stock_name", pd.Series("", index=work.index)).fillna("").astype(str)
        keep &= ~names.str.contains("ST", case=False, regex=False)
        keep &= ~names.str.contains("退", regex=False)
    if exclude_bj:
        codes = work["stock_code"].astype(str)
        keep &= ~codes.str.endswith(".BJ")
        keep &= ~codes.str[:2].isin({"43", "83", "87", "88", "89"})
    return work.loc[keep].copy()


def attach_market_features(
    predictions: pd.DataFrame,
    *,
    samples_path: str | Path | None,
    metric_path: str | Path | None,
) -> pd.DataFrame:
    if metric_path is None:
        raise ValueError("metric_path is required when strategy uses cap_min_pct or cap_bonus")
    work = predictions.copy()
    if "feature_asof_date" not in work.columns:
        if samples_path is None:
            raise ValueError("samples_path is required to recover feature_asof_date for cap-aware strategy scoring")
        samples = ds.dataset(str(samples_path), format="parquet").to_table(
            columns=["sample_id", "feature_asof_date"]
        ).to_pandas()
        samples["sample_id"] = samples["sample_id"].astype(str)
        work["sample_id"] = work["sample_id"].astype(str)
        samples = samples.loc[samples["sample_id"].isin(set(work["sample_id"]))]
        work = work.merge(samples, on="sample_id", how="left")
    work["feature_asof_date"] = pd.to_datetime(work["feature_asof_date"], errors="coerce").dt.normalize()
    start = work["feature_asof_date"].min()
    end = work["feature_asof_date"].max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("feature_asof_date is missing for cap-aware strategy scoring")
    metric = ds.dataset(str(metric_path), format="parquet").to_table(
        columns=["stock_code", "trade_date", "total_mv", "circ_mv", "turnover_rate"],
        filter=(ds.field("trade_date") >= pd.Timestamp(start)) & (ds.field("trade_date") <= pd.Timestamp(end)),
    ).to_pandas()
    metric["stock_code"] = metric["stock_code"].astype(str)
    metric["trade_date"] = pd.to_datetime(metric["trade_date"], errors="coerce").dt.normalize()
    work = work.merge(
        metric,
        left_on=["stock_code", "feature_asof_date"],
        right_on=["stock_code", "trade_date"],
        how="left",
    )
    work["log_total_mv"] = np.log(pd.to_numeric(work["total_mv"], errors="coerce").clip(lower=1.0))
    work["cap_pct"] = work.groupby("target_trade_date")["log_total_mv"].rank(pct=True)
    return work


def attach_trend_features(
    predictions: pd.DataFrame,
    *,
    samples_path: str | Path | None,
    metric_path: str | Path | None,
    price_path: str | Path | None = None,
    moneyflow_path: str | Path | None = None,
) -> pd.DataFrame:
    work = predictions.copy()
    work = _ensure_feature_asof_date(work, samples_path=samples_path)
    work["feature_asof_date"] = pd.to_datetime(work["feature_asof_date"], errors="coerce").dt.normalize()
    start = work["feature_asof_date"].min()
    end = work["feature_asof_date"].max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("feature_asof_date is missing for trend-aware strategy scoring")

    price_source = _resolve_processed_path(price_path, metric_path, "price.parquet")
    price = _load_price_trend_frame(price_source, start=start - pd.Timedelta(days=140), end=end)
    stock_features, industry_features, market_features = _compute_price_trend_features(price)
    del price

    work = work.merge(
        stock_features,
        left_on=["stock_code", "feature_asof_date"],
        right_on=["stock_code", "trade_date"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    work = work.merge(
        industry_features,
        left_on=["industry", "feature_asof_date"],
        right_on=["industry", "trade_date"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    work = work.merge(
        market_features,
        left_on="feature_asof_date",
        right_on="trade_date",
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")

    if metric_path is not None:
        metric = _load_metric_trend_frame(metric_path, start=start, end=end)
        if not metric.empty:
            work = work.merge(
                metric,
                left_on=["stock_code", "feature_asof_date"],
                right_on=["stock_code", "trade_date"],
                how="left",
            ).drop(columns=["trade_date"], errors="ignore")
    if "log_total_mv" not in work.columns and "total_mv" in work.columns:
        work["log_total_mv"] = np.log(pd.to_numeric(work["total_mv"], errors="coerce").clip(lower=1.0))
    if "cap_pct" not in work.columns and "log_total_mv" in work.columns:
        work["cap_pct"] = work.groupby("target_trade_date")["log_total_mv"].rank(pct=True)

    mf_source = _resolve_processed_path(moneyflow_path, metric_path, "moneyflow.parquet")
    if mf_source is not None and mf_source.exists():
        moneyflow = _load_moneyflow_trend_frame(mf_source, start=start, end=end)
        if not moneyflow.empty:
            work = work.merge(
                moneyflow,
                left_on=["stock_code", "feature_asof_date"],
                right_on=["stock_code", "trade_date"],
                how="left",
            ).drop(columns=["trade_date"], errors="ignore")

    amount = pd.to_numeric(work.get("trend_amount"), errors="coerce")
    net_mf = pd.to_numeric(work.get("net_mf_amount"), errors="coerce")
    work["trend_mf_ratio"] = (net_mf / amount.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return work


def strategy_score(
    predictions: pd.DataFrame,
    *,
    variant: str = "y_score",
    score_weight_y: float = 1.0,
    score_weight_return: float = 0.0,
    score_weight_direction: float = 0.0,
    score_weight_cap: float = 0.0,
    cap_min_pct: float = 0.0,
    cap_bonus: float = 0.0,
) -> pd.Series:
    work = predictions.copy()
    variant = str(variant or "y_score")
    if variant == "y_score":
        score = pd.to_numeric(work["y_score"], errors="coerce")
    elif variant == "final_score":
        score = pd.to_numeric(work["final_score"], errors="coerce")
    elif variant == "return_pred":
        score = _daily_zscore(work, "return_pred")
    elif variant == "direction_prob":
        score = _daily_zscore(work, "direction_prob")
    elif variant == "z_y_plus_return":
        score = _daily_zscore(work, "y_score") + _daily_zscore(work, "return_pred")
    elif variant == "z_y_plus_direction":
        score = _daily_zscore(work, "y_score") + _daily_zscore(work, "direction_prob")
    elif variant == "z_all_equal":
        score = _daily_zscore(work, "y_score") + _daily_zscore(work, "return_pred") + _daily_zscore(work, "direction_prob")
    elif variant == "direct_multihead":
        score = (
            0.62 * _daily_zscore(work, "final_score")
            + 0.10 * _daily_zscore(work, "y_score")
            + 0.22 * _daily_zscore(work, "return_pred")
            + 0.06 * _daily_logit_zscore(work, "direction_prob")
        )
    elif variant == "weighted":
        score = (
            float(score_weight_y) * _daily_zscore(work, "y_score")
            + float(score_weight_return) * _daily_zscore(work, "return_pred")
            + float(score_weight_direction) * _daily_zscore(work, "direction_prob")
        )
        if float(score_weight_cap) != 0.0:
            if "log_total_mv" not in work.columns:
                raise KeyError("Missing log_total_mv for weighted cap strategy scoring")
            score = score + float(score_weight_cap) * _daily_zscore(work, "log_total_mv")
    elif _is_trend_score_variant(variant):
        if variant.startswith("exact_"):
            components = _exact_score_components(work)
        else:
            components = _context_score_components(work)
        if variant == "direct_theme_soft":
            score = _direct_multihead_base_score(work) + _theme_overlay_score(components, theme_weight=0.06)
        elif variant == "direct_theme_medium":
            score = _direct_multihead_base_score(work) + _theme_overlay_score(components, theme_weight=0.12)
        elif variant.startswith("direct_theme_soft_cluster_boost"):
            score = _direct_theme_soft_cluster_boost_score(
                work,
                components,
                boost_weight=_cluster_boost_weight_from_variant(variant),
            )
        elif variant.startswith("direct_theme_soft_cluster"):
            score = _direct_theme_soft_cluster_score(work, components, cluster_count=_cluster_count_from_variant(variant))
        elif variant in {"context_a4", "context_a4"}:
            score = (
                0.50 * components["F"]
                + 0.06 * components["Y"]
                + 0.16 * components["R"]
                + 0.06 * components["D_eff"]
                + 0.07 * components["NEWS"]
                + 0.07 * components["TR"]
                - 0.05 * components["OH"]
                - 0.08 * components["BR"]
                + 0.03 * components["CAP"]
            )
        elif variant in {"context_a4_no_news", "context_a4_no_news"}:
            score = (
                0.53 * components["F"]
                + 0.06 * components["Y"]
                + 0.17 * components["R"]
                + 0.06 * components["D_eff"]
                + 0.08 * components["TR"]
                - 0.06 * components["OH"]
                - 0.09 * components["BR"]
                + 0.03 * components["CAP"]
            )
        elif variant == "exact_s5_soft":
            score = _context_s2_score(components)
        elif variant == "context_s1":
            score = _context_s1_score(components)
        elif variant in {"context_s2", "context_s5", "context_s5_soft"}:
            score = _context_s2_score(components)
        elif variant == "context_s2_no_news":
            score = _context_s2_no_news_score(components)
        elif variant == "context_s3":
            score = _context_s3_score(components, work)
        elif variant == "context_theme_s2":
            score = _context_theme_s2_score(components)
        elif variant == "context_s5_rerank":
            base = 0.60 * components["F"] + 0.20 * components["R"] + 0.10 * components["Y"] + 0.10 * components["D_eff"]
            rerank = _context_s2_score(components)
            candidate_rank = base.groupby(pd.to_datetime(work["target_trade_date"]).dt.normalize()).rank(method="first", ascending=False)
            candidate = candidate_rank.le(80)
            confirm = components["F"].gt(2.0) & components["R"].gt(1.0) & components["MF"].gt(0.8) & components["NEWS"].gt(0.0)
            broken = components["BR"].gt(0.75) & ~confirm
            overheat = components["OH"].gt(0.80) & components["HP"].lt(0.3)
            score = rerank.where(candidate, -1e9)
            score = score.where(~broken, -1e9)
            score = score.where(~overheat, score - 0.6)
        elif variant == "context_theme_s5_rerank":
            dates = pd.to_datetime(work["target_trade_date"]).dt.normalize()
            base = (
                0.50 * components["F"]
                + 0.18 * components["R"]
                + 0.08 * components["Y"]
                + 0.06 * components["D_eff"]
                + 0.18 * components["THEME"]
            )
            rerank = _context_theme_s2_score(components)
            candidate_rank = base.groupby(dates).rank(method="first", ascending=False)
            theme_rank = components["THEME"].groupby(dates).rank(method="first", ascending=False)
            candidate = candidate_rank.le(100) | theme_rank.le(80)
            confirm = (
                components["F"].gt(1.5)
                & components["R"].gt(0.6)
                & components["MF"].gt(0.5)
            ) | (
                components["THEME"].gt(1.4)
                & components["MF"].gt(0.2)
                & components["BR"].lt(0.70)
            )
            broken = components["BR"].gt(0.78) & ~confirm
            overheat = components["OH"].gt(0.85) & components["THEME_HP"].lt(0.2) & components["MF"].lt(0.2)
            score = rerank.where(candidate, -1e9)
            score = score.where(~broken, -1e9)
            score = score.where(~overheat, score - 0.6)
        else:
            raise ValueError(f"Unsupported strict strategy score variant: {variant}")
    else:
        raise ValueError(f"Unsupported strategy score variant: {variant}")

    if float(cap_bonus) != 0.0:
        if "log_total_mv" not in work.columns:
            raise KeyError("Missing log_total_mv for cap_bonus strategy scoring")
        score = score + float(cap_bonus) * _daily_zscore(work, "log_total_mv")
    if float(cap_min_pct) > 0.0:
        if "cap_pct" not in work.columns:
            raise KeyError("Missing cap_pct for cap_min_pct strategy scoring")
        score = score.where(pd.to_numeric(work["cap_pct"], errors="coerce").fillna(0.0) >= float(cap_min_pct), -1e9)
    return score.replace([np.inf, -np.inf], np.nan).fillna(-1e9)


def _daily_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Missing score column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    grouped = values.groupby(pd.to_datetime(frame["target_trade_date"]).dt.normalize())
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((values - mean) / std).fillna(0.0)


def _daily_logit_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Missing score column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    logits = np.log(values / (1.0 - values))
    temp = frame.copy()
    temp[f"__logit_{column}"] = logits
    return _daily_zscore(temp, f"__logit_{column}")


def _daily_robust_zscore(frame: pd.DataFrame, column: str, *, clip: float = 4.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    dates = pd.to_datetime(frame["target_trade_date"]).dt.normalize()
    median = values.groupby(dates).transform("median")
    mad = (values - median).abs().groupby(dates).transform("median")
    scale = (1.4826 * mad).replace(0.0, np.nan)
    std = values.groupby(dates).transform("std").replace(0.0, np.nan)
    scale = scale.fillna(std).replace(0.0, np.nan)
    z = ((values - median) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.clip(lower=-clip, upper=clip)


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(values, errors="coerce").clip(lower=-20.0, upper=20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _context_score_components(work: pd.DataFrame) -> dict[str, pd.Series]:
    required = (
        "context_tr",
        "context_mf",
        "context_news",
        "context_oh",
        "context_br",
        "context_hp",
        "context_h",
        "context_cap",
    )
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise KeyError(f"Missing strict strategy context columns: {missing}")
    eps = 1e-6
    direction = pd.to_numeric(work.get("direction_prob"), errors="coerce").clip(eps, 1.0 - eps)
    frame = work.copy()
    frame["__direction_logit"] = np.log(direction / (1.0 - direction))
    d = _daily_robust_zscore(frame, "__direction_logit")
    h = pd.to_numeric(frame["context_h"], errors="coerce").clip(0.0, 1.0).fillna(0.0)
    d_eff = d.clip(lower=0.0) + (1.0 - h) * d.clip(upper=0.0)
    final_score = "final_score" if "final_score" in frame.columns else "y_score"
    return {
        "F": _daily_robust_zscore(frame, final_score),
        "Y": _daily_robust_zscore(frame, "raw_y_score" if "raw_y_score" in frame.columns else "y_score"),
        "R": _daily_robust_zscore(frame, "return_pred"),
        "D_eff": d_eff.fillna(0.0),
        "TR": pd.to_numeric(frame["context_tr"], errors="coerce").fillna(0.0),
        "MF": pd.to_numeric(frame["context_mf"], errors="coerce").fillna(0.0),
        "NEWS": pd.to_numeric(frame["context_news"], errors="coerce").fillna(0.0),
        "HP": pd.to_numeric(frame["context_hp"], errors="coerce").fillna(0.0),
        "OH": pd.to_numeric(frame["context_oh"], errors="coerce").clip(0.0, 1.0).fillna(0.0),
        "BR": pd.to_numeric(frame["context_br"], errors="coerce").clip(0.0, 1.0).fillna(0.0),
        "CAP": pd.to_numeric(frame["context_cap"], errors="coerce").fillna(0.0),
        "THEME": pd.to_numeric(frame.get("context_theme_strength"), errors="coerce").fillna(0.0),
        "THEME_HP": pd.to_numeric(frame.get("context_theme_hp"), errors="coerce").fillna(0.0),
        "CLUSTER": pd.to_numeric(frame.get("context_cluster_strength"), errors="coerce").fillna(0.0),
        "CLUSTER_MF": pd.to_numeric(frame.get("context_cluster_mf"), errors="coerce").fillna(0.0),
        "CLUSTER_HP": pd.to_numeric(frame.get("context_cluster_hp"), errors="coerce").fillna(0.0),
        "CLUSTER_SIZE": _daily_robust_zscore(frame, "context_cluster_size"),
    }


def _exact_score_components(work: pd.DataFrame) -> dict[str, pd.Series]:
    required = (
        "context_roc3",
        "context_roc5",
        "context_roc20",
        "context_roc60",
        "context_rsv20",
        "context_volume_ratio",
        "context_news_exact",
        "context_log_total_mv",
        "context_broken_ma",
    )
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise KeyError(f"Missing exact strategy context columns: {missing}")
    eps = 1e-6
    direction = pd.to_numeric(work.get("direction_prob"), errors="coerce").clip(eps, 1.0 - eps)
    frame = work.copy()
    frame["__direction_logit"] = np.log(direction / (1.0 - direction))
    roc3 = _daily_robust_zscore(frame, "context_roc3")
    roc5 = _daily_robust_zscore(frame, "context_roc5")
    roc20 = _daily_robust_zscore(frame, "context_roc20")
    roc60 = _daily_robust_zscore(frame, "context_roc60")
    rsv20 = _daily_robust_zscore(frame, "context_rsv20")
    volume = _daily_robust_zscore(frame, "context_volume_ratio")
    final_score = "final_score" if "final_score" in frame.columns else "y_score"
    return {
        "F": _daily_robust_zscore(frame, final_score),
        "Y": _daily_robust_zscore(frame, "raw_y_score" if "raw_y_score" in frame.columns else "y_score"),
        "R": _daily_robust_zscore(frame, "return_pred"),
        "D_eff": _daily_robust_zscore(frame, "__direction_logit"),
        "TR": 0.55 * roc20 + 0.35 * roc60 - 0.10 * rsv20,
        "MF": pd.Series(0.0, index=frame.index, dtype="float64"),
        "NEWS": _daily_robust_zscore(frame, "context_news_exact"),
        "HP": pd.Series(0.0, index=frame.index, dtype="float64"),
        "OH": _sigmoid(0.80 * roc3 + 0.80 * roc5 + 0.50 * rsv20 + 0.30 * volume),
        "BR": _sigmoid(
            pd.to_numeric(frame["context_broken_ma"], errors="coerce").fillna(0.0)
            - 0.60 * roc20
            - 0.40 * roc60
        ),
        "CAP": _daily_robust_zscore(frame, "context_log_total_mv"),
    }


def _context_s1_score(components: dict[str, pd.Series]) -> pd.Series:
    return (
        0.50 * components["F"]
        + 0.15 * components["Y"]
        + 0.18 * components["R"]
        + 0.07 * components["D_eff"]
        + 0.08 * components["MF"]
        + 0.05 * components["NEWS"]
        + 0.07 * components["TR"]
        + 0.06 * components["HP"]
        - 0.12 * components["OH"]
        - 0.20 * components["BR"]
        + 0.03 * components["CAP"]
    )


def _context_s2_score(components: dict[str, pd.Series]) -> pd.Series:
    return (
        0.42 * components["F"]
        + 0.08 * components["Y"]
        + 0.18 * components["R"]
        + 0.04 * components["D_eff"]
        + 0.12 * components["TR"]
        + 0.15 * components["HP"]
        + 0.10 * components["MF"]
        + 0.06 * components["NEWS"]
        - 0.10 * components["OH"]
        - 0.28 * components["BR"]
        + 0.03 * components["CAP"]
    )


def _context_s2_no_news_score(components: dict[str, pd.Series]) -> pd.Series:
    return (
        0.45 * components["F"]
        + 0.08 * components["Y"]
        + 0.19 * components["R"]
        + 0.04 * components["D_eff"]
        + 0.13 * components["TR"]
        + 0.16 * components["HP"]
        + 0.11 * components["MF"]
        - 0.11 * components["OH"]
        - 0.30 * components["BR"]
        + 0.03 * components["CAP"]
    )


def _context_s3_score(components: dict[str, pd.Series], work: pd.DataFrame) -> pd.Series:
    score = (
        0.38 * components["F"]
        + 0.12 * components["Y"]
        + 0.25 * components["R"]
        + 0.10 * components["D_eff"]
        + 0.08 * components["MF"]
        + 0.07 * components["TR"]
        + 0.06 * components["HP"]
        + 0.05 * components["NEWS"]
        - 0.10 * components["OH"]
        - 0.22 * components["BR"]
    )
    direction = pd.to_numeric(work.get("direction_prob"), errors="coerce")
    healthy = pd.to_numeric(work.get("context_h"), errors="coerce")
    guard = components["R"].lt(-0.8) & direction.lt(0.48) & healthy.lt(0.5)
    dates = pd.to_datetime(work["target_trade_date"]).dt.normalize()
    top40_floor = score.groupby(dates).transform(_group_top40_floor)
    return score.where(~guard, np.minimum(score, top40_floor - 1e-6))


def _context_theme_s2_score(components: dict[str, pd.Series]) -> pd.Series:
    return (
        0.36 * components["F"]
        + 0.07 * components["Y"]
        + 0.17 * components["R"]
        + 0.04 * components["D_eff"]
        + 0.09 * components["TR"]
        + 0.16 * components["THEME"]
        + 0.10 * components["HP"]
        + 0.07 * components["THEME_HP"]
        + 0.09 * components["MF"]
        + 0.05 * components["NEWS"]
        - 0.10 * components["OH"]
        - 0.25 * components["BR"]
        + 0.03 * components["CAP"]
    )


def _direct_multihead_base_score(work: pd.DataFrame) -> pd.Series:
    return (
        0.62 * _daily_zscore(work, "final_score")
        + 0.10 * _daily_zscore(work, "raw_y_score" if "raw_y_score" in work.columns else "y_score")
        + 0.22 * _daily_zscore(work, "return_pred")
        + 0.06 * _daily_logit_zscore(work, "direction_prob")
    )


def _theme_overlay_score(components: dict[str, pd.Series], *, theme_weight: float) -> pd.Series:
    return (
        float(theme_weight) * components["THEME"]
        + 0.03 * components["TR"]
        + 0.02 * components["MF"]
        - 0.02 * components["OH"]
        - 0.03 * components["BR"]
    )


def _cluster_count_from_variant(variant: str) -> int:
    suffix = variant.removeprefix("direct_theme_soft_cluster")
    if not suffix:
        return 8
    try:
        value = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Unsupported cluster score variant: {variant}") from exc
    if value <= 0:
        raise ValueError(f"Cluster count must be positive in score variant: {variant}")
    return value


def _cluster_boost_weight_from_variant(variant: str) -> float:
    suffix = variant.removeprefix("direct_theme_soft_cluster_boost")
    if not suffix:
        return 0.14
    if suffix == "strong":
        return 0.22
    if suffix == "light":
        return 0.08
    try:
        return float(suffix) / 100.0
    except ValueError as exc:
        raise ValueError(f"Unsupported cluster boost score variant: {variant}") from exc


def _direct_theme_soft_cluster_boost_score(
    work: pd.DataFrame,
    components: dict[str, pd.Series],
    *,
    boost_weight: float,
) -> pd.Series:
    base = _direct_multihead_base_score(work) + _theme_overlay_score(components, theme_weight=0.06)
    cluster_score = (
        0.45 * components["CLUSTER"]
        + 0.18 * components["CLUSTER_MF"]
        + 0.18 * components["CLUSTER_HP"]
        + 0.10 * components["THEME"]
        + 0.08 * components["TR"]
        + 0.04 * components["CLUSTER_SIZE"]
        - 0.10 * components["OH"]
        - 0.18 * components["BR"]
    )
    frame = pd.DataFrame(
        {
            "__date": pd.to_datetime(work["target_trade_date"]).dt.normalize(),
            "__cluster": pd.to_numeric(work.get("context_cluster_id"), errors="coerce"),
            "__cluster_size": pd.to_numeric(work.get("context_cluster_size"), errors="coerce"),
            "__base": base,
            "__cluster_score": cluster_score,
        },
        index=work.index,
    )
    out = base.copy()
    for _, day in frame.groupby("__date", sort=False):
        valid = day.loc[day["__cluster"].notna() & day["__cluster_size"].ge(8)].copy()
        if valid.empty:
            continue
        cluster_mean = valid.groupby("__cluster", sort=False)["__cluster_score"].mean().sort_values(ascending=False)
        rank_map = {cluster_id: rank for rank, cluster_id in enumerate(cluster_mean.index, start=1)}
        cluster_rank = valid["__cluster"].map(rank_map).astype("float64")
        top_cluster = cluster_rank.le(8)
        weak_cluster = cluster_rank.gt(max(20, int(np.ceil(len(cluster_mean) * 0.75))))
        bonus = boost_weight * valid["__cluster_score"] + 0.04 * top_cluster.astype("float64") - 0.05 * weak_cluster.astype("float64")
        out.loc[valid.index] = valid["__base"] + bonus
    return out


def _direct_theme_soft_cluster_score(
    work: pd.DataFrame,
    components: dict[str, pd.Series],
    *,
    cluster_count: int,
) -> pd.Series:
    base = _direct_multihead_base_score(work) + _theme_overlay_score(components, theme_weight=0.06)
    cluster_score = (
        0.45 * components["CLUSTER"]
        + 0.18 * components["CLUSTER_MF"]
        + 0.18 * components["CLUSTER_HP"]
        + 0.10 * components["THEME"]
        + 0.08 * components["TR"]
        + 0.04 * components["CLUSTER_SIZE"]
        - 0.10 * components["OH"]
        - 0.18 * components["BR"]
    )
    frame = pd.DataFrame(
        {
            "__date": pd.to_datetime(work["target_trade_date"]).dt.normalize(),
            "__cluster": pd.to_numeric(work.get("context_cluster_id"), errors="coerce"),
            "__cluster_size": pd.to_numeric(work.get("context_cluster_size"), errors="coerce"),
            "__base": base,
            "__cluster_score": cluster_score,
        },
        index=work.index,
    )
    out = pd.Series(-1e9, index=work.index, dtype="float64")
    for _, day in frame.groupby("__date", sort=False):
        valid = day.loc[day["__cluster"].notna() & day["__cluster_size"].ge(8)].copy()
        if valid.empty:
            out.loc[day.index] = day["__base"]
            continue
        cluster_rank = (
            valid.groupby("__cluster", sort=False)["__cluster_score"]
            .mean()
            .sort_values(ascending=False)
            .head(cluster_count)
        )
        selected = valid.loc[valid["__cluster"].isin(set(cluster_rank.index))].copy()
        if selected.empty:
            out.loc[day.index] = day["__base"]
            continue
        rank_map = {cluster_id: rank for rank, cluster_id in enumerate(cluster_rank.index, start=1)}
        selected["__cluster_rank"] = selected["__cluster"].map(rank_map).astype("float64")
        selected["__within_rank"] = selected.groupby("__cluster")["__base"].rank(method="first", ascending=False)
        hierarchical = -(
            (selected["__within_rank"] - 1.0) * float(max(cluster_count, 1)) + selected["__cluster_rank"]
        )
        out.loc[selected.index] = hierarchical + 1e-4 * selected["__base"]
        fallback = day.index.difference(selected.index)
        out.loc[fallback] = -10000.0 + day.loc[fallback, "__base"]
    return out


def _group_top40_floor(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return -1e9
    if len(clean) < 40:
        return float(clean.min())
    return float(clean.nlargest(40).iloc[-1])


def _trend_score_components(work: pd.DataFrame) -> dict[str, pd.Series]:
    eps = 1e-6
    direction = pd.to_numeric(work.get("direction_prob"), errors="coerce").clip(eps, 1.0 - eps)
    direction_logit = np.log(direction / (1.0 - direction))
    frame = work.copy()
    frame["__direction_logit"] = direction_logit
    frame["__trend_strength_raw"] = (
        0.55 * _daily_robust_zscore(frame, "trend_roc20")
        + 0.35 * _daily_robust_zscore(frame, "trend_roc60")
        + 0.30 * _daily_robust_zscore(frame, "trend_industry_excess20")
        + 0.15 * _daily_robust_zscore(frame, "trend_industry_excess60")
    )
    frame["__pullback_raw"] = (
        _daily_robust_zscore(frame.assign(__neg_roc3=-pd.to_numeric(frame.get("trend_roc3"), errors="coerce")), "__neg_roc3").clip(0.0, 1.5)
        + _daily_robust_zscore(frame.assign(__neg_roc5=-pd.to_numeric(frame.get("trend_roc5"), errors="coerce")), "__neg_roc5").clip(0.0, 1.5)
    ) / 2.0
    frame["__mf_raw"] = _daily_robust_zscore(frame, "trend_mf_ratio")
    frame["__rsv_raw"] = _daily_robust_zscore(frame, "trend_rsv20")
    frame["__volume_raw"] = _daily_robust_zscore(frame, "trend_volume_ratio")
    frame["__drawdown_raw"] = _daily_robust_zscore(frame.assign(__neg_drawdown=-pd.to_numeric(frame.get("trend_drawdown20"), errors="coerce")), "__neg_drawdown")
    frame["__broken_ma"] = pd.to_numeric(frame.get("trend_broken_ma"), errors="coerce").fillna(0.0)

    frame["trend_TR"] = _daily_robust_zscore(frame, "__trend_strength_raw")
    frame["trend_MF"] = frame["__mf_raw"]
    frame["trend_OH"] = _daily_robust_zscore(
        frame.assign(
            __oh_raw=(
                0.80 * _daily_robust_zscore(frame, "trend_roc3")
                + 0.80 * _daily_robust_zscore(frame, "trend_roc5")
                + 0.50 * frame["__rsv_raw"]
                + 0.40 * frame["__volume_raw"]
                - 0.30 * frame["__mf_raw"]
            )
        ),
        "__oh_raw",
    )
    severe_pullback = pd.to_numeric(frame.get("trend_roc5"), errors="coerce").lt(-0.12) | pd.to_numeric(frame.get("trend_drawdown20"), errors="coerce").lt(-0.18)
    frame["trend_BR"] = _daily_robust_zscore(
        frame.assign(
            __br_raw=(
                -0.70 * _daily_robust_zscore(frame, "trend_roc20")
                -0.50 * _daily_robust_zscore(frame, "trend_roc60")
                -0.50 * frame["__mf_raw"]
                -0.40 * _daily_robust_zscore(frame, "trend_industry_excess20")
                + 0.80 * frame["__broken_ma"]
                + 0.50 * severe_pullback.astype("float64")
            )
        ),
        "__br_raw",
    )
    frame["trend_HP"] = _daily_robust_zscore(
        frame.assign(
            __hp_raw=(
                0.55 * frame["trend_TR"]
                + 0.45 * frame["__pullback_raw"]
                + 0.25 * frame["trend_MF"]
                - 0.25 * frame["trend_OH"].clip(lower=0.0)
                - 0.40 * frame["trend_BR"].clip(lower=0.0)
            )
        ),
        "__hp_raw",
    )
    hp_positive = frame["trend_HP"].clip(lower=0.0, upper=2.0) / 2.0
    d = _daily_robust_zscore(frame, "__direction_logit")
    d_eff = d.clip(lower=0.0) + (1.0 - hp_positive) * d.clip(upper=0.0)
    news = _daily_robust_zscore(frame, "news_count") if "news_count" in frame.columns else pd.Series(0.0, index=frame.index)
    cap = _daily_robust_zscore(frame, "log_total_mv") if "log_total_mv" in frame.columns else pd.Series(0.0, index=frame.index)
    final_score = "final_score" if "final_score" in frame.columns else "y_score"
    return {
        "F": _daily_robust_zscore(frame, final_score),
        "Y": _daily_robust_zscore(frame, "raw_y_score" if "raw_y_score" in frame.columns else "y_score"),
        "R": _daily_robust_zscore(frame, "return_pred"),
        "D": d,
        "D_eff": d_eff.fillna(0.0),
        "TR": frame["trend_TR"].fillna(0.0),
        "HP": frame["trend_HP"].fillna(0.0),
        "MF": frame["trend_MF"].fillna(0.0),
        "NEWS": news.fillna(0.0),
        "OH": frame["trend_OH"].fillna(0.0),
        "BR": frame["trend_BR"].fillna(0.0),
        "CAP": cap.fillna(0.0),
    }


def _s2_trend_pullback_score(components: dict[str, pd.Series]) -> pd.Series:
    return (
        0.42 * components["F"]
        + 0.08 * components["Y"]
        + 0.18 * components["R"]
        + 0.04 * components["D_eff"]
        + 0.12 * components["TR"]
        + 0.15 * components["HP"]
        + 0.10 * components["MF"]
        + 0.06 * components["NEWS"]
        - 0.10 * components["OH"]
        - 0.28 * components["BR"]
        + 0.03 * components["CAP"]
    )


def _ensure_feature_asof_date(work: pd.DataFrame, *, samples_path: str | Path | None) -> pd.DataFrame:
    if "feature_asof_date" in work.columns:
        return work
    if samples_path is None:
        raise ValueError("samples_path is required to recover feature_asof_date for trend-aware strategy scoring")
    samples = ds.dataset(str(samples_path), format="parquet").to_table(columns=["sample_id", "feature_asof_date"]).to_pandas()
    samples["sample_id"] = samples["sample_id"].astype(str)
    out = work.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    samples = samples.loc[samples["sample_id"].isin(set(out["sample_id"]))]
    return out.merge(samples, on="sample_id", how="left")


def _resolve_processed_path(
    explicit_path: str | Path | None,
    metric_path: str | Path | None,
    filename: str,
) -> Path | None:
    if explicit_path is not None:
        return Path(explicit_path)
    if metric_path is None:
        return None
    return Path(metric_path).with_name(filename)


def _load_price_trend_frame(path: Path | None, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if path is None:
        raise ValueError("price_path or metric_path is required for trend-aware strategy scoring")
    table = ds.dataset(str(path), format="parquet").to_table(
        columns=["stock_code", "trade_date", "close", "high", "low", "volume", "amount", "industry"],
        filter=(ds.field("trade_date") >= pd.Timestamp(start)) & (ds.field("trade_date") <= pd.Timestamp(end)),
    )
    price = table.to_pandas()
    price["stock_code"] = price["stock_code"].astype(str)
    price["industry"] = price["industry"].fillna("").astype(str)
    price["trade_date"] = pd.to_datetime(price["trade_date"], errors="coerce").dt.normalize()
    for column in ("close", "high", "low", "volume", "amount"):
        price[column] = pd.to_numeric(price[column], errors="coerce")
    return price.dropna(subset=["stock_code", "trade_date"]).sort_values(["stock_code", "trade_date"])


def _compute_price_trend_features(price: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_stock = price.groupby("stock_code", sort=False)
    price["trend_ret1"] = by_stock["close"].pct_change(fill_method=None)
    for window in (3, 5, 20, 60):
        price[f"trend_roc{window}"] = by_stock["close"].transform(lambda s, w=window: s / s.shift(w) - 1.0)
    price["trend_ma20"] = by_stock["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    price["trend_ma60"] = by_stock["close"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    high20 = by_stock["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    low20 = by_stock["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    vol20 = by_stock["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    price["trend_drawdown20"] = price["close"] / high20.replace(0.0, np.nan) - 1.0
    price["trend_rsv20"] = (price["close"] - low20) / (high20 - low20).replace(0.0, np.nan)
    price["trend_volume_ratio"] = price["volume"] / vol20.replace(0.0, np.nan) - 1.0
    price["trend_broken_ma"] = (price["close"].lt(price["trend_ma20"]) & price["close"].lt(price["trend_ma60"])).astype("float64")
    stock_columns = [
        "stock_code",
        "trade_date",
        "trend_roc3",
        "trend_roc5",
        "trend_roc20",
        "trend_roc60",
        "trend_drawdown20",
        "trend_rsv20",
        "trend_volume_ratio",
        "trend_broken_ma",
        "amount",
    ]
    stock_features = price[stock_columns].rename(columns={"amount": "trend_amount"})

    industry_daily = (
        price.dropna(subset=["industry", "trade_date"])
        .groupby(["industry", "trade_date"], sort=True)["trend_ret1"]
        .mean()
        .reset_index()
        .sort_values(["industry", "trade_date"])
    )
    industry_daily["industry_index"] = (1.0 + industry_daily["trend_ret1"].fillna(0.0)).groupby(industry_daily["industry"]).cumprod()
    by_industry = industry_daily.groupby("industry", sort=False)
    industry_daily["trend_industry_roc20"] = by_industry["industry_index"].transform(lambda s: s / s.shift(20) - 1.0)
    industry_daily["trend_industry_roc60"] = by_industry["industry_index"].transform(lambda s: s / s.shift(60) - 1.0)

    market_daily = price.groupby("trade_date", sort=True)["trend_ret1"].mean().reset_index()
    market_daily["market_index"] = (1.0 + market_daily["trend_ret1"].fillna(0.0)).cumprod()
    market_daily["trend_market_roc20"] = market_daily["market_index"] / market_daily["market_index"].shift(20) - 1.0
    market_daily["trend_market_roc60"] = market_daily["market_index"] / market_daily["market_index"].shift(60) - 1.0
    industry_features = industry_daily.merge(
        market_daily[["trade_date", "trend_market_roc20", "trend_market_roc60"]],
        on="trade_date",
        how="left",
    )
    industry_features["trend_industry_excess20"] = industry_features["trend_industry_roc20"] - industry_features["trend_market_roc20"]
    industry_features["trend_industry_excess60"] = industry_features["trend_industry_roc60"] - industry_features["trend_market_roc60"]
    industry_features = industry_features[
        ["industry", "trade_date", "trend_industry_excess20", "trend_industry_excess60"]
    ]
    market_features = market_daily[["trade_date", "trend_market_roc20", "trend_market_roc60"]]
    return stock_features, industry_features, market_features


def _load_metric_trend_frame(path: str | Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    table = ds.dataset(str(path), format="parquet").to_table(
        columns=["stock_code", "trade_date", "total_mv", "circ_mv", "turnover_rate", "volume_ratio"],
        filter=(ds.field("trade_date") >= pd.Timestamp(start)) & (ds.field("trade_date") <= pd.Timestamp(end)),
    )
    metric = table.to_pandas()
    metric["stock_code"] = metric["stock_code"].astype(str)
    metric["trade_date"] = pd.to_datetime(metric["trade_date"], errors="coerce").dt.normalize()
    return metric


def _load_moneyflow_trend_frame(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    table = ds.dataset(str(path), format="parquet").to_table(
        columns=["stock_code", "trade_date", "net_mf_amount"],
        filter=(ds.field("trade_date") >= pd.Timestamp(start)) & (ds.field("trade_date") <= pd.Timestamp(end)),
    )
    moneyflow = table.to_pandas()
    moneyflow["stock_code"] = moneyflow["stock_code"].astype(str)
    moneyflow["trade_date"] = pd.to_datetime(moneyflow["trade_date"], errors="coerce").dt.normalize()
    moneyflow["net_mf_amount"] = pd.to_numeric(moneyflow["net_mf_amount"], errors="coerce")
    return moneyflow


def generate_trade_signals(
    predictions: pd.DataFrame,
    current_positions: pd.DataFrame | None = None,
    top_n: int = 20,
    daily_replace_k: int = 3,
) -> pd.DataFrame:
    """Generate one-day executable signals for Tonghuashun simulation."""
    _require_columns(predictions, ("target_trade_date", "stock_code", "y_score"), "predictions")
    latest_date = pd.to_datetime(predictions["target_trade_date"]).max()
    day = predictions.loc[pd.to_datetime(predictions["target_trade_date"]).eq(latest_date)].copy()
    day = day.sort_values("y_score", ascending=False).reset_index(drop=True)
    day["rank"] = range(1, len(day) + 1)

    held = _current_holding_set(current_positions)
    if held:
        held_scores = day.loc[day["stock_code"].isin(held)].sort_values("y_score", ascending=True)
        sell = set(held_scores.head(daily_replace_k)["stock_code"].astype(str))
        remaining = held - sell
        buy = []
        for code in day["stock_code"].astype(str):
            if code not in remaining and code not in sell:
                buy.append(code)
            if len(buy) >= daily_replace_k:
                break
        target = (remaining | set(buy))
        if len(target) < top_n:
            for code in day["stock_code"].astype(str):
                if code not in target and code not in sell:
                    target.add(code)
                if len(target) >= top_n:
                    break
    else:
        sell = set()
        target = set(day.head(top_n)["stock_code"].astype(str))
        buy = list(target)

    signal = day.copy()
    signal["current_position"] = signal["stock_code"].astype(str).isin(held)
    signal["suggested_action"] = "watch"
    target_codes = signal["stock_code"].astype(str).isin(target)
    current_codes = signal["stock_code"].astype(str).isin(held)
    signal.loc[target_codes & current_codes, "suggested_action"] = "hold"
    signal.loc[target_codes & ~current_codes, "suggested_action"] = "buy"
    signal.loc[current_codes & ~target_codes, "suggested_action"] = "sell"
    signal["target_weight"] = 0.0
    if target:
        signal.loc[signal["stock_code"].astype(str).isin(target), "target_weight"] = 1.0 / len(target)
    signal["order_note"] = signal["suggested_action"].map(
        {
            "buy": "manual order during 09:30-15:00; adjust price if not filled",
            "sell": "T+1 sell candidate; adjust price if not filled",
            "hold": "keep position to maintain full investment",
            "watch": "not selected today",
        }
    )
    keep = [
        "target_trade_date",
        "stock_code",
        "stock_name",
        "industry",
        "y_score",
        "rank",
        "g_price",
        "g_text",
        "g_fundamental",
        "current_position",
        "suggested_action",
        "target_weight",
        "order_note",
    ]
    for column in keep:
        if column not in signal.columns:
            signal[column] = "" if column in {"stock_name", "industry"} else 0.0
    return signal[keep]


def write_competition_signals(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    current_positions: pd.DataFrame | None = None,
    top_n: int = 20,
    daily_replace_k: int = 3,
) -> Path:
    signals = generate_trade_signals(predictions, current_positions, top_n, daily_replace_k)
    date = pd.to_datetime(signals["target_trade_date"]).max().strftime("%Y%m%d")
    output_path = Path(output_dir) / f"signals_{date}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_path, index=False)
    return output_path


def _current_holding_set(current_positions: pd.DataFrame | None) -> set[str]:
    if current_positions is None or current_positions.empty or "stock_code" not in current_positions.columns:
        return set()
    if "weight" in current_positions.columns:
        frame = current_positions.loc[pd.to_numeric(current_positions["weight"], errors="coerce").fillna(0).gt(0)]
    else:
        frame = current_positions
    return set(frame["stock_code"].astype(str))


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {name} columns: {missing}")
