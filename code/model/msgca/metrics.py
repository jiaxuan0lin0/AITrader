from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationSummary:
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_ir: float
    direction_accuracy: float
    topk_return_mean: float
    day_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rank_ic_mean": self.rank_ic_mean,
            "rank_ic_std": self.rank_ic_std,
            "rank_ic_ir": self.rank_ic_ir,
            "direction_accuracy": self.direction_accuracy,
            "topk_return_mean": self.topk_return_mean,
            "day_count": self.day_count,
        }


def add_daily_ranks(predictions: pd.DataFrame, score_col: str = "y_score") -> pd.DataFrame:
    out = predictions.copy()
    out["rank"] = out.groupby("target_trade_date")[score_col].rank(method="first", ascending=False)
    return out


def daily_rank_ic(predictions: pd.DataFrame, label_col: str = "label_next_open_return", score_col: str = "y_score") -> pd.DataFrame:
    required = {"target_trade_date", label_col, score_col}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for date, group in predictions.groupby("target_trade_date", sort=True):
        valid = group[[score_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 2 or valid[score_col].nunique() < 2 or valid[label_col].nunique() < 2:
            rank_ic = np.nan
        else:
            rank_ic = valid[score_col].rank().corr(valid[label_col].rank())
        rows.append({"target_trade_date": date, "rank_ic": rank_ic, "pair_count": len(valid)})
    return pd.DataFrame(rows)


def topk_returns(
    predictions: pd.DataFrame,
    k: int = 20,
    label_col: str = "label_next_open_return",
    score_col: str = "y_score",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in predictions.groupby("target_trade_date", sort=True):
        valid = group[[score_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
        selected = valid.sort_values(score_col, ascending=False).head(k)
        rows.append(
            {
                "target_trade_date": date,
                "topk": k,
                "topk_return": float(selected[label_col].mean()) if not selected.empty else np.nan,
                "selected_count": len(selected),
            }
        )
    return pd.DataFrame(rows)


def direction_accuracy(predictions: pd.DataFrame, label_col: str = "label_next_open_return", score_col: str = "y_score") -> float:
    valid = predictions[[score_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return float("nan")
    if score_col == "direction_prob":
        predicted_up = valid[score_col].gt(0.5)
    else:
        predicted_up = valid[score_col].gt(valid[score_col].median())
    return float((predicted_up == valid[label_col].gt(0)).mean())


def direction_prediction_metrics(
    predictions: pd.DataFrame,
    *,
    prob_col: str = "direction_prob",
    label_col: str = "label_next_open_return",
) -> dict[str, float]:
    if prob_col not in predictions or label_col not in predictions:
        return {}
    frame = pd.DataFrame(
        {
            "prob": pd.to_numeric(predictions[prob_col], errors="coerce"),
            "label_return": pd.to_numeric(predictions[label_col], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {
            "direction_pred_bce": float("nan"),
            "direction_pred_accuracy": float("nan"),
            "direction_pred_up_rate": float("nan"),
            "direction_label_up_rate": float("nan"),
        }
    probs = frame["prob"].clip(1e-7, 1.0 - 1e-7)
    labels = frame["label_return"].gt(0).astype("float64")
    bce = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)).mean()
    predicted_up = probs.gt(0.5)
    return {
        "direction_pred_bce": float(bce),
        "direction_pred_accuracy": float((predicted_up == labels.astype(bool)).mean()),
        "direction_pred_up_rate": float(predicted_up.mean()),
        "direction_label_up_rate": float(labels.mean()),
    }


def return_prediction_metrics(
    predictions: pd.DataFrame,
    *,
    pred_col: str = "return_pred",
    primary_label_col: str = "label_next_open_return",
    secondary_label_col: str = "label_next_vwap_return",
    secondary_weight: float = 0.0,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if pred_col not in predictions or primary_label_col not in predictions:
        return metrics
    metrics.update(_single_return_metrics(predictions, pred_col, primary_label_col, "open"))
    if secondary_label_col in predictions:
        metrics.update(_single_return_metrics(predictions, pred_col, secondary_label_col, "vwap"))
        blend = _blend_return_labels(
            predictions[primary_label_col],
            predictions[secondary_label_col],
            secondary_weight=secondary_weight,
        )
    else:
        blend = pd.to_numeric(predictions[primary_label_col], errors="coerce")
    metrics.update(_return_metrics_from_target(predictions[pred_col], blend, "blend"))
    return metrics


def topk_prediction_metrics(
    predictions: pd.DataFrame,
    *,
    ks: Sequence[int],
    label_col: str = "label_next_open_return",
    score_col: str = "y_score",
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_k in ks:
        k = int(raw_k)
        if k <= 0:
            continue
        frame = topk_returns(predictions, k=k, label_col=label_col, score_col=score_col)
        values = pd.to_numeric(frame["topk_return"], errors="coerce")
        metrics[f"top{k}_return_mean"] = float(values.mean())
    return metrics


def summarize_predictions(predictions: pd.DataFrame, topk: int = 20, score_col: str = "y_score") -> EvaluationSummary:
    rank_ic = daily_rank_ic(predictions, score_col=score_col)
    rank_values = pd.to_numeric(rank_ic["rank_ic"], errors="coerce").dropna()
    mean = float(rank_values.mean()) if not rank_values.empty else float("nan")
    std = float(rank_values.std(ddof=0)) if not rank_values.empty else float("nan")
    ir = mean / std if std and np.isfinite(std) and std > 0 else float("nan")
    topk_frame = topk_returns(predictions, k=topk, score_col=score_col)
    topk_mean = float(pd.to_numeric(topk_frame["topk_return"], errors="coerce").mean())
    return EvaluationSummary(
        rank_ic_mean=mean,
        rank_ic_std=std,
        rank_ic_ir=ir,
        direction_accuracy=direction_accuracy(predictions, score_col=score_col),
        topk_return_mean=topk_mean,
        day_count=int(rank_ic["target_trade_date"].nunique()) if not rank_ic.empty else 0,
    )


def _single_return_metrics(predictions: pd.DataFrame, pred_col: str, label_col: str, suffix: str) -> dict[str, float]:
    return _return_metrics_from_target(predictions[pred_col], predictions[label_col], suffix)


def _return_metrics_from_target(prediction: pd.Series, target: pd.Series, suffix: str) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "prediction": pd.to_numeric(prediction, errors="coerce"),
            "target": pd.to_numeric(target, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    prefix = f"return_pred_{suffix}"
    if frame.empty:
        return {
            f"{prefix}_mse": float("nan"),
            f"{prefix}_mae": float("nan"),
            f"{prefix}_bias": float("nan"),
            f"{prefix}_pearson": float("nan"),
            f"{prefix}_spearman": float("nan"),
            f"{prefix}_direction_accuracy": float("nan"),
        }
    error = frame["prediction"] - frame["target"]
    return {
        f"{prefix}_mse": float(np.square(error).mean()),
        f"{prefix}_mae": float(np.abs(error).mean()),
        f"{prefix}_bias": float(error.mean()),
        f"{prefix}_pearson": _safe_corr(frame["prediction"], frame["target"], method="pearson"),
        f"{prefix}_spearman": _safe_corr(frame["prediction"], frame["target"], method="spearman"),
        f"{prefix}_direction_accuracy": float((frame["prediction"].gt(0) == frame["target"].gt(0)).mean()),
    }


def _blend_return_labels(primary: pd.Series, secondary: pd.Series, *, secondary_weight: float) -> pd.Series:
    weight = min(max(float(secondary_weight), 0.0), 1.0)
    primary_values = pd.to_numeric(primary, errors="coerce")
    secondary_values = pd.to_numeric(secondary, errors="coerce")
    if weight <= 0:
        return primary_values
    if weight >= 1:
        return secondary_values.where(secondary_values.notna(), primary_values)
    blended = (1.0 - weight) * primary_values + weight * secondary_values
    blended = blended.where(primary_values.notna() & secondary_values.notna(), primary_values)
    blended = blended.where(primary_values.notna() | secondary_values.isna(), secondary_values)
    return blended


def _safe_corr(left: pd.Series, right: pd.Series, *, method: str) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    value = left.corr(right, method=method)
    return float(value) if pd.notna(value) else float("nan")


def predictions_from_batch(
    batch: dict[str, object],
    y_score: Sequence[float],
    return_pred: Sequence[float],
    direction_prob: Sequence[float],
    gates: np.ndarray,
    final_score: Sequence[float] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": batch["sample_id"],
            "stock_code": batch["stock_code"],
            "stock_name": batch.get("stock_name", [""] * len(y_score)),
            "industry": batch.get("industry", [""] * len(y_score)),
            "feature_asof_date": pd.to_datetime(batch["feature_asof_date"]),
            "target_trade_date": pd.to_datetime(batch["target_trade_date"]),
            "y_score": y_score,
            "return_pred": return_pred,
            "direction_prob": direction_prob,
            "g_price": gates[:, 0],
            "g_text": gates[:, 1],
            "g_fundamental": gates[:, 2],
        }
    )
    if final_score is not None:
        frame["final_score"] = final_score
    for label in ("label_next_open_return", "label_next_vwap_return"):
        if label in batch:
            values = batch[label]
            if hasattr(values, "detach"):
                frame[label] = values.detach().cpu().numpy()
            else:
                frame[label] = values
    return add_daily_ranks(frame)


def write_evaluation_outputs(predictions: pd.DataFrame, output_root: str | Path, prefix: str = "") -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    factor_group_cols = [column for column in predictions.columns if column.startswith("__factor_group__")]
    public_predictions = predictions.drop(columns=factor_group_cols) if factor_group_cols else predictions
    paths = {
        "predictions": root / f"{stem}predictions.parquet",
        "daily_rankic": root / f"{stem}daily_rankic.csv",
        "portfolio_topk": root / f"{stem}portfolio_topk.csv",
        "validation_metrics": root / f"{stem}validation_metrics.json",
    }
    if factor_group_cols:
        paths["factor_group_attribution"] = root / f"{stem}factor_group_attribution.csv"
    public_predictions.to_parquet(paths["predictions"], index=False)
    daily_rank_ic(public_predictions).to_csv(paths["daily_rankic"], index=False)
    topk_returns(public_predictions).to_csv(paths["portfolio_topk"], index=False)
    summary = summarize_predictions(public_predictions)
    pd.Series(summary.to_dict()).to_json(paths["validation_metrics"], force_ascii=False, indent=2)
    if factor_group_cols:
        _factor_group_attribution(predictions, factor_group_cols).to_csv(paths["factor_group_attribution"], index=False)
    return paths


def _factor_group_attribution(predictions: pd.DataFrame, factor_group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.to_datetime(predictions["target_trade_date"], errors="coerce") if "target_trade_date" in predictions.columns else None
    for column in factor_group_cols:
        values = pd.to_numeric(predictions[column], errors="coerce")
        rows.append(
            {
                "group": column.removeprefix("__factor_group__"),
                "mean_weight": float(values.mean()),
                "nonzero_rate": float(values.gt(0).mean()),
                "row_count": int(values.notna().sum()),
                "date_min": "" if dates is None or dates.dropna().empty else str(dates.min().date()),
                "date_max": "" if dates is None or dates.dropna().empty else str(dates.max().date()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_weight", ascending=False).reset_index(drop=True)
