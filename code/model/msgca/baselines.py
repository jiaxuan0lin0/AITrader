from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from model.msgca.metrics import summarize_predictions


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    enable_price: bool = True
    enable_news: bool = True
    enable_fundamental: bool = True
    use_gate: bool = True
    use_cross_attention: bool = True


def default_ablation_specs() -> list[AblationSpec]:
    return [
        AblationSpec("msgca_full"),
        AblationSpec("price_only_msgca", enable_news=False, enable_fundamental=False),
        AblationSpec("no_news", enable_news=False),
        AblationSpec("no_fundamental", enable_fundamental=False),
        AblationSpec("no_gate_concat", use_gate=False, use_cross_attention=False),
    ]


def simple_factor_topk_predictions(
    samples: pd.DataFrame,
    factor_col: str,
    label_col: str = "label_next_open_return",
) -> pd.DataFrame:
    if factor_col not in samples.columns:
        raise KeyError(f"Missing factor column for baseline: {factor_col}")
    required = ["sample_id", "stock_code", "target_trade_date", factor_col, label_col]
    frame = samples[[column for column in required if column in samples.columns]].copy()
    frame["y_score"] = pd.to_numeric(frame[factor_col], errors="coerce")
    frame["return_pred"] = frame["y_score"]
    frame["direction_prob"] = 1.0 / (1.0 + np.exp(-frame["y_score"].fillna(0.0)))
    frame["g_price"] = 1.0
    frame["g_text"] = 0.0
    frame["g_fundamental"] = 0.0
    frame["rank"] = frame.groupby("target_trade_date")["y_score"].rank(method="first", ascending=False)
    return frame


def write_baseline_summary(results: dict[str, pd.DataFrame], output_path: str | Path) -> Path:
    rows = []
    for name, predictions in results.items():
        summary = summarize_predictions(predictions)
        rows.append({"baseline": name, **summary.to_dict()})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
