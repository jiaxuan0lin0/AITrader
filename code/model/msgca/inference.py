from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.msgca.config import MSGCAConfig
from model.msgca.dataset import DayBatchSampler, ScalerState, build_datasets, collate_msgca_batch
from model.msgca.metrics import predictions_from_batch
from model.msgca.modules import MSGCA


def evaluate_checkpoint(
    config: MSGCAConfig,
    checkpoint_path: str | Path | None,
    split: str = "validation",
    limit: int | None = None,
) -> pd.DataFrame:
    state = _load_checkpoint(checkpoint_path)
    feature_scaler = _feature_scaler_from_checkpoint(state)
    dataset, layout, _ = build_datasets(config, split=split, limit=limit, feature_scaler=feature_scaler)
    if state is not None:
        _validate_checkpoint_layout(state, layout)
    model = build_model_from_layout(config, layout)
    if state is not None:
        _load_model_state(model, state)
    return predict_dataset(model, dataset, batch_days=config.train.batch_days)


def _load_checkpoint(checkpoint_path: str | Path | None) -> object | None:
    if checkpoint_path is None:
        return None
    return torch.load(checkpoint_path, map_location="cpu")


def _feature_scaler_from_checkpoint(state: object | None) -> ScalerState | None:
    if not isinstance(state, dict):
        return None
    payload = state.get("feature_scaler")
    if payload is None:
        return None
    return ScalerState.from_dict(payload)


def _validate_checkpoint_layout(state: object, layout) -> None:
    if not isinstance(state, dict):
        return
    saved = state.get("layout")
    if not isinstance(saved, dict):
        return
    for key in ("price_columns", "text_columns", "fundamental_columns", "text_group_ids", "fundamental_group_ids"):
        previous = list(saved.get(key, []))
        current = list(getattr(layout, key, []))
        if previous != current:
            raise ValueError(f"Checkpoint layout mismatch for {key}: {len(previous)} saved vs {len(current)} current")


def _load_model_state(model: torch.nn.Module, state: object) -> None:
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint model state")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"final_score_head.weight", "final_score_head.bias"}
    extra_missing = set(missing) - allowed_missing
    if extra_missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={sorted(extra_missing)} unexpected={sorted(unexpected)}")


def build_model_from_layout(config: MSGCAConfig, layout) -> MSGCA:
    return MSGCA(
        price_variables=len(layout.price_columns),
        lookback=config.data.lookback,
        text_features=len(layout.text_columns),
        fundamental_features=len(layout.fundamental_columns),
        hidden_dim=config.model.hidden_dim,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout,
        price_layers=config.model.price_layers,
        enable_price=config.model.enable_price,
        enable_news=config.model.enable_news,
        enable_fundamental=config.model.enable_fundamental,
        use_gate=config.model.use_gate,
        use_cross_attention=config.model.use_cross_attention,
        max_text_features=config.model.max_text_features,
        max_fundamental_features=config.model.max_fundamental_features,
        factor_encoder=config.model.factor_encoder,
        factor_layers=config.model.factor_layers,
        factor_group_layers=config.model.factor_group_layers,
        factor_group_prototypes=config.model.factor_group_prototypes,
        use_factor_gate=config.model.use_factor_gate,
        factor_gate_activation=config.model.factor_gate_activation,
        text_group_ids=layout.text_group_ids,
        text_group_names=layout.text_group_names,
        fundamental_group_ids=layout.fundamental_group_ids,
        fundamental_group_names=layout.fundamental_group_names,
    )


@torch.no_grad()
def predict_dataset(
    model: MSGCA,
    dataset,
    batch_days: int = 1,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> pd.DataFrame:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    workers = max(int(num_workers or 0), 0)
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_sampler": DayBatchSampler(dataset.samples, batch_days=batch_days, shuffle=False),
        "collate_fn": collate_msgca_batch,
        "num_workers": workers,
        "pin_memory": bool(pin_memory) and device.type == "cuda",
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        **loader_kwargs,
    )
    frames: list[pd.DataFrame] = []
    for batch in loader:
        tensors = {
            key: value.to(device, non_blocking=bool(pin_memory) and device.type == "cuda")
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        output = model(
            tensors["price_window"],
            tensors["price_mask"],
            tensors["text_features"],
            tensors["text_mask"],
            tensors["fundamental_features"],
            tensors["fundamental_mask"],
        )
        frame = predictions_from_batch(
            batch,
            output.y_score.detach().cpu().numpy(),
            output.return_pred.detach().cpu().numpy(),
            torch.sigmoid(output.direction_logit).detach().cpu().numpy(),
            output.gates.detach().cpu().numpy(),
            None if output.final_score is None else output.final_score.detach().cpu().numpy(),
        )
        if output.factor_group_weights is not None:
            names = getattr(model, "factor_group_names", [])
            weights = output.factor_group_weights.detach().cpu().numpy()
            for index in range(weights.shape[1]):
                name = names[index] if index < len(names) else f"group_{index}"
                frame[f"__factor_group__{name}"] = weights[:, index]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "stock_code",
                "stock_name",
                "industry",
                "target_trade_date",
                "y_score",
                "rank",
                "return_pred",
                "direction_prob",
                "final_score",
                "label_next_open_return",
                "label_next_vwap_return",
                "g_price",
                "g_text",
                "g_fundamental",
            ]
        )
    predictions = pd.concat(frames, ignore_index=True)
    predictions["rank"] = predictions.groupby("target_trade_date")["y_score"].rank(method="first", ascending=False)
    return predictions
