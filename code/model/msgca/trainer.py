from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.msgca.config import MSGCAConfig, write_resolved_config
from model.msgca.competition_metrics import score_competition_predictions
from model.msgca.dataset import (
    DayBatchSampler,
    ScalerState,
    build_datasets,
    build_train_dataset,
    build_train_validation_datasets,
    collate_msgca_batch,
)
from model.msgca.inference import build_model_from_layout, predict_dataset
from model.msgca.losses import AuxRankLoss, AuxTopKLoss, LossWeights, msgca_loss, set_torch_seed, strategy_window_return_loss_from_output
from model.msgca.metrics import (
    direction_prediction_metrics,
    return_prediction_metrics,
    summarize_predictions,
    topk_prediction_metrics,
    write_evaluation_outputs,
)


def train_msgca(
    config: MSGCAConfig,
    limit: int | None = None,
    device: str | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_weights_only: bool = False,
) -> dict[str, object]:
    set_torch_seed(config.train.seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    started = time.monotonic()
    _log(f"device={device_obj} limit={limit} epochs={config.train.epochs}")
    _log("build_datasets_start")
    valid_dataset = None
    if _needs_validation_dataset(config):
        train_dataset, valid_dataset, layout, feature_scaler = build_train_validation_datasets(
            config,
            limit=limit,
        )
    else:
        train_dataset, layout, feature_scaler = build_train_dataset(
            config,
            limit=limit,
        )
    _log(
        "build_datasets_done "
        f"elapsed_sec={time.monotonic() - started:.1f} "
        f"train_rows={len(train_dataset)} valid_rows={len(valid_dataset) if valid_dataset is not None else 0} "
        f"price_vars={len(layout.price_columns)} text_features={len(layout.text_columns)} "
        f"fundamental_features={len(layout.fundamental_columns)}"
    )
    model = build_model_from_layout(config, layout).to(device_obj)
    resume_state = _load_resume_checkpoint(resume_checkpoint, device_obj)
    start_epoch = 0
    if resume_state is not None:
        _validate_resume_layout(resume_state, layout)
        _load_model_state(model, resume_state)
        start_epoch = 0 if resume_weights_only else _checkpoint_epoch(resume_state)
        mode = "weights_only" if resume_weights_only else "resume"
        _log(f"resume_checkpoint_loaded path={resume_checkpoint} mode={mode} start_epoch={start_epoch}")
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp and device_obj.type == "cuda")
    amp_scaler_state = _checkpoint_amp_scaler(resume_state)
    if amp_scaler_state is not None and not resume_weights_only:
        try:
            scaler.load_state_dict(amp_scaler_state)
            _log("resume_amp_scaler_loaded")
        except RuntimeError as exc:
            if "source state dict is empty" not in str(exc):
                raise
            _log("resume_amp_scaler_skipped empty_state")
    if resume_state is not None and not resume_weights_only and _restore_rng_state(resume_state):
        _log("resume_rng_state_loaded")
    root = config.paths.output_root
    checkpoint_path = root / "checkpoints" / "msgca_latest.pt"
    train_log_path = root / "train_log.csv"
    train_log: list[dict[str, object]] = []
    best_metric_value: float | None = None
    best_metric_name = str(getattr(config.train, "best_checkpoint_metric", "competition_score"))
    best_metric_mode = str(getattr(config.train, "best_checkpoint_mode", "max"))
    if start_epoch > 0:
        best_metric_value = _load_existing_best_metric(root, best_metric_name, best_metric_mode)
        if best_metric_value is not None:
            _log(f"resume_best_metric_loaded metric={best_metric_name} value={best_metric_value}")
        train_log = _load_existing_train_log(train_log_path, start_epoch)
        if train_log:
            _log(f"resume_train_log_loaded rows={len(train_log)}")
    loader = DataLoader(
        **_data_loader_kwargs(train_dataset, config, shuffle=True, device=device_obj),
    )
    stages = _stage_specs(config, start_epoch=start_epoch)
    strategy_dataset = None
    strategy_day_indices: list[tuple[pd.Timestamp, list[int]]] = []
    if _needs_strategy_window_dataset(config, stages):
        if str(getattr(config.train, "strategy_window_pool_split", "custom") or "custom").lower() == "train":
            strategy_dataset = train_dataset
            _log("strategy_window_dataset_use_train_split")
        else:
            strategy_dataset, strategy_layout, _ = _build_strategy_window_dataset(
                config,
                feature_scaler,
                limit=limit,
            )
            _validate_strategy_layout(strategy_layout, layout)
        strategy_day_indices = _strategy_day_indices(strategy_dataset.samples)
        _log(
            "strategy_window_dataset_done "
            f"rows={len(strategy_dataset)} days={len(strategy_day_indices)} "
            f"pool={config.train.strategy_window_pool_start}:{config.train.strategy_window_pool_end}"
        )
    _log("stage_plan " + " ".join(f"{stage.name}:{stage.epochs}" for stage in stages))
    global_epoch = start_epoch

    for stage_index, stage in enumerate(stages):
        if stage.epochs <= 0:
            continue
        trainable_count, total_count = _set_trainable_parameters(model, stage.trainable)
        _log(
            "stage_start "
            f"name={stage.name} epochs={stage.epochs} trainable={stage.trainable} "
            f"params={trainable_count}/{total_count} lr={stage.learning_rate}"
        )
        optimizer = torch.optim.AdamW(
            _trainable_parameters(model),
            lr=stage.learning_rate,
            weight_decay=stage.weight_decay,
        )
        if (
            resume_state is not None
            and not resume_weights_only
            and stage_index == 0
            and "optimizer" in resume_state
            and _can_resume_optimizer(resume_state, stage)
        ):
            optimizer.load_state_dict(resume_state["optimizer"])
            _log("resume_optimizer_loaded")
        weights = _loss_weights_for_stage(config, stage)
        stage_best_metric_value: float | None = None
        stage_best_path = root / "checkpoints" / f"msgca_stage_{stage.name}_best.pt"
        if start_epoch > 0:
            stage_best_metric_value = _load_best_metric_json(
                root / "checkpoints" / f"msgca_stage_{stage.name}_best.json",
                stage.best_metric_name,
                stage.best_metric_mode,
            )
            if stage_best_metric_value is not None:
                _log(f"resume_stage_best_metric_loaded stage={stage.name} metric={stage.best_metric_name} value={stage_best_metric_value}")
        stage_wait = 0

        for stage_epoch in range(1, stage.epochs + 1):
            global_epoch += 1
            epoch = global_epoch
            epoch_started = time.monotonic()
            _log(f"epoch_start epoch={epoch} stage={stage.name} stage_epoch={stage_epoch}")
            model.train()
            epoch_losses: list[float] = []
            epoch_parts: dict[str, list[float]] = {}
            for batch in loader:
                tensors = _batch_tensors_to_device(batch, config, device_obj)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=config.train.amp and device_obj.type == "cuda"):
                    output = model(
                        tensors["price_window"],
                        tensors["price_mask"],
                        tensors["text_features"],
                        tensors["text_mask"],
                        tensors["fundamental_features"],
                        tensors["fundamental_mask"],
                    )
                    loss, parts = msgca_loss(
                        output,
                        tensors["label_next_open_return"],
                        tensors["label_next_vwap_return"],
                        tensors["label_direction"],
                        batch["target_trade_date"],
                        weights,
                        config.train.max_pairs_per_day,
                        sample_weights=_batch_time_weights(batch["target_trade_date"], config, device_obj),
                        context=tensors.get("loss_context"),
                        context_columns=getattr(train_dataset, "context_columns", []),
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(float(loss.detach().cpu()))
                for key, value in parts.items():
                    epoch_parts.setdefault(key, []).append(float(value.detach().cpu()))

            strategy_parts = _run_strategy_window_training(
                model,
                strategy_dataset,
                strategy_day_indices,
                optimizer,
                scaler,
                config,
                stage,
                device_obj,
                epoch=epoch,
                stage_epoch=stage_epoch,
            )
            for key, value in strategy_parts.items():
                epoch_parts.setdefault(key, []).append(float(value))

            should_validate = valid_dataset is not None and _should_validate_epoch(config, stage, stage_epoch)
            summary = None
            competition_summary: dict[str, object] = {}
            return_metrics: dict[str, float] = {}
            direction_metrics: dict[str, float] = {}
            topk_metrics: dict[str, float] = {}
            if should_validate:
                valid_predictions = predict_dataset(
                    model,
                    valid_dataset,
                    batch_days=config.train.batch_days,
                    device=device_obj,
                    num_workers=config.train.dataloader_workers,
                    pin_memory=config.train.dataloader_pin_memory,
                )
                if not valid_predictions.empty:
                    summary = summarize_predictions(valid_predictions, topk=config.strategy.top_n, score_col=stage.validation_score_col)
                    return_metrics = return_prediction_metrics(
                        valid_predictions,
                        secondary_weight=float(_stage_value(stage.raw, "return_secondary_weight", config.train.return_secondary_weight)),
                    )
                    direction_metrics = direction_prediction_metrics(valid_predictions)
                    topk_metrics = topk_prediction_metrics(
                        valid_predictions,
                        ks=_stage_topk_metric_ks(config, stage),
                        score_col=stage.validation_score_col,
                    )
                if getattr(config.train, "competition_validate", True) and not valid_predictions.empty:
                    competition_summary = score_competition_predictions(
                        valid_predictions,
                        _stage_strategy(config, stage),
                        samples_path=config.paths.samples_path,
                        metric_path=config.paths.metric_path,
                        price_path=config.paths.price_path,
                        feature_registry_path=config.paths.feature_registry_path,
                        news_path=config.paths.news_path,
                        news_scores_path=config.paths.news_scores_path,
                        context_cache_path=getattr(config.train, "context_cache_path", None),
                        news_cache_path=getattr(config.train, "context_news_cache_path", None),
                        window_days=config.train.competition_window_days,
                        recent_window_count=config.train.competition_recent_window_count,
                    )
            log_row = {
                "epoch": epoch,
                "stage": stage.name,
                "stage_epoch": stage_epoch,
                "trainable": stage.trainable,
                "train_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                **_mean_loss_parts(epoch_parts),
                **({} if summary is None else summary.to_dict()),
                **return_metrics,
                **direction_metrics,
                **topk_metrics,
                **_competition_log_columns(competition_summary),
            }
            train_log.append(log_row)
            _log(
                "epoch_done "
                f"epoch={epoch} stage={stage.name} elapsed_sec={time.monotonic() - epoch_started:.1f} "
                f"train_loss={train_log[-1]['train_loss']}"
            )
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scaler,
                config,
                layout,
                feature_scaler,
                epoch,
                stage_name=stage.name,
                stage_epoch=stage_epoch,
            )
            if getattr(config.train, "save_epoch_checkpoints", True):
                _save_checkpoint(
                    root / "checkpoints" / f"msgca_epoch_{epoch}.pt",
                    model,
                    optimizer,
                    scaler,
                    config,
                    layout,
                    feature_scaler,
                    epoch,
                    stage_name=stage.name,
                    stage_epoch=stage_epoch,
                )

            candidate_metric = train_log[-1].get(best_metric_name)
            if _metric_improved(candidate_metric, best_metric_value, mode=best_metric_mode):
                best_metric_value = float(candidate_metric)
                best_path = root / "checkpoints" / "msgca_best.pt"
                _save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scaler,
                    config,
                    layout,
                    feature_scaler,
                    epoch,
                    stage_name=stage.name,
                    stage_epoch=stage_epoch,
                )
                _write_best_checkpoint_json(
                    root / "checkpoints" / "msgca_best.json",
                    best_path,
                    epoch,
                    stage_epoch,
                    best_metric_name,
                    best_metric_mode,
                    best_metric_value,
                    stage,
                )

            stage_candidate_metric = train_log[-1].get(stage.best_metric_name)
            stage_improved = _metric_improved(stage_candidate_metric, stage_best_metric_value, mode=stage.best_metric_mode)
            if stage_improved:
                stage_best_metric_value = float(stage_candidate_metric)
                stage_wait = 0
                _save_checkpoint(
                    stage_best_path,
                    model,
                    optimizer,
                    scaler,
                    config,
                    layout,
                    feature_scaler,
                    epoch,
                    stage_name=stage.name,
                    stage_epoch=stage_epoch,
                )
                _write_best_checkpoint_json(
                    root / "checkpoints" / f"msgca_stage_{stage.name}_best.json",
                    stage_best_path,
                    epoch,
                    stage_epoch,
                    stage.best_metric_name,
                    stage.best_metric_mode,
                    stage_best_metric_value,
                    stage,
                )
            elif should_validate and stage.patience > 0:
                stage_wait += 1

            pd.DataFrame(train_log).to_csv(train_log_path, index=False)

            if should_validate and stage.patience > 0 and stage_epoch >= stage.min_epochs and stage_wait >= stage.patience:
                _log(
                    "stage_early_stop "
                    f"name={stage.name} stage_epoch={stage_epoch} patience={stage.patience} "
                    f"metric={stage.best_metric_name} best={stage_best_metric_value}"
                )
                break

        if stage.restore_best and stage_best_path.exists():
            state = torch.load(stage_best_path, map_location=device_obj)
            _load_model_state(model, state)
            _log(f"stage_best_restored name={stage.name} checkpoint={stage_best_path}")

    write_resolved_config(config, root / "config.resolved.yaml")
    validation_predictions_path = None
    if valid_dataset is not None and config.train.final_validate:
        _log("final_validation_start")
        validation_predictions = predict_dataset(
            model,
            valid_dataset,
            batch_days=config.train.batch_days,
            device=device_obj,
            num_workers=config.train.dataloader_workers,
            pin_memory=config.train.dataloader_pin_memory,
        )
        output_paths = write_evaluation_outputs(validation_predictions, root, prefix="validation")
        validation_predictions_path = output_paths["predictions"]
        _log(f"final_validation_done elapsed_sec={time.monotonic() - started:.1f}")
    else:
        _log(f"final_validation_skipped elapsed_sec={time.monotonic() - started:.1f}")
    return {
        "checkpoint_path": checkpoint_path,
        "train_log_path": train_log_path,
        "validation_predictions_path": validation_predictions_path,
        "layout": layout,
        "feature_scaler": feature_scaler,
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: MSGCAConfig,
    layout,
    feature_scaler: ScalerState,
    epoch: int,
    *,
    stage_name: str | None = None,
    stage_epoch: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "amp_scaler": scaler.state_dict(),
            "scaler": scaler.state_dict(),
            "feature_scaler": feature_scaler.to_dict(),
            "epoch": int(epoch),
            "config": config.to_dict(),
            "layout": layout.__dict__,
            "rng_state": _capture_rng_state(),
            "stage": None if stage_name is None else {"name": stage_name, "epoch": int(stage_epoch or 0)},
        },
        path,
    )


def _data_loader_kwargs(dataset, config: MSGCAConfig, *, shuffle: bool, device: torch.device) -> dict[str, object]:
    workers = max(int(getattr(config.train, "dataloader_workers", 0) or 0), 0)
    pin_memory = bool(getattr(config.train, "dataloader_pin_memory", False)) and device.type == "cuda"
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_sampler": DayBatchSampler(
            dataset.samples,
            batch_days=config.train.batch_days,
            shuffle=shuffle,
            seed=config.train.seed,
        ),
        "collate_fn": collate_msgca_batch,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(getattr(config.train, "dataloader_persistent_workers", True))
        kwargs["prefetch_factor"] = max(int(getattr(config.train, "dataloader_prefetch_factor", 2) or 2), 1)
    return kwargs


def _batch_tensors_to_device(batch: dict[str, object], config: MSGCAConfig, device: torch.device) -> dict[str, torch.Tensor]:
    non_blocking = bool(getattr(config.train, "dataloader_pin_memory", False)) and device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def _can_resume_optimizer(checkpoint: dict[str, object], stage: "_StageSpec") -> bool:
    saved_stage = checkpoint.get("stage")
    if not isinstance(saved_stage, dict):
        return True
    saved_name = saved_stage.get("name")
    return saved_name is None or str(saved_name) == stage.name


def _load_resume_checkpoint(path: str | Path | None, device: torch.device) -> dict[str, object] | None:
    if path is None:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid resume checkpoint: {checkpoint_path}")
    return checkpoint


def _load_model_state(model: torch.nn.Module, checkpoint: dict[str, object]) -> None:
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model state")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"final_score_head.weight", "final_score_head.bias"}
    extra_missing = set(missing) - allowed_missing
    if extra_missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={sorted(extra_missing)} unexpected={sorted(unexpected)}")


def _checkpoint_epoch(checkpoint: dict[str, object]) -> int:
    value = checkpoint.get("epoch")
    if value is not None:
        return int(value)
    raw_config = checkpoint.get("config", {})
    if isinstance(raw_config, dict):
        train_config = raw_config.get("train", {})
        if isinstance(train_config, dict) and train_config.get("epochs") is not None:
            return int(train_config["epochs"])
    return 0


def _checkpoint_amp_scaler(checkpoint: dict[str, object] | None) -> object | None:
    if checkpoint is None:
        return None
    state = checkpoint.get("amp_scaler") or checkpoint.get("scaler")
    if isinstance(state, dict) and not state:
        return None
    return state


def _capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(checkpoint: dict[str, object]) -> bool:
    state = checkpoint.get("rng_state")
    if not isinstance(state, dict):
        return False
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, dict):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.detach().cpu())
    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list) and cuda_state:
        torch.cuda.set_rng_state_all([item.detach().cpu() if isinstance(item, torch.Tensor) else item for item in cuda_state])
    return True


@dataclass(frozen=True)
class _StageSpec:
    name: str
    epochs: int
    min_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    trainable: str
    restore_best: bool
    validation_interval: int
    best_metric_name: str
    best_metric_mode: str
    validation_score_col: str
    validation_score_variant: str | None
    raw: dict[str, object]


@dataclass(frozen=True)
class _StrategyWindowSettings:
    weight: float
    pool_split: str
    pool_start: str | None
    pool_end: str | None
    window_days: int
    samples_per_epoch: int
    label: str
    score_field: str
    top_n: int
    mode: str
    temperature: float
    return_weight: float
    excess_weight: float
    downside_weight: float
    seed: int
    month_cap: float


def _stage_specs(config: MSGCAConfig, *, start_epoch: int) -> list[_StageSpec]:
    raw_stages = getattr(config.train, "stages", None) or []
    if not raw_stages:
        remaining = max(int(config.train.epochs) - int(start_epoch), 0)
        return [
            _StageSpec(
                name="default",
                epochs=remaining,
                min_epochs=int(getattr(config.train, "early_stop_min_epochs", 0) or 0),
                patience=int(getattr(config.train, "early_stop_patience", 0) or 0),
                learning_rate=float(config.train.learning_rate),
                weight_decay=float(config.train.weight_decay),
                trainable="all",
                restore_best=bool(getattr(config.train, "restore_best_checkpoint", False)),
                validation_interval=int(config.train.validation_interval),
                best_metric_name=str(getattr(config.train, "best_checkpoint_metric", "competition_score")),
                best_metric_mode=str(getattr(config.train, "best_checkpoint_mode", "max")),
                validation_score_col=str(getattr(config.train, "validation_score_col", "y_score") or "y_score"),
                validation_score_variant=getattr(config.train, "validation_score_variant", None),
                raw={},
            )
        ]

    stages: list[_StageSpec] = []
    remaining_skip = max(int(start_epoch), 0)
    for index, item in enumerate(raw_stages, start=1):
        if not isinstance(item, dict):
            raise TypeError("train.stages entries must be mappings")
        name = str(item.get("name") or f"stage{index}")
        configured_epochs = int(_stage_value(item, "epochs", config.train.epochs))
        if remaining_skip >= configured_epochs:
            remaining_skip -= configured_epochs
            continue
        epochs = configured_epochs - remaining_skip
        min_epochs = max(int(_stage_value(item, "min_epochs", getattr(config.train, "early_stop_min_epochs", 0) or 0)) - remaining_skip, 0)
        remaining_skip = 0
        score_col = str(_stage_value(item, "validation_score_col", getattr(config.train, "validation_score_col", "y_score")) or "y_score")
        stages.append(
            _StageSpec(
                name=_safe_stage_name(name),
                epochs=epochs,
                min_epochs=min_epochs,
                patience=int(_stage_value(item, "patience", getattr(config.train, "early_stop_patience", 0) or 0)),
                learning_rate=float(_stage_value(item, "learning_rate", config.train.learning_rate)),
                weight_decay=float(_stage_value(item, "weight_decay", config.train.weight_decay)),
                trainable=str(_stage_value(item, "trainable", "all")),
                restore_best=bool(_stage_value(item, "restore_best", True)),
                validation_interval=int(_stage_value(item, "validation_interval", config.train.validation_interval)),
                best_metric_name=str(_stage_value(item, "best_checkpoint_metric", getattr(config.train, "best_checkpoint_metric", "competition_score"))),
                best_metric_mode=str(_stage_value(item, "best_checkpoint_mode", getattr(config.train, "best_checkpoint_mode", "max"))),
                validation_score_col=score_col,
                validation_score_variant=_stage_value(item, "validation_score_variant", getattr(config.train, "validation_score_variant", None)),
                raw=dict(item),
            )
        )
    return stages


def _stage_value(stage: dict[str, object], key: str, default: object) -> object:
    value = stage.get(key, default)
    return default if value is None else value


def _safe_stage_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "stage"


def _needs_validation_dataset(config: MSGCAConfig) -> bool:
    if config.train.final_validate or config.train.validate_each_epoch:
        return True
    raw_stages = getattr(config.train, "stages", None) or []
    if raw_stages:
        return any(int(item.get("patience", 0) or 0) > 0 for item in raw_stages if isinstance(item, dict))
    return int(getattr(config.train, "early_stop_patience", 0) or 0) > 0


def _should_validate_epoch(config: MSGCAConfig, stage: _StageSpec, stage_epoch: int) -> bool:
    if not (config.train.validate_each_epoch or stage.patience > 0):
        return False
    return stage.validation_interval <= 0 or stage_epoch % stage.validation_interval == 0


def _loss_weights_for_stage(config: MSGCAConfig, stage: _StageSpec) -> LossWeights:
    return LossWeights(
        rank=float(_stage_value(stage.raw, "rank_loss_weight", config.train.rank_loss_weight)),
        return_mse=float(_stage_value(stage.raw, "return_loss_weight", config.train.return_loss_weight)),
        secondary_return=float(
            _stage_value(stage.raw, "secondary_return_loss_weight", config.train.secondary_return_loss_weight)
        ),
        direction_bce=float(_stage_value(stage.raw, "direction_loss_weight", config.train.direction_loss_weight)),
        return_loss_type=str(_stage_value(stage.raw, "return_loss_type", config.train.return_loss_type)),
        return_huber_delta=float(_stage_value(stage.raw, "return_huber_delta", config.train.return_huber_delta)),
        rank_label_mode=str(_stage_value(stage.raw, "rank_label_mode", config.train.rank_label_mode)),
        topk_label_mode=str(_stage_value(stage.raw, "topk_label_mode", config.train.topk_label_mode)),
        rank_topk_k=int(_stage_value(stage.raw, "rank_topk_k", config.train.rank_topk_k)),
        direction_label_mode=str(_stage_value(stage.raw, "direction_label_mode", config.train.direction_label_mode)),
        direction_label_epsilon=float(
            _stage_value(stage.raw, "direction_label_epsilon", config.train.direction_label_epsilon)
        ),
        direction_label_std_fraction=float(
            _stage_value(stage.raw, "direction_label_std_fraction", config.train.direction_label_std_fraction)
        ),
        topk_return=float(_stage_value(stage.raw, "topk_return_loss_weight", config.train.topk_return_loss_weight)),
        return_secondary_weight=float(_stage_value(stage.raw, "return_secondary_weight", config.train.return_secondary_weight)),
        topk_secondary_weight=float(_stage_value(stage.raw, "topk_secondary_weight", config.train.topk_secondary_weight)),
        topk_k=int(_stage_value(stage.raw, "topk_return_k", config.train.topk_return_k or config.strategy.top_n)),
        topk_mode=str(_stage_value(stage.raw, "topk_return_mode", config.train.topk_return_mode)),
        topk_temperature=float(_stage_value(stage.raw, "topk_temperature", config.train.topk_temperature)),
        gate_entropy=float(_stage_value(stage.raw, "gate_entropy_weight", config.train.gate_entropy_weight)),
        rank_score_field=str(_stage_value(stage.raw, "rank_score_field", getattr(config.train, "rank_score_field", "y_score"))),
        topk_score_field=str(_stage_value(stage.raw, "topk_score_field", getattr(config.train, "topk_score_field", "y_score"))),
        aux_rank=_aux_rank_losses(_stage_value(stage.raw, "aux_rank_losses", getattr(config.train, "aux_rank_losses", []))),
        aux_topk=_aux_topk_losses(_stage_value(stage.raw, "aux_topk_return_losses", getattr(config.train, "aux_topk_return_losses", []))),
        combined_topk=float(_stage_value(stage.raw, "combined_topk_loss_weight", config.train.combined_topk_loss_weight)),
        combined_topk_k=int(_stage_value(stage.raw, "combined_topk_k", config.train.combined_topk_k)),
        combined_topk_mode=str(_stage_value(stage.raw, "combined_topk_mode", config.train.combined_topk_mode)),
        combined_topk_temperature=float(
            _stage_value(stage.raw, "combined_topk_temperature", config.train.combined_topk_temperature)
        ),
        combined_weight_final=float(_stage_value(stage.raw, "combined_weight_final", config.train.combined_weight_final)),
        combined_weight_return=float(_stage_value(stage.raw, "combined_weight_return", config.train.combined_weight_return)),
        combined_weight_direction=float(
            _stage_value(stage.raw, "combined_weight_direction", config.train.combined_weight_direction)
        ),
        combined_weight_y=float(_stage_value(stage.raw, "combined_weight_y", config.train.combined_weight_y)),
        consistency=float(_stage_value(stage.raw, "consistency_loss_weight", config.train.consistency_loss_weight)),
        consistency_score_field=str(
            _stage_value(stage.raw, "consistency_score_field", config.train.consistency_score_field)
        ),
        consistency_topk_k=int(_stage_value(stage.raw, "consistency_topk_k", config.train.consistency_topk_k)),
        consistency_temperature=float(
            _stage_value(stage.raw, "consistency_temperature", config.train.consistency_temperature)
        ),
        consistency_return_weight=float(
            _stage_value(stage.raw, "consistency_return_weight", config.train.consistency_return_weight)
        ),
        consistency_direction_weight=float(
            _stage_value(stage.raw, "consistency_direction_weight", config.train.consistency_direction_weight)
        ),
        context_score_topk=float(
            _stage_value(stage.raw, "context_score_topk_loss_weight", config.train.context_score_topk_loss_weight)
        ),
        context_score_variant=str(_stage_value(stage.raw, "context_score_variant", config.train.context_score_variant)),
        context_score_topk_k=int(_stage_value(stage.raw, "context_score_topk_k", config.train.context_score_topk_k)),
        context_score_topk_mode=str(_stage_value(stage.raw, "context_score_topk_mode", config.train.context_score_topk_mode)),
        context_score_topk_temperature=float(
            _stage_value(stage.raw, "context_score_topk_temperature", config.train.context_score_topk_temperature)
        ),
        trend_adjusted_topk=float(
            _stage_value(stage.raw, "trend_adjusted_topk_loss_weight", config.train.trend_adjusted_topk_loss_weight)
        ),
        trend_adjusted_topk_k=int(_stage_value(stage.raw, "trend_adjusted_topk_k", config.train.trend_adjusted_topk_k)),
        trend_adjusted_topk_mode=str(_stage_value(stage.raw, "trend_adjusted_topk_mode", config.train.trend_adjusted_topk_mode)),
        trend_adjusted_topk_temperature=float(
            _stage_value(stage.raw, "trend_adjusted_topk_temperature", config.train.trend_adjusted_topk_temperature)
        ),
        trend_adjusted_score_field=str(
            _stage_value(stage.raw, "trend_adjusted_score_field", config.train.trend_adjusted_score_field)
        ),
        trend_adjusted_positive_weight=float(
            _stage_value(stage.raw, "trend_adjusted_positive_weight", config.train.trend_adjusted_positive_weight)
        ),
        trend_overheat_negative_weight=float(
            _stage_value(stage.raw, "trend_overheat_negative_weight", config.train.trend_overheat_negative_weight)
        ),
        trend_broken_negative_weight=float(
            _stage_value(stage.raw, "trend_broken_negative_weight", config.train.trend_broken_negative_weight)
        ),
        direction_healthy_pullback_discount=float(
            _stage_value(
                stage.raw,
                "direction_healthy_pullback_discount",
                config.train.direction_healthy_pullback_discount,
            )
        ),
        cluster_topk=float(_stage_value(stage.raw, "cluster_topk_loss_weight", config.train.cluster_topk_loss_weight)),
        cluster_column=str(_stage_value(stage.raw, "cluster_column", config.train.cluster_column)),
        cluster_topk_score_field=str(
            _stage_value(stage.raw, "cluster_topk_score_field", config.train.cluster_topk_score_field)
        ),
        cluster_topk_label_mode=str(
            _stage_value(stage.raw, "cluster_topk_label_mode", config.train.cluster_topk_label_mode)
        ),
        cluster_topk_k=int(_stage_value(stage.raw, "cluster_topk_k", config.train.cluster_topk_k)),
        cluster_topk_mode=str(_stage_value(stage.raw, "cluster_topk_mode", config.train.cluster_topk_mode)),
        cluster_topk_temperature=float(
            _stage_value(stage.raw, "cluster_topk_temperature", config.train.cluster_topk_temperature)
        ),
        cluster_topk_min_size=int(_stage_value(stage.raw, "cluster_topk_min_size", config.train.cluster_topk_min_size)),
        cluster_topk_member_k=int(_stage_value(stage.raw, "cluster_topk_member_k", config.train.cluster_topk_member_k)),
        cluster_topk_member_temperature=float(
            _stage_value(stage.raw, "cluster_topk_member_temperature", config.train.cluster_topk_member_temperature)
        ),
        cluster_rank=float(_stage_value(stage.raw, "cluster_rank_loss_weight", config.train.cluster_rank_loss_weight)),
        cluster_rank_score_field=str(
            _stage_value(stage.raw, "cluster_rank_score_field", config.train.cluster_rank_score_field)
        ),
        cluster_rank_label_mode=str(
            _stage_value(stage.raw, "cluster_rank_label_mode", config.train.cluster_rank_label_mode)
        ),
        cluster_rank_min_size=int(_stage_value(stage.raw, "cluster_rank_min_size", config.train.cluster_rank_min_size)),
        cluster_rank_member_k=int(_stage_value(stage.raw, "cluster_rank_member_k", config.train.cluster_rank_member_k)),
        cluster_rank_member_temperature=float(
            _stage_value(stage.raw, "cluster_rank_member_temperature", config.train.cluster_rank_member_temperature)
        ),
        cluster_rank_max_pairs_per_day=int(
            _stage_value(stage.raw, "cluster_rank_max_pairs_per_day", config.train.cluster_rank_max_pairs_per_day)
        ),
        in_cluster_rank=float(
            _stage_value(stage.raw, "in_cluster_rank_loss_weight", config.train.in_cluster_rank_loss_weight)
        ),
        in_cluster_rank_score_field=str(
            _stage_value(stage.raw, "in_cluster_rank_score_field", config.train.in_cluster_rank_score_field)
        ),
        in_cluster_rank_label_mode=str(
            _stage_value(stage.raw, "in_cluster_rank_label_mode", config.train.in_cluster_rank_label_mode)
        ),
        in_cluster_rank_topk_k=int(
            _stage_value(stage.raw, "in_cluster_rank_topk_k", config.train.in_cluster_rank_topk_k)
        ),
        in_cluster_rank_min_size=int(
            _stage_value(stage.raw, "in_cluster_rank_min_size", config.train.in_cluster_rank_min_size)
        ),
        in_cluster_rank_max_clusters_per_day=int(
            _stage_value(
                stage.raw,
                "in_cluster_rank_max_clusters_per_day",
                config.train.in_cluster_rank_max_clusters_per_day,
            )
        ),
        in_cluster_rank_max_pairs_per_cluster=int(
            _stage_value(
                stage.raw,
                "in_cluster_rank_max_pairs_per_cluster",
                config.train.in_cluster_rank_max_pairs_per_cluster,
            )
        ),
    )


def _stage_topk_metric_ks(config: MSGCAConfig, stage: _StageSpec) -> list[int]:
    values = {
        int(getattr(config.strategy, "top_n", 20) or 20),
        int(_stage_value(stage.raw, "topk_return_k", config.train.topk_return_k or config.strategy.top_n)),
        int(_stage_value(stage.raw, "strategy_window_top_n", config.train.strategy_window_top_n or config.strategy.top_n)),
    }
    for aux in _aux_topk_losses(_stage_value(stage.raw, "aux_topk_return_losses", getattr(config.train, "aux_topk_return_losses", []))):
        values.add(int(aux.k))
    return sorted(value for value in values if value > 0)


def _strategy_window_settings_for_stage(config: MSGCAConfig, stage: _StageSpec) -> _StrategyWindowSettings:
    seed_value = getattr(config.train, "strategy_window_seed", None)
    seed = int(config.train.seed + 100_003 if seed_value is None else seed_value)
    return _StrategyWindowSettings(
        weight=float(_stage_value(stage.raw, "strategy_window_loss_weight", config.train.strategy_window_loss_weight)),
        pool_split=str(_stage_value(stage.raw, "strategy_window_pool_split", config.train.strategy_window_pool_split)),
        pool_start=getattr(config.train, "strategy_window_pool_start", None),
        pool_end=getattr(config.train, "strategy_window_pool_end", None),
        window_days=max(int(_stage_value(stage.raw, "strategy_window_days", config.train.strategy_window_days)), 1),
        samples_per_epoch=max(int(_stage_value(stage.raw, "strategy_window_samples_per_epoch", config.train.strategy_window_samples_per_epoch)), 0),
        label=str(_stage_value(stage.raw, "strategy_window_label", config.train.strategy_window_label)),
        score_field=str(_stage_value(stage.raw, "strategy_window_score_field", config.train.strategy_window_score_field)),
        top_n=max(int(_stage_value(stage.raw, "strategy_window_top_n", config.train.strategy_window_top_n or config.strategy.top_n)), 1),
        mode=str(_stage_value(stage.raw, "strategy_window_mode", config.train.strategy_window_mode)),
        temperature=float(_stage_value(stage.raw, "strategy_window_temperature", config.train.strategy_window_temperature)),
        return_weight=float(_stage_value(stage.raw, "strategy_window_return_weight", config.train.strategy_window_return_weight)),
        excess_weight=float(_stage_value(stage.raw, "strategy_window_excess_weight", config.train.strategy_window_excess_weight)),
        downside_weight=float(_stage_value(stage.raw, "strategy_window_downside_weight", config.train.strategy_window_downside_weight)),
        seed=seed,
        month_cap=float(_stage_value(stage.raw, "strategy_window_month_cap", config.train.strategy_window_month_cap)),
    )


def _needs_strategy_window_dataset(config: MSGCAConfig, stages: list[_StageSpec]) -> bool:
    return any(
        (settings := _strategy_window_settings_for_stage(config, stage)).weight > 0.0
        and settings.samples_per_epoch > 0
        for stage in stages
    )


def _build_strategy_window_dataset(
    config: MSGCAConfig,
    feature_scaler: ScalerState,
    *,
    limit: int | None,
):
    pool_start = getattr(config.train, "strategy_window_pool_start", None)
    pool_end = getattr(config.train, "strategy_window_pool_end", None)
    if not pool_start or not pool_end:
        raise ValueError("strategy_window_pool_start and strategy_window_pool_end are required when strategy window loss is enabled")
    _validate_strategy_window_pool(config, str(pool_start), str(pool_end))
    pool_config = deepcopy(config)
    pool_config.data.validation_start = str(pool_start)
    pool_config.data.validation_end = str(pool_end)
    _log(f"strategy_window_dataset_start pool={pool_start}:{pool_end}")
    return build_datasets(
        pool_config,
        split="validation",
        limit=limit,
        feature_scaler=feature_scaler,
    )


def _validate_strategy_window_pool(config: MSGCAConfig, pool_start: str, pool_end: str) -> None:
    pool = (pd.Timestamp(pool_start), pd.Timestamp(pool_end))
    protected = {
        "train": (config.data.train_start, config.data.train_end),
        "validation": (config.data.validation_start, config.data.validation_end),
        "holdout": (config.data.holdout_start, config.data.holdout_end),
    }
    for name, raw_range in protected.items():
        if _date_ranges_overlap(pool, _coerce_date_range(raw_range)):
            raise ValueError(
                "strategy_window_pool must not overlap "
                f"{name}: pool={pool_start}:{pool_end} {name}={raw_range[0]}:{raw_range[1]}"
            )


def _coerce_date_range(raw_range: tuple[str | None, str | None]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start, end = raw_range
    start_ts = None if start is None else pd.Timestamp(start)
    end_ts = None if end is None else pd.Timestamp(end)
    return start_ts, end_ts


def _date_ranges_overlap(
    left: tuple[pd.Timestamp | None, pd.Timestamp | None],
    right: tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> bool:
    left_start, left_end = left
    right_start, right_end = right
    if left_start is None or left_end is None or right_start is None:
        return False
    effective_right_end = pd.Timestamp.max.normalize() if right_end is None else right_end
    return left_start <= effective_right_end and right_start <= left_end


def _validate_strategy_layout(strategy_layout, train_layout) -> None:
    for key in ("price_columns", "text_columns", "fundamental_columns", "text_group_ids", "fundamental_group_ids"):
        current = list(getattr(train_layout, key, []))
        candidate = list(getattr(strategy_layout, key, []))
        if candidate != current:
            raise ValueError(f"Strategy window layout mismatch for {key}: {len(candidate)} pool vs {len(current)} train")


def _strategy_day_indices(samples: pd.DataFrame) -> list[tuple[pd.Timestamp, list[int]]]:
    dates = pd.to_datetime(samples["target_trade_date"], errors="coerce").dt.normalize()
    valid = pd.notna(dates).to_numpy()
    frame = pd.DataFrame(
        {
            "date": dates[valid].to_numpy(),
            "position": np.arange(len(samples), dtype=np.int64)[valid],
        }
    )
    grouped = frame.groupby("date", sort=True).indices
    return [(pd.Timestamp(day), [int(index) for index in indices]) for day, indices in grouped.items()]


def _run_strategy_window_training(
    model: torch.nn.Module,
    dataset,
    day_indices: list[tuple[pd.Timestamp, list[int]]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: MSGCAConfig,
    stage: _StageSpec,
    device: torch.device,
    *,
    epoch: int,
    stage_epoch: int,
) -> dict[str, float]:
    settings = _strategy_window_settings_for_stage(config, stage)
    if settings.weight <= 0.0 or settings.samples_per_epoch <= 0:
        return {}
    if dataset is None:
        raise ValueError("Strategy window loss is enabled but strategy dataset was not built")
    windows = _sample_strategy_windows(day_indices, settings, epoch=epoch, stage_epoch=stage_epoch)
    if not windows:
        _log(
            "strategy_window_skipped "
            f"stage={stage.name} reason=insufficient_days available={len(day_indices)} window_days={settings.window_days}"
        )
        return {"strategy_window_sampled_windows": 0.0, "strategy_window_sampled_days": 0.0}

    model.train()
    raw_losses: list[float] = []
    weighted_losses: list[float] = []
    part_values: dict[str, list[float]] = {}
    processed_days = 0
    skipped_no_grad = 0
    for window in windows:
        for _, indices in window:
            if not indices:
                continue
            batch = collate_msgca_batch([dataset[int(index)] for index in indices])
            tensors = _batch_tensors_to_device(batch, config, device)
            if settings.label not in tensors:
                raise KeyError(f"Unsupported strategy_window_label: {settings.label}")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config.train.amp and device.type == "cuda"):
                output = model(
                    tensors["price_window"],
                    tensors["price_mask"],
                    tensors["text_features"],
                    tensors["text_mask"],
                    tensors["fundamental_features"],
                    tensors["fundamental_mask"],
                )
                raw_loss, raw_parts = strategy_window_return_loss_from_output(
                    output,
                    tensors[settings.label],
                    batch["target_trade_date"],
                    score_field=settings.score_field,
                    topk_k=settings.top_n,
                    mode=settings.mode,
                    temperature=settings.temperature,
                    return_weight=settings.return_weight,
                    excess_weight=settings.excess_weight,
                    downside_weight=settings.downside_weight,
                    sample_weights=_batch_time_weights(batch["target_trade_date"], config, device),
                )
                loss = raw_loss * settings.weight
            raw_losses.append(float(raw_loss.detach().cpu()))
            weighted_losses.append(float(loss.detach().cpu()))
            for key, value in raw_parts.items():
                part_values.setdefault(key, []).append(float(value.detach().cpu()))
            if not loss.requires_grad:
                skipped_no_grad += 1
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            processed_days += 1

    result = {key: float(np.mean(values)) for key, values in part_values.items() if values}
    result["strategy_window_loss"] = float(np.mean(raw_losses)) if raw_losses else 0.0
    result["strategy_window_weighted_loss"] = float(np.mean(weighted_losses)) if weighted_losses else 0.0
    result["strategy_window_sampled_windows"] = float(len(windows))
    result["strategy_window_sampled_days"] = float(processed_days)
    result["strategy_window_skipped_no_grad_days"] = float(skipped_no_grad)
    return result


def _sample_strategy_windows(
    day_indices: list[tuple[pd.Timestamp, list[int]]],
    settings: _StrategyWindowSettings,
    *,
    epoch: int,
    stage_epoch: int,
) -> list[list[tuple[pd.Timestamp, list[int]]]]:
    if len(day_indices) < settings.window_days or settings.samples_per_epoch <= 0:
        return []
    starts = list(range(0, len(day_indices) - settings.window_days + 1))
    rng = random.Random(int(settings.seed) + int(epoch) * 104_729 + int(stage_epoch) * 9_176)
    selected_starts = _sample_window_starts_with_month_cap(starts, day_indices, settings, rng)
    return [day_indices[start : start + settings.window_days] for start in selected_starts]


def _sample_window_starts_with_month_cap(
    starts: list[int],
    day_indices: list[tuple[pd.Timestamp, list[int]]],
    settings: _StrategyWindowSettings,
    rng: random.Random,
) -> list[int]:
    count = int(settings.samples_per_epoch)
    if count <= 0:
        return []
    cap = float(settings.month_cap or 0.0)
    if cap <= 0.0:
        if count <= len(starts):
            return rng.sample(starts, count)
        return [rng.choice(starts) for _ in range(count)]

    per_month_cap = max(int(np.ceil(count * min(max(cap, 0.0), 1.0))), 1)
    by_month: dict[str, list[int]] = {}
    for start in starts:
        month = pd.Timestamp(day_indices[start][0]).strftime("%Y-%m")
        by_month.setdefault(month, []).append(start)
    months = list(by_month)
    rng.shuffle(months)
    selected: list[int] = []
    for month in months:
        candidates = list(by_month[month])
        rng.shuffle(candidates)
        take = min(per_month_cap, len(candidates), count - len(selected))
        selected.extend(candidates[:take])
        if len(selected) >= count:
            break
    if len(selected) < count:
        remaining = [start for start in starts if start not in set(selected)]
        if remaining:
            if count - len(selected) <= len(remaining):
                selected.extend(rng.sample(remaining, count - len(selected)))
            else:
                selected.extend(rng.choice(remaining) for _ in range(count - len(selected)))
        else:
            selected.extend(rng.choice(starts) for _ in range(count - len(selected)))
    rng.shuffle(selected)
    return selected[:count]


def _aux_topk_losses(raw: object) -> tuple[AuxTopKLoss, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise TypeError("aux_topk_return_losses must be a list")
    losses: list[AuxTopKLoss] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("aux_topk_return_losses entries must be mappings")
        losses.append(
            AuxTopKLoss(
                weight=float(item.get("weight", 0.0)),
                k=int(item.get("k", 20)),
                score_field=str(item.get("score_field", "y_score")),
                secondary_weight=None if item.get("secondary_weight") is None else float(item["secondary_weight"]),
                mode=None if item.get("mode") is None else str(item["mode"]),
                temperature=None if item.get("temperature") is None else float(item["temperature"]),
            )
        )
    return tuple(losses)


def _aux_rank_losses(raw: object) -> tuple[AuxRankLoss, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise TypeError("aux_rank_losses must be a list")
    losses: list[AuxRankLoss] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("aux_rank_losses entries must be mappings")
        losses.append(
            AuxRankLoss(
                weight=float(item.get("weight", 0.0)),
                score_field=str(item.get("score_field", "y_score")),
                label_mode=None if item.get("label_mode") is None else str(item["label_mode"]),
                topk_k=None if item.get("topk_k") is None else int(item["topk_k"]),
            )
        )
    return tuple(losses)


def _stage_strategy(config: MSGCAConfig, stage: _StageSpec):
    variant = stage.validation_score_variant or _strategy_variant_for_score_col(stage.validation_score_col)
    return replace(config.strategy, score_variant=variant)


def _strategy_variant_for_score_col(score_col: str) -> str:
    if score_col in {"y_score", "return_pred", "direction_prob", "final_score"}:
        return score_col
    if score_col == "direction_logit":
        return "direction_prob"
    return "y_score"


def _set_trainable_parameters(model: torch.nn.Module, policy: str) -> tuple[int, int]:
    normalized = str(policy or "all")
    head_prefixes = {
        "score": "score_head.",
        "return": "return_head.",
        "direction": "direction_head.",
        "final": "final_score_head.",
    }
    total_count = 0
    trainable_count = 0
    for name, parameter in model.named_parameters():
        total_count += int(parameter.numel())
        trainable = _parameter_trainable(name, normalized, head_prefixes)
        parameter.requires_grad = trainable
        if trainable:
            trainable_count += int(parameter.numel())
    if trainable_count <= 0:
        raise ValueError(f"No trainable parameters for trainable policy: {policy}")
    return trainable_count, total_count


def _parameter_trainable(name: str, policy: str, head_prefixes: dict[str, str]) -> bool:
    if policy == "all":
        return True
    if policy == "heads_only":
        return any(name.startswith(prefix) for prefix in head_prefixes.values())
    if policy.endswith("_head_only"):
        key = policy.removesuffix("_head_only")
        if key == "final_score":
            key = "final"
        prefix = head_prefixes.get(key)
        if prefix is None:
            raise ValueError(f"Unsupported trainable policy: {policy}")
        return name.startswith(prefix)
    if policy.startswith("encoder_and_"):
        key = policy.removeprefix("encoder_and_")
        if key == "final_score":
            key = "final"
        keep_prefix = head_prefixes.get(key)
        if keep_prefix is None:
            raise ValueError(f"Unsupported trainable policy: {policy}")
        disabled = [prefix for prefix in head_prefixes.values() if prefix != keep_prefix]
        return not any(name.startswith(prefix) for prefix in disabled)
    raise ValueError(f"Unsupported trainable policy: {policy}")


def _trainable_parameters(model: torch.nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _mean_loss_parts(parts: dict[str, list[float]]) -> dict[str, float]:
    return {f"train_{key}": float(np.mean(values)) for key, values in parts.items() if key != "loss" and values}


def _write_best_checkpoint_json(
    path: Path,
    checkpoint: Path,
    epoch: int,
    stage_epoch: int,
    metric: str,
    mode: str,
    value: float,
    stage: _StageSpec,
) -> None:
    path.write_text(
        pd.Series(
            {
                "checkpoint": str(checkpoint),
                "epoch": int(epoch),
                "stage": stage.name,
                "stage_epoch": int(stage_epoch),
                "metric": metric,
                "mode": mode,
                "value": float(value),
            }
        ).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )


def _competition_log_columns(summary: dict[str, object]) -> dict[str, object]:
    if not summary:
        return {}
    keys = {
        "period_return",
        "period_excess_equal",
        "market_equal_period_return",
        "sharpe",
        "max_drawdown",
        "turnover_mean",
        "rolling_return_mean",
        "rolling_excess_equal_mean",
        "rolling_win_rate",
        "rolling_return_min",
        "latest_window_return",
        "recent_return_mean",
        "recent_return_min",
        "competition_score",
    }
    out: dict[str, object] = {}
    for key in keys:
        if key in summary:
            out[f"competition_{key}" if key != "competition_score" else key] = summary[key]
    return out


def _metric_improved(value: object, best: float | None, *, mode: str) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not pd.notna(numeric):
        return False
    if best is None:
        return True
    if mode == "min":
        return numeric < best
    return numeric > best


def _load_existing_best_metric(root: Path, metric_name: str, mode: str) -> float | None:
    return _load_best_metric_json(root / "checkpoints" / "msgca_best.json", metric_name, mode)


def _load_existing_train_log(path: Path, max_epoch: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if "epoch" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["epoch"], errors="coerce").fillna(0).astype(int) <= int(max_epoch)]
    return frame.to_dict(orient="records")


def _load_best_metric_json(path: Path, metric_name: str, mode: str) -> float | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("metric") != metric_name or payload.get("mode") != mode:
        return None
    value = payload.get("value")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if pd.notna(numeric) else None


def _validate_resume_layout(checkpoint: dict[str, object], layout) -> None:
    saved = checkpoint.get("layout")
    if not isinstance(saved, dict):
        return
    for key in ("price_columns", "text_columns", "fundamental_columns", "text_group_ids", "fundamental_group_ids"):
        current = list(getattr(layout, key, []))
        previous = list(saved.get(key, []))
        if current != previous:
            raise ValueError(f"Resume checkpoint layout mismatch for {key}: {len(previous)} saved vs {len(current)} current")


def _log(message: str) -> None:
    print(f"[{pd.Timestamp.now(tz='UTC').isoformat()}] {message}", flush=True)


def _batch_time_weights(target_trade_dates, config, device: torch.device) -> torch.Tensor | None:
    bins = getattr(config.train, "time_weight_bins", None) or []
    if not bins:
        return None
    dates = pd.to_datetime(target_trade_dates, errors="coerce").normalize()
    weights = pd.Series(1.0, index=range(len(dates)), dtype="float64")
    for rule in bins:
        if not isinstance(rule, dict):
            raise TypeError("train.time_weight_bins entries must be mappings")
        value = float(rule.get("weight", 1.0))
        mask = pd.Series(True, index=weights.index)
        start = rule.get("start")
        end = rule.get("end")
        if start:
            mask &= dates >= pd.Timestamp(start)
        if end:
            mask &= dates <= pd.Timestamp(end)
        weights.loc[mask] = value
    values = weights.to_numpy(dtype="float32")
    if getattr(config.train, "normalize_time_weights", True):
        mean = float(values.mean()) if len(values) else 1.0
        if mean > 0:
            values = values / mean
    return torch.as_tensor(values, dtype=torch.float32, device=device)
