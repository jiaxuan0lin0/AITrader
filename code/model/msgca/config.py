from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from aitrader_paths import DATASETS_ROOT, EXPERIMENTS_ROOT


DEFAULT_PROCESSED_DIR = DATASETS_ROOT / "processed"
DEFAULT_EVALUATION_DIR = DATASETS_ROOT / "factors" / "evaluation" / "final"
DEFAULT_MODEL_ROOT = EXPERIMENTS_ROOT / "msgca" / "ad_hoc"


@dataclass
class PathConfig:
    processed_dir: str = str(DEFAULT_PROCESSED_DIR)
    feature_registry_path: str = str(DATASETS_ROOT / "features" / "feature_registry.json")
    evaluation_dir: str = str(DEFAULT_EVALUATION_DIR)
    model_root: str = str(DEFAULT_MODEL_ROOT)

    @property
    def samples_path(self) -> Path:
        return Path(self.processed_dir) / "samples.parquet"

    @property
    def price_path(self) -> Path:
        return Path(self.processed_dir) / "price.parquet"

    @property
    def moneyflow_path(self) -> Path:
        return Path(self.processed_dir) / "moneyflow.parquet"

    @property
    def metric_path(self) -> Path:
        return Path(self.processed_dir) / "metric.parquet"

    @property
    def news_path(self) -> Path:
        return Path(self.processed_dir) / "news.parquet"

    @property
    def news_scores_path(self) -> Path:
        return DATASETS_ROOT / "factors" / "news_llm_scores.parquet"

    @property
    def output_root(self) -> Path:
        return Path(self.model_root)


@dataclass
class DataConfig:
    lookback: int = 60
    primary_label: str = "label_next_open_return"
    secondary_label: str = "label_next_vwap_return"
    strict_lookback: bool = True
    fast_loader: bool = True
    use_polars: bool = True
    price_window_cache: str = "memory"
    price_window_cache_dir: str | None = None
    train_start: str = "2019-01-01"
    train_end: str = "2025-09-30"
    validation_start: str = "2025-10-01"
    validation_end: str = "2025-12-31"
    holdout_start: str = "2026-01-01"
    holdout_end: str | None = None
    price_columns: list[str] = field(
        default_factory=lambda: [
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "amount",
            "mf_net_amount_ratio",
            "mf_main_net_amount_ratio",
            "mf_small_order_pressure",
        ]
    )
    text_prefixes: list[str] = field(default_factory=lambda: ["news_", "market_news"])
    fundamental_prefixes: list[str] = field(default_factory=lambda: ["metric_", "mf_", "alpha158_"])


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    price_layers: int = 2
    fusion_layers: int = 1
    factor_encoder: str = "simple"
    factor_layers: int = 2
    factor_group_layers: int = 1
    factor_group_prototypes: int = 1
    factor_group_source: str = "selected_blocks"
    factor_gate_activation: str = "softmax"
    max_price_variables: int = 256
    max_text_features: int = 512
    max_fundamental_features: int = 2048
    enable_price: bool = True
    enable_news: bool = True
    enable_fundamental: bool = True
    use_gate: bool = True
    use_cross_attention: bool = True
    use_factor_gate: bool = True


@dataclass
class TrainConfig:
    seed: int = 2026
    epochs: int = 5
    batch_days: int = 1
    final_validate: bool = False
    validate_each_epoch: bool = False
    validation_interval: int = 0
    max_pairs_per_day: int = 4096
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    dataloader_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    competition_validate: bool = True
    competition_window_days: int = 10
    competition_recent_window_count: int = 5
    best_checkpoint_metric: str = "competition_score"
    best_checkpoint_mode: str = "max"
    save_epoch_checkpoints: bool = True
    early_stop_min_epochs: int = 0
    early_stop_patience: int = 0
    restore_best_checkpoint: bool = False
    validation_score_col: str = "y_score"
    validation_score_variant: str | None = None
    rank_loss_weight: float = 1.0
    return_loss_weight: float = 0.1
    secondary_return_loss_weight: float = 0.0
    direction_loss_weight: float = 0.05
    return_loss_type: str = "mse"
    return_huber_delta: float = 0.02
    rank_label_mode: str = "raw"
    topk_label_mode: str = "raw"
    rank_topk_k: int = 0
    direction_label_mode: str = "raw"
    direction_label_epsilon: float = 0.0
    direction_label_std_fraction: float = 0.0
    topk_return_loss_weight: float = 0.0
    return_secondary_weight: float = 0.0
    topk_secondary_weight: float = 0.0
    topk_return_k: int = 20
    topk_return_mode: str = "soft_equal_topk"
    topk_temperature: float = 0.1
    gate_entropy_weight: float = 0.001
    rank_score_field: str = "y_score"
    topk_score_field: str = "y_score"
    aux_rank_losses: list[dict[str, Any]] = field(default_factory=list)
    aux_topk_return_losses: list[dict[str, Any]] = field(default_factory=list)
    combined_topk_loss_weight: float = 0.0
    combined_topk_k: int = 20
    combined_topk_mode: str = "soft_equal_topk"
    combined_topk_temperature: float = 0.1
    combined_weight_final: float = 0.60
    combined_weight_return: float = 0.22
    combined_weight_direction: float = 0.08
    combined_weight_y: float = 0.10
    consistency_loss_weight: float = 0.0
    consistency_score_field: str = "final_score"
    consistency_topk_k: int = 40
    consistency_temperature: float = 0.1
    consistency_return_weight: float = 0.7
    consistency_direction_weight: float = 0.3
    loss_context_columns: list[str] = field(default_factory=list)
    context_cache_path: str | None = None
    context_news_cache_path: str | None = None
    context_score_topk_loss_weight: float = 0.0
    context_score_variant: str = "context_s5"
    context_score_topk_k: int = 20
    context_score_topk_mode: str = "soft_equal_topk"
    context_score_topk_temperature: float = 0.1
    trend_adjusted_topk_loss_weight: float = 0.0
    trend_adjusted_topk_k: int = 20
    trend_adjusted_topk_mode: str = "soft_equal_topk"
    trend_adjusted_topk_temperature: float = 0.1
    trend_adjusted_score_field: str = "final_score"
    trend_adjusted_positive_weight: float = 0.25
    trend_overheat_negative_weight: float = 0.35
    trend_broken_negative_weight: float = 0.45
    direction_healthy_pullback_discount: float = 0.0
    cluster_topk_loss_weight: float = 0.0
    cluster_column: str = "context_cluster_id"
    cluster_topk_score_field: str = "final_score"
    cluster_topk_label_mode: str = "excess"
    cluster_topk_k: int = 6
    cluster_topk_mode: str = "soft_equal_topk"
    cluster_topk_temperature: float = 0.1
    cluster_topk_min_size: int = 8
    cluster_topk_member_k: int = 5
    cluster_topk_member_temperature: float = 0.1
    cluster_rank_loss_weight: float = 0.0
    cluster_rank_score_field: str = "final_score"
    cluster_rank_label_mode: str = "excess"
    cluster_rank_min_size: int = 8
    cluster_rank_member_k: int = 5
    cluster_rank_member_temperature: float = 0.1
    cluster_rank_max_pairs_per_day: int = 512
    in_cluster_rank_loss_weight: float = 0.0
    in_cluster_rank_score_field: str = "final_score"
    in_cluster_rank_label_mode: str = "excess"
    in_cluster_rank_topk_k: int = 5
    in_cluster_rank_min_size: int = 8
    in_cluster_rank_max_clusters_per_day: int = 32
    in_cluster_rank_max_pairs_per_cluster: int = 512
    strategy_window_loss_weight: float = 0.0
    strategy_window_pool_split: str = "custom"
    strategy_window_pool_start: str | None = None
    strategy_window_pool_end: str | None = None
    strategy_window_days: int = 10
    strategy_window_samples_per_epoch: int = 0
    strategy_window_label: str = "label_next_open_return"
    strategy_window_score_field: str = "final_score"
    strategy_window_top_n: int = 20
    strategy_window_mode: str = "soft_equal_topk"
    strategy_window_temperature: float = 0.1
    strategy_window_return_weight: float = 0.0
    strategy_window_excess_weight: float = 1.0
    strategy_window_downside_weight: float = 0.1
    strategy_window_seed: int | None = None
    strategy_window_month_cap: float = 0.0
    stages: list[dict[str, Any]] = field(default_factory=list)
    time_weight_bins: list[dict[str, Any]] = field(default_factory=list)
    normalize_time_weights: bool = False


@dataclass
class StrategyConfig:
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


@dataclass
class MSGCAConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def ensure_output_dirs(self) -> None:
        root = self.paths.output_root
        for child in (
            root,
            root / "checkpoints",
            root / "competition_signals",
            root / "report_assets",
        ):
            child.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> MSGCAConfig:
    """Load config from YAML and merge it into dataclass defaults."""
    data: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a mapping: {config_path}")
            data = loaded
        else:
            raise FileNotFoundError(f"Missing config file: {config_path}")
    if overrides:
        data = _deep_merge(data, dict(overrides))
    return _from_mapping(MSGCAConfig(), data)


def write_resolved_config(config: MSGCAConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _from_mapping(instance: Any, values: Mapping[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise TypeError(f"Config section must be a mapping: {key}")
            setattr(instance, key, _from_mapping(current, value))
        else:
            setattr(instance, key, value)
    return instance


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
