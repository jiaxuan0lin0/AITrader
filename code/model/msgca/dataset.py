from __future__ import annotations

from dataclasses import dataclass, field
import gc
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from model.msgca.config import MSGCAConfig
from model.msgca.context_features import CONTEXT_COLUMNS, attach_context_features
from model.msgca.feature_set import (
    SampleFeatureMatrix,
    load_sample_feature_panel,
    load_sample_feature_matrix,
    load_sample_feature_matrix_by_row_positions,
    load_selected_features,
    split_features_by_modality,
)


PRICE_BASE_COLUMNS = ("stock_code", "trade_date")
SAMPLE_BASE_COLUMNS = (
    "sample_id",
    "stock_code",
    "stock_name",
    "industry",
    "feature_asof_date",
    "target_trade_date",
    "decision_ts",
    "label_next_open_return",
    "label_next_vwap_return",
)


@dataclass(frozen=True)
class FeatureLayout:
    price_columns: list[str]
    text_columns: list[str]
    fundamental_columns: list[str]
    selected_feature_mode: str
    selected_feature_path: str | None
    text_group_ids: list[int] = field(default_factory=list)
    text_group_names: list[str] = field(default_factory=list)
    fundamental_group_ids: list[int] = field(default_factory=list)
    fundamental_group_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScalerState:
    columns: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "medians": {str(key): float(value) for key, value in self.medians.items()},
            "means": {str(key): float(value) for key, value in self.means.items()},
            "stds": {str(key): float(value) for key, value in self.stds.items()},
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ScalerState":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid scaler payload: {type(payload)!r}")
        columns = [str(column) for column in payload.get("columns", [])]
        medians = _float_mapping(payload.get("medians", {}))
        means = _float_mapping(payload.get("means", {}))
        stds = _float_mapping(payload.get("stds", {}))
        return cls(columns=columns, medians=medians, means=means, stds=stds)

    def transform_frame(self, frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        columns = list(columns)
        if not columns:
            return frame.copy()
        numeric = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
        medians = pd.Series({column: self.medians.get(column, 0.0) for column in columns}, dtype="float64")
        means = pd.Series({column: self.means.get(column, 0.0) for column in columns}, dtype="float64")
        stds = pd.Series({column: self.stds.get(column, 1.0) or 1.0 for column in columns}, dtype="float64")
        transformed = numeric.fillna(medians).sub(means, axis=1).div(stds, axis=1)
        base = frame.drop(columns=[column for column in columns if column in frame.columns])
        return pd.concat([base, transformed], axis=1)


@dataclass
class FeatureArrayData:
    text_values: np.ndarray
    text_present: np.ndarray
    fundamental_values: np.ndarray
    fundamental_present: np.ndarray


@dataclass(frozen=True)
class _PriceSeries:
    dates: np.ndarray
    values: np.ndarray
    mask: np.ndarray


class FastPriceStore:
    """Stock-indexed NumPy store for as-of price windows."""

    def __init__(self, by_stock: dict[str, _PriceSeries], price_columns: Sequence[str]) -> None:
        self.by_stock = by_stock
        self.price_columns = list(price_columns)
        self.variable_count = len(self.price_columns)

    @classmethod
    def from_frame(cls, price: pd.DataFrame, price_columns: Sequence[str]) -> "FastPriceStore":
        by_stock: dict[str, _PriceSeries] = {}
        for stock_code, frame in price.groupby("stock_code", sort=False):
            ordered = frame.sort_values("trade_date")
            dates = _datetime_series_to_day_int(ordered["trade_date"])
            raw = ordered[list(price_columns)].to_numpy(dtype="float32", copy=True)
            valid = np.isfinite(raw)
            values = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
            by_stock[str(stock_code)] = _PriceSeries(dates=dates, values=values, mask=valid)
        return cls(by_stock, price_columns)

    def has_full_window(self, stock_code: str, feature_asof_day: int, lookback: int) -> bool:
        series = self.by_stock.get(stock_code)
        if series is None:
            return False
        return int(np.searchsorted(series.dates, feature_asof_day, side="right")) >= lookback

    def get_window(self, stock_code: str, feature_asof_day: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((self.variable_count, lookback), dtype="float32")
        mask = np.zeros((self.variable_count, lookback), dtype=bool)
        series = self.by_stock.get(stock_code)
        if series is None:
            return values, mask

        end = int(np.searchsorted(series.dates, feature_asof_day, side="right"))
        if end <= 0:
            return values, mask
        start = max(0, end - lookback)
        offset = lookback - (end - start)
        values[:, offset:] = series.values[start:end].T
        mask[:, offset:] = series.mask[start:end].T
        return values, mask


class MSGCASampleDataset(Dataset):
    """Sample-level dataset that never joins daily features on target_trade_date."""

    def __init__(
        self,
        samples: pd.DataFrame,
        price: pd.DataFrame,
        layout: FeatureLayout,
        lookback: int,
        scaler: ScalerState | None = None,
        strict_lookback: bool = True,
        fast_loader: bool = True,
        feature_arrays: FeatureArrayData | None = None,
        price_window_cache: str = "none",
        price_window_cache_dir: str | Path | None = None,
        price_window_cache_name: str = "dataset",
        context_columns: Sequence[str] | None = None,
    ) -> None:
        _require_columns(samples, ("sample_id", "stock_code", "feature_asof_date", "target_trade_date"), "samples")
        _require_columns(price, PRICE_BASE_COLUMNS, "price")
        if lookback <= 0:
            raise ValueError("lookback must be positive")

        self.samples = _prepare_samples(samples).reset_index(drop=True)
        self.layout = layout
        self.lookback = int(lookback)
        self.strict_lookback = strict_lookback
        self.fast_loader = fast_loader
        self.feature_arrays = feature_arrays
        self.price_window_cache = str(price_window_cache or "none")
        self.price_window_cache_dir = None if price_window_cache_dir is None else Path(price_window_cache_dir)
        self.price_window_cache_name = str(price_window_cache_name or "dataset")
        self.context_columns = list(context_columns or [])
        self._cached_price_values: np.ndarray | None = None
        self._cached_price_mask: np.ndarray | None = None
        if feature_arrays is not None and not fast_loader:
            raise ValueError("feature_arrays require fast_loader=True")
        if feature_arrays is not None and len(feature_arrays.text_values) != len(self.samples):
            raise ValueError("feature_arrays length must match samples")
        self.price = _prepare_price(price, layout.price_columns)
        self.scaler = scaler
        self.price_store = FastPriceStore.from_frame(self.price, layout.price_columns) if fast_loader else None
        self._price_by_stock = (
            {}
            if fast_loader
            else {
                stock_code: frame.sort_values("trade_date").reset_index(drop=True)
                for stock_code, frame in self.price.groupby("stock_code", sort=False)
            }
        )

        if scaler is not None and feature_arrays is None:
            all_numeric = [*layout.text_columns, *layout.fundamental_columns]
            self.samples = scaler.transform_frame(self.samples, all_numeric)

        if strict_lookback:
            keep_array = self._full_price_window_mask()
            self.samples = self.samples.loc[keep_array].reset_index(drop=True)
            if self.feature_arrays is not None:
                self.feature_arrays = FeatureArrayData(
                    text_values=self.feature_arrays.text_values[keep_array],
                    text_present=self.feature_arrays.text_present[keep_array],
                    fundamental_values=self.feature_arrays.fundamental_values[keep_array],
                    fundamental_present=self.feature_arrays.fundamental_present[keep_array],
                )

        if self.fast_loader:
            self._prepare_fast_arrays()
            self._prepare_price_window_cache()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        if self.fast_loader:
            return self._getitem_fast(index)

        row = self.samples.iloc[index]
        price_window, price_mask = self._price_window(str(row["stock_code"]), row["feature_asof_date"])
        text_values, text_mask = _row_values(row, self.layout.text_columns)
        fundamental_values, fundamental_mask = _row_values(row, self.layout.fundamental_columns)
        label = _float_or_nan(row.get("label_next_open_return", np.nan))
        secondary_label = _float_or_nan(row.get("label_next_vwap_return", np.nan))
        direction = 1.0 if np.isfinite(label) and label > 0 else (0.0 if np.isfinite(label) else np.nan)
        item = {
            "sample_id": str(row["sample_id"]),
            "stock_code": str(row["stock_code"]),
            "stock_name": "" if pd.isna(row.get("stock_name", "")) else str(row.get("stock_name", "")),
            "industry": "" if pd.isna(row.get("industry", "")) else str(row.get("industry", "")),
            "target_trade_date": pd.Timestamp(row["target_trade_date"]),
            "feature_asof_date": pd.Timestamp(row["feature_asof_date"]),
            "decision_ts": pd.Timestamp(row["decision_ts"]) if "decision_ts" in row else pd.NaT,
            "price_window": torch.as_tensor(price_window, dtype=torch.float32),
            "price_mask": torch.as_tensor(price_mask, dtype=torch.bool),
            "text_features": torch.as_tensor(text_values, dtype=torch.float32),
            "text_mask": torch.as_tensor(text_mask, dtype=torch.bool),
            "fundamental_features": torch.as_tensor(fundamental_values, dtype=torch.float32),
            "fundamental_mask": torch.as_tensor(fundamental_mask, dtype=torch.bool),
            "label_next_open_return": torch.tensor(label, dtype=torch.float32),
            "label_next_vwap_return": torch.tensor(secondary_label, dtype=torch.float32),
            "label_direction": torch.tensor(direction, dtype=torch.float32),
        }
        if self.context_columns:
            context_values, _ = _row_values(row, self.context_columns)
            item["loss_context"] = torch.as_tensor(context_values, dtype=torch.float32)
        return item

    def _prepare_fast_arrays(self) -> None:
        self._sample_ids = self.samples["sample_id"].astype(str).to_numpy(dtype=object)
        self._stock_codes = self.samples["stock_code"].astype(str).to_numpy(dtype=object)
        self._stock_names = _string_array(self.samples, "stock_name")
        self._industries = _string_array(self.samples, "industry")
        self._target_dates = pd.to_datetime(self.samples["target_trade_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        self._feature_dates = pd.to_datetime(self.samples["feature_asof_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        if "decision_ts" in self.samples.columns:
            self._decision_ts = pd.to_datetime(self.samples["decision_ts"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        else:
            self._decision_ts = np.full(len(self.samples), np.datetime64("NaT"), dtype="datetime64[ns]")
        self._feature_asof_days = _datetime_series_to_day_int(self.samples["feature_asof_date"])
        if self.feature_arrays is None:
            self._text_values, self._text_present = _frame_values(self.samples, self.layout.text_columns)
            self._fundamental_values, self._fundamental_present = _frame_values(self.samples, self.layout.fundamental_columns)
        else:
            self._text_values = self.feature_arrays.text_values
            self._text_present = self.feature_arrays.text_present
            self._fundamental_values = self.feature_arrays.fundamental_values
            self._fundamental_present = self.feature_arrays.fundamental_present
        self._label_open = _float_array(self.samples, "label_next_open_return")
        self._label_vwap = _float_array(self.samples, "label_next_vwap_return")
        if self.context_columns:
            self._loss_context_values, _ = _frame_values(self.samples, self.context_columns)
        else:
            self._loss_context_values = np.zeros((len(self.samples), 0), dtype="float32")

    def _getitem_fast(self, index: int) -> dict[str, object]:
        assert self.price_store is not None
        stock_code = str(self._stock_codes[index])
        if self._cached_price_values is not None and self._cached_price_mask is not None:
            price_window = self._cached_price_values[index]
            price_mask = self._cached_price_mask[index]
        else:
            price_window, price_mask = self.price_store.get_window(stock_code, int(self._feature_asof_days[index]), self.lookback)
        label = float(self._label_open[index])
        secondary_label = float(self._label_vwap[index])
        direction = 1.0 if np.isfinite(label) and label > 0 else (0.0 if np.isfinite(label) else np.nan)
        item = {
            "sample_id": str(self._sample_ids[index]),
            "stock_code": stock_code,
            "stock_name": str(self._stock_names[index]),
            "industry": str(self._industries[index]),
            "target_trade_date": pd.Timestamp(self._target_dates[index]),
            "feature_asof_date": pd.Timestamp(self._feature_dates[index]),
            "decision_ts": pd.Timestamp(self._decision_ts[index]),
            "price_window": torch.as_tensor(price_window, dtype=torch.float32),
            "price_mask": torch.as_tensor(price_mask, dtype=torch.bool),
            "text_features": torch.as_tensor(self._text_values[index], dtype=torch.float32),
            "text_mask": torch.as_tensor(self._text_present[index], dtype=torch.bool),
            "fundamental_features": torch.as_tensor(self._fundamental_values[index], dtype=torch.float32),
            "fundamental_mask": torch.as_tensor(self._fundamental_present[index], dtype=torch.bool),
            "label_next_open_return": torch.tensor(label, dtype=torch.float32),
            "label_next_vwap_return": torch.tensor(secondary_label, dtype=torch.float32),
            "label_direction": torch.tensor(direction, dtype=torch.float32),
        }
        if self.context_columns:
            item["loss_context"] = torch.as_tensor(self._loss_context_values[index], dtype=torch.float32)
        return item

    def _prepare_price_window_cache(self) -> None:
        mode = self.price_window_cache.lower()
        if mode in {"", "none", "false", "0"}:
            return
        if not self.fast_loader:
            raise ValueError("price_window_cache requires fast_loader=True")
        if self.price_store is None:
            raise ValueError("price_window_cache requires a FastPriceStore")
        if mode not in {"memory", "memmap"}:
            raise ValueError(f"Unsupported price_window_cache: {self.price_window_cache}")

        shape = (len(self.samples), self.price_store.variable_count, self.lookback)
        _log_data_stage(f"price_window_cache_start name={self.price_window_cache_name} mode={mode} shape={shape}")
        if mode == "memory":
            values = np.zeros(shape, dtype="float32")
            mask = np.zeros(shape, dtype=bool)
        else:
            if self.price_window_cache_dir is None:
                raise ValueError("price_window_cache_dir is required for memmap cache")
            self.price_window_cache_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _safe_cache_name(self.price_window_cache_name)
            values_path = self.price_window_cache_dir / f"{safe_name}_price_windows.float32.memmap"
            mask_path = self.price_window_cache_dir / f"{safe_name}_price_masks.bool.memmap"
            values = np.memmap(values_path, dtype="float32", mode="w+", shape=shape)
            mask = np.memmap(mask_path, dtype=bool, mode="w+", shape=shape)
            values[:] = 0.0
            mask[:] = False

        sample_days = self._feature_asof_days
        positions = np.arange(len(self.samples), dtype=np.int64)
        by_stock = pd.DataFrame({"stock_code": self._stock_codes, "position": positions})
        filled = 0
        for stock_code, group in by_stock.groupby("stock_code", sort=False):
            series = self.price_store.by_stock.get(str(stock_code))
            if series is None:
                continue
            group_positions = group["position"].to_numpy(dtype=np.int64, copy=False)
            ends = np.searchsorted(series.dates, sample_days[group_positions], side="right")
            for position, end in zip(group_positions, ends, strict=False):
                end = int(end)
                if end <= 0:
                    continue
                start = max(0, end - self.lookback)
                offset = self.lookback - (end - start)
                values[position, :, offset:] = series.values[start:end].T
                mask[position, :, offset:] = series.mask[start:end].T
            filled += len(group_positions)
            if filled and filled % 500_000 < len(group_positions):
                _log_data_stage(f"price_window_cache_progress name={self.price_window_cache_name} rows={filled}")
        if isinstance(values, np.memmap):
            values.flush()
        if isinstance(mask, np.memmap):
            mask.flush()
        self._cached_price_values = values
        self._cached_price_mask = mask
        _log_data_stage(f"price_window_cache_done name={self.price_window_cache_name} rows={len(self.samples)}")

    def _has_full_price_window(self, row: pd.Series) -> bool:
        if self.price_store is not None:
            return self.price_store.has_full_window(str(row["stock_code"]), _datetime_to_day_int(row["feature_asof_date"]), self.lookback)
        stock_frame = self._price_by_stock.get(str(row["stock_code"]))
        if stock_frame is None:
            return False
        eligible = stock_frame.loc[stock_frame["trade_date"].le(row["feature_asof_date"])]
        return len(eligible) >= self.lookback

    def _full_price_window_mask(self) -> np.ndarray:
        if self.samples.empty:
            return np.zeros(0, dtype=bool)
        if self.price_store is None:
            return np.asarray([self._has_full_price_window(row) for _, row in self.samples.iterrows()], dtype=bool)

        keep = np.zeros(len(self.samples), dtype=bool)
        sample_days = _datetime_series_to_day_int(self.samples["feature_asof_date"])
        stock_codes = self.samples["stock_code"].astype(str).to_numpy(dtype=object)
        positions = np.arange(len(self.samples), dtype=np.int64)
        by_stock = pd.DataFrame({"stock_code": stock_codes, "position": positions})
        for stock_code, group in by_stock.groupby("stock_code", sort=False):
            series = self.price_store.by_stock.get(str(stock_code))
            if series is None:
                continue
            group_positions = group["position"].to_numpy(dtype=np.int64, copy=False)
            available_counts = np.searchsorted(series.dates, sample_days[group_positions], side="right")
            keep[group_positions] = available_counts >= self.lookback
        return keep

    def _price_window(self, stock_code: str, feature_asof_date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        stock_frame = self._price_by_stock.get(stock_code)
        n_vars = len(self.layout.price_columns)
        values = np.zeros((n_vars, self.lookback), dtype="float32")
        mask = np.zeros((n_vars, self.lookback), dtype=bool)
        if stock_frame is None:
            return values, mask
        eligible = stock_frame.loc[stock_frame["trade_date"].le(pd.Timestamp(feature_asof_date))].tail(self.lookback)
        if eligible.empty:
            return values, mask
        offset = self.lookback - len(eligible)
        raw = eligible[self.layout.price_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32").T
        valid = np.isfinite(raw)
        values[:, offset:] = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        mask[:, offset:] = valid
        return values, mask


class DayBatchSampler(Sampler[list[int]]):
    """Yield batches grouped by target_trade_date."""

    def __init__(self, samples: pd.DataFrame, batch_days: int = 1, shuffle: bool = False, seed: int = 2026) -> None:
        if batch_days <= 0:
            raise ValueError("batch_days must be positive")
        dates = pd.to_datetime(samples["target_trade_date"], errors="coerce").dt.normalize()
        grouped = pd.DataFrame({"__date": dates}).groupby("__date", sort=True).indices
        self.day_indices = [list(indices) for _, indices in grouped.items()]
        self.batch_days = batch_days
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        order = list(range(len(self.day_indices)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(order)
        for start in range(0, len(order), self.batch_days):
            batch: list[int] = []
            for day_pos in order[start : start + self.batch_days]:
                batch.extend(self.day_indices[day_pos])
            yield batch

    def __len__(self) -> int:
        return int(np.ceil(len(self.day_indices) / self.batch_days))


def collate_msgca_batch(items: list[dict[str, object]]) -> dict[str, object]:
    tensor_keys = (
        "price_window",
        "price_mask",
        "text_features",
        "text_mask",
        "fundamental_features",
        "fundamental_mask",
        "label_next_open_return",
        "label_next_vwap_return",
        "label_direction",
    )
    batch: dict[str, object] = {}
    for key in tensor_keys:
        batch[key] = torch.stack([item[key] for item in items])  # type: ignore[arg-type]
    if items and "loss_context" in items[0]:
        batch["loss_context"] = torch.stack([item["loss_context"] for item in items])  # type: ignore[arg-type]
    for key in ("sample_id", "stock_code", "stock_name", "industry"):
        batch[key] = [str(item[key]) for item in items]
    for key in ("target_trade_date", "feature_asof_date", "decision_ts"):
        batch[key] = pd.to_datetime([item[key] for item in items])
    return batch


def build_datasets(
    config: MSGCAConfig,
    split: str = "train",
    limit: int | None = None,
    feature_scaler: ScalerState | dict[str, object] | None = None,
) -> tuple[MSGCASampleDataset, FeatureLayout, ScalerState]:
    base_samples, selected = load_base_samples_and_selection(config)
    scaler = _coerce_scaler(feature_scaler)
    train_base = filter_samples_by_split(base_samples, config, "train") if scaler is None else pd.DataFrame()
    if limit is not None:
        train_base = train_base.head(limit).copy()
    target_base = filter_samples_by_split(base_samples, config, split)
    if limit is not None:
        target_base = target_base.head(limit).copy()
    context_columns = _loss_context_columns(config)
    if context_columns:
        if scaler is None and not train_base.empty:
            train_base = _attach_loss_context_features(config, train_base, context_columns)
        target_base = _attach_loss_context_features(config, target_base, context_columns)

    if config.data.fast_loader and config.data.use_polars:
        return _build_target_dataset_matrix(config, selected, train_base, target_base, split, feature_scaler=scaler)

    needed_frames = [target_base["sample_id"]]
    if scaler is None:
        needed_frames.insert(0, train_base["sample_id"])
    needed_ids = pd.concat(needed_frames, ignore_index=True).drop_duplicates()
    feature_panel = load_sample_feature_panel(
        config.paths.feature_registry_path,
        selected.selected_features,
        sample_ids=needed_ids,
        strict=True,
        use_polars=config.data.use_polars,
    )
    target_samples = attach_selected_features(target_base, feature_panel, selected)
    layout = infer_feature_layout(config, target_samples)
    if scaler is None:
        train_samples = attach_selected_features(train_base, feature_panel, selected)
        scaler = fit_scaler(train_samples, [*layout.text_columns, *layout.fundamental_columns])
    _validate_scaler_columns(scaler, [*layout.text_columns, *layout.fundamental_columns])
    price = build_price_feature_frame(config, layout.price_columns)
    dataset = MSGCASampleDataset(
        target_samples,
        price,
        layout,
        lookback=config.data.lookback,
        scaler=scaler,
        strict_lookback=config.data.strict_lookback,
        fast_loader=config.data.fast_loader,
        context_columns=context_columns,
        **_price_window_cache_kwargs(config, split),
    )
    return dataset, layout, scaler


def _build_target_dataset_matrix(
    config: MSGCAConfig,
    selected,
    train_base: pd.DataFrame,
    target_base: pd.DataFrame,
    split: str,
    feature_scaler: ScalerState | None = None,
) -> tuple[MSGCASampleDataset, FeatureLayout, ScalerState]:
    layout = infer_feature_layout_from_features(config, selected.selected_features, selected.mode, selected.source_path, selected.blocks)
    _log_data_stage(
        "target_matrix_layout "
        f"split={split} selected={len(selected.selected_features)} "
        f"text={len(layout.text_columns)} fundamental={len(layout.fundamental_columns)}"
    )
    if feature_scaler is None:
        _log_data_stage(f"target_train_matrix_load_start rows={len(train_base)}")
        train_matrix = _load_feature_matrix_for_samples(config, selected, train_base)
        _log_data_stage(f"target_train_matrix_load_done shape={train_matrix.values.shape}")
        _log_data_stage("target_train_scaler_fit_start")
        scaler = fit_scaler_matrix(train_matrix)
        del train_matrix
        gc.collect()
        _log_data_stage("target_train_scaler_fit_done")
    else:
        scaler = feature_scaler
        _log_data_stage(f"target_feature_scaler_reuse columns={len(scaler.columns)}")
    _validate_scaler_columns(scaler, [*layout.text_columns, *layout.fundamental_columns])

    _log_data_stage(f"target_matrix_load_start split={split} rows={len(target_base)}")
    target_matrix = _load_feature_matrix_for_samples(config, selected, target_base)
    _log_data_stage(f"target_matrix_load_done split={split} shape={target_matrix.values.shape}")
    _log_data_stage(f"target_arrays_prepare_start split={split}")
    target_arrays = feature_arrays_from_matrix(target_matrix, layout, scaler)
    del target_matrix
    gc.collect()
    _log_data_stage(f"target_arrays_prepare_done split={split}")

    _log_data_stage(f"target_price_frame_build_start split={split}")
    price = build_price_feature_frame(config, layout.price_columns)
    _log_data_stage(f"target_price_frame_build_done split={split} rows={len(price)}")
    context_columns = _loss_context_columns(config)
    dataset = MSGCASampleDataset(
        target_base,
        price,
        layout,
        lookback=config.data.lookback,
        scaler=None,
        strict_lookback=config.data.strict_lookback,
        fast_loader=True,
        feature_arrays=target_arrays,
        context_columns=context_columns,
        **_price_window_cache_kwargs(config, split),
    )
    _log_data_stage(f"target_dataset_done split={split} rows={len(dataset)}")
    return dataset, layout, scaler


def build_train_validation_datasets(
    config: MSGCAConfig,
    limit: int | None = None,
) -> tuple[MSGCASampleDataset, MSGCASampleDataset, FeatureLayout, ScalerState]:
    base_samples, selected = load_base_samples_and_selection(config)
    train_base = filter_supervised_train_samples(filter_samples_by_split(base_samples, config, "train"), config.data.primary_label)
    valid_base = filter_samples_by_split(base_samples, config, "validation")
    if limit is not None:
        train_base = train_base.head(limit).copy()
        valid_base = valid_base.head(limit).copy()
    context_columns = _loss_context_columns(config)
    if context_columns:
        train_base = _attach_loss_context_features(config, train_base, context_columns)
        valid_base = _attach_loss_context_features(config, valid_base, context_columns)

    if config.data.fast_loader and config.data.use_polars:
        return _build_train_validation_datasets_matrix(config, selected, train_base, valid_base)

    needed_ids = pd.concat([train_base["sample_id"], valid_base["sample_id"]], ignore_index=True).drop_duplicates()
    feature_panel = load_sample_feature_panel(
        config.paths.feature_registry_path,
        selected.selected_features,
        sample_ids=needed_ids,
        strict=True,
        use_polars=config.data.use_polars,
    )
    train_samples = attach_selected_features(train_base, feature_panel, selected)
    valid_samples = attach_selected_features(valid_base, feature_panel, selected)
    layout_source = train_samples if not train_samples.empty else valid_samples
    layout = infer_feature_layout(config, layout_source)
    scaler = fit_scaler(train_samples, [*layout.text_columns, *layout.fundamental_columns])
    price = build_price_feature_frame(config, layout.price_columns)
    dataset_kwargs = {
        "price": price,
        "layout": layout,
        "lookback": config.data.lookback,
        "scaler": scaler,
        "strict_lookback": config.data.strict_lookback,
        "fast_loader": config.data.fast_loader,
        "context_columns": context_columns,
        **_price_window_cache_kwargs(config, "train_validation"),
    }
    train_dataset = MSGCASampleDataset(train_samples, **{**dataset_kwargs, "price_window_cache_name": "train"})
    valid_dataset = MSGCASampleDataset(valid_samples, **{**dataset_kwargs, "price_window_cache_name": "validation"})
    return train_dataset, valid_dataset, layout, scaler


def _build_train_validation_datasets_matrix(
    config: MSGCAConfig,
    selected,
    train_base: pd.DataFrame,
    valid_base: pd.DataFrame,
) -> tuple[MSGCASampleDataset, MSGCASampleDataset, FeatureLayout, ScalerState]:
    layout = infer_feature_layout_from_features(config, selected.selected_features, selected.mode, selected.source_path, selected.blocks)
    _log_data_stage(
        "matrix_layout "
        f"selected={len(selected.selected_features)} text={len(layout.text_columns)} "
        f"fundamental={len(layout.fundamental_columns)}"
    )
    _log_data_stage(f"train_matrix_load_start rows={len(train_base)}")
    train_matrix = _load_feature_matrix_for_samples(config, selected, train_base)
    _log_data_stage(f"train_matrix_load_done shape={train_matrix.values.shape}")
    _log_data_stage("train_scaler_fit_start")
    scaler = fit_scaler_matrix(train_matrix)
    _log_data_stage("train_scaler_fit_done")
    _log_data_stage("train_arrays_prepare_start")
    train_arrays = feature_arrays_from_matrix(train_matrix, layout, scaler)
    del train_matrix
    gc.collect()
    _log_data_stage("train_arrays_prepare_done")

    _log_data_stage(f"valid_matrix_load_start rows={len(valid_base)}")
    valid_matrix = _load_feature_matrix_for_samples(config, selected, valid_base)
    _log_data_stage(f"valid_matrix_load_done shape={valid_matrix.values.shape}")
    _log_data_stage("valid_arrays_prepare_start")
    valid_arrays = feature_arrays_from_matrix(valid_matrix, layout, scaler)
    del valid_matrix
    gc.collect()
    _log_data_stage("valid_arrays_prepare_done")

    _log_data_stage("price_frame_build_start")
    price = build_price_feature_frame(config, layout.price_columns)
    _log_data_stage(f"price_frame_build_done rows={len(price)}")
    context_columns = _loss_context_columns(config)
    dataset_kwargs = {
        "price": price,
        "layout": layout,
        "lookback": config.data.lookback,
        "scaler": None,
        "strict_lookback": config.data.strict_lookback,
        "fast_loader": True,
        "context_columns": context_columns,
        **_price_window_cache_kwargs(config, "train_validation"),
    }
    train_dataset = MSGCASampleDataset(
        train_base,
        feature_arrays=train_arrays,
        **{**dataset_kwargs, "price_window_cache_name": "train"},
    )
    valid_dataset = MSGCASampleDataset(
        valid_base,
        feature_arrays=valid_arrays,
        **{**dataset_kwargs, "price_window_cache_name": "validation"},
    )
    _log_data_stage("matrix_datasets_done")
    return train_dataset, valid_dataset, layout, scaler


def build_train_dataset(
    config: MSGCAConfig,
    limit: int | None = None,
) -> tuple[MSGCASampleDataset, FeatureLayout, ScalerState]:
    base_samples, selected = load_base_samples_and_selection(config)
    train_base = filter_supervised_train_samples(filter_samples_by_split(base_samples, config, "train"), config.data.primary_label)
    if limit is not None:
        train_base = train_base.head(limit).copy()
    context_columns = _loss_context_columns(config)
    if context_columns:
        train_base = _attach_loss_context_features(config, train_base, context_columns)

    if config.data.fast_loader and config.data.use_polars:
        return _build_train_dataset_matrix(config, selected, train_base)

    feature_panel = load_sample_feature_panel(
        config.paths.feature_registry_path,
        selected.selected_features,
        sample_ids=train_base["sample_id"],
        strict=True,
        use_polars=config.data.use_polars,
    )
    train_samples = attach_selected_features(train_base, feature_panel, selected)
    layout = infer_feature_layout(config, train_samples)
    scaler = fit_scaler(train_samples, [*layout.text_columns, *layout.fundamental_columns])
    price = build_price_feature_frame(config, layout.price_columns)
    train_dataset = MSGCASampleDataset(
        train_samples,
        price,
        layout,
        lookback=config.data.lookback,
        scaler=scaler,
        strict_lookback=config.data.strict_lookback,
        fast_loader=config.data.fast_loader,
        context_columns=context_columns,
        **_price_window_cache_kwargs(config, "train"),
    )
    return train_dataset, layout, scaler


def _build_train_dataset_matrix(
    config: MSGCAConfig,
    selected,
    train_base: pd.DataFrame,
) -> tuple[MSGCASampleDataset, FeatureLayout, ScalerState]:
    layout = infer_feature_layout_from_features(config, selected.selected_features, selected.mode, selected.source_path, selected.blocks)
    _log_data_stage(
        "train_only_matrix_layout "
        f"selected={len(selected.selected_features)} text={len(layout.text_columns)} "
        f"fundamental={len(layout.fundamental_columns)}"
    )
    _log_data_stage(f"train_only_matrix_load_start rows={len(train_base)}")
    train_matrix = _load_feature_matrix_for_samples(config, selected, train_base)
    _log_data_stage(f"train_only_matrix_load_done shape={train_matrix.values.shape}")
    _log_data_stage("train_only_scaler_fit_start")
    scaler = fit_scaler_matrix(train_matrix)
    _log_data_stage("train_only_scaler_fit_done")
    _log_data_stage("train_only_arrays_prepare_start")
    train_arrays = feature_arrays_from_matrix(train_matrix, layout, scaler)
    del train_matrix
    gc.collect()
    _log_data_stage("train_only_arrays_prepare_done")

    _log_data_stage("train_only_price_frame_build_start")
    price = build_price_feature_frame(config, layout.price_columns)
    _log_data_stage(f"train_only_price_frame_build_done rows={len(price)}")
    context_columns = _loss_context_columns(config)
    train_dataset = MSGCASampleDataset(
        train_base,
        price,
        layout,
        lookback=config.data.lookback,
        scaler=None,
        strict_lookback=config.data.strict_lookback,
        fast_loader=True,
        feature_arrays=train_arrays,
        context_columns=context_columns,
        **_price_window_cache_kwargs(config, "train"),
    )
    _log_data_stage(f"train_only_dataset_done rows={len(train_dataset)}")
    return train_dataset, layout, scaler


def load_samples_with_features(
    config: MSGCAConfig,
    limit: int | None = None,
) -> pd.DataFrame:
    samples, selected = load_base_samples_and_selection(config)
    if limit is not None:
        samples = samples.head(limit).copy()
    feature_panel = load_sample_feature_panel(
        config.paths.feature_registry_path,
        selected.selected_features,
        sample_ids=samples["sample_id"],
        strict=True,
        use_polars=config.data.use_polars,
    )
    return attach_selected_features(samples, feature_panel, selected)


def _loss_context_columns(config: MSGCAConfig) -> list[str]:
    raw = list(getattr(config.train, "loss_context_columns", []) or [])
    if not raw:
        return []
    if any(str(item).lower() in {"context_all", "all"} for item in raw):
        return list(CONTEXT_COLUMNS)
    return list(dict.fromkeys(str(item) for item in raw))


def _attach_loss_context_features(config: MSGCAConfig, samples: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    _log_data_stage(f"context_attach_start rows={len(samples)} columns={len(columns)}")
    result = attach_context_features(
        samples,
        samples_path=config.paths.samples_path,
        price_path=config.paths.price_path,
        metric_path=config.paths.metric_path,
        feature_registry_path=config.paths.feature_registry_path,
        news_path=config.paths.news_path,
        news_scores_path=config.paths.news_scores_path,
        context_cache_path=config.train.context_cache_path,
        news_cache_path=config.train.context_news_cache_path,
        context_columns=columns,
        strict=True,
    )
    _log_data_stage(f"context_attach_done rows={len(result)} columns={len(columns)}")
    return result


def _load_feature_matrix_for_samples(config: MSGCAConfig, selected, samples: pd.DataFrame) -> SampleFeatureMatrix:
    if "__row_pos" in samples.columns:
        return load_sample_feature_matrix_by_row_positions(
            config.paths.feature_registry_path,
            selected.selected_features,
            row_positions=samples["__row_pos"].to_numpy(dtype=np.int64, copy=False),
            strict=True,
        )
    return load_sample_feature_matrix(
        config.paths.feature_registry_path,
        selected.selected_features,
        sample_ids=samples["sample_id"],
        strict=True,
    )


def load_base_samples_and_selection(config: MSGCAConfig):
    samples_path = config.paths.samples_path
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples parquet: {samples_path}")
    available_columns = set(_parquet_columns(samples_path))
    requested_columns = list(SAMPLE_BASE_COLUMNS)
    columns = [column for column in dict.fromkeys(requested_columns) if column in available_columns]
    samples = pd.read_parquet(samples_path, columns=columns)
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["__row_pos"] = np.arange(len(samples), dtype=np.int64)

    selected = load_selected_features(config.paths.evaluation_dir)
    samples = _prepare_samples(samples)
    samples.attrs["selected_feature_mode"] = selected.mode
    samples.attrs["selected_feature_path"] = None if selected.source_path is None else str(selected.source_path)
    return samples, selected


def attach_selected_features(samples, feature_panel, selected):
    samples = samples.copy()
    if not feature_panel.empty and len(feature_panel.columns) > 1:
        samples = samples.merge(feature_panel, on="sample_id", how="left")
    missing = [column for column in selected.selected_features if column not in samples.columns]
    if missing:
        samples = pd.concat([samples, pd.DataFrame(np.nan, index=samples.index, columns=missing)], axis=1)
    else:
        samples = samples.copy()
    samples.attrs["selected_feature_mode"] = selected.mode
    samples.attrs["selected_feature_path"] = None if selected.source_path is None else str(selected.source_path)
    return _prepare_samples(samples)


def infer_feature_layout(config: MSGCAConfig, samples: pd.DataFrame) -> FeatureLayout:
    selected_obj = load_selected_features(config.paths.evaluation_dir)
    selected = [feature for feature in selected_obj.selected_features if feature in samples.columns]
    selected_blocks = selected_obj.blocks
    mode = str(samples.attrs.get("selected_feature_mode", selected_obj.mode))
    source_path = samples.attrs.get("selected_feature_path")
    text, fundamental = split_features_by_modality(
        selected,
        config.data.text_prefixes,
        config.data.fundamental_prefixes,
        selected_blocks,
    )
    extra_text = [column for column in ("market_news_count", "news_count") if column in samples.columns and column not in text]
    text.extend(extra_text)
    fundamental = [column for column in fundamental if column not in set(text)]
    text = list(dict.fromkeys(text))
    fundamental = list(dict.fromkeys(fundamental))
    text_group_ids, text_group_names = infer_feature_groups(text, selected_blocks)
    fundamental_group_ids, fundamental_group_names = infer_feature_groups(fundamental, selected_blocks)
    return FeatureLayout(
        price_columns=list(config.data.price_columns),
        text_columns=text,
        fundamental_columns=fundamental,
        selected_feature_mode=mode,
        selected_feature_path=None if source_path is None else str(source_path),
        text_group_ids=text_group_ids,
        text_group_names=text_group_names,
        fundamental_group_ids=fundamental_group_ids,
        fundamental_group_names=fundamental_group_names,
    )


def infer_feature_layout_from_features(
    config: MSGCAConfig,
    features: Sequence[str],
    mode: str,
    source_path: str | Path | None,
    selected_blocks: dict[str, list[str]] | None = None,
) -> FeatureLayout:
    selected = list(dict.fromkeys(str(feature) for feature in features))
    text, fundamental = split_features_by_modality(
        selected,
        config.data.text_prefixes,
        config.data.fundamental_prefixes,
        selected_blocks,
    )
    extra_text = [column for column in ("market_news_count", "news_count") if column in selected and column not in text]
    text.extend(extra_text)
    fundamental = [column for column in fundamental if column not in set(text)]
    text = list(dict.fromkeys(text))
    fundamental = list(dict.fromkeys(fundamental))
    text_group_ids, text_group_names = infer_feature_groups(text, selected_blocks or {})
    fundamental_group_ids, fundamental_group_names = infer_feature_groups(fundamental, selected_blocks or {})
    return FeatureLayout(
        price_columns=list(config.data.price_columns),
        text_columns=text,
        fundamental_columns=fundamental,
        selected_feature_mode=mode,
        selected_feature_path=None if source_path is None else str(source_path),
        text_group_ids=text_group_ids,
        text_group_names=text_group_names,
        fundamental_group_ids=fundamental_group_ids,
        fundamental_group_names=fundamental_group_names,
    )


def infer_feature_groups(features: Sequence[str], selected_blocks: dict[str, list[str]] | None = None) -> tuple[list[int], list[str]]:
    ordered = list(dict.fromkeys(str(feature) for feature in features))
    if not ordered:
        return [], []
    feature_to_block: dict[str, str] = {}
    for block_name, block_features in (selected_blocks or {}).items():
        for feature in block_features:
            feature_to_block[str(feature)] = str(block_name)
    names: list[str] = []
    ids: list[int] = []
    name_to_id: dict[str, int] = {}
    for feature in ordered:
        group = _canonical_feature_group(feature, feature_to_block.get(feature))
        if group not in name_to_id:
            name_to_id[group] = len(names)
            names.append(group)
        ids.append(name_to_id[group])
    return ids, names


def _canonical_feature_group(feature: str, block_name: str | None = None) -> str:
    key = f"{block_name or ''}:{feature}".lower()
    if block_name and block_name.lower().startswith("gpt_mined_"):
        return _gpt_mined_semantic_group(block_name)
    if "news" in key:
        return "news"
    if "moneyflow" in key or feature.startswith("mf_"):
        return "moneyflow"
    if "metric" in key or feature.startswith("metric_"):
        return "metric"
    if "alpha158_kbar" in key or feature.startswith(("alpha158_k",)):
        return "alpha158_kbar"
    if "alpha158_price" in key or any(feature.startswith(prefix) for prefix in ("alpha158_low", "alpha158_high", "alpha158_vwap")):
        return "alpha158_price"
    if "alpha158_return" in key or feature.startswith("alpha158_roc"):
        return "alpha158_return"
    if "alpha158_rolling" in key:
        return "alpha158_rolling"
    if feature.startswith("alpha158_"):
        return "alpha158"
    return "other"


def _gpt_mined_semantic_group(block_name: str) -> str:
    name = block_name.lower().removesuffix("_sample")
    gpt_group_map = {
        "gpt_mined_industry_relative": "industry_relative",
        "gpt_mined_interaction": "interaction",
        "gpt_mined_liquidity": "liquidity",
        "gpt_mined_moneyflow": "moneyflow",
        "gpt_mined_news_state": "news",
        "gpt_mined_regime_momentum": "regime_momentum",
        "gpt_mined_reversal": "reversal",
        "gpt_mined_risk_control": "risk_control",
        "gpt_mined_valuation": "valuation",
    }
    if name in gpt_group_map:
        return gpt_group_map[name]
    return name.removeprefix("gpt_mined_") or "other"


def build_price_feature_frame(config: MSGCAConfig, price_columns: Sequence[str]) -> pd.DataFrame:
    price_path = config.paths.price_path
    moneyflow_path = config.paths.moneyflow_path
    if not price_path.exists():
        raise FileNotFoundError(f"Missing price parquet: {price_path}")
    base_columns = ["stock_code", "trade_date", "open", "high", "low", "close", "vwap", "volume", "amount"]
    price = pd.read_parquet(price_path, columns=[column for column in base_columns if column in _parquet_columns(price_path)])
    if "volume" not in price.columns and "vol" in price.columns:
        price["volume"] = price["vol"]

    if moneyflow_path.exists():
        mf_columns = [
            "stock_code",
            "trade_date",
            "buy_sm_amount",
            "sell_sm_amount",
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
            "net_mf_amount",
        ]
        moneyflow = pd.read_parquet(moneyflow_path, columns=[column for column in mf_columns if column in _parquet_columns(moneyflow_path)])
        price = price.merge(moneyflow, on=["stock_code", "trade_date"], how="left")
        amount = pd.to_numeric(price.get("amount"), errors="coerce")
        price["mf_net_amount_ratio"] = _safe_div(_numeric_column(price, "net_mf_amount") * 10.0, amount)
        main = (
            _numeric_column(price, "buy_lg_amount")
            + _numeric_column(price, "buy_elg_amount")
            - _numeric_column(price, "sell_lg_amount")
            - _numeric_column(price, "sell_elg_amount")
        )
        price["mf_main_net_amount_ratio"] = _safe_div(main * 10.0, amount)
        small_pressure = _numeric_column(price, "sell_sm_amount") - _numeric_column(price, "buy_sm_amount")
        price["mf_small_order_pressure"] = _safe_div(small_pressure * 10.0, amount)

    for column in price_columns:
        if column not in price.columns:
            price[column] = np.nan
    return _prepare_price(price[["stock_code", "trade_date", *price_columns]], price_columns)


def filter_samples_by_split(samples: pd.DataFrame, config: MSGCAConfig, split: str) -> pd.DataFrame:
    dates = pd.to_datetime(samples["target_trade_date"], errors="coerce").dt.normalize()
    if split == "train":
        start, end = config.data.train_start, config.data.train_end
    elif split in {"validation", "valid", "val"}:
        start, end = config.data.validation_start, config.data.validation_end
    elif split == "holdout":
        start, end = config.data.holdout_start, config.data.holdout_end
    elif split == "all":
        return samples.copy()
    else:
        raise ValueError(f"Unsupported split: {split}")
    mask = dates.ge(pd.Timestamp(start))
    if end is not None:
        mask &= dates.le(pd.Timestamp(end))
    return samples.loc[mask].copy().reset_index(drop=True)


def filter_supervised_train_samples(samples: pd.DataFrame, label_column: str) -> pd.DataFrame:
    if label_column not in samples.columns:
        raise KeyError(f"Missing training label column: {label_column}")
    values = pd.to_numeric(samples[label_column], errors="coerce").to_numpy(dtype="float64", copy=False)
    keep = np.isfinite(values)
    dropped = int((~keep).sum())
    if dropped:
        _log_data_stage(f"train_label_filter dropped={dropped} label={label_column}")
    return samples.loc[keep].copy().reset_index(drop=True)


def fit_scaler(frame: pd.DataFrame, columns: Sequence[str]) -> ScalerState:
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype="float64")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            medians[column] = 0.0
            means[column] = 0.0
            stds[column] = 1.0
        else:
            medians[column] = float(finite.median())
            means[column] = float(finite.mean())
            std = float(finite.std(ddof=0))
            stds[column] = std if np.isfinite(std) and std > 1e-12 else 1.0
    return ScalerState(list(columns), medians, means, stds)


def fit_scaler_matrix(matrix: SampleFeatureMatrix) -> ScalerState:
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for index, column in enumerate(matrix.columns):
        values = matrix.values[:, index]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            medians[column] = 0.0
            means[column] = 0.0
            stds[column] = 1.0
        else:
            medians[column] = float(np.median(finite_values))
            means[column] = float(finite_values.mean(dtype="float64"))
            std = float(finite_values.std(dtype="float64"))
            stds[column] = std if np.isfinite(std) and std > 1e-12 else 1.0
    return ScalerState(list(matrix.columns), medians, means, stds)


def _coerce_scaler(payload: ScalerState | dict[str, object] | None) -> ScalerState | None:
    if payload is None:
        return None
    return ScalerState.from_dict(payload)


def _float_mapping(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            out[str(key)] = 0.0
    return out


def _validate_scaler_columns(scaler: ScalerState, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in scaler.means or column not in scaler.stds]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Feature scaler missing columns: {preview}")


def feature_arrays_from_matrix(matrix: SampleFeatureMatrix, layout: FeatureLayout, scaler: ScalerState) -> FeatureArrayData:
    column_to_index = {column: index for index, column in enumerate(matrix.columns)}
    text_columns = [column for column in layout.text_columns if column in column_to_index]
    fundamental_columns = [column for column in layout.fundamental_columns if column in column_to_index]
    present = np.isfinite(matrix.values)

    for index, column in enumerate(matrix.columns):
        values = matrix.values[:, index]
        finite = present[:, index]
        median = scaler.medians.get(column, 0.0)
        mean = scaler.means.get(column, 0.0)
        std = scaler.stds.get(column, 1.0) or 1.0
        values[~finite] = median
        values -= mean
        values /= std

    text_values = _matrix_columns(matrix, text_columns, column_to_index)
    fundamental_values = _matrix_columns(matrix, fundamental_columns, column_to_index)
    text_present = _matrix_columns_bool(present, text_columns, column_to_index)
    fundamental_present = _matrix_columns_bool(present, fundamental_columns, column_to_index)
    return FeatureArrayData(
        text_values=text_values,
        text_present=text_present,
        fundamental_values=fundamental_values,
        fundamental_present=fundamental_present,
    )


def _prepare_samples(samples: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    out["stock_code"] = out["stock_code"].astype(str)
    for column in ("feature_asof_date", "target_trade_date", "decision_ts"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    if out["sample_id"].duplicated().any():
        examples = out.loc[out["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"samples.sample_id must be unique: {examples}")
    return out


def _prepare_price(price: pd.DataFrame, price_columns: Sequence[str]) -> pd.DataFrame:
    out = price.copy()
    out["stock_code"] = out["stock_code"].astype(str)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["trade_date"])
    for column in price_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce") if column in out.columns else np.nan
    return out.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


def _row_values(row: pd.Series, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.zeros((0,), dtype="float32"), np.zeros((0,), dtype=bool)
    values = np.array([_float_or_nan(row.get(column, np.nan)) for column in columns], dtype="float32")
    valid = np.isfinite(values)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), valid


def _frame_values(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.zeros((len(frame), 0), dtype="float32"), np.zeros((len(frame), 0), dtype=bool)
    values = frame.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32", copy=True)
    valid = np.isfinite(values)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), valid


def _matrix_columns(matrix: SampleFeatureMatrix, columns: Sequence[str], column_to_index: dict[str, int]) -> np.ndarray:
    if not columns:
        return np.zeros((matrix.values.shape[0], 0), dtype="float32")
    indices = [column_to_index[column] for column in columns]
    return np.ascontiguousarray(matrix.values[:, indices], dtype="float32")


def _matrix_columns_bool(values: np.ndarray, columns: Sequence[str], column_to_index: dict[str, int]) -> np.ndarray:
    if not columns:
        return np.zeros((values.shape[0], 0), dtype=bool)
    indices = [column_to_index[column] for column in columns]
    return np.ascontiguousarray(values[:, indices], dtype=bool)


def _float_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype="float32")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float32", copy=True)


def _string_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), "", dtype=object)
    return frame[column].fillna("").astype(str).to_numpy(dtype=object)


def _datetime_series_to_day_int(values: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(values, errors="coerce").dt.normalize().to_numpy(dtype="datetime64[D]")
    return dates.astype("int64", copy=False)


def _datetime_to_day_int(value: object) -> int:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return int(np.iinfo(np.int64).min)
    return int(timestamp.normalize().to_datetime64().astype("datetime64[D]").astype("int64"))


def _float_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _price_window_cache_kwargs(config: MSGCAConfig, name: str) -> dict[str, object]:
    mode = str(getattr(config.data, "price_window_cache", "none") or "none")
    cache_dir = getattr(config.data, "price_window_cache_dir", None)
    if mode.lower() == "memmap" and not cache_dir:
        cache_dir = str(config.paths.output_root / "price_window_cache")
    return {
        "price_window_cache": mode,
        "price_window_cache_dir": cache_dir,
        "price_window_cache_name": name,
    }


def _safe_cache_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))
    return safe.strip("_") or "dataset"


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _parquet_columns(path: Path) -> list[str]:
    return pq.read_schema(path).names


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {name} columns: {missing}")


def _log_data_stage(message: str) -> None:
    print(f"[{pd.Timestamp.now(tz='UTC').isoformat()}] data_{message}", flush=True)
