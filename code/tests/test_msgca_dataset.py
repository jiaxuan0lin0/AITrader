from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import torch

from model.msgca.dataset import (
    FeatureLayout,
    MSGCASampleDataset,
    FastPriceStore,
    ScalerState,
    build_datasets,
    filter_samples_by_split,
    filter_supervised_train_samples,
    fit_scaler,
    infer_feature_groups,
)
from model.msgca.config import MSGCAConfig


def test_dataset_uses_feature_asof_date_not_target_trade_date() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["000001.SZ_2020-01-03"],
            "stock_code": ["000001.SZ"],
            "stock_name": ["平安银行"],
            "industry": ["银行"],
            "feature_asof_date": [pd.Timestamp("2020-01-02")],
            "target_trade_date": [pd.Timestamp("2020-01-03")],
            "decision_ts": [pd.Timestamp("2020-01-03 09:25:00")],
            "label_next_open_return": [0.01],
            "label_next_vwap_return": [0.02],
            "metric_pb": [1.0],
            "news_count": [3.0],
        }
    )
    price = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "close": [10.0, 11.0, 999.0],
        }
    )
    layout = FeatureLayout(
        price_columns=["close"],
        text_columns=["news_count"],
        fundamental_columns=["metric_pb"],
        selected_feature_mode="selected",
        selected_feature_path=None,
    )

    dataset = MSGCASampleDataset(samples, price, layout, lookback=2, strict_lookback=True)
    item = dataset[0]

    assert item["price_window"].tolist() == [[10.0, 11.0]]


def test_dataset_returns_feature_level_masks() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["000001.SZ_2020-01-03"],
            "stock_code": ["000001.SZ"],
            "feature_asof_date": [pd.Timestamp("2020-01-02")],
            "target_trade_date": [pd.Timestamp("2020-01-03")],
            "label_next_open_return": [0.01],
            "label_next_vwap_return": [0.02],
            "news_count": [float("nan")],
            "news_stock_count_10d": [2.0],
            "metric_pb": [float("nan")],
            "metric_pe": [12.0],
        }
    )
    price = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000001.SZ"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "close": [10.0, 11.0],
        }
    )
    layout = FeatureLayout(
        price_columns=["close"],
        text_columns=["news_count", "news_stock_count_10d"],
        fundamental_columns=["metric_pb", "metric_pe"],
        selected_feature_mode="selected",
        selected_feature_path=None,
    )

    dataset = MSGCASampleDataset(samples, price, layout, lookback=2, strict_lookback=True, fast_loader=True)
    item = dataset[0]

    assert item["text_mask"].tolist() == [False, True]
    assert item["fundamental_mask"].tolist() == [False, True]


def test_fast_price_store_matches_pandas_window() -> None:
    price = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "close": [10.0, None, 999.0],
            "volume": [100.0, 200.0, 300.0],
        }
    )
    store = FastPriceStore.from_frame(price, ["close", "volume"])

    values, mask = store.get_window("000001.SZ", pd.Timestamp("2020-01-02").to_datetime64().astype("datetime64[D]").astype("int64"), 2)

    assert values.tolist() == [[10.0, 0.0], [100.0, 200.0]]
    assert mask.tolist() == [[True, False], [True, True]]


def test_fast_and_pandas_dataset_windows_are_equivalent() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["000001.SZ_2020-01-03"],
            "stock_code": ["000001.SZ"],
            "feature_asof_date": [pd.Timestamp("2020-01-02")],
            "target_trade_date": [pd.Timestamp("2020-01-03")],
            "label_next_open_return": [0.01],
            "label_next_vwap_return": [0.02],
        }
    )
    price = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "close": [10.0, 11.0, 999.0],
        }
    )
    layout = FeatureLayout(
        price_columns=["close"],
        text_columns=[],
        fundamental_columns=[],
        selected_feature_mode="selected",
        selected_feature_path=None,
    )

    fast = MSGCASampleDataset(samples, price, layout, lookback=2, strict_lookback=True, fast_loader=True)
    slow = MSGCASampleDataset(samples, price, layout, lookback=2, strict_lookback=True, fast_loader=False)

    assert fast[0]["price_window"].tolist() == slow[0]["price_window"].tolist()
    assert fast[0]["price_mask"].tolist() == slow[0]["price_mask"].tolist()


def test_price_window_memory_cache_matches_fast_loader() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["A_2020-01-03", "A_2020-01-04"],
            "stock_code": ["A", "A"],
            "feature_asof_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "target_trade_date": pd.to_datetime(["2020-01-03", "2020-01-04"]),
            "label_next_open_return": [0.01, 0.02],
            "label_next_vwap_return": [0.01, 0.02],
        }
    )
    price = pd.DataFrame(
        {
            "stock_code": ["A", "A", "A"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "close": [10.0, 11.0, 12.0],
            "volume": [100.0, 200.0, 300.0],
        }
    )
    layout = FeatureLayout(
        price_columns=["close", "volume"],
        text_columns=[],
        fundamental_columns=[],
        selected_feature_mode="selected",
        selected_feature_path=None,
    )

    cached = MSGCASampleDataset(
        samples,
        price,
        layout,
        lookback=2,
        strict_lookback=True,
        fast_loader=True,
        price_window_cache="memory",
    )
    uncached = MSGCASampleDataset(samples, price, layout, lookback=2, strict_lookback=True, fast_loader=True)

    assert cached[0]["price_window"].tolist() == uncached[0]["price_window"].tolist()
    assert cached[1]["price_window"].tolist() == uncached[1]["price_window"].tolist()
    assert cached[1]["price_mask"].tolist() == uncached[1]["price_mask"].tolist()


def test_price_window_memmap_cache_writes_files(tmp_path: Path) -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["A_2020-01-03"],
            "stock_code": ["A"],
            "feature_asof_date": [pd.Timestamp("2020-01-02")],
            "target_trade_date": [pd.Timestamp("2020-01-03")],
            "label_next_open_return": [0.01],
            "label_next_vwap_return": [0.01],
        }
    )
    price = pd.DataFrame(
        {
            "stock_code": ["A", "A"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "close": [10.0, 11.0],
        }
    )
    layout = FeatureLayout(
        price_columns=["close"],
        text_columns=[],
        fundamental_columns=[],
        selected_feature_mode="selected",
        selected_feature_path=None,
    )

    dataset = MSGCASampleDataset(
        samples,
        price,
        layout,
        lookback=2,
        strict_lookback=True,
        fast_loader=True,
        price_window_cache="memmap",
        price_window_cache_dir=tmp_path,
        price_window_cache_name="case",
    )

    assert dataset[0]["price_window"].tolist() == [[10.0, 11.0]]
    assert (tmp_path / "case_price_windows.float32.memmap").exists()
    assert (tmp_path / "case_price_masks.bool.memmap").exists()


def test_strict_lookback_filters_samples_by_stock_vectorized() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["A_2020-01-02", "A_2020-01-03", "B_2020-01-03"],
            "stock_code": ["A", "A", "B"],
            "feature_asof_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "target_trade_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-03"]),
            "label_next_open_return": [0.01, 0.02, 0.03],
            "label_next_vwap_return": [0.01, 0.02, 0.03],
        }
    )
    price = pd.DataFrame(
        {
            "stock_code": ["A", "A", "B"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "close": [10.0, 11.0, 20.0],
        }
    )
    layout = FeatureLayout(
        price_columns=["close"],
        text_columns=[],
        fundamental_columns=[],
        selected_feature_mode="selected",
        selected_feature_path=None,
    )

    dataset = MSGCASampleDataset(samples, price, layout, lookback=2, strict_lookback=True, fast_loader=True)

    assert dataset.samples["sample_id"].tolist() == ["A_2020-01-03"]


def test_scaler_is_fit_on_train_split_only() -> None:
    config = MSGCAConfig()
    config.data.train_start = "2020-01-01"
    config.data.train_end = "2020-01-31"
    frame = pd.DataFrame(
        {
            "target_trade_date": pd.to_datetime(["2020-01-02", "2020-02-03"]),
            "factor": [1.0, 1000.0],
        }
    )
    train = filter_samples_by_split(frame, config, "train")
    scaler = fit_scaler(train, ["factor"])

    assert scaler.means["factor"] == 1.0
    transformed = scaler.transform_frame(frame, ["factor"])
    assert transformed["factor"].tolist() == [0.0, 999.0]


def test_holdout_split_respects_optional_end_date() -> None:
    config = MSGCAConfig()
    config.data.holdout_start = "2026-05-21"
    config.data.holdout_end = "2026-05-31"
    frame = pd.DataFrame(
        {
            "sample_id": ["before", "inside", "after"],
            "target_trade_date": pd.to_datetime(["2026-05-20", "2026-05-29", "2026-06-01"]),
        }
    )

    holdout = filter_samples_by_split(frame, config, "holdout")

    assert holdout["sample_id"].tolist() == ["inside"]


def test_scaler_state_round_trips_dict() -> None:
    scaler = ScalerState(
        columns=["metric_factor"],
        medians={"metric_factor": 1.0},
        means={"metric_factor": 2.0},
        stds={"metric_factor": 3.0},
    )

    restored = ScalerState.from_dict(scaler.to_dict())

    assert restored == scaler


def test_build_datasets_reuses_checkpoint_feature_scaler(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    samples = pd.DataFrame(
        {
            "sample_id": ["A_2020-01-02", "A_2020-02-03"],
            "stock_code": ["A", "A"],
            "stock_name": ["A", "A"],
            "industry": ["I", "I"],
            "feature_asof_date": pd.to_datetime(["2020-01-01", "2020-02-02"]),
            "target_trade_date": pd.to_datetime(["2020-01-02", "2020-02-03"]),
            "decision_ts": pd.to_datetime(["2020-01-02 09:25", "2020-02-03 09:25"]),
            "label_next_open_return": [0.01, 0.02],
            "label_next_vwap_return": [0.01, 0.02],
        }
    )
    samples.to_parquet(processed / "samples.parquet", index=False)
    pd.DataFrame(
        {
            "stock_code": ["A", "A"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-02-02"]),
            "close": [10.0, 20.0],
        }
    ).to_parquet(processed / "price.parquet", index=False)

    feature_root = tmp_path / "features"
    block_dir = feature_root / "blocks" / "sample"
    manifest_dir = feature_root / "manifests"
    block_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": ["A_2020-01-02", "A_2020-02-03"],
            "metric_factor": [1.0, 120.0],
        }
    ).to_parquet(block_dir / "manual_metric_sample.parquet", index=False)
    (manifest_dir / "manual_metric_sample.json").write_text(
        json.dumps([{"name": "metric_factor"}]),
        encoding="utf-8",
    )
    (feature_root / "feature_registry.json").write_text(
        json.dumps(
            [
                {
                    "name": "manual_metric_sample",
                    "granularity": "sample",
                    "factor_path": "blocks/sample/manual_metric_sample.parquet",
                    "manifest_path": "manifests/manual_metric_sample.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "selected_features.json").write_text(
        json.dumps({"selected_features": ["metric_factor"], "blocks": {"manual_metric_sample": ["metric_factor"]}}),
        encoding="utf-8",
    )

    config = MSGCAConfig()
    config.paths.processed_dir = str(processed)
    config.paths.feature_registry_path = str(feature_root / "feature_registry.json")
    config.paths.evaluation_dir = str(evaluation)
    config.data.train_start = "2020-01-01"
    config.data.train_end = "2020-01-31"
    config.data.validation_start = "2020-02-01"
    config.data.validation_end = "2020-02-28"
    config.data.lookback = 1
    config.data.price_columns = ["close"]
    config.data.fast_loader = False
    config.data.use_polars = False
    checkpoint_scaler = ScalerState(
        columns=["metric_factor"],
        medians={"metric_factor": 100.0},
        means={"metric_factor": 100.0},
        stds={"metric_factor": 10.0},
    )

    dataset, _, scaler = build_datasets(config, split="validation", feature_scaler=checkpoint_scaler)

    assert scaler == checkpoint_scaler
    assert dataset[0]["fundamental_features"].tolist() == [2.0]


def test_missing_return_label_keeps_direction_label_missing() -> None:
    samples = pd.DataFrame(
        {
            "sample_id": ["000001.SZ_2020-01-03"],
            "stock_code": ["000001.SZ"],
            "feature_asof_date": [pd.Timestamp("2020-01-02")],
            "target_trade_date": [pd.Timestamp("2020-01-03")],
            "label_next_open_return": [float("nan")],
            "label_next_vwap_return": [float("nan")],
        }
    )
    price = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000001.SZ"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "close": [10.0, 11.0],
        }
    )
    layout = FeatureLayout(
        price_columns=["close"],
        text_columns=[],
        fundamental_columns=[],
        selected_feature_mode="selected",
        selected_feature_path=None,
    )

    dataset = MSGCASampleDataset(samples, price, layout, lookback=2, strict_lookback=True, fast_loader=True)

    assert torch.isnan(dataset[0]["label_direction"])


def test_supervised_train_filter_drops_missing_primary_labels() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "label_next_open_return": [0.1, float("nan"), -0.2],
        }
    )

    filtered = filter_supervised_train_samples(frame, "label_next_open_return")

    assert filtered["sample_id"].tolist() == ["a", "c"]


def test_infer_feature_groups_prefers_selected_blocks_then_prefix() -> None:
    ids, names = infer_feature_groups(
        [
            "metric_pb",
            "alpha158_KLEN",
            "mf_net_amount_ratio",
            "news_stock_count_10d",
            "gpt_regime_signal",
            "gpt_moneyflow_signal",
            "gpt_news_signal",
        ],
        {
            "manual_metric_sample": ["metric_pb"],
            "manual_alpha158_kbar_sample": ["alpha158_KLEN"],
            "manual_moneyflow_sample": ["mf_net_amount_ratio"],
            "gpt_mined_regime_momentum_sample": ["gpt_regime_signal"],
            "gpt_mined_moneyflow_sample": ["gpt_moneyflow_signal"],
            "gpt_mined_news_state_sample": ["gpt_news_signal"],
        },
    )

    assert names == ["metric", "alpha158_kbar", "moneyflow", "news", "regime_momentum"]
    assert ids == [0, 1, 2, 3, 4, 2, 3]
