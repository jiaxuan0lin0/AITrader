from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd
import pytest

from FactorMiner.core.factor_block import FactorBlock, write_factor_block
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.pools.news_sample import SAMPLE_KEY_COLUMNS
from FactorMiner.core.registry import FactorRegistry, upsert_block, validate_registry


def _daily_result(name: str = "factor_a") -> FactorResult:
    return FactorResult(
        pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                name: [1.0, 2.0],
            }
        ),
        [FactorSpec(name, "test", "daily", ("close",), "close")],
    )


def _sample_result(name: str = "news_factor") -> FactorResult:
    return FactorResult(
        pd.DataFrame({"sample_id": ["A_2024-01-03", "B_2024-01-03"], name: [0.1, 0.2]}),
        [FactorSpec(name, "test", "sample", ("sentiment_score",), "sentiment_score", availability="decision_ts")],
        key_columns=SAMPLE_KEY_COLUMNS,
    )


def test_write_factor_block_materializes_parquet_and_manifest(tmp_path) -> None:
    factor_path = tmp_path / "blocks" / "daily" / "manual_alpha158.parquet"
    manifest_path = tmp_path / "manifests" / "manual_alpha158.json"

    block = write_factor_block(_daily_result(), "manual_alpha158", "daily", factor_path, manifest_path)

    assert block.name == "manual_alpha158"
    assert block.granularity == "daily"
    assert block.key_columns == ("stock_code", "trade_date")
    assert block.factor_count == 1
    assert block.row_count == 2
    assert pd.read_parquet(factor_path).columns.tolist() == ["stock_code", "trade_date", "factor_a"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))[0]["name"] == "factor_a"


def test_factor_block_rejects_wrong_key_columns_for_granularity(tmp_path) -> None:
    with pytest.raises(ValueError, match="FactorResult key_columns must be"):
        FactorBlock.from_result(
            name="bad_sample",
            granularity="sample",
            result=_daily_result(),
            factor_path=tmp_path / "bad.parquet",
            manifest_path=tmp_path / "bad.json",
        )


def test_registry_upserts_and_validates_relative_paths(tmp_path) -> None:
    daily = write_factor_block(
        _daily_result("factor_a"),
        "manual_alpha158",
        "daily",
        tmp_path / "blocks" / "daily" / "manual_alpha158.parquet",
        tmp_path / "manifests" / "manual_alpha158.json",
    )
    sample = write_factor_block(
        _sample_result("news_factor"),
        "news_llm_sample",
        "sample",
        tmp_path / "blocks" / "sample" / "news_llm_sample.parquet",
        tmp_path / "manifests" / "news_llm_sample.json",
    )
    daily = replace(daily, factor_path="blocks/daily/manual_alpha158.parquet", manifest_path="manifests/manual_alpha158.json")
    sample = replace(sample, factor_path="blocks/sample/news_llm_sample.parquet", manifest_path="manifests/news_llm_sample.json")
    registry_path = tmp_path / "factor_registry.json"

    registry = FactorRegistry()
    registry.upsert(daily)
    registry.upsert(sample)
    registry.save(registry_path)

    loaded = FactorRegistry.load(registry_path)
    loaded.validate(tmp_path)
    assert [block.name for block in loaded.blocks] == ["manual_alpha158", "news_llm_sample"]


def test_upsert_block_replaces_same_named_block(tmp_path) -> None:
    registry_path = tmp_path / "factor_registry.json"
    first = write_factor_block(_daily_result("factor_a"), "manual_metric", "daily", tmp_path / "a.parquet", tmp_path / "a.json")
    second = write_factor_block(_daily_result("factor_b"), "manual_metric", "daily", tmp_path / "b.parquet", tmp_path / "b.json")

    upsert_block(registry_path, first)
    registry = upsert_block(registry_path, second)

    assert len(registry.blocks) == 1
    assert registry.blocks[0].factor_path == str(tmp_path / "b.parquet")


def test_validate_registry_rejects_duplicate_factors_across_blocks(tmp_path) -> None:
    first = write_factor_block(_daily_result("shared_factor"), "first", "daily", tmp_path / "first.parquet", tmp_path / "first.json")
    second = write_factor_block(_sample_result("shared_factor"), "second", "sample", tmp_path / "second.parquet", tmp_path / "second.json")
    registry_path = tmp_path / "factor_registry.json"
    FactorRegistry([first, second]).save(registry_path)

    with pytest.raises(ValueError, match="Duplicate factor column across blocks"):
        validate_registry(registry_path)


def test_validate_registry_rejects_manifest_factor_missing_from_parquet(tmp_path) -> None:
    block = write_factor_block(_daily_result("factor_a"), "manual_metric", "daily", tmp_path / "metric.parquet", tmp_path / "metric.json")
    (tmp_path / "metric.json").write_text(
        json.dumps(
            [
                {
                    "name": "missing_factor",
                    "source": "test",
                    "category": "bad",
                    "inputs": [],
                    "expression": "missing",
                }
            ]
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "factor_registry.json"
    FactorRegistry([block]).save(registry_path)

    with pytest.raises(KeyError, match="Manifest factors missing from parquet"):
        validate_registry(registry_path)
