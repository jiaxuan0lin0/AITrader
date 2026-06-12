from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from FactorMiner.build.sample_features import main
from FactorMiner.core.factor_block import write_factor_block
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.core.registry import upsert_block, validate_registry


def test_build_sample_features_aligns_daily_block_on_feature_asof_date(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.parquet"
    source_registry_path = _write_source_daily_registry(tmp_path)
    output_root = tmp_path / "features"
    feature_registry_path = output_root / "feature_registry.json"

    pd.DataFrame(
        {
            "sample_id": ["A_20240103"],
            "stock_code": ["A"],
            "feature_asof_date": pd.to_datetime(["2024-01-02"]),
            "target_trade_date": pd.to_datetime(["2024-01-03"]),
        }
    ).to_parquet(samples_path, index=False)

    exit_code = main(
        [
            "--samples-path",
            str(samples_path),
            "--source-registry-path",
            str(source_registry_path),
            "--output-root",
            str(output_root),
            "--feature-registry-path",
            str(feature_registry_path),
            "--blocks",
            "manual_metric",
            "--workers",
            "2",
            "--validate-source",
        ]
    )

    assert exit_code == 0
    block_path = output_root / "blocks" / "sample" / "manual_metric_sample.parquet"
    manifest_path = output_root / "manifests" / "manual_metric_sample.json"
    features = pd.read_parquet(block_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(feature_registry_path.read_text(encoding="utf-8"))

    assert features.columns.tolist() == ["sample_id", "factor_a"]
    assert features.loc[0, "sample_id"] == "A_20240103"
    assert features.loc[0, "factor_a"] == 1.0
    assert manifest[0]["name"] == "factor_a"
    assert registry[0]["name"] == "manual_metric_sample"
    assert registry[0]["granularity"] == "sample"
    assert registry[0]["factor_path"] == "blocks/sample/manual_metric_sample.parquet"
    validate_registry(feature_registry_path)


def test_build_sample_features_rejects_limit_on_default_output_root() -> None:
    with pytest.raises(ValueError, match="limit requires a non-default --output-root"):
        main(["--limit", "10"])


def _write_source_daily_registry(tmp_path: Path) -> Path:
    source_root = tmp_path / "factors"
    registry_path = source_root / "factor_registry.json"
    result = FactorResult(
        pd.DataFrame(
            {
                "stock_code": ["A", "A"],
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "factor_a": [1.0, 99.0],
            }
        ),
        [
            FactorSpec(
                name="factor_a",
                source="metric",
                category="metric.raw",
                inputs=("pb",),
                expression="pb",
            )
        ],
    )
    block = write_factor_block(
        result,
        "manual_metric",
        "daily",
        source_root / "blocks" / "daily" / "manual_metric.parquet",
        source_root / "manifests" / "manual_metric.json",
    )
    block = replace(
        block,
        factor_path="blocks/daily/manual_metric.parquet",
        manifest_path="manifests/manual_metric.json",
    )
    upsert_block(registry_path, block)
    return registry_path
