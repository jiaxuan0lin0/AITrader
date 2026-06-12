from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from FactorMiner.core.factor_block import write_factor_block
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.core.registry import upsert_block
from FactorMiner.evaluation.quality import main


def test_quality_writes_factor_and_block_reports(tmp_path: Path) -> None:
    samples_path, registry_path = _write_feature_registry(tmp_path)
    output_dir = tmp_path / "evaluation"

    exit_code = main(
        [
            "--samples-path",
            str(samples_path),
            "--feature-registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--max-missing-rate",
            "0.5",
            "--min-non-missing",
            "2",
            "--workers",
            "2",
        ]
    )

    assert exit_code == 0
    quality = pd.read_csv(output_dir / "sample_feature_quality.csv")
    blocks = pd.read_csv(output_dir / "sample_feature_block_quality.csv")
    by_factor = quality.set_index("factor_name")

    assert blocks.loc[0, "block"] == "manual_test_sample"
    assert blocks.loc[0, "sample_match_rate"] == 1.0
    assert by_factor.loc["factor_good", "quality_pass"] == np.True_
    assert by_factor.loc["factor_good", "missing_rate"] == 0.0
    assert by_factor.loc["factor_constant", "quality_flags"] == "constant"
    assert "has_inf" in by_factor.loc["factor_bad", "quality_flags"]
    assert "missing_rate_high" in by_factor.loc["factor_bad", "quality_flags"]


def test_quality_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    samples_path, registry_path = _write_feature_registry(tmp_path)
    pd.DataFrame(
        {
            "sample_id": ["S1", "S1"],
            "target_trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    ).to_parquet(samples_path, index=False)

    with pytest.raises(ValueError, match="samples.sample_id must be unique"):
        main(
            [
                "--samples-path",
                str(samples_path),
                "--feature-registry-path",
                str(registry_path),
                "--output-dir",
                str(tmp_path / "evaluation"),
            ]
        )


def test_quality_filters_samples_by_target_trade_date(tmp_path: Path) -> None:
    samples_path, registry_path = _write_feature_registry(tmp_path)
    output_dir = tmp_path / "evaluation"

    exit_code = main(
        [
            "--samples-path",
            str(samples_path),
            "--feature-registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--since",
            "2024-01-01",
            "--until",
            "2024-12-31",
            "--min-non-missing",
            "1",
        ]
    )

    assert exit_code == 0
    quality = pd.read_csv(output_dir / "sample_feature_quality.csv")
    blocks = pd.read_csv(output_dir / "sample_feature_block_quality.csv")

    assert blocks.loc[0, "sample_count"] == 2
    assert blocks.loc[0, "row_count"] == 2
    assert quality.set_index("factor_name").loc["factor_good", "row_count"] == 2
    assert quality.set_index("factor_name").loc["factor_good", "year_count"] == 1


def _write_feature_registry(tmp_path: Path) -> tuple[Path, Path]:
    samples_path = tmp_path / "samples.parquet"
    feature_root = tmp_path / "features"
    registry_path = feature_root / "feature_registry.json"
    pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "target_trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2025-01-02", "2025-01-03"]),
        }
    ).to_parquet(samples_path, index=False)

    result = FactorResult(
        pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3", "S4"],
                "factor_good": [1.0, 2.0, 3.0, 4.0],
                "factor_constant": [0.0, 0.0, 0.0, 0.0],
                "factor_bad": [1.0, np.inf, np.nan, np.nan],
            }
        ),
        [
            FactorSpec("factor_good", "unit", "unit.good", ("x",), "x"),
            FactorSpec("factor_constant", "unit", "unit.constant", ("x",), "0"),
            FactorSpec("factor_bad", "unit", "unit.bad", ("x",), "bad"),
        ],
        key_columns=("sample_id",),
    )
    block = write_factor_block(
        result,
        "manual_test_sample",
        "sample",
        feature_root / "blocks" / "sample" / "manual_test_sample.parquet",
        feature_root / "manifests" / "manual_test_sample.json",
    )
    block = replace(
        block,
        factor_path="blocks/sample/manual_test_sample.parquet",
        manifest_path="manifests/manual_test_sample.json",
    )
    upsert_block(registry_path, block)
    return samples_path, registry_path
