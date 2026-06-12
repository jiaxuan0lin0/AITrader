from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from FactorMiner.core.factor_block import write_factor_block
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.core.registry import upsert_block
from FactorMiner.evaluation.single_factor import main


def test_single_factor_writes_ic_rankic_group_and_summary(tmp_path: Path) -> None:
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
            "--labels",
            "label_next_open_return",
            "--min-pairs",
            "3",
            "--groups",
            "2",
            "--workers",
            "2",
        ]
    )

    assert exit_code == 0
    ic = pd.read_csv(output_dir / "factor_ic.csv")
    rankic = pd.read_csv(output_dir / "factor_rankic.csv")
    groups = pd.read_csv(output_dir / "group_return.csv")
    summary = pd.read_csv(output_dir / "factor_summary.csv")
    summary_by_factor = summary.set_index("factor_name")

    assert set(ic["factor_name"]) == {"factor_good", "factor_constant"}
    assert set(rankic["factor_name"]) == {"factor_good", "factor_constant"}
    assert summary_by_factor.loc["factor_good", "ic_mean"] == pytest.approx(1.0)
    assert summary_by_factor.loc["factor_good", "rank_ic_mean"] == pytest.approx(1.0)
    assert summary_by_factor.loc["factor_good", "group_spread_mean"] > 0
    assert pd.isna(summary_by_factor.loc["factor_constant", "ic_mean"])
    assert groups.loc[groups["factor_name"].eq("factor_good"), "group"].isin([1, 2]).all()


def test_single_factor_rejects_missing_label(tmp_path: Path) -> None:
    samples_path, registry_path = _write_feature_registry(tmp_path)

    with pytest.raises(Exception, match="missing_label"):
        main(
            [
                "--samples-path",
                str(samples_path),
                "--feature-registry-path",
                str(registry_path),
                "--output-dir",
                str(tmp_path / "evaluation"),
                "--labels",
                "missing_label",
            ]
        )


def _write_feature_registry(tmp_path: Path) -> tuple[Path, Path]:
    samples_path = tmp_path / "samples.parquet"
    feature_root = tmp_path / "features"
    registry_path = feature_root / "feature_registry.json"
    samples = pd.DataFrame(
        {
            "sample_id": [f"S{day}_{rank}" for day in (1, 2) for rank in (1, 2, 3, 4)],
            "target_trade_date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4),
            "label_next_open_return": [0.01, 0.02, 0.03, 0.04, 0.02, 0.04, 0.06, 0.08],
        }
    )
    samples.to_parquet(samples_path, index=False)
    features = pd.DataFrame(
        {
            "sample_id": samples["sample_id"],
            "factor_good": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "factor_constant": [1.0] * 8,
        }
    )
    result = FactorResult(
        features,
        [
            FactorSpec("factor_good", "unit", "unit.good", ("x",), "x"),
            FactorSpec("factor_constant", "unit", "unit.constant", ("x",), "1"),
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
