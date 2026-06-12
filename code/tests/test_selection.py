from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from FactorMiner.core.factor_block import write_factor_block
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.core.registry import upsert_block
from FactorMiner.evaluation.selection import main


def test_selection_outputs_auto_list_and_correlation_artifacts(tmp_path: Path) -> None:
    registry_path = _write_feature_registry(tmp_path)
    output_dir = tmp_path / "evaluation"
    _write_quality_and_summary(output_dir)

    exit_code = main(
        [
            "--quality-path",
            str(output_dir / "sample_feature_quality.csv"),
            "--factor-summary-path",
            str(output_dir / "factor_summary.csv"),
            "--feature-registry-path",
            str(registry_path),
            "--output-dir",
            str(output_dir),
            "--primary-label",
            "label_next_open_return",
            "--min-rank-ic-days",
            "5",
            "--min-coverage",
            "0.1",
            "--min-abs-rank-ic",
            "0.01",
            "--corr-threshold",
            "0.95",
            "--min-corr-pairs",
            "5",
            "--corr-row-limit",
            "10",
        ]
    )

    assert exit_code == 0
    selected = pd.read_csv(output_dir / "selected_features.csv")
    candidates = pd.read_csv(output_dir / "candidate_features.csv")
    rejected = pd.read_csv(output_dir / "rejected_features.csv")
    conflicts = pd.read_csv(output_dir / "correlation_conflicts.csv")
    clusters = pd.read_csv(output_dir / "correlation_clusters.csv")
    selected_json = json.loads((output_dir / "selected_features.json").read_text(encoding="utf-8"))
    review_packet = json.loads((output_dir / "review_packet.json").read_text(encoding="utf-8"))

    assert selected["factor_name"].tolist() == ["factor_a", "factor_c"]
    assert selected_json["selected_features"] == ["factor_a", "factor_c"]
    assert selected_json["blocks"] == {"manual_test_sample": ["factor_a", "factor_c"]}
    assert candidates.set_index("factor_name").loc["factor_b", "final_reject_reason"] == "high_corr_with:factor_a"
    assert "factor_bad" in rejected["factor_name"].tolist()
    assert conflicts[["factor_a", "factor_b"]].iloc[0].tolist() == ["factor_a", "factor_b"]
    assert clusters.loc[clusters["factor_name"].eq("factor_a"), "is_representative"].iloc[0]
    assert review_packet["summary_counts"]["selected_count"] == 2


def _write_feature_registry(tmp_path: Path) -> Path:
    feature_root = tmp_path / "features"
    registry_path = feature_root / "feature_registry.json"
    sample_ids = [f"S{i:03d}" for i in range(20)]
    values = list(range(20))
    result = FactorResult(
        pd.DataFrame(
            {
                "sample_id": sample_ids,
                "factor_a": values,
                "factor_b": [value * 2 for value in values],
                "factor_c": [0, 3, 1, 4, 2] * 4,
                "factor_bad": [1.0] * 20,
            }
        ),
        [
            FactorSpec("factor_a", "unit", "unit.good", ("x",), "x"),
            FactorSpec("factor_b", "unit", "unit.good", ("x",), "2*x"),
            FactorSpec("factor_c", "unit", "unit.good", ("x",), "independent"),
            FactorSpec("factor_bad", "unit", "unit.bad", ("x",), "1"),
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
    return registry_path


def _write_quality_and_summary(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_rows = []
    for factor_name in ("factor_a", "factor_b", "factor_c", "factor_bad"):
        quality_rows.append(
            {
                "block": "manual_test_sample",
                "factor_name": factor_name,
                "source": "unit",
                "category": "unit.good" if factor_name != "factor_bad" else "unit.bad",
                "availability": "feature_asof_date",
                "window": "",
                "lookback": 0,
                "row_count": 20,
                "sample_count": 20,
                "non_missing_count": 20,
                "missing_rate": 0.0,
                "constant_flag": factor_name == "factor_bad",
                "quality_pass": factor_name != "factor_bad",
                "quality_flags": "" if factor_name != "factor_bad" else "constant",
            }
        )
    pd.DataFrame(quality_rows).to_csv(output_dir / "sample_feature_quality.csv", index=False)

    summary_rows = [
        _summary_row("factor_a", 0.05, 0.8),
        _summary_row("factor_b", 0.03, 0.6),
        _summary_row("factor_c", -0.04, -0.7),
        _summary_row("factor_bad", 0.10, 1.0),
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "factor_summary.csv", index=False)


def _summary_row(factor_name: str, rank_ic_mean: float, rank_ic_ir: float) -> dict[str, object]:
    return {
        "block": "manual_test_sample",
        "factor_name": factor_name,
        "source": "unit",
        "category": "unit.good" if factor_name != "factor_bad" else "unit.bad",
        "availability": "feature_asof_date",
        "window": "",
        "lookback": 0,
        "label": "label_next_open_return",
        "target_day_count": 10,
        "ic_day_count": 10,
        "rank_ic_day_count": 10,
        "pair_count_mean": 20,
        "coverage_mean": 1.0,
        "ic_mean": rank_ic_mean,
        "ic_std": 0.01,
        "ic_ir": rank_ic_ir,
        "ic_positive_rate": 0.8 if rank_ic_mean > 0 else 0.2,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": 0.01,
        "rank_ic_ir": rank_ic_ir,
        "rank_ic_positive_rate": 0.8 if rank_ic_mean > 0 else 0.2,
        "group_spread_mean": rank_ic_mean / 10,
        "group_spread_positive_rate": 0.8 if rank_ic_mean > 0 else 0.2,
    }
