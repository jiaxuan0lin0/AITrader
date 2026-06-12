from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from FactorMiner.core.factor_block import write_factor_block
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.core.registry import upsert_block
from FactorMiner.run_factor_workflow import build_parser, main


def test_factor_workflow_select_defaults_to_full_correlation_rows() -> None:
    args = build_parser().parse_args(
        [
            "--mode",
            "select",
            "--select-since",
            "2024-01-01",
            "--select-until",
            "2024-12-31",
        ]
    )

    assert args.corr_row_limit == 0


def test_factor_workflow_inference_uses_target_date_without_labels(tmp_path: Path) -> None:
    samples_path, registry_path = _write_feature_registry(tmp_path)
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "selected_features_reviewed.json").write_text(
        json.dumps(
            {
                "selected_features": ["factor_a", "factor_c"],
                "blocks": {"manual_test_sample": ["factor_a", "factor_c"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "model_features" / "features.parquet"

    exit_code = main(
        [
            "--mode",
            "inference",
            "--samples-path",
            str(samples_path),
            "--feature-registry-path",
            str(registry_path),
            "--evaluation-dir",
            str(evaluation_dir),
            "--target-date",
            "2024-01-03",
            "--output-path",
            str(output_path),
        ]
    )

    assert exit_code == 0
    output = pd.read_parquet(output_path)
    metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))

    assert output["sample_id"].tolist() == ["S20240103_1", "S20240103_2", "S20240103_3"]
    assert output["target_trade_date"].dt.normalize().eq(pd.Timestamp("2024-01-03")).all()
    assert output["factor_a"].tolist() == [10.0, 11.0, 12.0]
    assert output["factor_c"].tolist() == [1.0, 0.0, 1.0]
    assert "label_next_open_return" not in output.columns
    assert metadata["feature_count"] == 2
    assert metadata["row_count"] == 3


def test_factor_workflow_select_filters_by_training_dates(tmp_path: Path) -> None:
    samples_path, registry_path = _write_feature_registry(tmp_path)
    evaluation_dir = tmp_path / "evaluation_select"

    exit_code = main(
        [
            "--mode",
            "select",
            "--samples-path",
            str(samples_path),
            "--feature-registry-path",
            str(registry_path),
            "--evaluation-dir",
            str(evaluation_dir),
            "--select-since",
            "2024-01-01",
            "--select-until",
            "2024-12-31",
            "--labels",
            "label_next_open_return,label_next_vwap_return",
            "--quality-min-non-missing",
            "1",
            "--min-pairs",
            "2",
            "--groups",
            "2",
            "--min-rank-ic-days",
            "1",
            "--min-coverage",
            "0.1",
            "--min-abs-rank-ic",
            "0",
            "--min-corr-pairs",
            "2",
            "--corr-row-limit",
            "10",
        ]
    )

    assert exit_code == 0
    quality_summary = json.loads((evaluation_dir / "sample_feature_quality_summary.json").read_text(encoding="utf-8"))
    single_summary = json.loads((evaluation_dir / "single_factor_summary.json").read_text(encoding="utf-8"))
    selected = json.loads((evaluation_dir / "selected_features.json").read_text(encoding="utf-8"))

    assert quality_summary["sample_count"] == 6
    assert single_summary["sample_count"] == 6
    assert quality_summary["since"] == "2024-01-01"
    assert quality_summary["until"] == "2024-12-31"
    assert set(selected["selected_features"]).issubset({"factor_a", "factor_b", "factor_c"})


def test_factor_workflow_select_slice_reuses_daily_reports(tmp_path: Path) -> None:
    samples_path, registry_path = _write_feature_registry(tmp_path)
    source_dir = tmp_path / "evaluation_source"
    evaluation_dir = tmp_path / "evaluation_slice"
    _write_daily_reports(source_dir)

    exit_code = main(
        [
            "--mode",
            "select",
            "--select-engine",
            "slice",
            "--source-evaluation-dir",
            str(source_dir),
            "--samples-path",
            str(samples_path),
            "--feature-registry-path",
            str(registry_path),
            "--evaluation-dir",
            str(evaluation_dir),
            "--select-since",
            "2024-01-01",
            "--select-until",
            "2024-12-31",
            "--quality-min-non-missing",
            "1",
            "--min-rank-ic-days",
            "1",
            "--min-coverage",
            "0.1",
            "--min-abs-rank-ic",
            "0",
            "--min-corr-pairs",
            "2",
            "--corr-row-limit",
            "10",
            "--review-profile",
            "competition",
            "--prepare-review",
        ]
    )

    assert exit_code == 0
    workflow = json.loads((evaluation_dir / "factor_workflow_summary.json").read_text(encoding="utf-8"))
    factor_summary = pd.read_csv(evaluation_dir / "factor_summary.csv")
    selected = json.loads((evaluation_dir / "selected_features.json").read_text(encoding="utf-8"))

    assert [stage["stage"] for stage in workflow["stages"]] == ["quality", "slice_summary", "selection", "review_prepare"]
    assert factor_summary["target_day_count"].max() == 2
    assert selected["selected_features"]
    assert (evaluation_dir / "review_prompt.md").exists()
    assert (evaluation_dir / "review_inputs.txt").exists()
    assert workflow["config"]["review_profile"] == "competition"
    assert "profile: competition" in (evaluation_dir / "review_prompt.md").read_text(encoding="utf-8")


def _write_feature_registry(tmp_path: Path) -> tuple[Path, Path]:
    samples_path = tmp_path / "samples.parquet"
    feature_root = tmp_path / "features"
    registry_path = feature_root / "feature_registry.json"
    samples = pd.DataFrame(
        {
            "sample_id": [
                "S20240102_1",
                "S20240102_2",
                "S20240102_3",
                "S20240103_1",
                "S20240103_2",
                "S20240103_3",
                "S20250102_1",
                "S20250102_2",
                "S20250102_3",
            ],
            "stock_code": ["000001", "000002", "000003"] * 3,
            "industry": ["bank", "tech", "steel"] * 3,
            "target_trade_date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3 + ["2025-01-02"] * 3),
            "feature_asof_date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-12-31"] * 3),
            "decision_ts": pd.to_datetime(["2024-01-01 15:00:00"] * 3 + ["2024-01-02 15:00:00"] * 3 + ["2024-12-31 15:00:00"] * 3),
            "label_next_open_return": [0.01, 0.02, 0.03, 0.02, 0.04, 0.06, 0.10, 0.11, 0.12],
            "label_next_vwap_return": [0.02, 0.04, 0.06, 0.01, 0.02, 0.03, 0.12, 0.11, 0.10],
        }
    )
    samples.to_parquet(samples_path, index=False)

    features = pd.DataFrame(
        {
            "sample_id": samples["sample_id"],
            "factor_a": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 100.0, 101.0, 102.0],
            "factor_b": [3.0, 2.0, 1.0, 12.0, 11.0, 10.0, 102.0, 101.0, 100.0],
            "factor_c": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    result = FactorResult(
        features,
        [
            FactorSpec("factor_a", "unit", "unit.test", ("x",), "a"),
            FactorSpec("factor_b", "unit", "unit.test", ("x",), "b"),
            FactorSpec("factor_c", "unit", "unit.test", ("x",), "c"),
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


def _write_daily_reports(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    factors = ["factor_a", "factor_b", "factor_c"]
    pd.DataFrame(
        {
            "block": ["manual_test_sample"] * len(factors),
            "factor_name": factors,
            "source": ["unit"] * len(factors),
            "category": ["unit.test"] * len(factors),
            "availability": ["feature_asof_date"] * len(factors),
            "window": [""] * len(factors),
            "lookback": [0] * len(factors),
            "label": ["label_next_open_return"] * len(factors),
        }
    ).to_csv(output_dir / "factor_summary.csv", index=False)
    rows = []
    for factor_name, base_value in zip(factors, [0.05, -0.04, 0.03], strict=True):
        for date, multiplier in [("2024-01-02", 1.0), ("2024-01-03", 1.2), ("2025-01-02", 5.0)]:
            rows.append(
                {
                    "block": "manual_test_sample",
                    "factor_name": factor_name,
                    "source": "unit",
                    "category": "unit.test",
                    "label": "label_next_open_return",
                    "target_trade_date": date,
                    "row_count": 3,
                    "pair_count": 3,
                    "coverage": 1.0,
                    "ic": base_value * multiplier,
                    "rank_ic": base_value * multiplier,
                }
            )
    daily = pd.DataFrame(rows)
    daily.drop(columns=["rank_ic"]).to_csv(output_dir / "factor_ic.csv", index=False)
    daily.drop(columns=["ic"]).to_csv(output_dir / "factor_rankic.csv", index=False)
    group_rows = []
    for factor_name in factors:
        for date in ["2024-01-02", "2024-01-03", "2025-01-02"]:
            group_rows.extend(
                [
                    {
                        "block": "manual_test_sample",
                        "factor_name": factor_name,
                        "label": "label_next_open_return",
                        "target_trade_date": date,
                        "group": 1,
                        "count": 1,
                        "mean_return": 0.01,
                    },
                    {
                        "block": "manual_test_sample",
                        "factor_name": factor_name,
                        "label": "label_next_open_return",
                        "target_trade_date": date,
                        "group": 2,
                        "count": 2,
                        "mean_return": 0.02,
                    },
                ]
            )
    pd.DataFrame(group_rows).to_csv(output_dir / "group_return.csv", index=False)
