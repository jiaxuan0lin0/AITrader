from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from FactorMiner.evaluation.slice_summary import main


def test_slice_summary_rebuilds_factor_summary_from_daily_reports(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    _write_source_reports(source_dir)

    exit_code = main(
        [
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--since",
            "2024-01-01",
            "--until",
            "2024-12-31",
            "--chunksize",
            "3",
        ]
    )

    assert exit_code == 0
    summary = pd.read_csv(output_dir / "factor_summary.csv")
    row = summary.set_index(["factor_name", "label"]).loc[("factor_a", "label_next_open_return")]

    assert row["target_day_count"] == 2
    assert row["ic_day_count"] == 2
    assert row["rank_ic_day_count"] == 2
    assert row["pair_count_mean"] == 3
    assert row["coverage_mean"] == pytest.approx(1.0)
    assert row["ic_mean"] == pytest.approx(0.2)
    assert row["rank_ic_mean"] == pytest.approx(0.3)
    assert row["group_spread_mean"] == pytest.approx(0.01)
    assert row["group_spread_positive_rate"] == pytest.approx(0.5)
    assert (output_dir / "sample_feature_quality.csv").exists()


def _write_source_reports(source_dir: Path) -> None:
    metadata = pd.DataFrame(
        {
            "block": ["manual_test_sample"],
            "factor_name": ["factor_a"],
            "source": ["unit"],
            "category": ["unit.test"],
            "availability": ["feature_asof_date"],
            "window": [""],
            "lookback": [0],
            "label": ["label_next_open_return"],
        }
    )
    metadata.to_csv(source_dir / "factor_summary.csv", index=False)
    daily_base = pd.DataFrame(
        {
            "block": ["manual_test_sample"] * 3,
            "factor_name": ["factor_a"] * 3,
            "source": ["unit"] * 3,
            "category": ["unit.test"] * 3,
            "label": ["label_next_open_return"] * 3,
            "target_trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2025-01-02"]),
            "row_count": [3, 3, 3],
            "pair_count": [3, 3, 3],
            "coverage": [1.0, 1.0, 1.0],
        }
    )
    daily_base.assign(ic=[0.1, 0.3, 0.9]).to_csv(source_dir / "factor_ic.csv", index=False)
    daily_base.assign(rank_ic=[0.2, 0.4, 0.8]).to_csv(source_dir / "factor_rankic.csv", index=False)
    pd.DataFrame(
        {
            "block": ["manual_test_sample"] * 6,
            "factor_name": ["factor_a"] * 6,
            "label": ["label_next_open_return"] * 6,
            "target_trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2025-01-02", "2025-01-02"]),
            "group": [1, 2, 1, 2, 1, 2],
            "count": [2, 1, 2, 1, 2, 1],
            "mean_return": [0.01, 0.05, 0.03, 0.01, 0.01, 0.50],
        }
    ).to_csv(source_dir / "group_return.csv", index=False)
    pd.DataFrame({"block": ["manual_test_sample"], "factor_name": ["factor_a"], "quality_pass": [True]}).to_csv(
        source_dir / "sample_feature_quality.csv",
        index=False,
    )
