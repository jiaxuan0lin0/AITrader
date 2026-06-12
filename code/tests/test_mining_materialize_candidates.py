from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from FactorMiner.core.factor_block import write_factor_block
from FactorMiner.core.factor_spec import FactorResult, FactorSpec
from FactorMiner.core.registry import upsert_block
from FactorMiner.mining.materialize_candidates import main


def test_materialize_candidates_writes_blocks_and_registry(tmp_path: Path) -> None:
    paths = _write_materialize_inputs(tmp_path)

    exit_code = main(
        [
            "--round-dir",
            str(paths["round_dir"]),
            "--processed-dir",
            str(paths["processed_dir"]),
            "--feature-root",
            str(paths["feature_root"]),
            "--feature-registry-path",
            str(paths["feature_registry_path"]),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    output_path = paths["feature_root"] / "blocks" / "sample" / "gpt_mined_regime_momentum_sample.parquet"
    manifest_path = paths["feature_root"] / "manifests" / "gpt_mined_regime_momentum_sample.json"
    summary = json.loads((paths["round_dir"] / "materialized" / "materialization_summary.json").read_text(encoding="utf-8"))
    registry = json.loads(paths["feature_registry_path"].read_text(encoding="utf-8"))
    output = pd.read_parquet(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["status"] == "ok"
    assert output["sample_id"].tolist() == ["s1", "s2", "s3", "s4", "s5", "s6"]
    assert {"gpt_raw_mom1", "gpt_mixed_mom_factor", "gpt_industry_mom1"}.issubset(output.columns)
    assert output["gpt_raw_mom1"].notna().sum() == 4
    assert output["gpt_mixed_mom_factor"].notna().sum() == 4
    assert output["gpt_industry_mom1"].notna().sum() == 4
    assert [record["name"] for record in manifest] == ["gpt_raw_mom1", "gpt_mixed_mom_factor", "gpt_industry_mom1"]
    assert "gpt_mined_regime_momentum_sample" in {record["name"] for record in registry}


def _write_materialize_inputs(root: Path) -> dict[str, Path]:
    processed_dir = root / "processed"
    feature_root = root / "features"
    round_dir = root / "gpt_mining" / "round_test"
    processed_dir.mkdir(parents=True)
    round_dir.mkdir(parents=True)

    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    samples = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(1, 7)],
            "stock_code": ["000001.SZ", "000002.SZ"] * 3,
            "feature_asof_date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "industry": ["tech", "bank"] * 3,
        }
    )
    samples.to_parquet(processed_dir / "samples.parquet", index=False)
    price = pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000002.SZ"] * 3,
            "trade_date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "close": [10.0, 20.0, 11.0, 19.0, 12.0, 18.0],
            "amount": [100.0, 200.0, 120.0, 180.0, 140.0, 160.0],
            "volume": [10.0, 20.0, 12.0, 18.0, 14.0, 16.0],
        }
    )
    price.to_parquet(processed_dir / "price.parquet", index=False)
    pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000002.SZ"] * 3,
            "trade_date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "volume_ratio": [1.0, 1.1, 1.2, 0.9, 1.3, 0.8],
        }
    ).to_parquet(processed_dir / "metric.parquet", index=False)

    feature_registry_path = feature_root / "feature_registry.json"
    existing = pd.DataFrame({"sample_id": samples["sample_id"], "factor_a": [1.0, 2.0, 2.0, 1.0, 3.0, 0.5]})
    result = FactorResult(
        existing,
        [FactorSpec("factor_a", "unit", "unit.test", ("x",), "factor_a")],
        key_columns=("sample_id",),
    )
    block = write_factor_block(
        result,
        "manual_test_sample",
        "sample",
        feature_root / "blocks" / "sample" / "manual_test_sample.parquet",
        feature_root / "manifests" / "manual_test_sample.json",
    )
    block = block.__class__(
        **{
            **block.to_record(),
            "key_columns": tuple(block.key_columns),
            "factor_path": "blocks/sample/manual_test_sample.parquet",
            "manifest_path": "manifests/manual_test_sample.json",
        }
    )
    upsert_block(feature_registry_path, block)

    validated = [
        _candidate("gpt_raw_mom1", "rank_cs(return(close, 1))", ["close"], {"close": "processed:price"}),
        _candidate(
            "gpt_mixed_mom_factor",
            "rank_cs(return(close, 1)) * rank_cs(factor_a)",
            ["close", "factor_a"],
            {"close": "processed:price", "factor_a": "feature:manual_test_sample"},
        ),
        _candidate(
            "gpt_industry_mom1",
            "rank_cs(industry_neutralize(return(close, 1)))",
            ["close"],
            {"close": "processed:price"},
        ),
    ]
    (round_dir / "validated").mkdir()
    (round_dir / "validated" / "candidates_validated.json").write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "processed_dir": processed_dir,
        "feature_root": feature_root,
        "feature_registry_path": feature_registry_path,
        "round_dir": round_dir,
    }


def _candidate(name: str, formula: str, inputs: list[str], input_sources: dict[str, str]) -> dict[str, object]:
    return {
        "factor_name": name,
        "formula": formula,
        "inputs": inputs,
        "windows": [1],
        "category": "regime_momentum",
        "hypothesis": "unit test",
        "regime_link": "unit test",
        "expected_direction": 1,
        "leakage_risk": "low",
        "redundancy_risk": "low",
        "implementation_notes": "unit test",
        "priority": 1,
        "input_sources": input_sources,
    }
