from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from FactorMiner.mining.build_packet import REQUIRED_PACKET_FILES, main, validate_packet


def test_build_packet_writes_complete_gpt_inputs(tmp_path: Path) -> None:
    paths = _write_packet_inputs(tmp_path)
    packet_dir = tmp_path / "gpt_mining" / "round_test" / "packet"

    exit_code = main(
        [
            "--processed-dir",
            str(paths["processed_dir"]),
            "--factor-root",
            str(paths["factor_root"]),
            "--feature-root",
            str(paths["feature_root"]),
            "--feature-registry-path",
            str(paths["feature_registry_path"]),
            "--evaluation-dir",
            str(paths["evaluation_dir"]),
            "--packet-dir",
            str(packet_dir),
            "--round-name",
            "round_test",
            "--profile",
            "competition",
            "--cutoff-date",
            "2026-05-20",
            "--candidate-count",
            "25",
        ]
    )

    assert exit_code == 0
    for name in REQUIRED_PACKET_FILES:
        assert (packet_dir / name).exists(), name
        assert (packet_dir / name).stat().st_size > 0, name

    validation = validate_packet(packet_dir)
    manifest = json.loads((packet_dir / "packet_manifest.json").read_text(encoding="utf-8"))
    fields = json.loads((packet_dir / "01_available_fields.json").read_text(encoding="utf-8"))
    selected = json.loads((packet_dir / "03_selected_features_reviewed.json").read_text(encoding="utf-8"))
    schema = json.loads((packet_dir / "candidate_schema.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(packet_dir / "02_existing_factor_summary.csv")
    inputs_text = (packet_dir / "gpt_inputs.txt").read_text(encoding="utf-8")
    prompt = (packet_dir / "prompt_generate_candidates.md").read_text(encoding="utf-8")
    news_text = (packet_dir / "04_existing_news_features.md").read_text(encoding="utf-8")

    assert validation["status"] == "ok"
    assert manifest["profile"] == "competition"
    assert manifest["cutoff_date"] == "2026-05-20"
    assert manifest["candidate_count_request"] == 25
    assert fields["processed_tables"]["samples"]["row_count"] == 3
    assert fields["feature_blocks"][0]["name"] == "manual_test_sample"
    assert "news_llm_stock_sample" in {block["name"] for block in fields["feature_blocks"]}
    assert selected["selected_features"] == ["factor_a", "news_stock_count_5d"]
    assert set(summary["factor_name"]) >= {"factor_a", "news_stock_count_5d", "factor_b"}
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"web_research_summary", "candidates"}
    candidate_schema = schema["properties"]["candidates"]["items"]
    assert candidate_schema["properties"]["windows"]["items"]["enum"] == [1, 3, 5, 10, 20, 60]
    assert "regime_link" in candidate_schema["required"]
    assert "research_source_ids" not in candidate_schema["properties"]
    assert str((packet_dir / "prompt_generate_candidates.md").resolve()) in inputs_text
    assert "不要上传原始 parquet 大表" in inputs_text
    assert "只输出 JSON" in prompt
    assert "必须先联网调研" in prompt
    assert "不要求每个候选逐条绑定来源" in prompt
    assert "candidate_schema.json" in prompt
    assert "Do not request new LLM scoring" in news_text
    assert "ai_compute" in news_text


def test_validate_packet_rejects_missing_file(tmp_path: Path) -> None:
    paths = _write_packet_inputs(tmp_path)
    packet_dir = tmp_path / "packet"
    main(
        [
            "--processed-dir",
            str(paths["processed_dir"]),
            "--factor-root",
            str(paths["factor_root"]),
            "--feature-root",
            str(paths["feature_root"]),
            "--feature-registry-path",
            str(paths["feature_registry_path"]),
            "--evaluation-dir",
            str(paths["evaluation_dir"]),
            "--packet-dir",
            str(packet_dir),
            "--skip-news-keyword-coverage",
        ]
    )
    (packet_dir / "candidate_schema.json").unlink()

    with pytest.raises(FileNotFoundError, match="candidate_schema.json"):
        validate_packet(packet_dir)


def _write_packet_inputs(root: Path) -> dict[str, Path]:
    processed_dir = root / "processed"
    factor_root = root / "factors"
    feature_root = root / "features"
    evaluation_dir = factor_root / "evaluation"
    processed_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)
    (feature_root / "manifests").mkdir(parents=True)

    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "stock_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "industry": ["半导体", "通信设备", "银行"],
            "feature_asof_date": pd.to_datetime(["2026-05-18", "2026-05-18", "2026-05-18"]),
            "target_trade_date": pd.to_datetime(["2026-05-19", "2026-05-19", "2026-05-19"]),
            "label_next_open_return": [0.01, 0.02, -0.01],
            "label_next_vwap_return": [0.011, 0.015, -0.008],
        }
    ).to_parquet(processed_dir / "samples.parquet", index=False)
    pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000002.SZ"],
            "trade_date": pd.to_datetime(["2026-05-18", "2026-05-18"]),
            "close": [10.0, 20.0],
            "volume": [1000.0, 2000.0],
            "amount": [10000.0, 40000.0],
        }
    ).to_parquet(processed_dir / "price.parquet", index=False)
    pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000002.SZ"],
            "trade_date": pd.to_datetime(["2026-05-18", "2026-05-18"]),
            "pe_ttm": [30.0, 80.0],
            "pb": [3.0, 8.0],
            "total_mv": [100.0, 200.0],
        }
    ).to_parquet(processed_dir / "metric.parquet", index=False)
    pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000002.SZ"],
            "trade_date": pd.to_datetime(["2026-05-18", "2026-05-18"]),
            "buy_lg_amount": [10.0, 20.0],
            "sell_lg_amount": [3.0, 7.0],
        }
    ).to_parquet(processed_dir / "moneyflow.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-05-18", "2026-05-18", "2026-05-18"]),
            "matched_stock_count": [1, 0, 1],
            "news_text": ["AI 算力和 CPO 光模块景气提升", "半导体 HBM 存储需求增加", "银行分红稳定"],
        }
    ).to_parquet(processed_dir / "news.parquet", index=False)
    pd.DataFrame(
        {
            "news_text_hash": ["a", "b"],
            "sentiment_score": [0.8, 0.2],
            "impact_score": [0.9, 0.5],
            "risk_score": [0.1, 0.2],
            "relevance_score": [0.9, 0.7],
            "novelty_score": [0.6, 0.4],
            "event_type": ["company", "macro"],
        }
    ).to_parquet(factor_root / "news_llm_scores.parquet", index=False)

    selected = {
        "selected_features": ["factor_a", "news_stock_count_5d"],
        "blocks": {"manual_test_sample": ["factor_a"], "news_llm_stock_sample": ["news_stock_count_5d"]},
        "config": {"since": "2016-01-05", "until": "2026-05-20"},
    }
    (evaluation_dir / "selected_features_reviewed.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            _candidate("manual_test_sample", "factor_a", "unit", "momentum", True, 1.0),
            _candidate("manual_test_sample", "factor_b", "unit", "liquidity", False, 0.8),
            _candidate("news_llm_stock_sample", "news_stock_count_5d", "news_llm", "news.coverage", True, 0.6),
        ]
    ).to_csv(evaluation_dir / "candidate_features.csv", index=False)
    pd.DataFrame(
        [
            {"block": "manual_test_sample", "factor_name": "factor_a", "label": "label_next_open_return", "rank_ic_mean": 0.03},
            {"block": "manual_test_sample", "factor_name": "factor_b", "label": "label_next_open_return", "rank_ic_mean": 0.02},
        ]
    ).to_csv(evaluation_dir / "factor_summary.csv", index=False)
    pd.DataFrame(
        [
            {"block": "manual_test_sample", "factor_name": "factor_a", "quality_pass": True, "missing_rate": 0.0, "constant_flag": False},
            {"block": "manual_test_sample", "factor_name": "factor_b", "quality_pass": True, "missing_rate": 0.0, "constant_flag": False},
        ]
    ).to_csv(evaluation_dir / "sample_feature_quality.csv", index=False)
    pd.DataFrame([{"cluster_id": "cluster_0001", "factor_name": "factor_a"}]).to_csv(evaluation_dir / "correlation_clusters.csv", index=False)
    pd.DataFrame([{"factor_a": "factor_a", "factor_b": "factor_b", "corr": 0.96}]).to_csv(evaluation_dir / "correlation_conflicts.csv", index=False)

    _write_manifest(
        feature_root / "manifests" / "manual_test_sample.json",
        [
            {"name": "factor_a", "source": "unit", "category": "momentum", "inputs": ["close"], "expression": "return(close, 5)", "window": 5},
            {"name": "factor_b", "source": "unit", "category": "liquidity", "inputs": ["volume"], "expression": "rolling_mean(volume, 5)", "window": 5},
        ],
    )
    _write_manifest(
        feature_root / "manifests" / "news_llm_stock_sample.json",
        [
            {
                "name": "news_stock_count_5d",
                "source": "news_llm",
                "category": "news.coverage",
                "inputs": ["sentiment_score", "impact_score"],
                "expression": "count(events) over natural 5d window",
                "window": 5,
            }
        ],
    )
    registry = [
        {
            "name": "manual_test_sample",
            "granularity": "sample",
            "key_columns": ["sample_id"],
            "factor_path": "blocks/sample/manual_test_sample.parquet",
            "manifest_path": "manifests/manual_test_sample.json",
            "factor_count": 2,
            "row_count": 3,
            "created_at": "2026-05-20T00:00:00+00:00",
        },
        {
            "name": "news_llm_stock_sample",
            "granularity": "sample",
            "key_columns": ["sample_id"],
            "factor_path": "blocks/sample/news_llm_stock_sample.parquet",
            "manifest_path": "manifests/news_llm_stock_sample.json",
            "factor_count": 1,
            "row_count": 3,
            "created_at": "2026-05-20T00:00:00+00:00",
        },
    ]
    feature_registry_path = feature_root / "feature_registry.json"
    feature_registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "processed_dir": processed_dir,
        "factor_root": factor_root,
        "feature_root": feature_root,
        "evaluation_dir": evaluation_dir,
        "feature_registry_path": feature_registry_path,
    }


def _candidate(block: str, factor_name: str, source: str, category: str, selected: bool, score: float) -> dict[str, object]:
    return {
        "block": block,
        "factor_name": factor_name,
        "source": source,
        "category": category,
        "selected": selected,
        "selection_status": "candidate",
        "selection_score": score,
        "rank_ic_mean": 0.03,
        "rank_ic_ir": 0.5,
        "coverage_mean": 0.9,
        "quality_pass": True,
        "missing_rate": 0.0,
        "constant_flag": False,
    }


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
