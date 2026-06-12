from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from FactorMiner.mining import build_packet
from FactorMiner.mining.validate_candidates import main
from tests.test_mining_build_packet import _write_packet_inputs


def test_validate_candidates_accepts_and_rejects_with_dependency_report(tmp_path: Path) -> None:
    paths = _write_packet_inputs(tmp_path)
    round_dir = tmp_path / "gpt_mining" / "round_validate"
    packet_dir = round_dir / "packet"
    build_packet.main(
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
            "round_validate",
            "--skip-news-keyword-coverage",
        ]
    )
    response_path = round_dir / "gpt_response.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(_candidate_response(), ensure_ascii=False, indent=2), encoding="utf-8")

    exit_code = main(["--round-dir", str(round_dir)])

    assert exit_code == 0
    output_dir = round_dir / "validated"
    validated = json.loads((output_dir / "candidates_validated.json").read_text(encoding="utf-8"))
    rejected = pd.read_csv(output_dir / "candidates_rejected_by_parser.csv")
    dependency = pd.read_csv(output_dir / "candidate_dependency_report.csv")
    summary = json.loads((output_dir / "validation_summary.json").read_text(encoding="utf-8"))

    assert [item["factor_name"] for item in validated] == ["new_raw_mom5", "new_existing_news_interact", "new_mixed_mom_feature"]
    assert {item["dependency_type"] for item in validated} == {"raw_only", "existing_feature", "mixed"}
    assert {item["compute_class"] for item in validated} >= {"cross_sectional", "interaction"}
    assert len(rejected) == 7
    assert "unknown_inputs" in ";".join(rejected["reject_reasons"])
    assert "unknown_functions" in ";".join(rejected["reject_reasons"])
    assert "name_collision_selected_feature" in ";".join(rejected["reject_reasons"])
    assert "label_inputs" in ";".join(rejected["reject_reasons"])
    assert "duplicate_factor_name" in ";".join(rejected["reject_reasons"])
    assert "news_rescoring_term" in ";".join(rejected["reject_reasons"])
    assert summary["total_candidates"] == 10
    assert summary["accepted_count"] == 3
    assert summary["rejected_count"] == 7
    assert dependency.set_index("factor_name").loc["new_mixed_mom_feature", "dependency_type"] == "mixed"


def test_validate_candidates_reads_fenced_object_response(tmp_path: Path) -> None:
    paths = _write_packet_inputs(tmp_path)
    round_dir = tmp_path / "gpt_mining" / "round_fenced"
    packet_dir = round_dir / "packet"
    build_packet.main(
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
    response_path = round_dir / "gpt_response.json"
    payload = {"web_research_summary": _web_research_summary(), "candidates": [_valid_raw_candidate()]}
    response_path.write_text("```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```", encoding="utf-8")

    exit_code = main(["--round-dir", str(round_dir)])

    assert exit_code == 0
    summary = json.loads((round_dir / "validated" / "validation_summary.json").read_text(encoding="utf-8"))
    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 0


def _candidate_response() -> list[dict[str, object]]:
    return {
        "web_research_summary": _web_research_summary(),
        "candidates": [
        _valid_raw_candidate(),
        {
            "factor_name": "new_existing_news_interact",
            "formula": "interaction(rank_cs(factor_a), rank_cs(news_stock_count_5d))",
            "inputs": ["factor_a", "news_stock_count_5d"],
            "windows": [],
            "category": "interaction",
            "hypothesis": "Existing selected factor and existing news state may reinforce each other.",
            "regime_link": "Uses existing news state as a gate for risk appetite and event-driven participation.",
            "expected_direction": 1,
            "leakage_risk": "low",
            "redundancy_risk": "medium",
            "implementation_notes": "Use existing sample feature columns only.",
            "priority": 3,
        },
        {
            "factor_name": "new_mixed_mom_feature",
            "formula": "interaction(rank_cs(return(close, 5)), rank_cs(factor_a))",
            "inputs": ["close", "factor_a"],
            "windows": [5],
            "category": "interaction",
            "hypothesis": "Raw momentum and existing factor strength can combine.",
            "regime_link": "Captures momentum continuation when a stock is already strong under the current growth regime.",
            "expected_direction": 1,
            "leakage_risk": "low",
            "redundancy_risk": "medium",
            "implementation_notes": "Requires raw close and existing feature factor_a.",
            "priority": 4,
        },
        {
            "factor_name": "bad_unknown_input",
            "formula": "rank_cs(does_not_exist)",
            "inputs": ["does_not_exist"],
            "windows": [],
            "category": "liquidity",
            "hypothesis": "Bad field.",
            "regime_link": "Invalid test candidate.",
            "expected_direction": 1,
            "leakage_risk": "low",
            "redundancy_risk": "low",
            "implementation_notes": "Should reject.",
            "priority": 1,
        },
        {
            "factor_name": "bad_operator",
            "formula": "foo(close)",
            "inputs": ["close"],
            "windows": [],
            "category": "liquidity",
            "hypothesis": "Bad operator.",
            "regime_link": "Invalid test candidate.",
            "expected_direction": 1,
            "leakage_risk": "low",
            "redundancy_risk": "low",
            "implementation_notes": "Should reject.",
            "priority": 1,
        },
        {
            "factor_name": "factor_a",
            "formula": "rank_cs(close)",
            "inputs": ["close"],
            "windows": [],
            "category": "liquidity",
            "hypothesis": "Name collision.",
            "regime_link": "Invalid test candidate.",
            "expected_direction": 1,
            "leakage_risk": "low",
            "redundancy_risk": "low",
            "implementation_notes": "Should reject.",
            "priority": 1,
        },
        {
            "factor_name": "bad_label_input",
            "formula": "rank_cs(label_next_open_return)",
            "inputs": ["label_next_open_return"],
            "windows": [],
            "category": "liquidity",
            "hypothesis": "Leaks label.",
            "regime_link": "Invalid test candidate.",
            "expected_direction": 1,
            "leakage_risk": "high",
            "redundancy_risk": "low",
            "implementation_notes": "Should reject.",
            "priority": 1,
        },
        {
            "factor_name": "bad_rescore_request",
            "formula": "rank_cs(close)",
            "inputs": ["close"],
            "windows": [],
            "category": "news_state",
            "hypothesis": "Would require a new scoring pass.",
            "regime_link": "Invalid test candidate.",
            "expected_direction": 1,
            "leakage_risk": "low",
            "redundancy_risk": "low",
            "implementation_notes": "需要重新打分后再聚合新闻。",
            "priority": 1,
        },
        {
            "factor_name": "dup_candidate",
            "formula": "rank_cs(close)",
            "inputs": ["close"],
            "windows": [],
            "category": "liquidity",
            "hypothesis": "Duplicate.",
            "regime_link": "Invalid test candidate.",
            "expected_direction": 1,
            "leakage_risk": "low",
            "redundancy_risk": "low",
            "implementation_notes": "Should reject duplicate.",
            "priority": 1,
        },
        {
            "factor_name": "dup_candidate",
            "formula": "rank_cs(amount)",
            "inputs": ["amount"],
            "windows": [],
            "category": "liquidity",
            "hypothesis": "Duplicate.",
            "regime_link": "Invalid test candidate.",
            "expected_direction": 1,
            "leakage_risk": "low",
            "redundancy_risk": "low",
            "implementation_notes": "Should reject duplicate.",
            "priority": 1,
        },
        ],
    }


def _valid_raw_candidate() -> dict[str, object]:
    return {
        "factor_name": "new_raw_mom5",
        "formula": "rank_cs(return(close, 5))",
        "inputs": ["close"],
        "windows": [5],
        "category": "regime_momentum",
        "hypothesis": "Short-term momentum can persist in a strong technology-growth regime.",
        "regime_link": "Captures short-term continuation under technology-growth risk appetite.",
        "expected_direction": 1,
        "leakage_risk": "low",
        "redundancy_risk": "medium",
        "implementation_notes": "Use backward 5-day close return and cross-sectional rank.",
        "priority": 5,
    }


def _web_research_summary() -> dict[str, object]:
    return {
        "status": "completed",
        "cutoff_date": "2026-05-20",
        "research_queries": ["A股 2025 AI CPO 半导体 行情 资金流 动量"],
        "market_regime_summary": "A-share risk appetite favored technology-growth themes before the cutoff.",
        "factor_design_notes": ["Prefer regime-aware momentum, money-flow interaction, and liquidity expansion formulas."],
        "sources": [
            {
                "source_id": "S01",
                "title": "Example public market source",
                "url": "https://example.com/a-share-tech-regime",
                "published_or_accessed_date": "2026-05-20",
                "evidence": "Used only as a test source for mandatory top-level web research summary.",
            }
        ],
    }
