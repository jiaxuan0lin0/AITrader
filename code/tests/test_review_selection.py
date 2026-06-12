from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from FactorMiner.evaluation.review_selection import main


def test_review_selection_prepares_prompt_and_template(tmp_path: Path) -> None:
    _write_review_inputs(tmp_path)

    exit_code = main(["--output-dir", str(tmp_path), "--prepare"])

    assert exit_code == 0
    prompt = (tmp_path / "review_prompt.md").read_text(encoding="utf-8")
    template = json.loads((tmp_path / "review_response_template.json").read_text(encoding="utf-8"))
    review_inputs = (tmp_path / "review_inputs.txt").read_text(encoding="utf-8")
    assert "Factor Selection Review Prompt" in prompt
    assert "profile: research" in prompt
    assert "factor_a" in prompt
    assert "factor_b" in prompt
    assert set(template) == {"remove", "add_back", "flags", "allowed_flags", "global_notes"}
    assert "sparse_event_signal" in template["allowed_flags"]
    assert "review_profile: research" in review_inputs
    assert "review_prompt.md:" in review_inputs
    assert str((tmp_path / "review_prompt.md").resolve()) in review_inputs
    assert str((tmp_path / "candidate_features.csv").resolve()) in review_inputs


def test_review_selection_applies_response(tmp_path: Path) -> None:
    _write_review_inputs(tmp_path)
    response_path = tmp_path / "review_response.json"
    response_path.write_text(
        json.dumps(
            {
                "remove": [{"factor_name": "factor_c", "reason": "Less interpretable than the remaining selected factor."}],
                "add_back": [{"factor_name": "factor_b", "reason": "Highly redundant but preferred by reviewer for stability."}],
                "flags": [
                    {"factor_name": "factor_b", "flag": "watchlist", "reason": "Keep an eye on redundancy."},
                    {"factor_name": "factor_b", "flag": "manual_review", "reason": "Reviewer explicitly overrode correlation pruning."},
                ],
                "global_notes": ["Reviewed manually in ChatGPT."],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = main(["--output-dir", str(tmp_path), "--apply", "--response-path", str(response_path)])

    assert exit_code == 0
    reviewed = json.loads((tmp_path / "selected_features_reviewed.json").read_text(encoding="utf-8"))
    reviewed_csv = pd.read_csv(tmp_path / "selected_features_reviewed.csv")
    audit = pd.read_csv(tmp_path / "selection_review_audit.csv")
    report = (tmp_path / "selection_review_report.md").read_text(encoding="utf-8")

    assert reviewed["selected_features"] == ["factor_a", "factor_b"]
    assert reviewed["blocks"] == {"manual_test_sample": ["factor_a", "factor_b"]}
    assert reviewed["review_flags"] == {"factor_b": ["watchlist", "manual_review"]}
    assert reviewed_csv.set_index("factor_name").loc["factor_b", "review_action"] == "added_back"
    assert set(audit["action"]) >= {"remove", "add_back", "flag:watchlist", "global_note"}
    assert "Reviewed selected feature count: 2" in report


def test_review_selection_rejects_quality_failed_add_back(tmp_path: Path) -> None:
    _write_review_inputs(tmp_path)
    response_path = tmp_path / "review_response.json"
    response_path.write_text(
        json.dumps(
            {
                "remove": [],
                "add_back": [{"factor_name": "factor_bad", "reason": "Try to force a bad factor."}],
                "flags": [],
                "global_notes": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="quality_failed/constant"):
        main(["--output-dir", str(tmp_path), "--apply", "--response-path", str(response_path)])


def test_review_selection_normalizes_review_flag_aliases(tmp_path: Path) -> None:
    _write_review_inputs(tmp_path)
    response_path = tmp_path / "review_response.json"
    response_path.write_text(
        json.dumps(
            {
                "remove": [],
                "add_back": [],
                "flags": [{"factor_name": "factor_a", "flag": "sparse_event_factor", "reason": "Alias from AI response."}],
                "global_notes": [],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--output-dir", str(tmp_path), "--apply", "--response-path", str(response_path)])

    assert exit_code == 0
    reviewed = json.loads((tmp_path / "selected_features_reviewed.json").read_text(encoding="utf-8"))
    assert reviewed["review_flags"] == {"factor_a": ["sparse_event_signal"]}


def test_review_selection_writes_competition_profile(tmp_path: Path) -> None:
    _write_review_inputs(tmp_path)

    exit_code = main(["--output-dir", str(tmp_path), "--prepare", "--review-profile", "competition"])

    assert exit_code == 0
    prompt = (tmp_path / "review_prompt.md").read_text(encoding="utf-8")
    review_inputs = (tmp_path / "review_inputs.txt").read_text(encoding="utf-8")
    assert "profile: competition" in prompt
    assert "purpose: competition_final" in prompt
    assert "review_profile: competition" in review_inputs


def _write_review_inputs(root: Path) -> None:
    selected = {
        "version": "auto_test",
        "selection_mode": "auto",
        "primary_label": "label_next_open_return",
        "selected_features": ["factor_a", "factor_c"],
        "directions": {"factor_a": 1, "factor_c": -1},
        "blocks": {"manual_test_sample": ["factor_a", "factor_c"]},
        "config": {"corr_threshold": 0.95},
    }
    (root / "selected_features.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates = pd.DataFrame(
        [
            _candidate("factor_a", selected=True, score=1.0, quality_pass=True, constant=False, reason=""),
            _candidate("factor_b", selected=False, score=0.9, quality_pass=True, constant=False, reason="high_corr_with:factor_a"),
            _candidate("factor_c", selected=True, score=0.8, quality_pass=True, constant=False, reason=""),
            _candidate("factor_bad", selected=False, score=0.0, quality_pass=False, constant=True, reason="quality_failed;constant"),
        ]
    )
    candidates.to_csv(root / "candidate_features.csv", index=False)
    candidates.loc[~candidates["selected"]].to_csv(root / "rejected_features.csv", index=False)
    pd.DataFrame(
        [
            {
                "cluster_id": "cluster_0001",
                "factor_name": "factor_a",
                "representative": "factor_a",
                "is_representative": True,
                "cluster_size": 2,
                "score": 1.0,
                "rank_ic_mean": 0.03,
                "rank_ic_ir": 0.5,
                "coverage_mean": 0.9,
                "source": "unit",
                "category": "unit.good",
            },
            {
                "cluster_id": "cluster_0001",
                "factor_name": "factor_b",
                "representative": "factor_a",
                "is_representative": False,
                "cluster_size": 2,
                "score": 0.9,
                "rank_ic_mean": 0.028,
                "rank_ic_ir": 0.6,
                "coverage_mean": 0.9,
                "source": "unit",
                "category": "unit.good",
            },
        ]
    ).to_csv(root / "correlation_clusters.csv", index=False)
    pd.DataFrame(
        [
            {
                "factor_a": "factor_a",
                "factor_b": "factor_b",
                "corr": 0.98,
                "abs_corr": 0.98,
                "overlap_count": 100,
                "corr_method": "spearman",
            }
        ]
    ).to_csv(root / "correlation_conflicts.csv", index=False)
    review_packet = {
        "selection_config": {"primary_label": "label_next_open_return"},
        "summary_counts": {"selected_count": 2, "conflict_count": 1},
        "selected_features": [{"factor_name": "factor_a"}, {"factor_name": "factor_c"}],
        "borderline_cases": [],
        "correlation_clusters": [],
    }
    (root / "review_packet.json").write_text(json.dumps(review_packet, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate(factor_name: str, selected: bool, score: float, quality_pass: bool, constant: bool, reason: str) -> dict[str, object]:
    return {
        "block": "manual_test_sample",
        "factor_name": factor_name,
        "source": "unit",
        "category": "unit.good" if factor_name != "factor_bad" else "unit.bad",
        "selection_score": score,
        "direction": 1 if factor_name != "factor_c" else -1,
        "rank_ic_mean": 0.03 if factor_name != "factor_c" else -0.02,
        "rank_ic_ir": 0.5,
        "rank_ic_day_count": 20,
        "coverage_mean": 0.9,
        "group_spread_mean": 0.001,
        "quality_pass": quality_pass,
        "constant_flag": constant,
        "quality_flags": "" if quality_pass else "constant",
        "selection_status": "candidate" if quality_pass else "rejected",
        "selected": selected,
        "final_reject_reason": reason,
        "cluster_id": "cluster_0001" if factor_name in {"factor_a", "factor_b"} else "cluster_0002",
    }
