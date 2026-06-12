from __future__ import annotations

import json
from pathlib import Path

from model.msgca.feature_set import load_selected_features, split_features_by_modality


def test_selected_features_prefers_reviewed_then_auto(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "selected_features.json").write_text(
        json.dumps({"selected_features": ["auto_a"], "blocks": {"b": ["auto_a"]}}),
        encoding="utf-8",
    )
    (evaluation / "selected_features_reviewed.json").write_text(
        json.dumps({"selected_features": ["reviewed_a"], "blocks": {"b": ["reviewed_a"]}}),
        encoding="utf-8",
    )

    selected = load_selected_features(evaluation)

    assert selected.mode == "reviewed"
    assert selected.selected_features == ["reviewed_a"]
    assert selected.blocks == {"b": ["reviewed_a"]}


def test_selected_features_uses_auto_when_reviewed_missing(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "selected_features.json").write_text(
        json.dumps({"selected_features": [{"factor_name": "auto_a"}]}),
        encoding="utf-8",
    )

    selected = load_selected_features(evaluation)

    assert selected.mode == "auto"
    assert selected.selected_features == ["auto_a"]


def test_split_features_by_modality_uses_news_blocks() -> None:
    text, fundamental = split_features_by_modality(
        ["gpt_news_macro", "gpt_flow_signal", "metric_pb"],
        text_prefixes=["news_"],
        fundamental_prefixes=["metric_", "mf_"],
        selected_blocks={
            "gpt_mined_news_state_sample": ["gpt_news_macro"],
            "gpt_mined_moneyflow_sample": ["gpt_flow_signal"],
        },
    )

    assert text == ["gpt_news_macro"]
    assert fundamental == ["gpt_flow_signal", "metric_pb"]
