from __future__ import annotations

import json
from pathlib import Path

from FactorMiner import run_pipeline


class _FakeBlock:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "granularity": "sample" if self.name.endswith("_sample") else "daily",
            "key_columns": ["sample_id"],
            "factor_path": f"blocks/{self.name}.parquet",
            "manifest_path": f"manifests/{self.name}.json",
            "factor_count": 1,
            "row_count": 10,
            "created_at": "2026-05-21T00:00:00+00:00",
            "description": "",
        }


def test_run_pipeline_executes_stages_in_order(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    summary_path = tmp_path / "evaluation" / "pipeline_summary.json"

    def fake_validate(path: Path, metadata_only: bool = False) -> None:
        calls.append(f"validate:{path.name}")

    monkeypatch.setattr(run_pipeline, "validate_registry", fake_validate)
    monkeypatch.setattr(
        run_pipeline.daily_build,
        "build_daily_blocks",
        lambda args: calls.append(f"daily:{args.block}:aw{args.alpha_workers}") or [_FakeBlock(f"manual_{args.block}")],
    )
    monkeypatch.setattr(
        run_pipeline.sample_features_build,
        "build_sample_feature_blocks",
        lambda args: calls.append(f"sample_features:w{args.workers}") or [_FakeBlock("manual_metric_sample")],
    )
    monkeypatch.setattr(
        run_pipeline.news_sample_build,
        "build_news_sample_blocks",
        lambda args: calls.append("news_sample") or [_FakeBlock("news_llm_market_sample"), _FakeBlock("news_llm_stock_sample")],
    )
    monkeypatch.setattr(
        run_pipeline.quality_eval,
        "run_quality",
        lambda args: calls.append("quality") or {"quality_report_path": str(tmp_path / "quality.csv")},
    )
    monkeypatch.setattr(
        run_pipeline.single_factor_eval,
        "run_single_factor",
        lambda args: calls.append("single_factor") or {"summary_path": str(tmp_path / "factor_summary.csv")},
    )
    monkeypatch.setattr(
        run_pipeline.selection_eval,
        "run_selection",
        lambda args: calls.append("selection") or {"selected_json_path": str(tmp_path / "selected_features.json")},
    )
    monkeypatch.setattr(
        run_pipeline.review_selection_eval,
        "prepare_review",
        lambda args: calls.append("review_prepare") or {"review_prompt": str(tmp_path / "review_prompt.md")},
    )

    exit_code = run_pipeline.main(
        [
            "--skip-precheck",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--factor-root",
            str(tmp_path / "factors"),
            "--feature-root",
            str(tmp_path / "features"),
            "--evaluation-dir",
            str(tmp_path / "evaluation"),
            "--summary-path",
            str(summary_path),
            "--alpha-workers",
            "3",
            "--sample-feature-workers",
            "4",
            "--prepare-review",
        ]
    )

    assert exit_code == 0
    assert calls == [
        "daily:metric:aw3",
        "daily:moneyflow:aw3",
        "daily:alpha158:aw3",
        "validate:factor_registry.json",
        "sample_features:w4",
        "news_sample",
        "validate:feature_registry.json",
        "quality",
        "single_factor",
        "selection",
        "review_prepare",
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert [stage["status"] for stage in summary["stages"]] == ["ok"] * len(summary["stages"])
