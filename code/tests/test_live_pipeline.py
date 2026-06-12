from __future__ import annotations

import json
from pathlib import Path

import yaml

from workflow import run_live_pipeline


def test_write_live_model_config_overrides_runtime_paths(tmp_path: Path) -> None:
    source_config = tmp_path / "config.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "processed_dir": "/old/processed",
                    "feature_registry_path": "/old/features/feature_registry.json",
                    "evaluation_dir": "/old/evaluation",
                    "model_root": "/old/model",
                },
                "train": {
                    "context_cache_path": "/old/context.parquet",
                    "context_news_cache_path": "/old/news_context.parquet",
                },
                "strategy": {"score_variant": "direct_theme_medium"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    args = run_live_pipeline.build_parser().parse_args(
        [
            "--summary-path",
            str(tmp_path / "run" / "summary.json"),
            "--model-config",
            str(source_config),
            "--processed-dir",
            str(tmp_path / "processed"),
            "--feature-root",
            str(tmp_path / "features"),
            "--evaluation-dir",
            str(tmp_path / "evaluation"),
            "--model-root",
            str(tmp_path / "model"),
            "--model-score-variant",
            "direct_theme_soft",
        ]
    )
    run_live_pipeline._resolve_paths(args)
    result = run_live_pipeline._write_live_model_config(args, "2026-06-02")
    live_config = yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))

    assert live_config["paths"]["processed_dir"] == str(tmp_path / "processed")
    assert live_config["paths"]["feature_registry_path"] == str(tmp_path / "features" / "feature_registry.json")
    assert live_config["paths"]["evaluation_dir"] == str(tmp_path / "evaluation")
    assert live_config["paths"]["model_root"] == str(tmp_path / "model")
    assert live_config["data"]["holdout_start"] == "2026-06-02"
    assert live_config["data"]["holdout_end"] == "2026-06-02"
    assert live_config["train"]["context_cache_path"] is None
    assert live_config["train"]["context_news_cache_path"] is None
    assert live_config["strategy"]["score_variant"] == "direct_theme_soft"


def test_run_model_inference_uses_absolute_config_and_checkpoint_paths(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model" / "msgca_best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "run" / "model_live_config.yaml"
    config_path.parent.mkdir()
    config_path.write_text("paths: {}\n", encoding="utf-8")
    args = run_live_pipeline.build_parser().parse_args(
        [
            "--summary-path",
            str(tmp_path / "run" / "summary.json"),
            "--checkpoint",
            str(checkpoint),
            "--model-score-variant",
            "direct_theme_soft",
        ]
    )
    run_live_pipeline._resolve_paths(args)
    captured: dict[str, object] = {}

    def fake_run_command(command, passed_args):
        captured["command"] = command
        captured["args"] = passed_args
        return {"command": "ok"}

    monkeypatch.setattr(run_live_pipeline, "_run_command", fake_run_command)
    run_live_pipeline._run_model_inference(args, "2026-06-05", config_path)

    command = captured["command"]
    assert str(config_path.resolve()) in command
    assert str(checkpoint.resolve()) in command
    assert "--score-variant" in command


def test_write_live_samples_adds_unlabeled_target_rows(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    live_processed = tmp_path / "live" / "processed"
    processed.mkdir()
    live_processed.mkdir(parents=True)

    samples = [
        {
            "sample_id": "000001_2026-05-29",
            "stock_code": "000001",
            "stock_name": "A",
            "industry": "bank",
            "feature_asof_date": "2026-05-28",
            "target_trade_date": "2026-05-29",
            "trade_date": "2026-05-29",
            "decision_ts": "2026-05-29 09:25:00",
            "label_next_open_return": 0.01,
            "label_next_vwap_return": 0.02,
        },
        {
            "sample_id": "000002_2026-05-29",
            "stock_code": "000002",
            "stock_name": "B",
            "industry": "tech",
            "feature_asof_date": "2026-05-28",
            "target_trade_date": "2026-05-29",
            "trade_date": "2026-05-29",
            "decision_ts": "2026-05-29 09:25:00",
            "label_next_open_return": 0.03,
            "label_next_vwap_return": 0.04,
        },
    ]
    panel = [
        {"stock_code": "000001", "stock_name": "A", "industry": "bank", "trade_date": "2026-05-28"},
        {"stock_code": "000002", "stock_name": "B", "industry": "tech", "trade_date": "2026-05-28"},
        {"stock_code": "000001", "stock_name": "A", "industry": "bank", "trade_date": "2026-05-29"},
        {"stock_code": "000002", "stock_name": "B", "industry": "tech", "trade_date": "2026-05-29"},
        {"stock_code": "000001", "stock_name": "A", "industry": "bank", "trade_date": "2026-06-01"},
        {"stock_code": "000002", "stock_name": "B", "industry": "tech", "trade_date": "2026-06-01"},
    ]
    import pandas as pd

    pd.DataFrame(samples).to_parquet(processed / "samples.parquet", index=False)
    pd.DataFrame(panel).to_parquet(processed / "panel.parquet", index=False)

    result = run_live_pipeline._write_live_samples(
        source_processed_dir=processed,
        live_processed_dir=live_processed,
        target=pd.Timestamp("2026-06-02"),
        context_start=pd.Timestamp("2026-01-01"),
    )
    output = pd.read_parquet(live_processed / "samples.parquet")
    target_rows = output.loc[pd.to_datetime(output["target_trade_date"]).dt.normalize().eq(pd.Timestamp("2026-06-02"))]

    assert result["target_row_count"] == 2
    assert result["generated_feature_dates"] == ["2026-05-29", "2026-06-01"]
    assert set(target_rows["sample_id"]) == {"000001_2026-06-02", "000002_2026-06-02"}
    assert pd.to_datetime(target_rows["feature_asof_date"]).dt.normalize().unique().tolist() == [pd.Timestamp("2026-06-01")]
    assert target_rows["label_next_open_return"].isna().all()
