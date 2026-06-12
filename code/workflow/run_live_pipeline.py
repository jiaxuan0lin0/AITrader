#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Sequence
from urllib import request
from urllib.error import URLError
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq
import yaml


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from aitrader_paths import DATASETS_ROOT, EXPERIMENTS_ROOT, LOG_DIR, RUNTIME_DIR  # noqa: E402


DEFAULT_FINAL_MODEL_RUN = (
    EXPERIMENTS_ROOT
    / "msgca"
    / "final"
    / "model"
)
DEFAULT_MODEL_CONFIG = DEFAULT_FINAL_MODEL_RUN / "config.resolved.yaml"
DEFAULT_CHECKPOINT = DEFAULT_FINAL_MODEL_RUN / "checkpoints" / "msgca_best.pt"
DEFAULT_PROCESSED_DIR = DATASETS_ROOT / "processed"
DEFAULT_FACTOR_ROOT = DATASETS_ROOT / "factors"
DEFAULT_FEATURE_ROOT = DATASETS_ROOT / "features"
DEFAULT_EVALUATION_DIR = DEFAULT_FACTOR_ROOT / "evaluation" / "final"
DEFAULT_NEWS_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_GPT_MINING_ROUND_DIR = DEFAULT_FACTOR_ROOT / "gpt_mining" / "final"
DEFAULT_LIVE_CONTEXT_START = "2025-01-01"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the live AITrader flow: data update, news LLM scoring, factor construction, "
            "inference feature assembly, and MSGCA buy-signal generation."
        )
    )
    parser.add_argument("--target-date", default=None, help="Prediction target_trade_date. Defaults to latest samples date after data update.")
    parser.add_argument("--python-bin", default=os.environ.get("AITRADER_PYTHON_BIN", sys.executable or "python3"))
    parser.add_argument("--summary-path", type=Path, default=None)

    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTOR_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--factor-registry-path", type=Path, default=None)
    parser.add_argument("--feature-registry-path", type=Path, default=None)
    parser.add_argument("--selected-features-path", type=Path, default=None)
    parser.add_argument(
        "--live-sample-mode",
        choices=("auto", "off", "force"),
        default="auto",
        help=(
            "Build an auditable runtime live sample set when target_date is not present in "
            "processed/samples.parquet. Use off to require pre-existing samples."
        ),
    )
    parser.add_argument("--live-context-start", default=DEFAULT_LIVE_CONTEXT_START)
    parser.add_argument("--live-workspace", type=Path, default=None)

    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--data-update-script", type=Path, default=CODE_ROOT / "data" / "daily_update_a_share.sh")
    parser.add_argument("--sync-start-date", default=None)
    parser.add_argument("--sync-end-date", default=None)
    parser.add_argument("--no-auto-start-sync", action="store_true", help="Do not pass --auto-start to data sync when start date is omitted.")

    parser.add_argument("--skip-news-scoring", action="store_true")
    parser.add_argument("--news-service", choices=("auto", "assume-running", "off"), default="auto")
    parser.add_argument("--news-server-script", type=Path, default=CODE_ROOT / "FactorMiner" / "news_scoring" / "serve_qwen3_32b_awq_vllm.sh")
    parser.add_argument("--news-server-log", type=Path, default=LOG_DIR / "live_news_vllm.log")
    parser.add_argument("--news-server-timeout-sec", type=int, default=600)
    parser.add_argument("--keep-news-service", action="store_true", help="Keep a service started by this script running after scoring.")
    parser.add_argument("--news-base-url", default=DEFAULT_NEWS_BASE_URL)
    parser.add_argument("--news-model", default="qwen3-news")
    parser.add_argument("--news-api-key", default="EMPTY")
    parser.add_argument("--news-since", default=None)
    parser.add_argument("--news-until", default=None)
    parser.add_argument("--news-limit", type=int, default=None)
    parser.add_argument("--news-checkpoint-size", type=int, default=1000)
    parser.add_argument("--news-concurrency", type=int, default=20)
    parser.add_argument("--news-request-batch-size", type=int, default=4)
    parser.add_argument("--news-max-tokens", type=int, default=1024)
    parser.add_argument("--use-guided-json", action="store_true")

    parser.add_argument("--skip-factor-build", action="store_true")
    parser.add_argument(
        "--skip-daily-factor-build",
        action="store_true",
        help="Reuse the existing daily factor registry while rebuilding sample/news live feature blocks.",
    )
    parser.add_argument("--daily-blocks", default="metric,moneyflow,alpha158")
    parser.add_argument("--sample-blocks", default="all")
    parser.add_argument("--news-windows", default="1,3,5,10")
    parser.add_argument("--news-scope", choices=("split", "all", "market", "stock"), default="split")
    parser.add_argument("--alpha-workers", type=int, default=6)
    parser.add_argument("--alpha-layout", choices=("split", "single"), default="split")
    parser.add_argument("--sample-feature-workers", type=int, default=4)
    parser.add_argument("--disable-neutral", action="store_true")
    parser.add_argument("--run-selection", action="store_true", help="Also run quality/single-factor/selection. Default assumes fixed factor combo.")
    parser.add_argument("--prepare-review", action="store_true")
    parser.add_argument("--gpt-mining-round-dir", type=Path, default=DEFAULT_GPT_MINING_ROUND_DIR)
    parser.add_argument("--skip-gpt-live-features", action="store_true")

    parser.add_argument("--skip-feature-assembly", action="store_true")
    parser.add_argument("--feature-output-path", type=Path, default=None)

    parser.add_argument("--skip-model-inference", action="store_true")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-root", type=Path, default=None, help="Override paths.model_root in the live inference config.")
    parser.add_argument("--model-score-variant", default=None, help="Override strategy.score_variant for live inference.")
    parser.add_argument("--positions-path", type=Path, default=None)
    parser.add_argument("--prediction-limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_paths(args)
    summary = _initial_summary(args)
    _write_json(args.summary_path, summary)

    started_news_proc: subprocess.Popen[str] | None = None
    try:
        if not args.skip_data_update:
            _run_stage(summary, args, "data_update", lambda: _run_data_update(args))

        target_date = _resolve_target_date(args)
        summary["target_date"] = target_date
        _write_json(args.summary_path, summary)

        if not args.skip_news_scoring:
            started_news_proc = _run_stage(summary, args, "news_service", lambda: _ensure_news_service(args))
            _run_stage(summary, args, "news_scoring", lambda: _run_news_scoring(args))
            if started_news_proc is not None and not args.keep_news_service:
                _run_stage(summary, args, "news_service_stop", lambda: _stop_started_process(started_news_proc))
                started_news_proc = None

        if args.live_sample_mode != "off":
            _run_stage(summary, args, "live_inputs", lambda: _prepare_live_inputs(args, target_date))
            _refresh_summary_paths(summary, args)
            _write_json(args.summary_path, summary)

        if not args.skip_factor_build:
            _run_stage(summary, args, "factor_build", lambda: _run_factor_build(args))

        if getattr(args, "live_inputs_active", False) and not args.skip_factor_build and not args.skip_gpt_live_features:
            _run_stage(summary, args, "gpt_live_features", lambda: _run_gpt_live_features(args))

        if not args.skip_feature_assembly:
            _run_stage(summary, args, "feature_assembly", lambda: _run_feature_assembly(args, target_date))

        if not args.skip_model_inference:
            live_config = _run_stage(summary, args, "live_model_config", lambda: _write_live_model_config(args, target_date))
            _run_stage(summary, args, "model_inference", lambda: _run_model_inference(args, target_date, Path(live_config["config_path"])))
            _run_stage(summary, args, "signal_summary", lambda: _summarize_signals(args, target_date))

        summary["status"] = "ok"
        summary["finished_at"] = _utc_now()
        _write_json(args.summary_path, summary)
        print(f"live_pipeline_summary={args.summary_path}")
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["finished_at"] = _utc_now()
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(args.summary_path, summary)
        raise
    finally:
        if started_news_proc is not None and not args.keep_news_service:
            _stop_started_process(started_news_proc)


def _resolve_paths(args: argparse.Namespace) -> None:
    args.factor_registry_path = args.factor_registry_path or args.factor_root / "factor_registry.json"
    args.feature_registry_path = args.feature_registry_path or args.feature_root / "feature_registry.json"
    if args.selected_features_path is None:
        reviewed = args.evaluation_dir / "selected_features_reviewed.json"
        auto = args.evaluation_dir / "selected_features.json"
        args.selected_features_path = reviewed if reviewed.exists() else auto
    if args.summary_path is None:
        label = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.summary_path = RUNTIME_DIR / "live_pipeline" / label / "summary.json"
    args.live_workspace = args.live_workspace or args.summary_path.parent / "live_inputs"
    args.live_inputs_active = False


def _run_stage(summary: dict[str, Any], args: argparse.Namespace, name: str, action):
    record: dict[str, Any] = {"stage": name, "started_at": _utc_now()}
    summary["stages"].append(record)
    _write_json(args.summary_path, summary)
    started = time.monotonic()
    try:
        result = action()
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "duration_sec": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(args.summary_path, summary)
        raise
    record.update(
        {
            "status": "ok",
            "finished_at": _utc_now(),
            "duration_sec": round(time.monotonic() - started, 3),
            "result": _jsonable(result),
        }
    )
    _write_json(args.summary_path, summary)
    return result


def _run_data_update(args: argparse.Namespace) -> dict[str, Any]:
    command = ["bash", str(args.data_update_script)]
    if args.sync_start_date:
        command.extend(["--start-date", args.sync_start_date])
    elif not args.no_auto_start_sync:
        command.append("--auto-start")
    if args.sync_end_date:
        command.extend(["--end-date", args.sync_end_date])
    return _run_command(command, args)


def _ensure_news_service(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if args.news_service == "off":
        return None
    if _news_service_ready(args.news_base_url):
        return None
    if args.news_service == "assume-running":
        raise RuntimeError(f"News scoring service is not reachable: {args.news_base_url}")
    env = _base_env(args)
    env.setdefault("GPU_MEMORY_UTILIZATION", "0.82")
    env.setdefault("MAX_NUM_SEQS", "32")
    parsed = urlparse(args.news_base_url)
    if parsed.port is not None:
        env["PORT"] = str(parsed.port)
    args.news_server_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.news_server_log.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", str(args.news_server_script)],
        cwd=str(CODE_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + args.news_server_timeout_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"News scoring service exited early with code {proc.returncode}. See {args.news_server_log}")
        if _news_service_ready(args.news_base_url):
            return proc
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for news scoring service: {args.news_base_url}. See {args.news_server_log}")


def _run_news_scoring(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        args.python_bin,
        "-u",
        "-m",
        "FactorMiner.news_scoring.score_news_items",
        "--news-path",
        str(args.processed_dir / "news.parquet"),
        "--scores-path",
        str(args.factor_root / "news_llm_scores.parquet"),
        "--base-url",
        args.news_base_url,
        "--model",
        args.news_model,
        "--api-key",
        args.news_api_key,
        "--checkpoint-size",
        str(args.news_checkpoint_size),
        "--concurrency",
        str(args.news_concurrency),
        "--request-batch-size",
        str(args.news_request_batch_size),
        "--max-tokens",
        str(args.news_max_tokens),
    ]
    if args.use_guided_json:
        command.append("--use-guided-json")
    if args.news_since:
        command.extend(["--since", args.news_since])
    if args.news_until:
        command.extend(["--until", args.news_until])
    if args.news_limit is not None:
        command.extend(["--limit", str(args.news_limit)])
    return _run_command(command, args)


def _run_factor_build(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        args.python_bin,
        "-m",
        "FactorMiner.run_pipeline",
        "--processed-dir",
        str(args.processed_dir),
        "--factor-root",
        str(args.factor_root),
        "--feature-root",
        str(args.feature_root),
        "--factor-registry-path",
        str(args.factor_registry_path),
        "--feature-registry-path",
        str(args.feature_registry_path),
        "--news-scores-path",
        str(args.factor_root / "news_llm_scores.parquet"),
        "--daily-blocks",
        args.daily_blocks,
        "--sample-blocks",
        args.sample_blocks,
        "--news-windows",
        args.news_windows,
        "--news-scope",
        args.news_scope,
        "--alpha-workers",
        str(args.alpha_workers),
        "--alpha-layout",
        args.alpha_layout,
        "--sample-feature-workers",
        str(args.sample_feature_workers),
    ]
    if args.disable_neutral:
        command.append("--disable-neutral")
    if args.skip_daily_factor_build:
        command.append("--skip-daily")
    if not args.run_selection:
        command.extend(["--skip-quality", "--skip-single-factor", "--skip-selection"])
    if args.prepare_review:
        command.append("--prepare-review")
    return _run_command(command, args)


def _run_gpt_live_features(args: argparse.Namespace) -> dict[str, Any]:
    if not args.gpt_mining_round_dir.exists():
        raise FileNotFoundError(f"Missing GPT mining round dir: {args.gpt_mining_round_dir}")
    command = [
        args.python_bin,
        "-m",
        "FactorMiner.mining.materialize_candidates",
        "--round-dir",
        str(args.gpt_mining_round_dir),
        "--processed-dir",
        str(args.processed_dir),
        "--samples-path",
        str(args.processed_dir / "samples.parquet"),
        "--price-path",
        str(args.processed_dir / "price.parquet"),
        "--metric-path",
        str(args.processed_dir / "metric.parquet"),
        "--feature-root",
        str(args.feature_root),
        "--feature-registry-path",
        str(args.feature_registry_path),
        "--summary-path",
        str(args.summary_path.parent / "gpt_live_materialization_summary.json"),
        "--overwrite",
    ]
    return _run_command(command, args)


def _run_feature_assembly(args: argparse.Namespace, target_date: str) -> dict[str, Any]:
    command = [
        args.python_bin,
        "-m",
        "FactorMiner.run_factor_workflow",
        "--mode",
        "inference",
        "--processed-dir",
        str(args.processed_dir),
        "--factor-root",
        str(args.factor_root),
        "--feature-root",
        str(args.feature_root),
        "--samples-path",
        str(args.processed_dir / "samples.parquet"),
        "--feature-registry-path",
        str(args.feature_registry_path),
        "--evaluation-dir",
        str(args.evaluation_dir),
        "--selected-features-path",
        str(args.selected_features_path),
        "--target-date",
        target_date,
    ]
    output_path = args.feature_output_path or _default_feature_output_path(target_date)
    command.extend(["--output-path", str(output_path)])
    return {**_run_command(command, args), "output_path": str(output_path)}


def _write_live_model_config(args: argparse.Namespace, target_date: str) -> dict[str, Any]:
    output_path = args.summary_path.parent / "model_live_config.yaml"
    if not args.model_config.exists():
        raise FileNotFoundError(f"Missing model config: {args.model_config}")
    config = yaml.safe_load(args.model_config.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a mapping: {args.model_config}")
    paths = config.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Model config paths section must be a mapping")
    paths["processed_dir"] = str(args.processed_dir)
    paths["feature_registry_path"] = str(args.feature_registry_path)
    paths["evaluation_dir"] = str(args.evaluation_dir)
    paths["model_root"] = str(args.model_root or DEFAULT_FINAL_MODEL_RUN)
    strategy = config.setdefault("strategy", {})
    if not isinstance(strategy, dict):
        raise ValueError("Model config strategy section must be a mapping")
    if args.model_score_variant is not None:
        strategy["score_variant"] = str(args.model_score_variant)
    train = config.setdefault("train", {})
    if not isinstance(train, dict):
        raise ValueError("Model config train section must be a mapping")
    train["context_cache_path"] = None
    train["context_news_cache_path"] = None
    data = config.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("Model config data section must be a mapping")
    data["holdout_start"] = target_date
    data["holdout_end"] = target_date
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"config_path": str(output_path), "source_config": str(args.model_config)}


def _run_model_inference(args: argparse.Namespace, target_date: str, live_config_path: Path) -> dict[str, Any]:
    checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    command = [
        args.python_bin,
        "-m",
        "model.msgca.predict_live",
        "--config",
        str(live_config_path.resolve()),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--target-date",
        target_date,
    ]
    if args.model_score_variant is not None:
        command.extend(["--score-variant", str(args.model_score_variant)])
    if args.positions_path is not None:
        command.extend(["--positions-path", str(args.positions_path)])
    if args.prediction_limit is not None:
        command.extend(["--limit", str(args.prediction_limit)])
    return _run_command(command, args)


def _summarize_signals(args: argparse.Namespace, target_date: str) -> dict[str, Any]:
    signals_path = _signals_path(args, target_date)
    if not signals_path.exists():
        raise FileNotFoundError(f"Missing live signal output: {signals_path}")
    signals = pd.read_csv(signals_path)
    if "suggested_action" in signals.columns:
        actions = signals["suggested_action"].astype(str)
    else:
        actions = pd.Series("", index=signals.index, dtype="object")
    buys = signals.loc[actions.eq("buy")].copy()
    holds = signals.loc[actions.eq("hold")].copy()
    buy_list_path = signals_path.with_name(f"buy_list_{_date_label(target_date)}.csv")
    buys.to_csv(buy_list_path, index=False)
    return {
        "signals_path": str(signals_path),
        "buy_list_path": str(buy_list_path),
        "row_count": int(len(signals)),
        "buy_count": int(len(buys)),
        "hold_count": int(len(holds)),
    }


def _run_command(command: Sequence[str | Path], args: argparse.Namespace) -> dict[str, Any]:
    command = [str(item) for item in command]
    printable = shlex.join(command)
    subprocess.run(command, cwd=str(CODE_ROOT), env=_base_env(args), check=True)
    return {"command": printable}


def _prepare_live_inputs(args: argparse.Namespace, target_date: str) -> dict[str, Any]:
    target = pd.Timestamp(target_date).normalize()
    samples_path = args.processed_dir / "samples.parquet"
    target_available = _target_samples_available(samples_path, target)
    if args.live_sample_mode == "auto" and target_available:
        args.live_inputs_active = False
        return {"active": False, "reason": "target_samples_already_available", "samples_path": str(samples_path)}

    source_processed_dir = args.processed_dir
    live_processed_dir = args.live_workspace / "processed"
    live_feature_root = args.live_workspace / "features"
    live_processed_dir.mkdir(parents=True, exist_ok=True)
    live_feature_root.mkdir(parents=True, exist_ok=True)
    _ensure_basic_input(source_processed_dir)
    _link_processed_inputs(source_processed_dir, live_processed_dir)
    metadata = _write_live_samples(
        source_processed_dir=source_processed_dir,
        live_processed_dir=live_processed_dir,
        target=target,
        context_start=_parse_context_start(args.live_context_start),
    )

    args.processed_dir = live_processed_dir
    args.feature_root = live_feature_root
    args.feature_registry_path = live_feature_root / "feature_registry.json"
    if args.feature_output_path is None:
        args.feature_output_path = args.live_workspace / "model_features" / f"target_date={_date_label(target_date)}" / "features.parquet"
    args.live_inputs_active = True
    return {
        "active": True,
        "workspace": str(args.live_workspace),
        "processed_dir": str(args.processed_dir),
        "feature_root": str(args.feature_root),
        "feature_registry_path": str(args.feature_registry_path),
        **metadata,
    }


def _target_samples_available(samples_path: Path, target: pd.Timestamp) -> bool:
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples parquet: {samples_path}")
    available = set(pq.ParquetFile(samples_path).schema.names)
    if "target_trade_date" not in available:
        raise KeyError(f"samples parquet must include target_trade_date: {samples_path}")
    dates = pd.read_parquet(samples_path, columns=["target_trade_date"])
    normalized = pd.to_datetime(dates["target_trade_date"], errors="coerce").dt.normalize()
    return bool(normalized.eq(target).any())


def _link_processed_inputs(source_dir: Path, live_dir: Path) -> None:
    for source in source_dir.iterdir():
        if not source.is_file() or source.name in {"samples.parquet", "samples_preview.csv"}:
            continue
        target = live_dir / source.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())


def _ensure_basic_input(source_dir: Path) -> None:
    basic_path = source_dir / "basic.parquet"
    if basic_path.exists():
        return
    panel_path = source_dir / "panel.parquet"
    if not panel_path.exists():
        return
    columns = _existing_parquet_columns(panel_path, ("stock_code", "stock_name", "industry", "trade_date"))
    if "stock_code" not in columns or "industry" not in columns:
        return
    panel = pd.read_parquet(panel_path, columns=columns)
    panel["stock_code"] = panel["stock_code"].astype(str)
    if "trade_date" in panel.columns:
        panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
        panel = panel.sort_values(["stock_code", "trade_date"], kind="mergesort")
    basic_columns = [column for column in ("stock_code", "stock_name", "industry") if column in panel.columns]
    basic = panel.loc[:, basic_columns].dropna(subset=["stock_code"]).drop_duplicates("stock_code", keep="last")
    basic.to_parquet(basic_path, index=False)


def _write_live_samples(
    *,
    source_processed_dir: Path,
    live_processed_dir: Path,
    target: pd.Timestamp,
    context_start: pd.Timestamp,
) -> dict[str, Any]:
    source_samples_path = source_processed_dir / "samples.parquet"
    source_panel_path = source_processed_dir / "panel.parquet"
    if not source_panel_path.exists():
        raise FileNotFoundError(f"Missing panel parquet for live sample generation: {source_panel_path}")

    samples = pd.read_parquet(source_samples_path)
    _require_columns(samples, ("sample_id", "stock_code", "feature_asof_date", "target_trade_date"), "samples")
    samples = samples.copy()
    for column in ("feature_asof_date", "target_trade_date", "trade_date", "decision_ts", "label_start_date", "label_end_date"):
        if column in samples.columns:
            samples[column] = pd.to_datetime(samples[column], errors="coerce")
    official_feature_max = pd.to_datetime(samples["feature_asof_date"], errors="coerce").max()
    if pd.isna(official_feature_max):
        raise ValueError(f"No valid feature_asof_date found in {source_samples_path}")
    target_dates = pd.to_datetime(samples["target_trade_date"], errors="coerce").dt.normalize()
    historical = samples.loc[target_dates.ge(context_start) & target_dates.le(target)].copy()

    panel_columns = _existing_parquet_columns(source_panel_path, ("stock_code", "stock_name", "industry", "trade_date"))
    panel = pd.read_parquet(source_panel_path, columns=panel_columns)
    _require_columns(panel, ("stock_code", "trade_date"), "panel")
    panel = _attach_basic_fields_for_live_panel(panel, source_processed_dir / "basic.parquet")
    panel["stock_code"] = panel["stock_code"].astype(str)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.normalize()
    trade_dates = sorted(pd.Timestamp(item).normalize() for item in panel["trade_date"].dropna().unique())
    calendar = sorted(set(trade_dates) | {target})
    next_by_feature = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
    official_feature_day = pd.Timestamp(official_feature_max).normalize()
    feature_dates = [
        date
        for date in trade_dates
        if date > official_feature_day
        and date < target
        and next_by_feature.get(date) is not None
        and next_by_feature[date] <= target
    ]
    if not feature_dates:
        raise ValueError(
            "No live feature dates can be generated. "
            f"official_feature_max={official_feature_day.date()} target={target.date()}"
        )

    sample_columns = samples.columns.tolist()
    live_frames = [
        _live_rows_for_feature_date(panel, sample_columns, feature_date, next_by_feature[feature_date])
        for feature_date in feature_dates
    ]
    live = pd.concat(live_frames, ignore_index=True)
    output = pd.concat([historical.loc[:, sample_columns], live.loc[:, sample_columns]], ignore_index=True)
    output = _attach_basic_fields_for_live_panel(output, source_processed_dir / "basic.parquet")
    output["sample_id"] = output["sample_id"].astype(str)
    output = output.drop_duplicates("sample_id", keep="last")
    output = output.sort_values(["stock_code", "target_trade_date"], kind="mergesort").reset_index(drop=True)

    samples_output_path = live_processed_dir / "samples.parquet"
    output.to_parquet(samples_output_path, index=False)
    output.head(2000).to_csv(live_processed_dir / "samples_preview.csv", index=False)
    target_row_count = int(pd.to_datetime(output["target_trade_date"], errors="coerce").dt.normalize().eq(target).sum())
    manifest = {
        "target_date": target.date().isoformat(),
        "context_start": context_start.date().isoformat(),
        "source_samples_path": str(source_samples_path),
        "source_panel_path": str(source_panel_path),
        "official_feature_max": official_feature_day.date().isoformat(),
        "generated_feature_dates": [date.date().isoformat() for date in feature_dates],
        "row_count": int(len(output)),
        "generated_row_count": int(len(live)),
        "target_row_count": target_row_count,
        "samples_path": str(samples_output_path),
    }
    _write_json(live_processed_dir / "live_samples_manifest.json", manifest)
    if target_row_count <= 0:
        raise ValueError(f"Live samples did not create target rows for {target.date().isoformat()}")
    return manifest


def _live_rows_for_feature_date(
    panel: pd.DataFrame,
    sample_columns: list[str],
    feature_date: pd.Timestamp,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    day = panel.loc[panel["trade_date"].eq(feature_date)].copy()
    if day.empty:
        raise ValueError(f"No panel rows for live feature date: {feature_date.date().isoformat()}")
    rows = pd.DataFrame(index=day.index)
    rows["sample_id"] = day["stock_code"].astype(str) + "_" + target_date.strftime("%Y-%m-%d")
    rows["stock_code"] = day["stock_code"].astype(str)
    rows["stock_name"] = day["stock_name"].astype("string") if "stock_name" in day.columns else ""
    rows["industry"] = day["industry"].astype("string") if "industry" in day.columns else ""
    rows["feature_asof_date"] = feature_date
    rows["target_trade_date"] = target_date
    rows["trade_date"] = target_date
    rows["decision_ts"] = target_date.normalize() + pd.Timedelta(hours=9, minutes=25)
    rows["label_start_date"] = target_date
    rows["label_end_date"] = pd.NaT
    for column in sample_columns:
        if column not in rows.columns:
            rows[column] = pd.NA
    for column in [item for item in sample_columns if item.startswith("label_")]:
        rows[column] = pd.NA
    return rows.loc[:, sample_columns]


def _attach_basic_fields_for_live_panel(panel: pd.DataFrame, basic_path: Path) -> pd.DataFrame:
    if {"stock_name", "industry"}.issubset(panel.columns):
        return panel
    if not basic_path.exists():
        for column in ("stock_name", "industry"):
            if column not in panel.columns:
                panel[column] = ""
        return panel
    basic_columns = _existing_parquet_columns(basic_path, ("stock_code", "stock_name", "industry"))
    basic = pd.read_parquet(basic_path, columns=basic_columns)
    if "stock_code" not in basic.columns:
        return panel
    panel = panel.copy()
    panel["stock_code"] = panel["stock_code"].astype(str)
    basic["stock_code"] = basic["stock_code"].astype(str)
    merged = panel.merge(basic.drop_duplicates("stock_code"), on="stock_code", how="left", suffixes=("", "_basic"))
    for column in ("stock_name", "industry"):
        fallback = f"{column}_basic"
        if column not in merged.columns and fallback in merged.columns:
            merged[column] = merged[fallback]
        elif fallback in merged.columns:
            merged[column] = merged[column].fillna(merged[fallback])
        if fallback in merged.columns:
            merged = merged.drop(columns=[fallback])
        if column not in merged.columns:
            merged[column] = ""
    return merged


def _existing_parquet_columns(path: Path, columns: Sequence[str]) -> list[str]:
    available = set(pq.ParquetFile(path).schema.names)
    return [column for column in columns if column in available]


def _parse_context_start(value: str | None) -> pd.Timestamp:
    if not value:
        return pd.Timestamp(DEFAULT_LIVE_CONTEXT_START)
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --live-context-start: {value}")
    return pd.Timestamp(parsed).normalize()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {name} columns: {missing}")


def _resolve_target_date(args: argparse.Namespace) -> str:
    if args.target_date:
        return pd.Timestamp(args.target_date).date().isoformat()
    samples_path = args.processed_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples parquet: {samples_path}")
    dates = pd.read_parquet(samples_path, columns=["target_trade_date"])
    latest = pd.to_datetime(dates["target_trade_date"], errors="coerce").max()
    if pd.isna(latest):
        raise ValueError(f"No valid target_trade_date found in {samples_path}")
    return pd.Timestamp(latest).date().isoformat()


def _news_service_ready(base_url: str) -> bool:
    models_url = base_url.rstrip("/") + "/models"
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(models_url, timeout=2) as response:
            return 200 <= int(response.status) < 500
    except (OSError, URLError):
        return False


def _stop_started_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return None
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    return None


def _signals_path(args: argparse.Namespace, target_date: str) -> Path:
    model_root = args.model_root
    if model_root is None and args.model_config.exists():
        config = yaml.safe_load(args.model_config.read_text(encoding="utf-8")) or {}
        paths = config.get("paths", {}) if isinstance(config, dict) else {}
        if isinstance(paths, dict) and paths.get("model_root"):
            model_root = Path(paths["model_root"])
    model_root = model_root or DEFAULT_FINAL_MODEL_RUN
    return Path(model_root) / "competition_signals" / f"signals_{_date_label(target_date)}.csv"


def _default_feature_output_path(target_date: str) -> Path:
    return DATASETS_ROOT / "model_features" / "inference" / f"target_date={_date_label(target_date)}" / "features.parquet"


def _date_label(target_date: str) -> str:
    return pd.Timestamp(target_date).strftime("%Y%m%d")


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    python_bin = Path(args.python_bin).expanduser()
    if not python_bin.is_absolute():
        resolved_python_bin = shutil.which(str(args.python_bin))
        python_bin = Path(resolved_python_bin) if resolved_python_bin else python_bin
    if python_bin.is_absolute():
        env["PATH"] = str(python_bin.parent) + os.pathsep + env.get("PATH", "")
    tmp_dir = RUNTIME_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("TMPDIR", str(tmp_dir))
    env.setdefault("TMP", str(tmp_dir))
    env.setdefault("TEMP", str(tmp_dir))
    env.setdefault("AITRADER_ROOT", str(CODE_ROOT.parent))
    env.setdefault("AITRADER_DATA_ROOT", str(CODE_ROOT.parent / "data"))
    env.setdefault("AITRADER_DATASETS_ROOT", str(DATASETS_ROOT))
    env.setdefault("AITRADER_LOG_DIR", str(LOG_DIR))
    env.setdefault("AITRADER_RUNTIME_DIR", str(RUNTIME_DIR))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _initial_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "target_date": args.target_date,
        "paths": {
            "processed_dir": str(args.processed_dir),
            "factor_root": str(args.factor_root),
            "feature_root": str(args.feature_root),
            "evaluation_dir": str(args.evaluation_dir),
            "selected_features_path": str(args.selected_features_path),
            "model_config": str(args.model_config),
            "checkpoint": str(args.checkpoint),
        },
        "stages": [],
    }


def _refresh_summary_paths(summary: dict[str, Any], args: argparse.Namespace) -> None:
    paths = summary.setdefault("paths", {})
    if not isinstance(paths, dict):
        return
    paths.update(
        {
            "processed_dir": str(args.processed_dir),
            "feature_root": str(args.feature_root),
            "feature_registry_path": str(args.feature_registry_path),
        }
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, subprocess.Popen):
        return {"pid": value.pid}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
