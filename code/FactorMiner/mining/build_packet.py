from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq

from FactorMiner.core.registry import FactorRegistry, resolve_registry_path
from aitrader_paths import DATASETS_ROOT


DEFAULT_DATASETS_ROOT = DATASETS_ROOT
DEFAULT_PROCESSED_DIR = DEFAULT_DATASETS_ROOT / "processed"
DEFAULT_FACTOR_ROOT = DEFAULT_DATASETS_ROOT / "factors"
DEFAULT_FEATURE_ROOT = DEFAULT_DATASETS_ROOT / "features"
DEFAULT_EVALUATION_DIR = DEFAULT_FACTOR_ROOT / "evaluation" / "final"
DEFAULT_FEATURE_REGISTRY_PATH = DEFAULT_FEATURE_ROOT / "feature_registry.json"
DEFAULT_GPT_MINING_ROOT = DEFAULT_FACTOR_ROOT / "gpt_mining"
DEFAULT_OUTPUT_ROOT = DEFAULT_GPT_MINING_ROOT / "experiment"
PROFILES = {"research", "competition"}
REQUIRED_PACKET_FILES = (
    "gpt_inputs.txt",
    "00_context.md",
    "01_available_fields.json",
    "02_existing_factor_summary.csv",
    "03_selected_features_reviewed.json",
    "04_existing_news_features.md",
    "05_market_regime_instruction.md",
    "06_allowed_operators.md",
    "07_leakage_rules.md",
    "candidate_schema.json",
    "prompt_generate_candidates.md",
    "packet_manifest.json",
)
PACKET_UPLOAD_ORDER = (
    "prompt_generate_candidates.md",
    "00_context.md",
    "01_available_fields.json",
    "02_existing_factor_summary.csv",
    "03_selected_features_reviewed.json",
    "04_existing_news_features.md",
    "05_market_regime_instruction.md",
    "06_allowed_operators.md",
    "07_leakage_rules.md",
    "candidate_schema.json",
)
DEFAULT_MARKET_REGIME = (
    "2025-2026 A股风险偏好明显偏向科技成长主线，重点关注 AI、CPO、算力、半导体、"
    "数据中心、机器人、PCB、液冷、存储等方向。候选因子应捕捉主线行情中的动量延续、"
    "成交额扩张、资金流入、换手提升、估值容忍、拥挤回撤和新闻状态强化风险偏好。"
)
NEWS_KEYWORD_PATTERNS = {
    "ai_compute": r"(?i)\bAI\b|人工智能|大模型|AIGC|生成式|算力|智算|推理|训练|Agent",
    "cpo_optical": r"(?i)\bCPO\b|光模块|硅光|光通信|光芯片|800G|1\.6T|高速交换机",
    "semiconductor": r"半导体|芯片|晶圆|封测|先进封装|HBM|GPU|EDA|光刻|存储",
    "datacenter": r"数据中心|服务器|液冷|交换机|IDC|云计算|智算中心",
    "robotics": r"机器人|人形机器人|具身智能|减速器|伺服|机器视觉",
    "pcb": r"(?i)\bPCB\b|印制电路板|覆铜板|HDI|载板",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a small GPT factor-mining packet from local FactorMiner metadata.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTOR_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--feature-registry-path", type=Path, default=DEFAULT_FEATURE_REGISTRY_PATH)
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument("--price-path", type=Path, default=None)
    parser.add_argument("--metric-path", type=Path, default=None)
    parser.add_argument("--moneyflow-path", type=Path, default=None)
    parser.add_argument("--news-path", type=Path, default=None)
    parser.add_argument("--news-scores-path", type=Path, default=None)
    parser.add_argument("--candidate-features-path", type=Path, default=None)
    parser.add_argument("--factor-summary-path", type=Path, default=None)
    parser.add_argument("--quality-path", type=Path, default=None)
    parser.add_argument("--selected-features-path", type=Path, default=None)
    parser.add_argument("--correlation-clusters-path", type=Path, default=None)
    parser.add_argument("--correlation-conflicts-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--packet-dir", type=Path, default=None)
    parser.add_argument("--round-name", default=None)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="competition")
    parser.add_argument("--cutoff-date", default=None)
    parser.add_argument("--select-since", default=None)
    parser.add_argument("--select-until", default=None)
    parser.add_argument("--candidate-count", type=int, default=150)
    parser.add_argument("--max-factor-summary-rows", type=int, default=1500)
    parser.add_argument("--max-selected-features", type=int, default=1200)
    parser.add_argument("--max-news-feature-preview", type=int, default=220)
    parser.add_argument("--market-regime", default=DEFAULT_MARKET_REGIME)
    parser.add_argument("--skip-news-keyword-coverage", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_paths(args)
    result = build_mining_packet(args)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


def _resolve_paths(args: argparse.Namespace) -> None:
    args.samples_path = args.samples_path or args.processed_dir / "samples.parquet"
    args.price_path = args.price_path or args.processed_dir / "price.parquet"
    args.metric_path = args.metric_path or args.processed_dir / "metric.parquet"
    args.moneyflow_path = args.moneyflow_path or args.processed_dir / "moneyflow.parquet"
    args.news_path = args.news_path or args.processed_dir / "news.parquet"
    args.news_scores_path = args.news_scores_path or args.factor_root / "news_llm_scores.parquet"
    args.candidate_features_path = args.candidate_features_path or args.evaluation_dir / "candidate_features.csv"
    args.factor_summary_path = args.factor_summary_path or args.evaluation_dir / "factor_summary.csv"
    args.quality_path = args.quality_path or args.evaluation_dir / "sample_feature_quality.csv"
    reviewed = args.evaluation_dir / "selected_features_reviewed.json"
    args.selected_features_path = args.selected_features_path or (reviewed if reviewed.exists() else args.evaluation_dir / "selected_features.json")
    args.correlation_clusters_path = args.correlation_clusters_path or args.evaluation_dir / "correlation_clusters.csv"
    args.correlation_conflicts_path = args.correlation_conflicts_path or args.evaluation_dir / "correlation_conflicts.csv"
    if args.cutoff_date is None:
        args.cutoff_date = datetime.now(timezone.utc).date().isoformat()
    if args.round_name is None:
        safe_cutoff = str(args.cutoff_date).replace("-", "")
        args.round_name = f"round_regime_{safe_cutoff}_{_timestamp_compact()}"
    args.packet_dir = args.packet_dir or args.output_root / args.round_name / "packet"


def build_mining_packet(args: argparse.Namespace) -> dict[str, str]:
    packet_dir = Path(args.packet_dir)
    packet_dir.mkdir(parents=True, exist_ok=True)

    selected = _load_json(args.selected_features_path, "selected_features")
    candidate_summary = _build_existing_factor_summary(args)
    available_fields = _build_available_fields(args)
    news_features = _build_news_features_text(args, available_fields)
    context = _build_context(args, selected, candidate_summary)
    market_regime = _build_market_regime_instruction(args)
    allowed_operators = _allowed_operators_text()
    leakage_rules = _leakage_rules_text(args)
    candidate_schema = _candidate_schema(args)
    prompt = _build_prompt(args)

    _write_text(packet_dir / "00_context.md", context)
    _write_json(packet_dir / "01_available_fields.json", available_fields)
    candidate_summary.to_csv(packet_dir / "02_existing_factor_summary.csv", index=False)
    _write_json(packet_dir / "03_selected_features_reviewed.json", _trim_selected_features(selected, args.max_selected_features))
    _write_text(packet_dir / "04_existing_news_features.md", news_features)
    _write_text(packet_dir / "05_market_regime_instruction.md", market_regime)
    _write_text(packet_dir / "06_allowed_operators.md", allowed_operators)
    _write_text(packet_dir / "07_leakage_rules.md", leakage_rules)
    _write_json(packet_dir / "candidate_schema.json", candidate_schema)
    _write_text(packet_dir / "prompt_generate_candidates.md", prompt)

    manifest = _packet_manifest(args, packet_dir, candidate_summary, selected, available_fields)
    _write_json(packet_dir / "packet_manifest.json", manifest)
    _write_text(packet_dir / "gpt_inputs.txt", _gpt_inputs_text(args, packet_dir))

    validation = validate_packet(packet_dir)
    manifest["validation"] = validation
    _write_json(packet_dir / "packet_manifest.json", manifest)
    return {
        "packet_dir": str(packet_dir),
        "profile": str(args.profile),
        "cutoff_date": str(args.cutoff_date),
        "required_files": str(len(REQUIRED_PACKET_FILES)),
        "validation": "ok",
    }


def validate_packet(packet_dir: str | Path) -> dict[str, Any]:
    packet_dir = Path(packet_dir)
    missing = [name for name in REQUIRED_PACKET_FILES if not (packet_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Mining packet missing required files: {missing}")
    empty = [name for name in REQUIRED_PACKET_FILES if (packet_dir / name).stat().st_size == 0]
    if empty:
        raise ValueError(f"Mining packet has empty required files: {empty}")
    for name in ("01_available_fields.json", "03_selected_features_reviewed.json", "candidate_schema.json", "packet_manifest.json"):
        _load_json(packet_dir / name, name)
    existing_summary = pd.read_csv(packet_dir / "02_existing_factor_summary.csv")
    required_columns = {"factor_name", "block", "source", "category"}
    missing_columns = sorted(required_columns - set(existing_summary.columns))
    if missing_columns:
        raise ValueError(f"02_existing_factor_summary.csv missing columns: {missing_columns}")
    prompt = (packet_dir / "prompt_generate_candidates.md").read_text(encoding="utf-8")
    if "只输出 JSON" not in prompt or "candidate_schema.json" not in prompt:
        raise ValueError("prompt_generate_candidates.md is missing required output constraints")
    inputs_text = (packet_dir / "gpt_inputs.txt").read_text(encoding="utf-8")
    for name in PACKET_UPLOAD_ORDER:
        if str((packet_dir / name).resolve()) not in inputs_text:
            raise ValueError(f"gpt_inputs.txt missing absolute path for {name}")
    return {"status": "ok", "file_count": len(REQUIRED_PACKET_FILES), "summary_rows": int(len(existing_summary))}


def _build_available_fields(args: argparse.Namespace) -> dict[str, Any]:
    processed_paths = {
        "samples": args.samples_path,
        "price": args.price_path,
        "metric": args.metric_path,
        "moneyflow": args.moneyflow_path,
        "news": args.news_path,
        "news_llm_scores": args.news_scores_path,
    }
    processed = {name: _parquet_profile(path) for name, path in processed_paths.items()}
    feature_blocks = _feature_block_profiles(args.feature_registry_path)
    selected_features = _load_selected_names(args.selected_features_path)
    selected_set = set(selected_features)
    block_counts = []
    for block in feature_blocks:
        block_selected = [name for name in block.get("factor_names", []) if name in selected_set]
        block_counts.append(
            {
                "block": block.get("name"),
                "factor_count": block.get("factor_count"),
                "selected_count": len(block_selected),
                "selected_preview": block_selected[:25],
            }
        )
    return {
        "created_at": _utc_now(),
        "rule": "GPT may only use fields listed here or factor columns from registered feature blocks. It must not invent unavailable fields.",
        "processed_tables": processed,
        "feature_registry_path": str(Path(args.feature_registry_path).resolve()),
        "feature_blocks": feature_blocks,
        "selected_feature_block_counts": block_counts,
        "news_score_fields": [
            "sentiment_score",
            "impact_score",
            "risk_score",
            "relevance_score",
            "novelty_score",
            "event_type",
        ],
        "usable_news_sample_prefixes": ["news_market_", "news_stock_"],
    }


def _feature_block_profiles(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return []
    registry = FactorRegistry.load(registry_path)
    profiles = []
    for block in registry.blocks:
        manifest_path = resolve_registry_path(block.manifest_path, registry_path.parent)
        factor_path = resolve_registry_path(block.factor_path, registry_path.parent)
        manifest = _load_manifest_records(manifest_path)
        factor_names = [str(record.get("name")) for record in manifest if record.get("name")]
        categories = Counter(str(record.get("category", "")) for record in manifest)
        sources = Counter(str(record.get("source", "")) for record in manifest)
        windows = sorted({_normal_json_value(record.get("window")) for record in manifest if record.get("window") is not None}, key=str)
        profiles.append(
            {
                "name": block.name,
                "granularity": block.granularity,
                "factor_count": block.factor_count,
                "row_count": block.row_count,
                "factor_path": str(factor_path.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "factor_names": factor_names,
                "source_counts": dict(sources.most_common(20)),
                "category_counts": dict(categories.most_common(40)),
                "windows": windows,
                "preview": [
                    {
                        "name": record.get("name"),
                        "source": record.get("source"),
                        "category": record.get("category"),
                        "window": _normal_json_value(record.get("window")),
                        "expression": record.get("expression"),
                    }
                    for record in manifest[:20]
                ],
            }
        )
    return profiles


def _build_existing_factor_summary(args: argparse.Namespace) -> pd.DataFrame:
    candidates = _load_optional_csv(args.candidate_features_path)
    factor_summary = _load_optional_csv(args.factor_summary_path)
    quality = _load_optional_csv(args.quality_path)
    selected = set(_load_selected_names(args.selected_features_path))
    if not candidates.empty:
        summary = candidates.copy()
    elif not factor_summary.empty:
        summary = factor_summary.copy()
        summary["selected"] = summary["factor_name"].astype(str).isin(selected)
    else:
        summary = pd.DataFrame(columns=["block", "factor_name", "source", "category", "selected"])
    if "selected" in summary.columns:
        summary["auto_selected"] = summary["selected"]
    else:
        summary["auto_selected"] = summary["factor_name"].astype(str).isin(selected)
    summary["selected"] = summary["factor_name"].astype(str).isin(selected)
    if "quality_pass" not in summary.columns and not quality.empty:
        cols = [col for col in ("block", "factor_name", "quality_pass", "missing_rate", "constant_flag", "quality_flags") if col in quality.columns]
        if {"block", "factor_name"}.issubset(cols):
            summary = summary.merge(quality.loc[:, cols].drop_duplicates(["block", "factor_name"]), on=["block", "factor_name"], how="left")
    for column in ("source", "category", "selection_score", "rank_ic_mean", "rank_ic_ir", "coverage_mean", "group_spread_mean"):
        if column not in summary.columns:
            summary[column] = pd.NA
    keep_columns = [
        "block",
        "factor_name",
        "source",
        "category",
        "selected",
        "auto_selected",
        "selection_status",
        "final_reject_reason",
        "direction",
        "selection_score",
        "rank_ic_mean",
        "rank_ic_ir",
        "rank_ic_positive_rate",
        "coverage_mean",
        "group_spread_mean",
        "group_spread_positive_rate",
        "secondary_rank_ic_mean",
        "secondary_rank_ic_ir",
        "quality_pass",
        "missing_rate",
        "zero_rate",
        "constant_flag",
        "year_coverage_min",
        "cluster_id",
        "cluster_size",
        "high_corr_with",
        "corr_with_selected",
    ]
    existing = summary.loc[:, [col for col in keep_columns if col in summary.columns]].copy()
    existing["_sort_selected"] = existing["selected"].map(_truthy).astype(int)
    if "selection_score" in existing.columns:
        existing["_sort_score"] = pd.to_numeric(existing["selection_score"], errors="coerce").fillna(-1)
    else:
        existing["_sort_score"] = -1
    existing = existing.sort_values(["_sort_selected", "_sort_score", "factor_name"], ascending=[False, False, True])
    existing = existing.drop(columns=["_sort_selected", "_sort_score"])
    if args.max_factor_summary_rows and len(existing) > args.max_factor_summary_rows:
        selected_rows = existing.loc[existing["selected"].map(_truthy)].copy()
        remaining = existing.loc[~existing.index.isin(selected_rows.index)].head(max(0, args.max_factor_summary_rows - len(selected_rows)))
        existing = pd.concat([selected_rows, remaining], ignore_index=True).head(args.max_factor_summary_rows)
    return existing.reset_index(drop=True)


def _build_news_features_text(args: argparse.Namespace, available_fields: dict[str, Any]) -> str:
    feature_blocks = available_fields.get("feature_blocks", [])
    news_blocks = [block for block in feature_blocks if str(block.get("name", "")).startswith("news_")]
    news_score_profile = available_fields.get("processed_tables", {}).get("news_llm_scores", {})
    news_profile = available_fields.get("processed_tables", {}).get("news", {})
    coverage = [] if args.skip_news_keyword_coverage else _news_keyword_coverage(args.news_path)
    lines = [
        "# Existing News Features",
        "",
        "Use existing news scores and sample-aligned news factors only. Do not request new LLM scoring, new article summaries, or manual per-news judgement.",
        "",
        "## Raw News Table",
        "",
        f"- path: {Path(args.news_path).resolve()}",
        f"- exists: {news_profile.get('exists')}",
        f"- rows: {news_profile.get('row_count')}",
        f"- columns: {', '.join(news_profile.get('columns', []))}",
        "",
        "## Existing LLM Score Table",
        "",
        f"- path: {Path(args.news_scores_path).resolve()}",
        f"- exists: {news_score_profile.get('exists')}",
        f"- rows: {news_score_profile.get('row_count')}",
        "- reusable fields: sentiment_score, impact_score, risk_score, relevance_score, novelty_score, event_type",
        "- these scores are already computed; candidate formulas may aggregate them but must not ask to re-score news.",
        "",
        "## Sample-Aligned News Blocks",
        "",
    ]
    if not news_blocks:
        lines.append("- No news sample block found in feature registry.")
    for block in news_blocks:
        lines.extend(
            [
                f"### {block.get('name')}",
                "",
                f"- factor_count: {block.get('factor_count')}",
                f"- category_counts: {json.dumps(block.get('category_counts', {}), ensure_ascii=False)}",
                "- preview:",
            ]
        )
        for record in block.get("preview", [])[: args.max_news_feature_preview]:
            lines.append(
                f"  - {record.get('name')} | category={record.get('category')} | window={record.get('window')} | {record.get('expression')}"
            )
        lines.append("")
    lines.extend(["## Local Keyword Coverage Snapshot", ""])
    if coverage:
        lines.append("| theme_keyword_group | since | total_news | stock_news | market_news |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in coverage:
            lines.append(
                f"| {row['theme_keyword_group']} | {row['since']} | {row['total_news']} | {row['stock_news']} | {row['market_news']} |"
            )
    else:
        lines.append("- Not computed or news table unavailable.")
    lines.extend(
        [
            "",
            "## Constraints",
            "",
            "- Allowed: use existing news_market_* and news_stock_* sample factors.",
            "- Allowed: propose formula families that aggregate existing news score fields.",
            "- Not allowed: call Qwen/GPT again to classify, score, summarize, or relabel all news.",
            "- Not allowed: use post-cutoff competition-period news or returns.",
        ]
    )
    return "\n".join(lines)


def _news_keyword_coverage(news_path: Path) -> list[dict[str, Any]]:
    if not news_path.exists():
        return []
    try:
        frame = pd.read_parquet(news_path, columns=["trade_date", "matched_stock_count", "news_text"])
    except Exception:
        return []
    if frame.empty:
        return []
    trade_date = pd.to_datetime(frame["trade_date"], errors="coerce")
    text = frame["news_text"].fillna("").astype(str)
    matched_count = pd.to_numeric(frame["matched_stock_count"], errors="coerce").fillna(0)
    rows: list[dict[str, Any]] = []
    for group, pattern in NEWS_KEYWORD_PATTERNS.items():
        mask = text.str.contains(pattern, regex=True, na=False)
        for since in ("2024-01-01", "2025-01-01", "2026-01-01"):
            since_ts = pd.Timestamp(since)
            subset = mask & trade_date.ge(since_ts)
            rows.append(
                {
                    "theme_keyword_group": group,
                    "since": since,
                    "total_news": int(subset.sum()),
                    "stock_news": int((subset & matched_count.gt(0)).sum()),
                    "market_news": int((subset & matched_count.eq(0)).sum()),
                }
            )
    return rows


def _build_context(args: argparse.Namespace, selected: dict[str, Any], factor_summary: pd.DataFrame) -> str:
    selected_features = selected.get("selected_features", [])
    source_counts = factor_summary.loc[factor_summary.get("selected", pd.Series(dtype=bool)).map(_truthy), "source"].fillna("unknown").astype(str).value_counts().head(20).to_dict() if "source" in factor_summary else {}
    category_counts = factor_summary.loc[factor_summary.get("selected", pd.Series(dtype=bool)).map(_truthy), "category"].fillna("unknown").astype(str).value_counts().head(30).to_dict() if "category" in factor_summary else {}
    return "\n".join(
        [
            "# GPT Factor Mining Packet Context",
            "",
            f"- profile: {args.profile}",
            f"- cutoff_date: {args.cutoff_date}",
            f"- select_since: {args.select_since or selected.get('config', {}).get('since', 'unknown')}",
            f"- select_until: {args.select_until or selected.get('config', {}).get('until', 'unknown')}",
            "- target_labels: label_next_open_return, label_next_vwap_return",
            f"- existing_reviewed_feature_count: {len(selected_features) if isinstance(selected_features, list) else 'unknown'}",
            f"- existing_factor_summary_rows: {len(factor_summary)}",
            "",
            "## Purpose",
            "",
            "Generate regime-guided candidate factor formulas. GPT may use public market research before cutoff_date as hypothesis input, but every candidate must be computable from local fields listed in 01_available_fields.json.",
            "",
            "## Existing Selected Source Counts",
            "",
            "```json",
            json.dumps(source_counts, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Existing Selected Category Counts",
            "",
            "```json",
            json.dumps(category_counts, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Hard Boundary",
            "",
            "- GPT output is only a candidate proposal; local validation and backtest decide whether a candidate survives.",
            "- Do not use fields outside the packet.",
            "- Do not request news re-scoring.",
            "- Do not use future labels or post-cutoff market data.",
        ]
    )


def _build_market_regime_instruction(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Market Regime Instruction",
            "",
            f"profile: {args.profile}",
            f"cutoff_date: {args.cutoff_date}",
            "",
            "GPT should first reason about market structure and then generate candidate factor formulas.",
            "",
            "This is a mandatory web-research profile. GPT must browse public sources before generating candidates and must cite those sources in web_research_summary.sources.",
            "",
            "Research background to emphasize:",
            "",
            args.market_regime,
            "",
            "Candidate formulas should capture tradable structures, not literal theme labels:",
            "",
            "- momentum continuation after high turnover",
            "- amount/volume expansion with positive money flow",
            "- industry-relative strength",
            "- small/mid-cap elasticity under strong market state",
            "- high impact market news interacting with stock momentum or money flow",
            "- crowded short-term reversal after extreme volume, return, or turnover",
            "- valuation tolerance in growth regimes",
            "",
            "Do not require local theme membership tables unless they are explicitly listed as available fields.",
        ]
    )


def _allowed_operators_text() -> str:
    return "\n".join(
        [
            "# Allowed Operators",
            "",
            "Only use these operators in candidate formulas. Use past data only.",
            "",
            "- rolling_mean(x, window)",
            "- rolling_sum(x, window)",
            "- rolling_std(x, window)",
            "- rolling_min(x, window)",
            "- rolling_max(x, window)",
            "- delta(x, window)",
            "- pct_change(x, window)",
            "- return(close, window)",
            "- rank_cs(x)",
            "- zscore_cs(x)",
            "- winsorize(x)",
            "- industry_neutralize(x)",
            "- safe_div(x, y)",
            "- log1p(x)",
            "- abs(x)",
            "- sign(x)",
            "- x + y",
            "- x - y",
            "- x * y",
            "- x / y through safe_div(x, y)",
            "- interaction(x, y) as x * y after both inputs are locally computable",
            "",
            "Allowed windows: 3, 5, 10, 20, 60. Use window=1 only for already daily fields or existing 1d news sample factors.",
        ]
    )


def _leakage_rules_text(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Leakage Rules",
            "",
            f"- cutoff_date: {args.cutoff_date}",
            "- Features must be available at feature_asof_date / decision_ts.",
            "- Labels are future returns and must never be used inside formulas.",
            "- Rolling windows must look backward only.",
            "- Do not use target_trade_date returns when building same-row features.",
            "- News factors may use only news published by decision_ts and already aggregated into news_market_* or news_stock_* sample factors.",
            "- Do not use post-competition-start market outcomes to tune factor design.",
            "- Do not use a manually curated stock list unless a local theme_membership field/table is explicitly listed in 01_available_fields.json.",
            "- Do not call external APIs from local validation/materialization.",
        ]
    )


def _candidate_schema(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "GPTRegimeGuidedCandidateFactors",
        "type": "object",
        "required": ["web_research_summary", "candidates"],
        "additionalProperties": False,
        "properties": {
            "web_research_summary": {
                "type": "object",
                "required": ["status", "cutoff_date", "research_queries", "market_regime_summary", "factor_design_notes", "sources"],
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["completed", "web_unavailable"]},
                    "cutoff_date": {"type": "string"},
                    "research_queries": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 20},
                    "market_regime_summary": {"type": "string"},
                    "factor_design_notes": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 30},
                    "sources": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 40,
                        "items": {
                            "type": "object",
                            "required": ["source_id", "title", "url", "published_or_accessed_date", "evidence"],
                            "additionalProperties": False,
                            "properties": {
                                "source_id": {"type": "string", "pattern": "^S[0-9]{2,3}$"},
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "published_or_accessed_date": {"type": "string"},
                                "evidence": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "candidates": {
                "type": "array",
                "minItems": 0,
                "maxItems": int(args.candidate_count),
                "items": {
                    "type": "object",
                    "required": [
                        "factor_name",
                        "formula",
                        "inputs",
                        "windows",
                        "category",
                        "hypothesis",
                        "regime_link",
                        "expected_direction",
                        "leakage_risk",
                        "redundancy_risk",
                        "implementation_notes",
                        "priority",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "factor_name": {"type": "string", "pattern": "^[a-zA-Z][a-zA-Z0-9_]{3,120}$"},
                        "formula": {"type": "string"},
                        "inputs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "windows": {"type": "array", "items": {"type": "integer", "enum": [1, 3, 5, 10, 20, 60]}},
                        "category": {
                            "type": "string",
                            "enum": [
                                "regime_momentum",
                                "moneyflow",
                                "liquidity",
                                "valuation",
                                "industry_relative",
                                "news_state",
                                "interaction",
                                "reversal",
                                "risk_control",
                            ],
                        },
                        "hypothesis": {"type": "string"},
                        "regime_link": {"type": "string"},
                        "expected_direction": {"type": "integer", "enum": [-1, 0, 1]},
                        "leakage_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                        "redundancy_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                        "implementation_notes": {"type": "string"},
                        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                },
            },
        },
    }


def _build_prompt(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Generate Candidate Factors",
            "",
            "你是量化因子研究员。必须先联网调研 cutoff_date 之前公开市场信息，形成对 A 股市场结构、市场情绪和前沿因子设计的研究背景，再基于本 packet 生成可本地计算的候选因子。",
            "",
            f"本轮目标：生成最多 {args.candidate_count} 个候选因子。",
            "",
            "强制联网调研要求：",
            "",
            "- 生成候选前必须使用联网搜索/深度研究能力调研 cutoff_date 之前的公开资料。",
            "- 联网调研的目的不是给每个因子逐条找出处，而是把握 A 股阶段性市场结构、风险偏好、资金偏好、行业主线、拥挤交易和前沿量化因子设计思路。",
            "- 至少覆盖 A股 2025-2026 科技成长主线、AI/CPO/算力/半导体/数据中心/机器人/PCB/液冷/存储等方向，以及这些行情对应的成交、资金流、动量、估值、新闻状态和风险偏好特征。",
            "- 同时调研可迁移的前沿因子设计方向，例如 regime-aware factors、flow-momentum interaction、liquidity expansion、industry-relative strength、crowding/reversal、news-state gating 等。",
            "- 返回 JSON 顶层必须包含 web_research_summary。",
            "- web_research_summary.status 必须是 completed；如果当前会话没有联网能力，请返回 status=web_unavailable 且 candidates=[]，不要编造来源。",
            "- web_research_summary.sources 必须列出真实公开来源 URL，并用 source_id 标识。",
            "- 每个 candidate 必须填写 regime_link，说明该候选试图捕捉哪一种市场结构或交易行为；不要求每个候选逐条绑定来源。",
            "- 不要把联网得到的不可结构化信息直接当成本地字段；候选公式仍必须只用 packet 中列出的本地字段和算子。",
            "",
            "必须先阅读：",
            "",
            "1. 00_context.md",
            "2. 01_available_fields.json",
            "3. 02_existing_factor_summary.csv",
            "4. 03_selected_features_reviewed.json",
            "5. 04_existing_news_features.md",
            "6. 05_market_regime_instruction.md",
            "7. 06_allowed_operators.md",
            "8. 07_leakage_rules.md",
            "9. candidate_schema.json",
            "",
            "研究方向：",
            "",
            "- 2025-2026 A股科技成长主线：AI、CPO、算力、半导体、数据中心、机器人、PCB、液冷、存储等。",
            "- 不要求本地精确拆主题股票池；优先生成能捕捉这种市场状态的通用可计算公式。",
            "- 重点考虑价量、资金流、换手、估值、市值、行业相对强弱、已有新闻状态因子的组合。",
            "",
            "硬约束：",
            "",
            "- 只输出 JSON 对象，不能输出 Markdown、解释文字或代码块。",
            "- 输出必须符合 candidate_schema.json。",
            "- JSON 顶层必须是 {\"web_research_summary\": ..., \"candidates\": [...]}。",
            "- factor_name 必须唯一，且不能和 03_selected_features_reviewed.json 中已有因子重名。",
            "- 只能使用 01_available_fields.json 中列出的本地字段、特征列或 06_allowed_operators.md 中列出的算子。",
            "- 不允许要求重新新闻打分、重新新闻总结、人工逐条新闻判断。",
            "- 不允许创造本地不存在的主题股票池或外部字段。",
            "- 不允许使用未来收益、未来成交、未来新闻、post-cutoff 信息。",
            "- 候选公式应尽量避免与已有 alpha158 简单价量因子完全重复；如果重复风险高，必须写明原因。",
            "",
            "输出 JSON schema 见 candidate_schema.json。每个候选必须包含经济假设、regime_link、输入字段、窗口、预期方向、泄露风险和实现说明。",
        ]
    )


def _gpt_inputs_text(args: argparse.Namespace, packet_dir: Path) -> str:
    lines = [
        "# GPT Factor Mining Inputs",
        "",
        "把“建议上传”里的文件提供给 GPT5.5Pro 做候选因子生成。不要上传原始 parquet 大表。",
        "",
        f"profile: {args.profile}",
        f"cutoff_date: {args.cutoff_date}",
        f"packet_dir: {packet_dir.resolve()}",
        "",
        "## 建议上传",
        "",
    ]
    for name in PACKET_UPLOAD_ORDER:
        lines.append(f"{name}: {(packet_dir / name).resolve()}")
    lines.extend(
        [
            "",
            "## 不建议上传",
            "",
            f"samples.parquet: {Path(args.samples_path).resolve()}",
            f"price.parquet: {Path(args.price_path).resolve()}",
            f"metric.parquet: {Path(args.metric_path).resolve()}",
            f"moneyflow.parquet: {Path(args.moneyflow_path).resolve()}",
            f"news.parquet: {Path(args.news_path).resolve()}",
            f"news_llm_scores.parquet: {Path(args.news_scores_path).resolve()}",
            "",
            "## 使用方式",
            "",
            "1. 先上传 prompt_generate_candidates.md 和 candidate_schema.json。",
            "2. 再上传 context、available_fields、existing_factor_summary、selected_features、news、operators、leakage 文件。",
            "3. 要求 GPT 只返回 JSON。",
            "4. 要求 GPT 必须联网调研，并在 JSON 顶层返回 web_research_summary.sources。",
            "5. 将返回保存为本轮目录下的 gpt_response.json，供后续 Candidate Validation 使用。",
        ]
    )
    return "\n".join(lines)


def _packet_manifest(
    args: argparse.Namespace,
    packet_dir: Path,
    factor_summary: pd.DataFrame,
    selected: dict[str, Any],
    available_fields: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": "gpt_mining_packet_v1",
        "created_at": _utc_now(),
        "profile": args.profile,
        "cutoff_date": args.cutoff_date,
        "packet_dir": str(packet_dir.resolve()),
        "source_paths": {
            "samples": str(Path(args.samples_path).resolve()),
            "price": str(Path(args.price_path).resolve()),
            "metric": str(Path(args.metric_path).resolve()),
            "moneyflow": str(Path(args.moneyflow_path).resolve()),
            "news": str(Path(args.news_path).resolve()),
            "news_scores": str(Path(args.news_scores_path).resolve()),
            "feature_registry": str(Path(args.feature_registry_path).resolve()),
            "evaluation_dir": str(Path(args.evaluation_dir).resolve()),
            "selected_features": str(Path(args.selected_features_path).resolve()),
        },
        "file_order": list(PACKET_UPLOAD_ORDER),
        "selected_feature_count": len(selected.get("selected_features", [])) if isinstance(selected.get("selected_features"), list) else None,
        "existing_factor_summary_rows": int(len(factor_summary)),
        "feature_block_count": len(available_fields.get("feature_blocks", [])),
        "candidate_count_request": int(args.candidate_count),
    }


def _parquet_profile(path: Path) -> dict[str, Any]:
    path = Path(path)
    profile: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.exists(),
    }
    if not path.exists():
        return profile
    parquet = pq.ParquetFile(path)
    profile.update(
        {
            "row_count": int(parquet.metadata.num_rows),
            "column_count": int(parquet.metadata.num_columns),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
            "columns": parquet.schema.names,
        }
    )
    return profile


def _load_manifest_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = _load_json(path, "manifest")
    if not isinstance(data, list):
        return []
    return [record for record in data if isinstance(record, dict)]


def _load_selected_names(path: Path) -> list[str]:
    data = _load_json(path, "selected_features")
    names = data.get("selected_features", []) if isinstance(data, dict) else []
    return [str(name) for name in names] if isinstance(names, list) else []


def _trim_selected_features(selected: dict[str, Any], max_features: int) -> dict[str, Any]:
    trimmed = dict(selected)
    names = trimmed.get("selected_features")
    if isinstance(names, list) and max_features and len(names) > max_features:
        trimmed["selected_features"] = names[:max_features]
        trimmed["truncated"] = True
        trimmed["original_selected_feature_count"] = len(names)
    return trimmed


def _load_json(path: Path, name: str) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, str, bytes)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normal_json_value(value: Any) -> Any:
    if value is None:
        return None
    return _jsonable(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _timestamp_compact() -> str:
    return datetime.now(timezone.utc).strftime("%H%M%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
