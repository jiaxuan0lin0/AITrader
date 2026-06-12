from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from aitrader_paths import DATASETS_ROOT


DEFAULT_EVALUATION_DIR = DATASETS_ROOT / "factors" / "evaluation" / "experiment" / "ad_hoc"
ALLOWED_REVIEW_FLAGS = {
    "state_factor",
    "interaction_candidate",
    "watchlist",
    "redundancy_risk",
    "sparse_event_signal",
    "manual_review",
    "other",
}
REVIEW_PROFILES = {"research", "competition"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or apply a manual ChatGPT review of selected factors.")
    parser.add_argument("--selected-features-path", type=Path, default=None)
    parser.add_argument("--candidate-features-path", type=Path, default=None)
    parser.add_argument("--review-packet-path", type=Path, default=None)
    parser.add_argument("--correlation-clusters-path", type=Path, default=None)
    parser.add_argument("--correlation-conflicts-path", type=Path, default=None)
    parser.add_argument("--response-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--response-template-path", type=Path, default=None)
    parser.add_argument("--review-inputs-path", type=Path, default=None)
    parser.add_argument("--reviewed-json-path", type=Path, default=None)
    parser.add_argument("--reviewed-csv-path", type=Path, default=None)
    parser.add_argument("--audit-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--review-profile", choices=sorted(REVIEW_PROFILES), default="research")
    parser.add_argument("--prepare", action="store_true", help="Generate review_prompt.md and a response template.")
    parser.add_argument("--apply", action="store_true", help="Apply --response-path to selected_features.json.")
    parser.add_argument("--max-selected-preview", type=int, default=250)
    parser.add_argument("--max-rejected-preview", type=int, default=150)
    parser.add_argument("--max-borderline-preview", type=int, default=120)
    parser.add_argument("--max-cluster-preview", type=int, default=80)
    parser.add_argument("--allow-quality-failed-add-back", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_paths(args)
    if not args.prepare and not args.apply:
        args.prepare = True
    outputs: dict[str, Any] = {}
    if args.prepare:
        outputs.update(prepare_review(args))
    if args.apply:
        outputs.update(apply_review(args))
    print(" ".join(f"{key}={value}" for key, value in outputs.items()))
    return 0


def _resolve_paths(args: argparse.Namespace) -> None:
    args.selected_features_path = args.selected_features_path or args.output_dir / "selected_features.json"
    args.candidate_features_path = args.candidate_features_path or args.output_dir / "candidate_features.csv"
    args.review_packet_path = args.review_packet_path or args.output_dir / "review_packet.json"
    args.correlation_clusters_path = args.correlation_clusters_path or args.output_dir / "correlation_clusters.csv"
    args.correlation_conflicts_path = args.correlation_conflicts_path or args.output_dir / "correlation_conflicts.csv"
    args.prompt_path = args.prompt_path or args.output_dir / "review_prompt.md"
    args.response_template_path = args.response_template_path or args.output_dir / "review_response_template.json"
    args.review_inputs_path = args.review_inputs_path or args.output_dir / "review_inputs.txt"
    args.reviewed_json_path = args.reviewed_json_path or args.output_dir / "selected_features_reviewed.json"
    args.reviewed_csv_path = args.reviewed_csv_path or args.output_dir / "selected_features_reviewed.csv"
    args.audit_path = args.audit_path or args.output_dir / "selection_review_audit.csv"
    args.report_path = args.report_path or args.output_dir / "selection_review_report.md"


def prepare_review(args: argparse.Namespace) -> dict[str, str]:
    selected_json = _load_json(args.selected_features_path, "selected_features")
    candidates = _load_csv(args.candidate_features_path, "candidate_features")
    _validate_selected_features_known(selected_json, candidates)
    review_packet = _load_json(args.review_packet_path, "review_packet")
    clusters = _load_optional_csv(args.correlation_clusters_path)
    conflicts = _load_optional_csv(args.correlation_conflicts_path)

    prompt_path = args.prompt_path
    template_path = args.response_template_path
    review_inputs_path = getattr(args, "review_inputs_path", None) or args.output_dir / "review_inputs.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    review_inputs_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(_build_review_prompt(args, selected_json, candidates, review_packet, clusters, conflicts), encoding="utf-8")
    template_path.write_text(json.dumps(_response_template(), ensure_ascii=False, indent=2), encoding="utf-8")
    review_inputs_path.write_text(_build_review_inputs_text(args, review_inputs_path), encoding="utf-8")
    return {"review_prompt": str(prompt_path), "response_template": str(template_path), "review_inputs": str(review_inputs_path)}


def apply_review(args: argparse.Namespace) -> dict[str, str]:
    if args.response_path is None:
        raise ValueError("--response-path is required when --apply is used")
    selected_json = _load_json(args.selected_features_path, "selected_features")
    candidates = _load_csv(args.candidate_features_path, "candidate_features")
    _validate_selected_features_known(selected_json, candidates)
    response = _normalize_review_response(_load_json(args.response_path, "review_response"))
    reviewed_json_path = args.reviewed_json_path
    reviewed_csv_path = args.reviewed_csv_path
    audit_path = args.audit_path
    report_path = args.report_path

    selected_auto = set(map(str, selected_json.get("selected_features", [])))
    candidate_by_name = candidates.drop_duplicates("factor_name", keep="first").set_index("factor_name", drop=False)
    _validate_response(response, selected_auto, candidate_by_name, args)

    audit_records: list[dict[str, Any]] = []
    final_selected = set(selected_auto)
    review_reason_by_factor: dict[str, str] = {}
    review_action_by_factor: dict[str, str] = {factor: "kept_auto" for factor in final_selected}
    flags_by_factor: dict[str, list[str]] = {}

    for item in response["remove"]:
        factor_name = item["factor_name"]
        reason = item["reason"]
        final_selected.discard(factor_name)
        review_reason_by_factor[factor_name] = reason
        review_action_by_factor[factor_name] = "removed"
        audit_records.append(_audit("remove", factor_name, True, reason, "Removed from automatic selection."))

    for item in response["add_back"]:
        factor_name = item["factor_name"]
        reason = item["reason"]
        final_selected.add(factor_name)
        review_reason_by_factor[factor_name] = reason
        review_action_by_factor[factor_name] = "added_back"
        audit_records.append(_audit("add_back", factor_name, True, reason, "Added to reviewed selection."))

    for item in response["flags"]:
        factor_name = item["factor_name"]
        flag = item["flag"]
        reason = item["reason"]
        flags_by_factor.setdefault(factor_name, []).append(flag)
        audit_records.append(_audit(f"flag:{flag}", factor_name, factor_name in final_selected, reason, "Flag recorded."))

    for note in response["global_notes"]:
        audit_records.append(_audit("global_note", "", True, str(note), "Global review note."))

    reviewed_rows = _reviewed_rows(candidates, final_selected, review_action_by_factor, review_reason_by_factor, flags_by_factor)
    reviewed_json = _reviewed_json(selected_json, reviewed_rows, response, args.response_path, args.review_profile)

    reviewed_json_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed_json_path.write_text(json.dumps(_jsonable(reviewed_json), ensure_ascii=False, indent=2), encoding="utf-8")
    reviewed_rows.to_csv(reviewed_csv_path, index=False)
    pd.DataFrame(audit_records).to_csv(audit_path, index=False)
    report_path.write_text(_review_report(reviewed_json, audit_records), encoding="utf-8")

    return {
        "reviewed_json": str(reviewed_json_path),
        "reviewed_csv": str(reviewed_csv_path),
        "audit": str(audit_path),
        "report": str(report_path),
        "reviewed_count": str(len(final_selected)),
    }


def _build_review_prompt(
    args: argparse.Namespace,
    selected_json: dict[str, Any],
    candidates: pd.DataFrame,
    review_packet: dict[str, Any],
    clusters: pd.DataFrame,
    conflicts: pd.DataFrame,
) -> str:
    selected_names = set(map(str, selected_json.get("selected_features", [])))
    selected = candidates.loc[candidates["factor_name"].astype(str).isin(selected_names)].copy()
    rejected = candidates.loc[~candidates["factor_name"].astype(str).isin(selected_names)].copy()
    borderline = candidates.loc[candidates.get("selection_status", pd.Series(index=candidates.index, dtype=object)).eq("borderline")].copy()
    high_score_rejected = rejected.sort_values("selection_score", ascending=False).head(args.max_rejected_preview)
    selected_preview = selected.sort_values("selection_score", ascending=False).head(args.max_selected_preview)
    borderline_preview = borderline.sort_values("selection_score", ascending=False).head(args.max_borderline_preview)
    cluster_preview = _cluster_prompt_records(clusters, args.max_cluster_preview)
    source_counts = selected["source"].fillna("").astype(str).value_counts().to_dict() if "source" in selected else {}
    category_counts = selected["category"].fillna("").astype(str).value_counts().head(30).to_dict() if "category" in selected else {}
    payload = {
        "review_profile": _review_profile_payload(args, selected_json),
        "selection_config": selected_json.get("config", {}),
        "summary_counts": review_packet.get("summary_counts", {}),
        "auto_selected_count": len(selected_names),
        "auto_selected_source_counts": source_counts,
        "auto_selected_top_category_counts": category_counts,
        "auto_selected_preview": _records(selected_preview, _review_columns()),
        "high_score_rejected_preview": _records(high_score_rejected, _review_columns(extra=("final_reject_reason", "high_corr_with", "corr_with_selected"))),
        "borderline_preview": _records(borderline_preview, _review_columns(extra=("reject_reason",))),
        "correlation_cluster_preview": cluster_preview,
        "correlation_conflict_count": int(len(conflicts)),
    }
    return "\n".join(
        [
            "# Factor Selection Review Prompt",
            "",
            "你是一个量化因子研究审查员。请审查自动筛选出的因子列表，并只输出符合下方 schema 的 JSON。",
            "",
            "目标：在不重新计算指标的前提下，基于已给出的质量、IC/RankIC、分组收益、相关性聚类和因子语义，给出可执行的修改建议。",
            "",
            *_review_profile_prompt_lines(args, selected_json),
            "",
            "硬约束：",
            "- 不要添加材料中不存在的因子。",
            "- 不要添加 quality_failed 或 constant 因子，除非你非常明确地写出 override 理由；当前工具默认会拒绝这种 add_back。",
            "- remove/add_back/flags 中每一项必须有 reason。",
            "- 不要输出 Markdown，不要解释过程，只输出 JSON。",
            "- remove 表示从自动 selected_features 中删除。",
            "- add_back 表示从 candidate_features/rejected_features 中加回 reviewed 版本。",
            "- flags 只用于标注，不自动删除或添加。",
            "- flags.flag 只能使用以下枚举：state_factor、sparse_event_signal、interaction_candidate、watchlist、redundancy_risk、manual_review、other。",
            "- 不要自造 flag 名称；例如稀疏事件因子必须写 sparse_event_signal，不要写 sparse_event_factor。",
            "- 如果 rejected 因子本身应该进入下游模型，必须写入 add_back；只写 flags 不会进入最终 selected_features。",
            "- 对 market-level/news state 因子，如果你判断应作为状态变量给模型使用，请同时 add_back 并在 flags 中标记 state_factor。",
            "",
            "文件优先级和关系：",
            "- 一级依据：review_prompt.md。它是压缩后的审查入口，先用来理解自动筛选结果、主要风险和输出 schema。",
            "- 建议上传：review_prompt.md、selected_features.json、candidate_features.csv、factor_summary.csv、sample_feature_quality.csv、review_packet.json、correlation_clusters.csv、correlation_conflicts.csv。",
            "- 当前自动入选清单：selected_features.json 是机器可读的最终自动清单；selected_features.csv 是同一清单的指标明细版。",
            "- 全量候选工作台：candidate_features.csv 是逐因子的主审查表，已合并质量、主标签 RankIC/IR/覆盖率、选择状态、得分和最终拒绝原因。",
            "- 补充依据：factor_summary.csv 提供双标签完整单因子明细；sample_feature_quality.csv 提供独立质量表；review_packet.json 提供压缩审查包的机器可读原始版本。",
            "- 不建议上传：selected_features.csv 和 rejected_features.csv 是 candidate_features.csv 的拆分子集，会增加重复上下文。",
            "- 冗余关系依据：correlation_clusters.csv 给出高相关簇和代表因子，correlation_conflicts.csv 给出具体两两高相关冲突。",
            "- 决策顺序建议：先看质量底线，再看单因子效果，再看相关性簇内代表选择，最后结合因子语义和新闻/状态因子的特殊性做 remove/add_back/flags。",
            "",
            "建议关注：",
            "- 同一高相关簇中代表因子是否选得合理。",
            "- 是否存在同一 source/category 保留过多的问题。",
            "- 新闻稀疏因子是否可能是有经济意义的事件信号。",
            "- 市场级新闻因子如果质量通过且能作为模型状态输入，应 add_back 并标记 state_factor，而不是只 flag。",
            "- 负向因子不是坏因子，只要方向稳定、解释合理即可保留。",
            "",
            "返回 JSON schema：",
            "```json",
            json.dumps(_response_template(), ensure_ascii=False, indent=2),
            "```",
            "",
            "审查材料：",
            "```json",
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
            "```",
        ]
    )


def _build_review_inputs_text(args: argparse.Namespace, review_inputs_path: Path) -> str:
    upload_records = [
        ("review_prompt.md", args.prompt_path),
        ("selected_features.json", args.selected_features_path),
        ("candidate_features.csv", args.candidate_features_path),
        ("factor_summary.csv", args.output_dir / "factor_summary.csv"),
        ("sample_feature_quality.csv", args.output_dir / "sample_feature_quality.csv"),
        ("review_packet.json", args.review_packet_path),
        ("correlation_clusters.csv", args.correlation_clusters_path),
        ("correlation_conflicts.csv", args.correlation_conflicts_path),
    ]
    skip_records = [
        ("selected_features.csv", args.output_dir / "selected_features.csv"),
        ("rejected_features.csv", args.output_dir / "rejected_features.csv"),
    ]
    lines = [
        "# Factor Selection AI Review Inputs",
        "",
        "把“建议上传”里的文件提供给 AI 做因子复核。“不建议上传”里的文件是重复拆分表，默认不要上传。",
        "",
        *_review_profile_input_lines(args),
        "",
        f"review_inputs.txt: {_abs_path(review_inputs_path)}",
        "",
        "## 建议上传",
        "",
    ]
    for label, path in upload_records:
        lines.append(f"{label}: {_abs_path(path)}")
    lines.extend(["", "## 不建议上传", ""])
    for label, path in skip_records:
        lines.append(f"{label}: {_abs_path(path)}")
    lines.extend(
        [
            "",
            "## 文件关系和优先级",
            "",
            "1. 先读 review_prompt.md，理解输出 schema、硬约束和审查目标。",
            "2. 用 selected_features.json 确认当前自动入选清单。",
            "3. 用 candidate_features.csv 作为主审查表。它已包含 selected/rejected、质量字段、主标签单因子指标、selection_score、final_reject_reason、high_corr_with。",
            "4. 用 correlation_clusters.csv 看簇内代表因子是否合理；用 correlation_conflicts.csv 查具体两两高相关关系。",
            "5. 用 factor_summary.csv 复核第二标签完整表现和单因子明细。",
            "6. 用 sample_feature_quality.csv 独立核查缺失、极端值、常数列、质量失败原因。",
            "7. 用 review_packet.json 读取压缩审查包的机器可读原始版本。",
            "8. 不上传 selected_features.csv 和 rejected_features.csv；它们是 candidate_features.csv 的拆分子集。",
            "",
            "AI 返回 JSON 后建议保存到:",
            f"review_response.json: {_abs_path(args.output_dir / 'review_response.json')}",
            "",
            "应用复核结果命令:",
            "python3 -m FactorMiner.evaluation.review_selection \\",
            f"  --apply --output-dir {_abs_path(args.output_dir)} \\",
            f"  --response-path {_abs_path(args.output_dir / 'review_response.json')}",
            "",
        ]
    )
    return "\n".join(lines)


def _abs_path(path: Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _review_profile_payload(args: argparse.Namespace, selected_json: dict[str, Any]) -> dict[str, Any]:
    profile = str(getattr(args, "review_profile", "research"))
    config = dict(selected_json.get("config", {}))
    since = config.get("since")
    until = config.get("until")
    if profile == "competition":
        purpose = "competition_final"
        instruction = "用于正式比赛最终因子清单。允许使用 cutoff 及以前已经可见的历史特征和收益评价，目标是预测 cutoff 之后未知样本。"
        emphasis = "可以更关注 cutoff 前最近市场结构和 2025/2026 近期行情，但不能使用 cutoff 之后的信息。"
    else:
        purpose = "research_validation"
        instruction = "用于研究验证和方法评估。只能使用训练窗口内的标签评价，禁止根据验证集或测试集表现调整因子。"
        emphasis = "更重视稳健性、跨年份一致性和不过拟合；不要为了贴合验证/测试行情修改因子。"
    return {
        "profile": profile,
        "purpose": purpose,
        "select_since": since,
        "select_until": until,
        "instruction": instruction,
        "emphasis": emphasis,
    }


def _review_profile_prompt_lines(args: argparse.Namespace, selected_json: dict[str, Any]) -> list[str]:
    payload = _review_profile_payload(args, selected_json)
    lines = [
        "本次复核版本：",
        f"- profile: {payload['profile']}",
        f"- purpose: {payload['purpose']}",
        f"- 可用标签/筛选窗口: {payload.get('select_since')} 到 {payload.get('select_until')}",
        f"- {payload['instruction']}",
        f"- {payload['emphasis']}",
    ]
    if payload["profile"] == "competition":
        lines.append("- 如果因子在最近市场结构中更有意义，且 cutoff 前证据充分，可以在质量通过的前提下更积极保留或 add_back。")
    else:
        lines.append("- 不能因为验证期或测试期表现做事后筛选；review 只应依据本训练窗口材料。")
    return lines


def _review_profile_input_lines(args: argparse.Namespace) -> list[str]:
    profile = str(getattr(args, "review_profile", "research"))
    if profile == "competition":
        return [
            "review_profile: competition",
            "用途: 正式比赛最终复核。允许使用筛选截止日及以前的历史材料，目标是预测截止日之后未知样本。",
        ]
    return [
        "review_profile: research",
        "用途: 研究验证复核。只使用训练窗口材料，不根据验证/测试表现调整因子。",
    ]


def _validate_response(
    response: dict[str, Any],
    selected_auto: set[str],
    candidate_by_name: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    known = set(candidate_by_name.index.astype(str))
    for key in ("remove", "add_back"):
        seen: set[str] = set()
        for item in response[key]:
            factor_name = item["factor_name"]
            if factor_name not in known:
                raise ValueError(f"Unknown factor in review {key}: {factor_name}")
            if factor_name in seen:
                raise ValueError(f"Duplicate factor in review {key}: {factor_name}")
            seen.add(factor_name)
            if not str(item.get("reason", "")).strip():
                raise ValueError(f"Review {key} item must include reason: {factor_name}")

    seen_flags: set[tuple[str, str]] = set()
    for item in response["flags"]:
        factor_name = item["factor_name"]
        flag = item["flag"]
        if factor_name not in known:
            raise ValueError(f"Unknown factor in review flags: {factor_name}")
        if flag not in ALLOWED_REVIEW_FLAGS:
            raise ValueError(f"Unsupported review flag for {factor_name}: {flag}")
        if (factor_name, flag) in seen_flags:
            raise ValueError(f"Duplicate review flag for {factor_name}: {flag}")
        seen_flags.add((factor_name, flag))
        if not str(item.get("reason", "")).strip():
            raise ValueError(f"Review flags item must include reason: {factor_name}")

    for item in response["remove"]:
        if item["factor_name"] not in selected_auto:
            raise ValueError(f"Cannot remove a factor that is not in selected_features: {item['factor_name']}")

    for item in response["add_back"]:
        factor_name = item["factor_name"]
        if factor_name in selected_auto:
            raise ValueError(f"Cannot add_back an already selected factor: {factor_name}")
        row = candidate_by_name.loc[factor_name]
        if not args.allow_quality_failed_add_back and (not _as_bool_value(row.get("quality_pass")) or _as_bool_value(row.get("constant_flag"))):
            raise ValueError(f"Cannot add_back quality_failed/constant factor without override: {factor_name}")


def _reviewed_rows(
    candidates: pd.DataFrame,
    final_selected: set[str],
    review_action_by_factor: dict[str, str],
    review_reason_by_factor: dict[str, str],
    flags_by_factor: dict[str, list[str]],
) -> pd.DataFrame:
    rows = candidates.loc[candidates["factor_name"].astype(str).isin(final_selected)].copy()
    rows["review_action"] = rows["factor_name"].map(review_action_by_factor).fillna("kept_auto")
    rows["review_reason"] = rows["factor_name"].map(review_reason_by_factor).fillna("")
    rows["review_flags"] = rows["factor_name"].map(lambda name: ";".join(flags_by_factor.get(str(name), [])))
    return rows.sort_values(["block", "factor_name"]).reset_index(drop=True)


def _reviewed_json(
    selected_json: dict[str, Any],
    reviewed_rows: pd.DataFrame,
    response: dict[str, Any],
    response_path: Path,
    review_profile: str = "research",
) -> dict[str, Any]:
    blocks: dict[str, list[str]] = {}
    for block, group in reviewed_rows.groupby("block", sort=True):
        blocks[str(block)] = group.sort_values("factor_name")["factor_name"].astype(str).tolist()
    selected_features = reviewed_rows.sort_values(["block", "factor_name"])["factor_name"].astype(str).tolist()
    directions = {
        str(row["factor_name"]): int(row["direction"])
        for _, row in reviewed_rows.iterrows()
        if "direction" in reviewed_rows.columns and pd.notna(row.get("direction"))
    }
    flags = {
        str(row["factor_name"]): str(row["review_flags"]).split(";")
        for _, row in reviewed_rows.iterrows()
        if str(row.get("review_flags", "")).strip()
    }
    return {
        "version": f"reviewed_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "selection_mode": "manual_chatgpt_review",
        "base_selection_version": selected_json.get("version"),
        "review_profile": review_profile,
        "primary_label": selected_json.get("primary_label"),
        "selected_features": selected_features,
        "directions": directions,
        "blocks": blocks,
        "review_flags": flags,
        "config": selected_json.get("config", {}),
        "review_response_path": str(response_path),
        "review_global_notes": response.get("global_notes", []),
    }


def _review_report(reviewed_json: dict[str, Any], audit_records: list[dict[str, Any]]) -> str:
    lines = [
        "# Selection Review Report",
        "",
        f"Reviewed selected feature count: {len(reviewed_json['selected_features'])}",
        f"Base selection version: {reviewed_json.get('base_selection_version')}",
        f"Review profile: {reviewed_json.get('review_profile')}",
        "",
        "## Changes",
        "",
    ]
    if not audit_records:
        lines.append("No review changes were applied.")
    else:
        for record in audit_records:
            factor = record["factor_name"] or "(global)"
            lines.append(f"- {record['action']} `{factor}`: {record['reason']}")
    if reviewed_json.get("review_global_notes"):
        lines.extend(["", "## Global Notes", ""])
        for note in reviewed_json["review_global_notes"]:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _normalize_review_response(data: dict[str, Any]) -> dict[str, Any]:
    remove = data.get("remove", data.get("removed", []))
    add_back = data.get("add_back", data.get("added_back", []))
    flags = data.get("flags", [])
    global_notes = data.get("global_notes", data.get("notes", []))
    normalized = {
        "remove": _normalize_items(remove, require_flag=False),
        "add_back": _normalize_items(add_back, require_flag=False),
        "flags": _normalize_items(flags, require_flag=True),
        "global_notes": [str(note) for note in global_notes],
    }
    return normalized


def _normalize_items(items: Any, require_flag: bool) -> list[dict[str, str]]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("Review response remove/add_back/flags must be lists")
    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Review response items must be objects")
        factor_name = str(item.get("factor_name", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not factor_name:
            raise ValueError("Review response item missing factor_name")
        record = {"factor_name": factor_name, "reason": reason}
        if require_flag:
            record["flag"] = _normalize_review_flag(str(item.get("flag", "")).strip())
            if not record["flag"]:
                raise ValueError(f"Review flag missing for {factor_name}")
        normalized.append(record)
    return normalized


def _normalize_review_flag(value: str) -> str:
    aliases = {
        "sparse_event_factor": "sparse_event_signal",
        "event_factor": "sparse_event_signal",
        "event_signal": "sparse_event_signal",
        "market_state": "state_factor",
        "regime_factor": "state_factor",
        "state": "state_factor",
    }
    return aliases.get(value, value)


def _response_template() -> dict[str, Any]:
    return {
        "remove": [
            {
                "factor_name": "factor_to_remove",
                "reason": "Why it should be removed from the automatic selection.",
            }
        ],
        "add_back": [
            {
                "factor_name": "factor_to_add_back",
                "reason": "Why it should be added despite not being selected automatically.",
            }
        ],
        "flags": [
            {
                "factor_name": "factor_to_flag",
                "flag": "state_factor",
                "reason": "Why this flag is useful.",
            }
        ],
        "allowed_flags": sorted(ALLOWED_REVIEW_FLAGS),
        "global_notes": ["Short overall review notes."],
    }


def _audit(action: str, factor_name: str, applied: bool, reason: str, detail: str) -> dict[str, Any]:
    return {
        "action": action,
        "factor_name": factor_name,
        "applied": applied,
        "reason": reason,
        "detail": detail,
    }


def _cluster_prompt_records(clusters: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if clusters.empty:
        return []
    records: list[dict[str, Any]] = []
    for cluster_id, group in clusters.groupby("cluster_id", sort=True):
        if len(group) <= 1:
            continue
        group = group.sort_values("score", ascending=False)
        representative_mask = group["is_representative"].map(_as_bool_value)
        records.append(
            {
                "cluster_id": cluster_id,
                "representative": str(group.loc[representative_mask, "factor_name"].iloc[0])
                if representative_mask.any()
                else str(group["factor_name"].iloc[0]),
                "members": _records(
                    group,
                    ("factor_name", "score", "rank_ic_mean", "rank_ic_ir", "coverage_mean", "source", "category"),
                    limit=25,
                ),
            }
        )
        if len(records) >= limit:
            break
    return records


def _review_columns(extra: Sequence[str] = ()) -> tuple[str, ...]:
    return (
        "factor_name",
        "block",
        "source",
        "category",
        "selection_score",
        "direction",
        "rank_ic_mean",
        "rank_ic_ir",
        "rank_ic_day_count",
        "coverage_mean",
        "group_spread_mean",
        "quality_flags",
        "cluster_id",
        *extra,
    )


def _records(frame: pd.DataFrame, columns: Sequence[str], limit: int | None = None) -> list[dict[str, Any]]:
    selected_columns = [column for column in columns if column in frame.columns]
    work = frame.loc[:, selected_columns]
    if limit is not None:
        work = work.head(limit)
    return [_jsonable(record) for record in work.to_dict("records")]


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return data


def _load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return pd.read_csv(path)


def _validate_selected_features_known(selected_json: dict[str, Any], candidates: pd.DataFrame) -> None:
    selected = set(map(str, selected_json.get("selected_features", [])))
    known = set(candidates.get("factor_name", pd.Series(dtype=object)).astype(str))
    missing = sorted(selected - known)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"selected_features contains factors missing from candidate_features: {preview}")


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _as_bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
