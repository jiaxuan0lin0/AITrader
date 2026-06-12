from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Sequence

import pandas as pd

from aitrader_paths import DATASETS_ROOT


DEFAULT_DATASETS_ROOT = DATASETS_ROOT
DEFAULT_GPT_MINING_ROOT = DEFAULT_DATASETS_ROOT / "factors" / "gpt_mining"
DEFAULT_OUTPUT_ROOT = DEFAULT_GPT_MINING_ROOT / "experiment"
ALLOWED_FUNCTIONS = {
    "rolling_mean",
    "rolling_sum",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "delta",
    "pct_change",
    "return",
    "rank_cs",
    "zscore_cs",
    "winsorize",
    "industry_neutralize",
    "safe_div",
    "log1p",
    "abs",
    "sign",
    "interaction",
}
WINDOW_FUNCTIONS = {
    "rolling_mean",
    "rolling_sum",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "delta",
    "pct_change",
    "return",
}
LEAKAGE_TERMS = (
    "label_next",
    "future_return",
    "future returns",
    "target_trade_date",
    "post-cutoff",
    "post_cutoff",
    "2026-06",
    "比赛期",
    "未来收益",
    "未来涨跌",
    "next_open",
    "next_vwap",
)
NEWS_RESCORING_TERMS = (
    "需要重新打分",
    "要求重新打分",
    "需要重新新闻打分",
    "要求重新新闻打分",
    "需要重新总结",
    "要求重新总结",
    "需要重新分类新闻",
    "要求重新分类新闻",
    "重新调用qwen",
    "重新调用 gpt",
    "调用qwen",
    "调用 gpt",
    "call qwen",
    "call gpt",
    "rescore news",
    "re-score news",
    "new llm scoring",
    "manual per-news",
)
OUTPUT_FILES = {
    "validated_json": "candidates_validated.json",
    "rejected_csv": "candidates_rejected_by_parser.csv",
    "dependency_csv": "candidate_dependency_report.csv",
    "summary_json": "validation_summary.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate GPT-generated candidate factor formulas before materialization.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--round-name", default=None)
    parser.add_argument("--round-dir", type=Path, default=None)
    parser.add_argument("--packet-dir", type=Path, default=None)
    parser.add_argument("--response-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-high-leakage-risk", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_paths(args)
    result = validate_candidates(args)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


def _resolve_paths(args: argparse.Namespace) -> None:
    if args.round_dir is None:
        if args.round_name is not None:
            args.round_dir = args.output_root / args.round_name
        elif args.response_path is not None:
            response_parent = args.response_path.parent
            args.round_dir = response_parent.parent if response_parent.name == "packet" else response_parent
        else:
            raise ValueError("One of --round-dir, --round-name, or --response-path is required")
    args.round_dir = Path(args.round_dir)
    args.packet_dir = args.packet_dir or args.round_dir / "packet"
    if args.response_path is None:
        root_response = args.round_dir / "gpt_response.json"
        packet_response = args.packet_dir / "gpt_response.json"
        args.response_path = root_response if root_response.exists() else packet_response
    args.output_dir = args.output_dir or args.round_dir / "validated"


def validate_candidates(args: argparse.Namespace) -> dict[str, Any]:
    packet = _load_packet(args.packet_dir)
    response = _load_response(args.response_path)
    candidates = response["candidates"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    available = _available_field_index(packet["available_fields"])
    existing_selected = set(map(str, packet["selected"].get("selected_features", [])))
    existing_features = set(available["feature_fields"])
    schema = packet["schema"]
    item_schema = _candidate_item_schema(schema)
    web_research_required = _web_research_required(schema)
    duplicate_names = _duplicates([str(item.get("factor_name", "")) for item in candidates if isinstance(item, dict)])
    if web_research_required:
        _validate_web_research_summary(response["web_research_summary"])

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    dependency_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        record = _validate_one_candidate(
            index=index,
            candidate=candidate,
            schema=schema,
            available=available,
            existing_selected=existing_selected,
            existing_features=existing_features,
            duplicate_names=duplicate_names,
            item_schema=item_schema,
            web_research_required=web_research_required,
            allow_high_leakage_risk=args.allow_high_leakage_risk,
        )
        if record["status"] == "accepted":
            accepted.append(record["candidate"])
        else:
            rejected.append(record["rejected"])
        dependency_rows.append(record["dependency"])

    if not accepted:
        raise ValueError("No GPT candidates passed validation")

    output_paths = {
        key: args.output_dir / filename
        for key, filename in OUTPUT_FILES.items()
    }
    _write_json(output_paths["validated_json"], accepted)
    pd.DataFrame(rejected).to_csv(output_paths["rejected_csv"], index=False)
    pd.DataFrame(dependency_rows).to_csv(output_paths["dependency_csv"], index=False)
    summary = _validation_summary(args, candidates, accepted, rejected, dependency_rows, packet, response)
    _write_json(output_paths["summary_json"], summary)
    return {
        "response_path": str(args.response_path),
        "output_dir": str(args.output_dir),
        "total": len(candidates),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "summary": str(output_paths["summary_json"]),
    }


def _load_packet(packet_dir: Path) -> dict[str, Any]:
    packet_dir = Path(packet_dir)
    required = {
        "available_fields": packet_dir / "01_available_fields.json",
        "selected": packet_dir / "03_selected_features_reviewed.json",
        "schema": packet_dir / "candidate_schema.json",
        "manifest": packet_dir / "packet_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing packet files for candidate validation: {missing}")
    return {name: _load_json(path, name) for name, path in required.items()}


def _load_response(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing GPT response JSON: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    data = _loads_json_or_fenced(raw)
    web_research_summary = None
    if isinstance(data, dict):
        for key in ("candidates", "candidate_factors", "factors"):
            if isinstance(data.get(key), list):
                web_research_summary = data.get("web_research_summary")
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("GPT response must be a JSON array, or an object containing candidates/candidate_factors/factors")
    return {"candidates": data, "web_research_summary": web_research_summary}


def _validate_one_candidate(
    *,
    index: int,
    candidate: Any,
    schema: dict[str, Any],
    available: dict[str, Any],
    existing_selected: set[str],
    existing_features: set[str],
    duplicate_names: set[str],
    item_schema: dict[str, Any],
    web_research_required: bool,
    allow_high_leakage_risk: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    if not isinstance(candidate, dict):
        return _validation_record(
            index=index,
            candidate={},
            reasons=[f"not_object:{type(candidate).__name__}"],
            warnings=[],
            available=available,
        )

    required = set(item_schema.get("required", []))
    properties = item_schema.get("properties", {})
    allowed_keys = set(properties)
    missing_required = sorted(required - set(candidate))
    extra_keys = sorted(set(candidate) - allowed_keys)
    if missing_required:
        reasons.append(f"missing_required:{'|'.join(missing_required)}")
    if extra_keys:
        reasons.append(f"extra_keys:{'|'.join(extra_keys)}")

    factor_name = str(candidate.get("factor_name", ""))
    name_pattern = properties.get("factor_name", {}).get("pattern")
    if name_pattern and not re.match(name_pattern, factor_name):
        reasons.append("bad_factor_name_pattern")
    if factor_name in duplicate_names:
        reasons.append("duplicate_factor_name")
    if factor_name in existing_selected:
        reasons.append("name_collision_selected_feature")
    if factor_name in existing_features:
        reasons.append("name_collision_existing_feature")

    category = candidate.get("category")
    if category not in set(properties.get("category", {}).get("enum", [])):
        reasons.append(f"bad_category:{category}")
    leakage_risk = candidate.get("leakage_risk")
    if leakage_risk not in set(properties.get("leakage_risk", {}).get("enum", [])):
        reasons.append(f"bad_leakage_risk:{leakage_risk}")
    if leakage_risk == "high" and not allow_high_leakage_risk:
        reasons.append("high_leakage_risk")
    redundancy_risk = candidate.get("redundancy_risk")
    if redundancy_risk not in set(properties.get("redundancy_risk", {}).get("enum", [])):
        reasons.append(f"bad_redundancy_risk:{redundancy_risk}")
    expected_direction = candidate.get("expected_direction")
    if expected_direction not in set(properties.get("expected_direction", {}).get("enum", [])):
        reasons.append(f"bad_expected_direction:{expected_direction}")
    priority = candidate.get("priority")
    if not isinstance(priority, int) or not 1 <= priority <= 5:
        reasons.append(f"bad_priority:{priority}")
    if web_research_required:
        if not str(candidate.get("regime_link", "")).strip():
            reasons.append("missing_regime_link")

    declared_windows = candidate.get("windows", [])
    allowed_windows = set(properties.get("windows", {}).get("items", {}).get("enum", []))
    if not isinstance(declared_windows, list) or any(not isinstance(item, int) or item not in allowed_windows for item in declared_windows):
        reasons.append(f"bad_windows:{declared_windows}")
        declared_windows_set: set[int] = set()
    else:
        declared_windows_set = set(declared_windows)

    inputs = candidate.get("inputs", [])
    if not isinstance(inputs, list) or not inputs or any(not isinstance(item, str) or not item for item in inputs):
        reasons.append("bad_inputs")
        input_names: list[str] = []
    else:
        input_names = list(dict.fromkeys(inputs))
    missing_inputs = [name for name in input_names if name not in available["all_fields"]]
    if missing_inputs:
        reasons.append(f"unknown_inputs:{'|'.join(missing_inputs)}")
    label_inputs = [name for name in input_names if name.startswith("label_")]
    if label_inputs:
        reasons.append(f"label_inputs:{'|'.join(label_inputs)}")

    formula = str(candidate.get("formula", ""))
    formula_report = _validate_formula(formula, input_names, declared_windows_set, available, allowed_windows)
    reasons.extend(formula_report["reasons"])
    warnings.extend(formula_report["warnings"])

    leak_hit = _first_text_hit(candidate, LEAKAGE_TERMS)
    if leak_hit:
        reasons.append(f"leakage_term:{leak_hit}")
    rescore_hit = _first_text_hit(candidate, NEWS_RESCORING_TERMS)
    if rescore_hit:
        reasons.append(f"news_rescoring_term:{rescore_hit}")

    input_sources = _input_sources(input_names, available)
    dependency_type = _dependency_type(input_sources)
    compute_flags = _compute_flags(formula_report["functions"])
    compute_class = _compute_class(compute_flags)
    if formula_report["unused_inputs"]:
        warnings.append(f"unused_inputs:{'|'.join(formula_report['unused_inputs'])}")

    enriched = {
        **candidate,
        "validation_status": "accepted" if not reasons else "rejected",
        "dependency_type": dependency_type,
        "compute_class": compute_class,
        "compute_flags": compute_flags,
        "input_sources": input_sources,
        "formula_functions": formula_report["functions"],
        "formula_windows": formula_report["windows"],
        "formula_fields": formula_report["fields"],
        "validation_warnings": warnings,
    }
    dependency = {
        "row_index": index,
        "factor_name": factor_name,
        "status": "accepted" if not reasons else "rejected",
        "reject_reasons": ";".join(reasons),
        "warnings": ";".join(warnings),
        "category": category,
        "priority": priority,
        "dependency_type": dependency_type,
        "compute_class": compute_class,
        "compute_flags": "|".join(compute_flags),
        "input_count": len(input_names),
        "raw_input_count": sum(1 for source in input_sources.values() if "processed" in source),
        "feature_input_count": sum(1 for source in input_sources.values() if "feature" in source),
        "missing_input_count": len(missing_inputs),
        "formula_functions": "|".join(formula_report["functions"]),
        "formula_windows": "|".join(map(str, formula_report["windows"])),
    }
    if not reasons:
        return {"status": "accepted", "candidate": enriched, "dependency": dependency}
    rejected = {
        "row_index": index,
        "factor_name": factor_name,
        "reject_reasons": ";".join(reasons),
        "warnings": ";".join(warnings),
        "formula": formula,
        "inputs": json.dumps(input_names, ensure_ascii=False),
        "category": category,
        "priority": priority,
    }
    return {"status": "rejected", "rejected": rejected, "dependency": dependency}


def _validate_formula(
    formula: str,
    inputs: list[str],
    declared_windows: set[int],
    available: dict[str, Any],
    allowed_windows: set[int],
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    calls = _function_calls(formula)
    functions = [call["name"] for call in calls]
    unknown_functions = sorted(set(functions) - ALLOWED_FUNCTIONS)
    if unknown_functions:
        reasons.append(f"unknown_functions:{'|'.join(unknown_functions)}")

    used_windows: list[int] = []
    for call in calls:
        if call["name"] in WINDOW_FUNCTIONS:
            numeric_args = [int(item) for item in re.findall(r"(?<![A-Za-z_])(?:window\s*=\s*)?([0-9]+)(?![A-Za-z_])", call["body"])]
            if not numeric_args:
                reasons.append(f"missing_window_arg:{call['name']}")
                continue
            window = numeric_args[-1]
            used_windows.append(window)
            if window not in allowed_windows:
                reasons.append(f"window_not_allowed:{call['name']}:{window}")
    undeclared_windows = sorted(set(used_windows) - declared_windows)
    if undeclared_windows:
        reasons.append(f"formula_window_not_declared:{'|'.join(map(str, undeclared_windows))}")

    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula))
    allowed_constants = {"nan", "NaN", "inf", "True", "False"}
    known_fields = available["all_fields"]
    unknown_tokens = sorted(tokens - ALLOWED_FUNCTIONS - known_fields - allowed_constants)
    if unknown_tokens:
        reasons.append(f"unknown_formula_tokens:{'|'.join(unknown_tokens[:20])}")
    formula_fields = sorted(tokens & known_fields)
    undeclared_inputs = sorted(set(formula_fields) - set(inputs))
    if undeclared_inputs:
        reasons.append(f"formula_uses_undeclared_inputs:{'|'.join(undeclared_inputs)}")
    unused_inputs = sorted(set(inputs) - set(formula_fields))

    return {
        "reasons": reasons,
        "warnings": warnings,
        "functions": sorted(set(functions), key=functions.index),
        "windows": sorted(set(used_windows)),
        "fields": formula_fields,
        "unused_inputs": unused_inputs,
    }


def _function_calls(formula: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", formula):
        name = match.group(1)
        open_index = formula.find("(", match.start())
        close_index = _matching_paren(formula, open_index)
        if close_index is None:
            calls.append({"name": name, "body": "", "closed": False})
            continue
        calls.append({"name": name, "body": formula[open_index + 1 : close_index], "closed": True})
    if any(not call["closed"] for call in calls):
        calls.append({"name": "__parse_error__", "body": "", "closed": False})
    return calls


def _matching_paren(text: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _validation_record(
    *,
    index: int,
    candidate: dict[str, Any],
    reasons: list[str],
    warnings: list[str],
    available: dict[str, Any],
) -> dict[str, Any]:
    factor_name = str(candidate.get("factor_name", f"row_{index}"))
    dependency = {
        "row_index": index,
        "factor_name": factor_name,
        "status": "rejected",
        "reject_reasons": ";".join(reasons),
        "warnings": ";".join(warnings),
        "category": candidate.get("category"),
        "priority": candidate.get("priority"),
        "dependency_type": "unknown",
        "compute_class": "unknown",
        "compute_flags": "",
        "input_count": 0,
        "raw_input_count": 0,
        "feature_input_count": 0,
        "missing_input_count": 0,
        "formula_functions": "",
        "formula_windows": "",
    }
    return {
        "status": "rejected",
        "rejected": {
            "row_index": index,
            "factor_name": factor_name,
            "reject_reasons": ";".join(reasons),
            "warnings": ";".join(warnings),
            "formula": candidate.get("formula", ""),
            "inputs": json.dumps(candidate.get("inputs", []), ensure_ascii=False),
            "category": candidate.get("category"),
            "priority": candidate.get("priority"),
        },
        "dependency": dependency,
    }


def _available_field_index(available_fields: dict[str, Any]) -> dict[str, Any]:
    processed_field_tables: dict[str, set[str]] = defaultdict(set)
    for table, profile in available_fields.get("processed_tables", {}).items():
        for column in profile.get("columns", []):
            processed_field_tables[str(column)].add(str(table))
    feature_field_blocks: dict[str, set[str]] = defaultdict(set)
    for block in available_fields.get("feature_blocks", []):
        for factor_name in block.get("factor_names", []):
            feature_field_blocks[str(factor_name)].add(str(block.get("name")))
    processed_fields = set(processed_field_tables)
    feature_fields = set(feature_field_blocks)
    return {
        "processed_fields": processed_fields,
        "feature_fields": feature_fields,
        "all_fields": processed_fields | feature_fields,
        "processed_field_tables": processed_field_tables,
        "feature_field_blocks": feature_field_blocks,
    }


def _input_sources(inputs: list[str], available: dict[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for name in inputs:
        parts = []
        if name in available["processed_fields"]:
            tables = sorted(available["processed_field_tables"].get(name, []))
            parts.append(f"processed:{'|'.join(tables)}")
        if name in available["feature_fields"]:
            blocks = sorted(available["feature_field_blocks"].get(name, []))
            parts.append(f"feature:{'|'.join(blocks)}")
        sources[name] = ",".join(parts) if parts else "missing"
    return sources


def _dependency_type(input_sources: dict[str, str]) -> str:
    has_raw = any("processed:" in source for source in input_sources.values())
    has_feature = any("feature:" in source for source in input_sources.values())
    if has_raw and has_feature:
        return "mixed"
    if has_feature:
        return "existing_feature"
    if has_raw:
        return "raw_only"
    return "unknown"


def _compute_flags(functions: list[str]) -> list[str]:
    flags: list[str] = []
    if any(func in WINDOW_FUNCTIONS for func in functions):
        flags.append("rolling")
    if any(func in {"rank_cs", "zscore_cs"} for func in functions):
        flags.append("cross_sectional")
    if "industry_neutralize" in functions:
        flags.append("industry_neutral")
    if "interaction" in functions:
        flags.append("interaction")
    return flags or ["simple"]


def _compute_class(flags: list[str]) -> str:
    for name in ("industry_neutral", "interaction", "cross_sectional", "rolling", "simple"):
        if name in flags:
            return name
    return "unknown"


def _first_text_hit(candidate: dict[str, Any], terms: Sequence[str]) -> str | None:
    text = json.dumps(candidate, ensure_ascii=False).lower()
    for term in terms:
        if term.lower() in text:
            return term
    return None


def _validation_summary(
    args: argparse.Namespace,
    candidates: list[Any],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    dependency_rows: list[dict[str, Any]],
    packet: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    dependency_counts = Counter(row["dependency_type"] for row in dependency_rows if row["status"] == "accepted")
    compute_counts = Counter(row["compute_class"] for row in dependency_rows if row["status"] == "accepted")
    category_counts = Counter(item.get("category") for item in accepted)
    priority_counts = Counter(str(item.get("priority")) for item in accepted)
    reject_reason_counts: Counter[str] = Counter()
    for item in rejected:
        for reason in str(item.get("reject_reasons", "")).split(";"):
            if reason:
                reject_reason_counts[reason.split(":", 1)[0]] += 1
    return {
        "version": "gpt_candidate_validation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_dir": str(args.round_dir),
        "packet_dir": str(args.packet_dir),
        "response_path": str(args.response_path),
        "packet_profile": packet["manifest"].get("profile"),
        "packet_cutoff_date": packet["manifest"].get("cutoff_date"),
        "web_research_required": _web_research_required(packet["schema"]),
        "web_research_status": _web_research_status(response["web_research_summary"]),
        "web_research_source_count": len(_research_source_ids(response["web_research_summary"])),
        "total_candidates": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "dependency_type_counts": dict(dependency_counts),
        "compute_class_counts": dict(compute_counts),
        "category_counts": dict(category_counts),
        "priority_counts": dict(priority_counts),
        "reject_reason_counts": dict(reject_reason_counts),
        "outputs": {key: str(Path(args.output_dir) / filename) for key, filename in OUTPUT_FILES.items()},
    }


def _candidate_item_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "array":
        return schema.get("items", {})
    if schema.get("type") == "object":
        return schema.get("properties", {}).get("candidates", {}).get("items", {})
    return schema.get("items", {})


def _web_research_required(schema: dict[str, Any]) -> bool:
    if schema.get("type") == "object" and "web_research_summary" in schema.get("required", []):
        return True
    item_schema = _candidate_item_schema(schema)
    return "regime_link" in item_schema.get("required", [])


def _validate_web_research_summary(summary: Any) -> None:
    if not isinstance(summary, dict):
        raise ValueError("GPT response missing required web_research_summary")
    status = summary.get("status")
    if status != "completed":
        raise ValueError(f"web_research_summary.status must be completed, got {status!r}")
    sources = summary.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("web_research_summary.sources must contain at least one public source")
    source_ids = _research_source_ids(summary)
    if len(source_ids) != len(sources):
        raise ValueError("web_research_summary.sources must have unique non-empty source_id values")


def _research_source_ids(summary: Any) -> set[str]:
    if not isinstance(summary, dict):
        return set()
    sources = summary.get("sources", [])
    if not isinstance(sources, list):
        return set()
    return {str(item.get("source_id")) for item in sources if isinstance(item, dict) and item.get("source_id")}


def _web_research_status(summary: Any) -> str | None:
    if not isinstance(summary, dict):
        return None
    return None if summary.get("status") is None else str(summary.get("status"))


def _loads_json_or_fenced(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        json_error = exc
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return json.loads(match.group(1))
    raise json_error


def _duplicates(values: list[str]) -> set[str]:
    counts = Counter(values)
    return {value for value, count in counts.items() if value and count > 1}


def _load_json(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
