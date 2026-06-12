from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

EVENT_TYPES = (
    "company",
    "earnings",
    "policy",
    "macro",
    "rates",
    "fx",
    "geopolitics",
    "commodity",
    "shipping",
    "industry",
    "market",
    "litigation",
    "contract",
    "other",
)
HORIZONS = ("intraday", "short", "medium", "unknown")
EVENT_TYPE_ALIASES = {
    "risk": "other",
    "risks": "other",
}


class NewsScorePayload(BaseModel):
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    impact_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    event_type: Literal[
        "company",
        "earnings",
        "policy",
        "macro",
        "rates",
        "fx",
        "geopolitics",
        "commodity",
        "shipping",
        "industry",
        "market",
        "litigation",
        "contract",
        "other",
    ] = "other"
    horizon: Literal["intraday", "short", "medium", "unknown"] = "unknown"
    summary: str = ""

    @field_validator("event_type", mode="before")
    @classmethod
    def normalize_event_type(cls, value: Any) -> str:
        normalized = str(value or "other").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in EVENT_TYPES:
            return normalized
        return EVENT_TYPE_ALIASES.get(normalized, "other")

    def to_record(self) -> dict[str, Any]:
        return self.model_dump()


class NewsBatchScorePayload(NewsScorePayload):
    id: int = Field(ge=0)


class NewsBatchResponse(BaseModel):
    scores: list[NewsBatchScorePayload]


class NewsBatchMapResponse(BaseModel):
    scores: dict[str, NewsScorePayload]


def parse_score_response(text: str) -> NewsScorePayload:
    """Parse and validate the model JSON response."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_extract_json_object(text))
    try:
        return NewsScorePayload.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid news score payload: {exc}") from exc


def single_score_json_schema() -> dict[str, Any]:
    return _score_object_schema(include_id=False)


def batch_score_json_schema(expected_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    str(index): _score_object_schema(include_id=False)
                    for index in range(expected_count)
                },
                "required": [str(index) for index in range(expected_count)],
                "additionalProperties": False,
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }


def parse_batch_score_response(text: str, expected_count: int) -> list[NewsScorePayload]:
    """Parse a batch model response and return payloads ordered by input id."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_extract_json_object(text))
    if isinstance(data.get("scores"), dict):
        return _parse_batch_score_map(data, expected_count)
    try:
        batch = NewsBatchResponse.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid batch news score payload: {exc}") from exc

    scores = batch.scores
    ids = [item.id for item in scores]
    expected_ids = set(range(expected_count))
    if len(scores) != expected_count or set(ids) != expected_ids or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid batch news score ids: expected 0..{expected_count - 1}, got {ids}")
    by_id = {item.id: item for item in scores}
    return [
        NewsScorePayload.model_validate(by_id[index].model_dump(exclude={"id"}))
        for index in range(expected_count)
    ]


def _parse_batch_score_map(data: dict[str, Any], expected_count: int) -> list[NewsScorePayload]:
    try:
        batch = NewsBatchMapResponse.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid batch news score payload: {exc}") from exc
    expected_keys = {str(index) for index in range(expected_count)}
    keys = set(batch.scores)
    if keys != expected_keys:
        raise ValueError(f"Invalid batch news score keys: expected {sorted(expected_keys)}, got {sorted(keys)}")
    return [batch.scores[str(index)] for index in range(expected_count)]


def _score_object_schema(include_id: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "sentiment_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "impact_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "risk_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "relevance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "novelty_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "event_type": {"type": "string", "enum": list(EVENT_TYPES)},
        "horizon": {"type": "string", "enum": list(HORIZONS)},
        "summary": {"type": "string", "minLength": 1},
    }
    required = [
        "sentiment_score",
        "impact_score",
        "risk_score",
        "relevance_score",
        "novelty_score",
        "event_type",
        "horizon",
        "summary",
    ]
    if include_id:
        properties = {"id": {"type": "integer", "minimum": 0}, **properties}
        required = ["id", *required]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _extract_json_object(text: str) -> str:
    matched = re.search(r"\{.*\}", text, flags=re.S)
    if not matched:
        raise ValueError("Model response does not contain a JSON object")
    return matched.group(0)
