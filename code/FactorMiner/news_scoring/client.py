from __future__ import annotations

from typing import Sequence

import httpx
from openai import OpenAI

from FactorMiner.news_scoring.config import NewsScoringConfig
from FactorMiner.news_scoring.prompt import SYSTEM_PROMPT, build_batch_user_prompt, build_user_prompt
from FactorMiner.news_scoring.schema import (
    NewsScorePayload,
    batch_score_json_schema,
    parse_batch_score_response,
    parse_score_response,
    single_score_json_schema,
)


class NewsScoringClient:
    """OpenAI-compatible client for a local vLLM scoring service."""

    def __init__(self, config: NewsScoringConfig):
        self.config = config
        http_client = httpx.Client(timeout=config.request_timeout, trust_env=False)
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, http_client=http_client)

    def score_text(self, news_text: str) -> NewsScorePayload:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(news_text)},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format=self._single_response_format(),
        )
        content = response.choices[0].message.content or ""
        return parse_score_response(content)

    def score_texts(self, news_texts: Sequence[str]) -> list[NewsScorePayload]:
        if not news_texts:
            return []
        if len(news_texts) == 1:
            return [self.score_text(news_texts[0])]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_batch_user_prompt(list(news_texts))},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format=self._batch_response_format(len(news_texts)),
        )
        content = response.choices[0].message.content or ""
        return parse_batch_score_response(content, expected_count=len(news_texts))

    def _single_response_format(self) -> dict:
        if self.config.use_guided_json:
            return _json_schema_response_format("news_score", single_score_json_schema())
        return {"type": "json_object"}

    def _batch_response_format(self, expected_count: int) -> dict:
        if self.config.use_guided_json:
            return _json_schema_response_format("news_batch_score", batch_score_json_schema(expected_count))
        return {"type": "json_object"}


def _json_schema_response_format(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": True,
        },
    }
