from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from FactorMiner.news_scoring.prompt import PROMPT_VERSION
from aitrader_paths import DATASETS_ROOT


@dataclass(frozen=True)
class NewsScoringConfig:
    model: str = "qwen3-news"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    prompt_version: str = PROMPT_VERSION
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout: float = 120.0
    checkpoint_size: int = 1000
    concurrency: int = 20
    request_batch_size: int = 4
    use_guided_json: bool = False
    news_path: Path = DATASETS_ROOT / "processed" / "news.parquet"
    scores_path: Path = DATASETS_ROOT / "factors" / "news_llm_scores.parquet"
