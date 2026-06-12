from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq

from FactorMiner.news_scoring.client import NewsScoringClient
from FactorMiner.news_scoring.config import NewsScoringConfig
from FactorMiner.pools.news_llm import prepare_news_items


SCORE_COLUMNS = (
    "news_id",
    "news_text_hash",
    "publish_time",
    "sentiment_score",
    "impact_score",
    "risk_score",
    "relevance_score",
    "novelty_score",
    "event_type",
    "horizon",
    "summary",
    "llm_model",
    "prompt_version",
    "scored_at",
)
_THREAD_LOCAL = threading.local()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score news items with a local OpenAI-compatible LLM service.")
    parser.add_argument("--news-path", type=Path, default=NewsScoringConfig.news_path)
    parser.add_argument("--scores-path", type=Path, default=NewsScoringConfig.scores_path)
    parser.add_argument("--base-url", default=NewsScoringConfig.base_url)
    parser.add_argument("--model", default=NewsScoringConfig.model)
    parser.add_argument("--api-key", default=NewsScoringConfig.api_key)
    parser.add_argument("--prompt-version", default=NewsScoringConfig.prompt_version)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--since", default=None, help="Inclusive publish_time lower bound, e.g. 2026-05-01")
    parser.add_argument("--until", default=None, help="Inclusive publish_time upper bound, e.g. 2026-05-18 09:25:00")
    parser.add_argument("--checkpoint-size", type=int, default=NewsScoringConfig.checkpoint_size)
    parser.add_argument("--concurrency", type=int, default=NewsScoringConfig.concurrency)
    parser.add_argument("--max-tokens", type=int, default=NewsScoringConfig.max_tokens)
    parser.add_argument("--request-batch-size", type=int, default=NewsScoringConfig.request_batch_size)
    parser.add_argument("--use-guided-json", action="store_true", default=NewsScoringConfig.use_guided_json)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = NewsScoringConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_version=args.prompt_version,
        checkpoint_size=args.checkpoint_size,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        request_batch_size=args.request_batch_size,
        use_guided_json=args.use_guided_json,
        news_path=args.news_path,
        scores_path=args.scores_path,
    )

    news_items = load_news_items(config.news_path, args.since, args.until)
    existing = load_existing_scores(config.scores_path)
    pending = select_pending_items(news_items, existing, config.model, config.prompt_version, args.limit)

    print(f"news_items={len(news_items)} existing_scores={len(existing)} pending={len(pending)}")
    if pending.empty:
        return 0

    records = existing.to_dict("records")
    total = len(pending)
    concurrency = max(1, config.concurrency)
    request_batch_size = max(1, config.request_batch_size)
    print(
        " ".join(
            [
                f"concurrency={concurrency}",
                f"checkpoint_size={config.checkpoint_size}",
                f"request_batch_size={request_batch_size}",
                f"max_tokens={config.max_tokens}",
                f"use_guided_json={config.use_guided_json}",
            ]
        )
    )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        scored = 0
        checkpoint_size = max(1, config.checkpoint_size)
        for start in range(0, total, checkpoint_size):
            checkpoint = pending.iloc[start : start + checkpoint_size]
            futures = [
                executor.submit(score_news_batch, batch, config)
                for batch in split_frame(checkpoint, request_batch_size)
            ]
            for future in as_completed(futures):
                batch_records = future.result()
                records.extend(batch_records)
                scored += len(batch_records)
            save_scores(records, config.scores_path)
            print(f"scored={scored}/{total}")

    save_scores(records, config.scores_path)
    print(f"scored={total}/{total}")
    return 0


def split_frame(frame: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    size = max(1, batch_size)
    return [frame.iloc[start : start + size] for start in range(0, len(frame), size)]


def score_news_batch(batch: pd.DataFrame, config: NewsScoringConfig) -> list[dict]:
    client = get_thread_client(config)
    texts = batch["news_text"].astype(str).tolist()
    try:
        payloads = client.score_texts(texts)
    except Exception as exc:
        if len(batch) == 1:
            raise
        if len(batch) > 2:
            split_size = max(1, len(batch) // 2)
            print(f"batch_failed size={len(batch)} fallback=split split_size={split_size} reason={exc}")
            records: list[dict] = []
            for sub_batch in split_frame(batch, split_size):
                records.extend(score_news_batch(sub_batch, config))
            return records
        print(f"batch_failed size={len(batch)} fallback=single reason={exc}")
        payloads = [client.score_text(text) for text in texts]

    if len(payloads) != len(batch):
        raise ValueError(f"Batch score count mismatch: expected {len(batch)}, got {len(payloads)}")
    return [
        build_score_record(row, payload, config)
        for row, payload in zip(batch.itertuples(index=False), payloads, strict=True)
    ]


def score_news_row(row: object, config: NewsScoringConfig) -> dict:
    client = get_thread_client(config)
    payload = client.score_text(row.news_text)
    return build_score_record(row, payload, config)


def build_score_record(row: object, payload: object, config: NewsScoringConfig) -> dict:
    return {
        "news_id": row.news_id,
        "news_text_hash": row.news_text_hash,
        "publish_time": row.publish_time,
        **payload.to_record(),
        "llm_model": config.model,
        "prompt_version": config.prompt_version,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def get_thread_client(config: NewsScoringConfig) -> NewsScoringClient:
    client = getattr(_THREAD_LOCAL, "news_scoring_client", None)
    if client is None or client.config != config:
        client = NewsScoringClient(config)
        _THREAD_LOCAL.news_scoring_client = client
    return client


def load_news_items(news_path: Path, since: str | None, until: str | None) -> pd.DataFrame:
    columns = ["stock_code", "matched_stock_codes", "matched_stock_count", "trade_date", "publish_time", "news_text", "__source_file"]
    available = set(pq.read_schema(news_path).names)
    news = pd.read_parquet(news_path, columns=[column for column in columns if column in available])
    items = prepare_news_items(news).news_items
    if since:
        items = items[items["publish_time"].ge(pd.to_datetime(since))]
    if until:
        items = items[items["publish_time"].le(pd.to_datetime(until))]
    return items.reset_index(drop=True)


def load_existing_scores(scores_path: Path) -> pd.DataFrame:
    if not scores_path.exists():
        return pd.DataFrame(columns=SCORE_COLUMNS)
    return pd.read_parquet(scores_path)


def select_pending_items(
    news_items: pd.DataFrame,
    existing_scores: pd.DataFrame,
    model: str,
    prompt_version: str,
    limit: int | None,
) -> pd.DataFrame:
    if existing_scores.empty:
        scored_keys: set[tuple[str, str, str]] = set()
    else:
        scored = existing_scores[
            existing_scores["llm_model"].eq(model) & existing_scores["prompt_version"].eq(prompt_version)
        ]
        scored_keys = set(zip(scored["news_text_hash"], scored["llm_model"], scored["prompt_version"], strict=True))
    keys = list(zip(news_items["news_text_hash"], [model] * len(news_items), [prompt_version] * len(news_items), strict=True))
    pending = news_items[[key not in scored_keys for key in keys]].copy()
    pending = pending.drop_duplicates("news_text_hash", keep="first")
    if limit is not None:
        pending = pending.head(limit)
    return pending.reset_index(drop=True)


def save_scores(records: list[dict], scores_path: Path) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=SCORE_COLUMNS)
    frame = frame.loc[:, [column for column in SCORE_COLUMNS if column in frame.columns]]
    frame = frame.drop_duplicates(["news_text_hash", "llm_model", "prompt_version"], keep="last")
    frame.to_parquet(scores_path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
