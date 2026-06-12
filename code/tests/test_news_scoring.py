import pandas as pd
import pytest

from FactorMiner.news_scoring.schema import parse_batch_score_response, parse_score_response
from FactorMiner.news_scoring.score_news_items import select_pending_items, split_frame


def test_parse_score_response_accepts_json_object() -> None:
    payload = parse_score_response(
        """
        {
          "sentiment_score": 0.5,
          "impact_score": 0.8,
          "risk_score": 0.1,
          "relevance_score": 0.9,
          "novelty_score": 0.7,
          "event_type": "earnings",
          "horizon": "short",
          "summary": "业绩超预期"
        }
        """
    )

    assert payload.sentiment_score == 0.5
    assert payload.horizon == "short"


def test_parse_score_response_normalizes_invalid_event_type() -> None:
    payload = parse_score_response(
        """
        {
          "sentiment_score": -0.2,
          "impact_score": 0.4,
          "risk_score": 0.8,
          "relevance_score": 0.6,
          "novelty_score": 0.5,
          "event_type": "risk",
          "horizon": "short",
          "summary": "市场风险偏好下降"
        }
        """
    )

    assert payload.event_type == "other"


def test_parse_score_response_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValueError, match="Invalid news score payload"):
        parse_score_response('{"sentiment_score": 2, "impact_score": 0, "risk_score": 0, "relevance_score": 0, "novelty_score": 0}')


def test_select_pending_items_uses_model_and_prompt_version() -> None:
    news_items = pd.DataFrame(
        {
            "news_id": ["a", "b"],
            "news_text_hash": ["hash-a", "hash-b"],
            "news_text": ["A", "B"],
        }
    )
    existing = pd.DataFrame(
        {
            "news_text_hash": ["hash-a", "hash-b"],
            "llm_model": ["qwen3-news", "other-model"],
            "prompt_version": ["news_score", "news_score"],
        }
    )

    pending = select_pending_items(news_items, existing, "qwen3-news", "news_score", None)

    assert pending["news_id"].tolist() == ["b"]


def test_parse_batch_score_response_orders_by_input_id() -> None:
    payloads = parse_batch_score_response(
        """
        {
          "scores": [
            {
              "id": 1,
              "sentiment_score": -0.5,
              "impact_score": 0.4,
              "risk_score": 0.7,
              "relevance_score": 0.3,
              "novelty_score": 0.8,
              "event_type": "geopolitics",
              "horizon": "short",
              "summary": "地缘风险升温"
            },
            {
              "id": 0,
              "sentiment_score": 0.5,
              "impact_score": 0.6,
              "risk_score": 0.1,
              "relevance_score": 0.9,
              "novelty_score": 0.7,
              "event_type": "policy",
              "horizon": "short",
              "summary": "政策支持行业发展"
            }
          ]
        }
        """,
        expected_count=2,
    )

    assert [payload.sentiment_score for payload in payloads] == [0.5, -0.5]


def test_parse_batch_score_response_accepts_score_map() -> None:
    payloads = parse_batch_score_response(
        """
        {
          "scores": {
            "0": {
              "sentiment_score": 0.5,
              "impact_score": 0.6,
              "risk_score": 0.1,
              "relevance_score": 0.9,
              "novelty_score": 0.7,
              "event_type": "policy",
              "horizon": "short",
              "summary": "政策支持行业发展"
            },
            "1": {
              "sentiment_score": -0.5,
              "impact_score": 0.4,
              "risk_score": 0.7,
              "relevance_score": 0.3,
              "novelty_score": 0.8,
              "event_type": "geopolitics",
              "horizon": "short",
              "summary": "地缘风险升温"
            }
          }
        }
        """,
        expected_count=2,
    )

    assert [payload.event_type for payload in payloads] == ["policy", "geopolitics"]


def test_parse_batch_score_response_rejects_missing_items() -> None:
    with pytest.raises(ValueError, match="Invalid batch news score ids"):
        parse_batch_score_response(
            """
            {
              "scores": [
                {
                  "id": 0,
                  "sentiment_score": 0,
                  "impact_score": 0,
                  "risk_score": 0,
                  "relevance_score": 0,
                  "novelty_score": 0,
                  "event_type": "other",
                  "horizon": "unknown",
                  "summary": "无关新闻摘要"
                }
              ]
            }
            """,
            expected_count=2,
        )


def test_split_frame_uses_request_batch_size() -> None:
    frame = pd.DataFrame({"value": range(5)})

    chunks = split_frame(frame, 2)

    assert [chunk["value"].tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]
