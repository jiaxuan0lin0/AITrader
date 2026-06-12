import pandas as pd
import pytest

from FactorMiner.core.factor_spec import FactorResult, FactorSpec, combine_factor_results


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["A", "B"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "factor_a": [1.0, 2.0],
            "factor_b": [3.0, 4.0],
        }
    )


def test_factor_spec_validates_required_metadata() -> None:
    with pytest.raises(ValueError, match="name cannot be empty"):
        FactorSpec(name="", source="alpha158", category="kbar", inputs=("close",), expression="close")

    with pytest.raises(ValueError, match="window must be positive"):
        FactorSpec(name="bad", source="alpha158", category="rolling", inputs=("close",), expression="x", window=0)

    with pytest.raises(ValueError, match="lookback cannot be negative"):
        FactorSpec(name="bad", source="alpha158", category="rolling", inputs=("close",), expression="x", lookback=-1)


def test_factor_spec_manifest_record_is_json_friendly() -> None:
    spec = FactorSpec(
        name="ROC3",
        source="alpha158",
        category="rolling.roc",
        inputs=("close",),
        expression="delay(close, 3) / close",
        window=3,
        lookback=3,
    )

    assert spec.to_record() == {
        "name": "ROC3",
        "source": "alpha158",
        "category": "rolling.roc",
        "inputs": ["close"],
        "expression": "delay(close, 3) / close",
        "window": 3,
        "lookback": 3,
        "availability": "feature_asof_date",
        "description": "",
    }


def test_factor_result_validates_and_selects_output_columns() -> None:
    result = FactorResult(
        factors=_base_frame(),
        specs=[
            FactorSpec("factor_a", "test", "a", ("close",), "close"),
            FactorSpec("factor_b", "test", "b", ("volume",), "volume"),
        ],
    )

    result.validate()
    selected = result.select_output_columns()

    assert selected.columns.tolist() == ["stock_code", "trade_date", "factor_a", "factor_b"]
    assert result.factor_names() == ["factor_a", "factor_b"]
    assert result.max_lookback() == 0


def test_factor_result_rejects_missing_key_columns() -> None:
    frame = _base_frame().drop(columns=["trade_date"])
    result = FactorResult(frame, [FactorSpec("factor_a", "test", "a", ("close",), "close")])

    with pytest.raises(KeyError, match="Missing factor key columns"):
        result.validate()


def test_factor_result_rejects_duplicate_keys() -> None:
    frame = pd.concat([_base_frame(), _base_frame().iloc[[0]]], ignore_index=True)
    result = FactorResult(frame, [FactorSpec("factor_a", "test", "a", ("close",), "close")])

    with pytest.raises(ValueError, match="Factor keys must be unique"):
        result.validate()


def test_factor_result_rejects_duplicate_factor_names() -> None:
    result = FactorResult(
        _base_frame(),
        [
            FactorSpec("factor_a", "test", "a", ("close",), "close"),
            FactorSpec("factor_a", "test", "a", ("close",), "close"),
        ],
    )

    with pytest.raises(ValueError, match="Factor names must be unique"):
        result.validate()


def test_factor_result_rejects_missing_factor_columns() -> None:
    result = FactorResult(_base_frame(), [FactorSpec("missing_factor", "test", "a", ("close",), "close")])

    with pytest.raises(KeyError, match="Missing factor columns"):
        result.validate()


def test_combine_factor_results_outer_merges_on_daily_keys() -> None:
    first = FactorResult(
        _base_frame()[["stock_code", "trade_date", "factor_a"]],
        [FactorSpec("factor_a", "test", "a", ("close",), "close", lookback=3)],
    )
    second = FactorResult(
        pd.DataFrame(
            {
                "stock_code": ["A", "C"],
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "factor_b": [5.0, 6.0],
            }
        ),
        [FactorSpec("factor_b", "test", "b", ("volume",), "volume", lookback=10)],
    )

    combined = combine_factor_results([first, second])

    combined.validate()
    assert combined.factor_names() == ["factor_a", "factor_b"]
    assert combined.max_lookback() == 10
    assert len(combined.factors) == 3


def test_combine_factor_results_rejects_overlapping_factor_columns() -> None:
    first = FactorResult(
        _base_frame()[["stock_code", "trade_date", "factor_a"]],
        [FactorSpec("factor_a", "test", "a", ("close",), "close")],
    )
    second = FactorResult(
        _base_frame()[["stock_code", "trade_date", "factor_a"]],
        [FactorSpec("factor_a", "test", "a", ("close",), "close")],
    )

    with pytest.raises(ValueError, match="Duplicate factor columns"):
        combine_factor_results([first, second])
