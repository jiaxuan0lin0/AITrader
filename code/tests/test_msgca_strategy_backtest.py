from __future__ import annotations

import json

import pandas as pd

from model.msgca.backtest import (
    daily_orders_from_trades,
    filter_first_trade_days,
    main as backtest_main,
    read_predictions,
    rolling_trade_date_windows,
    run_rolling_topk_backtest,
    run_topk_backtest,
)
from model.msgca.competition_metrics import score_competition_predictions
from model.msgca.ensemble_predictions import ensemble_predictions
from model.msgca.metrics import direction_accuracy, direction_prediction_metrics, return_prediction_metrics, topk_prediction_metrics
from model.msgca.predict_live_ensemble import filter_target_date, write_buy_list
from model.msgca.strategy import StrategyParams, generate_trade_signals, prepare_strategy_predictions
from model.msgca.strategy import strategy_score


def test_generate_trade_signals_required_columns_and_actions() -> None:
    predictions = pd.DataFrame(
        {
            "target_trade_date": [pd.Timestamp("2026-06-01")] * 4,
            "stock_code": ["A", "B", "C", "D"],
            "stock_name": ["a", "b", "c", "d"],
            "industry": ["i"] * 4,
            "y_score": [4.0, 3.0, 2.0, 1.0],
            "g_price": [1.0] * 4,
            "g_text": [0.0] * 4,
            "g_fundamental": [0.0] * 4,
        }
    )

    signals = generate_trade_signals(predictions, top_n=2, daily_replace_k=1)

    assert {"target_trade_date", "stock_code", "suggested_action", "target_weight", "order_note"} <= set(signals.columns)
    assert signals.set_index("stock_code").loc["A", "suggested_action"] == "buy"
    assert signals.set_index("stock_code").loc["B", "target_weight"] == 0.5


def test_validation_metrics_align_return_direction_and_topk_objectives() -> None:
    predictions = pd.DataFrame(
        {
            "target_trade_date": [pd.Timestamp("2026-06-01")] * 4,
            "stock_code": ["A", "B", "C", "D"],
            "return_pred": [0.10, 0.04, -0.02, -0.08],
            "direction_prob": [0.9, 0.8, 0.2, 0.1],
            "final_score": [4.0, 3.0, 2.0, 1.0],
            "label_next_open_return": [0.10, 0.04, -0.02, -0.08],
            "label_next_vwap_return": [0.08, 0.02, -0.04, -0.10],
        }
    )

    return_metrics = return_prediction_metrics(predictions, secondary_weight=0.5)
    direction_metrics = direction_prediction_metrics(predictions)
    topk_metrics = topk_prediction_metrics(predictions, ks=[1, 2], score_col="final_score")

    assert return_metrics["return_pred_open_mse"] == 0.0
    assert return_metrics["return_pred_blend_mse"] > 0.0
    assert direction_metrics["direction_pred_bce"] < 0.25
    assert direction_metrics["direction_pred_accuracy"] == 1.0
    assert abs(topk_metrics["top1_return_mean"] - 0.10) < 1e-12
    assert abs(topk_metrics["top2_return_mean"] - 0.07) < 1e-12


def test_prepare_strategy_predictions_filters_and_scores() -> None:
    predictions = pd.DataFrame(
        {
            "target_trade_date": [pd.Timestamp("2026-06-01")] * 5,
            "stock_code": ["A.SZ", "B.SZ", "C.SZ", "D.BJ", "E.SZ"],
            "stock_name": ["Alpha", "Beta", "*ST Case", "North", "Echo"],
            "y_score": [1.0, 5.0, 10.0, 20.0, 0.0],
            "return_pred": [0.02, 0.03, 0.50, 0.60, 0.01],
            "direction_prob": [0.5, 0.6, 0.9, 0.9, 0.4],
            "log_total_mv": [10.0, 12.0, 13.0, 14.0, 11.0],
        }
    )

    scored = prepare_strategy_predictions(
        predictions,
        StrategyParams(
            score_variant="return_pred",
            cap_min_pct=0.5,
            cap_bonus=0.5,
            exclude_st=True,
            exclude_bj=True,
        ),
    )

    assert scored["stock_code"].tolist() == ["A.SZ", "B.SZ", "E.SZ"]
    assert "raw_y_score" in scored.columns
    assert scored.sort_values("y_score", ascending=False).iloc[0]["stock_code"] == "B.SZ"
    assert scored.loc[scored["stock_code"].eq("A.SZ"), "y_score"].iloc[0] == -1e9


def test_prepare_strategy_predictions_supports_weighted_score() -> None:
    predictions = pd.DataFrame(
        {
            "target_trade_date": [pd.Timestamp("2026-06-01")] * 3,
            "stock_code": ["A.SZ", "B.SZ", "C.SZ"],
            "stock_name": ["Alpha", "Beta", "Charlie"],
            "y_score": [3.0, 2.0, 1.0],
            "return_pred": [0.0, 0.1, 0.2],
            "direction_prob": [0.5, 0.5, 0.5],
            "log_total_mv": [10.0, 10.0, 10.0],
        }
    )

    scored = prepare_strategy_predictions(
        predictions,
        StrategyParams(score_variant="weighted", score_weight_y=0.0, score_weight_return=1.0),
    )

    assert scored.sort_values("y_score", ascending=False).iloc[0]["stock_code"] == "C.SZ"
    assert scored["strategy_score_weight_return"].iloc[0] == 1.0


def test_prepare_strategy_predictions_supports_final_score() -> None:
    predictions = pd.DataFrame(
        {
            "target_trade_date": [pd.Timestamp("2026-06-01")] * 3,
            "stock_code": ["A.SZ", "B.SZ", "C.SZ"],
            "stock_name": ["Alpha", "Beta", "Charlie"],
            "y_score": [3.0, 2.0, 1.0],
            "final_score": [1.0, 5.0, 2.0],
        }
    )

    scored = prepare_strategy_predictions(predictions, StrategyParams(score_variant="final_score"))

    assert scored.sort_values("y_score", ascending=False).iloc[0]["stock_code"] == "B.SZ"
    assert scored["strategy_score_variant"].iloc[0] == "final_score"


def test_direct_theme_soft_cluster_score_round_robins_selected_clusters() -> None:
    predictions = pd.DataFrame(
        {
            "target_trade_date": [pd.Timestamp("2026-06-01")] * 6,
            "stock_code": ["A1", "A2", "A3", "B1", "B2", "B3"],
            "y_score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "final_score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "return_pred": [0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
            "direction_prob": [0.7] * 6,
            "context_tr": [1.0] * 6,
            "context_mf": [0.5] * 6,
            "context_news": [0.0] * 6,
            "context_oh": [0.1] * 6,
            "context_br": [0.1] * 6,
            "context_hp": [0.3] * 6,
            "context_h": [0.5] * 6,
            "context_cap": [0.0] * 6,
            "context_theme_strength": [1.0] * 6,
            "context_theme_hp": [0.2] * 6,
            "context_cluster_id": [1, 1, 1, 2, 2, 2],
            "context_cluster_size": [10] * 6,
            "context_cluster_strength": [1.0] * 3 + [0.9] * 3,
            "context_cluster_mf": [0.5] * 6,
            "context_cluster_hp": [0.2] * 6,
        }
    )

    scores = strategy_score(predictions, variant="direct_theme_soft_cluster2")
    ordered = predictions.assign(score=scores).sort_values("score", ascending=False)["stock_code"].tolist()

    assert ordered[:4] == ["A1", "B1", "A2", "B2"]


def test_direction_accuracy_uses_half_threshold_for_direction_prob() -> None:
    predictions = pd.DataFrame(
        {
            "direction_prob": [0.9, 0.8, 0.4, 0.3],
            "label_next_open_return": [0.01, -0.01, -0.02, 0.02],
        }
    )

    assert direction_accuracy(predictions, score_col="direction_prob") == 0.5


def test_ensemble_predictions_aligns_by_sample_id_and_averages_daily_zscores() -> None:
    first = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "target_trade_date": [pd.Timestamp("2026-06-01")] * 3,
            "stock_code": ["A", "B", "C"],
            "y_score": [1.0, 2.0, 3.0],
            "final_score": [1.0, 2.0, 3.0],
            "return_pred": [0.1, 0.2, 0.3],
            "direction_prob": [0.2, 0.6, 0.8],
        }
    )
    second = pd.DataFrame(
        {
            "sample_id": ["c", "b", "a"],
            "target_trade_date": [pd.Timestamp("2026-06-01")] * 3,
            "stock_code": ["C", "B", "A"],
            "y_score": [30.0, 20.0, 10.0],
            "final_score": [30.0, 20.0, 10.0],
            "return_pred": [0.3, 0.2, 0.1],
            "direction_prob": [0.7, 0.5, 0.3],
        }
    )

    out = ensemble_predictions([first, second])

    assert out["sample_id"].tolist() == ["a", "b", "c"]
    assert out.loc[out["sample_id"].eq("c"), "final_score"].iloc[0] > out.loc[out["sample_id"].eq("a"), "final_score"].iloc[0]
    assert abs(out.loc[out["sample_id"].eq("b"), "direction_prob"].iloc[0] - 0.55) < 1e-12


def test_live_ensemble_filters_target_date_and_writes_buy_list(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "target_trade_date": pd.to_datetime(["2026-06-05", "2026-06-06"]),
            "sample_id": ["a", "b"],
        }
    )

    filtered = filter_target_date(predictions, pd.Timestamp("2026-06-06"))

    assert filtered["sample_id"].tolist() == ["b"]

    signals_path = tmp_path / "signals_20260606.csv"
    pd.DataFrame(
        {
            "target_trade_date": ["2026-06-06", "2026-06-06"],
            "stock_code": ["A", "B"],
            "suggested_action": ["buy", "watch"],
        }
    ).to_csv(signals_path, index=False)

    buy_list_path = write_buy_list(signals_path)
    buy_list = pd.read_csv(buy_list_path)

    assert buy_list_path.name == "buy_list_20260606.csv"
    assert buy_list["stock_code"].tolist() == ["A"]


def test_direction_accuracy_uses_median_threshold_for_scores() -> None:
    predictions = pd.DataFrame(
        {
            "y_score": [10.0, 9.0, 2.0, 1.0],
            "label_next_open_return": [0.01, -0.01, -0.02, 0.02],
        }
    )

    assert direction_accuracy(predictions, score_col="y_score") == 0.5


def test_backtest_outputs_nav_trades_positions_and_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "target_trade_date": pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3),
            "stock_code": ["A", "B", "C", "A", "B", "C"],
            "y_score": [3.0, 2.0, 1.0, 1.0, 3.0, 2.0],
            "label_next_open_return": [0.01, 0.02, -0.01, 0.00, 0.03, 0.01],
        }
    )

    nav, trades, positions, metrics = run_topk_backtest(
        predictions,
        StrategyParams(initial_cash=1_000_000, top_n=2, daily_replace_k=1, fee_rate=0.0, slippage_rate=0.0),
    )

    assert len(nav) == 2
    assert not trades.empty
    assert not positions.empty
    assert "annual_return" in metrics
    daily_orders = daily_orders_from_trades(trades)
    assert {"target_trade_date", "buy_stock_codes", "sell_stock_codes"} <= set(daily_orders.columns)
    assert daily_orders.iloc[0]["buy_count"] == 2


def test_competition_metrics_include_rolling_and_excess() -> None:
    dates = pd.bdate_range("2020-01-02", periods=12)
    predictions = pd.DataFrame(
        {
            "target_trade_date": [date for date in dates for _ in range(2)],
            "stock_code": ["A", "B"] * len(dates),
            "stock_name": ["Alpha", "Beta"] * len(dates),
            "y_score": [2.0, 1.0] * len(dates),
            "label_next_open_return": [0.01, 0.0] * len(dates),
        }
    )

    metrics = score_competition_predictions(
        predictions,
        StrategyParams(top_n=1, daily_replace_k=0, fee_rate=0.0, slippage_rate=0.0),
        window_days=10,
        recent_window_count=3,
    )

    assert metrics["day_count"] == 12
    assert metrics["rolling_win_rate"] == 1.0
    assert metrics["recent_window_count"] == 3
    assert metrics["period_excess_equal"] > 0


def test_backtest_cli_reads_predictions_and_writes_outputs(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "target_trade_date": pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3),
            "stock_code": ["A", "B", "C", "A", "B", "C"],
            "y_score": [3.0, 2.0, 1.0, 1.0, 3.0, 2.0],
            "label_next_open_return": [0.01, 0.02, -0.01, 0.00, 0.03, 0.01],
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    exit_code = backtest_main(
        [
            "--predictions-path",
            str(predictions_path),
            "--output-root",
            str(tmp_path),
            "--output-prefix",
            "case",
            "--top-n",
            "2",
            "--daily-replace-k",
            "1",
            "--fee-rate",
            "0",
            "--slippage-rate",
            "0",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "case_nav.csv").exists()
    assert (tmp_path / "case_trades.csv").exists()
    assert (tmp_path / "case_positions.csv").exists()
    assert (tmp_path / "case_daily_orders.csv").exists()
    daily_orders = pd.read_csv(tmp_path / "case_daily_orders.csv")
    assert "buy_stock_codes" in daily_orders.columns
    metrics = json.loads((tmp_path / "case_metrics.json").read_text(encoding="utf-8"))
    assert "total_return" in metrics


def test_read_predictions_rejects_unknown_extension(tmp_path) -> None:
    path = tmp_path / "predictions.txt"
    path.write_text("x", encoding="utf-8")

    try:
        read_predictions(path)
    except ValueError as exc:
        assert "Unsupported predictions file extension" in str(exc)
    else:
        raise AssertionError("read_predictions should reject unsupported extensions")


def test_filter_first_trade_days_and_ten_day_return(tmp_path, capsys) -> None:
    dates = pd.bdate_range("2020-01-02", periods=12)
    predictions = pd.DataFrame(
        {
            "target_trade_date": [date for date in dates for _ in range(2)],
            "stock_code": ["A", "B"] * len(dates),
            "y_score": [2.0, 1.0] * len(dates),
            "label_next_open_return": [0.01, 0.0] * len(dates),
        }
    )

    filtered = filter_first_trade_days(predictions, 10)
    assert filtered["target_trade_date"].nunique() == 10

    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    exit_code = backtest_main(
        [
            "--predictions-path",
            str(predictions_path),
            "--output-root",
            str(tmp_path),
            "--output-prefix",
            "ten_day",
            "--initial-cash",
            "1000000",
            "--top-n",
            "1",
            "--daily-replace-k",
            "0",
            "--fee-rate",
            "0",
            "--slippage-rate",
            "0",
            "--max-days",
            "10",
        ]
    )

    assert exit_code == 0
    nav = pd.read_csv(tmp_path / "ten_day_nav.csv")
    metrics = json.loads((tmp_path / "ten_day_metrics.json").read_text(encoding="utf-8"))
    assert len(nav) == 10
    assert metrics["initial_cash"] == 1_000_000.0
    assert metrics["day_count"] == 10.0
    assert metrics["ten_day_return"] == metrics["period_return"]
    assert metrics["final_nav"] > 1_000_000.0
    output = capsys.readouterr().out
    assert "period_return=" in output
    assert "ten_day_return=" in output


def test_rolling_backtest_windows_and_metrics(tmp_path, capsys) -> None:
    dates = pd.bdate_range("2020-01-02", periods=12)
    predictions = pd.DataFrame(
        {
            "target_trade_date": [date for date in dates for _ in range(2)],
            "stock_code": ["A", "B"] * len(dates),
            "y_score": [2.0, 1.0] * len(dates),
            "label_next_open_return": [0.01, 0.0] * len(dates),
        }
    )

    windows = rolling_trade_date_windows(predictions, window_days=10, step_days=1)
    assert len(windows) == 3
    assert windows[0][0] == pd.Timestamp("2020-01-02")
    assert windows[-1][-1] == dates[-1]

    nav, trades, positions, window_frame, metrics = run_rolling_topk_backtest(
        predictions,
        StrategyParams(initial_cash=1_000_000, top_n=1, daily_replace_k=0, fee_rate=0.0, slippage_rate=0.0),
        window_days=10,
    )
    assert window_frame["window_id"].tolist() == [1, 2, 3]
    assert len(nav) == 30
    assert not trades.empty
    assert not positions.empty
    assert metrics["window_count"] == 3.0
    assert metrics["win_rate"] == 1.0
    assert metrics["ten_day_return_mean"] == metrics["return_mean"]

    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    exit_code = backtest_main(
        [
            "--predictions-path",
            str(predictions_path),
            "--output-root",
            str(tmp_path),
            "--output-prefix",
            "rolling",
            "--initial-cash",
            "1000000",
            "--top-n",
            "1",
            "--daily-replace-k",
            "0",
            "--fee-rate",
            "0",
            "--slippage-rate",
            "0",
            "--rolling-window-days",
            "10",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "rolling_rolling_nav.csv").exists()
    assert (tmp_path / "rolling_rolling_trades.csv").exists()
    assert (tmp_path / "rolling_rolling_positions.csv").exists()
    assert (tmp_path / "rolling_rolling_daily_orders.csv").exists()
    assert (tmp_path / "rolling_rolling_windows.csv").exists()
    rolling_orders = pd.read_csv(tmp_path / "rolling_rolling_daily_orders.csv")
    assert {"window_id", "buy_stock_codes", "sell_stock_codes"} <= set(rolling_orders.columns)
    rolling_metrics = json.loads((tmp_path / "rolling_rolling_metrics.json").read_text(encoding="utf-8"))
    assert rolling_metrics["window_count"] == 3.0
    output = capsys.readouterr().out
    assert "rolling_window_count=3" in output
    assert "rolling_ten_day_return_mean=" in output
