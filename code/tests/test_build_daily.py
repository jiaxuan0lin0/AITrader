from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from FactorMiner.build.daily import main


def test_build_daily_metric_filters_output_after_computing_history(tmp_path: Path) -> None:
    paths = _write_daily_inputs(tmp_path)
    output_root = tmp_path / "factors"
    registry_path = output_root / "factor_registry.json"

    exit_code = main(
        [
            "--block",
            "metric",
            "--metric-path",
            str(paths["metric"]),
            "--basic-path",
            str(paths["basic"]),
            "--output-root",
            str(output_root),
            "--registry-path",
            str(registry_path),
            "--since",
            "2024-01-02",
            "--until",
            "2024-01-02",
        ]
    )

    assert exit_code == 0
    block_path = output_root / "blocks" / "daily" / "manual_metric.parquet"
    manifest_path = output_root / "manifests" / "manual_metric.json"
    block = pd.read_parquet(block_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert block["trade_date"].dt.normalize().unique().tolist() == [pd.Timestamp("2024-01-02")]
    row_a = block.loc[block["stock_code"].eq("000001.SZ")].iloc[0]
    assert row_a["metric_pe_ttm_delta1"] == 2.0
    assert "metric_pe_ttm_delta1" in {record["name"] for record in manifest}
    assert registry[0]["name"] == "manual_metric"
    assert registry[0]["factor_path"] == "blocks/daily/manual_metric.parquet"


def test_build_daily_all_writes_registered_blocks(tmp_path: Path) -> None:
    paths = _write_daily_inputs(tmp_path)
    output_root = tmp_path / "factors"
    registry_path = output_root / "factor_registry.json"

    exit_code = main(
        [
            "--block",
            "all",
            "--price-path",
            str(paths["price"]),
            "--metric-path",
            str(paths["metric"]),
            "--moneyflow-path",
            str(paths["moneyflow"]),
            "--basic-path",
            str(paths["basic"]),
            "--output-root",
            str(output_root),
            "--registry-path",
            str(registry_path),
            "--alpha-workers",
            "2",
        ]
    )

    assert exit_code == 0
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    expected_names = [
        "manual_alpha158_kbar",
        "manual_alpha158_price",
        "manual_alpha158_return",
        "manual_alpha158_rolling3",
        "manual_alpha158_rolling5",
        "manual_alpha158_rolling10",
        "manual_alpha158_rolling20",
        "manual_alpha158_rolling60",
        "manual_metric",
        "manual_moneyflow",
    ]
    assert [record["name"] for record in registry] == expected_names

    for name in expected_names:
        block = pd.read_parquet(output_root / "blocks" / "daily" / f"{name}.parquet")
        manifest = json.loads((output_root / "manifests" / f"{name}.json").read_text(encoding="utf-8"))
        assert not block.empty
        assert manifest
        assert block.duplicated(["stock_code", "trade_date"]).sum() == 0

    moneyflow_block = pd.read_parquet(output_root / "blocks" / "daily" / "manual_moneyflow.parquet")
    assert "mf_main_net_amount_ratio_ind_neu" in moneyflow_block.columns
    assert not any(column.endswith("_x") or column.endswith("_y") for column in moneyflow_block.columns)


def test_build_daily_alpha_single_layout_keeps_legacy_block_name(tmp_path: Path) -> None:
    paths = _write_daily_inputs(tmp_path)
    output_root = tmp_path / "factors"
    registry_path = output_root / "factor_registry.json"

    exit_code = main(
        [
            "--block",
            "alpha158",
            "--alpha-layout",
            "single",
            "--price-path",
            str(paths["price"]),
            "--basic-path",
            str(paths["basic"]),
            "--output-root",
            str(output_root),
            "--registry-path",
            str(registry_path),
        ]
    )

    assert exit_code == 0
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert [record["name"] for record in registry] == ["manual_alpha158"]
    assert (output_root / "blocks" / "daily" / "manual_alpha158.parquet").exists()
    assert (output_root / "manifests" / "manual_alpha158.json").exists()


def test_build_daily_rejects_stock_limit_on_default_output_root(tmp_path: Path) -> None:
    paths = _write_daily_inputs(tmp_path)

    with pytest.raises(ValueError, match="stock-limit requires a non-default output root"):
        main(
            [
                "--block",
                "metric",
                "--metric-path",
                str(paths["metric"]),
                "--basic-path",
                str(paths["basic"]),
                "--stock-limit",
                "1",
            ]
        )


def _write_daily_inputs(tmp_path: Path) -> dict[str, Path]:
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"])
    rows = [
        ("000001.SZ", "bank", 10.0, 1000.0),
        ("000002.SZ", "tech", 20.0, 2000.0),
    ]

    price_records = []
    metric_records = []
    moneyflow_records = []
    for stock_code, industry, base_close, base_amount in rows:
        for offset, trade_date in enumerate(dates):
            close = base_close + offset
            price_records.append(
                {
                    "stock_code": stock_code,
                    "trade_date": trade_date,
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "vwap": close + 0.1,
                    "volume": 1000.0 + offset,
                    "amount": base_amount + offset * 10.0,
                    "industry": industry,
                }
            )
            metric_records.append(
                {
                    "stock_code": stock_code,
                    "trade_date": trade_date,
                    "pe_ttm": base_close + offset * 2.0,
                    "pb": 1.0 + offset,
                    "ps_ttm": 2.0 + offset,
                    "total_mv": 100.0 + base_close + offset,
                    "circ_mv": 80.0 + base_close + offset,
                    "turnover_rate": 1.0 + offset,
                    "turnover_rate_f": 0.8 + offset,
                    "volume_ratio": 0.5 + offset,
                    "industry": industry,
                }
            )
            moneyflow_records.append(
                {
                    "stock_code": stock_code,
                    "trade_date": trade_date,
                    "buy_sm_amount": 1.0 + offset,
                    "sell_sm_amount": 2.0 + offset,
                    "buy_lg_amount": 10.0 + offset,
                    "sell_lg_amount": 4.0 + offset,
                    "buy_elg_amount": 5.0 + offset,
                    "sell_elg_amount": 1.0 + offset,
                    "net_mf_amount": 8.0 + offset,
                    "industry": industry,
                }
            )

    paths = {
        "price": tmp_path / "price.parquet",
        "metric": tmp_path / "metric.parquet",
        "moneyflow": tmp_path / "moneyflow.parquet",
        "basic": tmp_path / "basic.parquet",
    }
    pd.DataFrame(price_records).to_parquet(paths["price"], index=False)
    pd.DataFrame(metric_records).to_parquet(paths["metric"], index=False)
    pd.DataFrame(moneyflow_records).to_parquet(paths["moneyflow"], index=False)
    pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000002.SZ"],
            "industry": ["bank", "tech"],
        }
    ).to_parquet(paths["basic"], index=False)
    return paths
