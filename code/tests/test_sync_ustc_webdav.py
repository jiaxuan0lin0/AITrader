from __future__ import annotations

from datetime import date
from pathlib import Path

from data.sync_ustc_webdav import RemoteItem, parse_date, resolve_date_window, select_items


def _remote(relative_path: str) -> RemoteItem:
    return RemoteItem(
        href_path=f"/seafdav/{relative_path}",
        relative_path=relative_path,
        is_dir=False,
        etag=None,
        size=None,
        modified=None,
        download_url=f"https://pan.ustc.edu.cn/seafdav/{relative_path}",
    )


def test_end_date_infers_start_from_latest_local_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    (raw_dir / "A股数据" / "daily").mkdir(parents=True)
    (raw_dir / "A股数据" / "daily" / "20260520.csv").write_text("local", encoding="utf-8")

    window = resolve_date_window(
        target_dir=raw_dir,
        start_date=None,
        end_date=parse_date("2026-05-22"),
        auto_start=False,
        local_fallback_to_mtime=False,
    )
    selected, skipped_without_date = select_items(
        [
            _remote("A股数据/daily/20260520.csv"),
            _remote("A股数据/daily/20260521.csv"),
            _remote("A股数据/daily/20260522.csv"),
            _remote("A股数据/daily/20260523.csv"),
            _remote("A股数据/daily/no_date.csv"),
        ],
        window,
        fallback_to_modified=False,
    )

    assert window.local_latest_date == date(2026, 5, 20)
    assert window.start_date == date(2026, 5, 21)
    assert window.end_date == date(2026, 5, 22)
    assert [entry.item.relative_path for entry in selected] == [
        "A股数据/daily/20260521.csv",
        "A股数据/daily/20260522.csv",
    ]
    assert skipped_without_date == 1


def test_explicit_start_date_overrides_local_auto_start(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    (raw_dir / "daily").mkdir(parents=True)
    (raw_dir / "daily" / "20260520.csv").write_text("local", encoding="utf-8")

    window = resolve_date_window(
        target_dir=raw_dir,
        start_date=parse_date("2026-05-19"),
        end_date=parse_date("2026-05-20"),
        auto_start=True,
        local_fallback_to_mtime=False,
    )

    assert window.start_date == date(2026, 5, 19)
    assert window.end_date == date(2026, 5, 20)
    assert window.local_latest_date is None
    assert not window.inferred_start
