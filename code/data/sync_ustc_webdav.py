#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote, urljoin, urlparse, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from aitrader_paths import RAW_MARKET_DATA_DIR


LOG = logging.getLogger("sync_ustc_webdav")
DAV_NS = {"d": "DAV:"}
MANIFEST_NAME = ".ustc_webdav_manifest.json"
DEFAULT_SOURCE_URL = "https://pan.ustc.edu.cn/seafdav/"
DEFAULT_TARGET_DIR = str(RAW_MARKET_DATA_DIR)
DATE_PATTERNS = (
    re.compile(r"(?<!\d)(20\d{2})[-_/\.]?([01]\d)[-_/\.]?([0-3]\d)(?!\d)"),
    re.compile(r"(?<!\d)([01]?\d)[-_/\.]([0-3]?\d)(?!\d)"),
)


class SyncError(RuntimeError):
    pass


@dataclass
class RemoteItem:
    href_path: str
    relative_path: str
    is_dir: bool
    etag: str | None
    size: int | None
    modified: str | None
    download_url: str


@dataclass(frozen=True)
class SelectedItem:
    item: RemoteItem
    item_date: date | None = None


@dataclass(frozen=True)
class DateWindow:
    start_date: date | None
    end_date: date | None
    local_latest_date: date | None = None
    inferred_start: bool = False


class WebDAVSyncer:
    def __init__(self, source_url: str, username: str, password: str, timeout: int = 60) -> None:
        if not username or not password:
            raise SyncError("USTC_WEBDAV_USERNAME and USTC_WEBDAV_PASSWORD are required.")
        self.source_url = self._ensure_trailing_slash(source_url)
        self.source_path = unquote(urlparse(self.source_url).path)
        self.origin = f"{urlparse(self.source_url).scheme}://{urlparse(self.source_url).netloc}"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.session.headers.update({"User-Agent": "AITrader-WebDAV-Sync/1.0"})

    @staticmethod
    def _ensure_trailing_slash(url: str) -> str:
        return url if url.endswith("/") else url + "/"

    def list_dir(self, directory_url: str) -> list[RemoteItem]:
        body = """<?xml version='1.0' encoding='UTF-8'?>
<d:propfind xmlns:d='DAV:'>
  <d:prop>
    <d:resourcetype/>
    <d:getcontentlength/>
    <d:getlastmodified/>
    <d:getetag/>
    <d:displayname/>
  </d:prop>
</d:propfind>
"""
        response = self.session.request(
            "PROPFIND",
            self._ensure_trailing_slash(directory_url),
            data=body.encode("utf-8"),
            headers={"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items: list[RemoteItem] = []
        for index, node in enumerate(root.findall("d:response", DAV_NS)):
            href = node.findtext("d:href", default="", namespaces=DAV_NS)
            if not href:
                continue
            href_path = unquote(urlparse(href).path if "://" in href else href)
            if index == 0:
                continue
            if not href_path.startswith(self.source_path.rstrip("/")):
                continue
            relative = href_path[len(self.source_path.rstrip("/")):].lstrip("/")
            if not relative:
                continue
            prop = node.find("d:propstat/d:prop", DAV_NS)
            if prop is None:
                continue
            is_dir = prop.find("d:resourcetype/d:collection", DAV_NS) is not None
            etag = prop.findtext("d:getetag", default=None, namespaces=DAV_NS)
            size_text = prop.findtext("d:getcontentlength", default=None, namespaces=DAV_NS)
            modified = prop.findtext("d:getlastmodified", default=None, namespaces=DAV_NS)
            size = int(size_text) if size_text and size_text.isdigit() else None
            items.append(
                RemoteItem(
                    href_path=href_path,
                    relative_path=relative.rstrip("/") if is_dir else relative,
                    is_dir=is_dir,
                    etag=etag,
                    size=size,
                    modified=modified,
                    download_url=quote_url_path(urljoin(self.origin, href if href.startswith("/") else urlparse(href).path)),
                )
            )
        return items

    def walk(self) -> list[RemoteItem]:
        collected: list[RemoteItem] = []
        stack = [self.source_url]
        while stack:
            current = stack.pop()
            for item in self.list_dir(current):
                collected.append(item)
                if item.is_dir:
                    stack.append(urljoin(self.origin, self._quote_href(item.href_path) + "/"))
        return collected

    @staticmethod
    def _quote_href(href_path: str) -> str:
        return quote(href_path, safe="/")

    def download(self, item: RemoteItem, target: Path, max_retries: int, retry_wait: float) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_name(target.name + ".part")
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                response = self.session.get(item.download_url, stream=True, timeout=self.timeout)
                response.raise_for_status()
                with temp_target.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                temp_target.replace(target)
                if item.modified:
                    try:
                        ts = parsedate_to_datetime(item.modified).timestamp()
                        os.utime(target, (ts, ts))
                    except Exception:
                        pass
                return
            except requests.RequestException as exc:
                last_error = exc
                if temp_target.exists():
                    temp_target.unlink(missing_ok=True)
                if attempt > max_retries:
                    break
                LOG.warning(
                    "Download failed for %s (attempt %d/%d): %s; retrying in %.1fs",
                    item.relative_path,
                    attempt,
                    max_retries + 1,
                    exc,
                    retry_wait,
                )
                time.sleep(retry_wait)
        assert last_error is not None
        raise last_error


def env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def env_bool(*names: str, default: bool = False) -> bool:
    value = env_first(*names)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_date(value: str) -> date:
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y.%m.%d", "%m-%d", "%m.%d", "%m/%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt.startswith("%m"):
                return parsed.replace(year=date.today().year).date()
            return parsed.date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Invalid date: {value}. Use YYYY-MM-DD, YYYYMMDD, or MM-DD.")


def parse_optional_date(value: str | None) -> date | None:
    return parse_date(value) if value else None


def quote_url_path(url: str) -> str:
    parts = urlsplit(url)
    quoted_path = quote(parts.path, safe="/%")
    return urlunsplit((parts.scheme, parts.netloc, quoted_path, parts.query, parts.fragment))


def infer_date(relative_path: str, modified: str | None = None, fallback_to_modified: bool = False) -> date | None:
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(relative_path):
            parts = match.groups()
            try:
                if len(parts[0]) == 4:
                    return date(int(parts[0]), int(parts[1]), int(parts[2]))
                return date(date.today().year, int(parts[0]), int(parts[1]))
            except ValueError:
                continue
    if fallback_to_modified and modified:
        try:
            return parsedate_to_datetime(modified).date()
        except Exception:
            return None
    return None


def infer_local_latest_date(target_dir: Path, fallback_to_mtime: bool = False) -> date | None:
    if not target_dir.exists():
        return None
    latest: date | None = None
    for path in target_dir.rglob("*"):
        if not path.is_file() or path.name == MANIFEST_NAME or path.suffix == ".part":
            continue
        relative_path = path.relative_to(target_dir).as_posix()
        item_date = infer_date(relative_path)
        if item_date is None and fallback_to_mtime:
            item_date = datetime.fromtimestamp(path.stat().st_mtime).date()
        if item_date is not None and (latest is None or item_date > latest):
            latest = item_date
    return latest


def resolve_date_window(
    target_dir: Path,
    start_date: date | None,
    end_date: date | None,
    auto_start: bool,
    local_fallback_to_mtime: bool,
) -> DateWindow:
    local_latest_date: date | None = None
    inferred_start = False
    if start_date is None and (auto_start or end_date is not None):
        local_latest_date = infer_local_latest_date(target_dir, fallback_to_mtime=local_fallback_to_mtime)
        if local_latest_date is not None:
            start_date = local_latest_date + timedelta(days=1)
            inferred_start = True
    if start_date is not None and end_date is not None and start_date > end_date and not inferred_start:
        raise SyncError(f"start-date {start_date.isoformat()} is later than end-date {end_date.isoformat()}.")
    return DateWindow(
        start_date=start_date,
        end_date=end_date,
        local_latest_date=local_latest_date,
        inferred_start=inferred_start,
    )


def has_date_filter(window: DateWindow) -> bool:
    return window.start_date is not None or window.end_date is not None


def select_items(
    items: Iterable[RemoteItem],
    window: DateWindow,
    fallback_to_modified: bool,
) -> tuple[list[SelectedItem], int]:
    if not has_date_filter(window):
        return [SelectedItem(item=item) for item in items], 0

    selected: list[SelectedItem] = []
    skipped_without_date = 0
    for item in items:
        if item.is_dir:
            continue
        item_date = infer_date(item.relative_path, item.modified, fallback_to_modified=fallback_to_modified)
        if item_date is None:
            skipped_without_date += 1
            continue
        if window.start_date is not None and item_date < window.start_date:
            continue
        if window.end_date is not None and item_date > window.end_date:
            continue
        selected.append(SelectedItem(item=item, item_date=item_date))
    selected.sort(key=lambda entry: (entry.item_date or date.min, entry.item.relative_path))
    return selected, skipped_without_date


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def should_download(item: RemoteItem, target: Path, old_manifest: dict[str, Any]) -> bool:
    if item.is_dir:
        return False
    return not target.exists()


def prune_deleted(target_dir: Path, remote_files: set[str], manifest_path: Path) -> None:
    for path in target_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(target_dir).as_posix()
        if relative not in remote_files:
            path.unlink()
    for path in sorted(target_dir.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if manifest_path.exists() and MANIFEST_NAME not in remote_files:
        pass


def run_sync(args: argparse.Namespace) -> None:
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / MANIFEST_NAME
    old_manifest = load_manifest(manifest_path)
    if manifest_path.exists():
        LOG.info("Loaded manifest: %s (entries=%d)", manifest_path, len(old_manifest))
    else:
        LOG.warning("Manifest not found at %s; this run will still skip files that already exist locally.", manifest_path)

    syncer = WebDAVSyncer(args.source_url, args.username, args.password, timeout=args.timeout)
    items = syncer.walk()
    remote_files = {item.relative_path for item in items if not item.is_dir}
    window = resolve_date_window(
        target_dir=target_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        auto_start=args.auto_start,
        local_fallback_to_mtime=args.local_fallback_to_mtime,
    )
    selected_items, skipped_without_date = select_items(
        items,
        window,
        fallback_to_modified=args.fallback_to_modified,
    )
    date_filtered = has_date_filter(window)
    checkpoint_manifest: dict[str, Any] = dict(old_manifest)
    final_manifest: dict[str, Any] = dict(old_manifest) if date_filtered else {}

    if window.inferred_start:
        LOG.info(
            "Auto start date resolved from local latest file date: local_latest=%s start=%s",
            window.local_latest_date.isoformat() if window.local_latest_date else "none",
            window.start_date.isoformat() if window.start_date else "none",
        )
    if date_filtered:
        LOG.info(
            "Date window: start=%s end=%s selected_files=%d skipped_without_date=%d",
            window.start_date.isoformat() if window.start_date else "-",
            window.end_date.isoformat() if window.end_date else "-",
            len(selected_items),
            skipped_without_date,
        )

    downloaded = 0
    skipped = 0
    reused_local = 0
    for entry in selected_items:
        item = entry.item
        target = target_dir / item.relative_path
        if item.is_dir:
            target.mkdir(parents=True, exist_ok=True)
            continue
        needs_download = should_download(item, target, old_manifest)
        if needs_download:
            date_label = f"[{entry.item_date.isoformat()}] " if entry.item_date else ""
            LOG.info("Downloading %s%s", date_label, item.relative_path)
            try:
                syncer.download(item, target, max_retries=args.max_retries, retry_wait=args.retry_wait)
                downloaded += 1
            except requests.RequestException as exc:
                if target.exists():
                    reused_local += 1
                    LOG.warning(
                        "Download failed for %s but local file exists; keeping local copy and continuing: %s",
                        item.relative_path,
                        exc,
                    )
                    record = checkpoint_manifest.get(item.relative_path) or old_manifest.get(item.relative_path)
                    if record:
                        final_manifest[item.relative_path] = record
                    save_manifest(manifest_path, checkpoint_manifest)
                    continue
                raise
        else:
            skipped += 1

        record = {
            "etag": item.etag,
            "size": item.size,
            "modified": item.modified,
        }
        checkpoint_manifest[item.relative_path] = record
        final_manifest[item.relative_path] = record
        save_manifest(manifest_path, checkpoint_manifest)

    if args.delete:
        prune_deleted(target_dir, remote_files, manifest_path)

    save_manifest(manifest_path, final_manifest)
    LOG.info(
        "Sync finished: downloaded=%d skipped=%d reused_local=%d selected_files=%d total_remote_files=%d",
        downloaded,
        skipped,
        reused_local,
        sum(1 for entry in selected_items if not entry.item.is_dir),
        len(remote_files),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync USTC Pan WebDAV folder to local directory.")
    parser.add_argument("--source-url", default=env_first("A_SHARE_SYNC_URL", "USTC_WEBDAV_URL", "DATASET_CLOUD_URL", default=DEFAULT_SOURCE_URL))
    parser.add_argument("--target-dir", default=env_first("A_SHARE_SYNC_TARGET", "USTC_WEBDAV_TARGET", "DATASET_CLOUD_TARGET", default=DEFAULT_TARGET_DIR))
    parser.add_argument("--username", default=env_first("A_SHARE_SYNC_USERNAME", "USTC_WEBDAV_USERNAME", "DATASET_CLOUD_USERNAME", default=""))
    parser.add_argument("--password", default=env_first("A_SHARE_SYNC_PASSWORD", "USTC_WEBDAV_PASSWORD", "DATASET_CLOUD_PASSWORD", default=""))
    parser.add_argument("--timeout", type=int, default=int(env_first("A_SHARE_SYNC_TIMEOUT", "USTC_WEBDAV_TIMEOUT", "DATASET_CLOUD_TIMEOUT", default="60") or "60"))
    parser.add_argument("--max-retries", type=int, default=int(env_first("A_SHARE_SYNC_MAX_RETRIES", "USTC_WEBDAV_MAX_RETRIES", "DATASET_CLOUD_MAX_RETRIES", default="3") or "3"))
    parser.add_argument("--retry-wait", type=float, default=float(env_first("A_SHARE_SYNC_RETRY_WAIT", "USTC_WEBDAV_RETRY_WAIT", "DATASET_CLOUD_RETRY_WAIT", default="2") or "2"))
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=parse_optional_date(env_first("A_SHARE_SYNC_START_DATE", "USTC_WEBDAV_START_DATE", "DATASET_CLOUD_START_DATE")),
        help="Only sync files whose inferred date is on or after this date.",
    )
    parser.add_argument(
        "--end-date",
        "--latest-date",
        dest="end_date",
        type=parse_date,
        default=parse_optional_date(env_first("A_SHARE_SYNC_END_DATE", "USTC_WEBDAV_END_DATE", "DATASET_CLOUD_END_DATE")),
        help="Only sync files whose inferred date is on or before this date. If start-date is omitted, local latest date + 1 is used.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        default=env_bool("A_SHARE_SYNC_AUTO_START", "USTC_WEBDAV_AUTO_START", "DATASET_CLOUD_AUTO_START"),
        help="Infer start-date from the latest dated local file and begin at the following day.",
    )
    parser.add_argument(
        "--fallback-to-modified",
        action="store_true",
        default=env_bool("A_SHARE_SYNC_FALLBACK_TO_MODIFIED", "USTC_WEBDAV_FALLBACK_TO_MODIFIED", "DATASET_CLOUD_FALLBACK_TO_MODIFIED"),
        help="Use WebDAV last-modified date when a remote file path does not contain a date.",
    )
    parser.add_argument(
        "--local-fallback-to-mtime",
        action="store_true",
        default=env_bool("A_SHARE_SYNC_LOCAL_FALLBACK_TO_MTIME", "USTC_WEBDAV_LOCAL_FALLBACK_TO_MTIME", "DATASET_CLOUD_LOCAL_FALLBACK_TO_MTIME"),
        help="Use local file mtime when inferring the local latest date and the local path has no date.",
    )
    parser.add_argument("--delete", action="store_true", help="Delete local files that no longer exist remotely.")
    parser.add_argument("--log-level", default=env_first("A_SHARE_SYNC_LOG_LEVEL", "USTC_WEBDAV_LOG_LEVEL", "DATASET_CLOUD_LOG_LEVEL", default="INFO"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        run_sync(args)
    except SyncError as exc:
        LOG.error("%s", exc)
        return 2
    except requests.RequestException as exc:
        LOG.error("WebDAV request failed: %s", exc)
        return 3
    except ET.ParseError as exc:
        LOG.error("Failed to parse WebDAV XML: %s", exc)
        return 4
    except Exception:
        LOG.exception("Unexpected sync failure.")
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
