"""Raw source files downloaded from upstream (GeoNames, the language-code
lists) and the record of what was downloaded.

Each raw folder keeps a `sources.json`: when its files were downloaded, and
each file's URL, size, SHA-256 and the server's Last-Modified date. From it:

- `source_id`: a short hash of the files' checksums. A dataset stores the
  source_id it was loaded from in `build_metadata`; when the files are
  downloaded again and differ, its readiness check fails and `nte init`
  reloads it. Without a record (files downloaded by an older NTE, or none
  yet) there is nothing to compare and the check is skipped.
- `sources_age_days`: how long ago the files were downloaded. GeoNames
  exports a new dump every day, so there is no "newer version" to look for;
  `nte update` downloads the files again once they are older than a
  maximum age instead.

A refresh downloads every file into a staging folder first and replaces the
old files only when all of them arrived, so a failed download never leaves
a mix of old and new files.
"""

import hashlib
import json
import os
import shutil

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Final

from .shared import build_session, get_and_save_file

SOURCES_FILE: Final[str] = "sources.json"
_STAGING_DIR: Final[str] = ".download"


def read_sources(raw_dir: Path) -> dict[str, Any] | None:
    path = raw_dir / SOURCES_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data.get("files"), dict) else None


def source_id(raw_dir: Path, names: list[str] | tuple[str, ...]) -> str | None:
    """Hash of the recorded checksums of `names`; None when the record does
    not cover them all or a file is missing or changed size"""
    record = read_sources(raw_dir)
    if record is None:
        return None
    entries = []
    for name in sorted(names):
        entry = record["files"].get(name)
        path = raw_dir / name
        if entry is None or not path.is_file() or path.stat().st_size != entry["bytes"]:
            return None
        entries.append(f"{name}={entry['sha256']}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]


def downloaded_at(raw_dir: Path, names: list[str] | tuple[str, ...]) -> datetime | None:
    """When the files were downloaded: the oldest of them. Without a record,
    the files' modification times. None when a file is missing"""
    record = read_sources(raw_dir) or {"files": {}}
    times = []
    for name in names:
        path = raw_dir / name
        if not path.is_file():
            return None
        entry = record["files"].get(name)
        if entry and entry.get("downloaded_at"):
            times.append(_parse_time(entry["downloaded_at"]))
        else:
            times.append(datetime.fromtimestamp(path.stat().st_mtime, UTC))
    return min(times) if times else None


def sources_age_days(
    raw_dir: Path, names: list[str] | tuple[str, ...], now: datetime | None = None
) -> float | None:
    when = downloaded_at(raw_dir, names)
    if when is None:
        return None
    now = now or datetime.now(UTC)
    return (now - when).total_seconds() / 86400


def snapshot_date(raw_dir: Path, name: str) -> str | None:
    """The upstream date of a file (YYYY-MM-DD): its Last-Modified header when
    recorded, else when it was downloaded"""
    record = read_sources(raw_dir)
    entry = (record or {"files": {}})["files"].get(name)
    if not entry:
        return None
    value = entry.get("last_modified") or entry.get("downloaded_at")
    return _parse_time(value).strftime("%Y-%m-%d") if value else None


def download_sources(
    raw_dir: Path, urls: dict[str, str], *, force: bool = False, label: str
) -> None:
    """Download the files that are missing, or all of them with `force`, and
    record them in sources.json"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = {
        name: url
        for name, url in urls.items()
        if force or not (raw_dir / name).is_file()
    }
    if not wanted:
        print(f"{label}: all files present.")
        return

    print(f"{label} download started ({', '.join(wanted)}).")
    staging = raw_dir / _STAGING_DIR
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    entries: dict[str, dict[str, Any]] = {}
    try:
        with build_session() as session:
            for name, url in wanted.items():
                print(f"Fetching {name}...")
                headers: dict[str, str] = {}
                path = get_and_save_file(
                    session=session,
                    url=url,
                    output_path=staging / name,
                    response_headers=headers,
                )
                entries[name] = {
                    "url": url,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "last_modified": _http_date(headers.get("Last-Modified")),
                    "downloaded_at": _now(),
                }
    except BaseException:
        # The old files stay as they were
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Every file arrived: replace the old ones together
    for name in wanted:
        os.replace(staging / name, raw_dir / name)
    shutil.rmtree(staging, ignore_errors=True)

    record = read_sources(raw_dir) or {"files": {}}
    record["files"].update(entries)
    # Drop entries of files that are gone or no longer downloaded
    record["files"] = {
        name: entry
        for name, entry in record["files"].items()
        if name in urls and (raw_dir / name).is_file()
    }
    tmp = raw_dir / (SOURCES_FILE + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, raw_dir / SOURCES_FILE)
    print(f"{label} download finished.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _http_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
