import json
import time
import requests
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIRECTORY: Final[Path] = Path("./data/raw/wikidata")
WIKIDATA_URL: Final[str] = (
    "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2"
)
CHUNK_SIZE: Final[int] = 8 * 1024 * 1024  # 8 MiB


@dataclass
class RemoteFileInfo:
    url: str
    size: int | None
    etag: str | None
    last_modified: str | None
    accept_ranges: bool


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "NameTransductionEngine/0.1 (+local data bootstrap)",
        }
    )

    retry_config = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry_config)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _human_bytes(num: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _probe_remote_file(session: requests.Session, url: str) -> RemoteFileInfo:
    response = session.head(url, allow_redirects=True, timeout=(10, 30))
    response.raise_for_status()

    content_length = response.headers.get("Content-Length")
    accept_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"

    return RemoteFileInfo(
        url=response.url,
        size=int(content_length) if content_length else None,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        accept_ranges=accept_ranges,
    )


def _save_meta(meta_path: Path, info: RemoteFileInfo) -> None:
    meta_path.write_text(
        json.dumps(
            {
                "url": info.url,
                "size": info.size,
                "etag": info.etag,
                "last_modified": info.last_modified,
                "accept_ranges": info.accept_ranges,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _same_remote_file(old_meta: dict | None, info: RemoteFileInfo) -> bool:
    if not old_meta:
        return False

    if old_meta.get("url") != info.url:
        return False

    old_etag = old_meta.get("etag")
    new_etag = info.etag
    if old_etag and new_etag and old_etag != new_etag:
        return False

    old_size = old_meta.get("size")
    new_size = info.size
    if old_size and new_size and old_size != new_size:
        return False

    old_modified = old_meta.get("last_modified")
    new_modified = info.last_modified
    if old_modified and new_modified and old_modified != new_modified:
        return False

    return True


def download_wikidata(force: bool = False) -> Path:
    print("Wikidata download started.")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    filename = WIKIDATA_URL.rsplit("/", 1)[-1]
    output_path = OUTPUT_DIRECTORY / filename
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")

    with _build_session() as session:
        info = _probe_remote_file(session=session, url=WIKIDATA_URL)

        if output_path.exists() and not force:
            local_size = output_path.stat().st_size
            if info.size is None or local_size == info.size:
                print(f"Wikidata already downloaded: {output_path.name}")
                return output_path

        if force:
            if output_path.exists():
                output_path.unlink()
            if part_path.exists():
                part_path.unlink()
            if meta_path.exists():
                meta_path.unlink()

        _save_meta(meta_path, info)

        _download_with_resume(
            session=session,
            info=info,
            output_path=output_path,
            part_path=part_path,
            meta_path=meta_path,
        )

    print("Wikidata download finished.")
    return output_path


def _print_progress(message: str, width: int = 120) -> None:
    padded = message.ljust(width)
    sys.stdout.write("\r" + padded)
    sys.stdout.flush()


def _download_with_resume(
    session: requests.Session,
    info: RemoteFileInfo,
    output_path: Path,
    part_path: Path,
    meta_path: Path,
) -> None:
    existing_meta = _load_meta(meta_path)
    existing_size = part_path.stat().st_size if part_path.exists() else 0

    can_resume = (
        existing_size > 0
        and info.accept_ranges
        and _same_remote_file(existing_meta, info)
        and (info.size is None or existing_size < info.size)
    )

    headers: dict[str, str] = {}
    mode = "wb"
    downloaded_before = 0

    if can_resume:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
        downloaded_before = existing_size
        print(f"Resuming from {_human_bytes(existing_size)}.")
    elif part_path.exists() and existing_size > 0:
        print("Existing partial file is not safely resumable. Restarting from scratch.")
        part_path.unlink()

    with session.get(
        info.url,
        headers=headers,
        stream=True,
        timeout=(10, 120),
    ) as response:
        response.raise_for_status()

        if downloaded_before:
            if response.status_code != 206:
                # Server ignored Range; restart cleanly.
                print("Server did not honor range request. Restarting from scratch.")
                downloaded_before = 0
                mode = "wb"
                if part_path.exists():
                    part_path.unlink()
            else:
                content_range = response.headers.get("Content-Range", "")
                expected_prefix = f"bytes {existing_size}-"
                if not content_range.startswith(expected_prefix):
                    raise IOError(
                        f"Unexpected Content-Range: {content_range!r}; "
                        f"expected prefix {expected_prefix!r}"
                    )

        total_size = info.size
        bytes_written_this_run = 0
        started_at = time.monotonic()
        last_report_at = started_at

        with part_path.open(mode) as output_file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue

                output_file.write(chunk)
                bytes_written_this_run += len(chunk)

                now = time.monotonic()
                if now - last_report_at >= 2.0:
                    current_size = downloaded_before + bytes_written_this_run
                    elapsed = max(now - started_at, 0.001)
                    speed = bytes_written_this_run / elapsed

                    if total_size:
                        pct = current_size / total_size * 100.0
                        remaining = max(total_size - current_size, 0)
                        eta = remaining / speed if speed > 0 else float("inf")
                        eta_text = (
                            f"{eta/3600:.1f}h" if eta != float("inf") else "unknown"
                        )
                        _print_progress(
                            f"{pct:6.2f}%  "
                            f"{_human_bytes(current_size)} / {_human_bytes(total_size)}  "
                            f"at {_human_bytes(speed)}/s  ETA {eta_text}"
                        )
                    else:
                        print()
                        print(
                            f"{_human_bytes(current_size)} downloaded  "
                            f"at {_human_bytes(speed)}/s"
                        )

                    last_report_at = now

        final_size = part_path.stat().st_size
        if total_size is not None and final_size != total_size:
            raise IOError(
                f"Incomplete download for {output_path.name}: "
                f"expected {total_size} bytes, got {final_size} bytes."
            )

    part_path.replace(output_path)
    print()
    print(f"Saved {output_path.name} ({_human_bytes(output_path.stat().st_size)}).")
