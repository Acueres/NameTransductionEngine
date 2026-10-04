import re
import threading
import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterator

import requests

from name_transduction_engine.datasets.shared import build_session

DEFAULT_MIRROR: Final[str] = "https://dumps.wikimedia.org"
ENTITIES_PATH: Final[str] = "/wikidatawiki/entities/"
USER_AGENT: Final[str] = (
    "NameTransductionEngine/0.1 "
    "(+https://github.com/Acueres/NameTransductionEngine; dataset build)"
)

CHUNK_SIZE: Final[int] = 1024 * 1024  # bytes handed to the decoder at a time
RANGE_WINDOW: Final[int] = 64 * 1024 * 1024  # bytes per HTTP range request
# A run gives up after this long without receiving a single byte
MAX_STALL_SECONDS: Final[int] = 2 * 60 * 60
MIN_DUMP_SIZE: Final[int] = 10 * 1024**3  # a complete dump is ~100 GB

_DATE_DIR_RE: Final = re.compile(r'href="(\d{8})/"')
_SNAPSHOT_RE: Final = re.compile(r"(\d{8})")


class DumpUnavailableError(RuntimeError):
    """The dump is gone from the server (404/410)"""


class DumpChangedError(RuntimeError):
    """The file on the server is no longer the one the run started on"""


@dataclass(frozen=True)
class DumpInfo:
    url: str  # http(s) URL, or an absolute local path
    size: int
    etag: str | None
    last_modified: str | None
    snapshot: str  # YYYYMMDD of the dump, or "local-YYYYMMDD" for undated files

    @property
    def is_local(self) -> bool:
        return not self.url.startswith(("http://", "https://"))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "DumpInfo":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})

    def same_file(self, other: "DumpInfo") -> bool:
        if self.url != other.url or self.size != other.size:
            return False
        if self.etag and other.etag and self.etag != other.etag:
            return False
        return True


def make_session() -> requests.Session:
    session = build_session()
    session.headers["User-Agent"] = USER_AGENT
    return session


# Resolving a dump


def dated_dump_url(mirror: str, snapshot: str) -> str:
    base = mirror.rstrip("/") + ENTITIES_PATH
    return f"{base}{snapshot}/wikidata-{snapshot}-all.json.bz2"


def probe_dump(session: requests.Session, url: str) -> DumpInfo | None:
    """HEAD a dump URL. None if it does not exist"""
    response = session.head(url, allow_redirects=True, timeout=(15, 60))
    if response.status_code in (404, 410):
        return None
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    if not length:
        raise RuntimeError(f"{url}: server did not report a size")
    if response.headers.get("Accept-Ranges", "").lower() != "bytes":
        raise RuntimeError(f"{url}: server does not support range requests")
    match = _SNAPSHOT_RE.search(url.rsplit("/", 1)[-1])
    return DumpInfo(
        url=response.url,
        size=int(length),
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        snapshot=match.group(1) if match else "unknown",
    )


def resolve_latest_dump(
    session: requests.Session, mirror: str = DEFAULT_MIRROR, max_dirs: int = 10
) -> DumpInfo:
    """The newest complete dated JSON dump on the mirror"""
    index_url = mirror.rstrip("/") + ENTITIES_PATH
    response = session.get(index_url, timeout=(15, 60))
    response.raise_for_status()
    dates = sorted(set(_DATE_DIR_RE.findall(response.text)), reverse=True)
    if not dates:
        raise RuntimeError(f"no dated dump directories found at {index_url}")

    for snapshot in dates[:max_dirs]:
        info = probe_dump(session, dated_dump_url(mirror, snapshot))
        if info is not None and info.size >= MIN_DUMP_SIZE:
            return info
    raise RuntimeError(
        f"no complete JSON dump among the newest {max_dirs} directories at {index_url}"
    )


def local_dump(path: Path) -> DumpInfo:
    path = path.expanduser().resolve()
    stat = path.stat()
    match = _SNAPSHOT_RE.search(path.name)
    if match:
        snapshot = match.group(1)
    else:
        stamp = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        snapshot = "local-" + stamp.strftime("%Y%m%d")
    return DumpInfo(
        url=str(path),
        size=stat.st_size,
        etag=f"{stat.st_size}-{int(stat.st_mtime)}",
        last_modified=None,
        snapshot=snapshot,
    )


def refresh_info(session: requests.Session | None, info: DumpInfo) -> DumpInfo | None:
    """Current state of the dump a run started on; None if it disappeared"""
    if info.is_local:
        path = Path(info.url)
        return local_dump(path) if path.is_file() else None
    assert session is not None
    return probe_dump(session, info.url)


# Readers


def iter_chunks(
    info: DumpInfo,
    start: int,
    *,
    session: requests.Session | None = None,
    stop: threading.Event | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[tuple[int, bytes]]:
    if info.is_local:
        yield from _iter_local(info, start, stop, chunk_size)
    else:
        assert session is not None
        yield from _iter_remote(session, info, start, stop, chunk_size)


def _iter_local(
    info: DumpInfo, start: int, stop: threading.Event | None, chunk_size: int
) -> Iterator[tuple[int, bytes]]:
    with open(info.url, "rb") as f:
        f.seek(start)
        offset = start
        while True:
            if stop is not None and stop.is_set():
                return
            data = f.read(chunk_size)
            if not data:
                return
            yield offset, data
            offset += len(data)


_TRANSIENT = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
    requests.exceptions.RetryError,
    ConnectionError,
    TimeoutError,
)


def _iter_remote(
    session: requests.Session,
    info: DumpInfo,
    start: int,
    stop: threading.Event | None,
    chunk_size: int,
) -> Iterator[tuple[int, bytes]]:
    offset = start
    failures = 0
    last_progress = time.monotonic()

    while offset < info.size:
        if stop is not None and stop.is_set():
            return
        end = min(offset + RANGE_WINDOW, info.size) - 1
        headers = {"Range": f"bytes={offset}-{end}"}
        if info.etag:
            headers["If-Range"] = info.etag
        elif info.last_modified:
            headers["If-Range"] = info.last_modified

        try:
            with session.get(
                info.url, headers=headers, stream=True, timeout=(20, 120)
            ) as response:
                if response.status_code in (404, 410):
                    raise DumpUnavailableError(f"{info.url} is no longer available")
                if response.status_code == 200:
                    raise DumpChangedError(
                        f"{info.url} changed on the server (range request answered "
                        "with the whole file)"
                    )
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise DumpChangedError(
                        f"unexpected Content-Range {content_range!r} for offset {offset}"
                    )
                total = content_range.rpartition("/")[2]
                if total.isdigit() and int(total) != info.size:
                    raise DumpChangedError(
                        f"{info.url} changed size on the server "
                        f"({info.size} -> {total})"
                    )

                for data in response.iter_content(chunk_size=chunk_size):
                    if not data:
                        continue
                    if stop is not None and stop.is_set():
                        return
                    yield offset, data
                    offset += len(data)
                    failures = 0
                    last_progress = time.monotonic()
        except (DumpUnavailableError, DumpChangedError):
            raise
        except _TRANSIENT as exc:
            failures += 1
            if time.monotonic() - last_progress > MAX_STALL_SECONDS:
                raise RuntimeError(
                    f"no data from {info.url} for {MAX_STALL_SECONDS // 60} minutes "
                    f"(last error: {exc})"
                ) from exc
            wait = min(10 * 2 ** min(failures - 1, 5), 300)
            print(
                f"[wikidata] connection problem at byte {offset:,} ({exc}); "
                f"retrying in {wait}s",
                flush=True,
            )
            _sleep(wait, stop)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 500 <= status < 600 or status == 429:
                failures += 1
                if time.monotonic() - last_progress > MAX_STALL_SECONDS:
                    raise
                wait = min(30 * 2 ** min(failures - 1, 4), 600)
                print(
                    f"[wikidata] server error {status} at byte {offset:,}; "
                    f"retrying in {wait}s",
                    flush=True,
                )
                _sleep(wait, stop)
            else:
                raise


def _sleep(seconds: float, stop: threading.Event | None) -> None:
    if stop is None:
        time.sleep(seconds)
    else:
        stop.wait(seconds)


def read_range(
    info: DumpInfo,
    start: int,
    length: int,
    *,
    session: requests.Session | None = None,
) -> bytes:
    """`length` bytes from `start` (fewer at the end of the file)"""
    out = bytearray()
    for _, data in iter_chunks(info, start, session=session):
        out += data
        if len(out) >= length:
            break
    return bytes(out[:length])


def human_bytes(num: float) -> str:
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"
