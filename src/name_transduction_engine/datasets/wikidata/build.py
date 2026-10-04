"""`nte data build wikidata-compact`: stream the Wikidata dump into the
compact dataset, resumably.

    download (HTTP ranges) -> bzip2 decode -> lines -> prefilter -> extract
                                                         -> shard files

Work directory (`data/build/wikidata/work`):

    state.json         the checkpoint: dump identity, where to resume, counters
    shards/            shard-000001.jsonl.gz ... (records that passed the
                       lenient extraction gate)

Every few minutes (and on Ctrl-C or SIGTERM) the open shard is closed and
state.json is rewritten atomically, so an interruption loses at most one
interval. The next run with the same command continues from there: it
re-reads a few MiB of the dump before the checkpoint, decodes from a block
boundary (bz2_resume.py) and skips to the line after the last one processed,
found by its first bytes.

When the dump is finished, the final gates are applied to the shards and the
compact dataset is written to `data/build/wikidata/compact` (compact.py).
Shards are kept, so the final gates can be changed and re-applied with
`--refinalize` without reading the dump again.
"""

import base64
import gzip
import json
import os
import queue
import shutil
import signal
import threading
import time
import orjson

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any, Final, Iterator

from name_transduction_engine.paths import (
    WIKIDATA_COMPACT_DIR,
    WIKIDATA_WORK_DIR,
)
from . import classes as C
from .bz2_resume import Bz2DataError, decode, find_block_starts
from .compact import format_manifest_summary, write_compact
from .dump_source import (
    DEFAULT_MIRROR,
    DumpChangedError,
    DumpInfo,
    DumpUnavailableError,
    human_bytes,
    iter_chunks,
    local_dump,
    make_session,
    probe_dump,
    read_range,
    refresh_info,
    resolve_latest_dump,
)
from .extract import EXTRACTOR_VERSION, Extractor

STATE_FORMAT: Final[int] = 1
STATE_NAME: Final[str] = "state.json"
SHARDS_DIR: Final[str] = "shards"
LOCK_NAME: Final[str] = "build.lock"

PREFIX_BYTES: Final[int] = 96  # bytes of the last processed line kept to find it again
RESUME_MARGIN: Final[int] = 8 * 1024 * 1024  # re-read this much before the checkpoint
RESUME_WINDOW: Final[int] = 16 * 1024 * 1024  # bytes searched for a block start
MAX_SEEK_BYTES: Final[int] = 512 * 1024 * 1024  # decoded bytes searched for the line
PROGRESS_SECONDS: Final[float] = 60.0
QUEUE_SIZE: Final[int] = 4


@dataclass
class BuildOptions:
    source: str | None = None  # dump URL or local .bz2 path; None = newest dated dump
    mirror: str = DEFAULT_MIRROR
    restart: bool = False
    stop_after_gb: float | None = None
    partial: bool = False  # also write a compact dataset when stopping early
    refinalize: bool = False
    checkpoint_minutes: float = 10.0
    work_dir: Path = WIKIDATA_WORK_DIR
    compact_dir: Path = WIKIDATA_COMPACT_DIR


class BuildError(RuntimeError):
    pass


def build_wikidata_compact(options: BuildOptions | None = None) -> int:
    """Returns a process exit code"""
    options = options or BuildOptions()
    options.work_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(options.work_dir / LOCK_NAME):
        if options.refinalize:
            return _refinalize(options)
        return _build(options)


# State


def read_state(work_dir: Path = WIKIDATA_WORK_DIR) -> dict[str, Any] | None:
    path = work_dir / STATE_NAME
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if state.get("format") == STATE_FORMAT else None


def _write_state(work_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    path = work_dir / STATE_NAME
    tmp = path.with_suffix(".json.part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _new_state(dump: DumpInfo, class_map: C.ClassMap) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "status": "running",
        "dump": dump.to_json(),
        "extractor_version": EXTRACTOR_VERSION,
        "extract_gates": C.extract_gates_fingerprint(),
        "classes_sha256": class_map.sha256,
        "shards": 0,
        "resume": None,
        "position": 0,
        "lines": 0,
        "candidates": 0,
        "stats": {},
        "active_seconds": 0.0,
        "started_at": _now(),
        "updated_at": _now(),
    }


def _incompatibility(state: dict[str, Any], class_map: C.ClassMap) -> str | None:
    if state.get("extractor_version") != EXTRACTOR_VERSION:
        return "the extractor changed"
    if state.get("extract_gates") != C.extract_gates_fingerprint():
        return "the extraction gates changed"
    if state.get("classes_sha256") != class_map.sha256:
        return "the class file changed"
    return None


def _shard_dir(work_dir: Path) -> Path:
    return work_dir / SHARDS_DIR


def committed_shards(work_dir: Path, state: dict[str, Any]) -> list[Path]:
    shard_dir = _shard_dir(work_dir)
    paths = [_shard_path(shard_dir, i) for i in range(1, state.get("shards", 0) + 1)]
    return [p for p in paths if p.is_file()]


def _shard_path(shard_dir: Path, index: int) -> Path:
    return shard_dir / f"shard-{index:06d}.jsonl.gz"


def _reset_work_dir(work_dir: Path) -> None:
    shard_dir = _shard_dir(work_dir)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    state = work_dir / STATE_NAME
    if state.exists():
        state.unlink()


def _remove_uncommitted(work_dir: Path, state: dict[str, Any]) -> None:
    shard_dir = _shard_dir(work_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    keep = {_shard_path(shard_dir, i).name for i in range(1, state["shards"] + 1)}
    for path in shard_dir.iterdir():
        if path.name not in keep:
            path.unlink()


# Choosing what to do


def _build(options: BuildOptions) -> int:
    class_map = C.load_class_map()
    session = make_session()
    work_dir = options.work_dir
    state = None if options.restart else read_state(work_dir)

    explicit = _explicit_dump(options, session)

    if state is not None:
        reason = _incompatibility(state, class_map)
        old_dump = DumpInfo.from_json(state["dump"])
        if reason is not None:
            print(f"[wikidata] {reason} since the last run; starting over.")
            state = None
        elif explicit is not None and not explicit.same_file(old_dump):
            print("[wikidata] a different dump was requested; starting over.")
            state = None
        elif state["status"] == "finished":
            latest = explicit or resolve_latest_dump(session, options.mirror)
            if latest.snapshot == old_dump.snapshot and latest.same_file(old_dump):
                print(
                    f"[wikidata] compact dataset for dump {old_dump.snapshot} is "
                    "up to date; nothing to do."
                )
                return 0
            print(
                f"[wikidata] new dump {latest.snapshot} "
                f"(last build: {old_dump.snapshot}); starting a new run."
            )
            state = None
            explicit = latest
        elif state["status"] == "running":
            current = refresh_info(None if old_dump.is_local else session, old_dump)
            if current is None:
                print(
                    f"[wikidata] dump {old_dump.snapshot} is no longer available; "
                    "starting over with the newest dump."
                )
                state = None
            elif not current.same_file(old_dump):
                print(
                    f"[wikidata] dump {old_dump.snapshot} changed on the server; "
                    "starting over."
                )
                state = None

    if state is not None and state["status"] == "read":
        print("[wikidata] the dump was fully read; writing the compact dataset.")
        _finalize(options, state, class_map, partial=False)
        state["status"] = "finished"
        _write_state(work_dir, state)
        return 0

    if state is None:
        dump = explicit or resolve_latest_dump(session, options.mirror)
        _reset_work_dir(work_dir)
        state = _new_state(dump, class_map)
        _shard_dir(work_dir).mkdir(parents=True, exist_ok=True)
        _write_state(work_dir, state)
        print(
            f"[wikidata] new run on dump {dump.snapshot} "
            f"({human_bytes(dump.size)}): {dump.url}"
        )
    else:
        _remove_uncommitted(work_dir, state)
        dump = DumpInfo.from_json(state["dump"])
        print(
            f"[wikidata] resuming dump {dump.snapshot} at "
            f"{state['position'] / dump.size:.1%} "
            f"({state['shards']} shards, {state['lines']:,} lines done)"
        )

    run = _Run(options, state, dump, class_map, session)
    try:
        outcome = run.execute()
    except (DumpUnavailableError, DumpChangedError) as exc:
        print(f"[wikidata] {exc}. The next run will start over on the newest dump.")
        return 1

    if outcome == "finished":
        state["status"] = "read"  # a crash during finalize resumes here
        _write_state(work_dir, state)
        _finalize(options, state, class_map, partial=False)
        state["status"] = "finished"
        _write_state(work_dir, state)
        return 0
    if outcome == "interrupted":
        return 130
    # stopped early (--stop-after-gb)
    if options.partial:
        _finalize(options, state, class_map, partial=True)
    return 0


def _explicit_dump(options: BuildOptions, session) -> DumpInfo | None:
    if not options.source:
        return None
    source = options.source
    if source.startswith(("http://", "https://")):
        info = probe_dump(session, source)
        if info is None:
            raise BuildError(f"dump not found: {source}")
        return info
    path = Path(source)
    if not path.is_file():
        raise BuildError(f"dump file not found: {path}")
    return local_dump(path)


def _refinalize(options: BuildOptions) -> int:
    state = read_state(options.work_dir)
    if state is None or not committed_shards(options.work_dir, state):
        print("[wikidata] no build shards to refinalize; run a build first.")
        return 1
    class_map = C.load_class_map()
    reason = _incompatibility(state, class_map)
    if reason is not None:
        print(
            f"[wikidata] note: {reason} since the shards were built; the final "
            "gates are applied to the shards as they are."
        )
    _finalize(options, state, class_map, partial=state["status"] != "finished")
    return 0


def _finalize(
    options: BuildOptions,
    state: dict[str, Any],
    class_map: C.ClassMap,
    *,
    partial: bool,
) -> None:
    shards = committed_shards(options.work_dir, state)
    dump = state["dump"]
    progress = state["position"] / dump["size"] if dump["size"] else 0.0
    what = "partial compact dataset" if partial else "compact dataset"
    print(f"[wikidata] writing {what} from {len(shards)} shards...")
    started = time.monotonic()
    manifest = write_compact(
        shards,
        options.compact_dir,
        dump=dump,
        classes=class_map,
        extraction_stats=dict(state.get("stats", {})),
        partial=partial,
        progress=progress,
    )
    print(
        f"[wikidata] {what} written to {options.compact_dir} "
        f"in {time.monotonic() - started:.0f}s"
        + (f" (covers {progress:.1%} of the dump)" if partial else "")
    )
    for line in format_manifest_summary(manifest):
        print(f"  {line}")
    print("Load it into names.sqlite with `nte init`.")


# The streaming run

_END = object()


class _Run:
    def __init__(
        self,
        options: BuildOptions,
        state: dict[str, Any],
        dump: DumpInfo,
        class_map: C.ClassMap,
        session,
    ) -> None:
        self.options = options
        self.state = state
        self.dump = dump
        self.session = session
        self.work_dir = options.work_dir
        self.shard_dir = _shard_dir(options.work_dir)
        self.extractor = Extractor(class_map)
        self.extractor.stats.update(state.get("stats", {}))

        self.stop = threading.Event()
        self.interrupted = False
        self.stopped_early = False

        self._writer: gzip.GzipFile | None = None
        self._writer_path: Path | None = None
        self._records_in_shard = 0

        self._last_line_pos: int | None = None
        self._last_line_prefix: bytes | None = None
        self._lines_since_commit = 0
        self._position = state["position"]
        self._session_start_pos = state["position"]

    # Entry point

    def execute(self) -> str:
        """'finished', 'stopped' (stop-after) or 'interrupted'"""
        stream, prefix = self._open_stream()
        q: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        producer = threading.Thread(
            target=self._produce, args=(stream, q), name="wikidata-reader", daemon=True
        )

        started = time.monotonic()
        self._last_commit = started
        self._last_progress = started
        self._progress_mark = (started, self._position, 0)
        finished = False

        with self._signals():
            producer.start()
            try:
                finished = self._consume(q, prefix)
            finally:
                self.stop.set()
                self._commit()
                self.state["active_seconds"] = round(
                    self.state.get("active_seconds", 0.0) + time.monotonic() - started,
                    1,
                )
                _write_state(self.work_dir, self.state)
                producer.join(timeout=5)

        if finished:
            print(
                f"[wikidata] dump fully read: {self.state['lines']:,} lines, "
                f"{self.state['candidates']:,} parsed, "
                + ", ".join(
                    f"{k.split(':', 1)[1]} {v:,}"
                    for k, v in sorted(self.extractor.stats.items())
                    if k.startswith("kept:")
                )
            )
            return "finished"
        if self.interrupted:
            print("[wikidata] stopped; progress saved. Run the same command to resume.")
            return "interrupted"
        print("[wikidata] stopped after the requested amount; progress saved.")
        return "stopped"

    @contextmanager
    def _signals(self):
        def handler(signum, frame):
            if self.interrupted:
                raise KeyboardInterrupt
            self.interrupted = True
            self.stop.set()
            print("\n[wikidata] stopping after the current chunk...", flush=True)

        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, handler)
        try:
            yield
        finally:
            for sig, old in previous.items():
                signal.signal(sig, old)

    # Producer thread

    def _produce(self, stream: Iterator[tuple[bytes, int]], q: queue.Queue) -> None:
        try:
            for item in stream:
                while not self.stop.is_set():
                    try:
                        q.put(item, timeout=1)
                        break
                    except queue.Full:
                        continue
                if self.stop.is_set():
                    return
            q.put(_END)
        except BaseException as exc:  # handed to the consumer
            q.put(exc)

    def _open_stream(self) -> tuple[Iterator[tuple[bytes, int]], bytes | None]:
        resume = self.state.get("resume")
        session = None if self.dump.is_local else self.session
        if not resume:
            return (
                decode(iter_chunks(self.dump, 0, session=session, stop=self.stop)),
                None,
            )

        prefix = base64.b64decode(resume["prefix"])
        search_start = max(0, resume["pos"] - RESUME_MARGIN)
        if search_start == 0:
            return (
                decode(iter_chunks(self.dump, 0, session=session, stop=self.stop)),
                prefix,
            )

        window = read_range(self.dump, search_start, RESUME_WINDOW, session=session)
        candidates = [b for b in find_block_starts(window, search_start)]
        for bit in candidates[:32]:
            stream = decode(
                iter_chunks(self.dump, bit // 8, session=session, stop=self.stop),
                bit,
                spliced=True,
            )
            try:
                first = next(stream)
            except (Bz2DataError, StopIteration):
                stream.close()
                continue
            return chain([first], stream), prefix
        raise BuildError(
            f"could not find a bzip2 block near byte {resume['pos']:,}; "
            "run with --restart"
        )

    # Consumer (main thread)

    def _consume(self, q: queue.Queue, prefix: bytes | None) -> bool:
        """True when the whole dump was read"""
        buf = b""
        buf_pos = self._position
        seeking = prefix is not None
        seek_buf = b""
        seek_at_stream_start = True
        seeked = 0
        skipping_rest_of_line = False

        while True:
            try:
                item = q.get(timeout=1)
            except queue.Empty:
                if self.stop.is_set():
                    return False
                continue
            if item is _END:
                break
            if isinstance(item, BaseException):
                raise item
            data, pos = item
            self._position = pos

            if seeking:
                seek_buf += data
                seeked += len(data)
                idx = _find_line_prefix(seek_buf, prefix, seek_at_stream_start)
                if idx < 0:
                    if seeked > MAX_SEEK_BYTES:
                        raise BuildError(
                            "could not find the checkpoint line again; "
                            "run with --restart"
                        )
                    keep = len(prefix) + 1
                    if len(seek_buf) > keep:
                        seek_buf = seek_buf[-keep:]
                        seek_at_stream_start = False
                    continue
                seeking = False
                data = seek_buf[idx:]
                seek_buf = b""
                skipping_rest_of_line = True

            if skipping_rest_of_line:
                nl = data.find(b"\n")
                if nl < 0:
                    continue
                data = data[nl + 1 :]
                skipping_rest_of_line = False
                buf, buf_pos = b"", pos

            if buf:
                data = buf + data
                first_pos = buf_pos
            else:
                first_pos = pos
            lines = data.split(b"\n")
            buf = lines.pop()
            for i, line in enumerate(lines):
                self._process_line(line, first_pos if i == 0 else pos)
            buf_pos = pos if lines else first_pos

            now = time.monotonic()
            if now - self._last_progress >= PROGRESS_SECONDS:
                self._report(now)
            if now - self._last_commit >= self.options.checkpoint_minutes * 60:
                self._commit()
            if self._should_stop_early():
                self.stopped_early = True
                self.stop.set()
                return False
            if self.interrupted:
                return False

        if seeking:
            raise BuildError(
                "reached the end of the dump while looking for the checkpoint"
            )
        if buf.strip():
            self._process_line(buf, buf_pos)
        return True

    def _should_stop_early(self) -> bool:
        limit = self.options.stop_after_gb
        if limit is None:
            return False
        return self._position - self._session_start_pos >= limit * 1024**3

    def _process_line(self, line: bytes, pos: int) -> None:
        self.state["lines"] += 1
        self._lines_since_commit += 1
        self._last_line_pos = pos
        self._last_line_prefix = line[:PREFIX_BYTES]

        if len(line) < 16 or line[0:1] != b"{":
            return  # "[" / "]" lines of the JSON array
        if line.endswith(b","):
            line = line[:-1]
        extractor = self.extractor
        if not extractor.prefilter(line):
            return
        self.state["candidates"] += 1
        try:
            record = extractor.extract_line(line)
        except (orjson.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            extractor.stats["skip:bad_entity"] += 1
            if extractor.stats["skip:bad_entity"] <= 20:
                print(f"[wikidata] skipped an unreadable entity ({exc}): {line[:80]!r}")
            return
        if record is not None:
            self._write(orjson.dumps(record) + b"\n")

    # Shards

    def _write(self, data: bytes) -> None:
        if self._writer is None:
            index = self.state["shards"] + 1
            self._writer_path = _shard_path(self.shard_dir, index)
            part = self._writer_path.with_name(self._writer_path.name + ".part")
            self._raw = open(part, "wb")
            self._writer = gzip.GzipFile(
                filename="", mode="wb", fileobj=self._raw, compresslevel=5, mtime=0
            )
        self._writer.write(data)
        self._records_in_shard += 1

    def _commit(self) -> None:
        """Close the open shard and move the resume point to the last
        processed line. Lines and their records become durable together"""
        self._last_commit = time.monotonic()
        if self._lines_since_commit == 0:
            return
        if self._writer is not None:
            self._writer.close()
            self._raw.flush()
            os.fsync(self._raw.fileno())
            self._raw.close()
            part = self._writer_path.with_name(self._writer_path.name + ".part")
            part.replace(self._writer_path)
            self._writer = None
            self.state["shards"] += 1
            self._records_in_shard = 0
        if self._last_line_pos is not None and self._last_line_prefix is not None:
            self.state["resume"] = {
                "pos": self._last_line_pos,
                "prefix": base64.b64encode(self._last_line_prefix).decode("ascii"),
            }
        self.state["position"] = self._position
        self.state["stats"] = dict(self.extractor.stats)
        self._lines_since_commit = 0
        _write_state(self.work_dir, self.state)

    # Progress

    def _report(self, now: float) -> None:
        then, then_pos, then_lines = self._progress_mark
        elapsed = max(now - then, 1e-6)
        rate = (self._position - then_pos) / elapsed
        lines_rate = (self.state["lines"] - then_lines) / elapsed
        remaining = self.dump.size - self._position
        eta = _duration(remaining / rate) if rate > 0 else "?"
        kept = self.extractor.stats
        groups = ", ".join(
            f"{k.split(':', 1)[1]} {v:,}"
            for k, v in sorted(kept.items())
            if k.startswith("kept:")
        )
        print(
            f"[wikidata] {self._position / self.dump.size:6.2%} "
            f"({human_bytes(self._position)}/{human_bytes(self.dump.size)}) "
            f"{human_bytes(rate)}/s, {lines_rate:,.0f} entities/s, ETA {eta} | "
            f"{self.state['lines']:,} entities, {groups or 'nothing kept yet'}",
            flush=True,
        )
        self._last_progress = now
        self._progress_mark = (now, self._position, self.state["lines"])


def _find_line_prefix(buf: bytes, prefix: bytes, at_stream_start: bool) -> int:
    """Index of `prefix` at a line start in `buf`, or -1"""
    if at_stream_start and buf.startswith(prefix):
        return 0
    idx = buf.find(b"\n" + prefix)
    return idx + 1 if idx >= 0 else -1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


@contextmanager
def _exclusive_lock(path: Path):
    try:
        import fcntl
    except ImportError:  # Windows: no locking
        yield
        return
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BuildError(
                "another Wikidata build is running (lock held on "
                f"{path}); stop it first"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
