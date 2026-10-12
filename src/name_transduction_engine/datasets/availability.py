import json
import os
import socket

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from name_transduction_engine import paths


class DataUnavailable(RuntimeError):
    pass


class DataBusy(RuntimeError):
    """Another `nte init` or `nte update` is running"""


@dataclass(frozen=True)
class MaintenanceNotice:
    reason: str
    started_at: str  # UTC, ISO 8601
    pid: int
    host: str
    active: bool  # False: its process is gone

    @property
    def message(self) -> str:
        return (
            f"NTE is unavailable: its data is being updated ({self.reason}) "
            f"since {self.started_at}. Try again later."
        )


def read_notice(path: Path | None = None) -> MaintenanceNotice | None:
    path = path or paths.MAINTENANCE_NOTICE_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = int(data["pid"])
        host = str(data["host"])
        notice = MaintenanceNotice(
            str(data["reason"]), str(data["started_at"]), pid, host, True
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        # Damaged (it is written atomically, so never half-written): stale
        return MaintenanceNotice("unknown", "unknown", 0, "", False)
    if host == socket.gethostname() and not _alive(pid):
        return MaintenanceNotice(
            notice.reason, notice.started_at, pid, host, active=False
        )
    return notice


def check_data_available() -> None:
    """Raise DataUnavailable while an update is rebuilding the database"""
    notice = read_notice()
    if notice is not None and notice.active:
        raise DataUnavailable(notice.message)


@contextmanager
def maintenance(reason: str) -> Generator[None, None, None]:
    """Show the maintenance notice for the duration of the block"""
    path = paths.MAINTENANCE_NOTICE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "reason": reason,
        "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def clear_stale_notice() -> None:
    notice = read_notice()
    if notice is not None and not notice.active:
        paths.MAINTENANCE_NOTICE_PATH.unlink(missing_ok=True)


@contextmanager
def data_lock() -> Generator[None, None, None]:
    """Held while names.sqlite is (re)built by `nte init` or `nte update`"""
    path = paths.DATA_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # Windows: no locking
        yield
        return
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DataBusy(
                "another `nte init` or `nte update` is running; wait for it to finish"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True
