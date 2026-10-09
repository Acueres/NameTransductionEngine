"""Models learned from GeoNames, as built artifacts.

A model is trained from the GeoNames tables in names.sqlite and written to
`data/models/`, so the code that reads it needs no database. The file
records what it was built from:

- the input fingerprint: the GeoNames build (its registry fingerprint and row
  counts, from `build_metadata`). A reloaded or changed GeoNames makes the
  model stale;
- the rules fingerprint: what shapes the model besides the data (letter
  readings, context windows, thresholds). Editing them makes it stale too.

Each model describes itself with a `GeoNamesModelSpec`; `nte init` builds the
models after the datasets and `nte data build models` rebuilds them alone.
"""

import datetime
import gzip
import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from name_transduction_engine.datasets.geonames.load import (
    REGISTRY_FINGERPRINT_KEY as GEONAMES_REGISTRY_FINGERPRINT_KEY,
    is_geonames_ready,
)
from name_transduction_engine.paths import DB_PATH

# build_metadata keys that identify the GeoNames build a model learns from
_INPUT_KEYS = (
    GEONAMES_REGISTRY_FINGERPRINT_KEY,
    "geoname_count",
    "alternate_name_count",
)


@dataclass(frozen=True)
class ModelState:
    ready: bool
    reason: str | None  # why it is not ready
    meta: dict  # the built model's `meta`, empty when there is none


@dataclass(frozen=True)
class TrainedModel:
    model: dict  # the stored model, without its `meta`
    meta: dict  # training counts for `meta` (names aligned, table sizes...)
    summary: str  # one line for the build message


@dataclass(frozen=True)
class GeoNamesModelSpec:
    name: str  # "persian"
    title: str  # "Persian romanization model", for messages
    path: Path
    format_version: int
    source: str  # attribution, recorded in `meta`
    rules_fingerprint: Callable[[], str]
    # Reads the training data from names.sqlite and trains: (data, version)
    extract: Callable[[sqlite3.Connection], list]
    train: Callable[[list, str], TrainedModel]
    # Called after a new file is written (the reader's cache is cleared)
    on_built: Callable[[], None] = lambda: None


def input_fingerprint(conn: sqlite3.Connection) -> str | None:
    """Fingerprint of the GeoNames build in the database, None without one"""
    try:
        rows = dict(
            conn.execute(
                f"SELECT key, value FROM build_metadata WHERE key IN "
                f"({', '.join('?' for _ in _INPUT_KEYS)})",
                _INPUT_KEYS,
            )
        )
    except sqlite3.DatabaseError:
        return None
    if any(k not in rows for k in _INPUT_KEYS):
        return None
    text = "|".join(f"{k}={rows[k]}" for k in _INPUT_KEYS)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def read_model_meta(path: Path) -> dict | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    meta = raw.get("meta") if isinstance(raw, dict) else None
    return meta if isinstance(meta, dict) else None


def model_state(
    spec: GeoNamesModelSpec, db_path: Path = DB_PATH, model_path: Path | None = None
) -> ModelState:
    model_path = model_path or spec.path
    meta = read_model_meta(model_path)
    if meta is None:
        if model_path.exists():
            return ModelState(False, "unreadable; run `nte init`", {})
        return ModelState(False, "not built; run `nte init`", {})
    if meta.get("format") != spec.format_version:
        return ModelState(False, "built in an older format; run `nte init`", meta)
    if meta.get("rules_fingerprint") != spec.rules_fingerprint():
        return ModelState(False, "training rules changed; run `nte init`", meta)
    current = None
    if db_path.exists():
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                current = input_fingerprint(conn)
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            current = None
    if current is None:
        # The model still works; it just cannot be checked against its source
        return ModelState(True, "GeoNames not built; source not checked", meta)
    if meta.get("input_fingerprint") != current:
        return ModelState(
            False, "built from a different GeoNames build; run `nte init`", meta
        )
    return ModelState(True, None, meta)


def is_model_ready(
    spec: GeoNamesModelSpec, db_path: Path = DB_PATH, model_path: Path | None = None
) -> bool:
    state = model_state(spec, db_path, model_path)
    return state.ready and state.reason is None


def ensure_model(spec: GeoNamesModelSpec, force: bool = False) -> None:
    title = spec.title
    if not force and is_model_ready(spec):
        print(f"{title} is ready: {spec.path}")
        return
    if not is_geonames_ready(DB_PATH):
        raise RuntimeError(
            f"The {title} is learned from GeoNames, which is missing or out of "
            "date. Build the datasets first (`nte init` does this in order)."
        )

    print(f"Building the {title} from GeoNames...")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        source_fingerprint = input_fingerprint(conn)
        data = spec.extract(conn)
    finally:
        conn.close()

    rules = spec.rules_fingerprint()
    version = hashlib.sha256(f"{source_fingerprint}|{rules}".encode()).hexdigest()[:12]
    trained = spec.train(data, version)
    model = dict(trained.model)
    model["meta"] = {
        "format": spec.format_version,
        "name": spec.name,
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "input_fingerprint": source_fingerprint,
        "rules_fingerprint": rules,
        "source": spec.source,
        **trained.meta,
    }
    write_atomically(model, spec.path)
    spec.on_built()

    if not is_model_ready(spec):
        raise RuntimeError(f"{title} build finished, but validation failed.")
    print(f"{title} built: {spec.path} ({trained.summary})")


def write_atomically(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode()
    temp = path.with_name(path.name + ".part")
    try:
        # mtime=0: the file depends only on its content
        with gzip.GzipFile(temp, "wb", mtime=0) as f:
            f.write(data)
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
