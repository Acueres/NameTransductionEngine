"""The Persian display-romanization model as a built artifact.

Learned from GeoNames (see `train.py`) and written to `data/models/`, next to
the database it came from. The file records what it was built from:

- the input fingerprint: the GeoNames build (its registry fingerprint and row
  counts, from `build_metadata`). A reloaded or changed GeoNames makes the
  model stale;
- the rules fingerprint: the letter readings, context windows and thresholds
  of the trainer. Editing them makes the model stale too.

`nte init` builds it after the datasets; `nte data build models` rebuilds it
alone. The romanizer reads the file without the database and, when it is
missing, falls back to the generic Arabic-script rules with a warning.
"""

import datetime
import gzip
import hashlib
import json
import sqlite3

from dataclasses import dataclass
from pathlib import Path

from name_transduction_engine.datasets.geonames.load import (
    REGISTRY_FINGERPRINT_KEY as GEONAMES_REGISTRY_FINGERPRINT_KEY,
    is_geonames_ready,
)
from name_transduction_engine.paths import DB_PATH, PERSIAN_MODEL_PATH
from name_transduction_engine.transliteration.romanization_packs import persian as fa

from . import train

MODEL_NAME = "persian"
FORMAT_VERSION = 1

# build_metadata keys that identify the GeoNames build the model learns from
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


def read_model_meta(path: Path = PERSIAN_MODEL_PATH) -> dict | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    meta = raw.get("meta") if isinstance(raw, dict) else None
    return meta if isinstance(meta, dict) else None


def persian_model_state(
    db_path: Path = DB_PATH, model_path: Path = PERSIAN_MODEL_PATH
) -> ModelState:
    meta = read_model_meta(model_path)
    if meta is None:
        if model_path.exists():
            return ModelState(False, "unreadable; run `nte init`", {})
        return ModelState(False, "not built; run `nte init`", {})
    if meta.get("format") != FORMAT_VERSION:
        return ModelState(False, "built in an older format; run `nte init`", meta)
    if meta.get("rules_fingerprint") != train.rules_fingerprint():
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


def is_persian_model_ready(
    db_path: Path = DB_PATH, model_path: Path = PERSIAN_MODEL_PATH
) -> bool:
    state = persian_model_state(db_path, model_path)
    return state.ready and state.reason is None


def ensure_persian_model(force: bool = False) -> None:
    if not force and is_persian_model_ready():
        print(f"Persian romanization model is ready: {PERSIAN_MODEL_PATH}")
        return
    if not is_geonames_ready(DB_PATH):
        raise RuntimeError(
            "The Persian romanization model is learned from GeoNames, which is "
            "missing or out of date. Build the datasets first (`nte init` does "
            "this in order)."
        )

    print("Building the Persian romanization model from GeoNames...")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        source_fingerprint = input_fingerprint(conn)
        pairs = train.extract_pairs(conn)
    finally:
        conn.close()
    print(f"Aligning {len(pairs):,} Persian names with their romanizations...")
    counts = train.count(pairs)

    rules = train.rules_fingerprint()
    version = hashlib.sha256(f"{source_fingerprint}|{rules}".encode()).hexdigest()[:12]
    model = train.build_model(counts, version)
    model["meta"] = {
        "format": FORMAT_VERSION,
        "name": MODEL_NAME,
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "input_fingerprint": source_fingerprint,
        "rules_fingerprint": rules,
        "source": train.SOURCE,
        "names_total": counts.names_total,
        "names_aligned": counts.names_aligned,
        "contexts": len(model["table"]),
        "words": len(model["lexicon"]),
        "ezafe_heads": len(model["ezafe_head"]),
    }
    _write_atomically(model, PERSIAN_MODEL_PATH)
    fa.load_model.cache_clear()

    if not is_persian_model_ready():
        raise RuntimeError("Persian model build finished, but validation failed.")
    print(
        f"Persian romanization model built: {PERSIAN_MODEL_PATH} "
        f"({counts.names_aligned:,}/{counts.names_total:,} names aligned; "
        f"{len(model['table']):,} contexts, {len(model['lexicon']):,} words)"
    )


def _write_atomically(model: dict, path: Path) -> None:
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
