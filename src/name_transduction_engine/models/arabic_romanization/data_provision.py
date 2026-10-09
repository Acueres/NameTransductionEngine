"""The Arabic display-romanization model as a built artifact.

Learned from GeoNames (see `train.py`) and written to `data/models/`; built,
fingerprinted and checked like every GeoNames model (`models/geonames_model.py`).
The romanizer reads the file without the database and, when it is missing,
falls back to the rule-based Arabic-script reader with a warning.
"""

import sqlite3
from pathlib import Path

from name_transduction_engine.paths import ARABIC_MODEL_PATH, DB_PATH
from name_transduction_engine.transliteration.romanization_packs import arabic as ar

from ..geonames_model import (
    GeoNamesModelSpec,
    ModelState,
    TrainedModel,
    ensure_model,
    model_state,
)
from . import train

__all__ = ["SPEC", "arabic_model_state", "ensure_arabic_model"]

MODEL_NAME = "arabic"
FORMAT_VERSION = 1


def _extract(conn: sqlite3.Connection) -> list[train.Pair]:
    return train.extract_pairs(conn)


def _train(pairs: list[train.Pair], version: str) -> TrainedModel:
    print(f"Aligning {len(pairs):,} Arabic names with their romanizations...")
    counts = train.count(pairs)
    model = train.build_model(counts, version)
    meta = {
        "names_total": counts.names_total,
        "names_aligned": counts.names_aligned,
        "contexts": len(model["table"]),
        "words": len(model["lexicon"]),
        "construct_heads": len(model["construct_head"]),
    }
    summary = (
        f"{counts.names_aligned:,}/{counts.names_total:,} names aligned; "
        f"{len(model['table']):,} contexts, {len(model['lexicon']):,} words"
    )
    return TrainedModel(model, meta, summary)


SPEC = GeoNamesModelSpec(
    name=MODEL_NAME,
    title="Arabic romanization model",
    path=ARABIC_MODEL_PATH,
    format_version=FORMAT_VERSION,
    source=train.SOURCE,
    rules_fingerprint=train.rules_fingerprint,
    extract=_extract,
    train=_train,
    on_built=ar.load_model.cache_clear,
)


def arabic_model_state(
    db_path: Path = DB_PATH, model_path: Path = ARABIC_MODEL_PATH
) -> ModelState:
    return model_state(SPEC, db_path, model_path)


def ensure_arabic_model(force: bool = False) -> None:
    ensure_model(SPEC, force)
