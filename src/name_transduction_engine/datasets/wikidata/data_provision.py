import sqlite3

from name_transduction_engine.paths import DB_PATH, WIKIDATA_LOCATIONS_PATH
from .build import build_wikidata_compact_dataset
from .load import (
    is_wikidata_ready,
    load_locations_dataset
)
from .schema import create_schema, build_indexes
from .download import download_wikidata_locations_data
from .download_raw import download_wikidata_raw

__all__ = [
    "download_wikidata_raw",
    "ensure_wikidata_sqlite",
    "build_wikidata_compact_dataset",
]


def ensure_wikidata_sqlite(force: bool = False) -> None:
    """Download the published compact locations dataset from GitHub and (re)build the Wikidata tables in names.sqlite"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force and is_wikidata_ready(DB_PATH):
        print(f"Wikidata is ready: {DB_PATH}")
        return

    print("Wikidata missing or invalid. Rebuilding from scratch...")

    download_wikidata_locations_data(force)

    conn = sqlite3.connect(DB_PATH)
    try:
        _configure_connection(conn)
        create_schema(conn)
        load_locations_dataset(conn, WIKIDATA_LOCATIONS_PATH)
        build_indexes(conn)
        conn.execute("ANALYZE;")
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not is_wikidata_ready(DB_PATH):
        raise RuntimeError("Wikidata build finished, but validation failed.")

    print(f"Wikidata built successfully: {DB_PATH}")


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
