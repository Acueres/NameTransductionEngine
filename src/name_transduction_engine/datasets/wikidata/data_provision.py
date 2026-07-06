import sqlite3

from name_transduction_engine.paths import DB_PATH
from .load import (
    ensure_locations_dataset,
    is_wikidata_ready,
    load_locations_dataset,
    build_wikidata_compact_dataset,
)
from .schema import create_schema, build_indexes
from .download_raw import download_wikidata_raw


def ensure_wikidata_sqlite(force: bool = False) -> None:
    """
    The full Wikidata dump is optional. If it exists, it can be used to build
    the compact local wikidata_locations JSONL dataset. If it does not exist,
    the importer expects that compact dataset to exist or be fetched later.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force and is_wikidata_ready(DB_PATH):
        print(f"Wikidata is ready: {DB_PATH}")
        return

    dataset_path = ensure_locations_dataset(force)

    print("Wikidata missing or invalid. Rebuilding from scratch...")

    conn = sqlite3.connect(DB_PATH)
    try:
        _configure_connection(conn)
        create_schema(conn)
        load_locations_dataset(conn, dataset_path)
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
