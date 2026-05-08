import sqlite3

from pathlib import Path

from name_transduction_engine.paths import RAW_DIR_WIKIDATA
from .load import ensure_locations_dataset, database_is_ready, load_locations_dataset
from .schema import create_schema, build_indexes
from .download_raw import download_wikidata

RAW_DUMP_PATH = RAW_DIR_WIKIDATA / "latest-all.json.bz2"


def download_wikidata_raw(force=False):
    download_wikidata(force)


def ensure_wikidata_sqlite(
    db_path: Path,
    raw_dump_path: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Create or validate the local Wikidata locations SQLite database.

    The full Wikidata dump is optional. If it exists, it can be used to build
    the compact local wikidata_locations JSONL dataset. If it does not exist,
    the importer expects that compact dataset to exist or be fetched later.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and database_is_ready(db_path):
        print(f"Wikidata DB is ready: {db_path}")
        return db_path

    dataset_path = ensure_locations_dataset(
        raw_dump_path=raw_dump_path or RAW_DUMP_PATH,
        force=force,
    )

    print("Wikidata DB missing or invalid. Rebuilding from scratch...")

    conn = sqlite3.connect(db_path)
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

    if not database_is_ready(db_path):
        raise RuntimeError("Wikidata DB build finished, but validation failed.")

    print(f"Wikidata DB built successfully: {db_path}")
    return db_path


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
