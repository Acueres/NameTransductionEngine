import sqlite3

from pathlib import Path
from .download import download_geonames_data
from .schema import create_schema, build_indexes
from .load import (
    is_geonames_ready,
    configure_connection,
    load_all_data,
    write_build_metadata,
)
from name_transduction_engine.paths import DB_PATH


def ensure_geonames_sqlite(force=False) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force and is_geonames_ready(DB_PATH):
        print(f"GeoNames is ready: {DB_PATH}")
        return

    print("GeoNames missing or invalid. Rebuilding from scratch...")

    download_geonames_data(force)

    conn = sqlite3.connect(DB_PATH)
    try:
        configure_connection(conn)
        create_schema(conn)
        load_all_data(conn)
        build_indexes(conn)
        write_build_metadata(conn)

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

    if not is_geonames_ready(DB_PATH):
        raise RuntimeError("GeoNames build finished, but validation failed.")

    print(f"GeoNames built successfully: {DB_PATH}")
