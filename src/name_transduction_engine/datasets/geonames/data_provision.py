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


def ensure_geonames_sqlite(db_path: Path, force=False) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and is_geonames_ready(db_path):
        print(f"GeoNames is ready: {db_path}")
        return db_path

    print("GeoNames missing or invalid. Rebuilding from scratch...")

    download_geonames_data(force)

    conn = sqlite3.connect(db_path)
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

    if not is_geonames_ready(db_path):
        raise RuntimeError("GeoNames build finished, but validation failed.")

    print(f"GeoNames built successfully: {db_path}")
    return db_path
