import sqlite3

from pathlib import Path
from .download import download_geonames_data
from .schema import create_schema
from .load import (
    database_is_ready,
    configure_connection,
    load_all_data,
    write_build_metadata,
)


def ensure_geonames_sqlite(db_path: Path, force_download=False) -> Path:
    """
    Create or validate the local GeoNames SQLite database.

    If the DB is missing or invalid, rebuild it from scratch.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if database_is_ready(db_path):
        print(f"GeoNames DB is ready: {db_path}")
        return db_path

    print("GeoNames DB missing or invalid. Rebuilding from scratch...")

    if db_path.exists():
        db_path.unlink()

    download_geonames_data(force_download)

    conn = sqlite3.connect(db_path)
    try:
        configure_connection(conn)
        create_schema(conn)
        load_all_data(conn)
        write_build_metadata(conn)

        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        if db_path.exists():
            db_path.unlink()
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not database_is_ready(db_path):
        if db_path.exists():
            db_path.unlink()
        raise RuntimeError("GeoNames DB build finished, but validation failed.")

    print(f"GeoNames DB built successfully: {db_path}")
    return db_path
