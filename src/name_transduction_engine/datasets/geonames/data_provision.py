import sqlite3

from .download import download_geonames_data
from .schema import create_schema, build_indexes
from .load import (
    is_geonames_ready,
    load_all_data,
    write_build_metadata,
)
from name_transduction_engine.datasets.shared import (
    configure_connection,
    ensure_build_metadata_table,
)
from name_transduction_engine.datasets.language_codes.data_provision import (
    read_registry,
)
from name_transduction_engine.normalization.language_code_normalization import (
    RegistryError,
)
from name_transduction_engine.paths import DB_PATH

__all__ = [
    "ensure_geonames_sqlite",
]


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
        ensure_build_metadata_table(conn)
        registry = _read_registry_or_explain(conn)
        create_schema(conn)
        load_all_data(conn, registry)
        build_indexes(conn)
        write_build_metadata(conn, registry)

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


def _read_registry_or_explain(conn: sqlite3.Connection):
    try:
        return read_registry(conn)
    except (sqlite3.DatabaseError, RegistryError) as exc:
        raise RuntimeError(
            "GeoNames needs the language registry, which is missing or out of "
            "date. Build language codes first (`nte init` does this in order)."
        ) from exc
