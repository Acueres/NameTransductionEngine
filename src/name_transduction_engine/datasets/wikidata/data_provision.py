import sqlite3

from name_transduction_engine.paths import DB_PATH, WIKIDATA_LOCATIONS_PATH
from name_transduction_engine.datasets.shared import (
    configure_connection,
    ensure_build_metadata_table,
)
from name_transduction_engine.datasets.language_codes.data_provision import (
    read_registry,
)
from name_transduction_engine.normalization.language_code_normalization import (
    LanguageRegistry,
    RegistryError,
)
from .build import build_wikidata_compact_dataset
from .load import (
    is_wikidata_ready,
    load_locations_dataset,
    write_build_metadata,
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
        configure_connection(conn)
        ensure_build_metadata_table(conn)
        # Read before create_schema: if the registry is missing, fail without
        # having dropped the existing Wikidata tables
        registry = _read_registry_or_explain(conn)
        create_schema(conn)
        load_locations_dataset(conn, WIKIDATA_LOCATIONS_PATH, registry)
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

    if not is_wikidata_ready(DB_PATH):
        raise RuntimeError("Wikidata build finished, but validation failed.")

    print(f"Wikidata built successfully: {DB_PATH}")


def _read_registry_or_explain(conn: sqlite3.Connection) -> LanguageRegistry:
    try:
        return read_registry(conn)
    except (sqlite3.DatabaseError, RegistryError) as exc:
        raise RuntimeError(
            "Wikidata needs the language registry, which is missing or out of "
            "date. Build language codes first (`nte init` does this in order)."
        ) from exc
