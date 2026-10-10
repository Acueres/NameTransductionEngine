import sqlite3

from name_transduction_engine.paths import DB_PATH, WIKIDATA_DOWNLOAD_DIR
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
from .build import BuildOptions, build_wikidata_compact
from .classes import refresh_class_map as refresh_wikidata_classes
from .compact import CompactDataset, has_loaded_groups, open_compact_dataset
from .load import (
    is_wikidata_ready,
    load_compact_dataset,
    write_build_metadata,
)
from .schema import create_schema, build_indexes
from .download import fetch_published_dataset
from .download_raw import download_wikidata_raw

__all__ = [
    "BuildOptions",
    "build_wikidata_compact",
    "current_wikidata_dataset",
    "download_wikidata_raw",
    "ensure_wikidata_sqlite",
    "fetch_published_dataset",
    "refresh_wikidata_classes",
]


def current_wikidata_dataset() -> CompactDataset | None:
    """What `nte init` loads: the published dataset downloaded by
    `nte data fetch wikidata`, if it holds every loaded group. The output of
    `nte data build wikidata-compact` is never loaded; it is the source of
    new releases"""
    dataset = open_compact_dataset(WIKIDATA_DOWNLOAD_DIR)
    return dataset if dataset is not None and has_loaded_groups(dataset) else None


def ensure_wikidata_sqlite(force: bool = False) -> None:
    """(Re)build the Wikidata tables in names.sqlite from the compact dataset
    (see current_wikidata_dataset). Without a downloaded copy, the newest
    published dataset is downloaded first; a newer release is only picked
    up by `nte data fetch wikidata`"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = current_wikidata_dataset()
    if dataset is None:
        fetch_published_dataset()
        dataset = current_wikidata_dataset()
        if dataset is None:
            raise RuntimeError("Wikidata compact dataset is missing after download.")

    if not force and is_wikidata_ready(DB_PATH, dataset.dataset_id):
        print(f"Wikidata is ready: {DB_PATH}")
        return

    print("Wikidata missing, outdated or invalid. Rebuilding from scratch...")

    conn = sqlite3.connect(DB_PATH)
    try:
        configure_connection(conn)
        ensure_build_metadata_table(conn)
        registry = _read_registry_or_explain(conn)
        create_schema(conn)
        load_compact_dataset(conn, dataset, registry)
        build_indexes(conn)
        write_build_metadata(conn, registry, dataset)

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

    if not is_wikidata_ready(DB_PATH, dataset.dataset_id):
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
