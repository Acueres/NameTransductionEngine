import sqlite3

from .download import download_language_codes_data
from .schema import create_schema
from .load import (
    is_language_codes_ready,
    load_all_data,
    print_language_tag_summary,
    read_registry,
    stored_registry_fingerprint,
    write_build_metadata,
    write_language_tag_report,
)
from .schema import language_tag_report_ddl
from name_transduction_engine.datasets.shared import (
    configure_connection,
    ensure_build_metadata_table,
)
from name_transduction_engine.paths import DB_PATH

__all__ = [
    "ensure_language_codes_sqlite",
    "language_tag_report_ddl",
    "print_language_tag_summary",
    "read_registry",
    "stored_registry_fingerprint",
    "write_language_tag_report",
]


def ensure_language_codes_sqlite(force=False) -> None:
    """Build the language registry. Must run before GeoNames and Wikidata,
    which canonicalize their language tags against it.

    Name tables do not hold foreign keys to `language`: with foreign keys on,
    SQLite refuses to drop a table that other rows reference, which would block
    this rebuild. They record the registry fingerprint they were built with
    instead, and their readiness checks compare it with the current one
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force and is_language_codes_ready(DB_PATH):
        print(f"Language codes are ready: {DB_PATH}")
        return

    print("Language codes missing or invalid. Rebuilding from scratch...")

    download_language_codes_data(force)

    conn = sqlite3.connect(DB_PATH)
    try:
        configure_connection(conn)
        ensure_build_metadata_table(conn)
        create_schema(conn)
        load_all_data(conn)
        # No build_indexes: the registry is read into memory whole, and its
        # PRIMARY KEY / UNIQUE constraints already index every lookup column
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

    if not is_language_codes_ready(DB_PATH):
        raise RuntimeError("Language codes build finished, but validation failed.")

    print(f"Language codes built successfully: {DB_PATH}")
