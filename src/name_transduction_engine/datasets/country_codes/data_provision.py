import sqlite3

from .download import download_country_codes_data
from .load import is_country_codes_ready, load_country_codes
from .schema import create_schema
from name_transduction_engine.datasets.shared import (
    configure_connection,
    ensure_build_metadata_table,
)
from name_transduction_engine.paths import DB_PATH

__all__ = ["ensure_country_codes_sqlite"]


def ensure_country_codes_sqlite(force=False) -> None:
    """Country names from CLDR, for entity context and `--country`. Does not
    depend on the language registry"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force and is_country_codes_ready(DB_PATH):
        print(f"Country codes are ready: {DB_PATH}")
        return

    print("Country codes missing, outdated or invalid. Rebuilding...")
    download_country_codes_data(force)

    conn = sqlite3.connect(DB_PATH)
    try:
        configure_connection(conn)
        ensure_build_metadata_table(conn)
        create_schema(conn)
        load_country_codes(conn)
        conn.commit()
    finally:
        conn.close()  # without a commit, the load is rolled back

    if not is_country_codes_ready(DB_PATH):
        raise RuntimeError("Country codes build finished, but validation failed.")
    print(f"Country codes built successfully: {DB_PATH}")
