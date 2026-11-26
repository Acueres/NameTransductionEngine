import json
import re
import sqlite3

from pathlib import Path

from name_transduction_engine.paths import RAW_DIR_COUNTRY_CODES
from .download import CLDR_TERRITORIES_FILENAME, CLDR_VERSION

VERSION_KEY = "country_codes_cldr_version"

# Two-letter CLDR territories that are not countries: groupings, unknown and
# pseudo-locale codes, and codes ISO reserves for special areas (Ascension,
# Clipperton, Sark, Diego Garcia, Ceuta and Melilla, Canary Islands, Tristan
# da Cunha). Number codes (world regions) and `-alt-` names are skipped by
# their form
_NOT_COUNTRIES = frozenset(
    {"EU", "EZ", "UN", "QO", "ZZ", "XA", "XB", "AC", "CP", "CQ", "DG", "EA", "IC", "TA"}
)


def read_countries(path: Path) -> dict[str, str]:
    """ISO 3166-1 alpha-2 code -> English name"""
    data = json.loads(path.read_text(encoding="utf-8"))
    territories = data["main"]["en"]["localeDisplayNames"]["territories"]
    countries = {
        code: name
        for code, name in territories.items()
        if re.fullmatch("[A-Z]{2}", code) and code not in _NOT_COUNTRIES
    }

    if "XK" not in countries or len(countries) < 249:
        raise ValueError(
            f"{path.name}: expected the 249 ISO 3166-1 countries and XK, "
            f"found {len(countries)} codes"
        )
    return countries


def is_country_codes_ready(db_path: Path) -> bool:
    """Loaded from this code's CLDR version, and not empty (a failed load
    leaves the recreated table empty)"""
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(db_path)
        try:
            version = conn.execute(
                "SELECT value FROM build_metadata WHERE key = ?", (VERSION_KEY,)
            ).fetchone()
            count = conn.execute("SELECT COUNT(*) FROM country_codes").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.DatabaseError:  # also a missing table
        return False
    return version == (CLDR_VERSION,) and count > 0


def load_country_codes(conn: sqlite3.Connection) -> None:
    countries = read_countries(RAW_DIR_COUNTRY_CODES / CLDR_TERRITORIES_FILENAME)
    print(f"Storing {len(countries)} country codes (CLDR {CLDR_VERSION})")
    conn.executemany(
        "INSERT INTO country_codes (code, name) VALUES (?, ?)",
        sorted(countries.items()),
    )
    conn.execute(
        "INSERT OR REPLACE INTO build_metadata (key, value) VALUES (?, ?)",
        (VERSION_KEY, CLDR_VERSION),
    )
