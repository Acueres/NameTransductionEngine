import sqlite3

from name_transduction_engine.datasets.country_codes.download import CLDR_VERSION
from name_transduction_engine.datasets.country_codes.load import (
    VERSION_KEY,
    is_country_codes_ready,
)


def test_country_list(nte_data):
    db = nte_data / "names.sqlite"
    with sqlite3.connect(db) as conn:
        countries = dict(conn.execute("SELECT code, name FROM country_codes"))
    # The 249 ISO 3166-1 countries plus XK, as GeoNames uses them
    assert len(countries) == 250
    assert countries["XK"] == "Kosovo"
    assert countries["DE"] == "Germany"
    assert countries["QA"] == "Qatar"
    assert not {"EU", "UN", "ZZ", "XA", "IC", "001", "US-alt-short"} & countries.keys()
    assert is_country_codes_ready(db)


def test_not_ready(tmp_path):
    db = tmp_path / "names.sqlite"
    assert not is_country_codes_ready(db)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE build_metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE country_codes (code TEXT PRIMARY KEY, name TEXT)")
        conn.execute(
            "INSERT INTO build_metadata VALUES (?, ?)", (VERSION_KEY, CLDR_VERSION)
        )
    # The table a failed load leaves behind: recreated, but empty
    assert not is_country_codes_ready(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO country_codes VALUES ('DE', 'Germany')")
        conn.execute("UPDATE build_metadata SET value = '0.0.0'")
    # Loaded from another CLDR version
    assert not is_country_codes_ready(db)
