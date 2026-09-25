import sqlite3

from name_transduction_engine.datasets.language_codes.data_provision import (
    language_tag_report_ddl,
)

# One row per distinct raw isolanguage value, including non-name tags, which
# are counted even though their rows are not stored. Read by `nte data status`
LANGUAGE_TAG_TABLE = "geonames_language_tag"


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS alternate_name;
        DROP TABLE IF EXISTS geoname;

        CREATE TABLE geoname (
            geonameid       INTEGER PRIMARY KEY,
            name            TEXT NOT NULL,
            asciiname       TEXT,
            latitude        REAL,
            longitude       REAL,
            country_code    TEXT,
            admin1_code     TEXT,
            admin2_code     TEXT,
            feature_class   TEXT,
            feature_code    TEXT,
            population      INTEGER,
            normalized_name TEXT NOT NULL
        );

        -- Rows whose isolanguage is not a name (postcodes, airport codes,
        -- links) are not stored. `lang` holds a registry code or NULL; it has
        -- no FOREIGN KEY to `language` on purpose (see the language_codes
        -- data_provision docstring), the registry fingerprint links them.
        CREATE TABLE alternate_name (
            alternate_name_id   INTEGER PRIMARY KEY,
            geonameid           INTEGER NOT NULL,
            isolanguage         TEXT NOT NULL DEFAULT '',  -- raw tag, provenance
            alternate_name      TEXT NOT NULL,
            is_preferred_name   INTEGER NOT NULL DEFAULT 0,
            is_short_name       INTEGER NOT NULL DEFAULT 0,
            is_colloquial       INTEGER NOT NULL DEFAULT 0,
            is_historic         INTEGER NOT NULL DEFAULT 0,
            from_date           TEXT,
            to_date             TEXT,
            lang                TEXT,
            lang_script         TEXT,
            lang_region         TEXT,
            lang_variant        TEXT,
            tag_status          TEXT NOT NULL
                CHECK (tag_status IN ('language', 'untagged', 'multiple', 'unmapped')),
            normalized_name     TEXT,

            FOREIGN KEY (geonameid) REFERENCES geoname(geonameid)
        );
        """ + language_tag_report_ddl(LANGUAGE_TAG_TABLE))


def build_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE INDEX idx_geoname_normalized_name
        ON geoname(normalized_name);

        CREATE INDEX idx_alt_normalized_name
        ON alternate_name(normalized_name, geonameid);

        CREATE INDEX idx_alt_geoname_lang
        ON alternate_name(geonameid, lang);

        ANALYZE;
        """)