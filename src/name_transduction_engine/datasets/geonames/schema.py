import sqlite3


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS alternate_name;
        DROP TABLE IF EXISTS geoname;
        DROP TABLE IF EXISTS language_code;
        DROP TABLE IF EXISTS build_metadata;

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

        CREATE TABLE alternate_name (
            alternate_name_id   INTEGER PRIMARY KEY,
            geonameid           INTEGER NOT NULL,
            isolanguage         TEXT NOT NULL DEFAULT '',
            alternate_name      TEXT NOT NULL,
            is_preferred_name   INTEGER NOT NULL DEFAULT 0,
            is_short_name       INTEGER NOT NULL DEFAULT 0,
            is_colloquial       INTEGER NOT NULL DEFAULT 0,
            is_historic         INTEGER NOT NULL DEFAULT 0,
            from_date           TEXT,
            to_date             TEXT,
            row_kind            TEXT NOT NULL,
            normalized_name     TEXT,

            FOREIGN KEY (geonameid) REFERENCES geoname(geonameid)
        );

        CREATE TABLE language_code (
            iso_639_3       TEXT,
            iso_639_2       TEXT,
            iso_639_1       TEXT,
            language_name   TEXT NOT NULL
        );

        CREATE TABLE build_metadata (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL
        );
        """)


def build_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE INDEX idx_geoname_normalized_name
        ON geoname(normalized_name);

        CREATE INDEX idx_alt_normalized_name
        ON alternate_name(normalized_name, geonameid);

        CREATE INDEX idx_alt_geoname_lang
        ON alternate_name(geonameid, isolanguage);
        """)
