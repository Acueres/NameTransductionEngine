import sqlite3


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS alternate_name;
        DROP TABLE IF EXISTS geoname;
        DROP TABLE IF EXISTS language_code;
        DROP TABLE IF EXISTS build_metadata;

        CREATE TABLE geoname (
            geonameid       INTEGER PRIMARY KEY,
            name            TEXT NOT NULL,
            asciiname       TEXT,
            country_code    TEXT,
            admin1_code     TEXT,
            admin2_code     TEXT,
            feature_class   TEXT,
            feature_code    TEXT,
            population      INTEGER
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

        CREATE INDEX idx_geoname_country_code
            ON geoname(country_code);

        CREATE INDEX idx_geoname_feature
            ON geoname(feature_class, feature_code);

        CREATE INDEX idx_geoname_population
            ON geoname(population);

        CREATE INDEX idx_alt_geonameid
            ON alternate_name(geonameid);

        CREATE INDEX idx_alt_isolanguage
            ON alternate_name(isolanguage);

        CREATE INDEX idx_alt_row_kind
            ON alternate_name(row_kind);

        CREATE INDEX idx_alt_normalized_name
            ON alternate_name(normalized_name);

        CREATE INDEX idx_alt_lang_name
            ON alternate_name(isolanguage, normalized_name);

        CREATE INDEX idx_alt_geoname_lang
            ON alternate_name(geonameid, isolanguage);

        CREATE INDEX idx_alt_lookup_candidates
            ON alternate_name(isolanguage, normalized_name, geonameid)
            WHERE row_kind IN ('name_lang', 'name_untyped');
        """
    )
