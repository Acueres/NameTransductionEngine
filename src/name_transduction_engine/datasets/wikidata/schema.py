import sqlite3


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS wikidata_location;
        DROP TABLE IF EXISTS wikidata_location_geonames;
        DROP TABLE IF EXISTS wikidata_location_p31;
        DROP TABLE IF EXISTS wikidata_lang_norm;
        DROP TABLE IF EXISTS wikidata_location_name;

        CREATE TABLE wikidata_location (
            qid TEXT PRIMARY KEY,

            -- Coarse NTE classification derived from accepted P31 values:
            -- country, city, river, island, hamlet, etc.
            kind TEXT NOT NULL,

            -- Optional coordinates from P625.
            lat REAL,
            lon REAL
        );

        CREATE TABLE wikidata_location_geonames (
            qid TEXT NOT NULL,
            geonames_id TEXT NOT NULL,

            PRIMARY KEY (qid, geonames_id),

            FOREIGN KEY (qid)
            REFERENCES wikidata_location(qid)
            ON DELETE CASCADE
            );

        CREATE TABLE wikidata_location_p31 (
            qid TEXT NOT NULL,
            p31_qid TEXT NOT NULL,

            PRIMARY KEY (qid, p31_qid),

            FOREIGN KEY (qid)
                REFERENCES wikidata_location(qid)
                ON DELETE CASCADE
        );

        CREATE TABLE wikidata_lang_norm (
            wd_lang TEXT PRIMARY KEY,

            -- NTE/GeoNames-style lookup bucket.
            -- Usually ISO-639-1 if available, else ISO-639-3.
            -- May contain explicit exceptions such as map-bms / roa-tara.
            geo_lang TEXT,

            iso639_3 TEXT,
            iso639_1 TEXT
        );

        CREATE TABLE wikidata_location_name (
            qid TEXT NOT NULL,

            -- Original Wikidata/Wikimedia language code.
            -- Examples: en, fr, zh-hant, be-tarask, sr-el.
            wd_lang TEXT NOT NULL,

            -- Collapsed NTE/GeoNames-style language bucket.
            -- Examples: en, fr, zh, be, sr, grc, ang.
            geo_lang TEXT,

            -- Original Wikidata label text, preserved in original script.
            name TEXT NOT NULL,

            -- Conservative search key:
            -- Unicode-normalized, whitespace-normalized, casefolded.
            -- Not Latinized/transliterated.
            name_norm TEXT NOT NULL,

            -- For now this will be "label".
            -- Later aliases can be added as "alias" with lower priority.
            term_type TEXT NOT NULL DEFAULT 'label',

            PRIMARY KEY (qid, wd_lang, term_type, name),

            FOREIGN KEY (qid)
                REFERENCES wikidata_location(qid)
                ON DELETE CASCADE,

            FOREIGN KEY (wd_lang)
                REFERENCES wikidata_lang_norm(wd_lang)
        );
        """)


def build_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE INDEX idx_wd_location_geonames_id
        ON wikidata_location_geonames (geonames_id);
        
        CREATE INDEX idx_wd_location_p31_qid
        ON wikidata_location_p31 (p31_qid);

        CREATE INDEX idx_wd_lang_norm_geo_lang
        ON wikidata_lang_norm (geo_lang);

        CREATE INDEX idx_wd_location_name_lookup
        ON wikidata_location_name (geo_lang, name_norm);

        CREATE INDEX idx_wd_location_name_qid
        ON wikidata_location_name (qid);

        CREATE INDEX idx_wd_location_name_name_norm
        ON wikidata_location_name (name_norm);
        """)
