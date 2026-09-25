import sqlite3

from name_transduction_engine.datasets.language_codes.data_provision import (
    language_tag_report_ddl,
)

# One row per distinct raw wd_lang value: what it mapped to, how many labels
# carried it. Read by `nte data status`
LANGUAGE_TAG_TABLE = "wikidata_language_tag"


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS wikidata_location_name;
        DROP TABLE IF EXISTS wikidata_location_p31;
        DROP TABLE IF EXISTS wikidata_location_geonames;
        DROP TABLE IF EXISTS wikidata_lang_norm;  -- replaced by the language registry
        DROP TABLE IF EXISTS wikidata_location;

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

            geonames_id INTEGER NOT NULL,

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

        CREATE TABLE wikidata_location_name (
            qid TEXT NOT NULL,

            -- Original Wikidata/Wikimedia language code, kept for provenance.
            -- Examples: en, fr, zh-hant, be-tarask, sr-el.
            wd_lang TEXT NOT NULL,

            -- Registry code and subtags, computed at load time from wd_lang.
            -- Examples: sr-el -> lang=sr, lang_script=Latn;
            -- be-tarask -> lang=be, lang_variant=tarask.
            lang TEXT,
            lang_script TEXT,
            lang_region TEXT,
            lang_variant TEXT,
            tag_status TEXT NOT NULL
                CHECK (tag_status IN ('language', 'untagged', 'multiple', 'unmapped')),

            -- Original Wikidata label text, preserved in original script
            name TEXT NOT NULL,

            -- Search key
            normalized_name TEXT NOT NULL,

            term_type TEXT NOT NULL DEFAULT 'label',

            PRIMARY KEY (qid, wd_lang, term_type, name),

            FOREIGN KEY (qid)
                REFERENCES wikidata_location(qid)
                ON DELETE CASCADE
        );
        """ + language_tag_report_ddl(LANGUAGE_TAG_TABLE))


def build_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE INDEX idx_wd_location_geonames_id
        ON wikidata_location_geonames (geonames_id);

        CREATE INDEX idx_wd_location_p31_qid
        ON wikidata_location_p31 (p31_qid);

        CREATE INDEX idx_wd_name_resolve
        ON wikidata_location_name (normalized_name, qid);

        CREATE INDEX idx_wd_name_hop
        ON wikidata_location_name (qid, lang);

        ANALYZE;
        """)
