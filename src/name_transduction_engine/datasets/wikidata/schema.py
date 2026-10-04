import sqlite3

from name_transduction_engine.datasets.language_codes.data_provision import (
    language_tag_report_ddl,
)

# One row per distinct raw wd_lang value: what it mapped to, how many names
# carried it. Read by `nte data status`
LANGUAGE_TAG_TABLE = "wikidata_language_tag"

TABLES = (
    "wikidata_entity",
    "wikidata_entity_class",
    "wikidata_name",
    "wikidata_link",
    "wikidata_geonames",
    "wikidata_external_id",
)

# Tables of the locations-only schema this one replaced
_OLD_TABLES = (
    "wikidata_location_name",
    "wikidata_location_p31",
    "wikidata_location_geonames",
    "wikidata_lang_norm",
    "wikidata_location",
)


def create_schema(conn: sqlite3.Connection) -> None:
    drops = "\n".join(
        f"DROP TABLE IF EXISTS {table};" for table in (*_OLD_TABLES, *reversed(TABLES))
    )
    conn.executescript(
        drops
        + """
        CREATE TABLE wikidata_entity (
            qid TEXT PRIMARY KEY,

            -- place | historical_place (the groups loaded into the database)
            entity_group TEXT NOT NULL,

            -- NTE classification from P31: city, village, admin1, river,
            -- historical_country, ancient_city, ...
            kind TEXT NOT NULL,

            lat REAL,
            lon REAL,

            -- Years, negative = BCE: inception and dissolution
            start_year INTEGER,
            end_year INTEGER,

            -- Number of Wikipedia articles: a rough notability measure
            sitelinks INTEGER NOT NULL DEFAULT 0,

            population INTEGER
        );

        -- Direct P31 (instance of) values
        CREATE TABLE wikidata_entity_class (
            qid TEXT NOT NULL,
            class_qid TEXT NOT NULL,

            PRIMARY KEY (qid, class_qid),
            FOREIGN KEY (qid) REFERENCES wikidata_entity(qid) ON DELETE CASCADE
        );

        CREATE TABLE wikidata_name (
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

            -- The name as written in Wikidata, original script
            name TEXT NOT NULL,

            -- Search key
            normalized_name TEXT NOT NULL,

            -- label | alias | official | native | short | name | nickname
            name_type TEXT NOT NULL,

            -- When the name was in use, if Wikidata says (years, negative = BCE)
            start_year INTEGER,
            end_year INTEGER,

            FOREIGN KEY (qid) REFERENCES wikidata_entity(qid) ON DELETE CASCADE
        );

        -- Links to other items: replaces, replaced_by, follows, followed_by,
        -- country, located_in, located_on, part_of, capital, capital_of,
        -- same_as. The target need not be in wikidata_entity
        CREATE TABLE wikidata_link (
            qid TEXT NOT NULL,
            rel TEXT NOT NULL,
            target_qid TEXT NOT NULL,
            start_year INTEGER,
            end_year INTEGER,

            FOREIGN KEY (qid) REFERENCES wikidata_entity(qid) ON DELETE CASCADE
        );

        -- GeoNames bridge (P1566), an INTEGER for joins with geoname
        CREATE TABLE wikidata_geonames (
            qid TEXT NOT NULL,
            geonames_id INTEGER NOT NULL,

            PRIMARY KEY (qid, geonames_id),
            FOREIGN KEY (qid) REFERENCES wikidata_entity(qid) ON DELETE CASCADE
        );

        -- Other gazetteers: pleiades, dare, trismegistos, topostext, tgn
        CREATE TABLE wikidata_external_id (
            qid TEXT NOT NULL,
            source TEXT NOT NULL,
            ext_id TEXT NOT NULL,

            PRIMARY KEY (qid, source, ext_id),
            FOREIGN KEY (qid) REFERENCES wikidata_entity(qid) ON DELETE CASCADE
        );
        """
        + language_tag_report_ddl(LANGUAGE_TAG_TABLE)
    )


def build_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE INDEX idx_wd_name_resolve
        ON wikidata_name (normalized_name, qid);

        CREATE INDEX idx_wd_name_hop
        ON wikidata_name (qid, lang);

        CREATE INDEX idx_wd_entity_class
        ON wikidata_entity_class (class_qid);

        CREATE INDEX idx_wd_link_from
        ON wikidata_link (qid, rel);

        CREATE INDEX idx_wd_link_to
        ON wikidata_link (target_qid, rel);

        CREATE INDEX idx_wd_geonames_id
        ON wikidata_geonames (geonames_id);

        CREATE INDEX idx_wd_external_id
        ON wikidata_external_id (source, ext_id);

        ANALYZE;
        """)
