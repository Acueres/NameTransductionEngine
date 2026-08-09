import sqlite3


def create_pack_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
                -- Pack identity: what a pack IS.
        -- Operational metadata (paths, hashes) lives in pack_install.
        CREATE TABLE IF NOT EXISTS pack (
            pack_id         TEXT    PRIMARY KEY,
            display_name    TEXT    NOT NULL,
            bcp47           TEXT    NOT NULL,
            version         TEXT    NOT NULL,
            kind            TEXT    NOT NULL,
            default_mode    TEXT    NOT NULL,
            manifest_json   TEXT    NOT NULL,
            enabled         INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            installed_at    TEXT    NOT NULL DEFAULT (datetime('now'))
        );
 
        -- Operational metadata: paths, content hash, build timestamp.
        -- Kept separate so queries on pack identity are never joined with
        -- tooling concerns.
        CREATE TABLE IF NOT EXISTS pack_install (
            pack_id         TEXT    PRIMARY KEY
                                    REFERENCES pack(pack_id) ON DELETE CASCADE,
            source_path     TEXT    NOT NULL,
            content_hash    TEXT    NOT NULL,
            built_at        TEXT    NOT NULL DEFAULT (datetime('now'))
        );
 
        -- Ordered resolution pipeline for a pack.  Steps are tried in
        -- step_index order.  See pack.yaml [pipeline] for step_type values.
        CREATE TABLE IF NOT EXISTS pack_pipeline_step (
            pack_id         TEXT    NOT NULL
                                    REFERENCES pack(pack_id) ON DELETE CASCADE,
            step_index      INTEGER NOT NULL,
            step_type       TEXT    NOT NULL,
            enabled         INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            config_json     TEXT    NOT NULL DEFAULT '{}',
            PRIMARY KEY (pack_id, step_index)
        );
 
        -- Entity-anchored name mappings.
        -- Maps (namespace, entity_id) → output_name, optionally date-scoped.
        -- output_name_normal is a generated column; the importer must never
        -- supply it directly.
        CREATE TABLE IF NOT EXISTS pack_entity_name (
            pack_id             TEXT    NOT NULL
                                        REFERENCES pack(pack_id) ON DELETE CASCADE,
            namespace           TEXT    NOT NULL,
            entity_id           TEXT    NOT NULL,
            output_name         TEXT    NOT NULL,
            output_name_normal  TEXT    GENERATED ALWAYS AS (lower(output_name)) STORED,
            valid_from          TEXT    NOT NULL,
            valid_to            TEXT    NOT NULL,
            priority            INTEGER NOT NULL DEFAULT 100,
            note                TEXT,
            PRIMARY KEY (pack_id, namespace, entity_id, output_name, valid_from, valid_to)
        );
 
        -- Tags on entity name rows, stored one-per-row for queryability.
        CREATE TABLE IF NOT EXISTS pack_entity_name_tag (
            pack_id         TEXT    NOT NULL,
            namespace       TEXT    NOT NULL,
            entity_id       TEXT    NOT NULL,
            output_name     TEXT    NOT NULL,
            valid_from      TEXT,
            valid_to        TEXT,
            tag             TEXT    NOT NULL,
            PRIMARY KEY (pack_id, namespace, entity_id, output_name,
                         valid_from, valid_to, tag),
            FOREIGN KEY (pack_id, namespace, entity_id, output_name,
                         valid_from, valid_to)
                REFERENCES pack_entity_name(pack_id, namespace, entity_id,
                                            output_name, valid_from, valid_to)
                ON DELETE CASCADE
        );
 
        -- String-anchored exonyms.
        -- Maps a normalised input string → output_name when no entity ID is
        -- known.  Scores lower than entity_lookup hits; see pack.yaml [ranking].
        CREATE TABLE IF NOT EXISTS pack_string_exonym (
            pack_id             TEXT    NOT NULL
                                        REFERENCES pack(pack_id) ON DELETE CASCADE,
            input_normal        TEXT    NOT NULL,
            output_name         TEXT    NOT NULL,
            output_name_normal  TEXT    GENERATED ALWAYS AS (lower(output_name)) STORED,
            valid_from          TEXT,
            valid_to            TEXT,
            priority            INTEGER NOT NULL DEFAULT 100,
            note                TEXT,
            PRIMARY KEY (pack_id, input_normal, output_name, valid_from, valid_to)
        );
 
        -- One record per imported data file.
        CREATE TABLE IF NOT EXISTS pack_import (
            import_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            pack_id         TEXT    NOT NULL
                                    REFERENCES pack(pack_id) ON DELETE CASCADE,
            source_file     TEXT    NOT NULL,
            imported_at     TEXT    NOT NULL DEFAULT (datetime('now'))
        );
 
        -- Maps each data row back to the import batch that produced it.
        -- row_pk_json is a canonical JSON encoding of the row's primary key.
        CREATE TABLE IF NOT EXISTS pack_row_provenance (
            pack_id         TEXT    NOT NULL,
            table_name      TEXT    NOT NULL
                                    CHECK (table_name IN (
                                        'pack_entity_name',
                                        'pack_string_exonym'
                                    )),
            row_pk_json     TEXT    NOT NULL,
            import_id       INTEGER NOT NULL
                                    REFERENCES pack_import(import_id)
                                    ON DELETE CASCADE,
            source_line     INTEGER,
            PRIMARY KEY (pack_id, table_name, row_pk_json)
        );
 
        CREATE INDEX IF NOT EXISTS idx_pen_entity
            ON pack_entity_name(pack_id, namespace, entity_id);
 
        CREATE INDEX IF NOT EXISTS idx_pen_entity_priority
            ON pack_entity_name(pack_id, namespace, entity_id, priority DESC);
 
        CREATE INDEX IF NOT EXISTS idx_pent_tag
            ON pack_entity_name_tag(pack_id, tag);
 
        CREATE INDEX IF NOT EXISTS idx_pse_input
            ON pack_string_exonym(pack_id, input_normal);
 
        CREATE INDEX IF NOT EXISTS idx_pse_input_priority
            ON pack_string_exonym(pack_id, input_normal, priority DESC);
 
        CREATE INDEX IF NOT EXISTS idx_pps_order
            ON pack_pipeline_step(pack_id, step_index);
 
        CREATE INDEX IF NOT EXISTS idx_prp_import
            ON pack_row_provenance(import_id);
        """
    )
