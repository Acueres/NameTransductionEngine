import sqlite3


def create_pack_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pack (
            pack_id          TEXT PRIMARY KEY,
            display_name     TEXT NOT NULL,
            origin           TEXT NOT NULL,
            version          TEXT NOT NULL,
            default_mode     TEXT,
            enabled          INTEGER NOT NULL DEFAULT 1,
            source_path      TEXT NOT NULL,
            manifest_json    TEXT NOT NULL,
            content_hash     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pack_build (
            pack_id          TEXT PRIMARY KEY,
            built_at         TEXT NOT NULL,
            content_hash     TEXT NOT NULL,
            FOREIGN KEY (pack_id) REFERENCES pack(pack_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS pack_fallback_step (
            pack_id          TEXT NOT NULL,
            step_index       INTEGER NOT NULL,
            step_type        TEXT NOT NULL,
            step_target      TEXT,
            config_json      TEXT NOT NULL,
            PRIMARY KEY (pack_id, step_index),
            FOREIGN KEY (pack_id) REFERENCES pack(pack_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS pack_place_override (
            pack_id                  TEXT NOT NULL,
            namespace                TEXT NOT NULL,
            external_id              TEXT NOT NULL,
            geonameid                INTEGER,
            output_name              TEXT NOT NULL,
            output_name_normalized   TEXT NOT NULL,
            valid_from               TEXT,
            valid_to                 TEXT,
            priority                 INTEGER NOT NULL,
            tags                     TEXT,
            note                     TEXT,
            source_file              TEXT NOT NULL,
            source_line              INTEGER NOT NULL,
            PRIMARY KEY (
                pack_id,
                namespace,
                external_id,
                output_name,
                valid_from,
                valid_to
            ),
            FOREIGN KEY (pack_id) REFERENCES pack(pack_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS pack_exonym (
            pack_id                  TEXT NOT NULL,
            input_normalized         TEXT NOT NULL,
            output_name              TEXT NOT NULL,
            output_name_normalized   TEXT NOT NULL,
            valid_from               TEXT,
            valid_to                 TEXT,
            priority                 INTEGER NOT NULL,
            note                     TEXT,
            source_file              TEXT NOT NULL,
            source_line              INTEGER NOT NULL,
            PRIMARY KEY (
                pack_id,
                input_normalized,
                output_name,
                valid_from,
                valid_to
            ),
            FOREIGN KEY (pack_id) REFERENCES pack(pack_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_pack_place_override_lookup
            ON pack_place_override(pack_id, namespace, external_id, priority DESC);

        CREATE INDEX IF NOT EXISTS idx_pack_place_override_geoname
            ON pack_place_override(pack_id, geonameid, priority DESC);

        CREATE INDEX IF NOT EXISTS idx_pack_exonym_lookup
            ON pack_exonym(pack_id, input_normalized, priority DESC);

        CREATE INDEX IF NOT EXISTS idx_pack_fallback_step
            ON pack_fallback_step(pack_id, step_index);
        """
    )
