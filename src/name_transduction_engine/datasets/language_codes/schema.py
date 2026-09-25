import sqlite3


def create_schema(conn: sqlite3.Connection) -> None:
    """Drops and recreates the registry tables. Name tables reference them, so
    rebuilding the registry means reloading both datasets"""
    conn.executescript("""
        DROP TABLE IF EXISTS language_alias;
        DROP TABLE IF EXISTS language_retirement;
        DROP TABLE IF EXISTS language_subtag;
        DROP TABLE IF EXISTS language;

        CREATE TABLE language (
                code           TEXT PRIMARY KEY,
                iso639_3       TEXT NOT NULL UNIQUE,
                iso639_1       TEXT UNIQUE,
                iso639_2b      TEXT,
                iso639_2t      TEXT,
                name           TEXT NOT NULL,
                scope          TEXT NOT NULL CHECK (scope IN ('individual', 'macrolanguage')),
                macrolanguage  TEXT REFERENCES language(code)
            );
        
            CREATE TABLE language_alias (
                code   TEXT NOT NULL REFERENCES language(code),
                alias  TEXT NOT NULL,
                PRIMARY KEY (code, alias)
            );
        
            CREATE TABLE language_retirement (
                code       TEXT PRIMARY KEY,
                change_to  TEXT
            );
        
            CREATE TABLE language_subtag (
                kind    TEXT NOT NULL CHECK (kind IN ('script', 'region')),
                subtag  TEXT NOT NULL,
                PRIMARY KEY (kind, subtag)
            );
        """)


def language_tag_report_ddl(table: str) -> str:
    """DDL for a dataset's language tag report table. Each dataset owns its
    table and recreates it with its own schema; the columns match
    `write_language_tag_report`"""
    return f"""
        DROP TABLE IF EXISTS {table};

        CREATE TABLE {table} (
            raw_tag       TEXT PRIMARY KEY,
            tag_status    TEXT NOT NULL,
            lang          TEXT,
            lang_script   TEXT,
            lang_region   TEXT,
            lang_variant  TEXT,
            row_count     INTEGER NOT NULL,
            note          TEXT
        );
    """
