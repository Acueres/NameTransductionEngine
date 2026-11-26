import sqlite3


def create_schema(conn: sqlite3.Connection) -> None:
    """Drops and recreates the country code table"""
    conn.executescript("""
        DROP TABLE IF EXISTS country_codes;

        CREATE TABLE country_codes (
            code  TEXT PRIMARY KEY,
            name  TEXT NOT NULL
        );
        """)
