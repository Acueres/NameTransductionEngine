import sqlite3
import pandas as pd

from pathlib import Path
from contextlib import closing


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def run_query(
    db_path: str | Path, sql: str, params: dict | None = None
) -> pd.DataFrame:
    with closing(get_conn(db_path)) as conn:
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except pd.errors.DatabaseError as exc:
            # pandas wraps sqlite errors in its own type; callers expect
            # sqlite3.DatabaseError (a missing table, a damaged file)
            if isinstance(exc.__cause__, sqlite3.DatabaseError):
                raise exc.__cause__ from None
            raise
