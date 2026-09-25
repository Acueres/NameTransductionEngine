import sqlite3
import pandas as pd

from pathlib import Path


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def run_query(db_path: str | Path, sql: str, params: dict = {}) -> pd.DataFrame:
    with get_conn(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)
