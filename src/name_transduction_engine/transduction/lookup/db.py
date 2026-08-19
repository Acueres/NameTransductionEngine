import sqlite3
import pandas as pd

from src.name_transduction_engine.paths import DB_PATH


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def run_query(sql: str, params: dict = {}) -> pd.DataFrame:
    with _get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)
