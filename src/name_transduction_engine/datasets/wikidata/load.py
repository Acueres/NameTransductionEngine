import gzip
import sqlite3
import orjson

from pathlib import Path
from .build import normalize_wd_lang

REQUIRED_TABLES = {
    "wikidata_location",
    "wikidata_location_geonames",
    "wikidata_location_p31",
    "wikidata_lang_norm",
    "wikidata_location_name",
}


def load_locations_dataset(
    conn: sqlite3.Connection,
    dataset_path: Path,
    batch_size: int = 50_000,
) -> None:
    insert_location = """
        INSERT INTO wikidata_location (qid, kind, lat, lon)
        VALUES (?, ?, ?, ?)
    """
    insert_geonames = """
        INSERT OR IGNORE INTO wikidata_location_geonames (qid, geonames_id)
        VALUES (?, ?)
    """
    insert_p31 = """
        INSERT OR IGNORE INTO wikidata_location_p31 (qid, p31_qid)
        VALUES (?, ?)
    """
    insert_lang = """
        INSERT OR IGNORE INTO wikidata_lang_norm
            (wd_lang, geo_lang, iso639_3, iso639_1)
        VALUES (?, ?, ?, ?)
    """
    insert_name = """
        INSERT OR IGNORE INTO wikidata_location_name
            (qid, wd_lang, geo_lang, name, name_norm, term_type)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    location_rows = []
    geonames_rows = []
    p31_rows = []
    lang_rows = []
    name_rows = []
    seen_langs = set()

    print("Loading Wikidata locations dataset...")

    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = orjson.loads(line)
            qid = record["qid"]

            location_rows.append(
                (
                    qid,
                    record["kind"],
                    record.get("lat"),
                    record.get("lon"),
                )
            )

            for geonames_id in record.get("geonames_ids", []):
                geonames_rows.append((qid, geonames_id))

            for p31_qid in record.get("p31_qids", []):
                p31_rows.append((qid, p31_qid))

            for name in record.get("names", []):
                wd_lang = name["wd_lang"]

                if wd_lang not in seen_langs:
                    norm = normalize_wd_lang(wd_lang)
                    lang_rows.append(
                        (
                            norm["wd_lang"],
                            norm["geo_lang"],
                            norm["iso639_3"],
                            norm["iso639_1"],
                        )
                    )
                    seen_langs.add(wd_lang)

                name_rows.append(
                    (
                        qid,
                        wd_lang,
                        name.get("geo_lang"),
                        name["name"],
                        name["name_norm"],
                        name.get("term_type", "label"),
                    )
                )

            if len(location_rows) >= batch_size or len(name_rows) >= batch_size:
                _flush_batches(
                    conn,
                    insert_location,
                    insert_geonames,
                    insert_p31,
                    insert_lang,
                    insert_name,
                    location_rows,
                    geonames_rows,
                    p31_rows,
                    lang_rows,
                    name_rows,
                )

    _flush_batches(
        conn,
        insert_location,
        insert_geonames,
        insert_p31,
        insert_lang,
        insert_name,
        location_rows,
        geonames_rows,
        p31_rows,
        lang_rows,
        name_rows,
    )


def _flush_batches(
    conn: sqlite3.Connection,
    insert_location: str,
    insert_geonames: str,
    insert_p31: str,
    insert_lang: str,
    insert_name: str,
    location_rows: list,
    geonames_rows: list,
    p31_rows: list,
    lang_rows: list,
    name_rows: list,
) -> None:
    if lang_rows:
        conn.executemany(insert_lang, lang_rows)
        lang_rows.clear()
    if location_rows:
        conn.executemany(insert_location, location_rows)
        location_rows.clear()
    if geonames_rows:
        conn.executemany(insert_geonames, geonames_rows)
        geonames_rows.clear()
    if p31_rows:
        conn.executemany(insert_p31, p31_rows)
        p31_rows.clear()
    if name_rows:
        conn.executemany(insert_name, name_rows)
        name_rows.clear()
    conn.commit()


def is_wikidata_ready(db_path: Path) -> bool:
    if not db_path.exists():
        return False

    try:
        conn = sqlite3.connect(db_path)
        try:
            existing_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table';"
                )
            }
            if not REQUIRED_TABLES <= existing_tables:
                return False

            location_count = conn.execute(
                "SELECT COUNT(*) FROM wikidata_location;"
            ).fetchone()[0]
            name_count = conn.execute(
                "SELECT COUNT(*) FROM wikidata_location_name;"
            ).fetchone()[0]
            lang_count = conn.execute(
                "SELECT COUNT(*) FROM wikidata_lang_norm;"
            ).fetchone()[0]

            return location_count > 0 and name_count > 0 and lang_count > 0
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False
