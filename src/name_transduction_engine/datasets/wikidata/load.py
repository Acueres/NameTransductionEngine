import gzip
import sqlite3
import orjson

from pathlib import Path
from name_transduction_engine.normalization.name_normalization import normalize_name
from name_transduction_engine.normalization.language_code_normalization import (
    LanguageRegistry,
    SourceTagCanonicalizer,
)
from name_transduction_engine.datasets.language_codes.data_provision import (
    print_language_tag_summary,
    stored_registry_fingerprint,
    write_language_tag_report,
)
from .schema import LANGUAGE_TAG_TABLE

# build_metadata key: the registry fingerprint the `lang` columns were built with
REGISTRY_FINGERPRINT_KEY = "wikidata_language_registry_fingerprint"

REQUIRED_TABLES = {
    "wikidata_location",
    "wikidata_location_geonames",
    "wikidata_location_p31",
    "wikidata_location_name",
    LANGUAGE_TAG_TABLE,
    "build_metadata",
}

REQUIRED_NAME_COLUMNS = {
    "qid",
    "wd_lang",
    "lang",
    "lang_script",
    "lang_region",
    "lang_variant",
    "tag_status",
    "name",
    "normalized_name",
    "term_type",
}


def load_locations_dataset(
    conn: sqlite3.Connection,
    dataset_path: Path,
    registry: LanguageRegistry,
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
    insert_name = """
        INSERT OR IGNORE INTO wikidata_location_name (
            qid,
            wd_lang,
            lang,
            lang_script,
            lang_region,
            lang_variant,
            tag_status,
            name,
            normalized_name,
            term_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    location_rows = []
    geonames_rows = []
    p31_rows = []
    name_rows = []

    canon = SourceTagCanonicalizer(registry, "wikidata")
    invalid_geonames_ids: list[tuple[str, str]] = []

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

            for raw_id in record.get("geonames_ids", []):
                geonames_id = _to_geonames_id(raw_id)
                if geonames_id is None:
                    invalid_geonames_ids.append((qid, str(raw_id)))
                    continue
                geonames_rows.append((qid, geonames_id))

            for p31_qid in record.get("p31_qids", []):
                p31_rows.append((qid, p31_qid))

            for name in record.get("names", []):
                wd_lang = name["wd_lang"]
                source_tag = canon(wd_lang)
                if source_tag.status == "non_name":
                    continue
                tag = source_tag.tag
                lang = tag.lang if tag else None
                normalized_name = normalize_name(name["name"])

                name_rows.append(
                    (
                        qid,
                        wd_lang,
                        lang,
                        tag.script if tag else None,
                        tag.region if tag else None,
                        tag.variant if tag else None,
                        source_tag.status,
                        name["name"],
                        normalized_name,
                        name.get("term_type", "label"),
                    )
                )

            if len(location_rows) >= batch_size or len(name_rows) >= batch_size:
                _flush_batches(
                    conn,
                    insert_location,
                    insert_geonames,
                    insert_p31,
                    insert_name,
                    location_rows,
                    geonames_rows,
                    p31_rows,
                    name_rows,
                )

    _flush_batches(
        conn,
        insert_location,
        insert_geonames,
        insert_p31,
        insert_name,
        location_rows,
        geonames_rows,
        p31_rows,
        name_rows,
    )

    report = canon.report()
    write_language_tag_report(conn, LANGUAGE_TAG_TABLE, report)
    conn.commit()
    print_language_tag_summary(report)

    if invalid_geonames_ids:
        examples = ", ".join(f"{q}={v!r}" for q, v in invalid_geonames_ids[:5])
        print(
            f"Skipped {len(invalid_geonames_ids):,} non-numeric GeoNames IDs "
            f"(P1566), e.g. {examples}"
        )


def _to_geonames_id(value) -> int | None:
    text = str(value).strip()
    return int(text) if text.isascii() and text.isdigit() else None


def _flush_batches(
    conn: sqlite3.Connection,
    insert_location: str,
    insert_geonames: str,
    insert_p31: str,
    insert_name: str,
    location_rows: list,
    geonames_rows: list,
    p31_rows: list,
    name_rows: list,
) -> None:
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


def write_build_metadata(conn: sqlite3.Connection, registry: LanguageRegistry) -> None:
    metadata = {
        REGISTRY_FINGERPRINT_KEY: registry.fingerprint,
        "wikidata_location_count": str(
            conn.execute("SELECT COUNT(*) FROM wikidata_location").fetchone()[0]
        ),
        "wikidata_location_name_count": str(
            conn.execute("SELECT COUNT(*) FROM wikidata_location_name").fetchone()[0]
        ),
    }

    conn.executemany(
        "INSERT OR REPLACE INTO build_metadata (key, value) VALUES (?, ?)",
        metadata.items(),
    )
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

            # A database built before the registry has geo_lang, not lang
            name_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(wikidata_location_name);")
            }
            if not REQUIRED_NAME_COLUMNS <= name_columns:
                return False

            for table in (
                "wikidata_location",
                "wikidata_location_name",
                LANGUAGE_TAG_TABLE,
            ):
                if conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0] == 0:
                    return False

            # The `lang` columns were built with the current language registry
            built_with = conn.execute(
                "SELECT value FROM build_metadata WHERE key = ?",
                (REGISTRY_FINGERPRINT_KEY,),
            ).fetchone()
            current = stored_registry_fingerprint(conn)
            if built_with is None or current is None or built_with[0] != current:
                return False

            return True
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False
