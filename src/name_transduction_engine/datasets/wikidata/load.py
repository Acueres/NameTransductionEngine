import sqlite3

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
from . import classes as C
from .compact import CompactDataset
from .schema import LANGUAGE_TAG_TABLE, TABLES

# build_metadata keys: the registry fingerprint the `lang` columns were built
# with, and the compact dataset that was loaded
REGISTRY_FINGERPRINT_KEY = "wikidata_language_registry_fingerprint"
DATASET_ID_KEY = "wikidata_dataset_id"

REQUIRED_TABLES = {*TABLES, LANGUAGE_TAG_TABLE, "build_metadata"}

# Link relations stored in wikidata_link (the people/name relations stay in
# the compact dataset until those groups are loaded)
LOADED_RELATIONS = frozenset(
    {
        "replaces",
        "replaced_by",
        "follows",
        "followed_by",
        "country",
        "located_in",
        "located_on",
        "part_of",
        "capital",
        "capital_of",
        "same_as",
    }
)

_INSERT_ENTITY = """
    INSERT OR IGNORE INTO wikidata_entity (
        qid, entity_group, kind, lat, lon, start_year, end_year, sitelinks, population
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_INSERT_CLASS = (
    "INSERT OR IGNORE INTO wikidata_entity_class (qid, class_qid) VALUES (?, ?)"
)
_INSERT_NAME = """
    INSERT INTO wikidata_name (
        qid, wd_lang, lang, lang_script, lang_region, lang_variant, tag_status,
        name, normalized_name, name_type, start_year, end_year
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_INSERT_LINK = """
    INSERT INTO wikidata_link (qid, rel, target_qid, start_year, end_year)
    VALUES (?, ?, ?, ?, ?)
"""
_INSERT_GEONAMES = (
    "INSERT OR IGNORE INTO wikidata_geonames (qid, geonames_id) VALUES (?, ?)"
)
_INSERT_EXTERNAL = (
    "INSERT OR IGNORE INTO wikidata_external_id (qid, source, ext_id) VALUES (?, ?, ?)"
)


def load_compact_dataset(
    conn: sqlite3.Connection,
    dataset: CompactDataset,
    registry: LanguageRegistry,
    batch_size: int = 50_000,
) -> None:
    rows: dict[str, list[tuple]] = {
        "entity": [],
        "class": [],
        "name": [],
        "link": [],
        "geonames": [],
        "external": [],
    }
    canon = SourceTagCanonicalizer(registry, "wikidata")
    invalid_geonames_ids: list[tuple[str, str]] = []
    seen: set[str] = set()
    entities = 0

    source = "legacy locations file" if dataset.is_legacy else dataset.dataset_id
    print(f"Loading Wikidata ({source})...")
    if dataset.partial:
        print(
            "  note: this compact dataset is partial "
            f"({dataset.manifest.get('progress', 0):.1%} of the dump)"
        )

    for record in dataset.iter_records(C.LOADED_GROUPS):
        qid = record["qid"]
        if qid in seen:
            continue
        seen.add(qid)
        entities += 1

        coord = record.get("coord") or (None, None)
        rows["entity"].append(
            (
                qid,
                record["group"],
                record["kind"],
                coord[0],
                coord[1],
                record.get("start"),
                record.get("end"),
                record.get("sitelinks", 0),
                record.get("population"),
            )
        )
        rows["class"].extend((qid, c) for c in record.get("classes", ()))

        for name in record.get("names", ()):
            text = name["text"]
            normalized = normalize_name(text)
            if normalized is None:
                continue
            for wd_lang in name["langs"]:
                source_tag = canon(wd_lang)
                if source_tag.status == "non_name":
                    continue
                tag = source_tag.tag
                rows["name"].append(
                    (
                        qid,
                        wd_lang,
                        tag.lang if tag else None,
                        tag.script if tag else None,
                        tag.region if tag else None,
                        tag.variant if tag else None,
                        source_tag.status,
                        text,
                        normalized,
                        name["type"],
                        name.get("start"),
                        name.get("end"),
                    )
                )

        for link in record.get("links", ()):
            if link["rel"] in LOADED_RELATIONS:
                rows["link"].append(
                    (qid, link["rel"], link["qid"], link.get("start"), link.get("end"))
                )

        for source_name, values in (record.get("ids") or {}).items():
            for value in values:
                if source_name == "geonames":
                    geonames_id = _to_geonames_id(value)
                    if geonames_id is None:
                        invalid_geonames_ids.append((qid, str(value)))
                    else:
                        rows["geonames"].append((qid, geonames_id))
                else:
                    rows["external"].append((qid, source_name, str(value)))

        if len(rows["entity"]) >= batch_size or len(rows["name"]) >= batch_size:
            _flush(conn, rows)

    _flush(conn, rows)

    report = canon.report()
    write_language_tag_report(conn, LANGUAGE_TAG_TABLE, report)
    conn.commit()
    print(f"Loaded {entities:,} Wikidata entities.")
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


def _flush(conn: sqlite3.Connection, rows: dict[str, list[tuple]]) -> None:
    for key, sql in (
        ("entity", _INSERT_ENTITY),
        ("class", _INSERT_CLASS),
        ("name", _INSERT_NAME),
        ("link", _INSERT_LINK),
        ("geonames", _INSERT_GEONAMES),
        ("external", _INSERT_EXTERNAL),
    ):
        if rows[key]:
            conn.executemany(sql, rows[key])
            rows[key].clear()
    conn.commit()


def write_build_metadata(
    conn: sqlite3.Connection, registry: LanguageRegistry, dataset: CompactDataset
) -> None:
    def count(table: str) -> str:
        return str(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    metadata = {
        REGISTRY_FINGERPRINT_KEY: registry.fingerprint,
        DATASET_ID_KEY: dataset.dataset_id,
        "wikidata_entity_count": count("wikidata_entity"),
        "wikidata_name_count": count("wikidata_name"),
    }
    if dataset.manifest is not None:
        metadata["wikidata_snapshot"] = dataset.manifest["snapshot"]

    # Keys of the replaced schema
    conn.execute(
        "DELETE FROM build_metadata WHERE key IN "
        "('wikidata_location_count', 'wikidata_location_name_count')"
    )
    conn.executemany(
        "INSERT OR REPLACE INTO build_metadata (key, value) VALUES (?, ?)",
        metadata.items(),
    )
    conn.commit()


def loaded_dataset_id(db_path: Path) -> str | None:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.DatabaseError:
        return None
    try:
        row = conn.execute(
            "SELECT value FROM build_metadata WHERE key = ?", (DATASET_ID_KEY,)
        ).fetchone()
        return row[0] if row else None
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()


def is_wikidata_ready(db_path: Path, expected_dataset_id: str | None = None) -> bool:
    """The tables exist, are filled, match the current language registry and,
    when `expected_dataset_id` is given, were loaded from that dataset"""
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

            for table in ("wikidata_entity", "wikidata_name", LANGUAGE_TAG_TABLE):
                if conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0] == 0:
                    return False

            metadata = dict(
                conn.execute(
                    "SELECT key, value FROM build_metadata WHERE key IN (?, ?)",
                    (REGISTRY_FINGERPRINT_KEY, DATASET_ID_KEY),
                )
            )
            # The `lang` columns were built with the current language registry
            current = stored_registry_fingerprint(conn)
            built_with = metadata.get(REGISTRY_FINGERPRINT_KEY)
            if built_with is None or current is None or built_with != current:
                return False

            if (
                expected_dataset_id is not None
                and metadata.get(DATASET_ID_KEY) != expected_dataset_id
            ):
                return False

            return True
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False
