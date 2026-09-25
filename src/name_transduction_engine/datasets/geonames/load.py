import csv
import sqlite3

from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile
from name_transduction_engine.paths import RAW_DIR_GEONAMES
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
REGISTRY_FINGERPRINT_KEY = "geonames_language_registry_fingerprint"

GEONAME_COLUMNS = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
]

ALT_COLUMNS = [
    "alternate_name_id",
    "geonameid",
    "isolanguage",
    "alternate_name",
    "is_preferred_name",
    "is_short_name",
    "is_colloquial",
    "is_historic",
    "from_date",
    "to_date",
]

GEONAME_KEEP_COLUMNS = [
    "geonameid",
    "name",
    "asciiname",
    "latitude",
    "longitude",
    "country_code",
    "admin1_code",
    "admin2_code",
    "feature_class",
    "feature_code",
    "population",
]

GEONAME_COLUMN_INDEX = {name: index for index, name in enumerate(GEONAME_COLUMNS)}
GEONAME_KEEP_INDICES = [GEONAME_COLUMN_INDEX[name] for name in GEONAME_KEEP_COLUMNS]

REQUIRED_TABLE_COLUMNS = {
    "geoname": {
        "geonameid",
        "name",
        "asciiname",
        "country_code",
        "admin1_code",
        "admin2_code",
        "feature_class",
        "feature_code",
        "population",
    },
    "alternate_name": {
        "alternate_name_id",
        "geonameid",
        "isolanguage",
        "alternate_name",
        "is_preferred_name",
        "is_short_name",
        "is_colloquial",
        "is_historic",
        "from_date",
        "to_date",
        "lang",
        "lang_script",
        "lang_region",
        "lang_variant",
        "tag_status",
        "normalized_name",
    },
    LANGUAGE_TAG_TABLE: {
        "raw_tag",
        "tag_status",
        "lang",
        "row_count",
    },
}

REQUIRED_POPULATED_TABLES = ("geoname", "alternate_name", LANGUAGE_TAG_TABLE)


def is_geonames_ready(db_path: Path) -> bool:
    if not db_path.exists():
        return False

    try:
        conn = sqlite3.connect(db_path)
        try:
            # Required tables exist
            existing_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table';"
                )
            }

            if not REQUIRED_TABLE_COLUMNS.keys() <= existing_tables:
                return False

            # Required columns exist
            for table_name, expected_columns in REQUIRED_TABLE_COLUMNS.items():
                actual_columns = {
                    row[1] for row in conn.execute(f"PRAGMA table_info({table_name});")
                }
                if not expected_columns <= actual_columns:
                    return False

            # Required tables are populated
            for table_name in REQUIRED_POPULATED_TABLES:
                row_count = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name};"
                ).fetchone()[0]
                if row_count == 0:
                    return False

            # The `lang` columns were built with the current language registry.
            # A rebuilt registry with different contents makes this false, so
            # GeoNames is rebuilt against it
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


def load_all_data(conn: sqlite3.Connection, registry: LanguageRegistry) -> None:
    print("Loading geoname table...")
    _load_geoname_table(conn, RAW_DIR_GEONAMES / "allCountries.zip")

    print("Loading alternate_name table...")
    canon = SourceTagCanonicalizer(registry, "geonames")
    _load_alternate_name_table(conn, RAW_DIR_GEONAMES / "alternateNamesV2.zip", canon)

    report = canon.report()
    write_language_tag_report(conn, LANGUAGE_TAG_TABLE, report)
    conn.commit()
    print_language_tag_summary(report)


def _load_geoname_table(
    conn: sqlite3.Connection,
    zip_path: Path,
    batch_size: int = 50_000,
) -> None:
    insert_sql = """
        INSERT INTO geoname (
            geonameid,
            name,
            asciiname,
            latitude,
            longitude,
            country_code,
            admin1_code,
            admin2_code,
            feature_class,
            feature_code,
            population,
            normalized_name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    batch: list[tuple] = []

    with ZipFile(zip_path) as zf:
        member_name = _find_zip_member(zf, preferred_name="allCountries.txt")
        with zf.open(member_name) as raw_file:
            text_file = TextIOWrapper(raw_file, encoding="utf-8", newline="")
            reader = csv.reader(text_file, delimiter="\t")

            for row in reader:
                if not row:
                    continue

                values = [
                    row[index] if index < len(row) else ""
                    for index in GEONAME_KEEP_INDICES
                ]

                name = values[1].strip()

                record = (
                    _to_int(values[0]),  # geonameid
                    name,
                    _empty_to_none(values[2]),
                    _to_float(values[3]),
                    _to_float(values[4]),
                    _empty_to_none(values[5]),
                    _empty_to_none(values[6]),
                    _empty_to_none(values[7]),
                    _empty_to_none(values[8]),
                    _empty_to_none(values[9]),
                    _to_int(values[10]),  # population
                    normalize_name(name),
                )

                batch.append(record)

                if len(batch) >= batch_size:
                    conn.executemany(insert_sql, batch)
                    conn.commit()
                    batch.clear()

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()


def _load_alternate_name_table(
    conn: sqlite3.Connection,
    zip_path: Path,
    canon: SourceTagCanonicalizer,
    batch_size: int = 50_000,
) -> None:
    insert_sql = """
        INSERT INTO alternate_name (
            alternate_name_id,
            geonameid,
            isolanguage,
            alternate_name,
            is_preferred_name,
            is_short_name,
            is_colloquial,
            is_historic,
            from_date,
            to_date,
            lang,
            lang_script,
            lang_region,
            lang_variant,
            tag_status,
            normalized_name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    batch: list[tuple] = []

    with ZipFile(zip_path) as zf:
        with zf.open("alternateNamesV2.txt") as raw_file:
            text_file = TextIOWrapper(raw_file, encoding="utf-8", newline="")
            reader = csv.reader(text_file, delimiter="\t")

            for row in reader:
                if not row:
                    continue

                padded_row = row + [""] * (len(ALT_COLUMNS) - len(row))

                alternate_name_id = _to_int(padded_row[0])
                geonameid = _to_int(padded_row[1])
                isolanguage = padded_row[2].strip()
                alternate_name = padded_row[3].strip()

                # Every row goes through the canonicalizer, including the ones
                # skipped next, so the tag report counts them too
                source_tag = canon(isolanguage)
                if source_tag.status == "non_name":
                    continue
                tag = source_tag.tag

                record = (
                    alternate_name_id,
                    geonameid,
                    isolanguage,
                    alternate_name,
                    _flag_to_int(padded_row[4]),
                    _flag_to_int(padded_row[5]),
                    _flag_to_int(padded_row[6]),
                    _flag_to_int(padded_row[7]),
                    _empty_to_none(padded_row[8].strip()),
                    _empty_to_none(padded_row[9].strip()),
                    tag.lang if tag else None,
                    tag.script if tag else None,
                    tag.region if tag else None,
                    tag.variant if tag else None,
                    source_tag.status,
                    normalize_name(alternate_name),
                )

                batch.append(record)

                if len(batch) >= batch_size:
                    conn.executemany(insert_sql, batch)
                    conn.commit()
                    batch.clear()

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()


def write_build_metadata(conn: sqlite3.Connection, registry: LanguageRegistry) -> None:
    metadata = {
        REGISTRY_FINGERPRINT_KEY: registry.fingerprint,
        "source": "GeoNames",
        "geoname_count": str(
            conn.execute("SELECT COUNT(*) FROM geoname").fetchone()[0]
        ),
        "alternate_name_count": str(
            conn.execute("SELECT COUNT(*) FROM alternate_name").fetchone()[0]
        ),
    }

    conn.executemany(
        "INSERT OR REPLACE INTO build_metadata (key, value) VALUES (?, ?)",
        metadata.items(),
    )
    conn.commit()


def _find_zip_member(zf: ZipFile, preferred_name: str | None = None) -> str:
    names = zf.namelist()

    if preferred_name and preferred_name in names:
        return preferred_name

    txt_names = [name for name in names if name.endswith(".txt")]
    if len(txt_names) == 1:
        return txt_names[0]

    raise ValueError(f"Could not determine text member in archive: {names}")


def _flag_to_int(value: str) -> int:
    return 1 if value.strip() == "1" else 0


def _empty_to_none(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _to_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


def _to_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)
