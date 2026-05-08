import bz2
import gzip
import sqlite3
import unicodedata

import langcodes
import orjson

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from name_transduction_engine.paths import RAW_DIR_WIKIDATA

RAW_DUMP_PATH = RAW_DIR_WIKIDATA / "latest-all.json.bz2"
LOCATIONS_DATASET_PATH = RAW_DIR_WIKIDATA / "wikidata_locations.jsonl.gz"

LOCATION_CLASS_QIDS = {
    "Q6256": "country",
    "Q3624078": "country",
    "Q3024240": "historical_country",
    "Q56061": "admin_division",
    "Q108640": "admin_division",
    "Q34876": "province",
    "Q28575": "county",
    "Q149621": "district",
    "Q35657": "state_subdivision",
    "Q107390": "federated_state",
    "Q486972": "settlement",
    "Q515": "city",
    "Q1549591": "city",
    "Q200250": "city",
    "Q174844": "city",
    "Q3957": "town",
    "Q532": "village",
    "Q5084": "hamlet",
    "Q15284": "municipality",
    "Q4022": "river",
    "Q23397": "lake",
    "Q165": "sea",
    "Q9430": "ocean",
    "Q39594": "bay",
    "Q1322134": "gulf",
    "Q37901": "strait",
    "Q46831": "mountain_range",
    "Q93352": "coast",
    "Q34763": "peninsula",
    "Q23442": "island",
    "Q33837": "archipelago",
    "Q8514": "desert",
    "Q8072": "volcano",
}

LOCATION_KIND_PRIORITY = [
    "country",
    "historical_country",
    "city",
    "town",
    "municipality",
    "village",
    "hamlet",
    "settlement",
    "island",
    "archipelago",
    "peninsula",
    "coast",
    "mountain_range",
    "desert",
    "volcano",
    "river",
    "lake",
    "sea",
    "ocean",
    "bay",
    "gulf",
    "strait",
    "state_subdivision",
    "federated_state",
    "province",
    "county",
    "district",
    "admin_division",
]

WIKIDATA_LANG_OVERRIDES = {
    "als": "gsw",
    "simple": "en",
    "be-x-old": "be",
    "be-tarask": "be",
    "zh-classical": "lzh",
    "zh-min-nan": "nan",
    "zh-yue": "yue",
    "fiu-vro": "vro",
    "bat-smg": "sgs",
    "roa-rup": "rup",
    "nrm": "nrf",
    "cbk-zam": "cbk",
}

SPECIAL_GEO_LANG_OVERRIDES = {
    "map-bms": "map-bms",
    "roa-tara": "roa-tara",
}

HARDCODED_LANG_NORMS: dict[str, dict[str, str | None]] = {
    "tl": {"wd_lang": "tl", "geo_lang": "tl", "iso639_3": "tgl", "iso639_1": "tl"},
}

REQUIRED_TABLES = {
    "wikidata_location",
    "wikidata_location_geonames",
    "wikidata_location_p31",
    "wikidata_lang_norm",
    "wikidata_location_name",
}


def ensure_locations_dataset(raw_dump_path: Path, force: bool) -> Path:
    RAW_DIR_WIKIDATA.mkdir(parents=True, exist_ok=True)

    jsonl_part_path, gzip_part_path = _dataset_part_paths(LOCATIONS_DATASET_PATH)

    if LOCATIONS_DATASET_PATH.exists() and not force:
        return LOCATIONS_DATASET_PATH

    if force:
        for path in (LOCATIONS_DATASET_PATH, jsonl_part_path, gzip_part_path):
            if path.exists():
                path.unlink()

    if jsonl_part_path.exists():
        print(f"Using existing partial Wikidata locations file: {jsonl_part_path}")
        _finish_locations_dataset(
            jsonl_part_path, gzip_part_path, LOCATIONS_DATASET_PATH
        )
        return LOCATIONS_DATASET_PATH

    if raw_dump_path.exists():
        _build_locations_dataset(raw_dump_path, LOCATIONS_DATASET_PATH)
        return LOCATIONS_DATASET_PATH

    return _fetch_locations_dataset(LOCATIONS_DATASET_PATH)


def _fetch_locations_dataset(output_path: Path) -> Path:
    raise NotImplementedError(
        "Fetching hosted wikidata_locations.jsonl.gz is not implemented yet."
    )


def _dataset_part_paths(output_path: Path) -> tuple[Path, Path]:
    jsonl_path = output_path.with_suffix("")
    jsonl_part_path = jsonl_path.with_suffix(jsonl_path.suffix + ".part")
    gzip_part_path = output_path.with_suffix(output_path.suffix + ".part")
    return jsonl_part_path, gzip_part_path


def _build_locations_dataset(raw_dump_path: Path, output_path: Path) -> None:
    print("Building compact Wikidata locations dataset...")

    jsonl_part_path, gzip_part_path = _dataset_part_paths(output_path)
    count = 0
    scanned = 0

    try:
        with jsonl_part_path.open("w", encoding="utf-8", newline="") as out:
            for entity in _iter_wikidata_entities(raw_dump_path):
                scanned += 1

                if scanned % 1_000_000 == 0:
                    print(
                        f"Scanned {scanned:,} entities; extracted {count:,} locations..."
                    )

                record = _extract_location_record(entity)
                if record is None:
                    continue

                out.write(orjson.dumps(record).decode("utf-8"))
                out.write("\n")
                count += 1

                if count % 100_000 == 0:
                    print(f"Extracted {count:,} locations...")
    except (OSError, EOFError, UnicodeDecodeError, orjson.JSONDecodeError) as exc:
        print(
            "Stopped while reading Wikidata dump; "
            f"keeping partial dataset ({type(exc).__name__}: {exc})."
        )

    if not jsonl_part_path.exists() or jsonl_part_path.stat().st_size == 0:
        raise RuntimeError("Wikidata extraction produced no usable partial dataset.")

    _finish_locations_dataset(jsonl_part_path, gzip_part_path, output_path)


def _finish_locations_dataset(
    jsonl_part_path: Path,
    gzip_part_path: Path,
    output_path: Path,
) -> None:
    print("Compressing compact Wikidata locations dataset...")

    count = 0
    with jsonl_part_path.open("rb") as src:
        with gzip.open(gzip_part_path, "wb", compresslevel=6) as dst:
            for line_number, line in enumerate(src, start=1):
                if not line.strip():
                    continue

                try:
                    orjson.loads(line)
                except orjson.JSONDecodeError:
                    print(f"Ignoring invalid final JSONL row at line {line_number:,}.")
                    break

                dst.write(line)
                count += 1

                if count % 100_000 == 0:
                    print(f"Compressed {count:,} locations...")

    if count == 0:
        raise RuntimeError("Wikidata partial dataset contains no valid JSONL rows.")

    jsonl_part_path.unlink()
    gzip_part_path.replace(output_path)

    print(f"Wikidata locations dataset built: {output_path} ({count:,} locations)")


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
    loaded = 0

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
                    norm = _normalize_wd_lang(wd_lang)
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

            loaded += 1
            if loaded % 100_000 == 0:
                print(f"Prepared {loaded:,} Wikidata location records...")

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


def _extract_location_record(entity: dict[str, Any]) -> dict[str, Any] | None:
    kind = _classify_location(entity)
    if kind is None:
        return None

    coords = _coordinate_value(entity)
    qid = entity["id"]

    return {
        "qid": qid,
        "kind": kind,
        "p31_qids": _dedupe_preserve_order(_item_ids(entity, "P31")),
        "geonames_ids": _dedupe_preserve_order(_string_values(entity, "P1566")),
        "lat": coords[0] if coords else None,
        "lon": coords[1] if coords else None,
        "names": _extract_label_rows(entity),
    }


def _iter_wikidata_entities(path: Path) -> Iterator[dict[str, Any]]:
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line in {"[", "]"}:
                continue
            if line.endswith(","):
                line = line[:-1]
            if line:
                yield orjson.loads(line)


def _extract_label_rows(entity: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []

    for lang, label_obj in entity.get("labels", {}).items():
        name = label_obj.get("value")
        if not name:
            continue

        wd_lang = label_obj.get("language", lang).strip().lower()
        lang_norm = _normalize_wd_lang(wd_lang)

        rows.append(
            {
                "wd_lang": wd_lang,
                "geo_lang": lang_norm["geo_lang"],
                "name": name,
                "name_norm": _normalize_name(name),
                "term_type": "label",
            }
        )

    return rows


def _claim_values(entity: dict[str, Any], prop: str) -> list[dict[str, Any]]:
    values = []

    for stmt in entity.get("claims", {}).get(prop, []):
        if stmt.get("rank") == "deprecated":
            continue

        mainsnak = stmt.get("mainsnak", {})
        if mainsnak.get("snaktype") != "value":
            continue

        datavalue = mainsnak.get("datavalue")
        if not datavalue:
            continue

        values.append(
            {
                "value": datavalue.get("value"),
                "type": datavalue.get("type"),
                "datatype": mainsnak.get("datatype"),
                "rank": stmt.get("rank", "normal"),
            }
        )

    return values


def _item_ids(entity: dict[str, Any], prop: str) -> list[str]:
    ids = []

    for value in _claim_values(entity, prop):
        raw_value = value["value"]
        if isinstance(raw_value, dict) and raw_value.get("id"):
            ids.append(raw_value["id"])

    return ids


def _string_values(entity: dict[str, Any], prop: str) -> list[str]:
    values = []

    for value in _claim_values(entity, prop):
        raw_value = value["value"]
        if isinstance(raw_value, str):
            values.append(raw_value)

    return values


def _coordinate_value(entity: dict[str, Any]) -> tuple[float, float] | None:
    values = _claim_values(entity, "P625")
    if not values:
        return None

    preferred = [value for value in values if value.get("rank") == "preferred"]
    chosen = preferred[0] if preferred else values[0]
    raw_value = chosen["value"]

    if not isinstance(raw_value, dict):
        return None

    lat = raw_value.get("latitude")
    lon = raw_value.get("longitude")

    if lat is None or lon is None:
        return None

    return float(lat), float(lon)


def _classify_location(entity: dict[str, Any]) -> str | None:
    kinds = {
        LOCATION_CLASS_QIDS[qid]
        for qid in _item_ids(entity, "P31")
        if qid in LOCATION_CLASS_QIDS
    }

    if not kinds:
        return None

    for kind in LOCATION_KIND_PRIORITY:
        if kind in kinds:
            return kind

    return sorted(kinds)[0]


@lru_cache(maxsize=8192)
def _normalize_wd_lang(wd_code: str) -> dict[str, str | None]:
    code = wd_code.strip().lower()

    if code in HARDCODED_LANG_NORMS:
        return HARDCODED_LANG_NORMS[code]

    if code in SPECIAL_GEO_LANG_OVERRIDES:
        return {
            "wd_lang": code,
            "geo_lang": SPECIAL_GEO_LANG_OVERRIDES[code],
            "iso639_3": None,
            "iso639_1": None,
        }

    if code in WIKIDATA_LANG_OVERRIDES:
        canonical_input = WIKIDATA_LANG_OVERRIDES[code]
    else:
        base, sep, rest = code.partition("-")
        canonical_input = WIKIDATA_LANG_OVERRIDES.get(base, base)
        if sep and base in WIKIDATA_LANG_OVERRIDES:
            canonical_input += sep + rest
        elif sep:
            canonical_input = code

    try:
        lang = langcodes.Language.get(canonical_input)

        try:
            iso3 = lang.to_alpha3()
        except LookupError:
            iso3 = None

        primary = lang.language
        iso1 = primary if primary and len(primary) == 2 else None

        return {
            "wd_lang": code,
            "geo_lang": iso1 or iso3,
            "iso639_3": iso3,
            "iso639_1": iso1,
        }
    except Exception:
        return {
            "wd_lang": code,
            "geo_lang": None,
            "iso639_3": None,
            "iso639_1": None,
        }


def _normalize_name(name: str) -> str:
    return unicodedata.normalize("NFC", " ".join(name.strip().split())).casefold()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []

    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)

    return out


def database_is_ready(db_path: Path) -> bool:
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
