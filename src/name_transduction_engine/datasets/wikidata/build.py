import bz2
import gzip
import unicodedata

import langcodes
import orjson

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
from name_transduction_engine.paths import (
    WIKIDATA_LOCATIONS_BUILD_PATH,
    WIKIDATA_RAW_DUMP_PATH,
)

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


def build_wikidata_compact_dataset() -> None:
    if not WIKIDATA_RAW_DUMP_PATH.exists():
        raise FileNotFoundError(
            f"Raw Wikidata dump not found: {WIKIDATA_RAW_DUMP_PATH}. "
            "Run `nte data fetch wikidata-raw` first."
        )

    _build_locations_dataset(WIKIDATA_RAW_DUMP_PATH, WIKIDATA_LOCATIONS_BUILD_PATH)


def _dataset_part_paths(output_path: Path) -> tuple[Path, Path]:
    jsonl_path = output_path.with_suffix("")
    jsonl_part_path = jsonl_path.with_suffix(jsonl_path.suffix + ".part")
    gzip_part_path = output_path.with_suffix(output_path.suffix + ".part")
    return jsonl_part_path, gzip_part_path


def _build_locations_dataset(raw_dump_path: Path, output_path: Path) -> None:
    print("Building compact Wikidata locations dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        lang_norm = normalize_wd_lang(wd_lang)

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
def normalize_wd_lang(wd_code: str) -> dict[str, str | None]:
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
