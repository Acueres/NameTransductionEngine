"""Data for the MCN quality benchmark (mcn_benchmark_normalization.ipynb).

Collects, for every More Cultural Names place linked to Wikidata or GeoNames:

- MCN's name in each language, with the language resolved to NTE's registry;
- NTE's names for the same place in that language (GeoNames and Wikidata,
  following the GeoNames <-> Wikidata links both ways), in NTE's lookup order;
- the place's reference names (GeoNames primary name, English names), which
  NTE's lookup passes to the display romanizer as reading hints;
- how MCN's name resolves through NTE's lookup key (`normalize_name`) over the
  whole database: whether the place is found, and how many other entities
  (GeoNames and Wikidata counted separately) share the key.

Reading names.sqlite takes a few minutes, so the result is cached:

    python analysis/mcn_quality_data.py      # from the project root

writes data/mcn_eval/mcn_quality.json.gz, which the notebook loads.
"""

import gzip
import json
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from name_transduction_engine.datasets.language_codes.data_provision import (  # noqa: E402
    read_registry,
)
from name_transduction_engine.normalization.language_code_normalization import (  # noqa: E402
    UnknownLanguageError,
    resolve_user_language,
)
from name_transduction_engine.normalization.name_normalization import (  # noqa: E402
    normalize_name,
)

MCN_DIR = PROJECT_ROOT / "data" / "repos" / "more-cultural-names"
DB_PATH = PROJECT_ROOT / "data" / "names.sqlite"
CACHE_PATH = PROJECT_ROOT / "data" / "mcn_eval" / "mcn_quality.json.gz"
FORMAT = 2

# MCN language ids for historical stages: their names follow older spellings,
# which NTE's (modern) data does not aim at
HISTORIC = re.compile(
    r"_(Before\d+|Old|Middle|Ancient|Classical|Archaic|Medieval|Early|Late)(_|$)"
    r"|^Proto|_Proto"
)


def parse_languages() -> dict[str, dict]:
    """MCN language id -> its ISO codes and whether it is a historical stage"""
    out = {}
    for lang in ET.parse(MCN_DIR / "languages.xml").getroot().findall("Language"):
        mcn_id = (lang.findtext("Id") or "").strip()
        if not mcn_id:
            continue
        codes = {}
        for node in lang.findall("Code"):
            for key in ("iso-639-1", "iso-639-3", "iso-639-2"):
                if node.attrib.get(key):
                    codes.setdefault(key, node.attrib[key])
        out[mcn_id] = {"codes": codes, "historic": bool(HISTORIC.search(mcn_id))}
    return out


def parse_locations() -> list[dict]:
    """MCN places with a Wikidata or GeoNames id and at least one name"""
    out = []
    for loc in ET.parse(MCN_DIR / "locations.xml").getroot().iter("LocationEntity"):
        qid = (loc.findtext("WikidataId") or "").strip() or None
        gid_text = (loc.findtext("GeoNamesId") or "").strip()
        gid = int(gid_text) if gid_text.isdigit() else None
        if qid is None and gid is None:
            continue
        names = {
            n.attrib["language"]: n.attrib["value"]
            for n in loc.findall("Names/Name")
            if n.attrib.get("value", "").strip()
        }
        if names:
            out.append(
                {
                    "mcn_id": (loc.findtext("Id") or "").strip(),
                    "qid": qid,
                    "gid": gid,
                    "names": names,
                }
            )
    return out


def resolve_codes(languages: dict[str, dict], registry) -> dict[str, str | None]:
    """MCN language id -> NTE registry language (None when unresolvable)"""
    out = {}
    for mcn_id, info in languages.items():
        code = None
        for key in ("iso-639-1", "iso-639-3", "iso-639-2"):
            raw = info["codes"].get(key)
            if not raw:
                continue
            try:
                code = resolve_user_language(raw, registry).tag.lang
                break
            except UnknownLanguageError:
                continue
        out[mcn_id] = code
    return out


def _linked_entities(conn, qid: str | None, gid: int | None):
    qids = {qid} if qid else set()
    gids = {gid} if gid else set()
    if qid:
        gids |= {
            r[0]
            for r in conn.execute(
                "SELECT geonames_id FROM wikidata_location_geonames WHERE qid = ?", (qid,)
            )
        }
    if gid:
        qids |= {
            r[0]
            for r in conn.execute(
                "SELECT qid FROM wikidata_location_geonames WHERE geonames_id = ?", (gid,)
            )
        }
    # Keep only entities NTE actually has
    gids = {
        g
        for g in gids
        if conn.execute("SELECT 1 FROM geoname WHERE geonameid = ?", (g,)).fetchone()
    }
    qids = {
        q
        for q in qids
        if conn.execute("SELECT 1 FROM wikidata_location WHERE qid = ?", (q,)).fetchone()
    }
    return sorted(gids), sorted(qids)


def _names(conn, gids, qids, codes) -> list[dict]:
    """NTE's names in the given languages, in lookup order: GeoNames first,
    entity id ascending, then record order"""
    if not codes:
        return []
    marks = ",".join("?" for _ in codes)
    out = []
    for g in gids:
        for name, lang, script, region, variant in conn.execute(
            f"SELECT alternate_name, lang, lang_script, lang_region, lang_variant "
            f"FROM alternate_name WHERE geonameid = ? AND lang IN ({marks}) "
            f"ORDER BY alternate_name_id",
            (g, *codes),
        ):
            out.append(_name("geonames", str(g), name, lang, script, region, variant))
    for q in qids:
        for name, lang, script, region, variant in conn.execute(
            f"SELECT name, lang, lang_script, lang_region, lang_variant "
            f"FROM wikidata_location_name WHERE qid = ? AND lang IN ({marks}) "
            f"ORDER BY term_type <> 'label', rowid",
            (q, *codes),
        ):
            out.append(_name("wikidata", q, name, lang, script, region, variant))
    return out


def _name(source, entity, name, lang, script, region, variant) -> dict:
    tag = "-".join(p for p in (lang, script, region, variant) if p)
    return {"source": source, "entity": entity, "name": name, "lang": lang, "tag": tag}


def _reference_names(conn, gids, qids) -> list[str]:
    """What NTE's lookup passes as hints: GeoNames primary name and English
    names, Wikidata English label and aliases"""
    out: list[str] = []
    for g in gids:
        row = conn.execute("SELECT name FROM geoname WHERE geonameid = ?", (g,)).fetchone()
        if row:
            out.append(row[0])
        out += [
            r[0]
            for r in conn.execute(
                "SELECT alternate_name FROM alternate_name WHERE geonameid = ? "
                "AND lang = 'en' ORDER BY alternate_name_id LIMIT 5",
                (g,),
            )
        ]
    for q in qids:
        out += [
            r[0]
            for r in conn.execute(
                "SELECT name FROM wikidata_location_name WHERE qid = ? AND lang = 'en' "
                "ORDER BY term_type <> 'label', name LIMIT 5",
                (q,),
            )
        ]
    return list(dict.fromkeys(out))


def _resolve(conn, name: str) -> tuple[set[int], set[str]]:
    """Entities whose names normalize to the same key as `name` (NTE's
    resolve step), over the whole database"""
    key = normalize_name(name)
    if not key:
        return set(), set()
    gids = {
        r[0]
        for r in conn.execute(
            "SELECT geonameid FROM geoname WHERE normalized_name = ? "
            "UNION SELECT geonameid FROM alternate_name WHERE normalized_name = ?",
            (key, key),
        )
    }
    qids = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT qid FROM wikidata_location_name WHERE normalized_name = ?",
            (key,),
        )
    }
    return gids, qids


def build() -> dict:
    t0 = time.time()
    languages = parse_languages()
    locations = parse_locations()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        registry = read_registry(conn)
        codes = resolve_codes(languages, registry)
        places = []
        resolved_cache: dict[str, tuple[set[int], set[str]]] = {}
        for k, loc in enumerate(locations):
            gids, qids = _linked_entities(conn, loc["qid"], loc["gid"])
            wanted = sorted({codes[m] for m in loc["names"] if codes.get(m)})
            rows = []
            for mcn_lang, mcn_name in loc["names"].items():
                if mcn_name not in resolved_cache:
                    resolved_cache[mcn_name] = _resolve(conn, mcn_name)
                res_gids, res_qids = resolved_cache[mcn_name]
                rows.append(
                    {
                        "mcn_lang": mcn_lang,
                        "code": codes.get(mcn_lang),
                        "historic": languages.get(mcn_lang, {}).get("historic", False),
                        "mcn_name": mcn_name,
                        # Does the lookup key find this place, and what else?
                        "found": bool(set(gids) & res_gids or set(qids) & res_qids),
                        "n_other": len(res_gids - set(gids)) + len(res_qids - set(qids)),
                    }
                )
            places.append(
                {
                    "mcn_id": loc["mcn_id"],
                    "qid": loc["qid"],
                    "gid": loc["gid"],
                    "nte_gids": gids,
                    "nte_qids": qids,
                    "reference_names": _reference_names(conn, gids, qids),
                    "nte_names": _names(conn, gids, qids, wanted),
                    "rows": rows,
                }
            )
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1:,}/{len(locations):,} places ({time.time() - t0:.0f}s)")
    finally:
        conn.close()
    return {
        "format": FORMAT,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "languages": {
            m: {"code": codes.get(m), "historic": info["historic"]}
            for m, info in languages.items()
        },
        "places": places,
    }


def load(rebuild: bool = False) -> dict:
    """The cached benchmark data, built first if missing or `rebuild`"""
    if not rebuild and CACHE_PATH.exists():
        with gzip.open(CACHE_PATH, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") == FORMAT:
            return data
    data = build()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE_PATH, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


if __name__ == "__main__":
    started = time.time()
    data = load(rebuild=True)
    rows = sum(len(p["rows"]) for p in data["places"])
    print(
        f"{CACHE_PATH}: {len(data['places']):,} places, {rows:,} MCN names "
        f"({time.time() - started:.0f}s)"
    )
