"""One dump line -> one compact record (or nothing).

Most of the ~110M entities in the dump are irrelevant (scholarly articles,
genes, astronomical objects...). `Extractor.prefilter` rejects them from the
raw bytes with a few substring searches, without parsing JSON. It only needs
to be a superset: anything it lets through is parsed with orjson and
classified exactly by `Extractor.extract`.

Compact record (one JSON object per line; empty fields are omitted):

    qid        "Q16869"
    group      place | historical_place | person_name | dynasty | ...
    kind       city, historical_country, given_name, ... (see classes.py)
    classes    direct P31 values
    sitelinks  number of Wikipedia articles
    population latest population (places)
    coord      [lat, lon] (Earth only)
    start/end  years (negative = BCE): inception/dissolution for places,
               birth/death for people
    names      [{"text", "type", "langs", "start"?, "end"?}]; one entry per
               distinct (text, type, period), listing every language that
               uses it. type: label | alias | official | native | short |
               name | nickname | birth
    links      [{"rel", "qid", "start"?, "end"?}] (see LINK_PROPERTIES)
    ids        {"geonames": [...], "pleiades": [...], ...}
"""

import re

from collections import Counter
from typing import Any, Final

import orjson

from name_transduction_engine.normalization.name_normalization import normalize_name
from . import classes as C

# Bump when the records produced from the same dump would change. A build
# in progress with another version is restarted rather than mixed
EXTRACTOR_VERSION: Final[int] = 1

# `"P31"` appears as the claims key and in every P31 snak; the item value
# follows within a few hundred bytes. Whitespace-tolerant on purpose
_P31_MARK: Final = b'"P31"'
_ITEM_VALUE_RE: Final = re.compile(rb'"(?:numeric-id"\s*:\s*|id"\s*:\s*"Q)(\d+)')
_P31_WINDOW: Final = 400  # bytes after the property mark that hold its value
_P31_GAP: Final = 50_000  # max distance between consecutive P31 statements
_COORD_MARK: Final = b'"P625"'
_GAZETTEER_MARKS: Final = tuple(
    f'"{prop}"'.encode()
    for prop, source in C.ID_PROPERTIES.items()
    if source in C.GAZETTEER_SOURCES
)
_LINEAGE_MARKS: Final = (b'"P53"', b'"P97"')
_SITE_MARK: Final = b'"site"'

# Sitelink keys that end in "wiki" but are not Wikipedias
_NON_WIKIPEDIA: Final = frozenset(
    {
        "commonswiki",
        "specieswiki",
        "metawiki",
        "mediawikiwiki",
        "wikidatawiki",
        "sourceswiki",
        "wikimaniawiki",
        "outreachwiki",
        "foundationwiki",
        "incubatorwiki",
        "wikifunctionswiki",
        "testwiki",
        "test2wiki",
        "testwikidatawiki",
        "otrs_wikiwiki",
        "strategywiki",
    }
)

MAX_NAME_LENGTH: Final[int] = 150


class Extractor:
    def __init__(self, class_map: C.ClassMap) -> None:
        self.kinds = class_map.kinds
        self.specs = C.SPEC_BY_KIND
        self.priority = {spec.kind: i for i, spec in enumerate(C.KIND_SPECS)}
        self.person_kinds = frozenset(
            spec.kind for spec in C.KIND_SPECS if spec.group == C.PERSON
        )
        self.stats: Counter[str] = Counter()

    # Prefilter (raw bytes)

    def prefilter(self, line: bytes) -> bool:
        kinds = {self.kinds.get(q) for q in _p31_ids(line)}
        kinds.discard(None)
        if kinds:
            if kinds <= self.person_kinds:
                return any(m in line for m in _LINEAGE_MARKS) or (
                    line.count(_SITE_MARK) >= 10
                )
            return True
        # Classical gazetteer route: needs coordinates and a gazetteer ID
        return _COORD_MARK in line and any(m in line for m in _GAZETTEER_MARKS)

    # Extraction (parsed entity)

    def extract_line(self, line: bytes) -> dict[str, Any] | None:
        entity = orjson.loads(line)
        return self.extract(entity)

    def extract(self, entity: dict[str, Any]) -> dict[str, Any] | None:
        if entity.get("type", "item") != "item":
            self.stats["skip:not_item"] += 1
            return None
        claims = entity.get("claims") or {}

        classes = _dedupe(_item_values(claims, C.P_INSTANCE_OF))
        kind = self._best_kind(classes)
        ids = _external_ids(claims)
        has_gazetteer = any(source in C.GAZETTEER_SOURCES for source in ids)

        coord, offworld = _coordinates(claims)
        if kind is None:
            if has_gazetteer and coord is not None:
                kind = C.GAZETTEER_KIND.kind
            else:
                self.stats["skip:no_class"] += 1
                return None

        spec = self.specs[kind]
        group = spec.group

        start, end = _period(claims, group)
        if group == C.PLACE and _is_defunct(claims):
            group = C.HISTORICAL_PLACE

        if group in C.GEO_GROUPS:
            if offworld:
                self.stats["skip:offworld"] += 1
                return None
            if coord is None and not any(
                claims.get(p) for p in C.GEO_SIGNAL_PROPERTIES
            ):
                self.stats["skip:no_location"] += 1
                return None

        names = _names(entity, claims)
        if not names:
            self.stats["skip:no_names"] += 1
            return None

        record: dict[str, Any] = {
            "qid": entity["id"],
            "group": group,
            "kind": kind,
            "classes": classes,
            "sitelinks": _wikipedia_count(entity.get("sitelinks") or {}),
        }
        if group in C.GEO_GROUPS:
            population = _population(claims)
            if population is not None:
                record["population"] = population
        if coord is not None:
            record["coord"] = coord
        if start is not None:
            record["start"] = start
        if end is not None:
            record["end"] = end
        record["names"] = names
        links = _links(claims)
        if links:
            record["links"] = links
        if ids:
            record["ids"] = ids

        if not spec.extract_gate.passes(gate_facts(record)):
            self.stats[f"gate:{kind}"] += 1
            return None

        self.stats[f"kept:{group}"] += 1
        return record

    def _best_kind(self, classes: list[str]) -> str | None:
        best: str | None = None
        for q in classes:
            kind = self.kinds.get(q)
            if kind is not None and (
                best is None or self.priority[kind] < self.priority[best]
            ):
                best = kind
        return best


# Gates on records


def gate_facts(record: dict[str, Any]) -> C.GateFacts:
    names = {
        key
        for name in record.get("names", ())
        if (key := normalize_name(name["text"])) is not None
    }
    links = record.get("links", ())
    ids = record.get("ids", {})
    is_person = record.get("group") == C.PERSON
    return C.GateFacts(
        sitelinks=record.get("sitelinks", 0),
        names=len(names),
        population=record.get("population"),
        gazetteer=any(source in C.GAZETTEER_SOURCES for source in ids),
        lineage=any(link["rel"] in ("family", "noble_title") for link in links),
        birth_year=record.get("start") if is_person else None,
    )


def passes_final_gate(record: dict[str, Any]) -> bool:
    spec = C.SPEC_BY_KIND.get(record["kind"])
    if spec is None:
        return False
    return spec.final_gate.passes(gate_facts(record))


# Raw-bytes helpers


def _p31_ids(line: bytes) -> list[str]:
    """Item IDs found shortly after each `"P31"` in the raw line. A superset
    of the entity's P31 values (it also catches P31 used as a qualifier, and
    sees each value more than once)"""
    out: list[str] = []
    i = line.find(_P31_MARK)
    while i >= 0:
        match = _ITEM_VALUE_RE.search(line, i, i + _P31_WINDOW)
        if match:
            out.append("Q" + match.group(1).decode("ascii"))
        i = line.find(_P31_MARK, i + len(_P31_MARK), i + _P31_GAP)
    return out


# Entity helpers


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _statements(claims: dict, prop: str) -> list[dict]:
    """Non-deprecated statements with a value, preferred ones first"""
    out = []
    for stmt in claims.get(prop) or ():
        rank = stmt.get("rank", "normal")
        if rank == "deprecated":
            continue
        snak = stmt.get("mainsnak") or {}
        if snak.get("snaktype") != "value" or "datavalue" not in snak:
            continue
        out.append(stmt)
    out.sort(key=lambda s: s.get("rank") != "preferred")
    return out


def _value(stmt: dict) -> Any:
    return stmt["mainsnak"]["datavalue"].get("value")


def _item_id(value: Any) -> str | None:
    if isinstance(value, dict):
        qid = value.get("id")
        if qid:
            return qid
        numeric = value.get("numeric-id")
        if numeric is not None and value.get("entity-type", "item") == "item":
            return f"Q{numeric}"
    return None


def _item_values(claims: dict, prop: str) -> list[str]:
    out = []
    for stmt in _statements(claims, prop):
        qid = _item_id(_value(stmt))
        if qid:
            out.append(qid)
    return out


def _year(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    text = value.get("time")
    if not text or value.get("precision", 9) < 6:  # coarser than millennium
        return None
    sign = -1 if text[0] == "-" else 1
    digits = text.lstrip("+-").split("-", 1)[0]
    if not digits.isdigit():
        return None
    year = sign * int(digits)
    return year or None


def _years(claims: dict, prop: str) -> list[int]:
    out = []
    for stmt in _statements(claims, prop):
        year = _year(_value(stmt))
        if year is not None:
            out.append(year)
    return out


def _qualifier_year(stmt: dict, prop: str) -> int | None:
    for snak in (stmt.get("qualifiers") or {}).get(prop) or ():
        if snak.get("snaktype") == "value" and "datavalue" in snak:
            year = _year(snak["datavalue"].get("value"))
            if year is not None:
                return year
    return None


def _statement_period(stmt: dict) -> tuple[int | None, int | None]:
    start = _qualifier_year(stmt, C.P_START)
    end = _qualifier_year(stmt, C.P_END)
    if start is None and end is None:
        point = _qualifier_year(stmt, C.P_POINT_IN_TIME)
        if point is not None:
            return point, point
    return start, end


def _period(claims: dict, group: str) -> tuple[int | None, int | None]:
    if group == C.PERSON:
        births = _years(claims, C.P_BIRTH)
        deaths = _years(claims, C.P_DEATH)
        return (births[0] if births else None, deaths[0] if deaths else None)
    starts = _years(claims, C.P_INCEPTION) or _years(claims, C.P_START)
    ends = _years(claims, C.P_DISSOLVED) or _years(claims, C.P_END)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _is_defunct(claims: dict) -> bool:
    return bool(_statements(claims, C.P_DISSOLVED) or _statements(claims, C.P_END))


def _coordinates(claims: dict) -> tuple[list[float] | None, bool]:
    """([lat, lon] or None, whether the coordinates are not on Earth)"""
    for stmt in _statements(claims, C.P_COORDINATES):
        value = _value(stmt)
        if not isinstance(value, dict):
            continue
        globe = value.get("globe") or ""
        if globe and not globe.endswith("/" + C.EARTH_QID):
            return None, True
        lat, lon = value.get("latitude"), value.get("longitude")
        if lat is None or lon is None:
            continue
        return [round(float(lat), 5), round(float(lon), 5)], False
    return None, False


def _population(claims: dict) -> int | None:
    best: tuple[bool, int, int] | None = None  # (preferred, year, amount)
    for stmt in _statements(claims, C.P_POPULATION):
        value = _value(stmt)
        if not isinstance(value, dict):
            continue
        try:
            amount = int(float(value.get("amount", "")))
        except ValueError:
            continue
        year = _qualifier_year(stmt, C.P_POINT_IN_TIME) or -10_000
        key = (stmt.get("rank") == "preferred", year, amount)
        if best is None or key > best:
            best = key
    return best[2] if best else None


def _wikipedia_count(sitelinks: dict) -> int:
    return sum(
        1 for site in sitelinks if site.endswith("wiki") and site not in _NON_WIKIPEDIA
    )


def _clean_text(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text or len(text) > MAX_NAME_LENGTH:
        return None
    return text


def _is_junk_alias(text: str) -> bool:
    """Codes and abbreviations ("FR-75", "NYC") rather than names"""
    if any(ch.isdigit() for ch in text):
        return True
    return len(text) <= 5 and text.isascii() and text.isupper()


def _names(entity: dict, claims: dict) -> list[dict[str, Any]]:
    acc: dict[tuple, list[str]] = {}

    def add(text: Any, lang: Any, kind: str, start=None, end=None) -> None:
        text = _clean_text(text)
        if text is None or not isinstance(lang, str) or not lang:
            return
        langs = acc.setdefault((text, kind, start, end), [])
        if lang not in langs:
            langs.append(lang)

    for lang, obj in (entity.get("labels") or {}).items():
        add(obj.get("value"), obj.get("language", lang), "label")

    for lang, objs in (entity.get("aliases") or {}).items():
        for obj in objs:
            text = obj.get("value")
            if isinstance(text, str) and not _is_junk_alias(text.strip()):
                add(text, obj.get("language", lang), "alias")

    for prop, kind in C.NAME_PROPERTIES.items():
        for stmt in _statements(claims, prop):
            value = _value(stmt)
            if not isinstance(value, dict):
                continue
            start, end = _statement_period(stmt)
            add(value.get("text"), value.get("language"), kind, start, end)

    out = []
    for (text, kind, start, end), langs in acc.items():
        name: dict[str, Any] = {"text": text, "type": kind, "langs": langs}
        if start is not None:
            name["start"] = start
        if end is not None:
            name["end"] = end
        out.append(name)
    return out


def _links(claims: dict) -> list[dict[str, Any]]:
    out = []
    for prop, rel in C.LINK_PROPERTIES.items():
        seen = set()
        for stmt in _statements(claims, prop):
            qid = _item_id(_value(stmt))
            if qid is None:
                continue
            start, end = _statement_period(stmt)
            key = (qid, start, end)
            if key in seen:
                continue
            seen.add(key)
            link: dict[str, Any] = {"rel": rel, "qid": qid}
            if start is not None:
                link["start"] = start
            if end is not None:
                link["end"] = end
            out.append(link)
            if len(seen) >= C.MAX_LINKS_PER_RELATION:
                break
    return out


def _external_ids(claims: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for prop, source in C.ID_PROPERTIES.items():
        values = []
        for stmt in _statements(claims, prop):
            value = _value(stmt)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
        if values:
            out[source] = _dedupe(values)
    return out
