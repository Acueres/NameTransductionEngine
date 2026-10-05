"""What the Wikidata build keeps: groups, kinds, their root classes, and gates.

An entity is classified by its `P31` (instance of) values. Each kind lists
root classes; `nte data build wikidata-classes` expands every root to all of
its subclasses (`P279*`) with one SPARQL query per root and writes the result
to `data/raw/wikidata/wikidata_classes.tsv`. The build reads only that file,
so a run never depends on a SPARQL endpoint. Its hash is recorded in the
build state and the compact dataset's manifest.

Kinds are listed in priority order. A class reachable from several roots gets
the first kind that reaches it (an "ancient city" is also a "city"; the
historical kind comes first, so it wins). An entity with several P31 classes
gets the highest-priority kind among them.

Two gate levels:
- the extraction gate (`extract_gate`) runs while reading the dump. It is
  deliberately lenient, because changing it means re-reading the dump;
- the final gate (`final_gate`) runs when the compact dataset is written from
  the build shards. It is the size lever: change it, then run
  `nte data build wikidata-compact --refinalize` (minutes, no dump access).
"""

import difflib
import hashlib
import re
import sys
import time

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable

from name_transduction_engine.paths import WIKIDATA_CLASSES_PATH

CLASSES_PATH: Path = WIKIDATA_CLASSES_PATH
CLASSES_FORMAT: Final[str] = "nte-wikidata-classes 1"

DEFAULT_SPARQL_ENDPOINT: Final[str] = "https://qlever.dev/api/wikidata"
WDQS_ENDPOINT: Final[str] = "https://query.wikidata.org/sparql"

# Groups

PLACE: Final = "place"
HISTORICAL_PLACE: Final = "historical_place"
PERSON_NAME: Final = "person_name"
DYNASTY: Final = "dynasty"
ETHNONYM: Final = "ethnonym"
LANGUAGE: Final = "language"
TITLE: Final = "title"
PERSON: Final = "person"

GROUPS: Final[tuple[str, ...]] = (
    PLACE,
    HISTORICAL_PLACE,
    PERSON_NAME,
    DYNASTY,
    ETHNONYM,
    LANGUAGE,
    TITLE,
    PERSON,
)

# Groups loaded into names.sqlite. The others are kept in the compact dataset
# for later use
LOADED_GROUPS: Final[tuple[str, ...]] = (PLACE, HISTORICAL_PLACE)

# Groups whose entities must be on Earth and locatable
GEO_GROUPS: Final[frozenset[str]] = frozenset({PLACE, HISTORICAL_PLACE})

# Properties

P_INSTANCE_OF: Final = "P31"
P_SUBCLASS_OF: Final = "P279"
P_COORDINATES: Final = "P625"
P_POPULATION: Final = "P1082"
P_GEONAMES: Final = "P1566"

P_INCEPTION: Final = "P571"
P_DISSOLVED: Final = "P576"
P_START: Final = "P580"
P_END: Final = "P582"
P_POINT_IN_TIME: Final = "P585"
P_BIRTH: Final = "P569"
P_DEATH: Final = "P570"

# Name statements (monolingual text): property -> name type
NAME_PROPERTIES: Final[dict[str, str]] = {
    "P1448": "official",  # official name
    "P1705": "native",  # native label
    "P1813": "short",  # short name
    "P2561": "name",  # name
    "P1449": "nickname",  # nickname
    "P1477": "birth",  # birth name (people)
    "P1559": "native",  # name in native language (people)
}

# Links to other items: property -> relation name
LINK_PROPERTIES: Final[dict[str, str]] = {
    "P1365": "replaces",
    "P1366": "replaced_by",
    "P155": "follows",
    "P156": "followed_by",
    "P17": "country",
    "P131": "located_in",
    "P706": "located_on",  # located in/on physical feature
    "P361": "part_of",
    "P36": "capital",
    "P1376": "capital_of",
    "P460": "same_as",  # said to be the same as
    # People and names
    "P735": "given_name",
    "P734": "family_name",
    "P53": "family",
    "P97": "noble_title",
    "P27": "citizenship",
    "P282": "writing_system",
    "P407": "language",
}
MAX_LINKS_PER_RELATION: Final = 30

# External identifiers kept: property -> source name
ID_PROPERTIES: Final[dict[str, str]] = {
    P_GEONAMES: "geonames",
    "P1584": "pleiades",
    "P1936": "dare",  # Digital Atlas of the Roman Empire
    "P1958": "trismegistos",  # Trismegistos Geo
    "P8068": "topostext",
    "P1667": "tgn",  # Getty Thesaurus of Geographic Names
}

# Classical gazetteers. An item with one of these IDs and coordinates is kept
# as a historical place even if none of its classes is known
GAZETTEER_SOURCES: Final[frozenset[str]] = frozenset(
    {"pleiades", "dare", "trismegistos", "topostext"}
)

# Properties any of which makes a place locatable when it has no coordinates
GEO_SIGNAL_PROPERTIES: Final[tuple[str, ...]] = ("P17", "P131", "P706", "P30")

# Expected English labels, checked by `nte data build wikidata-classes`
PROPERTY_LABELS: Final[dict[str, str]] = {
    "P31": "instance of",
    "P279": "subclass of",
    "P625": "coordinate location",
    "P1082": "population",
    "P1566": "GeoNames ID",
    "P571": "inception",
    "P576": "dissolved, abolished or demolished date",
    "P580": "start time",
    "P582": "end time",
    "P585": "point in time",
    "P569": "date of birth",
    "P570": "date of death",
    "P1448": "official name",
    "P1705": "native label",
    "P1813": "short name",
    "P2561": "name",
    "P1449": "nickname",
    "P1477": "birth name",
    "P1559": "name in native language",
    "P1365": "replaces",
    "P1366": "replaced by",
    "P155": "follows",
    "P156": "followed by",
    "P17": "country",
    "P131": "located in the administrative territorial entity",
    "P706": "located in/on physical feature",
    "P361": "part of",
    "P36": "capital",
    "P1376": "capital of",
    "P460": "said to be the same as",
    "P735": "given name",
    "P734": "family name",
    "P53": "family",
    "P97": "noble title",
    "P27": "country of citizenship",
    "P282": "writing system",
    "P407": "language of work or name",
    "P1584": "Pleiades ID",
    "P1936": "Digital Atlas of the Roman Empire ID",
    "P1958": "Trismegistos Geo ID",
    "P8068": "ToposText place ID",
    "P1667": "Getty Thesaurus of Geographic Names ID",
    "P30": "continent",
}

# Gates


@dataclass(frozen=True)
class Condition:
    """All thresholds must hold. `names` counts distinct normalized name
    strings across all languages and name types"""

    sitelinks: int = 0
    names: int = 0
    population: int = 0
    gazetteer: bool = False  # has a classical gazetteer ID
    lineage: bool = False  # people: has a family/dynasty or a noble title
    born_before: int | None = None  # people: birth year strictly below

    def holds(self, facts: "GateFacts") -> bool:
        if facts.sitelinks < self.sitelinks or facts.names < self.names:
            return False
        if self.population and (facts.population or 0) < self.population:
            return False
        if self.gazetteer and not facts.gazetteer:
            return False
        if self.lineage and not facts.lineage:
            return False
        if self.born_before is not None and (
            facts.birth_year is None or facts.birth_year >= self.born_before
        ):
            return False
        return True


@dataclass(frozen=True)
class Gate:
    """Keep if `always`, or if any condition holds"""

    always: bool = False
    any_of: tuple[Condition, ...] = ()

    def passes(self, facts: "GateFacts") -> bool:
        return self.always or any(c.holds(facts) for c in self.any_of)

    def describe(self) -> str:
        if self.always:
            return "always"
        parts = []
        for c in self.any_of:
            bits = []
            if c.sitelinks:
                bits.append(f"sitelinks>={c.sitelinks}")
            if c.names:
                bits.append(f"names>={c.names}")
            if c.population:
                bits.append(f"population>={c.population:,}")
            if c.gazetteer:
                bits.append("gazetteer id")
            if c.lineage:
                bits.append("family or title")
            if c.born_before is not None:
                bits.append(f"born<{c.born_before}")
            parts.append(" & ".join(bits) or "true")
        return " | ".join(parts) or "never"


@dataclass(frozen=True)
class GateFacts:
    sitelinks: int
    names: int
    population: int | None = None
    gazetteer: bool = False
    lineage: bool = False
    birth_year: int | None = None


ALWAYS: Final = Gate(always=True)


def _any(*conditions: Condition) -> Gate:
    return Gate(any_of=conditions)


# Lenient extraction gates (reading the dump). Kept wide so that the final
# gates can be tuned later without re-reading the dump
EXTRACT_PLACE_GATE: Final = _any(
    Condition(names=2), Condition(sitelinks=1), Condition(population=1000)
)
EXTRACT_PERSON_GATE: Final = _any(
    Condition(lineage=True), Condition(sitelinks=10, born_before=1900)
)

# Reused final gates
_MAJOR_NATURAL: Final = _any(Condition(sitelinks=1, names=2), Condition(names=3))
_MINOR_NATURAL: Final = _any(Condition(sitelinks=3, names=3))

# Kinds


@dataclass(frozen=True)
class KindSpec:
    group: str
    kind: str
    roots: tuple[tuple[str, str], ...]  # (QID, expected English label)
    final_gate: Gate = ALWAYS
    extract_gate: Gate = ALWAYS
    subclasses: bool = True  # expand roots with P279*


def _k(
    group: str,
    kind: str,
    roots: Iterable[tuple[str, str]],
    final_gate: Gate = ALWAYS,
    extract_gate: Gate | None = None,
    subclasses: bool = True,
) -> KindSpec:
    if extract_gate is None:
        extract_gate = EXTRACT_PLACE_GATE if group == PLACE else ALWAYS
    return KindSpec(group, kind, tuple(roots), final_gate, extract_gate, subclasses)


# Priority order: historical before current, specific before generic
KIND_SPECS: Final[tuple[KindSpec, ...]] = (
    # Historical places
    _k(HISTORICAL_PLACE, "historical_country", [("Q3024240", "historical country")]),
    _k(
        HISTORICAL_PLACE,
        "historical_admin",
        [
            ("Q19953632", "former administrative territorial entity"),
            ("Q182547", "Roman province"),
        ],
        final_gate=_any(Condition(sitelinks=1, names=2), Condition(names=4)),
    ),
    _k(HISTORICAL_PLACE, "historical_region", [("Q1620908", "historical region")]),
    _k(HISTORICAL_PLACE, "ancient_city", [("Q15661340", "ancient city")]),
    _k(
        HISTORICAL_PLACE,
        "abandoned_settlement",
        [("Q350895", "abandoned village"), ("Q74047", "ghost town")],
        final_gate=_any(Condition(sitelinks=1, names=2), Condition(gazetteer=True)),
    ),
    _k(
        HISTORICAL_PLACE,
        "archaeological_site",
        [("Q839954", "archaeological site")],
        final_gate=_any(Condition(sitelinks=2), Condition(gazetteer=True)),
        extract_gate=_any(Condition(sitelinks=1), Condition(gazetteer=True)),
    ),
    # Current places: political
    _k(
        PLACE,
        "country",
        [
            ("Q6256", "country"),
            ("Q3624078", "sovereign state"),
            ("Q3336843", "constituent country"),
        ],
    ),
    # Settlements before admin levels: a commune or a city-state is both,
    # call it by what it is on the ground
    _k(PLACE, "city", [("Q515", "city")]),
    _k(
        PLACE,
        "admin1",
        [("Q10864048", "first-level administrative division")],
    ),
    _k(
        PLACE,
        "town",
        [("Q3957", "town")],
        final_gate=_any(Condition(sitelinks=1), Condition(names=3)),
    ),
    _k(
        PLACE,
        "village",
        [("Q532", "village")],
        final_gate=_any(Condition(sitelinks=2, names=3), Condition(population=5000)),
    ),
    _k(
        PLACE,
        "hamlet",
        [("Q5084", "hamlet")],
        final_gate=_any(Condition(sitelinks=3, names=3)),
    ),
    _k(
        PLACE,
        "admin2",
        [("Q13220204", "second-level administrative division")],
        final_gate=_any(Condition(sitelinks=1), Condition(names=3)),
    ),
    _k(
        PLACE,
        "admin3",
        [("Q13221722", "third-level administrative division")],
        final_gate=_any(Condition(sitelinks=2, names=2), Condition(names=4)),
    ),
    # Current places: natural features
    _k(PLACE, "continent", [("Q5107", "continent")], final_gate=ALWAYS),
    _k(PLACE, "ocean", [("Q9430", "ocean")], final_gate=ALWAYS),
    _k(PLACE, "sea", [("Q165", "sea")], final_gate=_MAJOR_NATURAL),
    _k(PLACE, "gulf", [("Q1322134", "gulf")], final_gate=_MAJOR_NATURAL),
    _k(
        PLACE,
        "bay",
        [("Q39594", "bay")],
        final_gate=_any(Condition(sitelinks=2), Condition(names=3)),
    ),
    _k(PLACE, "strait", [("Q37901", "strait")], final_gate=_MAJOR_NATURAL),
    _k(PLACE, "archipelago", [("Q33837", "archipelago")], final_gate=_MAJOR_NATURAL),
    _k(PLACE, "peninsula", [("Q34763", "peninsula")], final_gate=_MAJOR_NATURAL),
    _k(
        PLACE,
        "mountain_range",
        [("Q46831", "mountain range")],
        final_gate=_MAJOR_NATURAL,
    ),
    _k(PLACE, "desert", [("Q8514", "desert")], final_gate=_MAJOR_NATURAL),
    _k(PLACE, "volcano", [("Q8072", "volcano")], final_gate=_MINOR_NATURAL),
    _k(PLACE, "island", [("Q23442", "island")], final_gate=_MINOR_NATURAL),
    _k(PLACE, "river", [("Q4022", "river")], final_gate=_MINOR_NATURAL),
    _k(PLACE, "lake", [("Q23397", "lake")], final_gate=_MINOR_NATURAL),
    _k(PLACE, "mountain", [("Q8502", "mountain")], final_gate=_MINOR_NATURAL),
    _k(PLACE, "cape", [("Q185113", "cape")], final_gate=_MINOR_NATURAL),
    _k(PLACE, "plateau", [("Q75520", "plateau")], final_gate=_MINOR_NATURAL),
    _k(PLACE, "plain", [("Q160091", "plain")], final_gate=_MINOR_NATURAL),
    _k(PLACE, "valley", [("Q39816", "valley")], final_gate=_MINOR_NATURAL),
    # Generic catch-alls last: their subclass trees are huge and reach classes
    # that have a kind of their own (in Wikidata, "continent" is a subclass of
    # "administrative territorial entity")
    _k(
        PLACE,
        "settlement",
        [("Q486972", "human settlement")],
        final_gate=_any(Condition(sitelinks=2, names=3), Condition(population=5000)),
    ),
    _k(
        PLACE,
        "admin",
        [("Q56061", "administrative territorial entity")],
        final_gate=_any(Condition(sitelinks=3, names=3)),
    ),
    _k(
        PLACE,
        "region",
        [("Q82794", "geographic region")],
        final_gate=_any(Condition(sitelinks=2, names=2)),
    ),
    # Kept for later, not loaded into names.sqlite yet
    _k(DYNASTY, "dynasty", [("Q171541", "dynasty"), ("Q13417114", "noble family")]),
    _k(PERSON_NAME, "given_name", [("Q202444", "given name")]),
    _k(PERSON_NAME, "family_name", [("Q101352", "family name")]),
    _k(ETHNONYM, "ethnic_group", [("Q41710", "ethnic group")]),
    _k(TITLE, "noble_title", [("Q355567", "noble title")]),
    _k(
        LANGUAGE,
        "language",
        [("Q34770", "language")],
        final_gate=_any(Condition(sitelinks=1)),
    ),
    _k(
        PERSON,
        "person",
        [("Q5", "human")],
        final_gate=EXTRACT_PERSON_GATE,
        extract_gate=EXTRACT_PERSON_GATE,
        subclasses=False,
    ),
)

# Kind used for items kept only because they carry a classical gazetteer ID
GAZETTEER_KIND: Final = KindSpec(
    HISTORICAL_PLACE, "ancient_place", (), final_gate=ALWAYS, extract_gate=ALWAYS
)

# Subtrees removed from every kind: too fine-grained, or not real places.
# Kept narrow on purpose; broad classes such as "organization" reach
# settlement classes through Wikidata's modelling of local government
EXCLUDED_ROOTS: Final[tuple[tuple[str, str], ...]] = (
    ("Q79007", "street"),
    ("Q34442", "road"),
    ("Q41176", "building"),
    ("Q55488", "railway station"),
    ("Q1248784", "airport"),
    ("Q123705", "neighborhood"),
    ("Q22698", "park"),
    ("Q39614", "cemetery"),
    ("Q3914", "school"),
    ("Q16970", "church building"),
    ("Q192611", "electoral unit"),
    ("Q14897293", "fictional entity"),
)

EARTH_QID: Final = "Q2"

SPEC_BY_KIND: Final[dict[str, KindSpec]] = {
    spec.kind: spec for spec in (*KIND_SPECS, GAZETTEER_KIND)
}


def rules_fingerprint() -> str:
    """Changes when the class rules change (roots, kinds, priority, exclusions),
    which means the class file must be regenerated"""
    h = hashlib.sha256()
    for spec in KIND_SPECS:
        h.update(f"{spec.group}|{spec.kind}|{spec.subclasses}|".encode())
        h.update(",".join(q for q, _ in spec.roots).encode())
        h.update(b"\n")
    h.update(b"exclude:" + ",".join(q for q, _ in EXCLUDED_ROOTS).encode())
    return h.hexdigest()[:16]


def final_gates_fingerprint() -> str:
    h = hashlib.sha256()
    for spec in (*KIND_SPECS, GAZETTEER_KIND):
        h.update(f"{spec.kind}={spec.final_gate!r}\n".encode())
    return h.hexdigest()[:16]


def extract_gates_fingerprint() -> str:
    h = hashlib.sha256()
    for spec in (*KIND_SPECS, GAZETTEER_KIND):
        h.update(f"{spec.kind}={spec.extract_gate!r}\n".encode())
    return h.hexdigest()[:16]


# Class file


@dataclass
class ClassMap:
    """class QID -> kind, as resolved by the class refresh"""

    kinds: dict[str, str]
    rules: str
    generated_at: str = ""
    endpoint: str = ""
    sha256: str = ""
    counts: dict[str, int] = field(default_factory=dict)

    def kind_of(self, class_qid: str) -> str | None:
        return self.kinds.get(class_qid)


class ClassFileError(RuntimeError):
    pass


def load_class_map(path: Path | None = None) -> ClassMap:
    path = path or CLASSES_PATH
    if not path.is_file():
        raise ClassFileError(
            f"Wikidata class file not found: {path}. "
            "Run `nte data build wikidata-classes` first (needs internet, "
            "takes a few minutes)."
        )

    raw = path.read_bytes()
    meta: dict[str, str] = {}
    kinds: dict[str, str] = {}
    known_kinds = {spec.kind for spec in KIND_SPECS}

    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            key, _, value = line[1:].strip().partition("\t")
            meta[key.strip()] = value.strip()
            continue
        if line.startswith("class_qid\t"):
            continue
        qid, kind = line.split("\t")[:2]
        if kind not in known_kinds:
            raise ClassFileError(
                f"{path.name}: unknown kind {kind!r}; regenerate it with "
                "`nte data build wikidata-classes`"
            )
        kinds[qid] = kind

    if meta.get("format") != CLASSES_FORMAT:
        raise ClassFileError(f"{path.name}: unsupported format {meta.get('format')!r}")

    rules = meta.get("rules", "")
    if rules != rules_fingerprint():
        raise ClassFileError(
            f"{path.name} was generated for different class rules "
            f"({rules or 'unknown'}, current {rules_fingerprint()}). "
            "Run `nte data build wikidata-classes` to regenerate it."
        )

    counts: dict[str, int] = {}
    for kind in kinds.values():
        counts[kind] = counts.get(kind, 0) + 1

    return ClassMap(
        kinds=kinds,
        rules=rules,
        generated_at=meta.get("generated_at", ""),
        endpoint=meta.get("endpoint", ""),
        sha256=hashlib.sha256(raw).hexdigest(),
        counts=counts,
    )


def write_class_map(
    kinds: dict[str, str], endpoint: str, path: Path | None = None
) -> None:
    path = path or CLASSES_PATH
    order = {spec.kind: i for i, spec in enumerate(KIND_SPECS)}
    rows = sorted(kinds.items(), key=lambda kv: (order[kv[1]], int(kv[0][1:])))
    lines = [
        f"# format\t{CLASSES_FORMAT}",
        f"# rules\t{rules_fingerprint()}",
        f"# generated_at\t{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"# endpoint\t{endpoint}",
        "class_qid\tkind",
        *(f"{qid}\t{kind}" for qid, kind in rows),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


# Refresh (SPARQL)

_PREFIXES: Final = (
    "PREFIX wd: <http://www.wikidata.org/entity/>\n"
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX wikibase: <http://wikiba.se/ontology#>\n"
)
# Labels at least this similar to the expected one count as the same item
LABEL_SIMILARITY: Final = 0.6
_ENTITY_RE: Final = re.compile(r"/entity/([PQ]\d+)>?")


def _sparql_tsv(
    session, endpoint: str, query: str, timeout: int = 600
) -> list[list[str]]:
    """Rows of a SELECT query as lists of cells (header dropped)"""
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            response = session.post(
                endpoint,
                data={"query": _PREFIXES + query},
                headers={"Accept": "text/tab-separated-values"},
                timeout=(15, timeout),
            )
            response.raise_for_status()
            lines = response.content.decode("utf-8").splitlines()
            return [line.split("\t") for line in lines[1:] if line.strip()]
        except Exception as exc:  # network errors, 5xx, timeouts
            last_exc = exc
            wait = 10 * (attempt + 1)
            print(f"  query failed ({exc}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"SPARQL query failed at {endpoint}: {last_exc}")


def _entity_ids(cells: list[str]) -> list[str]:
    out = []
    for cell in cells:
        m = _ENTITY_RE.search(cell)
        if m:
            out.append(m.group(1))
    return out


def _check_labels(session, endpoint: str) -> set[str]:
    """Compare the English labels of every root, exclusion and property with
    what the rules expect. Returns the roots whose label is unrelated to the
    expected one; those are skipped so a wrong QID cannot pull in an unrelated
    subtree. A similar label (an item renamed on Wikidata) is accepted with a
    note"""
    expected: dict[str, str] = {}
    for spec in KIND_SPECS:
        expected.update(dict(spec.roots))
    expected.update(dict(EXCLUDED_ROOTS))
    expected.update(PROPERTY_LABELS)

    labels: dict[str, str] = {}
    ids = sorted(expected)
    for i in range(0, len(ids), 80):
        values = " ".join(f"wd:{x}" for x in ids[i : i + 80])
        rows = _sparql_tsv(
            session,
            endpoint,
            f"SELECT ?x ?l WHERE {{ VALUES ?x {{ {values} }} "
            f'?x rdfs:label ?l FILTER(LANG(?l) = "en") }}',
        )
        for row in rows:
            got = _entity_ids(row[:1])
            if got and len(row) > 1:
                labels[got[0]] = row[1].strip().strip('"').rsplit('"@', 1)[0]

    bad: set[str] = set()
    for qid, want in sorted(expected.items()):
        got = labels.get(qid)
        if got is not None and got.casefold() == want.casefold():
            continue
        similarity = (
            difflib.SequenceMatcher(None, got.casefold(), want.casefold()).ratio()
            if got is not None
            else 0.0
        )
        if similarity >= LABEL_SIMILARITY:
            print(f"  label check: {qid} is now {got!r} (was {want!r}); using it")
            continue
        print(f"  label check: {qid} is {got!r}, expected {want!r}; skipped")
        if qid.startswith("Q"):
            bad.add(qid)
    if not bad:
        print(f"  label check: all {len(expected)} IDs are usable")
    return bad


def _closure(session, endpoint: str, root: str, subclasses: bool) -> set[str]:
    if not subclasses:
        return {root}
    rows = _sparql_tsv(
        session, endpoint, f"SELECT DISTINCT ?c WHERE {{ ?c wdt:P279* wd:{root} . }}"
    )
    found = set(_entity_ids([row[0] for row in rows]))
    found.add(root)
    return found


def refresh_class_map(
    endpoint: str = DEFAULT_SPARQL_ENDPOINT,
    path: Path | None = None,
    forecast: bool = False,
) -> None:
    path = path or CLASSES_PATH
    from name_transduction_engine.datasets.shared import build_session

    print(f"Refreshing Wikidata class file from {endpoint}")
    with build_session(use_env_proxy=True) as session:
        session.headers["User-Agent"] = USER_AGENT

        bad_roots = _check_labels(session, endpoint)
        if bad_roots:
            print(
                f"  skipping {len(bad_roots)} root(s) whose label does not match: "
                + ", ".join(sorted(bad_roots))
            )

        excluded: set[str] = set()
        for qid, label in EXCLUDED_ROOTS:
            if qid in bad_roots:
                continue
            sub = _closure(session, endpoint, qid, True)
            print(f"  exclude {qid} ({label}): {len(sub):,} classes")
            excluded |= sub

        kinds: dict[str, str] = {}
        for spec in KIND_SPECS:
            for qid, label in spec.roots:
                if qid in bad_roots:
                    continue
                if qid in excluded:
                    raise RuntimeError(
                        f"root {qid} ({label}) falls inside an excluded subtree; "
                        "fix EXCLUDED_ROOTS"
                    )
                if qid in kinds:
                    print(
                        f"  note: {qid} ({label}) is a subclass of an earlier "
                        f"kind's root and stays {kinds[qid]!r}"
                    )
                sub = _closure(session, endpoint, qid, spec.subclasses)
                new = 0
                for c in sub:
                    if c in excluded or c in kinds:
                        continue
                    kinds[c] = spec.kind
                    new += 1
                print(
                    f"  {spec.kind:<22} {qid} ({label}): {len(sub):,} classes, "
                    f"{new:,} assigned here"
                )

        write_class_map(kinds, endpoint, path)
        print(f"Wrote {len(kinds):,} classes to {path}")

        if forecast:
            _forecast(session, endpoint, bad_roots)


def _forecast(session, endpoint: str, bad_roots: set[str]) -> None:
    """Rough item counts per kind (before the name gates, with overlaps)"""
    print("Forecast (items per kind; overlapping, before name gates):")
    print(f"  {'kind':<22} {'items':>12} {'sitelinks>=2':>14}")
    for spec in KIND_SPECS:
        roots = [q for q, _ in spec.roots if q not in bad_roots]
        if not roots:
            continue
        values = " ".join(f"wd:{q}" for q in roots)
        path = "wdt:P31/wdt:P279*" if spec.subclasses else "wdt:P31"
        try:
            counts = []
            for extra in ("", "?i wikibase:sitelinks ?s . FILTER(?s >= 2)"):
                rows = _sparql_tsv(
                    session,
                    endpoint,
                    f"SELECT (COUNT(DISTINCT ?i) AS ?n) WHERE {{ "
                    f"VALUES ?r {{ {values} }} ?i {path} ?r . {extra} }}",
                    timeout=900,
                )
                match = re.match(r'\s*"?(\d+)', rows[0][0]) if rows else None
                counts.append(int(match.group(1)) if match else 0)
            print(f"  {spec.kind:<22} {counts[0]:>12,} {counts[1]:>14,}")
        except Exception as exc:
            print(f"  {spec.kind:<22} failed: {exc}")


USER_AGENT: Final = (
    "NameTransductionEngine/0.1 "
    "(+https://github.com/Acueres/NameTransductionEngine; dataset build)"
)
