import csv
import sqlite3

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from name_transduction_engine.normalization.language_code_normalization import (
    ISO639_2B_TO_3,
    WITHDRAWN_ISO639_2B,
    RegistryError,
    LanguageRegistry,
    LanguageRecord,
    SourceTag,
    TAG_RULES_VERSION,
)
from name_transduction_engine.paths import RAW_DIR_LANGUAGE_CODES
from .download import IANA_REGISTRY_FILENAME, ISO_LANGUAGECODES_FILENAME

REQUIRED_TABLES = {
    "language",
    "language_alias",
    "language_retirement",
    "language_subtag",
    "build_metadata",
}

REGISTRY_FINGERPRINT_KEY = "language_registry_fingerprint"

# IANA scopes that never become registry records. `special` codes (mul, und,
# zxx, mis) are handled by _SPECIAL_STATUS; `collection` codes (sla, ber, ...)
# are language families, not languages, and have no ISO 639-3 identity
_EXCLUDED_IANA_SCOPES = frozenset({"special", "private-use", "collection"})

# Two reference files:
#   IANA Language Subtag Registry -> which languages exist, canonical codes,
#       names, macrolanguages, deprecations, valid script/region subtags
#   GeoNames iso-languagecodes.txt -> the 639-3 code of languages whose
#       canonical code is 2 letters (fr <-> fra), which IANA omits. Its 639-2
#       column is ignored: B codes come from ISO639_2B_TO_3 instead


@dataclass(frozen=True)
class IanaLanguage:
    subtag: str
    descriptions: tuple[str, ...]
    scope: str = "individual"
    macrolanguage: str | None = None
    deprecated: bool = False
    preferred_value: str | None = None


@dataclass(frozen=True)
class IanaRegistry:
    file_date: str
    languages: tuple[IanaLanguage, ...]
    scripts: frozenset[str] = field(default_factory=frozenset)
    regions: frozenset[str] = field(default_factory=frozenset)


def read_iana_registry(path: str | Path) -> IanaRegistry:
    """Parse the IANA Language Subtag Registry (RFC 5646 record-jar format).

    Range entries (qaa..qtz, Qaaa..Qabx, QM..QZ) are private use and skipped.
    Deprecated scripts and regions are kept as valid subtags: data written
    before the deprecation still uses them
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    blocks = text.split("\n%%\n")

    header = _parse_record_jar(blocks[0])
    file_date = (header.get("File-Date") or [""])[0]
    if not file_date:
        raise RegistryError(f"{path}: missing File-Date; not an IANA registry file")

    languages: list[IanaLanguage] = []
    scripts: set[str] = set()
    regions: set[str] = set()

    for block in blocks[1:]:
        rec = _parse_record_jar(block)
        kind = (rec.get("Type") or [""])[0]
        subtag = (rec.get("Subtag") or [""])[0]
        if not subtag or ".." in subtag:
            continue
        if kind == "script":
            scripts.add(subtag.title())
        elif kind == "region":
            regions.add(subtag.upper())
        elif kind == "language":
            languages.append(
                IanaLanguage(
                    subtag=subtag.lower(),
                    descriptions=tuple(rec.get("Description", ())),
                    scope=(rec.get("Scope") or ["individual"])[0],
                    macrolanguage=(rec.get("Macrolanguage") or [None])[0],
                    deprecated=bool(rec.get("Deprecated")),
                    preferred_value=(rec.get("Preferred-Value") or [None])[0],
                )
            )

    return IanaRegistry(
        file_date=file_date,
        languages=tuple(sorted(languages, key=lambda lang: lang.subtag)),
        scripts=frozenset(scripts),
        regions=frozenset(regions),
    )


def _parse_record_jar(block: str) -> dict[str, list[str]]:
    record: dict[str, list[str]] = {}
    key: str | None = None
    for line in block.splitlines():
        if not line.strip():
            continue
        if line[0].isspace():  # continuation of the previous field
            if key is not None:
                record[key][-1] += " " + line.strip()
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        record.setdefault(key, []).append(value.strip())
    return record


def read_geonames_language_codes(path: str | Path) -> Iterator[LanguageRecord]:
    """GeoNames iso-languagecodes.txt: ISO 639-3, ISO 639-2, ISO 639-1, name"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) < 4:
                continue
            iso3, iso2, iso1, name = (c.strip() for c in row[:4])
            if not iso3 or not name:
                continue
            b, t = _split_639_2(iso2)
            yield LanguageRecord(
                iso639_3=iso3,
                name=name,
                iso639_1=iso1 or None,
                iso639_2b=b,
                iso639_2t=t,
            )


def _split_639_2(value: str) -> tuple[str | None, str | None]:
    """'fre / fra' -> ('fre', 'fra'); 'deu' -> (None, 'deu')"""
    parts = [p.strip() for p in value.split("/") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, (parts[0] if parts else None)


def build_language_records(
    iana: IanaRegistry, geonames: Iterable[LanguageRecord]
) -> tuple[list[LanguageRecord], dict[str, str | None]]:
    """Join IANA languages with GeoNames codes -> (records, retirements).

    IANA decides which languages exist. GeoNames only adds codes and names to
    them; a GeoNames row with no IANA language is ignored. An IANA 2-letter
    language with no GeoNames row has no 639-3 code, which stops the build:
    skipping it would make `--to <code>` fail with no explanation
    """
    geo = [r.cleaned() for r in geonames]
    by1 = _unique_index(geo, lambda r: r.iso639_1, "ISO 639-1")
    by3 = _unique_index(geo, lambda r: r.iso639_3, "ISO 639-3")
    b_code_of = {iso3: b for b, iso3 in ISO639_2B_TO_3.items()}

    records: list[LanguageRecord] = []
    retirements: dict[str, str | None] = dict(WITHDRAWN_ISO639_2B)
    missing: list[str] = []

    for lang in iana.languages:
        if lang.deprecated:
            retirements[lang.subtag] = lang.preferred_value
            # IANA only knows the retired 2-letter code (`mo`). If GeoNames
            # still lists the language, retire its 3-letter codes (`mol`) too.
            # A code that is still active elsewhere is left alone by the
            # registry, so this can never shadow a live language
            old = by1.get(lang.subtag) if len(lang.subtag) == 2 else None
            if old is not None and old.iso639_3:
                retirements.setdefault(old.iso639_3, lang.preferred_value)
            continue
        if lang.scope in _EXCLUDED_IANA_SCOPES:
            continue

        if len(lang.subtag) == 2:
            geo_row = by1.get(lang.subtag)
            if geo_row is None:
                missing.append(lang.subtag)
                continue
            iso3, iso1 = geo_row.iso639_3, lang.subtag
        else:
            geo_row = by3.get(lang.subtag)
            iso3, iso1 = lang.subtag, None

        names = list(lang.descriptions) or [lang.subtag]
        if geo_row is not None:
            names.append(geo_row.name)

        records.append(
            LanguageRecord(
                iso639_3=iso3,
                name=names[0],
                iso639_1=iso1,
                # T code only recorded where it differs from B; otherwise the
                # 639-3 identifier already covers it
                iso639_2b=b_code_of.get(iso3),
                iso639_2t=iso3 if iso3 in b_code_of else None,
                macrolanguage=lang.macrolanguage,
                scope=(
                    "macrolanguage" if lang.scope == "macrolanguage" else "individual"
                ),
                aliases=tuple(names[1:]),
            )
        )

    if missing:
        raise RegistryError(
            "IANA languages with no row in iso-languagecodes.txt (no ISO 639-3 "
            f"code available): {', '.join(sorted(missing))}"
        )
    return records, retirements


def _unique_index(
    records: Iterable[LanguageRecord], key, label: str
) -> dict[str, LanguageRecord]:
    index: dict[str, LanguageRecord] = {}
    for rec in records:
        k = key(rec)
        if not k:
            continue
        prev = index.get(k)
        if prev is not None and prev.iso639_3 != rec.iso639_3:
            raise RegistryError(
                f"iso-languagecodes.txt: {label} {k!r} on both "
                f"{prev.iso639_3!r} and {rec.iso639_3!r}"
            )
        index[k] = rec
    return index


def build_registry(
    iana_path: str | Path, geonames_path: str | Path
) -> tuple[LanguageRegistry, dict[str, str]]:
    """Build the registry from the two reference files -> (registry, sources).

    `sources` goes into build_metadata via `store_registry`
    """
    iana = read_iana_registry(iana_path)
    records, retirements = build_language_records(
        iana, read_geonames_language_codes(geonames_path)
    )
    registry = LanguageRegistry(records, retirements, iana.scripts, iana.regions)
    return registry, {"language_registry_iana_file_date": iana.file_date}


# Readiness


def is_language_codes_ready(db_path: Path) -> bool:
    """Ready means the stored registry reads back intact under this code's tag
    rules. That single check covers missing tables, an old rules version, a
    partial build (fingerprint mismatch) and an empty table"""
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
            return len(read_registry(conn)) > 0
        finally:
            conn.close()
    except (sqlite3.DatabaseError, RegistryError):
        return False


def stored_registry_fingerprint(conn: sqlite3.Connection) -> str | None:
    """For GeoNames/Wikidata: stamp this at build time, compare it in their
    readiness checks. A changed registry then makes them rebuild"""
    row = conn.execute(
        "SELECT value FROM build_metadata WHERE key = ?", (REGISTRY_FINGERPRINT_KEY,)
    ).fetchone()
    return row[0] if row else None


# Loading


def load_all_data(conn: sqlite3.Connection) -> LanguageRegistry:
    print("Building language registry...")
    registry, sources = build_registry(
        RAW_DIR_LANGUAGE_CODES / IANA_REGISTRY_FILENAME,
        RAW_DIR_LANGUAGE_CODES / ISO_LANGUAGECODES_FILENAME,
    )
    print(
        f"Storing {len(registry):,} languages "
        f"(IANA registry {sources['language_registry_iana_file_date']})..."
    )
    store_registry(conn, registry, sources)
    return registry


def write_build_metadata(conn: sqlite3.Connection) -> None:
    counts = {
        "language_count": "language",
        "language_alias_count": "language_alias",
        "language_retirement_count": "language_retirement",
        "language_subtag_count": "language_subtag",
    }
    conn.executemany(
        "INSERT OR REPLACE INTO build_metadata (key, value) VALUES (?, ?)",
        [
            (key, str(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]))
            for key, table in counts.items()
        ],
    )


# Persistence


def store_registry(
    conn: sqlite3.Connection,
    registry: LanguageRegistry,
    sources: Mapping[str, str] | None = None,
) -> None:
    if registry.scripts is None or registry.regions is None:
        raise RegistryError("refusing to store a registry without script/region data")

    # Macrolanguages first, so the self-referencing FK holds row by row
    rows = sorted(
        registry.records(), key=lambda r: (r.macrolanguage is not None, r.code)
    )
    conn.executemany(
        "INSERT INTO language (code, iso639_3, iso639_1, iso639_2b, iso639_2t, "
        "name, scope, macrolanguage) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r.code,
                r.iso639_3,
                r.iso639_1,
                r.iso639_2b,
                r.iso639_2t,
                r.name,
                r.scope,
                r.macrolanguage,
            )
            for r in rows
        ],
    )
    conn.executemany(
        "INSERT INTO language_alias (code, alias) VALUES (?, ?)",
        [(r.code, a) for r in registry.records() for a in r.aliases],
    )
    conn.executemany(
        "INSERT INTO language_retirement (code, change_to) VALUES (?, ?)",
        sorted(registry.retirements().items()),
    )
    conn.executemany(
        "INSERT INTO language_subtag (kind, subtag) VALUES (?, ?)",
        [("script", s) for s in sorted(registry.scripts)]
        + [("region", r) for r in sorted(registry.regions)],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO build_metadata (key, value) VALUES (?, ?)",
        [
            ("language_tag_rules_version", TAG_RULES_VERSION),
            (REGISTRY_FINGERPRINT_KEY, registry.fingerprint),
            *sorted((sources or {}).items()),
        ],
    )


def read_registry(conn: sqlite3.Connection) -> LanguageRegistry:
    """Rebuild the registry from the database. Query-time code (lookup, CLI)
    uses this, so resolution always uses the registry the rows were built with"""
    meta = dict(
        conn.execute(
            "SELECT key, value FROM build_metadata WHERE key IN (?, ?)",
            ("language_tag_rules_version", REGISTRY_FINGERPRINT_KEY),
        ).fetchall()
    )
    stored_rules = meta.get("language_tag_rules_version")
    if stored_rules != TAG_RULES_VERSION:
        raise RegistryError(
            f"database was built with language tag rules v{stored_rules}, "
            f"this code uses v{TAG_RULES_VERSION}; rebuild with `nte init --force`"
        )

    aliases: dict[str, list[str]] = {}
    for code, alias in conn.execute(
        "SELECT code, alias FROM language_alias ORDER BY code, alias"
    ):
        aliases.setdefault(code, []).append(alias)

    records = [
        LanguageRecord(
            iso639_3=r[1],
            iso639_1=r[2],
            iso639_2b=r[3],
            iso639_2t=r[4],
            name=r[5],
            scope=r[6],
            macrolanguage=r[7],
            aliases=tuple(aliases.get(r[0], ())),
        )
        for r in conn.execute(
            "SELECT code, iso639_3, iso639_1, iso639_2b, iso639_2t, name, scope, "
            "macrolanguage FROM language ORDER BY code"
        )
    ]
    retirements = dict(
        conn.execute(
            "SELECT code, change_to FROM language_retirement ORDER BY code"
        ).fetchall()
    )
    subtags: dict[str, list[str]] = {"script": [], "region": []}
    for kind, subtag in conn.execute(
        "SELECT kind, subtag FROM language_subtag ORDER BY kind, subtag"
    ):
        subtags[kind].append(subtag)

    registry = LanguageRegistry(
        records, retirements, subtags["script"], subtags["region"]
    )
    if registry.fingerprint != meta.get(REGISTRY_FINGERPRINT_KEY):
        raise RegistryError(
            "stored language registry does not match its recorded fingerprint"
        )
    return registry


# Tag reports (written by GeoNames and Wikidata, one table each)


def write_language_tag_report(
    conn: sqlite3.Connection, table: str, report: list[tuple[SourceTag, int]]
) -> None:
    """One row per distinct raw tag: what it mapped to, how many rows had it"""
    conn.executemany(
        f"""
        INSERT INTO {table} (
            raw_tag, tag_status, lang, lang_script, lang_region, lang_variant,
            row_count, note
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                st.raw,
                st.status,
                st.tag.lang if st.tag else None,
                st.tag.script if st.tag else None,
                st.tag.region if st.tag else None,
                st.tag.variant if st.tag else None,
                count,
                st.note,
            )
            for st, count in report
        ],
    )


def print_language_tag_summary(
    report: list[tuple[SourceTag, int]], limit: int = 10
) -> None:
    rows_by_status: dict[str, int] = {}
    for st, count in report:
        rows_by_status[st.status] = rows_by_status.get(st.status, 0) + count
    print(
        "Language tags: "
        + ", ".join(f"{status} {n:,}" for status, n in sorted(rows_by_status.items()))
    )

    unmapped = [(st, n) for st, n in report if st.status == "unmapped"]
    if unmapped:
        print(f"{len(unmapped)} unmapped tag(s), most rows first:")
        for st, n in unmapped[:limit]:
            print(f"  {st.raw!r}: {n:,} rows ({st.note})")
