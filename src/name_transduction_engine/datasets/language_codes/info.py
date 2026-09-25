import sqlite3

from dataclasses import dataclass

from name_transduction_engine.normalization.language_code_normalization import (
    LanguageRecord,
    LanguageRegistry,
    LanguageTag,
)

# (source name, tag report table). Kept here rather than imported from the
# dataset schemas so this module works even if one dataset was never built.
TAG_REPORT_TABLES: tuple[tuple[str, str], ...] = (
    ("geonames", "geonames_language_tag"),
    ("wikidata", "wikidata_language_tag"),
)


@dataclass(frozen=True)
class RawTagCount:
    source: str
    raw_tag: str
    tag: LanguageTag
    rows: int


@dataclass(frozen=True)
class Coverage:
    """Name counts per language and source, built once from the tag reports"""

    by_lang: dict[str, dict[str, int]]  # lang -> {source: rows}
    sources: tuple[str, ...]  # sources whose report table exists

    def total(self, lang: str) -> int:
        return sum(self.by_lang.get(lang, {}).values())

    def per_source(self, lang: str) -> dict[str, int]:
        counts = self.by_lang.get(lang, {})
        return {source: counts.get(source, 0) for source in self.sources}


def _existing_report_tables(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    existing = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    return [(source, table) for source, table in TAG_REPORT_TABLES if table in existing]


def read_coverage(conn: sqlite3.Connection) -> Coverage:
    tables = _existing_report_tables(conn)
    by_lang: dict[str, dict[str, int]] = {}
    for source, table in tables:
        for lang, rows in conn.execute(
            f"SELECT lang, SUM(row_count) FROM {table} "
            "WHERE lang IS NOT NULL GROUP BY lang ORDER BY lang"
        ):
            by_lang.setdefault(lang, {})[source] = int(rows)
    return Coverage(by_lang=by_lang, sources=tuple(source for source, _ in tables))


def raw_tags_for(conn: sqlite3.Connection, lang: str) -> list[RawTagCount]:
    """Every raw source tag normalized to `lang`, most rows first"""
    out: list[RawTagCount] = []
    for source, table in _existing_report_tables(conn):
        for raw, script, region, variant, rows in conn.execute(
            f"SELECT raw_tag, lang_script, lang_region, lang_variant, row_count "
            f"FROM {table} WHERE lang = ?",
            (lang,),
        ):
            out.append(
                RawTagCount(
                    source, raw, LanguageTag(lang, script, region, variant), int(rows)
                )
            )
    return sorted(out, key=lambda r: (-r.rows, r.source, r.raw_tag))


def rows_for_tag(raw_tags: list[RawTagCount], tag: LanguageTag) -> dict[str, int]:
    """Rows per source matching `tag` the way the lookup matches it: subtags the
    query leaves out match anything"""
    counts: dict[str, int] = {}
    for r in raw_tags:
        if (
            (tag.script is None or r.tag.script == tag.script)
            and (tag.region is None or r.tag.region == tag.region)
            and (tag.variant is None or r.tag.variant == tag.variant)
        ):
            counts[r.source] = counts.get(r.source, 0) + r.rows
    return counts


# Search


@dataclass(frozen=True)
class SearchHit:
    code: str
    rank: int  # 0 code, 1 exact name, 2 name prefix, 3 name substring
    matched: str  # the code or name that matched


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()


def search_languages(registry: LanguageRegistry, text: str) -> list[SearchHit]:
    """Codes and names (including aliases) containing `text`, best matches first.

    Order is deterministic: rank, then code. Callers may re-sort within a rank
    (e.g. by data coverage)
    """
    query = _fold(text)
    if not query:
        return []

    hits: list[SearchHit] = []
    for rec in registry.records():
        hit = _match_record(rec, query)
        if hit is not None:
            hits.append(hit)
    return sorted(hits, key=lambda h: (h.rank, h.code))


def _match_record(rec: LanguageRecord, query: str) -> SearchHit | None:
    codes = [
        c
        for c in (rec.code, rec.iso639_1, rec.iso639_3, rec.iso639_2b, rec.iso639_2t)
        if c
    ]
    if query in codes:
        return SearchHit(rec.code, 0, query)

    labels = (rec.name, *rec.aliases)
    best: SearchHit | None = None
    for label in labels:
        folded = _fold(label)
        if folded == query:
            rank = 1
        elif folded.startswith(query):
            rank = 2
        elif query in folded:
            rank = 3
        else:
            continue
        if best is None or rank < best.rank:
            best = SearchHit(rec.code, rank, label)
    return best
