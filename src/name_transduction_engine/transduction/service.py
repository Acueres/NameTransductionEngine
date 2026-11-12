"""What NTE can be asked, shared by the CLI and the API.

Each function returns a frozen result from `results` and never prints.
Failures are raised:

- DataUnavailable: `nte update` is rebuilding the data (CLI exit 3, API 503)
- DataMissing: the database is missing, incomplete or unreadable
- UnknownLanguageError: the language did not resolve; carries suggestions
- InvalidInput: the request cannot be answered as given

Front ends import everything they need from this module.
"""

import sqlite3

from collections.abc import Callable
from contextlib import closing
from functools import wraps
from typing import ParamSpec, TypeVar

from name_transduction_engine.datasets.availability import (
    DataUnavailable,
    check_data_available,
)
from name_transduction_engine.datasets.language_codes.info import (
    raw_tags_for,
    rows_for_tag,
    search_languages,
)
from name_transduction_engine.normalization.language_code_normalization import (
    RegistryError,
    UnknownLanguageError,
    resolve_user_language,
)
from name_transduction_engine.paths import DB_PATH
from name_transduction_engine.transduction.lookup.lookup_engine import lookup_name

from .db import get_conn
from .resources import get_coverage, get_registry, get_romanizer
from .results import (
    LanguageListResult,
    LanguageMatch,
    LanguageResult,
    LanguageSearchResult,
    LanguageSummary,
    LookupResult,
    RelatedLanguage,
    RomanizationResult,
    TagCount,
)

__all__ = [
    # Operations
    "get_lookup_result",
    "get_romanization_result",
    "get_language_result",
    "get_language_list_result",
    "get_language_search_result",
    # Failures
    "DataMissing",
    "DataUnavailable",
    "InvalidInput",
    "UnknownLanguageError",
    # Results
    "LanguageListResult",
    "LanguageMatch",
    "LanguageResult",
    "LanguageSearchResult",
    "LanguageSummary",
    "LookupResult",
    "RelatedLanguage",
    "RomanizationResult",
    "TagCount",
]


class DataMissing(Exception):
    """The database is missing, incomplete or unreadable: run `nte init`"""

    def __init__(self, what: str, cause: Exception):
        super().__init__(f"{what} data unavailable ({cause})")


class InvalidInput(ValueError):
    """The request cannot be answered as given (an empty name)"""


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _operation(what: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Refuse while the data is being rebuilt; report database errors as
    DataMissing("<what> data unavailable (...)")"""

    def decorate(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            check_data_available()
            try:
                return func(*args, **kwargs)
            except (RegistryError, sqlite3.DatabaseError) as exc:
                raise DataMissing(what, exc) from exc

        return wrapper

    return decorate


@_operation("lookup")
def get_lookup_result(lang: str, name: str) -> LookupResult:
    if not name.strip():
        raise InvalidInput("name is empty")
    language = resolve_user_language(lang, get_registry(DB_PATH))
    entities = lookup_name(name, language.tag, get_romanizer(), DB_PATH)
    return LookupResult(entities=tuple(entities), language=language)


@_operation("language")
def get_romanization_result(lang: str, text: str) -> RomanizationResult:
    language = resolve_user_language(lang, get_registry(DB_PATH))
    romanization = get_romanizer().romanize(text, str(language.tag))
    return RomanizationResult(romanization, language)


@_operation("language")
def get_language_list_result() -> LanguageListResult:
    registry = get_registry(DB_PATH)
    coverage = get_coverage(DB_PATH)
    codes = sorted(coverage.by_lang, key=lambda code: (-coverage.total(code), code))
    languages = tuple(
        LanguageSummary(
            code=code,
            name=registry.record(code).name if code in registry else None,
            counts=coverage.per_source(code),
        )
        for code in codes
    )
    return LanguageListResult(sources=coverage.sources, languages=languages)


@_operation("language")
def get_language_search_result(text: str) -> LanguageSearchResult:
    registry = get_registry(DB_PATH)
    coverage = get_coverage(DB_PATH)

    hits = search_languages(registry, text)
    # Within each match quality, languages with more names first
    hits.sort(key=lambda h: (h.rank, -coverage.total(h.code), h.code))

    matches = []
    for hit in hits:
        name = registry.record(hit.code).name
        matched = None if hit.matched in (name, hit.code) else hit.matched
        matches.append(LanguageMatch(hit.code, name, matched, coverage.total(hit.code)))
    return LanguageSearchResult(matches=tuple(matches))


@_operation("language")
def get_language_result(query: str) -> LanguageResult:
    registry = get_registry(DB_PATH)
    coverage = get_coverage(DB_PATH)
    language = resolve_user_language(query, registry)
    tag = language.tag
    record = registry.record(tag.lang)

    # Coverage: what `nte lookup --to <query>` can actually return
    with closing(get_conn(DB_PATH)) as conn:
        raw_tags = raw_tags_for(conn, tag.lang)
    counts = rows_for_tag(raw_tags, tag)
    tags = tuple(TagCount(r, bool(rows_for_tag([r], tag))) for r in raw_tags)

    # Macrolanguage relations, so the caller can see where names are
    def related(code: str) -> RelatedLanguage:
        name = registry.record(code).name if code in registry else None
        return RelatedLanguage(code, name, coverage.total(code))

    macrolanguage = related(record.macrolanguage) if record.macrolanguage else None
    members = tuple(
        sorted(
            (related(m) for m in registry.members_of(tag.lang)),
            key=lambda m: (-m.names, m.code),
        )
    )

    # Nothing here, but a related language may have names
    suggestion = None
    if coverage.sources and not sum(counts.values()):
        candidates = [r for r in (macrolanguage, *members) if r and r.names]
        if candidates:
            suggestion = max(candidates, key=lambda r: (r.names, r.code))

    return LanguageResult(
        language=language,
        record=record,
        sources=coverage.sources,
        counts=counts,
        tags=tags,
        macrolanguage=macrolanguage,
        members=members,
        suggestion=suggestion,
    )
