from dataclasses import dataclass
from enum import StrEnum

from name_transduction_engine.transduction.lookup.lookup_entity import (
    LookupEntity,
    LookupOutcome,
)
from name_transduction_engine.transliteration.romanization import Romanization
from name_transduction_engine.normalization.language_code_normalization import (
    LanguageQuery,
    LanguageRecord,
)
from name_transduction_engine.datasets.language_codes.info import RawTagCount


@dataclass(frozen=True)
class LookupResult:
    outcome: LookupOutcome
    entities: tuple[LookupEntity, ...]
    language: LanguageQuery


class RomanizeOutcome(StrEnum):
    ROMANIZED = "romanized"
    ALREADY_LATIN = "already_latin"
    NATIVE_KEPT = "native_kept"  # no romanization available for the script
    UNTRANSDUCIBLE = "untransducible"  # nothing to romanize (spaces, emoji)


@dataclass(frozen=True)
class RomanizationResult:
    outcome: RomanizeOutcome
    romanization: Romanization
    language: LanguageQuery


@dataclass(frozen=True)
class LanguageSummary:
    """One language that has names in the database"""

    code: str
    name: str | None  # None when the code is not in the registry
    counts: dict[str, int]  # names per source


@dataclass(frozen=True)
class LanguageListResult:
    sources: tuple[str, ...]  # datasets with name counts; empty: none built
    languages: tuple[LanguageSummary, ...]  # most names first


@dataclass(frozen=True)
class LanguageMatch:
    """One language whose code or name contains the search text"""

    code: str
    name: str
    matched: str | None  # the alias or code that matched, if not `name`/`code`
    names: int


@dataclass(frozen=True)
class LanguageSearchResult:
    # Best matches first; within each match quality, more names first
    matches: tuple[LanguageMatch, ...]


@dataclass(frozen=True)
class RelatedLanguage:
    """A macrolanguage or member language, with how many names it has"""

    code: str
    name: str | None  # None when the code is not in the registry
    names: int


@dataclass(frozen=True)
class TagCount:
    """One raw source tag normalized to the language. `counted` is False when
    the query's subtags exclude it (`zh-Hant` does not count `zh-hans` rows)"""

    raw: RawTagCount
    counted: bool


@dataclass(frozen=True)
class LanguageResult:
    """Everything known about one language: its registry record, the names
    `nte lookup --to` can return for it, and related languages that have names"""

    language: LanguageQuery
    record: LanguageRecord
    sources: tuple[str, ...]  # datasets with name counts; empty: none built
    counts: dict[str, int]  # names per source matching the tag
    tags: tuple[TagCount, ...]  # raw tags, most names first
    macrolanguage: RelatedLanguage | None
    members: tuple[RelatedLanguage, ...]  # most names first
    # The related language with the most names, when this one has none
    suggestion: RelatedLanguage | None

    @property
    def total(self) -> int:
        return sum(self.counts.values())
