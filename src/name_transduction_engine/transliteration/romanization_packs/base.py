"""Shared types for display-romanization providers.

A provider romanizes one script run of a name. It returns a Rendering, returns
None when it does not apply (wrong language, nothing to do), or raises:
- ProviderUnavailable: an optional dependency is missing; the chain continues
- NoRomanization: romanizing this run would be wrong rather than merely rough
  (Japanese kanji read as Mandarin, say); the chain stops at the native form
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

import langcodes

from ..romanization import Confidence

CONFIDENCE_RANK: dict[Confidence, int] = {"low": 0, "normal": 1, "high": 2}


@dataclass(frozen=True)
class Context:
    """What is known about the name's language, from its tag"""

    lang: str | None  # primary subtag, shortest form: "uk", "zh", "ckb"
    script: str | None  # ISO 15924 subtag from the tag: "Hant", "Latn"
    region: str | None  # "TW", "KP"
    tag: str | None  # the tag as given
    # Latin-script names of the same entity ("Venice" for ونیز). A provider
    # whose script leaves sounds unwritten may use them to choose a reading
    hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rendering:
    text: str
    transform: str  # e.g. "icu:ru-ru_Latn/BGN", "nte:hepburn"
    confidence: Confidence
    warnings: tuple[str, ...] = ()
    engine: str | None = None  # e.g. "ICU 78.3", "pypinyin 0.55.0"
    # False when the provider produced lowercase output from a caseless script;
    # the orchestrator then capitalizes each word
    cased: bool = True


Provider = Callable[[str, Context], Rendering | None]


class ProviderUnavailable(Exception):
    """An optional dependency of this provider is not installed"""


class NoRomanization(Exception):
    """Any romanization of this run would be misleading: keep the native form"""


# Valid ISO 639 codes that name no particular language
_NO_LANGUAGE = frozenset({"und", "mul", "mis", "zxx"})

# Individual languages that should use their macrolanguage's rules
_MACRO_ALIASES = {
    "cmn": "zh",
    "arb": "ar",
    # Spoken Arabic varieties, written in the same script: Arabic rules
    "arz": "ar",  # Egyptian
    "apc": "ar",  # Levantine
    "ajp": "ar",  # South Levantine
    "acm": "ar",  # Mesopotamian
    "afb": "ar",  # Gulf
    "ary": "ar",  # Moroccan
    "arq": "ar",  # Algerian
    "aeb": "ar",  # Tunisian
    "ayl": "ar",  # Libyan
    "apd": "ar",  # Sudanese
    "ars": "ar",  # Najdi
    "acw": "ar",  # Hijazi
    "pes": "fa",
    "prs": "fa",  # Dari
    "uzn": "uz",
    "khk": "mn",
    "ekk": "et",
    "ydd": "yi",
    "zsm": "ms",
    "azj": "az",
    "pbu": "ps",
    "pst": "ps",
    "urd": "ur",
    "npi": "ne",
}


@lru_cache(maxsize=1 << 10)
def parse_context(tag: str | None) -> tuple[Context, tuple[str, ...]]:
    if not tag:
        return Context(None, None, None, tag), ()
    try:
        parsed = langcodes.Language.get(tag)
    except (LookupError, ValueError):
        return Context(None, None, None, tag), (
            f"unrecognized language tag {tag!r}; language rules not applied",
        )
    lang = parsed.language
    if lang is None or lang in _NO_LANGUAGE or lang.startswith("x-"):
        lang = None
    else:
        lang = _MACRO_ALIASES.get(lang, lang)
    return Context(lang, parsed.script, parsed.territory, tag), ()


def package_version(package: str, label: str | None = None) -> str:
    try:
        return f"{label or package} {version(package)}"
    except PackageNotFoundError:
        return f"{label or package} (version unknown)"
