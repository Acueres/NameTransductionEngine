import icu

from functools import lru_cache
from dataclasses import dataclass
from typing import Literal

LATIN = "Latn"
IDENTITY = "identity"

# Script codes carrying no script identity of their own: digits, punctuation,
# combining marks, unassigned
_NEUTRAL_SCRIPTS = frozenset({"Zyyy", "Zinh", "Zzzz"})

Confidence = Literal["high", "normal", "low"]


@dataclass(frozen=True)
class Romanization:
    text: str
    source_script: str
    transform: str
    confidence: Confidence
    warnings: tuple[str, ...]
    engine_version: str


def dominant_script(text: str) -> tuple[str | None, tuple[str, ...]]:
    """Most frequent script with an identity of its own, or None"""
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    for index, char in enumerate(text):
        script = _script_of(char)
        if script in _NEUTRAL_SCRIPTS:
            continue
        counts[script] = counts.get(script, 0) + 1
        first_seen.setdefault(script, index)

    if not counts:
        return None, ()

    dominant = min(counts, key=lambda s: (-counts[s], first_seen[s]))

    warnings: tuple[str, ...] = ()
    if len(counts) > 1:
        others = ", ".join(sorted(s for s in counts if s != dominant))
        warnings = (f"mixed scripts present: {dominant} (dominant), {others}",)

    return dominant, warnings


def non_latin_scripts(text: str) -> set[str]:
    return {
        script
        for script in (_script_of(char) for char in text)
        if script not in _NEUTRAL_SCRIPTS and script != LATIN
    }


def script_of(char: str) -> str:
    """ISO 15924 short code for one character"""
    return _script_of(char)


def is_neutral_script(script: str) -> bool:
    """Digits, punctuation, combining marks: no script identity of their own"""
    return script in _NEUTRAL_SCRIPTS


@lru_cache(maxsize=1 << 14)
def _script_of(char: str) -> str:
    """ISO 15924 short code for one character"""
    try:
        return icu.Script.getScript(char).getShortName()
    except icu.ICUError:
        return "Zzzz"
