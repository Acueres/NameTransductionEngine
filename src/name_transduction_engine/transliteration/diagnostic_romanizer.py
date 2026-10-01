"""Diagnostic romanization: a readable Latin handle for any name, for debugging.

Contract:
- Input with no non-Latin content (Latin letters, digits, punctuation only) is
  returned unchanged as an identity result, diacritics included.
- Anything else is romanized by uroman as one string, so every script run in
  mixed-script input is covered, whichever script dominates.
- Output is Latin and mostly ASCII, but ASCII is not guaranteed: uroman keeps a
  few marks (the Tibetan tsheg comes out as "·").
- Not reversible and not a display form. Always show it next to the native name.
"""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import langcodes
import uroman as ur

from .romanization import (
    IDENTITY,
    Confidence,
    Romanization,
    dominant_script,
    non_latin_scripts,
)

TRANSFORM = "uroman"

# Known limits of uroman readings, keyed by source script
_SCRIPT_CAVEATS: dict[str, str] = {
    "Hani": "Han is read as Mandarin pinyin regardless of source language",
    "Arab": "abjad: unwritten short vowels are not recovered",
    "Hebr": "abjad: unwritten short vowels are not recovered",
}

# Valid ISO 639 codes that name no particular language
_NO_LANGUAGE = frozenset({"und", "mul", "mis", "zxx"})


class DiagnosticRomanizer:
    def romanize(self, text: str, lang: str | None = None) -> Romanization:
        """Romanize `text`. `lang` is the name's language tag in any form
        langcodes accepts ("uk", "ukr", "zh-Hant"); uroman applies
        language-specific rules for some languages (ukr, bel, bul, ...)"""
        script, warnings = dominant_script(text)
        source_script = script or "Zzzz"

        present = non_latin_scripts(text)
        if not present:
            if script is None:
                warnings += ("no romanizable content",)
            return _identity(text, source_script, "high", warnings)

        lcode, lang_warnings = _uroman_lcode(lang)
        warnings += lang_warnings

        try:
            result = _romanize_str(text, lcode)
        except Exception as exc:  # uroman raises assorted errors on odd input
            return _identity(
                text, source_script, "low", warnings + (f"uroman failed: {exc}",)
            )

        # In case of failure fall back to the native form
        if not result.strip():
            return _identity(
                text, source_script, "low", warnings + ("uroman produced empty output",)
            )

        confidence: Confidence = "normal"

        caveats = tuple(
            _SCRIPT_CAVEATS[s] for s in sorted(present) if s in _SCRIPT_CAVEATS
        )
        if caveats:
            confidence = "low"
            warnings += tuple(dict.fromkeys(caveats))  # Arab and Hebr share one

        # Postcondition: the output must actually be Latin
        residue = non_latin_scripts(result)
        if residue:
            confidence = "low"
            warnings += (
                f"output retains non-Latin scripts: {', '.join(sorted(residue))}",
            )

        return Romanization(
            text=result,
            source_script=source_script,
            transform=TRANSFORM,
            confidence=confidence,
            warnings=warnings,
            engine_version=_engine_version(),
        )


def _identity(
    text: str, source_script: str, confidence: Confidence, warnings: tuple[str, ...]
) -> Romanization:
    return Romanization(
        text=text,
        source_script=source_script,
        transform=IDENTITY,
        confidence=confidence,
        warnings=warnings,
        engine_version=_engine_version(),
    )


@lru_cache(maxsize=1 << 10)
def _uroman_lcode(lang: str | None) -> tuple[str | None, tuple[str, ...]]:
    """ISO 639-3 code for uroman, which ignores any other form ("uk" does nothing)"""
    if not lang:
        return None, ()
    try:
        code = langcodes.Language.get(lang).to_alpha3()
    except (LookupError, ValueError):
        return None, (f"unrecognized language tag {lang!r}; no language rules applied",)
    return (None if code in _NO_LANGUAGE else code), ()


def _romanize_str(text: str, lcode: str | None) -> str:
    result = _uroman_normalizer().romanize_string(
        text, lcode, rom_format=ur.RomFormat.STR
    )
    if not isinstance(result, str):
        raise TypeError(f"expected str, got {type(result).__name__}")
    return result


@lru_cache(maxsize=1)
def _uroman_normalizer() -> ur.Uroman:
    # Loading takes seconds: keep this lazy, never call it at import time
    return ur.Uroman()


@lru_cache(maxsize=1)
def _engine_version() -> str:
    try:
        return f"uroman {version('uroman')}"
    except PackageNotFoundError:
        return "uroman (version unknown)"
