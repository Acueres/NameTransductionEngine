import icu

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

Confidence = Literal["high", "normal", "low"]

LATIN = "Latn"
IDENTITY = "identity"

# Script codes carrying no script identity of their own: digits, punctuation,
# combining marks, unassigned
_NEUTRAL_SCRIPTS = frozenset({"Zyyy", "Zinh", "Zzzz"})

# Terminal fallback
_GENERIC_TRANSFORM = "Any-Latin"


@dataclass(frozen=True)
class Romanization:
    text: str
    source_script: str
    transform: str
    confidence: Confidence
    warnings: tuple[str, ...]
    icu_version: str


@dataclass(frozen=True)
class _Route:
    transform: str
    confidence: Confidence
    caveat: str | None = None


_ROUTES: dict[str, _Route] = {
    # Alphabets with a reversible, diacritic-based standard mapping
    "Cyrl": _Route("Cyrillic-Latin", "normal"),
    "Grek": _Route("Greek-Latin", "normal"),
    "Armn": _Route("Armenian-Latin", "normal"),
    "Geor": _Route("Georgian-Latin", "normal"),
    "Beng": _Route("Bengali-Latin", "normal"),
    "Deva": _Route("Devanagari-Latin", "normal"),
    "Gujr": _Route("Gujarati-Latin", "normal"),
    "Guru": _Route("Gurmukhi-Latin", "normal"),
    "Knda": _Route("Kannada-Latin", "normal"),
    "Mlym": _Route("Malayalam-Latin", "normal"),
    "Orya": _Route("Oriya-Latin", "normal"),
    "Taml": _Route("Tamil-Latin", "normal"),
    "Telu": _Route("Telugu-Latin", "normal"),
    "Thai": _Route("Thai-Latin", "normal"),
    # Abjads: short vowels are not written, so they are not recovered
    "Arab": _Route(
        "Arabic-Latin",
        "low",
        "abjad: unwritten short vowels are not recovered",
    ),
    "Hebr": _Route(
        "Hebrew-Latin",
        "low",
        "abjad: unwritten short vowels are not recovered",
    ),
    # CJK. Han is romanized as Mandarin pinyin regardless of the language the
    # text is actually in, so Japanese and Korean names written in Han come out
    # as Chinese readings. Acceptable as a stable debug handle, never as a name
    "Hani": _Route(
        "Han-Latin",
        "low",
        "Han is romanized as Mandarin pinyin regardless of source language",
    ),
    "Hira": _Route("Hiragana-Latin", "normal"),
    "Kana": _Route("Katakana-Latin", "normal"),
    "Hang": _Route(
        "Hangul-Latin",
        "low",
        "optimized for reversibility, not conventional romanization",
    ),
}


class Romanizer:
    def romanize(self, text: str) -> Romanization:
        normalized = _nfc(text)
        script, warnings = _dominant_script(normalized)

        # Already Latin, or nothing with a script identity at all
        if script in (LATIN, None):
            if script is None:
                warnings += ("no romanizable content",)
            return Romanization(
                text=normalized,
                source_script=script or "Zzzz",
                transform=IDENTITY,
                confidence="high",
                warnings=warnings,
                icu_version=icu.ICU_VERSION,
            )

        route, resolve_warnings = _resolve_route(script)
        warnings += resolve_warnings

        result, transform, warnings = self._apply(normalized, route.transform, warnings)
        confidence = route.confidence
        if route.caveat:
            warnings += (route.caveat,)

        # Postcondition: the output must actually be Latin
        residue = _non_latin_scripts(result)
        if residue and transform != _GENERIC_TRANSFORM:
            # Usually mixed-script input: the script-specific transform left the
            # other runs alone. Retry with the generic transform once
            retry, transform, warnings = self._apply(
                normalized, _GENERIC_TRANSFORM, warnings
            )
            if not _non_latin_scripts(retry):
                result = retry
                confidence = "low"
                warnings += (f"mixed script; romanized via {_GENERIC_TRANSFORM}",)
            else:
                residue = _non_latin_scripts(retry)
                result, confidence, warnings = (
                    retry,
                    "low",
                    warnings
                    + (
                        f"output retains non-Latin scripts: "
                        f"{', '.join(sorted(residue))}",
                    ),
                )
        elif residue:
            confidence = "low"
            warnings += (
                f"output retains non-Latin scripts: {', '.join(sorted(residue))}",
            )

        # In case of failure fall back to the native form
        if normalized and not result.strip():
            return Romanization(
                text=normalized,
                source_script=script,
                transform=IDENTITY,
                confidence="low",
                warnings=warnings + (f"{transform} produced empty output",),
                icu_version=icu.ICU_VERSION,
            )

        return Romanization(
            text=result,
            source_script=script,
            transform=transform,
            confidence=confidence,
            warnings=warnings,
            icu_version=icu.ICU_VERSION,
        )

    def _apply(
        self, text: str, transform_id: str, warnings: tuple[str, ...]
    ) -> tuple[str, str, tuple[str, ...]]:
        """Apply one transform, degrading to the generic one if it fails"""
        try:
            return (
                _nfc(_transliterator(transform_id).transliterate(text)),
                transform_id,
                warnings,
            )
        except icu.ICUError as exc:
            if transform_id == _GENERIC_TRANSFORM:
                raise
            warnings += (f"{transform_id} failed ({exc}); used {_GENERIC_TRANSFORM}",)
            generic = _transliterator(_GENERIC_TRANSFORM)
            return _nfc(generic.transliterate(text)), _GENERIC_TRANSFORM, warnings


@lru_cache(maxsize=None)
def _transliterator(transform_id: str) -> icu.Transliterator:
    return icu.Transliterator.createInstance(transform_id)


@lru_cache(maxsize=1)
def _available_transform_ids() -> frozenset[str]:
    return frozenset(icu.Transliterator.getAvailableIDs())


@lru_cache(maxsize=1)
def _nfc_normalizer() -> icu.Normalizer2:
    return icu.Normalizer2.getNFCInstance()


def _nfc(text: str) -> str:
    """ICU output can carry combining marks in arbitrary order. This function fixes that"""
    return _nfc_normalizer().normalize(text)


@lru_cache(maxsize=1 << 14)
def _script_of(char: str) -> str:
    """ISO 15924 short code for one character"""
    try:
        return icu.Script.getScript(char).getShortName()
    except icu.ICUError:
        return "Zzzz"


def _dominant_script(text: str) -> tuple[str | None, tuple[str, ...]]:
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


def _non_latin_scripts(text: str) -> set[str]:
    return {
        script
        for script in (_script_of(char) for char in text)
        if script not in _NEUTRAL_SCRIPTS and script != LATIN
    }


def _resolve_route(script: str) -> tuple[_Route, tuple[str, ...]]:
    route = _ROUTES.get(script)

    if route is None:
        return (
            _Route(_GENERIC_TRANSFORM, "low"),
            (f"no authored route for script {script}; used {_GENERIC_TRANSFORM}",),
        )

    if route.transform not in _available_transform_ids():
        return (
            _Route(_GENERIC_TRANSFORM, "low", route.caveat),
            (
                f"{route.transform} unavailable in ICU {icu.ICU_VERSION}; "
                f"used {_GENERIC_TRANSFORM}",
            ),
        )

    return route, ()
