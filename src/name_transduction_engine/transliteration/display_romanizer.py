"""Display romanization: the Latin form of a name shown to end users.

Not reversible and not for matching. Rules are chosen by the name's language
(and region where it matters), not only by its script:
- 東京 tagged ja is Tōkyō, 北京 tagged zh is Beijing
- 종로구 is Jongno-gu, but McCune-Reischauer for a name tagged ko-KP
- Київ tagged uk is Kyiv (Ukrainian national system), Москва tagged ru is
  Moskva (BGN/PCGN)

Each script run of the name goes through a chain and the first usable result
wins:
1. the provider for the run's script and language (see display/)
2. ICU's generic script transform, for scripts where it reads well
3. uroman
4. the native form, unchanged
A result is usable when it is non-empty and fully Latin. A provider can also
stop the chain at the native form when every romanization would be
misleading (Japanese kanji with no Japanese reader available: a Chinese
reading is wrong, not rough).

Latin runs, digits and punctuation are kept as they are.
"""

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Callable

from .romanization_packs import (
    alphabets,
    arabic,
    arabic_script,
    chinese,
    fallbacks,
    hebrew,
    indic,
    japanese,
    korean,
    persian,
    thai,
)
from .romanization_packs.base import (
    CONFIDENCE_RANK,
    Context,
    NoRomanization,
    ProviderUnavailable,
    Rendering,
    parse_context,
)
from .romanization_packs.text import titlecase_words
from .romanization import (
    IDENTITY,
    LATIN,
    Confidence,
    Romanization,
    dominant_script,
    is_neutral_script,
    non_latin_scripts,
    script_of,
)

RunProvider = Callable[[str, str, Context], Rendering | None]


def _for_script(fn: Callable[[str, Context], Rendering | None]) -> RunProvider:
    return lambda script, text, ctx: fn(text, ctx)


# Providers per script group, most specific first
_PROVIDERS: dict[str, list[tuple[str, RunProvider]]] = {
    "Cyrl": [("cyrillic", _for_script(alphabets.romanize_cyrillic))],
    "Jpan": [("japanese", _for_script(japanese.romanize_japanese))],
    "Hani": [("chinese", _for_script(chinese.romanize_chinese))],
    "Kore": [("korean", _for_script(korean.romanize_korean))],
    "Arab": [
        ("persian", _for_script(persian.romanize_persian)),
        ("arabic", _for_script(arabic.romanize_arabic)),
        ("arabic-script", _for_script(arabic_script.romanize_arabic_script)),
    ],
    "Hebr": [("hebrew", _for_script(hebrew.romanize_hebrew_script))],
    "Thai": [("thai", _for_script(thai.romanize_thai))],
}
for _script in ("Grek", "Armn", "Geor", "Thaa", "Ethi", "Sinh"):
    _PROVIDERS[_script] = [("alphabet", alphabets.romanize_alphabet)]
for _script in ("Deva", "Beng", "Guru", "Gujr", "Orya", "Taml", "Telu", "Knda", "Mlym"):
    _PROVIDERS[_script] = [("indic", indic.romanize_indic)]

_FALLBACKS: list[tuple[str, RunProvider]] = [
    ("icu-generic", fallbacks.romanize_icu_generic),
    ("uroman", _for_script(fallbacks.romanize_uroman)),
]


@dataclass(frozen=True)
class _Run:
    script: str  # script group: an ISO 15924 code, or Jpan/Kore for CJK
    text: str


class DisplayRomanizer:
    def romanize(
        self, text: str, lang: str | None = None, hints: Sequence[str] = ()
    ) -> Romanization:
        """Romanize `text` for display. `lang` is the name's language tag
        ("ja", "uk", "zh-Hant-TW", "ko-KP"); without it, the language is
        assumed from the script and results carry a warning. `hints` are
        other names of the same entity; Latin-script ones may guide the
        reading of scripts that leave vowels unwritten (Persian)"""
        text = unicodedata.normalize("NFC", text)
        source, warnings = dominant_script(text)
        source_script = source or "Zzzz"

        if not non_latin_scripts(text):
            if source is None:
                warnings += ("no romanizable content",)
            return Romanization(text, source_script, IDENTITY, "high", warnings, "nte")

        ctx, ctx_warnings = parse_context(lang)
        warnings += ctx_warnings
        latin_hints = tuple(dict.fromkeys(h for h in hints if _is_latin_name(h)))
        if latin_hints:
            ctx = replace(ctx, hints=latin_hints)

        pieces: list[str] = []
        transforms: list[str] = []
        engines: list[str] = []
        confidence: Confidence = "high"
        for run in _runs(text, ctx):
            if run.script == LATIN:
                pieces.append(run.text)
                continue
            rendering = _romanize_run(run, ctx)
            pieces.append(rendering.text)
            transforms.append(rendering.transform)
            if rendering.engine:
                engines.append(rendering.engine)
            warnings += rendering.warnings
            if CONFIDENCE_RANK[rendering.confidence] < CONFIDENCE_RANK[confidence]:
                confidence = rendering.confidence

        out = unicodedata.normalize("NFC", "".join(pieces))
        # One transform for the whole name only if every run kept its native form
        transform = " + ".join(dict.fromkeys(transforms)) or IDENTITY
        return Romanization(
            text=out,
            source_script=source_script,
            transform=transform,
            confidence=confidence,
            warnings=tuple(dict.fromkeys(warnings)),
            engine_version="; ".join(dict.fromkeys(engines)) or "nte",
        )


def _is_latin_name(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all(script_of(ch) == LATIN for ch in letters)


def _romanize_run(run: _Run, ctx: Context) -> Rendering:
    warnings: tuple[str, ...] = ()
    chain = _PROVIDERS.get(run.script, []) + _FALLBACKS
    script = _base_script(run.script)
    for name, provider in chain:
        try:
            rendering = provider(script, run.text, ctx)
        except ProviderUnavailable as exc:
            warnings += (f"{name}: {exc}",)
            continue
        except NoRomanization as exc:
            return _native(run, warnings + (f"native form kept: {exc}",))
        except Exception as exc:  # a provider bug must not break a lookup
            warnings += (f"{name} failed: {type(exc).__name__}: {exc}",)
            continue
        if rendering is None:
            continue

        text = rendering.text if rendering.cased else titlecase_words(rendering.text)
        residue = non_latin_scripts(text)
        if not text.strip() or residue:
            problem = (
                f"left {', '.join(sorted(residue))}" if residue else "gave empty output"
            )
            warnings += (f"{rendering.transform} {problem}; next fallback used",)
            continue
        return Rendering(
            text,
            rendering.transform,
            rendering.confidence,
            warnings + rendering.warnings,
            rendering.engine,
        )
    return _native(
        run,
        warnings + (f"no romanization available for {run.script}; native form kept",),
    )


def _native(run: _Run, warnings: tuple[str, ...]) -> Rendering:
    return Rendering(run.text, IDENTITY, "low", warnings)


def _base_script(group: str) -> str:
    return {"Jpan": "Hani", "Kore": "Hang"}.get(group, group)


def _runs(text: str, ctx: Context) -> list[_Run]:
    """Split into runs of one script group. Digits, punctuation, spaces and
    combining marks join the run they follow (or the first run)."""
    has_kana = any(script_of(c) in ("Hira", "Kana") for c in text)
    has_hangul = any(script_of(c) == "Hang" for c in text)

    def group(script: str) -> str:
        if script in ("Hira", "Kana"):
            return "Jpan"
        if script == "Hang":
            return "Kore"
        if script == "Hani":
            if ctx.lang == "ja" or (ctx.lang is None and has_kana):
                return "Jpan"
            if ctx.lang == "ko" or (ctx.lang is None and has_hangul):
                return "Kore"
            return "Hani"
        return script

    runs: list[list[str]] = []  # [group, text]
    leading = ""
    for ch in text:
        script = script_of(ch)
        if is_neutral_script(script):
            if runs:
                runs[-1][1] += ch
            else:
                leading += ch
            continue
        g = group(script)
        if runs and runs[-1][0] == g:
            runs[-1][1] += ch
        else:
            runs.append([g, ch])
    if not runs:
        return [_Run(LATIN, text)]
    runs[0][1] = leading + runs[0][1]

    # Trailing spaces and punctuation of a non-Latin run belong between runs,
    # not inside the provider's input
    out: list[_Run] = []
    for g, chunk in runs:
        if g != LATIN:
            core = chunk.rstrip(" -,.;:()/")
            tail = chunk[len(core) :]
            head_core = core.lstrip(" -,.;:()/")
            head = core[: len(core) - len(head_core)]
            if head:
                out.append(_Run(LATIN, head))
            out.append(_Run(g, head_core))
            if tail:
                out.append(_Run(LATIN, tail))
        else:
            out.append(_Run(g, chunk))
    return out
