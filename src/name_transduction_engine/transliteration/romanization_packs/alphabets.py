"""Alphabetic scripts with a national or BGN/PCGN standard: Cyrillic, Greek,
Armenian, Georgian, Thaana, Ethiopic, Sinhala.

The output is the romanization used for that language's place names:
- the country's own official system where there is one (Ukrainian 2010,
  Serbian and Uzbek Latin alphabets, Mongolian MNS, Greek ELOT 743/UNGEGN);
- BGN/PCGN otherwise, without the prime marks for soft and hard signs.
"""

import re
from dataclasses import dataclass
from typing import Callable

from ..romanization import Confidence
from . import icu_util
from .base import Context, Rendering
from .text import TableTransliterator, strip_combining


@dataclass(frozen=True)
class _Route:
    transform: str  # ICU transform ID, or "nte:<name>" for a table below
    system: str  # the standard, for warnings and the transform label
    cased: bool = True  # False: ICU emits lowercase for a caseless script


# Soft/hard sign primes and the BGN separator dot are dropped in display
_BGN_MARKS = str.maketrans("", "", "ʹʺ·")


def _clean_bgn(text: str) -> str:
    return text.translate(_BGN_MARKS)


# --------------------------------------------------------------------------- #
# Cyrillic
# --------------------------------------------------------------------------- #

# Ukrainian national system (Cabinet of Ministers resolution 55, 2010), also
# the UN and BGN/PCGN standard since 2012: Kyiv, Zaporizhzhia, Lviv
_UKRAINIAN = TableTransliterator(
    {
        "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e",
        "є": "ie", "ж": "zh", "з": "z", "зг": "zgh", "и": "y", "і": "i",
        "ї": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
        "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh",
        "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ь": "", "ю": "iu",
        "я": "ia", "'": "", "’": "", "ʼ": "",
    },
    initial={"є": "ye", "ї": "yi", "й": "y", "ю": "yu", "я": "ya"},
)  # fmt: skip

# Tajik has no ICU transform; common Latin practice for Tajik names
_TAJIK = TableTransliterator(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "ғ": "gh", "д": "d", "е": "e",
        "ё": "yo", "ж": "zh", "з": "z", "и": "i", "ӣ": "i", "й": "y", "к": "k",
        "қ": "q", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
        "с": "s", "т": "t", "у": "u", "ӯ": "ū", "ф": "f", "х": "kh", "ҳ": "h",
        "ч": "ch", "ҷ": "j", "ш": "sh", "ъ": "", "э": "e", "ю": "yu", "я": "ya",
    },
    initial={"е": "ye"},
)  # fmt: skip

_TABLES: dict[str, Callable[[str], str]] = {
    "nte:uk-national-2010": _UKRAINIAN,
    "nte:tg-latin": _TAJIK,
}

_CYRILLIC: dict[str, _Route] = {
    "ru": _Route("ru-ru_Latn/BGN", "BGN/PCGN"),
    "uk": _Route("nte:uk-national-2010", "Ukrainian national 2010"),
    "be": _Route("be-be_Latn/BGN", "BGN/PCGN"),
    "bg": _Route("bg-bg_Latn/BGN", "BGN/PCGN (Bulgarian 2009)"),
    "sr": _Route("sr-sr_Latn/BGN", "Serbian Latin alphabet"),
    "mk": _Route("mk-mk_Latn/BGN", "BGN/PCGN"),
    "kk": _Route("kk-kk_Latn/BGN", "BGN/PCGN"),
    "ky": _Route("ky-ky_Latn/BGN", "BGN/PCGN"),
    "mn": _Route("mn-mn_Latn/MNS", "Mongolian national standard MNS 5217"),
    "tk": _Route("tk_Cyrl-tk/BGN", "Turkmen Latin alphabet"),
    "uz": _Route("uz_Cyrl-uz_Latn", "Uzbek Latin alphabet"),
    "az": _Route("az_Cyrl-az/BGN", "Azerbaijani Latin alphabet"),
    "tg": _Route("nte:tg-latin", "Tajik Latin practice"),
}
# Other Cyrillic languages: Russian rules, then generic transliteration for
# the letters Russian lacks (Tatar ә, Bashkir ҡ, Ossetian æ, ...)
_CYRILLIC_DEFAULT = _Route("ru-ru_Latn/BGN; Cyrillic-Latin", "BGN/PCGN (Russian)")

# --------------------------------------------------------------------------- #
# Other alphabets
# --------------------------------------------------------------------------- #

_OTHER: dict[str, dict[str | None, _Route]] = {
    "Grek": {None: _Route("Greek-Latin/UNGEGN", "ELOT 743 / UNGEGN")},
    "Armn": {None: _Route("hy-hy_Latn/BGN", "BGN/PCGN")},
    "Geor": {None: _Route("ka-ka_Latn/BGN", "Georgian national 2002", cased=False)},
    "Thaa": {None: _Route("dv-dv_Latn/BGN", "BGN/PCGN", cased=False)},
    "Ethi": {None: _Route("am-am_Latn/BGN", "BGN/PCGN (Amharic)", cased=False)},
    "Sinh": {None: _Route("si-si_Latn", "ISO 15919 (Sinhala)", cased=False)},
    "Mong": {},  # traditional Mongolian: no ICU transform, falls through
}
_OWN_LANGUAGE = {
    "Grek": {"el"},
    "Armn": {"hy"},
    "Geor": {"ka"},
    "Thaa": {"dv"},
    "Ethi": {"am"},
    "Sinh": {"si"},
}

# ELOT 743 marks η and ω with a macron below and keeps the stress accent; road
# signs and maps use neither
_GREEK_MARKS = frozenset({"̱", "́"})


def romanize_cyrillic(text: str, ctx: Context) -> Rendering | None:
    lang = ctx.lang
    warnings: tuple[str, ...] = ()
    confidence: Confidence = "high"
    if lang is None:
        lang = _guess_cyrillic_language(text)
        if lang is not None:
            confidence = "normal"
            warnings = (f"language unknown; {lang!r} guessed from its letters",)
    route = _CYRILLIC.get(lang or "")
    if route is None:
        route = _CYRILLIC_DEFAULT
        confidence = "normal"
        warnings = (
            (f"no romanization standard for {ctx.lang!r}; Russian rules applied",)
            if ctx.lang
            else ("language unknown; Russian rules assumed",)
        )
    out = _run(route, text)
    return Rendering(
        _clean_bgn(out),
        _label(route),
        confidence,
        warnings,
        engine=None if route.transform.startswith("nte:") else icu_util.ICU_ENGINE,
    )


# Letters found in one Cyrillic language only (or nearly), checked in order
_CYRILLIC_MARKERS: tuple[tuple[str, str], ...] = (
    ("ґєї", "uk"),
    ("ҷӣӯ", "tg"),
    ("ҳ", "uz"),
    ("әұһқғ", "kk"),
    ("ңөү", "ky"),
    ("ѓќѕ", "mk"),
    ("ђћџљњј", "sr"),
    ("ў", "be"),
    ("і", "uk"),  # also Belarusian and Kazakh, which the rules above catch
)


def _guess_cyrillic_language(text: str) -> str | None:
    """A likely language from its distinctive letters; None for plain
    Russian-looking text (the default route is Russian anyway)"""
    lower = set(text.lower())
    for letters, lang in _CYRILLIC_MARKERS:
        if lower & set(letters):
            return lang
    return None


def romanize_alphabet(script: str, text: str, ctx: Context) -> Rendering | None:
    routes = _OTHER.get(script)
    if not routes:
        return None
    route = routes.get(ctx.lang) or routes.get(None)
    if route is None or not icu_util.has_transform(route.transform):
        return None

    confidence: Confidence = "high"
    warnings: tuple[str, ...] = ()
    own = _OWN_LANGUAGE.get(script, set())
    if ctx.lang not in own:
        confidence = "normal"
        warnings = (
            (f"{route.system} rules applied to {ctx.lang!r}",)
            if ctx.lang
            else (f"language unknown; {route.system} rules assumed",)
        )

    out = _run(route, text)
    if script == "Grek":
        out = strip_combining(out, _GREEK_MARKS)
    elif script == "Ethi":
        out = _ethiopic_cleanup(out)
        confidence = "normal" if confidence == "high" else confidence
    return Rendering(
        _clean_bgn(out),
        _label(route),
        confidence,
        warnings,
        engine=icu_util.ICU_ENGINE,
        cased=route.cased,
    )


def _run(route: _Route, text: str) -> str:
    if route.transform.startswith("nte:"):
        return _TABLES[route.transform](text)
    return icu_util.apply(route.transform, text)


def _label(route: _Route) -> str:
    if route.transform.startswith("nte:"):
        return route.transform
    return f"icu:{route.transform}"


# BGN writes the sixth-order (vowelless or schwa) syllable with a plain "i"
# and the third order with "ī". A sixth-order "i" at the end of a word is
# silent (ጎንደር "gonideri" is Gonder), and so is one between a vowel-final
# syllable's consonant and the next consonant+vowel
_ETHI_FINAL_I = re.compile(r"(?<=[^\Waeiouāēīōū])i\b")
_ETHI_MEDIAL_I = re.compile(
    r"(?<=[aeouāēīōū][^\Waeiouāēīōū'])i(?=[^\Waeiouāēīōū'][aeiouāēīōū])"
)


def _ethiopic_cleanup(text: str) -> str:
    text = _ETHI_FINAL_I.sub("", text)
    return _ETHI_MEDIAL_I.sub("", text)
