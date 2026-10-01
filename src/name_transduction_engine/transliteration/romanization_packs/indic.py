"""Brahmic scripts of India: Hunterian-style romanization, as on Indian maps
and signs (Rajasthan, Kanpur, Lakhnau, Chennai, Thiruvananthapuram).

The nine main Indic blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Odia,
Tamil, Telugu, Kannada, Malayalam) share one layout inherited from ISCII, so a
letter's offset within its block identifies it in every script. Words are
parsed into aksharas (consonant cluster + vowel), then:
- Indo-Aryan languages drop the inherent "a" where it is silent: always at
  the end of a word (राजस्थान Rajasthan), and in Hindi-type languages also in
  the middle by the standard VC_CV rule (पटना Patna, लखनऊ Lakhnau), with
  common place-name endings (-pur, -abad, -nagar, -garh...) treated as
  separate words (भरतपुर Bharatpur, अहमदाबाद Ahmadabad);
- Dravidian languages keep every vowel;
- Tamil voices stops between vowels and after nasals (மதுரை Madurai) and
  writes an initial dental t as "th" (தஞ்சாவூர் Thanjavur); Malayalam writes
  every dental t as "th".
"""

import re
from dataclasses import dataclass, field

from .base import Context, Rendering
from .text import capitalize_first

_BLOCKS = {
    "Deva": 0x0900, "Beng": 0x0980, "Guru": 0x0A00, "Gujr": 0x0A80,
    "Orya": 0x0B00, "Taml": 0x0B80, "Telu": 0x0C00, "Knda": 0x0C80,
    "Mlym": 0x0D00,
}  # fmt: skip

# Language assumed when the tag gives none
_SCRIPT_LANG = {
    "Deva": "hi", "Beng": "bn", "Guru": "pa", "Gujr": "gu", "Orya": "or",
    "Taml": "ta", "Telu": "te", "Knda": "kn", "Mlym": "ml",
}  # fmt: skip

# Languages whose inherent vowel is silent at the end of a word...
_FINAL_SCHWA_DELETION = frozenset(
    {
        "hi", "mr", "ne", "pa", "gu", "bn", "as", "or", "mai", "bho", "awa",
        "mag", "raj", "hne", "doi", "kok", "gom", "sd", "ks", "mwr", "bgc",
    }
)  # fmt: skip
# ...and also in the middle of a word
_MEDIAL_SCHWA_DELETION = frozenset(
    {"hi", "mr", "ne", "pa", "gu", "bn", "as", "mai", "bho", "awa", "mag", "raj", "hne", "doi", "mwr", "bgc"}
)  # fmt: skip
# Languages writing a word-final anusvara as "m"
_FINAL_ANUSVARA_M = frozenset({"ml", "te", "kn", "ta", "sa"})

# Letters by offset, as internal phoneme symbols (rendered at the end)
_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ṅ",
    0x1A: "c", 0x1B: "ch", 0x1C: "j", 0x1D: "jh", 0x1E: "ñ",
    0x1F: "ṭ", 0x20: "ṭh", 0x21: "ḍ", 0x22: "ḍh", 0x23: "ṇ",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "ṉ",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "ṟ", 0x32: "l", 0x33: "ḷ", 0x34: "ḻ",
    0x35: "v", 0x36: "ś", 0x37: "ṣ", 0x38: "s", 0x39: "h",
    # Letters with nukta that stay precomposed in NFC (Bengali, Gurmukhi, Odia)
    0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "ṛ", 0x5D: "ṛh", 0x5E: "f",
    0x5F: "ẏ", 0x71: "w",
}  # fmt: skip
_NUKTA = {"k": "q", "g": "gh", "j": "z", "ḍ": "ṛ", "ḍh": "ṛh", "ph": "f", "y": "ẏ"}

_INDEPENDENT_VOWELS = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}  # fmt: skip
_VOWEL_SIGNS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
    0x4A: "o", 0x4B: "o", 0x4C: "au", 0x57: "au", 0x62: "li", 0x63: "li",
}  # fmt: skip
_VIRAMA, _NUKTA_SIGN = 0x4D, 0x3C
_NASALS = (0x01, 0x02)  # candrabindu, anusvara
_VISARGA = 0x03
_GURMUKHI_TIPPI, _GURMUKHI_ADDAK = "ੰ", "ੱ"

# Letters outside the shared layout: Bengali khanda ta, Assamese ra/wa,
# Malayalam chillus (vowelless finals)
_EXTRA = {
    "ৎ": "t", "ৰ": "r", "ৱ": "w", "ൺ": "ṇ", "ൻ": "n",
    "ർ": "ṟ", "ൽ": "l", "ൾ": "ḷ", "ൿ": "k", "ൔ": "m",
    "ൕ": "y", "ൖ": "ḻ",
}  # fmt: skip

# Hunterian rendering of the internal symbols
_RENDER = {
    "ṅ": "n", "c": "ch", "ch": "chh", "ñ": "n", "ṭ": "t", "ṭh": "th",
    "ḍ": "d", "ḍh": "dh", "ṇ": "n", "ṉ": "n", "ṟ": "r", "ḷ": "l", "ḻ": "zh",
    "ś": "sh", "ṣ": "sh", "ṛ": "r", "ṛh": "rh", "ẏ": "y",
}  # fmt: skip

# Place-name endings treated as separate words for schwa deletion, as
# (consonants, vowel) per akshara, None standing for the inherent vowel
_SUFFIXES: tuple[tuple[tuple[str, str | None], ...], ...] = (
    (("p", "u"), ("r", "a")),  # -pura
    (("p", "u"), ("r", None)),  # -pur
    (("b", "a"), ("d", None)),  # -abad, -bad
    (("n", None), ("g", None), ("r", None)),  # -nagar
    (("g", None), ("ṛh", None)),  # -garh
    (("k", "o"), ("ṭ", None)),  # -kot
    (("gh", "a"), ("ṭ", None)),  # -ghat
    (("s", None), ("r", None)),  # -sar
    (("g", "a"), ("v", None)),  # -gaon
)

_VOWEL_CHARS = "aeiou"


@dataclass
class _Akshara:
    consonants: list[str] = field(default_factory=list)
    vowel: str | None = None  # None: inherent vowel; "" after a virama
    nasal: bool = False
    visarga: bool = False
    silent: bool = False  # inherent vowel not pronounced

    @property
    def has_inherent(self) -> bool:
        return self.vowel is None and bool(self.consonants)


def romanize_indic(script: str, text: str, ctx: Context) -> Rendering | None:
    base = _BLOCKS.get(script)
    if base is None:
        return None
    lang = ctx.lang or _SCRIPT_LANG[script]
    in_block = re.compile(f"[\\u{base:04x}-\\u{base + 0x7F:04x}\\u200c\\u200d]+")
    out = in_block.sub(lambda m: _romanize_word(m.group(), base, script, lang), text)

    warnings: tuple[str, ...] = ()
    if lang in _MEDIAL_SCHWA_DELETION:
        warnings = (
            "silent inherent vowels dropped by rule; established spellings may differ",
        )
    if ctx.lang is None:
        warnings += (f"language unknown; {lang!r} conventions assumed",)
    return Rendering(out, "nte:hunterian", "normal", warnings)


def _romanize_word(word: str, base: int, script: str, lang: str) -> str:
    aksharas = _parse(word, base, script)
    if not aksharas:
        return ""
    if lang in _FINAL_SCHWA_DELETION:
        _delete_schwas(aksharas, medial=lang in _MEDIAL_SCHWA_DELETION)
    return capitalize_first(_render(aksharas, script, lang))


def _parse(word: str, base: int, script: str) -> list[_Akshara]:
    aks: list[_Akshara] = []
    joining = False  # the previous consonant took a virama: the next one joins it
    geminate = False
    for idx, ch in enumerate(word):
        off = ord(ch) - base
        prev_off = ord(word[idx - 1]) - base if idx > 0 else -1
        if ch in _EXTRA:
            aks.append(_Akshara([_EXTRA[ch]], vowel=""))
            joining = False
        elif ch == _GURMUKHI_ADDAK:
            geminate = True
        elif ch == _GURMUKHI_TIPPI and aks:
            aks[-1].nasal = True
        elif off in _CONSONANTS:
            letter = _CONSONANTS[off]
            if script in ("Beng", "Orya") and letter == "y":
                # য/ଯ is "j" as a syllable onset, "y" as the second member of
                # a cluster (the ya-phala)
                letter = "y" if joining else "j"
            if geminate:
                aks.append(_Akshara([letter], vowel=""))
                geminate = False
                joining = True
            if joining and aks:
                aks[-1].consonants.append(letter)
                aks[-1].vowel = None
            else:
                aks.append(_Akshara([letter]))
            joining = False
        elif off == _NUKTA_SIGN and aks and aks[-1].consonants:
            last = aks[-1].consonants[-1]
            if script in ("Beng", "Orya") and last == "j" and prev_off == 0x2F:
                aks[-1].consonants[-1] = "ẏ"  # য় is y, not a nukta j
            else:
                aks[-1].consonants[-1] = _NUKTA.get(last, last)
        elif off == _VIRAMA and aks:
            aks[-1].vowel = ""
            joining = True
        elif off in _VOWEL_SIGNS and aks:
            aks[-1].vowel = _VOWEL_SIGNS[off]
            joining = False
        elif off in _INDEPENDENT_VOWELS:
            aks.append(_Akshara([], _INDEPENDENT_VOWELS[off]))
            joining = False
        elif off in _NASALS and aks:
            aks[-1].nasal = True
        elif off == _VISARGA and aks:
            aks[-1].visarga = True
        # ZWJ/ZWNJ, avagraha, digits and other signs carry no sound here
    return aks


def _delete_schwas(aks: list[_Akshara], medial: bool) -> None:
    cut = _suffix_start(aks)
    parts = [aks[:cut], aks[cut:]] if cut else [aks]
    for part in parts:
        _delete_in_part(part, medial)


def _suffix_start(aks: list[_Akshara]) -> int:
    for suffix in _SUFFIXES:
        n = len(suffix)
        if len(aks) <= n:
            continue
        if all(
            "".join(a.consonants) == c and a.vowel == v and not a.nasal
            for a, (c, v) in zip(aks[-n:], suffix, strict=True)
        ):
            return len(aks) - n
    return 0


def _delete_in_part(part: list[_Akshara], medial: bool) -> None:
    if len(part) < 2:
        return
    last = part[-1]
    # Final: silent after a single consonant or a geminate (कच्छ Kachchh),
    # kept after a cluster ending in r, y or v (इंद्र Indra, मित्र Mitra)
    if (
        last.has_inherent
        and not last.nasal
        and (len(last.consonants) == 1 or last.consonants[-1] not in ("r", "y", "v"))
    ):
        last.silent = True
    if not medial:
        return
    # Medial VC_CV: the inherent vowel of a lone consonant, between a spoken
    # vowel and a lone consonant with a spoken vowel, is silent. Left to
    # right, never two in a row
    for i in range(1, len(part) - 1):
        ak, prev, nxt = part[i], part[i - 1], part[i + 1]
        if not ak.has_inherent or len(ak.consonants) != 1 or ak.nasal or ak.visarga:
            continue
        if prev.silent or prev.vowel == "" or prev.nasal:
            continue
        if len(nxt.consonants) != 1 or nxt.silent or nxt.vowel == "":
            continue
        ak.silent = True


def _render(aks: list[_Akshara], script: str, lang: str) -> str:
    out = ""
    for i, ak in enumerate(aks):
        for j, cons in enumerate(ak.consonants):
            out += _consonant(cons, out, ak, i, j, script)
        if ak.vowel is None:
            out += "" if ak.silent else "a"
        else:
            out += ak.vowel
        if ak.nasal:
            nxt = (
                aks[i + 1].consonants[0]
                if i + 1 < len(aks) and aks[i + 1].consonants
                else ""
            )
            if nxt in ("m", "n", "ṇ"):
                pass  # nasal sign before a nasal consonant doubles it: ਅੰਮ੍ਰਿਤਸਰ Amritsar
            elif not nxt and lang in _FINAL_ANUSVARA_M:
                out += "m"
            elif nxt[:1] in ("p", "b", "m"):
                out += "m"
            elif nxt == "h":
                out += "ng"  # सिंह Singh
            else:
                out += "n"
        if ak.visarga:
            out += "h"
    return out


_DRAVIDIAN = frozenset({"Taml", "Telu", "Knda", "Mlym"})
# Between vowels Tamil voices k, ṭ, t and softens c; p is written p there
# by convention (காஞ்சிபுரம் Kanjipuram)
_TAMIL_VOICED = {"k": "g", "c": "s", "ṭ": "d", "t": "d"}
_TAMIL_AFTER_NASAL = {"k": "g", "c": "j", "ṭ": "d", "t": "d", "p": "b"}


def _consonant(
    cons: str, before: str, ak: _Akshara, i: int, j: int, script: str
) -> str:
    prev_cons = ak.consonants[j - 1] if j > 0 else None
    next_cons = ak.consonants[j + 1] if j + 1 < len(ak.consonants) else None
    if script == "Taml":
        # Doubled stops stay voiceless: க்க kk, த்த tt; ச்ச is written "ch"
        if prev_cons == cons:
            return "" if cons == "c" else _RENDER.get(cons, cons)
        if next_cons == cons:
            return _RENDER.get(cons, cons)
        if i == 0 and j == 0:
            return "th" if cons == "t" else _RENDER.get(cons, cons)
        if cons in _TAMIL_VOICED and prev_cons is None:
            # Single stop after a vowel (the previous akshara's)
            if before[-1:] in _VOWEL_CHARS:
                return _TAMIL_VOICED[cons]
        if cons in _TAMIL_AFTER_NASAL and prev_cons in ("ṅ", "ñ", "ṇ", "n", "m", "ṉ"):
            return _TAMIL_AFTER_NASAL[cons]
        return _RENDER.get(cons, cons)
    if script == "Mlym" and cons == "t":
        return "th"
    if script == "Mlym" and cons == "ś" and (prev_cons == "ś" or next_cons == "ś"):
        return "s"  # തൃശ്ശൂർ Thrissur
    if cons == "c" and prev_cons == "c":
        return "ch"  # कच्छ Kachchh
    if cons == "j" and next_cons == "ñ" and script not in _DRAVIDIAN:
        return "gy"  # ज्ञान gyan in the Indo-Aryan languages
    if cons == "ñ" and prev_cons == "j" and script not in _DRAVIDIAN:
        return ""
    return _RENDER.get(cons, cons)
