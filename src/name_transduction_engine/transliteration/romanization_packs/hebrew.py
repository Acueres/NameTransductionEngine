"""Hebrew script: Hebrew (abjad) and Yiddish (written with full vowels).

Hebrew follows the simplified Academy style of Israeli road signs: ח h,
כ/ך kh, צ ts, ע and א as an apostrophe between vowels, no underdots. With
niqqud the vowels are exact. Without them, as for Arabic, long vowels come
from the vowel letters (ו o/u, י i, final ה a) and short vowels are restored
by syllable rules, marked low confidence. The b/v, k/kh, p/f alternation
follows the restored vowels: hard at the start of a word and after a
consonant, soft after a vowel.

Yiddish uses YIVO romanization.
"""

from ..romanization import Confidence
from .base import Context, Rendering
from .text import capitalize_first

# Consonants; "b|v" alternates by position
_HEBREW = {
    "ב": "b|v", "ג": "g", "ד": "d", "ה": "h", "ז": "z", "ח": "h", "ט": "t",
    "כ": "k|kh", "ך": "kh", "ל": "l", "מ": "m", "ם": "m", "נ": "n", "ן": "n",
    "ס": "s", "פ": "p|f", "ף": "f", "צ": "ts", "ץ": "ts", "ק": "k", "ר": "r",
    "ש": "sh", "ת": "t",
}  # fmt: skip
_GERESH = {"ג": "j", "ז": "zh", "צ": "ch", "ץ": "ch", "ת": "t"}
_GERESH_CHARS = frozenset("׳'’")

_NIQQUD = {
    "ַ": "a", "ָ": "a", "ֶ": "e", "ֵ": "e", "ִ": "i",
    "ֹ": "o", "ֺ": "o", "ֻ": "u", "ֲ": "a", "ֱ": "e",
    "ֳ": "o", "ׇ": "o",
}  # fmt: skip
_SHVA, _DAGESH, _SHIN_DOT, _SIN_DOT, _RAFE = "ְ", "ּ", "ׁ", "ׂ", "ֿ"
_MARKS = frozenset([*_NIQQUD, _SHVA, _DAGESH, _SHIN_DOT, _SIN_DOT, _RAFE])


class _Unit:
    __slots__ = ("c", "vowel", "shva", "hard", "carrier")

    def __init__(self, c: str, vowel: str | None = None, carrier: bool = False):
        self.c = c
        self.vowel = vowel  # written (mater or niqqud) vowel
        self.shva = False
        self.hard = False  # dagesh
        self.carrier = carrier  # א/ע: no sound of its own


def romanize_hebrew_script(text: str, ctx: Context) -> Rendering | None:
    if ctx.lang in ("yi", "ydd"):
        return Rendering(_yiddish(text), "nte:yiddish-yivo", "normal")
    if ctx.lang not in (None, "he", "iw"):
        return None  # Ladino, Judeo-Arabic...: no rules here

    words: list[str] = []
    voweled_all = True
    for word in text.split(" "):
        roman, voweled = _romanize_word(word)
        voweled_all &= voweled
        words.append(roman)
    confidence: Confidence = "high" if voweled_all else "low"
    warnings: tuple[str, ...] = (
        () if voweled_all else ("short vowels restored by rule; not attested",)
    )
    if ctx.lang is None:
        warnings += ("language unknown; Hebrew assumed",)
    return Rendering(" ".join(words), "nte:hebrew-romanization", confidence, warnings)


def _romanize_word(word: str) -> tuple[str, bool]:
    # Hyphenated compounds and maqaf: each part is its own word
    for sep in ("-", "־"):
        if sep in word:
            parts = [_romanize_word(p) for p in word.split(sep)]
            return "-".join(p for p, _ in parts), all(v for _, v in parts)
    if not word:
        return word, True
    units = _parse(word)
    if units is None:
        return word, True
    voweled = any(u.vowel is not None and u.c for u in units) and sum(
        1 for ch in word if ch in _NIQQUD or ch == _SHVA
    ) >= max(1, len([u for u in units if u.c]) // 2)
    return capitalize_first(_render(units, voweled)), voweled


def _parse(word: str) -> list[_Unit] | None:
    units: list[_Unit] = []
    chars = list(word)
    has_niqqud = any(ch in _NIQQUD for ch in chars)
    for i, ch in enumerate(chars):
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        last = units[-1] if units else None
        if ch in _MARKS:
            if last is None:
                continue
            if ch in _NIQQUD:
                last.vowel = _NIQQUD[ch]
            elif ch == _SHVA:
                last.shva = True
            elif ch == _DAGESH:
                if last.c == "v" and last.vowel is None:  # shuruk וּ
                    last.c, last.vowel = "", "u"
                    if len(units) > 1:
                        units.pop()
                        units[-1].vowel = "u"
                else:
                    last.hard = True
            elif ch == _SIN_DOT:
                last.c = "s"
            continue
        if ch in _GERESH_CHARS:
            if last is not None and last.c in ("g", "z", "ts", "t"):
                last.c = {"g": "j", "z": "zh", "ts": "ch", "t": "t"}[last.c]
            continue
        if ch in ("א", "ע"):
            if not nxt and last is not None and last.c:
                last.vowel = last.vowel or "a"  # final: שבע Sheva
            else:
                units.append(_Unit("'" if units else "", carrier=True))
            continue
        if ch == "ו":
            if nxt == "ו":
                units.append(_Unit("v"))
                chars[i + 1] = ""
            elif nxt == "ֹ" or (
                last is not None and not has_niqqud and _vowel_slot(last)
            ):
                if last is None:
                    units.append(_Unit("v"))
                else:
                    last.vowel = "o"
            else:
                units.append(_Unit("v"))
            continue
        if ch == "י":
            bare = nxt not in _MARKS  # this yod carries no vowel of its own
            if nxt == "י":
                units.append(_Unit("y"))
                chars[i + 1] = ""
            elif last is None:
                units.append(_Unit("y"))
            elif has_niqqud and bare and last.vowel in ("i", "e", "a"):
                # Vowel letter after hiriq, tsere, patah: i, ei, ai
                last.vowel = {"i": "i", "e": "ei", "a": "ai"}[last.vowel]
            elif not has_niqqud and nxt in ("ה", "א") and i + 2 == len(chars):
                units.append(_Unit("y"))  # final -יה is -ya: נתניה Netanya
            elif not has_niqqud and _vowel_slot(last):
                last.vowel = "ei" if last.carrier and not last.c else "i"
            else:
                units.append(_Unit("y"))
            continue
        if ch == "ה" and not nxt and last is not None and last.c:
            last.vowel = last.vowel or "a"
            continue
        if ch == "":
            continue
        if ch in _HEBREW:
            units.append(_Unit(_HEBREW[ch]))
            continue
        if ch.isalpha():
            return None
        units.append(_Unit(ch, vowel=""))
    return units


def _vowel_slot(last: _Unit) -> bool:
    """A vowel letter after this unit writes its vowel (rather than a glide)"""
    return last.vowel is None and not last.shva


def _render(units: list[_Unit], voweled: bool) -> str:
    out = ""
    after_vowel = False
    for k, u in enumerate(units):
        nxt = units[k + 1] if k + 1 < len(units) else None
        after = units[k + 2] if k + 2 < len(units) else None
        c = u.c
        if "|" in c:
            hard, soft = c.split("|")
            c = hard if (u.hard or not after_vowel) else soft
        if c == "'" and not after_vowel:
            c = ""
        out += c
        if u.vowel is not None:
            out += u.vowel
            after_vowel = bool(u.vowel)
            continue
        if u.shva:
            # Vocal shva at the start of a word, silent elsewhere
            if k == 0 or (k == 1 and not units[0].c):
                out += "e"
                after_vowel = True
            else:
                after_vowel = False
            continue
        if voweled or nxt is None:
            if u.carrier and not u.c and nxt is not None:
                out += "a"
                after_vowel = True
            else:
                after_vowel = False
            continue
        next_can_start = nxt.vowel is not None or after is not None
        if after_vowel and next_can_start and k > 0 and not u.carrier:
            after_vowel = False
            continue
        out += "a"
        after_vowel = True
    return out


# --------------------------------------------------------------------------- #
# Yiddish (YIVO)
# --------------------------------------------------------------------------- #

_YIDDISH = [
    ("ײַ", "ay"), ("ײ", "ey"), ("יי", "ey"), ("ױ", "oy"), ("וי", "oy"), ("װ", "v"),
    ("וו", "v"), ("אַ", "a"), ("אָ", "o"), ("וּ", "u"), ("יִ", "i"), ("בֿ", "v"),
    ("כּ", "k"), ("פּ", "p"), ("פֿ", "f"), ("תּ", "t"), ("שׂ", "s"), ("דזש", "dzh"),
    ("זש", "zh"), ("טש", "tsh"), ("א", ""), ("ב", "b"), ("ג", "g"), ("ד", "d"),
    ("ה", "h"), ("ו", "u"), ("ז", "z"), ("ח", "kh"), ("ט", "t"), ("י", "i"),
    ("כ", "kh"), ("ך", "kh"), ("ל", "l"), ("מ", "m"), ("ם", "m"), ("נ", "n"),
    ("ן", "n"), ("ס", "s"), ("ע", "e"), ("פ", "f"), ("ף", "f"), ("צ", "ts"),
    ("ץ", "ts"), ("ק", "k"), ("ר", "r"), ("ש", "sh"), ("ת", "s"),
]  # fmt: skip


def _yiddish(text: str) -> str:
    out = ""
    i = 0
    while i < len(text):
        # Unpointed alef before a consonant stands for a/o; before a vowel
        # letter it is the silent alef
        if text[i] == "א" and text[i + 1 : i + 2] not in (
            "",
            " ",
            "ו",
            "י",
            "ע",
            "\u05b7",
            "\u05b8",
            "ײ",
            "ױ",
            "װ",
        ):
            out += "a"
            i += 1
            continue
        for src, dst in _YIDDISH:
            if text.startswith(src, i):
                # Consonantal yud before a vowel letter
                if (
                    src == "י"
                    and text[i + 1 : i + 2] in ("א", "ע", "ו")
                    or (src == "י" and (i == 0 or text[i - 1] == " "))
                ):
                    dst = "y"
                out += dst
                i += len(src)
                break
        else:
            if text[i] not in _MARKS:
                out += text[i]
            i += 1
    return " ".join(capitalize_first(w) for w in out.split(" "))
