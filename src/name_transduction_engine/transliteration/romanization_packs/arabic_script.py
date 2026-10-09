"""Arabic script, rule-based: Urdu and Pashto (abjads), Uyghur and Sorani
Kurdish (fully voweled alphabets).

Arabic and Persian have their own learned readers (arabic.py, persian.py);
these rules are their fallback when the models are missing, and the reader of
any untagged Arabic-script name they do not take.

Abjads leave short vowels unwritten, so there are two modes per word:
- voweled text (harakat on most consonants): exact, from the marks;
- plain text: long vowels come from the letters that write them (ا و ي),
  short vowels are restored by syllable rules. A consonant that starts a
  syllable gets a vowel; before a consonant+vowel it closes the syllable
  instead; after a long vowel it always starts a new one. The restored vowel
  is "a", or "i"/"u" before y/w + vowel (al-Riyāḍ). Right for some names
  (Baghdād, Tabrīz, Mashhad, Karāchī), wrong for many, so the result is
  marked low confidence.

Output style: simplified ALA-LC, long vowels with macrons, ʿ for ʿayn and ʾ
for medial hamza, no dots under emphatic consonants, article "al-" without
assimilation, tāʾ marbūṭa as "a" ("at" in a construct with a generic noun:
Madīnat al-Kuwayt).
"""

from dataclasses import dataclass

from ..romanization import Confidence
from .base import Context, Rendering
from .text import capitalize_first

_ABJAD_LANGS = frozenset({"ar", "fa", "ur", "ps"})

# Consonants common to the abjad languages, then per-language overrides
_CONSONANTS = {
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh", "د": "d",
    "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d",
    "ط": "t", "ظ": "z", "ع": "ʿ", "غ": "gh", "ف": "f", "ق": "q", "ك": "k",
    "ک": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "ہ": "h", "ۀ": "h",
    "ء": "ʾ", "ؤ": "ʾ", "ئ": "ʾ", "پ": "p", "چ": "ch", "ژ": "zh", "گ": "g",
    "ڤ": "v", "ٹ": "t", "ڈ": "d", "ڑ": "r", "ں": "n", "ټ": "t", "ډ": "d",
    "ړ": "r", "ږ": "zh", "ښ": "sh", "ځ": "dz", "څ": "ts", "ڼ": "n", "ګ": "g",
}  # fmt: skip
_PERSIAN_CONSONANTS = {"ث": "s", "ذ": "z", "ض": "z", "ظ": "z", "ع": "ʼ", "و": "v"}
_LANG_CONSONANTS = {
    "fa": _PERSIAN_CONSONANTS,
    "ur": _PERSIAN_CONSONANTS,
    "ps": {"ث": "s", "ذ": "z", "ض": "z", "ظ": "z", "ع": "ʼ"},
}

_ALEFS = {"ا": None, "ٱ": None, "أ": "a", "إ": "i", "آ": "ā"}
_WAW = "و"
_YEHS = frozenset("يیې")
_ALEF_MAQSURA = "ى"
_TA_MARBUTA = frozenset("ةۃ")
_FINAL_HEH = frozenset("هہ")
_URDU_ASPIRATE = "ھ"
_URDU_YEH_BARREE = frozenset("ےۓ")
_PASHTO_AI = "ۍ"
_SILENT = frozenset("ـ‌‍")  # tatweel, ZWNJ, ZWJ

_FATHA, _DAMMA, _KASRA = "َ", "ُ", "ِ"
_TANWIN = {"ً": "an", "ٌ": "un", "ٍ": "in"}
_SUKUN, _SHADDA, _SUPERSCRIPT_ALEF = "ْ", "ّ", "ٰ"
_SHORT = {_FATHA: "a", _DAMMA: "u", _KASRA: "i"}
_HARAKAT = frozenset([*_SHORT, *_TANWIN, _SUKUN, _SHADDA, _SUPERSCRIPT_ALEF])

_LETTER_VOWELS = frozenset(["ا", "ٱ", "آ", _WAW, *_YEHS, _ALEF_MAQSURA, *_TA_MARBUTA])

# Word-initial bare alif, whose vowel is not written
_INITIAL_VOWEL = {"ar": "i", "fa": "e", "ur": "i", "ps": "a"}
# Word-final silent heh after a consonant (Persian خانه khāne, Urdu کوئٹہ Kwaṭa)
_FINAL_HEH_VOWEL = {"fa": "e", "ur": "a", "ps": "a"}
_WAW_CONSONANT = {"ar": "w", "fa": "v", "ur": "v", "ps": "w"}

# Generic nouns taking the construct "at" before another word
_CONSTRUCT_NOUNS = frozenset(
    {
        "مدينة", "جزيرة", "قرية", "محافظة", "ولاية", "منطقة", "بلدة", "واحة",
        "بحيرة", "قلعة", "مملكة", "دولة", "جمهورية", "إمارة", "ناحية", "مديرية",
        "هضبة", "ساحة", "مقاطعة", "بلدية",
    }
)  # fmt: skip


@dataclass
class _Unit:
    c: str = ""  # consonant ("" for a vowel carrier)
    long: str | None = None  # ā ī ū e ai (written vowel)
    short: str | None = None  # from harakat
    sukun: bool = False
    shadda: bool = False
    marbuta: bool = False
    tanwin: str = ""


class _Unsupported(Exception):
    pass


def romanize_arabic_script(text: str, ctx: Context) -> Rendering | None:
    lang = ctx.lang
    if lang == "ug":
        return Rendering(_UYGHUR(text), "nte:uyghur-uly", "high")
    if lang == "ckb" or lang == "ku" and ctx.script == "Arab":
        return Rendering(
            _kurdish(text),
            "nte:kurdish-hawar",
            "normal",
            ("short i is not written in Sorani and is not restored",),
        )
    if lang not in _ABJAD_LANGS and lang is not None:
        return None  # Sindhi, Kashmiri, Malay Jawi...: no rules here
    lang = lang or "ar"

    words = text.split(" ")
    out: list[str] = []
    voweled_all = True
    try:
        for i, word in enumerate(words):
            nxt = words[i + 1] if i + 1 < len(words) else None
            roman, voweled = _romanize_word(
                word, lang, construct=_is_construct(word, nxt)
            )
            voweled_all &= voweled
            out.append(roman)
    except _Unsupported:
        return None

    text_out = " ".join(out)
    text_out = capitalize_first(text_out)
    confidence: Confidence = "high" if voweled_all else "low"
    warnings: tuple[str, ...] = (
        () if voweled_all else ("short vowels restored by rule; not attested",)
    )
    if ctx.lang is None:
        warnings += ("language unknown; Arabic assumed",)
    return Rendering(text_out, f"nte:{lang}-romanization", confidence, warnings)


def _is_construct(word: str, nxt: str | None) -> bool:
    return nxt is not None and _strip_harakat(word) in _CONSTRUCT_NOUNS


def _strip_harakat(word: str) -> str:
    return "".join(ch for ch in word if ch not in _HARAKAT)


def _romanize_word(word: str, lang: str, construct: bool) -> tuple[str, bool]:
    if not word:
        return word, True
    prefix = ""
    body = word
    bare = _strip_harakat(word)
    # The article (Arabic, and Arabic names in the other languages)
    if bare.startswith("ال") and len(bare) > 3:
        cut = _index_after_letters(word, 2)
        prefix, body = "al-", word[cut:]

    units = _parse(body, lang)
    if not units:
        # No letters at all (digits, punctuation): pass through
        return word, True
    consonants = sum(1 for u in units if u.c)
    marked = sum(1 for ch in body if ch in _SHORT or ch in _TANWIN or ch == _SUKUN)
    voweled = marked >= max(1, (consonants + 1) // 2)
    roman = _render(units, lang, voweled, construct)
    head = capitalize_first(roman)
    return (prefix + head if prefix else head), voweled


def _index_after_letters(word: str, n: int) -> int:
    seen = 0
    for i, ch in enumerate(word):
        if ch not in _HARAKAT:
            seen += 1
            if seen == n:
                # Skip marks on the second letter (a sukun on the lam)
                j = i + 1
                while j < len(word) and word[j] in _HARAKAT:
                    j += 1
                return j
    return len(word)


def _parse(word: str, lang: str) -> list[_Unit]:
    consonants = {**_CONSONANTS, **_LANG_CONSONANTS.get(lang, {})}
    units: list[_Unit] = []
    chars = [c for c in word if c not in _SILENT]
    for idx, ch in enumerate(chars):
        nxt = chars[idx + 1] if idx + 1 < len(chars) else ""
        last = units[-1] if units else None

        if ch in _HARAKAT:
            if last is None:
                continue
            if ch in _SHORT:
                last.short = _SHORT[ch]
            elif ch in _TANWIN:
                last.tanwin = _TANWIN[ch]
            elif ch == _SUKUN:
                last.sukun = True
            elif ch == _SHADDA:
                last.shadda = True
            elif ch == _SUPERSCRIPT_ALEF:
                last.long = "ā"
            continue

        if ch in _ALEFS:
            if last is None:
                units.append(
                    _Unit(
                        "",
                        long="ā" if ch == "آ" else None,
                        short=_ALEFS[ch] if ch != "آ" else None,
                    )
                )
                if units[-1].short is None and units[-1].long is None:
                    units[-1].short = _INITIAL_VOWEL.get(lang, "a")
            elif ch in ("أ", "إ"):
                units.append(_Unit("ʾ", short=_ALEFS[ch]))
            elif ch == "آ":
                # In Persian and Urdu a medial آ starts a new word part with
                # no glottal stop: حیدرآباد Haidarābād
                units.append(_Unit("" if lang in ("fa", "ur", "ps") else "ʾ", long="ā"))
            elif last.long is None and not last.tanwin:
                last.long = "ā"
            continue

        if ch == _WAW or ch in _YEHS:
            is_waw = ch == _WAW
            glide = _WAW_CONSONANT[lang] if is_waw else "y"
            if ch == "ې":  # Pashto e
                if last is None:
                    units.append(_Unit("y", long="e"))
                else:
                    last.long = "e"
                continue
            consonantal = (
                last is None
                or last.long is not None
                or (last.c == "" and last.short is not None)
                or nxt in _LETTER_VOWELS
                or nxt in (_FATHA, _DAMMA, _KASRA, _SHADDA)
                or last.short == "a"  # diphthong aw/ay in voweled text
                or last.sukun
            )
            if consonantal or last is None:
                units.append(_Unit(glide))
            else:
                last.long = "ū" if is_waw else "ī"
                if last.short in ("u", "i"):
                    last.short = None
            continue

        if ch == _ALEF_MAQSURA:
            if last is None:
                raise _Unsupported
            if lang in ("fa", "ur", "ps"):  # used for yeh in these languages
                last.long = "ī"
            else:
                last.long = "ā" if last.short != "i" else "ī"
            continue

        if ch in _TA_MARBUTA:
            units.append(_Unit("", marbuta=True))
            continue

        if (
            ch in _FINAL_HEH
            and not nxt
            and lang in _FINAL_HEH_VOWEL
            and last
            and last.c
            and last.long is None
            and len(units) >= 2
        ):
            last.long = _FINAL_HEH_VOWEL[lang]
            continue

        if ch == _URDU_ASPIRATE:
            if last is not None and last.c:
                last.c += "h"
            else:
                units.append(_Unit("h"))
            continue

        if ch in _URDU_YEH_BARREE:
            if last is None:
                units.append(_Unit("y", long="e"))
            else:
                last.long = "ai" if last.short == "a" else "e"
            continue

        if ch == _PASHTO_AI:
            if last is not None:
                last.long = "ai"
            continue

        if ch in consonants:
            units.append(_Unit(consonants[ch]))
            continue

        if ch.isalpha():
            raise _Unsupported  # a letter this table does not know
        # Digits, punctuation: keep as a bare unit
        units.append(_Unit(ch, long=""))
    return units


def _render(units: list[_Unit], lang: str, voweled: bool, construct: bool) -> str:
    out = ""
    after_vowel = False  # the previous output ended in a vowel
    last_nucleus_long = False
    for k, u in enumerate(units):
        nxt = units[k + 1] if k + 1 < len(units) else None
        after = units[k + 2] if k + 2 < len(units) else None

        if u.marbuta:
            if construct and nxt is None:
                out += "at"
            elif not out.endswith("a"):
                out += "a"
            after_vowel = True
            continue

        c = u.c
        if k == 0 and c == "ʼ":
            c = ""  # Persian/Urdu initial ʿayn is silent: عباس Abbās
        out += c + (c if u.shadda and c else "")
        if u.long is not None:
            if u.short and u.short != "a" and u.long in ("ā",):
                out += u.short  # rare: explicit short before a long vowel
            out += u.long
            after_vowel, last_nucleus_long = bool(u.long), True
            continue
        if u.short:
            out += u.short + u.tanwin[1:] if u.tanwin else u.short
            after_vowel, last_nucleus_long = True, False
            continue
        if u.tanwin:
            out += u.tanwin
            after_vowel = False
            continue
        if u.sukun or voweled or not u.c:
            after_vowel = False
            continue
        if nxt is None:
            after_vowel = False  # word-final consonant: pausal form, no vowel
            continue
        if not nxt.c and (nxt.marbuta or nxt.long):
            after_vowel = False  # the next unit supplies the vowel
            continue

        # Restore a short vowel. This consonant closes the syllable if a vowel
        # precedes it and the next consonant can start a syllable (it has a
        # vowel, or is followed by more letters). Arabic avoids a closed
        # syllable after a long vowel inside a word; Persian and Urdu do not
        # (کرمانشاه Kermānshāh)
        next_can_start = (
            nxt.long is not None or nxt.short is not None or after is not None
        )
        long_ok = lang in ("fa", "ur", "ps") or not last_nucleus_long
        if after_vowel and long_ok and next_can_start and k > 0:
            after_vowel = False
            continue
        out += _choose_vowel(nxt)
        after_vowel, last_nucleus_long = True, False
    return out


def _choose_vowel(nxt: _Unit | None) -> str:
    if nxt is not None and (nxt.long or nxt.short):
        if nxt.c == "y":
            return "i"
        if nxt.c in ("w", "v"):
            return "u"
    return "a"


# --------------------------------------------------------------------------- #
# Uyghur (Uyghur Latin Yéziqi) and Sorani Kurdish (Hawar-style Latin)
# --------------------------------------------------------------------------- #

_UYGHUR_MAP = {
    "ا": "a", "ە": "e", "ب": "b", "پ": "p", "ت": "t", "ج": "j", "چ": "ch",
    "خ": "x", "د": "d", "ر": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh",
    "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "گ": "g", "ڭ": "ng", "ل": "l",
    "م": "m", "ن": "n", "ھ": "h", "و": "o", "ۇ": "u", "ۆ": "ö", "ۈ": "ü",
    "ۋ": "w", "ې": "ë", "ى": "i", "ی": "i", "ي": "y",
}  # fmt: skip
_UYGHUR_VOWELS = frozenset("aeiouöüë")


def _UYGHUR(text: str) -> str:
    out = ""
    for ch in text:
        if ch == "ئ":
            # Hamza carrier before a vowel: silent at a word start, an
            # apostrophe between vowels
            if out and out[-1] in _UYGHUR_VOWELS:
                out += "'"
            continue
        out += _UYGHUR_MAP.get(ch, ch)
    return " ".join(capitalize_first(w) for w in out.split(" "))


_KURDISH_MAP = {
    "ا": "a", "ب": "b", "پ": "p", "ت": "t", "ج": "c", "چ": "ç", "ح": "h",
    "خ": "x", "د": "d", "ر": "r", "ڕ": "r", "ز": "z", "ژ": "j", "س": "s",
    "ش": "ş", "ع": "", "غ": "x", "ف": "f", "ڤ": "v", "ق": "q", "ک": "k",
    "ك": "k", "گ": "g", "ل": "l", "ڵ": "ll", "م": "m", "ن": "n", "ه": "h",
    "ە": "e", "ۆ": "o", "ێ": "ê",
}  # fmt: skip
_KURDISH_VOWELS = frozenset("aeêo")


def _kurdish(text: str) -> str:
    chars = list(text)
    out = ""
    for i, ch in enumerate(chars):
        prev = out[-1:] if out else ""
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if ch == "ئ":
            continue  # vowel carrier
        if ch == "و":
            if nxt == "و":
                continue  # first half of وو û
            if (
                prev == ""
                or prev == " "
                or prev in _KURDISH_VOWELS
                or nxt in ("ا", "ە", "ێ", "ۆ")
            ):
                out += "w" if not (i > 0 and chars[i - 1] == "و") else "û"
            else:
                out += "û" if i > 0 and chars[i - 1] == "و" else "u"
            continue
        if ch in ("ی", "ي"):
            if (
                prev == ""
                or prev == " "
                or prev in _KURDISH_VOWELS
                or nxt in ("ا", "ە", "ێ", "ۆ")
            ):
                out += "y"
            else:
                out += "î"
            continue
        out += _KURDISH_MAP.get(ch, ch)
    return " ".join(capitalize_first(w) for w in out.split(" "))
