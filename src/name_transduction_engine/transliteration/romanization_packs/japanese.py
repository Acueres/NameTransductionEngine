"""Japanese: modified Hepburn with macrons (Tōkyō, Ōsaka-fu, Shinjuku-ku).

Kana is converted directly. Kanji needs readings, which come from the
MeCab/UniDic analyzer (fugashi + unidic-lite). Without it, or when a kanji
has no reading, the native form is kept: a Chinese reading of a Japanese name
would be wrong, not merely rough.

Four parts, in the order a name goes through them:

1. Kana to Hepburn works on morae, over a whole word at once, so sokuon,
   syllabic n and small kana behave the same whatever the analyzer's token
   boundaries (ブラック|リー Burakkurī, サト|ゥ Satu). Extended katakana are
   composed by rule from the base kana and the small one (ニェ nye, ヴュ vyu),
   not looked up in a table.
2. Readings. UniDic gives each token an orthographic reading and a
   pronunciation; the pronunciation says which vowels are long (東京
   トーキョー Tōkyō, but 井上 イノウエ Inoue). Katakana the analyzer does not
   know is read as written, a long vowel being written ー, except ou, which
   Japanese pronounces ō.
3. Word division (Higashi Rōma Teikoku, but Nihonbashi), generic terms after
   a name (Tone-gawa, Erube-gawa, Kitonosu-tō), particles between words
   (Atenai no Akuroporisu).
4. Choosing between UniDic's readings with the entity's Latin-script names
   (Context.hints): a romanized Japanese name picks the reading it spells
   (羽田 Haneda, not Hata); a pinyin name, or a generic term only Chinese
   places have (鎮, 省), marks a Chinese place, which Japanese reads with
   Sino-Japanese readings (南充市 Nanjū-shi).
"""

import itertools
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache

from .abjad import fold_latin
from .base import (
    Context,
    NoRomanization,
    ProviderUnavailable,
    Rendering,
    package_version,
)
from .text import capitalize_first

# --------------------------------------------------------------------------- #
# 1. Kana to modified Hepburn
# --------------------------------------------------------------------------- #
#
# A mora is a base kana, optionally composed with a small kana after it.
# Every base kana is an onset and a vowel; composition keeps the base's
# onset (palatalized for the i column: ニ+ェ nye, キ+ャ kya) and takes the
# small kana's vowel. ウ and イ have no consonant of their own and give w
# and y (ウィ wi, イェ ye); ク and グ give kw and gw (クァ kwa).
#
# Small ャュョ always compose (yōon). A small vowel composes only into a
# syllable the regular syllabary cannot write: that is what it is for
# (ティ ti, スィ si, ファ fa, ツァ tsa, ドゥ du, ニェ nye). Where the
# syllable exists already, the small vowel is a vowel of its own, written
# small to show it is short or gliding: ブィ bui, ルィ rui (Russian ы),
# ニァ nia, リィ rii.

_BASE: dict[str, tuple[str, str]] = {
    "ア": ("", "a"), "イ": ("", "i"), "ウ": ("", "u"), "エ": ("", "e"), "オ": ("", "o"),
    "カ": ("k", "a"), "キ": ("k", "i"), "ク": ("k", "u"), "ケ": ("k", "e"), "コ": ("k", "o"),
    "ガ": ("g", "a"), "ギ": ("g", "i"), "グ": ("g", "u"), "ゲ": ("g", "e"), "ゴ": ("g", "o"),
    "サ": ("s", "a"), "シ": ("sh", "i"), "ス": ("s", "u"), "セ": ("s", "e"), "ソ": ("s", "o"),
    "ザ": ("z", "a"), "ジ": ("j", "i"), "ズ": ("z", "u"), "ゼ": ("z", "e"), "ゾ": ("z", "o"),
    "タ": ("t", "a"), "チ": ("ch", "i"), "ツ": ("ts", "u"), "テ": ("t", "e"), "ト": ("t", "o"),
    "ダ": ("d", "a"), "ヂ": ("j", "i"), "ヅ": ("z", "u"), "デ": ("d", "e"), "ド": ("d", "o"),
    "ナ": ("n", "a"), "ニ": ("n", "i"), "ヌ": ("n", "u"), "ネ": ("n", "e"), "ノ": ("n", "o"),
    "ハ": ("h", "a"), "ヒ": ("h", "i"), "フ": ("f", "u"), "ヘ": ("h", "e"), "ホ": ("h", "o"),
    "バ": ("b", "a"), "ビ": ("b", "i"), "ブ": ("b", "u"), "ベ": ("b", "e"), "ボ": ("b", "o"),
    "パ": ("p", "a"), "ピ": ("p", "i"), "プ": ("p", "u"), "ペ": ("p", "e"), "ポ": ("p", "o"),
    "マ": ("m", "a"), "ミ": ("m", "i"), "ム": ("m", "u"), "メ": ("m", "e"), "モ": ("m", "o"),
    "ヤ": ("y", "a"), "ユ": ("y", "u"), "ヨ": ("y", "o"),
    "ラ": ("r", "a"), "リ": ("r", "i"), "ル": ("r", "u"), "レ": ("r", "e"), "ロ": ("r", "o"),
    "ワ": ("w", "a"), "ヰ": ("", "i"), "ヱ": ("", "e"), "ヲ": ("", "o"),
    "ヴ": ("v", "u"), "ヵ": ("k", "a"), "ヶ": ("k", "e"),
    "ヷ": ("v", "a"), "ヸ": ("v", "i"), "ヹ": ("v", "e"), "ヺ": ("v", "o"),
}  # fmt: skip

# Syllables written by single kana or regular yōon, which a small vowel
# never needs to spell
_REGULAR_SYLLABLES = frozenset(
    {o + v for k, (o, v) in _BASE.items() if k not in "ヴヷヸヹヺヵヶヰヱヲ"}
    | {o + "y" + v for o in "kgnhbpmr" for v in "auo"}
    | {o + v for o in ("sh", "ch", "j") for v in "auo"}
)
_SMALL_Y = {"ャ": "a", "ュ": "u", "ョ": "o"}
_SMALL_VOWEL = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o", "ヮ": "a"}
_PALATAL = {"sh": "sh", "ch": "ch", "j": "j", "": "y"}  # others: onset + y
_SOKUON = "ッ"
_N = "ン"
_CHOON = "ー"
_MACRON = {"a": "ā", "i": "ī", "u": "ū", "e": "ē", "o": "ō"}


def _to_katakana(text: str) -> str:
    # Hiragana U+3041..U+3096 sit exactly 0x60 below their katakana
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def _palatal(onset: str) -> str:
    return _PALATAL.get(onset, onset + "y")


def _compose(base: str, small: str) -> tuple[str, str] | None:
    """Onset and vowel of base + small kana, or None if they do not combine
    (a vowel kana followed by another vowel: アァ)"""
    onset, vowel = _BASE[base]
    if small in _SMALL_Y:
        # Yōon (キャ kya, シュ shu) and its loanword extensions (フュ fyu,
        # テュ tyu, デュ dyu)
        return (_palatal(onset) if vowel == "i" else onset + "y"), _SMALL_Y[small]
    new = _SMALL_VOWEL[small]
    if new == vowel and small != "ヮ":
        return None  # リィ rii
    if small == "ヮ" or base in "クグ":
        composed = onset + "w"  # クァ kwa, グィ gwi, クヮ kwa
    elif base == "ウ":
        composed = "w"  # ウィ wi, ウォ wo
    elif vowel == "i":
        composed = _palatal(onset)  # ニェ nye, シェ she, イェ ye
    elif not onset:
        return None  # アァ
    else:
        composed = onset  # ファ fa, ツァ tsa, ティ ti, ドゥ du, ヴァ va
    if composed + new in _REGULAR_SYLLABLES:
        return None  # ブィ bui, ニァ nia
    return composed, new


@dataclass(frozen=True)
class _Mora:
    text: str  # the kana (or other character) it was read from
    onset: str | None = None  # None: not a regular mora (ッ, ン, ー, other)
    vowel: str = ""


def _morae(kana: str) -> list[_Mora]:
    kana = _to_katakana(kana)
    out: list[_Mora] = []
    i = 0
    while i < len(kana):
        ch = kana[i]
        nxt = kana[i + 1] if i + 1 < len(kana) else ""
        if ch in _BASE:
            if nxt in _SMALL_Y or nxt in _SMALL_VOWEL:
                composed = _compose(ch, nxt)
                if composed is not None:
                    out.append(_Mora(ch + nxt, *composed))
                    i += 2
                    continue
            out.append(_Mora(ch, *_BASE[ch]))
        elif ch in _SMALL_VOWEL or ch in _SMALL_Y:
            # A small kana with nothing to combine with is read as its vowel
            vowel = _SMALL_VOWEL.get(ch) or _SMALL_Y[ch]
            out.append(_Mora(ch, "y" if ch in _SMALL_Y else "", vowel))
        else:
            out.append(_Mora(ch))
        i += 1
    return out


def _starts_with_vowel_or_y(mora: _Mora | None) -> bool:
    return mora is not None and mora.onset is not None and mora.onset in ("", "y")


def render_kana(kana: str) -> str:
    """Kana to Hepburn, reading every vowel as written: only ー lengthens
    (use mark_long_vowels or UniDic's pronunciation for the rest).
    ッ doubles the next consonant (ッチ tchi), ン is n' before a vowel or y.
    Other characters pass through and break the mora sequence."""
    morae = _morae(kana)
    out: list[str] = []
    geminate = False
    for idx, m in enumerate(morae):
        if m.text == _SOKUON:
            geminate = True
            continue
        if m.text == _CHOON:
            if out and out[-1] and out[-1][-1] in _MACRON:
                out[-1] = out[-1][:-1] + _MACRON[out[-1][-1]]
            geminate = False
            continue
        if m.text == _N:
            nxt = morae[idx + 1] if idx + 1 < len(morae) else None
            out.append("n'" if _starts_with_vowel_or_y(nxt) else "n")
            geminate = False
            continue
        if m.onset is None:
            out.append(m.text)
            geminate = False
            continue
        roman = m.onset + m.vowel
        if geminate and m.onset:
            roman = ("t" if m.onset.startswith("ch") else m.onset[0]) + roman
        geminate = False
        out.append(roman)
    return "".join(out)


# Vowel kana that lengthen the vowel before them in native words
_LENGTHENERS = {"o": "ウオ", "u": "ウ", "a": "ア", "e": "エ"}


# Katakana writes long vowels with ー, so its vowel letters are read as
# written, except ou: Japanese pronounces it ō in loanwords too (ソウル is
# pronounced ソール), and katakana spells foreign "ow", "ou" with it
# (チェプストウ Chepusutō, コウヴォラ Kōvora)
_KATAKANA_LENGTHENERS = {"o": "ウ"}


def mark_long_vowels(kana: str, katakana: bool = False) -> str:
    """Spelling rule for words with no pronunciation to go by: in native
    words ou, oo, uu, aa, ee are one long vowel (とうきょう Tōkyō); ei and ii
    are not. In katakana only ou is. Returns the kana with the lengthening
    vowels replaced by ー."""
    lengtheners = _KATAKANA_LENGTHENERS if katakana else _LENGTHENERS
    out: list[str] = []
    prev = ""
    for m in _morae(kana):
        if m.onset == "" and m.text in lengtheners.get(prev, ""):
            out.append(_CHOON)
            continue  # the vowel stays the previous one
        out.append(m.text)
        prev = m.vowel if m.onset is not None else ""
    return "".join(out)


def kana_to_hepburn(kana: str) -> str:
    """Kana with no analyzer: hiragana spells native words, whose long
    vowels follow the spelling rule; katakana writes length with ー"""
    parts = re.split(r"([ァ-ヺー]+)", kana)
    return render_kana(
        "".join(mark_long_vowels(p, katakana=i % 2 == 1) for i, p in enumerate(parts))
    )


# --------------------------------------------------------------------------- #
# 2. Readings
# --------------------------------------------------------------------------- #


def phonetic_reading(
    kana: str, pron: str | None, native: bool, katakana: bool = False
) -> str:
    """The kana to romanize for one token: its orthographic reading with each
    long vowel written ー, as UniDic's pronunciation has it (トウキョウ /
    トーキョー -> トーキョー, イノウエ / イノウエ unchanged). ei and ii stay
    two letters, as Hepburn writes them. Without a usable pronunciation,
    native words follow the spelling rule.

    Particles are not given their sound (は wa): place names have no topic
    or direction particles, and a は the analyzer takes for one is part of
    a name (なは Naha)."""
    kana = _to_katakana(kana)
    if pron and pron != "*" and len(pron) == len(kana):
        return "".join(
            _CHOON if p == _CHOON and k in "アウエオ" else k for k, p in zip(kana, pron)
        )
    if native or katakana:
        return mark_long_vowels(kana, katakana=katakana and not native)
    return kana


# --------------------------------------------------------------------------- #
# The analyzer
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _tagger():
    try:
        import fugashi  # noqa: PLC0415 - optional dependency, loaded on demand
    except ImportError as exc:
        raise ProviderUnavailable(
            "Japanese readings need fugashi and unidic-lite"
        ) from exc
    try:
        return fugashi.Tagger()
    except RuntimeError as exc:  # installed, but no dictionary found
        raise ProviderUnavailable(f"Japanese analyzer failed to start: {exc}") from exc


@lru_cache(maxsize=1)
def _lattice_tagger():
    """The analyzer in all-morphs mode: every dictionary entry that matches
    anywhere in the text, not only the best path"""
    import fugashi  # noqa: PLC0415 - _tagger() has checked it is installed

    return fugashi.Tagger("-a")


@lru_cache(maxsize=1)
def _engine() -> str:
    return (
        f"{package_version('fugashi')}, {package_version('unidic-lite', 'UniDic-lite')}"
    )


def _is_kana(ch: str) -> bool:
    return "ぁ" <= ch <= "ヿ" or ch == _CHOON


def _is_han(ch: str) -> bool:
    return (
        "一" <= ch <= "鿿"
        or "㐀" <= ch <= "䶿"
        or "豈" <= ch <= "﫿"
        or "\U00020000" <= ch <= "\U0003134f"
        or ch in "々〆ヶ"
    )


def _usable(reading: str | None) -> bool:
    return bool(reading) and reading != "*"


@dataclass(frozen=True)
class _Entry:
    """A dictionary entry: one way of reading a stretch of text"""

    kana: str  # orthographic reading, katakana
    pron: str | None
    pos1: str
    pos2: str
    pos3: str
    goshu: str | None

    @property
    def native(self) -> bool:
        return self.goshu not in ("外", "記号")


def _entry(node) -> _Entry:
    f = node.feature
    return _Entry(
        f.kana if _usable(f.kana) else "",
        f.pron if _usable(f.pron) else None,
        f.pos1 or "",
        f.pos2 or "",
        f.pos3 or "",
        f.goshu,
    )


@lru_cache(maxsize=1 << 12)
def _entries_by_surface(text: str) -> dict[str, tuple[_Entry, ...]]:
    """Every reading the dictionary has for each stretch of `text`"""
    found: dict[str, list[_Entry]] = defaultdict(list)
    for node in _lattice_tagger().parseToNodeList(text):
        if node.is_unk or not any(_is_han(c) for c in node.surface):
            continue
        e = _entry(node)
        if e.kana and e not in found[node.surface]:
            found[node.surface].append(e)
    return {s: tuple(es) for s, es in found.items()}


# --------------------------------------------------------------------------- #
# 3. Word division
# --------------------------------------------------------------------------- #
#
# Hepburn writes a name as separate words, each capitalized (Higashi Rōma
# Teikoku, Kansai Kokusai Kūkō), but a Japanese toponym is one word however
# many morphemes it has (Nihonbashi, Higashiōsaka). UniDic already knows most
# toponyms as single proper-noun tokens, so words are divided between tokens
# by their part of speech and word origin (goshu):
#
# - a katakana word stands alone: Higashi Rōma, Roppongi Hiruzu; inside one
#   token too, where a kanji prefix meets a katakana name (南アフリカ Minami
#   Afurika), and after a prefix (高アトラス Kō Atorasu). Katakana next to
#   katakana stays one word: UniDic splits unknown foreign names into pieces
#   (ス|ファックス for Sfax), and Japanese writes loanword compounds without
#   spaces anyway (スーパーアリーナ Sūpāarīna)
# - a Sino-Japanese common noun of two or more kanji starts a word: Teikoku,
#   Kokusai Kūkō, Tōhoku Chihō
# - after a proper noun, any common noun of two or more characters starts a
#   word (Fushimi Inari Taisha), and so does another proper noun (Nikkō
#   Tōshō-gū, Tōkyō-to Chiyoda-ku)
# - a particle between two words is a word of its own, in lowercase
#   (アテナイのアクロポリス Atenai no Akuroporisu); inside a toponym it
#   attaches (Amanohashidate, Ochanomizu)
# - prefixes, suffixes and single-kanji common nouns attach: Dainippon,
#   Kyōwakoku, Toshokan, Higashinippon
# - generic terms attach with a hyphen: administrative units after any word
#   (Tōkyō-to, Tokubetsu-ku), geographic and other classifiers after a name
#   (Tone-gawa, Biwa-ko, Ōsaka-jō, Erube-gawa, Habusuburuku-ke); peoples and
#   languages are written solid (Nihonjin, Kurashovajin)

_PREFIX = "prefix"
_SUFFIX = "suffix"
_GENERIC = "generic"  # hyphenated generic term: -to, -shi, -gawa, -jō
_PROPER = "proper"
_COMMON = "common"  # common noun of 2+ characters, native or mixed origin
_SINO = "sino"  # Sino-Japanese common noun of 2+ characters
_SHORT = "short"  # single-character common noun, numeral, counter
_KATAKANA = "katakana"
_CHINESE = "chinese"  # part of a Chinese place name read in Sino-Japanese
_LATIN = "latin"
_PARTICLE = "particle"
_BOUND = "bound"  # other non-nouns: never split off
_PUNCT = "punct"  # hyphens, brackets: kept, and the next word is capitalized

# Kinds that name something: a generic term after them is hyphenated
_NAMES = frozenset({_PROPER, _KATAKANA, _CHINESE, _LATIN})
# Names read in a foreign way: generic terms take the reading Japanese uses
# after foreign names (キトノス島 Kitonosu-tō, ナイル川 Nairu-gawa)
_FOREIGN = frozenset({_KATAKANA, _CHINESE, _LATIN})

# Japanese administrative units: hyphenated after any word but the first
_ADMIN = frozenset("都道府県市区町村郡")

# Generic terms and how they are read in a compound after a name: (after a
# Japanese name, after a foreign one). None keeps the analyzer's reading:
# 町 is machi or chō, 村 mura or son, depending on the place, and a Japanese
# island is -shima, -jima or -tō (屋久島 Yakushima, 本州島 Honshūtō), while
# a foreign one is always -tō. The entity's own names can still choose
# another reading the dictionary has (石垣島 Ishigaki-jima).
_GENERIC_TERMS: dict[str, tuple[str | None, str | None]] = {
    # administrative units, Japanese and foreign
    "都": ("ト", "ト"), "道": ("ドウ", "ドウ"), "府": ("フ", "フ"),
    "県": ("ケン", "ケン"), "市": ("シ", "シ"), "区": ("ク", "ク"),
    "郡": ("グン", "グン"), "町": (None, None), "村": (None, None),
    "州": ("シュウ", "シュウ"), "省": ("ショウ", "ショウ"), "鎮": ("チン", "チン"),
    "郷": (None, "キョウ"), "国": (None, "コク"), "領": ("リョウ", "リョウ"),
    # landforms and waters
    "川": ("ガワ", "ガワ"), "山": ("サン", "サン"), "岳": ("ダケ", "ダケ"),
    "峰": ("ホウ", "ホウ"), "島": (None, "トウ"), "湖": ("コ", "コ"),
    "海": ("カイ", "カイ"), "湾": ("ワン", "ワン"), "岬": ("ミサキ", "ミサキ"),
    "峠": ("トウゲ", "トウゲ"), "礁": ("ショウ", "ショウ"), "池": (None, "イケ"),
    # buildings and places
    "城": ("ジョウ", "ジョウ"), "宮": ("グウ", "キュウ"), "寺": ("ジ", "ジ"),
    "駅": ("エキ", "エキ"), "港": ("コウ", "コウ"), "橋": (None, "キョウ"),
    "線": ("セン", "セン"),
    # families and dynasties: Habusuburuku-ke, Burubon-chō
    "家": ("ケ", "ケ"), "朝": ("チョウ", "チョウ"),
    # peoples, languages, countries: written as one word (Nihonjin, Nihongo,
    # Avantikoku), but read as in a compound (クラショヴァ人 -jin, not -nin)
    "人": ("ジン", "ジン"), "族": ("ゾク", "ゾク"), "語": ("ゴ", "ゴ"),
}  # fmt: skip
_SOLID_TERMS = frozenset("人族語国")


@dataclass
class _Piece:
    surface: str
    # What to romanize: phonetic kana (see phonetic_reading), or the surface
    # itself for Latin, digits and punctuation. The first is the default;
    # the others are the dictionary's other readings, for the hints to choose
    readings: list[str]
    kind: str
    entry: _Entry | None = None  # the analyzer's entry for the token
    choice: int = 0
    guessed: bool = False  # a Sino-Japanese reading predicted from pinyin

    @property
    def kana(self) -> str:
        return self.readings[self.choice]


def _is_katakana_word(text: str) -> bool:
    return len(text) > 1 and all("ァ" <= c <= "ヺ" or c == _CHOON for c in text)


def _all_katakana(text: str) -> bool:
    return bool(text) and all("ァ" <= c <= "ヺ" or c == _CHOON for c in text)


def _all_hiragana(text: str) -> bool:
    return bool(text) and all("ぁ" <= c <= "ゖ" for c in text)


def _kind(surface: str, e: _Entry) -> str:
    if e.pos1 == "接頭辞":
        return _PREFIX
    if e.pos1 == "接尾辞":
        return _SUFFIX
    if surface.isascii() and any(c.isalpha() for c in surface):
        return _LATIN
    if e.pos1 in ("補助記号", "記号") and not any(
        _is_han(c) or _is_kana(c) for c in surface
    ):
        return _PUNCT
    if _is_katakana_word(surface):
        return _KATAKANA  # loanwords of any part of speech: ビッグ, ユニバーサル
    if e.pos1 == "助詞" and _all_hiragana(surface):
        # a hiragana particle; katakana ノ, ガ, ヶ are part of a name
        # (御茶ノ水) or of a katakana word split by the analyzer (クラス|ノ|ダル)
        return _PARTICLE
    if e.pos1 != "名詞":
        return _BOUND
    if e.pos2 == "固有名詞":
        return _PROPER
    if e.pos2 == "普通名詞" and e.pos3 != "助数詞可能" and len(surface) > 1:
        return _SINO if e.goshu == "漢" else _COMMON
    return _SHORT


def _starts_word(prev: _Piece, cur: _Piece) -> bool:
    if _all_katakana(prev.surface) and _all_katakana(cur.surface):
        return False
    if cur.kind == _KATAKANA and prev.kind == _PREFIX:
        return True  # 高アトラス Kō Atorasu
    if cur.kind in (_SUFFIX, _GENERIC, _BOUND, _PARTICLE) or prev.kind in (
        _PREFIX,
        _BOUND,
        _PARTICLE,
    ):
        return False
    if _LATIN in (prev.kind, cur.kind) or cur.kind == _KATAKANA:
        return True
    if prev.kind == _CHINESE and cur.kind == _CHINESE:
        return False
    if prev.kind == _KATAKANA:
        # a katakana name followed by a kanji word; a single kanji attaches
        return cur.kind != _SHORT
    if cur.kind == _SINO:
        return True
    if cur.kind == _COMMON:
        return prev.kind in (_PROPER, _CHINESE)
    if cur.kind in (_PROPER, _CHINESE):
        return prev.kind in (_PROPER, _CHINESE, _SUFFIX, _GENERIC)
    return False


# A katakana run of two or more characters inside a mixed token: 南|アフリカ.
# Single ones are particles or linking kana inside toponyms (御茶ノ水, 関ケ原).
_KATAKANA_RUN = re.compile(r"[ァ-ヺー]{2,}")


def _split_mixed(piece: _Piece) -> list[_Piece]:
    """Split a token where kanji meet a katakana name (南アフリカ), sharing
    out the reading by matching the katakana part literally"""
    surface, reading = piece.surface, piece.kana
    runs = [m for m in _KATAKANA_RUN.finditer(surface) if m.group() != surface]
    if not runs or piece.kind not in (_PROPER, _COMMON, _SINO, _SHORT):
        return [piece]
    parts: list[str] = []
    pattern = ""
    pos = 0
    for m in runs:
        if m.start() > pos:
            parts.append(surface[pos : m.start()])
            pattern += "(.+)"
        parts.append(m.group())
        # the reading writes a long vowel ー where the katakana may have a
        # vowel letter (ソウル / ソール)
        letters = (f"[{c}ー]" if c in "アイウエオ" else re.escape(c) for c in m.group())
        pattern += f"({''.join(letters)})"
        pos = m.end()
    if pos < len(surface):
        parts.append(surface[pos:])
        pattern += "(.+)"
    aligned = re.fullmatch(pattern, reading)
    if aligned is None:
        return [piece]
    return [
        _Piece(
            part,
            [r],
            _KATAKANA if _is_katakana_word(part) else piece.kind,
            piece.entry,
        )
        for part, r in zip(parts, aligned.groups())
    ]


def _katakana_groups(nodes: list) -> list[list]:
    """Consecutive tokens, with runs of katakana tokens grouped: the analyzer
    splits foreign names it does not know (ス|ファックス, クラス|ノ|ダル for
    Krasnodar), and a katakana word is never divided anyway"""
    groups: list[list] = []
    for node in nodes:
        if (
            groups
            and _all_katakana(node.surface)
            and _all_katakana(groups[-1][-1].surface)
        ):
            groups[-1].append(node)
        else:
            groups.append([node])
    return groups


def _pieces(chunk: str) -> list[_Piece]:
    pieces: list[_Piece] = []
    for group in _katakana_groups(list(_tagger()(chunk))):
        if len(group) > 1:
            # Pieces of one foreign name: read as written. A piece that
            # happens to match a word (ヴァー|ドウス, Vardøhus: ドウス is
            # also a name pronounced ドース) says nothing about its vowels.
            surface = "".join(n.surface for n in group)
            reading = phonetic_reading(surface, None, native=False, katakana=True)
            pieces.append(_Piece(surface, [reading], _KATAKANA))
            continue
        node = group[0]
        surface = node.surface
        e = _entry(node)
        kind = _kind(surface, e)
        if any(_is_han(c) for c in surface):
            # Kanji the dictionary cannot read at all are left unread: only a
            # Chinese name can be read without it (曲麻莱県), see below
            reading = phonetic_reading(e.kana, e.pron, e.native) if e.kana else ""
        elif _all_katakana(surface):
            # Katakana writes long vowels with ー, so it is read as written,
            # unless the dictionary knows the word (ソウル, pronounced ソール)
            reading = phonetic_reading(surface, e.pron, native=False, katakana=True)
        elif any(_is_kana(c) for c in surface):
            # Hiragana: the pronunciation, or the spelling rule, says which
            # vowels are long (とうきょう Tōkyō)
            reading = phonetic_reading(surface, e.pron, native=True)
        else:
            reading = surface  # digits, Latin, punctuation
        pieces.extend(_split_mixed(_Piece(surface, [reading], kind, e)))
    return pieces


def _mark_generics(pieces: list[_Piece]) -> None:
    """Generic terms after a name, and administrative units after any word,
    are hyphenated (or written solid) and take their compound reading"""
    for j in range(1, len(pieces)):
        piece, prev = pieces[j], pieces[j - 1]
        if piece.surface not in _GENERIC_TERMS or piece.kind == _CHINESE:
            continue  # part of a Chinese name: 宿州 Shukushū
        if piece.surface not in _ADMIN and prev.kind not in _NAMES:
            continue
        if prev.kind in (_PUNCT, _PARTICLE):
            continue
        native, foreign = _GENERIC_TERMS[piece.surface]
        reading = foreign if prev.kind in _FOREIGN else native
        if reading:
            reading = mark_long_vowels(reading)  # Sino-Japanese: ジョウ jō
        if reading and reading != piece.kana:
            piece.readings = [reading] + [r for r in piece.readings if r != reading]
            piece.choice = 0
        piece.kind = _SUFFIX if piece.surface in _SOLID_TERMS else _GENERIC


# --------------------------------------------------------------------------- #
# 4. Readings chosen with the entity's other names
# --------------------------------------------------------------------------- #
#
# A kanji name can often be read in several ways, and the analyzer's choice
# is a guess (羽田 Hata or Haneda, 国立 Kokuritsu or Kunitachi). The entity's
# Latin-script names are evidence:
#
# - a Japanese place usually has a romanized name (GeoNames: "Haneda",
#   "Kunitachi", "Matsusaka"): among the readings the dictionary has for each
#   token, the combination that spells it is taken;
# - a Chinese place has a pinyin name ("Nanchong" for 南充). Japanese reads
#   Chinese names with Sino-Japanese (on) readings, character by character,
#   unless the dictionary knows the name (北京 Pekin, 江蘇 Chansū). The
#   analyzer, not knowing the name, gives Japanese name readings instead
#   (南 Minami, 充 Mitsuru).


def _japanese_key(text: str) -> str:
    """Latin letters folded the way romanizations of Japanese differ: long
    vowels written or not (Tōkyō, Tokyo, Toukyou, Ohno), m or n before a
    labial (Nihombashi)"""
    key = fold_latin(text)
    key = re.sub(r"oh(?![aiueo])", "o", key)
    key = re.sub(r"ou|oo|uu|aa|ee", lambda m: m.group()[0], key)
    return re.sub(r"m(?=[bmp])", "n", key)


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _allowed_distance(key: str) -> int:
    """Edits accepted between a reading and a name: none for short words,
    where another reading is too easily one letter away"""
    return 0 if len(key) < 5 else len(key) // 5


@dataclass
class _Hints:
    texts: tuple[str, ...]
    # each hint split into words, and runs of up to three words joined:
    # "Haneda Airport" -> haneda, airport, hanedaairport
    plain: dict[str, str] = field(default_factory=dict)  # key -> hint
    japanese: dict[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, hints: tuple[str, ...]) -> "_Hints":
        out = cls(hints)
        for hint in hints:
            words = [w for w in re.split(r"[\s\-‐,;:()/]+", hint) if w]
            for n in (1, 2, 3):
                for i in range(len(words) - n + 1):
                    run = "".join(words[i : i + n])
                    out.plain.setdefault(fold_latin(run), hint)
                    out.japanese.setdefault(_japanese_key(run), hint)
        out.plain.pop("", None)
        out.japanese.pop("", None)
        return out

    @staticmethod
    def _closest(key: str, keys: dict[str, str]) -> tuple[int, str]:
        best = (len(key) + 1, "")
        for k, hint in keys.items():
            if abs(len(k) - len(key)) > best[0]:
                continue
            d = _levenshtein(key, k)
            if d < best[0]:
                best = (d, hint)
        return best

    def match_pinyin(self, pinyin: str) -> str | None:
        d, hint = self._closest(pinyin, self.plain)
        return hint if d <= _allowed_distance(pinyin) else None

    def distance(self, romanized: str) -> tuple[int, str]:
        return self._closest(_japanese_key(romanized), self.japanese)


# Sino-Japanese syllables: one mora (a vowel, or a consonant with or without
# y and a vowel), then at most one of the codas Middle Chinese left in
# Japanese: u, i, n, ku, ki, tsu, chi (and fu in older readings)
_SINO_SYLLABLE = re.compile(r"[^ャュョァィゥェォッンー][ャュョ]?[ウインクキツチフ]?")
_GOSHU_RANK = {"漢": 0, "記号": 1, "固": 2}  # 記号 entries name the character


def _sino_readings(char: str) -> list[str]:
    """The dictionary's Sino-Japanese readings of one kanji, best attested
    first: from Sino-Japanese entries, the character's own name, then name
    entries of Sino-Japanese shape"""
    found: list[tuple[int, str]] = []
    for e in _entries_by_surface(char).get(char, ()):
        rank = _GOSHU_RANK.get(e.goshu or "")
        if rank is None or not _SINO_SYLLABLE.fullmatch(e.kana):
            continue
        found.append((rank, e.kana))
    out: list[str] = []
    for _, kana in sorted(found, key=lambda x: x[0]):
        if kana not in out:
            out.append(kana)
    return out


_PINYIN_INITIALS = re.compile(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)(.*)$")
_HEPBURN_ONSET = re.compile(r"^([^aiueoāīūēō]*)(.*)$")


def _halves(pattern: re.Pattern, text: str) -> tuple[str, str]:
    m = pattern.match(text)
    return (m.group(1), m.group(2)) if m else ("", text)


@dataclass(frozen=True)
class _SinoModel:
    """How Mandarin syllables correspond to Sino-Japanese readings, counted
    over every kanji the dictionary gives a Sino-Japanese reading (about
    2,200). Both descend from Middle Chinese, so the syllable tells which of
    a kanji's readings is the usual one (北 bei: hoku, not boku), and gives a
    likely reading to a kanji the dictionary has none for (塘 tang: tō).
    Syllables seen rarely are helped by the initial and final separately:
    the Japanese onset follows the Mandarin initial (m- m-), the rest the
    final (-ai -ai): 邁 mai is mai, though the only other "mai" kanji
    counted reads myaku."""

    syllables: dict[str, Counter]
    onsets: dict[str, Counter]  # Mandarin initial -> Japanese onset
    rimes: dict[str, Counter]  # Mandarin final -> rest of the reading
    readings: Counter  # how many kanji have each reading

    def score(self, syllable: str, reading: str) -> float:
        initial, final = _halves(_PINYIN_INITIALS, syllable)
        onset, rime = _halves(_HEPBURN_ONSET, render_kana(mark_long_vowels(reading)))
        onsets, rimes = self.onsets.get(initial), self.rimes.get(final)
        parts = 0.0
        if onsets and rimes:
            parts = (onsets[onset] / onsets.total()) * (rimes[rime] / rimes.total())
        return self.syllables.get(syllable, Counter())[reading] + parts

    def best(self, syllable: str, readings) -> str | None:
        return max(
            readings,
            key=lambda r: (self.score(syllable, r), self.readings[r]),
            default=None,
        )


@lru_cache(maxsize=1)
def _sino_model() -> _SinoModel:
    from pypinyin import Style, pinyin  # noqa: PLC0415 - optional dependency

    model = _SinoModel(
        defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter()
    )
    for code in range(0x4E00, 0x9FA6):
        char = chr(code)
        readings = [
            _entry(n).kana
            for n in _lattice_tagger().parseToNodeList(char)
            if n.surface == char and n.feature.goshu in ("漢", "記号")
        ]
        readings = list(
            dict.fromkeys(r for r in readings if r and _SINO_SYLLABLE.fullmatch(r))
        )
        if not readings:
            continue
        syllables = pinyin(char, style=Style.NORMAL, heteronym=True)[0]
        weight = 1 / (len(readings) * len(syllables))
        for syllable in syllables:
            initial, final = _halves(_PINYIN_INITIALS, syllable)
            for r in readings:
                onset, rime = _halves(_HEPBURN_ONSET, render_kana(mark_long_vowels(r)))
                model.syllables[syllable][r] += weight
                model.onsets[initial][onset] += weight
                model.rimes[final][rime] += weight
                model.readings[r] += weight
    return model


def _sino_reading(char: str, syllable: str) -> tuple[str, bool] | None:
    """A Sino-Japanese reading for `char` read `syllable` in Mandarin, and
    whether it was predicted rather than found in the dictionary"""
    model = _sino_model()
    own = _sino_readings(char)
    if own:
        # the reading usual for this Mandarin syllable, else the best attested
        return max(own, key=lambda r: (model.score(syllable, r), -own.index(r))), False
    guess = model.best(syllable, model.readings)
    return (guess, True) if guess else None


def _all_han(text: str) -> bool:
    return bool(text) and all(_is_han(c) for c in text)


# One syllable of a Sino-Japanese reading, or of the Mandarin-based readings
# Japanese has for some Chinese places (シャン|ハイ, チャン|スー, チン|タオ)
_SINO_LIKE = (
    r"[^ャュョァィゥェォッンー][ャュョァィゥェォ]?[アイウエオンクキツチフーッ]?"
)


def _kept_whole(piece: _Piece) -> bool:
    """A token the dictionary knows as a place name or a Sino-Japanese word,
    read one syllable per character: its reading is the established Japanese
    one for a Chinese name (北京 Pekin, 上海 Shanhai, 九寨 Kyūsai). A
    Japanese place of the same spelling has a Japanese reading instead
    (小金 Kogane, the Chinese county Xiaojin being Shōkin)."""
    e = piece.entry
    if e is None or len(piece.surface) < 2 or not (e.pos3 == "地名" or e.goshu == "漢"):
        return False
    return re.fullmatch(f"(?:{_SINO_LIKE}){{{len(piece.surface)}}}", e.kana) is not None


# Generic terms only Chinese places have: a name before them is Chinese
# whatever the hints say (吉隆鎮, Tibetan Kyirong)
_CHINESE_ONLY_TERMS = frozenset("鎮省")


def _read_chinese(run: list[_Piece], hints: _Hints) -> tuple[list[_Piece], str] | None:
    """Re-read a run of kanji tokens as a Chinese place name if one of the
    hints is its pinyin, or a Chinese generic term follows it; returns the
    new pieces and what showed the name to be Chinese"""
    try:
        from pypinyin import Style, lazy_pinyin  # noqa: PLC0415
    except ImportError:
        return None
    # Common nouns closing the run are generic words of their own (遼東半島
    # Ryōtō Hantō), and so is a generic term (省, 市, 鎮)
    after: list[_Piece] = []
    while len(run) > 1 and run[-1].kind in (_SINO, _COMMON) and run[-1].kana:
        after.insert(0, run[-1])
        run = run[:-1]
    text = "".join(p.surface for p in run)
    tail = (
        run[-1].surface
        if len(run) > 1 and run[-1].surface in _GENERIC_TERMS
        else text[-1]
        if text[-1] in _GENERIC_TERMS and not _kept_whole(run[-1])
        else ""
    )
    name = text[: len(text) - len(tail)]
    if len(name) < 2:
        return None
    syllables = lazy_pinyin(name, style=Style.NORMAL)
    if len(syllables) != len(name):
        return None
    hint = hints.match_pinyin("".join(syllables))
    if hint is None and tail not in _CHINESE_ONLY_TERMS:
        return None
    evidence = f"matching {hint!r}" if hint else f"before {tail}"

    out: list[_Piece] = []
    offset = 0
    reread = False
    for piece in run:
        span = piece.surface[: max(0, len(name) - offset)]
        if not span:
            break
        if span == piece.surface and _kept_whole(piece):
            out.append(replace(piece, kind=_CHINESE, readings=[piece.kana]))
        else:
            reread = True
            for k, char in enumerate(span):
                found = _sino_reading(char, syllables[offset + k])
                if found is None:
                    return None
                kana, guessed = found
                out.append(
                    _Piece(char, [mark_long_vowels(kana)], _CHINESE, guessed=guessed)
                )
        offset += len(span)
    if tail:
        # its reading is set with the other generic terms when it has one
        if run[-1].surface == tail:
            kana = run[-1].kana
        else:
            found = _sino_reading(tail, lazy_pinyin(tail, style=Style.NORMAL)[0])
            if found is None:
                return None
            kana = mark_long_vowels(found[0])
        out.append(_Piece(tail, [kana], _SHORT))
    if not hint and not reread:
        return None  # a word the dictionary knows: 外務省 Gaimushō
    return out + after, evidence


def _apply_chinese(
    pieces: list[_Piece], hints: _Hints, notes: list[str]
) -> list[_Piece]:
    out: list[_Piece] = []
    i = 0
    while i < len(pieces):
        if not _all_han(pieces[i].surface):
            out.append(pieces[i])
            i += 1
            continue
        j = i
        while j < len(pieces) and _all_han(pieces[j].surface):
            j += 1
        read = _read_chinese(pieces[i:j], hints)
        if read is None:
            out.extend(pieces[i:j])
        else:
            out.extend(read[0])
            notes.append(f"read as a Chinese name ({read[1]})")
        i = j
    return out


_POS_WITH_READINGS = ("名詞", "接尾辞", "接頭辞")


def _add_alternatives(pieces: list[_Piece], chunk: str) -> None:
    """Give every kanji token the other readings the dictionary has for it"""
    entries = _entries_by_surface(chunk)
    for piece in pieces:
        if piece.kind == _CHINESE or not any(_is_han(c) for c in piece.surface):
            continue
        for e in entries.get(piece.surface, ()):
            if e.pos1 not in _POS_WITH_READINGS:
                continue
            r = phonetic_reading(e.kana, e.pron, e.native)
            if r not in piece.readings:
                piece.readings.append(r)


# --------------------------------------------------------------------------- #
# Putting the words together
# --------------------------------------------------------------------------- #


@dataclass
class _Word:
    glue: str  # what goes before the word: " " or ""
    segments: list[list[_Piece]]  # joined with hyphens: Tōkyō-to
    capitalized: bool = True

    def render(self) -> str:
        out = "-".join(
            render_kana("".join(p.kana for p in seg)) for seg in self.segments
        )
        return capitalize_first(out) if self.capitalized else out


# Words that stand alone: a particle after one is a word of its own
_STANDALONE = frozenset({_KATAKANA, _LATIN, _SINO, _COMMON, _PROPER, _CHINESE})


def _particle_separates(prev: _Piece, nxt: _Piece) -> bool:
    """A particle stands alone between two words (霧のロンドン Kiri no
    Rondon, ゴルバツの村 Gorubatsu no Mura); after a single kanji it is
    inside a toponym (天の橋立 Amanohashidate)"""
    if nxt.kind in (_PUNCT, _PARTICLE, _BOUND, _SUFFIX, _GENERIC):
        return False
    if _all_hiragana(prev.surface) or _all_hiragana(nxt.surface):
        # in a name spelled in hiragana the analyzer finds particles that are
        # not there (しお|や|まち Shioyamachi)
        return False
    return prev.kind in _STANDALONE or _starts_word(prev, nxt)


def _divide(pieces: list[_Piece]) -> list[_Word]:
    words: list[_Word] = []
    prev: _Piece | None = None
    after_particle = False  # the previous piece was a particle standing alone
    for idx, piece in enumerate(pieces):
        if prev is None or _PUNCT in (prev.kind, piece.kind):
            words.append(_Word("", [[piece]]))
        elif after_particle:
            words.append(_Word(" ", [[piece]]))
        elif piece.kind == _PARTICLE:
            nxt = pieces[idx + 1] if idx + 1 < len(pieces) else None
            if nxt is not None and _particle_separates(prev, nxt):
                # between two words: a word of its own (Atenai no Akuroporisu)
                words.append(_Word(" ", [[piece]], capitalized=False))
                after_particle = True
                prev = piece
                continue
            words[-1].segments[-1].append(piece)
        elif _starts_word(prev, piece):
            words.append(_Word(" ", [[piece]]))
        elif piece.kind == _GENERIC:
            words[-1].segments.append([piece])
        else:
            words[-1].segments[-1].append(piece)
        after_particle = False
        prev = piece
    return words


_MAX_COMBINATIONS = 512


def _choose_readings(word: _Word, hints: _Hints, notes: list[str]) -> None:
    """Take the combination of readings that spells one of the hints, if it
    is clearly closer to it than the analyzer's own choice"""
    pieces = [p for seg in word.segments for p in seg]
    choosable = [p for p in pieces if len(p.readings) > 1]
    if not choosable:
        return
    options = [range(len(p.readings)) for p in choosable]
    size = 1
    for o in options:
        size *= len(o)
    if size > _MAX_COMBINATIONS:
        return

    def score(combo: tuple[int, ...]) -> tuple[tuple[int, int, int], str, str]:
        """(distance, distance of the whole word, readings changed), the
        hint, the text matched. The name may match without its
        generic term (Kunitachi-shi / Kunitachi); the whole word decides
        between readings of the generic term itself (Kiyomizu-dera / -ji)"""
        for p, c in zip(choosable, combo):
            p.choice = c
        text = word.render()
        d_word, hint = hints.distance(text)
        d, matched = d_word, text
        if len(word.segments) > 1:
            name = render_kana("".join(p.kana for p in word.segments[0]))
            d_name, hint_name = hints.distance(name)
            if d_name < d:
                d, hint, matched = d_name, hint_name, name
        return (d, d_word, sum(c > 0 for c in combo)), hint, matched

    default = tuple(0 for _ in choosable)
    default_score = score(default)[0]
    best_combo, best = default, None
    for combo in itertools.product(*options):
        if combo == default:
            continue
        scored = score(combo)
        if best is None or scored[0] < best[0]:
            best_combo, best = combo, scored
    # A reading is changed only if that makes the word, or its name part,
    # match a hint (an unrelated hint must not pick readings by chance: 州
    # stays shū, not su, for "Chiapas")
    accepted = False
    if best is not None:
        (d, d_word, _), _, matched = best
        if d < default_score[0]:
            accepted = d <= _allowed_distance(_japanese_key(matched))
        elif d == default_score[0] and d_word < default_score[1]:
            # the name matched already: another reading of its generic term
            # only if the hint spells it (Ishigaki-jima)
            accepted = d_word == 0
    final = best_combo if accepted else default
    for p, c in zip(choosable, final):
        p.choice = c
    if accepted:
        notes.append(f"reading chosen to match the entity's name ({best[1]!r})")


# Characters that separate words in the source: middle dot and double hyphen
# between parts of foreign names (ボスニア・ヘルツェゴビナ), ideographic space
_WORD_SEPARATORS = str.maketrans({"・": " ", "＝": " ", "゠": " ", "　": " "})

# Hiragana that look exactly like their katakana, and get typed for them
# inside katakana words (アピンへダム, Appingedam, with hiragana へ)
_LOOKALIKES = re.compile(r"(?<=[ァ-ヺー])[へべぺ](?=[ァ-ヺー])")


def _fix_lookalikes(text: str) -> str:
    return _LOOKALIKES.sub(lambda m: _to_katakana(m.group()), text)


def _romanize_chunk(chunk: str, hints: _Hints, notes: list[str]) -> tuple[str, bool]:
    """Romanize one space-free chunk: words separated by spaces, each
    capitalized; punctuation kept in place (東京-大阪 Tōkyō-Ōsaka).
    Returns the text and whether a reading was predicted (low confidence)"""
    pieces = _pieces(chunk)
    pieces = _apply_chinese(pieces, hints, notes)
    for piece in pieces:
        if not piece.kana and any(_is_han(c) for c in piece.surface):
            raise NoRomanization(f"no Japanese reading for {piece.surface!r}")
    _mark_generics(pieces)
    if hints.texts:
        _add_alternatives(pieces, chunk)
    words = _divide(pieces)
    if hints.texts:
        for word in words:
            _choose_readings(word, hints, notes)
    text = "".join(w.glue + w.render() for w in words)
    return text, any(p.guessed for p in pieces)


def romanize_japanese(text: str, ctx: Context) -> Rendering | None:
    text = _fix_lookalikes(" ".join(text.translate(_WORD_SEPARATORS).split()))
    has_han = any(_is_han(c) for c in text)
    try:
        _tagger()
    except ProviderUnavailable as exc:
        if has_han:
            # Any other reader would give Chinese readings: keep the kanji
            raise NoRomanization(f"{exc}; kanji kept") from exc
        # Kana only: readable without the analyzer, but words are divided
        # only where the source has spaces
        words = [capitalize_first(kana_to_hepburn(w)) for w in text.split(" ")]
        return Rendering(" ".join(words), "nte:hepburn", "high")

    hints = _Hints.of(ctx.hints)
    notes: list[str] = []
    rendered = [_romanize_chunk(chunk, hints, notes) for chunk in text.split(" ")]
    out = " ".join(r for r, _ in rendered)
    if not has_han:
        # Kana spellings are exact; the analyzer only divides the words
        return Rendering(out, "nte:hepburn", "high", engine=_engine())
    guessed = any(g for _, g in rendered)
    return Rendering(
        out,
        "nte:hepburn+unidic",
        "low" if guessed else "normal",
        tuple(dict.fromkeys(notes)),
        engine=_engine(),
    )
