"""Japanese: modified Hepburn with macrons (Tōkyō, Ōsaka-fu, Shinjuku-ku).

Kana is converted directly. Kanji needs readings, which come from the
MeCab/UniDic analyzer (fugashi + unidic-lite). Without it, or when a kanji
has no reading, the native form is kept: a Chinese reading of a Japanese name
would be wrong, not merely rough.

The analyzer also divides the name into words (Higashi Rōma Teikoku) while
keeping Japanese toponyms whole (Nihonbashi); see "Word division" below.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from .base import (
    Context,
    NoRomanization,
    ProviderUnavailable,
    Rendering,
    package_version,
)
from .text import capitalize_first

# --------------------------------------------------------------------------- #
# Kana to modified Hepburn
# --------------------------------------------------------------------------- #

_DIGRAPHS = {
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo", "シャ": "sha", "シュ": "shu",
    "ショ": "sho", "チャ": "cha", "チュ": "chu", "チョ": "cho", "ニャ": "nya",
    "ニュ": "nyu", "ニョ": "nyo", "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo",
    "ミャ": "mya", "ミュ": "myu", "ミョ": "myo", "リャ": "rya", "リュ": "ryu",
    "リョ": "ryo", "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo", "ジャ": "ja",
    "ジュ": "ju", "ジョ": "jo", "ヂャ": "ja", "ヂュ": "ju", "ヂョ": "jo",
    "ビャ": "bya", "ビュ": "byu", "ビョ": "byo", "ピャ": "pya", "ピュ": "pyu",
    "ピョ": "pyo",
    # Extended katakana for loanwords
    "シェ": "she", "ジェ": "je", "チェ": "che", "ティ": "ti", "ディ": "di",
    "トゥ": "tu", "ドゥ": "du", "デュ": "dyu", "テュ": "tyu", "ファ": "fa",
    "フィ": "fi", "フェ": "fe", "フォ": "fo", "フュ": "fyu", "ウィ": "wi",
    "ウェ": "we", "ウォ": "wo", "ヴァ": "va", "ヴィ": "vi", "ヴェ": "ve",
    "ヴォ": "vo", "ヴュ": "vyu", "ツァ": "tsa", "ツィ": "tsi", "ツェ": "tse",
    "ツォ": "tso", "イェ": "ye", "クァ": "kwa", "クィ": "kwi", "クェ": "kwe",
    "クォ": "kwo", "グァ": "gwa",
}  # fmt: skip

_MONOGRAPHS = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヰ": "i", "ヱ": "e", "ヲ": "o", "ン": "n",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "ヴ": "vu", "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
    "ャ": "ya", "ュ": "yu", "ョ": "yo", "ヮ": "wa", "ヵ": "ka", "ヶ": "ke",
}  # fmt: skip

_SOKUON = "ッ"
_CHOON = "ー"
_MACRON = {"a": "ā", "i": "ī", "u": "ū", "e": "ē", "o": "ō"}


def _to_katakana(text: str) -> str:
    # Hiragana U+3041..U+3096 sit exactly 0x60 below their katakana
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def _morae(kana: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(kana):
        two = kana[i : i + 2]
        if two in _DIGRAPHS:
            out.append(two)
            i += 2
        else:
            out.append(kana[i])
            i += 1
    return out


def kana_to_hepburn(kana: str) -> str:
    """Katakana or hiragana to modified Hepburn: long vowels with macrons
    (oo, ou -> ō; uu -> ū; aa -> ā; ee -> ē; ー), ii and ei kept, syllabic n
    as "n" with an apostrophe before a vowel or y, っ doubling (っち -> tchi)"""
    morae = _morae(_to_katakana(kana))
    out: list[str] = []
    geminate = False
    for idx, mora in enumerate(morae):
        if mora == _SOKUON:
            geminate = True
            continue
        if mora == _CHOON:
            if out and out[-1] and out[-1][-1] in _MACRON:
                out[-1] = out[-1][:-1] + _MACRON[out[-1][-1]]
            continue

        roman = _DIGRAPHS.get(mora) or _MONOGRAPHS.get(mora)
        if roman is None:
            out.append(mora)  # not kana: pass through (the caller checks)
            geminate = False
            continue

        # Vowel length: a bare vowel mora lengthening the previous one
        prev = out[-1] if out else ""
        if (
            prev
            and roman in ("u", "o")
            and prev[-1] == "o"
            or (prev and roman == "u" and prev[-1] == "u")
            or (prev and roman == "a" and prev[-1] == "a")
            or (prev and roman == "e" and prev[-1] == "e")
        ):
            out[-1] = prev[:-1] + _MACRON[prev[-1]]
            continue

        if mora == "ン":
            nxt = morae[idx + 1] if idx + 1 < len(morae) else ""
            nxt_roman = _DIGRAPHS.get(nxt) or _MONOGRAPHS.get(nxt) or ""
            if nxt_roman[:1] in ("a", "i", "u", "e", "o", "y"):
                roman = "n'"

        if geminate:
            roman = ("t" + roman) if roman.startswith("ch") else (roman[0] + roman)
            geminate = False
        out.append(roman)
    return "".join(out)


# --------------------------------------------------------------------------- #
# Readings for kanji
# --------------------------------------------------------------------------- #

# Administrative suffixes, written after a hyphen: Tōkyō-to, Yokohama-shi
_ADMIN_SUFFIXES = frozenset("都道府県市区町村郡")


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
        or "豈" <= ch <= "﫿"
        or "\U00020000" <= ch <= "\U0003134f"
        or ch in "々〆ヶ"
    )


# Characters that separate words in the source: middle dot and double hyphen
# between parts of foreign names (ボスニア・ヘルツェゴビナ), ideographic space
_WORD_SEPARATORS = str.maketrans({"・": " ", "＝": " ", "゠": " ", "\u3000": " "})


def romanize_japanese(text: str, ctx: Context) -> Rendering | None:
    text = " ".join(text.translate(_WORD_SEPARATORS).split())
    has_han = any(_is_han(c) for c in text)
    try:
        tagger = _tagger()
    except ProviderUnavailable as exc:
        if has_han:
            # Any other reader would give Chinese readings: keep the kanji
            raise NoRomanization(f"{exc}; kanji kept") from exc
        # Kana only: readable without the analyzer, but words are divided
        # only where the source has spaces
        words = [_romanize_kana_word(w) for w in text.split(" ")]
        return Rendering(" ".join(words), "nte:hepburn", "high")

    out = " ".join(_romanize_chunk(tagger, chunk) for chunk in text.split(" "))
    if not has_han:
        # Kana spellings are exact; the analyzer only divides the words
        return Rendering(out, "nte:hepburn", "high", engine=_engine())
    return Rendering(out, "nte:hepburn+unidic", "normal", engine=_engine())


def _romanize_kana_word(word: str) -> str:
    return capitalize_first(kana_to_hepburn(word)) if word else word


# --------------------------------------------------------------------------- #
# Word division
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
#   Afurika). Katakana next to katakana stays one word: UniDic splits unknown
#   foreign names into pieces (ス|ファックス for Sfax, レンツ|ブルク for
#   Rendsburg), and Japanese writes loanword compounds without spaces anyway
#   (スーパーアリーナ Sūpāarīna)
# - a Sino-Japanese common noun of two or more kanji starts a word: Teikoku,
#   Kokusai Kūkō, Tōhoku Chihō
# - after a proper noun, any common noun of two or more characters starts a
#   word (Fushimi Inari Taisha), and so does another proper noun (Nikkō
#   Tōshōgū, Tōkyō-to Chiyoda-ku)
# - prefixes, suffixes, particles and single-kanji common nouns attach:
#   Dainippon, Kyōwakoku, Toshokan, Higashinippon, Amanohashidate
# - administrative suffixes attach with a hyphen: Tōkyō-to
#
# The kinds below are what the rules need from a token.

_PREFIX = "prefix"
_SUFFIX = "suffix"
_ADMIN = "admin"
_PROPER = "proper"
_COMMON = "common"  # common noun of 2+ characters, native or mixed origin
_SINO = "sino"  # Sino-Japanese common noun of 2+ characters
_SHORT = "short"  # single-character common noun, numeral, counter
_KATAKANA = "katakana"
_LATIN = "latin"
_BOUND = "bound"  # particles and other non-nouns: never split off
_PUNCT = "punct"  # hyphens, brackets: kept, and the next word is capitalized


@dataclass(frozen=True)
class _Piece:
    surface: str
    kana: str  # reading in katakana, "" for non-Japanese text
    kind: str


def _is_katakana_word(text: str) -> bool:
    return len(text) > 1 and all("ァ" <= c <= "ヺ" or c == _CHOON for c in text)


def _kind(tok, first: bool) -> str:
    surface = tok.surface
    f = tok.feature
    if not first and surface in _ADMIN_SUFFIXES:
        return _ADMIN
    if f.pos1 == "接頭辞":
        return _PREFIX
    if f.pos1 == "接尾辞":
        return _SUFFIX
    if surface.isascii() and any(c.isalpha() for c in surface):
        return _LATIN
    if f.pos1 in ("補助記号", "記号") and not any(
        _is_han(c) or _is_kana(c) for c in surface
    ):
        return _PUNCT
    if _is_katakana_word(surface):
        return _KATAKANA  # loanwords of any part of speech: ビッグ, ユニバーサル
    if f.pos1 != "名詞":
        return _BOUND
    if f.pos2 == "固有名詞":
        return _PROPER
    if f.pos2 == "普通名詞" and f.pos3 != "助数詞可能" and len(surface) > 1:
        return _SINO if f.goshu == "漢" else _COMMON
    return _SHORT


def _all_katakana(text: str) -> bool:
    return bool(text) and all("ァ" <= c <= "ヺ" or c == _CHOON for c in text)


def _starts_word(prev: _Piece, cur: _Piece) -> bool:
    if cur.kind in (_SUFFIX, _ADMIN, _BOUND) or prev.kind in (_PREFIX, _BOUND):
        return False
    if _all_katakana(prev.surface) and _all_katakana(cur.surface):
        return False
    if _LATIN in (prev.kind, cur.kind) or cur.kind == _KATAKANA:
        return True
    if prev.kind == _KATAKANA:
        # a katakana name followed by a kanji word; a single kanji attaches
        return cur.kind != _SHORT
    if cur.kind == _SINO:
        return True
    if cur.kind == _COMMON:
        return prev.kind == _PROPER
    if cur.kind == _PROPER:
        return prev.kind in (_PROPER, _SUFFIX, _ADMIN)
    return False


# A katakana run of two or more characters inside a mixed token: 南|アフリカ.
# Single ones are particles or linking kana inside toponyms (御茶ノ水, 関ケ原).
_KATAKANA_RUN = re.compile(r"[ァ-ヺー]{2,}")


def _split_mixed(surface: str, kana: str, kind: str) -> list[_Piece]:
    """Split a token where kanji meet a katakana name (南アフリカ), sharing
    out the reading by matching the katakana part literally"""
    runs = [m for m in _KATAKANA_RUN.finditer(surface) if m.group() != surface]
    if not runs or kind not in (_PROPER, _COMMON, _SINO, _SHORT):
        return [_Piece(surface, kana, kind)]
    parts: list[str] = []
    pattern = ""
    pos = 0
    for m in runs:
        if m.start() > pos:
            parts.append(surface[pos : m.start()])
            pattern += "(.+)"
        parts.append(m.group())
        pattern += f"({re.escape(m.group())})"
        pos = m.end()
    if pos < len(surface):
        parts.append(surface[pos:])
        pattern += "(.+)"
    aligned = re.fullmatch(pattern, _to_katakana(kana))
    if aligned is None:
        return [_Piece(surface, kana, kind)]
    return [
        _Piece(part, reading, _KATAKANA if _is_katakana_word(part) else kind)
        for part, reading in zip(parts, aligned.groups())
    ]


def _pieces(tagger, chunk: str) -> list[_Piece]:
    pieces: list[_Piece] = []
    for i, tok in enumerate(tagger(chunk)):
        surface = tok.surface
        reading = tok.feature.kana
        if i > 0 and surface == "山" and reading == "ヤマ":
            # 山 closing a mountain's name reads san: 富士山 Fujisan
            reading = "サン"
        if any(_is_han(c) for c in surface):
            if not reading or reading == "*":
                raise NoRomanization(f"no Japanese reading for {surface!r}")
            kana = reading
        elif any(_is_kana(c) for c in surface):
            kana = surface
        else:
            kana = ""  # digits, Latin
        pieces.extend(_split_mixed(surface, kana, _kind(tok, first=i == 0)))
    return pieces


_VOWEL_START = tuple("aiueoāīūēōy")


def _romanize_chunk(tagger, chunk: str) -> str:
    """Romanize one space-free chunk: words separated by spaces, each
    capitalized; punctuation kept in place (東京-大阪 Tōkyō-Ōsaka)"""
    words: list[list[str]] = []  # romanized pieces of each word
    glue: list[str] = []  # what goes before each word: " " or ""
    prev: _Piece | None = None
    for piece in _pieces(tagger, chunk):
        roman = kana_to_hepburn(piece.kana) if piece.kana else piece.surface
        if prev is None or _PUNCT in (prev.kind, piece.kind):
            words.append([roman])
            glue.append("")
        elif _starts_word(prev, piece):
            words.append([roman])
            glue.append(" ")
        elif piece.kind == _ADMIN:
            words[-1].append("-" + roman)
        else:
            # Syllabic n before a vowel or y across the token boundary:
            # 新大阪 Shin'ōsaka
            if prev.kana.endswith(("ン", "ん")) and roman.startswith(_VOWEL_START):
                roman = "'" + roman
            words[-1].append(roman)
        prev = piece
    return "".join(g + capitalize_first("".join(w)) for g, w in zip(glue, words))
