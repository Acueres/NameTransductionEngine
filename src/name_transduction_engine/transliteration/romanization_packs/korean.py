"""Korean: Revised Romanization (South Korea, 2000) with its sound-change
rules, and McCune-Reischauer for names tagged as North Korean (region KP).

RR follows pronunciation between syllables: liaison (설악 Seorak), nasal
assimilation (종로 Jongno, 왕십리 Wangsimni, 압록 Amnok), lateralization
(신라 Silla, 별내 Byeollae), aspiration by ㅎ (좋고 joko) and palatalization
(같이 gachi). As RR prescribes for nouns, ㄱ/ㄷ/ㅂ + ㅎ keeps the h (묵호
Mukho). Administrative units are hyphenated and no sound change is applied
across the hyphen (종로구 Jongno-gu, 삼죽면 Samjuk-myeon). Not modeled: n
insertion (학여울 is Hangnyeoul, this gives Hagyeoul) and tensification, which
RR does not write anyway.
"""

import unicodedata

from . import icu_util
from .base import Context, NoRomanization, Rendering
from .text import capitalize_first, titlecase_words

_INITIALS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]  # fmt: skip
_VOWELS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]  # fmt: skip
# Coda as pronounced before a pause or a consonant with no special rule
_CODA = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t"]  # fmt: skip

# Final jamo indices
F_NONE, F_G, F_GG, F_GS, F_N, F_NJ, F_NH, F_D, F_L, F_LG, F_LM, F_LB, F_LS, F_LT, F_LP, F_LH, F_M, F_B, F_BS, F_S, F_SS, F_NG, F_J, F_CH, F_K, F_T, F_P, F_H = range(28)  # fmt: skip
# Initial jamo indices
I_G, I_GG, I_N, I_D, I_DD, I_R, I_M, I_B, I_BB, I_S, I_SS, I_NULL, I_J, I_JJ, I_CH, I_K, I_T, I_P, I_H = range(19)  # fmt: skip
V_I = 20  # ㅣ

# Final consonant moving to the next syllable's onset before a vowel:
# (what stays as coda, the onset it becomes)
_LIAISON = {
    F_G: ("", "g"), F_GG: ("", "kk"), F_GS: ("k", "s"), F_N: ("", "n"),
    F_NJ: ("n", "j"), F_NH: ("", "n"), F_D: ("", "d"), F_L: ("", "r"),
    F_LG: ("l", "g"), F_LM: ("l", "m"), F_LB: ("l", "b"), F_LS: ("l", "s"),
    F_LT: ("l", "t"), F_LP: ("l", "p"), F_LH: ("", "r"), F_M: ("", "m"),
    F_B: ("", "b"), F_BS: ("p", "s"), F_S: ("", "s"), F_SS: ("", "ss"),
    F_NG: ("ng", ""), F_J: ("", "j"), F_CH: ("", "ch"), F_K: ("", "k"),
    F_T: ("", "t"), F_P: ("", "p"), F_H: ("", ""),
}  # fmt: skip

_ASPIRATED = {I_G: "k", I_D: "t", I_J: "ch"}
_NASAL_OF = {"k": "ng", "t": "n", "p": "m"}

# Administrative units written after a hyphen, longest first
_UNITS = (
    "특별자치도", "특별자치시", "특별시", "광역시", "도", "시", "군", "구", "읍",
    "면", "리", "동", "가", "로", "길",
)  # fmt: skip
# 도 is also the "island" of 독도 Dokdo and 울릉도 Ulleungdo: hyphenate it only
# after a province name
_PROVINCES = frozenset(
    {
        "경기", "강원", "충청북", "충청남", "전라북", "전라남", "경상북", "경상남",
        "제주", "황해", "황해북", "황해남", "평안북", "평안남", "함경북", "함경남",
        "자강", "양강", "강원특별자치", "전북특별자치",
    }
)  # fmt: skip
# One-syllable district names: 중구 Jung-gu, 서구 Seo-gu
_SHORT_DISTRICTS = frozenset("중동서남북")


def romanize_korean(text: str, ctx: Context) -> Rendering | None:
    if any("一" <= c <= "鿿" for c in text):
        raise NoRomanization("no Korean reading of Han characters (hanja)")

    if ctx.region == "KP":
        # ICU's Korean BGN transform is McCune-Reischauer and needs jamo
        out = icu_util.apply("ko-ko_Latn/BGN", unicodedata.normalize("NFD", text))
        return Rendering(
            titlecase_words(out),
            "icu:ko-ko_Latn/BGN",
            "high",
            engine=icu_util.ICU_ENGINE,
        )

    words = [_romanize_word(w) for w in text.split(" ")]
    return Rendering(" ".join(words), "nte:revised-romanization", "high")


def _romanize_word(word: str) -> str:
    if not word:
        return word
    specific, unit = _split_unit(word)
    out = capitalize_first(_rr(specific))
    if unit:
        out += "-" + _rr(unit)
    return out


def _split_unit(word: str) -> tuple[str, str]:
    for unit in _UNITS:
        if not word.endswith(unit) or len(word) == len(unit):
            continue
        specific = word[: -len(unit)]
        if unit == "도":
            return (specific, unit) if specific in _PROVINCES else (word, "")
        if len(specific) >= 2 or (unit == "구" and specific in _SHORT_DISTRICTS):
            return specific, unit
    return word, ""


def _rr(text: str) -> str:
    """Revised Romanization of a run of text with sound changes between
    adjacent Hangul syllables; other characters pass through"""
    syllables: list[tuple[int, int, int] | str] = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            syllables.append((code // 588, (code % 588) // 28, code % 28))
        else:
            syllables.append(ch)

    out: list[str] = []
    for i, syl in enumerate(syllables):
        if isinstance(syl, str):
            out.append(syl)
            continue
        initial, vowel, final = syl
        prev = syllables[i - 1] if i > 0 else None
        nxt = syllables[i + 1] if i + 1 < len(syllables) else None

        onset = _INITIALS[initial] if initial != I_R else "r"
        if isinstance(prev, tuple):
            onset = _onset_after(prev[2], initial, vowel)
        elif initial == I_R:
            onset = "r"

        coda = _CODA[final]
        if isinstance(nxt, tuple):
            coda = _coda_before(final, nxt[0])
        out.append(onset + _VOWELS[vowel] + coda)
    return "".join(out)


def _coda_before(final: int, initial: int) -> str:
    """How the coda of a syllable is written before the given initial"""
    if final == F_NONE:
        return ""
    if initial == I_NULL:
        return _LIAISON[final][0]
    if final in (F_H, F_NH, F_LH):
        if initial in _ASPIRATED or initial == I_S:
            return {F_H: "", F_NH: "n", F_LH: "l"}[final]
        if initial == I_N:
            return {F_H: "n", F_NH: "n", F_LH: "l"}[final]
    coda = _CODA[final]
    if initial in (I_N, I_M) or (initial == I_R and final not in (F_N, F_L)):
        if coda == "l" and initial == I_N:
            return "l"  # 별내 Byeollae
        if coda == "l" and final in (F_LB,):
            return "m"  # ㄼ before a nasal patterns as ㅂ (밟는 bamneun)
        return _NASAL_OF.get(coda, coda)
    if initial == I_R and final in (F_N, F_L):
        return "l"  # 신라 Silla, 달리 dalli
    return coda


def _onset_after(final: int, initial: int, vowel: int) -> str:
    """How an initial is written after the given coda"""
    if initial == I_NULL:
        if final == F_NONE:
            return ""
        onset = _LIAISON[final][1]
        if vowel == V_I:
            # Palatalization: 굳이 guji, 같이 gachi
            onset = {"d": "j", "t": "ch"}.get(onset, onset)
        return onset
    if final in (F_H, F_NH, F_LH):
        if initial in _ASPIRATED:
            return _ASPIRATED[initial]
        if initial == I_N:
            return "n"
        if initial == I_S:
            return "ss"
    if initial == I_R:
        if final in (F_N, F_L, F_LH):
            return "l"
        return "r" if final == F_NONE else "n"  # 종로 Jongno
    if initial == I_N and _CODA[final] == "l":
        return "l"  # 별내 Byeollae
    return _INITIALS[initial]
