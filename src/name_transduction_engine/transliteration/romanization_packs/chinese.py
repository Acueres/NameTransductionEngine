"""Chinese: Hanyu Pinyin as used for place names (GB/T 16159, UN 1977).

- toneless, syllables of a name joined into one word: Beijing, Xiamen
- apostrophe before a/o/e starting a non-initial syllable: Xi'an, Lu'an
- administrative generics written apart and capitalized: Hebei Sheng,
  Zhangjiakou Shi, Guangxi Zhuangzu Zizhiqu
Readings of characters with more than one reading come from pypinyin's phrase
dictionary (厦门 Xiamen, 重庆 Chongqing, 六安 Lu'an).
"""

from functools import lru_cache

from .base import (
    Context,
    NoRomanization,
    ProviderUnavailable,
    Rendering,
    package_version,
)
from .text import capitalize_first

# Longest first. Only administrative generics: natural-feature generics (山,
# 江, 湖) are also the last syllable of many city names (中山 Zhongshan)
_GENERICS: tuple[str, ...] = (
    "特别行政区", "特別行政區", "自治区", "自治區", "自治州", "自治县", "自治縣",
    "自治旗", "地区", "地區", "省", "市", "县", "縣", "区", "區", "镇", "鎮",
    "乡", "鄉", "村", "旗", "盟",
)  # fmt: skip

_SINITIC = frozenset(
    {
        "zh",
        "yue",
        "wuu",
        "hak",
        "nan",
        "gan",
        "hsn",
        "cdo",
        "cjy",
        "cpx",
        "czh",
        "mnp",
        "lzh",
    }
)


@lru_cache(maxsize=1)
def _pypinyin():
    try:
        import pypinyin  # noqa: PLC0415 - optional dependency, loaded on demand
    except ImportError as exc:
        raise ProviderUnavailable("Chinese readings need pypinyin") from exc
    return pypinyin


@lru_cache(maxsize=1)
def _engine() -> str:
    return package_version("pypinyin")


def romanize_chinese(text: str, ctx: Context) -> Rendering | None:
    if ctx.lang is not None and ctx.lang not in _SINITIC:
        # Han in a Korean or Vietnamese name: no reader for those readings
        raise NoRomanization(f"no reading of Han characters for language {ctx.lang!r}")

    pypinyin = _pypinyin()
    warnings: tuple[str, ...] = ()
    if ctx.lang not in (None, "zh", "lzh"):
        warnings += (f"{ctx.lang!r} name romanized with Standard Mandarin readings",)
    if ctx.region in ("TW", "HK", "MO") or ctx.script == "Hant" and ctx.region is None:
        warnings += (
            "established local spellings may differ (Wade–Giles in Taiwan, "
            "Cantonese in Hong Kong and Macau)",
        )

    words = [_romanize_word(pypinyin, w) for w in text.split(" ")]
    return Rendering(
        " ".join(words), "nte:pinyin", "normal", warnings, engine=_engine()
    )


def _romanize_word(pypinyin, word: str) -> str:
    if not word:
        return word
    specific, generic = _split_generic(word)
    parts = []
    if generic.startswith("自治") and len(specific) >= 4 and specific.endswith("族"):
        # The people named in an autonomous area is its own word:
        # 广西壮族自治区 Guangxi Zhuangzu Zizhiqu
        parts.append(_pinyin_word(pypinyin, specific[:-2]))
        specific = specific[-2:]
    parts.append(_pinyin_word(pypinyin, specific))
    if generic:
        parts.append(_pinyin_word(pypinyin, generic))
    return " ".join(p for p in parts if p)


def _split_generic(word: str) -> tuple[str, str]:
    for generic in _GENERICS:
        # Split only when at least two characters remain: 沙市 is Shashi
        if word.endswith(generic) and len(word) - len(generic) >= 2:
            return word[: -len(generic)], generic
    return word, ""


def _pinyin_word(pypinyin, word: str) -> str:
    syllables = pypinyin.lazy_pinyin(
        word, style=pypinyin.Style.NORMAL, v_to_u=True, errors=_keep
    )
    out = ""
    for syl in syllables:
        if not syl:
            continue
        if any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in syl):
            raise NoRomanization(f"no reading for {syl!r}")
        if out and out[-1].isalpha() and syl[0] in "aoe":
            out += "'"
        out += syl
    return capitalize_first(out)


def _keep(chars: str) -> list[str]:
    # Non-Han characters (digits, Latin, punctuation) pass through as one piece
    return [chars]
