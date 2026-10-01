"""Thai: Royal Thai General System of Transcription (RTGS), as on Thai road
signs (Chiang Mai, Khon Kaen, Hat Yai, Udon Thani).

Thai writes vowels around consonants and leaves some unwritten. A word is
first split into syllables that carry a written vowel (with their onset,
which may be a true cluster or have a silent leading ห), and runs of
consonants with no written vowel between them. Each run is then divided by
rule:
- one consonant closes the previous syllable (ขอนแก่น Khon Kaen)
- two: open "Co" + consonant at the start or after an open vowel (ชลบุรี
  Chon Buri, อุดร Udon), closing + "Ca" after a vowel that needs a final
  (จันทบุรี Chanthaburi)
- three: "Ca" + "CoC" at the end of a word (นคร Nakhon), closing + "Ca" + "Ca"
  before more syllables (กาญจนบุรี Kanchanaburi)
Thai spells no word boundaries, so the syllables of one word are joined:
the result is "Chiangmai" where signs write "Chiang Mai". Pali and Sanskrit
compounds linked by an unwritten vowel (ราชสีมา Ratchasima) lose it. Hence
low confidence.
"""

from dataclasses import dataclass

from .base import Context, Rendering
from .text import capitalize_first

_INITIAL = {
    "ก": "k", "ข": "kh", "ฃ": "kh", "ค": "kh", "ฅ": "kh", "ฆ": "kh", "ง": "ng",
    "จ": "ch", "ฉ": "ch", "ช": "ch", "ซ": "s", "ฌ": "ch", "ญ": "y", "ฎ": "d",
    "ฏ": "t", "ฐ": "th", "ฑ": "th", "ฒ": "th", "ณ": "n", "ด": "d", "ต": "t",
    "ถ": "th", "ท": "th", "ธ": "th", "น": "n", "บ": "b", "ป": "p", "ผ": "ph",
    "ฝ": "f", "พ": "ph", "ฟ": "f", "ภ": "ph", "ม": "m", "ย": "y", "ร": "r",
    "ล": "l", "ว": "w", "ศ": "s", "ษ": "s", "ส": "s", "ห": "h", "ฬ": "l",
    "อ": "", "ฮ": "h",
}  # fmt: skip
_FINAL = {
    "ก": "k", "ข": "k", "ค": "k", "ฆ": "k", "ง": "ng", "จ": "t", "ช": "t",
    "ซ": "t", "ฌ": "t", "ญ": "n", "ฎ": "t", "ฏ": "t", "ฐ": "t", "ฑ": "t",
    "ฒ": "t", "ณ": "n", "ด": "t", "ต": "t", "ถ": "t", "ท": "t", "ธ": "t",
    "น": "n", "บ": "p", "ป": "p", "พ": "p", "ฟ": "p", "ภ": "p", "ม": "m",
    "ร": "n", "ล": "n", "ฬ": "n", "ศ": "t", "ษ": "t", "ส": "t", "ย": "i",
    "ว": "o", "อ": "",
}  # fmt: skip

_CONSONANTS = frozenset(_INITIAL)
_PRE = frozenset("เแโไใ")
_MAI_HAN_AKAT, _MAITAIKHU = "ั", "็"
_ABOVE_BELOW = frozenset("ัิีึืุู็")
_FOLLOW = frozenset("ะาำๅ")  # ะ า ำ ๅ
_MARKS = _ABOVE_BELOW | _FOLLOW
_TONES = frozenset("่้๊๋ํ")
_KARAN = "์"
_MAI_YAMOK, _PAIYANNOI = "ๆ", "ฯ"

# True clusters (both consonants pronounced)
_CLUSTERS = frozenset(
    {
        "กร", "กล", "กว", "ขร", "ขล", "ขว", "คร", "คล", "คว", "ปร", "ปล", "พร",
        "พล", "ผล", "ตร", "บร", "บล", "ดร", "ฟร", "ฟล",
    }
)  # fmt: skip
# ร written but not pronounced as r
_FALSE_CLUSTERS = {"ทร": "s", "จร": "ch", "ซร": "s", "ศร": "s", "สร": "s"}
# Leading ห: silent, marks the tone of the sonorant after it
_LEADING_H = frozenset("งญนมยรลว")

# Stops after which a written ร is silent (สมุทร samut, จักร chak, บุตร but)
_SILENT_R_AFTER = frozenset("กจชตทฏ")

# Vowels ending in a glide or with an inherent final: nothing can follow
_OPEN_ONLY = frozenset(
    {"ai", "ao", "am", "oi", "ui", "oei", "io", "eo", "aeo", "uai", "iao"}
)


@dataclass
class _Syl:
    onset: str
    vowel: str
    needs_coda: bool = False  # ั and ็ always close their syllable
    open_only: bool = False  # a glide vowel, or a short vowel written with ะ


def romanize_thai(text: str, ctx: Context) -> Rendering | None:
    words: list[str] = []
    for word in text.replace(_PAIYANNOI, "").split(" "):
        if word == _MAI_YAMOK and words:
            words.append(words[-1])
            continue
        repeat = word.endswith(_MAI_YAMOK)
        roman = _romanize_word(word.rstrip(_MAI_YAMOK))
        words.append(roman)
        if repeat:
            words.append(roman)
    return Rendering(
        " ".join(w for w in words if w),
        "nte:rtgs",
        "low",
        ("unwritten vowels restored by rule",),
    )


def _romanize_word(word: str) -> str:
    s = _silence(word)
    items = _segment(s)
    return capitalize_first(_render(items))


def _silence(word: str) -> list[str]:
    """Drop tone marks, and letters silenced by a karan (with a vowel mark
    written on them: ศักดิ์ sak)"""
    chars = [c for c in word if c not in _TONES]
    out: list[str] = []
    for ch in chars:
        if ch == _KARAN:
            while out and out[-1] in _ABOVE_BELOW:
                out.pop()
            if out and out[-1] in _CONSONANTS:
                out.pop()
                # A silent cluster after a final: สุราษฎร์ surat, จันทร์ chan
                if (
                    len(out) >= 2
                    and out[-1] in _CONSONANTS
                    and out[-2] in _CONSONANTS
                    and out[-2:] != ["ร", "ร"]  # สวรรค์ sawan keeps its รร
                ):
                    out.pop()
            continue
        out.append(ch)
    return out


def _pair(c1: str, c2: str) -> str | None:
    """Onset of the pair c1+c2 when they form one, else None"""
    pair = c1 + c2
    if pair in _CLUSTERS:
        return _INITIAL[c1] + _INITIAL[c2]
    if pair in _FALSE_CLUSTERS:
        return _FALSE_CLUSTERS[pair]
    if c1 == "ห" and c2 in _LEADING_H:
        return _INITIAL[c2]
    return None


def _segment(s: list[str]) -> list[_Syl | str]:
    """Syllables with a written vowel, and loose consonants/other characters
    as plain strings"""
    items: list[_Syl | str] = []
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch in _PRE:
            j = i + 1
            onset = ""
            if j < n and s[j] in _CONSONANTS:
                onset = _INITIAL[s[j]]
                j += 1
                if j < n and s[j] in _CONSONANTS:
                    joined = _pair(s[j - 1], s[j])
                    if joined is not None:
                        onset = joined
                        j += 1
            marks, j = _take_marks(s, j)
            vowel, j, needs = _vowel(ch, marks, s, j)
            items.append(_Syl(onset, vowel, needs, _is_open(vowel, marks)))
            i = j
            continue

        if ch == "ฤ" or ch == "ฦ":
            # After a consonant: r/l + i (อังกฤษ angkrit); alone: rue/lue
            letter = "r" if ch == "ฤ" else "l"
            if items and isinstance(items[-1], str) and items[-1] in _CONSONANTS:
                prev = items.pop()
                items.append(_Syl(_INITIAL[prev] + letter, "i"))  # type: ignore[index]
            else:
                items.append(_Syl(letter, "ue"))
            i += 1
            continue

        if ch in _CONSONANTS:
            nxt = s[i + 1] if i + 1 < n else ""
            after = s[i + 2] if i + 2 < n else ""
            if nxt == "ร" and after == "ร":
                # รร reads "a" with the next consonant as final (ธรรม tham),
                # or "an" with none (สวรรค์ sawan)
                j = i + 3
                if (
                    j < n
                    and s[j] in _CONSONANTS
                    and (j + 1 >= n or s[j + 1] not in _MARKS)
                ):
                    items.append(
                        _Syl(
                            _onset_with_previous(items, ch),
                            "a" + _FINAL.get(s[j], ""),
                            open_only=True,
                        )
                    )
                    i = j + 1
                else:
                    items.append(
                        _Syl(_onset_with_previous(items, ch), "an", open_only=True)
                    )
                    i = j
                continue
            if nxt in _MARKS:
                marks, j = _take_marks(s, i + 1)
                vowel, j, needs = _vowel("", marks, s, j)
                items.append(
                    _Syl(
                        _onset_with_previous(items, ch),
                        vowel,
                        needs,
                        _is_open(vowel, marks),
                    )
                )
                i = j
                continue
            if nxt == "อ" and after not in _MARKS:
                # C+อ: vowel "o" (ขอน khon); C+อ+ย: "oi"
                if after == "ย":
                    items.append(_Syl(_onset_with_previous(items, ch), "oi"))
                    i += 3
                else:
                    items.append(_Syl(_onset_with_previous(items, ch), "o"))
                    i += 2
                continue
            if (
                nxt == "ว"
                and after in _CONSONANTS
                and (i + 3 >= n or s[i + 3] not in _MARKS)
                and not (after == "ร" and i + 3 < n and s[i + 3] == "ร")  # สวรรค์ sawan
            ):
                # C+ว+final: vowel "ua" (สวน suan)
                items.append(_Syl(_onset_with_previous(items, ch), "ua"))
                i += 2
                continue
            items.append(ch)
            i += 1
            continue

        items.append(ch)
        i += 1
    return items


def _is_open(vowel: str, marks: str) -> bool:
    return vowel in _OPEN_ONLY or "\u0e30" in marks


def _onset_with_previous(items: list[_Syl | str], ch: str) -> str:
    """This consonant's onset, merged with a loose consonant just before it
    when the two form a cluster or a leading ห"""
    if items and isinstance(items[-1], str) and items[-1] in _CONSONANTS:
        joined = _pair(items[-1], ch)
        if joined is not None:
            items.pop()
            return joined
    return _INITIAL[ch]


def _take_marks(s: list[str], j: int) -> tuple[str, int]:
    marks = ""
    while j < len(s) and s[j] in _MARKS:
        marks += s[j]
        j += 1
    return marks, j


def _render(items: list[_Syl | str]) -> str:
    out = ""
    prev: _Syl | None = None
    k = 0
    while k < len(items):
        item = items[k]
        if isinstance(item, _Syl):
            out += item.onset + item.vowel
            prev = item
            k += 1
            continue
        if item not in _CONSONANTS:
            out += item
            prev = None
            k += 1
            continue
        run: list[str] = []
        while k < len(items) and isinstance(items[k], str) and items[k] in _CONSONANTS:
            run.append(items[k])  # type: ignore[arg-type]
            k += 1
        at_end = k >= len(items) or not isinstance(items[k], _Syl)
        out += _render_run(run, prev, at_end)
        prev = None
    return out


def _render_run(run: list[str], prev: _Syl | None, at_end: bool) -> str:
    can_close = prev is not None and not prev.open_only
    needs = prev is not None and prev.needs_coda
    r = len(run)
    ini = [_INITIAL[c] for c in run]
    fin = [_FINAL.get(c, "") for c in run]

    if run[0] == "อ":
        # อ is a vowel carrier, never a final: อยุธยา a-yut-ya
        return "a" + _render_run(run[1:], None, at_end) if r > 1 else "a"
    if r == 1:
        if can_close:
            return fin[0]
        return ini[0] + ("o" if at_end and prev is None else "a")
    if r == 2:
        silent_r = (
            can_close
            and run[1] == "ร"
            and run[0] in _SILENT_R_AFTER
            and (
                prev is not None
                and prev.vowel in ("u", "i", "a", "e")
                and (needs or prev.vowel != "a")
            )
        )
        if at_end:
            if silent_r:
                return fin[0]  # สมุทร samut, จักร chak: ร after a final stop is silent
            return ini[0] + "o" + fin[1]  # สาคร sa-khon, สีลม si-lom, สาทร sa-thon
        if silent_r:
            return fin[0]  # สมุทรปราการ samut-prakan
        if prev is None or (not needs and prev.vowel in ("u", "i", "ue")):
            return ini[0] + "o" + fin[1]  # ชลบุรี chon-buri, อุดร u-don
        return fin[0] + ini[1] + "a"  # จันทบุรี chan-tha-buri, เทพมหา thep-ma-ha
    if r == 3:
        if at_end:
            if needs:
                return fin[0] + ini[1] + "o" + fin[2]
            return ini[0] + "a" + ini[1] + "o" + fin[2]  # นคร na-khon
        if run[2] == "ร":
            return (
                (fin[0] if can_close else ini[0] + "a") + ini[1] + "o" + fin[2]
                if can_close
                else ini[0] + "a" + ini[1] + "o" + fin[2]
            )  # นครราช na-khon-rat
        if can_close:
            return fin[0] + ini[1] + "a" + ini[2] + "a"  # กาญจน kan-cha-na
        return ini[0] + "o" + fin[1] + ini[2] + "a"  # นนท non-tha
    # Longer runs: close the previous syllable, then "Ca-CoC" groups of three
    # (นครพนม na-khon-pha-nom, สกลนคร sa-kon-na-khon), "CoC" for a pair
    out = ""
    rest = list(range(r))
    if can_close:
        out += fin[0]
        rest = rest[1:]
    while rest:
        if len(rest) >= 3 and (len(rest) != 4 or run[rest[2]] == "ร"):
            a, b, c = rest[:3]
            out += ini[a] + "a" + ini[b] + "o" + fin[c]
            rest = rest[3:]
        elif len(rest) >= 2:
            out += ini[rest[0]] + "o" + fin[rest[1]]
            rest = rest[2:]
        else:
            out += ini[rest[0]] + "a"
            rest = []
    return out


def _vowel(pre: str, marks: str, s: list[str], i: int) -> tuple[str, int, bool]:
    """The vowel spelled by a leading vowel, the marks on the onset and the
    vowel letters after it. Returns (vowel, next index, needs a final)"""
    n = len(s)
    nxt = s[i] if i < n else ""
    nxt2 = s[i + 1] if i + 1 < n else ""
    if nxt in ("ย", "ว", "อ") and nxt2 in _MARKS - {"\u0e30"}:
        # The letter starts the next syllable (ธิวา thi-wa), it is not part
        # of this vowel
        nxt = ""

    if pre == "เ":
        if marks == "ี" and nxt == "ย":
            if nxt2 == "ว":
                return "iao", i + 2, False
            return "ia", i + (2 if nxt2 == "ะ" else 1), False
        if marks == "ื" and nxt == "อ":
            return "uea", i + 1, False
        if marks == "ิ":
            return "oe", i, False
        if marks == "าะ":
            return "o", i, False
        if marks == "า":
            return "ao", i, False
        if marks == _MAITAIKHU:
            if nxt == "ว":
                return "eo", i + 1, False
            return "e", i, True
        if marks == "" and nxt == "อ":
            return "oe", i + 1, False
        if marks == "" and nxt == "ย":
            return "oei", i + 1, False
        if marks == "" and nxt == "ว":
            return "eo", i + 1, False
        return "e", i, False
    if pre == "แ":
        if nxt == "ว":
            return "aeo", i + 1, False
        return "ae", i, marks == _MAITAIKHU
    if pre == "โ":
        return "o", i, False
    if pre in ("ไ", "ใ"):
        return "ai", i, False

    if marks == _MAI_HAN_AKAT:
        if nxt == "ว":
            if nxt2 == "ะ":
                return "ua", i + 2, False
            return "ua", i + 1, False
        if nxt == "ย":
            return "ai", i + 1, False
        return "a", i, True
    if marks and marks[0] in "ะาๅ":
        if marks == "า" and nxt == "ย":
            return "ai", i + 1, False
        if marks == "า" and nxt == "ว":
            return "ao", i + 1, False
        return "a", i, False
    if marks == "ำ":
        return "am", i, False
    if marks in ("ิ", "ี"):
        if marks == "ิ" and nxt == "ว":
            return "io", i + 1, False
        return "i", i, False
    if marks in ("ึ", "ื"):
        return "ue", i + (1 if nxt == "อ" else 0), False
    if marks in ("ุ", "ู"):
        if nxt == "ย":
            return "ui", i + 1, False
        return "u", i, False
    if marks == _MAITAIKHU:
        return "o", i, True
    return "a", i, False
