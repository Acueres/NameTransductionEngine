"""Training the Persian display-romanization model from GeoNames.

The training data: Iranian features whose GeoNames primary name is a BGN/PCGN
romanization (Kermānshāh, Bandar-e ‘Abbās) and which also carry a Persian
(`fa`) alternate name. Each Persian word is aligned letter by letter with its
romanization; the model records, for each letter in its context, the
romanization together with the short vowel before it. See
`transliteration/romanization_packs/persian.py` for how the model is read.

Built by the models stage (`models/persian_romanization/data_provision.py`);
evaluated by
`python -m name_transduction_engine.tools.persian_model eval`.
"""

import collections
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from name_transduction_engine.transliteration.romanization_packs import abjad
from name_transduction_engine.transliteration.romanization_packs import persian as fa

# Bump when the alignment, counting or model format changes in a way the
# fingerprinted tables below do not show; a new value makes built models stale
TRAINING_RULES_VERSION = 1

MIN_CONTEXT = 1  # a context must be seen this often to be kept
MIN_WORD = 2  # a word must be attested this often to enter the lexicon
WORD_AGREEMENT = 0.6  # share of its attestations the majority reading needs
MIN_EZAFE = 3  # attestations before a word's ezāfe tendency is kept
EZAFE_SMOOTHING = 2.0  # pseudo-counts at the overall rate

SOURCE = (
    "GeoNames (CC BY 4.0): Iranian features with BGN/PCGN primary names and "
    "Persian alternate names"
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Pair:
    geonameid: int
    feature_code: str
    population: int
    latin: str
    persian: str


PAIRS_QUERY = """
    SELECT g.geonameid, g.feature_code, COALESCE(g.population, 0),
           g.name, a.alternate_name
    FROM geoname g
    JOIN alternate_name a ON a.geonameid = g.geonameid AND a.lang = 'fa'
    WHERE g.country_code = 'IR'
    ORDER BY g.geonameid, a.alternate_name_id
"""


def extract_pairs(conn: sqlite3.Connection) -> list[Pair]:
    return [
        Pair(int(gid), code or "", int(pop or 0), latin, persian)
        for gid, code, pop, latin, persian in conn.execute(PAIRS_QUERY)
    ]


def write_pairs_tsv(pairs: list[Pair], out: Path) -> None:
    """For offline experiments (`tools/persian_model.py extract`)"""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("geonameid\tfeature_code\tpopulation\tname\tfa_name\n")
        for p in pairs:
            row = (p.geonameid, p.feature_code, p.population, p.latin, p.persian)
            f.write("\t".join(str(v).replace("\t", " ") for v in row) + "\n")


def load_pairs_tsv(path: Path) -> list[Pair]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            gid, code, pop, latin, persian = line.rstrip("\n").split("\t")
            pairs.append(Pair(int(gid), code, int(pop or 0), latin, persian))
    return pairs


def is_test(p: Pair) -> bool:
    """The held-out 10% used by the evaluation"""
    return p.geonameid % 10 == 7


# --------------------------------------------------------------------------- #
# Rules fingerprint
# --------------------------------------------------------------------------- #


def rules_fingerprint() -> str:
    """Fingerprint of everything that shapes a trained model besides the data:
    the letter readings, the context windows and the thresholds. Editing any
    of them makes a built model stale"""
    samples: dict[str, list] = {}
    for ch in sorted(fa.LETTERS):
        for label, word, i in (
            ("first", ch + "ب", 0),
            ("medial", "ب" + ch + "ب", 1),
            ("final", "ب" + ch, 1),
            ("after_hamza", "ء" + ch, 1),
        ):
            samples[f"{ch}:{label}"] = [
                list(fa.letter_options(word, i)),
                list(fa.slot_options(word, i)),
            ]
    payload = {
        "training_rules_version": TRAINING_RULES_VERSION,
        "levels": fa.LEVELS,
        "letters": samples,
        "thresholds": [
            MIN_CONTEXT,
            MIN_WORD,
            WORD_AGREEMENT,
            MIN_EZAFE,
            EZAFE_SMOOTHING,
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Folding BGN text to the display style
# --------------------------------------------------------------------------- #

_APOSTROPHES = str.maketrans(
    {"‘": "ʿ", "`": "ʿ", "ʻ": "ʿ", "’": "ʾ", "'": "ʾ", "ʼ": "ʾ"}
)
_DROP_MARKS = frozenset("\u0323\u0327\u0331\u0326\u0306")  # dot below, cedilla...


def fold(text: str) -> str:
    """BGN with dotted letters -> display style: Eşfahān -> esfahān,
    Moşţafá -> mostafā; lowercase"""
    text = unicodedata.normalize("NFD", text.translate(_APOSTROPHES))
    out: list[str] = []
    for ch in text:
        if ch in _DROP_MARKS:
            continue
        if ch == "\u0304" and out and out[-1].lower() in "sz":
            continue  # s̄ z̄ for ث ذ
        if ch == "\u0301":  # á (alef maqsura) -> ā
            ch = "\u0304"
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)).lower()


def compare_key(text: str) -> str:
    """Comparison form: folded, apostrophes (ʿ ʾ) ignored"""
    return re.sub(r"[ʿʾ]", "", fold(text)).strip()


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


def joint_options(word: str, i: int) -> list[tuple[str, int]]:
    return abjad.joint_options(fa.ORTHOGRAPHY, word, i)


def align(word: str, latin: str) -> list[str] | None:
    """Joint outputs per letter such that they concatenate to `latin`"""
    return abjad.align(fa.ORTHOGRAPHY, word, latin)


@dataclass
class AlignedWord:
    persian: str  # normalized, harakat removed
    latin: str  # folded, lowercase
    outputs: list[str]
    ezafe: str | None  # "e", "ye" or "" (none); None for the last word


def _latin_words(name: str) -> list[tuple[str, str]]:
    """Latin words with their ezāfe: Chāh-e Kheyrābād ->
    [(chāh, "e"), (kheyrābād, "")]"""
    words: list[tuple[str, str]] = []
    for token in fold(name).replace("-", " - ").split():
        if token == "-":
            continue
        if token in ("e", "ye") and words:
            words[-1] = (words[-1][0], token)
            continue
        words.append((token, ""))
    return words


def _clean_persian(word: str) -> str:
    return "".join(ch for ch in fa.normalize(word) if ch not in fa._MARKS)


def align_name(persian: str, latin: str) -> list[AlignedWord] | None:
    """Pair the words of a name, allowing two Persian words written apart to
    make one Latin word (خرم آباد Khorramābād), then align each pair"""
    if re.search(r"[0-9()/,]", latin + persian):
        return None
    pw = [_clean_persian(w) for w in persian.split()]
    pw = [w.replace(fa.EZAFE_MARK, "") for w in pw]
    lw = _latin_words(latin)
    if not pw or not lw:
        return None
    # dp[a][b]: aligned words for the first a Persian and b Latin words
    dp: dict[tuple[int, int], list[AlignedWord]] = {(0, 0): []}
    for a in range(len(pw) + 1):
        for b in range(len(lw) + 1):
            if (a, b) not in dp:
                continue
            done = dp[(a, b)]
            if a < len(pw) and b < len(lw):
                for take in (1, 2):
                    if a + take > len(pw):
                        continue
                    p = fa.ZWNJ.join(pw[a : a + take])
                    lat, ez = lw[b]
                    outs = ["va"] if p == "و" and lat == "va" else align(p, lat)
                    if outs is not None and (a + take, b + 1) not in dp:
                        dp[(a + take, b + 1)] = [*done, AlignedWord(p, lat, outs, ez)]
    result = dp.get((len(pw), len(lw)))
    if result is None:
        return None
    last = result[-1]
    result[-1] = AlignedWord(last.persian, last.latin, last.outputs, None)
    return result


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class Counts:
    contexts: dict[str, collections.Counter] = field(
        default_factory=lambda: collections.defaultdict(collections.Counter)
    )
    words: dict[str, collections.Counter] = field(
        default_factory=lambda: collections.defaultdict(collections.Counter)
    )
    ezafe: dict[str, collections.Counter] = field(
        default_factory=lambda: collections.defaultdict(collections.Counter)
    )
    ezafe_next: dict[str, collections.Counter] = field(
        default_factory=lambda: collections.defaultdict(collections.Counter)
    )
    names_total: int = 0
    names_aligned: int = 0


def count(pairs: list[Pair]) -> Counts:
    c = Counts()
    for p in pairs:
        c.names_total += 1
        aligned = align_name(p.persian, p.latin)
        if aligned is None:
            continue
        c.names_aligned += 1
        for k, w in enumerate(aligned):
            if w.persian == "و":
                continue
            c.words[w.persian][w.latin] += 1
            if w.ezafe is not None and aligned[k + 1].persian != "و":
                c.ezafe[w.persian][w.ezafe] += 1
                c.ezafe_next[aligned[k + 1].persian][bool(w.ezafe)] += 1
            abjad.count_contexts(fa.ORTHOGRAPHY, c.contexts, w.persian, w.outputs)
            # Parts of compounds, for the part lexicon
            spans = fa.part_spans(w.persian)
            if len(spans) > 1:
                for start, end in spans:
                    part = w.persian[start:end]
                    c.words[part]["".join(w.outputs[start:end])] += 1
    return c


def build_model(c: Counts, version: str) -> dict:
    """The model as stored (see `persian.make_model`), without its `meta`"""
    table, ranked = abjad.build_tables(fa.ORTHOGRAPHY, c.contexts, MIN_CONTEXT)
    lexicon = abjad.build_lexicon(c.words, MIN_WORD, WORD_AGREEMENT)

    total_all = sum(sum(cn.values()) for cn in c.ezafe.values())
    with_all = sum(cn["e"] + cn["ye"] for cn in c.ezafe.values())
    p0 = min(max(with_all / max(total_all, 1), 0.01), 0.99)
    prior = _logit(p0)

    def delta(with_: int, total: int) -> float:
        rate = (with_ + EZAFE_SMOOTHING * p0) / (total + EZAFE_SMOOTHING)
        return round(_logit(rate) - prior, 2)

    ezafe_head: dict[str, list] = {}
    for word, counter in c.ezafe.items():
        total = sum(counter.values())
        with_ = counter["e"] + counter["ye"]
        if total < MIN_EZAFE:
            continue
        form = "" if not with_ else ("e" if counter["e"] >= counter["ye"] else "ye")
        d = delta(with_, total)
        if abs(d) >= 0.1 or form:
            ezafe_head[word] = [d, form]
    ezafe_next: dict[str, float] = {}
    for word, counter in c.ezafe_next.items():
        total = sum(counter.values())
        if total >= MIN_EZAFE:
            d = delta(counter[True], total)
            if abs(d) >= 0.1:
                ezafe_next[word] = d

    return {
        "version": version,
        "source": SOURCE,
        "table": table,
        "lexicon": lexicon,
        "ranked": ranked,
        "ezafe_prior": round(prior, 3),
        "ezafe_head": dict(sorted(ezafe_head.items())),
        "ezafe_next": dict(sorted(ezafe_next.items())),
    }


def _logit(p: float) -> float:
    return math.log(p / (1 - p))
