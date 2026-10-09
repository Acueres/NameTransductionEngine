"""Training the Arabic display-romanization model from GeoNames.

The training data: names with a BGN/PCGN romanization, paired with an Arabic
(`ar`) name of the same feature:

- primary names of features in Arab countries, which GeoNames takes from the
  NGA's GNS, written in BGN/PCGN (Ar Riyāḑ, Madīnat al Kuwayt);
- alternate names without a language tag that carry BGN marks (Al Qāhirah,
  Tall ash Shīḩ), for features anywhere.

A pair is kept when every word aligns letter by letter (so a variant that is
another name altogether drops out). The model records, for each letter in
its context, the reading together with the short vowel before it; the words
seen with one dominant reading; and how often a word ending in tāʾ marbūṭa
heads a construct (-at). See `transliteration/romanization_packs/arabic.py`
for how the model is read.

Built by the models stage (`models/arabic_romanization/data_provision.py`);
evaluated by `python -m name_transduction_engine.tools.arabic_model eval`.
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
from name_transduction_engine.transliteration.romanization_packs import arabic as ar

# Bump when the alignment, counting or model format changes in a way the
# fingerprinted tables below do not show; a new value makes built models stale
TRAINING_RULES_VERSION = 1

MIN_CONTEXT = 1  # a context must be seen this often to be kept
MIN_WORD = 2  # a word must be attested this often to enter the lexicon
WORD_AGREEMENT = 0.6  # share of its attestations the majority reading needs
MIN_CONSTRUCT = 3  # attestations before a word's construct tendency is kept
CONSTRUCT_SMOOTHING = 2.0  # pseudo-counts at the overall rate

# Members of the Arab League, plus Western Sahara: their GeoNames primary
# names are BGN/PCGN romanizations of Arabic
ARAB_COUNTRIES = (
    "AE", "BH", "DJ", "DZ", "EG", "EH", "IQ", "JO", "KM", "KW", "LB", "LY",
    "MA", "MR", "OM", "PS", "QA", "SA", "SD", "SO", "SY", "TN", "YE",
)  # fmt: skip

SOURCE = (
    "GeoNames (CC BY 4.0): BGN/PCGN names of features in Arab countries and "
    "BGN/PCGN variant names, paired with Arabic alternate names"
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Pair:
    geonameid: int
    latin: str
    arabic: str


_PRIMARY_QUERY = f"""
    SELECT g.geonameid, g.name, a.alternate_name
    FROM geoname g
    JOIN alternate_name a ON a.geonameid = g.geonameid AND a.lang = 'ar'
    WHERE g.country_code IN ({", ".join(f"'{c}'" for c in ARAB_COUNTRIES)})
    ORDER BY g.geonameid, a.alternate_name_id
"""

_VARIANT_QUERY = """
    SELECT a.geonameid, u.alternate_name, a.alternate_name
    FROM alternate_name a
    JOIN alternate_name u ON u.geonameid = a.geonameid
        AND u.lang IS NULL AND u.tag_status = 'untagged'
    WHERE a.lang = 'ar'
    ORDER BY a.geonameid, a.alternate_name_id, u.alternate_name_id
"""

# A Latin name written with BGN marks (macrons, dots, cedillas, ‘ ’)
_BGN_MARKS = re.compile(r"[āīūáĀĪŪÁḩḨḑḐţŢşŞẓẒḤḥṢṣṬṭḌḍ‘’]")


def is_bgn(latin: str) -> bool:
    return bool(_BGN_MARKS.search(latin))


def extract_pairs(conn: sqlite3.Connection) -> list[Pair]:
    seen: set[tuple[int, str, str]] = set()
    pairs: list[Pair] = []
    for query, needs_marks in ((_PRIMARY_QUERY, False), (_VARIANT_QUERY, True)):
        for gid, latin, arabic in conn.execute(query):
            if needs_marks and not is_bgn(latin):
                continue
            key = (int(gid), latin, arabic)
            if key not in seen:
                seen.add(key)
                pairs.append(Pair(*key))
    return pairs


def write_pairs_tsv(pairs: list[Pair], out: Path) -> None:
    """For offline experiments (`tools/arabic_model.py extract`)"""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("geonameid\tname\tar_name\n")
        for p in pairs:
            row = (p.geonameid, p.latin, p.arabic)
            f.write("\t".join(str(v).replace("\t", " ") for v in row) + "\n")


def load_pairs_tsv(path: Path) -> list[Pair]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            gid, latin, arabic = line.rstrip("\n").split("\t")[:3]
            pairs.append(Pair(int(gid), latin, arabic))
    return pairs


def is_test(p: Pair) -> bool:
    """The held-out 10% used by the evaluation"""
    return p.geonameid % 10 == 7


# --------------------------------------------------------------------------- #
# Rules fingerprint
# --------------------------------------------------------------------------- #


def rules_fingerprint() -> str:
    """Fingerprint of everything that shapes a trained model besides the
    data: the letter readings, context windows, thresholds and the data
    selection. Editing any of them makes a built model stale"""
    payload = {
        "training_rules_version": TRAINING_RULES_VERSION,
        "levels": ar.ORTHOGRAPHY.levels,
        "letters": ar.ORTHOGRAPHY.rules_sample(),
        "thresholds": [
            MIN_CONTEXT,
            MIN_WORD,
            WORD_AGREEMENT,
            MIN_CONSTRUCT,
            CONSTRUCT_SMOOTHING,
        ],
        "countries": ARAB_COUNTRIES,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Folding BGN text to the model's form
# --------------------------------------------------------------------------- #

_APOSTROPHES = str.maketrans(
    {"‘": "ʿ", "`": "ʿ", "ʻ": "ʿ", "’": "ʾ", "'": "ʾ", "ʼ": "ʾ"}
)
# Dot below, cedilla, macron below, comma below, breve, dot above
_DROP_MARKS = frozenset("̧̣̱̦̆̇")


def fold(text: str) -> str:
    """BGN with dotted letters -> the model's form: Ar Riyāḑ -> ar riyād,
    Al Minyá -> al minyá; lowercase"""
    text = unicodedata.normalize("NFD", text.translate(_APOSTROPHES))
    out: list[str] = []
    for ch in text:
        if ch in _DROP_MARKS:
            continue
        if ch == "́" and not (out and out[-1].lower() == "a"):
            continue  # acute: kept only on á (alef maqsura)
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)).lower()


_SUN = frozenset(ar.CONSONANTS[ch] for ch in ar.SUN_LETTERS)


def latin_words(name: str) -> list[tuple[bool, str]] | None:
    """(has article, word) per word of a BGN name: Ar Riyāḑ ->
    [(True, riyād)]; Madīnat al Kuwayt -> [(False, madīnat), (True, kuwayt)].
    None when an article has no word after it"""
    tokens = [t for t in re.split(r"[\s\-]+", fold(name)) if t]
    out: list[tuple[bool, str]] = []
    article = False
    for k, t in enumerate(tokens):
        nxt = tokens[k + 1] if k + 1 < len(tokens) else None
        if t == "al" or (
            t[:1] == "a" and t[1:] in _SUN and nxt and nxt.startswith(t[1:])
        ):
            if nxt is None:
                return None
            article = True
            continue
        out.append((article, t))
        article = False
    return out


def compare_key(text: str) -> str:
    """Comparison form of a name, BGN or display: folded, hyphens as
    spaces, ʿ ʾ ignored"""
    return re.sub(r"[ʿʾ]", "", fold(text)).replace("-", " ").strip()


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


@dataclass
class AlignedWord:
    article: bool
    body: str  # Arabic without the article and harakat
    outputs: list[str]  # pausal readings per letter
    construct: bool  # the word heads a construct (-at)


def align_name(arabic: str, latin: str) -> list[AlignedWord] | None:
    """Pair the words of a name one to one (articles must agree) and align
    each pair letter by letter"""
    if re.search(r"[0-9()/,]", latin + arabic):
        return None
    lw = latin_words(latin)
    aw = [ar.split_article(ar.strip_marks(w)) for w in ar.normalize(arabic).split()]
    if not lw or len(lw) != len(aw):
        return None
    out: list[AlignedWord] = []
    for (a_art, body), (l_art, lat) in zip(aw, lw, strict=True):
        if body == "و" and lat == "wa" and not a_art and not l_art:
            out.append(AlignedWord(False, body, ["wa"], False))
            continue
        if a_art != l_art:
            return None
        outputs = abjad.align(ar.ORTHOGRAPHY, body, lat)
        if outputs is None:
            return None
        is_construct = body.endswith(ar.TA_MARBUTA) and outputs[-1] in ("at", "t")
        out.append(AlignedWord(a_art, body, ar.pausal(body, outputs), is_construct))
    return out


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
    # Construct outcomes (True: -at) per feature, head word and next word
    construct_feature: dict[str, collections.Counter] = field(
        default_factory=lambda: collections.defaultdict(collections.Counter)
    )
    construct_head: dict[str, collections.Counter] = field(
        default_factory=lambda: collections.defaultdict(collections.Counter)
    )
    construct_next: dict[str, collections.Counter] = field(
        default_factory=lambda: collections.defaultdict(collections.Counter)
    )
    construct_all: collections.Counter = field(default_factory=collections.Counter)
    names_total: int = 0
    names_aligned: int = 0


def count(pairs: list[Pair]) -> Counts:
    c = Counts()
    for p in pairs:
        c.names_total += 1
        aligned = align_name(p.arabic, p.latin)
        if aligned is None:
            continue
        c.names_aligned += 1
        for k, w in enumerate(aligned):
            if w.body == "و":
                continue
            c.words[w.body]["".join(w.outputs)] += 1
            abjad.count_contexts(ar.ORTHOGRAPHY, c.contexts, w.body, w.outputs)
            nxt = aligned[k + 1] if k + 1 < len(aligned) else None
            if nxt is not None and nxt.body != "و" and w.body.endswith(ar.TA_MARBUTA):
                outcome = w.construct
                c.construct_all[outcome] += 1
                c.construct_feature[f"head:{int(w.article)}"][outcome] += 1
                c.construct_feature[f"next:{int(nxt.article)}"][outcome] += 1
                c.construct_head[w.body][outcome] += 1
                c.construct_next[nxt.body][outcome] += 1
    return c


def build_model(c: Counts, version: str) -> dict:
    """The model as stored (see `arabic.make_model`), without its `meta`"""
    table, ranked = abjad.build_tables(ar.ORTHOGRAPHY, c.contexts, MIN_CONTEXT)
    lexicon = abjad.build_lexicon(c.words, MIN_WORD, WORD_AGREEMENT)

    total = sum(c.construct_all.values())
    p0 = min(max(c.construct_all[True] / max(total, 1), 0.01), 0.99)
    prior = _logit(p0)

    def deltas(
        table: dict[str, collections.Counter], min_count: int
    ) -> dict[str, float]:
        out = {}
        for key, counter in table.items():
            n = sum(counter.values())
            if n < min_count:
                continue
            rate = (counter[True] + CONSTRUCT_SMOOTHING * p0) / (
                n + CONSTRUCT_SMOOTHING
            )
            d = round(_logit(rate) - prior, 2)
            if abs(d) >= 0.1:
                out[key] = d
        return dict(sorted(out.items()))

    return {
        "version": version,
        "source": SOURCE,
        "table": table,
        "lexicon": lexicon,
        "ranked": ranked,
        "construct_prior": round(prior, 3),
        "construct_feature": deltas(c.construct_feature, 1),
        "construct_head": deltas(c.construct_head, MIN_CONSTRUCT),
        "construct_next": deltas(c.construct_next, MIN_CONSTRUCT),
    }


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))
