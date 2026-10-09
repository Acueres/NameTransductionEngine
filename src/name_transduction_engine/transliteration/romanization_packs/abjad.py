"""Reading words in an abjad: scripts that leave short vowels unwritten and
use one letter for several sounds (Arabic, Persian; later Urdu, Pashto,
Hebrew).

A language describes its spelling as an `Orthography`: for each letter of a
word, the ways it can be read, and the short vowels that may stand before it
unwritten. A reading of a word is one choice per letter of
"short vowel + letter reading" (a *joint output*). Everything else here is
shared by the languages:

- the letter model: for each letter, the joint output chosen by the letters
  around it, learned from attested romanizations, with backoff through nested
  context windows (`predict`, `read_letters`);
- the hint-guided search: the reading allowed by the letters that is closest
  to a Latin-script name of the same entity (`guided_reading`);
- for the trainers: aligning a word with its attested romanization letter by
  letter (`align`), and turning counted contexts into the stored tables
  (`build_tables`).

What is not shared stays in the language modules: their letters, syllable
rules, lexicons, and grammar between words (the Persian ezāfe, the Arabic
article and construct state).
"""

import collections
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

# --------------------------------------------------------------------------- #
# Orthography
# --------------------------------------------------------------------------- #

# Context windows of the letter model: offsets around the letter, most
# specific first. Each level is a subset of the one before, so a trainer can
# prune entries the next level already answers. "#" pads the word edges.
DEFAULT_LEVELS: tuple[tuple[int, ...], ...] = (
    (-4, -3, -2, -1, 0, 1, 2, 3),
    (-3, -2, -1, 0, 1, 2),
    (-2, -1, 0, 1, 2),
    (-2, -1, 0, 1),
    (-1, 0, 1),
    (-1, 0),
    (0,),
)
_PAD = 4

LATIN_VOWELS = frozenset("aeiouāīūēōá")
SHORT_VOWELS = frozenset("aeiou")
_LENGTHENED = {"a": "ā", "i": "ī", "u": "ū", "e": "ī", "o": "ū"}


class Orthography:
    """How a language writes words in an abjad. Subclasses fill in the
    letters and their readings; the defaults describe what is common.

    Letters are those of a normalized word without harakat. A letter outside
    `letters` (a digit, an unknown letter) is copied through."""

    name: str = ""
    letters: frozenset[str] = frozenset()
    # Letters read as consonants, with their romanization: a shadda doubles it
    consonants: Mapping[str, str] = {}
    # Harakat: the short vowel each vowel mark writes, and all marks
    mark_vowels: Mapping[str, str] = {}
    sukun: str = "\u0652"
    shadda: str = "\u0651"
    marks: frozenset[str] = frozenset()
    # Letters that carry a word-initial vowel (alef): a mark on them is the
    # letter's own vowel, not a vowel after it
    carriers: frozenset[str] = frozenset()
    # A letter that starts a new word part (Persian zero-width non-joiner)
    part_break: str | None = None
    levels: tuple[tuple[int, ...], ...] = DEFAULT_LEVELS
    # Letter pairs that spell one foreign sound (Arabic تش for ch): the
    # hint-guided search also matches them as a whole
    units: frozenset[str] = frozenset()

    # Hint matching: letter pairs that stand for each other across
    # orthographies (Persian و v for English w) cost half a substitution
    near_pairs: frozenset[frozenset[str]] = frozenset(
        frozenset(p)
        for p in ("ae", "ei", "ou", "vw", "ck", "cs", "cz", "sz", "kq", "gj", "iy", "uw")
    )

    def letter_options(self, word: str, i: int) -> tuple[str, ...]:
        """Every way letter i can be romanized, most usual first (the order
        breaks ties when aligning training data)"""
        raise NotImplementedError

    def slot_options(self, word: str, i: int) -> tuple[str, ...]:
        """Short vowels that may stand, unwritten, before letter i"""
        raise NotImplementedError

    def allowed(self, prefix: str, out: str, strict: bool = False) -> bool:
        """Whether reading `out` may follow the reading `prefix` (its
        `shape`) under the language's syllable rules. `strict` is used by
        the hint-guided search, which can choose any reading; the letter
        model follows attested readings and needs fewer rules"""
        return True

    def may_be_silent(self, word: str, i: int) -> bool:
        """Whether letter i is regularly silent, its sound carried by
        another letter (the hint-guided search then does not charge for
        silencing it)"""
        return False

    def fully_voweled(self, bare: str, after: dict[int, str]) -> bool:
        """Whether a word's vowel marks are its whole reading (an unmarked
        consonant then closes its syllable): marks on at least half its
        consonants"""
        consonants = [k for k, ch in enumerate(bare) if ch in self.consonants]
        marked = sum(1 for k in consonants if after.get(k))
        return bool(consonants) and marked >= max(1, len(consonants) // 2)

    def is_doubled(self, reading: str) -> bool:
        """A reading of a doubled (shadda) letter"""
        return len(reading) > 1 and reading[-1] == reading[-2] and reading[-1].isalpha()

    def allowed_end(self, shape: str) -> bool:
        """Whether a reading may end in this `shape` (strict mode)"""
        return True

    def shape(self, prefix: str) -> str:
        """What `allowed` needs to know about a reading so far; the
        hint-guided search keeps one best reading per shape"""
        return shape(prefix)

    def at_part_start(self, word: str, i: int) -> bool:
        return i == 0 or (
            self.part_break is not None and word[i - 1] == self.part_break
        )

    def forced_vowel_fits(self, vowel: str, letter: str, forced: str) -> bool:
        """Whether a joint output (`vowel` + `letter`) agrees with the short
        vowel written by a mark on the previous letter: the same short
        vowel, or a vowel letter that starts with it or lengthens it
        (damma + و is ū, not u + w)"""
        if vowel == forced:
            return True
        if vowel or not forced or not letter:
            return False
        return letter[0] in (forced, _LENGTHENED.get(forced))

    # ------------------------------------------------------------------ #

    def joint_outputs(self, word: str, i: int) -> list[str]:
        """Every short vowel + letter reading possible for letter i"""
        return [
            v + x
            for v in self.slot_options(word, i)
            for x in self.letter_options(word, i)
        ]

    def context_keys(self, word: str, i: int) -> list[str]:
        padded = "#" * _PAD + word + "#" * _PAD
        return [
            f"{n}|" + "".join(padded[i + _PAD + off] for off in offsets)
            for n, offsets in enumerate(self.levels)
        ]

    def split_joint(
        self, joint: str, word: str, i: int, prefer_letter: bool = False
    ) -> tuple[str, str]:
        """Split a joint output into (short vowel, letter part). By default a
        leading vowel is taken as the short vowel (a mark then replaces it);
        with `prefer_letter`, a whole output that is the letter's own reading
        stays whole (و read o, not o + silent و)"""
        letter_opts = self.letter_options(word, i)
        if prefer_letter and joint in letter_opts:
            return "", joint
        for v in sorted(self.slot_options(word, i), key=len, reverse=True):
            if v and joint.startswith(v) and joint[len(v) :] in letter_opts:
                return v, joint[len(v) :]
        return "", joint

    def split_marks(self, word: str) -> tuple[str, dict[int, str], set[int]]:
        """Separate harakat: the bare word, the short vowel written after
        each letter index ("" for sukun), and the letters with shadda"""
        bare: list[str] = []
        after: dict[int, str] = {}
        doubled: set[int] = set()
        for ch in word:
            if ch in self.marks:
                if not bare:
                    continue
                k = len(bare) - 1
                if ch in self.mark_vowels:
                    after[k] = self.mark_vowels[ch]
                elif ch == self.sukun:
                    after[k] = ""
                elif ch == self.shadda:
                    doubled.add(k)
                continue
            bare.append(ch)
        return "".join(bare), after, doubled

    def strip_marks(self, word: str) -> str:
        return "".join(ch for ch in word if ch not in self.marks)

    def has_vowel_marks(self, word: str) -> bool:
        return any(ch in self.mark_vowels for ch in word)

    def rules_sample(self) -> dict[str, list]:
        """The letter readings in a few positions, for a trainer's rules
        fingerprint: editing the readings makes a built model stale"""
        samples: dict[str, list] = {}
        for ch in sorted(self.letters):
            for label, word, i in self._sample_positions(ch):
                samples[f"{ch}:{label}"] = [
                    list(self.letter_options(word, i)),
                    list(self.slot_options(word, i)),
                ]
        return samples

    def _sample_positions(self, ch: str) -> Iterable[tuple[str, str, int]]:
        filler = next(iter(sorted(self.consonants)), "ب")
        return (
            ("first", ch + filler, 0),
            ("medial", filler + ch + filler, 1),
            ("final", filler + ch, 1),
        )


def shape(prefix: str) -> str:
    """What syllable rules need to know about a reading so far: the vowel it
    ends in, "ab" for a consonant after a vowel, or the consonants of a
    reading with no vowel yet"""
    if not prefix:
        return ""
    if prefix[-1] in LATIN_VOWELS:
        return prefix[-1]
    if any(c in LATIN_VOWELS for c in prefix):
        return "ab"
    return prefix


# --------------------------------------------------------------------------- #
# The letter model
# --------------------------------------------------------------------------- #


class LetterTables(Protocol):
    table: dict[str, str]  # context key -> joint output
    ranked: dict[str, list[str]]  # smallest contexts: outputs by frequency


def predict(
    orth: Orthography,
    model: LetterTables,
    word: str,
    i: int,
    prefix: str,
    accept: Callable[[str], bool] | None = None,
) -> str:
    """The joint output for letter i: the most specific context the model
    knows, unless it breaks the syllable rules after `prefix`, then the
    smaller contexts' ranked alternatives. `accept` restricts the outputs
    (to those agreeing with written marks)"""
    valid = orth.joint_outputs(word, i)
    if accept is not None:
        valid = [v for v in valid if accept(v)] or valid
    if len(valid) == 1:
        return valid[0]
    keys = orth.context_keys(word, i)
    candidates: list[str | None] = [model.table.get(key) for key in keys]
    for key in keys[-2:]:
        candidates.extend(model.ranked.get(key, ()))
    candidates.extend(valid)
    first = None
    for out in candidates:
        if out is None or out not in valid:
            continue
        if first is None:
            first = out
        if orth.allowed(prefix, out):
            return out
    return first or valid[0]


def read_letters(orth: Orthography, model: LetterTables, word: str) -> list[str]:
    """Letter-model reading of one normalized word, one joint output per
    letter of the word without harakat. Written marks override the model:
    a vowel mark sets the short vowel after its letter, a shadda doubles it,
    and in a fully voweled word an unmarked consonant closes its syllable"""
    bare, forced_after, doubled = orth.split_marks(word)
    # A mark on a vowel carrier (initial alef) is that letter's own vowel
    carrier_vowel: dict[int, str] = {}
    for k in list(forced_after):
        if bare[k] in orth.carriers and orth.at_part_start(bare, k):
            carrier_vowel[k] = forced_after[k]
            forced_after[k] = ""
    if orth.fully_voweled(bare, forced_after):
        for k, ch in enumerate(bare):
            if ch in orth.consonants:
                forced_after.setdefault(k, "")

    out: list[str] = []
    for i, ch in enumerate(bare):
        if ch not in orth.letters:
            out.append(ch)  # digits, punctuation, unknown letters
            continue
        if i in carrier_vowel:
            out.append(carrier_vowel[i])
            continue
        forced = forced_after.get(i - 1) if i > 0 else None
        accept = None
        constrain_vowel = forced is not None and "" in orth.slot_options(bare, i)
        if constrain_vowel or i in doubled:
            fv = forced if forced in orth.slot_options(bare, i) else ""

            def accept(
                joint: str, fv: str = fv, i: int = i, cv: bool = constrain_vowel
            ) -> bool:
                v, letter = orth.split_joint(joint, bare, i)
                if cv and not orth.forced_vowel_fits(v, letter, fv):
                    return False
                return i not in doubled or orth.is_doubled(letter)

        joint = predict(orth, model, bare, i, "".join(out), accept)
        if forced is not None or i in doubled:
            vowel, letter = orth.split_joint(joint, bare, i)
            if forced is not None and "" in orth.slot_options(bare, i):
                if not orth.forced_vowel_fits(vowel, letter, forced):
                    vowel = forced if forced in orth.slot_options(bare, i) else ""
            if i in doubled and ch in orth.consonants:
                letter = orth.consonants[ch] * 2
            joint = vowel + letter
        out.append(joint)
    return out


# --------------------------------------------------------------------------- #
# Reading guided by a Latin-script name of the same entity
# --------------------------------------------------------------------------- #
#
# A spelling allows many readings (Persian ونیز: vanīz, venīz, vonīz...).
# When the entity has Latin-script names (Venice), the reading closest to one
# of them is taken, if the letters clearly correspond: the language decides
# the accepted distance. For a native place the hint is often a romanization
# of the name itself (BGN), so the result is attested; for a foreign place
# the vowels follow the original name.


@dataclass(frozen=True, eq=False)  # hashed by identity: a cache key
class GuideCosts:
    off_model: float = 0.1  # a reading the letter model would not choose
    doubling: float = 0.5  # a doubled consonant, when the hint is a foreign name
    skip_doubled: float = 0.2  # ignoring the second of a doubled hint letter
    tail: float = 0.5  # a letter at the end of a foreign hint the word lacks
    silenced: float = 0.6  # reading a sounded letter as nothing or ʾ
    # (reading, hint, cost): how the language's readings of one letter are
    # spelled in the hints' orthographies, cheaper than letter-by-letter
    # edits: Arabic ش "sh" is "ch" in a French name, "sz" in a Polish one
    equivalences: tuple[tuple[str, str, float], ...] = ()
    # (letter, reading, cost): readings of a letter that need more evidence
    # from the hint than the model's choice does
    reading_costs: tuple[tuple[str, str, float], ...] = ()


DEFAULT_GUIDE_COSTS = GuideCosts()


@lru_cache(maxsize=1 << 12)
def fold_latin(text: str) -> str:
    """Lowercase ASCII letters for matching: Vanīz -> vaniz"""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if "a" <= ch <= "z")


def _sub_cost(near: frozenset[frozenset[str]], a: str, b: str) -> float:
    if a == b:
        return 0.0
    return 0.5 if frozenset((a, b)) in near else 1.0


@lru_cache(maxsize=1 << 17)
def _distance(
    near: frozenset[frozenset[str]],
    costs: GuideCosts,
    out: str,
    seg: str,
    before: str,
    strict: bool,
) -> float:
    """Edit distance between a reading and a stretch of the hint word.
    `before` is the hint letter preceding `seg`: unless `strict`, skipping
    the second of a doubled hint letter (Illinois) is nearly free"""
    n, m = len(out), len(seg)

    def skip(j: int) -> float:
        previous = seg[j - 1] if j > 0 else before
        return costs.skip_doubled if seg[j] == previous and not strict else 1.0

    rows = [[0.0] * (m + 1)]
    for j in range(1, m + 1):
        rows[0][j] = rows[0][j - 1] + skip(j - 1)
    for i in range(1, n + 1):
        prev = rows[i - 1]
        cur = [prev[0] + 1.0] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j] + 1.0,  # a letter the hint does not have
                cur[j - 1] + skip(j - 1),
                prev[j - 1] + _sub_cost(near, out[i - 1], seg[j - 1]),
            )
        for a, b, c in costs.equivalences:
            if len(a) <= i and out.startswith(a, i - len(a)):
                for j in range(len(b), m + 1):
                    if seg.startswith(b, j - len(b)):
                        cur[j] = min(cur[j], rows[i - len(a)][j - len(b)] + c)
        rows.append(cur)
    return rows[n][m]


def _guide_options(
    orth: Orthography,
    word: str,
    i: int,
    model_choice: list[str],
    romanized: bool,
    costs: "GuideCosts",
    fold: Callable[[str], str],
) -> list[tuple[str, str, float, bool]]:
    """(reading, folded, penalty, check syllable rules) for letter i: what
    each reading costs, whatever comes before it"""
    if word[i] not in orth.letters:
        return [(word[i], fold(word[i]), 0.0, False)]
    chosen = orth.split_joint(model_choice[i], word, i, prefer_letter=True)[1]
    chosen_sounded = bool(fold_latin(chosen))
    options = []
    for out in orth.joint_outputs(word, i):
        doubled = orth.is_doubled(out)
        if doubled and i == 0:
            continue
        penalty = 0.0 if out == model_choice[i] else costs.off_model
        letter = orth.split_joint(out, word, i, prefer_letter=True)[1]
        if (
            not fold_latin(letter)
            and chosen_sounded
            and not orth.may_be_silent(word, i)
        ):
            penalty += costs.silenced
        if doubled and not romanized:
            penalty += costs.doubling
        for ch, reading, cost in costs.reading_costs:
            if word[i] == ch and letter == reading:
                penalty += cost
        options.append((out, fold(out), penalty, True))
    return options


def guided_reading(
    orth: Orthography,
    word: str,
    target: str,
    model_choice: list[str],
    romanized: bool = False,
    costs: "GuideCosts | None" = None,
    max_cost: float = float("inf"),
    fold: Callable[[str], str] = None,  # type: ignore[assignment]
) -> tuple[float, tuple[str, ...]]:
    """The reading of `word` (no harakat) closest to the folded hint word
    `target`: (cost, joint output per letter). Readings keep to the syllable
    rules (strict); one the letter model would not choose (`model_choice`)
    costs a little, so the hint must earn it. A `romanized` hint (BGN) is
    followed letter for letter, doubled consonants included; letters a
    foreign hint has at its end and the word lacks (Venice, Illinois) cost
    less than other differences. Readings costing more than `max_cost` are
    abandoned (costs only grow along a reading); (inf, ()) if none is left.
    `fold` brings readings to the form of `target` (`fold_latin` by
    default)"""
    costs = costs or DEFAULT_GUIDE_COSTS
    fold = fold or fold_latin
    n, m = len(word), len(target)
    near = orth.near_pairs
    Key = tuple[int, str]
    Paths = dict[Key, tuple[float, tuple[str, ...]]]
    letter_options = [
        _guide_options(orth, word, i, model_choice, romanized, costs, fold)
        for i in range(n)
    ]
    # best[i]: readings of the first i letters, one per (hint position, shape)
    best: list[Paths] = [{} for _ in range(n + 1)]
    best[0][(0, "")] = (0.0, ())
    for i in range(n):
        if not best[i]:
            continue
        # A unit (two letters spelling one foreign sound, تش for ch) is
        # also matched as a whole, so the hint's spelling of the sound can
        # be compared with both letters' readings together
        steps = [
            (
                1,
                [
                    ((out,), folded, pen, check)
                    for out, folded, pen, check in letter_options[i]
                ],
            )
        ]
        if i + 1 < n and word[i : i + 2] in orth.units:
            pairs = [
                ((o1, o2), f1 + f2, p1 + p2, c1 and c2)
                for o1, f1, p1, c1 in letter_options[i]
                for o2, f2, p2, c2 in letter_options[i + 1]
                if orth.split_joint(o2, word, i + 1)[0] == ""
            ]
            steps.append((2, pairs))
        for width, options in steps:
            nxt = best[i + width]
            for (j, shp), (cost, path) in best[i].items():
                before = target[j - 1] if j > 0 else ""
                for outs, folded, penalty, check in options:
                    joined = "".join(outs)
                    if check and not all(
                        orth.allowed(
                            orth.shape(shp + "".join(outs[:k])), outs[k], strict=True
                        )
                        for k in range(len(outs))
                    ):
                        continue
                    base = cost + penalty
                    if base > max_cost:
                        continue
                    key_shape = orth.shape(shp + joined)
                    for j2 in range(j, min(m, j + len(folded) + 2) + 1):
                        c = base + _distance(
                            near, costs, folded, target[j:j2], before, romanized
                        )
                        if c > max_cost:
                            continue
                        key = (j2, key_shape)
                        old = nxt.get(key)
                        if old is None or c < old[0]:
                            nxt[key] = (c, (*path, *outs))
    if not best[n]:
        return float("inf"), ()
    tail = costs.tail if not romanized else 1.0
    final = [
        (cost + tail * (m - j), path)
        for (j, shp), (cost, path) in best[n].items()
        if orth.allowed_end(shp)
    ]
    if not final:
        return float("inf"), ()
    return min(final, key=lambda cp: cp[0])


# --------------------------------------------------------------------------- #
# Training: alignment and tables
# --------------------------------------------------------------------------- #


def joint_options(orth: Orthography, word: str, i: int) -> list[tuple[str, int]]:
    """(joint output, cost) for letter i. Cost prefers no vowel and the
    earlier (more usual) letter readings, so ambiguous alignments are
    resolved the same way every time"""
    out = []
    for vc, v in enumerate(orth.slot_options(word, i)):
        for lc, x in enumerate(orth.letter_options(word, i)):
            out.append((v + x, (1 if v else 0) * 10 + vc + lc))
    return out


def align(orth: Orthography, word: str, latin: str) -> list[str] | None:
    """Joint outputs per letter of `word` that concatenate to `latin`"""
    if any(ch not in orth.letters for ch in word):
        return None
    n, m = len(word), len(latin)
    inf = 10**9
    best = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0
    for i in range(n):
        opts = joint_options(orth, word, i)
        for j in range(m + 1):
            if best[i][j] >= inf:
                continue
            for s, cost in opts:
                if latin.startswith(s, j):
                    c = best[i][j] + cost
                    if c < best[i + 1][j + len(s)]:
                        best[i + 1][j + len(s)] = c
                        back[i + 1][j + len(s)] = (j, s)
    if best[n][m] >= inf:
        return None
    outs: list[str] = []
    j = m
    for i in range(n, 0, -1):
        prev = back[i][j]
        assert prev is not None
        j, s = prev
        outs.append(s)
    return outs[::-1]


def count_contexts(
    orth: Orthography,
    contexts: dict[str, collections.Counter],
    word: str,
    outputs: list[str],
) -> None:
    """Add one aligned word to the context counts (letters with a single
    possible reading teach nothing and are skipped)"""
    for i, out in enumerate(outputs):
        if len(orth.joint_outputs(word, i)) == 1:
            continue
        for key in orth.context_keys(word, i):
            contexts[key][out] += 1


def argmax(counter: collections.Counter) -> str:
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def build_tables(
    orth: Orthography, contexts: dict[str, collections.Counter], min_context: int
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """The stored letter model: (table, ranked). The table keeps a context
    only if it changes what the smaller contexts would answer (pruned bottom
    up); `ranked` has the full rankings of the two smallest levels, which
    the reader falls back on when the best readings break syllable rules"""
    n_levels = len(orth.levels)
    levels: dict[int, dict[str, str]] = collections.defaultdict(dict)
    for key, counter in contexts.items():
        if sum(counter.values()) >= min_context or key.startswith(f"{n_levels - 1}|"):
            level = int(key.split("|", 1)[0])
            levels[level][key] = argmax(counter)

    table: dict[str, str] = {}
    for level in range(n_levels - 1, -1, -1):
        for key, out in levels[level].items():
            if _fallback(orth, table, key, level) != out:
                table[key] = out

    small = {f"{n_levels - 2}|", f"{n_levels - 1}|"}
    ranked = {
        key: [
            out for out, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ][:8]
        for key, counter in contexts.items()
        if key[: key.index("|") + 1] in small
    }
    return dict(sorted(table.items())), dict(sorted(ranked.items()))


def _fallback(
    orth: Orthography, table: dict[str, str], key: str, level: int
) -> str | None:
    """What the reader would answer for this position if `key` were absent"""
    window = key.split("|", 1)[1]
    by_offset = dict(zip(orth.levels[level], window, strict=True))
    for lower in range(level + 1, len(orth.levels)):
        k = f"{lower}|" + "".join(by_offset[o] for o in orth.levels[lower])
        if k in table:
            return table[k]
    return None


def build_lexicon(
    words: dict[str, collections.Counter], min_word: int, agreement: float
) -> dict[str, str]:
    """Words attested at least `min_word` times whose majority reading has at
    least `agreement` of the attestations"""
    lexicon = {}
    for word, counter in words.items():
        total = sum(counter.values())
        top = argmax(counter)
        if total >= min_word and counter[top] / total >= agreement:
            lexicon[word] = top
    return dict(sorted(lexicon.items()))


# --------------------------------------------------------------------------- #
# Names: words and what stands between them
# --------------------------------------------------------------------------- #

_PUNCTUATION = str.maketrans(
    {"،": ",", "؛": ";", "؟": "?", "٪": "%", "٫": ".", "٬": ",", "«": "“", "»": "”"}
)


def is_word_char(ch: str) -> bool:
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


def tokenize(
    text: str, word_char: Callable[[str], bool] = is_word_char
) -> list[tuple[bool, str]]:
    """(is_word, text) tokens: words, and the spaces and punctuation between"""
    tokens: list[tuple[bool, str]] = []
    for ch in text:
        is_word = word_char(ch)
        if tokens and tokens[-1][0] == is_word:
            tokens[-1] = (is_word, tokens[-1][1] + ch)
        else:
            tokens.append((is_word, ch))
    return tokens


def separator(text: str) -> str:
    """Spaces collapse to one; Arabic-script punctuation becomes Latin,
    followed by a space where the source had one"""
    converted = text.translate(_PUNCTUATION)
    core = "".join(converted.split())
    if not core:
        return " "
    lead = " " if converted[:1].isspace() and core[0] in "(“" else ""
    trail = " " if converted[-1:].isspace() else ""
    return lead + core + trail
