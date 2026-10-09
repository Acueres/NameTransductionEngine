"""Evaluate the Arabic display-romanization model (a developer tool).

The model itself is built by `nte init` / `nte data build models`
(models/arabic_romanization/). This trains a throwaway model on 90% of the
GeoNames pairs and scores it on the held-out 10%, against the rule-based
Arabic-script reader, so changes to the trainer or the reader can be measured:

- exact: the whole name as BGN writes it (ignoring case, hyphens, ʿ ʾ);
- words: share of words right, in names with as many words as BGN's;
- with hint: the same, given the BGN name itself as a reading hint (what
  `nte lookup` does for Arab places, whose GeoNames name is BGN), and given
  another name's BGN form (an unrelated hint must not hurt).

    python -m name_transduction_engine.tools.arabic_model eval [--show N]
    python -m name_transduction_engine.tools.arabic_model extract  # pairs -> TSV

Both read names.sqlite, or a TSV written by `extract` with `--pairs`.
"""

import argparse
import collections
import json
import random
import sqlite3
import sys
from pathlib import Path

from name_transduction_engine.models.arabic_romanization import train
from name_transduction_engine.paths import DB_PATH
from name_transduction_engine.transliteration.romanization_packs import arabic as ar
from name_transduction_engine.transliteration.romanization_packs.arabic_script import (
    romanize_arabic_script,
)
from name_transduction_engine.transliteration.romanization_packs.base import (
    parse_context,
)

DEFAULT_TSV = Path("data/arabic_eval/ar_pairs.tsv")


def evaluate(
    pairs: list[train.Pair], model: dict, show: int = 0, limit: int = 8000
) -> dict[str, float]:
    loaded = ar.make_model(model)
    ctx, _ = parse_context("ar")
    test = [p for p in pairs if train.align_name(p.arabic, p.latin) is not None]
    random.Random(0).shuffle(test)
    test = test[:limit]
    others = [p.latin for p in test]
    random.Random(1).shuffle(others)
    stats: collections.Counter = collections.Counter()
    shown = 0
    for p, other in zip(test, others, strict=True):
        want = train.compare_key(p.latin)
        got = train.compare_key(ar.romanize_with_model(loaded, p.arabic).text)
        old = romanize_arabic_script(p.arabic, ctx)
        got_old = train.compare_key(old.text if old else p.arabic)
        stats["names"] += 1
        stats["exact"] += got == want
        stats["old_exact"] += got_old == want
        want_words, got_words = want.split(), got.split()
        stats["words_total"] += len(want_words)
        if len(want_words) == len(got_words):
            stats["words_right"] += sum(a == b for a, b in zip(want_words, got_words))
        own = ar.romanize_with_model(loaded, p.arabic, (p.latin,)).text
        unrelated = ar.romanize_with_model(loaded, p.arabic, (other,)).text
        stats["exact_own_hint"] += train.compare_key(own) == want
        stats["exact_unrelated_hint"] += train.compare_key(unrelated) == want
        if got != want and shown < show:
            print(f"  {p.arabic:28} want {want:28} got {got:28} old {got_old}")
            shown += 1
    n = stats["names"] or 1
    result = {
        k: round(stats[k] / n, 4)
        for k in ("exact", "old_exact", "exact_own_hint", "exact_unrelated_hint")
    }
    result["words"] = round(stats["words_right"] / max(stats["words_total"], 1), 4)
    result["names"] = stats["names"]
    return result


def _pairs(args: argparse.Namespace) -> list[train.Pair]:
    if args.pairs:
        return train.load_pairs_tsv(args.pairs)
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        return train.extract_pairs(conn)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["eval", "extract"])
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--pairs", type=Path, help="read pairs from this TSV instead")
    ap.add_argument("--out", type=Path, default=DEFAULT_TSV, help="for extract")
    ap.add_argument("--show", type=int, default=0, help="print N mismatches")
    ap.add_argument("--limit", type=int, default=8000, help="held-out names scored")
    args = ap.parse_args(argv)

    if args.command == "extract":
        if args.pairs:
            ap.error("extract reads the database; --pairs is for eval")
        pairs = _pairs(args)
        train.write_pairs_tsv(pairs, args.out)
        print(f"{args.out}: {len(pairs):,} pairs")
        return 0

    pairs = _pairs(args)
    counts = train.count([p for p in pairs if not train.is_test(p)])
    model = train.build_model(counts, "eval")
    print(
        f"train: {counts.names_aligned}/{counts.names_total} names aligned; "
        f"{len(model['table'])} contexts, {len(model['lexicon'])} words, "
        f"{len(model['construct_head'])} construct heads"
    )
    result = evaluate(
        [p for p in pairs if train.is_test(p)], model, args.show, args.limit
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
