"""Evaluate the Persian display-romanization model (a developer tool).

The model itself is built by `nte init` / `nte data build models`
(models/persian_romanization/). This trains a throwaway model on 90% of the GeoNames
pairs and scores it on the held-out 10%, against the generic Arabic-script
rules, so changes to the trainer or the reader can be measured.

    python -m name_transduction_engine.tools.persian_model eval [--show N]
    python -m name_transduction_engine.tools.persian_model extract  # pairs -> TSV

Both read names.sqlite, or a TSV written by `extract` with `--pairs`.
"""

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

from name_transduction_engine.models.persian_romanization import train
from name_transduction_engine.paths import DB_PATH
from name_transduction_engine.transliteration.romanization_packs import persian as fa
from name_transduction_engine.transliteration.romanization_packs.arabic_script import (
    romanize_arabic_script,
)
from name_transduction_engine.transliteration.romanization_packs.base import (
    parse_context,
)

DEFAULT_TSV = Path("data/persian_eval/ir_pairs.tsv")


def evaluate(pairs: list[train.Pair], model: dict, show: int = 0) -> dict[str, float]:
    loaded = fa.make_model(model)
    ctx, _ = parse_context("fa")
    stats: collections.Counter = collections.Counter()
    shown = 0
    for p in pairs:
        if train.align_name(p.persian, p.latin) is None:
            continue  # not a BGN transliteration of this spelling
        want = train.compare_key(p.latin)
        got = train.compare_key(fa.romanize_with_model(loaded, p.persian).text)
        old = romanize_arabic_script(p.persian, ctx)
        got_old = train.compare_key(old.text if old else p.persian)
        stats["names"] += 1
        stats["new_exact"] += got == want
        stats["old_exact"] += got_old == want
        stats["new_skeleton"] += _skeleton(got) == _skeleton(want)
        stats["old_skeleton"] += _skeleton(got_old) == _skeleton(want)
        if got != want and shown < show:
            print(f"  {p.persian:28} want {want:28} got {got:28} old {got_old}")
            shown += 1
    n = stats["names"] or 1
    return {k: stats[k] / n for k in stats if k != "names"} | {"names": stats["names"]}


def _skeleton(name: str) -> list[str]:
    """Words of a folded name, ignoring the ezāfe and hyphens"""
    return re.sub(r"-(y?e)\b", "", name).replace("-", " ").split()


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
        f"{len(model['ezafe_head'])} ezāfe heads"
    )
    result = evaluate([p for p in pairs if train.is_test(p)], model, args.show)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
