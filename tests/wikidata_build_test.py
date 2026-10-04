"""The streaming Wikidata build on a small synthetic dump: bzip2 resume at
every block, extraction, and interrupted builds giving the same result as an
uninterrupted one.

Run: uv run python tests/wikidata_build_test.py   (under a minute)
"""

import bz2
import gzip
import json
import random
import sys
import tempfile

from pathlib import Path

from name_transduction_engine.datasets.wikidata import classes as C
from name_transduction_engine.datasets.wikidata.build import (
    BuildOptions,
    build_wikidata_compact,
)
from name_transduction_engine.datasets.wikidata.bz2_resume import (
    decode,
    find_block_starts,
)

rng = random.Random(1)
SYLLABLES = ["ka", "lo", "mi", "ra", "ta", "ne", "so", "vi", "ber", "gal", "dun"]
CYRILLIC = "абвгдежзиклмнопрстуфхцчш"


def word(n: int = 3) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(n)).capitalize()


def snak(prop, datavalue, datatype):
    return {
        "snaktype": "value",
        "property": prop,
        "hash": "%040x" % rng.getrandbits(160),
        "datavalue": datavalue,
        "datatype": datatype,
    }


def item(qid):
    n = int(qid[1:])
    return {
        "value": {"entity-type": "item", "numeric-id": n, "id": qid},
        "type": "wikibase-entityid",
    }


def year(y):
    return {
        "value": {
            "time": f"{'-' if y < 0 else '+'}{abs(y):04d}-00-00T00:00:00Z",
            "precision": 9,
            "timezone": 0,
            "before": 0,
            "after": 0,
            "calendarmodel": "http://www.wikidata.org/entity/Q1985727",
        },
        "type": "time",
    }


def entity(qid, p31, labels, *, coord=None, extra=None, sitelinks=0, rank="normal"):
    claims = {
        "P31": [
            {
                "mainsnak": snak("P31", item(p31), "wikibase-item"),
                "type": "statement",
                "id": f"{qid}$1",
                "rank": rank,
            }
        ]
    }
    if coord:
        claims["P625"] = [
            {
                "mainsnak": snak(
                    "P625",
                    {
                        "value": {
                            "latitude": coord[0],
                            "longitude": coord[1],
                            "globe": f"http://www.wikidata.org/entity/{coord[2] if len(coord) > 2 else 'Q2'}",
                        },
                        "type": "globecoordinate",
                    },
                    "globe-coordinate",
                ),
                "type": "statement",
                "id": f"{qid}$2",
                "rank": "normal",
            }
        ]
    for prop, value, dtype, qualifiers in extra or ():
        stmt = {
            "mainsnak": snak(prop, value, dtype),
            "type": "statement",
            "id": f"{qid}${prop}",
            "rank": "normal",
        }
        if qualifiers:
            stmt["qualifiers"] = qualifiers
        claims.setdefault(prop, []).append(stmt)
    sites = ["enwiki", "dewiki", "frwiki", "ruwiki", "commonswiki", "plwiki", "lawiki"]
    return {
        "type": "item",
        "id": qid,
        "labels": {lang: {"language": lang, "value": v} for lang, v in labels.items()},
        "aliases": {
            "en": [
                {"language": "en", "value": "FR-75"},
                {"language": "en", "value": "Old " + word(2)},
            ]
        },
        "claims": claims,
        "sitelinks": {
            s: {"site": s, "title": "t", "badges": []} for s in sites[:sitelinks]
        },
    }


def make_dump(path: Path) -> None:
    entities = []
    qid = 100
    for i in range(6000):
        qid += 1
        kind = rng.random()
        base = word()
        if kind < 0.3:
            labels = {
                "en": base,
                "de": base,
                "ru": "".join(rng.choice(CYRILLIC) for _ in range(6)),
            }
            official = (
                "P1448",
                {
                    "value": {"text": word(), "language": "ru"},
                    "type": "monolingualtext",
                },
                "monolingualtext",
                {"P580": [snak("P580", year(1914), "time")]},
            )
            entities.append(
                entity(
                    f"Q{qid}",
                    "Q515",
                    labels,
                    coord=(50.0, 10.0),
                    extra=[official],
                    sitelinks=rng.randint(0, 5),
                )
            )
        elif kind < 0.5:
            entities.append(
                entity(
                    f"Q{qid}",
                    "Q532",
                    {"en": base, "nl": base},
                    coord=(51.0, 11.0),
                    sitelinks=rng.randint(0, 3),
                )
            )
        elif kind < 0.55:
            dissolved = ("P576", year(rng.randint(-300, 1800)), "time", None)
            entities.append(
                entity(
                    f"Q{qid}",
                    "Q3024240",
                    {"en": base, "la": base + "ia"},
                    coord=(40.0, 20.0),
                    extra=[dissolved],
                )
            )
        elif kind < 0.57:
            entities.append(
                entity(
                    f"Q{qid}",
                    "Q8502",
                    {"en": base, "ru": base + "а"},
                    coord=(1.0, 1.0, "Q111"),
                    sitelinks=5,
                )
            )
        elif kind < 0.6:
            pleiades = (
                "P1584",
                {"value": str(qid), "type": "string"},
                "external-id",
                None,
            )
            entities.append(
                entity(
                    f"Q{qid}",
                    "Q109607",
                    {"en": base, "grc": base},
                    coord=(37.0, 23.0),
                    extra=[pleiades],
                )
            )
        else:
            cites = [
                ("P2860", item(f"Q{rng.randint(1, 10**7)}"), "wikibase-item", None)
                for _ in range(rng.randint(5, 60))
            ]
            entities.append(
                entity(f"Q{qid}", "Q13442814", {"en": "On " + word(5)}, extra=cites)
            )
    raw = (
        "[\n"
        + ",\n".join(json.dumps(e, ensure_ascii=False) for e in entities)
        + "\n]\n"
    )
    path.write_bytes(bz2.compress(raw.encode("utf-8"), compresslevel=1))


def check_block_resume(path: Path) -> None:
    data = path.read_bytes()
    original = bz2.decompress(data)

    def chunks(start, size=50_000):
        for i in range(start, len(data), size):
            yield i, data[i : i + size]

    whole = b"".join(out for out, _ in decode(chunks(0)))
    assert whole == original, "plain decode differs"
    blocks = find_block_starts(data)
    assert len(blocks) > 3, "test dump too small"
    sample = sorted(set(blocks[::20] + blocks[-3:]))
    for bit in sample:
        out = b"".join(o for o, _ in decode(chunks(bit // 8), bit, spliced=True))
        assert out and original.endswith(out), f"resume at bit {bit} failed"
    print(f"ok: resume from {len(sample)} of {len(blocks)} bzip2 blocks")


def compact_lines(directory: Path) -> list[bytes]:
    lines = []
    for part in sorted(directory.glob("*.jsonl.gz")):
        lines += gzip.open(part, "rb").read().splitlines()
    return sorted(lines)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dump = tmp / "wikidata-20990101-all.json.bz2"
        make_dump(dump)
        check_block_resume(dump)

        classes = tmp / "classes.tsv"
        C.write_class_map(
            {
                "Q515": "city",
                "Q532": "village",
                "Q3024240": "historical_country",
                "Q8502": "mountain",
            },
            "test",
            classes,
        )
        C.CLASSES_PATH = classes

        def run(name, **kw):
            options = BuildOptions(
                source=str(dump),
                work_dir=tmp / name / "work",
                compact_dir=tmp / name / "compact",
                **kw,
            )
            return build_wikidata_compact(options)

        assert run("clean") == 0
        expected = compact_lines(tmp / "clean" / "compact")
        manifest = json.loads((tmp / "clean" / "compact" / "manifest.json").read_text())
        stats = manifest["extraction"]
        assert stats.get("skip:offworld", 0) > 0, stats
        kinds = manifest["groups"]["historical_place"]["kinds"]
        assert (
            kinds.get("ancient_place", 0) > 0 and kinds.get("historical_country", 0) > 0
        )
        record = json.loads(expected[0])
        assert all(n["text"] != "FR-75" for n in record["names"]), "junk alias kept"

        steps = 0
        while True:
            steps += 1
            assert run("resumed", stop_after_gb=0.0002) == 0
            state = json.loads((tmp / "resumed" / "work" / "state.json").read_text())
            if state["status"] == "finished":
                break
            assert steps < 200
        assert compact_lines(tmp / "resumed" / "compact") == expected
        print(f"ok: {steps} stop/resume steps give the same {len(expected):,} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
