"""The compact Wikidata dataset on disk: writing it from build shards,
reading it back, and the legacy single-file format.

Layout of a compact dataset directory:

    manifest.json
    wikidata-<snapshot>-<group>-<n>.jsonl.gz   one or more parts per group

Each part stays under GitHub's 2 GiB release-asset limit. Records are
described in extract.py; the manifest lists files, counts and provenance.
"""

import gzip
import hashlib
import json
import shutil

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Iterator

import orjson

from . import classes as C
from .extract import EXTRACTOR_VERSION, passes_final_gate

FORMAT: Final[str] = "nte-wikidata-compact"
FORMAT_VERSION: Final[int] = 1
MANIFEST_NAME: Final[str] = "manifest.json"
MAX_PART_BYTES: Final[int] = 1900 * 1024 * 1024
GZIP_LEVEL: Final[int] = 6


@dataclass(frozen=True)
class CompactDataset:
    """A compact dataset ready to load: either a manifest directory or the
    legacy locations file"""

    dataset_id: str
    path: Path
    manifest: dict[str, Any] | None  # None for the legacy file

    @property
    def is_legacy(self) -> bool:
        return self.manifest is None

    @property
    def partial(self) -> bool:
        return bool(self.manifest and self.manifest.get("partial"))

    def files(self, group: str) -> list[Path]:
        if self.manifest is None:
            return [self.path] if group == C.PLACE else []
        entry = self.manifest["groups"].get(group) or {}
        return [self.path / f["name"] for f in entry.get("files", [])]

    def iter_records(self, groups: Iterable[str]) -> Iterator[dict[str, Any]]:
        if self.manifest is None:
            yield from _iter_legacy(self.path)
            return
        for group in groups:
            for path in self.files(group):
                yield from iter_jsonl_gz(path)


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path}: not an NTE Wikidata compact dataset")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format version {manifest.get('format_version')} is not "
            f"supported (expected {FORMAT_VERSION})"
        )
    return manifest


def open_compact_dataset(directory: Path, legacy_file: Path) -> CompactDataset | None:
    """The locally built dataset if there is one, else the legacy file"""
    manifest = read_manifest(directory)
    if manifest is not None:
        return CompactDataset(manifest["dataset_id"], directory, manifest)
    if legacy_file.is_file():
        stat = legacy_file.stat()
        return CompactDataset(f"legacy-{stat.st_size}", legacy_file, None)
    return None


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


# Legacy format: {"qid", "kind", "p31_qids", "geonames_ids", "lat", "lon",
# "names": [{"wd_lang", "name", "term_type", ...}]}


def _iter_legacy(path: Path) -> Iterator[dict[str, Any]]:
    for old in iter_jsonl_gz(path):
        kind = old["kind"]
        names: dict[tuple[str, str], list[str]] = {}
        for name in old.get("names", ()):
            key = (name["name"], name.get("term_type", "label"))
            names.setdefault(key, []).append(name["wd_lang"])
        record: dict[str, Any] = {
            "qid": old["qid"],
            "group": C.HISTORICAL_PLACE if kind == "historical_country" else C.PLACE,
            "kind": kind,
            "classes": old.get("p31_qids", []),
            "names": [
                {"text": text, "type": kind_, "langs": langs}
                for (text, kind_), langs in names.items()
            ],
        }
        if old.get("lat") is not None and old.get("lon") is not None:
            record["coord"] = [old["lat"], old["lon"]]
        if old.get("geonames_ids"):
            record["ids"] = {"geonames": old["geonames_ids"]}
        yield record


# Writing (finalize)


class _GroupWriter:
    def __init__(self, directory: Path, snapshot: str, group: str) -> None:
        self.directory = directory
        self.snapshot = snapshot
        self.group = group
        self.parts: list[dict[str, Any]] = []
        self._raw = None
        self._gz = None
        self._records = 0
        self._since_check = 0

    def write(self, line: bytes) -> None:
        if self._gz is None:
            self._open()
        assert self._gz is not None
        self._gz.write(line)
        self._records += 1
        self._since_check += 1
        if self._since_check >= 2000:
            self._since_check = 0
            if self._raw.tell() >= MAX_PART_BYTES:
                self._close()

    def _open(self) -> None:
        n = len(self.parts) + 1
        name = f"wikidata-{self.snapshot}-{self.group}-{n}.jsonl.gz"
        self._raw = open(self.directory / name, "wb")
        self._gz = gzip.GzipFile(
            filename="", mode="wb", fileobj=self._raw, compresslevel=GZIP_LEVEL, mtime=0
        )
        self.parts.append({"name": name, "records": 0})
        self._records = 0

    def _close(self) -> None:
        if self._gz is None:
            return
        self._gz.close()
        self._raw.close()
        self.parts[-1]["records"] = self._records
        self._gz = self._raw = None

    def close(self) -> list[dict[str, Any]]:
        self._close()
        for part in self.parts:
            path = self.directory / part["name"]
            part["bytes"] = path.stat().st_size
            part["sha256"] = _sha256(path)
        return self.parts


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_compact(
    shards: list[Path],
    target: Path,
    *,
    dump: dict[str, Any],
    classes: C.ClassMap,
    extraction_stats: dict[str, int],
    partial: bool,
    progress: float,
) -> dict[str, Any]:
    """Apply the final gates to the shard records and write a compact
    dataset to `target`, replacing any previous one only when complete"""
    snapshot = dump["snapshot"]
    work = target.with_name(target.name + ".part")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    writers: dict[str, _GroupWriter] = {}
    kept: dict[str, Counter[str]] = {}
    dropped: Counter[str] = Counter()

    for shard in shards:
        with gzip.open(shard, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                record = orjson.loads(line)
                if not passes_final_gate(record):
                    dropped[record["kind"]] += 1
                    continue
                group = record["group"]
                writer = writers.get(group)
                if writer is None:
                    writer = writers[group] = _GroupWriter(work, snapshot, group)
                writer.write(line if line.endswith(b"\n") else line + b"\n")
                kept.setdefault(group, Counter())[record["kind"]] += 1

    groups: dict[str, Any] = {}
    for group in C.GROUPS:
        if group not in writers:
            continue
        files = writers[group].close()
        groups[group] = {
            "records": sum(kept[group].values()),
            "kinds": dict(kept[group].most_common()),
            "files": files,
        }

    final_gates = C.final_gates_fingerprint()
    dataset_id = (
        f"{snapshot}-x{EXTRACTOR_VERSION}-c{classes.sha256[:8]}-g{final_gates[:8]}"
        + (f"-partial{int(progress * 1000)}" if partial else "")
    )
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "dataset_id": dataset_id,
        "snapshot": snapshot,
        "partial": partial,
        "progress": round(progress, 4),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dump": dump,
        "extractor_version": EXTRACTOR_VERSION,
        "classes": {
            "sha256": classes.sha256,
            "rules": classes.rules,
            "generated_at": classes.generated_at,
        },
        "final_gates": final_gates,
        "groups": groups,
        "dropped_by_final_gate": dict(dropped.most_common()),
        "extraction": extraction_stats,
        "license": "Wikidata content is CC0 (https://www.wikidata.org/wiki/Wikidata:Licensing)",
    }
    (work / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    old = target.with_name(target.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        target.rename(old)
    work.rename(target)
    if old.exists():
        shutil.rmtree(old)
    return manifest


def format_manifest_summary(manifest: dict[str, Any]) -> list[str]:
    lines = []
    for group, entry in manifest.get("groups", {}).items():
        size = sum(f["bytes"] for f in entry["files"])
        loaded = " (loaded)" if group in C.LOADED_GROUPS else ""
        lines.append(f"{group}: {entry['records']:,} records, {_mib(size)}{loaded}")
        top = list(entry["kinds"].items())[:8]
        lines.append("    " + ", ".join(f"{k} {n:,}" for k, n in top))
    return lines


def _mib(num: int) -> str:
    return f"{num / 1024 / 1024:,.1f} MiB"
