"""`nte data fetch wikidata`: download a published compact dataset from the
project's GitHub releases.

Each dataset is one release tagged `wikidata-YYYYMMDD` (the dump date) whose
assets are the files of a compact dataset directory: `manifest.json` and the
`wikidata-YYYYMMDD-<group>-<n>.jsonl.gz` parts. The newest such release is
the current one; `--tag` picks another.

Only the groups loaded into names.sqlite are downloaded unless all are asked
for. Every file is checked against the manifest's SHA-256 before the new copy
replaces the old one, and files already downloaded and intact are reused, so
an interrupted fetch continues where it stopped.
"""

import os
import shutil

from pathlib import Path
from typing import Any

from name_transduction_engine.datasets.shared import get_and_save_file
from name_transduction_engine.paths import WIKIDATA_DOWNLOAD_DIR
from . import classes as C
from .compact import (
    MANIFEST_NAME,
    CompactDataset,
    open_compact_dataset,
    parse_manifest,
    sha256_file,
)
from .github import GITHUB_REPO, GitHub, GitHubError, Release


class FetchError(RuntimeError):
    pass


def find_release(github: GitHub, tag: str | None = None) -> Release:
    try:
        releases = [r for r in github.releases() if not r.draft]
    except GitHubError as exc:
        raise FetchError(str(exc)) from exc
    if not releases:
        raise FetchError(
            f"no published Wikidata dataset (release tagged wikidata-YYYYMMDD) "
            f"in {GITHUB_REPO}"
        )
    if tag is None:
        return releases[0]
    for release in releases:
        if release.tag == tag:
            return release
    known = ", ".join(r.tag for r in releases[:5])
    raise FetchError(f"no release {tag!r} in {GITHUB_REPO} (newest: {known})")


def fetch_published_dataset(
    tag: str | None = None,
    *,
    all_groups: bool = False,
    force: bool = False,
    target: Path | None = None,
) -> CompactDataset:
    """Download the newest (or the tagged) published dataset into
    data/raw/wikidata/compact. Returns it"""
    target = target or WIKIDATA_DOWNLOAD_DIR
    with GitHub() as github:
        return _fetch(github, find_release(github, tag), all_groups, force, target)


def _fetch(
    github: GitHub, release: Release, all_groups: bool, force: bool, target: Path
) -> CompactDataset:
    session = github.session

    if MANIFEST_NAME not in release.assets:
        raise FetchError(f"release {release.tag} has no {MANIFEST_NAME}")
    manifest_bytes = github.download(release.assets[MANIFEST_NAME].url)
    manifest = parse_manifest(manifest_bytes, f"{release.tag}/{MANIFEST_NAME}")

    groups = list(manifest["groups"]) if all_groups else list(C.LOADED_GROUPS)
    wanted = [
        part
        for group in groups
        for part in (manifest["groups"].get(group) or {}).get("files", [])
    ]

    current = open_compact_dataset(target)
    if (
        not force
        and current is not None
        and current.dataset_id == manifest["dataset_id"]
        and all((target / part["name"]).is_file() for part in wanted)
    ):
        print(
            f"Wikidata dataset {manifest['dataset_id']} ({release.tag}) is "
            "already downloaded."
        )
        return current

    staging = target.with_name(target.name + ".part")
    staging.mkdir(parents=True, exist_ok=True)
    total = sum(part["bytes"] for part in wanted)
    print(
        f"Fetching Wikidata dataset {manifest['dataset_id']} from release "
        f"{release.tag}: {len(wanted)} file(s), {total / 1024**2:,.1f} MiB "
        f"({', '.join(groups)})"
    )

    for part in wanted:
        name = part["name"]
        path = staging / name
        if not path.is_file() and not force:
            _reuse(target / name, path, part)
        if path.is_file() and _intact(path, part):
            print(f"  {name}: already downloaded")
            continue
        if name not in release.assets:
            raise FetchError(f"release {release.tag} has no asset {name}")
        get_and_save_file(
            session=session, url=release.assets[name].url, output_path=path
        )
        if not _intact(path, part):
            path.unlink()
            raise FetchError(
                f"{name} does not match the manifest checksum; run the fetch again"
            )

    # Files of other datasets or groups no longer wanted
    keep = {part["name"] for part in wanted}
    for path in staging.iterdir():
        if path.name not in keep:
            path.unlink()
    (staging / MANIFEST_NAME).write_bytes(manifest_bytes)

    old = target.with_name(target.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        target.rename(old)
    staging.rename(target)
    if old.exists():
        shutil.rmtree(old)

    dataset = open_compact_dataset(target)
    assert dataset is not None
    print(f"Wikidata dataset {dataset.dataset_id} saved to {target}")
    return dataset


def _intact(path: Path, part: dict[str, Any]) -> bool:
    return path.stat().st_size == part["bytes"] and sha256_file(path) == part["sha256"]


def _reuse(source: Path, destination: Path, part: dict[str, Any]) -> None:
    """Take an intact file from the previous download instead of fetching it"""
    if source.is_file() and _intact(source, part):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
