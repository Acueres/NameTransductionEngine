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
import re
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import requests

from name_transduction_engine.datasets.shared import build_session, get_and_save_file
from name_transduction_engine.paths import WIKIDATA_DOWNLOAD_DIR
from . import classes as C
from .compact import (
    MANIFEST_NAME,
    CompactDataset,
    open_compact_dataset,
    parse_manifest,
    sha256_file,
)

GITHUB_REPO: Final[str] = "Acueres/NameTransductionEngine"
GITHUB_API: Final[str] = "https://api.github.com"
RELEASE_TAG_RE: Final = re.compile(r"^wikidata-(\d{8})$")
USER_AGENT: Final[str] = (
    "NameTransductionEngine/0.1 (+https://github.com/Acueres/NameTransductionEngine)"
)


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Release:
    tag: str
    snapshot: str
    assets: dict[str, str]  # asset name -> download URL


def _session() -> requests.Session:
    session = build_session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def list_releases(session: requests.Session) -> list[Release]:
    """Published Wikidata datasets, newest first"""
    # The repository is public: no token. Listing releases is one API call
    # per fetch (unauthenticated limit: 60 an hour per IP); the asset
    # downloads themselves are not rate limited
    headers = {"Accept": "application/vnd.github+json"}

    releases: list[Release] = []
    url: str | None = f"{GITHUB_API}/repos/{GITHUB_REPO}/releases?per_page=100"
    while url:
        response = session.get(url, headers=headers, timeout=(15, 60))
        if response.status_code == 403 and "rate limit" in response.text.lower():
            raise FetchError("GitHub API rate limit reached; try again in an hour")
        response.raise_for_status()
        for item in response.json():
            match = RELEASE_TAG_RE.match(item.get("tag_name", ""))
            if not match or item.get("draft"):
                continue
            assets = {
                asset["name"]: asset["browser_download_url"]
                for asset in item.get("assets", [])
            }
            releases.append(Release(item["tag_name"], match.group(1), assets))
        url = response.links.get("next", {}).get("url")

    return sorted(releases, key=lambda r: r.snapshot, reverse=True)


def find_release(session: requests.Session, tag: str | None = None) -> Release:
    releases = list_releases(session)
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
    session = _session()
    release = find_release(session, tag)

    if MANIFEST_NAME not in release.assets:
        raise FetchError(f"release {release.tag} has no {MANIFEST_NAME}")
    manifest_bytes = _get(session, release.assets[MANIFEST_NAME])
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
        get_and_save_file(session=session, url=release.assets[name], output_path=path)
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


def _get(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=(15, 120))
    response.raise_for_status()
    return response.content


def _intact(path: Path, part: dict[str, Any]) -> bool:
    return path.stat().st_size == part["bytes"] and sha256_file(path) == part["sha256"]


def _reuse(source: Path, destination: Path, part: dict[str, Any]) -> None:
    """Take an intact file from the previous download instead of fetching it"""
    if source.is_file() and _intact(source, part):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
