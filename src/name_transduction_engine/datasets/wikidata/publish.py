"""`nte data publish wikidata`: publish the finished build output
(data/build/wikidata/compact) as a GitHub release and prune old releases.

1. The local dataset: a partial one, or one built from an undated local
   dump, is refused. Every file listed in the manifest must be present with
   its size.
2. Release `wikidata-<snapshot>`:
   - already published with the same dataset_id: nothing to upload;
   - published with another dataset_id (the same dump finalized again, e.g.
     after `--refinalize`): refused unless `--replace`, which deletes it first;
   - otherwise the draft left by an interrupted run is continued, or a new
     draft created.
3. Each file not already in the draft intact is uploaded, hashed while it is
   sent and compared with the manifest; manifest.json goes last.
4. The draft's assets are checked against the manifest, then it is published.
   NTE ignores drafts, so nobody downloads a half-uploaded release.
5. Pruning: the newest `keep` published releases (and the one just
   published) stay; older ones are deleted with their tags, and so are
   abandoned drafts of older dumps. Releases with other tag names are never
   touched.

Running it again after success does nothing but prune, so the build service
can run it after every build, including builds that found nothing new.

The token comes from GITHUB_TOKEN or the systemd credential `nte-github-token`
(ImportCredential=, see deploy/systemd). It needs read and write access to the
repository's contents.
"""

import os
import re
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import requests

from name_transduction_engine.paths import WIKIDATA_COMPACT_DIR
from . import classes as C
from .compact import MANIFEST_NAME, parse_manifest, read_manifest, sha256_file
from .github import (
    Asset,
    GitHub,
    GitHubAuthError,
    GitHubError,
    Release,
    release_tag,
)

KEEP_RELEASES: Final[int] = 2
TOKEN_ENV: Final[str] = "GITHUB_TOKEN"
CREDENTIAL_NAME: Final[str] = "nte-github-token"
UPLOAD_ATTEMPTS: Final[int] = 3
RETRY_SECONDS: float = 60.0  # times the attempt number


class PublishError(RuntimeError):
    """Publishing failed; running it again later may succeed"""


class PublishAttention(PublishError):
    """Publishing cannot succeed until someone acts (token, conflict, local
    data)"""


@dataclass(frozen=True)
class PublishOptions:
    keep: int = KEEP_RELEASES
    replace: bool = False
    dry_run: bool = False
    compact_dir: Path = WIKIDATA_COMPACT_DIR


@dataclass(frozen=True)
class _File:
    name: str
    path: Path
    size: int
    sha256: str
    content_type: str


def read_token() -> str | None:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if token:
        return token
    credentials = os.environ.get("CREDENTIALS_DIRECTORY")
    if credentials:
        path = Path(credentials) / CREDENTIAL_NAME
        if path.is_file():
            return path.read_text(encoding="utf-8").strip() or None
    return None


def publish_wikidata_dataset(options: PublishOptions | None = None) -> None:
    options = options or PublishOptions()
    if options.keep < 1:
        raise PublishAttention("at least one release must be kept")

    try:
        manifest = read_manifest(options.compact_dir)
    except ValueError as exc:
        raise PublishAttention(str(exc)) from exc
    if manifest is None:
        print(f"[publish] no compact dataset in {options.compact_dir}; nothing to do.")
        return
    tag = _tag_for(manifest)
    files = _local_files(options.compact_dir, manifest)

    token = read_token()
    if token is None and not options.dry_run:
        raise PublishAttention(
            f"no GitHub token: set {TOKEN_ENV}, or give the service the "
            f"credential {CREDENTIAL_NAME} (see deploy/systemd)"
        )
    if options.dry_run:
        print("[publish] dry run: nothing is changed on GitHub.")
        if token is None:
            print("[publish] no token: drafts are not visible.")

    with GitHub(token) as github:
        try:
            _publish(github, manifest, tag, files, options)
            _prune(github, options.keep, tag, options.dry_run)
        except GitHubAuthError as exc:
            raise PublishAttention(str(exc)) from exc
        except GitHubError as exc:
            raise PublishError(str(exc)) from exc
        except requests.RequestException as exc:
            raise PublishError(f"network error: {exc}") from exc


# Local dataset


def _tag_for(manifest: dict[str, Any]) -> str:
    dataset_id = manifest["dataset_id"]
    if manifest.get("partial"):
        raise PublishAttention(
            f"the compact dataset {dataset_id} is partial "
            f"({manifest.get('progress', 0):.1%} of the dump); only complete "
            "builds are published"
        )
    snapshot = manifest["snapshot"]
    if not re.fullmatch(r"\d{8}", snapshot):
        raise PublishAttention(
            f"the compact dataset {dataset_id} was built from an undated dump "
            f"({snapshot}); only dated dumps are published"
        )
    return release_tag(snapshot)


def _local_files(directory: Path, manifest: dict[str, Any]) -> list[_File]:
    files: list[_File] = []
    for group, entry in manifest["groups"].items():
        for part in entry["files"]:
            path = directory / part["name"]
            if not path.is_file() or path.stat().st_size != part["bytes"]:
                raise PublishAttention(
                    f"{path} is missing or not the size the manifest lists "
                    f"({group}); write the dataset again with "
                    "`nte data build wikidata-compact --refinalize`"
                )
            files.append(
                _File(
                    part["name"],
                    path,
                    part["bytes"],
                    part["sha256"],
                    "application/gzip",
                )
            )
    manifest_path = directory / MANIFEST_NAME
    files.append(
        _File(
            MANIFEST_NAME,
            manifest_path,
            manifest_path.stat().st_size,
            sha256_file(manifest_path),
            "application/json",
        )
    )
    return files


# Publishing


def _publish(
    github: GitHub,
    manifest: dict[str, Any],
    tag: str,
    files: list[_File],
    options: PublishOptions,
) -> None:
    dataset_id = manifest["dataset_id"]
    same_tag = [r for r in github.releases() if r.tag == tag]
    live = next((r for r in same_tag if not r.draft), None)
    drafts = [r for r in same_tag if r.draft]

    if live is not None:
        remote_id = _remote_dataset_id(github, live)
        if remote_id == dataset_id:
            missing = _check_assets(github, live, files)
            if missing:
                raise PublishAttention(
                    f"release {tag} is published but its assets do not match "
                    f"the dataset ({', '.join(missing)}); fix it by hand or "
                    "run with --replace"
                )
            print(f"[publish] {tag} ({dataset_id}) is already published.")
            return
        if not options.replace:
            raise PublishAttention(
                f"release {tag} is already published with dataset "
                f"{remote_id or 'unknown'}, but this build is {dataset_id}. "
                "Run `nte data publish wikidata --replace` to replace it"
            )
        verb = "would replace" if options.dry_run else "replacing"
        print(f"[publish] {verb} {tag} ({remote_id or 'unknown dataset'}).")
        if not options.dry_run:
            github.delete_release(live)

    total = sum(f.size for f in files)
    if options.dry_run:
        how = "continuing its draft" if drafts else "as a new draft"
        print(
            f"[publish] would publish {tag} ({dataset_id}): {len(files)} files, "
            f"{total / 1024**2:,.1f} MiB, {how}."
        )
        return

    body = _release_body(manifest, files)
    if drafts:
        draft = drafts[0]
        print(f"[publish] continuing the draft of {tag}.")
    else:
        draft = github.create_draft(tag, _release_name(manifest), body)
        print(f"[publish] created a draft of {tag}.")
    for extra in drafts[1:]:
        github.delete_release(extra)

    print(
        f"[publish] uploading {dataset_id}: {len(files)} files, "
        f"{total / 1024**2:,.1f} MiB"
    )
    _upload_all(github, draft, files, fresh=dataset_id not in draft.body)

    missing = _check_assets(github, draft, files)
    if missing:
        raise PublishError(
            f"the draft of {tag} does not match the dataset after uploading "
            f"({', '.join(missing)}); run again"
        )
    # A build that finished meanwhile would have replaced the files
    current = read_manifest(options.compact_dir)
    if current is None or current["dataset_id"] != dataset_id:
        raise PublishError(
            "the compact dataset changed while it was uploaded; run again"
        )
    github.publish_draft(draft, body)
    print(f"[publish] published {tag} ({dataset_id}).")


def _upload_all(
    github: GitHub, draft: Release, files: list[_File], fresh: bool
) -> None:
    """Upload what the draft does not have intact. `fresh`: the draft was
    started for another dataset, so none of its assets is reused"""
    assets = github.release_assets(draft)
    wanted = {f.name for f in files}
    for name, asset in assets.items():
        if fresh or name not in wanted:
            github.delete_asset(asset)
    if fresh:
        assets = {}

    for f in files:
        asset = assets.get(f.name)
        if asset is not None and _intact(asset, f):
            print(f"  {f.name}: already uploaded")
            continue
        if asset is not None:
            github.delete_asset(asset)
        _upload_one(github, draft, f)


def _upload_one(github: GitHub, draft: Release, f: _File) -> None:
    last: Exception | None = None
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        print(f"  {f.name}: uploading {f.size / 1024**2:,.1f} MiB", flush=True)
        started = time.monotonic()
        try:
            asset, sent = github.upload_asset(draft, f.path, f.name, f.content_type)
        except GitHubAuthError:
            raise
        except (GitHubError, requests.RequestException, OSError) as exc:
            last = exc
            print(f"  {f.name}: upload failed ({exc})", flush=True)
            # A failed upload can leave a broken asset under this name
            broken = github.release_assets(draft).get(f.name)
            if broken is not None:
                github.delete_asset(broken)
            if attempt < UPLOAD_ATTEMPTS:
                time.sleep(RETRY_SECONDS * attempt)
            continue
        if sent != f.sha256 or asset.size != f.size:
            github.delete_asset(asset)
            raise PublishError(
                f"{f.name} does not match the manifest checksum; the dataset "
                "may have changed during the upload, run again"
            )
        seconds = max(time.monotonic() - started, 0.001)
        print(f"  {f.name}: done ({f.size / 1024**2 / seconds:.1f} MiB/s)", flush=True)
        return
    raise PublishError(f"could not upload {f.name}: {last}")


def _intact(asset: Asset, f: _File) -> bool:
    return (
        asset.state == "uploaded"
        and asset.size == f.size
        and asset.sha256 in (None, f.sha256)
    )


def _check_assets(github: GitHub, release: Release, files: list[_File]) -> list[str]:
    """Files of the dataset the release lacks or holds differently"""
    assets = github.release_assets(release)
    return [
        f.name for f in files if f.name not in assets or not _intact(assets[f.name], f)
    ]


def _remote_dataset_id(github: GitHub, release: Release) -> str | None:
    asset = release.assets.get(MANIFEST_NAME)
    if asset is None:
        return None
    try:
        data = github.download(asset.url)
        return parse_manifest(data, f"{release.tag}/{MANIFEST_NAME}")["dataset_id"]
    except (requests.RequestException, ValueError, KeyError):
        return None


def _release_name(manifest: dict[str, Any]) -> str:
    s = manifest["snapshot"]
    return f"Wikidata {s[:4]}-{s[4:6]}-{s[6:]}"


def _release_body(manifest: dict[str, Any], files: list[_File]) -> str:
    s = manifest["snapshot"]
    sizes = {f.name: f.size for f in files}
    lines = [
        f"Compact Wikidata dataset built from the dump of {s[:4]}-{s[4:6]}-{s[6:]}.",
        (
            "NTE downloads it with `nte data fetch wikidata` (or `nte init` "
            "when nothing is downloaded yet) and loads it into names.sqlite."
        ),
        "",
        f"- Dataset: `{manifest['dataset_id']}`",
        f"- Built: {manifest.get('built_at', '?')}",
        f"- Dump: {manifest.get('dump', {}).get('url', '?')}",
        (
            f"- Extractor version {manifest.get('extractor_version', '?')}, "
            f"class file `{manifest.get('classes', {}).get('sha256', '?')[:12]}`"
        ),
        "",
        "| Group | Records | Size | Loaded |",
        "|---|---:|---:|---|",
    ]
    for group, entry in manifest["groups"].items():
        size = sum(sizes.get(part["name"], 0) for part in entry["files"])
        loaded = "yes" if group in C.LOADED_GROUPS else "no"
        lines.append(
            f"| {group} | {entry['records']:,} | {size / 1024**2:,.1f} MiB | {loaded} |"
        )
    lines += [
        "",
        (
            "Each file's SHA-256 is listed in manifest.json. Wikidata content "
            "is CC0 (https://www.wikidata.org/wiki/Wikidata:Licensing)."
        ),
    ]
    return "\n".join(lines)


# Pruning


def _prune(github: GitHub, keep: int, tag: str, dry_run: bool) -> None:
    releases = github.releases()
    snapshot = tag.split("-", 1)[1]
    # In a dry run the new release is not there yet; count it all the same
    live_tags = sorted(
        {r.tag for r in releases if not r.draft} | {tag},
        key=lambda t: t.split("-", 1)[1],
        reverse=True,
    )
    kept = set(live_tags[:keep]) | {tag}
    doomed = [r for r in releases if not r.draft and r.tag not in kept]
    doomed += [r for r in releases if r.draft and r.snapshot < snapshot]

    names = ", ".join(t for t in live_tags if t in kept)
    if tag not in live_tags[:keep]:
        print(
            f"[publish] note: {tag} is older than the {keep} newest releases; "
            "it is kept anyway"
        )
    if not doomed:
        print(f"[publish] releases kept: {names}; nothing to prune.")
        return
    verb = "would delete" if dry_run else "deleting"
    for release in doomed:
        print(f"[publish] {verb} {release.tag}{' (draft)' if release.draft else ''}")
        if not dry_run:
            github.delete_release(release)
    print(f"[publish] releases kept: {names}.")
