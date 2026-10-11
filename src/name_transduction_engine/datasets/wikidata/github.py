"""The GitHub releases that carry the published Wikidata datasets.

Each dataset is one release tagged `wikidata-YYYYMMDD` (the dump date) whose
assets are the files of a compact dataset directory.

Reading (download.py) needs no token: the repository is public, and listing
releases is one API call per fetch (60 an hour per IP without a token; asset
downloads do not count). Writing (publish.py) needs a token with write access
to the repository's contents; with it, drafts are listed too.
"""

import hashlib
import re
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, Self

import requests

from name_transduction_engine.datasets.shared import build_session

GITHUB_REPO: Final[str] = "Acueres/NameTransductionEngine"
GITHUB_API: Final[str] = "https://api.github.com"
RELEASE_TAG_RE: Final = re.compile(r"^wikidata-(\d{8})$")
USER_AGENT: Final[str] = (
    "NameTransductionEngine/0.1 (+https://github.com/Acueres/NameTransductionEngine)"
)

# Uploads report progress this often (seconds); the service logs to journald
_PROGRESS_SECONDS: Final[float] = 30.0


class GitHubError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class GitHubAuthError(GitHubError):
    """The token is missing, invalid, expired or lacks write access"""


@dataclass(frozen=True)
class Asset:
    id: int
    name: str
    size: int
    state: str  # "uploaded" once complete
    digest: str | None  # "sha256:<hex>" where GitHub provides it
    url: str  # public download URL

    @property
    def sha256(self) -> str | None:
        if self.digest and self.digest.startswith("sha256:"):
            return self.digest.split(":", 1)[1]
        return None


@dataclass(frozen=True)
class Release:
    id: int
    tag: str
    snapshot: str
    draft: bool
    upload_url: str
    body: str
    assets: dict[str, Asset]


def release_tag(snapshot: str) -> str:
    return f"wikidata-{snapshot}"


class GitHub:
    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.session = build_session()
        self.session.headers["User-Agent"] = USER_AGENT

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Reading

    def releases(self) -> list[Release]:
        """Wikidata dataset releases, newest snapshot first. Drafts are
        included when the token can see them"""
        releases: list[Release] = []
        url: str | None = f"{GITHUB_API}/repos/{GITHUB_REPO}/releases?per_page=100"
        while url:
            response = self._request("GET", url)
            for item in response.json():
                release = _release(item)
                if release is not None:
                    releases.append(release)
            url = response.links.get("next", {}).get("url")
        return sorted(releases, key=lambda r: (r.snapshot, not r.draft), reverse=True)

    def release_assets(self, release: Release) -> dict[str, Asset]:
        """The release's assets, fetched again (the list in a release object
        can be stale)"""
        assets: dict[str, Asset] = {}
        url: str | None = (
            f"{GITHUB_API}/repos/{GITHUB_REPO}/releases/{release.id}/assets"
            "?per_page=100"
        )
        while url:
            response = self._request("GET", url)
            for item in response.json():
                asset = _asset(item)
                assets[asset.name] = asset
            url = response.links.get("next", {}).get("url")
        return assets

    def download(self, url: str) -> bytes:
        response = self.session.get(url, timeout=(15, 120))
        response.raise_for_status()
        return response.content

    # Writing (token required)

    def create_draft(self, tag: str, name: str, body: str) -> Release:
        response = self._request(
            "POST",
            f"{GITHUB_API}/repos/{GITHUB_REPO}/releases",
            ok=(201,),
            json={
                "tag_name": tag,
                "name": name,
                "body": body,
                "draft": True,
                "prerelease": False,
            },
        )
        release = _release(response.json())
        assert release is not None
        return release

    def publish_draft(self, release: Release, body: str) -> None:
        # Data releases leave the repository's "Latest" label to code releases
        self._request(
            "PATCH",
            f"{GITHUB_API}/repos/{GITHUB_REPO}/releases/{release.id}",
            json={"draft": False, "body": body, "make_latest": "false"},
        )

    def upload_asset(
        self, release: Release, path: Path, name: str, content_type: str
    ) -> tuple[Asset, str]:
        """Upload `path` as asset `name`. Returns the asset and the SHA-256
        of the bytes actually sent"""
        base = release.upload_url.split("{", 1)[0]
        with path.open("rb") as f:
            body = _HashingReader(f, path.stat().st_size, name)
            response = self._request(
                "POST",
                base,
                ok=(201,),
                params={"name": name},
                data=body,
                headers={"Content-Type": content_type},
                timeout=(30, 1800),
            )
        return _asset(response.json()), body.hexdigest()

    def delete_asset(self, asset: Asset) -> None:
        self._request(
            "DELETE",
            f"{GITHUB_API}/repos/{GITHUB_REPO}/releases/assets/{asset.id}",
            ok=(204, 404),
        )

    def delete_release(self, release: Release) -> None:
        """Delete the release and, for a published one, its tag"""
        self._request(
            "DELETE",
            f"{GITHUB_API}/repos/{GITHUB_REPO}/releases/{release.id}",
            ok=(204, 404),
        )
        if not release.draft:
            # 422: the tag does not exist (any more)
            self._request(
                "DELETE",
                f"{GITHUB_API}/repos/{GITHUB_REPO}/git/refs/tags/{release.tag}",
                ok=(204, 404, 422),
            )

    # Plumbing

    def _request(
        self,
        method: str,
        url: str,
        *,
        ok: tuple[int, ...] = (200,),
        headers: dict[str, str] | None = None,
        timeout: tuple[float, float] = (15, 60),
        **kwargs: Any,
    ) -> requests.Response:
        all_headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            all_headers["Authorization"] = f"Bearer {self.token}"
        all_headers.update(headers or {})
        response = self.session.request(
            method, url, headers=all_headers, timeout=timeout, **kwargs
        )
        if response.status_code in ok:
            return response
        raise _error(response, method, url)


def _release(item: dict[str, Any]) -> Release | None:
    match = RELEASE_TAG_RE.match(item.get("tag_name") or "")
    if not match:
        return None
    return Release(
        id=item["id"],
        tag=item["tag_name"],
        snapshot=match.group(1),
        draft=bool(item.get("draft")),
        upload_url=item.get("upload_url", ""),
        body=item.get("body") or "",
        assets={a["name"]: _asset(a) for a in item.get("assets", [])},
    )


def _asset(item: dict[str, Any]) -> Asset:
    return Asset(
        id=item["id"],
        name=item["name"],
        size=item["size"],
        state=item.get("state", "uploaded"),
        digest=item.get("digest"),
        url=item.get("browser_download_url", ""),
    )


def _error(response: requests.Response, method: str, url: str) -> GitHubError:
    status = response.status_code
    try:
        payload = response.json()
        detail = payload.get("message", "")
        errors = payload.get("errors")
        if errors:
            detail += f" {errors}"
    except ValueError:
        detail = response.text[:200]
    remaining = response.headers.get("X-RateLimit-Remaining")
    if status in (403, 429) and (remaining == "0" or "rate limit" in detail.lower()):
        return GitHubError(
            "GitHub API rate limit reached; try again in an hour", status
        )
    where = url.split("?", 1)[0]
    if status == 401:
        return GitHubAuthError(
            f"GitHub rejected the token ({detail}); it may have expired", status
        )
    if status in (403, 404) and method != "GET":
        return GitHubAuthError(
            f"GitHub refused {method} {where} ({status}: {detail}); the token "
            "needs read and write access to the repository's contents",
            status,
        )
    return GitHubError(f"{method} {where} failed ({status}: {detail})", status)


class _HashingReader:
    """A file object for a streamed upload that hashes what is sent and
    reports progress. Its length makes requests send Content-Length (GitHub
    does not accept chunked uploads)"""

    def __init__(self, f: BinaryIO, size: int, name: str) -> None:
        self._f = f
        self._size = size
        self._name = name
        self._hash = hashlib.sha256()
        self._sent = 0
        self._started = time.monotonic()
        self._last = self._started

    def __len__(self) -> int:
        return self._size

    def read(self, n: int = -1) -> bytes:
        data = self._f.read(n if n and n > 0 else 1024 * 1024)
        self._hash.update(data)
        self._sent += len(data)
        now = time.monotonic()
        if now - self._last >= _PROGRESS_SECONDS:
            self._last = now
            rate = self._sent / max(now - self._started, 0.001) / 1024**2
            print(
                f"    {self._name}: {self._sent / 1024**2:,.0f}/"
                f"{self._size / 1024**2:,.0f} MiB ({self._sent / self._size:.0%}), "
                f"{rate:.1f} MiB/s",
                flush=True,
            )
        return data

    def hexdigest(self) -> str:
        return self._hash.hexdigest()
