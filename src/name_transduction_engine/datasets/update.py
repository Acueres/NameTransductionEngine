import requests

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from name_transduction_engine import paths
from name_transduction_engine.language_packs.language_pack_provider import (
    ensure_builtin_packs,
)
from name_transduction_engine.models.model_provider import (
    collect_model_status,
    ensure_models,
)
from .availability import clear_stale_notice, data_lock, maintenance
from .dataset_provider import ensure_datasets
from .geonames.data_provision import current_geonames_source
from .geonames.download import GEONAMES_URLS
from .geonames.load import is_geonames_ready
from .language_codes.data_provision import current_language_codes_source
from .language_codes.download import LANGUAGE_CODES_URLS
from .language_codes.load import is_language_codes_ready
from .sources import download_sources, downloaded_at
from .wikidata.compact import open_compact_dataset
from .wikidata.data_provision import current_wikidata_dataset
from .wikidata.download import FetchError, fetch_published_dataset, find_release
from .wikidata.github import GitHub
from .wikidata.load import is_wikidata_ready

DEFAULT_MAX_AGE_DAYS: Final[int] = 60


@dataclass(frozen=True)
class _Source:
    label: str
    raw_dir: Path
    urls: dict[str, str]


def _sources() -> list[_Source]:
    return [
        _Source("Language codes", paths.RAW_DIR_LANGUAGE_CODES, LANGUAGE_CODES_URLS),
        _Source("GeoNames", paths.RAW_DIR_GEONAMES, GEONAMES_URLS),
    ]


def run_update(max_age_days: int = DEFAULT_MAX_AGE_DAYS, check: bool = False) -> int:
    """Returns a process exit code: 0 when everything is up to date
    afterwards, 1 when a source could not be refreshed"""
    with data_lock():
        clear_stale_notice()
        failed = False
        now = datetime.now(UTC)

        for source in _sources():
            if not _refresh_source(source, max_age_days, now, check):
                failed = True
        if not _refresh_wikidata(check):
            failed = True

        stale = stale_parts()
        if not stale:
            print("[update] names.sqlite and the models are up to date.")
            return 1 if failed else 0
        if check:
            print(f"[update] would rebuild: {', '.join(stale)}.")
            return 1 if failed else 0

        print(
            f"[update] rebuilding: {', '.join(stale)}. NTE is unavailable "
            "until this finishes."
        )
        with maintenance(f"rebuilding {', '.join(stale)}"):
            ensure_datasets()
            ensure_models()
            ensure_builtin_packs()
        print("[update] done; NTE is available again.")
        return 1 if failed else 0


def _refresh_source(
    source: _Source, max_age_days: int, now: datetime, check: bool
) -> bool:
    names = tuple(source.urls)
    when = downloaded_at(source.raw_dir, names)
    if when is None:
        state, due = "not downloaded yet", True
    else:
        age = (now - when).total_seconds() / 86400
        due = age >= max_age_days
        state = f"downloaded {when:%Y-%m-%d} ({age:.0f} days ago)"
        if not due:
            next_refresh = when + timedelta(days=max_age_days)
            state += f"; next refresh after {next_refresh:%Y-%m-%d}"
    if not due:
        print(f"[update] {source.label}: {state}.")
        return True
    if check:
        print(f"[update] {source.label}: {state}; would download again.")
        return True
    print(f"[update] {source.label}: {state}; downloading again.")
    try:
        download_sources(source.raw_dir, source.urls, force=True, label=source.label)
    except (requests.RequestException, OSError) as exc:
        print(
            f"[update] {source.label}: download failed ({exc}); keeping the old files."
        )
        return False
    return True


def _refresh_wikidata(check: bool) -> bool:
    current = open_compact_dataset(paths.WIKIDATA_DOWNLOAD_DIR)
    try:
        if check:
            with GitHub() as github:
                newest = find_release(github)
            if current is not None and current.snapshot == newest.snapshot:
                print(
                    f"[update] Wikidata: {current.dataset_id} downloaded; "
                    f"{newest.tag} is the newest release."
                )
            else:
                have = current.dataset_id if current else "nothing downloaded"
                print(f"[update] Wikidata: {have}; would download {newest.tag}.")
            return True
        # Keep the groups a previous `--all-groups` fetch downloaded
        all_groups = current is not None and all(
            current.has_group(group) for group in current.manifest["groups"]
        )
        dataset = fetch_published_dataset(all_groups=all_groups)
        print(f"[update] Wikidata: {dataset.dataset_id}.")
        return True
    except (FetchError, requests.RequestException, OSError, ValueError) as exc:
        print(f"[update] Wikidata: could not check for a new release ({exc}).")
        return False


def stale_parts() -> list[str]:
    """What `nte init` would rebuild now"""
    db = paths.DB_PATH
    parts = []
    if not is_language_codes_ready(db, current_language_codes_source()):
        parts.append("language codes (and what was loaded with the old registry)")
    if not is_geonames_ready(db, current_geonames_source()):
        parts.append("GeoNames")
    dataset = current_wikidata_dataset()
    if dataset is None or not is_wikidata_ready(db, dataset.dataset_id):
        parts.append("Wikidata")
    if any(not model.ready for model in collect_model_status()):
        parts.append("models")
    return parts
