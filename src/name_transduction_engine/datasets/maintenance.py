import sqlite3

from dataclasses import dataclass, field
from pathlib import Path

from name_transduction_engine.paths import (
    DB_PATH,
    RAW_DIR_GEONAMES,
    RAW_DIR_LANGUAGE_CODES,
    RAW_DIR_WIKIDATA,
    BUILD_DIR,
    WIKIDATA_COMPACT_DIR,
    WIKIDATA_DOWNLOAD_DIR,
    WIKIDATA_RAW_DUMP_GLOB,
    WIKIDATA_WORK_DIR,
)
from .language_codes.download import (
    IANA_REGISTRY_FILENAME,
    ISO_LANGUAGECODES_FILENAME,
)
from .language_codes.load import (
    REGISTRY_FINGERPRINT_KEY as LANGUAGE_REGISTRY_FINGERPRINT_KEY,
    is_language_codes_ready,
)
from .geonames.load import (
    REGISTRY_FINGERPRINT_KEY as GEONAMES_REGISTRY_FINGERPRINT_KEY,
    is_geonames_ready,
)
from .geonames.schema import LANGUAGE_TAG_TABLE as GEONAMES_LANGUAGE_TAG_TABLE
from .wikidata.load import (
    DATASET_ID_KEY as WIKIDATA_DATASET_ID_KEY,
    REGISTRY_FINGERPRINT_KEY as WIKIDATA_REGISTRY_FINGERPRINT_KEY,
    is_wikidata_ready,
)
from .wikidata.schema import LANGUAGE_TAG_TABLE as WIKIDATA_LANGUAGE_TAG_TABLE
from .wikidata.build import committed_shards, read_state as read_wikidata_state
from .wikidata.classes import ClassFileError, load_class_map
from .wikidata.compact import format_manifest_summary, read_manifest
from .wikidata.data_provision import current_wikidata_dataset
from name_transduction_engine.models.model_provider import (
    ModelStatus,
    collect_model_status,
)

# Tables whose row counts are worth reporting, per source
_LANGUAGE_CODES_TABLES = (
    "language",
    "language_alias",
    "language_retirement",
    "language_subtag",
)
_GEONAMES_TABLES = ("geoname", "alternate_name", GEONAMES_LANGUAGE_TAG_TABLE)
_WIKIDATA_TABLES = (
    "wikidata_entity",
    "wikidata_name",
    "wikidata_link",
    "wikidata_entity_class",
    "wikidata_geonames",
    "wikidata_external_id",
    WIKIDATA_LANGUAGE_TAG_TABLE,
)

_RAW_DIRS = (RAW_DIR_LANGUAGE_CODES, RAW_DIR_GEONAMES, RAW_DIR_WIKIDATA)

# How many unmapped tags to list per source in `nte data status`
_UNMAPPED_TAGS_SHOWN = 10


# Status
@dataclass(frozen=True)
class ArtifactStatus:
    """A file on disk that the data layer cares about"""

    name: str
    path: Path
    exists: bool
    size_bytes: int | None  # None when the file does not exist


@dataclass(frozen=True)
class LanguageTagSummary:
    """How a source's raw language tags mapped to the registry, from its
    language tag report table"""

    rows_by_status: dict[str, int]  # tag_status -> number of name rows
    distinct_tags: int
    unmapped: list[
        tuple[str, int, str | None]
    ]  # (raw tag, rows, reason), most rows first
    unmapped_total: int  # distinct unmapped tags, including those not listed


@dataclass(frozen=True)
class SourceStatus:
    """Readiness of one data source inside names.sqlite"""

    source: str
    ready: bool
    table_counts: dict[str, int]  # only tables that exist
    # Why the source is not ready, when that can be told from the outside
    reason: str | None = None
    language_tags: LanguageTagSummary | None = None


@dataclass(frozen=True)
class WikidataBuildStatus:
    """The Wikidata compact datasets: the published copy downloaded by
    `nte data fetch wikidata` (the one `nte init` loads), and the build that
    produces new releases (`nte data build wikidata-compact`, its run and
    output), which is never loaded"""

    state: dict | None  # work/state.json
    shard_bytes: int
    manifest: dict | None  # build/wikidata/compact/manifest.json
    compact_bytes: int
    classes: str  # state of the class file the build needs
    published: dict | None = None  # raw/wikidata/compact/manifest.json
    published_bytes: int = 0
    loadable: bool = False  # the downloaded copy holds every loaded group


@dataclass(frozen=True)
class DataStatus:
    db: ArtifactStatus
    sources: list[SourceStatus]
    build_metadata: dict[str, str]  # empty if table absent
    raw_artifacts: list[ArtifactStatus]
    partial_files: list[ArtifactStatus]
    models: list[ModelStatus] = field(default_factory=list)
    wikidata_build: WikidataBuildStatus | None = None


def _artifact(name: str, path: Path) -> ArtifactStatus:
    exists = path.is_file()
    return ArtifactStatus(
        name=name,
        path=path,
        exists=exists,
        size_bytes=path.stat().st_size if exists else None,
    )


def _table_counts(conn: sqlite3.Connection, tables: tuple[str, ...]) -> dict[str, int]:
    existing = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    }
    counts: dict[str, int] = {}
    for table in tables:
        if table in existing:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
    return counts


def _read_build_metadata(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        return dict(conn.execute("SELECT key, value FROM build_metadata;"))
    except sqlite3.DatabaseError:
        return {}


def _language_tag_summary(
    conn: sqlite3.Connection, table: str
) -> LanguageTagSummary | None:
    try:
        rows = conn.execute(
            f"SELECT raw_tag, tag_status, row_count, note FROM {table} "
            "ORDER BY row_count DESC, raw_tag"
        ).fetchall()
    except sqlite3.DatabaseError:
        return None
    if not rows:
        return None

    rows_by_status: dict[str, int] = {}
    for _, status, count, _ in rows:
        rows_by_status[status] = rows_by_status.get(status, 0) + count
    unmapped = [
        (raw, count, note) for raw, status, count, note in rows if status == "unmapped"
    ]

    return LanguageTagSummary(
        rows_by_status=dict(sorted(rows_by_status.items())),
        distinct_tags=len(rows),
        unmapped=unmapped[:_UNMAPPED_TAGS_SHOWN],
        unmapped_total=len(unmapped),
    )


def _registry_mismatch(metadata: dict[str, str], built_with_key: str) -> str | None:
    """Explain a dataset built against a different language registry"""
    current = metadata.get(LANGUAGE_REGISTRY_FINGERPRINT_KEY)
    built_with = metadata.get(built_with_key)
    if current is None or built_with is None or built_with == current:
        return None
    return "built with a different language registry; run `nte init` to rebuild"


def _wikidata_expected_dataset() -> str | None:
    """dataset_id of the downloaded dataset `nte init` would load"""
    try:
        dataset = current_wikidata_dataset()
    except (OSError, ValueError):
        return None
    return dataset.dataset_id if dataset else None


def _class_file_state() -> str:
    try:
        class_map = load_class_map()
    except ClassFileError as exc:
        return f"not usable: {exc}"
    return (
        f"{len(class_map.kinds):,} classes, generated {class_map.generated_at} "
        f"from {class_map.endpoint}"
    )


def _collect_wikidata_build() -> WikidataBuildStatus:
    state = read_wikidata_state(WIKIDATA_WORK_DIR)
    try:
        manifest = read_manifest(WIKIDATA_COMPACT_DIR)
    except (OSError, ValueError):
        manifest = None
    shard_bytes = (
        sum(p.stat().st_size for p in committed_shards(WIKIDATA_WORK_DIR, state))
        if state
        else 0
    )
    compact_bytes = (
        sum(p.stat().st_size for p in WIKIDATA_COMPACT_DIR.iterdir() if p.is_file())
        if manifest
        else 0
    )
    try:
        published = read_manifest(WIKIDATA_DOWNLOAD_DIR)
    except (OSError, ValueError):
        published = None
    published_bytes = (
        sum(p.stat().st_size for p in WIKIDATA_DOWNLOAD_DIR.iterdir() if p.is_file())
        if published
        else 0
    )
    return WikidataBuildStatus(
        state,
        shard_bytes,
        manifest,
        compact_bytes,
        _class_file_state(),
        published,
        published_bytes,
        _wikidata_expected_dataset() is not None,
    )


def collect_data_status() -> DataStatus:
    """Gather a read-only snapshot of everything the data layer owns"""
    db = _artifact("names.sqlite", DB_PATH)
    wikidata_expected = _wikidata_expected_dataset()

    sources: list[SourceStatus] = []
    build_metadata: dict[str, str] = {}

    if db.exists:
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                build_metadata = _read_build_metadata(conn)

                language_codes_ready = is_language_codes_ready(DB_PATH)
                sources.append(
                    SourceStatus(
                        source="language_codes",
                        ready=language_codes_ready,
                        table_counts=_table_counts(conn, _LANGUAGE_CODES_TABLES),
                    )
                )
                for source, ready, tables, tag_table, fingerprint_key in (
                    (
                        "geonames",
                        is_geonames_ready(DB_PATH),
                        _GEONAMES_TABLES,
                        GEONAMES_LANGUAGE_TAG_TABLE,
                        GEONAMES_REGISTRY_FINGERPRINT_KEY,
                    ),
                    (
                        "wikidata",
                        is_wikidata_ready(DB_PATH, wikidata_expected),
                        _WIKIDATA_TABLES,
                        WIKIDATA_LANGUAGE_TAG_TABLE,
                        WIKIDATA_REGISTRY_FINGERPRINT_KEY,
                    ),
                ):
                    reason = None
                    if not ready:
                        reason = (
                            "language registry not ready"
                            if not language_codes_ready
                            else _registry_mismatch(build_metadata, fingerprint_key)
                        )
                        if (
                            reason is None
                            and source == "wikidata"
                            and wikidata_expected is not None
                            and build_metadata.get(WIKIDATA_DATASET_ID_KEY)
                            not in (None, wikidata_expected)
                        ):
                            reason = (
                                "a different Wikidata compact dataset is "
                                "available; run `nte init` to load it"
                            )
                    sources.append(
                        SourceStatus(
                            source=source,
                            ready=ready,
                            table_counts=_table_counts(conn, tables),
                            reason=reason,
                            language_tags=_language_tag_summary(conn, tag_table),
                        )
                    )
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            # Corrupt or locked DB
            sources = [
                SourceStatus(source=name, ready=False, table_counts={})
                for name in ("language_codes", "geonames", "wikidata")
            ]

    raw_artifacts = [
        _artifact(name, RAW_DIR_LANGUAGE_CODES / name)
        for name in (IANA_REGISTRY_FILENAME, ISO_LANGUAGECODES_FILENAME)
    ]
    raw_artifacts += [
        _artifact(name, RAW_DIR_GEONAMES / name)
        for name in (
            "allCountries.zip",
            "alternateNamesV2.zip",
            "admin1CodesASCII.txt",
            "admin2Codes.txt",
        )
    ]
    if RAW_DIR_WIKIDATA.is_dir():
        raw_artifacts += [
            _artifact(path.name, path)
            for path in sorted(RAW_DIR_WIKIDATA.glob(WIKIDATA_RAW_DUMP_GLOB))
        ]

    partial_files = [
        _artifact(path.name, path)
        for raw_dir in _RAW_DIRS
        if raw_dir.is_dir()
        for path in sorted(raw_dir.glob("*.part"))
    ]

    return DataStatus(
        db=db,
        sources=sources,
        build_metadata=build_metadata,
        raw_artifacts=raw_artifacts,
        partial_files=partial_files,
        models=collect_model_status(),
        wikidata_build=_collect_wikidata_build(),
    )


def format_data_status(status: DataStatus) -> str:
    """Render a DataStatus as a human-readable report"""
    lines: list[str] = []

    if status.db.exists:
        lines.append(
            f"Database: {status.db.path} ({_human_bytes(status.db.size_bytes)})"
        )
    else:
        lines.append(f"Database: {status.db.path} (missing; run `nte init`)")

    for source in status.sources:
        marker = "ready" if source.ready else "NOT READY"
        if source.reason:
            marker += f" ({source.reason})"
        lines.append(f"  {source.source}: {marker}")
        for table, count in source.table_counts.items():
            lines.append(f"    {table}: {count:,} rows")
        if source.language_tags is not None:
            lines.extend(_format_language_tags(source.language_tags))

    if status.build_metadata:
        lines.append("  build metadata:")
        for key, value in sorted(status.build_metadata.items()):
            lines.append(f"    {key}: {value}")

    if status.wikidata_build is not None:
        lines.extend(_format_wikidata_build(status.wikidata_build))

    if status.models:
        lines.append("Models:")
        for model in status.models:
            lines.extend(_format_model(model))

    lines.append("Raw files:")
    for artifact in status.raw_artifacts:
        if artifact.exists:
            lines.append(f"  {artifact.name}: {_human_bytes(artifact.size_bytes)}")
        else:
            lines.append(f"  {artifact.name}: absent")

    if status.partial_files:
        lines.append("Partial downloads (removable with `nte data clean`):")
        for artifact in status.partial_files:
            lines.append(f"  {artifact.name}: {_human_bytes(artifact.size_bytes)}")

    return "\n".join(lines)


def _format_wikidata_build(build: WikidataBuildStatus) -> list[str]:
    lines = ["Wikidata datasets:"]
    published = build.published
    if published is None:
        lines.append(
            "  downloaded (loaded by `nte init`): none yet; `nte init` or "
            "`nte data fetch wikidata` downloads the newest release"
        )
    else:
        note = "" if build.loadable else "; incomplete, run `nte data fetch wikidata`"
        lines.append(
            f"  downloaded (loaded by `nte init`): {published['dataset_id']} "
            f"({_human_bytes(build.published_bytes)}{note})"
        )
        lines.extend(
            f"    {line}"
            for line in format_manifest_summary(published, WIKIDATA_DOWNLOAD_DIR)
        )
    # The build is for machines that build datasets; skip it where none was
    if (
        build.state is not None
        or build.manifest is not None
        or (not build.classes.startswith("not usable"))
    ):
        lines.append(
            f"  build for publishing (not loaded), class file: {build.classes}"
        )
    state = build.state
    if state is not None:
        dump = state["dump"]
        progress = state["position"] / dump["size"] if dump["size"] else 0.0
        status = {
            "running": f"in progress, {progress:.1%} read",
            "read": "dump read, compact dataset not written yet",
            "finished": "finished",
        }.get(state["status"], state["status"])
        lines.append(f"  run: dump {dump['snapshot']} ({status})")
        lines.append(f"    source: {dump['url']}")
        kept = ", ".join(
            f"{key.split(':', 1)[1]} {count:,}"
            for key, count in sorted(state.get("stats", {}).items())
            if key.startswith("kept:")
        )
        lines.append(
            f"    entities read: {state['lines']:,}; kept: {kept or 'none yet'}"
        )
        lines.append(
            f"    shards: {state['shards']} ({_human_bytes(build.shard_bytes)}); "
            f"last checkpoint {state.get('updated_at', '?')}; "
            f"active {state.get('active_seconds', 0) / 3600:.1f} h"
        )
    manifest = build.manifest
    if manifest is not None:
        partial = (
            f", partial: {manifest.get('progress', 0):.1%} of the dump"
            if manifest.get("partial")
            else ""
        )
        lines.append(
            f"  build output (to publish): {manifest['dataset_id']} "
            f"({_human_bytes(build.compact_bytes)}{partial})"
        )
        lines.extend(
            f"    {line}"
            for line in format_manifest_summary(manifest, mark_loaded=False)
        )
    elif state is not None:
        lines.append("  build output (to publish): none yet")
    return lines


def _format_model(model: ModelStatus) -> list[str]:
    marker = "ready" if model.ready else "NOT READY"
    if model.reason:
        marker += f" ({model.reason})"
    lines = [f"  {model.name}: {marker}"]
    if model.size_bytes is not None:
        lines.append(f"    file: {model.path} ({_human_bytes(model.size_bytes)})")
    meta = model.meta
    if meta:
        lines.append(f"    built: {meta.get('built_at', '?')}")
        if "names_aligned" in meta:
            lines.append(
                f"    training: {meta['names_aligned']:,}/{meta.get('names_total', 0):,}"
                f" names aligned; {meta.get('contexts', 0):,} contexts, "
                f"{meta.get('words', 0):,} words"
            )
        if "source" in meta:
            lines.append(f"    source: {meta['source']}")
    return lines


def _format_language_tags(summary: LanguageTagSummary) -> list[str]:
    by_status = ", ".join(
        f"{status} {count:,}" for status, count in summary.rows_by_status.items()
    )
    lines = [f"    language tags: {summary.distinct_tags} distinct; rows: {by_status}"]
    if summary.unmapped_total:
        shown = len(summary.unmapped)
        more = summary.unmapped_total - shown
        lines.append(
            f"    unmapped tags ({summary.unmapped_total}"
            + (f", top {shown} shown" if more else "")
            + "):"
        )
        for raw, count, note in summary.unmapped:
            lines.append(
                f"      {raw!r}: {count:,} rows" + (f" ({note})" if note else "")
            )
    return lines


# Clean
@dataclass(frozen=True)
class CleanReport:
    removed: list[ArtifactStatus] = field(default_factory=list)
    freed_bytes: int = 0
    preview: bool = False


def clean_data(include_raw: bool = False, preview: bool = False) -> CleanReport:
    """Remove temporary and (optionally) raw downloaded files.

    Always targeted: *.part files in every raw directory, plus orphaned
    *.meta.json resume-metadata files whose final artifact no longer exists.

    With include_raw=True, also removes the raw source files themselves.

    With preview=True, nothing is deleted; the report lists what would go.
    """
    targets: list[Path] = []

    # .part cleanup
    for d in (*_RAW_DIRS, BUILD_DIR):
        if not d.is_dir():
            continue

        targets.extend(sorted(d.glob("*.part")))

        for meta_path in sorted(d.glob("*.meta.json")):
            final_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json"))
            part_path = final_path.with_suffix(final_path.suffix + ".part")
            final_will_remain = final_path.exists() and not include_raw
            part_will_remain = part_path.exists() and part_path not in targets
            if not final_will_remain and not part_will_remain:
                targets.append(meta_path)

    # raw cleanup
    for d in _RAW_DIRS:
        if not d.is_dir():
            continue

        if include_raw:
            targets.extend(
                sorted(
                    path
                    for path in d.iterdir()
                    if path.is_file()
                    and path.suffix != ".part"
                    and not path.name.endswith(".meta.json")
                )
            )

    removed: list[ArtifactStatus] = []
    freed = 0

    for path in targets:
        if not path.is_file():
            continue
        size = path.stat().st_size
        if not preview:
            path.unlink()
        removed.append(
            ArtifactStatus(name=path.name, path=path, exists=False, size_bytes=size)
        )
        freed += size

    return CleanReport(removed=removed, freed_bytes=freed, preview=preview)


def format_clean_report(report: CleanReport) -> str:
    """Render a CleanReport as a human-readable summary"""
    verb = "Would remove" if report.preview else "Removed"

    if not report.removed:
        return "Nothing to clean."

    lines = [
        f"{verb} {artifact.path} ({_human_bytes(artifact.size_bytes)})"
        for artifact in report.removed
    ]
    lines.append(
        f"{verb} {len(report.removed)} file(s), "
        f"{_human_bytes(report.freed_bytes)} total."
    )
    return "\n".join(lines)


# Helpers
def _human_bytes(num: int | None) -> str:
    if num is None:
        return "?"
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"
