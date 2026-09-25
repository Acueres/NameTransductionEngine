import sqlite3

from dataclasses import dataclass, field
from pathlib import Path

from name_transduction_engine.paths import (
    DB_PATH,
    RAW_DIR_GEONAMES,
    RAW_DIR_LANGUAGE_CODES,
    RAW_DIR_WIKIDATA,
    BUILD_DIR,
    WIKIDATA_LOCATIONS_PATH,
    WIKIDATA_RAW_DUMP_PATH,
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
    REGISTRY_FINGERPRINT_KEY as WIKIDATA_REGISTRY_FINGERPRINT_KEY,
    is_wikidata_ready,
)
from .wikidata.schema import LANGUAGE_TAG_TABLE as WIKIDATA_LANGUAGE_TAG_TABLE

# Tables whose row counts are worth reporting, per source
_LANGUAGE_CODES_TABLES = (
    "language",
    "language_alias",
    "language_retirement",
    "language_subtag",
)
_GEONAMES_TABLES = ("geoname", "alternate_name", GEONAMES_LANGUAGE_TAG_TABLE)
_WIKIDATA_TABLES = (
    "wikidata_location",
    "wikidata_location_name",
    "wikidata_location_geonames",
    "wikidata_location_p31",
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
class DataStatus:
    db: ArtifactStatus
    sources: list[SourceStatus]
    build_metadata: dict[str, str]  # empty if table absent
    raw_artifacts: list[ArtifactStatus]
    partial_files: list[ArtifactStatus]


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


def collect_data_status() -> DataStatus:
    """Gather a read-only snapshot of everything the data layer owns"""
    db = _artifact("names.sqlite", DB_PATH)

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
                        is_wikidata_ready(DB_PATH),
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
    raw_artifacts.append(_artifact("latest-all.json.bz2", WIKIDATA_RAW_DUMP_PATH))
    raw_artifacts.append(
        _artifact("wikidata_locations.jsonl.gz", WIKIDATA_LOCATIONS_PATH)
    )

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
