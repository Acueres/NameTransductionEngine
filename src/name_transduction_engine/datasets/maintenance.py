import sqlite3

from dataclasses import dataclass, field
from pathlib import Path

from name_transduction_engine.paths import (
    DB_PATH,
    RAW_DIR_GEONAMES,
    RAW_DIR_WIKIDATA,
    BUILD_DIR,
    WIKIDATA_LOCATIONS_PATH,
    WIKIDATA_RAW_DUMP_PATH,
)
from .geonames.load import is_geonames_ready
from .wikidata.load import is_wikidata_ready

# Tables whose row counts are worth reporting, per source
_GEONAMES_TABLES = ("geoname", "alternate_name", "language_code")
_WIKIDATA_TABLES = (
    "wikidata_location",
    "wikidata_location_name",
    "wikidata_location_geonames",
    "wikidata_lang_norm",
)


# Status
@dataclass(frozen=True)
class ArtifactStatus:
    """A file on disk that the data layer cares about"""

    name: str
    path: Path
    exists: bool
    size_bytes: int | None  # None when the file does not exist


@dataclass(frozen=True)
class SourceStatus:
    """Readiness of one data source inside names.sqlite"""

    source: str
    ready: bool
    table_counts: dict[str, int]  # only tables that exist


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


def collect_data_status() -> DataStatus:
    """Gather a read-only snapshot of everything the data layer owns"""
    db = _artifact("names.sqlite", DB_PATH)

    sources: list[SourceStatus] = []
    build_metadata: dict[str, str] = {}

    if db.exists:
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                sources.append(
                    SourceStatus(
                        source="geonames",
                        ready=is_geonames_ready(DB_PATH),
                        table_counts=_table_counts(conn, _GEONAMES_TABLES),
                    )
                )
                sources.append(
                    SourceStatus(
                        source="wikidata",
                        ready=is_wikidata_ready(DB_PATH),
                        table_counts=_table_counts(conn, _WIKIDATA_TABLES),
                    )
                )
                build_metadata = _read_build_metadata(conn)
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            # Corrupt or locked DB
            sources = [
                SourceStatus(source="geonames", ready=False, table_counts={}),
                SourceStatus(source="wikidata", ready=False, table_counts={}),
            ]

    raw_artifacts = [
        _artifact(name, RAW_DIR_GEONAMES / name)
        for name in (
            "allCountries.zip",
            "alternateNamesV2.zip",
            "iso-languagecodes.txt",
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
        for raw_dir in (RAW_DIR_GEONAMES, RAW_DIR_WIKIDATA)
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
        lines.append(f"  {source.source}: {marker}")
        for table, count in source.table_counts.items():
            lines.append(f"    {table}: {count:,} rows")

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


# Clean
@dataclass(frozen=True)
class CleanReport:
    removed: list[ArtifactStatus] = field(default_factory=list)
    freed_bytes: int = 0
    preview: bool = False


def clean_data(include_raw: bool = False, preview: bool = False) -> CleanReport:
    """Remove temporary and (optionally) raw downloaded files.

    Always targeted: *.part files in both raw directories, plus orphaned
    *.meta.json resume-metadata files whose final artifact no longer exists.

    With include_raw=True, also removes the raw source files themselves.

    With preview=True, nothing is deleted; the report lists what would go.
    """
    targets: list[Path] = []

    # .part cleanup
    for d in (RAW_DIR_GEONAMES, RAW_DIR_WIKIDATA, BUILD_DIR):
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
    for d in (RAW_DIR_GEONAMES, RAW_DIR_WIKIDATA):
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
