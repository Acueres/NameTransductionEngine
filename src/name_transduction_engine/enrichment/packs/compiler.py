import csv
import json
import re
import sqlite3
import unicodedata

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from .resolver import BuiltinPackPaths


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------

# Accepts optional leading minus, then 4–6-digit year, then -MM-DD
PACK_DATE_RE = re.compile(r"^-?\d{4,6}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Required TSV columns
# ---------------------------------------------------------------------------

REQUIRED_ENTITY_NAME_COLUMNS = {
    "namespace",
    "entity_id",
    "output_name",
    "valid_from",
    "valid_to",
    "priority",
    "tags",
    "note",
}

REQUIRED_STRING_EXONYM_COLUMNS = {
    "input",
    "output_name",
    "valid_from",
    "valid_to",
    "priority",
    "note",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rebuild_builtin_pack(
    conn: sqlite3.Connection,
    paths: BuiltinPackPaths,
    manifest: dict[str, Any],
    content_hash: str,
) -> None:
    """
    (Re)build all DB rows for one pack inside an already-open transaction.
    The caller is responsible for committing.
    """
    pack_id = str(manifest["id"]).strip()

    if pack_id != paths.root.name:
        raise ValueError(
            f"Pack directory name {paths.root.name!r} does not match "
            f"manifest id {pack_id!r}"
        )

    _upsert_pack(conn, pack_id, manifest)
    _upsert_pack_install(conn, pack_id, paths, content_hash)
    _clear_pack_rows(conn, pack_id)
    _load_pipeline_steps(conn, pack_id, manifest)
    _load_entity_names(conn, pack_id, paths.entity_names)
    _load_string_exonyms(conn, pack_id, paths.string_exonyms)


# ---------------------------------------------------------------------------
# Pack and install upserts
# ---------------------------------------------------------------------------


def _upsert_pack(
    conn: sqlite3.Connection,
    pack_id: str,
    manifest: dict[str, Any],
) -> None:
    display_name = str(manifest["display_name"]).strip()
    bcp47 = str(manifest["bcp47"]).strip()
    version = str(manifest["version"]).strip()
    kind = str(manifest.get("kind", "historical")).strip()

    # Derive default_mode from the first pipeline step when not explicit.
    pipeline = manifest.get("pipeline") or []
    default_mode = str(
        manifest.get("default_mode")
        or (pipeline[0].get("step", "lookup_first") if pipeline else "lookup_first")
    ).strip()

    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)

    conn.execute(
        """
        INSERT INTO pack (
            pack_id, display_name, bcp47, version, kind,
            default_mode, manifest_json, enabled
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(pack_id) DO UPDATE SET
            display_name  = excluded.display_name,
            bcp47         = excluded.bcp47,
            version       = excluded.version,
            kind          = excluded.kind,
            default_mode  = excluded.default_mode,
            manifest_json = excluded.manifest_json,
            enabled       = excluded.enabled
        """,
        (pack_id, display_name, bcp47, version, kind, default_mode, manifest_json),
    )


def _upsert_pack_install(
    conn: sqlite3.Connection,
    pack_id: str,
    paths: BuiltinPackPaths,
    content_hash: str,
) -> None:
    conn.execute(
        """
        INSERT INTO pack_install (pack_id, source_path, content_hash, built_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(pack_id) DO UPDATE SET
            source_path  = excluded.source_path,
            content_hash = excluded.content_hash,
            built_at     = excluded.built_at
        """,
        (
            pack_id,
            str(paths.root),
            content_hash,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


# ---------------------------------------------------------------------------
# Clear existing pack rows
# ---------------------------------------------------------------------------


def _clear_pack_rows(conn: sqlite3.Connection, pack_id: str) -> None:
    conn.execute("DELETE FROM pack_pipeline_step WHERE pack_id = ?", (pack_id,))
    conn.execute("DELETE FROM pack_entity_name   WHERE pack_id = ?", (pack_id,))
    conn.execute("DELETE FROM pack_string_exonym WHERE pack_id = ?", (pack_id,))
    conn.execute("DELETE FROM pack_import        WHERE pack_id = ?", (pack_id,))


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _load_pipeline_steps(
    conn: sqlite3.Connection,
    pack_id: str,
    manifest: dict[str, Any],
) -> None:
    pipeline = manifest.get("pipeline")
    if not pipeline:
        return
    if not isinstance(pipeline, list):
        raise ValueError(f"manifest 'pipeline' must be a list in pack {pack_id!r}")

    rows: list[tuple[Any, ...]] = []

    for step_index, item in enumerate(pipeline):
        if not isinstance(item, dict):
            raise ValueError(
                f"Pipeline step must be a mapping in pack {pack_id!r}: {item!r}"
            )

        step_type = str(item.get("step", "")).strip()
        if not step_type:
            raise ValueError(
                f"Pipeline step {step_index} missing 'step' key in pack {pack_id!r}"
            )

        enabled = 0 if item.get("enabled") is False else 1

        rows.append(
            (
                pack_id,
                step_index,
                step_type,
                enabled,
                json.dumps(item, ensure_ascii=False, sort_keys=True),
            )
        )

    if rows:
        conn.executemany(
            """
            INSERT INTO pack_pipeline_step
                (pack_id, step_index, step_type, enabled, config_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


# ---------------------------------------------------------------------------
# Entity-anchored name mappings  (entity_names.tsv)
# ---------------------------------------------------------------------------


def _load_entity_names(
    conn: sqlite3.Connection,
    pack_id: str,
    tsv_path: Path,
) -> None:
    if not tsv_path.exists():
        return

    import_id = _register_import(conn, pack_id, tsv_path)

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(_strip_comments(f), delimiter="\t")
        _check_columns(reader, REQUIRED_ENTITY_NAME_COLUMNS, tsv_path)

        name_rows: list[tuple[Any, ...]] = []
        tag_rows: list[tuple[Any, ...]] = []
        prov_rows: list[tuple[Any, ...]] = []

        for line_no, raw in enumerate(reader, start=2):
            namespace = raw["namespace"].strip()
            entity_id = raw["entity_id"].strip()
            output_name = raw["output_name"].strip()
            valid_from = _clean_date(raw["valid_from"], "valid_from", line_no, tsv_path)
            valid_to = _clean_date(raw["valid_to"], "valid_to", line_no, tsv_path)
            priority = _parse_int(raw["priority"], "priority", line_no, tsv_path)
            note = _clean_text(raw["note"])
            tags = _parse_tags(raw["tags"])

            _require_nonempty(namespace, "namespace", line_no, tsv_path)
            _require_nonempty(entity_id, "entity_id", line_no, tsv_path)
            _require_nonempty(output_name, "output_name", line_no, tsv_path)
            _check_date_range(valid_from, valid_to, line_no, tsv_path)

            # For GeoNames entities, validate the ID is known in the DB.
            if namespace == "geonames":
                _validate_geonameid(conn, entity_id, line_no, tsv_path)

            name_rows.append(
                (
                    pack_id,
                    namespace,
                    entity_id,
                    output_name,
                    valid_from,
                    valid_to,
                    priority,
                    note,
                )
            )

            for tag in tags:
                tag_rows.append(
                    (
                        pack_id,
                        namespace,
                        entity_id,
                        output_name,
                        valid_from,
                        valid_to,
                        tag,
                    )
                )

            prov_rows.append(
                (
                    pack_id,
                    "pack_entity_name",
                    _entity_name_pk_json(
                        pack_id, namespace, entity_id, output_name, valid_from, valid_to
                    ),
                    import_id,
                    line_no,
                )
            )

    if name_rows:
        conn.executemany(
            """
            INSERT INTO pack_entity_name
                (pack_id, namespace, entity_id, output_name,
                 valid_from, valid_to, priority, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            name_rows,
        )
    if tag_rows:
        conn.executemany(
            """
            INSERT INTO pack_entity_name_tag
                (pack_id, namespace, entity_id, output_name,
                 valid_from, valid_to, tag)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            tag_rows,
        )
    if prov_rows:
        conn.executemany(
            """
            INSERT INTO pack_row_provenance
                (pack_id, table_name, row_pk_json, import_id, source_line)
            VALUES (?, ?, ?, ?, ?)
            """,
            prov_rows,
        )


# ---------------------------------------------------------------------------
# String-anchored exonyms  (string_exonyms.tsv)
# ---------------------------------------------------------------------------


def _load_string_exonyms(
    conn: sqlite3.Connection,
    pack_id: str,
    tsv_path: Path,
) -> None:
    if not tsv_path.exists():
        return

    import_id = _register_import(conn, pack_id, tsv_path)

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(_strip_comments(f), delimiter="\t")
        _check_columns(reader, REQUIRED_STRING_EXONYM_COLUMNS, tsv_path)

        exonym_rows: list[tuple[Any, ...]] = []
        prov_rows: list[tuple[Any, ...]] = []

        for line_no, raw in enumerate(reader, start=2):
            # The TSV column is 'input' (raw form); we normalise on import.
            input_normal = _normalise(raw["input"])
            output_name = raw["output_name"].strip()
            valid_from = _clean_date(raw["valid_from"], "valid_from", line_no, tsv_path)
            valid_to = _clean_date(raw["valid_to"], "valid_to", line_no, tsv_path)
            priority = _parse_int(raw["priority"], "priority", line_no, tsv_path)
            note = _clean_text(raw["note"])

            _require_nonempty(input_normal, "input", line_no, tsv_path)
            _require_nonempty(output_name, "output_name", line_no, tsv_path)
            _check_date_range(valid_from, valid_to, line_no, tsv_path)

            exonym_rows.append(
                (
                    pack_id,
                    input_normal,
                    output_name,
                    valid_from,
                    valid_to,
                    priority,
                    note,
                )
            )

            prov_rows.append(
                (
                    pack_id,
                    "pack_string_exonym",
                    _string_exonym_pk_json(
                        pack_id, input_normal, output_name, valid_from, valid_to
                    ),
                    import_id,
                    line_no,
                )
            )

    if exonym_rows:
        conn.executemany(
            """
            INSERT INTO pack_string_exonym
                (pack_id, input_normal, output_name,
                 valid_from, valid_to, priority, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            exonym_rows,
        )
    if prov_rows:
        conn.executemany(
            """
            INSERT INTO pack_row_provenance
                (pack_id, table_name, row_pk_json, import_id, source_line)
            VALUES (?, ?, ?, ?, ?)
            """,
            prov_rows,
        )


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _register_import(
    conn: sqlite3.Connection,
    pack_id: str,
    source_path: Path,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO pack_import (pack_id, source_file, imported_at)
        VALUES (?, ?, ?)
        """,
        (pack_id, source_path.name, datetime.now(timezone.utc).isoformat()),
    )
    return cursor.lastrowid  # type: ignore[return-value]


def _entity_name_pk_json(
    pack_id: str,
    namespace: str,
    entity_id: str,
    output_name: str,
    valid_from: str | None,
    valid_to: str | None,
) -> str:
    return json.dumps(
        {
            "pack_id": pack_id,
            "namespace": namespace,
            "entity_id": entity_id,
            "output_name": output_name,
            "valid_from": valid_from,
            "valid_to": valid_to,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _string_exonym_pk_json(
    pack_id: str,
    input_normal: str,
    output_name: str,
    valid_from: str | None,
    valid_to: str | None,
) -> str:
    return json.dumps(
        {
            "pack_id": pack_id,
            "input_normal": input_normal,
            "output_name": output_name,
            "valid_from": valid_from,
            "valid_to": valid_to,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# GeoNames validation
# ---------------------------------------------------------------------------


def _validate_geonameid(
    conn: sqlite3.Connection,
    entity_id: str,
    line_no: int,
    file_path: Path,
) -> None:
    try:
        geonameid = int(entity_id)
    except ValueError:
        raise ValueError(
            f"{file_path}:{line_no} entity_id for namespace 'geonames' must be "
            f"an integer, got {entity_id!r}"
        )

    exists = conn.execute(
        "SELECT 1 FROM geoname WHERE geonameid = ?",
        (geonameid,),
    ).fetchone()

    if exists is None:
        raise ValueError(
            f"{file_path}:{line_no} references unknown geonameid {geonameid} "
            "(not present in the geoname table — run geonames import first)"
        )


# ---------------------------------------------------------------------------
# TSV helpers
# ---------------------------------------------------------------------------


def _strip_comments(lines: Any) -> StringIO:
    """
    Lets TSV files carry documentation
    """
    buf = StringIO()
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.lstrip().startswith("#") or not stripped.strip():
            continue
        buf.write(stripped + "\n")
    buf.seek(0)
    return buf


def _check_columns(
    reader: csv.DictReader,
    required: set[str],
    file_path: Path,
) -> None:
    fieldnames = set(reader.fieldnames or [])
    missing = required - fieldnames
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {sorted(missing)}")


def _parse_tags(value: str) -> list[str]:
    """
    Parse a comma-separated tag string into a sorted list of non-empty tokens
    """
    return sorted(t.strip().lower() for t in value.split(",") if t.strip())


# ---------------------------------------------------------------------------
# Value cleaners and validators
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """NFC-normalise, casefold, strip.  Applied to all lookup keys on import"""
    return unicodedata.normalize("NFC", text.strip()).casefold()


def _clean_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _clean_date(
    value: str,
    field: str,
    line_no: int,
    file_path: Path,
) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    if not PACK_DATE_RE.fullmatch(stripped):
        raise ValueError(
            f"{file_path}:{line_no} invalid {field} {stripped!r}; "
            "expected (-)YYYY-MM-DD with a four-or-more digit year"
        )
    return stripped


def _parse_int(
    value: str,
    field: str,
    line_no: int,
    file_path: Path,
) -> int:
    try:
        return int(value.strip())
    except ValueError:
        raise ValueError(
            f"{file_path}:{line_no} invalid integer for {field!r}: {value!r}"
        )


def _require_nonempty(value: str, field: str, line_no: int, file_path: Path) -> None:
    if not value:
        raise ValueError(f"{file_path}:{line_no} {field!r} must not be empty")


def _check_date_range(
    valid_from: str | None,
    valid_to: str | None,
    line_no: int,
    file_path: Path,
) -> None:
    if valid_from is None or valid_to is None:
        return
    if _date_key(valid_from) > _date_key(valid_to):
        raise ValueError(
            f"{file_path}:{line_no} invalid range: "
            f"valid_from {valid_from} is after valid_to {valid_to}"
        )


def _date_key(value: str) -> tuple[int, int, int]:
    """
    Return (year, month, day) as a sortable tuple.
    Handles both '1066-09-15' and '-0660-01-01'.
    """
    sign = -1 if value.startswith("-") else 1
    body = value[1:] if sign == -1 else value
    y, m, d = body.split("-")
    return (sign * int(y), int(m), int(d))
