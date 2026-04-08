import csv
import json
import re
import sqlite3
import unicodedata

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .resolver import BuiltinPackPaths


PACK_DATE_RE = re.compile(r"^-?\d{4,6}-\d{2}-\d{2}$")

REQUIRED_OVERRIDE_COLUMNS = {
    "namespace",
    "external_id",
    "output_name",
    "valid_from",
    "valid_to",
    "priority",
    "tags",
    "note",
}
REQUIRED_EXONYM_COLUMNS = {
    "input_normalized",
    "output_name",
    "valid_from",
    "valid_to",
    "priority",
    "note",
}


def rebuild_builtin_pack(
    conn: sqlite3.Connection,
    paths: BuiltinPackPaths,
    manifest: dict[str, Any],
    content_hash: str,
) -> None:
    pack_id = str(manifest["id"]).strip()
    display_name = str(manifest["display_name"]).strip()
    version = str(manifest["version"]).strip()
    default_mode = (
        str(manifest.get("default_mode", "lookup_first")).strip() or "lookup_first"
    )

    if pack_id != paths.root.name:
        raise ValueError(
            f"Pack directory name {paths.root.name!r} does not match manifest id {pack_id!r}"
        )

    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)

    with conn:
        conn.execute(
            """
            INSERT INTO pack (
                pack_id,
                display_name,
                origin,
                version,
                default_mode,
                enabled,
                source_path,
                manifest_json,
                content_hash
            )
            VALUES (?, ?, 'builtin', ?, ?, 1, ?, ?, ?)
            ON CONFLICT(pack_id) DO UPDATE SET
                display_name = excluded.display_name,
                origin = excluded.origin,
                version = excluded.version,
                default_mode = excluded.default_mode,
                enabled = excluded.enabled,
                source_path = excluded.source_path,
                manifest_json = excluded.manifest_json,
                content_hash = excluded.content_hash
            """,
            (
                pack_id,
                display_name,
                version,
                default_mode,
                str(paths.root),
                manifest_json,
                content_hash,
            ),
        )

        _clear_pack_rows(conn, pack_id)
        _load_pack_fallbacks(conn, pack_id, manifest)
        _load_pack_place_overrides(conn, pack_id, paths.place_overrides)
        _load_pack_exonyms(conn, pack_id, paths.exonyms)

        conn.execute(
            """
            INSERT INTO pack_build (pack_id, built_at, content_hash)
            VALUES (?, ?, ?)
            ON CONFLICT(pack_id) DO UPDATE SET
                built_at = excluded.built_at,
                content_hash = excluded.content_hash
            """,
            (
                pack_id,
                datetime.now(timezone.utc).isoformat(),
                content_hash,
            ),
        )


def _clear_pack_rows(conn: sqlite3.Connection, pack_id: str) -> None:
    conn.execute("DELETE FROM pack_fallback_step WHERE pack_id = ?", (pack_id,))
    conn.execute("DELETE FROM pack_place_override WHERE pack_id = ?", (pack_id,))
    conn.execute("DELETE FROM pack_exonym WHERE pack_id = ?", (pack_id,))


def _load_pack_fallbacks(
    conn: sqlite3.Connection,
    pack_id: str,
    manifest: dict[str, Any],
) -> None:
    fallbacks = manifest.get("fallbacks", [])
    if fallbacks is None:
        return

    if not isinstance(fallbacks, list):
        raise ValueError(f"Manifest fallbacks must be a list for pack {pack_id}")

    rows: list[tuple[Any, ...]] = []

    for step_index, item in enumerate(fallbacks):
        if not isinstance(item, dict):
            raise ValueError(
                f"Fallback step must be a mapping in pack {pack_id}: {item!r}"
            )

        step_type = str(item.get("type", "")).strip()
        if not step_type:
            raise ValueError(f"Fallback step missing 'type' in pack {pack_id}")

        # Generic target field for simple steps.
        step_target = (
            item.get("target")
            or item.get("pack_id")
            or item.get("source")
            or item.get("language")
        )
        step_target_str = str(step_target).strip() if step_target is not None else None

        rows.append(
            (
                pack_id,
                step_index,
                step_type,
                step_target_str,
                json.dumps(item, ensure_ascii=False, sort_keys=True),
            )
        )

    if rows:
        conn.executemany(
            """
            INSERT INTO pack_fallback_step (
                pack_id,
                step_index,
                step_type,
                step_target,
                config_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def _load_pack_place_overrides(
    conn: sqlite3.Connection,
    pack_id: str,
    overrides_path: Path,
) -> None:
    if not overrides_path.exists():
        return

    with overrides_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_OVERRIDE_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"place_overrides.tsv for pack {pack_id} is missing columns {sorted(missing)}"
            )

        rows: list[tuple[Any, ...]] = []

        for line_no, raw in enumerate(reader, start=2):
            namespace = raw["namespace"].strip()
            external_id = raw["external_id"].strip()
            output_name = raw["output_name"].strip()
            valid_from = _clean_optional_date(
                raw["valid_from"].strip(),
                field="valid_from",
                line_no=line_no,
                file_path=overrides_path,
            )
            valid_to = _clean_optional_date(
                raw["valid_to"].strip(),
                field="valid_to",
                line_no=line_no,
                file_path=overrides_path,
            )
            priority = _parse_required_int(
                raw["priority"].strip(),
                field="priority",
                line_no=line_no,
                file_path=overrides_path,
            )
            tags = _clean_optional_text(raw["tags"])
            note = _clean_optional_text(raw["note"])

            if not namespace:
                raise ValueError(f"{overrides_path}:{line_no} namespace is required")
            if not external_id:
                raise ValueError(f"{overrides_path}:{line_no} external_id is required")
            if not output_name:
                raise ValueError(f"{overrides_path}:{line_no} output_name is required")

            _validate_date_range(
                valid_from=valid_from,
                valid_to=valid_to,
                file_path=overrides_path,
                line_no=line_no,
            )

            geonameid: int | None = None
            if namespace == "geonames":
                geonameid = _parse_required_int(
                    external_id,
                    field="external_id",
                    line_no=line_no,
                    file_path=overrides_path,
                )
                exists = conn.execute(
                    "SELECT 1 FROM geoname WHERE geonameid = ?",
                    (geonameid,),
                ).fetchone()
                if exists is None:
                    raise ValueError(
                        f"{overrides_path}:{line_no} references unknown geonameid {geonameid}"
                    )

            rows.append(
                (
                    pack_id,
                    namespace,
                    external_id,
                    geonameid,
                    output_name,
                    _normalize_lookup_key(output_name),
                    valid_from,
                    valid_to,
                    priority,
                    tags,
                    note,
                    overrides_path.name,
                    line_no,
                )
            )

        if rows:
            conn.executemany(
                """
                INSERT INTO pack_place_override (
                    pack_id,
                    namespace,
                    external_id,
                    geonameid,
                    output_name,
                    output_name_normalized,
                    valid_from,
                    valid_to,
                    priority,
                    tags,
                    note,
                    source_file,
                    source_line
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def _load_pack_exonyms(
    conn: sqlite3.Connection,
    pack_id: str,
    exonyms_path: Path,
) -> None:
    if not exonyms_path.exists():
        return

    with exonyms_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_EXONYM_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"exonyms.tsv for pack {pack_id} is missing columns {sorted(missing)}"
            )

        rows: list[tuple[Any, ...]] = []

        for line_no, raw in enumerate(reader, start=2):
            input_normalized = _normalize_lookup_key(raw["input_normalized"])
            output_name = raw["output_name"].strip()
            valid_from = _clean_optional_date(
                raw["valid_from"].strip(),
                field="valid_from",
                line_no=line_no,
                file_path=exonyms_path,
            )
            valid_to = _clean_optional_date(
                raw["valid_to"].strip(),
                field="valid_to",
                line_no=line_no,
                file_path=exonyms_path,
            )
            priority = _parse_required_int(
                raw["priority"].strip(),
                field="priority",
                line_no=line_no,
                file_path=exonyms_path,
            )
            note = _clean_optional_text(raw["note"])

            if not input_normalized:
                raise ValueError(
                    f"{exonyms_path}:{line_no} input_normalized is required"
                )
            if not output_name:
                raise ValueError(f"{exonyms_path}:{line_no} output_name is required")

            _validate_date_range(
                valid_from=valid_from,
                valid_to=valid_to,
                file_path=exonyms_path,
                line_no=line_no,
            )

            rows.append(
                (
                    pack_id,
                    input_normalized,
                    output_name,
                    _normalize_lookup_key(output_name),
                    valid_from,
                    valid_to,
                    priority,
                    note,
                    exonyms_path.name,
                    line_no,
                )
            )

        if rows:
            conn.executemany(
                """
                INSERT INTO pack_exonym (
                    pack_id,
                    input_normalized,
                    output_name,
                    output_name_normalized,
                    valid_from,
                    valid_to,
                    priority,
                    note,
                    source_file,
                    source_line
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def _normalize_lookup_key(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip()).casefold()


def _clean_optional_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _clean_optional_date(
    value: str,
    *,
    field: str,
    line_no: int,
    file_path: Path,
) -> str | None:
    if not value:
        return None

    if not PACK_DATE_RE.fullmatch(value):
        raise ValueError(
            f"{file_path}:{line_no} invalid {field} {value!r}; "
            "expected signed YYYY-MM-DD or YYYY-MM-DD"
        )

    return value


def _parse_required_int(
    value: str,
    *,
    field: str,
    line_no: int,
    file_path: Path,
) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{file_path}:{line_no} invalid integer for {field}: {value!r}"
        ) from exc


def _validate_date_range(
    *,
    valid_from: str | None,
    valid_to: str | None,
    file_path: Path,
    line_no: int,
) -> None:
    if valid_from is None or valid_to is None:
        return

    from_key = _date_key(valid_from)
    to_key = _date_key(valid_to)

    if from_key > to_key:
        raise ValueError(
            f"{file_path}:{line_no} invalid range: valid_from {valid_from} > valid_to {valid_to}"
        )


def _date_key(value: str) -> tuple[int, int, int]:
    # Handles both "1066-09-15" and "-0660-01-01"
    sign = -1 if value.startswith("-") else 1
    body = value[1:] if sign == -1 else value
    year_str, month_str, day_str = body.split("-")
    year = sign * int(year_str)
    month = int(month_str)
    day = int(day_str)
    return (year, month, day)
