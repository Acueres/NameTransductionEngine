import hashlib
import sqlite3
import yaml

from pathlib import Path
from typing import Any

from name_transduction_engine.enrichment.packs.resolver import (
    BuiltinPackPaths,
    discover_builtin_packs,
)
from name_transduction_engine.enrichment.packs.schema import create_pack_schema
from name_transduction_engine.enrichment.packs.compiler import rebuild_builtin_pack
from name_transduction_engine.paths import DB_PATH, BUILTIN_PACKS_DIR

__all__ = [
    "ensure_builtin_pack_enrichment",
]

REQUIRED_MANIFEST_KEYS = {"id", "display_name", "version", "bcp47", "kind"}


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a mapping: {manifest_path}")

    missing = REQUIRED_MANIFEST_KEYS - set(manifest.keys())
    if missing:
        raise ValueError(
            f"Manifest missing required keys {sorted(missing)}: {manifest_path}"
        )

    return manifest


def _compute_pack_hash(paths: BuiltinPackPaths) -> str:
    """
    SHA-256 over all files in the pack directory, sorted by relative path
    """
    hasher = hashlib.sha256()

    for file_path in sorted(p for p in paths.root.rglob("*") if p.is_file()):
        rel = file_path.relative_to(paths.root).as_posix().encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\0")

    return hasher.hexdigest()


def _pack_is_current(
    conn: sqlite3.Connection,
    pack_id: str,
    content_hash: str,
) -> bool:
    row = conn.execute(
        """
        SELECT content_hash
        FROM pack_install
        WHERE pack_id = ?
        """,
        (pack_id,),
    ).fetchone()

    return bool(row and row["content_hash"] == content_hash)


def ensure_builtin_pack_enrichment() -> None:
    """
    Ensure pack-scoped enrichment tables exist and are populated from all
    built-in packs found under builtin_packs_dir.

    Safe to call repeatedly: packs whose content hash has not changed since
    the last build are skipped.

    The DB at db_path must already exist and contain the GeoNames tables
    (geoname, alternate_name), as entity_names.tsv validation queries them.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB does not exist yet: {DB_PATH}")

    pack_paths = discover_builtin_packs(BUILTIN_PACKS_DIR)
    if not pack_paths:
        raise RuntimeError(f"No built-in packs found in {BUILTIN_PACKS_DIR}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        _configure_connection(conn)
        create_pack_schema(conn)

        for paths in pack_paths:
            manifest = _load_manifest(paths.manifest)
            pack_id = str(manifest["id"]).strip()
            content_hash = _compute_pack_hash(paths)

            if _pack_is_current(conn, pack_id=pack_id, content_hash=content_hash):
                print(f"Pack {pack_id!r}: up to date, skipping.")
                continue

            print(f"Pack {pack_id!r}: building enrichment.")
            rebuild_builtin_pack(
                conn=conn,
                paths=paths,
                manifest=manifest,
                content_hash=content_hash,
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()
