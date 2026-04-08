import hashlib
import sqlite3
import yaml

from pathlib import Path
from typing import Any
from name_transduction_engine.enrichment.packs.resolver import BuiltinPackPaths, discover_builtin_packs
from name_transduction_engine.enrichment.packs.schema import create_pack_schema
from name_transduction_engine.enrichment.packs.compiler import rebuild_builtin_pack


REQUIRED_MANIFEST_KEYS = {"id", "display_name", "version"}


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
    hasher = hashlib.sha256()

    for file_path in sorted(p for p in paths.root.rglob("*") if p.is_file()):
        rel = file_path.relative_to(paths.root).as_posix().encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
        hasher.update(b"\0")

    return hasher.hexdigest()


def _pack_build_is_current(
    conn: sqlite3.Connection,
    pack_id: str,
    content_hash: str,
) -> bool:
    row = conn.execute(
        """
        SELECT content_hash
        FROM pack_build
        WHERE pack_id = ?
        """,
        (pack_id,),
    ).fetchone()

    return bool(row and row["content_hash"] == content_hash)


def ensure_builtin_pack_enrichment(
    db_path: Path,
    builtin_packs_dir: Path,
) -> Path:
    """
    Ensure pack-scoped enrichment tables exist and are populated
    from all built-in packs.

    Safe to call repeatedly.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB does not exist yet: {db_path}")

    pack_paths = discover_builtin_packs(builtin_packs_dir)
    if not pack_paths:
        raise RuntimeError(f"No built-in packs found in {builtin_packs_dir}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        _configure_connection(conn)
        create_pack_schema(conn)

        for paths in pack_paths:
            manifest = _load_manifest(paths.manifest)
            pack_id = str(manifest["id"]).strip()
            content_hash = _compute_pack_hash(paths)

            if _pack_build_is_current(conn, pack_id=pack_id, content_hash=content_hash):
                print(f"Skipping pack {pack_id}: unchanged.")
                continue

            print(f"Building built-in pack enrichment: {pack_id}")
            rebuild_builtin_pack(
                conn=conn,
                paths=paths,
                manifest=manifest,
                content_hash=content_hash,
            )

        conn.commit()
    finally:
        conn.close()

    return db_path
