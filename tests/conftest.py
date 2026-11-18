"""Every test runs against a small names.sqlite, built once per session by the
real loaders from the extracts in tests/fixtures/:

- language_codes/: the IANA registry and GeoNames language list, complete
- geonames/: eight places and their alternate names, in the GeoNames format
- wikidata/: real records from the 20261026 dataset, trimmed to 19 languages
  (Moscow, Tokyo, Istanbul and Beijing are classified historical_admin there)

NTE finds data/ from the current directory (`paths.PROJECT_ROOT`), so the
session moves into a temporary directory before any test imports NTE. The
Persian and Arabic models are copied in when the repository has them built.
"""

import gzip
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).parent.parent
WORK = Path(tempfile.mkdtemp(prefix="nte-tests-"))


def pytest_sessionstart(session):
    # After pytest has resolved its own paths, before test modules are
    # imported (they import NTE, which reads the current directory once)
    os.chdir(WORK)
    # Before collection too: romanizer_test.py skips cases by model presence
    models = REPO / "data" / "models"
    if models.is_dir():
        shutil.copytree(models, WORK / "data" / "models")


def pytest_sessionfinish(session):
    os.chdir(session.config.invocation_params.dir)
    shutil.rmtree(WORK, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def nte_data() -> Path:
    """The session's data/ directory, with names.sqlite built"""
    from name_transduction_engine import paths

    assert paths.PROJECT_ROOT == WORK, "NTE was imported before the session started"
    _write_raw_files(paths)
    _build_names_db(paths)
    return paths.DATA_DIR


def _build_names_db(paths) -> None:
    """names.sqlite, the way `nte init` builds it, minus the downloads"""
    from name_transduction_engine.datasets.shared import (
        configure_connection,
        ensure_build_metadata_table,
    )
    from name_transduction_engine.datasets.language_codes import (
        load as language_codes,
        schema as language_codes_schema,
    )
    from name_transduction_engine.datasets.geonames import (
        load as geonames,
        schema as geonames_schema,
    )
    from name_transduction_engine.datasets.wikidata import (
        load as wikidata,
        schema as wikidata_schema,
    )
    from name_transduction_engine.datasets.wikidata.compact import (
        open_compact_dataset,
    )

    conn = sqlite3.connect(paths.DB_PATH)
    try:
        configure_connection(conn)
        ensure_build_metadata_table(conn)

        language_codes_schema.create_schema(conn)
        registry = language_codes.load_all_data(conn)
        language_codes.write_build_metadata(conn)

        geonames_schema.create_schema(conn)
        geonames.load_all_data(conn, registry)
        geonames_schema.build_indexes(conn)
        geonames.write_build_metadata(conn, registry)

        dataset = open_compact_dataset(paths.WIKIDATA_DOWNLOAD_DIR)
        wikidata_schema.create_schema(conn)
        wikidata.load_compact_dataset(conn, dataset, registry)
        wikidata_schema.build_indexes(conn)
        wikidata.write_build_metadata(conn, registry, dataset)
        conn.commit()
    finally:
        conn.close()


def _write_raw_files(paths) -> None:
    """The extracts, laid out as the loaders expect them under data/raw/"""
    paths.RAW_DIR_LANGUAGE_CODES.mkdir(parents=True)
    for gz in (FIXTURES / "language_codes").glob("*.gz"):
        (paths.RAW_DIR_LANGUAGE_CODES / gz.stem).write_bytes(
            gzip.decompress(gz.read_bytes())
        )

    paths.RAW_DIR_GEONAMES.mkdir(parents=True)
    for name in ("allCountries", "alternateNamesV2"):
        with zipfile.ZipFile(paths.RAW_DIR_GEONAMES / f"{name}.zip", "w") as zf:
            zf.write(FIXTURES / "geonames" / f"{name}.txt", f"{name}.txt")

    paths.WIKIDATA_DOWNLOAD_DIR.mkdir(parents=True)
    groups = {}
    for group in ("place", "historical_place"):
        data = (FIXTURES / "wikidata" / f"{group}.jsonl").read_bytes()
        name = f"wikidata-fixture-{group}-1.jsonl.gz"
        (paths.WIKIDATA_DOWNLOAD_DIR / name).write_bytes(gzip.compress(data))
        groups[group] = {"records": data.count(b"\n"), "files": [{"name": name}]}
    manifest = {
        "format": "nte-wikidata-compact",
        "format_version": 1,
        "dataset_id": "fixture",
        "snapshot": "20261026",
        "groups": groups,
    }
    (paths.WIKIDATA_DOWNLOAD_DIR / "manifest.json").write_text(json.dumps(manifest))
