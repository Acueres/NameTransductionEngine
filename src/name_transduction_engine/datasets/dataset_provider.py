from name_transduction_engine.paths import DB_PATH
from .geonames.data_provision import ensure_geonames_sqlite
from .wikidata.data_provision import (
    download_wikidata_raw,
    ensure_wikidata_sqlite,
    build_compact_dataset,
)


def ensure_datasets(
    force_geonames=False,
    force_wikidata=False,
):
    ensure_geonames_sqlite(DB_PATH, force=force_geonames)
    ensure_wikidata_sqlite(DB_PATH, force=force_wikidata)


def ensure_wikidata_raw(force_download_wikidata=False):
    download_wikidata_raw(force_download_wikidata)


def build_wikidata_compact():
    build_compact_dataset()
