from name_transduction_engine.paths import DB_PATH_NAMES
from .geonames.data_provision import ensure_geonames_sqlite
from .wikidata.data_provision import download_wikidata_raw, ensure_wikidata_sqlite


def ensure_datasets(force_download_geonames=False, force_download_wikidata=False):
    ensure_geonames_sqlite(DB_PATH_NAMES, force_download_geonames)
    ensure_wikidata_sqlite(DB_PATH_NAMES, force=force_download_wikidata)


def ensure_wikidata_raw(force_download_wikidata=False):
    download_wikidata_raw(force_download_wikidata)
