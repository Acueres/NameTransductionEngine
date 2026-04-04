from .paths import DB_PATH_GEONAMES
from .geonames.data_provision import ensure_geonames_sqlite


def ensure_datasets(force_download_geonames=False):
    ensure_geonames_sqlite(DB_PATH_GEONAMES, force_download_geonames)
