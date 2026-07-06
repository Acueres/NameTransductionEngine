from .geonames.data_provision import ensure_geonames_sqlite
from .wikidata.data_provision import (
    download_wikidata_raw,
    ensure_wikidata_sqlite,
    build_wikidata_compact_dataset,
)


def ensure_datasets(force=False):
    ensure_geonames_sqlite(force)
    ensure_wikidata_sqlite(force)
