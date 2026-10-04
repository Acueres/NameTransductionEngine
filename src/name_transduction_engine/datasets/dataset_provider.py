from .language_codes.data_provision import ensure_language_codes_sqlite, read_registry
from .geonames.data_provision import ensure_geonames_sqlite
from .wikidata.data_provision import (
    BuildOptions as WikidataBuildOptions,
    build_wikidata_compact,
    download_wikidata_raw,
    ensure_wikidata_sqlite,
    refresh_wikidata_classes,
)

__all__ = [
    "ensure_language_codes_sqlite",
    "read_registry",
    "ensure_geonames_sqlite",
    "WikidataBuildOptions",
    "build_wikidata_compact",
    "download_wikidata_raw",
    "ensure_wikidata_sqlite",
    "refresh_wikidata_classes",
]


def ensure_datasets(force=False):
    ensure_language_codes_sqlite(force)
    ensure_geonames_sqlite(force)
    ensure_wikidata_sqlite(force)
