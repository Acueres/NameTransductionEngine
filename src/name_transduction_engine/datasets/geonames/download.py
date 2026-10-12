from typing import Final

from name_transduction_engine.datasets.sources import download_sources
from name_transduction_engine.paths import RAW_DIR_GEONAMES

_BASE: Final[str] = "https://download.geonames.org/export/dump/"
GEONAMES_URLS: Final[dict[str, str]] = {
    name: _BASE + name
    for name in (
        "allCountries.zip",
        "alternateNamesV2.zip",
        "admin1CodesASCII.txt",
        "admin2Codes.txt",
    )
}
GEONAMES_FILES: Final[tuple[str, ...]] = tuple(GEONAMES_URLS)


def download_geonames_data(force: bool = False) -> None:
    """Download the GeoNames files that are missing (all with `force`).
    GeoNames exports a new dump every day; `nte update` downloads it again
    once the local copy is older than its maximum age"""
    download_sources(RAW_DIR_GEONAMES, GEONAMES_URLS, force=force, label="GeoNames")
