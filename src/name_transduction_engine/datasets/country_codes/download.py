from typing import Final

from name_transduction_engine.datasets.sources import download_sources
from name_transduction_engine.paths import RAW_DIR_COUNTRY_CODES

CLDR_VERSION: Final[str] = "49.0.0"
CLDR_TERRITORIES_FILENAME: Final[str] = f"cldr-{CLDR_VERSION}-territories.json"

COUNTRY_CODES_URLS: Final[dict[str, str]] = {
    CLDR_TERRITORIES_FILENAME: (
        "https://raw.githubusercontent.com/unicode-org/cldr-json/"
        f"{CLDR_VERSION}/cldr-json/cldr-localenames-full/main/en/territories.json"
    ),
}


def download_country_codes_data(force: bool = False) -> None:
    """Download the CLDR territory names when missing (again with `force`)"""
    download_sources(
        RAW_DIR_COUNTRY_CODES, COUNTRY_CODES_URLS, force=force, label="Country codes"
    )
