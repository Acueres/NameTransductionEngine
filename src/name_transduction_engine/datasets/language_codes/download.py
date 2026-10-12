from typing import Final

from name_transduction_engine.datasets.sources import download_sources
from name_transduction_engine.paths import RAW_DIR_LANGUAGE_CODES

IANA_REGISTRY_FILENAME: Final[str] = "language-subtag-registry.txt"
ISO_LANGUAGECODES_FILENAME: Final[str] = "iso-languagecodes.txt"

LANGUAGE_CODES_URLS: Final[dict[str, str]] = {
    ISO_LANGUAGECODES_FILENAME: "https://download.geonames.org/export/dump/iso-languagecodes.txt",
    IANA_REGISTRY_FILENAME: "https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry",
}


def download_language_codes_data(force: bool = False) -> None:
    """Download the language-code lists that are missing (all with `force`)"""
    download_sources(
        RAW_DIR_LANGUAGE_CODES, LANGUAGE_CODES_URLS, force=force, label="Language codes"
    )
