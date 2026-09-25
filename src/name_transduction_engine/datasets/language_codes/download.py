from pathlib import Path
from typing import Final
from name_transduction_engine.datasets.shared import get_and_save_file, build_session
from name_transduction_engine.paths import RAW_DIR_LANGUAGE_CODES

IANA_REGISTRY_FILENAME: Final[str] = "language-subtag-registry.txt"
ISO_LANGUAGECODES_FILENAME: Final[str] = "iso-languagecodes.txt"

LANGUAGE_CODES_URLS: Final[dict[str, str]] = {
    ISO_LANGUAGECODES_FILENAME: "http://download.geonames.org/export/dump/iso-languagecodes.txt",
    IANA_REGISTRY_FILENAME: "https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry",
}


def download_language_codes_data(force: bool = False) -> None:
    print("Language codes download started.")
    RAW_DIR_LANGUAGE_CODES.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []

    with build_session() as session:
        for filename, url in LANGUAGE_CODES_URLS.items():
            output_path = RAW_DIR_LANGUAGE_CODES / filename

            if output_path.exists() and not force:
                print(f"Skipping {filename}: already exists.")
                downloaded_files.append(output_path)
                continue

            print(f"Fetching {filename}...")
            downloaded_path = get_and_save_file(
                session=session,
                url=url,
                output_path=output_path,
            )
            downloaded_files.append(downloaded_path)

    print("Language codes download finished.")
