from pathlib import Path
from typing import Final
from name_transduction_engine.datasets.shared import get_and_save_file, build_session
from name_transduction_engine.paths import RAW_DIR_GEONAMES

GEONAMES_URLS: Final[dict[str, str]] = {
    "allCountries.zip": "http://download.geonames.org/export/dump/allCountries.zip",
    "alternateNamesV2.zip": "http://download.geonames.org/export/dump/alternateNamesV2.zip",
    "iso-languagecodes.txt": "http://download.geonames.org/export/dump/iso-languagecodes.txt",
    "admin1CodesASCII.txt": "http://download.geonames.org/export/dump/admin1CodesASCII.txt",
    "admin2Codes.txt": "http://download.geonames.org/export/dump/admin2Codes.txt",
}


def download_geonames_data(force: bool = False) -> None:
    print("GeoNames download started.")
    RAW_DIR_GEONAMES.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []

    with build_session() as session:
        for filename, url in GEONAMES_URLS.items():
            output_path = RAW_DIR_GEONAMES / filename

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

    print("GeoNames download finished.")
