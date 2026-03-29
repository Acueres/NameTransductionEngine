import requests

from pathlib import Path
from typing import Final
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIRECTORY: Final[Path] = Path("./data/raw/geonames")

GEONAMES_URLS: Final[dict[str, str]] = {
    "allCountries.zip": "https://download.geonames.org/export/dump/allCountries.zip",
    "alternateNamesV2.zip": "https://download.geonames.org/export/dump/alternateNamesV2.zip",
    "iso-languagecodes.txt": "https://download.geonames.org/export/dump/iso-languagecodes.txt",
    "admin1CodesASCII.txt": "https://download.geonames.org/export/dump/admin1CodesASCII.txt",
    "admin2Codes.txt": "https://download.geonames.org/export/dump/admin2Codes.txt",
}


def _build_session() -> requests.Session:
    session = requests.Session()

    retry_config = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )

    adapter = HTTPAdapter(max_retries=retry_config)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def _download_geonames_data(force: bool = False) -> list[Path]:
    print("GeoNames download started.")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []

    with _build_session() as session:
        for filename, url in GEONAMES_URLS.items():
            output_path = OUTPUT_DIRECTORY / filename

            if output_path.exists() and not force:
                print(f"Skipping {filename}: already exists.")
                downloaded_files.append(output_path)
                continue

            print(f"Fetching {filename}...")
            downloaded_path = _get_and_save_file(
                session=session,
                url=url,
                output_path=output_path,
            )
            downloaded_files.append(downloaded_path)

    print("GeoNames download finished.")
    return downloaded_files


def _get_and_save_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    chunk_size: int = 1024 * 1024,
) -> Path:
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        with session.get(url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            downloaded_size = 0

            with temp_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    output_file.write(chunk)
                    downloaded_size += len(chunk)

            if total_size and downloaded_size != total_size:
                raise IOError(
                    f"Incomplete download for {output_path.name}: "
                    f"expected {total_size} bytes, got {downloaded_size} bytes."
                )

        temp_path.replace(output_path)
        print(f"Saved {output_path.name} ({downloaded_size} bytes).")
        return output_path

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def main():
    _download_geonames_data()


if __name__ == "__main__":
    main()
