import requests

from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session() -> requests.Session:
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


def get_and_save_file(
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
