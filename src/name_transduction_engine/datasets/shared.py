import requests
import time

from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session(*, use_env_proxy: bool = False) -> requests.Session:
    session = requests.Session()
    session.trust_env = use_env_proxy
    session.headers.update(
        {
            "User-Agent": (
                "NameTransductionEngine/0.1 " "(dataset bootstrap)"
            ),
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }
    )

    retry_config = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry_config, pool_connections=1, pool_maxsize=1)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def get_and_save_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    chunk_size: int = 256 * 1024,
) -> Path:
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        started_at = time.monotonic()

        with session.get(url, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            downloaded_size = 0
            last_report = started_at

            with temp_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue

                    output_file.write(chunk)
                    downloaded_size += len(chunk)

                    now = time.monotonic()
                    if now - last_report >= 2.0:
                        elapsed = max(now - started_at, 0.001)
                        mib = downloaded_size / (1024 * 1024)
                        rate = mib / elapsed

                        if total_size:
                            pct = downloaded_size / total_size * 100
                            total_mib = total_size / (1024 * 1024)
                            print(
                                f"\r{output_path.name}: "
                                f"{mib:.1f}/{total_mib:.1f} MiB "
                                f"({pct:.1f}%), {rate:.2f} MiB/s",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r{output_path.name}: {mib:.1f} MiB, {rate:.2f} MiB/s",
                                end="",
                                flush=True,
                            )

                        last_report = now

            print()

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
