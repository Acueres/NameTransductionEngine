"""`nte data fetch wikidata-raw`: download a dated Wikidata JSON dump to
data/raw/wikidata, resumably.

Not needed for the build, which streams the dump; useful to build from a
local file (`nte data build wikidata-compact --source <file>`).

The file is always a dated dump, never `latest-all.json.bz2`, which moves
every week: resuming a download across a new dump used to splice two
different files together. A partial download is resumed only if the server
still has the same file (size and ETag, checked again by every range request
through If-Range).
"""

import json
import time

from pathlib import Path

from name_transduction_engine.paths import RAW_DIR_WIKIDATA
from .dump_source import (
    DEFAULT_MIRROR,
    DumpInfo,
    human_bytes,
    iter_chunks,
    make_session,
    probe_dump,
    resolve_latest_dump,
)


def download_wikidata_raw(
    force: bool = False, mirror: str = DEFAULT_MIRROR, url: str | None = None
) -> Path:
    RAW_DIR_WIKIDATA.mkdir(parents=True, exist_ok=True)
    session = make_session()

    if url:
        info = probe_dump(session, url)
        if info is None:
            raise RuntimeError(f"dump not found: {url}")
    else:
        info = resolve_latest_dump(session, mirror)

    path = RAW_DIR_WIKIDATA / f"wikidata-{info.snapshot}-all.json.bz2"
    part_path = path.with_name(path.name + ".part")
    meta_path = path.with_name(path.name + ".meta.json")

    if path.exists() and not force and path.stat().st_size == info.size:
        print(f"Wikidata dump already downloaded: {path}")
        return path

    old_meta = _load_meta(meta_path)  # read before it is overwritten
    have = part_path.stat().st_size if part_path.exists() else 0
    resume = (
        not force
        and have > 0
        and old_meta is not None
        and DumpInfo.from_json(old_meta).same_file(info)
        and have < info.size
    )
    if resume:
        print(f"Resuming {path.name} from {human_bytes(have)}.")
    else:
        if part_path.exists():
            part_path.unlink()
        have = 0
        meta_path.write_text(json.dumps(info.to_json(), indent=2), encoding="utf-8")

    print(f"Downloading {info.url} ({human_bytes(info.size)})")
    started = last = time.monotonic()
    done_this_run = 0
    with part_path.open("ab" if resume else "wb") as out:
        for _, data in iter_chunks(info, have, session=session):
            out.write(data)
            done_this_run += len(data)
            now = time.monotonic()
            if now - last >= 2.0:
                current = have + done_this_run
                rate = done_this_run / max(now - started, 1e-3)
                eta = (info.size - current) / rate / 3600 if rate else float("inf")
                print(
                    f"\r{current / info.size:6.2%}  {human_bytes(current)} / "
                    f"{human_bytes(info.size)}  at {human_bytes(rate)}/s  "
                    f"ETA {eta:.1f}h".ljust(100),
                    end="",
                    flush=True,
                )
                last = now
    print()

    final = part_path.stat().st_size
    if final != info.size:
        raise IOError(
            f"Incomplete download of {path.name}: expected {info.size} bytes, "
            f"got {final}. Run the same command to resume."
        )
    part_path.replace(path)
    print(f"Saved {path} ({human_bytes(final)}).")
    return path


def _load_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        return None
