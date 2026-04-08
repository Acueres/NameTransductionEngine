from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuiltinPackPaths:
    root: Path
    manifest: Path
    place_overrides: Path
    exonyms: Path


def discover_builtin_packs(builtin_packs_dir: Path) -> list[BuiltinPackPaths]:
    if not builtin_packs_dir.exists():
        raise FileNotFoundError(
            f"Built-in packs directory does not exist: {builtin_packs_dir}"
        )

    packs: list[BuiltinPackPaths] = []

    for pack_dir in sorted(p for p in builtin_packs_dir.iterdir() if p.is_dir()):
        manifest = pack_dir / "pack.yaml"
        if not manifest.exists():
            continue

        packs.append(
            BuiltinPackPaths(
                root=pack_dir,
                manifest=manifest,
                place_overrides=pack_dir / "place_overrides.tsv",
                exonyms=pack_dir / "exonyms.tsv",
            )
        )

    return packs
