import argparse
import shutil
import subprocess
from pathlib import Path

from name_transduction_engine.paths import PROJECT_ROOT

MCN_REPO_URL = "https://github.com/hmlendea/more-cultural-names"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "data" / "repos" / "more-cultural-names"


def clone_or_update_mcn(
    target_dir: Path = DEFAULT_TARGET_DIR, *, update: bool = False
) -> Path:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is not installed or is not available on PATH.")

    target_dir = target_dir.resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if not target_dir.exists():
        print(f"Cloning MCN into {target_dir}")
        subprocess.run(
            [git, "clone", "--depth", "1", MCN_REPO_URL, str(target_dir)],
            check=True,
        )
        return target_dir

    git_dir = target_dir / ".git"
    if not git_dir.exists():
        raise RuntimeError(
            f"Target path already exists but is not a git repository: {target_dir}"
        )

    if update:
        print(f"Updating existing MCN repo at {target_dir}")
        subprocess.run(
            [git, "-C", str(target_dir), "pull", "--ff-only"],
            check=True,
        )
    else:
        print(f"MCN repo already exists at {target_dir}; use --update to pull latest.")

    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone or update the more-cultural-names repository for benchmarking."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Target directory. Default: {DEFAULT_TARGET_DIR}",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="If the repo already exists, run git pull --ff-only.",
    )

    args = parser.parse_args()
    clone_or_update_mcn(args.target_dir, update=args.update)


if __name__ == "__main__":
    main()
