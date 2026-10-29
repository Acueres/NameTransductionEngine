import sys

from name_transduction_engine.cli import build_parser
from name_transduction_engine.datasets.availability import (
    DataUnavailable,
    check_data_available,
)

# Exit code of commands refused while `nte update` rebuilds the data
EXIT_UNAVAILABLE = 3


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "needs_data", False):
        try:
            check_data_available()
        except DataUnavailable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_UNAVAILABLE
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(["romanize", "--lang", "ru", "München"]))
