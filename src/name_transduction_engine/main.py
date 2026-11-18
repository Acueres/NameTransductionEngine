import sys

from name_transduction_engine.cli import build_parser
from name_transduction_engine.transduction.service import DataUnavailable

# Exit code of commands refused while `nte update` rebuilds the data
EXIT_UNAVAILABLE = 3


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DataUnavailable as exc:
        # Raised by the service layer while the data is being rebuilt
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE


if __name__ == "__main__":
    sys.exit(main(["lookup", "Мюнхен", "--to", "ja"]))
