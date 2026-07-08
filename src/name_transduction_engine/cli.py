import argparse
import sys

from name_transduction_engine.datasets.dataset_provider import (
    ensure_datasets,
    download_wikidata_raw,
    build_wikidata_compact_dataset,
)
from name_transduction_engine.enrichment.enrichment_provider import (
    ensure_builtin_pack_enrichment,
)
from name_transduction_engine.datasets.maintenance import (
    collect_data_status,
    format_data_status,
    clean_data,
    format_clean_report,
)

# --------------------------------------------------------------------------- #
# Command handlers
#
# Exit codes:
#   0  -> success or no result
#   1  -> error
#   2  -> reserved by argparse for usage errors
# --------------------------------------------------------------------------- #


def cmd_init(args: argparse.Namespace) -> int:
    ensure_datasets(args.force)
    ensure_builtin_pack_enrichment()
    return 0


def cmd_data_fetch(args: argparse.Namespace) -> int:
    if args.source == "wikidata-raw":
        download_wikidata_raw(args.force)
    return 0


def cmd_data_build(args: argparse.Namespace) -> int:
    if args.target == "wikidata-compact":
        build_wikidata_compact_dataset()
    return 0


def cmd_data_status(args: argparse.Namespace) -> int:
    print(format_data_status(collect_data_status()))
    return 0


def cmd_data_clean(args: argparse.Namespace) -> int:
    report = clean_data(include_raw=args.raw, preview=args.preview)
    print(format_clean_report(report))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    if not args.name.strip():
        print("error: name is empty", file=sys.stderr)
        return 1

    # TODO: result = transduce(args.name, to=args.to, source=args.source,
    #                          mode=args.mode, topk=args.topk)
    print(
        f"[transduce] name={args.name!r} from={args.source} to={args.to} "
        f"mode={args.mode} topk={args.topk} json={args.json}"
    )
    # Example of the no-result-is-still-exit-0 contract:
    #   if not result: return 0   # print nothing (or a marker to stderr)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    # TODO: print the pipeline trace: normalize -> lookup hit/miss -> ...
    print(f"[explain] name={args.name!r} to={args.to}")
    return 0


# Parser construction
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nte",
        description="Name Transduction Engine — convert and generate "
        "culturally plausible names.",
    )
    # Global options apply to every subcommand.
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (repeatable: -v, -vv)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress non-error output",
    )

    # Ensure that bare 'nte' with no verb errors out
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # Initialization
    p_init = sub.add_parser(
        "init",
        help="fetch lookup data and build names.sqlite",
    )
    p_init.add_argument(
        "--force", action="store_true", help="rebuild even if a valid DB already exists"
    )
    p_init.set_defaults(func=cmd_init)

    # data <subcommand>
    p_data = sub.add_parser("data", help="manage raw and built datasets")
    data_sub = p_data.add_subparsers(
        dest="data_command",
        required=True,
        metavar="<subcommand>",
    )

    p_fetch = data_sub.add_parser("fetch", help="download a raw dataset")
    # Ensure the parser rejects anything that isn't a known source
    p_fetch.add_argument("source", choices=["wikidata-raw"])
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=cmd_data_fetch)

    p_build = data_sub.add_parser(
        "build",
        help="build a compact dataset from a downloaded raw dump",
    )
    p_build.add_argument("target", choices=["wikidata-compact"])
    p_build.set_defaults(func=cmd_data_build)

    p_status = data_sub.add_parser("status", help="show dataset state")
    p_status.set_defaults(func=cmd_data_status)

    p_clean = data_sub.add_parser("clean", help="remove temp/partial files")
    p_clean.add_argument(
        "--raw", action="store_true", help="also delete downloaded raw dumps"
    )
    p_clean.add_argument(
        "--preview",
        action="store_true",
        help="show what would be removed without deleting",
    )
    p_clean.set_defaults(func=cmd_data_clean)

    # Transduction
    p_convert = sub.add_parser("convert", help="convert a name into a target form")
    p_convert.add_argument(
        "name",
        help="the name to convert (use -- before " "names that start with a dash)",
    )
    p_convert.add_argument(
        "--to", required=True, metavar="LANG", help="target language code, e.g. 'latin'"
    )
    p_convert.add_argument(
        "--from",
        dest="source",
        metavar="LANG",
        help="source language/script code (optional)",
    )
    p_convert.add_argument(
        "--mode", choices=["strict", "lookup", "adapt"], default="lookup"
    )
    p_convert.add_argument(
        "--topk", type=int, default=1, help="number of ranked candidates to return"
    )
    p_convert.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    p_convert.set_defaults(func=cmd_convert)

    # Explain
    p_ex = sub.add_parser("explain", help="show the transduction pipeline trace")
    p_ex.add_argument("name")
    p_ex.add_argument("--to", required=True, metavar="LANG")
    p_ex.set_defaults(func=cmd_explain)

    return parser
