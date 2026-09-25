import argparse
import sqlite3
import sys

from name_transduction_engine.datasets.dataset_provider import (
    ensure_datasets,
    download_wikidata_raw,
    build_wikidata_compact_dataset,
)
from name_transduction_engine.language_packs.language_pack_provider import (
    ensure_builtin_packs,
)
from name_transduction_engine.datasets.maintenance import (
    collect_data_status,
    format_data_status,
    clean_data,
    format_clean_report,
)
from name_transduction_engine import paths
from name_transduction_engine.transduction.lookup.lookup_engine import lookup_name
from name_transduction_engine.normalization.language_code_normalization import (
    LanguageRegistry,
    RegistryError,
    UnknownLanguageError,
    resolve_user_language,
)
from name_transduction_engine.datasets.language_codes.data_provision import (
    read_registry,
)
from name_transduction_engine.datasets.language_codes.info import (
    Coverage,
    raw_tags_for,
    read_coverage,
    rows_for_tag,
    search_languages,
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
    ensure_builtin_packs()
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


def cmd_lookup(args: argparse.Namespace) -> int:
    if not args.name.strip():
        print("error: name is empty", file=sys.stderr)
        return 1

    try:
        result = lookup_name(args.name, args.to)
    except UnknownLanguageError as exc:
        # The message already carries did-you-mean suggestions
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (RegistryError, sqlite3.DatabaseError) as exc:
        print(
            f"error: lookup data unavailable ({exc}); run `nte init`", file=sys.stderr
        )
        return 1

    entities = result.entities
    print(
        f"[lookup] name={args.name!r} "
        f"to={result.language.tag} "
        f"entities={len(entities)}"
    )

    for note in result.language.notes:
        print(f"[note] {note}")

    if not entities:
        print("[result] no match")
        return 0

    for entity in entities:
        if not entity.names and not args.all:
            continue

        location = ""
        if entity.latitude is not None and entity.longitude is not None:
            location = f" coords=({entity.latitude:.5f}, {entity.longitude:.5f})"

        print(
            f"[entity] source={entity.source} "
            f"id={entity.entity_id} "
            f"type={entity.entity_type}"
            f"{location}"
        )

        if not entity.names:
            print(f"  [name] no name for language={result.language.tag}")
            continue

        for name in entity.names:
            print(f"  [name] {_format_name(name)} lang={name.language_code}")

    return 0


def _format_name(name) -> str:
    """Native form first; the romanization only when it adds something (not for
    names already in Latin), flagged when its confidence is low"""
    romanization = name.romanization
    if romanization is None or romanization.transform == "identity":
        return name.name

    text = f"{name.name} ({romanization.text})"
    if romanization.confidence == "low":
        text += " [low-confidence romanization]"
    return text


# Language information


_MATCHED_VIA = {
    "iso639_1": "ISO 639-1 code",
    "iso639_2": "ISO 639-2 code",
    "iso639_3": "ISO 639-3 code",
    "retired": "retired code",
    "name": "language name",
}


def cmd_lang(args: argparse.Namespace) -> int:
    try:
        conn = sqlite3.connect(f"file:{paths.DB_PATH}?mode=ro", uri=True)
    except sqlite3.DatabaseError as exc:
        print(
            f"error: language data unavailable ({exc}); run `nte init`", file=sys.stderr
        )
        return 1
    try:
        registry = read_registry(conn)
        coverage = read_coverage(conn)
        if args.list:
            return _lang_list(registry, coverage, args.limit)
        if args.search is not None:
            return _lang_search(registry, coverage, args.search, args.limit)
        if args.query is None:
            print(
                "error: give a language to resolve, or use --search / --list",
                file=sys.stderr,
            )
            return 2
        return _lang_show(conn, registry, coverage, args.query)
    except (RegistryError, sqlite3.DatabaseError) as exc:
        print(
            f"error: language data unavailable ({exc}); run `nte init`", file=sys.stderr
        )
        return 1
    finally:
        conn.close()


def _lang_show(
    conn: sqlite3.Connection,
    registry: LanguageRegistry,
    coverage: Coverage,
    query: str,
) -> int:
    try:
        resolved = resolve_user_language(query, registry)
    except UnknownLanguageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tag = resolved.tag
    rec = registry.record(tag.lang)
    print(f"[language] {query!r} -> {tag}  {rec.name}")
    print(
        f"  matched   : {_MATCHED_VIA.get(resolved.matched_via, resolved.matched_via)}"
    )
    for note in resolved.notes:
        print(f"  note      : {note}")

    codes = [
        f"639-1 {rec.iso639_1}" if rec.iso639_1 else None,
        f"639-2/B {rec.iso639_2b}" if rec.iso639_2b else None,
        f"639-2/T {rec.iso639_2t}" if rec.iso639_2t else None,
        f"639-3 {rec.iso639_3}",
    ]
    print(f"  codes     : {' · '.join(c for c in codes if c)}")
    if rec.aliases:
        # "; " because a single alias can contain commas ("Greek, Modern (1453-)")
        print(f"  also named: {'; '.join(rec.aliases)}")
    print(f"  scope     : {rec.scope}")
    if tag.script or tag.region or tag.variant:
        parts = [
            f"script {tag.script}" if tag.script else None,
            f"region {tag.region}" if tag.region else None,
            f"variant {tag.variant}" if tag.variant else None,
        ]
        print(
            f"  narrowed  : {', '.join(p for p in parts if p)} "
            "(only names carrying these subtags are returned)"
        )

    # Coverage: what `nte lookup --to <query>` can actually return
    raw_tags = raw_tags_for(conn, tag.lang)
    if not coverage.sources:
        print("  names     : unknown (no dataset built; run `nte init`)")
    else:
        counts = rows_for_tag(raw_tags, tag)
        print(f"  names     : {_format_counts(counts, coverage.sources)}")
        if raw_tags:
            counted = {id(r) for r in raw_tags if rows_for_tag([r], tag)}
            shown = ", ".join(
                f"{r.source}:{r.raw_tag!r} ({r.rows:,})"
                + (f" = {r.tag}" if str(r.tag) != tag.lang else "")
                + ("" if id(r) in counted else " [excluded]")
                for r in raw_tags[:8]
            )
            more = f" … +{len(raw_tags) - 8} more" if len(raw_tags) > 8 else ""
            print(f"  from tags : {shown}{more}")

    # Macrolanguage relations, with coverage so the user can see where names are
    if rec.macrolanguage:
        macro = rec.macrolanguage
        print(
            f"  part of   : {registry.describe(macro)} [{coverage.total(macro):,} names]"
        )
    members = registry.members_of(tag.lang)
    if members:
        ranked = sorted(members, key=lambda m: (-coverage.total(m), m))
        with_data = [m for m in ranked if coverage.total(m)]
        listed = ", ".join(f"{m} [{coverage.total(m):,}]" for m in with_data[:10])
        print(
            f"  members   : {len(members)} languages, {len(with_data)} with names"
            + (f": {listed}" if listed else "")
            + (" …" if len(with_data) > 10 else "")
        )

    # The hint: nothing here, but a related language has names
    if coverage.sources and not sum(rows_for_tag(raw_tags, tag).values()):
        related = [x for x in (rec.macrolanguage, *members) if x and coverage.total(x)]
        if related:
            best = max(related, key=lambda x: (coverage.total(x), x))
            print(
                f"[hint] no names are tagged {tag}; try `--to {best}` "
                f"({registry.record(best).name}, {coverage.total(best):,} names)"
            )
        else:
            print(f"[hint] no names are tagged {tag} in either dataset")
    return 0


def _lang_search(
    registry: LanguageRegistry, coverage: Coverage, text: str, limit: int
) -> int:
    hits = search_languages(registry, text)
    if not hits:
        print(f"[search] no language code or name contains {text!r}")
        return 0
    # Within each match quality, languages with more names first
    hits.sort(key=lambda h: (h.rank, -coverage.total(h.code), h.code))
    shown = hits if limit <= 0 else hits[:limit]
    print(
        f"[search] {text!r}: {len(hits)} match(es)"
        + (f", showing {len(shown)}" if len(shown) < len(hits) else "")
    )
    width = max(len(h.code) for h in shown)
    for h in shown:
        rec = registry.record(h.code)
        via = "" if h.matched in (rec.name, h.code) else f"  (as {h.matched!r})"
        print(
            f"  {h.code:<{width}}  {rec.name}{via}  [{coverage.total(h.code):,} names]"
        )
    return 0


def _lang_list(registry: LanguageRegistry, coverage: Coverage, limit: int) -> int:
    if not coverage.sources:
        print("error: no dataset built; run `nte init`", file=sys.stderr)
        return 1
    langs = sorted(coverage.by_lang, key=lambda lang: (-coverage.total(lang), lang))
    shown = langs if limit <= 0 else langs[:limit]
    print(
        f"[languages] {len(langs)} languages have names"
        + (
            f", showing the top {len(shown)} (--limit 0 for all)"
            if len(shown) < len(langs)
            else ""
        )
    )
    width = max((len(x) for x in shown), default=4)
    for lang in shown:
        name = registry.record(lang).name if lang in registry else "?"
        print(
            f"  {lang:<{width}}  {_format_counts(coverage.per_source(lang), coverage.sources)}  {name}"
        )
    return 0


def _format_counts(counts: dict[str, int], sources: tuple[str, ...]) -> str:
    return " · ".join(f"{source} {counts.get(source, 0):,}" for source in sources)


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
    # Lookup
    p_lookup = sub.add_parser("lookup", help="look up a name from available sources")
    p_lookup.add_argument(
        "name",
        help="the name to convert (use -- before " "names that start with a dash)",
    )
    p_lookup.add_argument(
        "--to",
        required=True,
        metavar="LANG",
        help="target language: ISO 639-1/2/3 code or exact English name, "
        "optionally with subtags, e.g. 'os', 'oss', 'ossetian', 'zh-Hant'",
    )
    p_lookup.add_argument(
        "--all",
        action="store_true",
        help="also show matched entities that have no name in the target language",
    )
    p_lookup.set_defaults(func=cmd_lookup)

    # Language information
    p_lang = sub.add_parser(
        "lang",
        aliases=["language"],
        help="show what a language code or name resolves to, and how many names it has",
        description="Resolve a language the way `lookup --to` does, and show its "
        "codes, names, macrolanguage relations and name counts per dataset.",
    )
    p_lang.add_argument(
        "query",
        nargs="?",
        metavar="LANG",
        help="code or exact English name, e.g. 'os', 'oss', 'ossetian', 'zh-Hant'",
    )
    p_lang_mode = p_lang.add_mutually_exclusive_group()
    p_lang_mode.add_argument(
        "--search",
        metavar="TEXT",
        help="list languages whose code or name contains TEXT",
    )
    p_lang_mode.add_argument(
        "--list",
        action="store_true",
        help="list the languages that have names in the database, most names first",
    )
    p_lang.add_argument(
        "--limit",
        type=int,
        default=25,
        help="rows to show for --search/--list (0 = all; default 25)",
    )
    p_lang.set_defaults(func=cmd_lang)

    # Explain
    p_ex = sub.add_parser("explain", help="show the transduction pipeline trace")
    p_ex.add_argument("name")
    p_ex.add_argument("--to", required=True, metavar="LANG")
    p_ex.set_defaults(func=cmd_explain)

    return parser
