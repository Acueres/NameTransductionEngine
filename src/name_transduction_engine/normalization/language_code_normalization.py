import difflib
import hashlib
import json
import re

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

TAG_RULES_VERSION = "1"

Source = Literal["geonames", "wikidata"]
CodeKind = Literal["iso639_1", "iso639_2", "iso639_3", "retired", "name"]
LanguageScope = Literal["individual", "macrolanguage"]

# language   -> attested language; `tag` is set
# untagged   -> a name with no language tag; usable for resolve, never for retrieval
# multiple   -> Wikidata `mul`: one label shared by many languages
# non_name   -> not a name at all (postcodes, airport codes, URLs); do not store
# unmapped   -> a tag we could not map; keep the row, count it, report it
TagStatus = Literal["language", "untagged", "multiple", "non_name", "unmapped"]


class RegistryError(ValueError):
    """The registry data is inconsistent, or does not match this code"""


class UnknownLanguageError(ValueError):
    """User input did not resolve to exactly one registry language"""

    def __init__(self, text: str, reason: str, suggestions: tuple[str, ...] = ()):
        self.text = text
        self.reason = reason
        self.suggestions = suggestions
        message = f"{reason}: {text!r}"
        if suggestions:
            message += f"; did you mean: {', '.join(suggestions)}?"
        super().__init__(message)


@dataclass(frozen=True)
class LanguageRecord:
    iso639_3: str
    name: str
    iso639_1: str | None = None
    iso639_2b: str | None = None
    iso639_2t: str | None = None
    macrolanguage: str | None = None  # canonical code of the macrolanguage
    scope: LanguageScope = "individual"
    aliases: tuple[str, ...] = ()  # other names; resolvable, never displayed

    @property
    def code(self) -> str:
        return self.iso639_1 or self.iso639_3

    def cleaned(self) -> "LanguageRecord":
        def c(v: str | None) -> str | None:
            v = (v or "").strip().lower()
            return v or None

        name = " ".join(self.name.split())
        aliases = {" ".join(a.split()) for a in self.aliases} - {name, ""}

        return LanguageRecord(
            iso639_3=c(self.iso639_3) or "",
            name=name,
            iso639_1=c(self.iso639_1),
            iso639_2b=c(self.iso639_2b),
            iso639_2t=c(self.iso639_2t),
            macrolanguage=c(self.macrolanguage),
            scope=self.scope,
            aliases=tuple(sorted(aliases)),
        )


@dataclass(frozen=True)
class LanguageTag:
    lang: str
    script: str | None = None  # ISO 15924, title case: "Hant", "Latn"
    region: str | None = None  # ISO 3166-1 alpha-2 or UN M.49, upper case
    variant: str | None = None  # lower case, "-"-joined: "tarask", "1793", "x-bms"

    def __str__(self) -> str:
        return "-".join(
            p for p in (self.lang, self.script, self.region, self.variant) if p
        )


@dataclass(frozen=True)
class SourceTag:
    raw: str
    status: TagStatus
    tag: LanguageTag | None = None
    note: str | None = None


@dataclass(frozen=True)
class LanguageQuery:
    input: str
    tag: LanguageTag
    matched_via: CodeKind
    notes: tuple[str, ...] = ()


# GeoNames `isolanguage` pseudo-codes that are not names.
_GEONAMES_NON_NAME = frozenset(
    {
        "post",
        "iata",
        "icao",
        "faac",
        "tcid",
        "link",
        "wkdt",
        "unlc",
        "nuts",
        "lauc",
        "phon",
        "uicn",
        "geoid",
    }
)
# GeoNames values that are names without a language.
_GEONAMES_UNTAGGED = frozenset({"", "abbr"})
# GeoNames tags the generic parser cannot handle. (fr_1793 needs no entry: it
# parses generically as fr + variant 1793.)
_GEONAMES_TAG_EXCEPTIONS: dict[str, LanguageTag] = {
    "piny": LanguageTag("zh", script="Latn", variant="pinyin"),
}

# Wikimedia codes that are not ISO, or whose subtags the generic parser would
# misread. Keys are lower case. Based on MediaWiki's
# LanguageCode::getNonstandardLanguageCodeMapping
_WIKIMEDIA_TAG_EXCEPTIONS: dict[str, LanguageTag] = {
    # Not ISO, or clashes with ISO.
    "als": LanguageTag("gsw"),  # Alemannic. ISO `als` is Tosk Albanian
    "bat-smg": LanguageTag("sgs"),
    "be-x-old": LanguageTag("be", variant="tarask"),
    "bh": LanguageTag("bho"),
    "cbk-zam": LanguageTag("cbk"),
    "fiu-vro": LanguageTag("vro"),
    "map-bms": LanguageTag("jv", variant="x-bms"),
    "mo": LanguageTag("ro", script="Cyrl", region="MD"),
    "nrm": LanguageTag("nrf"),
    "roa-rup": LanguageTag("rup"),
    "roa-tara": LanguageTag("nap", variant="x-tara"),
    "simple": LanguageTag("en", variant="simple"),
    "zh-classical": LanguageTag("lzh"),
    "zh-min-nan": LanguageTag("nan"),
    "zh-yue": LanguageTag("yue"),
    # Parse generically but mean something else
    "sr-ec": LanguageTag("sr", script="Cyrl"),  # would parse as region EC
    "sr-el": LanguageTag("sr", script="Latn"),  # would parse as region EL
    "zh-cn": LanguageTag("zh", script="Hans", region="CN"),
    "zh-sg": LanguageTag("zh", script="Hans", region="SG"),
    "zh-my": LanguageTag("zh", script="Hans", region="MY"),
    "zh-tw": LanguageTag("zh", script="Hant", region="TW"),
    "zh-hk": LanguageTag("zh", script="Hant", region="HK"),
    "zh-mo": LanguageTag("zh", script="Hant", region="MO"),
}

# ISO 639-2 bibliographic (B) codes that differ from the terminology (T) code,
# mapped to the language's ISO 639-3 identifier. The list is closed: ISO froze
# it, and every T code equals the 639-3 identifier, which the registry already
# indexes. Used by the registry build; the GeoNames 639-2 column is not, because
# it lacks these codes and attaches some 639-2 codes to the wrong language
ISO639_2B_TO_3: dict[str, str] = {
    "alb": "sqi",  # Albanian
    "arm": "hye",  # Armenian
    "baq": "eus",  # Basque
    "bur": "mya",  # Burmese
    "chi": "zho",  # Chinese
    "cze": "ces",  # Czech
    "dut": "nld",  # Dutch
    "fre": "fra",  # French
    "geo": "kat",  # Georgian
    "ger": "deu",  # German
    "gre": "ell",  # Modern Greek
    "ice": "isl",  # Icelandic
    "mac": "mkd",  # Macedonian
    "mao": "mri",  # Maori
    "may": "msa",  # Malay
    "per": "fas",  # Persian
    "rum": "ron",  # Romanian
    "slo": "slk",  # Slovak
    "tib": "bod",  # Tibetan
    "wel": "cym",  # Welsh
}

# B codes withdrawn in 2008 (Serbian, Croatian). Still found in older data, so
# they resolve as retirements, with a note
WITHDRAWN_ISO639_2B: dict[str, str] = {
    "scc": "srp",
    "scr": "hrv",
}

# Special ISO codes, handled the same way for every source
_SPECIAL_STATUS: dict[str, TagStatus] = {
    "mul": "multiple",
    "und": "untagged",
    "zxx": "untagged",
}

_SUBTAG_SPLIT = re.compile(r"[-_]")
_TAG_SHAPED = re.compile(r"[A-Za-z0-9_-]+")
# A bare 2-3 letter input is a language code, never a name: `bih` (the Bihari
# collection code) must not resolve to the language *named* "Bih" (ibh)
_CODE_SHAPED = re.compile(r"[A-Za-z]{2,3}")
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


class LanguageRegistry:
    """The closed set of languages NTE knows about, plus lookup indexes.

    Construction checks the data and raises RegistryError on any conflict (one
    code naming two languages, a dangling macrolanguage). A conflict is never
    resolved silently, because every later lookup would inherit the result.

    `scripts` and `regions` are the valid subtags. None disables that check,
    which is only meant for tests: a production registry comes from
    `build_registry` and always has both
    """

    def __init__(
        self,
        records: Iterable[LanguageRecord],
        retirements: Mapping[str, str | None] | None = None,
        scripts: Iterable[str] | None = None,
        regions: Iterable[str] | None = None,
    ) -> None:
        self._records = _dedupe_records(records)
        self._codes = _index_codes(self._records.values())
        self.scripts = (
            None if scripts is None else frozenset(s.title() for s in scripts)
        )
        self.regions = (
            None if regions is None else frozenset(r.upper() for r in regions)
        )

        members: dict[str, list[str]] = {}
        for rec in self._records.values():
            if rec.macrolanguage is None:
                continue
            if rec.macrolanguage not in self._records:
                raise RegistryError(
                    f"{rec.code}: macrolanguage {rec.macrolanguage!r} is not in the registry"
                )
            members.setdefault(rec.macrolanguage, []).append(rec.code)
        self._members = {m: tuple(sorted(v)) for m, v in sorted(members.items())}

        # Retirements. If a code is still active, the active entry wins
        self._retired_raw: dict[str, str | None] = {}
        self._retired: dict[str, str | None] = {}
        merged = {
            k.strip().lower(): (v or "").strip().lower() or None
            for k, v in (retirements or {}).items()
        }
        for old, new in sorted(merged.items()):
            if not old or old in self._codes:
                continue
            self._retired_raw[old] = new
            target = self._codes.get(new) if new else None
            self._retired[old] = target[0] if target else None

        names: dict[str, set[str]] = {}
        for rec in self._records.values():
            for label in (rec.name, *rec.aliases):
                for key in _name_keys(label):
                    names.setdefault(key, set()).add(rec.code)
        self._names = {
            k: next(iter(v)) for k, v in sorted(names.items()) if len(v) == 1
        }
        self._ambiguous_names = {
            k: tuple(sorted(v)) for k, v in sorted(names.items()) if len(v) > 1
        }
        self._suggestion_pool = sorted(set(self._codes) | set(self._names))
        self.fingerprint = self._compute_fingerprint()

    # lookups

    def __contains__(self, code: object) -> bool:
        return isinstance(code, str) and code in self._records

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> tuple[LanguageRecord, ...]:
        return tuple(self._records.values())

    def record(self, code: str) -> LanguageRecord:
        return self._records[code]

    def lookup_code(self, code: str) -> tuple[str, CodeKind] | None:
        return self._codes.get(code.strip().lower())

    def retirement(self, code: str) -> tuple[str | None, str | None] | None:
        """(canonical replacement or None, raw change_to) if `code` is retired"""
        key = code.strip().lower()
        if key not in self._retired_raw:
            return None
        return self._retired[key], self._retired_raw[key]

    def retirements(self) -> dict[str, str | None]:
        return dict(self._retired_raw)

    def lookup_name(self, name: str) -> str | None:
        return self._names.get(_fold(name))

    def ambiguous_name(self, name: str) -> tuple[str, ...]:
        return self._ambiguous_names.get(_fold(name), ())

    def macrolanguage_of(self, code: str) -> str | None:
        rec = self._records.get(code)
        return rec.macrolanguage if rec else None

    def members_of(self, code: str) -> tuple[str, ...]:
        return self._members.get(code, ())

    def suggest(self, text: str, limit: int = 3) -> tuple[str, ...]:
        """Close registry codes for a failed input. Used only in error messages"""
        key = _fold(text)
        found: list[str] = []
        for hit in difflib.get_close_matches(
            key, self._suggestion_pool, n=limit * 2, cutoff=0.7
        ):
            code = self._codes[hit][0] if hit in self._codes else self._names[hit]
            if code not in found:
                found.append(code)
        return tuple(found[:limit])

    def describe(self, code: str) -> str:
        return f"{code} ({self._records[code].name})" if code in self._records else code

    def _compute_fingerprint(self) -> str:
        payload = {
            "rules": TAG_RULES_VERSION,
            "records": [
                [
                    r.code,
                    r.iso639_3,
                    r.iso639_1,
                    r.iso639_2b,
                    r.iso639_2t,
                    r.name,
                    r.macrolanguage,
                    r.scope,
                    list(r.aliases),
                ]
                for r in self._records.values()
            ],
            "retired": sorted(self._retired_raw.items()),
            "scripts": None if self.scripts is None else sorted(self.scripts),
            "regions": None if self.regions is None else sorted(self.regions),
        }
        blob = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _dedupe_records(records: Iterable[LanguageRecord]) -> dict[str, LanguageRecord]:
    out: dict[str, LanguageRecord] = {}
    for rec in sorted((r.cleaned() for r in records), key=lambda r: r.code):
        if not rec.iso639_3:
            raise RegistryError(f"record without ISO 639-3 code: {rec.name!r}")
        prev = out.get(rec.code)
        if prev is not None and prev != rec:
            raise RegistryError(
                f"conflicting records for {rec.code!r}: {prev} vs {rec}"
            )
        out[rec.code] = rec
    return out


def _index_codes(records: Iterable[LanguageRecord]) -> dict[str, tuple[str, CodeKind]]:
    index: dict[str, tuple[str, CodeKind]] = {}

    def put(code: str | None, canonical: str, kind: CodeKind) -> None:
        if not code:
            return
        prev = index.get(code)
        if prev is None:
            index[code] = (canonical, kind)
        elif prev[0] != canonical:
            raise RegistryError(
                f"code {code!r} names both {prev[0]!r} and {canonical!r}"
            )

    # Order matters only for `kind` when one string is several code types
    # (e.g. `fra` is both 639-3 and 639-2/T); the more specific label wins
    for rec in records:
        put(rec.iso639_1, rec.code, "iso639_1")
        put(rec.iso639_3, rec.code, "iso639_3")
        put(rec.iso639_2t, rec.code, "iso639_2")
        put(rec.iso639_2b, rec.code, "iso639_2")
    return dict(sorted(index.items()))


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()


def _name_keys(name: str) -> set[str]:
    keys = {_fold(name)}
    stripped = _fold(_PARENTHETICAL.sub("", name))
    if stripped:
        keys.add(stripped)
    return keys


# Parsing


def _is_ascii_alpha(s: str) -> bool:
    return s.isascii() and s.isalpha()


def _parse_subtags(
    subtags: list[str], registry: LanguageRegistry
) -> tuple[str | None, str | None, str | None, tuple[str, ...]]:
    """BCP 47-shaped subtags -> (script, region, variant, unparsed leftovers).

    A script or region that has the right shape but is not in the registry is
    a leftover, not a match: `zh-Hnat` must fail, not silently match nothing
    """
    script: str | None = None
    region: str | None = None
    variants: list[str] = []
    leftovers: list[str] = []

    for i, sub in enumerate(subtags):
        if not sub:
            continue
        if sub.lower() == "x":
            rest = [s.lower() for s in subtags[i + 1 :] if s]
            if rest:
                variants.append("x-" + "-".join(rest))
            else:
                leftovers.append(sub)
            break
        if (
            script is None
            and region is None
            and not variants
            and len(sub) == 4
            and _is_ascii_alpha(sub)
        ):
            candidate = sub.title()
            if registry.scripts is None or candidate in registry.scripts:
                script = candidate
            else:
                leftovers.append(sub)
        elif (
            region is None
            and not variants
            and (
                (len(sub) == 2 and _is_ascii_alpha(sub))
                or (len(sub) == 3 and sub.isdigit())
            )
        ):
            candidate = sub.upper()
            if registry.regions is None or candidate in registry.regions:
                region = candidate
            else:
                leftovers.append(sub)
        elif (
            sub.isascii()
            and sub.isalnum()
            and (5 <= len(sub) <= 8 or (len(sub) == 4 and sub[0].isdigit()))
        ):
            variants.append(sub.lower())
        else:
            leftovers.append(sub)

    return script, region, "-".join(variants) or None, tuple(leftovers)


def _resolve_base(
    base: str, registry: LanguageRegistry
) -> tuple[str | None, CodeKind | None, str | None]:
    hit = registry.lookup_code(base)

    if hit is not None:
        return hit[0], hit[1], None

    retired = registry.retirement(base)

    if retired is not None:
        target, raw = retired
        if target is not None:
            return (
                target,
                "retired",
                f"{base.lower()!r} is a retired code; using {target!r}",
            )
        detail = f" (replacement {raw!r} is not in the registry)" if raw else ""
        return (
            None,
            None,
            f"{base.lower()!r} is a retired code with no single replacement{detail}",
        )
    return None, None, None


def _exception_tag(
    spec: LanguageTag, raw: str, registry: LanguageRegistry
) -> SourceTag:
    hit = registry.lookup_code(spec.lang)

    if hit is None:
        return SourceTag(
            raw,
            "unmapped",
            None,
            f"exception target {spec.lang!r} is not in the registry",
        )

    return SourceTag(
        raw, "language", replace(spec, lang=hit[0]), "source-specific mapping"
    )


# Import side


def canonicalize_source_tag(
    raw: str | None, source: Source, registry: LanguageRegistry
) -> SourceTag:
    """Map one raw database tag to the registry vocabulary.

    More lenient than the user-input side: an unknown tag is kept and reported
    as `unmapped` rather than raising, because one odd row must not fail an
    import of millions
    """
    text = (raw or "").strip()
    key = text.lower()

    if source == "geonames":
        if key in _GEONAMES_NON_NAME:
            return SourceTag(text, "non_name", None, f"GeoNames pseudo-code {key!r}")
        if key in _GEONAMES_UNTAGGED:
            return SourceTag(text, "untagged")
        if key in _GEONAMES_TAG_EXCEPTIONS:
            return _exception_tag(_GEONAMES_TAG_EXCEPTIONS[key], text, registry)
    elif source == "wikidata":
        if key in _WIKIMEDIA_TAG_EXCEPTIONS:
            return _exception_tag(_WIKIMEDIA_TAG_EXCEPTIONS[key], text, registry)
    else:
        raise ValueError(f"unknown source {source!r}")

    if not key:
        return SourceTag(text, "untagged")
    if key in _SPECIAL_STATUS:
        return SourceTag(text, _SPECIAL_STATUS[key])

    base, *subtags = _SUBTAG_SPLIT.split(text)
    code, _, note = _resolve_base(base, registry)
    if code is None:
        return SourceTag(
            text, "unmapped", None, note or f"unknown language code {base.lower()!r}"
        )

    script, region, variant, leftovers = _parse_subtags(subtags, registry)
    if leftovers:
        extra = f"unparsed subtag(s) {', '.join(leftovers)} ignored"
        note = f"{note}; {extra}" if note else extra
    return SourceTag(text, "language", LanguageTag(code, script, region, variant), note)


class SourceTagCanonicalizer:
    def __init__(self, registry: LanguageRegistry, source: Source) -> None:
        self._registry = registry
        self._source = source
        self._cache: dict[str, SourceTag] = {}
        self._counts: Counter[str] = Counter()

    def __call__(self, raw: str | None) -> SourceTag:
        key = raw or ""
        result = self._cache.get(key)
        if result is None:
            result = canonicalize_source_tag(key, self._source, self._registry)
            self._cache[key] = result
        self._counts[key] += 1
        return result

    def report(self) -> list[tuple[SourceTag, int]]:
        """(result, row count), most rows first. Order is deterministic"""
        return sorted(
            ((self._cache[raw], n) for raw, n in self._counts.items()),
            key=lambda item: (item[0].status, -item[1], item[0].raw),
        )


# Query side


def resolve_user_language(text: str, registry: LanguageRegistry) -> LanguageQuery:
    """Resolve CLI input: ISO 639-1/2/3 code, optional subtags, or an exact name.

    Strict: anything that does not resolve to exactly one language raises
    UnknownLanguageError. Name matching is exact (case- and space-insensitive),
    never fuzzy; close matches only feed the error message. User input always
    has ISO meaning
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        raise UnknownLanguageError(text, "empty language")

    if _TAG_SHAPED.fullmatch(cleaned):
        base, *subtags = _SUBTAG_SPLIT.split(cleaned)
        code, kind, note = _resolve_base(base, registry)
        if code is not None and kind is not None:
            script, region, variant, leftovers = _parse_subtags(subtags, registry)
            if leftovers:
                raise UnknownLanguageError(
                    text, f"unrecognized subtag(s) {', '.join(leftovers)}"
                )
            return LanguageQuery(
                input=text,
                tag=LanguageTag(code, script, region, variant),
                matched_via=kind,
                notes=(note,) if note else (),
            )
        if note is not None:  # retired with no usable replacement
            raise UnknownLanguageError(
                text, note, _describe_all(registry.suggest(base), registry)
            )
        if _CODE_SHAPED.fullmatch(cleaned):
            raise UnknownLanguageError(
                text,
                "unknown language code",
                _describe_all(registry.suggest(cleaned), registry),
            )

    code = registry.lookup_name(cleaned)
    if code is not None:
        return LanguageQuery(
            input=text,
            tag=LanguageTag(code),
            matched_via="name",
            notes=(f"language name {cleaned!r} resolved to {code!r}",),
        )

    ambiguous = registry.ambiguous_name(cleaned)
    if ambiguous:
        raise UnknownLanguageError(
            text, "ambiguous language name", _describe_all(ambiguous, registry)
        )
    raise UnknownLanguageError(
        text, "unknown language", _describe_all(registry.suggest(cleaned), registry)
    )


def _describe_all(codes: Iterable[str], registry: LanguageRegistry) -> tuple[str, ...]:
    return tuple(registry.describe(c) for c in codes)
