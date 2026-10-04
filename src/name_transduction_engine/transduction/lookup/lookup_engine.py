import pandas as pd

from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from typing import Literal

from .db import run_query, get_conn
from .lookup_entity import LookupEntity
from .lookup_name import LookupName
from .lookup_candidate import LookupCandidate
from name_transduction_engine.normalization.name_normalization import normalize_name
from name_transduction_engine.normalization.language_code_normalization import (
    LanguageQuery,
    LanguageRegistry,
    LanguageTag,
    resolve_user_language,
)
from name_transduction_engine.transliteration.diagnostic_romanizer import (
    DiagnosticRomanizer,
)
from name_transduction_engine.transliteration.display_romanizer import (
    DisplayRomanizer,
)
from name_transduction_engine.datasets.dataset_provider import read_registry
from name_transduction_engine.paths import DB_PATH

# Resolve (step 1): every entity with any name, in any language, equal to
# the normalized input. Retrieve (step 2): that entity's names in the requested
# language. The optional script/region/variant filters narrow step 2 only.
#
# Wikidata: step 1 matches every name type (labels, aliases, official and
# historical names: "Tsargrad" finds Istanbul); step 2 returns labels only.
#
# Ordering: GeoNames before Wikidata, then entity ID numerically, then names in record order.
#
# Each row also carries the entity's reference names: the GeoNames primary
# name and up to five English names ("|"-separated). The display romanizer
# may use them to read scripts that leave vowels unwritten (Persian ونیز,
# guided by "Venice").

QUERY = """
WITH resolved_geonames AS (
    SELECT geonameid
    FROM geoname
    WHERE normalized_name = :name_norm

    UNION

    SELECT geonameid
    FROM alternate_name
    WHERE normalized_name = :name_norm
),

resolved_wikidata AS (
    SELECT DISTINCT qid
    FROM wikidata_name
    WHERE normalized_name = :name_norm
)

SELECT
    'geonames'                                  AS source,
    CAST(rg.geonameid AS TEXT)                  AS entity_id,
    rg.geonameid                                AS entity_order,
    gn.feature_class || ', ' || gn.feature_code AS entity_type,
    gn.latitude                                 AS latitude,
    gn.longitude                                AS longitude,
    alt.lang                                    AS lang,
    alt.lang_script                             AS lang_script,
    alt.lang_region                             AS lang_region,
    alt.lang_variant                            AS lang_variant,
    alt.alternate_name                          AS candidate_name,
    MIN(alt.alternate_name_id)                  AS name_order,
    gn.name || COALESCE('|' || (
        SELECT group_concat(en.alternate_name, '|')
        FROM (
            SELECT alternate_name
            FROM alternate_name
            WHERE geonameid = rg.geonameid AND lang = 'en'
            ORDER BY alternate_name_id
            LIMIT 5
        ) en
    ), '')                                      AS reference_names
FROM resolved_geonames rg
JOIN geoname gn ON gn.geonameid = rg.geonameid
LEFT JOIN alternate_name alt
    ON alt.geonameid = rg.geonameid
   AND alt.lang = :lang
   AND (:script  IS NULL OR alt.lang_script  = :script)
   AND (:region  IS NULL OR alt.lang_region  = :region)
   AND (:variant IS NULL OR alt.lang_variant = :variant)
GROUP BY
    rg.geonameid,
    alt.lang,
    alt.lang_script,
    alt.lang_region,
    alt.lang_variant,
    alt.alternate_name

UNION ALL

SELECT DISTINCT
    'wikidata'                                  AS source,
    rw.qid                                      AS entity_id,
    CAST(substr(rw.qid, 2) AS INTEGER)          AS entity_order,
    wd.entity_group || ', ' || wd.kind          AS entity_type,
    wd.lat                                      AS latitude,
    wd.lon                                      AS longitude,
    wd_name.lang                                AS lang,
    wd_name.lang_script                         AS lang_script,
    wd_name.lang_region                         AS lang_region,
    wd_name.lang_variant                        AS lang_variant,
    wd_name.name                                AS candidate_name,
    NULL                                        AS name_order,
    (
        SELECT group_concat(en.name, '|')
        FROM (
            SELECT name
            FROM wikidata_name
            WHERE qid = rw.qid AND lang = 'en' AND name_type = 'label'
            ORDER BY name
            LIMIT 5
        ) en
    )                                           AS reference_names
FROM resolved_wikidata rw
JOIN wikidata_entity wd ON wd.qid = rw.qid
LEFT JOIN wikidata_name wd_name
    ON wd_name.qid = rw.qid
   AND wd_name.lang = :lang
   AND wd_name.name_type = 'label'
   AND (:script  IS NULL OR wd_name.lang_script  = :script)
   AND (:region  IS NULL OR wd_name.lang_region  = :region)
   AND (:variant IS NULL OR wd_name.lang_variant = :variant)

ORDER BY
    source,
    entity_order,
    name_order,
    candidate_name,
    lang_script,
    lang_region,
    lang_variant;
"""


# "debug": uroman, a readable handle for any script (DiagnosticRomanizer)
# "pretty": the conventional romanization for end users (DisplayRomanizer)
RomanizationMode = Literal["debug", "pretty"]
ROMANIZATION_MODES: tuple[RomanizationMode, ...] = ("debug", "pretty")


@dataclass(frozen=True)
class LookupResult:
    language: LanguageQuery
    entities: list[LookupEntity]


def lookup_name(
    name: str,
    lang: str,
    db_path: str | Path = DB_PATH,
    romanization: RomanizationMode = "debug",
) -> LookupResult:
    language = resolve_user_language(lang, _registry(db_path))
    tag = language.tag

    df = run_query(
        db_path,
        QUERY,
        {
            "name_norm": normalize_name(name),
            "lang": tag.lang,
            "script": tag.script,
            "region": tag.region,
            "variant": tag.variant,
        },
    )
    candidates = [
        LookupCandidate(
            source=str(row.source),
            entity_id=str(row.entity_id),
            entity_type=_optional_str(row.entity_type),
            latitude=_optional_float(row.latitude),
            longitude=_optional_float(row.longitude),
            language_code=_language_code(row),
            candidate_name=_optional_str(row.candidate_name),
            reference_names=_split_names(row.reference_names),
        )
        for row in df.itertuples(index=False)
    ]

    grouped_candidates = _group_lookup_candidates(candidates)
    romanized_candidates = _romanize_lookup_names(
        grouped_candidates, _romanizer(romanization), query_name=name
    )
    return LookupResult(language=language, entities=romanized_candidates)


@lru_cache(maxsize=1)
def _registry(db_path: str | Path) -> LanguageRegistry:
    with closing(get_conn(db_path)) as conn:
        return read_registry(conn)


def _language_code(row) -> str | None:
    lang = _optional_str(row.lang)
    if lang is None:
        return None
    return str(
        LanguageTag(
            lang,
            _optional_str(row.lang_script),
            _optional_str(row.lang_region),
            _optional_str(row.lang_variant),
        )
    )


def _group_lookup_candidates(
    candidates: list[LookupCandidate],
) -> list[LookupEntity]:
    entities: list[LookupEntity] = []

    for _, group in groupby(
        candidates,
        key=attrgetter("source", "entity_id"),
    ):
        rows = list(group)
        first = rows[0]

        names = tuple(
            LookupName(
                name=row.candidate_name,
                romanization=None,
                language_code=row.language_code,
            )
            for row in rows
            if row.candidate_name is not None and row.language_code is not None
        )

        entities.append(
            LookupEntity(
                source=first.source,
                entity_id=first.entity_id,
                entity_type=first.entity_type,
                latitude=first.latitude,
                longitude=first.longitude,
                names=names,
                reference_names=first.reference_names,
            )
        )

    return entities


@lru_cache(maxsize=None)
def _romanizer(mode: RomanizationMode) -> DiagnosticRomanizer | DisplayRomanizer:
    if mode == "pretty":
        return DisplayRomanizer()
    if mode == "debug":
        return DiagnosticRomanizer()
    raise ValueError(f"unknown romanization mode {mode!r}")


def _romanize_lookup_names(
    entities: list[LookupEntity],
    rm: DiagnosticRomanizer | DisplayRomanizer,
    query_name: str | None = None,
) -> list[LookupEntity]:
    """Romanize every name. The display romanizer also gets the entity's
    reference names and the query (which resolved to this entity) as hints"""
    result: list[LookupEntity] = []

    for entity in entities:
        hints = (*entity.reference_names, *([query_name] if query_name else []))
        romanized_names = tuple(
            LookupName(
                name=n.name,
                romanization=(
                    rm.romanize(n.name, n.language_code, hints)
                    if isinstance(rm, DisplayRomanizer)
                    else rm.romanize(n.name, n.language_code)
                ),
                language_code=n.language_code,
            )
            for n in entity.names
        )

        new_entity = LookupEntity(
            source=entity.source,
            entity_id=entity.entity_id,
            entity_type=entity.entity_type,
            latitude=entity.latitude,
            longitude=entity.longitude,
            names=romanized_names,
            reference_names=entity.reference_names,
        )

        result.append(new_entity)

    return result


def _split_names(value) -> tuple[str, ...]:
    text = _optional_str(value)
    if not text:
        return ()
    return tuple(dict.fromkeys(n for n in text.split("|") if n))


def _optional_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _optional_str(value) -> str | None:
    return None if pd.isna(value) else str(value)
