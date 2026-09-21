import pandas as pd

from itertools import groupby
from operator import attrgetter

from name_transduction_engine.normalization import normalize_name
from .db import run_query
from .lookup_entity import LookupEntity
from .lookup_name import LookupName
from .lookup_candidate import LookupCandidate
from ..transliteration.romanization import Romanizer


def lookup_name(name: str, target: str) -> list[LookupEntity]:
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
    FROM wikidata_location_name
    WHERE name_norm = :name_norm
)

SELECT DISTINCT
    'geonames'                 AS source,
    CAST(rg.geonameid AS TEXT) AS entity_id,
    concat(gn.feature_class, ', ',
           gn.feature_code)    AS entity_type,
    gn.latitude,
    gn.longitude,
    alt.isolanguage            AS language_code,
    alt.alternate_name         AS candidate_name
FROM resolved_geonames rg
JOIN geoname gn ON gn.geonameid = rg.geonameid
LEFT JOIN alternate_name alt
    ON alt.geonameid = rg.geonameid
       AND alt.isolanguage = :target

UNION ALL

SELECT DISTINCT
    'wikidata'             AS source,
    rw.qid                 AS entity_id,
    wd_loc.kind            AS entity_type,
    lat                    AS latitude,
    lon                    AS longitude,
    wd_name.geo_lang       AS language_code,
    wd_name.name           AS candidate_name
FROM resolved_wikidata rw
JOIN wikidata_location wd_loc ON wd_loc.qid = rw.qid
LEFT JOIN wikidata_location_name wd_name
    ON wd_name.qid = rw.qid
       AND wd_name.geo_lang = :target

ORDER BY source, entity_id, candidate_name;
    """

    name_norm = normalize_name(name)

    df = run_query(QUERY, {"name_norm": name_norm, "target": target})
    candidates = [
        LookupCandidate(
            source=str(row.source),
            entity_id=str(row.entity_id),
            entity_type=str(row.entity_type),
            latitude=_optional_float(row.latitude),
            longitude=_optional_float(row.longitude),
            language_code=_optional_str(row.language_code),
            candidate_name=_optional_str(row.candidate_name),
        )
        for row in df.itertuples(index=False)
    ]

    grouped_candidates = _group_lookup_candidates(candidates)
    romanized_candidates = _romanize_lookup_names(grouped_candidates)
    return romanized_candidates


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
            )
        )

    return entities


def _romanize_lookup_names(entities: list[LookupEntity]):
    result: list[LookupEntity] = []
    rm = Romanizer()

    for entity in entities:
        romanized_names = tuple(
            LookupName(
                name=n.name,
                romanization=rm.romanize(n.name),
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
                    )

        result.append(new_entity)

    return result


def _optional_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _optional_str(value) -> str | None:
    return None if pd.isna(value) else str(value)
