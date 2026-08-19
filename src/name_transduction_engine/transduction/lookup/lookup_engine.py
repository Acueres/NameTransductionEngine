from .db import run_query
from .lookup_candidate import LookupCandidate


def lookup_name(name: str, target: str) -> list[LookupCandidate]:
    QUERY = """
    WITH resolved_geonames AS (
    SELECT geonameid
    FROM geoname
    WHERE name = :name

    UNION

    SELECT geonameid
    FROM alternate_name
    WHERE alternate_name = :name
),

resolved_wikidata AS (
    SELECT qid
    FROM wikidata_location_name
    WHERE name = :name
)

SELECT DISTINCT
    'geonames'             AS source,
    CAST(rg.geonameid AS TEXT) AS entity_id,
    alt.isolanguage        AS language_code,
    alt.alternate_name     AS candidate_name
FROM resolved_geonames rg
JOIN alternate_name alt
    ON alt.geonameid = rg.geonameid
WHERE alt.isolanguage = :target

UNION ALL

SELECT DISTINCT
    'wikidata'             AS source,
    rw.qid                 AS entity_id,
    wd_name.geo_lang       AS language_code,
    wd_name.name           AS candidate_name
FROM resolved_wikidata rw
JOIN wikidata_location_name wd_name
    ON wd_name.qid = rw.qid
WHERE wd_name.geo_lang = :target

ORDER BY source, entity_id, candidate_name;
    """

    df = run_query(QUERY, {"name": name, "target": target})
    geonames_results = [
        LookupCandidate(
            candidate_name=row.candidate_name,
            candidate_name_transliterated=None,
            source=row.source,
            entity_id=row.entity_id,
            language_code=row.language_code,
        )
        for row in df.itertuples(index=False)
    ]

    return geonames_results
