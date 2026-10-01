from dataclasses import dataclass


@dataclass(frozen=True)
class LookupCandidate:
    source: str
    entity_id: str

    entity_type: str | None

    latitude: float | None
    longitude: float | None

    language_code: str | None
    candidate_name: str | None

    # The entity's GeoNames primary name and English names
    reference_names: tuple[str, ...] = ()
