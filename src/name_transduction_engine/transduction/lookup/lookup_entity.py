from dataclasses import dataclass
from .lookup_name import LookupName


@dataclass(frozen=True)
class LookupEntity:
    source: str
    entity_id: str

    entity_type: str | None

    latitude: float | None
    longitude: float | None

    names: tuple[LookupName, ...]

    # Latin-script names of the entity (GeoNames primary name, English
    # names), used as reading hints by the display romanizer
    reference_names: tuple[str, ...] = ()
