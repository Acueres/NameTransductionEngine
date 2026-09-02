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
