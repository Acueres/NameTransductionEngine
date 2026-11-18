from dataclasses import dataclass, field
from enum import StrEnum
from .lookup_name import LookupName


class LookupOutcome(StrEnum):
    FOUND = "found"  # a name in the target language
    NO_TARGET_NAME = "no_target_name"  # found, but no name in that language
    NO_MATCH = "no_match"  # no entity has a name equal to the input


@dataclass(frozen=True)
class LookupEntity:
    source: str
    entity_id: str

    entity_type: str | None

    latitude: float | None
    longitude: float | None

    names: tuple[LookupName, ...]

    # Latin-script names of the entity (GeoNames primary name, English
    # names), used as reading hints by the romanizer
    reference_names: tuple[str, ...] = ()

    # Follows from `names`; never passed in
    outcome: LookupOutcome = field(init=False)

    def __post_init__(self) -> None:
        outcome = LookupOutcome.FOUND if self.names else LookupOutcome.NO_TARGET_NAME
        object.__setattr__(self, "outcome", outcome)
