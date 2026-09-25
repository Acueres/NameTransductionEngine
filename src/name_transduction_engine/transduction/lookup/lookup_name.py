from dataclasses import dataclass
from name_transduction_engine.transliteration.romanization import Romanization


@dataclass(frozen=True)
class LookupName:
    name: str
    romanization: Romanization | None
    language_code: str
