from dataclasses import dataclass
from ..transliteration.romanization import Romanization


@dataclass(frozen=True)
class LookupName:
    name: str
    romanization: Romanization | None
    language_code: str

    @property
    def romanized_name(self) -> str | None:
        return self.romanization.text if self.romanization is not None else None
