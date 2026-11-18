from dataclasses import dataclass
from typing import Literal

from name_transduction_engine.transliteration.romanization import Romanization


@dataclass(frozen=True)
class LookupName:
    name: str
    romanization: Romanization | None
    language_code: str
    # "attested": from a database. "generated" arrives with adaptation (Phase 3)
    origin: Literal["attested", "generated"] = "attested"
