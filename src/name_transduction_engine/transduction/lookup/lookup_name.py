from dataclasses import dataclass


@dataclass(frozen=True)
class LookupName:
    name: str
    language_code: str
