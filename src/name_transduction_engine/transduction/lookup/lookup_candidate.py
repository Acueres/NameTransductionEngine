from dataclasses import dataclass

@dataclass
class LookupCandidate:
    candidate_name: str
    candidate_name_transliterated: str | None
    source: str
    entity_id: str
    language_code: str | None