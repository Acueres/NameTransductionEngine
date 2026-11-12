from functools import lru_cache
from contextlib import closing

from name_transduction_engine.transliteration.romanizer import Romanizer
from name_transduction_engine.normalization.language_code_normalization import (
    LanguageRegistry,
)
from name_transduction_engine.datasets.language_codes.data_provision import (
    read_registry,
)
from name_transduction_engine.datasets.language_codes.info import (
    Coverage,
    read_coverage,
)
from .db import get_conn


@lru_cache(maxsize=1)
def get_romanizer() -> Romanizer:
    return Romanizer()


@lru_cache(maxsize=1)
def get_registry(db_path: str) -> LanguageRegistry:
    with closing(get_conn(db_path)) as conn:
        return read_registry(conn)


@lru_cache(maxsize=1)
def get_coverage(db_path: str) -> Coverage:
    with closing(get_conn(db_path)) as conn:
        return read_coverage(conn)
