import icu

from functools import lru_cache

ICU_ENGINE = f"ICU {icu.ICU_VERSION}"


@lru_cache(maxsize=None)
def transliterator(transform_id: str) -> icu.Transliterator:
    return icu.Transliterator.createInstance(transform_id)


@lru_cache(maxsize=1)
def available_ids() -> frozenset[str]:
    return frozenset(icu.Transliterator.getAvailableIDs())


def has_transform(transform_id: str) -> bool:
    # Compound IDs ("A; B") are checked part by part
    return all(part.strip() in available_ids() for part in transform_id.split(";"))


def apply(transform_id: str, text: str) -> str:
    return transliterator(transform_id).transliterate(text)
