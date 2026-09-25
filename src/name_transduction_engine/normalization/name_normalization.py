import unicodedata


def normalize_name(name: str) -> str | None:
    name = " ".join(name.strip().split())
    if not name:
        return None

    name = unicodedata.normalize("NFC", name)
    name = name.casefold()
    return unicodedata.normalize("NFC", name)
