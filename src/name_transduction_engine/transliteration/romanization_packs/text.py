"""Text helpers shared by the display providers"""

import unicodedata


def capitalize_first(word: str) -> str:
    """Upper-case the first letter, skipping leading ʿ, ʾ, apostrophes"""
    for i, ch in enumerate(word):
        if ch.isalpha() and unicodedata.category(ch) != "Lm":
            return word[:i] + ch.upper() + word[i + 1 :]
    return word


def titlecase_words(text: str) -> str:
    """Capitalize the first letter of each whitespace-separated word"""
    return " ".join(capitalize_first(w) for w in text.split(" "))


def strip_combining(text: str, marks: frozenset[str]) -> str:
    """Remove the given combining marks, keeping base letters and other marks"""
    decomposed = unicodedata.normalize("NFD", text)
    kept = "".join(ch for ch in decomposed if ch not in marks)
    return unicodedata.normalize("NFC", kept)


class TableTransliterator:
    """Longest-match character table with word-initial variants.

    `table` maps lowercase source sequences to output; `initial` overrides the
    output at the start of a word. Case is carried over from the source.
    Characters not in the table are passed through unchanged.
    """

    def __init__(self, table: dict[str, str], initial: dict[str, str] | None = None):
        self.table = table
        self.initial = initial or {}
        self.max_len = max(len(k) for k in [*table, *self.initial])

    def __call__(self, text: str) -> str:
        out: list[str] = []
        i = 0
        at_word_start = True
        while i < len(text):
            for n in range(min(self.max_len, len(text) - i), 0, -1):
                chunk = text[i : i + n]
                key = chunk.lower()
                if at_word_start and key in self.initial:
                    out.append(_case_like(chunk, self.initial[key], text, i + n))
                    break
                if key in self.table:
                    out.append(_case_like(chunk, self.table[key], text, i + n))
                    break
            else:
                n = 1
                out.append(text[i])
            at_word_start = (
                not text[i + n - 1].isalpha() and text[i + n - 1] not in "'’ʼ"
            )
            i += n
        return "".join(out)


def _case_like(chunk: str, mapped: str, text: str, end: int) -> str:
    if not mapped or not chunk[:1].isupper():
        return mapped
    # An all-caps word maps to all caps ("ЩУКА" -> "SHCHUKA"), otherwise only
    # the first letter is capitalized ("Щука" -> "Shchuka")
    nxt = text[end : end + 1]
    if len(chunk) > 1 and chunk.isupper() or (nxt.isalpha() and nxt.isupper()):
        return mapped.upper()
    return mapped[:1].upper() + mapped[1:]
