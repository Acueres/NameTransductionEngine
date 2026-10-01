"""Last resorts before the native form: ICU's generic script transform, for
the scripts where its scholarly output still reads well, then uroman"""

from ..diagnostic_romanizer import DiagnosticRomanizer
from ..romanization import IDENTITY
from . import icu_util
from .base import Context, Rendering

# ICU "<script>-Latin" is usable for display only where it is a clean,
# phonemic transliteration. Excluded on purpose: abjads (unvoweled consonant
# strings with dots and rings), Thai (tone letters as diacritics), Canadian
# syllabics (leaves characters behind), Han (always Mandarin)
_ICU_GENERIC = frozenset(
    {
        "Armn", "Beng", "Bopo", "Cyrl", "Deva", "Ethi", "Geor", "Grek", "Gujr",
        "Guru", "Hang", "Hira", "Kana", "Knda", "Mlym", "Orya", "Syrc", "Taml",
        "Telu", "Thaa",
    }
)  # fmt: skip


def romanize_icu_generic(script: str, text: str, ctx: Context) -> Rendering | None:
    transform = f"{script}-Latn"
    if script not in _ICU_GENERIC or not icu_util.has_transform(transform):
        return None
    return Rendering(
        icu_util.apply(transform, text),
        f"icu:{transform}",
        "low",
        (
            f"no display romanization for this {script} text; scholarly transliteration used",
        ),
        engine=icu_util.ICU_ENGINE,
        cased=script in _CASED_SCRIPTS,
    )


# Scripts with upper and lower case; ICU carries the case over from them. For
# all others the output is lowercase and gets capitalized word by word
_CASED_SCRIPTS = frozenset({"Armn", "Cyrl", "Grek"})


_DIAGNOSTIC = DiagnosticRomanizer()


def romanize_uroman(text: str, ctx: Context) -> Rendering | None:
    result = _DIAGNOSTIC.romanize(text, ctx.tag)
    if result.transform == IDENTITY:
        return None
    return Rendering(
        # uroman keeps the Tibetan syllable mark as a middle dot
        result.text.replace("·", ""),
        "uroman",
        "low",
        ("no display romanization for this text; universal romanization used",)
        + result.warnings,
        engine=result.engine_version,
        cased=False,
    )
