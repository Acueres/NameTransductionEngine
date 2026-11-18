"""`nte romanize` through the service layer"""

import pytest

from name_transduction_engine.transduction.service import (
    RomanizeOutcome,
    UnknownLanguageError,
    get_romanization_result,
)
from name_transduction_engine.transliteration.romanization import non_latin_scripts

ROMANIZED = RomanizeOutcome.ROMANIZED
ALREADY_LATIN = RomanizeOutcome.ALREADY_LATIN
NATIVE_KEPT = RomanizeOutcome.NATIVE_KEPT
UNTRANSDUCIBLE = RomanizeOutcome.UNTRANSDUCIBLE


@pytest.mark.parametrize(
    "lang, text, outcome, expected",
    [
        ("ru", "Москва", ROMANIZED, "Moskva"),
        ("ukrainian", "Київ", ROMANIZED, "Kyiv"),
        ("ja", "東京", ROMANIZED, "Tōkyō"),
        ("ko", "평양", ROMANIZED, "Pyeongyang"),
        ("ko-KP", "평양", ROMANIZED, "P’yŏngyang"),  # the region picks the system
        ("ru", "München", ALREADY_LATIN, "München"),
        ("ru", "Moskva 2", ALREADY_LATIN, "Moskva 2"),
        ("ru", "東京", NATIVE_KEPT, "東京"),  # no reading of Han for Russian
        ("ru", "   ", UNTRANSDUCIBLE, None),
        ("ru", "!!! ???", UNTRANSDUCIBLE, None),
        ("ru", "😀", UNTRANSDUCIBLE, None),
        ("ru", "", UNTRANSDUCIBLE, None),
    ],
)
def test_outcomes(lang, text, outcome, expected):
    result = get_romanization_result(lang, text)
    assert result.outcome is outcome
    if expected is not None:
        assert result.romanization.text == expected


def test_language_resolves_by_name_with_a_note():
    result = get_romanization_result("persian", "تهران")
    assert str(result.language.tag) == "fa"
    assert result.language.notes == ("language name 'persian' resolved to 'fa'",)


def test_unknown_language():
    with pytest.raises(UnknownLanguageError):
        get_romanization_result("jp", "東京")


# Names in rare and ancient scripts (§4.3): romanized without errors and
# marked `low`. A script nothing can read keeps its native form, also `low`
RARE_SCRIPTS = [
    ("got", "𐌲𐌿𐍄𐌸𐌹𐌿𐌳𐌰"),  # Gothic
    ("peo", "𐎧𐏁𐎠𐎹𐎰𐎡𐎹"),  # Old Persian cuneiform
    ("gmy", "𐀒𐀇𐀵"),  # Linear B
    ("phn", "𐤂𐤁𐤋"),  # Phoenician
    ("uga", "𐎜𐎂𐎗𐎚"),  # Ugaritic
    ("egy", "𓊃𓈖𓆑"),  # Egyptian hieroglyphs
    ("akk", "𒀭𒈗"),  # Cuneiform
    ("sux", "𒂗𒆤"),  # Cuneiform
    ("non", "ᚠᚢᚦᚨᚱᚲ"),  # Runic
    ("sga", "ᚑᚌᚐᚋ"),  # Ogham
    ("cop", "ⲕⲏⲙⲉ"),  # Coptic
    ("cu", "ⰔⰎⰑⰂⰑ"),  # Glagolitic
    ("otk", "𐰇𐰼𐰚"),  # Old Turkic
    ("syc", "ܐܬܘܪ"),  # Syriac
    ("ae", "𐬀𐬵𐬎𐬭𐬀"),  # Avestan
    ("arc", "𐡀𐡓𐡌"),  # Imperial Aramaic
    ("smp", "ࠔࠌࠓࠍ"),  # Samaritan
    ("xsa", "𐩪𐩨𐩱"),  # Old South Arabian
    ("ett", "𐌓𐌏𐌌𐌀"),  # Old Italic
    ("pi", "𑀅𑀲𑁄𑀓"),  # Brahmi
    ("pgd", "𐨐𐨿𐨮"),  # Kharoshthi (Gāndhārī)
    ("mn-Mong", "ᠮᠣᠩᠭᠣᠯ"),  # Mongolian script
]


@pytest.mark.parametrize("lang, text", RARE_SCRIPTS)
def test_rare_scripts(lang, text):
    result = get_romanization_result(lang, text)
    romanization = result.romanization

    assert result.outcome in (ROMANIZED, NATIVE_KEPT)
    assert romanization.confidence == "low"
    if result.outcome is ROMANIZED:
        assert romanization.text and not non_latin_scripts(romanization.text)
