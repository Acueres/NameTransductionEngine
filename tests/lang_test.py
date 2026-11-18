"""`nte lang` through the service layer: resolution, coverage, list, search"""

from name_transduction_engine.transduction.service import (
    get_language_list_result,
    get_language_result,
    get_language_search_result,
)


def test_name_resolves_with_coverage():
    result = get_language_result("persian")

    assert result.language.tag.lang == "fa"
    assert result.language.notes == ("language name 'persian' resolved to 'fa'",)
    assert result.counts == {"geonames": 2, "wikidata": 37}
    assert result.suggestion is None


def test_retired_code():
    result = get_language_result("iw")
    assert result.language.tag.lang == "he"
    assert result.language.notes == ("'iw' is a retired code; using 'he'",)


def test_script_subtag_excludes_other_scripts():
    result = get_language_result("zh-Hant")
    excluded = {t.raw.raw_tag for t in result.tags if not t.counted}

    assert result.counts == {"wikidata": 10}
    assert "zh-hans" in excluded


def test_suggests_the_macrolanguage_when_a_member_has_no_names():
    result = get_language_result("cmn")

    assert result.total == 0
    assert result.macrolanguage.code == "zh"
    assert result.suggestion == result.macrolanguage


def test_no_names_and_nothing_related():
    result = get_language_result("kk")
    assert result.total == 0
    assert result.suggestion is None


def test_list():
    result = get_language_list_result()
    languages = {lang.code: lang for lang in result.languages}

    assert result.sources == ("geonames", "wikidata")
    assert result.languages[0].code == "en"  # most names first
    assert languages["os"].counts == {"geonames": 1, "wikidata": 9}


def test_search():
    matches = get_language_search_result("persian").matches

    assert matches[0].code == "fa"
    assert matches[0].matched is None  # matched its own name
    assert any(m.matched == "Persian Sign Language" for m in matches)


def test_search_without_matches():
    assert get_language_search_result("qqqqq").matches == ()
