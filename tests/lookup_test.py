"""Lookup through the service layer, as the CLI and the API call it"""

import shutil
import sqlite3

import pytest

from name_transduction_engine.datasets.availability import maintenance
from name_transduction_engine.transduction import service
from name_transduction_engine.transduction.service import (
    DataMissing,
    DataUnavailable,
    InvalidInput,
    LookupOutcome,
    UnknownLanguageError,
    get_lookup_result,
)


def by_id(result):
    return {(e.source, e.entity_id): e for e in result.entities}


def names(entity):
    return [(n.name, n.language_code) for n in entity.names]


def test_entity_hop_russian_to_ossetian():
    result = get_lookup_result("os", "Москва")
    entities = by_id(result)

    assert result.outcome is LookupOutcome.FOUND
    assert names(entities["geonames", "524901"]) == [("Мæскуы", "os")]
    assert names(entities["wikidata", "Q649"]) == [("Мæскуы", "os")]


def test_ambiguous_name_gives_one_group_per_entity_in_fixed_order():
    result = get_lookup_result("en", "Moscow")
    keys = [(e.source, e.entity_id) for e in result.entities]

    assert len(keys) == len(set(keys)) > 20  # never merged
    assert {("geonames", "524901"), ("geonames", "5601538")} <= set(keys)
    # GeoNames before Wikidata, then IDs numerically
    assert keys == sorted(keys, key=lambda k: (k[0], int(k[1].lstrip("Q"))))


def test_resolves_on_any_wikidata_name_type():
    # A Russian alias of Moscow; retrieval returns labels only
    result = get_lookup_result("en", "Первопрестольная")
    assert [names(e) for e in result.entities] == [[("Moscow", "en")]]


def test_resolves_on_historic_geonames_names():
    result = get_lookup_result("en", "Constantinopolis")
    assert ("geonames", "745044") in by_id(result)


def test_subtags_narrow_the_names():
    serbian = by_id(get_lookup_result("sr", "Belgrade"))["wikidata", "Q3711"]
    latin = by_id(get_lookup_result("sr-Latn", "Belgrade"))["wikidata", "Q3711"]

    assert {code for _, code in names(serbian)} == {"sr", "sr-Latn", "sr-Cyrl"}
    assert names(latin) == [("Beograd", "sr-Latn")]


def test_names_are_attested_and_romanized():
    entity = by_id(get_lookup_result("ru", "Munich"))["geonames", "2867714"]
    (name,) = entity.names

    assert name.name == "Мюнхен"
    assert name.origin == "attested"
    assert name.romanization.text == "Myunkhen"


def test_region_subtag_picks_the_convention():
    entity = by_id(get_lookup_result("ko-KP", "Pyongyang"))["wikidata", "Q18808"]
    assert [n.romanization.text for n in entity.names] == ["P’yŏngyangsi"]


def test_no_target_name():
    result = get_lookup_result("kk", "Munich")

    assert result.outcome is LookupOutcome.NO_TARGET_NAME
    assert len(result.entities) == 2
    assert all(e.outcome is LookupOutcome.NO_TARGET_NAME for e in result.entities)
    assert all(e.names == () for e in result.entities)


def test_mixed_entity_outcomes():
    entities = by_id(get_lookup_result("sr-Latn", "Belgrade"))

    assert entities["wikidata", "Q3711"].outcome is LookupOutcome.FOUND
    assert entities["geonames", "792680"].outcome is LookupOutcome.NO_TARGET_NAME


def test_no_match():
    result = get_lookup_result("en", "Atlantis")
    assert result.outcome is LookupOutcome.NO_MATCH
    assert result.entities == ()


def test_language_resolves_by_name():
    result = get_lookup_result("ossetian", "Москва")
    assert str(result.language.tag) == "os"
    assert result.language.notes


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_name_is_invalid(name):
    with pytest.raises(InvalidInput):
        get_lookup_result("en", name)


def test_unknown_language_carries_suggestions():
    with pytest.raises(UnknownLanguageError) as error:
        get_lookup_result("jp", "Tokyo")
    assert "ojp (Old Japanese)" in error.value.suggestions


def test_refused_during_maintenance():
    with maintenance("test"):
        with pytest.raises(DataUnavailable):
            get_lookup_result("en", "Moscow")
    assert get_lookup_result("en", "Moscow").outcome is LookupOutcome.FOUND


def test_missing_database(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "DB_PATH", tmp_path / "names.sqlite")
    with pytest.raises(DataMissing):
        get_lookup_result("en", "Moscow")


def test_database_without_name_tables(monkeypatch, tmp_path, nte_data):
    # The registry loads, the names query fails: still DataMissing, not a
    # pandas error
    path = tmp_path / "names.sqlite"
    shutil.copy(nte_data / "names.sqlite", path)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE wikidata_name")
    monkeypatch.setattr(service, "DB_PATH", path)

    with pytest.raises(DataMissing, match="no such table: wikidata_name"):
        get_lookup_result("en", "Moscow")
