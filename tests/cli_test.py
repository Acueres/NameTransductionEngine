"""The CLI on top of the service: output lines and exit codes"""

from name_transduction_engine.datasets.availability import maintenance
from name_transduction_engine.main import main


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_lookup(capsys):
    code, out, _ = run(capsys, "lookup", "Москва", "--to", "os")
    assert code == 0
    assert "[entity] source=geonames id=524901" in out
    assert "  [name] Мæскуы (Mæskuy) lang=os" in out


def test_lookup_no_match(capsys):
    code, out, _ = run(capsys, "lookup", "Atlantis", "--to", "en")
    assert code == 0
    assert out.endswith("[result] no match\n")


def test_lookup_all_shows_entities_without_names(capsys):
    _, out, _ = run(capsys, "lookup", "Munich", "--to", "kk")
    assert "[entity]" not in out

    _, out, _ = run(capsys, "lookup", "Munich", "--to", "kk", "--all")
    assert out.count("  [name] no name for language=kk") == 2


def test_errors_exit_1(capsys):
    code, _, err = run(capsys, "lookup", "Tokyo", "--to", "jp")
    assert code == 1
    assert err.startswith("error: unknown language code: 'jp'; did you mean")

    code, _, err = run(capsys, "lookup", "  ", "--to", "en")
    assert (code, err) == (1, "error: name is empty\n")


def test_maintenance_exits_3(capsys):
    with maintenance("test"):
        code, _, err = run(capsys, "romanize", "Москва", "--lang", "ru")
    assert code == 3
    assert err.startswith("error: ")


def test_romanize(capsys):
    code, out, _ = run(capsys, "romanize", "평양", "--lang", "ko-KP")
    assert code == 0
    assert out.endswith("[result] P’yŏngyang confidence=high\n")


def test_romanize_untransducible(capsys):
    code, out, _ = run(capsys, "romanize", "😀", "--lang", "ru")
    assert code == 0
    assert out.endswith("[result] untransducible: nothing to romanize\n")


def test_lang(capsys):
    code, out, _ = run(capsys, "lang", "persian")
    assert code == 0
    assert out.startswith("[language] 'persian' -> fa  Persian\n")
