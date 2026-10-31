"""An NFsim load leaves the host process as it found it (issue #1314).

``create_system()`` changed into the XML's directory so NFsim could open a
TFUN's ``file=`` by relative path, and changed back after the load as plain
code: an exception out of ``initializeFromXML`` skipped the restore and left
the host in the XML's directory, and every later relative path resolved
against it. ``initCompartments`` is the one loader stage with no catch-all, so
a malformed compartment ``size`` is that exception.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_root import bngsim_source_root  # noqa: E402

_DATA = (bngsim_source_root() or Path(__file__).resolve().parents[2]) / "tests" / "data" / "nfsim"

if not bngsim._bngsim_core.HAS_NFSIM:
    pytest.skip("NFsim not compiled in", allow_module_level=True)

_EMPTY_COMPARTMENTS = "    <ListOfCompartments>\n    </ListOfCompartments>\n"
_BAD_COMPARTMENT = (
    "    <ListOfCompartments>\n"
    '      <Compartment id="C" spatialDimensions="3" size="1.0abc"/>\n'
    "    </ListOfCompartments>\n"
)


def _fixture(name: str) -> Path:
    path = _DATA / name
    if not path.is_file():
        pytest.skip(f"test data not found: {path}")
    return path


@pytest.fixture
def places(tmp_path, monkeypatch):
    """A model directory and another one to load it from: the cwd is the latter."""
    model, elsewhere = tmp_path / "model", tmp_path / "elsewhere"
    model.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return model, elsewhere


def _core(xml: Path, **params):
    """The core simulator, which loads at initialize(): bngsim's Python layer
    refuses a compartment model before NFsim reads it (#1067), so the loader's
    throw is reached here, the embeddable C++ surface the issue is about."""
    sim = bngsim._bngsim_core.NfsimSimulator(str(xml))
    for name, value in params.items():
        sim.set_param(name, value)
    return sim


def _bad_size(model: Path, text_edit=lambda t: t) -> Path:
    text = _fixture("simple_system.xml").read_text()
    assert _EMPTY_COMPARTMENTS in text
    out = model / "badsize.xml"
    out.write_text(text_edit(text.replace(_EMPTY_COMPARTMENTS, _BAD_COMPARTMENT)))
    return out


def test_a_load_that_throws_leaves_the_cwd_and_names_the_file(places):
    model, elsewhere = places
    xml = _bad_size(model)
    with pytest.raises(RuntimeError, match="Failed to load XML file") as err:
        _core(xml).initialize(1)
    assert Path(os.getcwd()).resolve() == elsewhere.resolve()
    assert str(xml) in str(err.value), str(err.value)
    assert "convertToDouble" in str(err.value), str(err.value)


def test_a_load_that_fails_by_returning_null_leaves_the_cwd(places):
    """The loader stages that catch their own throws return NULL: that path
    restored the cwd already. Pinned."""
    model, elsewhere = places
    xml = model / "composite.xml"
    shutil.copy(_fixture("composite_function_reads_observable.xml"), xml)
    with pytest.raises(bngsim.SimulationError, match="Failed to load XML file"):
        bngsim.NfsimSession(str(xml)).initialize(seed=1)
    assert Path(os.getcwd()).resolve() == elsewhere.resolve()


def test_a_load_through_the_override_copy_leaves_the_cwd(places):
    """A set_param before initialize() loads a rewritten copy from the temp
    directory, and changed into the original XML's directory all the same."""
    model, elsewhere = places
    with pytest.raises(RuntimeError, match="convertToDouble"):
        _core(_bad_size(model), kon=2.0).initialize(1)
    assert Path(os.getcwd()).resolve() == elsewhere.resolve()


def test_a_load_through_the_seed_round_copy_leaves_the_cwd(places):
    """A fractional seed concentration loads a rounded copy from the temp
    directory (GH #51)."""
    model, elsewhere = places

    def fractional(text):
        i = text.index('concentration="')
        j = text.index('"', i + len('concentration="'))
        return text[: i + len('concentration="')] + "10.4" + text[j:]

    with pytest.raises(RuntimeError, match="convertToDouble"):
        _core(_bad_size(model, fractional)).initialize(1)
    assert Path(os.getcwd()).resolve() == elsewhere.resolve()


def _tfun_model(model: Path) -> Path:
    """tfun_simple.xml and its table, copied into ``model``."""
    src = _fixture("tfun_test/tfun_simple.xml")
    shutil.copy(src, model / src.name)
    shutil.copy(src.parent / "tfun_data.dat", model / "tfun_data.dat")
    return model / src.name


def _decoy(where: Path) -> None:
    """A table of the same name with every multiplier 0: a path resolved
    against the wrong directory reads it, runs, and gives other numbers."""
    table = (_fixture("tfun_test/tfun_simple.xml").parent / "tfun_data.dat").read_text()
    rows = [
        line if line.startswith("#") or not line.strip() else f"{line.split()[0]}    0.0"
        for line in table.splitlines()
    ]
    (where / "tfun_data.dat").write_text("\n".join(rows) + "\n")


def _observables(xml: Path, **params) -> np.ndarray:
    sess = bngsim.NfsimSession(str(xml))
    for name, value in params.items():
        sess.set_param(name, value)
    sess.initialize(seed=1)
    r = sess.simulate(0.0, 10.0, 11)
    sess.destroy()
    return np.asarray(r.observables)


def _from_its_own_directory(xml: Path, monkeypatch, **params) -> np.ndarray:
    """The oracle: the run with the cwd at the XML, where a relative path
    resolves as written on main and here alike."""
    here = os.getcwd()
    monkeypatch.chdir(xml.parent)
    try:
        return _observables(xml, **params)
    finally:
        monkeypatch.chdir(here)


def test_a_tfun_file_by_relative_path_loads_from_elsewhere(places, monkeypatch):
    """Why the cwd was changed at all: a TFUN's file= is opened at load, by
    relative path. It must still read the XML's own table with the cwd
    elsewhere, and a table of the same name there must not be read."""
    model, elsewhere = places
    xml = _tfun_model(model)
    assert 'file="tfun_data.dat"' in xml.read_text()
    want = _from_its_own_directory(xml, monkeypatch)
    _decoy(elsewhere)
    np.testing.assert_array_equal(_observables(xml), want)
    assert Path(os.getcwd()).resolve() == elsewhere.resolve()


def test_a_tfun_file_resolves_against_the_original_xml_through_a_copy(places, monkeypatch):
    """A set_param before initialize() loads a rewritten copy from the temp
    directory: the TFUN's relative file= is the original XML's neighbour, not
    the copy's."""
    model, elsewhere = places
    xml = _tfun_model(model)
    param = {_first_param(xml): 1.5}
    want = _from_its_own_directory(xml, monkeypatch, **param)
    _decoy(elsewhere)
    np.testing.assert_array_equal(_observables(xml, **param), want)
    assert Path(os.getcwd()).resolve() == elsewhere.resolve()


def test_a_tfun_file_in_a_directory_xml_must_escape(tmp_path, monkeypatch):
    """The absolute path is written into an attribute: a directory named with
    an ampersand, an apostrophe and a space is escaped, and loads."""
    model = tmp_path / "a&b 'c'"
    model.mkdir()
    monkeypatch.chdir(tmp_path)
    xml = _tfun_model(model)
    want = _from_its_own_directory(xml, monkeypatch)
    _decoy(tmp_path)
    np.testing.assert_array_equal(_observables(xml), want)


_WRITTEN = 'file="tfun_data.dat"'
_ONE_LINE_OPEN = "<ListOfFunctions>\n"


def _on_the_list_line(text: str) -> str:
    """The first <Function> on the <ListOfFunctions> line, the comment between
    them dropped."""
    head, rest = text.split(_ONE_LINE_OPEN, 1)
    return head + "<ListOfFunctions>" + rest[rest.index("<Function") :]


# Spellings of one TFUN element that NFsim's TinyXML reads as the same
# attribute (TiXmlAttribute::Parse), and markup round it that TinyXML skips.
# The rewrite read the XML by line and matched `file="` or `file='` literally,
# so every layout but the attribute on its own line loaded from the XML's
# directory only, and from anywhere else read the cwd's table, if any. The
# comment and the CDATA hold the list's end tag: a scan that read inside
# either would stop there, before the real <Function>.
_LAYOUTS = {
    "spaces round =": lambda t: t.replace(_WRITTEN, 'file = "tfun_data.dat"'),
    "on the ListOfFunctions line": _on_the_list_line,
    "the whole file on one line": lambda t: t.replace("\n", " "),
    "single quotes, spaces": lambda t: t.replace(_WRITTEN, "file = 'tfun_data.dat'"),
    "unquoted": lambda t: t.replace(_WRITTEN, "file=tfun_data.dat"),
    "the attribute on its own line": lambda t: t.replace(_WRITTEN, "\n        " + _WRITTEN),
    "after a commented-out Function and end tag": lambda t: t.replace(
        "<Function ",
        '<!-- <Function id="old" file="old.dat"> </ListOfFunctions> --> <Function ',
        1,
    ),
    "after CDATA holding the end tag": lambda t: t.replace(
        "<Function ", "<![CDATA[ </ListOfFunctions> ]]> <Function ", 1
    ),
    "after another attribute ending in file": lambda t: t.replace(
        _WRITTEN, 'data-file="no.dat" ' + _WRITTEN
    ),
}


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_a_tfun_file_in_any_layout_tinyxml_reads_loads_from_elsewhere(places, monkeypatch, layout):
    model, elsewhere = places
    xml = _tfun_model(model)
    want = _from_its_own_directory(xml, monkeypatch)
    text = xml.read_text()
    changed = _LAYOUTS[layout](text)
    assert changed != text, "the layout's spelling is not in the fixture"
    xml.write_text(changed)
    _decoy(elsewhere)
    np.testing.assert_array_equal(_observables(xml), want)


def _first_param(xml: Path) -> str:
    text = xml.read_text()
    i = text.index('<Parameter id="') + len('<Parameter id="')
    return text[i : text.index('"', i)]


def test_the_streams_work_after_a_failed_load(places, capfd):
    model, _ = places
    with pytest.raises(RuntimeError, match="convertToDouble"):
        _core(_bad_size(model)).initialize(1)
    print("python stdout")
    print("python stderr", file=sys.stderr)
    os.write(1, b"fd 1\n")
    out, err = capfd.readouterr()
    assert "python stdout" in out and "fd 1" in out
    assert "python stderr" in err
    # and NFsim's own output is captured again on the next load, not leaked
    good = bngsim.NfsimSession(str(_fixture("simple_system.xml")))
    good.initialize(seed=1)
    good.simulate(0.0, 1.0, 2)
    good.destroy()


def _segments(xml: str, between=None):
    """Session A at seed 7 to t = 5, then on to 10; ``between`` runs between
    the two segments."""
    a = bngsim.NfsimSession(xml)
    a.initialize(seed=7)
    first = np.asarray(a.simulate(0.0, 5.0, 6).observables)
    if between:
        between()
    rest = np.asarray(a.simulate(5.0, 10.0, 6).observables)
    a.destroy()
    return np.concatenate([first, rest])


def _another_session(xml: str):
    def run():
        b = bngsim.NfsimSession(xml)
        b.initialize(seed=12345)
        b.simulate(0.0, 10.0, 3)
        b.destroy()

    return run


def test_exchangeable_molecules_hide_which_one_is_picked():
    """The control: on simple_system every molecule a rule can fire on is like
    every other, so which one the shared RNG picks never shows."""
    xml = str(_fixture("simple_system.xml"))
    np.testing.assert_array_equal(_segments(xml, _another_session(xml)), _segments(xml))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1438: ReactantList::pickRandom (reactantList.cpp L39, L46) and reaction.cpp "
    "L771 draw from NFutil's process-wide Mersenne Twister, which prepare_system() "
    "reseeds: a session prepared between another's segments moves its picks. A "
    "re-vendor of NFsim that changes those draws turns this into an XPASS",
)
def test_another_session_between_segments_does_not_move_a_seeded_trajectory():
    """In a TLBR model the molecules a rule can fire on sit in aggregates of
    every size, so which one is picked shows in the observables."""
    xml = str(_fixture("tlbr_shared_rng.xml"))
    np.testing.assert_array_equal(_segments(xml, _another_session(xml)), _segments(xml))
