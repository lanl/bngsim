"""A .net file whose bytes are not UTF-8 (issue #1653).

pybind11 hands every name, expression and message the loader makes to Python
through Python's strict UTF-8 decoder. A species name written in Latin-1
(``Aµ()``, the byte 0xb5) loaded, and its name then raised a bare
``UnicodeDecodeError`` that named neither the file nor the line; a refusal
quoting such a line lost its words to the same error.

The loader now refuses a line it reads (its comment off, in one of the five
blocks) that is not UTF-8, quoting it with the byte shown as ``\\xb5``. A byte
nothing reads, in a comment, a block passed over or an action line, still
loads, as it did. Any other refusal shows such a byte the same way. Measured
with the referees on the Latin-1 file: run_network 3.0 reads it, to
A(0.4) = 3 e^-0.8 = 1.347987; BNG2.pl 2.9.3's readFile refuses it ("Invalid
Molecule name"), and the UTF-8 one too, which keeps loading here.

``write_net`` now writes UTF-8 whatever the locale: under a Latin-1 or cp1252
locale (Windows) it wrote ``Aµ()`` as the byte this refuses, and could not
write ``AΩ()`` at all.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest

NET = (
    b"begin parameters\n 1 k 2\nend parameters\n"
    b"begin species\n 1 A() 3\n 2 B() 0\nend species\n"
    b"begin reactions\n 1 1 2 k #_R1\nend reactions\n"
    b"begin groups\n 1 Atot 1\nend groups\n"
)
NOT_UTF8 = "which is not UTF-8: bngsim reads the names and expressions of a .net file as UTF-8"
# CLOSED_FORM: A(0.4) against 3 e^-0.8 at rtol 1e-10 and atol 1e-12. Room for
# any platform; the check is that the file loads and runs the network it says.
CLOSED_FORM = 1e-7


def _write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "m.net"
    path.write_bytes(data)
    return path


def _edit(old: bytes, new: bytes) -> bytes:
    assert NET.count(old) == 1, old
    return NET.replace(old, new)


def _a_at(model: bngsim.Model) -> float:
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 0.4), n_points=2, rtol=1e-10, atol=1e-12
    )
    return float(np.asarray(result.species)[-1, 0])


# ── Refused, with the line and the byte ──────────────────────────────────────


def test_the_issue_s_latin_1_species_name_is_refused(tmp_path):
    """It loaded, and ``species_names`` raised a bare UnicodeDecodeError."""
    path = _write(tmp_path, _edit(b"A()", b"A\xb5()"))
    with pytest.raises(bngsim.ModelError, match=NOT_UTF8) as caught:
        bngsim.Model.from_net(str(path))
    assert "species line '1 A\\xb5() 3' holds the byte 0xb5" in str(caught.value)
    assert "m.net" in str(caught.value)
    with pytest.raises(ValueError, match=r"species line '1 A\\xb5\(\) 3' holds the byte 0xb5"):
        bngsim.parse_net_file(str(path))


IN_A_BLOCK = {
    "parameter": (b" 1 k 2\n", b" 1 k 2\n 2 k\xb5 3\n", "parameter line '2 k\\xb5 3'"),
    "species": (b"A()", b"A\xb5()", "species line '1 A\\xb5() 3'"),
    "functions": (
        b"begin reactions",
        b"begin functions\n 1 f() k*\xb5\nend functions\nbegin reactions",
        "functions line '1 f() k*\\xb5'",
    ),
    # The issue's second file: refused on main too, its words lost.
    "reaction": (b" 1 1 2 k #_R1", b" 1 1 2 k 7 \xb5", "reaction line '1 1 2 k 7 \\xb5'"),
    "group": (b"1 Atot 1", b"1 Atot\xb5 1", "group line '1 Atot\\xb5 1'"),
}


@pytest.mark.parametrize("block", sorted(IN_A_BLOCK))
def test_a_byte_in_each_block_s_text_is_refused(tmp_path, block):
    """Every block the loader reads, by the label its own refusals use."""
    old, new, quoted = IN_A_BLOCK[block]
    path = _write(tmp_path, _edit(old, new))
    with pytest.raises(bngsim.ModelError, match=NOT_UTF8) as caught:
        bngsim.Model.from_net(str(path))
    assert quoted in str(caught.value)


OUTSIDE_THE_BLOCKS = {
    "a-line-outside-any-block": (
        b"stray\xb5 line\n" + NET,
        "line outside any block: 'stray\\xb5 line'",
    ),
    "a-block-s-name": (
        NET + b"begin thing\xb5\nend thing\xb5\n",
        "Could not process block type 'thing\\xb5'",
    ),
    "an-end-without-a-begin": (NET + b"end widget\xb5\n", "'end widget\\xb5' without a 'begin"),
}


@pytest.mark.parametrize("case", sorted(OUTSIDE_THE_BLOCKS))
def test_a_refusal_of_a_line_no_block_reads_keeps_its_words(tmp_path, case):
    """Refused on main too, as a UnicodeDecodeError in place of its words."""
    data, words = OUTSIDE_THE_BLOCKS[case]
    path = _write(tmp_path, data)
    with pytest.raises(bngsim.ModelError) as caught:
        bngsim.Model.from_net(str(path))
    assert words in str(caught.value)
    with pytest.raises(ValueError) as parsed:
        bngsim.parse_net_file(str(path))
    assert words in str(parsed.value)


@pytest.mark.parametrize("run", [10, 400, 5000])
def test_a_cr_file_s_cut_stays_short_across_a_run_of_such_bytes(tmp_path, run):
    """A bare-CR file's refusal is cut at 300 characters (#1511). Each byte
    of a run across the cut read as a character's continuation, so the cut
    took the whole run, and the bytes then lost the refusal. Shown as ``\\xb5``
    first, the cut is where it was for any run."""
    data = (b"begin parameters\n 1 k" + b"\xb5" * run + b" 1\nend parameters\n").replace(
        b"\n", b"\r"
    )
    path = _write(tmp_path, data)
    with pytest.raises(ValueError, match="Could not process block type") as caught:
        bngsim.parse_net_file(str(path))
    message = str(caught.value)
    assert "carriage return that is not followed by a newline" in message
    # The cut's 300, the prefix and the CR hint; measured 603 for every run.
    assert len(message) < 700


@pytest.mark.parametrize("pad", [0, 1, 2, 3])
def test_a_cr_file_s_cut_ends_on_a_whole_escape(tmp_path, pad):
    """The cut lands at each of the four places a ``\\xb5`` can straddle it.
    Cut over the shown text, it left ``\\xb``, ``\\x`` or ``\\`` before the
    " ...", a byte the file does not hold."""
    data = b"begin parameters\n 1 k" + b"a" * pad + b"\xb5" * 400 + b" 1\nend parameters\n"
    path = _write(tmp_path, data.replace(b"\n", b"\r"))
    with pytest.raises(ValueError, match="carriage return") as caught:
        bngsim.parse_net_file(str(path))
    shown = str(caught.value).split(" ....")[0]
    assert shown.endswith("\\xb5"), shown[-12:]


# Bytes at every edge of the Unicode Standard's Table 3-7, which is what
# Python's strict decoder takes: overlong forms (C0, C1, E0 80, F0 80),
# surrogates (ED A0), past U+10FFFF (F4 90, F5), and the valid sequences on
# either side of each (E0 A0 80, ED 9F BF, EE 80 80, F0 90 80 80, F4 8F BF BF).
EDGE = [0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1, 0xEC]
EDGE += [0xED, 0xEE, 0xEF, 0xF0, 0xF1, 0xF3, 0xF4, 0xF5, 0xFF, ord("a")]


def _names():
    for b in range(0x80, 0x100):
        yield bytes([b])
    for a in EDGE:
        for b in EDGE:
            yield bytes([a, b])
    for cp in (0xB5, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF):
        yield chr(cp).encode()
    yield from (
        b"\xe0\xa0\x80",
        b"\xe0\x9f\xbf",
        b"\xed\x9f\xbf",
        b"\xed\xa0\x80",
        b"\xf0\x90\x80\x80",
    )
    yield from (
        b"\xf0\x8f\xbf\xbf",
        b"\xf4\x8f\xbf\xbf",
        b"\xf4\x90\x80\x80",
        b"\xe2\x82",
        b"\xf0\x9f\x98",
    )


def test_a_name_loads_exactly_when_python_can_decode_it(tmp_path):
    """Against Python's own decoder, as the names reach it: a name loads when
    its bytes decode, and is refused at the byte the decoder stops at,
    quoted as ``backslashreplace`` shows it."""
    disagree = []
    for name in _names():
        line = b"1 A" + name + b"() 3"
        path = _write(tmp_path, b"begin species\n " + line + b"\nend species\n")
        try:
            line.decode("utf-8")
            want = None
        except UnicodeDecodeError as err:
            shown = line.decode("utf-8", "backslashreplace")
            want = f"species line '{shown}' holds the byte 0x{line[err.start]:02x}"
        try:
            got = list(bngsim.Model.from_net(str(path)).species_names)
            if want is not None or got != ["A" + name.decode() + "()"]:
                disagree.append((name, got))
        except bngsim.ModelError as e:
            if want is None or want not in str(e):
                disagree.append((name, str(e)[-160:]))
    assert disagree == []


# ── Read as before ───────────────────────────────────────────────────────────


def test_a_utf_8_name_loads_and_runs(tmp_path):
    """Control: loaded on main too. BNG2.pl's readFile refuses it, as it
    refuses any name that is not ASCII; run_network reads it."""
    path = _write(tmp_path, _edit(b"A()", "Aµ()".encode()))
    model = bngsim.Model.from_net(str(path))
    assert list(model.species_names) == ["Aµ()", "B()"]
    assert _a_at(model) == pytest.approx(3 * math.exp(-0.8), rel=CLOSED_FORM)


NOT_READ = {
    "a-header-comment": b"# Created by BioNetGen 2.9.3, J\xf6rg's model\n" + NET,
    "a-comment-line-in-a-block": _edit(b" 1 A() 3\n", b"# note \xb5\n 1 A() 3\n"),
    "a-reaction-s-trailing-comment": _edit(b"#_R1", b"#_R\xb51"),
    "a-species-trailing-comment": _edit(b" 1 A() 3\n", b" 1 A() 3 # \xb5g/L\n"),
    "a-block-passed-over": b"begin compartments\n EC\xb5 3 1\nend compartments\n" + NET,
    "an-action-line": b'setOption("note","\xb5")\n' + NET,
}


@pytest.mark.parametrize("case", sorted(NOT_READ))
def test_a_byte_nothing_reads_still_loads(tmp_path, case):
    """Control: loaded on main too, and must not be refused: none of these
    bytes reaches Python. The model is the one without the byte."""
    model = bngsim.Model.from_net(str(_write(tmp_path, NOT_READ[case])))
    assert list(model.species_names) == ["A()", "B()"]
    assert _a_at(model) == pytest.approx(3 * math.exp(-0.8), rel=CLOSED_FORM)
    assert bngsim.parse_net_file(str(tmp_path / "m.net"))["species"]


# ── A table a .net reads ─────────────────────────────────────────────────────

TFUN_NET = (
    b"begin parameters\n 1 k0 0\nend parameters\n"
    b"begin functions\n 1 rate() tfun('rate.tfun')\nend functions\n"
    b"begin species\n 1 A() 1\n 2 B() 0\nend species\n"
    b"begin reactions\n 1 1 1,2 rate\nend reactions\n"
    b"begin groups\n 1 B_tot 2\nend groups\n"
)
TABLE = b"# time  rate\n0 0\n1 1\n2 1\n"


def _tfun(tmp_path: Path, table: bytes) -> Path:
    (tmp_path / "rate.tfun").write_bytes(table)
    return _write(tmp_path, TFUN_NET)


TABLE_REFUSALS = {
    # The reader quoted these raw, and they reach Python at the model's build.
    "a-data-row": (TABLE + b"3 \xb51\n", "invalid data line: '3 \\xb51'"),
    "a-header-column": (TABLE.replace(b"rate\n0", b"rat\xb5\n0"), "header column 2 'rat\\xb5'"),
    "a-header-index": (TABLE.replace(b"# time", b"# tim\xb5"), "header column 1 'tim\\xb5'"),
    "a-one-column-header": (b"# \xb5\n0 0\n1 1\n", "two column names (got: '# \\xb5')"),
}


@pytest.mark.parametrize("case", sorted(TABLE_REFUSALS))
def test_a_table_s_refusal_keeps_its_words(tmp_path, case):
    """The table is read when the model is built, after the .net's own lines,
    and its refusals quote the table's text."""
    table, words = TABLE_REFUSALS[case]
    path = _tfun(tmp_path, table)
    with pytest.raises(bngsim.ModelError) as caught:
        bngsim.Model.from_net(str(path))
    assert words in str(caught.value)
    with pytest.raises(ValueError) as parsed:
        bngsim.parse_net_file(str(path))
    assert words in str(parsed.value)


def test_a_table_read_from_a_parsed_dict_keeps_its_words(tmp_path):
    """``build_model_from_parsed`` reads the table again, through no loader's
    wrapper: the reader's own words are shown as UTF-8."""
    parsed = bngsim.parse_net_file(str(_tfun(tmp_path, TABLE)))
    (tmp_path / "rate.tfun").write_bytes(TABLE + b"3 \xb51\n")
    with pytest.raises(RuntimeError, match=r"invalid data line: '3 \\xb51'"):
        bngsim.build_model_from_parsed(parsed)


def test_a_byte_in_a_table_s_comment_still_loads(tmp_path):
    """Control: loaded on main too. Past its header, a table's comment lines
    are passed over, and so is a word after its two column names."""
    table = TABLE.replace(b"rate\n", b"rate  \xb5g/L\n") + b"# J\xf6rg's data\n3 1\n"
    model = bngsim.Model.from_net(str(_tfun(tmp_path, table)))
    assert list(model.species_names) == ["A()", "B()"]


_ROUND_TRIP = """
import locale, sys
import bngsim
from bngsim.convert import write_net
src, out = sys.argv[1], sys.argv[2]
print(locale.getpreferredencoding(False))
write_net(bngsim.Model.from_net(src), out)
print(ascii(list(bngsim.Model.from_net(out).species_names)))
"""


@pytest.mark.parametrize("name", ["Aµ()", "AΩ()"])
def test_write_net_writes_utf_8_whatever_the_locale(tmp_path, name):
    """Under the C locale, with Python's UTF-8 coercion off, the locale's
    encoding is ASCII: write_net raised UnicodeEncodeError on both names, as
    it raised on Ω under Latin-1, and wrote µ as the byte refused above."""
    src = _write(tmp_path, _edit(b"A()", name.encode()))
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONCOERCECLOCALE="0", PYTHONUTF8="0")
    proc = subprocess.run(
        [sys.executable, "-c", _ROUND_TRIP, str(src), str(tmp_path / "out.net")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
        env=env,
        timeout=120,
    )
    lines = proc.stdout.splitlines()
    if lines and lines[0].lower().replace("-", "") == "utf8":
        pytest.skip("this platform's C locale is UTF-8")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert lines[1] == ascii([name, "B()"])
    assert (tmp_path / "out.net").read_bytes().decode("utf-8")
