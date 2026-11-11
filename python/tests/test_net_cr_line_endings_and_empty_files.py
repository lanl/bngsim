"""A .net file from which no block is read, and one whose lines end in a bare
carriage return (issue #1511).

A 0-byte file, one of blank or comment lines only, and one whose lines end in
a bare CR (classic Mac OS) behind the ``# Created by BioNetGen`` line BNG2.pl
writes first loaded as a model of 0 species and 0 reactions, with no error:
the CR file is one line to the loader, and that line is a comment. run_network
3.0 refuses each ("Couldn't read rates array").

BNG2.pl and run_network read a bare CR as a blank inside its line, not as a
line end, and so does the loader: a CR in place of one newline joins two
lines, and a CR anywhere else inside a line changes nothing. Measured with
run_network 3.0 and BNG2.pl 2.9.3's readFile on the files below: each CR edit
that loads here both referees read as 2 species and 2 reactions, run_network
to A(0.4) = 3.839397, and run_network refuses every file refused here. The
blocks that are there and empty, at the end, run_network refuses as well
("Couldn't read rates array", "must have parameters and/or functions"); they
keep loading here, as a block was read.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

NET = """# Created by BioNetGen 2.9.3
begin parameters
    1 kf 2
    2 kr 0.5
end parameters
begin species
    1 A() 7
    2 B() 3
end species
begin reactions
    1 1 2 kf #_R1
    2 2 1 kr #_reverse__R1
end reactions
begin groups
    1 Atot 1
end groups
"""

CR_HINT = "carriage return that is not followed by a newline"
# Tolerances and bounds, each with its reason (measured on macOS arm64):
#
# CLOSED_FORM: a run at the default tolerances against A(0.4) = 2 + 5 e^-1, the
# closed form for A <-> B: measured 7.7e-09. 1e-6 leaves room for another
# platform, and is far from the 3.145 a dropped reaction gave.
CLOSED_FORM = 1e-6
# MESSAGE_ROOM: what a refusal holds past its path: the reading's own words cut
# at 300 characters, the "Failed to load" prefixes, and the CR hint; measured
# 583. 1000 leaves room for a longer hint, and an uncut quote, the whole file,
# runs to 33,000 to 41,000 (main, on the 2,000-parameter file below).
MESSAGE_ROOM = 1000
NO_BLOCK = "has no parameters, species, functions, reactions or groups block"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "m.net"
    # newline="" writes the bytes as given, on Windows too.
    with open(path, "w", newline="") as f:
        f.write(text)
    return path


def _a_at(model: bngsim.Model) -> np.ndarray:
    result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 0.4), n_points=2)
    return np.asarray(result.species)[-1]


# ── Refused ──────────────────────────────────────────────────────────────────


def test_a_file_whose_lines_end_in_a_bare_cr_is_refused(tmp_path):
    """The issue's file. It loaded as 0 species and 0 reactions."""
    path = _write(tmp_path, NET.replace("\n", "\r"))
    with pytest.raises(bngsim.ModelError, match=NO_BLOCK) as caught:
        bngsim.Model.from_net(str(path))
    assert CR_HINT in str(caught.value)


def test_parse_net_file_refuses_it_too(tmp_path):
    """The dictionary reading goes through the same reader, and refuses with
    the ValueError ``parse_net_file`` documents."""
    path = _write(tmp_path, NET.replace("\n", "\r"))
    with pytest.raises(ValueError, match=NO_BLOCK) as caught:
        bngsim.parse_net_file(str(path))
    assert CR_HINT in str(caught.value)


def test_a_bare_cr_file_without_the_header_comment_says_why(tmp_path):
    """Refused before too, as a block named by the whole file. The refusal
    now says what made the line."""
    path = _write(tmp_path, NET.split("\n", 1)[1].replace("\n", "\r"))
    with pytest.raises(bngsim.ModelError, match="Could not process block type") as caught:
        bngsim.Model.from_net(str(path))
    assert CR_HINT in str(caught.value)


def test_a_stray_cr_for_one_newline_says_why(tmp_path):
    """``#_reverse__R1\\rend reactions`` is one line, whose comment holds the
    ``end reactions``, as it is for BNG2.pl ("begin block before end of
    previous block") and run_network ("No reactions in the network")."""
    path = _write(tmp_path, NET.replace("R1\nend reactions", "R1\rend reactions"))
    with pytest.raises(bngsim.ModelError, match="'end reactions' is missing") as caught:
        bngsim.Model.from_net(str(path))
    assert CR_HINT in str(caught.value)


LARGE_CR_FILES = {
    # Read as a block named by the whole file.
    "a-block-name": ("", "Could not process block type"),
    # Read as one line outside any block, quoted as it is, CRs and all.
    "a-line-outside-any-block": ('version("2.9.3")\n', "a line outside any block"),
}


@pytest.mark.parametrize("shape", sorted(LARGE_CR_FILES))
def test_the_refusal_of_a_large_cr_file_stays_readable(tmp_path, shape):
    """With no comment to end it, the one line a CR file makes runs to the
    whole file, and the refusal quotes it. Its CRs are shown as ``\\r``, as a
    printed CR returns the cursor over the message, and it is cut short."""
    head, refusal = LARGE_CR_FILES[shape]
    params = "".join(f"    {i} k{i} {i}.5\n" for i in range(1, 2001))
    text = f"{head}begin parameters\n{params}end parameters\n"
    path = _write(tmp_path, text.replace("\n", "\r"))
    with pytest.raises(bngsim.ModelError, match=refusal) as caught:
        bngsim.Model.from_net(str(path))
    message = str(caught.value)
    assert "\r" not in message
    assert "k1 1.5" in message and " ..." in message
    assert CR_HINT in message
    assert len(message) < len(str(path)) + MESSAGE_ROOM


NOTHING_READ = {
    "empty": "",
    "blank-lines": "\n\n  \n",
    "comments": "# Created by BioNetGen 2.9.3\n# nothing else\n",
    # CRLF ends a line, and says nothing of a bare CR.
    "comments-crlf": "# Created by BioNetGen 2.9.3\r\n# nothing else\r\n",
    "actions": 'setOption("SpeciesLabel","HNauty")\nversion("2.9.3")\n',
    "model-markers": "begin model\nend model\n",
    "a-block-passed-over": "begin compartments\n  EC 3 1\nend compartments\n",
}


@pytest.mark.parametrize("case", sorted(NOTHING_READ))
def test_a_file_with_no_block_is_refused(tmp_path, case):
    """Each loaded as 0 species and 0 reactions. No bare CR, so no word of one."""
    path = _write(tmp_path, NOTHING_READ[case])
    with pytest.raises(bngsim.ModelError, match=NO_BLOCK) as caught:
        bngsim.Model.from_net(str(path))
    assert CR_HINT not in str(caught.value)
    with pytest.raises(ValueError, match=NO_BLOCK):
        bngsim.parse_net_file(str(path))


# ── Read as before ───────────────────────────────────────────────────────────

READ_AS_LF = {
    "crlf": lambda net: net.replace("\n", "\r\n"),
    "cr-cr-lf": lambda net: net.replace("\n", "\r\r\n"),
    "a-cr-opening-each-line": lambda net: net.replace("\n", "\n\r"),
    "a-cr-in-a-trailing-comment": lambda net: net.replace("#_R1\n", "#_R\r1\n"),
    "a-cr-between-tokens": lambda net: net.replace("1 1 2 kf", "1 1 2\rkf"),
    "a-cr-at-a-line-start": lambda net: net.replace("    1 A() 7", "\r    1 A() 7"),
    "a-cr-in-the-header-comment": lambda net: net.replace("BioNetGen 2.9.3", "BioNetGen\r2.9.3"),
    "a-cr-on-a-blank-line": lambda net: net.replace("end parameters\n", "end parameters\n\r\n"),
}


@pytest.mark.parametrize("edit", sorted(READ_AS_LF))
def test_a_cr_inside_a_line_is_a_blank(tmp_path, edit):
    """Control: loaded on main too. Each gives the LF file's trajectory, bit
    for bit, as both referees give its numbers."""
    lf = _a_at(bngsim.Model.from_net(str(_write(tmp_path, NET))))
    model = bngsim.Model.from_net(str(_write(tmp_path, READ_AS_LF[edit](NET))))
    assert (model.n_species, model.n_reactions) == (2, 2)
    np.testing.assert_array_equal(_a_at(model), lf)
    assert lf[0] == pytest.approx(2 + 5 * np.exp(-2.5 * 0.4), rel=CLOSED_FORM)


EMPTY_BLOCKS = {
    "parameters-only": ("begin parameters\n    1 kf 2\nend parameters\n", 0),
    "an-empty-parameters-block": ("begin parameters\nend parameters\n", 0),
    "species-and-an-empty-reactions-block": (
        "begin species\n    1 A() 7\nend species\nbegin reactions\nend reactions\n",
        1,
    ),
    # Every one of the five blocks counts, not only those that hold species:
    # a check that counted parameters, species and reactions passed every
    # other case. run_network refuses both ("Couldn't read rates array").
    "an-empty-functions-block": ("begin functions\nend functions\n", 0),
    "an-empty-groups-block": ("begin groups\nend groups\n", 0),
}


@pytest.mark.parametrize("case", sorted(EMPTY_BLOCKS))
def test_a_block_that_is_there_and_empty_still_loads(tmp_path, case):
    """Control: a block was read, so the file is a model, if an empty one."""
    text, n_species = EMPTY_BLOCKS[case]
    model = bngsim.Model.from_net(str(_write(tmp_path, text)))
    assert (model.n_species, model.n_reactions) == (n_species, 0)


def test_a_model_with_nothing_in_it_still_round_trips(tmp_path):
    """Control: ``write_net`` writes every block, empty or not, so a model of
    0 species and 0 reactions (an SBML time-only model, which ModelBuilder
    accepts on purpose) loads back from the file it writes."""
    from bngsim.convert import write_net

    model = bngsim.Model.from_sbml_string(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">\n'
        '  <model id="m">\n'
        '    <listOfParameters><parameter id="k" value="2" constant="true"/>'
        "</listOfParameters>\n"
        "  </model>\n"
        "</sbml>\n"
    )
    path = tmp_path / "rt.net"
    write_net(model, path)
    back = bngsim.Model.from_net(str(path))
    assert (back.n_species, back.n_reactions) == (0, 0)
    assert "k" in back.param_names
