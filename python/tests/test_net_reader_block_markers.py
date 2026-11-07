"""The ``.net`` reader finds a block's markers as BNG2.pl and run_network do
(issue #1498).

A hand-edited ``.net`` could load clean as a different network, with no error
and no warning. The reader found ``begin``/``end`` markers by a search of the
raw line for one spelling, ``begin reactions`` with a single blank, and passed
over any line it did not match:

* ``begin<TAB>reactions`` (or two blanks) opened no block, so its lines were
  passed over one by one and the model had 0 reactions: a flat trajectory,
  reported as success. run_network and BNG2.pl both read 2 reactions from it.
* a trailing comment that said ``end reactions`` ended the block: that reaction
  and every one after it were lost. Both referees read the line as a reaction.
* a negative species index in a reaction was filtered out with the null
  species ``0``, so ``C -> B`` loaded as ``0 -> B``. run_network refuses it.
* a block of a name the reader did not know (``begin reaction``) was passed
  over line by line, and loaded as nothing.

The markers are now read as BNG2.pl's ``readFile`` reads them
(Perl2/BNGModel.pm): the comment off, ``begin`` or ``end`` as the first word,
case-sensitive, and the name the words after it with their whitespace
collapsed. What BNG2.pl refuses is refused by name: an unknown block, another
block's ``end``, a ``begin`` inside a block, a block with no ``end``, a second
``begin model``, a block after ``end model``, and an index that is not a
species index. A data line outside every block is refused too, though BNG2.pl
only warns of it ("Unidentified input!") and run_network passes it over: it is
a block whose ``begin`` was not read, and BNG2.pl reads ``Begin reactions``
that way, as no reactions and exit code 0.

The referees' verdicts below were measured on run_network 3.0 (BNG 2.9.3) and
on BNG2.pl and run_network 2.9.2 (PyBioNetGen's bundle), and read from
BNGModel.pm's ``readFile`` and ``read_block_array``; the oracle for a loaded
network is the closed form of A -> B (k1 = 1) and C -> B (k2 = 0.5) from
A = 10, C = 2: B(2) = 12 - 10e^-2 - 2e^-1.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import Model, ModelError
from bngsim._net_reader import parse_net_file

NET = """begin parameters
    1 k1 1
    2 k2 0.5
end parameters
begin species
    1 A() 10
    2 B() 0
    3 C() 2
end species
begin reactions
    1 1 2 k1
    2 3 2 k2
end reactions
begin groups
    1 Btot 2
end groups
"""
B_AT_2 = 12.0 - 10.0 * math.exp(-2.0) - 2.0 * math.exp(-1.0)


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "m.net"
    path.write_text(text)
    return str(path)


def _b_at_2(path: str) -> float:
    model = Model.from_net(path)
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 2.0), n_points=2, rtol=1e-10, atol=1e-12
    )
    return float(np.asarray(result.species)[-1, list(result.species_names).index("B()")])


def _loads_the_network(path: str) -> None:
    """Both entry points read the two reactions, and the run is the closed form."""
    assert Model.from_net(path).n_reactions == 2
    assert len(parse_net_file(path)["reactions"]) == 2
    assert _b_at_2(path) == pytest.approx(B_AT_2, rel=1e-7)


def _refused(path: str, match: str) -> None:
    """Both entry points refuse the file, naming the line."""
    with pytest.raises(ModelError, match=match):
        Model.from_net(path)
    with pytest.raises(ValueError, match=match):
        parse_net_file(path)


class TestMarkerSpellings:
    """run_network reads 2 reactions from each; the reader read 0."""

    @pytest.mark.parametrize(
        "edit",
        [
            ("begin reactions", "begin\treactions"),
            ("begin reactions", "begin  reactions"),
            ("begin reactions", "begin reactions # the numeric block"),
            ("end reactions", "end\treactions"),
            ("end reactions", "end  reactions"),
            ("end reactions", "end reactions # done"),
            ("begin groups", "begin\tgroups"),
            ("begin species", "begin  species"),
        ],
        ids=lambda e: e[1].replace("\t", "<TAB>"),
    )
    def test_whitespace_and_a_comment_on_a_marker_are_read(self, tmp_path, edit):
        _loads_the_network(_write(tmp_path, NET.replace(*edit)))

    def test_the_file_as_written_is_unchanged(self, tmp_path):
        _loads_the_network(_write(tmp_path, NET))

    def test_an_explicitly_empty_reactions_block_still_loads(self, tmp_path):
        """Zero reactions as such is not refused: the block is there and empty."""
        net = _write(tmp_path, NET.replace("    1 1 2 k1\n    2 3 2 k2\n", ""))
        assert Model.from_net(net).n_reactions == 0
        assert parse_net_file(net)["reactions"] == []


class TestACommentIsNotAMarker:
    """A trailing comment naming the block's end ended it on the raw line, and
    the data line that carried it was lost with everything after it. BNG2.pl
    and run_network strip the comment first."""

    def test_a_reaction_line_whose_comment_says_end_reactions(self, tmp_path):
        """Read 1 reaction: B(2) = 8.65 for 9.91."""
        _loads_the_network(
            _write(tmp_path, NET.replace("    1 1 2 k1\n", "    1 1 2 k1 # see end reactions\n"))
        )

    def test_a_group_line_whose_comment_says_end_groups(self, tmp_path):
        """observables came back [] (the line itself was lost too)."""
        net = _write(
            tmp_path, NET.replace("    1 Btot 2\n", "    1 Btot 2 # end groups\n    2 Atot 1\n")
        )
        assert Model.from_net(net).observable_names == ["Btot", "Atot"]
        assert [name for name, _ in parse_net_file(net)["observables"]] == ["Btot", "Atot"]

    def test_a_parameter_line_whose_comment_says_end_parameters(self, tmp_path):
        """The block ended on k1, so k2 was 'unknown parameter k2' on reaction 2."""
        net = _write(tmp_path, NET.replace("    1 k1 1\n", "    1 k1 1 # end parameters\n"))
        assert Model.from_net(net).get_param("k2") == 0.5
        _loads_the_network(net)

    def test_a_species_line_whose_comment_says_end_species(self, tmp_path):
        net = _write(tmp_path, NET.replace("    1 A() 10\n", "    1 A() 10 # end species\n"))
        _loads_the_network(net)

    def test_a_function_line_whose_comment_says_end_functions(self, tmp_path):
        """An output-only function went silently; one a reaction reads failed
        the load against the wrong block."""
        net = _write(
            tmp_path,
            NET.replace(
                "begin reactions\n    1 1 2 k1\n",
                "begin functions\n    1 f() k1 # end functions\n    2 g() 2*k1\nend functions\n"
                "begin reactions\n    1 1 2 f\n",
            ),
        )
        _loads_the_network(net)

    def test_a_skipped_blocks_tab_end_marker_does_not_swallow_the_rest(self, tmp_path):
        """`end<TAB>observables` was not found by the skip, so every block after
        the observables went with them: 0 reactions, with no error."""
        net = _write(
            tmp_path,
            NET.replace(
                "begin reactions\n",
                "begin observables\n    1 Molecules Atot A()\nend\tobservables\nbegin reactions\n",
            ),
        )
        _loads_the_network(net)


class TestSpeciesIndices:
    """run_network refuses a negative index ("Reaction list not read because of
    errors"); the reader filtered it out with the null species."""

    @pytest.mark.parametrize(
        ("line", "match"),
        [
            ("    2 -3 2 k2\n", r"names reactant species '-3'"),
            ("    2 3 -2 k2\n", r"names product species '-2'"),
            ("    2 x 2 k2\n", r"names reactant species 'x'"),
            ("    2 3 2.0 k2\n", r"names product species '2.0'"),
            ("    2 1,-3 2 k2\n", r"names reactant species '-3'"),
            ("    2 9999999999 2 k2\n", r"names reactant species '9999999999'"),
        ],
        ids=[
            "negative reactant",
            "negative product",
            "a name",
            "a decimal",
            "second of two",
            "ten digits",
        ],
    )
    def test_an_index_that_is_no_species_index_is_refused_by_name(self, tmp_path, line, match):
        """-3 loaded as the null species: C -> B became 0 -> B, B(2) = 9.65."""
        _refused(_write(tmp_path, NET.replace("    2 3 2 k2\n", line)), match)

    def test_the_null_species_and_two_reactants_still_load(self, tmp_path):
        """Controls: `0` is the null species on either side, as run_network reads it."""
        net = _write(tmp_path, NET.replace("    2 3 2 k2\n", "    2 0 2 k2\n    3 1,3 0 k2\n"))
        assert Model.from_net(net).n_reactions == 3
        parsed = parse_net_file(net)["reactions"]
        assert len(parsed) == 3

    def test_a_plus_sign_and_a_leading_zero_are_the_species(self, tmp_path):
        """Controls: BNG2.pl and run_network read `+3` and `03` as species 3, and so
        did the reader before #1498."""
        _loads_the_network(_write(tmp_path, NET.replace("    2 3 2 k2\n", "    2 +3 02 k2\n")))


class TestUnknownBlocksAndStrayLines:
    """BNG2.pl refuses a block of a name it does not know ("Could not process
    block type") and warns of a line outside every block ("Unidentified
    input!"); the reader passed both over, and refuses both now."""

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            (
                NET.replace("begin reactions", "begin reaction").replace(
                    "end reactions", "end reaction"
                ),
                r"Could not process block type 'reaction'",
            ),
            (NET + "begin foo\n    1 x\nend foo\n", r"Could not process block type 'foo'"),
            (
                NET.replace("begin reactions", "Begin reactions"),
                r"outside any block: 'Begin reactions'",
            ),
            (NET + "1 A() 10\n", r"outside any block: '1 A\(\) 10'"),
            (NET + "end reactions\n", r"'end reactions' without a 'begin reactions'"),
        ],
        ids=["singular", "foo", "capital Begin", "data line", "stray end"],
    )
    def test_is_refused_by_name(self, tmp_path, text, match):
        """`begin reaction` loaded 0 reactions with no error."""
        _refused(_write(tmp_path, text), match)

    @pytest.mark.parametrize(
        "end", ["", ";", "; # a comment"], ids=["bare", "semicolon", "commented"]
    )
    def test_the_actions_bng2_writes_at_the_top_are_read_past(self, tmp_path, end):
        """BNG2.pl writes these with no `;` and reads them with one or without
        (readFile's action pattern ends `\\);?\\s*$`); run_network passes them over."""
        head = "".join(
            f"{action}{end}\n"
            for action in (
                'setOption("SpeciesLabel","HNauty")',
                'version("2.9.3")',
                'substanceUnits("Number")',
            )
        )
        _loads_the_network(_write(tmp_path, head + NET))

    @pytest.mark.parametrize(
        "text",
        [
            "begin model\n" + NET + "end model\n",
            "begin model\n" + NET,
            NET + "end model\n",
            NET + "end model\nbegin actions\n    writeNetwork({})\nend actions\n",
        ],
        ids=["begin and end", "begin only", "end only", "end, then actions"],
    )
    def test_a_begin_model_wrapper_is_read_past(self, tmp_path, text):
        """BNG2.pl reads a file as one model open from its first line, so it needs
        neither marker, and actions may follow `end model`; run_network reads
        each of these too."""
        _loads_the_network(_write(tmp_path, text))

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            (
                "begin model\n" + NET + "end model\nbegin model\nend model\n",
                r"a second 'begin model'",
            ),
            (
                NET.replace("begin groups", "end model\nbegin groups"),
                r"'begin groups' after 'end model'",
            ),
            (
                "begin model\n" + NET + "end model\nend model\n",
                r"'end model' after the model has ended",
            ),
        ],
        ids=["a second model", "a block after end model", "end model twice"],
    )
    def test_the_model_wrapper_bng2_refuses_is_refused(self, tmp_path, text, match):
        """BNG2.pl: "Only one model definition allowed per file", "groups cannot
        be defined outside of a model", "end model encountered without enclosing
        begin model"."""
        _refused(_write(tmp_path, text), match)

    def test_a_protocol_block_is_passed_over(self, tmp_path):
        """BNG2.pl reads `begin protocol` as it reads `begin actions`, inside a
        model or after it, and run_network passes it over."""
        protocol = 'begin protocol\n    simulate({method=>"ode",t_end=>1})\nend protocol\n'
        _loads_the_network(_write(tmp_path, NET + protocol))

    @pytest.mark.parametrize(
        "block",
        [
            "compartments",
            "molecule types",
            "observables",
            "energy patterns",
            "population types",
            "population maps",
            "reaction rules",
            "reactions_text",
            "actions",
        ],
    )
    def test_each_block_bng2_can_write_is_passed_over_whole(self, tmp_path, block):
        """With include_model=1 BNG2.pl writes these beside the five the reader
        reads. Their lines are no reactions, and `begin reactions_text` is not
        `begin reactions` (issue #13)."""
        extra = f"begin {block}\n    1 A -> B k1 # a line in {block}\nend {block}\n"
        _loads_the_network(
            _write(tmp_path, NET.replace("begin reactions\n", extra + "begin reactions\n"))
        )

    def test_seed_species_is_the_species_block(self, tmp_path):
        """BNG2.pl reads `seed species` as `species` (BNGModel.pm)."""
        net = _write(
            tmp_path,
            NET.replace("begin species", "begin seed species").replace(
                "end species", "end seed species"
            ),
        )
        _loads_the_network(net)


class TestBlockNesting:
    """What BNG2.pl's read_block_array refuses: another block's `end`, a `begin`
    before the `end`, and a block the file ends inside. run_network refuses the
    last ("Error in end command") and complains of the first."""

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            (
                NET.replace("    2 3 2 k2\n", "    2 3 2 k2\nend species\n"),
                r"'end species' does not match 'begin reactions'",
            ),
            (
                NET.replace("    2 3 2 k2\n", "begin groups\n    2 3 2 k2\n"),
                r"'begin groups' before the end of the reactions block",
            ),
            (
                NET.replace("end reactions\n", ""),
                r"'begin groups' before the end of the reactions block",
            ),
            (
                NET.replace("begin reactions\n", "begin observables\n    1 Molecules Atot A()\n"),
                r"'end reactions' does not match 'begin observables'",
            ),
            (
                NET.split("begin groups")[0].replace("end reactions\n", ""),
                r"'begin reactions' has no matching 'end reactions'",
            ),
            (
                NET.split("begin groups")[0] + "begin groups\n    1 Btot 2\n",
                r"'begin groups' has no matching 'end groups'",
            ),
            (
                NET.replace("begin reactions\n", "begin observables\n    1 Molecules Atot A()\n")
                .replace("end reactions\n", "")
                .split("begin groups")[0],
                r"'begin observables' has no matching 'end observables'",
            ),
            (
                NET.replace(
                    "begin reactions\n",
                    "begin observables\n    1 Molecules Atot A()\nbegin reactions\n"
                    "end observables\nbegin reactions\n",
                ),
                r"'begin reactions' before the end of the observables block",
            ),
        ],
        ids=[
            "end of another block",
            "begin inside",
            "end missing, next block begins",
            "a skipped block ended by another's end",
            "no end at the file's end",
            "no end of groups",
            "no end of a skipped block",
            "begin inside a skipped block",
        ],
    )
    def test_is_refused_by_name(self, tmp_path, text, match):
        _refused(_write(tmp_path, text), match)
