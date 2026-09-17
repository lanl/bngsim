"""Regression tests for issue #600: the leading index on a `.net` species line is
optional, and no species line may be dropped silently.

`parse_species()` required three whitespace fields, which meant it required the
leading index. BNG2.pl treats the index as optional — `Perl2/SpeciesList.pm`
strips one with `s/^\\s*\\d+\\s+//`, kept with the comment "Can't deprecate this
because indices used in NET files" — and BNG2.pl writes *both* shapes into a file
named `.net`: `generate_network` emits `1 A() A0`, while
`writeFile({format=>"net"})` emits a bare `A() A0`. `run_network` reads either and
reports A(0) = 100 for both.

bngsim skipped the unindexed form line by line, with no error and no warning:

* with reactions present the model failed against the *wrong* block
  (``reaction 0 has reactant species index 1 out of range [1, 0]``), and
* with no reactions it loaded clean as a zero-species model.

Both `.net` loaders are covered, since #554 requires them to agree.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from bngsim import Model
from bngsim._exceptions import ModelError
from bngsim._net_reader import parse_net_file

# The same network, written the two ways BNG2.pl writes it. `run_network` gives
# A(0) = 100, B(0) = 0 for both.
_INDEXED = """
    begin parameters
        1 k1 0.5
        2 A0 100
    end parameters
    begin species
        1 A() A0
        2 B() 0
    end species
    begin reactions
        1 1 2 k1
    end reactions
    begin groups
        1 Atot 1
        2 Btot 2
    end groups
"""

_UNINDEXED = """
    begin parameters
        1 k1 0.5
        2 A0 100
    end parameters
    begin species
        A() A0
        B() 0
    end species
    begin reactions
        1 1 2 k1
    end reactions
    begin groups
        1 Atot 1
        2 Btot 2
    end groups
"""


def _write(tmp_path: Path, body: str, name: str = "model.net") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).strip() + "\n")
    return p


class TestOptionalLeadingIndex:
    @pytest.mark.parametrize("body", [_INDEXED, _UNINDEXED], ids=["indexed", "unindexed"])
    def test_both_shapes_load_the_same_state(self, tmp_path: Path, body: str) -> None:
        """Both forms are valid `.net`; both must seed A=100, B=0 (as run_network does)."""
        model = Model.from_net(_write(tmp_path, body))
        assert list(model._core.get_initial_state()) == [100.0, 0.0]

    @pytest.mark.parametrize("body", [_INDEXED, _UNINDEXED], ids=["indexed", "unindexed"])
    def test_both_shapes_agree_in_the_python_reader(self, tmp_path: Path, body: str) -> None:
        """#554: the Python reader must not disagree with the C++ loader here."""
        parsed = parse_net_file(_write(tmp_path, body))
        assert parsed["species"] == [("A()", 100.0, False), ("B()", 0.0, False)]
        assert parsed["species_ic_params"] == [(0, "A0")]

    def test_unindexed_block_is_not_silently_emptied(self, tmp_path: Path) -> None:
        """The worst shape of the defect: no reactions, so nothing downstream objects.

        This used to load as a zero-species model with no error and no warning.
        """
        net = _write(
            tmp_path,
            """
            begin parameters
              1 A0 100
            end parameters
            begin species
              A() A0
              B() 0
            end species
            begin reactions
            end reactions
            begin groups
            end groups
            """,
        )
        model = Model.from_net(net)
        assert list(model._core.get_initial_state()) == [100.0, 0.0]
        assert parse_net_file(net)["species"] == [("A()", 100.0, False), ("B()", 0.0, False)]

    def test_fixed_marker_still_recognized_without_an_index(self, tmp_path: Path) -> None:
        """`$` clamping (issue #41) is read off the species field, wherever it sits.

        Covers the bare `$A()` form and the cBNGL `@CP::$Sink()` form, which sits
        after a compartment prefix — both must survive the index becoming optional.
        """
        net = _write(
            tmp_path,
            """
            begin parameters
              1 A0 100
            end parameters
            begin species
              $A() A0
              @CP::$Sink() 0
              B() 7
            end species
            begin reactions
            end reactions
            begin groups
            end groups
            """,
        )
        assert parse_net_file(net)["species"] == [
            ("A()", 100.0, True),
            ("@CP::Sink()", 0.0, True),
            ("B()", 7.0, False),
        ]
        assert list(Model.from_net(net)._core.get_initial_state()) == [100.0, 0.0, 7.0]

    def test_species_with_no_concentration_is_zero(self, tmp_path: Path) -> None:
        """BNG2.pl reads a concentration only if text remains, so an omitted one is 0."""
        parsed = parse_net_file(
            _write(
                tmp_path,
                """
                begin parameters
                  1 A0 100
                end parameters
                begin species
                  1 A()
                  2 B() 5
                end species
                begin reactions
                end reactions
                begin groups
                end groups
                """,
            )
        )
        assert parsed["species"] == [("A()", 0.0, False), ("B()", 5.0, False)]

    def test_index_with_no_species_pattern_is_refused(self, tmp_path: Path) -> None:
        """A line that carries only an index has nothing to parse — say so, don't skip."""
        net = _write(
            tmp_path,
            """
            begin parameters
              1 A0 100
            end parameters
            begin species
              1
            end species
            begin reactions
            end reactions
            begin groups
            end groups
            """,
        )
        with pytest.raises(ModelError, match="index but no species pattern"):
            Model.from_net(net)
        with pytest.raises(ValueError, match="index but no species pattern"):
            parse_net_file(net)


class TestOptionalIndexInParametersBlock:
    """The parameters block carries the same optional index, for the same reason.

    `writeFile({format=>"net"})` writes `A0 100`; `generate_network` writes
    `1 A0 100`. Requiring the index discarded the whole parameters block
    silently, so every species IC and rate law naming a parameter then failed to
    resolve against a block that had been thrown away.
    """

    def test_unindexed_parameters_resolve(self, tmp_path: Path) -> None:
        net = _write(
            tmp_path,
            """
            begin parameters
              k1 0.5
              A0 100
            end parameters
            begin species
              A() A0
              B() 0
            end species
            begin reactions
              1 1 2 k1
            end reactions
            begin groups
              1 Atot 1
            end groups
            """,
        )
        model = Model.from_net(net)
        assert model.get_param("A0") == 100.0
        assert list(model._core.get_initial_state()) == [100.0, 0.0]
        assert parse_net_file(net)["species"] == [("A()", 100.0, False), ("B()", 0.0, False)]

    def test_parameter_line_without_a_value_is_refused(self, tmp_path: Path) -> None:
        net = _write(
            tmp_path,
            """
            begin parameters
              A0
            end parameters
            begin species
              1 A() 1
            end species
            begin reactions
            end reactions
            begin groups
            end groups
            """,
        )
        with pytest.raises(ModelError, match="no name and value to read"):
            Model.from_net(net)
        with pytest.raises(ValueError, match="no name and value to read"):
            parse_net_file(net)


class TestWriteFileNetFormat:
    """The real shape BNG2.pl's `writeFile({format=>"net"})` produces.

    Mixed on purpose: parameters and species unindexed (model style), reactions
    and groups indexed (network style) — which is exactly what BNG2.pl 2.9.3
    emits. Expression seed species arrive as `_InitialConc<N>` parameters, so the
    ICs are 2*A0 = 200, A0/f = 50 and (frac*A0)+10 = 35. `run_network` on this
    same file reports 200 / 50 / 35; this pins that bngsim agrees.
    """

    _WRITEFILE = """
        begin parameters
          A0             100
          f              2
          frac           0.25
          _InitialConc1  2*A0
          _InitialConc2  A0/f
          _InitialConc3  (frac*A0)+10
          _rateLaw1      1.0
        end parameters
        begin species
          A() _InitialConc1
          B() _InitialConc2
          C() _InitialConc3
        end species
        begin reactions
            1 1 2 _rateLaw1
        end reactions
        begin groups
            1 Atot                 1
            2 Btot                 2
            3 Ctot                 3
        end groups
    """

    def test_cpp_loader_matches_run_network(self, tmp_path: Path) -> None:
        model = Model.from_net(_write(tmp_path, self._WRITEFILE))
        assert list(model._core.get_initial_state()) == [200.0, 50.0, 35.0]

    def test_python_reader_agrees(self, tmp_path: Path) -> None:
        """#554: both documented loaders must seed from the same numbers."""
        parsed = parse_net_file(_write(tmp_path, self._WRITEFILE))
        assert [conc for _name, conc, _fixed in parsed["species"]] == [200.0, 50.0, 35.0]


class TestGroupsBlockOptionalIndex:
    """`run_network` reads an unindexed groups block; bngsim refused the file.

    Taking the name from a fixed second field read the *entry* as the name, so
    `Atot 1` became an observable called "1" with no entries. The load then died
    on a duplicate-symbol error that blamed the `.net` for names it did not
    contain — loud, but pointing at the wrong thing entirely.
    """

    def test_unindexed_group_keeps_its_name_and_entries(self, tmp_path: Path) -> None:
        net = _write(
            tmp_path,
            """
            begin parameters
              1 k1 0.5
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 k1
            end reactions
            begin groups
              Atot 1
              Btot 2
            end groups
            """,
        )
        # run_network on this file reports Atot = 100, Btot = 0.
        model = Model.from_net(net)
        assert model.observable_names == ["Atot", "Btot"]
        assert parse_net_file(net)["observables"] == [("Atot", [(0, 1.0)]), ("Btot", [(1, 1.0)])]

    def test_observable_matching_no_species_still_has_no_entries(self, tmp_path: Path) -> None:
        """BNG2.pl writes a bare `<index> <name>` when a pattern matches nothing.

        55 such lines exist in the repo's `.net` corpus, so the entries column
        must stay optional in both the indexed and unindexed forms.
        """
        for groups in ("  1 Atot 1\n  2 Ghost", "  Atot 1\n  Ghost"):
            net = _write(
                tmp_path,
                f"""
                begin parameters
                  1 k1 0.5
                end parameters
                begin species
                  1 A() 100
                  2 B() 0
                end species
                begin reactions
                  1 1 2 k1
                end reactions
                begin groups
                {groups}
                end groups
                """,
                name=f"m{len(groups)}.net",
            )
            assert parse_net_file(net)["observables"] == [("Atot", [(0, 1.0)]), ("Ghost", [])]

    def test_index_with_no_observable_name_is_refused(self, tmp_path: Path) -> None:
        net = _write(
            tmp_path,
            """
            begin parameters
              1 k1 0.5
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 k1
            end reactions
            begin groups
              1
            end groups
            """,
        )
        with pytest.raises(ModelError, match="index but no observable name"):
            Model.from_net(net)
        with pytest.raises(ValueError, match="index but no observable name"):
            parse_net_file(net)


class TestReactionsBlockRefusesShortLines:
    """A reaction line's index is NOT optional — and a short line must not vanish.

    `run_network` refuses a reactions block written without indices ("Reaction
    list not read because of errors. ERROR: No reactions in the network."), so
    bngsim should refuse too. It instead skipped each such line and built a model
    with *zero reactions*, which loaded clean and integrated a flat trajectory
    while reporting success.
    """

    _UNINDEXED_REACTIONS = """
        begin parameters
          1 k1 0.5
        end parameters
        begin species
          1 A() 100
          2 B() 0
        end species
        begin reactions
          1 2 k1
        end reactions
        begin groups
          1 Atot 1
        end groups
    """

    def test_unindexed_reaction_line_is_refused_not_dropped(self, tmp_path: Path) -> None:
        net = _write(tmp_path, self._UNINDEXED_REACTIONS)
        with pytest.raises(ModelError, match="needs an index, reactants, products"):
            Model.from_net(net)
        with pytest.raises(ValueError, match="needs an index, reactants, products"):
            parse_net_file(net)

    def test_indexed_reactions_still_load(self, tmp_path: Path) -> None:
        """The corpus is entirely indexed; that path must be untouched."""
        model = Model.from_net(_write(tmp_path, _INDEXED))
        assert model.n_reactions == 1


class TestSpacedConcentration:
    """A concentration spread over several tokens is kept whole, not truncated.

    `1 A() 2 * A0` used to read only the token after the species field, so the
    initial condition became 2.0 and `* A0` was discarded silently. The remainder
    is now joined the way the parameters block joins its value tokens (#498), so
    the whole text is what gets resolved — today that is refused as an
    unresolvable token (#571); evaluating it is tracked in #600.
    """

    def test_spaced_expression_is_not_truncated_to_its_first_token(self, tmp_path: Path) -> None:
        net = _write(
            tmp_path,
            """
            begin parameters
              1 k1 0.5
              2 A0 100
            end parameters
            begin species
              1 A() 2 * A0
              2 B() 0
            end species
            begin reactions
              1 1 2 k1
            end reactions
            begin groups
              1 Atot 1
            end groups
            """,
        )
        # The point is that 2.0 is NOT silently accepted; the whole text is what
        # gets resolved, and it appears in the error.
        with pytest.raises(ModelError, match=r"2 \* A0"):
            Model.from_net(net)
        with pytest.raises(ValueError, match=r"2 \* A0"):
            parse_net_file(net)
