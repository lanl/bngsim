"""Regression tests for issue #606: a `.net` functions line is
`<index> <name>() <expression>`, and a line that is not one must fail rather than
be dropped or read under the wrong name.

The functions block was the last of the five `.net` block parsers still
discarding or mis-reading a line it did not recognise — and it did both:

* `rate_fn() (kcat*Atot)/(10+Atot)` (unindexed) has two fields, so it fell
  through the `< 3` gate and vanished silently, exactly as the reactions block
  did before #603; and
* `rate_fn() = (kcat*Atot)/(10+Atot)` — the form `writeFile({format=>"net"})`
  writes — has three, so it passed the gate and the name was read from the fixed
  second field, which is the `=`. bngsim invented a function literally called
  `=`, carrying the right expression under the wrong name.

Either way the C++ loader then failed against the *wrong* block
(``reaction 0 (Elementary) references unknown parameter 'rate_fn'`` — the
reaction was fine), and `parse_net_file` did not fail at all: it handed the
caller a wrong `functions` list with no error and no warning.

`run_network` 3.0 refuses both shapes with `ERROR: Found invalid line
"rate_fn()" while reading functions block`, so there is nothing here to
*support*. The functions block is the one block where BNG2.pl's two writers
differ in the separator and not just in the leading index, and `writeFile`'s
functions block is not a network any BNG reader loads. This is about failing
correctly, not about accepting more.

Both documented `.net` loaders are covered, since #554 requires them to agree.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import bngsim
import pytest
from bngsim import Model
from bngsim._exceptions import ModelError
from bngsim._net_reader import parse_net_file


def _write(tmp_path: Path, functions_line: str, name: str = "model.net") -> Path:
    """The issue's reproduction network, with one functions line substituted in."""
    p = tmp_path / name
    p.write_text(
        textwrap.dedent(
            f"""
            begin parameters
                1 kcat 0.3
            end parameters
            begin species
                1 A() 100
                2 B() 0
            end species
            begin functions
                {functions_line}
            end functions
            begin reactions
                1 1 2 rate_fn
            end reactions
            begin groups
                1 Atot 1
            end groups
            """
        ).strip()
        + "\n"
    )
    return p


def _both_loaders_refuse(net: Path, message: str) -> None:
    """#554: the two documented loaders must refuse the same file the same way."""
    with pytest.raises(ModelError, match=message):
        Model.from_net(net)
    with pytest.raises(ValueError, match=message):
        parse_net_file(net)


class TestUnindexedFunctionsLineIsRefused:
    """`run_network` refuses a functions line with no leading index; so must bngsim.

    This is the opposite fix from species, parameters and groups, where the index
    *is* optional (#601, #603) — and the same fix as reactions, for the same
    reason: matching BNG here means refusing the unindexed form, not reading it.
    """

    def test_unindexed_line_is_refused_not_dropped(self, tmp_path: Path) -> None:
        net = _write(tmp_path, "rate_fn() (kcat*Atot)/(10+Atot)")
        # `functions == []`, silently, before the fix — and then a C++ load that
        # blamed the reactions block for the missing `rate_fn`.
        _both_loaders_refuse(net, "needs a leading index")

    def test_writefile_equals_form_is_refused_not_renamed(self, tmp_path: Path) -> None:
        """`writeFile({format=>"net"})`'s own output, which run_network also refuses."""
        net = _write(tmp_path, "rate_fn() = (kcat*Atot)/(10+Atot)")
        _both_loaders_refuse(net, "needs a leading index")
        # Before the fix this was `[('=', '(kcat*Atot)/(10+Atot)')]`: the right
        # expression under a function named `=`.
        with pytest.raises(ValueError):
            parse_net_file(net)


class TestEqualsSeparatorWithAnIndex:
    """The `=` separator is refused wherever it sits, not only when it is field 2.

    An index in front of the `writeFile` shape — what adding indices to that file
    by hand produces — used to keep the `=` as the expression's first character
    (`'= (kcat*Atot)/(10+Atot)'`) in the Python reader and reach ExprTk in the
    C++ one, which failed with a bare `ERR248` that never named the line.
    """

    def test_spaced_equals_is_refused(self, tmp_path: Path) -> None:
        net = _write(tmp_path, "1 rate_fn() = (kcat*Atot)/(10+Atot)")
        _both_loaders_refuse(net, "separates its name and expression with '='")

    def test_unspaced_equals_is_refused(self, tmp_path: Path) -> None:
        """`1 fA()=Atot*2` is two fields, so it used to be dropped like an unindexed line."""
        net = _write(tmp_path, "1 rate_fn()=(kcat*Atot)/(10+Atot)")
        _both_loaders_refuse(net, "where a function name belongs")


class TestIncompleteFunctionsLine:
    def test_index_with_no_name(self, tmp_path: Path) -> None:
        net = _write(tmp_path, "1")
        _both_loaders_refuse(net, "has an index but no function name")

    def test_name_with_no_expression(self, tmp_path: Path) -> None:
        """run_network refuses this too (muParser: "Expression is empty")."""
        net = _write(tmp_path, "1 rate_fn()")
        _both_loaders_refuse(net, "has an index and a name but no expression")

    def test_function_with_arguments_is_refused(self, tmp_path: Path) -> None:
        """run_network: "Functions cannot contain arguments ('rate_fn(x)')"."""
        net = _write(tmp_path, "1 rate_fn(x) (kcat*Atot)/(10+Atot)")
        _both_loaders_refuse(net, "where a function name belongs")


class TestGenerateNetworkShapeStillLoads:
    """The corpus shape must be untouched: all 13,690 functions lines in the tree
    are `generate_network`'s `<index> <name>() <expression>`, indexed and `=`-free.
    """

    def test_indexed_function_drives_the_rate(self, tmp_path: Path) -> None:
        """A parameter-only function, so `run_network` can be the reference.

        `run_network -o par par.net 1 1` on this network reports
        A(1) = 5.488116231968e+01, B(1) = 4.511883768032e+01 — the function is
        the rate constant (0.6) and BNG multiplies by [A].
        """
        net = _write(tmp_path, "1 rate_fn() kcat*2")
        model = Model.from_net(net)
        assert model.function_names == ["rate_fn"]
        result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 1.0), n_points=2)
        assert result.species[-1, 0] == pytest.approx(5.488116231968e01, rel=1e-8)
        assert result.species[-1, 1] == pytest.approx(4.511883768032e01, rel=1e-8)

    def test_observable_referencing_function_still_loads(self, tmp_path: Path) -> None:
        """The issue's own expression, which bngsim reads and run_network cannot."""
        net = _write(tmp_path, "1 rate_fn() (kcat*Atot)/(10+Atot)")
        assert parse_net_file(net)["functions"] == [("rate_fn", "(kcat*Atot)/(10+Atot)")]
        assert Model.from_net(net).function_names == ["rate_fn"]

    def test_trailing_comment_is_still_stripped(self, tmp_path: Path) -> None:
        net = _write(tmp_path, "1 rate_fn() kcat*2  # the rate")
        assert parse_net_file(net)["functions"] == [("rate_fn", "kcat*2")]
        assert Model.from_net(net).function_names == ["rate_fn"]

    def test_name_without_the_empty_argument_list_still_reads(self, tmp_path: Path) -> None:
        """A deliberate divergence, kept: run_network refuses `1 rate_fn kcat*2`
        ("Functions cannot contain arguments ('rate_fn')"), but the name and the
        expression are unambiguous, and bngsim has always read this. The shape
        check is here to stop a *mis*-read, not to narrow what loads.
        """
        net = _write(tmp_path, "1 rate_fn kcat*2")
        assert parse_net_file(net)["functions"] == [("rate_fn", "kcat*2")]
        assert Model.from_net(net).function_names == ["rate_fn"]
