"""GH #498 — a .net parameter expression keeps the whitespace it was written with.

The ``.net`` reader splits a parameter line on whitespace and then joined the
value tokens back together with **nothing** between them, so any space that
separated two word characters inside the expression was deleted and the two
tokens merged. ``if(k > 0.1 and thr > 0.5, k, 0.0)`` came back as
``if(k>0.1andthr>0.5,k,0.0)``. ExprTk parses ``0.1andthr`` happily, so the
parameter took a wrong value that was finite, with no warning and no error.

The functions block in the same file has always joined with ``" "``, so the two
parsers disagreed about the same expression text. The parameters block now
matches it.

The property pinned here is that two spellings of one rate law, differing only in
whitespace, read back to the same number. Whitespace outside a string literal
carries no meaning in ExprTk, so any model where it changes the answer is a
parser bug by definition.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest

# `x` is the parameter under test. `k` and `thr` exist only to give the guard
# something to reference, so the merge has two word characters to weld together.
_NET = """begin parameters
    1 k   0.3
    2 thr 1.0
    3 x   {expr}
end parameters
begin species
    1 S() 10.0
end species
begin reactions
    1 1 0 x
end reactions
begin groups
    1 S_tot 1
end groups
"""


def _x(tmp_path: Path, expr: str) -> dict:
    """Load a .net whose third parameter is `expr` and return that parameter."""
    net = tmp_path / "m.net"
    net.write_text(_NET.format(expr=expr))
    return bngsim.Model.from_net(net)._core.codegen_data()["parameters"][2]


@pytest.mark.parametrize(
    "expr",
    [
        # The issue's rate law. `0.1`, `and` and `thr` are three separate tokens.
        "if(k > 0.1 and thr > 0.5, k, 0.0)",
        # `or` and `not` weld the same way.
        "if(k > 0.1 or thr > 9.0, k, 0.0)",
        # Spaces around an operator, no word-character welding, but the
        # expression text should still survive intact.
        "k * 1.0",
    ],
)
def test_spacing_is_preserved(tmp_path: Path, expr: str) -> None:
    assert _x(tmp_path, expr)["expression"] == expr


def test_spaced_and_parenthesized_guards_agree(tmp_path: Path) -> None:
    """One rate law, two spellings, one value.

    The parenthesized form never had spaces between word characters, so it was
    always read correctly. It is the control: before the fix the spaced form
    evaluated to 0.0 against its 0.3.
    """
    spaced = _x(tmp_path, "if(k > 0.1 and thr > 0.5, k, 0.0)")
    parenthesized = _x(tmp_path, "if((k>0.1)and(thr>0.5),k,0.0)")

    assert spaced["value"] == pytest.approx(0.3)
    assert spaced["value"] == pytest.approx(parenthesized["value"])


def test_a_plain_number_is_still_a_number(tmp_path: Path) -> None:
    """The join change must not turn a constant into a derived parameter."""
    p = _x(tmp_path, "0.25")
    assert p["value"] == pytest.approx(0.25)
    assert p["is_const"]
