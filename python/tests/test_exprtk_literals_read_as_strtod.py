"""A numeric literal in an expression reads as strtod reads it, interpreted or
compiled (issue #1013).

ExprTk's own number parser is not correctly rounded: it sums the digits in a
double and scales by a power of ten, rounding at each step, where strtod rounds
once. Generated C is read by a C compiler, which rounds once, so the same model
computed with different numbers depending on whether codegen ran — one ulp apart
for about a quarter of literals written with a negative exponent, ordinary rate
constants among them (``4.82e-06``, ``6.57e-06``, even ``1.14``). At a threshold
one ulp is a whole step: ``round(0.49999999999999997)`` was 1 interpreted and 0
compiled. The vendored ExprTk header stays unmodified (test_exprtk_vendoring.py);
the host respells a literal ExprTk would misread before ExprTk sees it
(``expr_compat::exact_literals``).

The oracle is Python's ``float()``, which is correctly rounded, as strtod is.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import bngsim
import pytest

#: ExprTk read each of these one ulp off.
MISREAD = [
    "4.82e-06",
    "6.57e-06",
    "1.14",
    "4.89e-05",
    "8.77e-12",
    "9.87e+11",
    "7.77e-05",
    "0.49999999999999997",
    "2.71828182845904523536",
    "123456789012345678",
    # past 1e22 a power of ten is inexact, so these take the exact fallback,
    # dividing and multiplying by powers of two
    "1e-25",
    "1e-30",
    "3e23",
]
#: And these it read right, which must not move.
READ_RIGHT = ["0.1", "2.5", "3.45e-06", "6.02214076e23", "1e-3", "1e23"]


def _net(tmp_path: Path, params: list[str], functions: list[str]) -> str:
    lines = ["begin parameters"]
    lines += [f"  {i} {p}" for i, p in enumerate(["k 1", *params], 1)]
    lines += ["end parameters", "begin species", "  1 B() 0", "end species"]
    lines += ["begin functions"]
    lines += [f"  {i} f{i}() {body}" for i, body in enumerate(functions, 1)]
    lines += ["end functions", "begin reactions", "  1 0 1 k", "end reactions"]
    lines += ["begin groups", "  1 Btot 1", "end groups"]
    path = tmp_path / "m.net"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _function_values(net: str, n: int, codegen: bool) -> list[float]:
    model = bngsim.Model.from_net(net)
    with contextlib.redirect_stderr(io.StringIO()):
        sim = bngsim.Simulator(model, method="ode", codegen=codegen)
        res = sim.run(t_span=(0, 1e-9), n_points=2)
    return [float(res.expressions[f"f{i}"][0]) for i in range(1, n + 1)]


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
def test_every_literal_reads_as_strtod_on_both_paths(tmp_path, codegen):
    literals = MISREAD + READ_RIGHT
    got = _function_values(_net(tmp_path, [], literals), len(literals), codegen)
    assert dict(zip(literals, got, strict=True)) == {s: float(s) for s in literals}


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
def test_a_threshold_one_ulp_below_a_half(tmp_path, codegen):
    """0.49999999999999997 is the double below 0.5. Read as 0.5, round() went up."""
    bodies = ["round(0.49999999999999997)", "if(0.49999999999999997 < 0.5, 1, 0)"]
    assert _function_values(_net(tmp_path, [], bodies), 2, codegen) == [0.0, 1.0]


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreter", "codegen"])
def test_a_subnormal_literal_compiles(tmp_path, codegen):
    """ExprTk refuses an exponent below the normal range (ERR006), so the model
    did not load at all, codegen or not; strtod and C give the subnormal."""
    assert _function_values(_net(tmp_path, [], ["5e-324"]), 1, codegen) == [5e-324]


def test_a_derived_parameter_reads_its_literal_as_strtod(tmp_path):
    model = bngsim.Model.from_net(_net(tmp_path, ["q 4.82e-06*k"], ["q"]))
    assert model.get_param("q") == float("4.82e-06")


def test_a_literal_in_context_is_respelled_and_a_name_is_not(tmp_path):
    """Digits inside a name are the name's, not a number: 3e23 alone is misread
    and respelled, so k3e23 split apart would no longer be the parameter. A
    misread literal, in either respelling, next to a sign, a power or a call
    still reads right."""
    bodies = [
        "k3e23*2",
        "a_1e5+1",
        "-4.82e-06^1",
        "max(4.82e-06, 0)",
        "2*(1.14)",
        "-3e23^1",
        "2^-1e-30",
        "1e-30*k3e23",
    ]
    net = _net(tmp_path, ["k3e23 3", "a_1e5 5"], bodies)
    got = _function_values(net, len(bodies), codegen=False)
    assert got == [
        6.0,
        6.0,
        -float("4.82e-06"),
        float("4.82e-06"),
        2 * float("1.14"),
        -float("3e23"),
        2.0 ** -float("1e-30"),
        float("1e-30") * 3,
    ]
