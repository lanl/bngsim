"""Issue #1506 — ``not(k)^n`` is ``(not(k))^n``, compiled as it is interpreted.

ExprTk parses ``not`` as a one-argument call, so ``not(k)^n`` is
``(not(k))^n``. The C generator renamed ``not`` to C's prefix ``!`` before the
power pass, and that pass took only the bracket ``(k)`` as the base, leaving
the ``!`` outside: ``!pow(k, n)``. At ``k = n = 0`` the interpreter gives
``1^0 = 1`` and the compiled RHS ``!(0^0) = 0``, so a species fed at that rate
reached 0 instead of 10, with no warning. On the exponent side ``2^not(k)``
emitted ``pow(2.0, )!(p[0])``, which does not compile.

The two readings differ whenever ``k^n`` and ``k == 0`` do: at ``n = 0``, at
``n < 0``, and at ``n > 0`` when a nonzero ``k^n`` underflows to 0. With a 0/1
base and a positive exponent they agree, which is why nothing caught it.
``not(...)`` is now written whole, as ExprTk's ``notl_impl``, before the power
pass sees it.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _expr_to_c

NET = """begin parameters
    1 k 0
    2 n 0
    3 m -1
    4 x 2
    5 tiny 1e-200
    6 two 2
end parameters
begin species
    1 X() 0
end species
begin functions
    1 r() {law}
end functions
begin reactions
    1 0 1 r
end reactions
begin groups
    1 X_tot 1
end groups
"""

# (rate law, its value). `0 -> X` at r() from X = 0, so X(10) = 10*r().
LAWS = [
    ("not(k)^n", 1.0),  # 1^0, not !(0^0)
    ("not(k)^m", 1.0),  # 1^-1, not !(0^-1) = !inf
    ("(not(k))^n", 1.0),  # the parenthesised control, right before too
    ("not (k)^n", 1.0),
    ("not((k))^n", 1.0),
    ("x*not(k)^n", 2.0),
    ("not(not(k))^n", 1.0),  # 0^0
    ("not(k)^n^x", 1.0),  # 1^(0^2)
    ("if(not(k)^n, 1, 2)", 1.0),
    # 1e-200^2 underflows to 0, so not(tiny^2) is 1 and (not(tiny))^2 is 0.
    ("not(tiny)^two + 0.5", 0.5),
    # The exponent side, which did not compile.
    ("2^not(k)", 2.0),
    ("x^not(k) - 1", 1.0),
    ("2^not(x)", 1.0),
]


def _x10(tmp_path, law: str, codegen: bool) -> float:
    path = tmp_path / "m.net"
    path.write_text(NET.format(law=law))
    with contextlib.redirect_stderr(io.StringIO()):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", codegen=codegen)
        result = sim.run(t_span=(0.0, 10.0), n_points=11)
    return float(np.asarray(result.species)[-1, 0])


@pytest.mark.parametrize("law, value", LAWS, ids=[law for law, _ in LAWS])
def test_the_compiled_run_reaches_the_closed_form(tmp_path, law, value):
    interpreted = _x10(tmp_path, law, codegen=False)
    compiled = _x10(tmp_path, law, codegen=True)
    assert interpreted == pytest.approx(10.0 * value, rel=1e-12)
    assert compiled == interpreted


@pytest.mark.parametrize(
    "expr, expect",
    [
        ("not(k)^n", "pow((((p[0]) != 0.0) ? 0.0 : 1.0), p[1])"),
        ("2^not(k)", "pow(2.0, (((p[0]) != 0.0) ? 0.0 : 1.0))"),
        ("not(k)", "(((p[0]) != 0.0) ? 0.0 : 1.0)"),
    ],
)
def test_not_is_one_operand_on_either_side_of_a_power(expr, expect):
    assert _expr_to_c(expr, ["k", "n"], [], [], []) == expect
