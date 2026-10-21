"""Issue #1127 — a rate law written with ``%`` is read as the ``mod()`` it is.

ExprTk's ``%`` on doubles is ``std::fmod``, which is exactly what its ``mod``
function computes. But the analysis of a rate law reads named calls only: the
crossing stops (#869) and state-switch roots (#1068) look for ``mod(``, and
codegen renames ``mod(`` to C's ``fmod(``. So a narrow pulse written with ``%``
got no stop and no root and was stepped over (Y = 0, no error), and
``codegen=True`` failed to compile (``%`` on two doubles is not C).

The model builder now spells every ``a % b`` in a function body as
``mod(a, b)``. ``%`` binds like ``*`` and ``/``, left to right, which the
value tests check against ExprTk itself: a ``.net`` derived parameter is
evaluated from its text as written, so each spelling is computed both ways
and the two must agree to the bit.
"""

from __future__ import annotations

import contextlib
import io
import math

import bngsim
import numpy as np
import pytest

# The #1068 model: a counter T at k_clock, and X -> Y at a pulse of height 100
# and width 0.01 in T at T = 50, so Y ends at 10 (1 - exp(-1/k_clock)).
PULSE_NET = """begin parameters
    1 k_clock {k}
    2 kx 1
end parameters
begin species
    1 T() 0
    2 X() 10
    3 Y() 0
end species
begin functions
    1 pulse() {pulse}
    2 r2() kx*pulse()
end functions
begin reactions
    1 0 1 k_clock
    2 2 3 r2
end reactions
begin groups
    1 Tobs 1
end groups
"""

PERCENT = "((Tobs+49.99)%100)-((Tobs+50)%100)+0.01"
MOD = "mod(Tobs+49.99,100)-mod(Tobs+50,100)+0.01"


def _load(tmp_path, text: str, name: str = "m.net"):
    p = tmp_path / name
    p.write_text(text)
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Model.from_net(str(p))


def _pulse_y(tmp_path, pulse: str, k: float, **sim) -> float:
    m = _load(tmp_path, PULSE_NET.format(k=k, pulse=pulse))
    with contextlib.redirect_stderr(io.StringIO()):
        r = bngsim.Simulator(m, method="ode", **sim).run(t_span=(0, 60 / k), n_points=7)
    return float(np.asarray(r.species)[-1, list(r.species_names).index("Y()")])


@pytest.mark.parametrize("k", [2.0, 1.0], ids=["counter", "clock"])
def test_a_percent_pulse_is_not_stepped_over(tmp_path, k):
    """On a counter it is rooted (#1068), on a unit-rate clock stopped on (#869)."""
    y = _pulse_y(tmp_path, PERCENT, k)
    assert y == pytest.approx(10 * (1 - math.exp(-1 / k)), rel=1e-5)
    assert y == _pulse_y(tmp_path, MOD, k)


@pytest.mark.parametrize(
    "written, read",
    [
        ("A%3", "mod(A, 3)"),
        ("kx*A%3/2", "mod(kx*A, 3)/2"),
        ("A%3%2", "mod(mod(A, 3), 2)"),
        ("1 + A % 3", "1 + mod(A, 3)"),
        ("A^2%3", "mod(A^2, 3)"),
        ("if(A%2<1,A%3,0)", "if(mod(A, 2)<1,mod(A, 3),0)"),
        ("kx*A", "kx*A"),
    ],
)
def test_the_function_text_is_spelled_with_mod(tmp_path, written, read):
    net = PULSE_NET.format(k=1, pulse=written.replace("A", "Tobs"))
    m = _load(tmp_path, net)
    assert m._core.function_expressions[0] == read.replace("A", "Tobs")


# Precedence, chaining, signs, a negative divisor, a non-integer modulus, and
# % inside comparisons, if() and a call's arguments.
VALUE_CASES = [
    "7.5%2",
    "-7.5%2",
    "2*7.5%4/2",
    "7.5%4%3",
    "2^3%5",
    "7%-3",
    "1e-1%3e-2",
    "(1+2)%2+1",
    "3-7.5%2*2",
    "7.5%2>1",
    "1 + -7.5 % 2",
    "7.5 % 2 * 3 % 2",
    "if(7.5%2>1, 9%4, 0)",
    "max(7.5%2, 9%4)",
    "7.5%(1+1)",
    "-(7.5%2)^2",
]


def test_each_spelling_computes_what_exprtk_computes_for_percent(tmp_path):
    params = "\n".join(f"    {i + 2} p{i} {c}" for i, c in enumerate(VALUE_CASES))
    funcs = "\n".join(f"    {i + 1} f{i}() {c}" for i, c in enumerate(VALUE_CASES))
    net = (
        f"begin parameters\n    1 k 1\n{params}\nend parameters\n"
        f"begin functions\n{funcs}\nend functions\n"
        "begin species\n    1 A() 1\nend species\n"
        "begin reactions\n    1 1 0 k\nend reactions\n"
        "begin groups\n    1 Aobs 1\nend groups\n"
    )
    m = _load(tmp_path, net)
    values = m._core._eval_functions(0.0, [1.0])
    for i, case in enumerate(VALUE_CASES):
        assert "%" not in m._core.function_expressions[i], case
        assert values[f"f{i}"] == m.get_param(f"p{i}"), case


def test_the_compiled_path_matches_the_interpreter(tmp_path):
    """Before the fix ``codegen=True`` failed: ``%`` on two doubles is not C."""
    net = PULSE_NET.format(k=1, pulse="(Tobs%3)")
    runs = []
    for codegen in (False, True):
        m = _load(tmp_path, net)
        with contextlib.redirect_stderr(io.StringIO()):
            r = bngsim.Simulator(m, method="ode", codegen=codegen).run(t_span=(0, 5), n_points=11)
        runs.append(np.asarray(r.species))
    assert runs[1] == pytest.approx(runs[0], rel=0, abs=0)
    assert runs[0][-1, 2] > 0  # the model moved, so the equality is not two flat lines


def test_a_model_with_its_own_mod_keeps_its_text(tmp_path):
    """The evaluator renames a model's own ``mod`` symbol, which would capture an
    inserted ``mod(`` call, so such a model is left as written."""
    net = (
        "begin parameters\n    1 mod 2\nend parameters\n"
        "begin functions\n    1 law() Aobs%mod\nend functions\n"
        "begin species\n    1 A() 5\nend species\n"
        "begin reactions\n    1 1 0 law\nend reactions\n"
        "begin groups\n    1 Aobs 1\nend groups\n"
    )
    m = _load(tmp_path, net)
    assert m._core.function_expressions == ["Aobs%mod"]
    assert m._core._eval_functions(0.0, [5.0])["law"] == 1.0
