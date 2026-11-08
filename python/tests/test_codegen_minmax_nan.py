"""GH #660: compiled ``max``/``min`` mean what ExprTk's do.

ExprTk's ``max``/``min`` are ``std::max``/``std::min``: ``(a < b) ? b : a``,
which returns the FIRST argument whenever the comparison is false, so a NaN in
first position wins. Codegen used to rename them to C's ``fmax``/``fmin``,
which return the non-NaN argument. A rate law the interpreter refuses as a NaN
right-hand side then ran to completion compiled, with plausible numbers.
"""

from __future__ import annotations

import os

import bngsim
import numpy as np
import pytest

_NET = """begin parameters
    1 k       0.3  # Constant
end parameters
begin functions
    1 law() {body}
end functions
begin species
    1 A() 5.0
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 A                    1
end groups
"""


def _run(tmp_path, body: str, codegen: bool):
    path = os.path.join(tmp_path, f"m_{abs(hash((body, codegen)))}.net")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_NET.format(body=body))
    m = bngsim.Model.from_net(path)
    sim = bngsim.Simulator(m, method="ode", codegen=codegen)
    return np.asarray(sim.run(t_span=(0.0, 2.0), n_points=3, timeout=60.0).observables["A"])


# ln(A - 10) is NaN for the whole run (A starts at 5 and decays).
@pytest.mark.parametrize("body", ["k*max(ln(A-10),1)", "k*min(ln(A-10),1)"])
def test_nan_first_argument_is_refused_by_both_engines(tmp_path, body):
    with pytest.raises(bngsim.SimulationError):
        _run(tmp_path, body, codegen=False)
    # fmax/fmin returned 1 here, so the compiled run used to complete.
    with pytest.raises(bngsim.SimulationError):
        _run(tmp_path, body, codegen=True)


@pytest.mark.parametrize("body", ["k*max(1,ln(A-10))", "k*min(1,ln(A-10))"])
def test_nan_second_argument_agrees_across_engines(tmp_path, body):
    interp = _run(tmp_path, body, codegen=False)
    compiled = _run(tmp_path, body, codegen=True)
    np.testing.assert_allclose(compiled, interp, rtol=1e-6)
    np.testing.assert_allclose(interp, 5.0 * np.exp(-0.3 * np.array([0.0, 1.0, 2.0])), rtol=1e-5)


# ── Issue #1507: a variadic call groups its arguments as ExprTk does ─────────
#
# ExprTk's vararg_max_op / vararg_min_op (exprtk.hpp) are a left fold of the
# binary call for three arguments and for six or more, but process_4 and
# process_5 pair the first four. With every argument finite the grouping cannot
# show; with a NaN it decides what the NaN takes with it. The compiled call was
# a left fold at every arity, so `max(0,0,NaN,1)` was 1 compiled and 0
# interpreted.


def _hi(a, b):
    return b if a < b else a  # std::max


def _lo(a, b):
    return b if b < a else a  # std::min


def _exprtk_vararg(f, v):
    """exprtk.hpp's vararg_max_op::process / vararg_min_op::process, line for line."""
    if len(v) == 4:
        return f(f(v[0], v[1]), f(v[2], v[3]))
    if len(v) == 5:
        return f(f(f(v[0], v[1]), f(v[2], v[3])), v[4])
    r = v[0]
    for x in v[1:]:
        r = f(r, x)
    return r


_SOURCE_NET = """begin parameters
    1 k 1.0
end parameters
begin species
    1 P() 0
    2 Q() 0
{species}end species
begin functions
    1 frac() P_tot/(P_tot+Q_tot)
{functions}end functions
begin reactions
{reactions}end reactions
begin groups
    1 P_tot 1
    2 Q_tot 2
end groups
"""


@pytest.mark.parametrize("name", ["max", "min"])
@pytest.mark.parametrize("arity", [3, 4, 5, 6])
def test_a_nan_in_any_later_position_agrees_with_exprtk(tmp_path, name, arity):
    """frac() is 0/0, a NaN, because the pool is empty. Each X_j is fed from
    nothing at the call with the NaN in position j, so X_j(10) is 10 times the
    call's value. A NaN in first position wins in every grouping and the run is
    refused (above), so it is left out."""
    f = _hi if name == "max" else _lo
    # The fourth argument is the extreme one and the fifth is not, so a NaN in
    # third position takes the fourth with it in ExprTk's pairing and not in a
    # left fold, at four arguments and at five.
    values = (
        [0.25, 0.5, 0.75, 1.0, 0.6, 0.4] if name == "max" else [1.0, 0.75, 0.5, 0.25, 0.6, 0.8]
    )[:arity]
    positions = range(1, arity)
    species, functions, reactions, expected = [], [], [], []
    for row, j in enumerate(positions):
        args = [repr(v) for v in values]
        args[j] = "frac()"
        species.append(f"    {row + 3} X{j}() 0\n")
        functions.append(f"    {row + 2} r{j}() {name}({','.join(args)})\n")
        reactions.append(f"    {row + 1} 0 {row + 3} r{j}\n")
        with_nan = list(values)
        with_nan[j] = float("nan")
        expected.append(10.0 * _exprtk_vararg(f, with_nan))
    path = os.path.join(tmp_path, f"{name}{arity}.net")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            _SOURCE_NET.format(
                species="".join(species),
                functions="".join(functions),
                reactions="".join(reactions),
            )
        )
    runs = []
    for codegen in (False, True):
        sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", codegen=codegen)
        runs.append(np.asarray(sim.run(t_span=(0.0, 10.0), n_points=2).species)[-1, 2:])
    interpreted, compiled = runs
    np.testing.assert_allclose(interpreted, expected, rtol=1e-12)
    np.testing.assert_array_equal(compiled, interpreted)


def test_the_issue_case_differs_between_the_two_groupings():
    """The test above can tell the groupings apart: at four and five arguments,
    a left fold gives another value than ExprTk for a NaN in third position."""
    nan = float("nan")
    for v in ([0, 0, nan, 1.0], [0, 0, nan, 1.0, 0.5]):
        fold = v[0]
        for x in v[1:]:
            fold = _hi(fold, x)
        assert fold != _exprtk_vararg(_hi, v)
