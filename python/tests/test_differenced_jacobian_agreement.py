"""A differenced Jacobian is confirmed at two steps, or its answer refused
(issue #1002).

A difference quotient of the right-hand side steps a species by √eps of the
larger of itself and the state's largest concentration. The floor keeps a
small species out of the cancellation noise of the rows it is read in, and it
steps that species by many times itself: beside a species at 1e8, one at 0.87
is stepped by 1.5, and ``k·X²`` is read as a secant across twice the species.
``steady_state(jacobian="fd", sensitivity_params=...)`` returned dX*/ds 7% off
beside an unrelated species at 1e7, 44% at 1e8 and 99% at 1e10, with no
error.

The species' own step has the opposite failure: a term that reads it through a
sum with something much larger, ``C·((K + X) − K)``, answers in steps of that
sum's last bit, which the own step does not see. Nothing in a row tells the two
apart in every case, so the columns are solved with both matrices. Where they
agree the floored one's are returned, as they were; where an entry is further
apart than 1e-4 of itself (1e-10 of its column's largest, at least), they are
refused. Held to 1e-4 of the column's largest instead, an entry small within
its column passed 12% off. The state-chain term
of a function's sensitivity was differenced at the default settings too, and
held to the same check; it is not differenced at all now (Step 2 of the plan
on #1002): a function the compiled chain rule has no derivative for is
refused by name, as after a run.

The own step reads a row only where its rounding over the step is under the
term, and a row that a much larger species is in it does not read: there a
term of second order in the small species was compared with nothing, and
``k·X²`` in the rate of a species at 1e8 came back 12% off. A species at 0
was not stepped twice at all. So the columns are solved a third time, with
those columns, and those of species at 0, taken whole at half the floored
step, which a secant moves with and a slope does not.

Every expected value is a closed form.

Since issue #1330 a model the dual evaluation follows takes these derivatives
exactly instead, by dual numbers through the interpreter, and none of them is
differenced: test_exact_derivatives_by_dual_numbers.py holds it to the same
closed forms. The differences above still run for a model with no dual
evaluation (a ring of functions that read each other, rateOf), so this module
switches the dual one off (BNGSIM_NO_DUAL_NUMBERS) to keep their checks tested.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _differenced(monkeypatch):
    """Every model of this module on the difference quotients (issue #1330)."""
    monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")


S, K1, K2, KD = 1.3, 0.21, 0.73, 0.5
# 0 -> X at s, X -> 0 at k1, X + X -> D at k2 (a net loss of 2*k2*X^2), D -> 0.
# G is unrelated: made at sg and removed at 1, it sits at sg.
X_SS = (-K1 + math.sqrt(K1 * K1 + 8 * K2 * S)) / (4 * K2)
DX_DS = 1.0 / (K1 + 4 * K2 * X_SS)
SQRT_EPS = 1.4901161193847656e-8

NET = """begin parameters
    1 s   {s!r}
    2 k1  {k1!r}
    3 k2  {k2!r}
    4 kd  {kd!r}
    5 sg  {big!r}
    6 dg  1.0
end parameters
begin functions
    1 xsq() abs(Xo)*Xo
    2 weak() 1e-3*Xo
end functions
begin species
    1 X() 0
    2 D() 0
    3 G() {big!r}
end species
begin reactions
    1 0 1 s
    2 1 0 k1
    3 1,1 2 k2
    4 2 0 kd
    5 0 3 sg
    6 3 0 dg
end reactions
begin groups
    1 Xo 1
end groups
"""


def _model(tmp_path, big, functions=("xsq", "weak"), s=S):
    text = NET.format(s=s, k1=K1, k2=K2, kd=KD, big=float(big))
    for name, line in (("xsq", "    1 xsq() abs(Xo)*Xo\n"), ("weak", "    2 weak() 1e-3*Xo\n")):
        if name not in functions:
            text = text.replace(line, "")
    path = tmp_path / "small.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


def _state(big, x=X_SS):
    return np.array([x, K2 * x * x / KD, float(big)])


def _fd(model, y):
    return np.asarray(model._core.fill_dense_fd_jacobian(0.0, np.asarray(y, dtype=float)))


def _first_quotient(model, y, j):
    """The one-sided quotient at the floored step, as the sweep takes it."""
    y = np.asarray(y, dtype=float)
    stepped = y.copy()
    stepped[j] = y[j] + SQRT_EPS * max(abs(y[j]), float(np.max(np.abs(y))))
    h = stepped[j] - y[j]
    return (np.asarray(model.rhs(stepped, 0.0)) - np.asarray(model.rhs(y, 0.0))) / h


def _steady(model, **kw):
    sim = bngsim.Simulator(model, method="ode", **kw)
    return sim.steady_state(sensitivity_params=["s"])


def _refused_by_name(out, name, construct):
    """``output_sensitivities`` refuses the function by its name and the
    construct the compiled chain rule has no derivative for (issue #1002)."""
    with pytest.raises(ValueError, match=rf"'expression:{name}' has no steady-state") as err:
        out.output_sensitivities([name])
    assert f"uses unsupported construct: {construct}" in str(err.value)
    assert "issue #1002" in str(err.value)


# ─── The steady state's columns ──────────────────────────────────────────────


@pytest.mark.parametrize("big", [1e6, 1e7, 1e8, 1e10])
def test_columns_the_step_decides_are_refused(tmp_path, big):
    """dX*/ds = 1/(k1 + 4·k2·X*), whatever G is. With ``jacobian="fd"`` it
    came back 0.8% off at G = 1e6, 7% at 1e7, 44% at 1e8 and 99% at 1e10,
    with no error. Now the error names the entry and both of its values."""
    model = _model(tmp_path, big, functions=())
    with pytest.raises(bngsim.SimulationError, match=r"issue #1002") as err:
        _steady(model, jacobian="fd")
    msg = str(err.value)
    assert "dY_ss/dp" in msg
    assert "stepped by 1.5e-8 of itself" in msg


@pytest.mark.parametrize("big", [1.0, 30.0, 1e4])
def test_columns_both_steps_agree_on_are_returned(tmp_path, big):
    """Control. Within 64 times the largest concentration no column is taken
    again; at G = 1e4 it is, and the two answers are 8e-5 of each other,
    which is agreement."""
    out = _steady(_model(tmp_path, big, functions=()), jacobian="fd")
    assert out.sens_jacobian_source == "finite-difference"
    assert out.sens_fd_disagree_species is None
    assert out.sens_fd_disagree_param is None
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(DX_DS, rel=1e-4)
    assert len(out.sens_fd_own_step_species) == (0 if big <= 30.0 else 1)


@pytest.mark.parametrize("big", [1e8, 1e10])
def test_a_closed_form_jacobian_is_not_asked(tmp_path, big):
    """Control. The default Jacobian is in closed form for this model, and
    its columns are right whatever G is: nothing is solved twice."""
    out = _steady(_model(tmp_path, big, functions=()))
    assert out.sens_jacobian_source != "finite-difference"
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(DX_DS, rel=1e-9)
    assert out.sens_fd_own_step_species == []
    assert out.sens_fd_half_step_species == []


GRAIN = """begin parameters
    1 s   {s!r}
    2 k1  0.21
    3 kd  0.5
    4 sg  {big!r}
    5 dg  1.0
    6 A   3.0
    7 C   -2.0
    8 K   {k!r}
end parameters
begin functions
    1 mixed() A*Xo + C*((K+Xo)-K)
end functions
begin species
    1 X() 0
    2 E() 0
    3 G() {big!r}
end species
begin reactions
    1 0 1 s
    2 1 0 k1
    3 0 2 mixed
    4 2 0 kd
    5 0 3 sg
    6 3 0 dg
end reactions
begin groups
    1 Xo 1
end groups
"""


def _grain(tmp_path, k):
    path = tmp_path / "grain.net"
    path.write_text(GRAIN.format(s=2.1e-3, big=1e10, k=k))
    return bngsim.Model.from_net(path)


def test_a_term_in_the_grain_of_a_larger_sum_is_refused(tmp_path):
    """E is made at ``A·X + C·((K + X) − K)``, which is ``(A + C)·X``, with
    K = 1e8 and X = 0.01 beside a species at 1e10. X is stepped by 149 at
    the floored step, which sees the whole slope, 1, and by 1.5e-10 at its
    own, and 9.5e-9 at 64 times that, both under the 1.5e-8 grain of K + X:
    those steps see only A, 3. dE*/ds = (A + C)/(k1·kd) is 9.52 at the first
    and 28.6 at the second. The floored answer is the right one, but nothing
    in a row tells this from a term of second order: refused."""
    with pytest.raises(bngsim.SimulationError, match=r"issue #1002") as err:
        _steady(_grain(tmp_path, 1e8), jacobian="fd")
    assert "dE()/ds is 9.52381" in str(err.value)
    assert "28.5714" in str(err.value)


def test_a_grain_inside_two_decades_of_the_own_step_is_returned(tmp_path):
    """Control. With K = 1e7 the grain, 1.9e-9, is under 64 times X's own
    step, where the quotient reads 1 and not 3: the two quotients do not
    agree, X's column is not taken again, and the floored answer is
    returned."""
    out = _steady(_grain(tmp_path, 1e7), jacobian="fd")
    assert out.sens_fd_own_step_species == []
    sens = np.asarray(out.sensitivity)
    assert sens[0, 0] == pytest.approx(1 / 0.21, rel=1e-6)
    assert sens[1, 0] == pytest.approx(1 / (0.21 * 0.5), rel=1e-6)


ALL_GRAIN = """begin parameters
    1 s   {s!r}
    2 k1  {k1!r}
    3 k2  {k2!r}
    4 kd  {kd!r}
    5 sg  {big!r}
    6 dg  1.0
    7 C   2.0
    8 K   {k!r}
end parameters
begin functions
    1 grain() C*((K+Xo)-K)
end functions
begin species
    1 X() 0
    2 D() 0
    3 E() 0
    4 G() {big!r}
end species
begin reactions
    1 0 1 s
    2 1 0 k1
    3 1,1 2 k2
    4 2 0 kd
    5 0 3 grain
    6 3 0 kd
    7 0 4 sg
    8 4 0 dg
end reactions
begin groups
    1 Xo 1
end groups
"""


@pytest.mark.parametrize("big, k", [(1e8, 1e8), (1e10, 1e10), (1e8, 1e10)])
def test_a_row_the_own_step_sees_nothing_of_does_not_hide_a_column_it_decides(tmp_path, big, k):
    """The issue's model with one reaction added, E made at ``C·((K + X) − K)``
    with K at 1e8 or more: the grain of K + X is wider than 64 of X's own
    steps, so that row reads 0 at the own step and C at the floored one.
    Taken whole or not at all, X's column was not taken, nothing was
    compared, and dX*/ds came back 0.202 (44% off) and 0.00454 (99% off)
    with no error, the very columns the check exists to refuse. A column is
    taken where any entry of it is; the second matrix is only compared
    with, never returned, so a mixed column can only refuse."""
    path = tmp_path / "grain_row.net"
    path.write_text(ALL_GRAIN.format(s=S, k1=K1, k2=K2, kd=KD, big=float(big), k=float(k)))
    model = bngsim.Model.from_net(path)
    with pytest.raises(bngsim.SimulationError, match=r"dY_ss/dp .*issue #1002") as err:
        _steady(model, jacobian="fd")
    # The entry furthest apart relative to itself is X's own: the secant at
    # the floored step, against the closed form at X's own step.
    msg = str(err.value)
    floored = {1e8: 0.202473, 1e10: 0.00453885}[big]
    assert f"dX()/ds is {floored:.6g} with every species stepped" in msg
    assert f"{DX_DS:.6g} with X() stepped by 1.5e-8 of itself" in msg


def test_a_grain_row_beside_a_column_both_steps_agree_on_is_returned(tmp_path):
    """Control. With K = 1 there is no grain, and at G = 30 no column is
    taken again: the floored answer is returned."""
    path = tmp_path / "grain_row.net"
    path.write_text(ALL_GRAIN.format(s=S, k1=K1, k2=K2, kd=KD, big=30.0, k=1.0))
    out = _steady(bngsim.Model.from_net(path), jacobian="fd")
    assert out.sens_fd_disagree_species is None
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(DX_DS, rel=1e-4)


# ─── A function's sensitivity ────────────────────────────────────────────────


@pytest.mark.parametrize("big", [1e4, 1e8])
def test_a_declined_function_is_refused_by_name(tmp_path, big):
    """``xsq = abs(X)·X`` has no compiled derivative, so its sensitivity at a
    steady state was assembled from a differenced ``∂xsq/∂X``, with the
    closed-form Jacobian too. d(xsq)/ds = 2·X*·dX*/ds; beside G = 1e8 it
    came back 85% off at the default settings, and #1100 refused the solve.
    Beside G = 1e4 the quotient was right to 8.5e-5, and was returned; a
    quotient does not tell the two apart. Neither is differenced now: the
    solve returns dY_ss/dp, and xsq is refused by name."""
    out = _steady(_model(tmp_path, big, functions=("xsq",)))
    assert float(np.asarray(out.sensitivity)[0, 0]) == pytest.approx(DX_DS, rel=1e-7)
    assert out.sens_output_source == "none"
    _refused_by_name(out, "xsq", "abs()")


@pytest.mark.parametrize("functions", [("xsq", "big"), ("big", "xsq")])
def test_a_compiled_function_beside_a_declined_one_keeps_its_row(tmp_path, functions):
    """``big = 1e8 + X`` has a compiled derivative and ``xsq = abs(X)·X`` has
    not. d(xsq)/ds came back 1.17 for 0.633 (85% off) with no error beside
    G = 1e8 and ``big`` (#1100's review), and both were refused with the
    solve after it. ``big`` is returned now, d(big)/ds = dX*/ds, and ``xsq``
    is refused by name, in either order."""
    text = NET.format(s=S, k1=K1, k2=K2, kd=KD, big=1e8)
    lines = {"xsq": "    1 xsq() abs(Xo)*Xo\n", "big": "    2 big() 1e8+Xo\n"}
    block = "".join(lines[name] for name in functions)
    text = text.replace("    1 xsq() abs(Xo)*Xo\n    2 weak() 1e-3*Xo\n", block)
    path = tmp_path / "mixed.net"
    path.write_text(text)
    model = bngsim.Model.from_net(path)
    out = _steady(model)
    assert out.sens_output_source == "partial"
    got = float(np.asarray(out.output_sensitivities(["big"])).ravel()[0])
    assert got == pytest.approx(DX_DS, rel=1e-7)
    _refused_by_name(out, "xsq", "abs()")


def test_a_function_linear_in_a_small_species_is_returned(tmp_path):
    """Control. ``weak = 1e-3·X`` has a compiled derivative: d(weak)/ds =
    1e-3·dX*/ds."""
    out = _steady(_model(tmp_path, 1e8, functions=("weak",)))
    assert out.sens_output_source == "codegen"
    got = float(np.asarray(out.output_sensitivities(["weak"])).ravel()[0])
    assert got == pytest.approx(1e-3 * DX_DS, rel=1e-7)


# ─── The matrices ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("big", [1e6, 1e8, 1e12])
def test_the_second_matrix_has_the_small_species_column_at_its_own_step(tmp_path, big):
    """∂f_X/∂X = -(k1 + 4·k2·X) and ∂f_D/∂X = 2·k2·X. The floored matrix
    reads them as secants (-4.94 and 2.36 for -2.76 and 1.28 at G = 1e8),
    and is what ``fill_dense_fd_jacobian`` still returns; the second has
    them, and every other column is the first's to the bit."""
    model = _model(tmp_path, big, functions=())
    y = _state(big)
    floored = _fd(model, y)
    np.testing.assert_array_equal(floored[:, 0], _first_quotient(model, y, 0))
    own, changed = model._core.fill_dense_fd_jacobian_own_step(0.0, y, np.asfortranarray(floored))
    own = np.asarray(own)
    assert list(changed) == [0]
    assert own[0, 0] == pytest.approx(-(K1 + 4 * K2 * X_SS), rel=1e-7)
    assert own[1, 0] == pytest.approx(2 * K2 * X_SS, rel=1e-7)
    np.testing.assert_array_equal(own[:, 1:], floored[:, 1:])


def test_the_second_matrix_of_a_state_of_one_scale_is_the_first(tmp_path):
    """Control."""
    model = _model(tmp_path, 1.0, functions=())
    y = _state(1.0)
    floored = _fd(model, y)
    own, changed = model._core.fill_dense_fd_jacobian_own_step(0.0, y, np.asfortranarray(floored))
    assert list(changed) == []
    np.testing.assert_array_equal(np.asarray(own), floored)


def test_model_jacobian_refuses_where_the_step_decides(tmp_path, monkeypatch):
    """``Model.jacobian()`` differences by the same sweep where the closed
    form is not to be had (issue #523). It returned the floored matrix beside
    a warning, and the matrix's column of X is a secant across X (issue
    #1002); it refuses now, naming the entry."""
    model = _model(tmp_path, 1e8, functions=())
    monkeypatch.setattr(type(model), "prepare_analytical_jacobian", lambda self: False)
    with pytest.raises(bngsim.SimulationError, match=r"Model\.jacobian\(\) does not return"):
        model.jacobian(_state(1e8))


def test_model_jacobian_is_quiet_where_the_steps_agree(tmp_path, monkeypatch):
    """Control."""
    model = _model(tmp_path, 1.0, functions=())
    monkeypatch.setattr(type(model), "prepare_analytical_jacobian", lambda self: False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jac = model.jacobian(_state(1.0))
    assert jac.source == "finite-difference"


# ─── Curvature in a row the own step does not read ───────────────────────────

# 0 -> X at s and X -> 0 at k1: X is linear in its own row, X* = s/k1.
# X + X -> X + X + G at k makes G at k·X², and G, made at sg and removed at
# dg = 1, sits at sg + k·X*². dG*/ds = 2·k·X*·dX*/ds = 2·k·s/k1², whatever sg
# is. G's row is G's: the rounding of a rate of 1e8 over X's own step, 9e-8,
# is more than the slope of k·X² there, so the own step reads nothing of it.
SECOND_ORDER = """begin parameters
    1 s   {s!r}
    2 k1  {k1!r}
    3 k   1.0
    4 sg  {big!r}
    5 dg  1.0
end parameters
begin species
    1 X() 0
    2 G() {big!r}
end species
begin groups
    1 Xo 1
    2 Go 2
end groups
begin functions
    1 gx() abs(Xo)*Xo + Go
end functions
begin reactions
    1 0 1 s
    2 1 0 k1
    3 1,1 1,1,2 k
    4 0 2 sg
    5 2 0 dg
end reactions
"""
SO_X = S / K1
SO_DG_DS = 2 * SO_X / K1


def _second_order(tmp_path, big, functions=False, s=S):
    text = SECOND_ORDER.format(s=s, k1=K1, big=float(big))
    if not functions:
        text = text.replace("    1 gx() abs(Xo)*Xo + Go\n", "")
    path = tmp_path / "second_order.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


def _secant_dg_ds(big, part=1.0):
    """dG*/ds from the secant of k·X² across X's floored step (or a part of
    it): k·(2·X* + h)·dX*/ds, the X row being linear. The step is floored at
    the largest concentration, G* = sg + k·X*²."""
    h = (SO_X + part * SQRT_EPS * (float(big) + SO_X * SO_X)) - SO_X
    return (2 * SO_X + h) / K1


@pytest.mark.parametrize("big", [1e6, 1e8, 1e10])
def test_curvature_in_a_row_a_larger_species_is_in_is_refused(tmp_path, big):
    """dG*/ds = 2·k·s/k1² = 58.957. With ``jacobian="fd"`` it came back
    59.028 at G = 1e6, 66.053 at 1e8 and 768.5 at 1e10, with no error: X's
    column was not taken again at its own step, since the X row is linear and
    the own step reads nothing of G's, and nothing was compared. At half the
    floored step the secant is k·(2X + h/2), and the two answers are refused,
    with both named."""
    with pytest.raises(bngsim.SimulationError, match=r"issue #1002") as err:
        _steady(_second_order(tmp_path, big), jacobian="fd")
    msg = str(err.value)
    assert "dG()/ds is " + format(_secant_dg_ds(big), ".6g") in msg
    assert (
        format(_secant_dg_ds(big, 0.5), ".6g") + " with X() stepped by half the first step" in msg
    )


def test_curvature_both_steps_agree_on_is_returned(tmp_path):
    """Control. At G = 1e4 X is stepped by 1.5e-4, 1,600 times its own step,
    so its column is taken again at half the step, and the two answers are
    6e-6 of each other: returned, the floored one."""
    out = _steady(_second_order(tmp_path, 1e4), jacobian="fd")
    sens = np.asarray(out.sensitivity)
    assert sens[1, 0] == pytest.approx(_secant_dg_ds(1e4), rel=1e-9)
    assert sens[1, 0] == pytest.approx(SO_DG_DS, rel=2e-5)
    assert out.sens_fd_disagree_species is None
    assert out.sens_fd_half_step_species == ["X()"]


@pytest.mark.parametrize("big", [1e4, 1e8])
def test_curvature_in_a_declined_function_over_a_larger_species_is_refused(tmp_path, big):
    """``gx = abs(X)·X + G`` has no compiled derivative, so its state-chain
    term was differenced at the default settings. d(gx)/ds = 2·X*·dX*/ds +
    dG*/ds = 117.91; beside G = 1e8 it came back 125.01 (6% off) with no
    error, the own step reading nothing of a function at 1e8, and #1294
    refused the solve. At G = 1e4 it was right to 3e-6 and returned. It is
    refused by name at both now, and dG*/ds comes from the closed form."""
    out = _steady(_second_order(tmp_path, big, functions=True))
    assert float(np.asarray(out.sensitivity)[1, 0]) == pytest.approx(SO_DG_DS, rel=1e-7)
    _refused_by_name(out, "gx", "abs()")


@pytest.mark.parametrize("big", [1e4, 1e8])
def test_a_species_at_0_the_parameter_moves_is_refused(tmp_path, big):
    """With s = 0 nothing makes X, and X* = 0; dX*/ds = 1/k1 = 4.7619. A
    species at 0 has no own step, so its column was never taken again, and
    ``-2·k2·X²`` was read as ``-2·k2·h`` for 0: dX*/ds came back 4.7570 at
    G = 1e4 and 0.4192 (91% off) at 1e8, with no error. At half the step it
    is 4.7594 and 0.7705."""
    with pytest.raises(bngsim.SimulationError, match=r"issue #1002") as err:
        _steady(_model(tmp_path, big, functions=(), s=0.0), jacobian="fd")
    msg = str(err.value)
    assert "stepped by half the first step" in msg
    # The entry furthest apart relative to itself: at G = 1e4, D's, whose
    # floored secant k2·h·dX*/ds/kd stands for 0; at 1e8, X's own.
    h = SQRT_EPS * float(big)
    floored_x = 1.0 / (K1 + 2 * K2 * h)
    if big < 1e6:
        assert f"dD()/ds is {K2 * h * floored_x / KD:.6g}" in msg
    else:
        assert f"dX()/ds is {floored_x:.6g}" in msg


@pytest.mark.parametrize("big", [1.0, 1e8])
def test_a_species_at_0_is_returned_where_the_step_does_not_decide(tmp_path, big):
    """Control. With s = 0 and the columns asked for sg, nothing moves X,
    and its column, wrong as it is at 1e8, is multiplied by 0: dG*/dsg = 1
    and dX*/dsg = 0 are returned."""
    model = _model(tmp_path, big, functions=(), s=0.0)
    sim = bngsim.Simulator(model, method="ode", jacobian="fd")
    out = sim.steady_state(sensitivity_params=["sg"])
    sens = np.asarray(out.sensitivity)
    assert sens[0, 0] == 0.0
    assert sens[2, 0] == pytest.approx(1.0, rel=1e-12)
    assert out.sens_fd_disagree_species is None
    assert out.sens_fd_half_step_species == ["X()", "D()"]


def test_an_entry_near_0_the_step_moves_is_refused(tmp_path):
    """With s = 0 beside G = 1, X* = 0 and dD*/ds = 2·k2·X*·dX*/ds/kd = 0.
    The floored step (1.5e-8) reads 1.04e-7 for it, and half the step half
    that. The column was right to 1e-7 of its largest entry (dX*/ds = 1/k1),
    and main returned it; held to 1e-4 of itself (1e-10 of the column's
    largest, at least), the entry is not confirmed, and the columns are
    refused (issue #1002)."""
    model = _model(tmp_path, 1.0, functions=(), s=0.0)
    sim = bngsim.Simulator(model, method="ode", jacobian="fd")
    with pytest.raises(bngsim.SimulationError, match=r"dD\(\)/ds is 1\.0\d+e-07"):
        sim.steady_state(sensitivity_params=["sg", "s"])


def test_an_entry_small_within_its_column_is_held_to_itself(tmp_path):
    """``X + X -> X + X + G`` at k = 1e-6 beside G at 1e8: dG*/ds =
    2·k·X*/k1 = 5.896e-5, and the floored step reads k·(2X* + h)/k1 =
    6.605e-5, 12% off. The half step reads it 6% off, 3.6e-6 from the first,
    which is 7.5e-7 of the column's largest entry (dX*/ds = 4.76): held to 1e-4 of that, main
    returned it with no error. Held to 1e-4 of itself, it is refused."""
    text = SECOND_ORDER.replace("    3 k   1.0\n", "    3 k   1e-6\n").format(s=S, k1=K1, big=1e8)
    text = text.replace("    1 gx() abs(Xo)*Xo + Go\n", "")
    path = tmp_path / "small_entry.net"
    path.write_text(text)
    with pytest.raises(bngsim.SimulationError, match=r"dG\(\)/ds is 6\.60476e-05"):
        _steady(bngsim.Model.from_net(path), jacobian="fd")


def test_a_declined_function_of_a_species_at_0_is_refused(tmp_path):
    """``xsq = abs(X)·X`` was differenced; with s = 0, X* = 0 and
    d(xsq)/ds = 2·|X*|·dX*/ds = 0. Its partial across the floored step is
    h for 0, so d(xsq)/ds came back 7.1e-8, the step's alone, at the default
    settings beside G = 1, with no error, and #1294 refused the solve. The
    solve returns dX*/ds = 1/k1 now, and xsq is refused by name."""
    out = _steady(_model(tmp_path, 1.0, functions=("xsq",), s=0.0))
    assert float(np.asarray(out.sensitivity)[0, 0]) == pytest.approx(1.0 / K1, rel=1e-7)
    _refused_by_name(out, "xsq", "abs()")


NAN_WINDOW = """begin parameters
    1 s   0.0
    2 k1  0.21
    3 sg  1e8
    4 dg  1.0
    5 kd  0.5
end parameters
begin species
    1 X() 0
    2 E() 0
    3 G() 1e8
end species
begin groups
    1 Xo 1
end groups
begin functions
    1 r() if(Xo > 0.6, if(Xo < 0.9, log(Xo - 1), 0), 0)
end functions
begin reactions
    1 0 1 s
    2 1 0 k1
    3 0 2 r
    4 2 0 kd
    5 0 3 sg
    6 3 0 dg
end reactions
"""


def test_a_half_step_answer_that_is_no_number_is_refused(tmp_path):
    """E is made at a rate that is no number for X between 0.6 and 0.9, and
    X sits at 0 beside G = 1e8: the floored step, 1.49, is past that window
    and half of it is inside. The floored answer, dX*/ds = 1/k1, is right
    here, and main returns it; but a column that is no number at half the
    step confirms nothing, and a column of the answer that reads it is
    refused, as one is where the own step's solve is no number."""
    path = tmp_path / "nan_window.net"
    path.write_text(NAN_WINDOW)
    with pytest.raises(bngsim.SimulationError, match=r"dY_ss/dp") as err:
        _steady(bngsim.Model.from_net(path), jacobian="fd")
    assert "dX()/ds is " + format(1 / K1, ".6g") in str(err.value)
    assert "nan with X(), E() stepped by half the first step" in str(err.value)


@pytest.mark.parametrize("kw", [{}, {"jacobian": "fd"}])
def test_a_half_step_column_that_is_no_number_is_not_read_where_its_entry_is_0(tmp_path, kw):
    """Control. Asked for sg, which moves neither X nor E, the answer does
    not read X's column: dG*/dsg = 1 and the rest 0 are returned, as on main,
    with the differenced Jacobian and with the default one. ``r`` reads an
    ``if()``, which has no compiled derivative, and is refused by name (it
    was differenced, and 0 here)."""
    path = tmp_path / "nan_window.net"
    path.write_text(NAN_WINDOW)
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", **kw)
    out = sim.steady_state(sensitivity_params=["sg"])
    sens = np.asarray(out.sensitivity)
    assert sens[0, 0] == 0.0
    assert sens[1, 0] == 0.0
    assert sens[2, 0] == pytest.approx(1.0, rel=1e-12)
    _refused_by_name(out, "r", "if() conditional")


# SECOND_ORDER with Z, made at kz = 1e-30 and removed at 1, which makes X at
# q·Z: dX*/dq = Z*/k1 and dG*/dq = 2·k·X*·dX*/dq, a column of 1e-29.
TINY_COLUMN = (
    SECOND_ORDER.replace("    5 dg  1.0\n", "    5 dg  1.0\n    6 kz  1e-30\n    7 q   1.0\n")
    .replace("    2 G() {big!r}\n", "    2 G() {big!r}\n    3 Z() 0\n")
    .replace("    5 2 0 dg\n", "    5 2 0 dg\n    6 0 3 kz\n    7 3 0 dg\n    8 3 3,1 q\n")
)


def _tiny_column(tmp_path, functions):
    text = TINY_COLUMN.format(s=S, k1=K1, big=1e8)
    if not functions:
        text = text.replace("    1 gx() abs(Xo)*Xo + Go\n", "")
    path = tmp_path / "tiny_column.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


def test_a_column_of_1e_29_is_held_to_the_same_check(tmp_path):
    """dG*/dq is 5.9e-29 beside G at 1e8, and the floored step reads it 12%
    high, as it reads dG*/ds: main returned it with ``jacobian="fd"`` and no
    error. A column is refused for what the step does to it, whatever its
    size; the half step moves this one by 5% of itself."""
    sim = bngsim.Simulator(_tiny_column(tmp_path, False), method="ode", jacobian="fd")
    with pytest.raises(bngsim.SimulationError, match=r"dG\(\)/dq is ") as err:
        sim.steady_state(sensitivity_params=["q"])
    assert "stepped by half the first step" in str(err.value)


def test_a_function_column_of_1e_28_is_refused_by_name(tmp_path):
    """``gx = abs(X)·X + G`` beside G at 1e8: d(gx)/dq is 1.2e-28, and the
    floored partial over X read it 6% high at the default settings, where
    main returned it with no error (and #1294 refused the solve). It is not
    differenced now, whatever its size."""
    sim = bngsim.Simulator(_tiny_column(tmp_path, True), method="ode")
    out = sim.steady_state(sensitivity_params=["q"])
    _refused_by_name(out, "gx", "abs()")


@pytest.mark.parametrize("s", [1e-12, 1e-9])
def test_a_parameter_at_a_small_value_is_held_to_the_same_check(tmp_path, s):
    """dG*/ds = 2·k·s/k1², 4.5e-11 at s = 1e-12, where X* = s/k1 is 4.8e-12
    beside G at 1e8. The floored step, 1.49, reads k·(2X + h): main returned
    dG*/ds = 7.0958 with ``jacobian="fd"`` and no error, 1.6e11 times the
    slope. (A floor on the half step's disagreement that was scaled by the
    parameter's size let it through.)"""
    x = s / K1
    h = (x + SQRT_EPS * (1e8 + x * x)) - x
    with pytest.raises(bngsim.SimulationError, match=r"issue #1002") as err:
        _steady(_second_order(tmp_path, 1e8, s=s), jacobian="fd")
    assert "dG()/ds is " + format((2 * x + h) / K1, ".6g") in str(err.value)
    assert "stepped by half the first step" in str(err.value)


# B <-> A at kf = kr = 1 with A + B = 1, B the law's dependent; G is made at
# k·B² beside sg = 1e8. B* = 1/2, dB*/dkf = 1/4 and dG*/dkf = 2·k·B*·dB*/dkf.
MASKED_DEPENDENT = """begin parameters
    1 kf  1.0
    2 kr  1.0
    3 k   1.0
    4 sg  1e8
    5 dg  1.0
    6 T   1.0
end parameters
begin species
    1 B() 0
    2 A() T
    3 G() 1e8
end species
begin reactions
    1 2 1 kf
    2 1 2 kr
    3 1,1 1,1,3 k
    4 0 3 sg
    5 3 0 dg
end reactions
"""

# $F at F0 = 0.5 makes G at k·F² beside sg = 1e8: dG*/dF0 = 2·k·F0 = 1.
MASKED_FIXED = """begin parameters
    1 F0  0.5
    2 k   1.0
    3 sg  1e8
    4 dg  1.0
    5 s   1.3
    6 k1  0.21
end parameters
begin species
    1 $F() F0
    2 X() 0
    3 G() 1e8
end species
begin reactions
    1 1,1 1,1,3 k
    2 0 3 sg
    3 3 0 dg
    4 0 2 s
    5 2 0 k1
end reactions
"""


@pytest.mark.parametrize(
    "text, param, mask, floored",
    [
        (MASKED_DEPENDENT, "kf", ["A()", "G()"], "dG()/dkf is 0.622529"),
        (MASKED_FIXED, "F0", ["X()", "G()"], "dG()/dF0 is 2.49012"),
    ],
    ids=["a law's dependent", "a fixed species a parameter sets"],
)
def test_a_species_the_mask_leaves_out_is_stepped_again_where_its_column_is_read(
    tmp_path, text, param, mask, floored
):
    """A species ``mask=`` leaves out is held, and its column is still read
    by the solve where it is a conservation law's dependent, rebuilt from the
    total, or a fixed species a parameter sets. Its secant across the floored
    step, k·(2B + h), came back as dG*/dkf 0.6225 for 0.25 and dG*/dF0 2.490
    for 1 with ``jacobian="fd"`` and no error. (Not stepping it again, which
    an earlier cut of this check did, let both through.)"""
    path = tmp_path / "masked.net"
    path.write_text(text)
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", jacobian="fd")
    with pytest.raises(bngsim.SimulationError, match=r"issue #1002") as err:
        sim.steady_state(sensitivity_params=[param], mask=mask)
    assert floored in str(err.value)
    assert "stepped by half the first step" in str(err.value)


def test_a_function_whose_half_step_partial_is_no_number_is_refused(tmp_path):
    """At the default settings r's partial over X was differenced, and it is
    no number at half the step, where dX*/ds is not 0: #1294 refused the
    solve. It is not differenced now: the solve returns dX*/ds = 1/k1, and
    ``r`` is refused by name."""
    path = tmp_path / "nan_window.net"
    path.write_text(NAN_WINDOW)
    out = _steady(bngsim.Model.from_net(path))
    assert float(np.asarray(out.sensitivity)[0, 0]) == pytest.approx(1.0 / K1, rel=1e-7)
    _refused_by_name(out, "r", "if() conditional")


def test_model_jacobian_refuses_where_the_half_step_is_no_number(tmp_path, monkeypatch):
    """``Model.jacobian()`` at X = 0 beside G = 1e8 in the model whose rate
    is no number for X between 0.6 and 0.9: the column of X at half the
    step is no number, and the refusal says so (it warned)."""
    path = tmp_path / "nan_window.net"
    path.write_text(NAN_WINDOW)
    model = bngsim.Model.from_net(path)
    monkeypatch.setattr(type(model), "prepare_analytical_jacobian", lambda self: False)
    with pytest.raises(bngsim.SimulationError, match=r"and nan stepped by half that"):
        model.jacobian(np.array([0.0, 0.0, 1e8]))


@pytest.mark.parametrize("big", [1e6, 1e8])
def test_the_third_matrix_has_the_columns_at_half_the_step(tmp_path, big):
    """The columns of X and D, both far below G, are each the quotient at
    half the floored step, whole; G's is the first matrix's to the bit."""
    model = _model(tmp_path, big, functions=())
    y = _state(big)
    floored = _fd(model, y)
    half, taken = model._core.fill_dense_fd_jacobian_half_step(0.0, y, np.asfortranarray(floored))
    half = np.asarray(half)
    assert list(taken) == [0, 1]
    for j in (0, 1):
        stepped = y.copy()
        stepped[j] = y[j] + 0.5 * SQRT_EPS * float(big)
        quotient = (np.asarray(model.rhs(stepped, 0.0)) - np.asarray(model.rhs(y, 0.0))) / (
            stepped[j] - y[j]
        )
        np.testing.assert_array_equal(half[:, j], quotient)
    # The secant of -2·k2·X² across h/2, and of k2·X².
    h = (y[0] + 0.5 * SQRT_EPS * float(big)) - y[0]
    assert half[0, 0] == pytest.approx(-(K1 + 2 * K2 * (2 * X_SS + h)), rel=1e-6)
    assert half[1, 0] == pytest.approx(K2 * (2 * X_SS + h), rel=1e-6)
    np.testing.assert_array_equal(half[:, 2], floored[:, 2])


def test_the_third_matrix_takes_a_species_at_0(tmp_path):
    """X and D at 0 beside G = 1e8: both columns are taken, X's reading
    -k1 - 2·k2·(h/2) for the floored -k1 - 2·k2·h."""
    model = _model(tmp_path, 1e8, functions=())
    y = np.array([0.0, 0.0, 1e8])
    floored = _fd(model, y)
    half, taken = model._core.fill_dense_fd_jacobian_half_step(0.0, y, np.asfortranarray(floored))
    assert list(taken) == [0, 1]
    h = SQRT_EPS * 1e8
    assert floored[0, 0] == pytest.approx(-(K1 + 2 * K2 * h), rel=1e-12)
    assert np.asarray(half)[0, 0] == pytest.approx(-(K1 + K2 * h), rel=1e-12)


def test_the_third_matrix_of_a_state_of_one_scale_is_the_first(tmp_path):
    """Control."""
    model = _model(tmp_path, 1.0, functions=())
    y = _state(1.0)
    floored = _fd(model, y)
    half, taken = model._core.fill_dense_fd_jacobian_half_step(0.0, y, np.asfortranarray(floored))
    assert list(taken) == []
    np.testing.assert_array_equal(np.asarray(half), floored)


def test_model_jacobian_refuses_where_the_half_step_decides(tmp_path, monkeypatch):
    """``Model.jacobian()`` refuses, naming the entry, where the half step
    moves it, in a row the own step cannot read: d(rate of G)/dX is
    k·(2X + h), 13.871, and 13.126 at half the step (it returned the
    floored matrix beside a warning)."""
    model = _second_order(tmp_path, 1e8)
    monkeypatch.setattr(type(model), "prepare_analytical_jacobian", lambda self: False)
    y = np.array([SO_X, 1e8 + SO_X * SO_X])
    with pytest.raises(
        bngsim.SimulationError,
        match=r"d\(rate of G\(\)\)/dX\(\) is 13\.8711 .* 13\.126 stepped by half that",
    ):
        model.jacobian(y)


def test_model_jacobian_holds_an_entry_small_within_its_column_to_itself(tmp_path, monkeypatch):
    """``X + X -> X + X + G`` at k = 1e-6 beside G at 1e8: in X's column,
    d(rate of G)/dX = 2·k·X* = 1.2e-5 sits beside -k1 = -0.21, and the
    floored step reads k·(2X* + h) for it, the half step k·(2X* + h/2). The
    two are 7.5e-7 apart, 3.5e-6 of the column's largest entry, which passed;
    held to 1e-4 of itself the entry is refused."""
    text = SECOND_ORDER.replace("    3 k   1.0\n", "    3 k   1e-6\n").format(s=S, k1=K1, big=1e8)
    text = text.replace("    1 gx() abs(Xo)*Xo + Go\n", "")
    path = tmp_path / "small_entry.net"
    path.write_text(text)
    model = bngsim.Model.from_net(path)
    monkeypatch.setattr(type(model), "prepare_analytical_jacobian", lambda self: False)
    with pytest.raises(bngsim.SimulationError, match=r"d\(rate of G\(\)\)/dX\(\)"):
        model.jacobian(np.array([SO_X, 1e8 + 1e-6 * SO_X * SO_X]))


RESIDUE = """begin parameters
    1 s   1.3
    2 k1  0.21
    3 sg  1e8
    4 dg  1.0
    5 k   1.0
end parameters
begin species
    1 X() 0
    2 G() 1e8
    3 Y() 0
end species
begin groups
    1 Xo 1
end groups
begin functions
    1 res() k*((Xo + 1) - 1 - Xo)
end functions
begin reactions
    1 0 1 s
    2 1 0 k1
    3 0 2 sg
    4 2 0 dg
    5 0 3 res
    6 3 0 dg
end reactions
"""


def test_an_entry_of_rounding_residue_is_not_held_to_itself(tmp_path, monkeypatch):
    """Control. Y is made at ``k·((X + 1) − 1 − X)``, the rounding of a sum,
    so dY*/ds is rounding too: -2.8e-15 beside dX*/ds = 4.76, and another
    rounding at half the step. An entry is held to 1e-4 of itself only down
    to 1e-10 of its column's largest; held to itself alone, this column, a
    right answer to its last bits, was refused. ``Model.jacobian()`` keeps the
    same floor."""
    path = tmp_path / "residue.net"
    path.write_text(RESIDUE)
    out = _steady(bngsim.Model.from_net(path), jacobian="fd")
    sens = np.asarray(out.sensitivity)[:, 0]
    assert sens[0] == pytest.approx(1 / K1, rel=1e-9)
    assert abs(sens[2]) < 1e-12
    assert "Y()" in out.sens_fd_half_step_species
    model = bngsim.Model.from_net(path)
    monkeypatch.setattr(type(model), "prepare_analytical_jacobian", lambda self: False)
    assert model.jacobian(np.array([1.3 / K1, 1e8, float(out.concentrations[2])])).source == (
        "finite-difference"
    )


def test_model_jacobian_does_not_hold_rounding_to_itself(tmp_path, monkeypatch):
    """Control. Y is made at ``k·((X + K) − K − X)`` with K = 3, the rounding
    of a sum: its entry in X's column is -3.7e-17 at the floored step and
    -7.5e-17 at half of it, beside -k1 = -0.21. Held to 1e-4 of itself the
    matrix was refused; the floor, 1e-10 of the column's largest, returns it,
    as ``steady_state`` does its columns."""
    text = RESIDUE.replace("    5 k   1.0\n", "    5 k   1.0\n    6 K   3.0\n").replace(
        "((Xo + 1) - 1 - Xo)", "((Xo + K) - K - Xo)"
    )
    path = tmp_path / "residue_k.net"
    path.write_text(text)
    model = bngsim.Model.from_net(path)
    monkeypatch.setattr(type(model), "prepare_analytical_jacobian", lambda self: False)
    assert model.jacobian(np.array([0.3, 1e8, 0.0])).source == "finite-difference"


def test_model_jacobian_checks_a_column_with_an_entry_that_is_no_number(tmp_path, monkeypatch):
    """W is made at ``sqrt(X − 10)``, no number at X = 6.19, so X's column has
    a NaN in W's row. The column's largest was NaN, which no entry exceeds,
    and the column was not checked: G's entry, 13.871 at the floored step and
    13.126 at half of it, was returned. The check is over the entries that
    are numbers, as ``steady_state``'s is, and refuses it."""
    text = SECOND_ORDER.format(s=S, k1=K1, big=1e8).replace("    1 gx() abs(Xo)*Xo + Go\n", "")
    text = (
        text.replace("    2 G() 100000000.0\n", "    2 G() 100000000.0\n    3 W() 0\n")
        .replace("begin functions\n", "begin functions\n    1 wr() sqrt(Xo - 10)\n")
        .replace("    5 2 0 dg\n", "    5 2 0 dg\n    6 0 3 wr\n")
    )
    path = tmp_path / "nan_row.net"
    path.write_text(text)
    model = bngsim.Model.from_net(path)
    monkeypatch.setattr(type(model), "prepare_analytical_jacobian", lambda self: False)
    with pytest.raises(bngsim.SimulationError, match=r"d\(rate of G\(\)\)/dX\(\) is 13\.8711"):
        model.jacobian(np.array([SO_X, 1e8 + SO_X * SO_X, 0.0]))
