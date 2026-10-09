"""The differenced state Jacobian of a species far below the state's scale
(issue #1002).

A difference quotient of the right-hand side steps a species by √eps of the
larger of itself and the state's largest concentration. The floor keeps a
small species out of the cancellation noise of the rows it is read in, and it
steps that species by many times itself: beside a species at 1e8, one at 0.87
is stepped by 1.5, and ``k·X²`` is read as a secant across twice the species,
``k·(2X + h)``.

- ``steady_state(jacobian="fd", sensitivity_params=...)`` returned dX*/ds 7%
  off beside an unrelated species at 1e7, 44% at 1e8 and 99% at 1e10, with no
  error. ``Model.jacobian()`` hands back the same matrix where it differences.
- A species at exactly 0 that dimerizes was given a decay rate it does not
  have: -2.39 for -0.21.
- The sensitivity of a function at a steady state is assembled from
  ``∂func/∂x`` by the same step, for a function the compiled code has no
  derivative of. ``abs(X)*X`` beside a species at 1e8 came back 85% off **at
  the default settings**, with the closed-form Jacobian.

A second quotient at half the step is now taken for such a species. Where the
two agree to what rounding can do to them, the entry is the first, bit for bit
as it was. Where they do not, the entry is extrapolated to a step of zero from
a ladder of halved steps, which for a mass-action law is exact in three.

Every expected value is a closed form.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

S, K1, K2, KD = 1.3, 0.21, 0.73, 0.5
# 0 -> X at s, X -> 0 at k1, X + X -> D at k2 (a net loss of 2*k2*X^2), D -> 0.
# G is unrelated: made at sg and removed at 1, it sits at sg.
X_SS = (-K1 + math.sqrt(K1 * K1 + 8 * K2 * S)) / (4 * K2)
DX_DS = 1.0 / (K1 + 4 * K2 * X_SS)

NET = """begin parameters
    1 s   {s!r}
    2 k1  {k1!r}
    3 k2  {k2!r}
    4 kd  {kd!r}
    5 sg  {big!r}
    6 dg  1.0
    7 k3  0.4
    8 KH  {kh!r}
    9 VH  0.9
end parameters
begin functions
    1 xsq() abs(Xo)*Xo
    2 hill() VH*Xo^4/(KH^4+Xo^4)
    3 weak() 1e-3*Xo
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
{more}end reactions
begin groups
    1 Xo 1
end groups
"""


def _model(tmp_path, big, more="", kh=1.0):
    path = tmp_path / "small.net"
    path.write_text(NET.format(s=S, k1=K1, k2=K2, kd=KD, big=float(big), more=more, kh=kh))
    return bngsim.Model.from_net(path)


def _state(big, x=X_SS):
    return np.array([x, K2 * x * x / KD, float(big)])


def _fd(model, y):
    return np.asarray(model._core.fill_dense_fd_jacobian(0.0, np.asarray(y, dtype=float)))


# ─── The steady state's columns ──────────────────────────────────────────────


@pytest.mark.parametrize("big", [1e4, 1e6, 1e7, 1e8, 1e10])
def test_a_steady_state_column_beside_a_large_species(tmp_path, big):
    """dX*/ds = 1/(k1 + 4·k2·X*), whatever G is. With ``jacobian="fd"`` it was
    8e-5 off at G = 1e4, 0.8% at 1e6, 7% at 1e7, 44% at 1e8 and 99% at 1e10."""
    sim = bngsim.Simulator(_model(tmp_path, big), method="ode", jacobian="fd")
    out = sim.steady_state(sensitivity_params=["s"])
    assert out.sens_jacobian_source == "finite-difference"
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(DX_DS, rel=1e-7)


@pytest.mark.parametrize("big", [1.0, 30.0])
def test_a_steady_state_column_beside_a_species_of_its_own_size(tmp_path, big):
    """Control. Within 64 times the largest concentration the floored step
    is at most 64·√eps of the species, and its column is not looked at
    again."""
    sim = bngsim.Simulator(_model(tmp_path, big), method="ode", jacobian="fd")
    out = sim.steady_state(sensitivity_params=["s"])
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(DX_DS, rel=1e-6)


@pytest.mark.parametrize("kw", [{}, {"jacobian": "fd"}], ids=["closed-form", "differenced"])
@pytest.mark.parametrize("big", [1e4, 1e8])
def test_a_function_that_is_not_linear_in_a_small_species(tmp_path, big, kw):
    """``xsq = abs(X)·X`` has no compiled derivative, so its sensitivity at a
    steady state is assembled from a differenced ``∂xsq/∂X``, with the
    closed-form Jacobian too. d(xsq)/ds = 2·X*·dX*/ds. At the default
    settings it was 8.5e-5 off at G = 1e4 and 85% off at 1e8."""
    out = bngsim.Simulator(_model(tmp_path, big), method="ode", **kw).steady_state(
        sensitivity_params=["s"]
    )
    got = float(np.asarray(out.output_sensitivities(["xsq"])).ravel()[0])
    assert got == pytest.approx(2 * X_SS * DX_DS, rel=1e-7)


def test_a_function_that_is_linear_in_a_small_species(tmp_path):
    """Control. ``weak = 1e-3·X``: d(weak)/ds = 1e-3·dX*/ds."""
    out = bngsim.Simulator(_model(tmp_path, 1e8), method="ode").steady_state(
        sensitivity_params=["s"]
    )
    got = float(np.asarray(out.output_sensitivities(["weak"])).ravel()[0])
    assert got == pytest.approx(1e-3 * DX_DS, rel=1e-7)


# ─── The matrix itself ───────────────────────────────────────────────────────


@pytest.mark.parametrize("big", [1e6, 1e8, 1e12])
def test_the_entry_of_a_second_order_term(tmp_path, big):
    """∂f_X/∂X = -(k1 + 4·k2·X) and ∂f_D/∂X = 2·k2·X. At G = 1e8 they came
    back -4.94 for -2.76 and 2.36 for 1.28."""
    jac = _fd(_model(tmp_path, big), _state(big))
    assert jac[0, 0] == pytest.approx(-(K1 + 4 * K2 * X_SS), rel=1e-9)
    assert jac[1, 0] == pytest.approx(2 * K2 * X_SS, rel=1e-9)


def test_a_species_at_zero_that_dimerizes(tmp_path):
    """At X = 0 the dimerization has no slope: ∂f_X/∂X = -k1. A step of 1.5
    gave it one, -2.39 for -0.21: a decay the stability of a state is read
    from."""
    jac = _fd(_model(tmp_path, 1e8), _state(1e8, x=0.0))
    assert jac[0, 0] == pytest.approx(-K1, rel=1e-12)
    assert jac[1, 0] == pytest.approx(0.0, abs=1e-12)


def test_a_term_of_third_order(tmp_path):
    """``3 X -> 0`` at k3 is of third order in X, which a ladder of four
    steps is exact for. ∂f_X/∂X = -(k1 + 4·k2·X + 9·k3·X²)."""
    model = _model(tmp_path, 1e8, more="    7 1,1,1 0 k3\n")
    jac = _fd(model, _state(1e8))
    assert jac[0, 0] == pytest.approx(-(K1 + 4 * K2 * X_SS + 9 * 0.4 * X_SS**2), rel=1e-9)


@pytest.mark.parametrize("x, kh", [(1e-3, 2e-3), (0.87, 0.5), (0.0, 1e-4)])
def test_a_saturating_law_in_a_small_species(tmp_path, x, kh):
    """``0 -> D`` at ``VH·X⁴/(KH⁴ + X⁴)``, no polynomial in X: the ladder runs
    until the estimates settle. ∂f_D/∂X = 2·k2·X + 4·VH·KH⁴·X³/(KH⁴ + X⁴)²."""
    model = _model(tmp_path, 1e8, more="    7 0 2 hill\n", kh=kh)
    jac = _fd(model, _state(1e8, x=x))
    want = 2 * K2 * x + 4 * 0.9 * kh**4 * x**3 / (kh**4 + x**4) ** 2
    assert jac[1, 0] == pytest.approx(want, rel=1e-6, abs=1e-9)


def test_the_closed_form_and_the_difference_agree(tmp_path):
    """The two matrices of the same model at the same state, entry by entry,
    against the largest entry of each row."""
    model = _model(tmp_path, 1e8, more="    7 1,1,1 0 k3\n    8 0 2 hill\n", kh=0.5)
    assert model.prepare_analytical_jacobian()
    y = _state(1e8)
    closed = np.asarray(model._core.fill_dense_analytical_jacobian(0.0, y))
    apart = np.abs(_fd(model, y) - closed) / np.max(np.abs(closed), axis=1, keepdims=True)
    assert np.max(apart) < 1e-7


# ─── What is left as it was ──────────────────────────────────────────────────

SQRT_EPS = 1.4901161193847656e-8


def _first_quotient(model, y, j):
    """The one-sided quotient at the floored step, as the sweep takes it."""
    y = np.asarray(y, dtype=float)
    stepped = y.copy()
    stepped[j] = y[j] + SQRT_EPS * max(abs(y[j]), float(np.max(np.abs(y))))
    h = stepped[j] - y[j]
    return (np.asarray(model.rhs(stepped, 0.0)) - np.asarray(model.rhs(y, 0.0))) / h


def test_an_entry_a_small_species_enters_linearly_is_bit_for_bit_what_it_was(tmp_path):
    """Control. G gains ``k3·X`` (reaction 7) beside fluxes of 1e8. Half the
    step gives the same quotient to rounding, so the entry is the first
    quotient: extrapolating it would only add the rounding of two more
    evaluations to a row whose terms are 1e8."""
    model = _model(tmp_path, 1e8, more="    7 1 1,3 k3\n")
    y = _state(1e8)
    jac = _fd(model, y)
    first = _first_quotient(model, y, 0)
    assert jac[2, 0] == first[2]
    assert jac[2, 0] == pytest.approx(0.4, rel=1e-6)


def test_the_column_of_a_species_at_the_states_scale_is_what_it_was(tmp_path):
    """Control. G is the largest species, and its column is one quotient."""
    model = _model(tmp_path, 1e8)
    y = _state(1e8)
    np.testing.assert_array_equal(_fd(model, y)[:, 2], _first_quotient(model, y, 2))


def test_a_refined_entry_is_not_the_first_quotient(tmp_path):
    """The entry of the dimerization is the one that changes."""
    model = _model(tmp_path, 1e8)
    y = _state(1e8)
    assert _fd(model, y)[0, 0] != _first_quotient(model, y, 0)[0]


def test_model_jacobian_hands_back_the_same_matrix(tmp_path, monkeypatch):
    """``Model.jacobian()`` differences by the same sweep where the closed
    form is not to be had (issue #523)."""
    model = _model(tmp_path, 1e8)
    monkeypatch.setattr(type(model), "prepare_analytical_jacobian", lambda self: False)
    jac = model.jacobian(_state(1e8))
    assert jac.source == "finite-difference"
    assert np.asarray(jac)[0, 0] == pytest.approx(-(K1 + 4 * K2 * X_SS), rel=1e-9)
