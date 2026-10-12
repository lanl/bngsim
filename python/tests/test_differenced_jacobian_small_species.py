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
- The sensitivity of a function at a steady state is assembled from
  ``∂func/∂x`` by the same step, for a function the compiled code has no
  derivative of. ``abs(X)*X`` beside a species at 1e8 came back 85% off **at
  the default settings**, with the closed-form Jacobian.

For such a species the row is now evaluated at the species' own step, √eps of
itself, at twice that, and a thousandth beyond it. An entry becomes the
quotient at the own step where the row's rounding there is under 1e-4 of it,
the quotient at twice the step agrees with it, the row resolves the step, and
it is further from the first quotient than it is known to. Elsewhere the
entry is the first quotient, bit for bit as it was: where the row's rounding
is too large for the species' own step (which is what the floor is for), and
for a species at exactly 0, which has no step of its own.

Two earlier cuts took the entry from a ladder of halved steps below the
floored one, extrapolated to a step of zero. A ladder shows what the row does
down to its last step: a kink 1.7e-5 of the species away came back 11% off,
and a ripple 1e-4 of the species wide with the wrong sign.

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
   10 kw  1e-4
   11 kbig 1e8
   12 xc  {xc!r}
end parameters
begin functions
    1 xsq() abs(Xo)*Xo
    2 hill() VH*Xo^4/(KH^4+Xo^4)
    3 weak() 1e-3*Xo
    4 root() sqrt(abs(Xo-xc))
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


def _model(tmp_path, big, more="", kh=1.0, xc=0.0):
    path = tmp_path / "small.net"
    path.write_text(
        NET.format(s=S, k1=K1, k2=K2, kd=KD, big=float(big), more=more, kh=kh, xc=float(xc))
    )
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
    model = _model(tmp_path, big)
    sim = bngsim.Simulator(model, method="ode", jacobian="fd")
    out = sim.steady_state(sensitivity_params=["s"])
    assert np.asarray(out.sensitivity)[0, 0] == pytest.approx(DX_DS, rel=1e-6)
    y = _state(big)
    np.testing.assert_array_equal(_fd(model, y)[:, 0], _first_quotient(model, y, 0))


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


@pytest.mark.parametrize("big", [1e6, 1e8, 1e12, 1e14])
def test_the_entry_of_a_second_order_term(tmp_path, big):
    """∂f_X/∂X = -(k1 + 4·k2·X) and ∂f_D/∂X = 2·k2·X. At G = 1e8 they came
    back -4.94 for -2.76 and 2.36 for 1.28."""
    jac = _fd(_model(tmp_path, big), _state(big))
    assert jac[0, 0] == pytest.approx(-(K1 + 4 * K2 * X_SS), rel=1e-7)
    assert jac[1, 0] == pytest.approx(2 * K2 * X_SS, rel=1e-7)


def test_the_column_of_a_species_at_zero_is_what_it_was(tmp_path):
    """Control. A species at exactly 0 has no step of its own, and its column
    is the first quotient: the dimerization's entries there are still the
    secant, -2.39 for -0.21 in its own row."""
    model = _model(tmp_path, 1e8)
    y = _state(1e8, x=0.0)
    np.testing.assert_array_equal(_fd(model, y)[:, 0], _first_quotient(model, y, 0))


def test_a_term_of_third_order(tmp_path):
    """``3 X -> 0`` at k3 is of third order in X.
    ∂f_X/∂X = -(k1 + 4·k2·X + 9·k3·X²)."""
    model = _model(tmp_path, 1e8, more="    7 1,1,1 0 k3\n")
    jac = _fd(model, _state(1e8))
    assert jac[0, 0] == pytest.approx(-(K1 + 4 * K2 * X_SS + 9 * 0.4 * X_SS**2), rel=1e-7)


@pytest.mark.parametrize("x, kh", [(1e-3, 2e-3), (0.87, 0.5)])
def test_a_saturating_law_in_a_small_species(tmp_path, x, kh):
    """``0 -> D`` at ``VH·X⁴/(KH⁴ + X⁴)``, no polynomial in X.
    ∂f_D/∂X = 2·k2·X + 4·VH·KH⁴·X³/(KH⁴ + X⁴)²."""
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


# ─── A law with a scale of its own far below the first step ─────────────────

LAWS = """begin parameters
    1 s   1.3
    2 k1  0.21
    3 k2  0.73
    4 kd  0.5
    5 sg  {big!r}
    6 dg  1.0
    7 V   {v!r}
    8 K   {k!r}
end parameters
begin functions
    1 mm() V*Xo/(K+Xo)
    2 hill2() V*Xo^2/(K^2+Xo^2)
    3 grows() exp(Xo/0.05)
    4 peaks() V*Xo*exp(-Xo/K)
    5 slow() abs(Xo)^1.5
    6 step() if(Xo>0.5,1,0)
    7 spike() if(Xo=={k!r},1e308*10,0)
    8 kink() V*abs(Xo-K)
    9 ripple() V*sin(Xo/K)
   10 nearly() V*ln(1+Xo/K)
   11 inside() V*((K+Xo)-K)
end functions
begin species
    1 X() 0
    2 D() 0
    3 G() {big!r}
    4 E() 0
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


def _laws(tmp_path, big, more, v=1.0, k=1.0):
    """The model above with a law of X added: into X (``0 1``), D (``0 2``)
    or E (``0 4``), which nothing else is in."""
    path = tmp_path / "laws.net"
    path.write_text(LAWS.format(big=float(big), more=more, v=v, k=k))
    return bngsim.Model.from_net(path)


def _is_what_it_was(model, y, row):
    assert _fd(model, y)[row, 0] == _first_quotient(model, y, 0)[row]


@pytest.mark.parametrize("big", [1e8, 1e12])
def test_a_kink_just_above_the_species(tmp_path, big):
    """``0 -> D`` at ``V·|X − K|`` with K 2.96e-8 above X = 1.7e-3, which is
    1.7e-5 of X and a thousand of its own steps: ∂f_D/∂X = 2·k2·X − V. The
    first quotient is 1.09 beside 1e8 and 10878 beside 1e12. A ladder of
    halved steps under it stopped above the kink and returned the slope
    beyond it, 0.00374 for 0.0042060 in the reviewer's row."""
    x, v = 1.7e-3, 4.66e-4
    model = _laws(tmp_path, big, "    7 0 2 kink\n", v=v, k=x + 2.96e-8)
    got = _fd(model, [x, 0.0, big, 0.0])[1, 0]
    assert got == pytest.approx(2 * K2 * x - v, rel=1e-6)


def test_a_ripple_far_narrower_than_the_first_step(tmp_path):
    """``0 -> E`` at ``V·sin(X/K)`` with K 1e-4 of X: the slope is
    ``V/K·cos(X/K)``, 0.00505, the first quotient -1.3e-12, and a ladder
    took the slope of the row without the ripple, with the wrong sign."""
    x, k, v = 1.7e-3, 1.74e-7, 9.05e-10
    model = _laws(tmp_path, 2.95e10, "    7 0 4 ripple\n", v=v, k=k)
    got = _fd(model, [x, 0.0, 2.95e10, 0.0])[3, 0]
    assert got == pytest.approx(v / k * math.cos(x / k), rel=1e-4)


def test_a_hill_term_far_under_its_half_saturation(tmp_path):
    """``0 -> E`` at ``V·X²/(K² + X²)`` with K = 2000 and X = 1e-9: the
    slope is ``2·V·X/K²`` = 4.5e-16, in a row with nothing else in it. The
    first quotient was 3.4e-7."""
    model = _laws(tmp_path, 1e8, "    7 0 4 hill2\n", v=0.9, k=2000.0)
    want = 2 * 0.9 * 1e-9 / 2000.0**2
    assert _fd(model, [1e-9, 0.0, 1e8, 0.0])[3, 0] == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize(
    "more, v, k",
    [
        ("", 1.0, 1.0),  # the dimerization alone
        ("    7 0 2 mm\n", 4e-7, 1e-6),  # a slope of 0.4 under it
        ("    7 0 2 grows\n", 1.0, 1.0),
        ("    7 0 2 peaks\n", 1.5, 1e-3),
        ("    7 0 2 mm\n", 1e-19, 1e-19),  # a scale no step reaches
        ("    7 0 2 slow\n", 1.0, 1.0),
    ],
)
def test_a_law_of_a_species_at_zero_is_what_it_was(tmp_path, more, v, k):
    """Control. A species at exactly 0 has no step of its own, whatever law
    reads it: its column is the first quotient. (``X + 0.73·X² +
    1e-19·X/(1e-19 + X)`` has a slope of 2 there, the first quotient is
    2.09, and a ladder of 40 halvings under it returned 1.)"""
    model = _laws(tmp_path, 1e8, more, v=v, k=k)
    y = [0.0, 0.0, 1e8, 0.0]
    np.testing.assert_array_equal(_fd(model, y)[:, 0], _first_quotient(model, y, 0))


@pytest.mark.parametrize("big", [1e10, 1e12])
def test_an_entry_under_its_rows_rounding_at_the_own_step_is_what_it_was(tmp_path, big):
    """Control. ``0 -> D`` at ``V·X/(K + X)`` with X = 1e-3 and K = 1e-6: the
    slope is 0.90 and the first quotient 109 and 10878. The row is 0.9, and
    X's own step is 1.5e-11: the row's rounding over it is 5e-4 of the
    slope, which is not known there. This is what the floor on the step is
    for, and the entry stays what the floor gives."""
    model = _laws(tmp_path, big, "    7 0 2 mm\n", v=0.9, k=1e-6)
    _is_what_it_was(model, [1e-3, 0.0, big, 0.0], 1)


@pytest.mark.parametrize("x, d, row", [(1e-9, 0.2, 0), (1e-9, 0.2, 1), (1e-30, 0.0, 0)])
def test_a_species_next_to_zero_is_what_it_was(tmp_path, x, d, row):
    """Control. X at a billionth, and next to nothing, in rows whose terms
    are of order 1: a step of √eps of X moves them by less than their
    rounding."""
    _is_what_it_was(_laws(tmp_path, 1e8, ""), [x, d, 1e8, 0.0], row)


def test_a_law_that_is_nearly_linear_is_what_it_was(tmp_path):
    """Control. ``0 -> E`` at ``V·ln(1 + X/K)`` with K 8000 times X: the first
    quotient is 1.5e-5 off, and a ladder's late estimate, 2.7e-5 off, was
    taken for it."""
    model = _laws(tmp_path, 1e9, "    7 0 4 nearly\n", v=2e-3, k=4.97e5)
    y = [62.9, 0.0, 1e9, 0.0]
    got = _fd(model, y)[3, 0]
    want = 2e-3 / (4.97e5 + 62.9)
    assert abs(got - want) <= abs(_first_quotient(model, y, 0)[3] - want)
    assert got == pytest.approx(want, rel=2e-5)


@pytest.mark.parametrize("x", [X_SS, 3.3])
def test_a_rounding_inside_one_rate_law_is_what_it_was(tmp_path, x):
    """Control. ``0 -> E`` at ``V·((K + X) − K)`` with K = 1e8 is X, in steps
    of 1.5e-8, the last bit of 1e8, and its row is X: neither the row's
    terms nor a small move of the state shows that rounding. The quotients
    at X's own step, 1.3e-8, and at twice it are the same wrong number, 1.14
    for 1. A thousandth beyond the step the row has not moved at all, so it
    does not resolve the step, and the entry stays the first quotient,
    which is 1."""
    model = _laws(tmp_path, 1e8, "    7 0 4 inside\n", v=1.0, k=1e8)
    y = [x, 0.0, 1e8, 0.0]
    _is_what_it_was(model, y, 3)
    assert _fd(model, y)[3, 0] == pytest.approx(1.0, rel=1e-6)


def test_a_step_inside_the_first_step_is_what_it_was(tmp_path):
    """Control. ``if(X > 0.5, 1, 0)`` with X at 1e-3 and a step of 1.5: the
    first quotient is across the switch, and X's own step does not move the
    row at all. A row that a step does not move is not told from one whose
    rounding hides the move, so the entry stays."""
    model = _laws(tmp_path, 1e8, "    7 0 4 step\n")
    _is_what_it_was(model, [1e-3, 0.0, 1e8, 0.0], 3)


def test_a_row_that_is_no_number_at_the_own_step_is_what_it_was(tmp_path):
    """Control. X's own row with a term that is infinite at one value of X,
    its own step above where it is, and nothing anywhere else."""
    x = 0.5
    own = x + SQRT_EPS * x
    model = _laws(tmp_path, 1e8, "    7 0 1 spike\n", k=own)
    y = [x, 0.0, 1e8, 0.0]
    assert not np.isfinite(np.asarray(model.rhs(np.array([own, 0.0, 1e8, 0.0]), 0.0))[0])
    _is_what_it_was(model, y, 0)


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
    """Control. G gains ``kw·X`` (reaction 7), 1e-4 a unit of X, beside
    fluxes of 1e8, with G at an eighth of where it settles and no round
    number: its row is 8.8e7 and does not cancel. (At a round G the step,
    2^-26 of it, moves the row by a whole number of its last bits, and the
    quotients carry no rounding at all.) The quotients at the step and at
    half of it differ by the rounding of that row, 8e-4 of the entry, which
    is nothing the entry was not known to already: it is the first quotient."""
    model = _model(tmp_path, 1e8, more="    7 1 1,3 kw\n")
    y = _state(1e8)
    y[2] = 12345678.9
    jac = _fd(model, y)
    first = _first_quotient(model, y, 0)
    half = y.copy()
    half[0] = y[0] + (SQRT_EPS * y[2]) / 2
    second = (np.asarray(model.rhs(half, 0.0)) - np.asarray(model.rhs(y, 0.0)))[2] / (
        half[0] - y[0]
    )
    assert abs(second - first[2]) > 1e-6 * abs(first[2])
    assert jac[2, 0] == first[2]
    assert jac[2, 0] == pytest.approx(1e-4, rel=1e-2)


def test_a_row_with_a_large_term_no_species_is_in(tmp_path):
    """Control. D is made at 1e8 (reaction 7), which no species is in, and
    the state is not a steady one. That flux is in the row's rounding, and
    the entry of a species at 1e-6 in it is the first quotient."""
    model = _model(tmp_path, 1.0, more="    7 0 2 kbig\n")
    y = np.array([1e-6, 0.0, 1.0])
    assert _fd(model, y)[1, 0] == _first_quotient(model, y, 0)[1]


@pytest.mark.parametrize("x", [0.0, 1e-3])
def test_a_law_that_is_not_smooth_at_the_species_is_what_it_was(tmp_path, x):
    """Control. ``0 -> D`` at ``sqrt(abs(X - xc))`` with X on ``xc``: the
    quotient is ``h^(-1/2)``, which has no limit. At twice X's own step it
    is 0.71 of what it is at the step, so the entry is the first quotient;
    at X = 0 there is no own step to ask."""
    model = _model(tmp_path, 1e8, more="    7 0 2 root\n", xc=x)
    y = _state(1e8, x=x)
    assert _fd(model, y)[1, 0] == _first_quotient(model, y, 0)[1]


SHARE = """begin parameters
    1 r      0.99
    2 delta  0.0001
    3 g      1.0
    4 alpha  1e-7
    5 p      0.3
end parameters
begin functions
    1 phi() ((r-delta)*Lo+g*Uo-alpha*Uo*Vo*(1-p))/(Lo+Uo)
    2 made() r*Lo+p*alpha*Uo*Vo
    3 lost() delta+phi()
end functions
begin species
    1 L() 1e6
    2 U() 4.3e-9
    3 V() 1.57e5
end species
begin reactions
    1 0 1 made
    2 1 0 lost
end reactions
begin groups
    1 Lo 1
    2 Uo 2
    3 Vo 3
end groups
"""


def test_a_row_that_cancels_in_its_value_and_in_its_entries(tmp_path):
    """Control. The lysogens of BIOMD0000000884: L is made at
    ``r·L + p·alpha·U·V`` and lost at ``(delta + phi)·L`` with
    ``phi = ((r - delta)·L + g·U - alpha·U·V·(1 - p))/(L + U)``. With U next
    to nothing, terms of 1e6 cancel in the row's value and in its entry for
    L, so neither says how large the row's rounding is: 1e-10. The entry of U
    is ``alpha·V - g + r - delta``, 0.0056, and a quotient of the row at a
    small step is all rounding (0.023 at a step of 1e-8), then exactly 0.
    The entry of a species the floor steps by many times itself is taken
    from its own step only where the row's rounding there is small: it is
    measured, 1e-10, and the entry stays the first quotient. The state is
    the model's at t = 100."""
    path = tmp_path / "share.net"
    path.write_text(SHARE)
    model = bngsim.Model.from_net(path)
    y = np.array([999999.9999999902, 4.312687078760201e-09, 157089.2524458467])
    jac = _fd(model, y)
    assert jac[0, 1] == _first_quotient(model, y, 1)[0]
    assert jac[0, 1] == pytest.approx(1e-7 * y[2] - 1.0 + 0.99 - 0.0001, rel=1e-4)


CANCEL = """begin parameters
    1 k 500
    2 c 1e-7
end parameters
begin species
    1 X() 0
    2 A() 0.7
    3 B() 0.1
    4 Y() 7.0
end species
begin reactions
    1 1,2 1,2,4 k
    2 1,3,4 1,3 k
    3 1 1,4 c
end reactions
"""


def test_a_row_whose_terms_cancel_in_proportion_to_the_stepped_species(tmp_path):
    """Control. Y is made at ``k·X·A + c·X`` and lost at ``k·X·B·Y``, with
    ``A = B·Y``: two terms of 350·X that cancel, and an entry for X of ``c``,
    1e-7. The quotients at the floored step and at half of it differ by the
    rounding of those terms, 2e-6 of the entry, and X's own step, 4.5e-21,
    moves the row by less than that rounding: the entry is the first
    quotient. (A ladder under the floored step read the scatter as curvature
    and took an estimate 3e-6 off, where the first quotient is 6e-7 off:
    BIOMD0000000499 at t = 100.)"""
    path = tmp_path / "cancel.net"
    path.write_text(CANCEL)
    model = bngsim.Model.from_net(path)
    y = np.array([3e-13, 0.7, 0.1, 7.0])
    first = _first_quotient(model, y, 0)
    half = y.copy()
    half[0] = y[0] + (SQRT_EPS * 7.0) / 2
    second = (np.asarray(model.rhs(half, 0.0)) - np.asarray(model.rhs(y, 0.0)))[3] / (
        half[0] - y[0]
    )
    assert abs(second - first[3]) > 1e-6 * abs(first[3])
    assert _fd(model, y)[3, 0] == first[3]
    assert first[3] == pytest.approx(1e-7, rel=1e-5)


def test_two_quotients_that_agree_to_a_millionth_are_one(tmp_path):
    """Control. Beside a species a hundred times its size, X is stepped by
    1.5e-6 of itself, and the first quotient of its dimerization is 7e-7 of
    the entry from the one at X's own step: no further than that one is
    known to, so it stays."""
    big = 100 * X_SS
    model = _model(tmp_path, big)
    y = _state(big)
    assert _fd(model, y)[0, 0] == _first_quotient(model, y, 0)[0]


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
