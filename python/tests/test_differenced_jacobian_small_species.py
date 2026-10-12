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

An entry changes only where the ladder shows what it is: two estimates agree
to a millionth of themselves, the row's rounding is under 1e-4 of them, every
estimate after them stays there, down to the species' own step or to where
the rounding takes over, and the estimate is further from the first quotient
than it is known to itself. A first cut asked the agreement to a millionth of
the first quotient and stopped at it. Beside a species at 1e9, a dimerization
and ``V·X/(K + X)`` with K at 1e-6 came back with 1.9e-5 for a slope of 40,
where the first quotient had been 10.9. A slope that is 0, or under the row's
rounding, is not told from one that is not, and stays the first quotient.

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


def test_a_term_of_third_order(tmp_path):
    """``3 X -> 0`` at k3 is of third order in X, which a ladder of four
    steps is exact for. ∂f_X/∂X = -(k1 + 4·k2·X + 9·k3·X²)."""
    model = _model(tmp_path, 1e8, more="    7 1,1,1 0 k3\n")
    jac = _fd(model, _state(1e8))
    assert jac[0, 0] == pytest.approx(-(K1 + 4 * K2 * X_SS + 9 * 0.4 * X_SS**2), rel=1e-9)


@pytest.mark.parametrize("x, kh", [(1e-3, 2e-3), (0.87, 0.5)])
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


@pytest.mark.parametrize("big, v", [(1e8, 4e-7), (1e9, 4e-5)])
def test_a_saturating_law_under_a_dimerization(tmp_path, big, v):
    """``0 -> D`` at ``V·X/(K + X)`` with K = 1e-6, beside the dimerization,
    at X = 0: ∂f_D/∂X = V/K, 0.4 and 40. To every step above K the law is
    flat, and the estimates agree on the dimerization's slope, 0. The first
    quotient was 1.09 and 10.9; an entry taken where the estimates first
    agreed, to a millionth of that quotient, was 1.9e-6 and 1.9e-5."""
    model = _laws(tmp_path, big, "    7 0 2 mm\n", v=v, k=1e-6)
    y = np.array([0.0, 0.0, big, 0.0])
    assert _fd(model, y)[1, 0] == pytest.approx(v / 1e-6, rel=1e-6)


def test_a_law_that_grows_as_an_exponential(tmp_path):
    """``0 -> D`` at ``exp(X/0.05)`` at X = 0: the slope is 20, the first
    quotient across a step of 1.5 is 5.9e12, and an entry taken where two
    estimates were within a millionth of that was 315."""
    model = _laws(tmp_path, 1e8, "    7 0 2 grows\n")
    assert _fd(model, [0.0, 0.0, 1e8, 0.0])[1, 0] == pytest.approx(20.0, rel=1e-6)


@pytest.mark.parametrize("big", [1e10, 1e12])
def test_a_saturating_law_beside_a_very_large_species(tmp_path, big):
    """X = 1e-3 with K = 1e-6 and steps of 150 and 15,000: the law's slope
    at X, ``V·K/(K + X)²`` = 0.898, is seen 17 and 24 halvings down. It was
    lost: 0.0015, the dimerization's alone."""
    x, v, k = 1e-3, 0.9, 1e-6
    model = _laws(tmp_path, big, "    7 0 2 mm\n", v=v, k=k)
    want = 2 * K2 * x + v * k / (k + x) ** 2
    assert _fd(model, [x, 0.0, big, 0.0])[1, 0] == pytest.approx(want, rel=1e-6)


def test_a_law_that_peaks_under_the_first_step(tmp_path):
    """``V·X·exp(-X/K)`` with K = 1e-3 is nothing at every step above 0.05,
    and its slope at 0 is V."""
    model = _laws(tmp_path, 1e8, "    7 0 2 peaks\n", v=1.5, k=1e-3)
    assert _fd(model, [0.0, 0.0, 1e8, 0.0])[1, 0] == pytest.approx(1.5, rel=1e-6)


@pytest.mark.parametrize("d", [0.0, 0.3])
def test_a_slope_under_a_plateau(tmp_path, d):
    """``0 -> X`` at ``V·X/(K + X)`` with V = 1e-7 and K = 1e-6, at X = 0:
    ∂f_X/∂X = -k1 + V/K = -0.11. Down to steps of 0.1 the estimates are at
    -0.21 to a millionth, where the law is flat, and they leave it below. An
    entry is the estimate that every later one stays at."""
    model = _laws(tmp_path, 1e8, "    7 0 1 mm\n", v=1e-7, k=1e-6)
    assert _fd(model, [0.0, d, 1e8, 0.0])[0, 0] == pytest.approx(-K1 + 0.1, rel=1e-5)


def test_a_hill_term_far_under_its_half_saturation(tmp_path):
    """``0 -> E`` at ``V·X²/(K² + X²)`` with K = 2000 and X = 1e-9: the
    slope is ``2·V·X/K²`` = 4.5e-16, in a row with nothing else in it. The
    first quotient was 3.4e-7."""
    model = _laws(tmp_path, 1e8, "    7 0 4 hill2\n", v=0.9, k=2000.0)
    want = 2 * 0.9 * 1e-9 / 2000.0**2
    assert _fd(model, [1e-9, 0.0, 1e8, 0.0])[3, 0] == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("x, d", [(1e-9, 0.2), (1e-9, 0.0), (1e-30, 0.0)])
def test_a_species_next_to_zero_that_dimerizes(tmp_path, x, d):
    """∂f_X/∂X = -(k1 + 4·k2·X), with X a billionth of a unit and next to
    nothing: a ladder that goes all the way down."""
    jac = _fd(_laws(tmp_path, 1e8, ""), [x, d, 1e8, 0.0])
    assert jac[0, 0] == pytest.approx(-(K1 + 4 * K2 * x), rel=1e-9)


def test_the_slope_of_a_dimerization_at_a_billionth(tmp_path):
    """∂f_D/∂X = 2·k2·X = 1.46e-9, in a row that holds ``kd·D`` = 0.1 and so
    has a rounding of its own. The first quotient was 1.09."""
    jac = _fd(_laws(tmp_path, 1e8, ""), [1e-9, 0.2, 1e8, 0.0])
    assert jac[1, 0] == pytest.approx(2 * K2 * 1e-9, rel=1e-5)


def _first_or_right(model, y, row, want):
    """An entry is what it was, or it is the slope."""
    got, was = _fd(model, y)[row, 0], _first_quotient(model, y, 0)[row]
    assert got == was or got == pytest.approx(want, rel=1e-3), (got, was, want)
    return got == was


@pytest.mark.parametrize("d", [0.3, 0.0])
def test_a_slope_of_zero_is_what_it_was(tmp_path, d):
    """Control. The dimerization into D at X = 0 has a slope of exactly 0,
    and its estimates are 0 to the rounding of the quotients they are made
    of. A slope of 1e-33, which a product with another species next to 0
    gives, reads the same, so the entry is not changed: it is the secant,
    1.09, as it was. With D at 0.3, where the row has a rounding of its own,
    and with D at 0."""
    model = _laws(tmp_path, 1e8, "")
    y = [0.0, d, 1e8, 0.0]
    assert _fd(model, y)[1, 0] == _first_quotient(model, y, 0)[1]
    assert _fd(model, y)[1, 0] == pytest.approx(K2 * SQRT_EPS * 1e8, rel=1e-6)


def test_a_hill_term_at_a_species_at_zero_is_what_it_was(tmp_path):
    """Control. ``0 -> D`` at ``VH·X⁴/(KH⁴ + X⁴)`` with KH = 1e-4, at X = 0:
    the slope is 0, and the estimates fall towards it without coming to
    rest on a value."""
    model = _model(tmp_path, 1e8, more="    7 0 2 hill\n", kh=1e-4)
    y = _state(1e8, x=0.0)
    assert _fd(model, y)[1, 0] == _first_quotient(model, y, 0)[1]


def test_a_slope_under_the_rows_rounding_is_what_it_was(tmp_path):
    """Control. ``0 -> X`` at ``V·X/(K + X)`` with V = 4e-9 and K = 1e-8:
    the slope -k1 + 0.4 is seen at steps of 1e-10, where the rounding of the
    row is 1e-4 of it. The estimates leave -0.21 and come to rest nowhere."""
    model = _laws(tmp_path, 1e8, "    7 0 1 mm\n", v=4e-9, k=1e-8)
    assert _first_or_right(model, [0.0, 0.0, 1e8, 0.0], 0, -K1 + 0.4)


def test_a_power_of_three_halves_at_zero_is_what_it_was(tmp_path):
    """Control. ``|X|^1.5`` has a slope of 0 at 0, and its quotient falls as
    the square root of the step: 1e-6 of the first at the ladder's end, which
    is no 0."""
    model = _laws(tmp_path, 1e8, "    7 0 4 slow\n")
    assert _first_or_right(model, [0.0, 0.0, 1e8, 0.0], 3, 0.0)


def test_a_step_inside_the_first_step_is_what_it_was(tmp_path):
    """Control. ``if(X > 0.5, 1, 0)`` with X at 1e-3 and a step of 1.5: the
    first quotient is across the switch, and every step under 0.5 leaves the
    row where it was, to the last bit."""
    model = _laws(tmp_path, 1e8, "    7 0 4 step\n")
    assert _first_or_right(model, [1e-3, 0.0, 1e8, 0.0], 3, 0.0)


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
    quotient is ``h^(-1/2)``, which has no limit, and no two estimates of
    the ladder agree. An entry changes only where the ladder settles, so
    this one is the first quotient."""
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
    With the rounding taken from the row's value and entries, a first cut of
    the ladder met those zeros and took one for the entry: 1e-11 for 0.0056.
    It is measured, and the entry stays the first quotient. The state is the
    model's at t = 100."""
    path = tmp_path / "share.net"
    path.write_text(SHARE)
    model = bngsim.Model.from_net(path)
    y = np.array([999999.9999999902, 4.312687078760201e-09, 157089.2524458467])
    jac = _fd(model, y)
    assert jac[0, 1] == _first_quotient(model, y, 1)[0]
    assert jac[0, 1] == pytest.approx(1e-7 * y[2] - 1.0 + 0.99 - 0.0001, rel=1e-4)


def test_two_quotients_that_agree_to_a_millionth_are_one(tmp_path):
    """Control. Beside a species a hundred times its size, X is stepped by
    1.5e-6 of itself, and the two quotients of its dimerization differ by
    3.4e-7 of the entry: under the millionth an entry is asked to, so it is
    the first quotient, as it was."""
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
