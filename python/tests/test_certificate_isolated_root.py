"""The stability certificate reads an isolated eigenvalue exactly (issue #1529).

A ``$``-fixed source S feeds X at k = 1e7, X decays at 1 and is read by
nothing, and S also feeds a pair Y <-> Z. S = 1, X = 1e7, Y = 2, Z = 1 is an
exact root, and the Jacobian is block lower triangular: its eigenvalues are
exactly 0 (S's zero row), -1 and (-3 +- sqrt 5)/2, all in the closed left
half-plane, so the root is "stable".

``steady_state(method="newton")`` called it "unstable", with a ``+6.9e-05`` for
the 0 and ``-1.000069`` for -1 (``+3.1e-05`` on another libm). The eigensolver's
balancing skipped S's isolated row without moving it, and the zero's condition
number is k. It now isolates such rows and columns as LAPACK's ``dgebal`` does,
and reads their eigenvalues exactly. The label depended on the roundoff's sign;
the spectrum is what these tests pin.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 k {k}
    2 one 1.0
end parameters
begin species
    1 $S() 1.0
    2 X() {k}
    3 Y() 2.0
    4 Z() 1.0
end species
begin reactions
    1 1 1,2 k
    2 2 0 one
    3 1 1,3 one
    4 3 4 one
    5 4 3 one
    6 4 0 one
end reactions
"""
GOLDEN = ((-3 - math.sqrt(5)) / 2, (-3 + math.sqrt(5)) / 2)
EXACT = np.array(sorted([0.0, -1.0, *GOLDEN]))
TOL = 1e-12 * abs(GOLDEN[0])  # the spectral radius is 2.618


def _model(tmp_path: Path, k: float = 1e7, *, sibling: bool = False) -> bngsim.Model:
    text = NET.format(k=repr(k))
    if sibling:
        # No $ species: S free, decaying at 1e-9, everything at the root 0.
        text = (
            text.replace("$S() 1.0", "S() 0.0")
            .replace(f"X() {k!r}", "X() 0.0")
            .replace("Y() 2.0", "Y() 0.0")
            .replace("Z() 1.0", "Z() 0.0")
            .replace("    2 one 1.0\n", "    2 one 1.0\n    3 eps 1e-9\n")
            .replace("    6 4 0 one\n", "    6 4 0 one\n    7 1 0 eps\n")
        )
    path = tmp_path / ("sib.net" if sibling else "m.net")
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


def _spectrum(result) -> np.ndarray:
    return np.sort(np.asarray(result.eigenvalues).real)


def test_the_issue_s_root_is_stable_with_its_exact_spectrum(tmp_path):
    result = bngsim.Simulator(_model(tmp_path), method="ode").steady_state(method="newton")
    assert result.converged
    np.testing.assert_allclose(_spectrum(result), EXACT, rtol=0, atol=TOL)
    assert result.root_stability == "stable"


def test_numpy_agrees_with_the_certificate(tmp_path):
    """The cross-check the ``eigenvalues`` docstring names."""
    model = _model(tmp_path)
    y = np.asarray(model.get_state())
    reference = np.sort(np.linalg.eigvals(model.jacobian(y)).real)
    result = bngsim.Simulator(model, method="ode").steady_state(method="newton")
    np.testing.assert_allclose(_spectrum(result), reference, rtol=0, atol=TOL)


def test_integrating_then_seeding_newton_reaches_the_same_certificate(tmp_path):
    """The ordinary workflow, under the default criterion."""
    model = _model(tmp_path)
    model.set_state(np.array([1.0, 0.0, 0.0, 0.0]))
    marched = bngsim.Simulator(model, method="ode").steady_state(method="integration")
    seeded = _model(tmp_path)
    seeded.set_state(np.asarray(marched.concentrations))
    result = bngsim.Simulator(seeded, method="ode").steady_state(method="newton")
    np.testing.assert_allclose(_spectrum(result), EXACT, rtol=0, atol=TOL)
    assert result.root_stability == "stable"


@pytest.mark.parametrize("k", [1e3, 1e5, 63095734.44801943, 1e9, 1e12])
def test_the_spectrum_is_exact_for_every_k(tmp_path, k):
    """At k = 6.31e7 the label was right by luck and -1 read as -0.968: 3%.
    At k = 1e3 main's error is under the tolerance, so that case is a control."""
    result = bngsim.Simulator(_model(tmp_path, k), method="ode").steady_state(method="newton")
    np.testing.assert_allclose(_spectrum(result), EXACT, rtol=0, atol=TOL)
    assert result.root_stability == "stable"


def test_a_batch_seeded_at_the_root_certifies_it_stable(tmp_path):
    """The start-at-a-root branch, through ``steady_state_batch``."""
    model = _model(tmp_path)
    sim = bngsim.Simulator(model, method="ode")
    (entry,) = sim.steady_state_batch(states=[np.asarray(model.get_state())], method="newton")
    np.testing.assert_allclose(_spectrum(entry), EXACT, rtol=0, atol=TOL)
    assert entry.root_stability == "stable"


def test_steady_states_still_calls_it_stable(tmp_path):
    """Control: passes on main. ``steady_states`` leaves the fixed S out of
    its unknowns, so it never had the zero row."""
    roots = bngsim.Simulator(_model(tmp_path), method="ode").steady_states(n_starts=5, seed=0)
    assert list(roots.stability) == ["stable"]


# ── With no fixed species ────────────────────────────────────────────────────


def test_the_sibling_s_slow_decay_is_read_exactly(tmp_path):
    """S free, decaying at 1e-9: its row is still isolated, and -1e-9 comes
    back as the diagonal it is, nine decades under the radius."""
    result = bngsim.Simulator(_model(tmp_path, sibling=True), method="ode").steady_state(
        method="newton"
    )
    spectrum = _spectrum(result)
    np.testing.assert_allclose(spectrum, sorted([-1e-9, -1.0, *GOLDEN]), rtol=0, atol=TOL)
    assert spectrum[-1] == pytest.approx(-1e-9, rel=1e-9)
    assert result.root_stability == "stable"


def test_the_sibling_s_sensitivities_are_not_refused(tmp_path):
    """The #995 refusal quoted the spurious positive eigenvalue, where the
    roundoff came out positive: on the issue's Linux libm main refused this
    with "an eigenvalue with a real part of 6.9e-05"; on macOS's, main's error
    was -1.03e-04 and it passed. dY_ss/dp exists here; it is 0, the root being
    0 for every k."""
    result = bngsim.Simulator(_model(tmp_path, sibling=True), method="ode").steady_state(
        sensitivity_params=["k", "one"]
    )
    assert result.converged
    np.testing.assert_allclose(np.asarray(result.sensitivity), 0.0, atol=1e-12)
