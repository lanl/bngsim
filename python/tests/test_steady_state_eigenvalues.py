"""``SteadyStateResult.eigenvalues`` (issue #523).

The issue #78 stability certificate computes the spectrum of the Jacobian
restricted to the species the Newton polish solved for, reads one bit off it
(is any real part positive?) and threw the rest away. A continuation needs
that spectrum along the branch — a Hopf bifurcation is a conjugate pair
crossing the imaginary axis — so the result now carries it.

These tests grade the field against numpy's own eigensolver on
``Model.jacobian`` at the returned root, not against the certificate, so they
do not check the solver against itself.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

_NETS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"
_GARDNER = "genetic_switch.net"
# Point 3 of the model's own alpha_2 scan, and the saddle it has there (see
# test_steady_state_gh78.py for where both numbers come from).
_SADDLE_DOSE = 53.526315789473685
_SADDLE = np.array([28.245215769071677, 1.8302588776274484])


def _net(name: str) -> str:
    path = _NETS_DIR / name
    if not _NETS_DIR.is_dir():
        pytest.skip(f"benchmark net corpus not available: {_NETS_DIR}")
    assert path.is_file(), f"vendored net missing from tracked corpus: {path}"
    return str(path)


def _sorted_like_the_result(ev: np.ndarray) -> np.ndarray:
    """Descending real part, then descending imaginary — the result's order."""
    return ev[np.lexsort((-ev.imag, -ev.real))]


def test_integration_result_carries_no_spectrum(data_dir):
    model = bngsim.Model.from_net(str(data_dir / "simple_decay.net"))
    ss = bngsim.Simulator(model, method="ode").steady_state(method="integration")
    assert ss.root_stability == ""
    assert ss.eigenvalues.dtype == np.complex128
    assert ss.eigenvalues.shape == (0,)


def test_one_unknown_one_eigenvalue(data_dir):
    """A -> B conserves A + B: one unknown, whose Jacobian is the scalar -k1."""
    model = bngsim.Model.from_net(str(data_dir / "simple_decay.net"))
    ss = bngsim.Simulator(model, method="ode").steady_state(method="newton")
    assert ss.method_used == "newton"
    assert ss.root_stability == "stable"
    assert ss.eigenvalues.dtype == np.complex128
    np.testing.assert_allclose(ss.eigenvalues, [-0.1 + 0j], rtol=1e-12)


def test_restricted_spectrum_is_the_full_one_less_a_zero_per_law(data_dir):
    """A + B <-> C: two laws over three species, so the certificate sees one
    unknown. The full Jacobian's spectrum is that eigenvalue plus two zeros."""
    model = bngsim.Model.from_net(str(data_dir / "two_species_reversible.net"))
    ss = bngsim.Simulator(model, method="ode").steady_state(method="newton")
    assert ss.method_used == "newton"
    assert model.conservation_laws["n_laws"] == 2
    assert ss.eigenvalues.shape == (1,)
    assert ss.eigenvalues[0].real < 0

    full = np.linalg.eigvals(model.jacobian(ss.concentrations))
    nonzero = full[np.abs(full) > 1e-9]
    assert nonzero.shape == (1,)
    np.testing.assert_allclose(ss.eigenvalues, nonzero, rtol=1e-8)


def test_saddle_reports_its_positive_eigenvalue():
    """Seeded AT the Gardner saddle the solve returns it as "unstable" — and now
    says by how much. No conservation laws, so this is the full spectrum."""
    model = bngsim.Model.from_net(_net(_GARDNER))
    model.set_param("alpha_2", _SADDLE_DOSE)
    model.set_state(_SADDLE)
    ss = bngsim.Simulator(model, method="ode").steady_state(method="newton")
    assert ss.method_used == "newton"
    assert ss.root_stability == "unstable"
    assert ss.eigenvalues.shape == (2,)
    assert ss.eigenvalues[0].real > 0 > ss.eigenvalues[1].real, "sorted by descending real part"

    expect = _sorted_like_the_result(np.linalg.eigvals(model.jacobian(ss.concentrations)))
    np.testing.assert_allclose(ss.eigenvalues, expect, rtol=1e-8, atol=1e-12)


def test_rejected_saddle_leaves_the_attractor_spectrum():
    """From the model's own initial condition the polish lands on the saddle,
    the certificate throws it out, and the accepted root's spectrum is the
    attractor's — every real part negative."""
    model = bngsim.Model.from_net(_net(_GARDNER))
    model.set_param("alpha_2", _SADDLE_DOSE)
    ss = bngsim.Simulator(model, method="ode").steady_state(method="newton")
    assert ss.converged
    assert ss.method_used == "newton"  # the polish lands after the rejection (GH #78)
    assert ss.root_stability == "stable"
    assert ss.n_unstable_roots_rejected >= 1
    assert ss.eigenvalues.shape == (2,)
    assert np.all(ss.eigenvalues.real < 0)
    expect = _sorted_like_the_result(np.linalg.eigvals(model.jacobian(ss.concentrations)))
    np.testing.assert_allclose(ss.eigenvalues, expect, rtol=1e-8, atol=1e-12)


def test_hopf_focus_reports_its_conjugate_pair(data_dir):
    """Goldbeter 1991 (BIOMD0000000003) oscillates; its steady state is an
    unstable focus. Seeded there, the result carries the pair — adjacent, the
    +imaginary member first — which is the crossing a Hopf detector watches."""
    pytest.importorskip("libsbml")
    fsolve = pytest.importorskip("scipy.optimize").fsolve
    model = bngsim.Model.from_sbml(str(data_dir / "BIOMD0000000003.xml"))
    root, _info, ier, _msg = fsolve(lambda y: model.rhs(y), model.get_state(), full_output=True)
    assert ier == 1
    assert np.abs(model.rhs(root)).max() < 1e-10

    model.set_state(root)
    ss = bngsim.Simulator(model, method="ode").steady_state(method="newton")
    assert ss.method_used == "newton"
    assert ss.root_stability == "unstable"
    assert ss.eigenvalues.shape == (3,)
    pair = ss.eigenvalues[:2]
    assert pair[0].imag > 0 > pair[1].imag
    np.testing.assert_allclose(pair[0], np.conj(pair[1]), rtol=1e-10)
    assert pair[0].real > 0

    expect = _sorted_like_the_result(np.linalg.eigvals(model.jacobian(ss.concentrations)))
    np.testing.assert_allclose(ss.eigenvalues, expect, rtol=1e-8, atol=1e-12)


def test_declined_certificate_leaves_it_empty(tmp_path):
    """Above 512 unknowns the certificate declines: "undetermined", and no
    spectrum rather than a partial or stale one. A 520-species decay chain
    conserves its total, so 519 unknowns."""
    n = 520
    lines = ["begin parameters", "    1 k 1.0", "end parameters", "begin species"]
    lines += [f"    {i} S{i}() {100.0 if i == 1 else 0.0}" for i in range(1, n + 1)]
    lines += ["end species", "begin reactions"]
    lines += [f"    {i} {i} {i + 1} k" for i in range(1, n)]
    lines += ["end reactions"]
    path = tmp_path / "chain.net"
    path.write_text("\n".join(lines) + "\n")

    model = bngsim.Model.from_net(str(path))
    assert model.n_species - model.conservation_laws["n_laws"] > 512
    ss = bngsim.Simulator(model, method="ode").steady_state(method="newton")
    assert ss.converged
    assert ss.method_used == "newton"
    assert ss.root_stability == "undetermined"
    assert ss.eigenvalues.shape == (0,)
