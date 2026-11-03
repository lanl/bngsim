"""The dense Newton solve reads a row the matrix holds alone as its own equation.

Where row i of the Newton matrix has no nonzero off its diagonal, its component
of the solve is fixed by that row alone, x_i = b_i / a_ii, and for a species held
at exactly 0 (b_i = 0) it is exactly 0. Partial pivoting does not see that: where
column i has a larger entry in another row, that row becomes the pivot, and the
back-solve leaves the rounding of sums that cancel in x_i. On an exact state
that is unstable, the residue grows.

MODEL1812040005's P is a rate-rule target whose law is P*g: it starts at 0 and
stays there, but g > 0 on t in [1.7, 9.5], where a departure grows by 4e26. V's
rate reads sqrt(P), so the matrix has a huge entry in P's column on V's row.
With the species in the file's order P's column comes last and P stays at 0;
reversed (the path-agreement check's ``reorder``), P's column comes first, V's
row is the pivot, and the residue left in P grew to 2.6e8, 25.6 times the
trajectory's peak, on x86-64. On arm64 it came out negative, and #1340's band
held P at -8e-12. KLU, whose block-triangular form takes such a row on its own,
keeps P at 0 in either order, as libRoadRunner does.

Oracles: the closed form of a two-species model whose P = 0 is unstable from the
start and whose V row reads P with a slope of 1e8; and, for MODEL1812040005 and
BIOMD0000000908, the unstable species at 0 exactly (its law is a multiple of
itself, so a start at 0 stays there) with the rest as the file's order gives it.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

_needs_klu = pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled")


def _unstable_zero(order: str) -> bngsim.Model:
    """P' = P (P = 0 unstable from t = 0) and V' = 1 - V + 1e8*P from V = 0: in
    exact arithmetic P = 0 and V = 1 - exp(-t). V's row reads P with a slope of
    1e8, so where P's column comes first V's row is its pivot."""
    decl = "V = 0, P = 0" if order == "VP" else "P = 0, V = 0"
    return bngsim.Model.from_antimony_string(
        f"model m\n  species {decl};\n  k = 1; c = 1e8;\n"
        "  J1: -> P; k*P;\n  J2: -> V; 1 + c*P;\n  J3: V -> ; V;\nend"
    )


def _species(model, run, **simulator) -> dict[str, np.ndarray]:
    r = bngsim.Simulator(model, method="ode", **simulator).run(**run)
    names = list(r.species_names)
    return {n: np.asarray(r.species)[:, names.index(n)] for n in names} | {"t": np.asarray(r.time)}


_UNSTABLE = {"t_span": (0, 40), "n_points": 41, "rtol": 1e-8, "atol": 1e-10}


@pytest.mark.parametrize("order", ["VP", "PV"])
def test_a_species_held_at_zero_stays_there_in_either_order(order):
    model = _unstable_zero(order)
    assert list(model.species_names) == list(order)
    x = _species(model, _UNSTABLE, force_dense_linear_solver=True)
    np.testing.assert_array_equal(x["P"], 0.0)
    np.testing.assert_allclose(x["V"], 1.0 - np.exp(-x["t"]), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("order", ["VP", "PV"])
@pytest.mark.parametrize("jacobian", ["analytical", "fd"])
def test_each_jacobian_route_reads_it(order, jacobian):
    x = _species(
        _unstable_zero(order), _UNSTABLE, force_dense_linear_solver=True, jacobian=jacobian
    )
    np.testing.assert_array_equal(x["P"], 0.0)


@pytest.mark.parametrize("order", ["VP", "PV"])
def test_a_run_with_sensitivities_reads_it(order):
    r = bngsim.Simulator(
        _unstable_zero(order),
        method="ode",
        force_dense_linear_solver=True,
        sensitivity_params=["c"],
    ).run(**_UNSTABLE)
    names = list(r.species_names)
    np.testing.assert_array_equal(np.asarray(r.species)[:, names.index("P")], 0.0)


@_needs_klu
@pytest.mark.parametrize("order", ["VP", "PV"])
def test_dense_now_agrees_with_klu_to_the_bit_on_p(order):
    """The control: KLU kept P at 0 before the fix as well."""
    x = _species(_unstable_zero(order), _UNSTABLE, force_sparse_linear_solver=True)
    np.testing.assert_array_equal(x["P"], 0.0)


def _reversed(src: Path, out: Path) -> Path:
    libsbml = pytest.importorskip("libsbml")
    doc = libsbml.readSBMLFromFile(str(src))
    m = doc.getModel()
    for lst in (m.getListOfSpecies(), m.getListOfParameters()):
        items = [lst.remove(0) for _ in range(lst.size())]
        for item in reversed(items):
            lst.appendAndOwn(item)
    libsbml.writeSBMLToFile(doc, str(out))
    return out


# The path-agreement nightly's window for this model.
_M18 = {"t_span": (0, 100), "n_points": 101, "rtol": 1e-6, "atol": 1e-9}


def test_model1812040005_keeps_p_at_zero_in_either_order(data_dir: Path, tmp_path: Path):
    src = data_dir / "MODEL1812040005.xml"
    rev = _reversed(src, tmp_path / "reversed.xml")
    file_order = _species(bngsim.Model.from_sbml(str(src)), _M18, force_dense_linear_solver=True)
    reversed_order = _species(
        bngsim.Model.from_sbml(str(rev)), _M18, force_dense_linear_solver=True
    )
    np.testing.assert_array_equal(file_order["P"], 0.0)
    np.testing.assert_array_equal(reversed_order["P"], 0.0)
    # The rest is the same trajectory, apart by rounding: within 1e-6 of each
    # species' own peak (V peaks at 5.3e6; the wrong answer peaked at 9.2e6).
    for name in ("T", "I1", "I2", "V"):
        a, b = file_order[name], reversed_order[name]
        assert np.abs(a - b).max() <= 1e-6 * np.abs(a).max() + _M18["atol"], name


def test_biomd0000000908_keeps_s_at_zero_in_either_order(data_dir: Path, tmp_path: Path):
    """dePillis2013: sunitinib S is never dosed, and its clearance law, -n*S on a
    reaction that consumes S, makes S' = +0.277*S, so S = 0 is exact and
    unstable. Reversed, the residue left in S grew to 1.09e134 of the peak on
    the Linux x86-64 nightly, and failed the run near t = 150 (CV_CONV_FAILURE)
    on macOS, arm64 and x86-64."""
    src = data_dir / "BIOMD0000000908.xml"
    rev = _reversed(src, tmp_path / "reversed.xml")
    run = {"t_span": (0, 300), "n_points": 1001, "rtol": 1e-6, "atol": 1e-9}
    file_order = _species(bngsim.Model.from_sbml(str(src)), run, force_dense_linear_solver=True)
    reversed_order = _species(
        bngsim.Model.from_sbml(str(rev)), run, force_dense_linear_solver=True
    )
    np.testing.assert_array_equal(file_order["S"], 0.0)
    np.testing.assert_array_equal(reversed_order["S"], 0.0)
    for name in ("T", "N", "L", "R", "C", "I"):
        a, b = file_order[name], reversed_order[name]
        assert np.abs(a - b).max() <= 1e-6 * np.abs(a).max() + run["atol"], name
