"""KLU's colored difference quotient reads f as the RHS does (GH #135, issue #1340).

Where f is not finite at a state, the RHS callback reads it again at the
nonnegative-clamped state (GH #135), and where that is still not finite, with
each slot the clamp leaves alone read at 0 if it is below 0 by less than its
atol (#1340). The dense difference quotient calls that callback for each
column, so its matrix is that of the f the solver integrates. The colored
quotient KLU factors where there is no closed-form Jacobian (``jacobian="fd"``,
or a law the closed form declines) called ``compute_derivs`` raw: at such a
state each column beside the slot came back NaN, and the run failed where the
dense one ran on. MODEL9811206584's s, a rate-rule target decayed to -5e-12
under ``log(s)``, failed on KLU at t = 99.3 (``CV_REPTD_RHSFUNC_ERR``, NaN in f
and h) and ran on dense to t = 100: the nightly's path agreement
(``klu_vs_dense``), once #1453 moved its steps onto such a state.

The oracle is a closed form where there is one: A held at exactly 1 where its
law reads P at 0, and A's decay through sqrt(1 - t) where S is drained by a
zero-order sink. Otherwise, the dense run of the same model.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import bngsim
import numpy as np
import pytest

_HAS_KLU = bngsim._bngsim_core.HAS_KLU
_needs_klu = pytest.mark.skipif(not _HAS_KLU, reason="KLU not compiled")

_RUN = {"t_span": (0, 10), "n_points": 11, "rtol": 1e-8, "atol": 1e-9}
_EXACT = {**_RUN, "atol": bngsim.ExactAtol(1e-9)}

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
_codegen = pytest.mark.parametrize(
    "codegen",
    [False, pytest.param(True, marks=pytest.mark.skipif(_CC is None, reason="no C compiler"))],
    ids=["interpreted", "compiled"],
)


def _species(model, run, solver: str, **simulator):
    r = bngsim.Simulator(
        model, method="ode", jacobian="fd", **{f"force_{solver}_linear_solver": True}, **simulator
    ).run(**run)
    names = list(r.species_names)
    return np.asarray(r.time), {n: np.asarray(r.species)[:, names.index(n)] for n in names}


def _held(p: str = "-5e-10") -> bngsim.Model:
    """P held at p by a rate rule (P' = 0), read by sqrt(P) in A's decay. At
    -5e-10 it is inside its atol of 1e-9, read at 0, so A is exactly 1."""
    return bngsim.Model.from_antimony_string(
        f"model m\n  species A = 1, P = {p};\n  k = 0.1;\n  P' = 0;\n  J: A -> ; k*A*P^0.5;\nend"
    )


def _sinking(order: str) -> bngsim.Model:
    """P' = -1e-30 from 0, inside its band to t = 10, read by sqrt(P)."""
    decl = "A = 1, P = 0" if order == "AP" else "P = 0, A = 1"
    return bngsim.Model.from_antimony_string(
        f"model m\n  species {decl};\n  k = 0.1; c = 1e-30;\n  P' = -c;\n"
        "  J: A -> ; k*A*P^0.5;\nend"
    )


def _drained() -> bngsim.Model:
    """S drained past 0 by a zero-order sink, read by sqrt(S) in A's decay."""
    return bngsim.Model.from_antimony_string(
        "model m\n  species S = 1, A = 1;\n  c = 1; k = 0.1;\n  J1: S -> ; c;\n"
        "  J2: A -> ; k*A*S^0.5;\nend"
    )


def _drained_exact(t):
    """A = exp(-k*(2/3)*(1 - (1 - t)^1.5)) to t = 1, held there after."""
    return np.exp(-0.1 * (2 / 3) * (1 - np.clip(1 - t, 0, None) ** 1.5))


_BAND = pytest.mark.parametrize(
    ("make", "run"),
    [(_held, _EXACT), (lambda: _sinking("AP"), _RUN), (lambda: _sinking("PA"), _RUN)],
    ids=["held", "sinking AP", "sinking PA"],
)


@_BAND
def test_the_dense_quotient_reads_a_rate_rule_target_in_its_band_at_zero(make, run):
    """The reference the KLU cases are held to, on every build: the dense
    quotient calls the RHS callback, retry and all, for each column, and ran
    each of these on main."""
    _, x = _species(make(), run, "dense")
    np.testing.assert_array_equal(x["A"], 1.0)


def test_the_dense_quotient_follows_the_drained_closed_form():
    t, x = _species(_drained(), _EXACT, "dense")
    # The run is at rtol 1e-8; what is left is the kink at t = 1, where sqrt's
    # slope is infinite (7.9e-7 from the closed form).
    np.testing.assert_allclose(x["A"], _drained_exact(t), rtol=2e-6)


@_needs_klu
@_codegen
@_BAND
def test_klu_reads_a_rate_rule_target_inside_its_atol_of_zero_at_zero(make, run, codegen):
    """The band of #1340. On main, KLU failed each of these with CV_CONV_FAILURE
    at t = 1, where dense ran. Compiled, CVODE's f is the compiled RHS and the
    quotient's base is read again on the interpreted one, through the retry."""
    _, x = _species(make(), run, "sparse", codegen=codegen)
    np.testing.assert_array_equal(x["A"], 1.0)


@_needs_klu
@_codegen
def test_klu_reads_a_drained_species_at_zero_as_the_dense_quotient_does(codegen):
    """The clamp of GH #135, which the colored quotient never had. On main KLU
    failed at t = 2 (CV_CONV_FAILURE); dense ran."""
    t, x = _species(_drained(), _EXACT, "sparse", codegen=codegen)
    np.testing.assert_allclose(x["A"], _drained_exact(t), rtol=2e-6)
    # A fresh Model: a run leaves its state on the one it ran.
    _, dense = _species(_drained(), _EXACT, "dense", codegen=codegen)
    np.testing.assert_allclose(x["A"], dense["A"], rtol=1e-9)


def test_both_routes_still_fail_where_no_reading_is_finite():
    """The control: P held at -3e-9 is past its atol of 1e-9, and sqrt of it
    is read as nothing on either route."""
    for solver in ("sparse", "dense") if _HAS_KLU else ("dense",):
        with pytest.raises(bngsim.SimulationError):
            _species(_held("-3e-9"), _EXACT, solver)


@_needs_klu
def test_model9811206584_runs_on_klu_as_on_dense(data_dir: Path):
    """The nightly's finding: s decays to 0 from t = 90, the solver takes it a
    hair below, and ``log(s)/log(10)`` there is NaN, read at 0 by the band.
    Each species within 1e-5 of its own peak plus the run's atol, which is
    what the two routes leave apart (5e-8 relative where they agree)."""
    run = {"t_span": (0, 100), "n_points": 101, "rtol": 1e-6, "atol": 1e-9}
    path = str(data_dir / "MODEL9811206584.xml")
    _, klu = _species(bngsim.Model.from_sbml(path), run, "sparse")
    _, dense = _species(bngsim.Model.from_sbml(path), run, "dense")
    for name, x in dense.items():
        bound = 1e-5 * float(np.abs(x).max()) + run["atol"]
        assert np.abs(klu[name] - x).max() <= bound, name
