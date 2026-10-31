"""``solver_stats["n_jac_rhs_evals"]``: what a difference-quotient Jacobian costs
in right-hand-side evaluations (issue #1333).

#1302 forms CVODE's dense difference quotient itself and tests each column for
a jump, one or three more evaluations of f per tested column. Steps and
Jacobian counts were unchanged, and ``n_rhs_evals`` is CVODE's count of the
integration's own evaluations, so none of that cost showed in any counter (it
was +21% to +78% of the evaluations on the ODE perf suite, +4% to +36% of the
wall). It is now counted, for bngsim's dense and colored quotients and CVODE's
own, and the nightly compares it.
"""

from __future__ import annotations

import bngsim
import pytest

_N = 10
_CHAIN = (
    "species X0 = 10;"
    + "".join(f" species X{i} = 0;" for i in range(1, _N))
    + "".join(f" J{i}: X{i} -> X{i + 1}; 0.5*X{i}*X{i}/(1 + X{i});" for i in range(_N - 1))
)


def _stats(text: str = _CHAIN, **kw) -> dict:
    run = {k: kw.pop(k) for k in list(kw) if k in ("t_span", "n_points")}
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode", **kw)
    return sim.run(**{"t_span": (0, 20), "n_points": 21, **run}).solver_stats


def test_a_closed_form_jacobian_costs_no_evaluations():
    stats = _stats()
    assert stats["n_jac_rhs_evals"] == 0


def test_the_dense_quotient_is_counted():
    """Each Jacobian formed takes a column per species and, past that, #1302's
    jump test, one more point at least: more than n_species per formation."""
    fd = _stats(jacobian="fd")
    assert fd["n_jac_formations"] > 0
    assert fd["n_jac_rhs_evals"] > _N * fd["n_jac_formations"]
    assert fd["n_jac_rhs_evals"] <= 4 * _N * fd["n_jac_formations"]


@pytest.mark.skipif(not bngsim._bngsim_core.HAS_KLU, reason="KLU not compiled in")
def test_the_colored_quotient_is_counted():
    stats = _stats(jacobian="fd", force_sparse_linear_solver=True)
    assert stats["linear_solver"] != 0  # not the dense solver
    # A coloring evaluates f once per color, fewer than the species here.
    assert stats["n_jac_formations"] > 0
    assert stats["n_jac_rhs_evals"] >= stats["n_jac_formations"]


def test_the_count_covers_a_run_its_events_restart():
    """An event's re-initialization zeroes CVODE's own counters, which the run
    banks; this one is the run's, so the segment after a restart adds to it."""
    text = _CHAIN + " E1: at (time > 10): X0 = 10;"
    whole = _stats(text, jacobian="fd")
    half = _stats(text, jacobian="fd", t_span=(0, 10), n_points=11)
    assert whole["n_jac_rhs_evals"] > half["n_jac_rhs_evals"] > 0


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(_CHAIN, id="warm path"),
        # A run with events takes the cold path every time.
        pytest.param(_CHAIN + " E1: at (time > 10): X0 = 10;", id="cold path"),
    ],
)
def test_a_second_run_counts_its_own(text):
    """The count belongs to the run: a Simulator run twice from the same state
    reports the same count, not the sum."""
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", jacobian="fd")
    first = sim.run(t_span=(0, 20), n_points=21).solver_stats["n_jac_rhs_evals"]
    model.reset()
    second = sim.run(t_span=(0, 20), n_points=21).solver_stats["n_jac_rhs_evals"]
    assert first == second > 0


def test_the_nightly_compares_it():
    """The nightly's work counters carry it and alert on it; a baseline that
    predates it has no value to compare and is skipped."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "verdicts", root / "parity_checks" / "nightly" / "verdicts.py"
    )
    verdicts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verdicts)
    assert "n_jac_rhs_evals" in verdicts.WORK_KEYS
    assert "n_jac_rhs_evals" in verdicts.WORK_ALERT_KEYS
    assert "n_jac_rhs_evals" in verdicts.WORK_MIN_DELTA
