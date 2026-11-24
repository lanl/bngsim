"""steady_state()'s run test resolves the model, whatever the march's atol (issue #1895).

``criterion="scaled"`` calls a state converged when the residual test passes
and a run taken on from the state moves no species past its tolerance. The run
was integrated at the march's own atol, which ``steady_state(atol=...)`` takes
as given (issue #1420 kept it so). BIOMD0000000609, a paracetamol model whose
species are all below 1.4e-13, asked at its SED-ML file's ``rtol=1e-6`` and
``atol=1e-9`` with its adduct accumulator masked, marched with error control
off, stopped on a root of f with two species near -1.5e-14 (the steady state
has them at 1.3e-14 and 0) and the accumulator at 3.9e-9, and came back
``converged=True``: a run at an atol of 1e-9 does not see a state at 1e-14
move. A run from it, as :meth:`Simulator.run` integrates one, fails at
t = 2.1 with the RHS non-finite. The run test is now integrated at an atol no
coarser than ``rtol`` times the largest species the solve starts with.

The models below are the issue's, transcribed to ``.net``, and a three-species
reduction of it. Their steady state has a closed form: ``S = bS/dS``,
``G = bG/dG``, ``N = P = 0``. The oracles are that closed form and a run of
the time course from the returned state; neither is the run test.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

# BIOMD0000000609, by reaction: S is sulphate (PAPS), G glutathione, N NAPQI,
# P paracetamol, A the protein adducts, which nothing consumes.
APAP_NET = """begin parameters
    1 dG    2.0
    2 bG    1.374e-14
    3 kGSH  1.6e+18
    4 kG    2.99
    5 kS    2.26e14
    6 bS    2.65e-14
    7 dS    2.0
    8 k450  0.315
    9 kN    0.0315
   10 kPSH  110.0
   11 P0    1.32e-13
   12 S0    1.325e-14
   13 G0    6.87e-15
end parameters
begin species
    1 S() S0
    2 G() G0
    3 N() 0
    4 P() P0
    5 A() 0
end species
begin reactions
    1 4 3 k450
    2 3 4 kN
    3 4 0 kG
    4 3 5 kPSH
    5 3,2 0 kGSH
    6 2 0 dG
    7 0 2 bG
    8 4,1 0 kS
    9 1 0 dS
   10 0 1 bS
end reactions
"""

# The same sulphate and paracetamol, P drained into an accumulator A.
SPA_NET = """begin parameters
    1 kG    2.99
    2 kS    2.26e14
    3 bS    2.65e-14
    4 dS    2.0
    5 P0    1.32e-13
    6 S0    1.325e-14
    7 kA    0.5
end parameters
begin species
    1 S() S0
    2 P() P0
    3 A() 0
end species
begin reactions
    1 2 0 kG
    2 2,1 0 kS
    3 1 0 dS
    4 0 1 bS
    5 2 3 kA
end reactions
"""

# The issue's call: its SED-ML file's tolerances, the differential harness's
# horizon, and the accumulator out of the test.
RTOL, ATOL, HORIZON = 1e-6, 1e-9, 1e6

MODELS = {"apap": APAP_NET, "spa": SPA_NET}

# The closed-form steady state of each species that has one; the accumulator
# has none (it ends at what the run drained into it).
ROOTS = {
    "apap": {"S()": 2.65e-14 / 2.0, "G()": 1.374e-14 / 2.0, "N()": 0.0, "P()": 0.0},
    "spa": {"S()": 2.65e-14 / 2.0, "P()": 0.0},
}


def _model(tmp_path: Path, name: str) -> bngsim.Model:
    p = tmp_path / f"{name}.net"
    p.write_text(MODELS[name])
    return bngsim.Model.from_net(str(p))


def _solve(tmp_path, name, method, atol=ATOL, state=None):
    """The solve, and the state it started from."""
    model = _model(tmp_path, name)
    if state is not None:
        model.set_state(np.asarray(state, dtype=float))
    start = np.asarray(model.get_state(), dtype=float).copy()
    sink = np.asarray(model.is_pure_sink(), dtype=bool)
    assert sink.tolist() == [n == "A()" for n in model.species_names]
    sim = bngsim.Simulator(model, method="ode")
    kw = {} if atol is None else {"atol": atol}
    ss = sim.steady_state(method=method, rtol=RTOL, max_time=HORIZON, mask=~sink, **kw)
    return ss, start


def _tolerance(start, x, atol=ATOL) -> np.ndarray:
    """Each species' tolerance at ``x`` under ``criterion="scaled"``, as the
    docstring gives it: ``rtol·|x_i| + atol_i``, ``atol_i`` no more than
    ``rtol`` times the largest species of the start and of ``x``. Strict
    (``atol=0``): ``rtol`` times the model's own scale, 1.4e-13, instead."""
    largest = max(float(np.abs(start).max()), float(np.abs(x).max()))
    floor = min(atol, RTOL * largest) if atol else RTOL * 1.4e-13
    return RTOL * np.abs(x) + floor


def _off_the_root(name: str, ss, start, atol=ATOL) -> list[str]:
    """The species of the closed form the solve's state is off by more than
    its tolerance."""
    x = np.asarray(ss.concentrations, dtype=float)
    tol = dict(zip(ss.species_names, _tolerance(start, x, atol), strict=True))
    at = dict(zip(ss.species_names, x, strict=True))
    return [s for s, v in ROOTS[name].items() if abs(at[s] - v) > tol[s]]


def _run_rests_at(tmp_path, name, start, x) -> bool:
    """Whether a run of the time course from ``x``, at the issue's rtol and
    atol (each species held to rtol of its own scale under the scalar, issue
    #1420), rests there: every species the solve measures within its
    tolerance at the end. One the solver cannot carry on does not."""
    model = _model(tmp_path, name)
    model.set_state(x)
    keep = ~np.asarray(model.is_pure_sink(), dtype=bool)
    try:
        r = bngsim.Simulator(model, method="ode").run(
            t_span=(0.0, HORIZON), n_points=5, rtol=RTOL, atol=ATOL
        )
    except bngsim.SimulationError as e:
        assert e.code.startswith("solver."), e
        return False
    end = np.asarray(r.species, dtype=float)[-1]
    return bool((np.abs(end - x) <= _tolerance(start, x))[keep].all())


@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("name", sorted(MODELS))
def test_converged_is_the_closed_form_steady_state(tmp_path, name, method):
    """At the issue's atol the march resolves nothing, and where it stops is
    its error. Wherever that is off the steady state, the solve must not say
    converged, and a run from it must not rest there. Main returned
    converged=True on a spurious root for apap by integration (S at -1.46e-14
    for 1.325e-14, the issue's state) and for spa by Newton (S at -1.54e-14);
    a run of the time course from either fails on the RHS."""
    ss, start = _solve(tmp_path, name, method)
    x = np.asarray(ss.concentrations, dtype=float)
    if _off_the_root(name, ss, start):
        assert not ss.converged, (x, ss.continued_run_distance)
        # Refused by the run test, which ran (the residual test passed), or
        # by the residual test itself.
        assert ss.continued_run_distance > 1.0 or not ss.residual < 1e-9
        assert not _run_rests_at(tmp_path, name, start, x)
    elif ss.converged:
        assert _run_rests_at(tmp_path, name, start, x)


@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_default_atol_finds_the_steady_state(tmp_path, name, method):
    """No atol given, each species is marched at rtol of its own scale and the
    solve reaches the closed form: the run test's atol is untouched there
    (the default is never coarser than rtol times the start's largest)."""
    ss, start = _solve(tmp_path, name, method, atol=None)
    assert ss.converged
    assert _off_the_root(name, ss, start, atol=0) == []
    assert ss.continued_run_distance <= 1.0


@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_steady_state_at_a_coarse_atol_stays_converged(tmp_path, name, method):
    """A start at the steady state, at the issue's atol: the run held to the
    start's scale does not refuse a state the system rests at."""
    model = _model(tmp_path, name)
    at = {s: ROOTS[name].get(s, 0.0) for s in model.species_names}
    ss, start = _solve(tmp_path, name, method, state=[at[s] for s in model.species_names])
    assert ss.converged
    assert _off_the_root(name, ss, start, atol=0) == []
    assert ss.continued_run_distance <= 1.0


def test_steady_state_batch_takes_the_same_test(tmp_path):
    """Each entry of a batch is solved by the same core call, and judged by
    the same run test."""
    model = _model(tmp_path, "apap")
    start = np.asarray(model.get_state(), dtype=float).copy()
    sink = np.asarray(model.is_pure_sink(), dtype=bool)
    sim = bngsim.Simulator(model, method="ode")
    results = sim.steady_state_batch(
        params=[{}, {"bS": 2.65e-14}], rtol=RTOL, atol=ATOL, max_time=HORIZON, mask=~sink
    )
    for ss in results:
        if ss.converged:
            assert _off_the_root("apap", ss, start) == [], ss.concentrations


_CORPUS = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"


def test_the_issues_corpus_model(tmp_path):
    """BIOMD0000000609 itself, as the issue calls it, where the corpus is in
    the checkout (it is fetched, not committed)."""
    hits = sorted(_CORPUS.glob("BIOMD0000000609/*.xml"))
    if not hits:
        pytest.skip("rr_parity corpus model BIOMD0000000609 not present")
    model = bngsim.Model.load(str(hits[0]))
    sink = np.asarray(model.is_pure_sink(), dtype=bool)
    ss = bngsim.Simulator(model, method="ode").steady_state(
        method="integration", rtol=RTOL, atol=ATOL, max_time=HORIZON, mask=~sink
    )
    at = dict(zip(ss.species_names, np.asarray(ss.concentrations, dtype=float), strict=True))
    if ss.converged:
        assert at["Sulphate__PAPS"] == pytest.approx(1.325e-14, rel=1e-3)
        assert abs(at["Paracetamol_APAP"]) < 1e-18
    else:
        assert ss.continued_run_distance > 1.0
