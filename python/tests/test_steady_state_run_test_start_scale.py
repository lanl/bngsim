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
t = 2.1 with the RHS non-finite. The run test now holds each species to an
atol no coarser than :meth:`Simulator.run` would hold it, given the same
scalar, from the solve's start: ``rtol`` of its own scale there (issue #1420's
rule). One cap for all, ``rtol`` times the start's largest species, was
blinded again by an accumulator that had already run, or by a species at the
atol beside the rest.

The models below are the issue's, transcribed to ``.net``, the same with a
larger species at the start, and a three-species reduction of it. Their steady
state has a closed form: ``S = bS/dS``, ``G = bG/dG``, ``N = P = 0``. The
reduction fed from a ``$``-fixed source, every other species at 0, is a start
whose only scale is the source's, which :meth:`Simulator.run` holds the rest
to; its steady state is a quadratic's root. The oracles are the closed form,
to the criterion's own tolerance, and a run of the time course from the
returned state; neither is the run test.
"""

from __future__ import annotations

import re
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

# The same, every species not $-fixed at 0, P fed from a $-fixed source at
# the issue's paracetamol: the start's only scale is the source's.
SRC_NET = """begin parameters
    1 kG    2.99
    2 kS    2.26e14
    3 bS    2.65e-14
    4 dS    2.0
    5 kA    0.5
    6 kp    {kp}
    7 Src0  1.32e-13
end parameters
begin species
    1 S() 0
    2 P() 0
    3 A() 0
    4 $Src() Src0
end species
begin reactions
    1 2 0 kG
    2 2,1 0 kS
    3 1 0 dS
    4 0 1 bS
    5 2 3 kA
    6 4 4,2 kp
end reactions
"""


def _src_root(kp: float) -> dict[str, float]:
    """SRC_NET's steady state: S = bS/(dS + kS·P), and P the positive root of
    the quadratic its own balance makes of that."""
    kG, kS, bS, dS, kA, src = 2.99, 2.26e14, 2.65e-14, 2.0, 0.5, 1.32e-13
    a = (kG + kA) * kS
    b = kp * src * kS - (kG + kA) * dS - kS * bS
    c = kp * src * dS
    p = (b + np.sqrt(b * b + 4.0 * a * c)) / (2.0 * a)
    return {"S()": bS / (dS + kS * p), "P()": float(p)}


# The issue's call: its SED-ML file's tolerances, the differential harness's
# horizon, and the accumulator out of the test.
RTOL, ATOL, HORIZON = 1e-6, 1e-9, 1e6

MODELS = {
    "apap": APAP_NET,
    "spa": SPA_NET,
    # The issue's model with the accumulator started where the issue's solve
    # left it, as a model that had already run would be.
    "apap_ran": APAP_NET.replace("    5 A() 0\n", "    5 A() 3.9e-9\n"),
    # The issue's model beside a species at the issue's atol that nothing
    # reads or moves.
    "apap_beside": APAP_NET.replace("    5 A() 0\n", "    5 A() 0\n    6 Z() 1e-9\n"),
    # And beside one at 1: there the criterion's own tolerance is 1e-9, the
    # sulphate and paracetamol are rounding beside Z, as run() holds them,
    # and the spurious root is within tolerance of the steady state.
    "apap_at_one": APAP_NET.replace("    5 A() 0\n", "    5 A() 0\n    6 Z() 1.0\n"),
    # Every species not $-fixed at 0, so the start's scale is the source's,
    # as run() reads it.
    "src": SRC_NET.format(kp=1.0),
}

# The closed-form steady state of each species that has one; the accumulator
# has none (it ends at what the run drained into it).
APAP_ROOT = {"S()": 2.65e-14 / 2.0, "G()": 1.374e-14 / 2.0, "N()": 0.0, "P()": 0.0}
ROOTS = {
    "apap": APAP_ROOT,
    "spa": {"S()": 2.65e-14 / 2.0, "P()": 0.0},
    "apap_ran": APAP_ROOT,
    "apap_beside": {**APAP_ROOT, "Z()": 1e-9},
    "apap_at_one": {**APAP_ROOT, "Z()": 1.0},
    "src": _src_root(1.0),
}


def _model(tmp_path: Path, name: str) -> bngsim.Model:
    p = tmp_path / f"{name}.net"
    p.write_text(MODELS[name])
    return bngsim.Model.from_net(str(p))


def _solve(tmp_path, name, method, atol=ATOL, state=None):
    """The solve, and the state it started from. ``atol="vector"`` is the
    issue's atol as one entry per species, the per-species path."""
    model = _model(tmp_path, name)
    if state is not None:
        model.set_state(np.asarray(state, dtype=float))
    start = np.asarray(model.get_state(), dtype=float).copy()
    sink = np.asarray(model.is_pure_sink(), dtype=bool)
    assert sink.tolist() == [n == "A()" for n in model.species_names]
    sim = bngsim.Simulator(model, method="ode")
    if isinstance(atol, str):
        atol = [ATOL] * len(model.species_names)
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


def _off_the_root(name: str, ss, start, atol=ATOL, root=None) -> list[str]:
    """The species of the closed form the solve's state is off by more than
    its tolerance."""
    x = np.asarray(ss.concentrations, dtype=float)
    tol = dict(zip(ss.species_names, _tolerance(start, x, atol), strict=True))
    at = dict(zip(ss.species_names, x, strict=True))
    return [s for s, v in (root or ROOTS[name]).items() if abs(at[s] - v) > tol[s]]


def _run_rests_at(model: bngsim.Model, start, x) -> bool:
    """Whether a run of the time course from ``x``, at the issue's rtol and
    atol, rests there: every species the solve measures within its tolerance
    at the end. ``run()`` reads the scalar as its ceiling, each species held
    to ``rtol`` of its own scale in ``x`` (issue #1420), by its own code. One
    the solver cannot carry on does not rest."""
    keep = ~np.asarray(model.is_pure_sink(), dtype=bool)
    model.set_state(x)
    try:
        r = bngsim.Simulator(model, method="ode").run(
            t_span=(0.0, HORIZON), n_points=5, rtol=RTOL, atol=ATOL
        )
    except bngsim.SimulationError as e:
        assert e.code.startswith("solver."), e
        return False
    end = np.asarray(r.species, dtype=float)[-1]
    return bool((np.abs(end - x) <= _tolerance(start, x))[keep].all())


@pytest.mark.parametrize("atol", [ATOL, "vector"], ids=["scalar", "vector"])
@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("name", sorted(MODELS))
def test_converged_is_a_rest_at_the_steady_state(tmp_path, name, method, atol):
    """At the issue's atol the march resolves nothing, and where it stops is
    its error. A state the solve calls converged must be the closed form, to
    its own tolerance, and a run that resolves every species must rest there.

    Main returned converged=True on the spurious root, S at -1.46e-14 for
    1.325e-14 (the issue's state), for apap by integration, for spa by Newton
    (S at -1.54e-14), and for apap_ran and apap_beside by integration; a run
    from each fails on the RHS. The same with the atol given per species.
    apap_at_one is converged at that root on both, rightly: within its 1e-9
    tolerance of the steady state, and a run from it rests. src, whose start
    has only the source's scale, was converged by integration with S at
    -7.1e-14 for 2.9e-15, a run test that took the coarse atol as given
    finding it moved by 0.004 of its tolerance; a run from it fails. Where the
    solve does not converge, the run test or the residual test said so."""
    ss, start = _solve(tmp_path, name, method, atol=atol)
    x = np.asarray(ss.concentrations, dtype=float)
    if ss.converged:
        assert _off_the_root(name, ss, start) == [], (x, ss.continued_run_distance)
        assert _run_rests_at(_model(tmp_path, name), start, x), x
    else:
        assert ss.continued_run_distance > 1.0 or not ss.residual < 1e-9


# SRC_NET's feeds, from a sweep of 51 between 0.01 and 1000, at which the
# march at the issue's atol stops with S and P below zero, far from where
# they start (S at -6.6e-12 at a feed of 100, past ten thousand times its
# steady value), and a run held to the start's scales rests there, where
# run() from it leaves (1.58) or fails (6.31, 100).
LEFT_BY_A_RUN = [1.584893192461114, 6.309573444801936, 100.0]


@pytest.mark.parametrize("atol", [ATOL, "vector"], ids=["scalar", "vector"])
@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("kp", LEFT_BY_A_RUN)
def test_a_state_below_zero_that_a_run_leaves_is_refused(tmp_path, kp, method, atol):
    """A species the state holds below zero is held, in the run test, as
    run() would hold it from that state, so a converged state is one run()
    from it rests at. Before, by integration, all three came back
    converged=True; run() from each leaves it. (A saddle that run() rests
    on is converged, as the guide says, and is not judged here.)"""
    p = tmp_path / "src.net"
    p.write_text(SRC_NET.format(kp=repr(kp)))
    model = bngsim.Model.from_net(str(p))
    start = np.asarray(model.get_state(), dtype=float).copy()
    sink = np.asarray(model.is_pure_sink(), dtype=bool)
    if isinstance(atol, str):
        atol = [ATOL] * len(model.species_names)
    ss = bngsim.Simulator(model, method="ode").steady_state(
        method=method, rtol=RTOL, atol=atol, max_time=HORIZON, mask=~sink
    )
    x = np.asarray(ss.concentrations, dtype=float)
    if ss.converged:
        assert _run_rests_at(bngsim.Model.from_net(str(p)), start, x), x
    else:
        assert ss.continued_run_distance > 1.0 or not ss.residual < 1e-9


@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("name", ["apap", "spa", "src"])
def test_the_default_atol_finds_the_steady_state(tmp_path, name, method):
    """No atol given, each species is marched at rtol of its own scale and the
    solve reaches the closed form, to the model's scale. The run test's atol
    is no coarser than that default, so the default's runs are unchanged."""
    ss, start = _solve(tmp_path, name, method, atol=None)
    assert ss.converged
    assert _off_the_root(name, ss, start, atol=0) == []
    assert ss.continued_run_distance <= 1.0


@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("name", ["apap", "spa"])
def test_the_steady_state_at_a_coarse_atol_stays_converged(tmp_path, name, method):
    """A start at the steady state, at the issue's atol: the run that holds
    each species to its own scale does not refuse a state the system rests
    at."""
    model = _model(tmp_path, name)
    at = {s: ROOTS[name].get(s, 0.0) for s in model.species_names}
    ss, start = _solve(tmp_path, name, method, state=[at[s] for s in model.species_names])
    assert ss.converged
    assert _off_the_root(name, ss, start, atol=0) == []
    assert ss.continued_run_distance <= 1.0


def test_steady_state_batch_takes_the_same_test(tmp_path):
    """Each entry of a batch is solved by the same core call and judged by
    the same run test: its first entry, at the model's own parameters, is
    steady_state()'s solve to the bit, and no entry is converged off its
    closed form. Main returned the first converged on the spurious root."""
    model = _model(tmp_path, "apap")
    start = np.asarray(model.get_state(), dtype=float).copy()
    sink = np.asarray(model.is_pure_sink(), dtype=bool)
    sim = bngsim.Simulator(model, method="ode")
    points = [{}, {"bS": 5.3e-14}]
    results = sim.steady_state_batch(
        params=points, rtol=RTOL, atol=ATOL, max_time=HORIZON, mask=~sink
    )
    single, _ = _solve(tmp_path, "apap", "integration")
    assert results[0].converged == single.converged
    assert np.array_equal(results[0].concentrations, single.concentrations)
    assert results[0].continued_run_distance == single.continued_run_distance
    for point, ss in zip(points, results, strict=True):
        root = {**APAP_ROOT, "S()": point.get("bS", 2.65e-14) / 2.0}
        if ss.converged:
            assert _off_the_root("apap", ss, start, root=root) == [], ss.concentrations
        else:
            assert ss.continued_run_distance > 1.0 or not ss.residual < 1e-9


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
        assert ss.continued_run_distance > 1.0 or not ss.residual < 1e-9


def test_the_run_tests_share_is_runs():
    """The run test holds a species as run() would from the same start: the
    share of the largest species below which a species is held as though at
    it is Simulator._ATOL_SCALE_RELEVANT on both sides. A finer share in the
    core (machine epsilon) held BIOMD0000000696's species at 0 to 6e-18 and
    its run ran out of steps at a stable root."""
    src = Path(__file__).resolve().parents[2] / "src" / "steady_state.cpp"
    found = re.findall(r"constexpr double kStartScaleShare = ([0-9.eE+-]+);", src.read_text())
    assert len(found) == 1, found
    assert float(found[0]) == bngsim.Simulator._ATOL_SCALE_RELEVANT
