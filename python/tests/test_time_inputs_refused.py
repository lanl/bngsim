"""Output times no run can honour are refused up front (issues #1504, #1523, #1524).

#1504: ``run(sample_times=[0, 5, inf])``, or a NaN last time, never returned on an
ODE model that settles, and ``timeout=`` could not stop it: CVODE was handed a
target it never reaches and spun inside one CVode call, where the wall-clock
budget is never checked. SSA ran until the budget or for ever, NFsim past it, and
a NaN came back as a row labelled ``t = nan``. ``run()`` never asked whether the
times were finite, nor did ``run_batch``, ``run_replicates``, ``run_until``,
``parameter_scan`` or the NFsim session.

#1523: ``compute_all_sensitivities`` checked neither ``t_span`` nor ``n_points``.
``n_points=0`` with a time switch inside the span killed the interpreter
(``CvodeSimulator::run`` read the ends of an empty output grid), and
``t_span=(10, 0)`` returned three rows at ``t = 10`` with zero sensitivities.

#1524: ``Result`` sized its buffers with ``int`` products, so 65537 output times
x 65536 species wrapped to a one-row buffer that every ``record`` wrote past.

Every entry point now refuses through one validator (``bngsim._sample_times``),
and each backend refuses an empty or non-finite grid at entry as well, for a
caller of ``_bngsim_core`` that skips the Python layer.
"""

from __future__ import annotations

import math
import subprocess
import sys
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import CvodeSimulator, SsaSimulator, TimeSpec

# x(t) = 20 (1 - exp(-0.1 t)): a model that settles, the case that hung.
SETTLES = "species x; x = 0; ks = 2; kd = 0.1\nJ0: -> x; ks\nJ1: x -> ; kd*x\n"
# Takes the run through the event loop, which hung the same way on a NaN.
WITH_EVENT = SETTLES + "E1: at (time > 2): x = 1\n"

NON_FINITE = [math.inf, -math.inf, math.nan]

_NFSIM = pytest.mark.skipif(
    not getattr(bngsim, "HAS_NFSIM", False), reason="NFsim not compiled in"
)


def _sim(text: str = SETTLES, method: str = "ode") -> bngsim.Simulator:
    extra = {"poplevel": 100} if method == "psa" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Simulator(bngsim.Model.from_antimony_string(text), method=method, **extra)


# ─── Issue #1504: Simulator.run and its siblings ────────────────────────────


@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize(
    ("text", "method"),
    [(SETTLES, "ode"), (WITH_EVENT, "ode"), (SETTLES, "ssa"), (SETTLES, "psa")],
    ids=["ode", "ode-event", "ssa", "psa"],
)
def test_run_refuses_sample_times_that_are_not_finite(text, method, bad):
    sim = _sim(text, method)
    # Last, where it hung, and in the middle, where sorting a list that holds a
    # NaN leaves it anywhere.
    for times in ([0.0, 5.0, bad], [0.0, bad, 10.0]):
        with pytest.raises(ValueError, match="sample_times must all be finite"):
            sim.run(sample_times=times, timeout=1.0)
    # The refusal moved nothing: the same Simulator runs a good list.
    r = sim.run(sample_times=[0.0, 5.0, 50.0], seed=1)
    assert r.n_times == 3


@pytest.mark.parametrize(
    "t_span", [(0.0, math.inf), (0.0, math.nan), (math.nan, 1.0), (-math.inf, 1.0)]
)
@pytest.mark.parametrize("method", ["ode", "ssa", "psa"])
def test_run_refuses_a_t_span_that_is_not_finite(method, t_span):
    # `t_end <= t_start` is false for a NaN and for inf alike, so both used to
    # pass: CV_CONV_FAILURE on ODE, a row at t = inf on SSA.
    with pytest.raises(ValueError, match="t_span must be finite"):
        _sim(method=method).run(t_span=t_span, n_points=3)


def test_the_finite_control_matches_the_closed_form():
    r = _sim().run(sample_times=[0.0, 5.0, 50.0], rtol=1e-10, atol=1e-12)
    expected = [20.0 * (1.0 - math.exp(-0.1 * t)) for t in (0.0, 5.0, 50.0)]
    np.testing.assert_allclose(r.species[:, 0], expected, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("bad", NON_FINITE)
def test_run_batch_refuses_a_t_span_that_is_not_finite(bad):
    with pytest.raises(ValueError, match="t_span must be finite"):
        _sim().run_batch(t_span=(0.0, bad), n_points=3, params=[{"kd": 0.1}, {"kd": 0.2}])


@pytest.mark.parametrize("bad", NON_FINITE)
def test_run_replicates_refuses_a_t_span_that_is_not_finite(bad):
    # SSA returned times [0, inf, inf] here with no error.
    with pytest.raises(ValueError, match="t_span must be finite"):
        _sim(method="ssa").run_replicates(2, t_span=(0.0, bad), n_points=3, seed=1)


@pytest.mark.parametrize("bad", NON_FINITE)
def test_run_until_refuses_a_target_that_is_not_finite(bad):
    # A NaN passed `t <= current_time`, which is false for it.
    sim = _sim()
    sim.run_until(2.0)
    with pytest.raises(ValueError, match="Target time must be finite"):
        sim.run_until(bad)
    assert sim.current_time == 2.0
    assert sim.run_until(4.0).time[-1] == 4.0


@pytest.mark.parametrize("scan", ["parameter_scan", "bifurcate"])
def test_a_scan_refuses_the_span_before_it_touches_the_model(scan):
    sim = _sim()
    before = sim.model.get_param("kd")
    with pytest.raises(ValueError, match="t_span must be finite"):
        getattr(sim, scan)("kd", par_min=0.1, par_max=0.2, n_scan_pts=2, t_span=(0.0, math.inf))
    assert sim.model.get_param("kd") == before
    assert sim.current_time == 0.0


@pytest.mark.parametrize("bad", NON_FINITE)
def test_compute_all_sensitivities_refuses_a_t_span_that_is_not_finite(bad):
    with pytest.raises(ValueError, match="t_span must be finite"):
        _sim().compute_all_sensitivities(t_span=(0.0, bad), n_points=3, params=["kd"])


@_NFSIM
@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_nfsim_refuses_times_that_are_not_finite(nfsim_xml: Path, simple_decay_net: Path, bad):
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(simple_decay_net)), method="nfsim", xml_path=str(nfsim_xml)
    )
    with pytest.raises(ValueError, match="sample_times must all be finite"):
        sim.run(sample_times=[0.0, 1.0, bad], seed=1, timeout=1.0)
    with pytest.raises(ValueError, match="t_span must be finite"):
        sim.run(t_span=(0.0, bad), n_points=3, seed=1)

    # The session takes bare times, refused in the C++ the way RuleMonkey's
    # session refuses them, so the two stay interchangeable: inf stepped for ever
    # and NaN labelled every row t = nan.
    with bngsim.NfsimSession(str(nfsim_xml)) as s:
        s.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="finite"):
            s.simulate(0.0, bad, n_points=3)
        with pytest.raises(bngsim.SimulationError, match="finite"):
            s.simulate(bad, 1.0, n_points=3)
        with pytest.raises(bngsim.SimulationError, match="finite"):
            s.step_to(bad)
        assert s.simulate(0.0, 1.0, n_points=2).n_times == 2


# ─── Issue #1523: compute_all_sensitivities ──────────────────────────────────

# A -> B at rate f() = if(time > tsw, 2k, k): a time switch at tsw = 5, the
# crossing stop whose filter read the ends of the empty grid.
SWITCH_NET = """begin parameters
    1 k 0.3
    2 tsw 5
end parameters
begin species
    1 A() 5
    2 B() 0
end species
begin functions
    1 f() if(time>tsw,2*k,k)
end functions
begin reactions
    1 1 2 f
end reactions
begin groups
    1 Atot 1
end groups
"""


@pytest.fixture
def switch_sim(tmp_path: Path) -> bngsim.Simulator:
    path = tmp_path / "switch.net"
    path.write_text(SWITCH_NET)
    return bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")


@pytest.mark.parametrize("params", [["k"], ["tsw"]])
@pytest.mark.parametrize("n_points", [0, 1, -1])
@pytest.mark.parametrize("t_span", [(0.0, 10.0), (0.0, 4.0)], ids=["switch-inside", "no-switch"])
def test_compute_all_sensitivities_refuses_the_grids_run_refuses(
    switch_sim, t_span, n_points, params
):
    # 0 with the switch inside the span was SIGSEGV; without it, a SimulationError
    # that blamed tolerances; -1 was std::length_error; 1 a one-row Result.
    with pytest.raises(ValueError, match=rf"n_points \({n_points}\) must be >= 2"):
        switch_sim.run(t_span=t_span, n_points=n_points)
    with pytest.raises(ValueError, match=rf"n_points \({n_points}\) must be >= 2"):
        switch_sim.compute_all_sensitivities(t_span=t_span, n_points=n_points, params=params)


@pytest.mark.parametrize("t_span", [(10.0, 0.0), (5.0, 5.0)])
def test_compute_all_sensitivities_refuses_a_span_that_does_not_advance(switch_sim, t_span):
    # (10, 0) returned rows at t = 10, A = 5 and dA/dk = 0, with nothing said.
    with pytest.raises(ValueError, match="must be > t_start"):
        switch_sim.compute_all_sensitivities(t_span=t_span, n_points=3, params=["k"])


def test_compute_all_sensitivities_still_differentiates_across_the_switch(switch_sim):
    # Closed form past the switch: A(t) = 5 exp(-k (2t - tsw)), so
    # dA/dk = -(2t - tsw) A and dA/dtsw = k A.
    r = switch_sim.compute_all_sensitivities(
        t_span=(0.0, 10.0), n_points=3, params=["k", "tsw"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(r.time, [0.0, 5.0, 10.0])
    k, tsw, t = 0.3, 5.0, 10.0
    a = 5.0 * math.exp(-k * (2.0 * t - tsw))
    columns = list(r.sensitivity_params)
    np.testing.assert_allclose(r.species[-1, 0], a, rtol=1e-7)
    np.testing.assert_allclose(
        r.sensitivities[-1, 0, columns.index("k")], -(2.0 * t - tsw) * a, rtol=1e-5
    )
    np.testing.assert_allclose(r.sensitivities[-1, 0, columns.index("tsw")], k * a, rtol=1e-5)


# ─── The backends' own refusal, for a caller of _bngsim_core ────────────────


def _spec(t_start=0.0, t_end=10.0, n_points=3, sample_times=None) -> TimeSpec:
    ts = TimeSpec()
    ts.t_start, ts.t_end, ts.n_points = t_start, t_end, n_points
    if sample_times is not None:
        ts.sample_times = sample_times
    return ts


_UNWALKABLE = [
    (_spec(n_points=0), "no points"),
    (_spec(n_points=-1), "no points"),
    (_spec(t_end=math.inf), "t_start and t_end must be finite"),
    (_spec(t_end=math.nan), "t_start and t_end must be finite"),
    (_spec(t_start=math.nan), "t_start and t_end must be finite"),
    (_spec(sample_times=[0.0, math.nan, 10.0]), "sample_times must all be finite"),
    (_spec(sample_times=[0.0, 5.0, math.inf]), "sample_times must all be finite"),
]
_UNWALKABLE_IDS = ["n0", "n-1", "end-inf", "end-nan", "start-nan", "st-nan", "st-inf"]


@pytest.mark.parametrize(("spec", "message"), _UNWALKABLE, ids=_UNWALKABLE_IDS)
@pytest.mark.parametrize(
    ("text", "backend"),
    [(SETTLES, "cvode"), (WITH_EVENT, "cvode"), (SETTLES, "ssa")],
    ids=["cvode-warm", "cvode-events", "ssa"],
)
def test_the_core_refuses_a_grid_it_cannot_walk(text, backend, spec, message):
    model = bngsim.Model.from_antimony_string(text)
    if backend == "cvode":
        sim = CvodeSimulator(model._core)
        with pytest.raises(ValueError, match=message):
            sim.run(spec)
        assert sim.run(_spec()).n_times == 3
    else:
        sim = SsaSimulator(model._core)
        with pytest.raises(ValueError, match=message):
            sim.run(spec, 1, 1.0)
        assert sim.run(_spec(), 1, 1.0).n_times == 3


def test_the_core_refuses_an_empty_grid_with_a_crossing_stop(switch_sim):
    # The path that died: a crossing stop inside the span (here the switch at
    # t = 5, on model time) sent the empty grid to the filter that reads
    # t_out.front() and t_out.back().
    from bngsim._bngsim_core import SolverOptions

    opts = SolverOptions()
    opts.set_crossing_stops([(5.0, -1, 0.0)], [])
    with pytest.raises(ValueError, match="no points"):
        switch_sim._sim.run(_spec(n_points=0), opts)
    assert switch_sim._sim.run(_spec(), opts).n_times == 3


@_NFSIM
@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_the_nfsim_core_refuses_times_that_are_not_finite(nfsim_xml: Path, bad):
    from bngsim._bngsim_core import NfsimSimulator

    sim = NfsimSimulator(str(nfsim_xml))
    with pytest.raises(ValueError, match="must be finite"):
        sim.run(_spec(t_end=bad), 42, 0.0)
    with pytest.raises(ValueError, match="sample_times must all be finite"):
        sim.run(_spec(sample_times=[0.0, 1.0, bad]), 42, 0.0)


# ─── Issue #1524: Result sizes its buffers in size_t ─────────────────────────
#
# The wrap needs 2^31 values, 16 GB of doubles, so the probe runs in a child
# that caps its own address space: the true request then fails as MemoryError at
# a fixed size, whatever the machine has, where the wrapped one was a buffer
# small enough to allocate and then be written past. Linux only, the one
# platform that enforces RLIMIT_AS; sanitizers.yml deselects it, because ASan
# reserves terabytes of address space up front.

_LINUX_ONLY = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="caps its own address space with RLIMIT_AS, which only Linux enforces",
)

_ADDRESS_SPACE_CAP = 4 * 1024**3
_N_SPECIES = 65536

_PROBE = f"""
import resource, sys, warnings
resource.setrlimit(resource.RLIMIT_AS, ({_ADDRESS_SPACE_CAP},) * 2)
warnings.simplefilter("ignore")
import bngsim
rows = [f"    {{i + 1}} S{{i}}() 1" for i in range({_N_SPECIES})]
net = "\\n".join(["begin parameters", "    1 k 1e-9", "end parameters", "begin species",
                  *rows, "end species", "begin reactions", "    1 1 2 k", "end reactions",
                  "begin groups", "    1 S0obs 1", "end groups", ""])
with open(sys.argv[1], "w") as fh:
    fh.write(net)
sim = bngsim.Simulator(bngsim.Model.from_net(sys.argv[1]), method="ssa")
for n_points in (3, 49152, 65537):
    try:
        r = sim.run(t_span=(0.0, 1.0), n_points=n_points, seed=1)
        print(n_points, "returned", r.species.shape)
    except MemoryError:
        print(n_points, "MemoryError")
"""


@_LINUX_ONLY
def test_a_result_too_large_to_allocate_is_a_memory_error(tmp_path: Path):
    # As int32, 49152 x 65536 wrapped negative ("ValueError: vector") and
    # 65537 x 65536 wrapped to 65536: one row, and SIGSEGV after the heap was
    # overwritten with species values. As size_t they are 25.8 and 34.4 GB.
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(tmp_path / "big.net")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"probe died with {proc.returncode}: {proc.stderr[-2000:]}"
    assert proc.stdout.split("\n")[:3] == [
        f"3 returned (3, {_N_SPECIES})",
        "49152 MemoryError",
        "65537 MemoryError",
    ], proc.stdout
