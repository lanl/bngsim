"""A rate law's condition on the clock is not stepped over, however it is
written (issues #899 and #904).

A ``.net`` rate law that is flat on both sides of a window gives CVODE no
error to see, so a step spans the window unless a stop sits on each of its
edges. The stop resolvers each read one shape (a residual linear in time, a
repeating schedule, a step of an affine argument, a sinusoid, a polynomial),
and a condition of any other shape got no stop and no warning:

* a ``mod`` of constants in a schedule's threshold (``mod(time(), P) <
  mod(P,7)``) was left as a symbolic call and the atom declined (#899): X(252)
  was 3 where 33 is exact;
* a comparison used as a number, with no ``if()`` around it
  (``k*((time()>=100)*(time()<=140))``), was never scanned at all (#904): A
  was 0 where 4 is exact, and the state-gated spelling returned Y = 0 too;
* ``trunc``, ``round``, ``frac``, ``sign``, a step of ``time()^2``, a window
  written with ``abs`` or ``min`` or ``exp``, and a pulse written with ``max``
  and ``abs`` and no comparison, each lost every pulse.

A condition no resolver reads is now evaluated over the window with the
model's own evaluator and stopped at wherever its value changes
(``bngsim._clock_scan``).

Oracles: closed forms. Each model is ``0 -> X`` at the rate law, so X(T) is
the integral of the rate law over the run: the on-time of a window, or the
area of a pulse.
"""

from __future__ import annotations

import contextlib
import math
import re

import bngsim
import numpy as np
import pytest
from bngsim import _clock_scan
from bngsim._switch_sensitivity import fixed_time_crossings

_NET = """begin parameters
    1 P 24
    2 w 0.5
    3 k 0.1
end parameters
begin species
    1 X() 0
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
"""

_T_END = 240.0


def _model(tmp_path, dose: str) -> bngsim.Model:
    path = tmp_path / "m.net"
    path.write_text(_NET.format(dose=dose), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _x_end(tmp_path, dose: str, t_end: float = _T_END, **kw) -> float:
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode", **kw)
    return float(np.asarray(sim.run(t_span=(0.0, t_end), n_points=3).species)[-1, 0])


# Pulses of time()^2 crossing 500n: on [sqrt(500n), sqrt(500n + 20)).
_SQUARE_PULSES = sum(math.sqrt(500 * n + 20) - math.sqrt(500 * n) for n in range(116))
# sqrt(time()) within 0.05 above an even number: t in [4j^2, (2j + 0.05)^2).
_ROOT_PULSES = sum((2 * j + 0.05) ** 2 - 4 * j * j for j in range(8))

# (rate law, exact X(240)). P = 24, w = 0.5, k = 0.1.
_WINDOWS = {
    # ── #904: a comparison as a number, outside every if() ──
    "product_of_comparisons": ("k*((time()>=100)*(time()<=140))", 4.0),
    "and_as_a_number": ("k*((time()>=100)&&(time()<=140))", 4.0),
    "word_and_as_a_number": ("k*((time()>=100) and (time()<=140))", 4.0),
    "difference_of_comparisons": ("k*((time()>=100)-(time()>140))", 4.0),
    "not_of_a_comparison": ("k*(not(time()<100))*(time()<=140)", 4.0),
    "comparisons_inside_max": ("k*max(0, (time()>=100)-(time()>=140))", 4.0),
    "comparisons_in_an_if_branch": ("if(k>0, k*((time()>=100)*(time()<=140)), 0)", 4.0),
    "narrow_pair": ("k*(time()>=100)*(time()<100.5)", 0.05),
    "schedule_as_a_number": ("k*(mod(time(), P) < w)", 0.5),
    "sinusoid_as_a_number": ("k*(sin(time()/5) > 0.999)", 0.1 * 8 * 10 * math.acos(0.999)),
    "abs_window_as_a_number": ("k*(abs(time()-120) < 0.5)", 0.1),
    # ── #899: a function of constants in the threshold ──
    "threshold_trunc_of_constants": ("if(mod(time(), P) < trunc(P/7), 1, 0)", 30.0),
    "threshold_round_of_constants": ("if(mod(time(), P) < round(P/7), 1, 0)", 30.0),
    # ── step calls no resolver read ──
    "sign_of_a_sawtooth": ("k*(1+sign(mod(time(),P)-12))/2", 12.0),
    "frac": ("if(frac(time()/P) < 0.02, 1, 0)", 4.8),
    "trunc_difference": ("trunc(time()/P) - trunc((time()-w)/P)", 4.5),
    "round_difference": ("round(time()/P) - round((time()-w)/P)", 5.0),
    "roundn_difference": ("roundn(time()/P, 0) - roundn((time()-w)/P, 0)", 5.0),
    "clamp_of_a_sawtooth": ("if(clamp(0, mod(time(),P), 1) < w, 1, 0)", 5.0),
    "inrange": ("inrange(0, mod(time(),P), w)", 5.0),
    "floor_of_time_squared": ("floor(time()^2/500) - floor((time()^2-20)/500)", _SQUARE_PULSES),
    "mod_of_a_square_root": ("if(mod(sqrt(time()), 2) < 0.05, 1, 0)", _ROOT_PULSES),
    # ── windows that are not monotone, and not a sinusoid or a polynomial ──
    "abs_window": ("if(abs(time()-120) < 0.5, 1, 0)", 1.0),
    "abs_window_a_thousandth_wide": ("if(abs(time()-120) < 0.0005, 1, 0)", 0.001),
    "gaussian_window": ("if(exp(-(time()-120)^2) > 0.5, 1, 0)", 2 * math.sqrt(math.log(2))),
    "min_window": ("if(min(time()-120, 121-time()) > 0, 1, 0)", 1.0),
    # ── a pulse with no comparison and no step call: it only bends ──
    "triangle": ("max(0, 1 - abs(time()-120)/0.25)", 0.25),
    "triangle_a_hundredth_as_wide": ("max(0, 1 - abs(time()-120)/0.0025)", 0.0025),
    "trapezoid": (
        "min(1, max(0, (time()-100)/0.01)) - min(1, max(0, (time()-100.02)/0.01))",
        0.02,
    ),
}


@pytest.mark.parametrize("dose, exact", _WINDOWS.values(), ids=_WINDOWS.keys())
def test_every_window_is_integrated(tmp_path, dose, exact):
    # The absolute part is the run's own atol (1e-8) on a pulse this small.
    assert _x_end(tmp_path, dose) == pytest.approx(exact, rel=5e-6, abs=2e-8)


@pytest.mark.parametrize(
    "name", ["product_of_comparisons", "abs_window", "trunc_difference", "triangle"]
)
def test_every_window_is_integrated_by_the_compiled_right_hand_side(tmp_path, name):
    dose, exact = _WINDOWS[name]
    assert _x_end(tmp_path, dose, codegen=True) == pytest.approx(exact, rel=5e-6)


# ── The same on a counter species, as a BNGL model spells the clock ─────────
_COUNTER_NET = """begin parameters
    1 P 24
    2 w 0.5
    3 k 0.1
    4 one 1
end parameters
begin species
    1 X() 0
    2 T() {t0}
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
    2 0 2 one #clock
end reactions
begin groups
    1 t 2
end groups
"""

# (rate law, the counter's value at the start, exact X(240))
_ON_A_COUNTER = {
    "abs_window": ("if(abs(t-120) < 0.5, 1, 0)", 0, 1.0),
    "abs_window_counter_from_7": ("if(abs(t-120) < 0.5, 1, 0)", 7, 1.0),
    "abs_window_already_open": ("if(abs(t-120) < 0.5, 1, 0)", 119.8, 0.7),
    "comparisons_as_numbers": ("k*((t>=100)*(t<=140))", 30, 4.0),
    "trunc_difference": ("trunc(t/P) - trunc((t-w)/P)", 0, 4.5),
    "frac_counter_from_5": ("if(frac(t/P) < 0.02, 1, 0)", 5, 4.8),
    "triangle": ("max(0, 1 - abs(t-120)/0.25)", 7, 0.25),
    "gaussian_window": ("if(exp(-(t-120)^2) > 0.5, 1, 0)", 3.3, 2 * math.sqrt(math.log(2))),
    "one_on_the_counter_one_on_time": (
        "if(abs(t-120) < 0.5, 1, 0) + if(abs(time()-60) < 0.5, 1, 0)",
        7,
        2.0,
    ),
    # t - time() is the counter's offset, 7: the first factor is 1 throughout.
    "both_in_one_condition": (
        "if(abs(t - time() - 7) < 0.5, 1, 0)*if(abs(time()-60) < 0.5, 1, 0)",
        7,
        1.0,
    ),
}


@pytest.mark.parametrize("dose, t0, exact", _ON_A_COUNTER.values(), ids=_ON_A_COUNTER.keys())
def test_every_window_on_a_counter_is_integrated(tmp_path, dose, t0, exact):
    path = tmp_path / "m.net"
    path.write_text(_COUNTER_NET.format(dose=dose, t0=t0), encoding="utf-8")
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    x_end = float(np.asarray(sim.run(t_span=(0.0, _T_END), n_points=3).species)[-1, 0])
    assert x_end == pytest.approx(exact, rel=5e-6, abs=2e-8)


# ── #899 ────────────────────────────────────────────────────────────────────
# mod(P,7) is 3 for P = 24: a 3-hour pulse a day, and a 1-hour pulse a day.
_MOD_OF_CONSTANTS = {
    "compared": ("if(mod(time(), P) < mod(P,7), 1, 0)", 33.0, [3, 24, 27, 48, 51]),
    "in_a_step": (
        "floor(time()/P + mod(P,7)) - floor((time()-1)/P + mod(P,7))",
        11.0,
        [1, 24, 25, 48, 49],
    ),
}


@pytest.mark.parametrize("dose, exact, stops", _MOD_OF_CONSTANTS.values(), ids=_MOD_OF_CONSTANTS)
def test_a_mod_of_constants_is_a_number_to_the_step_resolver(tmp_path, dose, exact, stops):
    """The issue's table. The exact resolver places these itself: the same
    stops the literal spelling gets."""
    assert _x_end(tmp_path, dose, 252.0) == pytest.approx(exact, rel=1e-6)
    m = _model(tmp_path, dose)
    found = fixed_time_crossings(m._core, 0.0, 60.0, m.time_discontinuity_conditions())
    assert found == pytest.approx(stops, abs=1e-9)
    literal = _model(tmp_path, dose.replace("mod(P,7)", "3"))
    assert found == fixed_time_crossings(
        literal._core, 0.0, 60.0, literal.time_discontinuity_conditions()
    )


# ── #904, what is scanned ───────────────────────────────────────────────────
def test_a_comparison_outside_an_if_is_a_condition(tmp_path):
    m = _model(tmp_path, "k*((time()>=100)*(time()<=140))")
    assert m.time_discontinuity_conditions() == ("time()>=100", "time()<=140")
    assert fixed_time_crossings(m._core, 0.0, _T_END, m.time_discontinuity_conditions()) == [
        100.0,
        140.0,
    ]


def test_a_comparison_of_parameters_alone_is_no_condition(tmp_path):
    """`k>0` reads no clock: nothing to stop at, and no scan of it."""
    m = _model(tmp_path, "if(k>0, k, 0)")
    assert m.time_discontinuity_conditions() == ()


def test_a_bending_call_of_the_clock_is_an_entry(tmp_path):
    m = _model(tmp_path, "max(0, 1 - abs(time()-120)/0.25)")
    assert m.time_discontinuity_conditions() == ("max(0, 1 - abs(time()-120)/0.25)",)


def test_a_bending_call_of_parameters_alone_is_no_entry(tmp_path):
    m = _model(tmp_path, "max(0, k - abs(w))")
    assert m.time_discontinuity_conditions() == ()


def test_the_sensitivity_scan_still_reads_step_calls_alone(tmp_path):
    """`steps_only` is what the sensitivity detector asks (issue #944). A
    comparison used as a number and a bending call are not step calls."""
    from bngsim._switch_sensitivity import time_discontinuity_conditions

    m = _model(tmp_path, "k*(time()>=100) + max(0, 1 - abs(time()-120)) + floor(time()/P)")
    assert time_discontinuity_conditions(m._core, steps_only=True) == ("floor(time()/P)",)


# ── #904, the state-gated spelling ──────────────────────────────────────────
# Issue #897's model with the window written as a product of comparisons. S
# decays as exp(-t/2), the window opens where t = 4*exp(-t/2) and is w wide.
_GATE_NET = """\
begin parameters
    1 k   10.0
    2 w   0.01
    3 kd  0.5
end parameters
begin functions
    1 onset() 4*Sobs
    2 r() {law}
end functions
begin species
    1 S() 1
    2 Y() 0
end species
begin reactions
    1 1 0 kd
    2 0 2 r
end reactions
begin groups
    1 Sobs 1
end groups
"""


def _gate_edge(w: float) -> float:
    t = 1.7
    for _ in range(60):
        t -= (t - 4.0 * math.exp(-t / 2.0) - w) / (1.0 + 2.0 * math.exp(-t / 2.0))
    return t


_GATE_EXACT = 10.0 * (_gate_edge(0.01) - _gate_edge(0.0))


@pytest.mark.parametrize(
    "law",
    [
        "k*((time()>=onset())*(time()<(onset()+w)))",
        "k*((time()>=onset())&&(time()<(onset()+w)))",
    ],
    ids=["product", "and"],
)
@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
def test_a_state_gated_window_written_as_a_number_is_rooted(tmp_path, law, rtol):
    path = tmp_path / "gate.net"
    path.write_text(_GATE_NET.format(law=law), encoding="utf-8")
    m = bngsim.Model.from_net(str(path))
    assert len(m.state_switch_root_conditions()) == 2
    r = bngsim.Simulator(m).run(sample_times=[0.0, 1.0, 3.0, 10.0], rtol=rtol, atol=rtol * 1e-2)
    y = float(np.asarray(r.species)[-1, list(r.species_names).index("Y()")])
    # Y is the difference of two edge times 0.0054 apart near 1.7, so an error
    # of order rtol in each is ~300 times larger relative to Y; the bound is
    # the if() spelling's own (issue #897), and far from the defect's Y = 0.
    assert y == pytest.approx(_GATE_EXACT, rel=1000 * rtol)


def test_a_sensitivity_run_still_refuses_a_comparison_outside_an_if(tmp_path):
    """The plain run roots it; a sensitivity run has a jump to apply there and
    no reading of it, and keeps its refusal."""
    path = tmp_path / "gate.net"
    path.write_text(
        _GATE_NET.format(law="k*((time()>=onset())*(time()<(onset()+w)))"), encoding="utf-8"
    )
    m = bngsim.Model.from_net(str(path))
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="not inside an if"):
        bngsim.Simulator(m, sensitivity_params=["k"]).run(t_span=(0.0, 10.0), n_points=3)


class _Options:
    """What `_apply_crossing_stops` writes to, kept where it can be read."""

    max_step_size = 0.0

    def __init__(self):
        self.stops = []

    def set_crossing_stops(self, stops, restarts=()):
        self.stops = [time for time, _clock, _value in stops]
        self.restarts = list(restarts)

    def set_crossing_probes(self, probes):
        pass


_SENS_NET = """begin parameters
    1 P 24
    2 w 0.5
    3 k 0.1
    4 t_c 120
    5 wd 0.25
    6 thr 20.085536923187668
    7 kd 0.01
    8 one 1
end parameters
begin species
    1 X() 0
    2 T() 7
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
    2 1 0 kd #deg
    3 0 2 one #clock
end reactions
begin groups
    1 t 2
end groups
"""


def _sens_model(tmp_path, dose: str) -> bngsim.Model:
    path = tmp_path / "sens.net"
    path.write_text(_SENS_NET.format(dose=dose), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _stops_of(tmp_path, dose: str, params) -> list[float]:
    sim = bngsim.Simulator(_sens_model(tmp_path, dose), method="ode", sensitivity_params=params)
    opts = _Options()
    sim._apply_crossing_stops(opts, 0.0, 10.0)
    return opts.stops


def test_a_sensitivity_run_stops_on_a_fixed_time_that_only_evaluating_places(tmp_path):
    """`exp(time()) >= e^3` reads no requested parameter: its change is a
    fixed time of the run with no jump to make in any column, and without the
    stop the columns were those of the stepped-over trajectory."""
    assert _stops_of(tmp_path, "if(exp(time())>=20.085536923187668,k,0)", ["k"]) == pytest.approx(
        [3.0], abs=1e-12
    )


def test_a_sensitivity_run_is_not_handed_a_stop_a_requested_parameter_moves(tmp_path):
    """The same condition on a threshold the run differentiates: the run reads
    it itself (and refuses it); a stop at today's value of the threshold is not
    a reading of how it moves."""
    assert _stops_of(tmp_path, "if(exp(time())>=thr,k,0)", ["k", "thr"]) == []
    assert _stops_of(tmp_path, "if(exp(time())>=thr,k,0)", ["k"]) == pytest.approx(
        [3.0], abs=1e-12
    )


def test_a_sensitivity_run_is_not_handed_a_scanned_comparison_on_a_counter(tmp_path):
    """A counter's `exp(t) >= e^3` is one of the sensitivity run's own roots,
    with a jump, and a stop there as well would be a second reading of one
    switch. The counter starts at 7, past the threshold, so only the absence
    of the stop is asked here."""
    plain = bngsim.Simulator(_sens_model(tmp_path, "if(exp(t - 7)>=thr,k,0)"), method="ode")
    opts = _Options()
    plain._apply_crossing_stops(opts, 0.0, 10.0)
    assert opts.stops == pytest.approx([3.0], abs=1e-9)
    assert _stops_of(tmp_path, "if(exp(t - 7)>=thr,k,0)", ["k"]) == []


def _columns(tmp_path, dose: str, params, **kw):
    sim = bngsim.Simulator(_sens_model(tmp_path, dose), sensitivity_params=params)
    r = sim.run(t_span=(0.0, _T_END), n_points=3, **kw)
    return float(np.asarray(r.species)[-1, 0]), np.asarray(r.sensitivities)[-1, 0]


def _decayed(windows) -> float:
    """X(240)/k for `X' = k on the windows - kd X`, kd = 0.01."""
    return sum(
        (math.exp(-0.01 * (_T_END - b)) - math.exp(-0.01 * (_T_END - a))) / 0.01
        for a, b in windows
    )


@pytest.mark.parametrize(
    ("dose", "windows"),
    [
        # trunc pulses on literal time: [24n, 24n + 0.5) for n >= 1 (trunc of
        # the small negative argument before 0.5 is 0, so none at 0).
        (
            "k*(trunc(time()/24) - trunc((time()-0.5)/24))",
            [(24.0 * n, 24.0 * n + 0.5) for n in range(1, 10)],
        ),
        # The same on a counter that starts at 7: the pulses come 7 early.
        (
            "k*(trunc(t/24) - trunc((t-0.5)/24))",
            [(24.0 * n - 7.0, 24.0 * n - 6.5) for n in range(1, 11)],
        ),
    ],
)
def test_a_sensitivity_run_integrates_the_pulses_a_plain_run_does(tmp_path, dose, windows):
    """With a column asked for a parameter the pulse does not read, the state
    is the plain run's and dX/dk is X/k. Before, the run with columns stepped
    over every pulse a plain run of the same model now stops at: X = 0 and a
    column of zeros."""
    x, column = _columns(tmp_path, dose, ["k"])
    exact = _decayed(windows)
    assert x == pytest.approx(0.1 * exact, rel=1e-5)
    assert column[0] == pytest.approx(exact, rel=1e-5)


def test_a_bend_is_stopped_at_in_a_sensitivity_run_whatever_it_reads(tmp_path):
    """A triangle of half-width 0.25 at t_c, with columns for t_c and wd. The
    rate law is continuous across each of its three bends, so there is no jump
    to make and the stop is all the run needs. X = 0 before."""
    x, column = _columns(tmp_path, "k*max(0, 1 - abs(time()-t_c)/wd)", ["k", "t_c", "wd"])
    # The area k*wd, decayed from t_c to the end.
    assert x == pytest.approx(0.1 * 0.25 * math.exp(-0.01 * 120.0), rel=1e-4)
    assert column[0] == pytest.approx(x / 0.1, rel=1e-5)
    assert column[2] == pytest.approx(x / 0.25, rel=1e-4)


@pytest.mark.parametrize(
    ("dose", "params", "names"),
    [
        ("k*(floor(time()/P) - floor((time()-w)/P))", ["k", "w"], "w"),
        ("k*(floor(time()/P) - floor((time()-w)/P))", ["k", "P"], "P"),
        ("k*(trunc(time()/P) - trunc((time()-w)/P))", ["P", "w"], "P"),
        ("k*(floor(time()^2/(20*P)) - floor((time()^2-20)/(20*P)))", ["P"], "P"),
        ("k*(trunc(t/P) - trunc((t-0.5)/P))", ["P"], "P"),
    ],
)
def test_a_step_call_a_requested_parameter_moves_is_refused_by_name(tmp_path, dose, params, names):
    """A step call outside every condition steps at times that move with a
    parameter it reads. Each step is then a jump in that column, which nothing
    computes. The run ground for minutes at the first edge (a pulse of floors,
    on main) or returned a column a quarter of the right size (`trunc`, once
    it was stopped at)."""
    sim = bngsim.Simulator(_sens_model(tmp_path, dose), sensitivity_params=params)
    with pytest.raises(bngsim.SensitivityUnsupportedError) as err:
        sim.run(t_span=(0.0, _T_END), n_points=3)
    message = str(err.value)
    # By the stops' own gate, or before it by the gate for a jump a requested
    # parameter moves the run across (issue #938), which names the call.
    if "step call outside any condition" in message:
        assert f"requested parameter {names}" in message
    else:
        assert "a requested parameter moves the run across" in message


# ── The SBML loader's models ────────────────────────────────────────────────
_ANTIMONY = "species X; X = 0; P = 24; w = 0.5; k = 0.1;\nJ: -> X; {law};\n"
_SBML_WINDOWS = {
    # The loader registers comparisons. A step or a bending call beside one had
    # no entry at all, where the same call alone was recovered and placed.
    "floor_pulse_beside_a_piecewise": (
        "floor(time/P) - floor((time-w)/P) + piecewise(0, time < 1, 0)",
        5.0,
    ),
    "triangle_beside_a_piecewise": (
        "max(0, 1 - abs(time-120)/0.25) + piecewise(0, time < 1, 0)",
        0.25,
    ),
    "triangle": ("max(0, 1 - abs(time-120)/0.25)", 0.25),
    "mod_of_a_square_root": ("piecewise(1, rem(sqrt(time), 2) < 0.05, 0)", _ROOT_PULSES),
    "comparisons_as_numbers": ("k*(time >= 100)*(time <= 140)", 4.0),
}


@pytest.mark.parametrize("law, exact", _SBML_WINDOWS.values(), ids=_SBML_WINDOWS.keys())
def test_an_sbml_models_window_is_integrated(law, exact):
    m = bngsim.Model.from_antimony_string(_ANTIMONY.format(law=law))
    r = bngsim.Simulator(m, method="ode").run(t_span=(0.0, _T_END), n_points=3)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(exact, rel=5e-6)


def test_a_recovered_entry_is_not_one_the_loader_registered():
    m = bngsim.Model.from_antimony_string(
        _ANTIMONY.format(law="floor(time/P) - floor((time-w)/P) + piecewise(0, time < 1, 0)")
    )
    conds = m.time_discontinuity_conditions()
    assert conds == ("(time()<1)", "floor((time()/P))", "floor(((time()-w)/P))")
    twice = bngsim.Model.from_antimony_string(
        _ANTIMONY.format(law="piecewise(1, (time >= 100) && (time <= 100.5), 0)")
    )
    assert twice.time_discontinuity_conditions() == ("(time()<=100.5)", "(time()>=100)")


# ── The scan itself, on an evaluator written here ───────────────────────────
_NUMPY = {
    "abs": np.abs,
    "floor": np.floor,
    "trunc": np.trunc,
    "mod": np.fmod,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "max": np.maximum,
    "min": np.minimum,
}


def _ev(text: str, times: np.ndarray) -> np.ndarray:
    """ExprTk's reading of the texts below, in numpy. A comparison is 1 or 0."""
    with np.errstate(invalid="ignore"):
        value = eval(text.replace("time()", "t").replace("^", "**"), {"t": times, **_NUMPY})  # noqa: S307
    return np.broadcast_to(np.asarray(value, dtype=float), times.shape).copy()


def _scan(text: str, lo=0.0, hi=240.0, **kw):
    sides = None
    kind = kw.pop("kind", "comparison")
    if kind == "comparison":
        lhs, rhs = text.split("<") if "<" in text else text.split(">")
        sides = (lhs, rhs)
    kw.setdefault("budget", 10_000)
    return _clock_scan.scan_changes(_ev, text, lo, hi, kind=kind, sides=sides, **kw)


def _before(x: float) -> float:
    return math.nextafter(x, -math.inf)


def test_an_edge_is_the_first_double_that_reads_the_new_value():
    edges = _scan("abs(time()-120) < 0.5")
    assert edges == pytest.approx([119.5, 120.5], abs=1e-12)
    for edge in edges:
        here, there = _ev("abs(time()-120) < 0.5", np.array([_before(edge), edge]))
        assert here != there


@pytest.mark.parametrize("half_width", [1e-3, 1e-6, 1e-9])
def test_a_window_narrower_than_a_grid_cell_is_found(half_width):
    """A grid cell is 240/16384, or 0.015. The window's residual has one turn
    there, which the search between grid points finds at any width."""
    edges = _scan(f"abs(time()-77.123) < {half_width!r}")
    assert len(edges) == 2
    assert edges[1] - edges[0] == pytest.approx(2 * half_width, rel=1e-6)


def test_a_smooth_window_narrower_than_a_grid_cell_is_found():
    edges = _scan("(time()-77.123)**2 < 1e-12")
    assert edges == pytest.approx([77.123 - 1e-6, 77.123 + 1e-6], abs=1e-9)


def test_a_window_a_few_doubles_wide_is_kept_for_an_event_trigger_alone():
    width = 8 * math.ulp(77.0)
    text = f"abs(time()-77) < {width!r}"
    assert _scan(text) == []
    assert len(_scan(text, touches=True)) == 2


def test_every_jump_of_a_step_is_found_whatever_its_argument():
    edges = _scan("floor(time()**2/500)", kind="value")
    assert edges == pytest.approx([math.sqrt(500 * n) for n in range(1, 116)], rel=1e-12)


def test_a_pulse_between_two_steps_is_found_at_any_width():
    """Each call's jumps are read from its own argument, so the pulse between
    the two calls' jumps is not something a grid has to land in."""
    edges = _scan("floor(time()/24) - floor((time()-1e-7)/24)", kind="truth")
    # Open from the start (the second floor is -1 there) to 1e-7, then at
    # 24, ..., 216 for 1e-7 each, and opening at 240, the end.
    assert len(edges) == 1 + 2 * 9 + 1
    assert edges[0] == pytest.approx(1e-7, rel=1e-6)
    assert edges[2] - edges[1] == pytest.approx(1e-7, rel=1e-3)


def test_a_bare_sawtooth_changes_only_where_it_jumps():
    """`mod(time(), 24)` is zero at one double a period, which is no window,
    and moves by an ulp at every other: neither is a change."""
    assert _scan("mod(time(), 24)", kind="value") == pytest.approx(
        [24.0 * n for n in range(1, 11)], abs=1e-9
    )


def test_a_bend_is_where_a_bending_call_turns():
    assert _scan("max(0, 1 - abs(time()-120)/0.25)", kind="bend") == pytest.approx(
        [119.75, 120.0, 120.25], abs=1e-9
    )


def test_a_side_that_is_not_a_number_at_first_does_not_hide_the_crossing():
    """sqrt(time()-5) is not a number before 5, and the comparison is false
    there, as it is until 9."""
    assert _scan("sqrt(time()-5) > 2") == pytest.approx([9.0], abs=1e-9)


def test_a_change_at_the_run_start_is_not_an_edge():
    assert _scan("time() > 0") == []
    assert _scan("time() > 100", lo=100.0) == []


def test_more_changes_than_the_budget_raises():
    with pytest.raises(_clock_scan.OverBudget):
        _scan("mod(time(), 0.01) < 0.003", budget=1000)


def test_step_calls_are_read_innermost_first():
    found = _clock_scan.phases("floor(floor(time()+3)/P) - trunc(mod(time(), P)/2)")
    # (`mod` and `trunc` by the size of their arguments: neither jumps at zero.)
    assert [phase.phase for phase in found] == [
        "(time()+3)",
        "(abs((time())/(P)))",
        "(floor(time()+3)/P)",
        "(abs(mod(time(), P)/2))",
    ]


# ── What the first review found ─────────────────────────────────────────────
# A level is crossed on two doubles: the first at which the argument has
# reached it and the first at which it has passed it. `floor` jumps on the
# first; `ceil`, `trunc` and `round` of a negative argument jump on the second,
# and where the argument is coarser than the clock (`time() - 1000` moves once
# in thirty doubles of the time) the two are far apart. Read for the first
# alone, each of these lost every pulse.
_SIDES = {
    # 30 windows of 0.1 before t = 30 (trunc is ceil there) and 209 after 31.
    "trunc(time()-30) - trunc(time()-30.1)": 23.9,
    "round(time()-30) - round(time()-30.1)": 24.0,
    # 7 for 0.8 of each unit and 6 for the rest.
    "round(time()) - round(time() - 6.8)": 1632.0,
    "ceil((time()-1000)/P) - ceil((time()-1000-w)/P)": 5.0,
    "trunc((time()-1000)/P) - trunc((time()-1000-w)/P)": 5.0,
    "round((time()+1000)/P) - round((time()+1000-w)/P)": 5.0,
    "k*(1+sign(mod(time()-1000,P)+12))/2": 12.0,
    "if(trunc(time()-30) - trunc(time()-30.001) > 0.5, 1, 0)": 0.239,
}


@pytest.mark.parametrize("dose, exact", _SIDES.items(), ids=range(len(_SIDES)))
def test_a_call_that_jumps_past_its_level_is_stopped_where_it_jumps(tmp_path, dose, exact):
    assert _x_end(tmp_path, dose) == pytest.approx(exact, rel=1e-4)


# Inside an if(), a bend is where the condition's residual turns; at the edge
# of a domain the residual stops being a number; and an equality of step calls
# holds for a stretch. None of the three had a grid point of its own.
_INSIDE = {
    "if(max(0, 1-abs(time()-120)/10) > 0.5, time()/100, 0)": 12.0,
    "if(min(time()-120, 121-time(), 5) > 0, 1, 0)": 1.0,
    "if(sqrt(time()-119.999) < 0.03, 1, 0)": 0.0009,
    "if(ln(time()-119.999) < -7, 1, 0)": math.exp(-7.0),
    "if(1/(time()-120.0073) > 1000, 1, 0)": 0.001,
    "if(mod(floor(time()),24) == 3, 1, 0)": 10.0,
    "if(floor(time()/0.5) != 240, 0, 1)": 0.5,
    # 10 sin(t/10) >= 9 on four stretches of 10 (pi - 2 asin 0.9).
    "if(floor(10*sin(time()/10)) == 9, 1, 0)": 40.0 * (math.pi - 2.0 * math.asin(0.9)),
}


@pytest.mark.parametrize("dose, exact", _INSIDE.items(), ids=range(len(_INSIDE)))
def test_a_window_inside_an_if_is_found_whatever_makes_it(tmp_path, dose, exact):
    assert _x_end(tmp_path, dose) == pytest.approx(exact, rel=1e-3)


def test_an_equality_of_things_that_move_is_still_an_instant(tmp_path):
    assert _model(tmp_path, "if(time() == 120, 1, 0)").time_discontinuity_conditions() == ()


def test_changes_a_few_doubles_apart_are_one_instant():
    """A comparison's rounding can flicker around a real crossing: true,
    true, false, true, false, false. Three changes are one, and the last of
    them is where the new value holds from. Read as a window with no measure,
    the pair was dropped and the real crossing with it."""
    at = np.array([10.0, np.nextafter(10.0, 11.0), np.nextafter(np.nextafter(10.0, 11.0), 11.0)])
    lone = np.array([50.0])
    edges = np.concatenate((at, lone))
    assert _clock_scan._one_instant(edges, keep_even=False).tolist() == [at[2], 50.0]
    # An even run is a window with no measure: nothing for a rate law, and the
    # whole of it for an event trigger.
    assert _clock_scan._one_instant(edges[:2], keep_even=False).tolist() == []
    assert _clock_scan._one_instant(edges[:2], keep_even=True).tolist() == edges[:2].tolist()


def test_a_crossing_the_comparison_flickers_around_is_integrated(tmp_path):
    """`sin(t) > cos(1.1 t)`: on-time by quadrature of the condition itself."""
    t = np.linspace(0.0, 240.0, 4_800_001)
    exact = float(np.mean(np.sin(t) > np.cos(1.1 * t)) * 240.0)
    got = _x_end(tmp_path, "if(sin(time()) > cos(time()*1.1), 1, 0)")
    assert got == pytest.approx(exact, abs=2e-4)


def test_a_grid_too_coarse_for_the_condition_is_refined_until_it_is_not():
    """5,000 windows of 0.01 in 5,000 time units, on a first grid of 4,096
    cells. A pass that found what its grid happened to catch was taken at its
    word: X = 20 of 50, then 17 of 1000."""
    edges = _scan("abs(mod(time(), 1) - 0.5) < 0.005", hi=5000.0, budget=1_000_000)
    assert len(edges) == 10_000


def test_an_oscillation_faster_than_the_grid_is_not_read_through_a_stroboscope():
    """`abs(sin(pi t))` against a level, over a window whose even grids all
    fall on the same few phases: two of them agreed on 3,392 changes of
    60,000. The points no period divides do not."""
    edges = _scan(
        "abs(sin(3.141592653589793*time())) > 0.9998766324816606", hi=30000.0, budget=1_000_000
    )
    assert len(edges) == 60_000


# sin(1/x) turns infinitely often toward x = 0: every finer grid finds more.
_ESSENTIAL = "sin(1/(time()-120.0003)) > 0.5"


def test_a_condition_that_never_settles_is_not_resolved():
    with pytest.raises(_clock_scan.Unresolved):
        _scan(_ESSENTIAL, hi=240.0, budget=10**9)


def test_a_condition_that_does_not_settle_is_refused_by_name(tmp_path):
    m = _model(tmp_path, f"if({_ESSENTIAL}, 1, 0)")
    with pytest.raises(bngsim.SimulationError, match="could not count its changes"):
        bngsim.Simulator(m, method="ode").run(t_span=(0.0, _T_END), n_points=3)


# A step call that jumps 2.4 million times, inside a comparison that changes
# once or never. Counting the jumps refused each.
_FEW = {
    "if(floor(time()/0.0001) < 5000, 1, 0)": 0.5,
    "(floor(time()/0.0001) < 5000)": 0.5,
    "if(ceil(time()*1e5) <= 5e4, 1, 0)": 0.5,
    "if(mod(time(), 1e-4) < 1, 1, 0)": 240.0,
    "if(abs(floor(time()*1e4)/1e4 - 120) < 0.25, 1, 0)": 0.4999,
}


@pytest.mark.parametrize("dose, exact", _FEW.items(), ids=range(len(_FEW)))
def test_many_jumps_inside_a_comparison_are_not_many_changes(tmp_path, dose, exact):
    assert _x_end(tmp_path, dose) == pytest.approx(exact, rel=1e-5)


def test_a_sawtooth_against_a_level_inside_its_range_is_still_over_budget(tmp_path):
    m = _model(tmp_path, "if(frac(time()/0.0001) < 0.5, 1, 0)")
    with pytest.raises(bngsim.SimulationError, match="#890"):
        bngsim.Simulator(m, method="ode").run(t_span=(0.0, _T_END), n_points=3)


def test_a_step_call_inside_a_comparison_is_not_a_condition_of_its_own(tmp_path):
    m = _model(tmp_path, "(floor(time()/0.5) < 100)*(abs(time()-10) < 3)")
    assert m.time_discontinuity_conditions() == ("floor(time()/0.5) < 100", "abs(time()-10) < 3")


def test_the_comparisons_of_a_deeply_nested_text_are_read_without_recursion():
    """A 3,000-term left-nested sum. Read by recursion it raised
    RecursionError, which was caught as "nothing found", and every clock
    condition of the model went with it."""
    from bngsim._switch_sensitivity import _iter_comparisons_anywhere

    text = "(time()>=0.5)"
    for i in range(1, 3000):
        text = f"({text}+(time()>={i + 0.5}))"
    atoms = list(_iter_comparisons_anywhere(text))
    assert len(atoms) == 3000
    assert atoms[0] == "time()>=0.5" and atoms[-1] == "time()>=2999.5"


def test_a_resolver_that_misreads_a_condition_is_overruled_by_the_evaluator(tmp_path):
    """`mod(rint(time()-0.01), 0.1) > floor(time())` has the shape the
    step-edge resolver reads and none of the edges it lists. What a resolver
    places is checked against the evaluator on a grid, and the condition is
    scanned where they disagree."""
    dose = "if(mod(rint(time()-0.01), 0.1) > floor(time()), 1, 0)"
    plain = _x_end(tmp_path, dose, t_end=20.0)
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    fine = float(np.asarray(sim.run(t_span=(0.0, 20.0), n_points=3, max_step=1e-3).species)[-1, 0])
    assert plain == pytest.approx(fine, rel=1e-6)


def test_the_counter_offset_is_handed_over_not_written_into_the_text(tmp_path):
    """A condition that reads a counter beside literal time is compiled once
    for every run: the counter's offset goes in a variable of the evaluator.
    Written into the text, each new offset was a new compiled expression kept
    for the life of the model."""
    m = _sens_model(tmp_path, "if(abs(t - time() - 7) < 1 && exp(time()) >= thr, k, 0)")
    core = m._core
    text = "exp(time()+bngsim_scan_shift_)"
    at = np.array([0.0, 1.0])
    assert core.evaluate_over_times(text, at, 0.0).tolist() == [1.0, math.e]
    assert core.evaluate_over_times(text, at, 1.0).tolist() == [math.e, math.exp(2.0)]
    # A clone has an evaluator of its own, and defines the variable again.
    assert m.clone()._core.evaluate_over_times(text, at, 1.0).tolist() == [math.e, math.exp(2.0)]


# ── What the second review found ────────────────────────────────────────────
# `time()*c` underflows to zero on the first few doubles after 0, so each of
# these changes value at about 2.5e-321: a stop CVODE cannot step to. The run
# returned zeros on two output points and died on eleven.
_FROM_ZERO = {
    "max(0, time()/1000)": 28.8,
    "k*abs(time()*0.001)": 2.88,
    "sign(time()/1000)": 240.0,
    "if(ceil(time()/1000) >= 1, 1, 0)": 240.0,
    "if(sqrt(time()/1000) > 0, 1, 0)": 240.0,
    "if(time()/1000 + time()/2000 > 0, 1, 0)": 240.0,
    "min(1, max(0, time()/200))": 140.0,
}


@pytest.mark.parametrize("n_points", [2, 11])
@pytest.mark.parametrize("dose, exact", _FROM_ZERO.items(), ids=range(len(_FROM_ZERO)))
def test_a_change_on_the_first_doubles_of_a_run_from_zero_is_its_start(
    tmp_path, dose, exact, n_points
):
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    r = sim.run(t_span=(0.0, _T_END), n_points=n_points)
    assert float(np.asarray(r.species)[-1, 0]) == pytest.approx(exact, rel=1e-6)


def test_the_start_of_a_run_is_measured_on_the_scale_of_its_window():
    assert _scan("time()/1000 > 0", lo=0.0, hi=240.0) == []
    # ...and a change a millionth of the window in is not the start.
    assert _scan("time() > 0.00024", lo=0.0, hi=240.0) == [math.nextafter(0.00024, 1.0)]


def _quadrature(tmp_path, dose: str, t_end: float, n: int = 1 << 22) -> float:
    """The integral of a rate law of the clock alone, by the midpoint rule on
    the model's own evaluator."""
    h = t_end / n
    core = _model(tmp_path, dose)._core
    return float(core.evaluate_over_times(dose, (np.arange(n) + 0.5) * h).sum() * h)


# Pulses bounded by the jumps of a step call whose argument turns. Within its
# rounding of a level the argument reads above and below it by turns, for a
# hundred doubles where it is turning, and the call flickers between two
# values. A jump asked to stand out from its neighbours there did not; a
# bracket a few doubles wide ended inside the flicker and read it backwards.
_TURNING = [
    "ceil((2*time()-8.458)*time()) - ceil((2*time()-8.458)*time() - 0.1)",
    "floor((1.378*time() - 10.953)*time() + -0.036) - "
    "floor((1.378*time() - 10.953)*time() + -0.036 - 0.2)",
    "round(0.985*time()^2 - 11.091*time() + -1.184) - "
    "round(0.985*time()^2 - 11.091*time() + -1.184 - 0.1)",
]


@pytest.mark.parametrize("dose", _TURNING, ids=range(len(_TURNING)))
def test_a_step_call_of_an_argument_that_turns_keeps_every_jump(tmp_path, dose):
    exact = _quadrature(tmp_path, dose, 10.0)
    assert _x_end(tmp_path, dose, t_end=10.0) == pytest.approx(exact, rel=2e-4)


def test_a_bracket_ends_where_the_argument_is_clear_of_its_rounding(tmp_path):
    """Every change of the call a dense sampling shows is one the scan has."""
    call = "floor((1.378*time() - 10.953)*time() + -0.036 - 0.2)"
    core = _model(tmp_path, call)._core
    t = np.linspace(0.0, 10.0, 2_000_001)
    v = core.evaluate_over_times(call, t)
    dense = t[1:][v[1:] != v[:-1]]
    edges = np.asarray(
        _clock_scan.scan_changes(
            lambda text, at: core.evaluate_over_times(text, np.ascontiguousarray(at)),
            call,
            0.0,
            10.0,
            kind="value",
            budget=10_000,
        )
    )
    assert len(edges) == len(dense)
    assert np.all(np.abs(edges - dense) <= 10.0 / 2_000_000 + 1e-12)


def test_the_search_for_a_change_ends_on_adjacent_doubles_however_it_is_cut():
    """One bracket is cut in 64 a round, a thousand in four, a million in two."""
    for n in (1, 1000, 5000):
        cut = np.linspace(0.3, 0.7, n)
        got = _clock_scan._bisect(
            lambda times, which, cut=cut: times >= cut[which],
            np.zeros(n),
            np.ones(n),
            np.zeros(n, bool),
        )
        assert np.array_equal(got, cut)


def test_the_count_reported_past_the_budget_is_of_every_call():
    """The second call's count alone was reported: "changes about 0 times,
    more than the 1,000,000 stops"."""
    with pytest.raises(_clock_scan.OverBudget) as err:
        _scan("floor(time()/0.5) - floor(time()/0.7)", kind="truth", budget=600)
    assert err.value.count > 600


def test_an_equality_of_calls_that_hold_their_value_is_a_window(tmp_path):
    assert _x_end(tmp_path, "if(max(abs(time()-120), 0.5) == 0.5, 1, 0)") == pytest.approx(1.0)
    assert _x_end(tmp_path, "if(clamp(119.5, time(), 120.5) == time(), 1, 0)") == pytest.approx(
        1.0
    )


def _trigger_stops(text: str) -> list[float]:
    model = bngsim.Model.from_antimony_string(text)
    opts = _Options()
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(opts, 0.0, 240.0)
    return [round(t, 6) for t in opts.stops]


def test_what_is_remembered_about_a_condition_is_about_what_its_names_stand_for():
    """Two models spell one trigger, `g > 0`, over an assignment rule that is
    a line in the first and a cubic through the same point in the second,
    which passes the linear resolver's probes. The second, run after the
    first, was answered with what was remembered of the first."""
    line = "species n = 0; g := time - 100; E: at (g > 0): n = n + 1;"
    cubic = (
        "species n = 0; g := (time - 100)*((time - 99.5)*(time - 156.5)/(99.5*156.5)); "
        "E: at (g > 0): n = n + 1;"
    )
    assert _trigger_stops(line) == [100.0]
    assert _trigger_stops(cubic) == [99.5, 100.0, 156.5]


def test_a_name_that_reads_the_clock_is_a_second_read_of_it():
    """`(time - 256)*g > 0` reads time once by its text, and the linear
    resolver's probes pass because the residual vanishes at 256. With
    `g := (time - 100)*(time - 100.5)` it changes at 100 and 100.5."""
    text = (
        "species n = 0; g := (time - 100)*(time - 100.5); E: at ((time - 256)*g > 0): n = n + 1;"
    )
    assert _trigger_stops(text) == [100.0, 100.5]


# ── What the third review found ─────────────────────────────────────────────
# A step call of an argument that itself jumps: the wrap of the `mod` inside.
# The bracket around each crossing was sized by how far the argument has to
# go to be clear of its own rounding, the rounding read as its largest move
# from one double to the next. At the wrap that move is the wrap, nothing is
# ever clear of it, and the bracket ran to a ten-thousandth of the time and
# took every crossing in it for one.
_ARGUMENT_JUMPS = {
    "sign(mod(time(), 60) - 20)": (172800.0, 57600.0),
    "sign(mod(time(), 0.05) - 0.02)": (240.0, 48.0),
    "trunc(mod(time() + 7, 60)/25)": (86400.0, 64800.0),
    "floor(4*frac(time()/60))": (86400.0, 129600.0),
}


@pytest.mark.parametrize(
    "dose, t_end, exact",
    [(k, *v) for k, v in _ARGUMENT_JUMPS.items()],
    ids=range(len(_ARGUMENT_JUMPS)),
)
def test_a_step_call_of_an_argument_that_jumps_keeps_every_jump(tmp_path, dose, t_end, exact):
    assert _x_end(tmp_path, dose, t_end=t_end) == pytest.approx(exact, rel=2e-4)


# Calls refused as ones whose changes could not be counted. Two calls of one
# text each found the jump they share, a few doubles apart; a sawtooth whose
# argument touches a level was given a jump there on one grid and not on the
# next; and the count went 46, 45, 46, 45.
_COUNTED = [
    ("floor(floor((4.053*sin(0.12*time() + 2.319) + ((time()-543.808) + 1.204*time())))/2)", 5000),
    ("mod(floor(3*sin(time()/7) + 2*cos(time()/3)), 2)", 5000),
    ("frac(1 + sin(2*_pi*time()/24))", 240),
    ("frac(3*sin(time()))", 100),
    ("mod(10*sin(time()/5)^2, 2)", 240),
]


@pytest.mark.parametrize("dose, t_end", _COUNTED, ids=range(len(_COUNTED)))
def test_a_step_call_is_counted_whatever_shares_or_grazes_its_jumps(tmp_path, dose, t_end):
    exact = _quadrature(tmp_path, dose, float(t_end))
    assert _x_end(tmp_path, dose, t_end=float(t_end)) == pytest.approx(exact, rel=1e-3)


def test_two_passes_agree_on_instants_not_on_edges():
    edges = np.array([1.0, np.nextafter(1.0, 2.0), 5.0, 5.0 + 1e-7])
    assert _clock_scan._instants(edges, 0.0, 10.0) == 3
    assert _clock_scan._instants(np.empty(0), 0.0, 10.0) == 0


# ── What the fourth review found ────────────────────────────────────────────
@pytest.mark.parametrize("height", [1e-8, 1e-10, 1e-11])
def test_a_window_where_the_argument_barely_passes_its_level_keeps_both_edges(tmp_path, height):
    """`(t - 5)^2 < height`: the argument is within its rounding of the level
    for hundreds of thousands of doubles on each side of the window. A
    bracket that stopped growing at 65,536 doubles ended inside that stretch,
    one edge or both were lost, and both passes agreed on what was left:
    X = 0 for 0.2."""
    dose = f"5000*(1 + sign({height!r} - (time()^2 - 10*time() + 25)))"
    width = 2.0 * math.sqrt(height)
    assert _x_end(tmp_path, dose, t_end=10.0) == pytest.approx(10000.0 * width, rel=2e-3)


# ── What the fifth review found ─────────────────────────────────────────────
# Windows a `sign` opens where a polynomial written out term by term turns
# just past a level. The argument is coarser than the clock there (one unit of
# its last place every thirty thousand doubles of the time for the first),
# so nearly every move it makes from one double to the next is zero, and its
# rounding measured by those was nothing: the bracket ended a few doubles on,
# inside the stretch the call flickers over. (centre, half-width squared)
_COARSE_TANGENCIES = [
    ("time()*time() - 8.302*time() + 17.230801", 1e-8),
    ("time()^2 - 16.06*time() + 64.48089999999999", 1e-8),
    ("time()^2 - 7.8*time() + 15.209999999999999", 1e-9),
    ("(time() - 3.151)*(time() - 5.151) + 1", 1e-7),
]


@pytest.mark.parametrize("poly, height", _COARSE_TANGENCIES, ids=range(len(_COARSE_TANGENCIES)))
def test_a_window_of_an_argument_coarser_than_the_clock_keeps_both_edges(tmp_path, poly, height):
    dose = f"5000.0*(1 + sign({height!r} - ({poly})))"
    exact = 10000.0 * 2.0 * math.sqrt(height)
    assert _x_end(tmp_path, dose, t_end=10.0) == pytest.approx(exact, rel=2e-3)


def test_rounding_that_comes_in_bursts_is_measured_by_the_bursts(tmp_path):
    """`frac(u)*4.891 + mod(time(), 47.062)`: the first term drops by 3.5e-14
    once in twelve doubles and the second climbs by 4.4e-16 on each. Measured
    by the many small moves, the argument was clear of a level sixteen doubles
    after dropping under it, where it had climbed back over; `ceil` read the
    same at both ends of the bracket and two of its 381 jumps had no stop.
    X(2.7) was 0.2% off."""
    dose = "ceil((frac(((time()-55.26))^2/77.056)*4.891 + mod(time(), 47.062)))"
    exact = _quadrature(tmp_path, dose, 2.7)
    assert _x_end(tmp_path, dose, t_end=2.7) == pytest.approx(exact, rel=2e-5)


def test_a_top_that_sits_on_a_level_has_its_pulse_on_every_grid(tmp_path):
    """Half way between two crossings of `3*sin(time())` through 2 is the top
    of the sine, which rounds to 3 on a stretch of doubles and makes `frac`
    of it drop to zero. Read as what the call holds between two crossings, it
    gave each top a pulse on one grid and none on the next, and the count of
    changes never settled. The top is searched for on every grid, and the
    pulse is the window between the two doubles its argument reaches 3 and
    leaves it on."""
    core = _model(tmp_path, "frac(3*sin(time()))")._core

    def ev(text, at):
        return core.evaluate_over_times(text, np.ascontiguousarray(at, dtype=float))

    phase = _clock_scan.phases("frac(3*sin(time()))")[0]
    f = lambda at: ev(phase.phase, at)  # noqa: E731
    top = 2.5 * math.pi
    crossed, _ = _clock_scan._crossings(f, _clock_scan._grid(0.0, 100.0, 4096), False, 10**6)
    jumps = _clock_scan._call_jumps(ev, phase.call, f, crossed, (0.0, 100.0), False)
    pulse = jumps[np.abs(jumps - top) < 1e-6]
    assert pulse.size == 2 and pulse[0] < top < pulse[1]
    stops = _scan_value(tmp_path, "frac(3*sin(time()))", 0.0, 100.0)
    assert np.count_nonzero(np.abs(stops - top) < 1e-6) == 2


def test_what_a_call_holds_between_its_jumps():
    def ev(text, at):
        return {
            "floor(x)": np.floor(at),
            "((x)-(frac(x)))": np.trunc(at),
            "frac(x)": at - np.trunc(at),
        }[text]

    held = _clock_scan._held(ev, "floor(x)")
    assert held.by == 0.0 and held.reading(np.array([1.5]))[0] == 1.0 and held.wraps is None
    held = _clock_scan._held(ev, "frac(x)")
    assert held.by == 0.5 and held.reading(np.array([2.5]))[0] == 2.0
    assert held.wraps(np.array([2.5]))[0] == 0.5
    # A modulus the clock moves holds nothing, and neither does an expression
    # that is not one call.
    assert _clock_scan._held(ev, "mod(x, time())") is None
    assert _clock_scan._held(ev, "floor(x) + 1") is None


# ── What the sixth review found ─────────────────────────────────────────────
def _scan_value(tmp_path, call: str, lo: float, hi: float, budget: int = 10**6) -> np.ndarray:
    core = _model(tmp_path, call)._core
    return np.array(
        _clock_scan.scan_changes(
            lambda text, at: core.evaluate_over_times(text, np.ascontiguousarray(at, dtype=float)),
            call,
            lo,
            hi,
            kind="value",
            budget=budget,
        )
    )


def _dense_jumps(tmp_path, call: str, lo: float, hi: float, n: int = 4_000_001) -> int:
    core = _model(tmp_path, call)._core
    values = core.evaluate_over_times(call, np.linspace(lo, hi, n))
    return int(np.count_nonzero(values[:-1] != values[1:]))


# A staircase on a slope too slow to move the argument in the doubles around a
# step. The step was the one move there and was taken for the argument's
# rounding; the bracket ran to its cap over the level the slope crosses next,
# and the two jumps were read as none.
_STAIRS_ON_A_SLOPE = [
    ("floor(floor(time()/1e-7)/3 + 5000*time())", 1000.0, 1000.001),
    ("ceil(floor(time()/1e-5)/7 + 30*time())", 30000.0, 30000.1),
    ("mod(rint(trunc(time()/0.000101)/2.64 + 1.606*time()), 2)", 100000.0, 100001.0),
]


@pytest.mark.parametrize("call, lo, hi", _STAIRS_ON_A_SLOPE, ids=range(len(_STAIRS_ON_A_SLOPE)))
def test_a_step_of_a_staircase_on_a_slope_is_not_its_rounding(tmp_path, call, lo, hi):
    stops = _scan_value(tmp_path, call, lo, hi)
    inside = stops[(stops > lo) & (stops < hi)]
    # (A jump on the window's own first or last double is the run's.)
    assert abs(inside.size - _dense_jumps(tmp_path, call, lo, hi)) <= 1


def test_a_slope_beside_a_pole_is_not_rounding(tmp_path):
    """Beside a zero of the sine `log(abs(sin))` moves by a whole number from
    one double to the next, on both sides of the pole. Taken for rounding,
    that ran the bracket of the innermost level over every level the argument
    climbs down to the pole, 2^24 doubles each way, and none had a stop."""
    call = "rint(log(abs(3.978*sin(0.327*time() + 2.414))) - -0.639)"
    lo, hi = 2400000.0, 2400020.0
    core = _model(tmp_path, call)._core
    stops = _scan_value(tmp_path, call, lo, hi)
    # Each level from -3 down to -9 is crossed on the way to a pole and on
    # the way back, a stop on each.
    zero = (math.pi * round((0.327 * 2400010.0 + 2.414) / math.pi) - 2.414) / 0.327
    assert lo < zero < hi
    for level in range(-9, -3):
        away = math.asin(math.exp(level + 0.5 - 0.639) / 3.978) / 0.327
        for at in (zero - away, zero + away):
            assert np.min(np.abs(stops - at)) < max(1e-6 * away, 4 * np.spacing(at)), (level, at)
    assert core.evaluate_over_times(call, np.array([zero - 1e-3]))[0] == -6.0


def test_a_sawtooth_is_not_stopped_where_only_its_argument_jumps(tmp_path):
    """`frac(floor(time()) + 0.3*time())` is `frac(0.3*time())`: at a whole
    number of the time its argument jumps by one and it does not move. What
    it takes off its argument does change there, and read as a missed wrap
    that put a stop on every whole number: 120 for 30."""
    stops = _scan_value(tmp_path, "frac(floor(time()) + 0.3*time())", 0.0, 100.0)
    assert stops.size == 30
    assert stops == pytest.approx(np.arange(1, 31) / 0.3, abs=1e-9)


@pytest.mark.parametrize("digits", [-1, -2, 0])
def test_roundn_rounds_to_no_fewer_digits_than_none(tmp_path, digits):
    """The evaluator takes the digits of `roundn` from 0 to 16, so
    `roundn(x, -1)` is `round(x)`. Read as rounding to tens, its jumps were
    looked for where `x/10 + 0.5` crosses a whole number: none of 13,260."""
    call = f"roundn(2.652*time(), {digits})"
    stops = _scan_value(tmp_path, call, 0.0, 50.0)
    assert stops == pytest.approx((np.arange(133) + 0.5) / 2.652, abs=1e-9)
    assert _clock_scan.phases(call)[0].phase.count("max(0,min(16,floor(") == 1


def test_which_moves_are_rounding():
    """`_clear` reads rounding from moves that go back and forth, from the
    smallest move made here or further out, and from a move onto the point
    itself that stands alone; a slope and a turn are none of it."""
    at = np.array([1000.0])
    ulp = np.spacing(1000.0)

    def steps(t):  # a step, on a slope that moves the argument once in 4,096 doubles
        return (
            np.floor((t - 1000.0) / (1e7 * ulp)) * 0.379
            + np.floor((t - 1000.0) / (4096 * ulp)) * 6e-8
        )

    # A step of the slope: the one move anywhere near, and the smallest.
    assert _clock_scan._rounding(steps, at + 3 * 4096 * ulp)[-1][0] == pytest.approx(6e-8)
    # The big step, asked about on the double it is made on, is what the
    # argument moves by there. (In a scan a step that a call inside the
    # argument makes is known for a jump and is not asked: `_clear_between`.)
    assert _clock_scan._rounding(steps, at)[-1][0] == pytest.approx(0.379, rel=1e-6)
    # ...and a few doubles beside it, it is no rounding of the slope's.
    assert _clock_scan._rounding(steps, at + 5 * ulp)[-1][0] == pytest.approx(6e-8)

    def pole(t):  # steep on both sides, one way on each
        return np.log(np.abs(t - 1000.0) / ulp + 0.5)

    assert _clock_scan._rounding(pole, at + 3 * ulp)[-1][0] < 0.1

    def noisy(t):  # back and forth by 3.5e-14 on a slope of 2.5e-15 a double
        k = np.rint((t - 1000.0) / ulp)
        return -2.5e-15 * k + 3.5e-14 * (np.mod(k, 12) / 12.0)

    assert _clock_scan._rounding(noisy, at)[-1][0] > 3e-14


# ── What the seventh review found ───────────────────────────────────────────
_Q4 = "(time()^4 - 20*time()^3 + 150*time()^2 - 500*time() + 625)"


def test_what_flickers_and_what_does_not():
    at = np.array([5.0])
    reach = np.array([2.0**24 * np.spacing(5.0)])
    rng = np.random.default_rng(7)

    def noisy(t):  # within its rounding of a level the whole way
        return 1e-13 * rng.integers(-3, 4, size=t.shape) + 1e-9 * (t - 5.0)

    def flickers(f):  # with zero the level, and an instant a quarter of the reach
        return _clock_scan._flickers(f, at, reach, True, 0.25 * reach[0])[0]

    assert flickers(noisy)
    assert not flickers(lambda t: np.zeros_like(t))  # flat
    assert not flickers(lambda t: np.floor((t - 5.0) / reach[0] * 4))  # a staircase
    assert not flickers(lambda t: -((t - 5.0) ** 2))  # a turn


def test_a_condition_that_could_not_be_counted_is_not_read_by_sampling(tmp_path, monkeypatch):
    """Past the stop budget a comparison is sampled for the few changes it
    may have. That reading sees nothing of a pulse between two samples, and a
    condition whose changes were found and could not be counted came back
    from it as one that never changes."""
    from bngsim import _switch_sensitivity as sw

    def never_settles(*args, **kwargs):
        raise _clock_scan.Unresolved(9)

    asked = []
    monkeypatch.setattr(_clock_scan, "scan_changes", never_settles)
    monkeypatch.setattr(sw, "_few_changes", lambda *a, **k: asked.append(a) or [])
    sw._STOP_MEMO.clear() if hasattr(sw, "_STOP_MEMO") else None
    dose = "1000*if(sign(sin(time())^3 - 0.5) > 0, 1, 0)"
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    with pytest.raises(bngsim.SimulationError, match="could not count its changes"):
        sim.run(t_span=(0.0, 10.0), n_points=3)
    assert not asked


@pytest.mark.parametrize("t_end", [500.0, 600.0, 3000.0])
def test_a_resolvers_stops_are_checked_between_the_grid_points_too(tmp_path, t_end):
    """`(time()-5)^4 < 1e-10` holds for 0.006 around 5. Over a run of 600 the
    polynomial resolver, which samples the window, had no stop for it; the
    check of its stops against the evaluator read the condition on a grid
    whose cells were wider than the window, saw no change, and agreed: X = 0
    for 6.32, on main too. The check now reads a comparison where the
    difference of its sides turns between two grid points."""
    dose = "1000*if((time()-5)^4 < 1e-10, 1, 0)"
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    x = float(np.asarray(sim.run(t_span=(0.0, t_end), n_points=7).species)[-1, 0])
    assert x == pytest.approx(1000.0 * 2.0 * 1e-10**0.25, rel=1e-4)


def test_a_stepping_call_jumps_across_a_bracket_it_reads_differently_at_the_ends():
    """Whatever it reads further out: a bracket that ends in the middle of a
    window has the window's other edge as far beyond its end as it is long."""

    def ev(text, at):
        return np.where(np.abs(at - 5.0) < 1.0, 1.0, -1.0)

    no = np.zeros(1, dtype=bool)
    jumps = _clock_scan._jumps_across(ev, "sign(q)", np.array([3.5]), np.array([5.0]), no, no)
    assert jumps == pytest.approx([4.0], abs=1e-12)


# ── What the eighth review found ────────────────────────────────────────────
def test_a_sawtooth_is_not_stopped_for_its_slope_across_one_double(tmp_path):
    """At t = 24000 one double of the clock moves this sawtooth by 1e-12
    through its slope, which is more than any rounding of it: read as a jump
    wherever its argument steps by a whole number, 3,144 stops for 96."""
    call = "frac(0.37*mod(time(), 10) + floor(mod(time(), 10)/0.01))"
    assert _scan_value(tmp_path, call, 24000.0, 24240.0).size == 96


def test_a_call_that_holds_nothing_beside_a_pole_is_judged_by_its_drift():
    def ev(text, at):
        return np.round(16.633 / (at - 5.0))

    lo, hi = np.array([5.0 - 1e-12]), np.array([5.0 + 1e-12])
    no = np.zeros(1, dtype=bool)
    assert _clock_scan._jumps_across(ev, "round(q)", lo, hi, no, no).size == 0


def test_flicker_is_asked_of_the_crossings_still_looked_at():
    """One crossing that never gets clear had every other one read across."""
    seen = []

    def f(t):
        seen.append(np.size(t))
        return np.where(np.abs(t) < 1.0, 0.0, t - np.round(t) + 0.5 * np.sign(t - np.round(t)))

    at = np.concatenate(([0.0], np.arange(10.0, 1010.0)))
    _clock_scan._clear(f, at, False, 0.0)
    # The value, the stencil and the points further out are read at every
    # crossing; nothing after them reads more points than there are crossings.
    assert max(seen[3:]) <= at.size


# ── What the ninth review found ─────────────────────────────────────────────
_P5 = "(time()^5 - 25*time()^4 + 250*time()^3 - 1250*time()^2 + 3125*time() - 3125)"
_P7 = (
    "(time()^7 - 35*time()^6 + 525*time()^5 - 4375*time()^4 + 21875*time()^3"
    " - 65625*time()^2 + 109375*time() - 78125)"
)
_P9 = (
    "(time()^9 - 45*time()^8 + 900*time()^7 - 10500*time()^6 + 78750*time()^5"
    " - 393750*time()^4 + 1312500*time()^3 - 2812500*time()^2 + 3515625*time() - 1953125)"
)


def test_rounding_that_repeats_every_few_doubles_is_seen_to_flicker():
    """Read at even places across a stretch a power of two of doubles long,
    a pattern of rounding eight doubles long was read at one place of itself
    by every point, and looked like a staircase of one step."""
    at = np.array([7300.0])
    ulp = np.spacing(7300.0)
    pattern = np.array([0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0])

    def f(t):
        k = np.rint((t - 7300.0) / ulp).astype(np.int64)
        return pattern[np.mod(k, 8)] + (k > 0)

    reach = np.array([2.0**24 * ulp])
    assert _clock_scan._flickers(f, at, reach, False, 0.25 * reach[0])[0]


def test_many_brackets_that_run_into_each_other_are_cut_in_one_pass():
    """A staircase with noise on every step: each crossing's bracket ends at
    the first cut after it. Cut by cut, 300,000 steps were hours."""
    n = 20_000
    near = np.arange(1, n + 1, dtype=float)
    rng = np.random.default_rng(5)

    def f(t):  # crosses a whole number at each, with a tenth of a step of noise
        return t + 0.1 * (rng.random(np.shape(t)) - 0.5)

    found = _clock_scan._brackets(f, near, False, 0.0)
    # One for each step (and one more where the last runs on past its step).
    assert n <= found.lo.size <= n + 2
    assert np.all(found.hi[:-1] < found.lo[1:])


def test_an_argument_flat_beside_a_jump_is_not_one_that_flickers(tmp_path):
    """`time() - mod(time() - 46.941, 31.695)` is flat before each wrap, a
    quarter of a step from the nearest level `round` reads, with a last
    place of rounding on it that goes back and forth. Taken for an argument
    that flickers across a level for as far as a bracket is grown, the call
    was refused."""
    stops = _scan_value(tmp_path, "round((time() - mod((time()-46.941), 31.695)))", 0.0, 100.0)
    assert stops == pytest.approx([46.941 - 31.695, 46.941 + 31.695], abs=1e-9)


# ── A call that flickers is refused by name ─────────────────────────────────
# An argument that is a small difference of large terms is within its own
# rounding of a level it crosses for longer than an instant of the run (a
# billionth of it), and the call reads on either side of that level by turns
# there. There is no time at which such a call changes. (Shorter than that, as
# for the squares and the turning arguments above, the flicker is one instant
# and has its stop.) Ten rounds of review went into placing a
# stop for it anyway: on a double where it reads its new value (the run
# restarted where it reads the old one again a double later, and one step
# crossed the window: X = 0), where the flicker ends (past whatever else
# closes a window before that: X = 0), with the rest of the text read inside
# the flicker (which cannot be told from the flicker). Each gave runs that
# returned nothing for a window they had stepped over, and it is refused.
_SHALLOW = "(time()*time() - 14600.0*time() + 53290000.0)"
_CUBE = "(time()^3 - 3000.0*time()^2 + 3000000.0*time() - 1000000000.0)"
_SQUARE = "(time()*time() - 10.0*time() + 25.0)"


def _pulse(k0: int, k1: int) -> str:
    """One from k0/100 to k1/100 and zero elsewhere, with two step calls."""
    return f"0.5*sign(floor(100*time()) - {k0 - 0.5}) - 0.5*sign(floor(100*time()) - {k1 - 0.5})"


# (rate law, run)
_FLICKERS = [
    # a quartic, and odd powers at their roots: billions, and more
    (f"5000.0*(1 + sign(1e-09 - {_Q4}))", (0.0, 10.0)),
    (f"1000*if(sign(1e-10 - {_Q4}) > 0, 1, 0)", (0.0, 1000.0)),
    (f"1000*if((sign(1e-10 - {_Q4}) >= 0) && (time() > 1), 1, 0)", (0.0, 1000.0)),
    (f"1000*if(sign(1e-09 - {_Q4}) > 0, 1, 0)", (4.99, 5.01)),
    (f"0.5 + 0.5*sign({_P5})", (0.0, 10.0)),
    (f"if(sign({_P7}) > 0 && time() < 5.3, 1, 0)", (0.0, 10.0)),
    (f"if(sign({_P9}) > 0, 1, 0)", (0.0, 10.0)),
    # a text that reads the flicker, and texts that do not
    (f"frac(0.3*(time() - 5.001) - 1.5*(1 - sign(1e-10 - {_Q4}))/2)", (0.0, 10.0)),
    (f"floor(10000*time() + sign({_P5}))", (4.9, 5.1)),
    ("sign(floor(2 + (time()^3 - 15*time()^2 + 75*time() - 125)) - 1.5)", (0.0, 10.0)),
    (f"floor(0.25*sign({_P7}) + 0.5 + {_pulse(488, 493)})", (0.0, 10.0)),
    (f"if({_pulse(508, 512)} + 0.2*sign({_P7}) > 0.5, 1, 0)", (0.0, 10.0)),
    # windows a few roundings deep, and one a rounding and a half
    (
        "1e6*(0.5 + 0.5*sign(1.73e-11 - (time()*time() - 246.912*time() + 15241.383936)))",
        (0.0, 160.0),
    ),
    (f"if(sign(1.12e-08 - {_SHALLOW}) > 0, 1, 0)", (0.0, 9490.0)),
    (f"if(sign(7.08e-08 - {_SHALLOW}) > 0, 1, 0)", (0.0, 9490.0)),
    # staircases with noise on every step
    (f"if(floor(3e5*{_CUBE}) > 50 && floor(3e5*{_CUBE}) < 80, 1, 0)", (999.9, 1000.1)),
    (f"if(floor(5e4*{_CUBE}) < 3, 1, 0)", (999.9, 1000.1)),
    (
        "if(mod(floor((time()*time() - 100.0*time() + 2500.0)/1e-10), 100) < 30, 1, 0)",
        (49.999, 50.001),
    ),
    (f"floor(3e13*{_SQUARE})", (4.99999, 5.00001)),
]


@pytest.mark.parametrize("dose, run", _FLICKERS, ids=range(len(_FLICKERS)))
def test_a_step_call_whose_argument_flickers_across_a_level_is_refused_by_name(
    tmp_path, dose, run
):
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    with pytest.raises(bngsim.SimulationError, match="within its own rounding of a level"):
        sim.run(t_span=run, n_points=3)


def test_the_refusal_names_the_call_and_where_it_flickers(tmp_path):
    call = f"sign({_P7})"
    core = _model(tmp_path, call)._core

    def ev(text, at):
        return core.evaluate_over_times(text, np.ascontiguousarray(at, dtype=float))

    with pytest.raises(_clock_scan.Flickers) as refused:
        _clock_scan.scan_changes(ev, call, 0.0, 10.0, kind="value", budget=1000)
    assert refused.value.call == call
    # The sign is noise from 4.95 to 5.06, and the stretch named is the first
    # of it that was found.
    assert 4.9 < refused.value.lo < refused.value.hi < 5.1
    # ...and it is one that could not be counted, to whoever asks only that.
    assert isinstance(refused.value, _clock_scan.Unresolved)


@pytest.mark.parametrize("max_step", [60.0, 5.0, 1.0, 0.1])
def test_no_step_bound_stands_for_the_stop_of_a_call_that_flickers(tmp_path, max_step):
    """A bound no wider than half the shortest stretch between two switches
    would, and what the rate law's other switches are is not known where a
    call of it flickers. `max_step=1` was taken for this one, whose window
    is 0.3 long, and the run returned 0."""
    dose = "if(sign(time()^3 - 15*time()^2 + 75*time() - 125) > 0 && time() < 5.3, 1, 0)"
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    with pytest.raises(bngsim.SimulationError, match="within its own rounding of a level"):
        sim.run(t_span=(0.0, 240.0), n_points=3, max_step=max_step)


def test_the_stretch_named_is_inside_the_run_and_the_first_of_those_found(tmp_path):
    dose = f"0.5 + 0.5*sign(1e-09 - {_Q4})"
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    with pytest.raises(bngsim.SimulationError) as refused:
        sim.run(t_span=(4.99, 10.0), n_points=3)
    text = str(refused.value)
    named = re.search(r"between t=([-0-9.e+]+) and t=([-0-9.e+]+) \(the first of (\d+) such", text)
    assert named is not None, text
    begins, ends = float(named.group(1)), float(named.group(2))
    # The window is 0.0056 either side of 5, and each of its edges flickers.
    assert 4.99 <= begins < ends < 5.0
    assert "run_until" not in text and "max_step" not in text


def test_a_flicker_of_a_few_doubles_is_still_one_jump(tmp_path):
    """The two doubles a level is crossed on, and the few around them on
    which a sum of terms rounds to either side, are a touch and no flicker:
    `floor` of this argument reads 12 and 13 by turns on four doubles at
    each of its steps."""
    call = "floor((time() - 3.7) + 0.2*time() + 12.3)"
    stops = _scan_value(tmp_path, call, 0.0, 10.0)
    assert stops.size == _dense_jumps(tmp_path, call, 0.0, 10.0, n=2_000_001)


def test_a_window_a_few_hundred_doubles_wide_is_no_flicker(tmp_path):
    """Both of its edges are clean, and its argument turns side once."""
    centre = 5.0
    half = 300.0 * float(np.spacing(centre))
    call = f"sign({half!r} - abs(time() - {centre!r}))"
    stops = _scan_value(tmp_path, call, 0.0, 10.0)
    assert stops == pytest.approx([centre - half, centre + half], abs=8 * np.spacing(centre))


def test_how_far_a_bracket_is_grown_where_the_argument_does_not_get_clear():
    """Sixteen million doubles; an instant where an instant is longer; and no
    more than four instants."""
    at = np.array([5.0])
    ulp = np.spacing(5.0)
    rng = np.random.default_rng(11)

    def flat_then_up(t):  # `max(0, t - 5)`: flat below, clear at once above
        return np.maximum(0.0, t - 5.0)

    for instant, grown in ((2.0**22, 2.0**24), (2.0**27, 2.0**27), (2.0**14, 2.0**16)):
        found = _clock_scan._clear(flat_then_up, np.nextafter(at, 6.0), True, instant * ulp)
        assert found.below[0] == pytest.approx(grown * ulp, rel=1e-3)
        assert not found.cleared_below[0] and found.cleared_above[0] and not found.flickers[0]

    def slow(t):  # crosses zero at 5 on a slope its rounding hides for 2^35 doubles
        k = np.rint((t - 5.0) / ulp)
        return 1e-22 * k + 1e-13 * rng.integers(-3, 4, size=np.shape(t))

    found = _clock_scan._clear(slow, at, True, 2.0**20 * ulp)
    assert not found.cleared_below[0] and not found.cleared_above[0] and found.flickers[0]


def test_a_piece_of_a_run_is_judged_by_an_instant_of_the_run(tmp_path):
    """A comparison whose floor jumps fifty million times and which changes
    once is read around that change alone, on a piece of the run a few cells
    long. The floor's argument is coarser than the clock and reads both
    values for a few thousand doubles at each step: more than a billionth of
    that piece, and nothing of the run, which is what it is held against.
    (0.117*t - 14.062815 = -10365469*5.44e-07 at t = 71.99999884.)"""
    dose = (
        "if(floor((((time() + 122698.9)*(3.464) + (time() + 3607.0)*(-3.9)) - (410909.28458)"
        " + 0.553*(time() - 120.195))/5.44e-07) < -10365469.0, 1, 0)"
    )
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    x = float(np.asarray(sim.run(t_span=(0.0, 240.0), n_points=3).species)[-1, 0])
    assert x == pytest.approx(71.99999884, abs=1e-6)


# A window a few roundings deep and narrower than an instant of the run. The
# argument is a coarse sum, never eight of its roundings clear of the level
# inside the window, so the brackets of its two edges run into each other and
# were one, which read the same at both of its ends: no jump, no stop and no
# word, where an earlier revision had kept both edges. Such a bracket is read
# from one end to the other.
# (call, run, where, the first and the last double of the window counted from
# there, by a walk over the doubles): 25,000 doubles four roundings deep; 2.4
# million, five; 570, six. Each edge is noise for a tenth of the window, and
# what the call does in that noise has stops of its own.
def _each_is_a_change(tmp_path, call: str, stops: np.ndarray) -> None:
    """Each stop is the first double of another value of the call."""
    core = _model(tmp_path, call)._core
    before = core.evaluate_over_times(call, np.nextafter(stops, -np.inf))
    at = core.evaluate_over_times(call, np.ascontiguousarray(stops))
    assert np.all(before != at)


_SPANNED = [
    (
        "sign(7.450580596923828e-09"
        " - abs(((1.07*time() + 10000000.0) - 0.42*time() - 10004745.0)))",
        (7292.7, 7307.3),
        (7300.0, -12442, 12441),
    ),
    (
        "sign(7.275957614183426e-11 - abs(((2.4*time() + 100000.0) - 1.37*time() - 100000.3811)))",
        (0.0, 7.37),
        (0.37, -1208920, 1194066),
    ),
    (
        "sign(6.821210263296962e-13 - abs(((2.47*time() + 1000.0) - 0.1*time() - 1011.85)))",
        (2.0, 12.0),
        (5.0, -274, 296),
    ),
]


@pytest.mark.parametrize("call, run, window", _SPANNED, ids=range(len(_SPANNED)))
def test_a_window_its_bracket_spans_keeps_both_edges(tmp_path, call, run, window):
    where, first, last = window
    ulp = float(np.spacing(where))
    stops = _scan_value(tmp_path, call, *run)
    assert stops.size >= 2
    slack = 0.2 * (last - first) * ulp + 4096 * ulp
    assert stops[0] == pytest.approx(where + first * ulp, abs=slack)
    assert stops[-1] == pytest.approx(where + last * ulp, abs=slack)
    _each_is_a_change(tmp_path, call, stops)
    # ...and the same window read by a condition runs.
    sim = bngsim.Simulator(_model(tmp_path, f"if({call} > 0, 1, 0)"), method="ode")
    sim.run(t_span=run, n_points=3)


def test_a_sawtooth_that_wraps_into_a_window_and_out_keeps_both_edges(tmp_path):
    """`frac` of an argument that is past 1 inside the window drops by one at
    each edge. Held against what the call does a bracket's width further out,
    which is the other edge's wrap and not how much it drifts, neither wrap
    was a jump."""
    call = (
        "frac(1 + 7.450580596923828e-09"
        " - abs(((1.07*time() + 10000000.0) - 0.42*time() - 10004745.0)))"
    )
    where, first, last = _SPANNED[0][2]
    ulp = float(np.spacing(where))
    stops = _scan_value(tmp_path, call, 7292.7, 7307.3)
    # (And it wraps where the argument passes -1, -2 and -3 on each side: six
    # instants more, some of them a few changes in the argument's rounding.)
    slack = 0.2 * (last - first) * ulp + 4096 * ulp
    edges = stops[np.abs(stops - where) < 1e-6]
    assert _clock_scan._instants(stops, 7292.7, 7307.3) == 8
    assert edges[0] == pytest.approx(where + first * ulp, abs=slack)
    assert edges[-1] == pytest.approx(where + last * ulp, abs=slack)


def test_a_bracket_that_is_not_plain_is_not_read_by_its_ends(tmp_path, monkeypatch):
    """What a call reads at the two ends of a bracket says what it does in
    between only where the argument is clear at both and crosses once. The
    bracket around this window is neither, and both edges are found with the
    reading of ends taken away altogether."""
    call, run, (where, first, last) = _SPANNED[0]
    monkeypatch.setattr(_clock_scan, "_jumps_across", lambda *args: np.empty(0))
    stops = _scan_value(tmp_path, call, *run)
    ulp = float(np.spacing(where))
    slack = 0.2 * (last - first) * ulp + 4096 * ulp
    assert stops[0] == pytest.approx(where + first * ulp, abs=slack)
    assert stops[-1] == pytest.approx(where + last * ulp, abs=slack)


@pytest.mark.parametrize("nudge", ["+ 1e-15", "- 1e-15"])
def test_a_top_a_rounding_or_two_past_a_level_is_a_window_with_clean_edges(tmp_path, nudge):
    """`3*sin(time()) + 1e-15` is past 3 by two of its last places for fifty
    million doubles at each top, and steps down from there one last place at
    a time: a window with clean edges, which is found and is no flicker."""
    call = f"floor(3*sin(time()) {nudge})"
    stops = _scan_value(tmp_path, call, 0.0, 100.0)
    # The 158 jumps of `floor(3*sin(time()))`, and the two edges of the window
    # at each of sixteen tops (or bottoms, with the nudge downward).
    assert stops.size == 190
    x = _x_end(tmp_path, f"3 + {call}", 100.0)
    assert x == pytest.approx(_quadrature(tmp_path, f"3 + {call}", 100.0), rel=1e-6)


def test_an_argument_that_comes_onto_a_level_is_a_window_while_it_stays(tmp_path):
    """`3*sin(time())` is 3 for ten million doubles around each top, and
    `floor` of it with it: a window rounding makes, 2e-8 wide, which the
    evaluator reads and the run is stopped at. (One bracket spans it and
    reads 2 at both ends; left at that, each top went unstopped, and a wider
    plateau of the kind with it: `floor(cos(0.001*t) + 2) > 2.5` is true for
    4.7e-5 at each top.)"""
    call = "floor(3*sin(time()))"
    core = _model(tmp_path, call)._core

    def ev(text, at):
        return core.evaluate_over_times(text, np.ascontiguousarray(at, dtype=float))

    def top(at):
        return ev("3*sin(time())", at)

    # The first double the argument reads 3 at.
    a, b = np.pi / 2 - 1e-6, np.pi / 2
    while np.nextafter(a, b) < b:
        mid = 0.5 * (a + b)
        a, b = (a, mid) if top(np.array([mid]))[0] >= 3.0 else (mid, b)
    found = _clock_scan._clear(top, np.array([b]), False, _clock_scan._instant(0.0, 100.0))
    assert not found.flickers[0]

    # The 158 jumps between its tops, and two at each of sixteen tops.
    assert len(_clock_scan.scan_changes(ev, call, 0.0, 100.0, kind="value", budget=10**6)) == 190
    pulse = _clock_scan.scan_changes(ev, call, 1.5707, 1.5709, kind="value", budget=10**6)
    assert len(pulse) == 2 and pulse[0] < np.pi / 2 < pulse[1]
    # Fifteen tops in a run to 1e5, each 4.71216e-5 wide (read on every
    # double at each of its edges).
    dose = "if(floor(cos(0.001*(time()-50000.5)) + 2) > 2.5, 1e4, 0)"
    assert _x_end(tmp_path, dose, 1e5) == pytest.approx(15 * 1e4 * 4.71216e-5, rel=1e-4)


# ── What the twelfth review found ───────────────────────────────────────────
def _x_over(tmp_path, dose: str, lo: float, hi: float, **kw) -> float:
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    return float(np.asarray(sim.run(t_span=(lo, hi), n_points=3, **kw).species)[-1, 0])


def _quadrature_over(tmp_path, dose: str, lo: float, hi: float, n: int = 1 << 22) -> float:
    h = (hi - lo) / n
    core = _model(tmp_path, dose)._core
    return float(core.evaluate_over_times(dose, lo + (np.arange(n) + 0.5) * h).sum() * h)


_CUBE5 = "(time()^3 - 15*time()^2 + 75*time() - 125)"

# (rate law, run): calls that have no jump where their argument crosses zero
_NO_LEVEL_AT_ZERO = [
    ("frac(time()^2 - 10*time() + 25)", (4.5, 5.5)),
    ("1 + trunc(time()^2 - 10*time() + 25)", (4.5, 5.5)),
    (f"5 + mod({_CUBE5}, 5)", (4.0, 6.0)),
    (f"1 + frac({_CUBE5})", (4.5, 5.5)),
]


@pytest.mark.parametrize("dose, run", _NO_LEVEL_AT_ZERO, ids=range(len(_NO_LEVEL_AT_ZERO)))
def test_zero_is_no_level_of_a_call_that_does_not_jump_there(tmp_path, dose, run):
    """`trunc`, `frac` and `mod` are continuous through zero. Read off the
    bare argument they had a level there, and an argument that flickers
    across zero (a square or a cube written out) had the call refused for a
    jump it does not have."""
    assert _x_over(tmp_path, dose, *run) == pytest.approx(
        _quadrature_over(tmp_path, dose, *run), rel=1e-6
    )


# Rate laws whose argument sits on a level and creeps off it, a last place at
# a time: c is where, on the clock the run is on.
_LEAVES_A_LEVEL = [
    "1000*(1 + round(0.5*tanh(-15 - 39*(time()-{c})^2)))",
    "1000*(1 + round(0 - 0.5/(1 + exp(-35 - 39*(time()-{c})^2))))",
    "1000*ceil(0.5*tanh(-15 - 39*(time()-{c})^2) + 0.5)",
]


@pytest.mark.parametrize("dose", _LEAVES_A_LEVEL, ids=range(len(_LEAVES_A_LEVEL)))
def test_a_call_changes_where_its_argument_leaves_a_level_it_sat_on(tmp_path, dose):
    """The argument of the first sits on -0.5, where `round` is -1, and
    leaves it upward for 0.6 around the middle of the run, where `round` is
    0. Read as `floor(x + 0.5)` the call had its change where x + 0.5 leaves
    zero; the evaluator computes `ceil(x - 0.5)` below zero, which leaves -1
    a fiftieth of a time unit later, far outside the bracket, and the window
    had no stop: X = 0 for 600. A phase is written so that the call is
    `floor` of it in the evaluator's own arithmetic.

    (On a clock where the stretch the argument creeps over is a few hundred
    instants of the run, and is read whole. Early in a run to 240 it is a
    million of them: see the next test.)"""
    dose = dose.format(c="1000000100")
    assert _x_over(tmp_path, dose, 1e9, 1e9 + 240.0) == pytest.approx(
        _quadrature_over(tmp_path, dose, 1e9, 1e9 + 240.0), rel=2e-4
    )


@pytest.mark.parametrize("dose", _LEAVES_A_LEVEL, ids=range(len(_LEAVES_A_LEVEL)))
def test_an_argument_that_creeps_off_a_level_is_refused_where_it_cannot_be_read(tmp_path, dose):
    """`tanh` of a large number leaves 1 one last place at a time, each held
    for a hundredth of a time unit, a hundred thousand instants of a run to
    240, and nothing says the next step is not back onto the level. The
    call would have to be read at every quarter of an instant across all of
    them. It is refused by name, with any step bound."""
    sim = bngsim.Simulator(_model(tmp_path, dose.format(c="100")), method="ode")
    for kw in ({}, {"max_step": 1e-3}):
        with pytest.raises(bngsim.SimulationError, match="more of the run than the call can be"):
            sim.run(t_span=(0.0, 240.0), n_points=3, **kw)


def _w_top(c: float, w: float) -> str:
    """Two windows 0.1 apart: `floor` of 1e9 plus a W-shaped top of five and
    three of its last places, with a dip of three between."""
    x = f"((time()-{c!r})/{w!r})"
    g = f"(-3 + 7.0*(2*{x}^2 - {x}^4) + (-1.0)*{x}*(1.5 - 0.5*{x}^2))"
    return f"1000*(floor(1000000000.0 + 1.1920928955078125e-07*{g}) - 999999999)"


_SAW = (
    "(1000000.0 + 3.4924596548080444e-10 - 1.3969838619232178e-09"
    "*((((time()-100000045757.61577)/328.9040256578399))^2 - 1)^2)"
)
# (rate law, run): on a clock that is large against the run
_FAR_CLOCK = [
    # a tread that sits on both bounds for 0.22
    ("1000*inrange(3, 3 + 2e-15*(time() - 240000100), 3)", (2.4e8, 2.4e8 + 240.0)),
    # a plateau that sits on a level for 1.5
    ("floor(tanh(40 - 39*(time()-1e9-50)^2)/2 + 0.5)", (1e9, 1e9 + 100.0)),
    # two shallow windows 0.1 apart
    (_w_top(2.4e8 + 100.0, 0.1), (2.4e8 + 100.0 - 0.145, 2.4e8 + 100.0 + 0.145)),
    # two windows of a sawtooth, 170 wide each
    (f"if(frac({_SAW}) < 0.5, 1, 0)", (1e11, 1e11 + 1e5)),
]


def test_a_stretch_the_argument_sits_on_zero_for_is_a_window_of_its_own(tmp_path):
    """`inrange(3, x, 3)` is one while x is 3 and zero on both sides of that.
    Here x is 3 for 0.002, two instants of a run to a million, on its way up:
    the bracket around it has x below 3 at one end and above at the other,
    and the call zero at both. It is split on the stretch x sits on 3 for."""
    dose = "1000*inrange(3, 3 + 2.2e-13*(time() - 1000000.5), 3)"
    # (3 + 2.2e-13*d rounds to 3 for |d| under 0.00100928: 2.01857 at 1000.)
    assert _x_over(tmp_path, dose, 0.0, 1e6 + 1.0) == pytest.approx(2.018566, rel=1e-4)
    assert _x_over(tmp_path, dose, 1e6, 1e6 + 1.0) == pytest.approx(2.018566, rel=1e-4)


_COARSE = "((0.65*time() + 100000.0) - 0.3*time() - 100001.75)"
_ZERO_FOR_A_TREAD = [
    # zero for one tread of the coarse sum at each edge, 50,000 doubles of t
    (f"sign(8.470329472543003e-20 - {_COARSE}*{_COARSE})", (3.5, 8.5)),
    (f"sign(8.470329472543003e-20 - {_COARSE}*{_COARSE})", (0.0, 11.0)),
    (f"sign(8.470329472543003e-20 - {_COARSE}*{_COARSE})", (4.45, 5.65)),
    (
        "sign(2.2351741790771484e-08 - sqrt((((2.07*time() + 10000000.0) - 1.36*time()"
        " - 10005183.0))^2))",
        (5110.0, 12410.0),
    ),
]


@pytest.mark.parametrize("call, run", _ZERO_FOR_A_TREAD, ids=range(len(_ZERO_FOR_A_TREAD)))
def test_a_tread_on_zero_at_each_edge_of_a_window_is_no_flicker(tmp_path, call, run):
    """`sign` of a constant less the square of a coarse sum is one in a
    window a third of an instant wide, and zero for one tread of the sum at
    each edge of it: -1, 0, 1, 0, -1. Split on the first double of a tread,
    the half that began on the second was held against the double before the
    tread, read as holding nothing there, and had no jump; the window was
    refused as a flicker in the runs it is less than an instant of. Each
    tread has a stop at both of its ends."""
    core = _model(tmp_path, call)._core
    stops = _scan_value(tmp_path, call, *run)
    before = core.evaluate_over_times(call, np.nextafter(stops, -np.inf))
    at = core.evaluate_over_times(call, stops)
    assert np.all(before != at)
    # -1 before the first and from the last on, and the window between them.
    # (More than four where an edge is not one step up but two, a step back
    # and two more: each is a change of the call, and has its stop.)
    assert stops.size >= 4 and before[0] == -1.0 and at[-1] == -1.0
    assert (at == 1.0).any() and (at == 0.0).any()


_PULSE = "(1 + sign(1e-08 - (time()*time() - 10*time() + 25)))/2"


@pytest.mark.parametrize(
    "call",
    [
        f"frac(0.3*time() + 0.2*{_PULSE})",
        f"frac(0.3*time() + 0.4*{_PULSE})",
        f"mod((0.3*time() + 0.2*{_PULSE})*2.5, 2.5)",
    ],
)
def test_a_sawtooth_moved_by_less_than_a_quarter_wrap_in_a_cut_bracket(tmp_path, call):
    """The pulse is one for 2e-4 around t = 5, and `sign` of the coarse
    quadratic is zero for one tread of it at each edge: the bracket there is
    split on that tread, and in each half the sawtooth around the pulse goes
    up by a tenth of a wrap, or a fifth. Asked, as a sawtooth with a cut end
    was, to take another whole number off its argument or to move by a
    quarter of a wrap, it had no jump, and the pulse no stop at either edge.
    A half has one end that is no cut, and what the sawtooth moves by is held
    against its drift beyond that end."""
    stops = _scan_value(tmp_path, call, 0.0, 10.0)
    assert np.any(np.abs(stops - 4.9999) < 1e-9)
    assert np.any(np.abs(stops - 5.0001) < 1e-9)


@pytest.mark.parametrize(
    "call, run, near",
    [
        # a coarse parabola that dips under zero for a hundred doubles
        (
            "sign((((time() + 2607.6)*(-2.6) + (time() + 619.615)*(1.393)) - (-5942.758199)"
            " + 0.00455*(time() - 21.642)^2))",
            (0.0, 100.0),
            21.642,
        ),
        # a pulse that moves a sawtooth by one whole wrap, half of it on each
        # side of the tread `sign` is zero for: no change of the sawtooth
        # across either edge but that tread
        (f"frac(0.3*time() + 1*{_PULSE})", (0.0, 10.0), 4.9999),
        (f"mod((0.3*time() + 1*{_PULSE})*2.5, 2.5)", (0.0, 10.0), 5.0001),
    ],
    ids=["dip", "frac", "mod"],
)
def test_what_goes_on_inside_a_bracket_shorter_than_an_instant_is_not_refused(
    tmp_path, call, run, near
):
    """A bracket cut in two is refused where one half has no jump and the
    call holds something else at one of its ends than at the other. In a
    bracket shorter than an instant that is one instant, like a flicker that
    short, and each of these was refused as a flicker."""
    stops = _scan_value(tmp_path, call, *run)
    assert np.any(np.abs(stops - near) < 1e-9)


def test_a_window_whose_edges_are_pulses_some_instants_long_is_not_lost(tmp_path):
    """`sign(1 - (L/c)^2)` for a coarse sum L is one in a window 180 instants
    wide of a run 0.005 long, and at each edge of it L goes from one tread to
    the next through a few pulses, each some instants long. A bracket grown
    four instants each way from a crossing began and ended between two
    pulses, read one side at both ends, and had no jump: every bracket of
    the window did, two passes agreed on no stops at all, and the window was
    stepped over with no word. A bracket that reads one side at both ends is
    looked into a touch either side of its crossing. The pulses are then
    found, as many as there are, or the passes do not settle and the call is
    refused."""
    call = (
        "sign(1 - ((((0.43*time() + 100000.0) - 0.24*time() - 100000.475))"
        "/7.275957614183426e-11)^2)"
    )
    try:
        stops = _scan_value(tmp_path, call, 2.4975, 2.5025)
    except _clock_scan.Unresolved:
        return
    # (The window is one from 2.4999999996 to 2.5000000005.)
    assert np.any(np.abs(stops - 2.4999999996) < 1e-10)
    assert np.any(np.abs(stops - 2.5000000005) < 1e-10)


@pytest.mark.parametrize("dose, run", _FAR_CLOCK, ids=range(len(_FAR_CLOCK)))
def test_a_bracket_is_no_longer_than_a_few_instants_on_a_large_clock(tmp_path, dose, run):
    """Sixteen million doubles of t = 2.4e8 are half a time unit, thousands
    of instants of a run 240 long. Grown that far, one bracket took in both
    edges of a tread 0.22 long and read `inrange` zero at both of its ends
    (X = 0 for 222); took in a plateau 1.5 long; took in two windows, was cut
    once, and lost the other; and held one window's wrap against the next
    window's. A bracket is grown no further than four instants."""
    assert _x_over(tmp_path, dose, *run) == pytest.approx(
        _quadrature_over(tmp_path, dose, *run), rel=2e-4
    )


_NO_JUMP_IN_A_HALF = [
    # identically 3: `ceil` of an argument that dips a last place under 3
    ("ceil(3 - max(0, 1e-15 - 1e-7*abs(time()-5)))", (0.0, 10.0)),
    # the call around does not change where the call inside does
    ("3 + floor(floor(3*sin(time()) + 1e-15)/2)", (0.0, 240.0)),
    ("ceil(floor(3*sin(time()) + 1e-15)/4)", (0.0, 240.0)),
]


@pytest.mark.parametrize("dose, run", _NO_JUMP_IN_A_HALF, ids=range(len(_NO_JUMP_IN_A_HALF)))
def test_a_half_the_call_does_not_change_in_is_no_refusal(tmp_path, dose, run):
    """A bracket cut inside a window of the argument has one edge of that
    window in each half. The call need not change there, and where it holds
    the same at both ends of a half there is nothing to find and nothing to
    refuse."""
    assert _x_over(tmp_path, dose, *run) == pytest.approx(
        _quadrature_over(tmp_path, dose, *run), rel=1e-6
    )


def test_a_comparison_past_the_budget_with_another_step_call_in_it_is_refused(tmp_path):
    """The samples that read a comparison past the budget for the few changes
    it has read nothing of another step call's jumps. This one is true for
    0.0117 between two samples 0.0146 apart, and came back as never
    changing: X = 0 for 11.7."""
    dose = (
        "if(floor(time()/0.0001)*1e-7"
        " + 5*(sign(time()-99.9917) - sign(time()-100.0034)) > 3, 1000, 0)"
    )
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    with pytest.raises(bngsim.SimulationError, match="more than the 1,000,000 stops"):
        sim.run(t_span=(0.0, 240.0), n_points=3)


_BETWEEN_TURNS = [
    ("1000*round(max(0, 1 - ((time()-100.01)/0.001)^2))", math.sqrt(2.0)),
    ("1000*round(clamp(0, 1 - ((time()-100.01)/0.001)^2, 1))", math.sqrt(2.0)),
    ("1000*round(max(0, 1 - abs(time()-100.01)/0.001))", 1.0),
    ("1000*round(1 - min(1, ((time()-100.01)/0.001)^2))", math.sqrt(2.0)),
]


@pytest.mark.parametrize("dose, exact", _BETWEEN_TURNS, ids=range(len(_BETWEEN_TURNS)))
def test_a_step_call_of_a_bump_changes_between_the_bumps_turns(tmp_path, dose, exact):
    """A bump 0.002 wide written with a bend is flat outside its two feet,
    and `round` of it is one between them where it is over a half. The feet
    were grid points and nothing between them was: stops at the feet alone,
    and X = 0."""
    assert _x_end(tmp_path, dose, 240.0) == pytest.approx(exact, rel=1e-5)


@pytest.mark.parametrize("centre", [0.01, 100.01, 239.99])
def test_a_pulse_that_turns_in_the_first_cell_of_the_grid_is_found(tmp_path, centre):
    """The extremum between grid points was searched for around interior
    samples that stand above or below both neighbours. A pulse 0.002 wide,
    0.01 after the start of the run, is in the first cell, where the first
    sample has one neighbour."""
    dose = f"1000*sign(1e-6 - (time()-{centre!r})*(time()-{centre!r})) + 1000"
    assert _x_end(tmp_path, dose, 240.0) == pytest.approx(4.0, rel=1e-5)


# ── What the thirteenth review found ────────────────────────────────────────
# All of it in brackets the argument is never clear of its rounding in: an
# argument that is a small sum on a large one (`1e9 + 3e-7*sin(...)`) moves by
# a few of its own last places in all. What the call reads at the two ends of
# such a bracket was taken to say what it does in between, and each way of
# taking it lost something with no word. Such a bracket is read from one end to
# the other, a quarter of an instant at a time, and as far out as the argument
# is within its rounding of a level.
_R9 = 1.1920928955078125e-07  # one last place of 1e9


# (rate law, run): one clean window of a sawtooth, wider than the brackets of
# its two edges are long. Held against what the call does a bracket's width
# further out, which was the window's other edge, neither wrap was a jump.
_WIDE_WINDOW = [
    (
        f"1000*frac(1000000001 + {_R9!r}*(-6 + 6.6*exp(-(((time() - 240000000.5556067)"
        "/0.0001220703125)/11.7)^4)))",
        (240000000.53338242, 240000000.5696129),
    ),
    (
        f"1000*mod(1000000000 + {_R9!r}*(-0.7 + 2.7*exp(-(((time() - 1000000000.5)"
        "/0.00048828125)/6)^4)), 1)",
        (1e9, 1e9 + 1.0),
    ),
    (
        "if(frac(2.9999999999999987 + 1.3322676295501878e-12*((time() - 10000.536855839891)"
        "/1.1947249278550851e-06)^2) > 0.5, 1000, 0)",
        (10000.0, 10001.0),
    ),
]


@pytest.mark.parametrize("dose, run", _WIDE_WINDOW, ids=range(len(_WIDE_WINDOW)))
def test_a_window_wider_than_its_edges_brackets_keeps_both_edges(tmp_path, dose, run):
    """X was 36.23 for 34.36, 1000 for 992.6 and 0 for 6.9e-5: no stop at
    either edge."""
    lo, hi = run
    n = 1 << 22
    core = _model(tmp_path, dose)._core
    # (Every double of the third's window: it is 7e-8 wide.)
    if "if(" in dose:
        at = 10000.536855839891 + np.spacing(10000.5) * np.arange(-60000, 60000)
        want = float(np.sum(core.evaluate_over_times(dose, at)[:-1] * np.diff(at)))
    else:
        want = _quadrature_over(tmp_path, dose, lo, hi, n)
    assert _x_over(tmp_path, dose, lo, hi) == pytest.approx(want, rel=5e-3)


def test_every_change_of_a_train_of_pulses_has_its_stop(tmp_path):
    """`floor` of 1e9 plus a sine two and a half last places high, eleven
    instants to the period: the brackets of its 372 changes run into each
    other. Cut once and read by its ends, the lot had one stop: X = 1000 for
    564."""
    call = "floor(1e9 + 3e-7*sin((time() - 1000000000)*1169.8148644639812))"
    stops = _scan_value(tmp_path, call, 1e9, 1e9 + 1.0)
    assert stops.size == _dense_jumps(tmp_path, call, 1e9, 1e9 + 1.0) == 372
    _each_is_a_change(tmp_path, call, stops)
    dose = f"1000*({call} - 999999999)"
    # (A stop on a clock of 1e9 is taken to within 2e-5 of it, and there are
    # 186 the rate law is on before.)
    assert _x_over(tmp_path, dose, 1e9, 1e9 + 1.0) == pytest.approx(
        _quadrature_over(tmp_path, dose, 1e9, 1e9 + 1.0), rel=2e-2
    )


@pytest.mark.parametrize("pulse", [-3.0, -6.0, -14.0])
def test_a_pulse_before_a_step_does_not_take_the_steps_stop(tmp_path, pulse):
    """A step of a coarse argument with a pulse a sixth of an instant wide a
    few instants before it. One bracket held both, its ends were on
    different treads, and its one stop was put on the first change the
    bisection met: the pulse. The step had none."""
    inst = 0.00048828125
    c = 1e9 + 50 * inst
    x = f"((time() - {c!r})/{inst!r})"
    call = f"floor(1000000000 + {_R9!r}*(3*tanh({x}/4) + 4*exp(-(({x} - ({pulse!r}))/0.1)^2)))"
    stops = (_scan_value(tmp_path, call, 1e9, 1e9 + 100 * inst) - c) / inst
    assert stops.size == 3
    assert stops == pytest.approx([pulse - 0.08, pulse + 0.08, -0.67], abs=0.03)


def test_a_call_that_is_one_on_a_level_alone_has_both_of_its_stretches(tmp_path):
    """`inrange(a, x, a)` for an x that goes through a twice, one tread of it
    on a each way: one for two stretches 0.375 long. A bracket cut on the
    hump between them had a half with no jump in it, and the refusal that
    stood behind that was asked only of brackets longer than an instant: X =
    0 for 749."""
    hump = f"1e9 + {_R9!r}*(-0.7 + 1.3*exp(-((time() - 100000000050)/0.345)^2))"
    for call in (
        f"inrange(1e9, {hump}, 1e9)",
        f"(1 - abs(sign(sqrt(({hump})^2) - 1e9)))",
    ):
        dose = f"1000*{call}"
        assert _x_over(tmp_path, dose, 1e11, 1e11 + 100.0) == pytest.approx(
            _quadrature_over(tmp_path, dose, 1e11, 1e11 + 100.0), rel=1e-2
        )


def test_two_clean_windows_of_a_sawtooth_are_no_flicker(tmp_path):
    """Two windows of a coarse argument 11 instants apart. One sawtooth's
    wrap, held against the next window's, was no jump; the half it was in
    had none; and the pair, clean to the double, was refused as a flicker."""
    inst = 4.76837158203125e-07
    c = 1000000.0305128287
    x = f"((time() - {c!r})/{inst!r})"
    arg = (
        f"(1000000 + 1.1641532182693481e-10*(-0.7 + 1.9*exp(-(({x})/7.366)^8)"
        f" + 1.3*exp(-(({x} - 23.4)/3.675)^8)))"
    )
    for call in (f"frac({arg})", f"floor(2*mod({arg}, 1))"):
        stops = (_scan_value(tmp_path, call, 1000000.0274615459, 1000000.0326721839) - c) / inst
        assert stops == pytest.approx([-8.15, 8.15, 19.43, 27.37], abs=0.02)


# A staircase of a coarse sum, one step in six of which comes as a pulse: up
# by a last place, back, and up again, each stretch thousands of instants of a
# run this short. A bracket is a few instants each way, the pulse was outside
# it, and the step had one stop for three changes: in 26 of 60 windows drawn.
# (run, changes of the call on a walk of two million points)
_STAIR = (
    "floor(((2.25*time() + 1000000000.0) - 0.9*time() - 1000000011.87055)/0.0005960464477539062)"
)
_STAIR_WINDOWS = [
    ((6.979184451578237, 6.979484451578237), 3),
    ((2.7099704051885984, 2.7109704051885983), 4),
    ((7.98147351198015, 7.98177351198015), 3),
    ((8.552442759765562, 8.555442759765562), 14),
]


@pytest.mark.parametrize("run, changes", _STAIR_WINDOWS, ids=range(len(_STAIR_WINDOWS)))
def test_a_pulse_beyond_a_steps_bracket_is_read_or_the_call_is_refused(tmp_path, run, changes):
    assert _dense_jumps(tmp_path, _STAIR, *run, n=2_000_001) == changes
    try:
        stops = _scan_value(tmp_path, _STAIR, *run)
    except _clock_scan.Flickers as refused:
        assert refused.unread
        return
    assert stops.size == changes
    _each_is_a_change(tmp_path, _STAIR, stops)


# (call, run): staircases that are no rounding. What moves the argument is a
# call or a comparison inside it, on one double, and what it does there is a
# jump: the argument is flat on both sides of it for as far as it is read.
_EXACT_STAIRS = [
    ("mod(((time() > 10) + (time() > 20) + (time() > 30)), 2)", (0.0, 100.0)),
    ("floor(0.8 + 0.4*(time() > 5.5) + 0.3*floor(time()/7))", (0.0, 100.0)),
    ("floor(if(time() < 5, 0.8, 1.2) + 0.3*(time() >= 7))", (0.0, 10.0)),
    ("ceil(((time() >= 100)*(time() <= 140))/2)", (0.0, 240.0)),
    ("mod(floor(time()), 24)", (0.5, 240.0)),
    ("floor(floor(time())/3)", (0.5, 30.0)),
    ("sign(if(sin(time()) > 0, sin(time()), 0))", (0.5, 20.0)),
    ("sign(max(0, sin(time())))", (0.5, 20.0)),
]


@pytest.mark.parametrize("call, run", _EXACT_STAIRS, ids=range(len(_EXACT_STAIRS)))
def test_a_staircase_of_calls_and_comparisons_is_not_its_own_rounding(tmp_path, call, run):
    stops = _scan_value(tmp_path, call, *run)
    assert stops.size == _dense_jumps(tmp_path, call, *run, n=2_000_001)
    _each_is_a_change(tmp_path, call, stops)


def test_a_jump_a_pass_missed_for_the_call_inside_is_a_jump_for_the_call_around(tmp_path):
    """`mod(trunc(u), 2.5)` for a sum of two sines u. A pass that did not
    find one of `trunc`'s jumps (u touches a whole number and comes back
    inside a cell) can still find `mod`'s there. With nothing known of what
    moved the argument, the staircase was taken for a coarse sum never clear
    of its level, and the call refused."""
    call = "mod(trunc((1.812*sin(2.283*time() + 0.026) + 4.769*sin(0.711*time() + 0.927))), 2.5)"
    assert _scan_value(tmp_path, call, 0.0, 5000.0).size == 14594  # by a walk


def test_an_argument_never_clear_of_a_level_is_read_where_the_run_is_short(tmp_path):
    """`1e9 + 3e-7*sin(...)` is within three of its own last places of 1e9
    throughout. On a clock of 1e9 a run of 100 is two hundred thousand
    instants and the call is read over all of it; from 0 it is a billion,
    and the call is refused by name."""
    far = "floor(1e9 + 3e-7*sin(time() - 1000000000))"
    stops = _scan_value(tmp_path, far, 1e9, 1e9 + 100.0)
    assert stops.size == _dense_jumps(tmp_path, far, 1e9, 1e9 + 100.0) == 31
    _each_is_a_change(tmp_path, far, stops)
    with pytest.raises(_clock_scan.Flickers) as refused:
        _scan_value(tmp_path, "floor(1e9 + 3e-7*sin(time()))", 0.0, 100.0)
    assert refused.value.unread
    sim = bngsim.Simulator(_model(tmp_path, "floor(1e9 + 3e-7*sin(time())) - 999999999"))
    with pytest.raises(bngsim.SimulationError, match="more of the run than the call can be"):
        sim.run(t_span=(0.0, 100.0), n_points=3)


# A coarse sum L beside a term that only shows while L is nothing: the argument
# moves by 1e-35 a double on the tread L is zero for, and by a whole last place
# of L onto it. Measured by the small moves the level was clear a few doubles
# on, the bracket was plain, and the pulse L makes some instants later (onto
# the tread, back, and on again) had no stop: in 10 of 40 windows drawn.
# (run, changes of the call on a walk of two million points)
_BESIDE_A_SMALL_TERM = [
    (
        "sign(((2.35*time() + 10000000000.0) - 1.59*time() - 10000000004.22381) + 1e-20*time())",
        (5.557276108625704, 5.558276108625704),
        5,
    ),
    (
        "sign(((1.54*time() + 100000000.0) - 0.85*time() - 100000002.17208105) + 1e-20*time())",
        (3.1479065563432016, 3.1480065563432014),
        5,
    ),
]


@pytest.mark.parametrize(
    "call, run, changes", _BESIDE_A_SMALL_TERM, ids=range(len(_BESIDE_A_SMALL_TERM))
)
def test_a_step_onto_a_level_that_stands_alone_is_the_arguments_rounding(
    tmp_path, call, run, changes
):
    assert _dense_jumps(tmp_path, call, *run, n=2_000_001) == changes
    try:
        stops = _scan_value(tmp_path, call, *run)
    except _clock_scan.Flickers as refused:
        assert refused.unread
        return
    assert stops.size == changes
    _each_is_a_change(tmp_path, call, stops)


def test_comparisons_in_a_text_are_found_each_as_a_text_of_its_own():
    found = _clock_scan._comparisons
    assert found("((time()>10)+(time()>20))") == ["time()>10", "time()>20"]
    assert found("if(time() > 5, 1.2, 0.8)") == ["time() > 5"]
    assert found("x*(a<=b and c!=d) + (e<f)") == ["a<=b and c!=d", "e<f"]
    assert found("1 + time() > 5") == ["1 + time() > 5"]
    assert found("min(1, time() >= 3) + 2") == ["time() >= 3"]
    assert found("(floor(time()/24))") == []


def test_a_bracket_is_read_every_quarter_instant_and_around_each_crossing():
    """The reading of a call across two brackets: a pulse 0.3 instants wide
    with no crossing known for it, a pulse a hundredth wide that begins on a
    crossing, and a step."""
    edges = np.array([3.3, 3.6, 5.0, 5.01, 7.0, 12.5])

    def reading(t):
        return np.searchsorted(edges, t, side="right").astype(float) % 2

    held = _clock_scan._Held(reading, 0.0, None)
    found = _clock_scan._combed(
        held, np.array([5.0, 7.0, 12.5]), np.array([0.0, 11.0]), np.array([10.0, 14.0]), 1.0
    )
    assert found == pytest.approx(edges, abs=1e-12)
    # ...and nothing is read between one bracket and the next.
    assert _clock_scan._combed(held, np.empty(0), np.array([8.0]), np.array([12.0]), 1.0).size == 0


def test_a_bracket_is_read_out_to_where_the_argument_is_clear():
    """A staircase of steps a tenth of a level high, one every seven time
    units on the way up and a last place of noise nowhere: within its
    rounding (the step) of the level at 0 for as long as it is under 0.8.
    The bracket a few instants long around the crossing is read out to there
    on the side the staircase goes on, and as far on the side it is flat."""

    def f(t):
        return 0.1 * np.floor(np.maximum(t, 0.0) / 7.0)

    near = np.array([7.0])
    no, yes = np.array([False]), np.array([True])
    found_at = _clock_scan._Brackets(
        lo=np.array([6.5]),
        hi=np.array([7.5]),
        lo_clear=no,
        hi_clear=no,
        rounding=np.array([0.1]),
        flickers=no,
        lo_cut=no,
        hi_cut=no,
        plain=no,
        lo_flat=yes,
        hi_flat=yes,
        jump=no,
    )
    lo, hi = _clock_scan._to_clear(f, found_at, np.array([True]), True, (-500.0, 500.0))
    assert near[0] - lo[0] == pytest.approx(hi[0] - near[0], rel=0.5)
    assert f(np.array([hi[0]]))[0] >= 0.8
    # Flat on both sides to the ends of the window: the bracket as it was.
    lo, hi = _clock_scan._to_clear(
        lambda t: (t >= 7.0).astype(float), found_at, np.array([True]), True, (-500.0, 500.0)
    )
    assert (lo[0], hi[0]) == (6.5, 7.5)


# A window narrower than a cell of the first grid is found by the search for an
# extremum between the grid's points. Where something inside the argument
# jumps, the grid has a cluster of adjacent doubles, and no extremum was
# looked for in the cell beside one: the two ends of that cell were taken to
# be as close as the cluster's.
_BESIDE_A_JUMP = (
    "sign(((time()*time() - 10.0*time() + 25.0) - 7.040103367572113e-13)"
    " + 1e-30*sign(((time()*time() - 10.003266628051168*time() + 25.01633580797055)"
    " - 2.670459911764231e-06)))"
)


def test_a_window_in_the_cell_beside_a_jump_inside_the_argument_is_found(tmp_path):
    """The parabola is under its level for 1.7e-6 around 5; the `sign` inside
    jumps 8.4e-7 before 5, which is where that window opens. The window had
    no stop: X = 0 for 1."""
    stops = _scan_value(tmp_path, _BESIDE_A_JUMP, 3.3, 7.7)
    half = 8.390532e-07  # sqrt(7.04e-13)
    assert stops.size >= 2
    assert stops.min() == pytest.approx(5.0 - half, abs=3e-9)
    assert stops.max() == pytest.approx(5.0 + half, abs=3e-9)
    dose = f"if({_BESIDE_A_JUMP} < 0, 595909.7434548252, 0)"
    assert _x_over(tmp_path, dose, 3.3, 7.7) == pytest.approx(1.0, rel=2e-3)


def test_an_extremum_is_looked_for_beside_a_cluster_of_adjacent_points():
    def f(t):
        return (t - 5.3) ** 2

    cluster = 5.0 + np.spacing(5.0) * np.arange(-3, 4)
    t = np.sort(np.concatenate((np.linspace(3.0, 7.0, 5), cluster)))
    found = _clock_scan._extrema(f, t, f(t))
    assert np.any(np.abs(found - 5.3) < 1e-6)


# ── The search for an extremum between grid points, and what it costs ────────
# The thirteenth review measured it: every scan searched the first cell of its
# grid and the last, eighty calls of the evaluator each, whenever the two
# samples there differed, which they do for anything that is not constant.
# `abs(time()-120) < 0.5` took 384 calls for the 52 it had taken.
def test_a_stretch_is_read_toward_the_point_at_its_end_for_where_it_turns():
    near, far = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    centre = np.array([1e-5, 0.3, 2.0])  # a turn close to the point, one far, none

    def f(t):
        which = np.broadcast_to(centre[:, None], (3, t.size // 3)).ravel()
        return (t - which) ** 2

    def one(row):
        g = lambda t: (t - centre[row]) ** 2  # noqa: E731
        return _clock_scan._ladder_turns(
            g,
            near[row : row + 1],
            g(near[row : row + 1]),
            far[row : row + 1],
            g(far[row : row + 1]),
        )

    for row in (0, 1):
        a, b, sign = one(row)
        assert a.size == 1 and a[0] < centre[row] < b[0] and sign[0] == 1.0
    assert one(2)[0].size == 0
    # ...all of them in one call of the evaluator.
    calls = []

    def counted(t):
        calls.append(t.size)
        return f(t)

    a, b, sign = _clock_scan._ladder_turns(counted, near, counted(near), far, counted(far))
    assert a.size == 2 and len(calls) == 3


def test_a_stretch_that_reads_the_same_and_then_moves_does_not_turn():
    # `time()^2 - 2000` is -2000 on every double up to 1e-7.
    def f(t):
        return t * t - 2000.0

    near, far = np.array([0.0]), np.array([240.0 / 4096])
    assert _clock_scan._ladder_turns(f, near, f(near), far, f(far))[0].size == 0


@pytest.mark.parametrize("many", [1, 3000])
def test_the_least_of_a_function_between_two_points(many):
    # By sections where there are few to find and by golden steps where there
    # are many: the same points, to the double or the one beside it.
    centre = 5.3 + 0.001 * np.arange(many)

    def f(t):
        return np.abs(np.asarray(t).reshape(many, -1) - centre[:, None]).ravel() ** 1.5

    found = _clock_scan._least(f, np.ones(many), np.full(many, 3.0), np.full(many, 9.0))
    assert np.max(np.abs(found - centre)) < 1e-7
    # A plateau: a point on it, not one past its edge.
    top = _clock_scan._least(
        lambda t: -np.minimum(1.0, 3.0 - np.abs(t - 6.0)),
        np.ones(1),
        np.array([3.0]),
        np.array([9.0]),
    )
    assert 4.0 <= top[0] <= 8.0


# (text, sides, kind): what the earlier revision took and what this one takes,
# in calls of the evaluator for a scan over (0, 240).
_ORDINARY = [
    ("abs(time()-120) < 0.5", ("abs(time()-120)", "0.5"), "comparison", 70),  # 52; was 384
    ("time()^2 > 2000", ("time()^2", "2000"), "comparison", 40),  # 28; was 194
    ("sqrt(time()) > 7.5", ("sqrt(time())", "7.5"), "comparison", 40),  # 28; was 194
    ("max(0, 1 - abs(time()-120)/0.25)", None, "bend", 70),  # 51; was 383
    (
        "min(time(), 100) > max(50, time() - 30)",
        ("min(time(), 100)", "max(50, time() - 30)"),
        "comparison",
        190,
    ),
]


@pytest.mark.parametrize("text, sides, kind, most", _ORDINARY, ids=range(len(_ORDINARY)))
def test_an_ordinary_condition_is_scanned_in_a_few_dozen_calls(tmp_path, text, sides, kind, most):
    core = _model(tmp_path, "1")._core
    calls = []

    def ev(what, at):
        calls.append(at.size)
        return core.evaluate_over_times(what, np.ascontiguousarray(at, dtype=float))

    _clock_scan.scan_changes(ev, text, 0.0, 240.0, kind=kind, sides=sides, budget=10**6)
    assert len(calls) <= most


def test_a_stretch_of_rounding_noise_is_searched_at_its_deepest_turn_alone():
    # Noise turns at every other rung. Each turn was searched and left a point
    # in the noise, at which the call was found to flip once more: 619 stops at
    # the two edges of one window for the 236 found without the ladder.
    rng = np.random.default_rng(7)
    near, far = np.array([0.0]), np.array([1.0])
    noise = {}

    def f(t):
        return np.array([noise.setdefault(float(x), float(rng.integers(-3, 4))) for x in t])

    a, b, sign = _clock_scan._ladder_turns(f, near, f(near), far, f(far))
    assert a.size == 1

    # A lean under the noise is the deepest turn there is.
    def g(t):
        return f(t) + 1e3 * (t - 0.3) ** 2

    a, b, sign = _clock_scan._ladder_turns(g, near, g(near), far, g(far))
    assert a.size == 1 and a[0] < 0.3 < b[0] and sign[0] == 1.0


# ── A line in time never steps back ──────────────────────────────────────────
# The clock times a number plus a number, by the text's own spelling, is a
# chain of sums, products and quotients with constants, and the double nearest
# each of those does not come down as its other operand goes up. However many
# of its last places such an argument stays beside a level for, it crosses it
# once, and its bracket is read by its ends. (Read like every other bracket
# the argument is not clear in, `ceil(1e12 + time())` over (0, 240), which
# moves once in 500 instants, was more than can be read, and refused.)
_LINES = [
    ("ceil(1e12 + time())", 240),
    ("floor(1e12 + time())", 240),
    ("floor((time() + 1e12)/24)", 10),
    ("floor(1e15 + 3*time())", 720),
    ("inrange(3, 3 + 2e-15*(time() - 100), 3)", 2),
    ("rint((ceil(1e12 + time())*2.02e-07 + 4.45e-06*time())*52.4)", 0),
]


@pytest.mark.parametrize("call, changes", _LINES, ids=range(len(_LINES)))
def test_a_step_call_of_a_line_in_time_is_read_however_coarse_the_line(tmp_path, call, changes):
    stops = _scan_value(tmp_path, call, 0.0, 240.0)
    assert stops.size == changes == _dense_jumps(tmp_path, call, 0.0, 240.0, 2_000_001)
    _each_is_a_change(tmp_path, call, stops)


@pytest.mark.parametrize(
    "call, line",
    [
        ("floor(1e12 + time())", True),
        ("ceil(-(3 - 2.5e-3*time())/-.5)", True),
        ("sign((time() - 100)/0.5 - 7)", True),
        ("inrange(3, 3 + 2e-15*(time() - 100), 3)", True),
        # two reads of the clock: this one does step back
        ("floor((1.77*time() + 1e9) - 0.68*time())", False),
        # a name can hide anything, a power turns, a quotient has a pole
        ("sign(k*time() - 3)", False),
        ("floor(time()^2/500)", False),
        ("floor(7.045/(time() - 687.424))", False),
        ("floor(time()/(2))", False),
        ("floor(exp(time()))", False),
        ("floor(time() % 7)", False),
    ],
)
def test_which_arguments_are_a_line_in_time_by_their_spelling(call, line):
    assert [_clock_scan._a_line_in_time(phase) for phase in _clock_scan.phases(call)] == [line]


def test_a_long_text_that_is_no_line_is_told_so_at_once():
    # (As one pattern over the whole text, a number of seventeen digits was cut
    # into pieces every way there is before the text was given up: a minute.)
    import time as clock

    hump = f"1e9 + {_R9!r}*(-0.7 + 1.3*exp(-((time() - 100000000050)/0.345)^2))"
    began = clock.perf_counter()
    for _ in range(100):
        assert _clock_scan._reads_of_the_clock_in_a_line(hump * 4) is None
    assert clock.perf_counter() - began < 0.5
    assert (
        _clock_scan._reads_of_the_clock_in_a_line("1.1920928955078125e-07*(time() - 3)/-2.5e3")
        == 1
    )
    assert _clock_scan._reads_of_the_clock_in_a_line("(1e9 + 3)*2/7") == 0
    assert _clock_scan._reads_of_the_clock_in_a_line("time()*time()") == 2
    for other in (
        "1e",
        "3x",
        "time",
        "2/(3)",
        "2**3",
        "time()/time()",
        "a + 1",
        "1 < 2",
        "tfun_k(time())",
    ):
        assert _clock_scan._reads_of_the_clock_in_a_line(other) is None


def test_two_arguments_a_call_compares_are_a_line_where_one_of_them_is_numbers():
    # `inrange(lo, x, hi)` compares x with each end: against another reading of
    # the clock the difference of the two can step back, against a number not.
    phases = _clock_scan.phases("inrange(1.77*time(), 0.68*time() + 1, 5)")
    assert [_clock_scan._a_line_in_time(phase) for phase in phases] == [False, True]
    # ...and where one of them is neither, the answer is no, and no error.
    phases = _clock_scan.phases("inrange(k, 0.68*time() + 1, exp(2))")
    assert [_clock_scan._a_line_in_time(phase) for phase in phases] == [False, False]


# The calls read through the size of their argument: `trunc`, `frac`, `round`
# and `mod` of a line are read by their brackets' ends too.
_SIZES_OF_LINES = [
    ("trunc(1e12 + time())", 240),
    ("round(1e12 + time())", 240),
    ("mod(time() + 1e12, 24)", 10),
    ("frac((time() + 1e12)/24)", 10),
    ("trunc(-1e12 - 0.25*time())", 60),
]


@pytest.mark.parametrize("call, changes", _SIZES_OF_LINES, ids=range(len(_SIZES_OF_LINES)))
def test_a_call_read_through_the_size_of_a_line_is_read_by_its_ends(tmp_path, call, changes):
    phases = _clock_scan.phases(call)
    assert [_clock_scan._a_line_in_time(phase) for phase in phases] == [True]
    stops = _scan_value(tmp_path, call, 0.0, 240.0)
    assert stops.size == changes
    _each_is_a_change(tmp_path, call, stops)


@pytest.mark.parametrize(
    "call",
    [
        "mod(time() + 1e12, 24 + time())",  # the modulus reads the clock
        "mod(time() + 1e12, period)",  # a name
        "frac(abs(time() - 5))",  # not a line
        "trunc(time()*time())",
        "floor(abs(time() - 5))",  # the size of a line, of a call that is not read through one
    ],
)
def test_a_call_whose_argument_is_no_line_is_not_read_by_that_rule(call):
    # (Of the calls that step. An `abs` inside is read for where it turns, by
    # a phase of its own, which may well be a line.)
    steps = [phase for phase in _clock_scan.phases(call) if not phase.bends]
    assert steps and not any(_clock_scan._a_line_in_time(phase) for phase in steps)


def test_a_sawtooth_of_an_argument_at_1e14_keeps_its_wraps(tmp_path):
    """The doubles of 1e14 are a sixty-fourth apart: `frac` of a line there
    climbs in sixty-fourths and drops by sixty-three of them where it wraps.
    What a jump of a sawtooth is held against was a touch of its argument's
    doubles, a whole wrap here, and no wrap was a jump: no stop on either of
    this one's (the revision before had both)."""
    call = "frac((time() - 1000000000.0)/131.76380996382414 + 100000000000000.9)"
    stops = _scan_value(tmp_path, call, 1e9, 1e9 + 240.0) - 1e9
    assert stops.size == 2 and stops[1] - stops[0] == pytest.approx(131.7638, abs=2e-3)
    core = _model(tmp_path, call)._core
    drop = core.evaluate_over_times(call, np.nextafter(stops + 1e9, -np.inf)) - (
        core.evaluate_over_times(call, stops + 1e9)
    )
    assert drop.tolist() == [0.984375, 0.984375]


# ── A sawtooth with nothing left of its argument to be one ───────────────────
# Beyond 2^48 the doubles of the whole number a sawtooth takes off are a
# sixteenth of a level apart or more: a period is a dozen doubles of the
# argument or fewer, the call goes up in a few steps and comes down by the
# rest, and a stop on each wrap is not a stop on each step.
_NO_SAWTOOTH = [
    ("mod(1000000000000.0 + time(), 0.00146484375)", (0.0, 0.146484375)),  # 12 doubles
    ("mod(1000000000000.0 + time(), 0.000146484375)", (0.0, 0.3)),  # 1.2
    ("mod(1e12 + time(), 0.0017)", (0.0, 1.7)),
    ("mod(time(), 1e-15)", (1.0, 1.0000001)),
    ("frac((time() - 1000000000.0)/131.76380996382414 + 1000000000000000.9)", (1e9, 1e9 + 240.0)),
]


@pytest.mark.parametrize("call, run", _NO_SAWTOOTH, ids=range(len(_NO_SAWTOOTH)))
def test_a_sawtooth_beyond_what_its_argument_resolves_is_refused(tmp_path, call, run):
    with pytest.raises(_clock_scan.Flickers) as refused:
        _scan_value(tmp_path, call, *run)
    assert refused.value.coarse and refused.value.call == call
    assert (refused.value.lo, refused.value.hi) == run


@pytest.mark.parametrize(
    "dose, run",
    [
        # With a stop on each of its 100 wraps X was 0.72 of the law's
        # integral (main, with none: 1.0008).
        ("100*mod(1000000000000.0 + time(), 0.00146484375)", (0.0, 0.146484375)),
        # No step of this one stands out as a wrap: no stop, and no word.
        ("100*mod(1000000000000.0 + time(), 0.000146484375)", (0.0, 0.3)),
    ],
)
def test_a_run_on_a_sawtooth_beyond_what_its_argument_resolves_says_so(tmp_path, dose, run):
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"beyond 2\^48") as refused:
        sim.run(t_span=run, n_points=3)
    assert "mod(1000000000000.0 + time()" in str(refused.value)
    assert "creep" not in str(refused.value)


# ── What goes wrong in the scan is not a condition that cannot be read ───────
# The run path declined a condition for anything the scan raised but its three
# refusals, with a line in the debug log: an IndexError inside the scan was a
# window stepped over and no word of it.
def test_a_defect_of_the_scan_is_said(tmp_path, monkeypatch, caplog):
    from bngsim import _switch_sensitivity

    def broken(*args, **kwargs):
        raise IndexError("index 3 is out of bounds for axis 0 with size 3")

    monkeypatch.setattr(_clock_scan, "scan_changes", broken)
    _switch_sensitivity._SCHEDULE_CACHE.clear()
    try:
        with caplog.at_level("WARNING", logger="bngsim"):
            _x_over(tmp_path, "if(exp(-((time() - 121.25)/3)^2) > 0.5, 1, 0)", 0.0, 240.0)
    finally:
        _switch_sensitivity._SCHEDULE_CACHE.clear()
    said = [
        r.getMessage() for r in caplog.records if "scan of the time condition" in r.getMessage()
    ]
    assert len(said) == 1
    assert "IndexError" in said[0] and "max_step" in said[0] and "121.25" in said[0]


def test_a_text_the_evaluator_cannot_be_asked_about_is_declined(tmp_path, monkeypatch, caplog):
    """...and that alone: the condition then has the stops it had before
    there was a scan, which is none, and the run is main's."""
    from bngsim import _switch_sensitivity

    asked = []

    def unreadable(ev, text, lo, hi, **kwargs):
        asked.append(text)
        return ev("no_such_call_anywhere(time())", np.array([lo, hi]))

    monkeypatch.setattr(_clock_scan, "scan_changes", unreadable)
    _switch_sensitivity._SCHEDULE_CACHE.clear()
    try:
        with caplog.at_level("WARNING", logger="bngsim"):
            x = _x_over(tmp_path, "if(exp(-((time() - 122.75)/3)^2) > 0.5, 1, 0)", 0.0, 240.0)
    finally:
        _switch_sensitivity._SCHEDULE_CACHE.clear()
    assert asked and x == 0.0
    assert not [r for r in caplog.records if "scan of the time condition" in r.getMessage()]


# ── A name that does not move in the run is a number ─────────────────────────
# A model writes `floor((time() - t0)/P)`, not the numbers. The run path knows
# which names no event or rule writes, and hands them to the scan.
_OFFSET_NET = """begin parameters
    1 t0 1e12
    2 P 7
end parameters
begin species
    1 X() 0
end species
begin functions
    1 dose() {dose}
end functions
begin reactions
    1 0 1 dose #_R1
end reactions
"""


def test_a_line_in_time_may_be_written_with_names_that_do_not_move():
    reads = _clock_scan._reads_of_the_clock_in_a_line
    assert reads("(time() - t0)/P", {"t0", "P"}) == 1
    assert reads("-(t0 + 2.5*time())*k/P", ["t0", "P", "k"]) == 1
    assert reads("t0*k + 3", {"t0", "k"}) == 0
    # A name the caller does not vouch for, a call, a dotted name, a name
    # that is the start of a call:
    assert reads("(time() - t0)/P", {"t0"}) is None
    assert reads("(time() - t0)/P") is None
    assert reads("time() - f()", {"f"}) is None
    assert reads("time() - k.f", {"k", "f"}) is None
    assert reads("time() - exp(t0)", {"exp", "t0"}) is None
    assert reads("time()/(P)", {"P"}) is None
    phase = _clock_scan.phases("ceil(t0 + time())")[0]
    assert _clock_scan._a_line_in_time(phase, {"t0"}) and not _clock_scan._a_line_in_time(phase)


def test_a_step_call_of_the_clock_and_a_parameter_is_read_however_coarse(tmp_path):
    """`ceil(t0 + time())` with t0 = 1e12 moves once in 500 instants of a run
    240 long. Without the names it was more than can be read."""
    path = tmp_path / "offset.net"
    path.write_text(_OFFSET_NET.format(dose="floor(ceil(t0 + time())/P)"), encoding="utf-8")
    model = bngsim.Model.from_net(str(path))
    core = model._core

    def ev(text, at):
        return core.evaluate_over_times(text, np.ascontiguousarray(at, dtype=float))

    call = "floor(ceil(t0 + time())/P)"
    with pytest.raises(_clock_scan.Flickers) as refused:
        _clock_scan.scan_changes(ev, call, 0.0, 240.0, kind="value", budget=10**6)
    assert refused.value.unread
    stops = np.array(
        _clock_scan.scan_changes(
            ev, call, 0.0, 240.0, kind="value", budget=10**6, constants={"t0", "P"}
        )
    )
    # 1e12 is 1 more than a multiple of seven, and `ceil` passes the next one
    # as the sum leaves 1e12 + 5: half a step of the sum (6.1e-5) after t = 5.
    assert stops.size == 34
    assert stops[:3] == pytest.approx([5.0, 12.0, 19.0], abs=1e-3)
    assert np.all(ev(call, np.nextafter(stops, -np.inf)) != ev(call, stops))
    # ...and the run asks the scan with the names it knows.
    opts = _Options()
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(opts, 0.0, 240.0)
    assert len(opts.stops) == 34


# ── A remainder is exact, and the quotient its wraps are read by is rounded ──
# `mod(x, m)` wraps where x passes a multiple of m. Its phase is x/m rounded,
# and where two values of a coarse x in a row give a quotient that rounds to
# the same whole number with the multiple between them, the wrap is on the
# step from one to the other: on neither of the two doubles the phase crosses
# the level on, and in neither's bracket. Read by its brackets alone, `mod`
# of a line had a stop for a third of its wraps; left out of the line rule,
# it was refused as more than can be read from an offset of 1e8 up. A call of
# a line is read between its brackets as well.
_REMAINDERS = [
    ("mod(1e12 + time(), 0.01)", (0.0, 3.0), 300),
    ("mod(1e12 + time(), 0.3)", (1.3995, 1.9), 2),
    ("mod(1e12 + time(), 0.3)", (1.3995, 1.6), 1),
    ("mod(240000000.0 + time(), 4e-6)", (100.0, 100.004), 1000),
    ("mod(1000000000.0 + time(), 0.001)", (0.0, 1.0), 1000),
    ("mod(3.3e12 + 0.7*time(), 0.03)", (0.0, 10.0), 233),
]


@pytest.mark.parametrize("call, run, wraps", _REMAINDERS, ids=range(len(_REMAINDERS)))
def test_a_remainder_of_a_coarse_line_has_every_wrap(tmp_path, call, run, wraps):
    stops = _scan_value(tmp_path, call, *run)
    assert stops.size == wraps
    # Each is the first double after a wrap: the remainder drops there.
    core = _model(tmp_path, call)._core
    before = core.evaluate_over_times(call, np.nextafter(stops, -np.inf))
    at = core.evaluate_over_times(call, np.ascontiguousarray(stops))
    assert np.all(at < before)
    # ...and no wrap is without one.
    dense = np.linspace(*run, 2_000_001)
    drops = dense[1:][np.diff(core.evaluate_over_times(call, dense)) < 0.0]
    reach = 4.0 * _clock_scan._instant(*run) + 2.0 * (dense[1] - dense[0])
    assert all(np.min(np.abs(stops - drop)) <= reach for drop in drops)


@pytest.mark.parametrize(
    "dose, run, exact",
    [
        ("if(mod(1000000000.0 + time(), 0.001) < 0.0005, 1, 0)", (0.0, 1.0), 0.5),
        ("if(mod(1700000000.0 + time(), 0.01) < 0.005, 1, 0)", (0.0, 10.0), 5.0),
        ("floor(mod(100000000.0 + time(), 0.001)/0.001 + 0.25)", (0.0, 0.5), 0.125),
        ("if(mod(100000000000.0 + time(), 0.3) < 0.15, 1, 0)", (0.0, 240.0), 120.0),
        # (a run that ends on a wrap)
        ("100*mod(100000000000.0 + time(), 0.3)", (7.0, 8.0), 15.99984),
        ("floor(mod(100000000000.0 + time(), 0.3)/0.3 + 0.25)", (7.0, 8.0), 0.300003),
    ],
)
def test_a_remainder_of_a_line_at_a_large_offset_is_integrated(tmp_path, dose, run, exact):
    """Each was refused once `mod` was read every quarter instant: more than
    can be read, in the words written for an argument that creeps."""
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    result = sim.run(t_span=run, n_points=2)
    assert float(np.asarray(result.species)[-1, 0]) == pytest.approx(exact, rel=2e-5)


# ── `rint` adds its half to the argument the evaluator computed ──────────────
def test_the_phase_of_rint_is_its_argument_evaluated_and_a_half_added(tmp_path):
    """Read as the text `((x)+0.5)`, the evaluator added x's constant to the
    half first, and that sum reached 1 at 99.8646, 0.03 before `rint` of x
    changes: with a window that begins a few doubles of the phase from 1,
    there was no stop for the change at all (main has one, at 99.92)."""
    call = "rint(0.4999999999999 + 1e-15*time())"
    (phase,) = _clock_scan.phases(call)
    assert phase.sides == ("0.4999999999999 + 1e-15*time()", "-0.5")
    assert _clock_scan._a_line_in_time(phase)
    for lo in (99.0, 90.0, 0.0):
        stops = _scan_value(tmp_path, call, lo, 110.0)
        assert stops == pytest.approx([99.89232], abs=1e-4)
        _each_is_a_change(tmp_path, call, stops)
    call = "rint(1e-13*time() + 0.49999999999)"
    stops = _scan_value(tmp_path, call, 99.99, 100.2)
    assert stops == pytest.approx([99.9991756], abs=1e-6)
    _each_is_a_change(tmp_path, call, stops)


# ── A run a few thousand doubles long ────────────────────────────────────────
# Every point of the grid over such a run is the double after the last, the
# whole grid is one neighbourhood, and the point beside its end was read off
# the end of the grid: an IndexError, which the run path turned into a
# warning and no stops, for the last leg of any run made in legs.
@pytest.mark.parametrize("start", [1.0, 120.0, 1e6, 1e11], ids=str)
@pytest.mark.parametrize("doubles", [2, 11, 64, 1000, 4097, 8193, 16000])
def test_a_run_a_few_thousand_doubles_long_is_scanned(start, doubles):
    end = start + doubles * float(np.spacing(start))
    for text, kind in (
        ("floor(time()^2/500)", "value"),
        ("abs(time() - 120) < 0.5", "comparison"),
        ("max(0, 1 - abs(time() - 120)/0.25)", "bend"),
    ):
        with contextlib.suppress(_clock_scan.OverBudget):  # a refusal by name is an answer
            _scan(text, start, end, kind=kind)


def test_nothing_turns_on_a_grid_of_adjacent_doubles():
    t = 1e11 + np.arange(6555) * float(np.spacing(1e11))
    for f in (lambda at: np.abs(at - t[3000]), lambda at: at - t[0], np.sin):
        assert _clock_scan._extrema(f, t, f(t)).size == 0


def test_the_last_leg_of_a_run_in_legs_is_scanned_like_any_other(tmp_path, caplog):
    model = _model(tmp_path, "if(abs(time() - 120) < 0.5, 1, 0) + floor(time()^2/500)")
    sim = bngsim.Simulator(model, method="ode")
    with caplog.at_level("WARNING", logger="bngsim"):
        sim.run(t_span=(9.99999999999998, 10.0), n_points=2)
    assert not [r for r in caplog.records if "scan of the time condition" in r.getMessage()]


def test_a_pulse_in_a_run_a_tenth_long_at_1e11_is_integrated(tmp_path, caplog):
    """6,554 doubles from one end of the run to the other."""
    model = _model(tmp_path, "if(abs(time() - 100000000000.05) < 0.02, 1, 0)")
    sim = bngsim.Simulator(model, method="ode")
    with caplog.at_level("WARNING", logger="bngsim"):
        result = sim.run(t_span=(1e11, 1e11 + 0.1), n_points=2)
    assert not [r for r in caplog.records if "scan of the time condition" in r.getMessage()]
    assert float(np.asarray(result.species)[-1, 0]) == pytest.approx(0.04, abs=3e-3)


# ── Two edges closer than the step resolver's arithmetic tells apart ─────────
def test_a_pulse_inside_the_step_resolvers_rounding_is_not_its_to_place(tmp_path):
    """`floor(mod(1e12 + time(), 0.3)/0.3 + 0.0005)` is one for 1.5e-4 before
    each wrap. The step resolver reads `(1e12 + time())/0.3` to 5e-4 of a
    whole number, took the middle of a pulse for past the wrap, and placed 158
    of the 1,600 changes in a run of 240: X was 0.0109 for 0.12, on main too,
    with no word. It declines now where the middle of a piece between two of
    its edges is within a few last places of a level, and the scan, which
    cannot read every double of that many pulses, says so."""
    with pytest.raises(bngsim.SimulationError, match="cannot be placed"):
        _x_end(tmp_path, "floor(mod(1e12 + time(), 0.3)/0.3 + 0.0005)", 241.3992)


@pytest.mark.parametrize("law", ["floor(time() - 1000000000.0)", "floor(time() - 300000000.0)"])
@pytest.mark.parametrize("doubles", [4, 8, 16])
def test_a_leg_that_begins_a_few_doubles_before_an_edge_has_the_edges_stop(tmp_path, law, doubles):
    """The piece from the run's start to the first edge is that short for no
    fault of the step resolver's arithmetic. Declined for it, the law went to
    the scan, which takes a change that close to the start for the start: no
    stop, and "CVODE made no progress" at t = 1e9."""
    edge = float(law.rsplit("- ", 1)[1].rstrip(")")) + 1.0
    start = edge - doubles * float(np.spacing(edge))
    model = _model(tmp_path, law)
    opts = _Options()
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(opts, start, edge + 0.5)
    assert opts.stops == [edge]
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(edge - 1.0, start), n_points=2)
    result = sim.run(t_span=(start, edge + 0.5), n_points=2)
    assert float(np.asarray(result.species)[-1, 0]) == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize(
    "dose, t_end, stops",
    [
        ("floor(mod(time(), 0.3)/0.3 + 0.25)", 30.0, 199),
        ("floor(mod(1e9 + time(), 0.3)/0.3 + 0.01)", 30.0, 200),
        ("floor(floor(time())/24)", 100.0, 4),
        ("floor((15 + 2e9 + time())/24)", 100.0, 5),
    ],
)
def test_the_step_resolver_still_places_edges_it_can_tell_apart(tmp_path, dose, t_end, stops):
    from bngsim import _switch_sensitivity

    model = _model(tmp_path, dose)
    scope = _switch_sensitivity.switch_condition_scope(model._core)
    times = _switch_sensitivity._step_edge_stop_times(dose, scope, 0.0, t_end)
    assert times is not None and len(times) == stops


# ── A sawtooth of a coarse line is left as main leaves it ────────────────────
# The scan finds every wrap of `mod(1.1e12 + time(), P)`. A run that restarts
# on each of them restarts where the rate is flat for a double of the
# argument, and where a period is a whole number of those doubles the rate is
# the same on every wrap: the integrator takes a period in one step and adds
# nothing for it (issue #998). Main's step resolver answers "no change" for
# these and the run, with no stop, has them right.
_COARSE = "100*mod(1100000000000.0 + time(), 0.0048828125)"  # 20 doubles of the sum


@pytest.mark.parametrize(
    "dose, t_end, exact",
    [
        (_COARSE, 20.2, 4.68507),  # 4,137 wraps: nearly 4,096 to a cell of the check
        ("100*mod(1100000000000.0 + time(), 0.004150390625)", 17.1, 3.33978),
    ],
)
def test_a_run_on_a_sawtooth_of_a_coarse_line_adds_every_period(tmp_path, dose, t_end, exact):
    """With a stop on each wrap, X was 0.0826 and 0.0592: about 72 periods of
    4,100. A resolver's answer for a sawtooth is not held against how few
    cells of a grid its value differs across."""
    model = _model(tmp_path, dose)
    opts = _Options()
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(opts, 0.0, t_end)
    assert opts.stops == []
    assert _x_end(tmp_path, dose, t_end=t_end) == pytest.approx(exact, rel=2e-3)


@pytest.mark.parametrize("x0", [0.0, 1e4])
def test_a_run_in_legs_on_a_sawtooth_of_a_coarse_line_adds_every_period(tmp_path, x0):
    """Ten legs of 0.2 fell to 0.22 of the integral; with a second stop to
    each wrap they held from X(0) = 0, and from X(0) = 1e4 one run did not
    (0.014 of it). (As main has them, which is with three outputs to the
    leg: with two, main is 20% over by the sixth, and so is this.)"""
    path = tmp_path / "coarse.net"
    path.write_text(_NET.format(dose=_COARSE).replace("X() 0", f"X() {x0!r}"), encoding="utf-8")
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    for leg in range(6):
        result = sim.run(t_span=(0.2 * leg, 0.2 * (leg + 1)), n_points=3)
    gain = float(np.asarray(result.species)[-1, 0]) - x0
    # (Main's own figure from X(0) = 1e4 is 4% over: the gain is 3e-5 of X.)
    assert gain == pytest.approx(0.2782, rel=6e-2)


def test_the_scan_is_not_asked_to_place_a_sawtooth_of_a_coarse_line(tmp_path):
    """`frac`, which no resolver reads: the scan has every wrap of it (below),
    and the run takes none for a stop while the line holds each of its
    doubles for longer than an instant."""
    coarse = "frac((1100000000000.0 + time())/0.0048828125)"
    assert _scan_value(tmp_path, coarse, 0.0, 0.1).size == 20
    opts = _Options()
    bngsim.Simulator(_model(tmp_path, f"100*{coarse}"), method="ode")._apply_crossing_stops(
        opts, 0.0, 0.1
    )
    assert opts.stops == []
    # A fine one has its stops, as before.
    opts = _Options()
    bngsim.Simulator(_model(tmp_path, "100*frac(time()/7)"), method="ode")._apply_crossing_stops(
        opts, 0.0, 240.0
    )
    assert len(opts.stops) == 34


# ── ...and what the rounds between took with it ──────────────────────────────
@pytest.mark.parametrize(
    "dose, run, exact",
    [
        # a run that begins on a wrap of a sawtooth of a line with an offset
        ("100*frac(time() + 4096)", (0.0, 24.0), 1200.0),
        ("100*frac(1e6 + time())", (3.0, 10.0), 350.0),
        ("100*frac((time() + 240000)/24)", (0.0, 240.0), 12000.0),
        # a sawtooth under a comparison, each change of which has its stop
        ("if(mod(1e15 + time(), 2) < 1, 1, 0)", (0.0, 240.0), 120.0),
        ("if(frac(1e15 + time()) < 0.5, 1, 0)", (0.0, 24.0), 12.0),
        ("if(mod(1e12 + time(), 0.00146484375) < 0.0007, 1, 0)", (0.0, 0.146484375), 0.0732421875),
        ("if(mod(1e9 + time(), 1e-6) + time() > 100, 1, 0)", (0.0, 240.0), 140.0),
        # a sawtooth guarded against the one point its argument is not a number at
        ("100*if(time() > 0.5, frac(ln(time())), 0)", (0.0, 10.0), 397.97616),
        ("100*if(time() > 0.5, mod(ln(time()), 5), 0)", (0.0, 10.0), 1387.2425),
    ],
)
def test_a_sawtooth_the_refusals_are_not_for_is_integrated(tmp_path, dose, run, exact):
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    result = sim.run(t_span=run, n_points=2)
    assert float(np.asarray(result.species)[-1, 0]) == pytest.approx(exact, rel=2e-6)


def test_a_sawtooth_of_a_line_coarser_than_the_clock_is_left_without_the_scans_stops(tmp_path):
    """At a clock of 1e6, `3e9 + time()` holds a double for 4,096 of the
    clock's: one instant of a run of 1, which was under the line. With a stop
    on each of its 1,024 wraps X was 0.0976 for 49.98 (#998); main, with no
    stop, has it right."""
    dose = "100*frac((3000000000.0 + time())/0.0009765625)"
    model = _model(tmp_path, dose)
    opts = _Options()
    bngsim.Simulator(model, method="ode")._apply_crossing_stops(opts, 1e6, 1e6 + 1.0)
    assert opts.stops == []
    sim = bngsim.Simulator(_model(tmp_path, dose), method="ode")
    result = sim.run(t_span=(1e6, 1e6 + 1.0), n_points=3)
    assert float(np.asarray(result.species)[-1, 0]) == pytest.approx(49.976, rel=1e-3)


# ── `round` is C's: a half away from zero, with no sum that can itself round ─
def test_the_phase_of_round_is_the_size_of_its_argument_less_a_half(tmp_path):
    """Since #684 the evaluator's `round` is `std::round`. Read as
    `abs(x) + 0.5`, which is what it was while `round` was `floor(x + 0.5)`,
    the phase is 1 on the double under a half, where `round` is still 0."""
    call = "round(0.4999999999999 + 1e-15*time())"
    (phase,) = _clock_scan.phases(call)
    assert phase.sides == ("abs(0.4999999999999 + 1e-15*time())", "0.5")
    assert _clock_scan._a_line_in_time(phase)
    core = _model(tmp_path, call)._core
    dense = np.linspace(99.0, 110.0, 2_000_001)
    values = core.evaluate_over_times(call, dense)
    size = core.evaluate_over_times("(abs(0.4999999999999 + 1e-15*time()))", dense)
    assert np.array_equal(
        np.flatnonzero(np.diff(values)), np.flatnonzero(np.diff(np.floor(size - 0.5)))
    )
    stops = _scan_value(tmp_path, call, 99.0, 110.0)
    assert stops.size == 1
    _each_is_a_change(tmp_path, call, stops)


@pytest.mark.parametrize(
    "call, run, changes",
    [
        ("round(4503599627370495.5 + time())", (0.0, 4.0), 4),  # across 2^52
        ("round(9007199254740990.0 + time())", (0.0, 8.0), 5),  # across 2^53
        ("round(-4503599627370495.5 - time())", (0.0, 4.0), 4),
    ],
)
def test_round_of_a_line_past_where_its_doubles_are_whole_numbers(tmp_path, call, run, changes):
    stops = _scan_value(tmp_path, call, *run)
    assert stops.size == changes == _dense_jumps(tmp_path, call, *run, 2_000_001)
    _each_is_a_change(tmp_path, call, stops)


@pytest.mark.parametrize("start", [1000000.023, 1000000.148, 1000000.0000002375])
@pytest.mark.parametrize("length", [1.0, 0.5, 2.0])
def test_how_coarse_a_sawtooths_line_is_does_not_go_by_where_the_run_begins(
    tmp_path, start, length
):
    """Read at 65 evenly spaced points, which all fall on one place in the
    argument's tread where the run is a whole number of its doubles long, the
    line of the test above was taken for a fine one in one window of 128: its
    1,024 stops, and 0.001 of its X."""
    dose = "100*frac((3000000000.0 + time())/0.0009765625)"
    opts = _Options()
    bngsim.Simulator(_model(tmp_path, dose), method="ode")._apply_crossing_stops(
        opts, start, start + length
    )
    assert opts.stops == []


def test_a_run_on_that_sawtooth_from_an_unlucky_start_adds_every_period(tmp_path):
    sim = bngsim.Simulator(
        _model(tmp_path, "100*frac((3000000000.0 + time())/0.0009765625)"), method="ode"
    )
    result = sim.run(t_span=(1000000.023, 1000001.023), n_points=3)
    assert float(np.asarray(result.species)[-1, 0]) == pytest.approx(49.976, rel=1e-3)


# ── `round` past 2^52, where every double is a whole number ──────────────────
@pytest.mark.parametrize(
    "call, run, changes",
    [
        ("round(4503599627370497 + time()*time()/16)", (1000000.0, 1000000.0001), 12),
        ("round(4503599627370496 + (time() - 1e6)*60 + (time() - 1e6)*60)", (1e6, 1e6 + 0.1), 6),
    ],
)
def test_round_of_an_argument_that_is_no_line_past_2_to_the_52_has_every_change(
    tmp_path, call, run, changes
):
    """`abs(x) - 0.5` is a tie on every double from 2^52, and `floor` of it
    changed on every other one: a change of `round` that was no crossing of
    its phase had no stop unless it lay between two that were."""
    stops = _scan_value(tmp_path, call, *run)
    assert stops.size == changes
    _each_is_a_change(tmp_path, call, stops)


@pytest.mark.parametrize("base", [4503599627370498, 4503599627370497, 6000000000000000])
def test_a_pulse_of_round_past_2_to_the_52_is_placed_or_refused(tmp_path, base):
    """One more than its base for 0.77 of time. It had no crossing, no stop and
    no word, and X was 0 for 0.767."""
    dose = f"round({base} + 0.9*exp(-((time() - 50)/0.5)^2)) - {base}"
    try:
        x = _x_end(tmp_path, dose, t_end=100.0)
    except bngsim.SimulationError as refused:
        assert "cannot be placed" in str(refused)
    else:
        assert x == pytest.approx(0.766672, rel=1e-4)
