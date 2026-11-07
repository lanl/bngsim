"""A trigger whose residual starts exactly ON its threshold (issues #340, #934).

``at (B > 0)`` with ``B(t_start) = 0`` is false at the start. Whether it fires is
the plain SBML rule: an event fires when its trigger goes from false to true, so
it fires at the start if the trajectory leaves the threshold into the trigger's
true side, at whatever order it leaves, and never if it stays on the threshold
or leaves into the false side. ``B = t^2/2`` (second order) and ``C = t^3/6``
(third) are positive for every ``t > 0``: the event fires at once, as in
libRoadRunner.

The first doubles of a run cannot be read for it: where ``t_start`` is not 0 the
elapsed time there is no larger than its rounding, and a stop or a root a few
ulps on (another event due, a row) reads the state with that rounding in its
sign. Read there, a trigger that leaves into its false side fired, and one that
leaves into its true side at second order fell and fired again. So such a
trigger is held, unless the flow carries it into its true side at first order
(that one fires at its first root, inside CVODE's root tolerance of the start,
as on main): it reads false until the hold's end and is read as it is from
there, where a departure into the true side at a higher order fires. A write
that changes its residual inside the hold ends the hold for it. The hold is
1e4·eps·|t_start|, the scale of that rounding: 1.6e-12 from 0.7, and none from
0, so a run from 0 fires at the first root, as libRoadRunner does.

A departure without first-order motion has no derivative for its fire time
(moving the threshold, or the flow's first motion, moves the fire from the
start to later), so a forward-sensitivity run refuses it by name, as issue
#960's envelope asks, when it fires at the start (within CVODE's root tolerance
of the hold's end). Whether the residual moves at first order is read off a
probe along the flow at δ and δ/2 (ratio 2), δ from the species' own time
scales, with the rates taken at the probe's time.

Issue #340 had refused that first root wherever the flow did not carry the
residual off at first order, on BIOMD0000000285 (``PIdeath > 0``, with the
aggregation cascade feeding PIdeath at zero), and so never fired a departure at
second order or higher (#934). SUNDIALS' own rule for a root function that is
zero at ``t0``, which AMICI inherits by rooting on the residual, does the same.
The maintainer's decision on #934 is the SBML rule: BIOMD0000000285's death
events fire at the start and the cell dies, as libRoadRunner has it.

The true side is the trigger's own, however it is written: ``at (A < 10)`` from
``A(0) = 10`` with A decaying fires at once, as ``at (10 > A)`` does.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

# The committed copy (the same model as the rr_parity corpus's, which is not
# checked in), so the BIOMD0000000285 tests run wherever the suite does.
_MODEL = Path(__file__).resolve().parents[2] / "benchmarks" / "sbml_events" / "BIOMD0000000285.xml"


def _sbml(species: str, rules: str, trigger: str, parameters: str = "") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="on_threshold">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      {species}
    </listOfSpecies>
    <listOfParameters>
      <parameter id="fired" value="0" constant="false"/>
      {parameters}
    </listOfParameters>
    <listOfRules>
      {rules}
    </listOfRules>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            {trigger}
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="fired">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""


_S = (
    '<species id="S" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"'
    ' boundaryCondition="true" constant="false"/>'
)
_A = (
    '<species id="A" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"'
    ' boundaryCondition="true" constant="false"/>'
)
_RATE_S_IS_A = (
    '<rateRule variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<ci>A</ci></math></rateRule>"
)
_RATE_A_IS_ONE = (
    '<rateRule variable="A"><math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<cn>1</cn></math></rateRule>"
)
_RATE_A_IS_TWO = (
    '<rateRule variable="A"><math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<cn>2</cn></math></rateRule>"
)
_RATE_S_IS_ONE = (
    '<rateRule variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML">'
    "<cn>1</cn></math></rateRule>"
)
_TRIGGER_S_GT_0 = "<apply><gt/><ci>S</ci><cn>0</cn></apply>"


def _run(xml: str, t_end: float = 2.0, n_points: int = 21):
    model = bngsim.Model.from_sbml_string(xml)
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=n_points, rtol=1e-9, atol=1e-12
    )
    # An event-assigned parameter is promoted to a state, so it comes back as an
    # observable column rather than an expression.
    obs = list(res.observable_names)
    fired = np.asarray(res.observables)[:, obs.index("fired")]
    return np.asarray(res.time), fired, res


def test_zero_flow_on_the_threshold_fires_at_the_start():
    """The BIOMD0000000285 shape, minimal: ``S(0) = 0`` with ``dS/dt(0) = 0``
    (issue #934).

    ``S = t^2/2`` is positive for every ``t > 0``, so the trigger is true from
    the first instant on and the event fires at the start, as in libRoadRunner.
    Issue #340's rule never fired it.
    """
    t, fired, res = _run(_sbml(_S + _A, _RATE_S_IS_A + _RATE_A_IS_ONE, _TRIGGER_S_GT_0))

    assert fired[0] == 0.0, "the t_start row is recorded before the root fires"
    assert fired[1:].all(), "the event did not fire at the start"
    s = np.asarray(res.species)[:, list(res.species_names).index("S")]
    assert s[-1] == pytest.approx(t[-1] ** 2 / 2, rel=1e-6)


def _fire_time(text: str, t_end: float = 2.0) -> tuple[float, float]:
    """(fired, the time the event recorded) for an Antimony model whose event
    sets ``fired = 1, tf = time``."""
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode").run(
        sample_times=[0.0, t_end / 2, t_end], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = np.asarray(run.species)[-1]
    return float(x[names.index("fired")]), float(x[names.index("tf")])


_CASCADE = "species X0 = 1, X1 = 0, X2 = 0, X3 = 0, X4 = 0, fired = 0, tf = -1\n" + "".join(
    f"J{j}: -> X{j}; X{j - 1}\n" for j in range(1, 5)
)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("spelling", ["X{k} > 0", "0 < X{k}", "X{k} - 0 > 0"])
def test_a_departure_at_any_order_into_the_true_side_fires_at_the_start(order, spelling):
    """X_k = t^k/k! leaves 0 at order k. The trigger is true for every t > 0,
    so the event fires at the start whatever k is (issue #934). From t = 0
    there is no start hold, and it fires at its first root."""
    trigger = spelling.format(k=order)
    fired, tf = _fire_time(_CASCADE + f"E: at ({trigger}): fired = 1, tf = time\n")
    assert fired == 1.0
    assert 0.0 <= tf <= 5e-12


def _near_start(t0, order, slope, due, rows, rtol=1e-10, atol=1e-12):
    """(fires of E0, the last fire's time) for X_k = slope*(t - t0)^k/k! from
    t0, beside events due at `due` (absolute times) and rows at `rows`."""
    text = "species X0 = 1" + "".join(f", X{j} = 0" for j in range(1, order + 1)) + ";\n"
    text += "".join(
        f"X{j}' = {(repr(slope) + '*') if j == order else ''}X{j - 1};\n"
        for j in range(1, order + 1)
    )
    text += f"n0 = 0; tf = -1;\nE0: at (X{order} > 0), t0=false: n0 = n0 + 1, tf = time;\n"
    text += "".join(
        f"tU{e} = 1; U{e}: at (time >= tU{e}), t0=false: q{e} = 1; q{e} = 0;\n"
        for e in range(len(due))
    )
    model = bngsim.Model.from_antimony_string(text)
    for e, t_due in enumerate(due):
        model.set_param(f"tU{e}", t_due)
    res = bngsim.Simulator(model, method="ode").run(
        sample_times=sorted({t0, *rows, t0 + 1.0}), rtol=rtol, atol=atol
    )
    last = dict(zip(res.observable_names, np.asarray(res.observables)[-1], strict=True))
    return last["n0"], last["tf"]


def _ulps(x, n):
    import math

    for _ in range(n):
        x = math.nextafter(x, math.inf)
    return x


@pytest.mark.parametrize(
    "t0, order, due, rows, tol",
    [
        (0.3, 1, [25], [1], (1e-10, 1e-12)),
        (0.3, 2, [10, 50, 10, 2], [], (1e-10, 1e-12)),
        (0.3, 3, [10, 50, 50], [], (1e-6, 1e-9)),
        (5.0, 2, [3, 200], [8, 100], (1e-8, 1e-14)),
    ],
)
def test_a_departure_into_the_false_side_beside_reads_a_few_ulps_on_does_not_fire(
    t0, order, due, rows, tol
):
    """X_k = -2(t - t0)^k/k! is negative for every t > t0, so E0 never fires.
    Other events due (and rows) a few ulps after a t_start that is not 0 read
    the state where its rounding decides X's sign; without the start hold E0
    fired there (issue #934's first review)."""
    n0, _ = _near_start(
        t0, order, -2.0, [_ulps(t0, k) for k in due], [_ulps(t0, k) for k in rows], *tol
    )
    assert n0 == 0.0


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("t0, slope", [(0.7, 2.0), (0.3, 1e-3)])
def test_a_departure_into_the_true_side_beside_reads_a_few_ulps_on_fires_once(t0, slope, order):
    """The true-side twin: E0 fires once, at the start (at the hold's end).
    Read at the rounding a few ulps on, the trigger fell after its fire and
    rose again, and E0 fired twice. (A first-order departure is not held and
    fires at its first root as on main, where a small slope can still read
    that rounding; left alone.)"""
    n0, tf = _near_start(t0, order, slope, [_ulps(t0, 100), _ulps(t0, 3), _ulps(t0, 40)], [])
    assert n0 == 1.0
    assert tf - t0 == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_a_departure_at_any_order_into_the_false_side_does_not_fire(order):
    """X_k = -t^k/k!: the trigger is false for every t, and nothing fires."""
    text = _CASCADE.replace(
        f"J{order}: -> X{order}; X{order - 1}", f"J{order}: X{order} -> ; X{order - 1}"
    )
    fired, _ = _fire_time(text + f"E: at (X{order} > 0): fired = 1, tf = time\n")
    assert fired == 0.0


def test_a_trigger_that_stays_on_its_threshold_does_not_fire():
    """Nothing moves X1 off 0: the trigger is never true."""
    text = "species X1 = 0, fired = 0, tf = -1\nE: at (X1 > 0): fired = 1, tf = time\n"
    fired, _ = _fire_time(text)
    assert fired == 0.0


def test_positive_flow_on_the_threshold_still_fires():
    """Same trigger, same ``S(0) = 0``, but ``dS/dt(0) = 1``: the first-order
    departure, which fired under issue #340's rule too."""
    t, fired, _ = _run(_sbml(_S, _RATE_S_IS_ONE, _TRIGGER_S_GT_0))

    assert fired[0] == 0.0, "the t_start row is recorded before the root fires"
    assert fired[-1] == 1.0
    assert fired[1] == 1.0, f"expected a fire by t={t[1]}"


def test_time_greater_than_zero_still_fires():
    """``time > 0`` — nine corpus models spell "at the start" this way.

    Its residual is exactly zero at ``t_start`` too, and the clock carries it
    into the true side at once.
    """
    _, fired, _ = _run(
        _sbml(
            _S,
            _RATE_S_IS_ONE,
            '<apply><gt/><csymbol encoding="text" '
            'definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
            "<cn>0</cn></apply>",
        )
    )

    assert fired[0] == 0.0
    assert fired[-1] == 1.0


def test_a_later_genuine_crossing_still_fires():
    """Starting on the threshold does not disarm the trigger for the whole run.

    ``A(0) = -1``, ``dA/dt = 2``, ``dS/dt = A`` gives ``S = t^2 - t``: the
    residual starts exactly on zero, leaves into the false side, and crosses
    back up at ``t = 1``. Nothing fires at the start; the crossing at 1 fires.
    """
    t, fired, _ = _run(
        _sbml(
            _S,
            _RATE_S_IS_A + _RATE_A_IS_TWO,
            _TRIGGER_S_GT_0,
            parameters='<parameter id="A" value="-1" constant="false"/>',
        ),
        t_end=2.0,
        n_points=201,
    )

    assert not fired[t < 0.99].any(), "fired before the crossing at t=1"
    assert fired[t > 1.01].all(), "the genuine crossing at t=1 did not fire"


def _corpus_xml() -> Path | None:
    return _MODEL if _MODEL.exists() else None


def test_biomd285s_cell_dies_at_the_start_as_in_libroadrunner():
    """The reported model (issue #934's decision): ``p38death`` and ``PIdeath``
    leave 0 at second and third order, both death triggers are true from the
    first instant on, and ``kalive := 0`` stops every reaction there.

    The values are libRoadRunner 2.9.2's at the same horizon and tolerances.
    Issue #340 had kept the cell alive, with AMICI, which does not fire a
    trigger whose residual is zero at the start (MisP(100) = 38.9 there).
    """
    xml = _corpus_xml()
    if xml is None:
        pytest.skip(f"model not present: {_MODEL}")

    model = bngsim.Model.from_sbml(str(xml))
    res = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 100.0), n_points=11, rtol=1e-9, atol=1e-12
    )
    obs = list(res.observable_names)
    values = np.asarray(res.observables)

    kalive = values[:, obs.index("kalive")]
    assert kalive[0] == 1.0
    assert (kalive[1:] == 0.0).all(), "no death event fired at the start"

    # libRoadRunner at t=100, rtol=1e-9 atol=1e-12: the state the cell died in.
    assert values[-1, obs.index("NatP")] == pytest.approx(19500.0, rel=1e-12)
    assert abs(values[-1, obs.index("MisP")]) < 1e-20  # rr: 6.0e-27
    assert abs(values[-1, obs.index("PIdeath")]) < 1e-40  # rr: 1.2e-65


def test_biomd285_forward_sensitivity_refuses_the_deaths_by_name():
    """The path issue #340 was reported through, which issue #934's decision
    brings back: both death triggers leave 0 with no first-order motion, so
    the time they fire has no derivative, and the run is refused by name."""
    xml = _corpus_xml()
    if xml is None:
        pytest.skip(f"model not present: {_MODEL}")

    model = bngsim.Model.from_sbml(str(xml))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kPIdeath", "kp38death"])
    with pytest.raises(
        bngsim.SensitivityUnsupportedError, match=r"no first-order motion \(issue #934\)"
    ):
        sim.run(t_span=(0.0, 100.0), n_points=11, rtol=1e-9, atol=1e-12)


def _sens_and_fd(text, param, times):
    """(forward sensitivity, central difference) of the last row, by species."""
    res = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=[param]
    ).run(sample_times=times, rtol=1e-10, atol=1e-12)
    names = list(res.species_names)
    sens = dict(zip(names, np.asarray(res.sensitivities)[-1, :, 0], strict=True))
    h = 1e-6
    ends = []
    for value in (1 - h, 1 + h):
        model = bngsim.Model.from_antimony_string(
            text.replace(f"{param} = 1", f"{param} = {value!r}")
        )
        run = bngsim.Simulator(model, method="ode").run(sample_times=times, rtol=1e-12, atol=1e-14)
        ends.append(np.asarray(run.species)[-1])
    fd = dict(zip(names, (ends[1] - ends[0]) / (2 * h), strict=True))
    return sens, fd


_FLAT = "species B = 0, A = 0, F = 0, G = 0; ka = 1; c = 0\nJA: -> A; ka\nJB: -> B; A\n"


@pytest.mark.parametrize("param", ["ka", "c"])
def test_a_start_fire_without_first_order_motion_refuses_sensitivities(param):
    """B = ka*t^2/2 leaves the threshold c = 0 at second order. Moving c moves
    the fire from the start to later (no derivative: one side 0, the other
    unbounded), and the run returned 1e27 for dG/dc unrefused. ka does not move
    it, but nothing at the fire tells the two apart, so both are refused."""
    text = _FLAT + "JG: -> G; F\nE: at (B > c), t0=false: F = 1000\n"
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=[param])
    with pytest.raises(
        bngsim.SensitivityUnsupportedError, match=r"no first-order motion \(issue #934\)"
    ):
        sim.run(sample_times=[0.0, 1.0, 2.0], rtol=1e-10, atol=1e-12)


def test_a_first_order_start_fire_is_differentiated():
    """S = ka*t leaves 0 at first order: a transversal departure, which fires at
    the start for every ka > 0. G' = F*(1 + S) integrates 1 + S from 0."""
    text = (
        "species S = 0, F = 0, G = 0; ka = 1\n"
        "JS: -> S; ka\nJG: -> G; F*(1 + S)\nE: at (S > 0): F = 1\n"
    )
    sens, fd = _sens_and_fd(text, "ka", [0.0, 1.0, 2.0])
    for name in ("S", "G"):
        assert sens[name] == pytest.approx(fd[name], rel=1e-5), name
    assert sens["G"] == pytest.approx(2.0, rel=1e-6)  # d/dka of 2 + 2*ka


def test_a_flat_start_that_crosses_back_later_is_differentiated():
    """S = -t^2/2 + k*t^3/6 starts on its threshold with no first-order
    motion, leaves into the false side, and crosses back transversally at
    t = 3/k. The rows read it off the threshold first, so that crossing is a
    crossing like any other: dG/dk = d/dk (6 - 3/k) = 3."""
    text = (
        "species S = 0, A = 0, C = 0, F = 0, G = 0; k = 1\n"
        "JC: -> C; k\nJA: -> A; C - 1\nJS: -> S; A\nJG: -> G; F\nE: at (S > 0): F = 1\n"
    )
    sens, fd = _sens_and_fd(text, "k", list(np.linspace(0.0, 6.0, 13)))
    assert sens["G"] == pytest.approx(3.0, rel=1e-6)
    assert sens["S"] == pytest.approx(fd["S"], rel=1e-5)


# ─── the true side is the trigger's own (issue #910's review) ──────────────


def _fires(dynamics: str, trigger: str, init: str = "A = 10") -> tuple[float, float]:
    """(F at the end, B at the end): F is set by the event and B grows at F."""
    text = (
        f"species A, F, B; {init}; F = 0; B = 0\n"
        f"{dynamics}\nJB: -> B; F\nE: at ({trigger}): F = 1\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode").run(
        sample_times=[0.0, 1.0, 2.0], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = np.asarray(run.species)[-1]
    return float(x[names.index("F")]), float(x[names.index("B")])


@pytest.mark.parametrize(
    "trigger",
    ["A < 10", "10 > A", "2*A < 20", "A - 10 < 0", "0 > A - 10", "floor(A) < 10", "10 > floor(A)"],
)
def test_a_trigger_the_trajectory_leaves_into_fires_however_it_is_written(trigger):
    """A decays from 10, so each of these is false at the start and true from
    the first instant on: the event fires at once and B(2) = 2. The spellings
    with ``<`` never fired."""
    fired, b = _fires("J0: A -> ; 0.5*A", trigger)
    assert fired == 1.0 and b == pytest.approx(2.0, rel=1e-6)


@pytest.mark.parametrize("trigger", ["A <= 10", "A > 10", "10 < A", "A >= 10.5"])
def test_a_trigger_the_trajectory_does_not_leave_into_does_not_fire(trigger):
    """``A <= 10`` is true at the start, so there is no rising edge. The others
    are false at the start and A moves away from them."""
    fired, b = _fires("J0: A -> ; 0.5*A", trigger)
    assert fired == 0.0 and b == 0.0


@pytest.mark.parametrize("trigger", ["A < 10", "10 > A"])
def test_a_rate_gated_on_time_past_the_start_counts(trigger):
    """A' is +1 at t = 0 exactly and -1 for every t > 0, so A leaves 10
    downward. Read at t_start itself the flow points the other way."""
    fired, b = _fires("J0: -> A; piecewise(-1, time > 0, 1)", trigger)
    assert fired == 1.0 and b == pytest.approx(2.0, rel=1e-6)


@pytest.mark.parametrize("trigger", ["L/(Kd + L) > 0", "0 < L/(Kd + L)"])
def test_a_trigger_with_a_pole_near_its_start_fires_at_once(trigger):
    """L starts at 0 and is made at 1e-10, in a model whose scale is
    Kd = 1e-10, with a pole at L = -Kd. Under issue #340's rule a finite
    difference reached past the pole and the first spelling never fired. Both
    fire at once."""
    text = (
        "species L, F, B; L = 0; F = 0; B = 0; Kd = 1e-10\n"
        f"J0: -> L; 1e-10\nJB: -> B; F\nE: at ({trigger}): F = 1\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode").run(
        sample_times=[0.0, 1.0, 2.0], rtol=1e-10, atol=1e-16
    )
    names = list(run.species_names)
    x = np.asarray(run.species)[-1]
    assert x[names.index("F")] == 1.0 and x[names.index("B")] == pytest.approx(2.0, rel=1e-6)


@pytest.mark.parametrize("trigger", ["abs(A - 9.5) < 0.5", "0.5 > abs(A - 9.5)"])
def test_a_trigger_true_only_briefly_after_the_start_fires(trigger):
    """The trigger is true while A is between 9 and 10, which is the first 0.21
    time units of a run that lasts a million. Under issue #340's rule a probe
    sized by the run alone landed past that window, read the trigger false
    there, and the event never fired."""
    text = (
        "species A, F, B; A = 10; F = 0; B = 0\n"
        f"J0: A -> ; 0.5*A\nJB: -> B; F\nE: at ({trigger}): F = 1\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode").run(
        sample_times=[0.0, 1.0, 1.0e6], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    x = np.asarray(run.species)
    assert x[1, names.index("F")] == 1.0
    assert x[1, names.index("B")] == pytest.approx(1.0, rel=1e-6)


# ─── What the second review found ─────────────────────────────────────────


def _obs_last(text, t0, t_end, n_points=8, **params):
    model = bngsim.Model.from_antimony_string(text)
    for name, value in params.items():
        model.set_param(name, value)
    run = bngsim.Simulator(model, method="ode").run(
        t_span=(t0, t_end), n_points=n_points, rtol=1e-8, atol=1e-10
    )
    values = dict(zip(run.species_names, np.asarray(run.species)[-1], strict=True))
    values.update(zip(run.observable_names, np.asarray(run.observables)[-1], strict=True))
    return values


@pytest.mark.parametrize("t0", [0.0, 0.5, 300.0])
def test_a_trigger_that_resets_itself_fires_each_time_from_any_start(t0):
    """X' = 1 from X(t0) = 0 with `at (X > 0): X = -1`: fires at t0, t0+1,
    t0+2, t0+3. Pinned true through the start hold, the reset's fall was not
    read and nothing fired after the first (from any t0 but 0)."""
    text = "species X = 0; n = 0;\nX' = 1;\nE: at (X > 0), t0=false: X = -1, n = n + 1;\n"
    last = _obs_last(text, t0, t0 + 3.5)
    assert last["n"] == 4.0
    assert last["X"] == pytest.approx(-0.5, abs=1e-6)


@pytest.mark.parametrize("t0", [0.0, 1.0, 10.0])
def test_a_ratchet_from_its_threshold_counts_every_unit(t0):
    """`at (X > lvl): lvl = lvl + 1` from X = lvl = 0 with X' = 1: n = 4 over
    3.5, as in libRoadRunner."""
    text = (
        "species X = 0; lvl = 0; n = 0;\nX' = 1;\n"
        "E: at (X > lvl), t0=false: lvl = lvl + 1, n = n + 1;\n"
    )
    assert _obs_last(text, t0, t0 + 3.5)["n"] == 4.0


def test_start_fires_at_different_orders_keep_their_priorities_from_zero():
    """E1 leaves its threshold at second order, E2 at first; from t = 0 both
    fire at the start, one instant, E1 (priority 10) before E2: v = 2, as in
    libRoadRunner. (From a t_start that is not 0, E1 is held and fires a hold
    after E2; that ordering is left alone.)"""
    t0 = 0.0
    text = (
        "species A = 0, B = 0, Y = 0; v = 0;\nA' = 1; B' = A; Y' = 1;\n"
        "E1: at (B > 0), t0=false, priority=10: v = 1;\n"
        "E2: at (Y > 0), t0=false, priority=0: v = 2;\n"
    )
    assert _obs_last(text, t0, t0 + 1.0, n_points=3)["v"] == 2.0


@pytest.mark.parametrize("t0", [0.0, 2.0])
@pytest.mark.parametrize("grid", [[0.0, 3.0], [0.0, 1.0, 3.0]])
def test_a_flat_start_that_crosses_back_is_differentiated_on_any_grid(t0, grid):
    """X = 1 - s^2/2 + a*s^3 leaves `X > 1` at second order into the false side
    and crosses back transversally at 1/(2a): dG/da = 1000/(2a^2) = 4500 at
    a = 1/3. With no row before the crossing, a refusal that waited for a read
    of the residual off its threshold refused it."""
    text = (
        "species S = 0, X = 1, F = 0, G = 0; c = 1; a = 0.3333333333333333;\n"
        "S' = 1;\nX' = -S + 3*a*S^2;\nG' = F;\nE: at (X > c), t0=false: F = 1000;\n"
    )
    res = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["a"]
    ).run(sample_times=[t0 + x for x in grid], rtol=1e-10, atol=1e-12)
    g = list(res.species_names).index("G")
    assert np.asarray(res.sensitivities)[-1, g, 0] == pytest.approx(4500.0, rel=1e-6)


@pytest.mark.parametrize("ulps", [5, 1000])
def test_a_write_that_opens_a_held_trigger_is_a_cascade_fire(ulps):
    """X(5) = 0 with X' = -k is held at the start; U sets X = k a few ulps on.
    The write ends E's hold, so E rises at U's write like any trigger an
    assignment opens, and a sensitivity run refuses it (issue #144). Deferred
    to the hold's end it was differentiated as a crossing: dG/dk = -1000, for
    a central difference of 0."""
    import math

    text = (
        "species X = 0, F = 0, G = 0; k = 1; tU = 1;\nX' = -k;\nG' = F;\n"
        "U: at (time >= tU), t0=false: X = k;\nE: at (X > 0), t0=false: F = 1000;\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("tU", 5.0 + ulps * math.ulp(5.0))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    with pytest.raises(bngsim.SimulationError, match="another event's assignment"):
        sim.run(sample_times=[5.0, 6.0, 7.0], rtol=1e-10, atol=1e-12)
    model = bngsim.Model.from_antimony_string(text)
    model.set_param("tU", 5.0 + ulps * math.ulp(5.0))
    run = bngsim.Simulator(model, method="ode").run(
        sample_times=[5.0, 6.0, 7.0], rtol=1e-10, atol=1e-12
    )
    names = list(run.species_names)
    assert np.asarray(run.species)[-1, names.index("G")] == pytest.approx(2000.0, rel=1e-9)


@pytest.mark.parametrize("trigger", ["X^2 > c", "X*Y > c"])
def test_a_residual_that_leaves_at_second_order_through_the_state_refuses(trigger):
    """X = Y = t: `X^2 > c` and `X*Y > c` leave c = 0 at second order though
    the state moves at first order, so a first-order step along the flow moved
    the residual and the fire was differentiated: dG/dc = -7e28."""
    text = (
        "species X = 0, Y = 0, F = 0, G = 0; c = 0\nX' = 1;\nY' = 1;\nG' = F;\n"
        f"E: at ({trigger}), t0=false: F = 1000;\n"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["c"]
    )
    with pytest.raises(
        bngsim.SensitivityUnsupportedError, match=r"no first-order motion \(issue #934\)"
    ):
        sim.run(sample_times=[0.0, 1.0, 2.0], rtol=1e-10, atol=1e-12)


# ─── What the third review found ──────────────────────────────────────────


@pytest.mark.parametrize("t_end", [100.0, 1e6])
def test_a_dose_that_starts_a_flat_species_is_a_crossing_however_long_the_run(t_end):
    """B = 0 with B' = k*A - kd*B and A = 0 until D doses A = 10 at tD = 0.5:
    H, `at (B > 0)`, starts on its threshold with no first-order motion, and
    crosses transversally at tD. dG/dtD = -1 (G' = flag). The refusal window
    once grew with the run's end and refused it on a run to 1e6."""
    text = (
        "species A = 0, B = 0, G = 0; k = 3; kd = 0.1; tD = 0.5; flag = 0\n"
        "B' = k*A - kd*B;\nG' = flag;\n"
        "D: at (time > tD), t0=false: A = 10;\nH: at (B > 0), t0=false: flag = 1;\n"
    )
    res = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["tD"]
    ).run(sample_times=[0.0, t_end], rtol=1e-10, atol=1e-12)
    g = list(res.species_names).index("G")
    assert np.asarray(res.sensitivities)[-1, g, 0] == pytest.approx(-1.0, rel=1e-6)


@pytest.mark.parametrize("t0", [0.0, 5.0, 300.0])
def test_a_time_gated_second_order_start_refuses_from_any_start(t0):
    """X' = k*(time - t0) from X(t0) = 0 leaves `X > c` at second order. Its
    rates read at t_start's next double made it look first order (k*ulp(t0)),
    and dG/dc = -9e10 came back unrefused from any t0 but 0."""
    text = (
        f"species X = 0, Z = 0, G = 0; k = 1; c = 0; s = {t0!r}\n"
        "X' = k*(time - s);\nG' = Z;\nE: at (X > c), t0=false: Z = 1;\n"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["c"]
    )
    with pytest.raises(
        bngsim.SensitivityUnsupportedError, match=r"no first-order motion \(issue #934\)"
    ):
        sim.run(sample_times=[t0, t0 + 1.0, t0 + 2.0], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("t0", [1e6, 1e8])
def test_a_first_order_start_fire_is_not_held(t0):
    """X' = 1 from X(t0) = 0 with `at (X > 0): X = -1` fires at the start's
    first root, as on main. Held for the start's rounding it fired a hold
    late, and every later fire with it: X(end) off by 2.3e-6 at 1e6, a
    hundred times main's error."""
    text = "species X = 0; n = 0;\nX' = 1;\nE: at (X > 0), t0=false: X = -1, n = n + 1;\n"
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode").run(
        t_span=(t0, t0 + 3.5), n_points=8, rtol=1e-10, atol=1e-12
    )
    x_end = float(np.asarray(run.species)[-1, list(run.species_names).index("X")])
    assert x_end == pytest.approx(-0.5, abs=1e3 * np.spacing(t0))  # main: ~220 doubles
