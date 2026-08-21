"""A BNGL rate law that switches on a clock must not be integrated over.

Issues #440 and #443 — one defect in two spellings, and the second is the one
BNGL models actually write.

Inside each branch of ``if(time() >= 100, k, 0)`` the right-hand side is a
constant, so CVODE's local error estimate over a step that spans the whole
branch is near zero and nothing stops the step from growing until it swallows
the window. The trajectory that comes back is then the one where the branch
never turned on, and no warning is emitted. Tightening ``rtol`` does not help,
because there is no error to see.

An SBML model does not have this problem: its loader walks the document at load
time and registers every ``time`` inequality as a CVODE root, and issue #305
resolves each of those to a crossing time and stops the step on it. A ``.net``
model is built entirely in C++, so its loader has no build-time seam to register
a root at, and until this issue it registered nothing at all.

The fix recovers the conditions from the built model's own function bodies and
hands their crossing times to the same ``CVodeSetStopTime`` machinery issue #305
already had. Every model here is a pure accumulator — ``0 -> A`` at a rate that
is ``k`` while the condition holds and zero otherwise, with nothing else in the
model — so the exact answer is ``k`` times the width of the on-window and the
oracle is arithmetic rather than another solver.

What this locks:

  1. a single window is not stepped over, and neither is a repeating schedule,
     which is worse because there are many windows to miss;
  2. the answer agrees with a fine ``max_step`` run of the same model (the only
     handle a user had before) and with the same model written in SBML, whose
     registered root does not by itself cover a repeating schedule;
  3. the warm CVODE fast path cannot swallow a stop — it has no stop-time
     handling of its own, and a model that carries stops has to leave it;
  4. a condition over model state resolves to nothing (its crossing moves with
     the trajectory, which is issue #150's business, not a fixed stop);
  5. a threshold written behind a derived parameter or a function call is still
     found, and one that reads live state is not;
  6. a model with no time condition gets no stops at all, so its stepping is
     untouched;
  7. a batch row stops where its own parameter point puts the crossing, and a
     crossing a fitted parameter moves keeps the sensitivity jump it already
     had;
  8. a schedule asking for more stops than the budget allows places none and
     says so, rather than stopping at a prefix of them in silence;
  9. the same window written against a *counter species* rather than against
     ``time()`` reaches the same answer (issue #443). That is the BNGL idiom: a
     species fed by a zeroth-order reaction at rate 1, read back through a
     group, conventionally called ``t``. Of the 585 ``.net`` models in this
     repository's corpus, 37 threshold such a counter and none thresholds
     ``time()``. A counter is *integrated*, so it needs two things a literal
     clock does not: its offset from time, and being put exactly on its
     threshold at the stop, without which the run restarts on the branch that
     just ended. Landing it is a repair for a stop that stands IN PLACE OF a
     root, so where a root is registered it stands down rather than stepping
     over it, on the forward-sensitivity path as well as this one.
"""

import logging
import os
import subprocess
import sys
import textwrap

import bngsim
import pytest
from bngsim._switch_sensitivity import fixed_time_crossings

# ── Model ───────────────────────────────────────────────────────────────────
# A pure accumulator: nothing consumes A, and the only reaction is a zeroth
# order synthesis whose rate is the function under test. So A(T) is exactly the
# integral of that function, and for a rate that is `k` on a window and 0 off it
# the answer is k times the total on-time.
_NET = """\
begin parameters
{params}
end parameters
begin functions
{functions}
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 dose #_R1
end reactions
begin groups
    1 A                    1
end groups
"""

_T_END = 240.0


def _net(tmp_path, body, *, params=("1 k       0.1  # Constant",), extra_functions=()):
    """Write a one-reaction accumulator whose rate law is *body*."""
    functions = [*extra_functions, f"    {len(extra_functions) + 1} dose() {body}"]
    path = tmp_path / "switch.net"
    path.write_text(
        _NET.format(
            params="\n".join("    " + p for p in params),
            functions="\n".join(functions),
        )
    )
    return path


def _final_A(path, **kw):
    """A(240) from a fresh model — fresh because a Model carries its live state
    across runs, so reusing one silently starts the next run where the last
    ended."""
    model = bngsim.Model.from_net(str(path))
    kw.setdefault("t_span", (0.0, _T_END))
    kw.setdefault("n_points", 3)
    return float(bngsim.Simulator(model).run(**kw).species[-1][0])


def _conditions_and_stops(path):
    model = bngsim.Model.from_net(str(path))
    conds = model.time_discontinuity_conditions()
    return conds, fixed_time_crossings(model._core, 0.0, _T_END, conds)


# ── 1. The windows the issue reports ────────────────────────────────────────
# Each case is (rate law, exact answer, expected number of stops). The exact
# answers are 0.1 x on-time: 40 units for the single window; 17 of every 24 over
# ten periods for the first schedule; 0.5 of every 3 over eighty for the second.
_WINDOWS = [
    ("if(time()>=100,if(time()<=140,k,0),0)", 4.0, 2),
    ("if(time()-24*floor(time()/24)>=7,k,0)", 17.0, 20),
    ("if(time()-3*floor(time()/3)>=2.5,k,0)", 4.0, 160),
]


@pytest.mark.parametrize(("body", "exact", "n_stops"), _WINDOWS)
def test_a_time_switched_rate_law_is_integrated_over_its_window(tmp_path, body, exact, n_stops):
    path = _net(tmp_path, body)
    conds, stops = _conditions_and_stops(path)
    assert conds, "the condition was not recovered from the function body"
    assert len(stops) == n_stops
    assert _final_A(path) == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize(("body", "exact", "_n"), _WINDOWS)
def test_the_answer_agrees_with_a_bounded_step_run(tmp_path, body, exact, _n):
    """The independent oracle, and the only handle a user had before this.

    A ``max_step`` small against the narrowest window forces the integrator to
    look inside every one of them, at the cost of many thousands of steps. It
    has to reach the same answer the stops do.
    """
    path = _net(tmp_path, body)
    bounded = _final_A(path, max_step=0.05, max_steps=1_000_000)
    assert bounded == pytest.approx(exact, rel=1e-6)
    assert _final_A(path) == pytest.approx(bounded, rel=1e-6)


# ── 2. The same model in the other format ───────────────────────────────────
# The issue's own framing: an SBML model with this rate law is fine, because its
# loader registers each condition as a root at load time and issue #305 stops the
# step on it. The BNGL model has to reach the same number.
_SBML_TWIN = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="window">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <piecewise>
              <piece>
                <ci> k </ci>
                <apply><and/>
                  <apply><geq/>{time}<cn> 100 </cn></apply>
                  <apply><leq/>{time}<cn> 140 </cn></apply>
                </apply>
              </piece>
              <otherwise><cn type="integer"> 0 </cn></otherwise>
            </piecewise>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
""".format(
    time=(
        '<csymbol encoding="text" '
        'definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>'
    )
)


_SBML_SCHEDULE = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="schedule">
    <listOfCompartments>
      <compartment id="C" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <piecewise>
              <piece>
                <ci> k </ci>
                <apply><geq/>
                  <apply><minus/>{time}
                    <apply><times/><cn> 24 </cn>
                      <apply><floor/>
                        <apply><divide/>{time}<cn> 24 </cn></apply>
                      </apply>
                    </apply>
                  </apply>
                  <cn> 7 </cn>
                </apply>
              </piece>
              <otherwise><cn type="integer"> 0 </cn></otherwise>
            </piecewise>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
""".format(
    time=(
        '<csymbol encoding="text" '
        'definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>'
    )
)


def test_a_registered_schedule_is_stopped_at_too(tmp_path):
    """A registered CVODE root does not cover a repeating schedule.

    The root is evaluated on the *boolean*, and the boolean of a schedule reads
    the same on both sides of a step that spans a whole period, so there is no
    sign change for the root finder to see. It is the same reason a BNGL model
    needs stops rather than roots, so the SBML side of the accumulator is wrong
    in the same way and by the same amount: it reported 10.6 where the answer is
    17, having registered its one root and stopped at none of the edges.
    """
    xml = tmp_path / "schedule.xml"
    xml.write_text(_SBML_SCHEDULE)
    model = bngsim.Model.from_sbml(str(xml))
    assert model._core.n_discontinuity_triggers == 1
    conds = model.time_discontinuity_conditions()
    assert len(fixed_time_crossings(model._core, 0.0, _T_END, conds)) == 20
    value = float(bngsim.Simulator(model).run(t_span=(0.0, _T_END), n_points=3).species[-1][0])
    assert value == pytest.approx(17.0, rel=1e-6)


def test_the_bngl_model_agrees_with_its_sbml_twin(tmp_path):
    """Two spellings of one model, which is the comparison that made the defect
    visible in the first place."""
    xml = tmp_path / "window.xml"
    xml.write_text(_SBML_TWIN)
    sbml_model = bngsim.Model.from_sbml(str(xml))
    assert sbml_model._core.n_discontinuity_triggers == 2
    sbml_A = float(
        bngsim.Simulator(sbml_model).run(t_span=(0.0, _T_END), n_points=3).species[-1][0]
    )
    assert sbml_A == pytest.approx(4.0, rel=1e-6)
    assert _final_A(_net(tmp_path, _WINDOWS[0][0])) == pytest.approx(sbml_A, rel=1e-6)


# ── 3. The warm fast path ───────────────────────────────────────────────────
def test_the_warm_fast_path_cannot_swallow_a_stop(tmp_path):
    """The warm CVODE path reuses persistent solver memory across calls and has
    no stop-time handling of its own, so a run carrying stops must not take it.

    Before this issue nothing could reach it with stops in hand: every model
    that had them had registered the roots that produced them, and a registered
    root already excludes the warm path. A ``.net`` model has the stops without
    the roots, so the exclusion has to be made on the stops themselves.

    ``BNGSIM_NO_WARM_CVODE`` forces the cold path. The two agree only if the
    warm path is genuinely not being taken here.
    """
    path = _net(tmp_path, _WINDOWS[0][0])
    script = textwrap.dedent(f"""
        import bngsim
        m = bngsim.Model.from_net({str(path)!r})
        print(repr(float(bngsim.Simulator(m).run(t_span=(0.0, 240.0), n_points=3).species[-1][0])))
    """)
    env = dict(os.environ, BNGSIM_NO_WARM_CVODE="1")
    cold = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
    )
    assert float(cold.stdout.strip().splitlines()[-1]) == pytest.approx(4.0, rel=1e-6)
    assert _final_A(path) == pytest.approx(4.0, rel=1e-6)


# ── 4. What is and is not a fixed time crossing ─────────────────────────────
def test_a_state_threshold_is_not_a_time_crossing(tmp_path):
    """``A >= 5`` crosses at a time nobody knows before the run, so there is no
    stop to place. That crossing is issue #150's business."""
    conds, stops = _conditions_and_stops(_net(tmp_path, "if(A>=5,k,0)"))
    assert conds == ()
    assert stops == []


def test_a_threshold_that_reads_state_is_declined(tmp_path):
    """Written the other way round it is the same crossing: the time is only
    known once the trajectory is."""
    conds, stops = _conditions_and_stops(_net(tmp_path, "if(time()>=A,k,0)"))
    assert conds == ()
    assert stops == []


def test_an_equality_over_time_is_not_stopped_at(tmp_path):
    """``time() == 100`` is true for one instant of measure zero, so its branch
    contributes nothing to the integral and there is no window to miss.

    Stopping there would be worse than leaving it alone: the step that restarts
    at the crossing reads the rate law where the equality holds and carries that
    value over a whole step, turning an instant into a pulse. The SBML scan
    registers only orderings for the same reason.
    """
    path = _net(tmp_path, "if(time()==100,k,0)")
    conds, stops = _conditions_and_stops(path)
    assert conds == ()
    assert stops == []
    assert _final_A(path) == pytest.approx(0.0, abs=1e-9)


def test_a_model_with_no_condition_gets_no_stops(tmp_path):
    """The common case, and the one whose stepping must be untouched."""
    conds, stops = _conditions_and_stops(_net(tmp_path, "k*2"))
    assert conds == ()
    assert stops == []


def test_a_schedule_with_too_many_edges_says_so(tmp_path, caplog):
    """A schedule can ask for an unbounded number of stops, since the count is
    the run window divided by the period rather than a property of the model.

    Past the budget bngsim places none of them and says so. Saying so is the
    point: this is the one case where it knows the schedule is there and cannot
    stop at it, and the alternative is a trajectory that is quietly wrong in
    exactly the way this issue is about.
    """
    # A hundredth of a time unit over 240 of them is 48,000 edges.
    path = _net(tmp_path, "if(time()-0.01*floor(time()/0.01)>=0.005,k,0)")
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        conds, stops = _conditions_and_stops(path)
    assert conds
    assert stops == []
    assert any("more than 8192 edges" in r.getMessage() for r in caplog.records)


def test_a_schedule_that_never_turns_over_places_no_stop(tmp_path):
    """``rem(time(), 3) >= 0`` holds at every instant of the run. It is
    recognized as a schedule and it is a condition over time alone, but its duty
    fills the whole period, so it has no edge and nothing to stop at."""
    conds, stops = _conditions_and_stops(_net(tmp_path, "if(time()-3*floor(time()/3)>=0,k,0)"))
    assert conds == ("time()-3*floor(time()/3)>=0",)
    assert stops == []


# ── 5. Where the threshold is written ───────────────────────────────────────
def test_a_derived_threshold_is_inlined(tmp_path):
    """``onset = t0 + delay`` is a parameter expression, so the condition names
    one symbol and the crossing is at the sum of two."""
    path = _net(
        tmp_path,
        "if(time()>=onset,k,0)",
        params=(
            "1 k       0.1  # Constant",
            "2 t0      60.0  # Constant",
            "3 delay   40.0  # Constant",
            "4 onset   t0+delay  # ConstantExpression",
        ),
    )
    conds, stops = _conditions_and_stops(path)
    assert conds == ("time()>=onset",)
    assert stops == [100.0]
    # On from t=100 to the end of the run: 140 units at 0.1.
    assert _final_A(path) == pytest.approx(14.0, rel=1e-6)


def test_a_threshold_behind_a_function_call_is_found(tmp_path):
    """The scan inlines function references before it splits the condition, so a
    threshold written as a call is found under its call site."""
    path = _net(
        tmp_path,
        "if(time()>=onset(),k,0)",
        params=("1 k       0.1  # Constant", "2 t0      60.0  # Constant"),
        extra_functions=("    1 onset() t0+40",),
    )
    conds, stops = _conditions_and_stops(path)
    assert stops == [100.0]
    assert _final_A(path) == pytest.approx(14.0, rel=1e-6)


# ── 6. The paths that run on a clone ────────────────────────────────────────
def test_every_batch_row_stops_at_its_own_crossing(tmp_path):
    """A batch row integrates a clone carrying that row's parameter point, so
    each row's crossing is at a different time and each has to be stopped at
    where it actually is."""
    path = _net(
        tmp_path,
        "if(time()>=onset,k,0)",
        params=("1 k       0.1  # Constant", "2 onset   100.0  # Constant"),
    )
    model = bngsim.Model.from_net(str(path))
    onsets = [40.0, 100.0, 200.0]
    results = bngsim.Simulator(model).run_batch(
        t_span=(0.0, _T_END), n_points=3, params=[{"onset": o} for o in onsets]
    )
    for onset, result in zip(onsets, results, strict=True):
        assert float(result.species[-1][0]) == pytest.approx(0.1 * (_T_END - onset), rel=1e-6)


def test_a_crossing_a_parameter_moves_keeps_its_sensitivity_jump(tmp_path):
    """A stop and an issue #48 switch time can land on the same instant.

    When the threshold is a requested sensitivity parameter, the switch-time
    machinery already stops there and carries a jump for it. The crossing stop
    must not pre-empt that stop, or the jump would be keyed on an instant the
    stop had already consumed. The core drops a crossing stop that a switch has
    claimed; this is the check that the gradient survives it.

    A(240) is ``k*(240 - onset)``, so ``dA/donset`` is exactly ``-k``.
    """
    path = _net(
        tmp_path,
        "if(time()>=onset,k,0)",
        params=("1 k       0.1  # Constant", "2 onset   100.0  # Constant"),
    )
    model = bngsim.Model.from_net(str(path))
    result = bngsim.Simulator(model, sensitivity_params=["onset"]).run(
        t_span=(0.0, _T_END), n_points=3
    )
    assert float(result.species[-1][0]) == pytest.approx(14.0, rel=1e-6)
    assert float(result.sensitivities[-1][0][0]) == pytest.approx(-0.1, rel=1e-6)


def test_a_clone_carries_the_answer_rather_than_rescanning(tmp_path):
    """The scan is the expensive half and the conditions are text, so they do
    not depend on any parameter value. A fan-out of clones must not repeat it."""
    path = _net(tmp_path, _WINDOWS[0][0])
    model = bngsim.Model.from_net(str(path))
    conds = model.time_discontinuity_conditions()
    assert conds
    clone = model.clone()
    assert clone._derived_time_disc_conditions is not None
    assert clone.time_discontinuity_conditions() == conds


def test_the_crossing_is_resolved_against_live_parameter_values(tmp_path):
    """Resolution happens per run, not at load: a fitted onset moves the stop.

    This is the same property issue #305 needs for an experimental-condition
    parameter that puts a crossing inside the window in one phase and outside
    it in another.
    """
    path = _net(
        tmp_path,
        "if(time()>=onset,k,0)",
        params=("1 k       0.1  # Constant", "2 onset   100.0  # Constant"),
    )
    model = bngsim.Model.from_net(str(path))
    conds = model.time_discontinuity_conditions()
    assert fixed_time_crossings(model._core, 0.0, _T_END, conds) == [100.0]
    model.set_param("onset", 200.0)
    assert fixed_time_crossings(model._core, 0.0, _T_END, conds) == [200.0]
    assert float(
        bngsim.Simulator(model).run(t_span=(0.0, _T_END), n_points=3).species[-1][0]
    ) == pytest.approx(4.0, rel=1e-6)


# ── 7. The counter-clock spelling (issue #443) ──────────────────────────────
# Everything above is written against literal simulation time, which is the
# spelling SBML uses. BNGL models mostly do not have it: they make time
# available to a rate law by feeding a species from a zeroth-order reaction at
# rate 1 and reading it back through a group, conventionally called `t`. Of the
# 585 .net models in this repository's corpus, 37 threshold such a counter and
# none thresholds `time()`, so this is the spelling the defect actually reaches.
#
# A counter obeys dc/dt = 1, so c(t) = t + (c(t_start) - t_start) for the whole
# run and a threshold on it is a threshold on time, placed at
# t_start + threshold - c(t_start). The models below are the same accumulator as
# above with the clock written that way, so the exact answers are unchanged.
_COUNTER_NET = """\
begin parameters
    1 rate    {rate}  # Constant
    2 k       0.1  # Constant
end parameters
begin functions
    1 dose() {body}
end functions
begin species
    1 counter() {c0}
    2 A() 0
end species
begin reactions
    1 0 1 rate #_R1
    2 0 2 dose #_R2
end reactions
begin groups
    1 t                    1
    2 A                    2
end groups
"""


def _counter_net(tmp_path, body, *, c0=0, rate=1, name="counter.net"):
    """An accumulator whose rate law *body* thresholds a counter species.

    ``c0`` seeds the counter, so the crossing moves without the window changing
    width. ``rate`` is what the counter fills at; anything but 1 is not a clock.
    """
    path = tmp_path / name
    path.write_text(_COUNTER_NET.format(body=body, c0=c0, rate=rate))
    return path


def _counter_final_A(path, **kw):
    """A(240) from a fresh model. Column 1 is the accumulator; column 0 is the
    clock."""
    model = bngsim.Model.from_net(str(path))
    kw.setdefault("t_span", (0.0, _T_END))
    kw.setdefault("n_points", 3)
    return float(bngsim.Simulator(model).run(**kw).species[-1][1])


# The same three windows as `_WINDOWS`, written on the counter instead of on
# `time()`. Same exact answers, same stop counts.
_COUNTER_WINDOWS = [
    ("if(t>=100,if(t<=140,k,0),0)", 4.0, 2),
    ("if(t-24*floor(t/24)>=7,k,0)", 17.0, 20),
    ("if(t-3*floor(t/3)>=2.5,k,0)", 4.0, 160),
]


@pytest.mark.parametrize(("body", "exact", "n_stops"), _COUNTER_WINDOWS)
def test_a_counter_switched_rate_law_is_integrated_over_its_window(tmp_path, body, exact, n_stops):
    """The issue's own report: this prints 0.0 where the answer is 4.0."""
    path = _counter_net(tmp_path, body)
    conds, stops = _conditions_and_stops(path)
    assert conds, "the condition was not recovered from the function body"
    assert len(stops) == n_stops
    assert _counter_final_A(path) == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize(("body", "exact", "_n"), _COUNTER_WINDOWS)
def test_the_counter_answer_agrees_with_a_bounded_step_run(tmp_path, body, exact, _n):
    """The independent oracle, as for the ``time()`` spelling above."""
    path = _counter_net(tmp_path, body)
    bounded = _counter_final_A(path, max_step=0.05, max_steps=1_000_000)
    assert bounded == pytest.approx(exact, rel=1e-6)
    assert _counter_final_A(path) == pytest.approx(bounded, rel=1e-6)


def test_the_two_spellings_of_one_model_agree(tmp_path):
    """A counter thresholded at 100 and ``time()`` thresholded at 100 are the
    same model, and the comparison is what makes the counter answer readable at
    all."""
    counter = _counter_final_A(_counter_net(tmp_path, _COUNTER_WINDOWS[0][0]))
    literal = _final_A(_net(tmp_path, _WINDOWS[0][0]))
    assert counter == pytest.approx(literal, rel=1e-6)


def test_the_stop_record_names_the_counter_and_what_it_reaches(tmp_path):
    """A counter stop carries more than a time.

    The core has to put the counter exactly on its threshold before restarting,
    so the record names which species to move and what to move it to. A crossing
    on literal simulation time needs neither and says so with a clock index of
    -1.
    """
    from bngsim._switch_sensitivity import fixed_crossing_stops

    model = bngsim.Model.from_net(str(_counter_net(tmp_path, _COUNTER_WINDOWS[0][0])))
    stops = fixed_crossing_stops(model._core, 0.0, _T_END, model.time_discontinuity_conditions())
    assert [(s.time, s.clock_species_idx, s.threshold) for s in stops] == [
        (100.0, 0, 100.0),
        (140.0, 0, 140.0),
    ]

    literal = bngsim.Model.from_net(str(_net(tmp_path, _WINDOWS[0][0])))
    stops = fixed_crossing_stops(
        literal._core, 0.0, _T_END, literal.time_discontinuity_conditions()
    )
    assert [(s.time, s.clock_species_idx) for s in stops] == [(100.0, -1), (140.0, -1)]


def test_a_counter_that_starts_ahead_crosses_earlier(tmp_path):
    """The conversion is ``t_start + threshold - c(t_start)``, not ``threshold``.

    A counter seeded at 30 reads 100 at t = 70, so the window opens 30 time units
    early and closes 30 early. Its width is unchanged, so the accumulated answer
    is the same 4.0 and only the stops move — which is what makes this a test of
    the offset rather than of the window.
    """
    from bngsim._switch_sensitivity import fixed_crossing_stops

    path = _counter_net(tmp_path, _COUNTER_WINDOWS[0][0], c0=30)
    model = bngsim.Model.from_net(str(path))
    stops = fixed_crossing_stops(model._core, 0.0, _T_END, model.time_discontinuity_conditions())
    assert [(s.time, s.threshold) for s in stops] == [(70.0, 100.0), (110.0, 140.0)]
    assert _counter_final_A(path) == pytest.approx(4.0, rel=1e-6)


def test_the_counter_is_landed_on_its_threshold_at_the_stop(tmp_path):
    """Issue #82, and the reason a counter stop is not a plain time.

    The stop puts t exactly on the crossing, but the condition is read off the
    counter, and the counter is integrated: it comes back a couple of parts in
    1e14 BELOW the threshold it is defined as reaching there. Left alone the run
    restarts on the branch that just ended and meets the jump inside the first
    step after a restart with no history, which is where issue #82's step-size
    collapse comes from. Sampling exactly at the crossing shows where the counter
    was put: without the repair this reads 99.99999999999993.
    """
    model = bngsim.Model.from_net(str(_counter_net(tmp_path, "if(t>=100,k,0)")))
    result = bngsim.Simulator(model).run(t_span=(0.0, 200.0), n_points=201)
    at_crossing = [
        row for t, row in zip(result.time, result.species, strict=False) if float(t) == 100.0
    ]
    assert len(at_crossing) == 1
    assert float(at_crossing[0][0]) >= 100.0
    assert float(result.species[-1][1]) == pytest.approx(10.0, rel=1e-6)


def test_a_species_that_is_not_a_unit_rate_clock_is_not_read_as_one(tmp_path):
    """A counter filling at rate 2 is not a clock, so nothing is placed.

    Its crossing time is knowable too, but the conversion this uses is written
    for dc/dt = 1 and reading any other slope off it would put the stop in the
    wrong place. Declining is the safe answer, and the stepping such a model
    gets is the stepping it had.
    """
    conds, stops = _conditions_and_stops(_counter_net(tmp_path, _COUNTER_WINDOWS[0][0], rate=2))
    assert conds == ()
    assert stops == []
    # And with the rate put back, the same model is admitted — so this is about
    # the slope rather than about anything else in the file.
    conds, stops = _conditions_and_stops(_counter_net(tmp_path, _COUNTER_WINDOWS[0][0], rate=1))
    assert len(stops) == 2


# A rate rule `dk1/dt = 1` makes k1 a counter in an SBML model too, and this one
# triggers an event on it. bngsim registers `k1 > 4.5` as a discontinuity
# condition, so the crossing stop lands there — and the event root has to still
# fire at it.
_SBML_COUNTER_EVENT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="counter_event">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0" constant="false"/>
      <parameter id="fired" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="k1">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci>k1</ci><cn> 4.5 </cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="fired">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>
"""


def test_a_registered_root_on_a_counter_is_not_stepped_over(tmp_path):
    """The stop must not pre-empt a root that is already there.

    CVODE finds a root by a sign change across a step it accepts. Landing the
    counter on its threshold moves the state during the restart instead, which
    presents no such step, and the root then never fires at all — an event lost
    outright, silently. So the repair above runs only on a run with no roots,
    which is the whole of what this issue is about: a ``.net`` model has stops
    precisely because it could register no root. The stop itself is still placed,
    and is still what makes the root reachable.
    """
    from bngsim._switch_sensitivity import fixed_crossing_stops

    xml = tmp_path / "counter_event.xml"
    xml.write_text(_SBML_COUNTER_EVENT)
    model = bngsim.Model.from_sbml(str(xml))
    conds = model.time_discontinuity_conditions()
    stops = fixed_crossing_stops(model._core, 0.0, 10.0, conds)
    assert [(s.time, s.clock_species_idx >= 0) for s in stops] == [(4.5, True)]

    result = bngsim.Simulator(model).run(t_span=(0.0, 10.0), n_points=11)
    assert float(result.observables["fired"][-1]) == pytest.approx(1.0)


def test_a_counter_reading_nan_places_no_stop(tmp_path):
    """A crossing computed from a nan clock is a nan, and a stop at nan is not
    a stop.

    Reachable because a nan concentration is a state bngsim runs rather than
    refuses: issue #353 substitutes a compartment size and warns instead of
    stopping the model. So the offset is checked rather than assumed finite.
    """
    from bngsim._switch_sensitivity import fixed_crossing_stops

    model = bngsim.Model.from_net(str(_counter_net(tmp_path, _COUNTER_WINDOWS[0][0])))
    conds = model.time_discontinuity_conditions()
    assert len(fixed_crossing_stops(model._core, 0.0, _T_END, conds)) == 2
    model.set_concentration("counter()", float("nan"))
    assert fixed_crossing_stops(model._core, 0.0, _T_END, conds) == []


# The same counter and event, plus a rate law thresholding the counter at a
# FITTED parameter. That is the one shape that reaches the issue #48 switch-time
# jump with a counter clock, and it is the shape whose events are most likely to
# trigger on the same value.
_SBML_COUNTER_FITTED_SWITCH = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="counter_fitted_switch">
    <listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0" constant="false"/>
      <parameter id="T" value="4.5" constant="true"/>
      <parameter id="kf" value="1.0" constant="true"/>
      <parameter id="fired" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="k1">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <piecewise>
              <piece><ci>kf</ci><apply><geq/><ci>k1</ci><ci>T</ci></apply></piece>
              <otherwise><cn type="integer">0</cn></otherwise>
            </piecewise>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
    <listOfEvents>
      <event id="E0" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><ci>k1</ci><ci>T</ci></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="fired">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer">1</cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>
"""


def test_the_sensitivity_jump_does_not_step_over_a_root_either(tmp_path):
    """The same rule on the forward-sensitivity path, which had the same hole.

    The issue #48 jump lands the clock on its threshold for its own reasons and
    has done since issue #82, before any of this. On a model with a registered
    root that lands the state past the root and the root never fires, exactly as
    a crossing stop would — and only on a sensitivity run, so the plain
    trajectory and the fitted one disagreed about whether the event happened at
    all. This is that model, and the answers are arithmetic: A accumulates ``kf``
    for the ``10 - T`` time units after the counter passes ``T``, so A(10) is 5.5
    and ``dA/dT`` is exactly ``-kf``.
    """
    xml = tmp_path / "fitted_switch.xml"
    xml.write_text(_SBML_COUNTER_FITTED_SWITCH)

    plain = bngsim.Simulator(bngsim.Model.from_sbml(str(xml))).run(t_span=(0.0, 10.0), n_points=11)
    assert float(plain.observables["fired"][-1]) == pytest.approx(1.0)
    assert float(plain.species[-1][0]) == pytest.approx(5.5, rel=1e-6)

    fitted = bngsim.Simulator(bngsim.Model.from_sbml(str(xml)), sensitivity_params=["T"]).run(
        t_span=(0.0, 10.0), n_points=11
    )
    assert float(fitted.observables["fired"][-1]) == pytest.approx(1.0)
    assert float(fitted.species[-1][0]) == pytest.approx(5.5, rel=1e-6)
    # (time, species, parameter). Landing the clock is what the gradient needed
    # in the first place, so this pins that standing it down here did not cost
    # it: the jump is still applied, only the ulp of state is not.
    assert float(fitted.sensitivities[-1][0][0]) == pytest.approx(-1.0, rel=1e-6)
