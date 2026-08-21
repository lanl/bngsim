"""A BNGL rate law that switches on time must not be integrated over (issue #440).

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
     handle a user had before) and with the same model written in SBML;
  3. the warm CVODE fast path cannot swallow a stop — it has no stop-time
     handling of its own, and a model that carries stops has to leave it;
  4. a condition over model state resolves to nothing (its crossing moves with
     the trajectory, which is issue #150's business, not a fixed stop);
  5. a threshold written behind a derived parameter or a function call is still
     found, and one that reads live state is not;
  6. a model with no time condition gets no stops at all, so its stepping is
     untouched.
"""

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
