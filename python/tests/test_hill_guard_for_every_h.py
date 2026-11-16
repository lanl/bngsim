"""Issue #1636: a Hill exponent moved into (0, 1) after load is read as 0 at S = 0.

#1629 (issue #1497) wrote the `.net` loader's Hill function
``Vmax*S^(h-1)/(Kh^h + S^h)`` as ``if((S > 0) or (h <= 0) or (h >= 1), body, 0)``,
which the reaction multiplies by S, but only where h was in (0, 1) at load: a
root on ``S > 0`` costs every step it is watched, and 300 Hill reactions with
h >= 1 ran 10x slower with it. So a model whose h was moved into (0, 1) after it
was loaded evaluated ``inf*0`` at S = 0, and every run that reached S = 0 failed.

Every Hill law now takes the guard, and a plain run folds a condition its
parameter values decide (``(h <= 0) or (h >= 1)`` true at h = 2) to a constant
before choosing its roots, so a model at h >= 1 roots nothing for it.

The model is the issue's: S fed at kin = 1 from 0 and drained by Hill(1, 2, h).
BNG2.pl 2.9.3 with run_network 3.0 gives S(5) = 2.796028127 at h = 0.5 and
2.92167346 at h = 2 (the issue's table); a stiff solve of the same ODE agrees.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil

import bngsim
import numpy as np
import pytest

# The loader says it rewrote a legacy Hill token, on every load here.
pytestmark = pytest.mark.filterwarnings("ignore:Legacy/deprecated BioNetGen .net rate law token")

S_AT_HALF, S_AT_TWO = 2.796028127, 2.92167346
TOL = 2e-8  # the reference's own nine digits, and rtol 1e-10 here

NET = """begin parameters
 1 Vmax 1
 2 Kh 2
 3 h {h}
 4 kin 1
end parameters
begin species
 1 S() 0
 2 P() 0
end species
begin reactions
 1 1 2 Hill Vmax Kh h
 2 0 1 kin
end reactions
begin groups
 1 S_obs 1
 2 P_obs 2
end groups
"""
RUN = {"t_span": (0.0, 5.0), "n_points": 2, "rtol": 1e-10, "atol": 1e-12}


def _model(tmp_path, h=2.0):
    path = tmp_path / "hill.net"
    path.write_text(NET.format(h=h), encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _s_end(result) -> float:
    return float(np.asarray(result.species)[-1, list(result.species_names).index("S()")])


def test_the_function_is_guarded_whatever_h_is_at_load(tmp_path):
    for h in (2.0, 1.0, 0.5, 0.0, -1.0):
        (body,) = _model(tmp_path, h)._core.function_expressions
        assert body.startswith("if((") and "or (h <= 0) or (h >= 1)" in body, (h, body)


def test_h_set_into_the_open_interval_after_load(tmp_path):
    # On main: CV_FIRST_RHSFUNC_ERR, `inf*0` at S = 0.
    model = _model(tmp_path, 2.0)
    model.set_param("h", 0.5)
    assert abs(_s_end(bngsim.Simulator(model, "ode").run(**RUN)) - S_AT_HALF) < TOL


def test_a_scan_and_a_batch_over_h(tmp_path):
    sim = bngsim.Simulator(_model(tmp_path, 2.0), "ode")
    scanned = [_s_end(r) for r in sim.parameter_scan("h", [0.5, 2.0], **RUN)]
    rows = [_s_end(r) for r in sim.run_batch(params=[{"h": 0.5}, {"h": 2.0}], **RUN)]
    for got in (scanned, rows):
        assert abs(got[0] - S_AT_HALF) < TOL and abs(got[1] - S_AT_TWO) < TOL, got


def test_a_run_at_h_2_then_one_at_one_half_on_the_same_simulator(tmp_path):
    # The roots kept for h = 2 must not be reused at h = 0.5.
    model = _model(tmp_path, 2.0)
    sim = bngsim.Simulator(model, "ode")
    assert abs(_s_end(sim.run(**RUN)) - S_AT_TWO) < TOL
    model.reset()
    model.set_param("h", 0.5)
    assert abs(_s_end(sim.run(**RUN)) - S_AT_HALF) < TOL
    model.reset()
    model.set_param("h", 2.0)
    assert abs(_s_end(sim.run(**RUN)) - S_AT_TWO) < TOL


def test_loaded_at_one_half_it_runs_as_before(tmp_path):
    assert abs(_s_end(bngsim.Simulator(_model(tmp_path, 0.5), "ode").run(**RUN)) - S_AT_HALF) < TOL


def _roots_handed_to_the_run(model, sim=None) -> list[str]:
    seen: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    log = logging.getLogger("bngsim")
    old = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        # At h <= 0 the law itself is infinite at S = 0 (run_network does not
        # read it as 0 there either), so the run from S = 0 fails at its first
        # step; the roots are handed to it before that.
        with contextlib.suppress(bngsim.SimulationError):
            (sim or bngsim.Simulator(model, "ode")).run(**RUN)
    finally:
        log.removeHandler(handler)
        log.setLevel(old)
    return [m for m in seen if m.startswith("State-switch roots for a plain run")]


@pytest.mark.parametrize(("h", "rooted"), [(2.0, False), (1.0, False), (-1.0, False), (0.5, True)])
def test_the_guard_is_rooted_only_where_h_is_in_the_open_interval(tmp_path, h, rooted):
    # The route, not only the answer: at h >= 1 or h <= 0 the guard is true
    # whatever S is, and the run is handed no root for it.
    assert bool(_roots_handed_to_the_run(_model(tmp_path, h))) is rooted


def test_the_root_comes_back_when_h_moves_into_the_interval(tmp_path):
    # The roots kept for one set of decisions are not those of another: the
    # same Simulator at h = 2, then 0.5, then 2.
    model = _model(tmp_path, 2.0)
    sim = bngsim.Simulator(model, "ode")
    assert not _roots_handed_to_the_run(model, sim)
    model.reset()
    model.set_param("h", 0.5)
    assert _roots_handed_to_the_run(model, sim)
    model.reset()
    model.set_param("h", 2.0)
    assert not _roots_handed_to_the_run(model, sim)


def test_a_sensitivity_run_at_h_2_gives_the_column(tmp_path):
    # dS/dVmax at t = 5 at h = 2, against a difference.
    model = _model(tmp_path, 2.0)
    sens = bngsim.Simulator(model, "ode", sensitivity_params=["Vmax"]).run(**RUN)
    column = float(np.asarray(sens.sensitivities)[-1, 0, 0])
    step = 1e-6
    up, down = _model(tmp_path, 2.0), _model(tmp_path, 2.0)
    up.set_param("Vmax", 1 + step)
    down.set_param("Vmax", 1 - step)
    fd = (
        _s_end(bngsim.Simulator(up, "ode").run(**RUN))
        - _s_end(bngsim.Simulator(down, "ode").run(**RUN))
    ) / (2 * step)
    assert abs(column - fd) < 1e-6 * max(1.0, abs(fd))


# ── The other readers of the roots ──────────────────────────────────────────
# Each read the roots of the laws as written, where a plain run read them at
# its parameter values: with the guard on every Hill law, a model at h >= 1
# was refused, slowed or bounded differently by each, where it had not been.

# S fed at kin and drained by Hill(1, 2, h) into P, which decays, so the model
# has a steady state: S* = 2*(kin/(1 - kin))^(1/h), P* = kin.
STEADY = """begin parameters
 1 Vmax 1
 2 Kh 2
 3 h {h}
 4 kin 0.4
 5 kdeg 1
 6 period 1
 7 width 0.1
end parameters
begin species
 1 S() 0
 2 P() 0
 3 X() 0
end species
{functions}begin reactions
 1 1 2 Hill Vmax Kh h
 2 0 1 kin
 3 2 0 kdeg
{dosed}end reactions
"""
SCHEDULE = (
    "begin functions\n"
    " 1 dose() if(time() - period*floor(time()/period) < width, 1, 0)\n"
    "end functions\n"
)


def _steady(tmp_path, h, schedule=False):
    path = tmp_path / "steady.net"
    text = STEADY.format(
        h=h,
        functions=SCHEDULE if schedule else "",
        dosed=" 4 0 3 dose\n" if schedule else "",
    )
    path.write_text(text, encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _s_star(h, kin=0.4):
    return 2.0 * (kin / (1.0 - kin)) ** (1.0 / h)


@pytest.fixture
def jumped(monkeypatch):
    """The conditions each sensitivity run is handed to root and jump."""
    from bngsim import _switch_sensitivity as sw

    handed: list[list[str]] = []
    found = sw.state_switch_conditions

    def spy(core, ctx=None):
        conditions = found(core, ctx)
        handed.append(list(conditions))
        return conditions

    monkeypatch.setattr(sw, "state_switch_conditions", spy)
    return handed


@pytest.mark.parametrize(("h", "rooted"), [(2.0, False), (1.0, False), (0.5, True)])
def test_a_sensitivity_run_roots_the_guard_only_inside_the_interval(tmp_path, jumped, h, rooted):
    # 300 Hill reactions at h = 2 with a column ran 162 ms with the guard's
    # roots, where main's bare body ran 12.
    model = _steady(tmp_path, h)
    model.set_param("kin", 0.4)
    # Away from S = 0, where d/dVmax of the body is not finite at h = 1.
    model.set_concentration("S()", 1.0)
    bngsim.Simulator(model, "ode", sensitivity_params=["Vmax"]).run(**RUN)
    assert [bool(conditions) for conditions in jumped] == [rooted]


def test_each_row_of_a_sensitivity_batch_is_read_at_its_own_h(tmp_path, jumped):
    model = _steady(tmp_path, 2.0)
    model.set_concentration("S()", 1.0)
    model.save_concentrations()
    sim = bngsim.Simulator(model, "ode", sensitivity_params=["Vmax"])
    rows = sim.run_batch(params=[{"h": 2.0}, {"h": 0.5}], **RUN)
    # Rows may run on worker threads, in either order.
    assert sorted(bool(conditions) for conditions in jumped) == [False, True]
    for h, row in zip((2.0, 0.5), rows, strict=True):
        alone = _steady(tmp_path, h)
        alone.set_concentration("S()", 1.0)
        single = bngsim.Simulator(alone, "ode", sensitivity_params=["Vmax"]).run(**RUN)
        assert np.allclose(
            np.asarray(row.sensitivities)[-1], np.asarray(single.sensitivities)[-1], rtol=1e-6
        )


@pytest.fixture
def marched(monkeypatch):
    """The roots each steady-state march is handed."""
    import bngsim._bngsim_core as core

    handed: list[list[str]] = []
    solve = core.find_steady_state

    def spy(model_core, opts):
        handed.append(list(opts.state_switch_conditions))
        return solve(model_core, opts)

    monkeypatch.setattr(core, "find_steady_state", spy)
    return handed


@pytest.mark.parametrize(("h", "rooted"), [(2.0, False), (1.0, False), (0.5, True)])
def test_a_steady_state_roots_the_guard_only_inside_the_interval(tmp_path, marched, h, rooted):
    found = bngsim.Simulator(_steady(tmp_path, h), "ode").steady_state()
    assert found.converged
    assert abs(float(np.asarray(found.concentrations)[0]) - _s_star(h)) < 1e-6
    assert marched and [bool(roots) for roots in marched] == [rooted] * len(marched)


def test_a_steady_state_sweep_reads_each_entry_at_its_own_h(tmp_path, marched):
    sim = bngsim.Simulator(_steady(tmp_path, 2.0), "ode")
    entries = sim.steady_state_batch(params=[{"h": 2.0}, {"h": 0.5}])
    assert sorted(bool(roots) for roots in marched) == [False, True]
    for h, entry in zip((2.0, 0.5), entries, strict=True):
        assert abs(float(np.asarray(entry.concentrations)[0]) - _s_star(h)) < 1e-6


@pytest.mark.parametrize("h", [1.0, 2.0])
def test_the_steady_state_column_in_h_at_an_h_of_1(tmp_path, h):
    # h = 1 sits on the guard's `h >= 1`, which is no kink of the law: at S > 0
    # the guard is true on both sides. Main returned this column from the bare
    # body; with the guard on every law it was refused as on a kink.
    # S* = 2*(2/3)^(1/h), so dS*/dh = -S*·ln(2/3)/h^2.
    found = bngsim.Simulator(_steady(tmp_path, h), "ode").steady_state(sensitivity_params=["h"])
    column = float(np.asarray(found.sensitivity).ravel()[0])
    assert abs(column - (-_s_star(h) * np.log(2.0 / 3.0) / h**2)) < 1e-6


def test_a_condition_on_its_switch_that_is_not_the_loaders_guard_is_still_a_kink(tmp_path):
    # The guard is read as its body only in the loader's own shape: the same
    # condition around another body is a kink at h = 1, as it was.
    path = tmp_path / "kink.net"
    text = STEADY.format(
        h=1.0,
        functions=(
            "begin functions\n 1 gate() if((S_obs > 0) or (h <= 0) or (h >= 1), kin, 0)\n"
            "end functions\n"
        ),
        dosed=" 4 0 3 gate\n",
    ).replace("begin reactions", "begin groups\n 1 S_obs 1\nend groups\nbegin reactions")
    path.write_text(text, encoding="utf-8")
    model = bngsim.Model.from_net(str(path))
    from bngsim._parameter_kinks import parameters_on_a_kink

    assert parameters_on_a_kink(model._core, ["h"], conditions=True)
    assert not parameters_on_a_kink(_steady(tmp_path, 1.0)._core, ["h"], conditions=True)


@pytest.fixture
def budget(monkeypatch):
    """The run's budget of stops lowered to 1000, as test_crossing_stop_budget
    lowers it, with the resolvers' memo emptied on each side."""
    from bngsim import _switch_sensitivity as sw

    sw._SCHEDULE_CACHE.clear()
    monkeypatch.setattr(sw, "_RUN_STOP_BUDGET", 1000)
    yield
    sw._SCHEDULE_CACHE.clear()


def test_a_schedule_past_the_budget_beside_a_hill_law_at_h_2_is_alone(tmp_path, budget):
    # Nothing else switches at h = 2, so the step is bounded as it was on main,
    # where the guard's root made the run ask for a max_step. X is the on-time.
    run = {"t_span": (0.0, 2000.0), "n_points": 3}
    result = bngsim.Simulator(_steady(tmp_path, 2.0, schedule=True), "ode").run(**run)
    assert float(np.asarray(result.species)[-1, 2]) == pytest.approx(200.0, rel=1e-4)
    # At h = 0.5 the guard does switch, and the run is refused, as on main.
    with pytest.raises(bngsim.SimulationError, match="#890"):
        bngsim.Simulator(_steady(tmp_path, 0.5, schedule=True), "ode").run(**run)


def test_run_diffrax_takes_a_hill_law_at_h_2(tmp_path):
    pytest.importorskip("diffrax")
    from bngsim._diffrax_solver import run_diffrax

    # Refused with the guard's root, though nothing switches at h = 2.
    out = run_diffrax(_steady(tmp_path, 2.0), 0.0, 5.0, 3)
    assert np.all(np.isfinite(np.asarray(out["species"])))
    with pytest.raises(ValueError, match="switches on the model's state"):
        run_diffrax(_steady(tmp_path, 0.5), 0.0, 5.0, 3)


def test_the_kept_roots_are_read_once_from_a_memo_another_row_may_clear(tmp_path):
    # The memo is shared with clones on worker threads and cleared whole at
    # its cap: a key found by `in` may be gone at the read after it.
    class ClearedBetween(dict):
        def __contains__(self, key):
            found = super().__contains__(key)
            if found and isinstance(key, tuple) and key[0] == "at values":
                self.clear()
            return found

    model = _model(tmp_path, 2.0)
    covered = model.time_discontinuity_conditions()
    first = model._state_switch_roots_at_values(covered)
    model._state_switch_root_memo = ClearedBetween(model._state_switch_root_memo)
    assert model._state_switch_roots_at_values(covered) == first == ((), ())


M = "http://www.w3.org/1998/Math/MathML"


def _sbml(event: bool) -> str:
    # S decays at k*S while (S > 1) or (p >= 1), the condition's parameter part
    # true at p = 2, beside an event that never fires (time >= 1e9). It assigns
    # S: an event that assigned p would make p part of the state, which no
    # fold reads as a constant; one that assigns anything is enough to keep
    # every root.
    ev = (
        '<listOfEvents><event id="E" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="false" persistent="true"><math xmlns="{M}"><apply><geq/>'
        '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
        "<cn>1e9</cn></apply></math></trigger><listOfEventAssignments>"
        f'<eventAssignment variable="S"><math xmlns="{M}"><cn>0</cn></math></eventAssignment>'
        "</listOfEventAssignments></event></listOfEvents>"
        if event
        else ""
    )
    law = (
        f'<math xmlns="{M}"><apply><piecewise><piece><apply><times/><ci>k</ci><ci>S</ci></apply>'
        "<apply><or/><apply><gt/><ci>S</ci><cn>1</cn></apply>"
        "<apply><geq/><ci>p</ci><cn>1</cn></apply></apply></piece><otherwise><cn>0</cn></otherwise>"
        "</piecewise></apply></math>"
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"><model id="m">
<listOfCompartments><compartment id="C" size="1" constant="true"/></listOfCompartments>
<listOfSpecies><species id="S" compartment="C" initialConcentration="3"
 hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/></listOfSpecies>
<listOfParameters><parameter id="k" value="1" constant="true"/>
<parameter id="p" value="2" constant="true"/></listOfParameters>
<listOfReactions><reaction id="R" reversible="false"><listOfReactants>
<speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
<kineticLaw>{law}</kineticLaw></reaction></listOfReactions>
{ev}</model></sbml>"""


def test_a_model_with_events_keeps_its_roots():
    # The condition is decided at p = 2 either way; with an event, which may
    # set p during the run, nothing is folded. The answer is e^-t both ways.
    plain = bngsim.Model.from_sbml_string(_sbml(event=False))
    with_event = bngsim.Model.from_sbml_string(_sbml(event=True))
    folded = plain._state_switch_roots_at_values(plain.time_discontinuity_conditions())
    assert folded is not None and folded[0] == ()
    assert (
        with_event._state_switch_roots_at_values(with_event.time_discontinuity_conditions())
        is None
    )
    for model in (plain, with_event):
        result = bngsim.Simulator(model, "ode").run((0.0, 5.0), 6, rtol=1e-10, atol=1e-12)
        s = np.asarray(result.species)[:, 0]
        assert np.abs(s - 3 * np.exp(-np.asarray(result.time))).max() < 1e-7


@pytest.mark.skipif(
    not (os.environ.get("BNGPATH") and shutil.which("perl")), reason="BNG2.pl not available"
)
def test_the_issues_bngl_setparameter_after_generate_network(tmp_path):
    bngl = tmp_path / "hill.bngl"
    bngl.write_text(
        "begin model\nbegin parameters\n  Vmax 1\n  Kh 2\n  h 2\n  kin 1\nend parameters\n"
        "begin molecule types\n  S()\n  P()\nend molecule types\n"
        "begin seed species\n  S() 0\n  P() 0\nend seed species\n"
        "begin observables\n  Molecules S_obs S()\n  Molecules P_obs P()\nend observables\n"
        "begin reaction rules\n  S() -> P() Hill(Vmax,Kh,h)\n  0 -> S() kin\nend reaction rules\n"
        'end model\ngenerate_network({overwrite=>1})\nsetParameter("h",0.5)\n'
        'simulate({method=>"ode",t_end=>5,n_steps=>1,atol=>1e-12,rtol=>1e-10})\n',
        encoding="utf-8",
    )
    result = bngsim.run_bngl(str(bngl)).runs[-1].result
    assert abs(_s_end(result) - S_AT_HALF) < TOL


# ── The fold itself ─────────────────────────────────────────────────────────

CONSTANTS = frozenset({"h", "k", "on"})


def fold_parameter_decided(text, constants, value_of):
    # Imported here, so the run tests above still run where it is absent.
    from bngsim._switch_sensitivity import fold_parameter_decided as fold

    return fold(text, constants, value_of)


def _fold(text, value_of):
    return fold_parameter_decided(text, CONSTANTS, value_of).replace(" ", "")


def _values(**at):
    def value_of(side: str):
        side = side.strip()
        try:
            return float(side)
        except ValueError:
            return at.get(side)

    return value_of


def test_an_or_with_a_true_parameter_part_folds_to_1():
    text = "if((S > 0) or (h <= 0) or (h >= 1), body, 0)"
    assert _fold(text, _values(h=2.0)) == "if(1,body,0)"
    assert _fold(text, _values(h=0.5)) == text.replace(" ", "")


def test_an_and_with_a_false_parameter_part_folds_to_0():
    text = "if((A > 1) and (on >= 1), k, 0)"
    assert _fold(text, _values(on=0.0)) == "if(0,k,0)"
    assert "A > 1" in fold_parameter_decided(text, CONSTANTS, _values(on=1.0))


def test_a_condition_with_no_parameter_part_is_left_alone():
    text = "if((A > 1) or (B < 2), k, 0)"
    assert _fold(text, _values()) == text.replace(" ", "")


@pytest.mark.parametrize(
    "part",
    ["(time() >= 1)", "(time >= 1)", "(S >= 1)", "(h >= unknown)"],
    ids=["the clock", "the clock, bare", "a species", "a name it cannot read"],
)
def test_a_part_that_is_not_over_constants_decides_nothing(part):
    # Every name here reads 5, the clock and the species included, so only the
    # rule that a part reads constants alone keeps these from folding.
    text = f"if((A > 1) or {part}, k, 0)"
    readable = _values(h=2.0, **{"time()": 5.0, "time": 5.0, "S": 5.0, "unknown": 1.0})
    assert "A > 1" in fold_parameter_decided(text, CONSTANTS, readable)


def test_nested_conditions_are_folded_inside_out():
    text = "if((A > 1) or (h >= 1), if((B > 1) and (h <= 0), 1, 2), 3)"
    assert _fold(text, _values(h=2.0)) == "if(1,if(0,1,2),3)"


def test_a_side_that_cannot_be_read_leaves_the_condition_alone():
    text = "if((S > 0) or (h >= 1), body, 0)"
    assert "S > 0" in fold_parameter_decided(text, CONSTANTS, lambda side: None)


def test_a_decided_if_is_written_as_its_branch_where_asked():
    from bngsim._switch_sensitivity import fold_parameter_decided as fold

    def taken(text, **at):
        return fold(text, CONSTANTS, _values(**at), take_branch=True).replace(" ", "")

    assert taken("if((S > 0) or (h >= 1), a*b, 0)", h=2.0) == "(a*b)"
    assert taken("if((S > 0) and (h >= 1), a, c + d)", h=0.5) == "(c+d)"
    assert taken("2*if((S > 0) or (h >= 1), if(T > 3, a, b), 0)", h=2.0) == "2*(if(T>3,a,b))"
    # Undecided, it stays an if().
    assert taken("if((S > 0) or (h >= 1), a, 0)", h=0.5) == "if((S>0)or(h>=1),a,0)"


BODY = "V * (s ^ (h - 1)) / ((K ^ h) + (s ^ h))"


@pytest.mark.parametrize(
    ("law", "read_as_its_body"),
    [
        (f"if((s > 0) or (h <= 0) or (h >= 1), {BODY}, 0)", True),
        # As write_net writes it back.
        ("if((s>0)||(h<=0)||(h>=1),V*(s^(h-1))/((K^h)+(s^h)),0)", True),
        # Another exponent in the guard, another species in the body, a law
        # that is more than the guard, and the guard around another body.
        (f"if((s > 0) or (h <= 0) or (g >= 1), {BODY}, 0)", False),
        ("if((s > 0) or (h <= 0) or (h >= 1), V * (t ^ (h - 1)) / ((K ^ h) + (s ^ h)), 0)", False),
        (f"if((s > 0) or (h <= 0) or (h >= 1), {BODY}, 0) + 1", False),
        ("if((s > 0) or (h <= 0) or (h >= 1), kin, 0)", False),
    ],
)
def test_only_the_loaders_own_guard_is_read_as_its_body_for_a_kink(law, read_as_its_body):
    from bngsim._parameter_kinks import _without_hill_guard

    assert (_without_hill_guard(law) != law) is read_as_its_body


def test_a_hill_law_written_back_and_loaded_has_its_column_in_h_at_1(tmp_path):
    from bngsim.convert import write_net

    written = tmp_path / "written.net"
    write_net(_steady(tmp_path, 1.0), str(written))
    found = bngsim.Simulator(bngsim.Model.from_net(str(written)), "ode").steady_state(
        sensitivity_params=["h"]
    )
    column = float(np.asarray(found.sensitivity).ravel()[0])
    assert abs(column - (-_s_star(1.0) * np.log(2.0 / 3.0))) < 1e-6
