"""A BNGL rate law gated on model state must not be stepped over (issue #897).

The ``.net`` twin of GH #194. A rate law such as

    r() = if((time() >= onset()) && (time() < onset() + w), k, 0)
    onset() = 4*Sobs

is constant on each side of its window, so a step that spans the whole window
reads it as off at both ends and CVODE's error estimate over that step is near
zero. The window is lost, and tightening ``rtol`` does not help, because there
is no error to see. Issues #440/#443 place a stop at every crossing whose time
is known before the run, but this crossing moves with ``S``, so no stop can be
placed at it. The SBML loader registers such a threshold as a CVODE root
(GH #194); the ``.net`` loader registered nothing, and only a sensitivity run
(issue #150) rooted the crossing. A plain run now roots it too, with no jump.

What this locks:

  1. the window the issue reports, and a window on a plain species threshold,
     are integrated over at loose and at tight tolerance, against exact answers;
  2. the answer agrees with the same model's sensitivity run and with its SBML
     twin, and the twin gains no second root for a crossing its loader already
     registered;
  3. every ``run_batch`` row roots its own window;
  4. what is rooted: only a state-reading atom of a rate law. A counter clock is
     left to its stop, and an output-only function and an equality are left
     alone;
  5. the scan runs once per model, a clone inherits it, and a batch scans once
     for all of its rows;
  6. a root restarts the run only at a genuine crossing, one the solver's own
     trajectory resolves past its tolerance and whose flow on the near side
     stays finite at the surface and carries the state across: GH #176's
     parked trajectory and five twins, a state relaxing onto its threshold, and
     one on an unstable surface all keep main's answers; a residual parked at
     exactly 0.0 is not read as a second root; and a real crossing on a curved
     residual, whose window opens and closes inside one step, still restarts;
  7. the band a crossing has to clear is what CVODE's error weights hold each
     species to, added in quadrature (issue #907): a window on an observable
     over a thousand species, and one under a tracking atol, are integrated;
  8. a residual too deep for ``ast.unparse`` keeps every root, and one that
     starts at exactly zero does not print a SUNDIALS warning.
"""

import math
import os
import re
import subprocess
import sys
import textwrap
import time

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import _surface_key

# ── Models ──────────────────────────────────────────────────────────────────
# The issue's reproduction. S decays as exp(-t/2), so the window opens where
# t = 4*exp(-t/2) and closes where t = 4*exp(-t/2) + w. Y integrates k over it.
_GATE_NET = """\
begin parameters
    1 k   10.0
    2 w   {w}
    3 kd  0.5
end parameters
begin functions
    1 onset() 4*Sobs
    2 r() if((time()>=onset())&&(time()<(onset()+w)),k,0)
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

_GATE_ANTIMONY = """
species S = 1, Y = 0
k = 10; w = 0.01
onset := 4*S
J1: S -> ; 0.5*S
J2: -> Y; piecewise(k, (time >= onset) && (time < onset + w), 0)
"""


def _gate_edge(w):
    """The root of t = 4*exp(-t/2) + w, by Newton's method from t = 1.7."""
    t = 1.7
    for _ in range(50):
        step = (t - 4.0 * math.exp(-t / 2.0) - w) / (1.0 + 2.0 * math.exp(-t / 2.0))
        t -= step
        if abs(step) < 1e-15:
            break
    return t


def _gate_exact(w):
    return 10.0 * (_gate_edge(w) - _gate_edge(0.0))


def _gate_model(tmp_path, w=0.01):
    path = tmp_path / "gate.net"
    path.write_text(_GATE_NET.format(w=w))
    return bngsim.Model.from_net(str(path))


def _final_Y(model, rtol, **kw):
    result = bngsim.Simulator(model, **kw).run(
        sample_times=[0.0, 1.0, 3.0, 10.0], rtol=rtol, atol=rtol * 1e-2
    )
    return float(result.species[-1][1])


# A species threshold with no clock in it. A is synthesized at rate 2, so it is
# not a unit-rate counter and the window A in [1, 1.002) is t in [0.5, 0.501):
# Y(2) = k * 0.001 exactly.
_SPECIES_NET = """\
begin parameters
    1 k   5.0
    2 two 2.0
end parameters
begin functions
    1 r() if((Aobs>=1)&&(Aobs<1.002),k,0)
end functions
begin species
    1 A() 0
    2 Y() 0
end species
begin reactions
    1 0 1 two
    2 0 2 r
end reactions
begin groups
    1 Aobs 1
end groups
"""


def _write(tmp_path, text, name="model.net"):
    path = tmp_path / name
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


# ── 1. The windows ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
def test_the_reported_window_is_integrated_over(tmp_path, rtol):
    """The issue's model gave Y = 0 at both tolerances: the whole window fell
    inside one step. The answer is k*(t2 - t1) = 0.0540116...

    Y is a difference of two edge times near 1.7 that are only 0.0054 apart,
    so an error of order ``rtol`` in each edge is ~300 times larger relative
    to Y. The bound allows for that and still rejects the Y = 0 of the defect.
    """
    model = _gate_model(tmp_path)
    assert model.time_discontinuity_conditions() == (), (
        "the crossing is state-dependent; if a fixed stop now covers it, this "
        "test no longer exercises the root"
    )
    assert _final_Y(model, rtol) == pytest.approx(_gate_exact(0.01), rel=1000 * rtol)


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
def test_a_window_on_a_species_threshold_is_integrated_over(tmp_path, rtol):
    """No clock at all: the gate reads a species through a group."""
    model = _write(tmp_path, _SPECIES_NET)
    assert model.state_switch_root_conditions() == ("Aobs>=1", "Aobs<1.002")
    result = bngsim.Simulator(model).run(sample_times=[0.0, 1.0, 2.0], rtol=rtol, atol=rtol * 1e-2)
    assert float(result.species[-1][1]) == pytest.approx(5.0 * 0.001, rel=100 * rtol)


# ── 2. Agreement with the other paths ───────────────────────────────────────
def test_a_plain_run_agrees_with_a_sensitivity_run(tmp_path):
    """The sensitivity run rooted this crossing before this issue (#150), so the
    two used to disagree about Y itself. dY/dk is the window width t2 - t1,
    because neither edge moves with k."""
    plain = _final_Y(_gate_model(tmp_path), 1e-8)
    model = _gate_model(tmp_path)
    result = bngsim.Simulator(model, sensitivity_params=["k"]).run(
        sample_times=[0.0, 1.0, 3.0, 10.0], rtol=1e-8, atol=1e-10
    )
    assert float(result.species[-1][1]) == pytest.approx(plain, rel=1e-6)
    width = _gate_edge(0.01) - _gate_edge(0.0)
    assert float(result.sensitivities[-1][1][0]) == pytest.approx(width, rel=1e-5)


def test_the_net_model_agrees_with_its_sbml_twin(tmp_path):
    """The comparison that made the defect visible. The twin's loader already
    registered both edges as roots (GH #194), so it gains no second root: its
    conditions name `onset`, a parameter slot, and are matched against the rate
    law's atoms only after `onset` is inlined."""
    pytest.importorskip("antimony")
    twin = bngsim.Model.from_antimony_string(_GATE_ANTIMONY)
    assert twin._core.n_discontinuity_triggers == 2
    assert twin.state_switch_root_conditions() == ()
    sbml_Y = float(
        bngsim.Simulator(twin)
        .run(sample_times=[0.0, 1.0, 3.0, 10.0], rtol=1e-10, atol=1e-12)
        .species[-1][1]
    )
    assert sbml_Y == pytest.approx(_gate_exact(0.01), rel=1e-7)
    assert _final_Y(_gate_model(tmp_path), 1e-10) == pytest.approx(sbml_Y, rel=1e-7)


# ── 3. Batch rows ───────────────────────────────────────────────────────────
def test_every_batch_row_roots_its_own_window(tmp_path):
    """A batch row integrates a clone carrying its own parameter point."""
    model = _gate_model(tmp_path)
    widths = [0.005, 0.01, 0.02]
    results = bngsim.Simulator(model).run_batch(
        t_span=(0.0, 10.0), n_points=3, params=[{"w": w} for w in widths], rtol=1e-8, atol=1e-10
    )
    for w, result in zip(widths, results, strict=True):
        assert float(result.species[-1][1]) == pytest.approx(_gate_exact(w), rel=1e-5), w


# ── 4. What is rooted ───────────────────────────────────────────────────────
_CLOCK_AND_STATE_NET = """\
begin parameters
    1 k       0.1
    2 one     1.0
end parameters
begin functions
    1 dose() if(t>=100,k,0)
    2 relay() if(A>1,k,0)
    3 flag() if(B>3,1,0)
    4 pulse() if(B==2,k,0)
end functions
begin species
    1 counter() 0
    2 A() 0
    3 B() 0
end species
begin reactions
    1 0 1 one
    2 0 2 dose
    3 0 3 relay
    4 0 3 pulse
end reactions
begin groups
    1 t                    1
    2 A                    2
    3 B                    3
end groups
"""


def test_only_state_atoms_of_rate_laws_are_rooted(tmp_path):
    """``t`` is a unit-rate counter, so ``t>=100`` is issue #443's stop and a
    root on it as well would put a second restart where the landing already
    works. ``flag()`` feeds no rate law, so its crossing changes an output and
    not the trajectory. ``B==2`` holds on a set of measure zero, which the
    sensitivity path declines to root for the same reason. What is left is
    ``A>1``."""
    model = _write(tmp_path, _CLOCK_AND_STATE_NET)
    assert model.time_discontinuity_conditions() == ("t>=100",)
    assert model.state_switch_root_conditions() == ("A>1",)


_HIDDEN_LOGIC_NET = """\
begin parameters
    1 k   1.0
    2 two 2.0
end parameters
begin functions
    1 early() Aobs<0.5
    2 flag() early()||(Aobs>3)
    3 gain() k*if(flag(),1,0)
    4 r() gain()
end functions
begin species
    1 A() 0
    2 Y() 0
end species
begin reactions
    1 0 1 two
    2 0 2 r
end reactions
begin groups
    1 Aobs 1
end groups
"""


def test_logic_that_inlining_exposes_is_split_too(tmp_path):
    """The condition is written ``flag``, a name with no comparison in it; the
    comparisons are two calls down, and the disjunction only exists once
    ``flag`` is inlined. The rate law reaches ``if()`` through ``r -> gain``.
    A compound has no residual, so it must be split after inlining or neither
    of its surfaces is rooted."""
    model = _write(tmp_path, _HIDDEN_LOGIC_NET)
    assert model.state_switch_root_conditions() == ("Aobs<0.5", "Aobs>3")


def test_a_model_with_no_conditional_rate_law_roots_nothing(tmp_path):
    model = _write(tmp_path, _SPECIES_NET.replace("if((Aobs>=1)&&(Aobs<1.002),k,0)", "k*Aobs"))
    assert model.state_switch_root_conditions() == ()


def test_one_surface_written_two_ways_is_one_key():
    """The residual is built from whatever text reaches it, so redundant
    parentheses and the side each operand sits on must not count."""
    assert _surface_key("(time())-(((4*S)))") == _surface_key("(time())-((4*S))")
    assert _surface_key("(time())-((4*S))") == _surface_key("((4*S))-(time())")
    assert _surface_key("(A)-(1)") != _surface_key("(A)-(2)")
    # ExprTk's own spellings still parse, and text that does not is its own key.
    assert _surface_key("(x)-(if((a<b)&&(c>=d),1,0))") == _surface_key("(if(a<b&&c>=d,1,0))-(x)")
    assert _surface_key("(lambda)-(1)") == "(lambda)-(1)"


# ── 5. Cost ─────────────────────────────────────────────────────────────────
def test_the_scan_runs_once_and_a_clone_inherits_it(tmp_path, monkeypatch):
    """The answer depends on structure alone, so a fit pays for it once."""
    import bngsim._switch_sensitivity as sw

    calls = []
    real = sw.state_switch_root_conditions

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sw, "state_switch_root_conditions", counting)
    model = _gate_model(tmp_path)
    sim = bngsim.Simulator(model)
    for _ in range(3):
        model.reset()
        sim.run(sample_times=[0.0, 10.0])
    clone = model.clone()
    assert clone.state_switch_root_conditions() == model.state_switch_root_conditions()
    assert len(calls) == 1


# ── 6. A root the flow does not carry ───────────────────────────────────────
_LTYPE = "ltype_calcium_discontinuous_jacobian.net"

_V_STEP = "if(((-70+Voltage_Level)<-20),0.5,0.05)"
_V_STEP_RAMP = "if(((-70+Voltage_Level)<(-20+rr*time())),0.5,0.05)"
_V_LATCH = "if(((-70+Voltage_Level)<-20),0,kup)"


def _ltype_variant(text, kind):
    """GH #176's fixture, or a twin that parks the same way, 1e-11 on the exact
    trajectory's side of the threshold, while defeating one earlier rule.
    Returns the network and its oracle: the same network with each step
    replaced by the branch the exact trajectory takes throughout.

    ``split`` carries Voltage_Level as V + V2 exchanging at kx*time() and kx, so
    dg/dt is f_V + f_V2, two terms of about 0.3 that cancel to 1e-11. ``ramp``
    moves the threshold at rr = 1e-8 per unit time and feeds V the same ramp, so
    dg/dt = f_V - rr, and the rounding in d/dt of rr*time() is 1e-10 whatever
    rr is. ``latch`` adds 0 -> Voltage at 1e-6 on the far branch only, so once
    across, the far-side flow carries V further across. ``damped`` makes the
    latch's V approach its rest point as a critically damped second-order
    system, V' = P - Ps, so the flow arriving at the threshold is carried by P
    and does not shrink as V nears it. ``moving`` makes the latch's threshold
    50 + 1e-6*sin(10t), with V resting 1e-11 below its minimum, so the flow
    arriving at the threshold is its own finite d/dt.
    """

    def swap(old, new, n=1):
        nonlocal text
        assert text.count(old) == n, old
        text = text.replace(old, new)

    last_param = "   19 _rateLaw2       0.01  # Constant\n"
    last_rxn = "   25 10 8 _rateLaw2 #_R8\n"
    last_species = "   10 LTCC(b,g~C,loc~mem,p~P,s~I) 0\n"
    steps = {_V_STEP: "0.5", "if((Phospho_LTCC>0),k_pka_shift,0)": "k_pka_shift"}
    if kind == "split":
        swap(last_param, last_param + "   20 kx 0.01\n")
        swap(f"    4 v_rec() {_V_STEP}\n", f"    4 v_rec() {_V_STEP}\n    5 xf() kx*time()\n")
        swap(last_species, last_species + "   11 Voltage2() 0\n")
        swap(last_rxn, last_rxn + "   26 3 11 xf\n   27 11 3 kx\n   28 11 0 k_v_leak\n")
        swap("    8 Voltage_Level        3\n", "    8 Voltage_Level        3,11\n")
    elif kind == "ramp":
        swap(last_param, last_param + "   20 rr 1e-08\n")
        swap(
            f"    4 v_rec() {_V_STEP}\n",
            f"    4 v_rec() {_V_STEP_RAMP}\n    5 stim() k_v_stim+rr*(time()+1)\n",
        )
        swap("    4 0 3 k_v_stim #_R9\n", "    4 0 3 stim #_R9\n")
        steps = {_V_STEP_RAMP: "0.5", "if((Phospho_LTCC>0),k_pka_shift,0)": "k_pka_shift"}
    elif kind in ("latch", "damped", "moving"):
        swap(last_param, last_param + "   20 kup 1e-06\n")
        swap(f"    4 v_rec() {_V_STEP}\n", f"    4 v_rec() {_V_STEP}\n    5 vup() {_V_LATCH}\n")
        swap(last_rxn, last_rxn + "   26 0 3 vup\n")
        steps[_V_LATCH] = "0"
        if kind == "damped":
            swap("   20 kup 1e-06\n", "   20 kup 1e-06\n   21 Ps 10\n")
            second_order = "    6 fv() Pvel-Ps\n    7 fp() -(Voltage_Level-k_v_stim)-2*(Pvel-Ps)\n"
            swap("    5 vup() ", second_order + "    5 vup() ")
            swap("    4 0 3 k_v_stim #_R9\n", "    4 0 3 fv #_R9\n")
            swap("    5 3 0 k_v_leak #_R10\n", "    5 0 11 fp #_R10\n")
            swap(last_species, last_species + "   11 P() 10\n")
            swap("    8 Voltage_Level        3\n", "    8 Voltage_Level        3\n    9 Pvel 11\n")
        elif kind == "moving":
            swap(
                "   16 k_v_stim        49.99999999999  # Constant\n",
                "   16 k_v_stim        49.999998999990005  # Constant\n",
            )
            swap("   20 kup 1e-06\n", "   20 kup 1e-06\n   21 amp 1e-06\n   22 om 10.0\n")
            moving = "((-70+Voltage_Level)<(-20+(amp*sin(om*time()))))"
            swap("((-70+Voltage_Level)<-20)", moving, n=2)
            steps = {
                f"if({moving},0.5,0.05)": "0.5",
                "if((Phospho_LTCC>0),k_pka_shift,0)": "k_pka_shift",
                f"if({moving},0,kup)": "0",
            }
    exact = text
    for condition, branch in steps.items():
        assert exact.count(condition) == 1, condition
        exact = exact.replace(condition, branch)
    return text, exact


@pytest.mark.parametrize("kind", ["fixture", "split", "ramp", "latch", "damped", "moving"])
def test_a_parked_root_does_not_restart_the_run(data_dir, tmp_path, kind):
    """GH #176's fixture parks Voltage 1e-11 below the step at 50, far inside
    the solver's own error at rtol 1e-8, so the interpolant can cross the step
    while the vector field points back: dV/dt = 49.99999999999 - V < 0 there.
    Restarting on that surface held the run on the branch the exact trajectory
    never takes. The observables came out 63% off after 2,019,910 steps, with no
    error.

    The core restarts at a lone state-switch root only at a genuine crossing:
    the solver's own trajectory has to go past the surface by more than its
    tolerance can blur, and the flow on the near side, where the arriving
    branch is live, has to stay finite at the surface and carry the state
    across. A parked trajectory fails the first test at least, so its root
    steps on as it did before the root existed: the analytical attempt fails
    loudly and ``auto`` retries with FD.

    Each twin defeated an earlier rule (review of PR #903):
    - Restarting unless the flow read as opposed sent split and ramp 9.4e-3 and
      0.62 off, worse at tighter tolerances.
    - Restarting whenever the flow at the located state carried sent latch 0.633
      off at every tolerance.
    - Restarting on a finite near-side flow alone sent damped and moving 1.085
      and 0.16 off: there the arriving flow is finite, and only the tolerance
      tells the crossing is not resolved.

    main is right on all six.
    """
    text, exact_text = _ltype_variant((data_dir / _LTYPE).read_text(), kind)
    model = _write(tmp_path, text, f"{kind}.net")
    assert any("Voltage_Level" in c for c in model.state_switch_root_conditions())
    window = dict(t_span=(0.0, 150.0), n_points=301, rtol=1e-8, atol=1e-8)
    # Keyed on explicit FD, as the #176 tests are: whether FD carries this
    # fixture is the host's business, but once it does, the default run failing
    # is this rule failing and must not read as a skip.
    try:
        bngsim.Simulator(_write(tmp_path, text, f"{kind}_fd.net"), jacobian="fd").run(**window)
    except bngsim.SimulationError:
        pytest.skip(
            "the finite-difference Jacobian does not carry this fixture on this "
            "build, so there is no rescue to compare (see lanl/bngsim#176)"
        )
    result = bngsim.Simulator(model).run(**window)
    assert result.solver_stats["n_steps"] < 10_000
    exact = bngsim.Simulator(_write(tmp_path, exact_text, f"{kind}_exact.net")).run(
        **{**window, "rtol": 1e-12, "atol": 1e-12}
    )
    got = np.asarray(result.observables)
    want = np.asarray(exact.observables)
    scale = np.maximum(np.abs(want).max(axis=0), 1e-12)
    assert float((np.abs(got - want) / scale).max()) < 1e-6


def test_a_state_resting_on_its_surface_steps_on(data_dir, tmp_path):
    """The rulehub model the #176 fixture was edited from: k_v_stim is 50.0, so
    Voltage relaxes to exactly the threshold. The exact trajectory never reaches
    it (V = 50 - 49 e^-t), and main converges to that at rtol 1e-10 (9.8e-10).
    The root finder, though, interpolates V onto 50.0 itself, where dV/dt = 0.
    A rule that restarted on a residual lying on its surface held V on 50.0 and
    on the far branch for the rest of the run: 0.633 off at every tolerance
    (review of PR #903). V relaxes onto the threshold, so its near-side flow
    vanishes there, and the root steps on.
    """
    text = (data_dir / _LTYPE).read_text()
    old = "   16 k_v_stim        49.99999999999  # Constant\n"
    assert text.count(old) == 1
    text = text.replace(old, "   16 k_v_stim        50.0  # Constant\n")
    window = dict(t_span=(0.0, 150.0), n_points=301, rtol=1e-10, atol=1e-10)
    result = bngsim.Simulator(_write(tmp_path, text, "upstream.net")).run(**window)
    exact_text = text
    for condition, branch in (
        (_V_STEP, "0.5"),
        ("if((Phospho_LTCC>0),k_pka_shift,0)", "k_pka_shift"),
    ):
        assert exact_text.count(condition) == 1, condition
        exact_text = exact_text.replace(condition, branch)
    exact = bngsim.Simulator(_write(tmp_path, exact_text, "upstream_exact.net")).run(
        **{**window, "rtol": 1e-12, "atol": 1e-12}
    )
    got = np.asarray(result.observables)
    want = np.asarray(exact.observables)
    scale = np.maximum(np.abs(want).max(axis=0), 1e-12)
    assert float((np.abs(got - want) / scale).max()) < 1e-6


# dY_dt() > 0, inlined: (Sig_Y() - Y_Out)/Tau.
_Y_RELAXATION = (
    "(((1/(1+exp(((-Gain)*(((H1_Act*(V1_Pos-V1_Neg))+(H2_Act*(V2_Pos-V2_Neg)))+Bias)))))"
    "-Y_Out)/Tau)>0"
)


def test_a_residual_parked_at_zero_is_not_a_second_root(data_dir):
    """Stepping on past a root leaves CVODE holding that root's value
    from the crossing. nn_xor's output relaxation settles with Y_Out equal to
    Sig_Y() to the last bit, so near t = 155.4 the root it passes reads exactly
    0.0, and CVODE's next call found it zero again at the same instant and
    refused the run: CV_ILL_INPUT, "Root found at and very near t", under every
    Jacobian. main integrates this model in 382 steps; a rooted run has to
    integrate it too, at a comparable cost (its restarts at real crossings take
    a few more), and to the same accuracy."""
    model = bngsim.Model.from_net(str(data_dir / "nnxor_parked_residual.net"))
    assert _Y_RELAXATION in model.state_switch_root_conditions()
    window = dict(t_span=(0.0, 160.0), n_points=201)
    result = bngsim.Simulator(model).run(**window, rtol=1e-8, atol=1e-8)
    assert result.solver_stats["n_steps"] <= 500
    reference = bngsim.Simulator(
        bngsim.Model.from_net(str(data_dir / "nnxor_parked_residual.net"))
    ).run(**window, rtol=1e-11, atol=1e-13)
    got = np.asarray(result.species)
    want = np.asarray(reference.species)
    scale = np.maximum(np.abs(want).max(axis=0), 1e-12)
    assert float((np.abs(got - want) / scale).max()) < 1e-5


# The window opens on a cubic, 3u - u^3 > 1.971 with u = A - 6, at u = 0.9,
# and closes on the linear A < 6.905. A = 2t, so it is open for 0.0025 time
# units and Y = 10 * 0.0025 = 0.025 exactly.
_CUBIC_NET = """\
begin parameters
    1 k    10.0
    2 two  2.0
    3 c    1.971
    4 s    6.0
    5 g0   1.9
    6 wc   0.905
end parameters
begin functions
    1 r() if(((3*(Aobs-s)-(Aobs-s)^3)>c)&&(Aobs<(s+wc))&&(Aobs>(s-g0)),k,0)
end functions
begin species
    1 A() 0
    2 Y() 0
end species
begin reactions
    1 0 1 two
    2 0 2 r
end reactions
begin groups
    1 Aobs 1
end groups
"""


@pytest.mark.parametrize("max_step", [None, 0.37, 0.385])
@pytest.mark.parametrize("rtol", [1e-6, 1e-8, 1e-10])
def test_a_crossing_on_a_curved_residual_restarts(tmp_path, rtol, max_step):
    """Whether the flow carries a root across is read from dg/dt at the located
    state.
    It used to be read as a secant along x +/- h*f over the whole last step,
    and 3u - u^3 curves enough across one that the secant pointed backwards
    once h > 0.3775: the opening was read as opposed, the restart skipped, and
    Y came out 0 at every tolerance (review of PR #903). max_step pins h on
    either side of that threshold."""
    model = _write(tmp_path, _CUBIC_NET)
    assert "(3*(Aobs-s)-(Aobs-s)^3)>c" in model.state_switch_root_conditions()
    kw = {} if max_step is None else {"max_step": max_step}
    result = bngsim.Simulator(model).run(
        t_span=(0.0, 5.0), n_points=3, rtol=rtol, atol=rtol * 1e-2, **kw
    )
    assert float(result.species[-1][1]) == pytest.approx(0.025, rel=1000 * rtol)


@pytest.mark.parametrize("rtol", [1e-6, 1e-8, 1e-10])
@pytest.mark.parametrize(
    ("t_end", "n_points"),
    [
        (3.6, 3),
        (4.0, 3),
        (5.0, 101),
        (6.0, 3),
        (7.0, 3),
        (8.0, 3),
        (10.0, 3),
        (10.0, 101),
        (20.0, 3),
        (20.0, 11),
        (50.0, 3),
    ],
)
def test_a_curved_residual_restarts_whatever_the_run_length(tmp_path, t_end, n_points, rtol):
    """Issue #907, case 1: the test above passed over (0, 5) with 3 points, and
    the window was lost on every grid here. CVODE grows its step about tenfold
    over this smooth stretch, so the step after the opening root ran many times
    the width of the cubic's hump, and the restart test's sixteen even samples
    of it straddled the hump: the crossing read as unresolved, the run stepped
    on, and Y came out 0. The test samples halving toward the root as well."""
    model = _write(tmp_path, _CUBIC_NET)
    result = bngsim.Simulator(model).run(
        t_span=(0.0, t_end), n_points=n_points, rtol=rtol, atol=rtol * 1e-2
    )
    assert float(result.species[-1][1]) == pytest.approx(0.025, rel=1000 * rtol)


@pytest.mark.parametrize("rtol", [4.9e-4, 5e-4])
@pytest.mark.parametrize("t_end", [4.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0])
def test_a_hump_that_only_just_clears_its_band_restarts(tmp_path, t_end, rtol):
    """At rtol 5e-4 the cubic's hump, 0.029 at its top, clears the 10*sqrt(2)
    bands the restart test asks of it (0.0283 there) only over a core from
    0.042 to 0.058 time units past the root. The halvings toward the root, a
    factor of two apart, and the even samples fell either side of that core
    on 25 runs of 242 near this tolerance, eleven of these fourteen among them,
    and Y came out 0. The best sample is now refined between its two
    neighbours."""
    model = _write(tmp_path, _CUBIC_NET)
    result = bngsim.Simulator(model).run(
        t_span=(0.0, t_end), n_points=3, rtol=rtol, atol=rtol * 1e-2
    )
    assert float(result.species[-1][1]) == pytest.approx(0.025, rel=1e-2)


def _clock_closed_window(n, x0_each, rate_each, thr, t_close, k=10.0):
    """`if((Obs > thr) && (Tobs < 1), k, 0)`: open from where Obs, the sum of n
    species each growing at rate_each, passes thr until a clock T (T' = 1)
    reaches 1 at t_close. Y integrates k over it."""
    species = [f"    {i + 1} X{i}() {x0_each!r}" for i in range(n)]
    reactions = [f"    {i + 1} 0 {i + 1} ri" for i in range(n)]
    members = ",".join(str(i + 1) for i in range(n))
    return (
        f"begin parameters\n    1 k {k}\n    2 ri {rate_each!r}\n    3 thr {thr!r}\n"
        "    4 one 1.0\nend parameters\n"
        "begin functions\n    1 r() if((Obs>thr)&&(Tobs<1),k,0)\nend functions\n"
        "begin species\n" + "\n".join(species) + f"\n    {n + 1} T() {1.0 - t_close!r}\n"
        f"    {n + 2} Y() 0\nend species\n"
        "begin reactions\n" + "\n".join(reactions) + f"\n    {n + 1} 0 {n + 1} one\n"
        f"    {n + 2} 0 {n + 2} r\nend reactions\n"
        f"begin groups\n    1 Obs {members}\n    2 Tobs {n + 1}\nend groups\n"
    )


# Where the window opens: five phases of it.
_PHASES = [5.0, 5.37, 6.13, 7.71, 9.19]
# The oracle: Obs reaches thr at t_open and T reaches 1 at t_open + 0.5, by
# construction, so r = k = 10 for 0.5 and Y = 5. A lost window gives Y = 0.
# The tolerance: CVODE holds Y to rtol 1e-8, and the 30 runs below are at most
# 3.1e-8 off (measured). 1e-6 leaves 30x and still fails a window that loses
# more than 5e-7 of its 0.5 (k*5e-7 = 5e-6 = 1e-6 of Y).
_WINDOW_Y = 5.0
_WINDOW_REL = 1e-6


@pytest.mark.parametrize("t_end", [20.0, 50.0])
@pytest.mark.parametrize("n", [1, 1000])
@pytest.mark.parametrize("t_open", _PHASES)
def test_a_window_on_an_observable_over_many_species_is_integrated(tmp_path, n, t_open, t_end):
    """Issue #907, case 3: Obs moves 0.1% of itself, 1e-3, across a window of
    0.5 that a clock closes. Summed over a thousand species of 1e-3 each, the
    band the restart test asked the crossing to clear was rtol*Obs + 1000*atol
    times 10*sqrt(N), 3.2e-3, past the whole excursion: the crossing read as
    unresolved, the run stepped on, and Y came out 0 on three phases of five.
    The WRMS norm bounds the residual's error by its terms in quadrature,
    sqrt(1000)*atol, so the bar is 1e-4. One species is the control, and two
    run lengths give the step after the root two geometries."""
    rate = 1e-3 / 0.5
    text = _clock_closed_window(n, (1.0 - rate * t_open) / n, rate / n, 1.0, t_open + 0.5)
    model = _write(tmp_path, text)
    result = bngsim.Simulator(model).run(t_span=(0.0, t_end), n_points=3, rtol=1e-8, atol=1e-8)
    assert float(result.species[-1][-1]) == pytest.approx(_WINDOW_Y, rel=_WINDOW_REL)


@pytest.mark.parametrize("t_end", [20.0, 50.0])
@pytest.mark.parametrize("t_open", _PHASES)
def test_a_window_under_a_tracking_atol_is_integrated(tmp_path, t_open, t_end):
    """Issue #907, case 5: under TrackingAtol(decades=6, ceiling=1e-8) CVODE
    holds S, about 1e-6, to 2e-14, but the restart test read its band from the
    1e-8 ceiling, and a window 5e-9 wide in S, closed by a clock, was lost on
    three phases of five. The band is now what the error weight's own rule
    holds S to, the tracked atol."""
    rate = 5e-9 / 0.5
    text = _clock_closed_window(1, 1e-6 - rate * t_open, rate, 1e-6, t_open + 0.5)
    model = _write(tmp_path, text)
    result = bngsim.Simulator(model).run(
        t_span=(0.0, t_end),
        n_points=3,
        rtol=1e-8,
        atol=bngsim.TrackingAtol(decades=6, ceiling=1e-8),
    )
    assert float(result.species[-1][-1]) == pytest.approx(_WINDOW_Y, rel=_WINDOW_REL)


def test_a_residual_too_deep_to_print_keeps_every_root(tmp_path):
    """A rate law over a sum of 400 observables has a residual that nests too
    deep for ``ast.unparse``. The RecursionError escaped the dedupe key, the
    model-level fallback caught it, and every root of the model went with it,
    the issue's own window included (Y = 0; review of PR #903)."""
    n = 400
    species = ["    1 S() 1", "    2 Y() 0", "    3 Z() 0"]
    species += [f"    {4 + i} A{i}() 0.001" for i in range(n)]
    groups = ["    1 Sobs 1"] + [f"    {2 + i} O{i} {4 + i}" for i in range(n)]
    total = "+".join(f"O{i}" for i in range(n))
    net = _GATE_NET.format(w=0.01)
    net = net.replace("    3 kd  0.5\n", "    3 kd  0.5\n    4 c   1e9\n")
    net = net.replace(
        "end functions", f"    3 tot() {total}\n    4 g() if(tot()>c,1,0)\nend functions"
    )
    net = net.replace("    1 S() 1\n    2 Y() 0", "\n".join(species))
    net = net.replace("    2 0 2 r\n", "    2 0 2 r\n    3 0 3 g\n")
    net = net.replace("    1 Sobs 1", "\n".join(groups))
    assert _surface_key(f"({total})-(c)") == f"({total})-(c)"
    model = _write(tmp_path, net)
    conditions = model.state_switch_root_conditions()
    assert "time()>=(4*Sobs)" in conditions and len(conditions) == 3
    assert _final_Y(model, 1e-10) == pytest.approx(_gate_exact(0.01), rel=1e-5)


def test_a_batch_scans_once_for_all_its_rows(tmp_path, monkeypatch):
    """Every row runs on a clone, which inherits the scan only if the parent
    has made it. The parent had not, so each row scanned again, and the next
    batch did it all over (review of PR #903)."""
    import bngsim._switch_sensitivity as sw

    calls = []
    real = sw.state_switch_root_conditions

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sw, "state_switch_root_conditions", counting)
    sim = bngsim.Simulator(_gate_model(tmp_path))
    for _ in range(2):
        sim.run_batch(t_span=(0.0, 10.0), n_points=3, params=[{"w": 0.01}] * 5)
    assert len(calls) == 1


# td_right() written as a mathematically identical sum. On main the model gives
# the same answer; for a rule that restarted on each carried wobble across the
# symmetric surface, it did not.
_QL_REORDERED = (
    "    8 td_right() (reward()+(gamma*max_q()))-qr()\n",
    "    8 td_right() (reward()-qr())+(gamma*max_q())\n",
)


@pytest.mark.parametrize("spelling", ["as written", "reordered"])
def test_a_residual_on_an_unstable_surface_steps_on(data_dir, tmp_path, spelling):
    """ml_q_learning's Q_Right and Q_Left meet exactly with identical
    derivatives, so roots of its argmax switch land where the residual is
    exactly zero. The symmetric state is unstable: near it the residual grows
    only in proportion to itself, so its near-side flow vanishes on the
    surface, no crossing there is genuine, and the run steps on as main does.

    Earlier rules got this wrong both ways (review of PR #903). Stepping on while
    restarting on every ~6e-11 wobble the flow carried gave about 140,000 root
    returns and 4.9 off at rtol 1e-11. Restarting on the surface instead was
    right as written, but 20x off once ``td_right`` was reordered, because each
    restart let rounding-level asymmetry grow. The reference is the same model
    with no state-switch roots at all, which is how main integrates it."""
    text = (data_dir / "qlearning_flat_residual.net").read_text()
    if spelling == "reordered":
        old, new = _QL_REORDERED
        assert text.count(old) == 1
        text = text.replace(old, new)
    model = _write(tmp_path, text, "ql.net")
    assert "(Q_Right-off)>(Q_Left-off)" in model.state_switch_root_conditions()
    window = dict(t_span=(0.0, 100.0), n_points=1001, rtol=1e-11, atol=1e-11)
    result = bngsim.Simulator(model).run(**window)
    assert result.solver_stats["n_steps"] < 5_000
    unrooted = _unrooted(tmp_path, text, "ql_unrooted.net")
    reference = bngsim.Simulator(unrooted).run(**window)
    got = np.asarray(result.species)
    want = np.asarray(reference.species)
    scale = np.maximum(np.abs(want).max(axis=0), 1e-12)
    assert float((np.abs(got - want) / scale).max()) < 1e-6


# ── 7. Output ───────────────────────────────────────────────────────────────
_ZERO_RESIDUAL_NET = """\
begin parameters
    1 k   1.0
end parameters
begin functions
    1 r() if(Eobs>0,k,0)
end functions
begin species
    1 E() 0
    2 Y() 0
end species
begin reactions
    1 0 2 r
end reactions
begin groups
    1 Eobs 1
end groups
"""


def test_a_residual_that_starts_at_zero_prints_nothing(tmp_path):
    """``E`` starts at 0 and nothing produces it, so the residual of ``E>0`` is
    identically zero and CVODE warns about it on stdout. A discontinuity root
    already had its warnings routed to the null sink; a state-switch root makes
    the same restarts and now shares that filter.

    In a subprocess, because SUNDIALS writes through C stdio, which is buffered
    when stdout is not a terminal and reaches the file descriptor only at exit.
    """
    path = tmp_path / "zero.net"
    path.write_text(_ZERO_RESIDUAL_NET)
    assert bngsim.Model.from_net(str(path)).state_switch_root_conditions() == ("Eobs>0",)
    script = textwrap.dedent(f"""
        import bngsim
        m = bngsim.Model.from_net({str(path)!r})
        r = bngsim.Simulator(m).run(t_span=(0.0, 10.0), n_points=11)
        print("Y", float(r.species[-1][1]))
    """)
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert "Y 0.0" in done.stdout
    assert "identically 0" not in done.stdout + done.stderr


# ── 8. A slide along the surface (issue #908) ───────────────────────────────
# The issue's reproduction. f() switches sign at S = 1, so S slides along it and
# crosses it on most steps. Each crossing was a root, located, returned and
# declined by the restart test.
_SLIDE_NET = """\
begin parameters
    1 amp 1.0
    2 one 1.0
end parameters
begin functions
    1 f() if(Sobs<1,amp,-amp)
end functions
begin species
    1 S() 0
    2 W() 0
end species
begin reactions
    1 0 1 f
    2 1 1,2 one
end reactions
begin groups
    1 Sobs 1
end groups
"""


def _unrooted(tmp_path, text, name):
    """The model with its state-switch roots stood down, the reference a
    rooted run is measured against. The roots are derived before they are
    blanked, so the blank sits under the key a run asks with (issue #726).
    Blanked before that, it was derived again at the first run, and the
    reference was rooted too: a comparison that could not fail."""
    model = _write(tmp_path, text, name)
    assert model.state_switch_root_conditions()
    model._state_switch_root_conditions = ()
    assert model.state_switch_root_conditions() == ()
    return model


def _within_budget(rooted, unrooted, **run):
    """Run ``unrooted`` and then ``rooted``, the second with a timeout of three
    times the first's wall time plus a quarter second. Measured here rather
    than fixed, so that the budget moves with the machine."""
    start = time.perf_counter()
    reference = bngsim.Simulator(unrooted).run(**run)
    budget = 3.0 * (time.perf_counter() - start) + 0.25
    return bngsim.Simulator(rooted).run(**run, timeout=budget), reference


@pytest.mark.parametrize("n_points", [None, 401])
def test_a_slide_costs_what_it_cost_without_its_root(tmp_path, n_points):
    """The issue's model ran 7.6 times as long as the same model with no
    state-switch roots (2.8 s against 0.37 s on arm64 macOS at rtol 1e-5), and
    12 times at rtol 1e-6, over the same 1.3 million steps. The slide's
    crossings are now held off the root search. The run takes every step the
    unrooted run takes, within three times its wall time, with two output
    points and with an output point at each of 401, where each crossing since
    the last is searched for and tested. Fails on main (timeout)."""
    run = (
        dict(sample_times=[0.0, 4.25])
        if n_points is None
        else dict(t_span=(0.0, 4.25), n_points=n_points)
    )
    result, reference = _within_budget(
        _write(tmp_path, _SLIDE_NET),
        _unrooted(tmp_path, _SLIDE_NET, "unrooted.net"),
        rtol=1e-5,
        atol=1e-5,
        **run,
    )
    assert result.solver_stats["n_steps"] == reference.solver_stats["n_steps"]
    np.testing.assert_array_equal(np.asarray(result.species), np.asarray(reference.species))


# S slides on S = 1 until t1, is then driven down for half a time unit and up at
# rate 1, and crosses S = 1 again at t_open = t2 + 1 - S(t2). Y integrates k
# from there until tc, which a time condition closes. One state residual, so a
# hold that outlived the slide would also hide the window's opening.
_SLIDE_THEN_WINDOW_NET = """\
begin parameters
    1 k   10.0
    2 t1  1.5
    3 t2  2.0
    4 tc  2.501
end parameters
begin functions
    1 amp() if(time()<t1,1,0)
    2 f() if(Sobs<1,amp(),-amp())
    3 g() if(time()>=t2,1,if(time()>=t1,-1,0))
    4 r() if((Sobs>1)&&(time()>t2)&&(time()<tc),k,0)
end functions
begin species
    1 S() 0
    2 Y() 0
end species
begin reactions
    1 0 1 f
    2 0 1 g
    3 0 2 r
end reactions
begin groups
    1 Sobs 1
end groups
"""


def test_a_residual_held_through_a_slide_still_opens_a_window(tmp_path):
    """The first version of the #908 fix reported a chattering residual as a
    constant for the rest of the run, so the window S > 1 that opens after the
    slide was never rooted, and Y came out 0 where it is 0.00999. Now a
    restart (here, the stop at t1) releases the hold, and the window is rooted
    and restarted at as on main. Main gets this right too (control)."""
    model = _write(tmp_path, _SLIDE_THEN_WINDOW_NET)
    assert model.state_switch_root_conditions() == ("Sobs<1",)
    result = bngsim.Simulator(model).run(sample_times=[0.0, 1.5, 2.0, 4.0], rtol=1e-5, atol=1e-7)
    s_t2 = float(result.species[1][0]) - 0.5
    exact = 10.0 * (2.501 - (2.0 + 1.0 - s_t2))
    assert float(result.species[-1][1]) == pytest.approx(exact, rel=1e-3)


# The same slide with the window gated by two residuals of S: two state
# switches, both of which slide.
_SLIDE_TWO_RESIDUALS_NET = """\
begin parameters
    1 k   10.0
    2 w   0.001
end parameters
begin functions
    1 amp() if(time()<1.5,1,0)
    2 f() if(Sobs<1,amp(),-amp())
    3 g() if(time()>=2,1,if(time()>=1.5,-1,0))
    4 r() if((Sobs>1)&&(Sobs<1+w)&&(time()>2),k,0)
end functions
begin species
    1 S() 0
    2 Y() 0
end species
begin reactions
    1 0 1 f
    2 0 1 g
    3 0 2 r
end reactions
begin groups
    1 Sobs 1
end groups
"""


def test_two_residuals_that_slide_are_both_held(tmp_path):
    """The first version of the #908 fix cleared every residual's chatter flag
    whenever any one was not chattering, so a model with two state switches
    paid in full: 3.6 s against 0.66 s with no state-switch roots, as on main.
    Both are now held, within three times the unrooted run, and the window S in
    (1, 1 + w), crossed at rate 1, still gives Y = k*w. Fails on main
    (timeout)."""
    model = _write(tmp_path, _SLIDE_TWO_RESIDUALS_NET)
    assert sorted(model.state_switch_root_conditions()) == ["Sobs<1", "Sobs<1+w"]
    result, _ = _within_budget(
        model,
        _unrooted(tmp_path, _SLIDE_TWO_RESIDUALS_NET, "unrooted.net"),
        sample_times=[0.0, 1.5, 3.0],
        rtol=1e-5,
        atol=1e-7,
    )
    assert float(result.species[-1][1]) == pytest.approx(0.01, rel=1e-3)


# S comes down onto thr() = 1 and slides on it until a() switches off at te.
# The drift dlt then lifts S off the surface, but not out of the band, and
# thr() sweeps up past S, fast. Only the residual and Y's rate read thr(), so
# CVODE's steps are not held back by the sweep, and one step goes from inside
# the band to far past the surface. Y integrates k while thr() - w < S < thr(),
# a window that opens on the slid residual's own crossing. z() keeps Y off
# during the slide.
_SWEEP_NET = """\
begin parameters
    1 k    10.0
    2 w    0.001
    3 A    0.5
    4 tp   30.05
    5 sig  0.001
    6 dlt  {dlt}
    7 te   30.0
end parameters
begin functions
    1 thr() 1+A*(1+tanh((time()-tp)/sig))/2
    2 a() 1/(1+exp((time()-te)/0.002))
    3 f() if(Sobs<thr(),a(),-a())
    4 up() dlt
    5 z() 1/(1+exp(-(time()-(te+0.02))/0.002))
    6 r() if((Sobs<thr())&&(Sobs>(thr()-w)),k,0)*z()
end functions
begin species
    1 S() 27.95
    2 Y() 0
end species
begin reactions
    1 0 1 f
    2 0 1 up
    3 0 2 r
end reactions
begin groups
    1 Sobs 1
end groups
"""


@pytest.mark.parametrize("rtol, dlt, rel", [(1e-3, 0.1, 1e-2), (1e-4, 0.01, 3e-2)])
def test_a_window_opened_by_a_moving_threshold_is_integrated(tmp_path, rtol, dlt, rel):
    """thr() sweeps past S in one step and opens a window on the slid residual's
    own crossing, which the run restarts at. S moves at dlt across the window
    and thr() is known, so the window's width is exact given S before the
    sweep. Main gets this right (control); the residual reads the time and is
    not held (see the next test)."""
    model = _write(tmp_path, _SWEEP_NET.format(dlt=dlt))
    t1, tp, amp, sig, w = 30.04, 30.05, 0.5, 0.001, 0.001
    result = bngsim.Simulator(model).run(
        sample_times=[0.0, t1, tp + 1.0], rtol=rtol, atol=rtol * 1e-2
    )
    s1 = float(result.species[1][0])

    def edge(offset):
        lo, hi = t1, tp + 0.5
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            thr = 1.0 + amp * (1.0 + math.tanh((mid - tp) / sig)) / 2.0
            if thr - offset - (s1 + dlt * (mid - t1)) < 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    exact = 10.0 * (edge(w) - edge(0.0))
    gained = float(result.species[-1][1]) - float(result.species[1][1])
    assert gained == pytest.approx(exact, rel=rel)


# The same sweep with the residual scaled by h(), which falls from 1 to 1e-3
# just before it, and Y's rate a pulse in time rather than a window with a
# second edge: nothing but the residual's own crossing ends the step.
_SCALED_SWEEP_NET = """\
begin parameters
    1 k    10.0
    2 A    0.5
    3 tp   30.05
    4 sig  0.001
    5 dlt  0.01
    6 te   30.0
end parameters
begin functions
    1 thr() 1+A*(1+tanh((time()-tp)/sig))/2
    2 h() 1e-3+1/(1+exp((time()-30.02)/0.002))
    3 a() 1/(1+exp((time()-te)/0.002))
    4 f() if(Sobs*h()<thr()*h(),a(),-a())
    5 up() dlt
    6 z() exp(-((time()-30.047)/0.001)^2)
    7 r() if(Sobs*h()<thr()*h(),k,0)*z()
end functions
begin species
    1 S() 27.95
    2 Y() 0
end species
begin reactions
    1 0 1 f
    2 0 1 up
    3 0 2 r
end reactions
begin groups
    1 Sobs 1
end groups
"""


def test_a_residual_that_reads_the_time_is_not_held(tmp_path):
    """A residual that reads the time is never held: its g and dg/dx move with
    the clock while the state stands still, which a band measured once cannot
    follow. Held with the band measured where h was 1, S*h < thr*h hid its own
    crossing once h had fallen to 1e-3, the run was not restarted there, and
    the pulse that crossing lets through was lost: Y gained 0 where it gains
    0.0134 (second review of the #908 fix). Main gets it right (control). The
    pulse is a Gaussian and S moves at dlt, so the gain is exact given S before
    the sweep."""
    model = _write(tmp_path, _SCALED_SWEEP_NET)
    t1, tp, amp, sig, dlt = 30.04, 30.05, 0.5, 0.001, 0.01
    result = bngsim.Simulator(model).run(sample_times=[0.0, t1, tp + 1.0], rtol=1e-3, atol=1e-5)
    s1 = float(result.species[1][0])
    lo, hi = t1, tp + 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        thr = 1.0 + amp * (1.0 + math.tanh((mid - tp) / sig)) / 2.0
        if thr - (s1 + dlt * (mid - t1)) < 0.0:
            lo = mid
        else:
            hi = mid
    t_open = 0.5 * (lo + hi)
    exact = 10.0 * 0.001 * math.sqrt(math.pi) / 2.0 * math.erfc((t_open - 30.047) / 0.001)
    gained = float(result.species[-1][1]) - float(result.species[1][1])
    assert gained == pytest.approx(exact, rel=2e-2)


# The scaled sweep again with the clock a species: C grows at rate 1, as a
# BNGL "Time" species does, and thr() and h() read C, so the residual reads
# only species and can be held. Written with {h} for h().
_CLOCK_SWEEP_NET = """\
begin parameters
    1 k    10.0
    2 A    0.5
    3 tp   30.05
    4 sig  0.001
    5 dlt  0.01
    6 te   30.0
    7 one  1.0
end parameters
begin functions
    1 thr() 1+A*(1+tanh((Cobs-tp)/sig))/2
    2 h() {h}
    3 a() 1/(1+exp((time()-te)/0.002))
    4 f() if(Sobs*h()<thr()*h(),a(),-a())
    5 up() dlt
    6 z() exp(-((time()-30.047)/0.001)^2)
    7 r() if(Sobs*h()<thr()*h(),k,0)*z()
end functions
begin species
    1 S() 27.95
    2 Y() 0
    3 C() 0
end species
begin reactions
    1 0 1 f
    2 0 1 up
    3 0 2 r
    4 0 3 one
end reactions
begin groups
    1 Sobs 1
    2 Cobs 3
end groups
"""


@pytest.mark.parametrize(
    "h",
    ["1", "1e-3+1/(1+exp((Cobs-30.02)/0.002))"],
    ids=["held, crossing restarted at", "dg/dx falls inside the box"],
)
def test_a_threshold_on_a_clock_species(tmp_path, h):
    """With h() = 1 the residual is held through the sweep, and its crossing,
    which CVODE does not locate, is found at the return that ends its step,
    tested, and restarted at: without that the pulse is lost and Y gains 0.
    With h() falling a thousandfold at C = 30.02, C's box at rtol 1e-3 (a
    quarter of a time unit) spans the fall, and a band measured where h was 1
    hid the crossing: Y gained 0 where it gains 0.0149 (third review of the
    #908 fix). dg/dx is now read across the box and that residual is not held.
    Main gets both right (control). C runs at rate 1 from 0, S moves at dlt,
    and the pulse is a Gaussian, so the gain is exact given S before the
    sweep."""
    model = _write(tmp_path, _CLOCK_SWEEP_NET.format(h=h))
    t1, tp, amp, sig, dlt = 30.04, 30.05, 0.5, 0.001, 0.01
    result = bngsim.Simulator(model).run(sample_times=[0.0, t1, tp + 1.0], rtol=1e-3, atol=1e-5)
    s1 = float(result.species[1][0])
    lo, hi = t1, tp + 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        thr = 1.0 + amp * (1.0 + math.tanh((mid - tp) / sig)) / 2.0
        if thr - (s1 + dlt * (mid - t1)) < 0.0:
            lo = mid
        else:
            hi = mid
    t_open = 0.5 * (lo + hi)
    exact = 10.0 * 0.001 * math.sqrt(math.pi) / 2.0 * math.erfc((t_open - 30.047) / 0.001)
    gained = float(result.species[-1][1]) - float(result.species[1][1])
    assert gained == pytest.approx(exact, rel=2e-2)


# S slides on S = 1, parks just above it once a() has decayed, and is pushed
# further up by v(): the residual leaves its band without crossing.
_SLIDE_PARK_PUSH_NET = """\
begin parameters
    1 V    2.0
    2 tp   3.0
    3 sig  0.1
end parameters
begin functions
    1 a() exp(-10*time())
    2 f() if(Sobs<1,a(),-a())
    3 v() V*exp(-((time()-tp)/sig)^2)
end functions
begin species
    1 S() 0.95
end species
begin reactions
    1 0 1 f
    2 0 1 v
end reactions
begin groups
    1 Sobs 1
end groups
"""


def test_a_held_residual_leaving_its_band_is_no_root(tmp_path):
    """Where a held residual's band runs out, CVODE returns a root of what it
    is shown, which is not a crossing and must not be handled as one: taken
    for a root of the residual, it is given the restart test at a point off
    the surface. The run takes every step the run with no state-switch roots
    takes. Main does too (control)."""
    run = dict(sample_times=[0.0, 3.5], rtol=1e-6, atol=1e-8, max_step=0.02)
    result = bngsim.Simulator(_write(tmp_path, _SLIDE_PARK_PUSH_NET)).run(**run)
    reference = bngsim.Simulator(_unrooted(tmp_path, _SLIDE_PARK_PUSH_NET, "u.net")).run(**run)
    assert result.solver_stats["n_steps"] == reference.solver_stats["n_steps"]
    np.testing.assert_array_equal(np.asarray(result.species), np.asarray(reference.species))


# S slides on S = 1 while a() decays; dn() then takes S down and out of its
# band, and much later v() carries it back up across S = 1 in large steps, a
# crossing the restart test carries. Written with {dlt} for the drift, {lam}
# for how fast a() decays, {kd} for K's decay, and {cond} for the condition.
_SLIDE_LEAVE_NET = """\
begin parameters
    1 V    0.2
    2 tp   3.0
    3 sig  0.3
    4 dlt  {dlt}
    5 kd   {kd}
end parameters
begin functions
    1 a() exp(-{lam}*time())
    2 f() if({cond},a(),-a())
    3 v() V*exp(-((time()-tp)/sig)^2)
    4 dn() -dlt
end functions
begin species
    1 S() 0.95
    2 K() 1
end species
begin reactions
    1 0 1 f
    2 0 1 v
    3 0 1 dn
    4 2 0 kd
end reactions
begin groups
    1 Sobs 1
    2 Kobs 2
end groups
"""

_HOLD_CASES = {
    # Held until the next restart instead, the crossing back went unseen in a
    # step nothing else ended, and the run kept a step main restarts in.
    "leaves its band, then crosses": dict(dlt=0.001, lam=10, kd=2, cond="Sobs<1"),
    # S leaves fast. Taken for a root of the residual, the band's end is given
    # the restart test off the surface, which carries it.
    "leaves its band fast": dict(dlt=0.1, lam=10, kd=2, cond="Sobs<1"),
    # dg/dS = K falls by e^-12 while S slides and parks. Measured once, the
    # band stays as wide as it was, and the crossing back lies inside it.
    "dg/dx shrinks while held": dict(dlt=0.001, lam=6, kd=4, cond="Sobs*Kobs<Kobs"),
}


@pytest.mark.parametrize("case", list(_HOLD_CASES))
def test_holding_takes_the_steps_main_takes(tmp_path, monkeypatch, case):
    """Holding a residual changes which crossings CVODE locates, never what
    the run does at one: every crossing main restarts at is still restarted
    at. Each run here takes every step it takes with holding off
    (BNGSIM_STATE_SWITCH_HOLD=0), which roots and tests every crossing as main
    does, and ends on the same bits. Each hold here ends where the state leaves
    its band. One that ends inside a slide, or a held crossing the run
    restarts at, lands within CVODE's root tolerance of main's restart instead,
    and the answers agree to the tolerance rather than to the bit."""
    text = _SLIDE_LEAVE_NET.format(**_HOLD_CASES[case])
    run = dict(sample_times=[0.0, 4.0], rtol=1e-6, atol=1e-8, max_step=0.05)
    result = bngsim.Simulator(_write(tmp_path, text)).run(**run)
    monkeypatch.setenv("BNGSIM_STATE_SWITCH_HOLD", "0")
    reference = bngsim.Simulator(_write(tmp_path, text, "off.net")).run(**run)
    assert result.solver_stats["n_steps"] == reference.solver_stats["n_steps"]
    np.testing.assert_array_equal(np.asarray(result.species), np.asarray(reference.species))


def test_a_held_slide_prints_nothing(tmp_path):
    """A held slide no longer returns at every step, so CVODE's calls end on
    their batch limit, and CVODE logs each end as an error: 132 lines of
    "mxstep steps taken before reaching tout" on this run. A run that holds a
    residual sends that log to the null sink. Main prints nothing (control).
    In a subprocess, for the reason given above."""
    path = tmp_path / "slide.net"
    path.write_text(_SLIDE_NET)
    script = textwrap.dedent(f"""
        import bngsim
        m = bngsim.Model.from_net({str(path)!r})
        r = bngsim.Simulator(m).run(sample_times=[0.0, 4.25], rtol=1e-5, atol=1e-5)
        print("S", float(r.species[-1][0]))
    """)
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert "S 0.99999" in done.stdout
    assert "mxstep" not in done.stdout + done.stderr


def _batch_ends_between(path, hold, start, stop):
    """The times CVODE logs a batch ending at, between ``start`` and ``stop``,
    in a run of ``path`` with holding on or off; in a subprocess, as above.
    A spent batch is logged only under BNGSIM_LOG_SPENT_BATCHES=1 (issue #1124,
    item 13), which leaves SUNDIALS' handler, and the hold's mute, as they were."""
    script = textwrap.dedent(f"""
        import bngsim
        m = bngsim.Model.from_net({str(path)!r})
        bngsim.Simulator(m).run(sample_times=[0.0, 4.0], rtol=1e-6, atol=1e-8,
                                max_step=0.05, max_steps=20)
    """)
    env = dict(os.environ, BNGSIM_STATE_SWITCH_HOLD=hold, BNGSIM_LOG_SPENT_BATCHES="1")
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
    )
    times = re.findall(r"At t = ([0-9.eE+-]+), mxstep", done.stdout + done.stderr)
    return [t for t in times if start < float(t) < stop]


def test_a_released_hold_logs_what_main_logs(tmp_path):
    """While a run holds a residual, CVODE's error log is in the null sink; once
    nothing is held it goes back. Kept in the sink for the rest of the run, the
    batch ends after S has left its band (in 20-step batches, between t = 1 and
    the restart at S's crossing back near t = 2.55) went unlogged, where the
    run with holding off logs them. Where each batch ends depends on where the
    run last returned, so the times are not compared."""
    path = tmp_path / "leave.net"
    path.write_text(_SLIDE_LEAVE_NET.format(**_HOLD_CASES["leaves its band, then crosses"]))
    assert _batch_ends_between(path, "0", 1.0, 2.4)
    assert _batch_ends_between(path, "1", 1.0, 2.4)


# Issue #1064: a slide along S = 1 written over a scale H. Each spelling is the
# same surface at every H.
_SCALED_SLIDE_NET = """\
begin parameters
    1 H {H}
    2 dlt 0.01
end parameters
begin functions
    1 f() if({cond},1,-1)
    2 up() dlt
end functions
begin species
    1 S() 27.95
end species
begin reactions
    1 0 1 f
    2 0 1 up
end reactions
begin groups
    1 Sobs 1
end groups
"""


def _scaled_slide(tmp_path, H, cond, hold):
    """n_steps, S(29.5) to the bit, and what CVODE logged, in a subprocess as
    above."""
    path = tmp_path / f"scaled_{H}.net"
    path.write_text(_SCALED_SLIDE_NET.format(H=H, cond=cond))
    script = textwrap.dedent(f"""
        import bngsim
        m = bngsim.Model.from_net({str(path)!r})
        r = bngsim.Simulator(m).run(sample_times=[0.0, 28.0, 29.5], rtol=1e-3, atol=1e-5)
        print("RUN", r.solver_stats["n_steps"], float(r.species[-1][0]).hex())
    """)
    env = dict(os.environ)
    if hold is None:  # the default, whatever the caller's shell sets
        env.pop("BNGSIM_STATE_SWITCH_HOLD", None)
    else:
        env["BNGSIM_STATE_SWITCH_HOLD"] = hold
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
    )
    run = re.search(r"^RUN (\d+) (\S+)$", done.stdout, re.MULTILINE)
    assert run is not None, done.stdout + done.stderr
    return int(run.group(1)), run.group(2), done.stdout.replace(run.group(0), "") + done.stderr


@pytest.mark.parametrize(
    "cond, hold",
    [("Sobs*H<H", "0"), ("Sobs*H<H+0*time()", None)],
    ids=["every crossing tested", "reads the time"],
)
def test_a_residual_scaled_by_a_constant_steps_alike(tmp_path, cond, hold):
    """A residual that reads exactly 0.0 while a run steps on past its root is
    reported as a tiny value on the side it is held to (#897). That value was
    DBL_MIN. CVODE's root search ranks a sign change by g(tmid) over
    g(tmid) - g(tlo), and against a residual past ~9e15 that ratio underflowed
    to 0: the search took the sign change for none, moved the wrong end of its
    bracket, and asked for times outside the step. At H = 5.18e21 this run took
    51,990 steps where H = 1 takes 55,002, and logged 105 "Illegal value for t"
    errors; at 1e23, 47,424 and 280. The surface is the same at every H, so the
    steps and the answer are too."""
    reference = _scaled_slide(tmp_path, "1", cond, hold)
    assert reference[2].strip() == ""
    for H in ("5.18e21", "1e23", "1e100", "1e-20"):
        steps, bits, log = _scaled_slide(tmp_path, H, cond, hold)
        assert "Illegal value for t" not in log, H
        assert (steps, bits) == reference[:2], H
