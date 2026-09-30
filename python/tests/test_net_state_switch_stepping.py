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
  6. a root restarts the run only where the flow clearly carries the residual
     across, or where the residual lies exactly on its surface: GH #176's
     parked trajectory, and two twins whose flow near the park cannot be
     resolved, keep their exact answers; a residual parked at exactly 0.0 is
     not read as a second root; one moving along its surface does not chatter,
     and one resting on it at a fixed point is not restarted onto it; and a
     real crossing on a curved residual still restarts;
  7. a residual too deep for ``ast.unparse`` keeps every root, and one that
     starts at exactly zero does not print a SUNDIALS warning.
"""

import math
import subprocess
import sys
import textwrap

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


def _ltype_variant(text, kind):
    """GH #176's fixture, or one of two twins that park the same way but whose
    residual's flow near the park is a near-cancel no difference can resolve.

    ``split`` carries Voltage_Level as V + V2 exchanging at kx*time() and kx, so
    dg/dt is f_V + f_V2, two terms of about 0.3 that cancel to 1e-11. ``ramp``
    moves the threshold at rr = 1e-8 per unit time and feeds V the same ramp, so
    dg/dt = f_V - rr, and the rounding in d/dt of rr*time() is 1e-10 whatever
    rr is. Each still parks 1e-11 on the exact trajectory's side.
    """

    def swap(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    last_param = "   19 _rateLaw2       0.01  # Constant\n"
    if kind == "split":
        swap(last_param, last_param + "   20 kx 0.01\n")
        swap(f"    4 v_rec() {_V_STEP}\n", f"    4 v_rec() {_V_STEP}\n    5 xf() kx*time()\n")
        last_species = "   10 LTCC(b,g~C,loc~mem,p~P,s~I) 0\n"
        swap(last_species, last_species + "   11 Voltage2() 0\n")
        last_rxn = "   25 10 8 _rateLaw2 #_R8\n"
        swap(last_rxn, last_rxn + "   26 3 11 xf\n   27 11 3 kx\n   28 11 0 k_v_leak\n")
        swap("    8 Voltage_Level        3\n", "    8 Voltage_Level        3,11\n")
    elif kind == "ramp":
        swap(last_param, last_param + "   20 rr 1e-08\n")
        swap(
            f"    4 v_rec() {_V_STEP}\n",
            f"    4 v_rec() {_V_STEP_RAMP}\n    5 stim() k_v_stim+rr*(time()+1)\n",
        )
        swap("    4 0 3 k_v_stim #_R9\n", "    4 0 3 stim #_R9\n")
    return text


@pytest.mark.parametrize("kind", ["fixture", "split", "ramp"])
def test_a_parked_root_does_not_restart_the_run(data_dir, tmp_path, kind):
    """GH #176's fixture parks Voltage 1e-11 below the step at 50, far inside
    the solver's own error at rtol 1e-8, so the interpolant can cross the step
    while the vector field points back: dV/dt = 49.99999999999 - V < 0 there.
    Restarting on that surface held the run on the branch the exact trajectory
    never takes. The observables came out 63% off after 2,019,910 steps, with no
    error.

    The core now restarts at a lone state-switch root only where the flow
    clearly carries the residual across, so a parked root steps on as it did
    before the root existed: the analytical attempt fails loudly and ``auto``
    retries with FD. "Clearly" is the point of the two twins. With a restart
    whenever the flow could not be read as opposed, their near-cancelling flows
    restarted on the surface and came out 9.4e-3 (split) and 0.62 (ramp) off,
    worse at tighter tolerances, while main was right (review of PR #903).

    The oracle is the same network with each step replaced by the branch the
    exact trajectory takes throughout: V < 50 always, so 0.5, and
    Phospho_LTCC > 0 for every t > 0.
    """
    text = _ltype_variant((data_dir / _LTYPE).read_text(), kind)
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

    exact_text = text
    step = _V_STEP_RAMP if kind == "ramp" else _V_STEP
    for condition, branch in (
        (step, "0.5"),
        ("if((Phospho_LTCC>0),k_pka_shift,0)", "k_pka_shift"),
    ):
        assert exact_text.count(condition) == 1, condition
        exact_text = exact_text.replace(condition, branch)
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
    The root finder, though, interpolates V onto 50.0 itself, where every term
    of the residual's flow is zero. Restarting there, as a residual lying on
    its surface does, held V on 50.0 and on the far branch for the rest of the
    run: 0.633 off at every tolerance (review of PR #903). Nothing the residual
    reads is moving, so this is a state resting on the surface, and it steps on.
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


def test_a_residual_lying_on_its_surface_restarts(data_dir):
    """ml_q_learning's Q_Right and Q_Left meet exactly with identical
    derivatives, so roots of its argmax switch land where the residual is
    exactly zero and neither difference moves it. That state lies on the
    surface, not beside it, and needs the restart that lets CVODE set the root
    aside. Stepping on under the zero hold kept it armed, so each ~6e-11 wobble
    across the surface was another root: about 140,000 of them, 119,478 steps,
    and a trajectory 4.9 off at rtol 1e-11. The reference is the same model with
    no state-switch roots at all, which is how main integrates it."""
    path = str(data_dir / "qlearning_flat_residual.net")
    model = bngsim.Model.from_net(path)
    assert "(Q_Right-off)>(Q_Left-off)" in model.state_switch_root_conditions()
    window = dict(t_span=(0.0, 100.0), n_points=1001, rtol=1e-11, atol=1e-11)
    result = bngsim.Simulator(model).run(**window)
    assert result.solver_stats["n_steps"] < 5_000
    unrooted = bngsim.Model.from_net(path)
    unrooted._state_switch_root_conditions = ()
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
