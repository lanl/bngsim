"""``steady_state(criterion="scaled")``, the default (issue #1001).

``||f(y)||_2 / n < tol`` is a number in the model's units of concentration per
time: a model in small units is under it where it starts, or one step later,
and that state came back ``converged=True``. The default asks that test and
that a run taken on from the state does not move it beyond each species' own
tolerance; the Newton step from the state to the root, in the same
tolerances, says where the march stops. These tests cover:

- the issue's cubic at scales 1e-9 and 1e-12 ends at its stable root, by both
  methods, from either side of it, beside large or ``$``-fixed species;
- the residual test is kept by name, and a run moves what it accepted;
- the reported step is the Newton step NumPy takes on the reduced system,
  with laws, masks and singular Jacobians (singular by rounding too);
- the run: a drift, a slow drift, an epidemic that burns out, a long
  ``max_time``, a tight ``rtol``, its own error, and states at rest that a
  run kicked off them would have refused;
- the march: one that stalls short of its root is polished, one that stops
  where the step cannot be had returns the state the residual test passed,
  above 512 unknowns too;
- the polish is kept only where it is nearer the root;
- the march's ``atol``: one the caller gives, "auto" and its floor, $-fixed
  species, batches;
- an unstable root that a run does not leave is converged, as under the
  residual test.
"""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _codegen_backend import threshold_forced  # noqa: E402

# X' = -k·x·(x - a)·(x - u), x = X/s: roots 0 and u = 1 stable, a = 0.3 not.
CUBIC_NET = """begin parameters
    1 k   1.0
    2 a   0.3
    3 u   1.0
    4 s   {s!r}
    5 X0  {x0!r}*s
    6 k3  k/(s*s)
    7 k2  k*(u+a)/s
    8 k1  k*a*u
end parameters
begin species
    1 X() X0
end species
begin reactions
    1 1,1,1 1,1 k3
    2 1,1 1,1,1 k2
    3 1 0 k1
end reactions
"""


def _cubic(tmp_path, s, x0):
    p = tmp_path / f"cubic_{s!r}_{x0!r}.net"
    p.write_text(CUBIC_NET.format(s=s, x0=x0))
    return bngsim.Model.from_net(str(p))


@pytest.mark.parametrize("s", [1e-9, 1e-12])
@pytest.mark.parametrize("x0", [0.5, 0.9, 1.5])
@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize("by_name", [False, True], ids=["default", "named"])
def test_scaled_reaches_the_stable_root_at_any_scale(tmp_path, s, x0, method, by_name):
    kw = {"criterion": "scaled"} if by_name else {}
    ss = bngsim.Simulator(_cubic(tmp_path, s, x0), method="ode").steady_state(method=method, **kw)
    assert ss.converged
    assert ss.criterion == "scaled"
    assert ss.concentrations[0] / s == pytest.approx(1.0, rel=1e-6)
    assert ss.newton_distance <= 1.0
    assert ss.continued_run_distance <= 1.0


def test_the_residual_test_is_kept_by_name(tmp_path):
    # The parity test, unchanged: it takes no Newton step (a Jacobian more per
    # solve) and no run on, so it reports neither.
    at_one = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.5), method="ode").steady_state(
        criterion="residual"
    )
    assert at_one.criterion == "residual"
    assert at_one.converged
    assert at_one.concentrations[0] == pytest.approx(1.0, rel=1e-6)
    assert math.isnan(at_one.newton_distance)
    assert math.isnan(at_one.continued_run_distance)


# The issue's table: the residual test at a scale of 1e-9 accepts the cubic one
# step from each start. Pinned so a change to the residual path shows.
RESIDUAL_AT_1E_9 = {
    ("integration", 0.5): 0.893865,
    ("integration", 0.9): 1.003209,
    ("integration", 1.5): 0.130999,
    ("newton", 0.5): 0.3,
    ("newton", 0.9): 1.038462,
    ("newton", 1.5): 1.214286,
}


@pytest.mark.parametrize("method,x0", sorted(RESIDUAL_AT_1E_9))
def test_the_residual_solve_is_the_one_main_took(tmp_path, method, x0):
    residual = bngsim.Simulator(_cubic(tmp_path, 1e-9, x0), method="ode").steady_state(
        method=method, criterion="residual"
    )
    assert residual.criterion == "residual"
    assert residual.converged and residual.n_steps == 1
    assert residual.concentrations[0] / 1e-9 == pytest.approx(
        RESIDUAL_AT_1E_9[(method, x0)], rel=1e-5
    )


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_scaled_with_a_conservation_law(data_dir, method):
    path = str(data_dir / "two_species_reversible.net")
    ref = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(
        method=method, tol=1e-14
    )
    ss = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(
        method=method, criterion="scaled"
    )
    assert ss.converged
    assert ss.newton_distance <= 1.0
    for got, want in zip(ss.concentrations, ref.concentrations, strict=False):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-12)


def test_a_given_atol_is_the_one_the_march_takes(tmp_path):
    # At the scalar 1e-8 the march runs a species of 1e-9 through zero; the
    # criterion then does not call where it stops converged.
    ss = bngsim.Simulator(_cubic(tmp_path, 1e-9, 0.5), method="ode").steady_state(
        criterion="scaled", atol=1e-8
    )
    assert not ss.converged
    assert ss.newton_distance > 1.0


def test_batch_passes_the_criterion_on(tmp_path):
    sim = bngsim.Simulator(_cubic(tmp_path, 1e-9, 0.5), method="ode")
    out = sim.steady_state_batch([{}, {"a": 0.2}], criterion="scaled")
    for ss in out:
        assert ss.converged
        assert ss.criterion == "scaled"
        assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


def test_an_unknown_criterion_is_refused(tmp_path):
    sim = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.5), method="ode")
    with pytest.raises((ValueError, bngsim.SimulationError), match="criterion"):
        sim.steady_state(criterion="relative")


# A + B <-> C, C -> D: two laws (A + C + D, B + C + D), so one unknown and three
# species that follow from it, and D a sink a mask can leave out.
BINDING_NET = """begin parameters
    1 kf 1.0
    2 kr 0.01
end parameters
begin species
    1 A() {a0!r}
    2 B() {b0!r}
    3 C() {c0!r}
    4 D() 0
end species
begin reactions
    1 1,2 3 kf
    2 3 1,2 kr
    3 3 4 kr
end reactions
"""


def _numpy_newton_distance(model, y, rtol, mask=None):
    """The step the criterion measures, by NumPy, and without choosing which
    species each law is solved for. Every species moves by its own equation
    but a $-fixed one and a masked one nothing reads (an empty column of the
    Jacobian's pattern), which is held; the laws are cut down to the ones no
    such species is in; the step is solved on all of it, with a rank check.
    Its size is taken over each kept species' rtol·|y| + min(atol_i,
    rtol·largest): atol_i the march's, the smaller of what atol="auto" derives
    from the start y0 and 1e-8, and `largest` the largest species of y0, y and
    y + δ."""
    ns = len(y)
    cl = model.conservation_laws
    laws = np.asarray(cl["coefficients"], float).reshape(cl["n_laws"], ns)
    fixed = np.asarray(model._core.species_is_fixed, bool)
    kept = (np.ones(ns, bool) if mask is None else np.asarray(mask, bool)) & ~fixed
    pattern = model.jacobian(np.asarray(model.get_state(), float), sparse=True).tocsc()
    unread = ~kept & ~fixed & (np.diff(pattern.indptr) == 0)
    if unread.any() and len(laws):
        u, sv, _ = np.linalg.svd(laws[:, unread])
        laws = u[:, int(np.sum(sv > 1e-12 * sv[0])) :].T @ laws
    moves = ~fixed & ~unread
    f = np.asarray(model.rhs(y), float)
    jac = np.asarray(model.jacobian(y), float)
    lhs = np.vstack([laws[:, moves], jac[moves][:, moves]])
    rhs = np.concatenate([np.zeros(len(laws)), -f[moves]])
    step, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    assert rank == moves.sum()
    assert np.allclose(lhs @ step, rhs, rtol=0, atol=1e-9 * np.max(np.abs(rhs)))
    delta = np.zeros(ns)
    delta[moves] = step
    y0 = np.asarray(model.get_state(), float)
    a0 = np.abs(y0)
    floor = a0[a0 > 1e-14 * a0.max()].min()
    atol = np.minimum(rtol * np.maximum(a0, floor), 1e-8)
    largest = max(a0[~fixed].max(), np.abs(y[~fixed]).max(), np.abs((y + delta)[~fixed]).max())
    scale = rtol * np.abs(y) + np.minimum(atol, rtol * largest)
    return float(np.max(np.abs(delta[kept]) / scale[kept]))


@pytest.mark.parametrize("a0,b0,c0", [(1.0, 1e-3, 0.0), (1e-9, 1e-12, 1e-10), (5.0, 2.0, 0.5)])
@pytest.mark.parametrize("mask", [None, [True, True, True, False]], ids=["all", "sink-out"])
def test_the_reported_step_is_the_newton_step(tmp_path, a0, b0, c0, mask):
    # Stopped short of the root (max_time 0.5), where the step is large and
    # every species' share of it shows.
    path = tmp_path / "binding.net"
    path.write_text(BINDING_NET.format(a0=a0, b0=b0, c0=c0))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", max_time=0.5, mask=mask
    )
    assert not ss.converged
    want = _numpy_newton_distance(
        bngsim.Model.from_net(str(path)), np.asarray(ss.concentrations), 1e-8, mask
    )
    assert ss.newton_distance == pytest.approx(want, rel=1e-6)


# 0 -> X at ksyn, X -> 0 at 1: X starts at 0 and ends at ksyn.
SYNTHESIS_NET = """begin parameters
    1 ksyn {ksyn!r}
    2 kdeg 1.0
end parameters
begin species
    1 X() 0
end species
begin reactions
    1 0 1 ksyn
    2 1 0 kdeg
end reactions
"""


@pytest.mark.parametrize("ksyn", [1e-9, 1e-12, 1e3])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_empty_start_is_held_to_its_own_size(tmp_path, ksyn, method):
    # With nothing above 0 at the start the floor was 1 in the model's units, a
    # tolerance of 1e-8 against a root of 1e-9: accepted at x = 0.69e-9 after
    # one step.
    path = tmp_path / "synthesis.net"
    path.write_text(SYNTHESIS_NET.format(ksyn=ksyn))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method
    )
    assert ss.converged
    assert ss.concentrations[0] / ksyn == pytest.approx(1.0, rel=1e-6)
    assert ss.newton_distance <= 1.0


# E + S -> E + P with no E: nothing moves, and the reduced Jacobian, k·E, is 0.
CATALYST_NET = """begin parameters
    1 k 1.0
end parameters
begin species
    1 E() 0
    2 S() 1e-9
    3 P() 0
end species
begin reactions
    1 1,2 1,3 k
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_root_with_a_singular_jacobian_is_converged(tmp_path, method):
    path = tmp_path / "catalyst.net"
    path.write_text(CATALYST_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method
    )
    assert ss.converged
    assert ss.newton_distance == 0.0
    assert list(ss.concentrations) == [0.0, 1e-9, 0.0]


def test_set_tolerances_without_atol_leaves_the_march_its_own(tmp_path):
    # set_tolerances(rtol=...) gave no atol, but it set the scalar 1e-8, which
    # took the march off "auto" and ran X through zero (x = -21.8).
    sim = bngsim.Simulator(_cubic(tmp_path, 1e-9, 0.5), method="ode")
    sim.set_tolerances(rtol=1e-8)
    ss = sim.steady_state(criterion="scaled")
    assert ss.converged
    assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


def test_the_distance_is_measured_on_the_retried_jacobian(data_dir):
    # The closed form stops this march and the solve is retried on
    # differences; measured on the closed form, the distance of the root the
    # retry accepted read inf beside converged=True.
    ss = bngsim.Simulator(
        bngsim.Model.from_net(str(data_dir / "nnxor_parked_residual.net")), method="ode"
    ).steady_state(criterion="scaled", method="newton")
    assert ss.solver_jacobian_retried
    assert ss.converged
    assert ss.newton_distance <= 1.0


def _wide_net(copies):
    # `copies` pairs A <-> B, fed and drained: 2·copies unknowns, no laws.
    species = []
    reactions = []
    for i in range(copies):
        a, b = 2 * i + 1, 2 * i + 2
        species += [f"    {a} A{i}() {1.0 + 0.01 * i!r}", f"    {b} B{i}() 0"]
        reactions += [
            f"    {4 * i + 1} {a} {b} kf",
            f"    {4 * i + 2} {b} {a} kf",
            f"    {4 * i + 3} 0 {a} ks",
            f"    {4 * i + 4} {b} 0 ks",
        ]
    return "\n".join(
        ["begin parameters", "    1 kf 1000.0", "    2 ks 1.0", "end parameters"]
        + ["begin species", *species, "end species"]
        + ["begin reactions", *reactions, "end reactions", ""]
    )


def test_above_512_unknowns_the_solve_is_the_residual_one(tmp_path):
    # The step is not taken there, and the solve stops where the residual one
    # does: with KINSOL's stopping rule left at the scaled one, the polish ran
    # on past the residual's answer (BaruaBCR_2012 was never accepted, 18 s
    # against 2 s). That answer is polished, to a thousandth of tol, and a run
    # taken on from it decides.
    path = tmp_path / "wide.net"
    path.write_text(_wide_net(260))
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method="newton", atol=1e-10)
    residual, scaled = out["residual"], out["scaled"]
    assert scaled.converged and scaled.method_used == "newton"
    assert scaled.criterion == "scaled"
    assert math.isnan(scaled.newton_distance)
    assert scaled.continued_run_distance <= 1.0
    assert scaled.residual < 1e-12
    np.testing.assert_allclose(scaled.concentrations, residual.concentrations, rtol=1e-8)


@pytest.mark.parametrize("name", ["BIOMD0000000252", "BIOMD0000000685"])
def test_a_march_that_stalls_short_of_its_root_is_polished(data_dir, name):
    # BIOMD0000000252's march stalls at step 5411 at a residual of 2.1e-7, a
    # step of 1.64 tolerances from its root, and BIOMD0000000685's 1.62 from
    # its, where main had stopped under tol at states no run moves. A short
    # Newton polish from there is kept where it passes the parity test and
    # lands where the step said.
    path = str(data_dir / f"{name}.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state()
    assert ss.converged
    assert ss.newton_distance <= 1.0
    assert ss.continued_run_distance <= 1.0
    run = bngsim.Model.from_sbml(path)
    bngsim.Simulator(run, method="ode").run(
        t_span=(0.0, 1e7), n_points=2, rtol=1e-11, atol=1e-16, max_steps=2_000_000
    )
    end = np.asarray(run._core.get_state(), float)
    y = np.asarray(ss.concentrations)
    np.testing.assert_allclose(y, end, rtol=1e-6, atol=1e-9 * np.abs(end).max())


@pytest.mark.parametrize("s", [1.0, 1e-9])
def test_burst_free_starts_find_the_same_roots_at_any_scale(tmp_path, s):
    # The solve steady_states() runs from each start (issue #524). At 1e-9 the
    # residual test calls most starts roots where they stand; the scaled one
    # finds the cubic's three roots, as at 1. The one at 0 is reached as
    # -5.9e-24, which the negativity check, relative to the largest species
    # and so to itself, threw out until it was given the species' tolerance.
    sim = bngsim.Simulator(_cubic(tmp_path, s, 0.5), method="ode")
    starts = (np.linspace(0.01, 1.6, 40) * s)[:, None]
    out = sim.steady_state_batch(
        states=starts, method="newton", newton_burst=False, criterion="scaled"
    )
    converged = [r for r in out if r.converged]
    assert all(r.newton_distance <= 1.0 for r in converged)
    roots = sorted({round(r.concentrations[0] / s, 6) + 0.0 for r in converged})
    assert roots == [0.0, 0.3, 1.0]


# A -> S, S a write-only sink, one law A + S. Listed A first, A is the law's
# dependent species, which the pure-sink mask keeps while it leaves S out.
DECAY_NETS = {
    "A first": """begin parameters
    1 k 1.0
end parameters
begin species
    1 A() 1.0
    2 S() 0
end species
begin reactions
    1 1 2 k
end reactions
""",
    "S first": """begin parameters
    1 k 1.0
end parameters
begin species
    1 S() 0
    2 A() 1.0
end species
begin reactions
    1 2 1 k
end reactions
""",
}


@pytest.mark.parametrize("order", sorted(DECAY_NETS))
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_masked_sink_takes_up_its_law(tmp_path, order, method):
    # The solvers hold S where integration left it; with A the law's dependent,
    # nothing moved A, and A first was accepted where it started, [1, 0].
    path = tmp_path / "decay.net"
    path.write_text(DECAY_NETS[order])
    model = bngsim.Model.from_net(str(path))
    a = model.species_names.index("A()")
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", method=method, mask=~model.is_pure_sink()
    )
    assert ss.converged
    assert abs(ss.concentrations[a]) < 1e-8
    assert ss.newton_distance <= 1.0


@pytest.mark.parametrize("order", sorted(DECAY_NETS))
def test_a_masked_law_member_follows_the_step(tmp_path, order):
    # Short of the root, the step is A's own: f = -k·A, so δ_A = -A exactly,
    # held to rtol·A + rtol·max(A, 1).
    path = tmp_path / "decay.net"
    path.write_text(DECAY_NETS[order])
    model = bngsim.Model.from_net(str(path))
    a = model.species_names.index("A()")
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", max_time=0.5, mask=~model.is_pure_sink()
    )
    assert not ss.converged
    y = ss.concentrations[a]
    assert ss.newton_distance == pytest.approx(y / (1e-8 * y + 1e-8 * max(y, 1.0)), rel=1e-6)


# A <-> B, one law; a mask that keeps only A, the law's dependent.
PAIR_NET = """begin parameters
    1 kf 1.0
    2 kr 0.5
end parameters
begin species
    1 A() 1.0
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_kept_dependent_whose_partner_is_masked_is_stepped(tmp_path, method):
    path = tmp_path / "pair.net"
    path.write_text(PAIR_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method, mask=[True, False]
    )
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(1.0 / 3.0, rel=1e-6)


# X' = if(X < 1, 0.1, 100): positive everywhere, no steady state.
JUMP_NET = """begin parameters
    1 klo 0.1
    2 khi 100
    3 x0 {x0!r}
end parameters
begin species
    1 X() x0
end species
begin reactions
    1 0 1 fr
end reactions
begin groups
    1 Xobs 1
end groups
begin functions
    1 fr() if(Xobs<1,klo,khi)
end functions
"""


@pytest.mark.parametrize("jacobian", ["fd", "auto"])
def test_a_differenced_jacobian_across_a_jump_certifies_nothing(tmp_path, jacobian):
    # The march stops on the switch, and one-sided across it the quotient is
    # the jump over the step: 20 of these 60 starts came back converged.
    for i, x0 in enumerate(np.linspace(0.0, 0.999, 60)):
        path = tmp_path / f"jump_{i}.net"
        path.write_text(JUMP_NET.format(x0=float(x0)))
        ss = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode", jacobian=jacobian
        ).steady_state(criterion="scaled", max_time=100.0)
        assert not ss.converged, x0


@pytest.mark.parametrize("s", [1.0, 1e-9])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_differenced_jacobian_still_certifies_a_smooth_root(tmp_path, s, method):
    ss = bngsim.Simulator(_cubic(tmp_path, s, 0.5), method="ode", jacobian="fd").steady_state(
        criterion="scaled", method=method
    )
    assert ss.converged
    assert ss.concentrations[0] / s == pytest.approx(1.0, rel=1e-6)


def _ltype(tmp_path, data_dir, k_v_stim=None):
    source = (data_dir / "ltype_calcium_discontinuous_jacobian.net").read_text()
    if k_v_stim is not None:
        source = re.sub(r"(16 k_v_stim\s+)\S+", rf"\g<1>{k_v_stim}", source)
    path = tmp_path / f"ltype_{k_v_stim}.net"
    path.write_text(source)
    return bngsim.Model.from_net(str(path))


def test_the_retried_march_past_a_switch_converges_away_from_it(tmp_path, data_dir):
    # GH #176's model: its closed-form Jacobian stops the march, the retry
    # differences across the switch on the way. Voltage relaxes to
    # k_v_stim/k_v_leak; at 45 it rests 5 below the switch at 50.
    ss = bngsim.Simulator(_ltype(tmp_path, data_dir, "45"), method="ode").steady_state()
    assert ss.converged
    assert ss.newton_distance <= 1.0
    assert ss.continued_run_distance <= 1.0


# The synthesis of SYNTHESIS_NET written the BNGL way, from a $-fixed source.
SOURCE_NET = """begin parameters
    1 ksyn {ksyn!r}
    2 kdeg 1.0
end parameters
begin species
    1 $Src() 1
    2 X() 0
end species
begin reactions
    1 1 1,2 ksyn
    2 2 0 kdeg
end reactions
"""


@pytest.mark.parametrize("ksyn", [1e-9, 1e-12])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_fixed_species_does_not_set_the_floor(tmp_path, ksyn, method):
    # $Src = 1 held X to 1e-8 against a root of 1e-9: 0.73 of it after a step.
    path = tmp_path / "source.net"
    path.write_text(SOURCE_NET.format(ksyn=ksyn))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method
    )
    assert ss.converged
    assert ss.concentrations[1] / ksyn == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_batch_entry_marches_at_its_own_auto_atol(tmp_path, method):
    # Taken from the model at s = 1, the "auto" ran the entry at s = 1e-9
    # through zero (x = -3.7).
    sim = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.5), method="ode")
    for out in (
        sim.steady_state_batch([{"s": 1e-9}], criterion="scaled", method=method)[0],
        sim.steady_state_batch(
            params=[{"s": 1e-9}], states=[[0.5e-9]], criterion="scaled", method=method
        )[0],
    ):
        assert out.converged
        assert out.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


# A -> S1 at 1, A + S1 -> S2 + S1 at 1e9 (S1 speeds A's loss to S2); one law
# A + S1 + S2, and only A kept. A's one steady state is 0.
CATALYZED_NET = """begin parameters
    1 k1 1.0
    2 k3 1e9
end parameters
begin species
{species}
end species
begin reactions
    1 {a} {s1} k1
    2 {a},{s1} {s2},{s1} k3
end reactions
"""


@pytest.mark.parametrize("order", [("A", "S1", "S2"), ("A", "S2", "S1")])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_two_masked_members_of_a_law_do_not_depend_on_species_order(tmp_path, order, method):
    # Holding one masked member and moving the other made the step depend on
    # which: listed A, S1, S2 the state was accepted where it started, A = 1.
    start = {"A": 1.0, "S1": 0.0, "S2": 0.0}
    index = {name: i + 1 for i, name in enumerate(order)}
    path = tmp_path / "catalyzed.net"
    path.write_text(
        CATALYZED_NET.format(
            species="\n".join(f"    {index[n]} {n}() {start[n]!r}" for n in order),
            a=index["A"],
            s1=index["S1"],
            s2=index["S2"],
        )
    )
    mask = [name == "A" for name in order]
    a = order.index("A")
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    early = sim.steady_state(criterion="scaled", method=method, mask=mask, max_time=1e-12)
    assert not early.converged
    ss = sim.steady_state(criterion="scaled", method=method, mask=mask)
    assert ss.converged
    assert abs(ss.concentrations[a]) < 1e-8


# A -> B at 1 and A -> C at 2: one law A + B + C, B and C write-only sinks.
BRANCH_NETS = {
    "A first": """begin parameters
    1 k1 1.0
    2 k2 2.0
end parameters
begin species
    1 A() 1.0
    2 B() 0
    3 C() 0
end species
begin reactions
    1 1 2 k1
    2 1 3 k2
end reactions
""",
    "A last": """begin parameters
    1 k1 1.0
    2 k2 2.0
end parameters
begin species
    1 B() 0
    2 C() 0
    3 A() 1.0
end species
begin reactions
    1 3 1 k1
    2 3 2 k2
end reactions
""",
}


@pytest.mark.parametrize("order", sorted(BRANCH_NETS))
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_two_sinks_of_one_law_converge(tmp_path, order, method):
    # Moving both sinks left no equation for how A's loss splits between them,
    # and the matrix was singular at every state: never converged.
    path = tmp_path / "branch.net"
    path.write_text(BRANCH_NETS[order])
    model = bngsim.Model.from_net(str(path))
    a = model.species_names.index("A()")
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", method=method, mask=~model.is_pure_sink()
    )
    assert ss.converged
    assert abs(ss.concentrations[a]) < 1e-8


# A + B <-> C, C -> D: two laws as found, D (a sink) in both; C drains.
DRAIN_NET = """begin parameters
    1 kf 2.0
    2 kr 0.3
    3 kd 0.7
end parameters
begin species
    1 A() 1.0
    2 B() 0.5
    3 C() 0
    4 D() 0
end species
begin reactions
    1 1,2 3 kf
    2 3 1,2 kr
    3 3 4 kd
end reactions
"""


@pytest.mark.parametrize("keep", ["AB", "ABC", "A"])
def test_a_masked_species_that_feeds_kept_ones_is_stepped(tmp_path, keep):
    # Holding C, as the solvers hold a masked species, stepped A and B as if
    # C did not drain; dropping both laws D is in lost B + C - A.
    path = tmp_path / "drain.net"
    path.write_text(DRAIN_NET)
    model = bngsim.Model.from_net(str(path))
    mask = [name[0] in keep for name in model.species_names]
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", max_time=2.0, mask=mask
    )
    assert not ss.converged
    want = _numpy_newton_distance(
        bngsim.Model.from_net(str(path)), np.asarray(ss.concentrations), 1e-8, mask
    )
    assert ss.newton_distance == pytest.approx(want, rel=1e-6)


# X' = if(X < c, klo, 1e3): positive everywhere. Y, settled at 1 beside it,
# makes the difference-quotient step 1.5e-8, past X's switch.
NEAR_ZERO_JUMP_NET = """begin parameters
    1 x0 {x0!r}
    2 one 1.0
end parameters
begin species
    1 X() x0
    2 Y() 1.0
end species
begin reactions
    1 0 1 fr
    2 0 2 one
    3 2 0 one
end reactions
begin groups
    1 Xobs 1
end groups
begin functions
    1 fr() if(Xobs<{c!r},{klo!r},1e3)
end functions
"""


@pytest.mark.parametrize(
    "c,klo,x0",
    [
        (1e-9, 1e-6, 0.0),
        (1e-9, 1e-6, 2.5e-10),
        (1e-9, 1e-6, 9e-10),
        (1e-12, 1e-6, 0.0),
    ],
)
@pytest.mark.parametrize("jacobian", ["fd", "auto"])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_jump_within_a_step_above_zero_certifies_nothing(tmp_path, c, klo, x0, jacobian, method):
    # The backward probe would go below 0. Keeping the forward quotient let the
    # jump through; the secant to 0 reads the law's own slope, 0, and where X
    # is too small for that secant to be read (9e-15, 3e-41), a forward quotient
    # at half the step grows across the jump.
    # The step steers where the solve stops and decides nothing; the parity
    # test and the runs decide, over max_time. (At c = 1e-20, klo = 1e-30 the
    # drift is too slow to show in it: X moves by 1e-24 in 1e6, within its
    # tolerance of 2.2e-24, and the state passes.) On differenced columns a
    # march of 1 from 9e-10 ends at X = -86, where X' is 1e-6 and its
    # tolerance 1.7e-6.
    path = tmp_path / "near_zero.net"
    path.write_text(NEAR_ZERO_JUMP_NET.format(x0=x0, c=c, klo=klo))
    short = not (jacobian == "fd" and x0 == 9e-10)
    for max_time in (1.0, 1e6) if short else (1e6,):
        ss = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode", jacobian=jacobian
        ).steady_state(criterion="scaled", method=method, max_time=max_time)
        assert not ss.converged, max_time


# A in a 1 mL medium taken up into a 1 fL cell: A -> S, S a write-only sink.
# The law weighs S by the volume ratio, A + 1e-12·S.
UPTAKE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="uptake">
    <listOfCompartments>
      <compartment id="medium" size="{medium!r}" constant="true" spatialDimensions="3"/>
      <compartment id="cell" size="1e-15" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="medium" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S" compartment="cell" initialConcentration="0"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="uptake" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci><ci>medium</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@pytest.mark.parametrize("medium", [1e-3, 1.0])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_sink_in_a_small_compartment_takes_up_its_law(medium, method):
    # Its coefficient, 1e-12 of A's (1e-15 at 1 L), read as rounding: the law
    # kept A for its dependent, and A was accepted where it started.
    model = bngsim.Model.from_sbml_string(UPTAKE_SBML.format(medium=medium))
    ss = bngsim.Simulator(model, method="ode").steady_state(
        criterion="scaled", method=method, mask=~model.is_pure_sink()
    )
    assert ss.converged
    assert abs(ss.concentrations[0]) < 1e-8


# A -> B with a rate that reads a counter t, run at 1 from 0: t has no root.
COUNTER_NET = """begin parameters
    1 k1  0.1
    2 one 1.0
end parameters
begin species
    1 A() 100
    2 B() 0
    3 t() 0
end species
begin functions
    1 _rateLaw1() k1*(1+0*t)
end functions
begin reactions
    1 1 2 _rateLaw1
    2 0 3 one
end reactions
begin groups
    1 A_tot  1
    2 B_tot  2
    3 t      3
end groups
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_masked_counter_is_held(tmp_path, method):
    # Its row reads no state, so stepped it made the matrix singular at every
    # state, and A never converged.
    path = tmp_path / "counter.net"
    path.write_text(COUNTER_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        criterion="scaled", method=method, mask=[True, False, False], max_time=2000.0
    )
    assert ss.converged
    assert ss.concentrations[0] < 1e-6


_B358 = (
    Path(__file__).resolve().parents[2]
    / "parity_checks/rr_parity/models/BIOMD0000000358/BIOMD0000000358_url.xml"
)


@pytest.mark.skipif(
    not _B358.exists(), reason="rr_parity corpus model BIOMD0000000358 not present"
)
@pytest.mark.parametrize("jacobian", ["fd", "auto"])
def test_a_root_that_is_not_isolated_reads_singular_on_every_platform(jacobian):
    # IIa_alpha2M and IIa_ATIII are fed by IIa alone, which is 0 at the root,
    # so their columns of the reduced Jacobian are one column and any split of
    # the two is a root. Its zero pivot was exact on x86-64 and -7.9e-43 on
    # arm64, where a step was read off it. No step is had on either; the run
    # from the state, which it does not leave, decides.
    ss = bngsim.Simulator(
        bngsim.Model.from_sbml(str(_B358)), method="ode", jacobian=jacobian
    ).steady_state(criterion="scaled")
    assert ss.converged
    assert ss.newton_distance == math.inf


# ─── The second test: runs taken on from the state (issue #1001) ─────────────


@pytest.mark.parametrize("s", [1.0, 1e-9, 1e-12])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_unstable_root_a_run_does_not_leave_is_converged(tmp_path, s, method):
    # Control: main converges. x = 0.3 is a root in floating point, and a run
    # from it stays: what the default asks, it passes. The march comes to rest
    # on it only where it starts there, and the Newton path labels it.
    ss = bngsim.Simulator(_cubic(tmp_path, s, 0.3), method="ode").steady_state(method=method)
    assert ss.converged
    assert ss.concentrations[0] / s == pytest.approx(0.3, rel=1e-9)
    if method == "newton":
        assert ss.root_stability == "unstable"


def test_a_burst_free_newton_solve_keeps_the_first_test(tmp_path):
    # steady_states() enumerates roots, unstable ones with them, and says which
    # are stable: no run is taken on.
    sim = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.3), method="ode")
    [ss] = sim.steady_state_batch(states=[[0.3]], method="newton", newton_burst=False)
    assert ss.converged
    assert ss.root_stability == "unstable"
    assert math.isnan(ss.continued_run_distance)


# X' = 0.001 + 1e8·|X - 1|: positive everywhere, with a kink at 1.
KINK_NET = """begin parameters
    1 x0 0.5
end parameters
begin species
    1 X() x0
end species
begin reactions
    1 0 1 fr
end reactions
begin groups
    1 Xobs 1
end groups
begin functions
    1 fr() if(Xobs<1,0.001+1e8*(1-Xobs),0.001+1e8*(Xobs-1))
end functions
"""


@pytest.mark.parametrize("jacobian", ["auto", "fd"])
def test_a_rate_with_a_floor_at_its_kink_is_not_converged(tmp_path, jacobian):
    # Control for the verdict: main refuses it too (its newton_distance is NaN).
    # The march parks at the kink, and the step from there, 5e-12, is within
    # tolerance: there is no root, and the parity test refuses it (f = 1e-3).
    path = tmp_path / "kink.net"
    path.write_text(KINK_NET)
    ss = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", jacobian=jacobian
    ).steady_state(max_time=100.0)
    assert ss.newton_distance <= 1.0
    assert not ss.converged


# 0 -> X at k, and nothing reads X: the reduced Jacobian is a zero.
DRIFT_NET = """begin parameters
    1 k {k!r}
end parameters
begin species
    1 X() 1.0
end species
begin reactions
    1 0 1 k
end reactions
"""


@pytest.mark.parametrize("k", [1.0, 1e-12])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_drift_with_no_step_to_take_is_not_converged(tmp_path, k, method):
    # There is no step to take. At k = 1 the residual test refuses; at 1e-12,
    # under tol, a run of max_time moves X by 1e-6, past its tolerance.
    path = tmp_path / "drift.net"
    path.write_text(DRIFT_NET.format(k=k))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert math.isinf(ss.newton_distance)
    assert not ss.converged
    if k < 1e-9:
        assert ss.continued_run_distance > 1.0
    else:
        assert math.isnan(ss.continued_run_distance)


def test_a_drift_too_slow_to_show_in_max_time_passes(tmp_path):
    # Control: the runs ask that the state stay over max_time, and X' = 1e-20
    # moves X by 1e-14 in 1e6.
    path = tmp_path / "drift.net"
    path.write_text(DRIFT_NET.format(k=1e-20))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state()
    assert ss.converged


SIR_NET = """begin parameters
    1 beta 0.5
    2 gamma 0.2
end parameters
begin species
    1 S() 0.99
    2 I() 0.01
    3 R() 0
end species
begin reactions
    1 1,2 2,2 beta
    2 2 3 gamma
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_an_epidemic_that_burns_out_is_converged(tmp_path, method):
    # Every state with I = 0 is a steady state, and the run ends at the one
    # with S = 0.1059, below gamma/beta: a kick of I dies away.
    path = tmp_path / "sir.net"
    path.write_text(SIR_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert ss.converged
    assert ss.continued_run_distance <= 1.0
    s, i, _ = np.asarray(ss.concentrations)
    assert s == pytest.approx(0.105894, rel=1e-4)
    assert abs(i) < 1e-9


def _wide_square_decay(kx, extra_species=(), extra_reactions=(), copies=260):
    # X' = kx·(1 - X²) beside 2·copies unknowns, and whatever species (named
    # after X, from n + 1, at 0) and reactions (numbered on from X's) are added.
    n = 2 * copies + 1
    species = [f"    {n} X() 0"]
    species += [f"    {n + 1 + k} {name}() 0" for k, name in enumerate(extra_species)]
    reactions = [f"    {4 * copies + 1} 0 {n} kx", f"    {4 * copies + 2} {n},{n} {n} kx"] + [
        f"    {4 * copies + 3 + k} {r}" for k, r in enumerate(extra_reactions)
    ]
    return n, (
        _wide_net(copies)
        .replace("end parameters", f"    3 kx {kx!r}\n    4 kc 1.0\nend parameters")
        .replace("end species", "\n".join([*species, "end species"]))
        .replace("end reactions", "\n".join([*reactions, "end reactions"]))
    )


@pytest.mark.parametrize("kx", [0.03, 1e-6])
def test_above_512_unknowns_a_march_stopped_short_is_polished_within_reach(tmp_path, kx):
    # Among 520 unknowns the march stops by the residual test, at kx = 0.03
    # 7.3e-6 short of 1 (360 tolerances) and at 1e-6 at X = 0.698, which the
    # residual criterion returns. With no step to tell how far off it is, and
    # a polish that would move it more than a tolerance, a run of 1e6 is taken
    # on from the state first, and the state polished from where it ends, to a
    # thousandth of tol, on a Jacobian at every iterate, and no further than a
    # thousand tolerances: at 0.03 the run ends at the root; at 1e-6 it ends at
    # tanh(1 + artanh(0.698)) = 0.953, still moving, beyond the polish's reach,
    # and the run test refuses that state.
    n, text = _wide_square_decay(kx)
    path = tmp_path / "wide_square_decay.net"
    path.write_text(text)
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method="integration")
    assert out["residual"].converged
    assert abs(out["residual"].concentrations[n - 1] - 1.0) > 1e-6
    scaled = out["scaled"]
    assert math.isnan(scaled.newton_distance)
    if kx == 0.03:
        assert scaled.converged
        assert abs(scaled.concentrations[n - 1] - 1.0) < 1e-10
    else:
        assert not scaled.converged
        assert scaled.continued_run_distance > 1.0
        reached = math.tanh(kx * 1e6 + math.atanh(out["residual"].concentrations[n - 1]))
        assert scaled.concentrations[n - 1] == pytest.approx(reached, rel=1e-5)


_X = 521  # _wide_square_decay's X


def test_above_512_unknowns_a_masked_accumulator_goes_on_with_the_run(tmp_path):
    # C' = 1, left out by the mask, beside 520 unknowns: the march stops with
    # C at 213, which the residual criterion returns, and the run taken on
    # from the state before the polish carries it on by the run's 1e6, to the
    # run's end with every other species. Put back where the march left it,
    # a species the mask leaves out broke sums the reactions do not show (the
    # tests below).
    c = _X + 1
    n, text = _wide_square_decay(0.03, ("C",), [f"0 {c} kc"])
    assert n == _X
    path = tmp_path / "wide_square_decay_accumulator.net"
    path.write_text(text)
    mask = [True] * n + [False]
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method="integration", mask=mask)
    scaled = out["scaled"]
    assert math.isnan(scaled.newton_distance)
    assert scaled.converged
    assert abs(scaled.concentrations[n - 1] - 1.0) < 1e-10
    march = out["residual"].concentrations[c - 1]
    assert 10.0 < march < 1e4
    assert scaled.concentrations[c - 1] - march == pytest.approx(1e6, rel=1e-9)


@pytest.mark.parametrize(
    ("copies", "kg"), [(260, 1e-3), (260, 5e-4), (260, 1e-4), (20, 1e-3), (20, 1e-4)]
)
def test_a_masked_species_that_overflows_in_the_run_is_refused(tmp_path, copies, kg):
    # C' = kg·C from 1, left out by the mask: a run of 1e6 overflows it at
    # kg = 1e-3 (e^1000) and not at 1e-4 (e^100); above 512 unknowns, where
    # the state returned is the end of a run already, at 5e-4 too. Started
    # again from zero, as a species stepped below zero is, C came back 0 with
    # converged=True beside 520 unknowns. A run that cannot be taken to its
    # time passes nothing: the solve is refused, at any size, with C a number,
    # where the march left it or, at 5e-4 above 512, where the first run
    # ended (e^500). The residual criterion converges all of them, by the
    # residual alone.
    c = 2 * copies + 2
    n, text = _wide_square_decay(0.03, ("C",), [f"{c} {c},{c} kg"], copies=copies)
    path = tmp_path / "wide_square_decay_growth.net"
    path.write_text(
        text.replace("end parameters", f"    5 kg {kg!r}\nend parameters").replace(
            f"    {c} C() 0", f"    {c} C() 1.0"
        )
    )
    mask = [True] * n + [False]
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method="integration", mask=mask)
    assert out["residual"].converged
    scaled = out["scaled"]
    assert math.isfinite(scaled.concentrations[c - 1])
    assert scaled.concentrations[c - 1] > 1.0
    assert scaled.converged == (kg == 1e-4)
    if kg > 1e-4:
        assert math.isinf(scaled.continued_run_distance)
    if copies == 260 and kg == 1e-3:
        assert scaled.concentrations[c - 1] == pytest.approx(
            out["residual"].concentrations[c - 1], rel=1e-9
        )


def test_above_512_unknowns_a_masked_species_another_reads_is_taken_on_with_it(tmp_path):
    # C, left out by the mask, relaxes to 1 at 1e-3, and W follows it at 1:
    # the march stops with both 4.8e-4 short, which the residual criterion
    # returns. The run taken on from the state before the polish takes both
    # to 1. Put back where the march left it, C would sit beside a W it is
    # 4.8e-4 from, and the state would fail the parity test.
    c, w = _X + 1, _X + 2
    n, text = _wide_square_decay(
        1e-3, ("C", "W"), [f"0 {c} kx", f"{c} 0 kx", f"{c} {c},{w} kc", f"{w} 0 kc"]
    )
    path = tmp_path / "wide_square_decay_reader.net"
    path.write_text(text)
    mask = [True] * (n + 2)
    mask[c - 1] = False
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method="integration", mask=mask)
    assert out["residual"].converged
    assert 1.0 - out["residual"].concentrations[w - 1] > 1e-4
    scaled = out["scaled"]
    assert math.isnan(scaled.newton_distance)
    assert scaled.converged
    assert scaled.concentrations[c - 1] == pytest.approx(1.0, abs=1e-10)
    assert scaled.concentrations[w - 1] == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("kx", [1e-3, 1e-4])
def test_above_512_unknowns_a_masked_sink_keeps_its_law(tmp_path, kx):
    # X -> S beside 520 unknowns, S left out by mask=~is_pure_sink(): the
    # march stops with X at 4.7e-4 (kx = 1e-3) and 4.8e-3 (1e-4), and the run
    # taken on from it before the polish carries X to 0, and S with it. Put
    # back where the march left it, as an earlier form of this change did, S
    # broke X + S, which came back 0.99953 and 0.99520 with converged=True.
    copies = 260
    x, s = 2 * copies + 1, 2 * copies + 2
    text = (
        _wide_net(copies)
        .replace("end parameters", f"    3 kx {kx!r}\nend parameters")
        .replace("end species", f"    {x} X() 1.0\n    {s} S() 0\nend species")
        .replace("end reactions", f"    {4 * copies + 1} {x} {s} kx\nend reactions")
    )
    path = tmp_path / "wide_sink.net"
    path.write_text(text)
    model = bngsim.Model.from_net(str(path))
    mask = ~np.asarray(model.is_pure_sink())
    assert not mask[s - 1]
    ss = bngsim.Simulator(model, method="ode").steady_state(method="integration", mask=mask)
    assert math.isnan(ss.newton_distance)
    assert ss.converged
    assert ss.concentrations[x - 1] + ss.concentrations[s - 1] == pytest.approx(1.0, abs=1e-12)
    assert ss.concentrations[s - 1] == pytest.approx(1.0, abs=1e-10)


_B341 = Path(__file__).resolve().parents[2] / "tests/data/BIOMD0000000341.xml"


def test_a_polish_with_a_species_held_goes_no_further_than_a_thousand_tolerances(tmp_path):
    # BIOMD0000000341 with G left out by the mask: G's own equation pins I
    # and B, and KINSOL, with G held, went 1e8 tolerances from the march's
    # state, at the dynamics' root (100, 10, 300), to a root of the rest,
    # I = B = 0, which was converged. Its step was 1.9e-6; a step is polished
    # from no further than a thousand tolerances, and the polish goes no
    # further than that.
    names = list(bngsim.Model.from_sbml(str(_B341)).species_names)
    ss = bngsim.Simulator(bngsim.Model.from_sbml(str(_B341)), method="ode").steady_state(
        method="integration", mask=[name != "G" for name in names]
    )
    assert ss.converged
    c = dict(zip(names, ss.concentrations, strict=True))
    assert c["I"] == pytest.approx(10.0, rel=1e-6)
    assert c["B"] == pytest.approx(300.0, rel=1e-6)


def _wide_hidden_sum_sbml(nz, kx, how):
    # Y in a sum with X that no reaction shows, beside nz species Z fed and
    # drained at 1 from 1: by rate rules X' = -kx·X and Y' = kx·X; by two
    # reactions, X -> 0 and 0 -> Y, at the one rate kx·X; or, by a dilution,
    # Y in a compartment V that a rate rule takes from 0.5 to 1 at kx, which
    # keeps Y·V.
    math_ns = 'xmlns="http://www.w3.org/1998/Math/MathML"'

    def species(name, value, compartment="c"):
        return (
            f'<species id="{name}" compartment="{compartment}" initialConcentration="{value}" '
            'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        )

    def reaction(rid, reactant, product, law):
        sides = [
            (
                f"<{tag}>"
                f'<speciesReference species="{sp}" stoichiometry="1" constant="true"/>'
                f"</{tag}>"
            )
            for tag, sp in (("listOfReactants", reactant), ("listOfProducts", product))
            if sp
        ]
        return (
            f'<reaction id="{rid}" reversible="false">{"".join(sides)}'
            f"<kineticLaw><math {math_ns}>{law}</math></kineticLaw></reaction>"
        )

    def times(*names):
        return "<apply><times/>" + "".join(f"<ci>{n}</ci>" for n in names) + "</apply>"

    reactions = []
    for i in range(nz):
        reactions.append(reaction(f"s{i}", None, f"Z{i}", "<ci>ks</ci>"))
        reactions.append(reaction(f"d{i}", f"Z{i}", None, times("kz", f"Z{i}")))
    rules = ""
    compartments = '<compartment id="c" size="1" constant="true" spatialDimensions="3"/>'
    y = species("Y", 0)
    if how == "rate rules":
        rules = (
            f'<rateRule variable="X"><math {math_ns}><apply><minus/>{times("kx", "X")}'
            f'</apply></math></rateRule><rateRule variable="Y"><math {math_ns}>'
            f"{times('kx', 'X')}</math></rateRule>"
        )
    elif how == "two reactions":
        reactions += [
            reaction("loseX", "X", None, times("kx", "X")),
            reaction("gainY", None, "Y", times("kx", "X")),
        ]
    else:
        compartments += '<compartment id="V" size="0.5" constant="false" spatialDimensions="3"/>'
        y = species("Y", 1, "V")
        rules = (
            f'<rateRule variable="V"><math {math_ns}><apply><times/><ci>kx</ci>'
            "<apply><minus/><cn>1</cn><ci>V</ci></apply></apply></math></rateRule>"
        )
    return "".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">',
            f'<model id="m"><listOfCompartments>{compartments}</listOfCompartments>',
            "<listOfSpecies>",
            species("X", 1),
            y,
            *(species(f"Z{i}", 1) for i in range(nz)),
            "</listOfSpecies><listOfParameters>",
            f'<parameter id="kx" value="{kx!r}" constant="true"/>',
            '<parameter id="ks" value="1" constant="true"/>',
            '<parameter id="kz" value="1" constant="true"/>',
            f"</listOfParameters><listOfRules>{rules}</listOfRules>",
            f"<listOfReactions>{''.join(reactions)}</listOfReactions></model></sbml>",
        ]
    )


@pytest.mark.parametrize("how", ["rate rules", "two reactions", "dilution"])
def test_above_512_unknowns_a_masked_species_keeps_a_sum_no_reaction_shows(tmp_path, how):
    # Y left out by the mask, beside 520 species: the march stops short, and
    # the run taken on from the state before the polish takes X (V) and Y on
    # together. Put back where the march left it, as an earlier form of this
    # change did for a species nothing else reads and no reaction moves with
    # another, Y broke the sum: X + Y came back 0.99950, and Y·V 0.50026,
    # with converged=True.
    path = tmp_path / "wide_hidden_sum.xml"
    path.write_text(_wide_hidden_sum_sbml(520, 1e-3, how))
    model = bngsim.Model.from_sbml(str(path))
    names = list(model.species_names)
    mask = [name != "Y" for name in names]
    ss = bngsim.Simulator(model, method="ode").steady_state(method="integration", mask=mask)
    assert math.isnan(ss.newton_distance)
    assert ss.converged
    c = dict(zip(names, ss.concentrations, strict=True))
    if how == "dilution":
        # To the march's own error: the state it stopped at holds Y·V to 7e-7.
        assert c["V"] == pytest.approx(1.0, abs=1e-12)
        assert c["Y"] * c["V"] == pytest.approx(0.5, abs=1e-5)
    else:
        assert c["X"] + c["Y"] == pytest.approx(1.0, abs=1e-12)


def _wide_cubic_net(x0, s=1e-9, z0=1e-20):
    # The issue's cubic at a scale of s (stable roots 0 and 1, unstable 0.3)
    # beside 520 fast decays from z0.
    species = [f"    1 X() {x0 * s!r}"] + [f"    {i + 2} Z{i}() {z0!r}" for i in range(520)]
    reactions = ["    1 1,1,1 1,1 k3", "    2 1,1 1,1,1 k2", "    3 1 0 k1"] + [
        f"    {i + 4} {i + 2} 0 kz" for i in range(520)
    ]
    return "\n".join(
        ["begin parameters", "    1 k 1.0", "    2 a 0.3", "    3 u 1.0", f"    4 s {s!r}"]
        + ["    5 k3 k/(s*s)", "    6 k2 k*(u+a)/s", "    7 k1 k*a*u", "    8 kz 1e6"]
        + ["end parameters", "begin species", *species, "end species"]
        + ["begin reactions", *reactions, "end reactions", ""]
    )


@pytest.mark.parametrize(
    ("x0", "s", "z0", "reached"),
    [
        (0.15, 1e-9, 1e-20, 0.0),
        (0.5, 1e-9, 1e-20, 1.0),
        (0.3 + 1e-6, 1e-3, 0.0, 1.0),
        (0.3 - 1e-6, 1e-3, 0.0, 0.0),
    ],
)
def test_above_512_unknowns_the_polish_does_not_leave_the_basin(tmp_path, x0, s, z0, reached):
    # The march stops at its first step, under tol, with no step to say how far
    # the root is. Polished from there, the state from 0.15, on its way to 0,
    # went on to the other root at 1, and the one from 0.5, on its way to 1, to
    # 0.2999999, beside the unstable root; bounded to a thousand tolerances, the
    # polish still took 0.3 + 1e-6 to beside it. All converged; main returns
    # the states the march stopped at. The dynamics are taken on first, and the
    # state is polished from where a run ends: the root they reach.
    path = tmp_path / "wide_cubic.net"
    path.write_text(_wide_cubic_net(x0, s, z0))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method="integration"
    )
    assert ss.converged
    assert ss.concentrations[0] / s == pytest.approx(reached, abs=1e-6)


def test_above_512_unknowns_a_march_stopped_far_short_is_taken_on_to_its_root():
    # Control: main converges it, at the state its march stops at. fceri_fyn
    # (1281 species) stops with a free receptor at 2.96e-4, 3e4 tolerances from
    # where a run ends. A polish bounded to a thousand tolerances refused it; a
    # run is taken on from the state, and the polish from where it ends.
    path = Path(__file__).resolve().parents[2] / "benchmarks/models/net/ode/fceri_fyn.net"
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state()
    assert ss.converged
    assert ss.residual < 1e-9


_B002 = Path(__file__).resolve().parents[2] / "tests/data/BIOMD0000000002.xml"
_B407 = Path(__file__).resolve().parents[2] / "tests/data/BIOMD0000000407.xml"


def _run_from(path, y, t_end):
    model = bngsim.Model.from_sbml(str(path))
    model._core.set_state(np.asarray(y, float))
    bngsim.Simulator(model, method="ode").run(t_span=(0.0, t_end), n_points=2, atol="auto")
    return np.asarray(model._core.get_state(), float)


def test_biomd2_comes_out_as_a_run_does():
    # The residual test accepted it 6% from its steady state in a species; a
    # run of 1e6 from where it is returned now moves it by 1e-10 of itself.
    ss = bngsim.Simulator(bngsim.Model.from_sbml(str(_B002)), method="ode").steady_state()
    assert ss.converged
    y = np.asarray(ss.concentrations)
    end = _run_from(_B002, y, 1e6)
    assert np.max(np.abs(end - y) / (np.abs(y) + 1e-6 * np.abs(y).max())) < 1e-8
    residual = bngsim.Simulator(bngsim.Model.from_sbml(str(_B002)), method="ode").steady_state(
        criterion="residual"
    )
    y = np.asarray(residual.concentrations)
    end = _run_from(_B002, y, 1e6)
    assert np.max(np.abs(end - y) / (np.abs(y) + 1e-6 * np.abs(y).max())) > 0.05


def test_biomd407_comes_out_as_a_run_does():
    # The residual test accepted it one step from where it starts. A run is
    # still moving at 1e6 and settles by 1e8: in max_time = 1e6 it is not
    # converged, and in 1e9 it is, where the run ends.
    ss = bngsim.Simulator(bngsim.Model.from_sbml(str(_B407)), method="ode").steady_state()
    assert not ss.converged
    y0 = np.asarray(bngsim.Model.from_sbml(str(_B407))._core.get_state(), float)
    ref = _run_from(_B407, y0, 1e10)
    for method in ("integration", "newton"):
        ss = bngsim.Simulator(bngsim.Model.from_sbml(str(_B407)), method="ode").steady_state(
            method=method, max_time=1e9
        )
        assert ss.converged
        y = np.asarray(ss.concentrations)
        assert np.max(np.abs(y - ref) / (np.abs(ref) + 1e-6 * np.abs(ref).max())) < 1e-8


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_step_along_a_continuum_says_nothing_and_the_runs_decide(data_dir, method):
    # IgE.bngl conserves R(a~1) - R(a~0) + 2·IgE_dimer because its two
    # reactions out of R(a~X) share one rate constant, a law of its rates that
    # no law of its stoichiometry holds: its steady states make a line, the
    # reduced Jacobian is singular but for rounding, and the step along the
    # line was 4e7 tolerances (rounding says how far) at a state no run moves.
    path = str(data_dir / "ige_receptor_cycle.net")
    ss = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(method=method)
    assert ss.converged
    assert ss.continued_run_distance <= 1.0
    y = np.asarray(ss.concentrations)
    run = bngsim.Model.from_net(path)
    run.set_state(y)
    bngsim.Simulator(run, method="ode").run(t_span=(0.0, 1e6), n_points=2, atol="auto")
    np.testing.assert_allclose(np.asarray(run.get_state()), y, rtol=1e-6)


_SHP2 = Path(__file__).resolve().parents[2] / "benchmarks/models/net/ode/SHP2_base_model.net"


@pytest.mark.skipif(not _SHP2.exists(), reason="benchmark net SHP2_base_model not present")
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_solve_seeded_with_its_own_answer_converges_again(method):
    # The species SHP2 drains to nothing sit at 1e-20 and below: taken as the
    # floor of the march's tolerances, they held every species at 0 to that,
    # and a solve seeded with its own answer was refused there.
    model = bngsim.Model.from_net(str(_SHP2))
    sim = bngsim.Simulator(model, method="ode")
    first = sim.steady_state(method="integration")
    assert first.converged
    model.set_state(np.asarray(first.concentrations))
    again = sim.steady_state(method=method)
    assert again.converged
    assert again.continued_run_distance <= 1.0
    np.testing.assert_allclose(
        np.asarray(again.concentrations),
        np.asarray(first.concentrations),
        rtol=1e-6,
        atol=1e-8,
    )


# ─── What the independent review found (issue #1001) ─────────────────────────

# 0 -> X at s, X -> 0 at k1, X + X -> D at k2, D -> 0; G, unrelated, at `big`.
BESIDE_A_LARGE_ONE_NET = """begin parameters
    1 s   1.3
    2 k1  0.21
    3 k2  0.73
    4 kd  0.5
    5 sg  {big!r}
    6 dg  1.0
end parameters
begin species
    1 X() 0
    2 D() 0
    3 G() {big!r}
end species
begin reactions
    1 0 1 s
    2 1 0 k1
    3 1,1 2 k2
    4 2 0 kd
    5 0 3 sg
    6 3 0 dg
end reactions
"""


@pytest.mark.parametrize("big", [1e8, 1e12])
@pytest.mark.parametrize("jacobian", ["auto", "fd"])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_beside_a_large_one_is_solved_to_its_root(tmp_path, big, jacobian, method):
    # Control: main solves it. X starts at 0 beside G at `big`; held to
    # rtol·big, as atol="auto" holds a species with no size of its own, X
    # came back converged 5% off at 1e8 and 129% off at 1e12 (differenced
    # columns) on the way to this change.
    path = tmp_path / "beside.net"
    path.write_text(BESIDE_A_LARGE_ONE_NET.format(big=big))
    ss = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", jacobian=jacobian
    ).steady_state(method=method)
    x_root = (-0.21 + math.sqrt(0.21**2 + 8 * 0.73 * 1.3)) / (4 * 0.73)
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(x_root, rel=1e-7)


# A + A -> 0 at 1, 0 -> A at 1e-6: A settles at 7.07e-4 from wherever it starts.
SETTLES_SMALL_NET = """begin parameters
    1 k 1.0
    2 s 1e-6
end parameters
begin species
    1 A() {a0!r}
end species
begin reactions
    1 1,1 0 k
    2 0 1 s
end reactions
"""


@pytest.mark.parametrize("a0", [1e5, 1e7, 1e8])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_that_settles_far_below_its_start_is_solved_to_its_root(tmp_path, a0, method):
    # Marched at atol="auto" from 1e7, A was held to 0.1 and came back
    # converged 91 times its root; from 1e5 the march ran to -1e19.
    path = tmp_path / "settles.net"
    path.write_text(SETTLES_SMALL_NET.format(a0=a0))
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(math.sqrt(1e-6 / 2), rel=1e-7)


def test_a_tol_the_call_gives_is_read(tmp_path):
    # Control: main reads it. A <-> B with A0 the rate back: A* = 9/4. Under
    # the default the parity test is asked too, at the tol given.
    path = tmp_path / "ab.net"
    path.write_text(
        "begin parameters\n    1 A0 3\n    2 kf 1\nend parameters\n"
        "begin species\n    1 A() A0\n    2 B() 0\nend species\n"
        "begin reactions\n    1 1 2 kf\n    2 2 1 A0\nend reactions\n"
    )
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(tol=1e-14)
    assert ss.converged and ss.residual < 1e-14
    np.testing.assert_allclose(np.asarray(ss.concentrations), [9 / 4, 3 / 4], rtol=1e-12)


@pytest.mark.parametrize(
    "text,want",
    [
        ("model neg\n V = 0\n J0: -> V; k*(Vrest - V)\n k = 0.5\n Vrest = -65\nend", [-65.0]),
        (
            "model neg2\n V = -60\n n = 0.3\n J0: -> V; k*(Vrest - V) - g*n*(V - Ek)\n"
            " J1: -> n; a*(1 - n) - b*n\n k = 0.5\n Vrest = -65\n g = 0.1\n Ek = -80\n"
            " a = 0.1\n b = 0.2\nend",
            [-65.9375, 1 / 3],
        ),
    ],
)
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_variable_that_rests_below_zero_is_converged(text, want, method):
    # Control: main converges. A membrane potential at -65; a guard on the
    # sign of every species refused it on the way to this change.
    model = bngsim.Model.from_antimony_string(text)
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged
    np.testing.assert_allclose(np.asarray(ss.concentrations), want, rtol=1e-7)


@pytest.mark.parametrize("x0", [0.5, 0.9, 1.5])
def test_a_state_set_at_another_scale_is_held_to_its_own(tmp_path, x0):
    # The cubic at a scale of 1e-9, declared at 1 and started by set_state()
    # and states=. Held to the floor its declaration gave, it was returned
    # where it was set.
    model = _cubic(tmp_path, 1e-9, 1.0 / 1e-9)
    model.set_state([x0 * 1e-9])
    ss = bngsim.Simulator(model, method="ode").steady_state()
    assert ss.converged
    assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)
    [batch] = bngsim.Simulator(
        _cubic(tmp_path, 1e-9, 1.0 / 1e-9), method="ode"
    ).steady_state_batch(states=[[x0 * 1e-9]])
    assert batch.converged
    assert batch.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


def test_a_species_at_roundoff_does_not_set_the_march_tolerance():
    # MODEL2307110001 declares a species at -1.6e-34. Taken as the floor of
    # atol="auto", it held every species at 0 to that, and the march failed.
    sim = (
        bngsim.Simulator(bngsim.Model.from_net(str(_SHP2)), method="ode")
        if _SHP2.exists()
        else None
    )
    if sim is None:
        pytest.skip("benchmark net SHP2_base_model not present")
    state = np.asarray(sim._model.get_state(), float)
    state[0] = -1.6e-34
    atol = np.asarray(sim._steady_state_atol(None, "scaled", state, 1e-8))
    assert atol.min() > 1e-30
    assert atol.max() <= 1e-8


def test_a_run_is_not_moved_by_its_own_error(data_dir):
    # Integrated at the criterion's own tolerances, the run moved the state by
    # 3.45 tolerances, at a hundredth of them by 1.41, at a thousandth by 0.46:
    # its own error. A run that moves the state is taken again tighter.
    path = str(data_dir / "BIOMD0000000038.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state()
    assert ss.converged
    assert ss.continued_run_distance <= 1.0


# ─── What the second review found (issue #1001) ──────────────────────────────

# 0 -> S at k0, S -> 0 at kd, from S = 1: S* = k0/kd, dS*/dkd = -k0/kd^2.
FROM_ABOVE_NET = """begin parameters
    1 k0  5e-14
    2 kd  5e-3
end parameters
begin species
    1 S() 1.0
end species
begin reactions
    1 0 1 k0
    2 1 0 kd
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_settling_far_below_its_start_has_its_root_and_column(tmp_path, method):
    # Control: main's steps land on the root. S settles at 1e-11 from 1, at an
    # atol of 1e-8: stopped within tolerance, S came back 0.4% off, and
    # dS*/dkd with it, on the way to this change. The state is polished.
    path = tmp_path / "from_above.net"
    path.write_text(FROM_ABOVE_NET)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method, sensitivity_params=["kd"]
    )
    assert ss.converged
    assert ss.concentrations[0] == pytest.approx(5e-14 / 5e-3, rel=1e-9)
    assert np.asarray(ss.sensitivity)[0, 0] == pytest.approx(-5e-14 / 5e-3**2, rel=1e-6)


@pytest.mark.parametrize("x0", [0.5, 0.9, 1.5])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_the_cubic_beside_a_species_at_1e6_reaches_its_root(tmp_path, x0, method):
    # 1e-14 of the largest species was taken for dust, and X at 5e-10 beside
    # Y at 1e6 was held to 1e-8: the issue's answers came back (0.894, 1.008,
    # 0.443, and 0.3 by Newton). A species keeps its own size.
    text = (
        CUBIC_NET.format(s=1e-9, x0=x0)
        .replace("end species", "    2 Y() 1e6\nend species")
        .replace("end reactions", "    4 0 2 big\n    5 2 0 k\nend reactions")
        .replace("end parameters", "    9 big 1e6\nend parameters")
    )
    path = tmp_path / "cubic_beside_big.net"
    path.write_text(text)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert ss.converged
    assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("max_time", [1e8, 1e10])
def test_a_long_max_time_does_not_lengthen_the_runs_past_the_default(data_dir, max_time):
    # Taken on for max_time, the runs from BIOMD0000000038's steady state at
    # 1e8 could not be taken to their time, and egfr_net's at 1e10 took 228 s;
    # main converged on both.
    path = str(data_dir / "BIOMD0000000038.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state(
        max_time=max_time
    )
    assert ss.converged
    assert ss.continued_run_distance <= 1.0


def test_a_burst_free_solve_says_it_was_decided_by_the_residual_test(tmp_path):
    # Control: main's default is the residual test.
    sim = bngsim.Simulator(_cubic(tmp_path, 1.0, 0.5), method="ode")
    [ss] = sim.steady_state_batch(states=[[0.5]], method="newton", newton_burst=False)
    assert ss.converged
    assert ss.criterion == "residual"


def test_a_march_that_stops_where_f_is_rounding_returns_the_state_it_passed(data_dir):
    # Control: main stops at the step that passes. Monro2008's R rests at 2e12,
    # where f is rounding: 0 at step 517, which falls between the steps the
    # step test is asked at, and 2.6e-6 at step 518, the march's max_time. The
    # march returns the last state that passed.
    path = str(data_dir / "BIOMD0000000776.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state()
    assert ss.converged
    assert ss.residual < 1e-9


SIR_AT_NO_INFECTION_ANT = """model sir0
  S = 0.99; I = 0; R = 0.01
  J1: S + I -> 2 I; kb*S*I
  J2: I -> R; kg*I
  kb = 0.5; kg = 0.2
end"""

COUNTER_AT_NO_PULSE_ANT = """model counter
  S0 = 100; S1 = 0; S2 = 0; S3 = 0; P = 0
  J1: S0 -> S1; k*S0*P
  J2: S1 -> S2; k*S1*P
  J3: S2 -> S3; k*S2*P
  J4: P -> ; kd*P
  k = 10; kd = 1
end"""


@pytest.mark.parametrize("text", [SIR_AT_NO_INFECTION_ANT, COUNTER_AT_NO_PULSE_ANT])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_state_with_an_absent_species_is_converged(text, method):
    # Control: main converges. SIR at I = 0 and a counter whose pulse is at 0
    # rest where they start. A run kicked off them by a millionth refused both:
    # the infection grew by 4e6 of the kick, and the pulse moved the counter 50
    # times what the kick was let grow to.
    model = bngsim.Model.from_antimony_string(text)
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged
    np.testing.assert_array_equal(ss.concentrations, model.get_state())


SQUARE_DECAY_NET = """begin parameters
    1 k 1.0
end parameters
begin species
    1 X() 1.0
end species
begin reactions
    1 1,1 1 k
end reactions
"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_reaching_zero_as_one_over_t_is_polished_there(tmp_path, method):
    # X' = -X² reaches its root, 0, as 1/t. The residual test returns X at
    # 3e-5, which a run of max_time halves; the march ends at max_time with X
    # near 1e-6, 50 tolerances short. The polish halves X at each of its ten
    # iterations, its steps never small enough to stop it, and so ends at its
    # iteration limit, where the iterate it ends at passes the parity test and
    # is kept.
    path = tmp_path / "square_decay.net"
    path.write_text(SQUARE_DECAY_NET)
    out = {}
    for criterion in ("residual", "scaled"):
        out[criterion] = bngsim.Simulator(
            bngsim.Model.from_net(str(path)), method="ode"
        ).steady_state(criterion=criterion, method=method)
    assert out["residual"].converged
    assert out["residual"].concentrations[0] > 1e-5
    scaled = out["scaled"]
    assert scaled.converged
    assert 0.0 < scaled.concentrations[0] < 1e-8
    assert scaled.continued_run_distance <= 1.0


LOPSIDED_LAW_ANT = """model lop
  B = 1e-6
  A = 1e6
  J1: A -> B; kf*A
  J2: B -> A; kr*B
  kf = 1e-12; kr = 1.0
end"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_polish_that_loses_a_dependents_digits_is_not_kept(method):
    # Control for integration: main's march leaves B exact. The start is the
    # steady state. KINSOL solves for A and makes B from the total, 1e6 + 1e-6,
    # which leaves B at 1.0000076e-6, 381 tolerances off, where main's Newton
    # solve returns it. The polish is kept only where it is nearer the root.
    model = bngsim.Model.from_antimony_string(LOPSIDED_LAW_ANT)
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged
    np.testing.assert_array_equal(ss.concentrations, model.get_state())


@pytest.mark.parametrize("x0", [0.5, 1.5])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_fixed_species_far_above_the_rest_does_not_coarsen_the_march(tmp_path, x0, method):
    # The cubic at 1e-9 beside a $-fixed Y at 1e20. Counted in the march's atol
    # floor, Y's double-precision resolution, 2e4, held X to the scalar 1e-8,
    # and the march ran X to 9.6 and 29 times its root and refused.
    text = CUBIC_NET.format(s=1e-9, x0=x0).replace("end species", "    2 $Y() 1e20\nend species")
    path = tmp_path / "cubic_beside_fixed.net"
    path.write_text(text)
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        method=method
    )
    assert ss.converged
    assert ss.concentrations[0] / 1e-9 == pytest.approx(1.0, rel=1e-6)


def test_a_tight_rtol_is_asked_no_finer_than_a_run_can_tell():
    # Control: main converges. Asked for rtol 1e-12, the runs, at 1e-14 and
    # 1e-15, could not be taken to their time and refused after 10.7 s; at
    # 1e-13 they moved the state 3.9 tolerances of 1e-12, their own error.
    path = Path(__file__).resolve().parents[2] / "benchmarks/models/net/ode/Kocieniewski_2012.net"
    ss = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state(
        rtol=1e-12, atol=1e-14
    )
    assert ss.converged


def test_a_march_the_step_cannot_stop_returns_the_state_the_residual_test_passed(data_dir):
    # BIOMD0000000463's step reads inf where the residual test passes (a jump
    # beside the state), so the march was taken on past it until it ran away,
    # to a residual of 2e16 with a species at -1.8e5. It returns the state the
    # residual test passed, main's, where the eigenvalues have a positive real
    # part (0.13) and a run from it leaves.
    #
    # The interpreted march, pinned (issue #1217): near an unstable state,
    # whether a march passes the residual test before it runs away is in the
    # last digits. On x86-64 Linux, interpreted with the closed form it passes
    # (3.9e-10), and so does compiled with jacobian="fd" (1.8e-10). Interpreted
    # with jacobian="fd" it runs away (2.6e8), and so does compiled with the
    # compiled closed form (4.6e17), never passing, so a forced codegen run had
    # no passed state to return.
    path = str(data_dir / "BIOMD0000000463.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode", codegen=False).steady_state()
    assert not ss.converged
    assert ss.residual < 1e-9
    assert np.min(ss.concentrations) > -1e-6
    assert ss.continued_run_distance > 1.0


BASIN_BESIDE_A_LOPSIDED_LAW_ANT = """model bs
  B = 1e-6
  A = 1e6
  W = 1e-6
  J1: A -> B; kf*A
  J2: B -> A; kr*B
  J3: A -> B; kc*B^2/(K^2 + B^2)
  J4: -> W; B
  J5: W -> ; W
  kf = 1e-12; kr = 1; kc = 2; K = 0.5
end"""


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_state_beside_a_second_basin_is_converged(method):
    # Control for integration: main converges there (its Newton solve leaves
    # B 7.6e-6 off, in its last digits). B, a law's dependent at 1e-6, has a
    # second basin at 1.7, which W follows. A kick of a millionth of A, at 1e6,
    # taken up by B crossed into it, and a run so kicked refused the state.
    model = bngsim.Model.from_antimony_string(BASIN_BESIDE_A_LOPSIDED_LAW_ANT)
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged
    assert ss.concentrations[1] == pytest.approx(1e6, rel=1e-12)
    assert ss.concentrations[2] == pytest.approx(ss.concentrations[0], rel=1e-6)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_state_resting_on_a_continuum_is_converged(data_dir, method):
    # Control: main converges. egfr_simple (RuleHub's tutorial, its network
    # generated by BNG2.pl, no EGF) rests on a continuum of steady states. A
    # run kicked off it by a millionth landed a species 47 times further than
    # the kick was let grow to, and refused it.
    path = str(data_dir / "egfr_simple_seed.net")
    ss = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(method=method)
    assert ss.converged


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_march_whose_integrator_fails_returns_the_state_it_passed(data_dir, method):
    # Control: main converges. glioblastoma-egfrviii-signaling (its network
    # generated by BNG2.pl, no actions run) passes the residual test at step
    # 2075, between the steps the step test is asked at, and its integrator
    # then fails, as it does again on difference quotients: the state it
    # failed at came back, with a residual of NaN.
    path = str(data_dir / "glioblastoma_egfrviii_seed.net")
    ss = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(method=method)
    assert ss.converged
    assert ss.residual < 1e-9


def _sir_in_order(order):
    init = {"S": "0.99", "I": "0", "R": "0.01"}
    decl = "\n".join(f"  {s} = {init[s]}" for s in order)
    return f"""model sir0
{decl}
  J1: S + I -> 2 I; kb*S*I
  J2: I -> R; kg*I
  kb = 0.5; kg = 0.2
end"""


@pytest.mark.parametrize("order", ["SIR", "SRI", "ISR", "IRS", "RSI", "RIS"])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_sir_at_no_infection_is_converged_in_every_order(order, method):
    # Control: main converges in every order. The order picks the dependent of
    # S + I + R; where it was I, a run kicked off the state made it follow S's
    # and R's kicks, the infection grew, and the state was refused.
    model = bngsim.Model.from_antimony_string(_sir_in_order(order))
    ss = bngsim.Simulator(model, method="ode").steady_state(method=method)
    assert ss.converged


# I feeds C1 at 3 and C2 at 1, and nothing else moves: at I = 0 every split of
# C1 and C2 is a root, and their columns of the reduced Jacobian are one.
SPLIT_NET = """begin parameters
    1 k1 3.0
    2 k2 1.0
end parameters
begin species
    1 I() 1.0
    2 C1() 0
    3 C2() 0
end species
begin reactions
    1 1 2 k1
    2 1 3 k2
end reactions
"""


@pytest.mark.parametrize("jacobian", ["auto", "fd"])
def test_a_singular_jacobian_is_singular_on_every_platform(tmp_path, jacobian):
    # The march stops with I near 1e-40, short of 0, so the rates are not 0
    # and the step is asked for. Its last pivot is -1 + (1/3)*3: 0 where the
    # product is rounded before the sum, as on x86-64, and -5.6e-17 where the
    # two are fused into one multiply-add, as on arm64. By its rounding the
    # matrix is singular on both, so no step is had; the run decides.
    path = tmp_path / "split.net"
    path.write_text(SPLIT_NET)
    ss = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", jacobian=jacobian
    ).steady_state(criterion="scaled", method="integration")
    assert ss.converged
    assert ss.newton_distance == math.inf
    assert list(ss.concentrations[1:]) == pytest.approx([0.75, 0.25], rel=1e-9)


# Compiled, the march comes to the same root without its integrator failing
# (residual 1.05e-15, within 1.6e-6 of the interpreter's state), so it is not
# retried: this pins the arm64 interpreter's route there, not its answer.
@threshold_forced
def test_a_march_whose_integrator_fails_is_still_retried_on_difference_quotients(data_dir):
    # MODEL1712050008's march passes the residual test, goes on, and its
    # integrator fails; the step does not take the state it passed (202
    # tolerances), which the run then leaves. Reported as a failure still, the
    # solve is retried on difference quotients, and that retry comes to rest.
    # Taking the passed state as the answer turned the retry off. Main
    # converges at a state a run moves 2.2e5 tolerances. That march is
    # arm64 macOS's: rounded as x86-64 Linux rounds it, the march reaches the
    # root by its step (4e-6) without its integrator failing, asks no retry,
    # and the run from it cannot be taken to its time, so the solve is
    # refused (residual 1.05e-15, on either linear solver). Everywhere, it is
    # not converged at a state a run moves.
    path = str(data_dir / "MODEL1712050008.xml")
    ss = bngsim.Simulator(bngsim.Model.from_sbml(path), method="ode").steady_state()
    assert not ss.converged or ss.continued_run_distance <= 1.0
    if sys.platform == "darwin" and platform.machine() == "arm64":
        assert ss.solver_jacobian_retried
        assert ss.converged


_CC_ON_PATH = any(shutil.which(c) for c in ("cc", "gcc", "clang"))


@pytest.mark.skipif(not _CC_ON_PATH, reason="no C compiler on PATH for the codegen leg")
def test_a_compiled_march_to_a_fractional_power_at_zero_is_not_refused(data_dir):
    """Issue #1357. glioblastoma-egfrviii raises AKT(s~P) to 3.5, and the march
    on the compiled right-hand side comes to it at 0 or within 1e-52 of it, at
    residual 2e-39. A corrector iterate an ulp below 0 makes the power NaN, so
    the run test could not be taken to its time (at t = 3.9e5 on arm64 macOS,
    at its first step on x86-64 Linux), and the state was refused with a run
    measure of inf. The run test now integrates as the time course does, which
    takes f again at the clamped state where it is NaN (GH #135). No retry is
    the answer's."""
    path = str(data_dir / "glioblastoma_egfrviii_seed.net")
    compiled = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", codegen=True
    ).steady_state(method="integration")
    assert compiled.converged
    assert compiled.continued_run_distance <= 1.0
    assert compiled.solver_jacobian_source == "codegen"
    assert not compiled.solver_jacobian_retried
    interpreted = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").steady_state(
        method="integration"
    )
    # At the march's own tolerances (rtol and atol 1e-8): AKT(s~P) is 0 here
    # and 1e-9 there, and the rest agree to the digit.
    np.testing.assert_allclose(
        compiled.concentrations, interpreted.concentrations, rtol=1e-8, atol=1e-8
    )


# Where the compiled march ended on x86-64 Linux, to the bit: AKT(s~P), the
# last, and Oncogenic_Output() are exactly 0, and the residual is 1.56e-40.
_GLIOBLASTOMA_AT_ZERO = [
    -4.5627044333513721e-44,
    2.0000000000000088e02,
    1.0000000000000007e03,
    -2.8516902708446076e-45,
    0.0,
    -6.0212043388956984e-45,
    -4.2897249239143463e-42,
    2.7026996748452282e-45,
    4.0693707403992688e-42,
    0.0,
]


@pytest.mark.parametrize(
    ("jacobian", "sparse"),
    [
        pytest.param("auto", False, id="auto"),
        pytest.param("fd", False, id="fd"),
        # A KLU-off build factors the march densely whatever is forced.
        pytest.param(
            "fd",
            True,
            id="fd colored",
            marks=pytest.mark.skipif(not bngsim.HAS_KLU, reason="KLU not compiled"),
        ),
    ],
)
def test_a_run_test_from_a_fractional_power_at_zero_is_taken_to_its_time(
    data_dir, jacobian, sparse
):
    """Issue #1357, on every platform: a solve started where the Linux march
    ended. On difference quotients, CVODE could not take the march or the run
    test one step from it (the corrector's iterate put AKT(s~P) below 0, where
    AKT(s~P)^3.5 is NaN), and the state was refused at residual 1.56e-40. A
    time course from it moves nothing. `fd colored` is the quotient KLU
    factors, formed from the same f."""
    path = str(data_dir / "glioblastoma_egfrviii_seed.net")
    sim = bngsim.Simulator(
        bngsim.Model.from_net(path),
        method="ode",
        jacobian=jacobian,
        force_sparse_linear_solver=sparse,
    )
    sim.set_state(np.array(_GLIOBLASTOMA_AT_ZERO))
    ss = sim.steady_state(method="integration")
    assert (ss.linear_solver == "klu") == sparse
    assert ss.converged
    assert ss.continued_run_distance <= 1.0
    np.testing.assert_allclose(ss.concentrations, _GLIOBLASTOMA_AT_ZERO, rtol=1e-8, atol=1e-8)
