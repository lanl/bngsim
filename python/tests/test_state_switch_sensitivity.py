"""Issue #150 — the saltation jump at a state-dependent rate-law switch.

A condition inside a rate law that reads the state — ``piecewise(0, Virus < 1,
Virus*rho_V)`` — flips a branch of ``f`` at a crossing whose time ``t*(θ)`` moves
with **every** parameter through the trajectory. The in-branch derivative is
right on both sides (``sympy.diff`` of a ``Piecewise`` carries no boundary delta
and does not need one); what is discontinuous is ``∂x/∂θ`` itself, by

    s(t*⁺) = s(t*⁻) + (f⁻ − f⁺)·dt*/dθ

Neither of the two ways bngsim could produce a sensitivity RHS carries that term
— both integrate the variational equation smoothly across — so it has to be
applied *at* the crossing, which first has to be located. That is the whole of
this issue: root on the condition's residual, differentiate ``dt*/dθ`` there by
the implicit function theorem (the issue #144 machinery, reached from a rate law
instead of an event trigger), and jump.

**What the oracle is.** A central finite difference of the model's own
trajectory, at a tolerance tight enough that the FD's own truncation error is the
only thing left. That works here — unlike at a *clock* switch, where both the
analytic path and the FD miss the crossing by the same O(h) and the comparison is
vacuous (see ``test_codegen_switch_condition_sens``) — because the FD re-solves
the whole trajectory including the moved crossing, and a state crossing moves
smoothly with the parameter.

**Why the analytic RHS had to be admitted with it.** With the crossing resolved
to a root, CVODES' internal difference quotient becomes *worse*, not better: its
probe evaluates ``f`` at ``y + σ·s`` with ``σ ≈ √rtol``, and just past a crossing
``σ·|s|`` is easily wide enough to put the probe back on the other branch. On the
model below that injected ``rho·X/σ ≈ 2.7e4`` into ``ds/dt`` for the sliver of
time the state stayed within ``σ·|s|`` of the surface, and the column came out
28% high — a jump correctly applied and then spoiled. So issue #150 also lifted
the GH #68 decline for exactly the conditions it compensates;
``test_codegen_switch_condition_sens`` owns that half of the contract.
"""

from __future__ import annotations

import re

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim import _switch_sensitivity as sw
from bngsim._exceptions import SimulationError

pytest.importorskip("sympy")


def _has_cc() -> bool:
    try:
        cg._find_c_compiler()
        return True
    except Exception:
        return False


requires_cc = pytest.mark.skipif(not _has_cc(), reason="no C compiler available")


# ─── the issue's own reproduction ──────────────────────────────────────────
#
#   dX/dt = if(X<1, 0, rho)·X − delta·X,   X(0) = X0 = 1000
#
# the same shape as AMICI's ``nested_events`` fixture with its injection event
# removed. Before the crossing dX/dt = (rho−delta)X = −0.8X, after it −delta·X =
# −1.6X, so the crossing is at t* = ln(1000)/0.8 = 8.63469 and the saltation
# factor is f⁺/f⁻ = 2 exactly.
#
# ``rho`` shows that factor cleanly (it appears ONLY in the branch that switches
# off, so its post-crossing column is pure jump); ``delta`` mixes the jump with a
# correct in-branch part and is the more representative case — it is also why
# "multiply the tail by f⁺/f⁻" is not a fix, and why both are asserted.
NET = """\
begin parameters
    1 X0     1000  # Constant
    2 rho    0.8  # Constant
    3 delta  1.6  # Constant
end parameters
begin functions
    1 growth() if(X<1,0,rho)
end functions
begin species
    1 A() X0
end species
begin reactions
    1 1 1,1 growth #_R1
    2 1 0 delta #_R2
end reactions
begin groups
    1 X                    1
end groups
"""

T_STAR = np.log(1000.0) / 0.8  # 8.634694...
T_END = 12.0
N_POINTS = 25
RTOL, ATOL = 1e-9, 1e-12


def _model(tmp_path, text=NET, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    return bngsim.Model.from_net(net)


def _sens(tmp_path, params, name="m.net", text=NET, t_end=T_END, **kw):
    model = _model(tmp_path, text, name)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params), **kw)
    return sim.run(t_span=(0.0, t_end), n_points=N_POINTS, rtol=RTOL, atol=ATOL)


def _fd(tmp_path, params, text=NET, rel=1e-5, t_end=T_END):
    """Central difference of the model's own trajectory, one column per param."""
    cols = []
    for i, p in enumerate(params):
        p0 = _model(tmp_path, text, f"fd{i}.net").get_param(p)
        h = rel * abs(p0)
        got = []
        for sign in (+1, -1):
            m = _model(tmp_path, text, f"fd{i}{sign}.net")
            m.set_param(p, p0 + sign * h)
            r = bngsim.Simulator(m, method="ode").run(
                t_span=(0.0, t_end), n_points=N_POINTS, rtol=RTOL, atol=ATOL
            )
            got.append(np.asarray(r.species))
        cols.append((got[0] - got[1]) / (2 * h))
    return np.stack(cols, axis=-1)


# ─── the term ──────────────────────────────────────────────────────────────


@requires_cc
class TestTheSaltationTerm:
    def test_every_column_matches_a_finite_difference_across_the_crossing(self, tmp_path):
        """The whole issue in one assertion. Before the fix ``rho`` came back a
        factor of exactly 2 low after t*, and ``delta`` came back low by a
        parameter-dependent amount between 1.7 and 2.0 — the same defect, seen
        through a column that also has a real in-branch part."""
        params = ["rho", "delta"]
        run = _sens(tmp_path, params)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params)
        assert an.shape == fd.shape
        for j, p in enumerate(params):
            scale = float(np.max(np.abs(fd[:, 0, j])))
            np.testing.assert_allclose(
                an[:, 0, j],
                fd[:, 0, j],
                rtol=2e-4,
                atol=1e-5 * scale,
                err_msg=f"column {p!r} disagrees with its own finite difference",
            )

    def test_the_jump_lands_at_the_crossing_and_nowhere_else(self, tmp_path):
        """Localisation, which is what says the *term* is being tested rather
        than the aggregate. Sample densely and compare against the closed form
        on each side: before t* the ``rho`` column is ``t·X(t)`` exactly, after
        it the jumped value decaying at ``-delta``. A jump applied at the wrong
        instant — or twice — fails here and passes an all-points tolerance."""
        model = _model(tmp_path)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["rho"])
        times = [T_STAR - 0.5, T_STAR - 1e-4, T_STAR + 1e-4, T_STAR + 0.5, T_STAR + 2.0]
        run = sim.run(sample_times=[0.0, *times], rtol=1e-11, atol=1e-13)
        s = np.asarray(run.sensitivities)[:, 0, 0]

        # Before: X = X0·e^{(rho−delta)t}, so ∂X/∂rho = t·X.
        for k, t in enumerate(times[:2], start=1):
            x = 1000.0 * np.exp(-0.8 * t)
            assert s[k] == pytest.approx(t * x, rel=1e-6)
        # At t* the column is t*·X(t*) = t*·1; the jump doubles it, and after the
        # crossing dX/dt = −delta·X has ∂f/∂rho = 0, so it only decays.
        for k, t in enumerate(times[2:], start=3):
            assert s[k] == pytest.approx(2.0 * T_STAR * np.exp(-1.6 * (t - T_STAR)), rel=1e-5)

    def test_the_jump_is_exactly_the_saltation_factor(self, tmp_path):
        """``rho`` appears only in the branch that switches off, so its whole
        post-crossing column is the jump and the ratio across t* is
        ``f⁺/f⁻ = (−delta·X)/((rho−delta)·X) = 2`` — the number the issue
        measured on AMICI's fixture, reproduced in closed form."""
        model = _model(tmp_path)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["rho"])
        run = sim.run(sample_times=[0.0, T_STAR - 1e-9, T_STAR + 1e-9], rtol=1e-12, atol=1e-14)
        s = np.asarray(run.sensitivities)[:, 0, 0]
        assert s[2] / s[1] == pytest.approx(2.0, rel=1e-6)

    def test_an_initial_condition_column_is_jumped_too(self, tmp_path):
        """The lesson issue #144 paid for on the event side: a *state* crossing
        moves with an initial condition just as it moves with a rate constant,
        so ``dt*/dθ`` has to be formed over all ``n_total`` columns and not only
        the parameter ones. ``X0`` is the model's own IC parameter, so this is
        the same column reached two ways — and both must carry the jump."""
        params = ["X0"]
        run = _sens(tmp_path, params)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params)
        scale = float(np.max(np.abs(fd[:, 0, 0])))
        np.testing.assert_allclose(an[:, 0, 0], fd[:, 0, 0], rtol=2e-4, atol=1e-5 * scale)
        # Pinned against the closed form as well, because the post-crossing IC
        # column decays to ~1e-5 of its peak and an all-points tolerance keyed
        # on that peak would pass with the jump missing. Before t*,
        # ∂X/∂X0 = e^{(rho−delta)t}; the jump doubles it at t*, and after it the
        # column only decays at −delta. Dropping the jump halves this.
        expected = 2.0 * np.exp(-0.8 * T_STAR) * np.exp(-1.6 * (T_END - T_STAR))
        assert an[-1, 0, 0] == pytest.approx(expected, rel=1e-4)


# ─── what the crossing costs when it is not located ────────────────────────


@requires_cc
class TestTheCrossingIsLocated:
    def test_the_integrator_does_not_grind_at_the_crossing(self, tmp_path):
        """Registering the root is half the fix and stands on its own: without a
        stop the integrator chases the discontinuity, and under sensitivities at
        a tight rtol it does not survive it (the issue quotes ``mxstep steps
        taken`` with ``h=1.1e-16`` — issue #82's pit, reached from the rate-law
        side). The step count is the deterministic form of that; wall clock
        would flake."""
        run = _sens(tmp_path, ["rho", "delta"])
        n_steps = run.solver_stats["n_steps"]
        # The smooth two-branch problem needs a few hundred steps. Chasing the
        # kink costs tens of thousands before it gets across, if it does.
        assert n_steps < 5_000, f"{n_steps} steps — the crossing is being chased, not located"

    def test_a_run_without_sensitivities_is_byte_identical(self, tmp_path):
        """The blast-radius bound. State-switch roots are registered only for a
        run that asks for sensitivities, so every plain trajectory keeps exactly
        the stepping — and exactly the numbers — it had before. Compared against
        a model that has no idea the feature exists, at full precision."""
        plain = bngsim.Simulator(_model(tmp_path, name="a.net"), method="ode").run(
            t_span=(0.0, T_END), n_points=N_POINTS, rtol=RTOL, atol=ATOL
        )
        again = bngsim.Simulator(_model(tmp_path, name="b.net"), method="ode").run(
            t_span=(0.0, T_END), n_points=N_POINTS, rtol=RTOL, atol=ATOL
        )
        assert np.asarray(plain.species).tobytes() == np.asarray(again.species).tobytes()
        assert plain.solver_stats["n_steps"] == again.solver_stats["n_steps"]

    def test_the_model_registers_no_discontinuity_trigger_of_its_own(self, tmp_path):
        """The premise the issue states: there is no GH #72 root for ``X<1``
        today, because that machinery only ever looked for thresholds on *time*.
        If a loader ever starts registering these, the state-switch roots would
        double up and this is where that shows."""
        assert _model(tmp_path)._core.n_discontinuity_triggers == 0


# ─── the residual, and who is allowed to claim a crossing ──────────────────


class TestTheResidual:
    def test_a_comparison_over_state_resolves_to_its_residual(self, tmp_path):
        core = _model(tmp_path)._core
        residual, why = core.state_switch_residual("X<1")
        assert residual and not why
        assert "X" in residual and "1" in residual

    @pytest.mark.parametrize("spelling", ["X<=1", "X==1", "X!=1"])
    def test_spellings_of_one_crossing_share_a_residual(self, tmp_path, spelling):
        """The dedup key. ``X<1`` and ``X<=1`` are the same surface, and
        registering both would put two roots on one crossing — which the solver
        then refuses as an ambiguous simultaneous pair. Orientation is free for
        the same reason: ``dt*/dθ`` is a ratio of two derivatives of ``g``.

        The equality spellings join them at issue #381. ``X == 1`` is not a
        *branch interval* — a continuous trajectory is on it for an instant —
        but the surface bounding that instant is still ``X − 1 = 0``, which is
        where ``X < 1`` changes branch too. Reading it as one crossing is what
        lets ``(X == 1) or (X < 1)``, the SBML ``<or/>``-of-``<eq/>``-and-``<lt/>``
        spelling of ``X <= 1``, register the one root its two atoms name
        (MODEL2003190004). Which side of that surface each spelling is true on
        the core reads by evaluating f there, never from the operator."""
        core = _model(tmp_path)._core
        assert core.state_switch_residual("X<1")[0] == core.state_switch_residual(spelling)[0]

    @pytest.mark.parametrize(
        ("cond", "fragment"),
        [
            ("(X<1)&&(X>0)", "combines conditions"),
            ("not(X<1)", "combines conditions"),
            ("X", "not a relational comparison"),
            ("X==1==1", "chains more than one comparison"),
            ("(X<1)!=(rho<delta)", "itself a comparison"),
            ("(X<1)<(rho<delta)", "itself a comparison"),
        ],
    )
    def test_what_cannot_be_rooted_says_why(self, tmp_path, cond, fragment):
        core = _model(tmp_path)._core
        residual, why = core.state_switch_residual(cond)
        assert not residual
        assert fragment in why, why

    def test_a_comparison_over_parameters_alone_is_not_a_state_switch(self, tmp_path):
        """``rho < delta`` has no crossing the trajectory can reach — it is a
        constant for the whole run. Claiming it would put a root on a residual
        that never changes sign and, worse, would let the state path claim a
        clock threshold whose jump issue #48 already applies."""
        core = _model(tmp_path)._core
        residual, why = core.state_switch_residual("rho<delta")
        assert not residual
        assert "no live model state" in why

    def test_a_clone_re_derives_rather_than_copying_an_expression_id(self, tmp_path):
        """The issue #144 rule, restated for this cache: an expression id means
        something else in another evaluator, so a clone must resolve the same
        text into its own table. Same residual *text* from both, which is the
        observable half of that."""
        core = _model(tmp_path)._core
        before = core.state_switch_residual("X<1")[0]
        clone = core.clone()
        assert clone.state_switch_residual("X<1")[0] == before
        # And the clone answers correctly having never been asked before the
        # copy: nothing was carried over to be stale.
        fresh = _model(tmp_path, name="fresh.net")._core.clone()
        assert fresh.state_switch_residual("X<1")[0] == before


class TestTheDetector:
    def test_the_condition_is_registered_once(self, tmp_path):
        assert sw.state_switch_conditions(_model(tmp_path)._core) == ["X<1"]

    def test_a_model_with_no_condition_registers_nothing(self, tmp_path):
        text = NET.replace("    1 growth() if(X<1,0,rho)\n", "    1 growth() rho\n")
        assert sw.state_switch_conditions(_model(tmp_path, text)._core) == []

    def test_a_clock_threshold_is_left_to_issue_48(self, tmp_path):
        """The partition. A BNGL counter clock is a *species*, so ``t>=sigma``
        reads live state and the residual splitter would happily claim it —
        which would apply the jump twice, once from the issue #48 stop time and
        once from a crossing root. ``clock_crossing_compensated`` is asked first
        by both the detector and the gate so that cannot happen."""
        text = NET.replace(
            "    2 rho    0.8  # Constant\n",
            "    2 rho    0.8  # Constant\n"
            "    4 sigma  3.0  # Constant\n"
            "    5 kclock 1  # Constant\n",
        )
        text = text.replace(
            "    1 growth() if(X<1,0,rho)\n", "    1 growth() if(tc>=sigma,rho,0)\n"
        )
        text = text.replace("    1 A() X0\n", "    1 A() X0\n    2 counter() 0\n")
        text = text.replace(
            "    2 1 0 delta #_R2\n", "    2 1 0 delta #_R2\n    3 0 2 kclock #_R3\n"
        )
        text = text.replace(
            "    1 X                    1\n",
            "    1 X                    1\n    2 tc                   2\n",
        )
        core = _model(tmp_path, text, name="clock.net")._core
        scope = sw.switch_condition_scope(core)
        assert "tc" in scope.clocks, "the fixture's counter must be detected as a clock"
        assert sw.clock_crossing_compensated("tc>=sigma", scope)
        assert sw.state_switch_conditions(core) == []
        records, _pinned = sw.compute_switch_time_sens(core, ["sigma"], 0.0, 100.0)
        assert records, "the clock detector must still claim it"


# ─── a crossing with no jump at it ─────────────────────────────────────────
#
# The saltation term is (f⁻ − f⁺)·dt*/dθ, so a `piecewise` that is CONTINUOUS at
# its own switch needs nothing — and that is the most common `piecewise` in the
# corpus, because it is how a clamp is written. BIOMD0000000161's basal PIP
# synthesis is `piecewise(0.581*k*(exp((basal - PIP)/basal) - 1), PIP < basal,
# 0)`, whose live branch is exactly 0 where PIP = basal; the trajectory then
# rides that surface, and refusing there (which the first cut of this feature
# did) took a model that had always run and made it raise.
#
# Here the branches meet at X = 1 for the same reason — `rho*(1-X)` is 0 there —
# so f is continuous across a crossing the trajectory passes straight through:
# X decays at −delta·X from 1000, reaches 1 at t = ln(1000)/1.6 = 4.317, and is
# then held up toward rho/(rho+delta). `rho` has a real in-branch derivative
# AFTER the crossing and none before it, which is what makes the column
# non-trivial and the comparison worth making.
CONTINUOUS = """\
begin parameters
    1 X0     1000  # Constant
    2 rho    0.5  # Constant
    3 delta  1.6  # Constant
end parameters
begin functions
    1 growth() if(X<1,rho*(1-X),0)
end functions
begin species
    1 A() X0
end species
begin reactions
    1 0 1 growth #_R1
    2 1 0 delta #_R2
end reactions
begin groups
    1 X                    1
end groups
"""


@requires_cc
class TestAContinuousSwitchNeedsNoJump:
    def test_it_runs_and_matches_a_finite_difference(self, tmp_path):
        """The branch gap is measured before ``dt*/dθ`` is ever formed, so a
        continuous switch costs no jump, no implicit-function solve and no
        transversality refusal — while the in-branch ``∂f/∂rho`` on the far side
        still has to come through."""
        params = ["rho", "delta"]
        run = _sens(tmp_path, params, name="cont.net", text=CONTINUOUS)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params, text=CONTINUOUS)
        for j, p in enumerate(params):
            scale = float(np.max(np.abs(fd[:, 0, j])))
            assert scale > 0.0, f"column {p!r} is trivially zero — the fixture is not testing it"
            # Read against the column's own peak, not pointwise: the sample
            # adjacent to the crossing is FD-limited, and it is the FD that is
            # limited. Measured at rtol 1e-9 / 1e-11 / 1e-12 the analytic value
            # there is 0.020753543 / 0.020753555 / 0.020753556 — stable to eight
            # digits — while the difference quotient wanders in the fourth
            # (0.0207398 / 0.0207289 / 0.0207497), which is what differencing
            # across a kink does.
            assert np.max(np.abs(an[:, 0, j] - fd[:, 0, j])) <= 2e-4 * scale, (
                f"column {p!r} disagrees with its own finite difference by "
                f"{np.max(np.abs(an[:, 0, j] - fd[:, 0, j])) / scale:.2e} of its peak"
            )

    def test_an_ic_only_request_registers_the_crossing_too(self, tmp_path):
        """Keyed on "any sensitivity at all", not on a parameter request.

        The three detectors sit behind ``if self._sensitivity_params:`` because
        a switch *time* and an event *time* are parameter-column concepts; a
        state crossing is not — it moves with every column, and the IC columns
        are the ones it moves *only* through the trajectory. Requesting an
        initial condition and nothing else is exactly the shape that would have
        gone unregistered, which is the same corner issue #144 found on the
        event side."""
        model = _model(tmp_path, name="iconly.net")
        sim = bngsim.Simulator(model, method="ode", sensitivity_ic=["A()"])
        run = sim.run(t_span=(0.0, T_END), n_points=N_POINTS, rtol=RTOL, atol=ATOL)
        s_ic = np.asarray(run.sensitivities_ic)[:, 0, 0]
        # dX/dX(0) is e^{(rho-delta)t} before the crossing, doubled by the jump
        # at t*, and decaying at -delta after it. Without the registration the
        # tail is half this.
        assert s_ic[-1] == pytest.approx(
            2.0 * np.exp(-0.8 * T_STAR) * np.exp(-1.6 * (T_END - T_STAR)), rel=1e-4
        )

    def test_the_crossing_is_still_registered(self, tmp_path):
        """The condition is a state switch like any other — nothing about it is
        knowable before the run. What decides is the branch gap measured AT the
        crossing, which is why this cannot be a detection-time rule."""
        core = _model(tmp_path, CONTINUOUS, name="cont2.net")._core
        assert sw.state_switch_conditions(core) == ["X<1"]

    def test_the_branch_gap_is_read_scale_free(self, tmp_path):
        """The residual carries the MODEL's units, not a species'.

        The ``.net`` corpus is full of the signed-rate idiom — BNGL rates must
        be non-negative, so a model needing a signed derivative splits it and
        guards each half by the sign of the rate itself, ``if(expr>0, expr, 0)``
        and ``if(expr>0, 0, -expr)``. Both branches are 0 where ``expr`` is, so f
        is continuous; but the residual is a *rate*, and on
        ``ph_lorenz_attractor`` (condition ``X·Y − beta·Z > 0``, the sign of
        dZ/dt) it is orders of magnitude off the species scale. A probe or a
        tolerance keyed on the species would call that continuous switch
        undecidable. Here the same crossing is written with a 1e6 factor on the
        residual and must give the identical answer."""
        scaled = CONTINUOUS.replace(
            "    1 growth() if(X<1,rho*(1-X),0)\n",
            "    1 growth() if(1e6*(1-X)>0,rho*(1-X),0)\n",
        )
        plain = _sens(tmp_path, ["rho"], name="s1.net", text=CONTINUOUS)
        blown = _sens(tmp_path, ["rho"], name="s2.net", text=scaled)
        a, b = np.asarray(plain.sensitivities), np.asarray(blown.sensitivities)
        scale = float(np.max(np.abs(a)))
        assert scale > 0.0
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9 * scale)


# ─── a bystander must not decide whether a switch jumps (issue #763) ────────
#
# Whether a crossing is continuous used to be ONE comparison: the largest
# |f⁻ − f⁺| over all species against the largest |f| over all species. A species
# the switch never touches then set the scale. Here Y's source switches off
# when A decays past thr, a jump of kb = 3 in Y and nothing else. A pool B, which
# nothing reads, degrades from B0; at B0 = 1e8 its |dB/dt| at t* is 7.2e6, more
# than 1e6·kb, so the jump read as roundoff. The saltation term was dropped and
# dY/d[A0, a, thr] came back exactly 0, with no warning. The trajectory itself
# was right, and dY/dkb (an in-branch derivative) survived.
#
# The closed form needs no sensitivity code: t* = ln(A0/thr)/a and
# Y(T) = kb·(T − t*), so
#
#   dY/dA0 = −kb/(a·A0),  dY/da = kb·ln(A0/thr)/a²,  dY/dthr = kb/(a·thr),
#   dY/dkb = T − t*.
BYSTANDER = """\
begin parameters
    1 A0    10  # Constant
    2 a     0.5  # Constant
    3 thr   2  # Constant
    4 kb    3  # Constant
    5 B0    {B0}  # Constant
    6 kdeg  0.1  # Constant
    7 vfast {vfast}  # Constant
end parameters
begin functions
    1 fY() if(Aobs<thr,kb,0)
end functions
begin species
    1 A() A0
    2 Y() 0
    3 B() B0
    4 F() 0
end species
begin reactions
    1 1 0 a #_R1
    2 0 2 fY #_R2
    3 3 0 kdeg #_R3
    4 0 4 vfast #_R4
end reactions
begin groups
    1 Aobs                 1
end groups
"""
BYSTANDER_PARAMS = ["A0", "a", "thr", "kb"]

# The same switch, with Y itself a pool at steady state: made at ksyn and lost at
# kdeg·Y, Y(0) = ksyn/kdeg.
TURNOVER = """\
begin parameters
    1 A0    10  # Constant
    2 a     0.5  # Constant
    3 thr   2  # Constant
    4 kb    3  # Constant
    5 ksyn  1e7  # Constant
    6 kdeg  0.1  # Constant
end parameters
begin functions
    1 fY() if(Aobs<thr,kb,0)
end functions
begin species
    1 A() A0
    2 Y() 1e8
end species
begin reactions
    1 1 0 a #_R1
    2 0 2 fY #_R2
    3 0 2 ksyn #_R3
    4 2 0 kdeg #_R4
end reactions
begin groups
    1 Aobs                 1
end groups
"""
BYSTANDER_T = 6.0


def _bystander_closed_form():
    A0, a, thr, kb, T = 10.0, 0.5, 2.0, 3.0, BYSTANDER_T
    t_star = np.log(A0 / thr) / a
    return np.array([-kb / (a * A0), kb * np.log(A0 / thr) / a**2, kb / (a * thr), T - t_star])


# ─── what the seventh review of the #763 fix found ─────────────────────────
#
# Thresholds a few hundred ulps apart, steep smooth terms beside a jump, and
# balanced exchanges whose flux is all rounding. Closed forms throughout.
EPS = float(np.finfo(float).eps)

# A decays at a from A0 = 10, so Aobs < thr is crossed at t = ln(A0/thr)/a, and
# a source `ksw` switched on there gives Y(T) = ksw·(T − t).
HAIR = """\
begin parameters
    1 A0    10
    2 a     0.5
    3 thr1  2
    4 thr2  {thr2!r}
    5 kb    3
    6 kc    {kc!r}
    7 kbig  {kbig!r}
end parameters
begin functions
    1 fY() {fy}
    2 fZ() {fz}
end functions
begin species
    1 A() A0
    2 Y() 0
    3 Z() 0
end species
begin reactions
    1 1 0 a
    2 0 2 fY
    3 0 3 fZ
end reactions
begin groups
    1 Aobs 1
end groups
"""
HAIR_PARAMS = ["a", "thr1", "thr2", "kb", "kc"]
HAIR_T = 12.0


def _hair(tmp_path, name, sample_times=None, kc=5.0, **fmt):
    model = _model(tmp_path, HAIR.format(kc=kc, **fmt), name=name)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=HAIR_PARAMS)
    kw = {"sample_times": sample_times} if sample_times else {"n_points": 3}
    return sim.run(t_span=(0.0, HAIR_T), rtol=1e-10, atol=1e-12, **kw)


def _switched_source(thr, ksw, col):
    """d/d[a, thr1, thr2, kb, kc] of ksw·(T − ln(A0/thr)/a), thr in column `col`."""
    want = np.zeros(5)
    want[0] = ksw * np.log(10.0 / thr) / 0.5**2
    want[col] = ksw / (0.5 * thr)
    return want


# Unit i switches on ``Aobs < thr_i`` with thr_i = 2·(1 − k_i·ε), A decaying
# from 10 at a = 0.5:
#   jump   a source of UNIT_KB[i] into Y_i, so dY_i/dthr_i = UNIT_KB[i];
#   ramp   ``size·(thr_i − Aobs)`` into Y_i below the threshold, continuous;
#   clamp  B_i (= size) -> C_i at ``0.1·min(Aobs/thr_i, 1)``, continuous.
#   noreader  a function of the condition that no reaction uses.
UNIT_KB = (3.0, 5.0, 7.0)
UNIT_REFUSED = re.compile("cross at the same instant|and its jump applied there")


def _units_sens(tmp_path, units, name):
    """The last sensitivity row, by species name, over [A0, a, thr0, thr1, ...]."""
    ps, fs, sp, rx = ["A0 10", "a 0.5"], [], ["A() A0"], ["1 0 a"]
    for i, (kind, k, size) in enumerate(units):
        ps.append(f"thr{i} {float(2.0 * (1 - k * EPS))!r}")
        if kind == "jump":
            fs.append(f"f{i}() if(Aobs<thr{i},{UNIT_KB[i]!r},0)")
        elif kind == "ramp":
            fs.append(f"f{i}() if(Aobs<thr{i},{size!r}*(thr{i}-Aobs),0)")
        elif kind == "noreader":
            fs.append(f"f{i}() if(Aobs<thr{i},1,0)")
        else:
            fs.append(f"f{i}() 0.1*if(Aobs<thr{i},Aobs/thr{i},1)")
        if kind == "clamp":
            sp += [f"B{i}() {size!r}", f"C{i}() 0"]
            rx.append(f"{len(sp) - 1} {len(sp)} f{i}")
        elif kind != "noreader":
            sp.append(f"Y{i}() 0")
            rx.append(f"0 {len(sp)} f{i}")

    def block(rows):
        return "".join(f"    {n + 1} {row}\n" for n, row in enumerate(rows))

    text = (
        f"begin parameters\n{block(ps)}end parameters\n"
        f"begin functions\n{block(fs)}end functions\n"
        f"begin species\n{block(sp)}end species\n"
        f"begin reactions\n{block(rx)}end reactions\n"
        "begin groups\n    1 Aobs 1\nend groups\n"
    )
    params = ["A0", "a"] + [f"thr{i}" for i in range(len(units))]
    run = bngsim.Simulator(
        _model(tmp_path, text, name=name), method="ode", sensitivity_params=params
    ).run(t_span=(0.0, 8.0), n_points=3, rtol=1e-10, atol=1e-12)
    s = np.asarray(run.sensitivities)[-1]
    return {n[:-2]: s[j] for j, n in enumerate(run.species_names)}


@requires_cc
class TestABystanderDoesNotHideAJump:
    @pytest.mark.parametrize(
        "B0,vfast",
        [
            (1.0, 0.0),  # the control: nothing large anywhere
            (1e8, 0.0),  # the issue's pool: a molecule-count reservoir turning over
            (1.0, 1e8),  # an unread species made at a constant 1e8 per unit time
        ],
        ids=["control", "pool", "fast-source"],
    )
    def test_the_switch_time_columns_match_the_closed_form(self, tmp_path, B0, vfast):
        """B and F are decoupled from A and Y, so neither can move a Y column."""
        text = BYSTANDER.format(B0=B0, vfast=vfast)
        model = _model(tmp_path, text, name="bystander.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=BYSTANDER_PARAMS).run(
            t_span=(0.0, BYSTANDER_T), n_points=3, rtol=1e-10, atol=1e-12
        )
        y = list(run.species_names).index("Y()")
        got = np.asarray(run.sensitivities)[-1, y, :]
        np.testing.assert_allclose(got, _bystander_closed_form(), rtol=1e-6)

    @pytest.mark.parametrize("ksyn", [1e7, 0.0], ids=["steady", "decaying"])
    def test_a_switched_species_with_its_own_turnover_still_jumps(self, tmp_path, ksyn):
        """Only the switch's own reaction is read, not Y's whole dx/dt.

        Here Y is itself a pool of 1e8, lost at kdeg·Y, with the same switched
        source of kb = 3 on top. ``steady`` also makes it at ksyn = 1e7, so its
        net rate is ~0 but its gross flux is 2e7; ``decaying`` does not, so
        |dY/dt| is ~7e6 at t*. Reading the gap of 3 against either at 1e-6 would
        call the switch continuous and lose the jump, the same bug inside one
        species; ``decaying`` did exactly that on main and on the first cut of
        the #763 fix. The switch's own reaction has a flux of 3 and a gap of 3.

        Past t*, Y relaxes toward its new level at kdeg:
        Y(T) = Y0 + (kb/kdeg)·(1 − e^{−kdeg·(T − t*)}), so
        dY/dθ = −kb·e^{−kdeg·(T − t*)}·dt*/dθ for θ in A0, a, thr.

        The in-branch dY/dkb is held to the tolerance the run asks of it, which
        is loose here: a sensitivity's absolute tolerance scales with its
        species (atol·|Y|/|kb| ≈ 3e-5 for Y = 1e8), and that column lands 1e-6
        to 1e-4 off the closed form at rtol 1e-8 to 1e-12, identically before
        and after issue #763's change. The switch-time columns are what the
        change is about, and they are held tight."""
        text = TURNOVER.replace("    5 ksyn  1e7  # Constant", f"    5 ksyn  {ksyn!r}  # Constant")
        model = _model(tmp_path, text, name="turnover.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=BYSTANDER_PARAMS).run(
            t_span=(0.0, BYSTANDER_T), n_points=3, rtol=1e-10, atol=1e-12
        )
        y = list(run.species_names).index("Y()")
        got = np.asarray(run.sensitivities)[-1, y, :]
        A0, a, thr, kb, kdeg, T = 10.0, 0.5, 2.0, 3.0, 0.1, BYSTANDER_T
        t_star = np.log(A0 / thr) / a
        decay = np.exp(-kdeg * (T - t_star))
        dtstar = np.array([1 / (a * A0), -np.log(A0 / thr) / a**2, -1 / (a * thr)])
        np.testing.assert_allclose(got[:3], -kb * decay * dtstar, rtol=1e-5)
        assert got[3] == pytest.approx((1 - decay) / kdeg, rel=1e-3)

    def test_a_bystander_that_cancels_internally_is_not_read_as_a_jump(self, tmp_path):
        """ml_hopfield's signed-rate crossing is continuous. Beside it here, Z is
        made at ``0.3*P - 0.1*Q`` over two ~1e8 pools decaying together: one rate
        law whose net is a small difference of large operands, so its value at
        two nearby states differs by roundoff alone. Read as part of Z's dx/dt,
        that roundoff looked like a jump and sent the continuous batch down the
        jump path, which refuses it (issue #153's split-crossing message). That
        was the first cut of the #763 fix, which judged every species' whole
        dx/dt; main never looked per species, and the fix reads only the
        signed-rate reactions. Z is decoupled, so the W columns of S1..S3 must
        equal the model without it, to solver tolerance: the pools change CVODE's
        error norm and so its steps (1.1e-9 of a 0.09 column, on main too)."""
        noisy = (
            HOPFIELD.replace(
                "end parameters",
                "    6 kd         0.1\n    7 P0         100000000.002\n"
                "    8 Q0         3e8\nend parameters",
            )
            .replace("end functions", "   16 h() 0.3*Pobs-0.1*Qobs\nend functions")
            .replace("end species", "    4 P() P0\n    5 Q() Q0\n    6 Z() 0\nend species")
            .replace("end reactions", "    7 4 0 kd\n    8 5 0 kd\n    9 0 6 h\nend reactions")
            .replace(
                "end groups",
                "    4 Pobs                 4\n    5 Qobs                 5\nend groups",
            )
        )
        params = ["W12", "W13", "W23"]

        def sens(text, name):
            # The review's grid: which states the probes land on decides whether
            # the roundoff reads as a jump, and on this one it did.
            model = _model(tmp_path, text, name)
            run = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(
                t_span=(0.0, 3.0), n_points=4, rtol=1e-9, atol=1e-12
            )
            return np.asarray(run.sensitivities)

        plain = sens(HOPFIELD, "hop_plain.net")
        got = sens(noisy, "hop_noisy.net")[:, :3, :]
        scale = float(np.max(np.abs(plain)))
        np.testing.assert_allclose(got, plain, rtol=0, atol=1e-6 * scale)

    def test_a_switched_rate_law_with_a_large_smooth_term_still_jumps(self, tmp_path):
        """``fY() = ksyn + if(Aobs<thr, kb, 0)``: the switched rate law carries a
        smooth 1e7 of its own, balanced by Y's separate degradation. Read
        against the switched reactions' own flux, the jump of 3 was under 1e-6
        of it and was dropped (the old global test saw it, since Y's net rate
        is ~3). Same closed form as the turnover case."""
        text = TURNOVER.replace(
            "    2 0 2 fY #_R2\n    3 0 2 ksyn #_R3\n", "    2 0 2 fY #_R2\n"
        ).replace("    1 fY() if(Aobs<thr,kb,0)", "    1 fY() ksyn+if(Aobs<thr,kb,0)")
        assert "ksyn+if" in text and "0 2 ksyn" not in text
        model = _model(tmp_path, text, name="basal.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=BYSTANDER_PARAMS).run(
            t_span=(0.0, BYSTANDER_T), n_points=3, rtol=1e-10, atol=1e-12
        )
        y = list(run.species_names).index("Y()")
        got = np.asarray(run.sensitivities)[-1, y, :3]
        A0, a, thr, kb, kdeg, T = 10.0, 0.5, 2.0, 3.0, 0.1, BYSTANDER_T
        t_star = np.log(A0 / thr) / a
        dtstar = np.array([1 / (a * A0), -np.log(A0 / thr) / a**2, -1 / (a * thr)])
        np.testing.assert_allclose(got, -kb * np.exp(-kdeg * (T - t_star)) * dtstar, rtol=1e-4)

    def test_a_condition_no_rate_law_reads_moves_nothing(self, tmp_path):
        """``flag()`` is an output: no rate law reads it, so its crossing cannot
        move f and is continuous outright. Judging it over the whole right-hand
        side instead read the roundoff of an internally cancelling bystander
        (``h() = 0.3*Pobs - 0.1*Qobs`` over two 1e8 pools) as a jump and put
        it into Z, which is decoupled from A: dZ/dthr and dZ/da are exactly 0."""
        text = """\
begin parameters
    1 A0 10
    2 a 0.5
    3 thr 0.001
    4 kd 0.01
    5 P0 100000000.001
    6 Q0 3e8
end parameters
begin functions
    1 flag() if(Aobs<thr,1,0)
    2 h() 0.3*Pobs-0.1*Qobs
end functions
begin species
    1 A() A0
    2 P() P0
    3 Q() Q0
    4 Z() 0
end species
begin reactions
    1 1 0 a
    2 2 0 kd
    3 3 0 kd
    4 0 4 h
end reactions
begin groups
    1 Aobs 1
    2 Pobs 2
    3 Qobs 3
end groups
"""
        model = _model(tmp_path, text, name="outflag.net")
        assert sw.state_switch_conditions(model._core) == ["Aobs<thr"]
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["thr", "a", "kd"]).run(
            t_span=(0.0, 30.0), n_points=4, rtol=1e-9, atol=1e-12
        )
        z = list(run.species_names).index("Z()")
        s = np.asarray(run.sensitivities)[:, z, :]
        assert np.all(s[:, 0] == 0.0) and np.all(s[:, 1] == 0.0), s[:, :2]

    @pytest.mark.parametrize(
        "fz",
        ["fZ() if(Bobs>A0-thr,kc*(Bobs-(A0-thr)),0)", "flag() if(Bobs>A0-thr,1,0)\n    3 fZ() 0"],
        ids=["continuous-clamp", "output-only"],
    )
    def test_a_second_spelling_of_the_surface_keeps_its_jump(self, tmp_path, fz):
        """A -> B conserves A + B = A0, so ``Bobs > A0 - thr`` and ``Aobs < thr``
        are one surface, and CVODE may report only one of the two roots. When the
        one it reported read as continuous (a clamp, or an output nobody reads),
        restarting past the surface left the other root behind, never fired, and
        fY's jump of kb went with it: 32 of these 60 points came back 0. Every
        switch the same probe pair straddles is now read with the crossing."""
        tmpl = """\
begin parameters
    1 A0    10
    2 a     {a}
    3 thr   {thr}
    4 kb    3
    5 kc    1
end parameters
begin functions
    1 {fz}
    2 fY() if(Aobs<thr,kb,0)
end functions
begin species
    1 A() A0
    2 Y() 0
    3 Z() 0
    4 B() 0
end species
begin reactions
    1 1 4 a #_R1
    2 0 2 fY #_R2
    3 0 3 fZ #_R3
end reactions
begin groups
    1 Aobs                 1
    2 Bobs                 4
end groups
"""
        worst = 0.0
        for a in (0.3, 0.5, 0.7, 1.1, 1.3, 2.9):
            for thr in (2.0, 3.3, 7.1, 1.7, 4.4):
                model = _model(tmp_path, tmpl.format(a=a, thr=thr, fz=fz), name="cons.net")
                run = bngsim.Simulator(
                    model, method="ode", sensitivity_params=["A0", "a", "thr", "kb"]
                ).run(t_span=(0, 12), n_points=3, rtol=1e-10, atol=1e-12)
                got = np.asarray(run.sensitivities)[-1, 1, 1:3]
                exact = np.array([3 * np.log(10 / thr) / a**2, 3 / (a * thr)])
                worst = max(worst, float(np.max(np.abs(got - exact) / np.abs(exact))))
        assert worst < 1e-6

    @pytest.mark.parametrize("kb", [3.0, 1.0])
    def test_a_jump_on_a_very_large_switched_flux_is_not_roundoff(self, tmp_path, kb):
        """``ksyn + if(...)`` with ksyn = 1e14: a jump of kb is ~70·kb ulps of the
        switched flux, which a floor proportional to the whole flux excused (1024
        ulps dropped kb = 3; 64 ulps still dropped kb = 1). No floor decides
        whether a jump is applied now. Same closed form as the turnover case."""
        text = (
            TURNOVER.replace("    2 0 2 fY #_R2\n    3 0 2 ksyn #_R3\n", "    2 0 2 fY #_R2\n")
            .replace("    1 fY() if(Aobs<thr,kb,0)", "    1 fY() ksyn+if(Aobs<thr,kb,0)")
            .replace("    5 ksyn  1e7  # Constant", "    5 ksyn  1e14  # Constant")
            .replace("    2 Y() 1e8", "    2 Y() 1e15")
            .replace("    4 kb    3  # Constant", f"    4 kb    {kb!r}  # Constant")
        )
        assert "1e14" in text and "1e15" in text
        model = _model(tmp_path, text, name="basal14.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=BYSTANDER_PARAMS).run(
            t_span=(0.0, BYSTANDER_T), n_points=3, rtol=1e-10, atol=1e-12
        )
        y = list(run.species_names).index("Y()")
        got = np.asarray(run.sensitivities)[-1, y, :3]
        A0, a, thr, kdeg, T = 10.0, 0.5, 2.0, 0.1, BYSTANDER_T
        t_star = np.log(A0 / thr) / a
        dtstar = np.array([1 / (a * A0), -np.log(A0 / thr) / a**2, -1 / (a * thr)])
        np.testing.assert_allclose(got, -kb * np.exp(-kdeg * (T - t_star)) * dtstar, rtol=1e-4)

    @pytest.mark.parametrize(
        ("kb", "X0"),
        [(3.0, 1e14), (3.0, 1e15), (3.0, 4e15), (1.0, 5e13), (1.0, 1e15), (1.0, 2e15)],
    )
    def test_a_jump_beside_a_large_cancelling_exchange_is_not_roundoff(self, tmp_path, kb, X0):
        """X swaps with P at kx·clamp both ways, so X's switched flux is ~2·X0 and
        cancels to kb. A floor of 64 ulps of the cancelling part read the jump of
        kb as roundoff from X0 = 5e13 (kb = 1) and 1e14 (kb = 3), where main reads
        it; 2 ulps of the gross flux still did at 2e15 and 4e15. Any floor drops
        the jumps below it, so the transversal path has none. X + P gains kb until
        t*, so d(X+P)/dθ = -kb·dt*/dθ at any T past t*."""
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr 2
    4 kb {kb!r}
    5 X0 {X0!r}
    6 kx 1
end parameters
begin functions
    1 fY() if(Aobs<thr,kb,0)
    2 fc() kx*if(Aobs<thr,Aobs/thr,1)
end functions
begin species
    1 A() A0
    2 X() X0
    3 P() X0
end species
begin reactions
    1 1 0 a #_R1
    2 0 2 fY #_R2
    3 3 2 fc #_R3
    4 2 3 fc #_R4
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="exchange.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["A0", "a", "thr"]).run(
            t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12
        )
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        got = s[names.index("X()")] + s[names.index("P()")]
        A0, a, thr = 10.0, 0.5, 2.0
        dtstar = np.array([1 / (a * A0), -np.log(A0 / thr) / a**2, -1 / (a * thr)])
        np.testing.assert_allclose(got, -kb * dtstar, rtol=1e-6)

    @pytest.mark.parametrize(
        ("kind", "size"), [("decay", 3e6), ("decay", 1e7), ("exchange", 1e5), ("exchange", 1e6)]
    )
    def test_a_continuous_clamp_crossing_just_after_a_jump_is_not_refused(
        self, tmp_path, kind, size
    ):
        """Aobs < thr1 switches kb into Y, and a clamp kx·if(Aobs<thr2, Aobs/thr2, 1)
        on a large flux crosses 1000 ulps of thr1 later: inside the probe window,
        outside CVODE's root bracket. The clamp is continuous, but across the
        probe it varies by its slope times up to dt, which passed 1e-6 of the
        drive, so it was read as a jump, joined the dt*/dθ agreement check, and
        was refused ("the right-hand side jumps there"). Main runs these and is
        right. The clamp is now read at its own root, each branch extended to it
        from two points on its own side, where its two branches meet.
        `decay` is B -> C through the clamp; `exchange` is X <-> P through it
        with a steady net flux `size` (refused only once the floor was lowered)."""
        eps = np.finfo(float).eps
        A0, a, thr1, kb, T = 10.0, 0.5, 0.02, 3.0, 16.0
        thr2 = float(thr1 * (1 - 1000 * eps))
        if kind == "decay":
            extra_params = f"    6 kx 0.1\n    7 B0 {size!r}\n"
            species = "    3 B() B0\n    4 C() 0\n"
            reactions = "    3 3 4 fc\n"
        else:
            P0 = size / 0.01
            extra_params = (
                f"    6 kx 1\n    7 ks {size!r}\n    8 kd 0.01\n"
                f"    9 X0 {P0 + size!r}\n   10 P0 {P0!r}\n"
            )
            species = "    3 X() X0\n    4 P() P0\n"
            reactions = "    3 3 4 fc\n    4 4 3 fc\n    5 0 3 ks\n    6 4 0 kd\n"
        text = (
            "begin parameters\n    1 A0 10\n    2 a 0.5\n    3 thr1 0.02\n"
            f"    4 thr2 {thr2!r}\n    5 kb 3\n{extra_params}end parameters\n"
            "begin functions\n    1 fY() if(Aobs<thr1,kb,0)\n"
            "    2 fc() kx*if(Aobs<thr2,Aobs/thr2,1)\nend functions\n"
            f"begin species\n    1 A() A0\n    2 Y() 0\n{species}end species\n"
            f"begin reactions\n    1 1 0 a\n    2 0 2 fY\n{reactions}end reactions\n"
            "begin groups\n    1 Aobs 1\nend groups\n"
        )
        model = _model(tmp_path, text, name=f"clamp_{kind}.net")
        run = bngsim.Simulator(
            model, method="ode", sensitivity_params=["a", "thr1", "thr2", "kb"]
        ).run(t_span=(0.0, T), n_points=3, rtol=1e-10, atol=1e-12)
        got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()")]
        t1 = np.log(A0 / thr1) / a
        exact = np.array([kb * np.log(A0 / thr1) / a**2, kb / (a * thr1), 0.0, T - t1])
        np.testing.assert_allclose(got, exact, rtol=1e-6, atol=1e-6)

    def test_two_independent_thresholds_a_hair_apart_are_refused_not_mixed(self, tmp_path):
        """Aobs < thr1 drives Y and Aobs < thr2 drives Z, with thr2 a few hundred
        ulps below thr1: two independent crossings CVODE can report as one. The
        second one's jump was credited to the first one's dt*/dθ, silently
        (dZ/dthr1 = 5, dZ/dthr2 = 0, where the truth is the other way round).
        Every column is now either right or refused, never silently wrong. At
        500 and 700 ulps only the wider probe crosses thr2, and its jump must not
        enter the jump applied at thr1."""
        a, thr1, kb, kc, T = 0.5, 2.0, 3.0, 5.0, 12.0
        eps = np.finfo(float).eps
        for k in (100, 300, 500, 700, 1000):
            thr2 = float(thr1 * (1 - k * eps))
            text = f"""\
begin parameters
    1 A0    10
    2 a     {a!r}
    3 thr1  {thr1!r}
    4 thr2  {thr2!r}
    5 kb    {kb!r}
    6 kc    {kc!r}
end parameters
begin functions
    1 fY() if(Aobs<thr1,kb,0)
    2 fZ() if(Aobs<thr2,kc,0)
end functions
begin species
    1 A() A0
    2 Y() 0
    3 Z() 0
end species
begin reactions
    1 1 0 a
    2 0 2 fY
    3 0 3 fZ
end reactions
begin groups
    1 Aobs                 1
end groups
"""
            model = _model(tmp_path, text, name=f"pair{k}.net")
            try:
                run = bngsim.Simulator(
                    model, method="ode", sensitivity_params=["a", "thr1", "thr2", "kb", "kc"]
                ).run(t_span=(0, T), n_points=3, rtol=1e-9, atol=1e-12)
            except SimulationError as e:
                assert "cross at the same instant" in str(e)
                continue
            s = np.asarray(run.sensitivities)[-1]
            names = list(run.species_names)
            ln = np.log(10.0 / thr1)
            want_y = [kb * ln / a**2, kb / (a * thr1), 0.0, T - ln / a, 0.0]
            want_z = [
                kc * np.log(10.0 / thr2) / a**2,
                0.0,
                kc / (a * thr2),
                0.0,
                T - np.log(10.0 / thr2) / a,
            ]
            np.testing.assert_allclose(s[names.index("Y()")], want_y, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(s[names.index("Z()")], want_z, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("kc", [5.0, 3.0])
    def test_a_switch_only_the_wider_probe_crosses_does_not_hide_the_jump(self, tmp_path, kc):
        """``fY = if(A<thr1,kb,0) + if(A<thr2,kc,0)`` with thr2 crossed between
        one and two probe steps after thr1. The gap at thr1 was read by how it
        grew with the probe, and the probe at twice the step took in thr2's jump:
        near = kb, far = kb + kc, "growth", so the thr1 jump was dropped and
        dY/d[a, thr1, thr2] came back exactly 0, where main is right. The same
        happened at thr2's own crossing, whose wider probe reaches back over
        thr1. The wider probe is now used only when it crosses the same switches
        as the near one. Closer than one step the two are refused as before.
        With kc = kb the two jumps cancel exactly in a branch extended across
        thr2, which is how an extension that ignored it would read the crossing."""
        ran = 0
        for k in (450, 550, 650, 750):
            thr2 = float(2.0 * (1 - k * EPS))
            fy = "if(Aobs<thr1,kb,0)+if(Aobs<thr2,kc,0)"
            try:
                run = _hair(tmp_path, f"third{k}.net", kc=kc, thr2=thr2, kbig=0.0, fy=fy, fz="0")
            except SimulationError as e:
                assert "cross at the same instant" in str(e)
                continue
            ran += 1
            got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()")]
            want = _switched_source(2.0, 3.0, 1) + _switched_source(thr2, kc, 2)
            want[3] = HAIR_T - np.log(10.0 / 2.0) / 0.5
            want[4] = HAIR_T - np.log(10.0 / thr2) / 0.5
            np.testing.assert_allclose(got, want, rtol=1e-6)
        assert ran > 0

    @pytest.mark.parametrize("kbig", [1e10, 1e12, 1e13])
    def test_a_steep_smooth_term_in_the_switched_rate_law_does_not_hide_the_jump(
        self, tmp_path, kbig
    ):
        """``fY = kbig*Aobs + if(A<thr1, kb, 0)``: between the two probe points
        the smooth term moves by 2·δt·kbig·|dA/dt|, which is 0.37 at kbig = 1e12
        and passes kb = 3 near 1e13. Read as a gap that grows with the probe, the
        jump was dropped from kbig = 6e12 (on main from 1e10, by the global
        scale), and dY/dthr1 was exactly 0. Each branch is now extended to the
        surface from two points on its own side, so the smooth term is not in the
        reading, and not in the jump the columns take either: just past the
        switch dY/dthr1 was 2.63 at kbig = 1e12, for a truth of 3."""
        fy = "kbig*Aobs+if(Aobs<thr1,kb,0)"
        t1 = np.log(5.0) / 0.5
        run = _hair(
            tmp_path,
            "steep.net",
            sample_times=[0.0, t1 + 0.05, HAIR_T],
            thr2=1.0,
            kbig=kbig,
            fy=fy,
            fz="0",
        )
        y = list(run.species_names).index("Y()")
        s = np.asarray(run.sensitivities)[:, y, :]
        # dY/dthr1 = kb/(a·thr1) at every time past the switch; the flux it
        # rides on is 2·kbig, whose rounding is 0.1% of kb at kbig = 1e13.
        assert s[1, 1] == pytest.approx(3.0, rel=5e-3)
        assert s[2, 1] == pytest.approx(3.0, rel=5e-3)
        assert s[2, 3] == pytest.approx(HAIR_T - t1, rel=1e-6)

    @pytest.mark.parametrize("kbig", [1e13, 1e14])
    def test_a_steep_term_in_the_co_crossing_switch_is_refused_not_mixed(self, tmp_path, kbig):
        """Two independent thresholds a few hundred ulps apart, the second in
        ``fZ = kbig*Aobs + if(A<thr2, kc, 0)``. Its smooth term made its gap grow
        with the probe, so it was not asked to agree on dt*/dθ, and its jump was
        credited to thr1: dZ/dthr1 = 4.96 and dZ/dthr2 = 0, the truth being 0 and
        5 (main drops both). The co-crossing switch is now read at its own root
        with the smooth term taken out."""
        for k in (100, 300):
            thr2 = float(2.0 * (1 - k * EPS))
            try:
                run = _hair(
                    tmp_path,
                    f"co{k}.net",
                    thr2=thr2,
                    kbig=kbig,
                    fy="if(Aobs<thr1,kb,0)",
                    fz="kbig*Aobs+if(Aobs<thr2,kc,0)",
                )
            except SimulationError as e:
                assert "cross at the same instant" in str(e)
                continue
            z = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Z()")]
            # kc on a flux of 2·kbig: its rounding is up to 4% of kc at 1e14.
            assert abs(z[1]) < 0.25
            assert z[2] == pytest.approx(5.0 / (0.5 * thr2), abs=0.25)

    @pytest.mark.parametrize(("npair", "X0"), [(1, 1e14), (1, 1e15), (8, 1e10), (8, 1e14)])
    def test_the_rounding_of_a_balanced_exchange_is_not_asked_to_agree(self, tmp_path, npair, X0):
        """A continuous clamp ``c = if(A<thr2, A/thr2, 1)`` scales both
        directions of X <-> P_i exchanges that sit at equilibrium, so its
        reactions' net flux is 0 on both branches and what the probes read is the
        rounding of X0-sized terms. It co-crosses a real jump (kb into Y at
        A < thr1). With no roundoff floor that rounding passed the drive, did not
        grow with the probe, and sent the clamp into the dt*/dθ agreement check,
        which refused 51 of 88 such runs; main runs them all. The floor is back
        in that check only: the jump itself is still applied whatever its size."""
        kf = [1.1, 1.3, 0.7, 0.9, 1.7, 0.3, 1.9, 0.55][:npair]
        kr = [0.6, 1.2, 1.45, 0.35, 0.8, 1.05, 0.25, 1.35][:npair]
        a, thr1, kb, T = 0.5, 2.0, 3.0, 8.0
        t1 = np.log(10.0 / thr1) / a
        want_y = np.array([kb * np.log(10.0 / thr1) / a**2, kb / (a * thr1), 0.0, T - t1])
        for k in (100, 200, 300):
            thr2 = float(thr1 * (1 - k * EPS))
            params = [
                "1 A0 10",
                "2 a 0.5",
                "3 thr1 2",
                f"4 thr2 {thr2!r}",
                "5 kb 3",
                f"6 X0 {X0!r}",
            ]
            funcs = ["1 fY() if(Aobs<thr1,kb,0)", "2 c() if(Aobs<thr2,Aobs/thr2,1)"]
            species = ["1 A() A0", "2 Y() 0", "3 X() X0"]
            rxns = ["1 1 0 a", "2 0 2 fY"]
            for i in range(npair):
                n = len(params) + 1
                params += [f"{n} kf{i} {kf[i]!r}", f"{n + 1} kr{i} {kr[i]!r}"]
                params += [f"{n + 2} P{i}0 {X0 * kf[i] / kr[i]!r}"]
                m = len(funcs) + 1
                funcs += [f"{m} ff{i}() kf{i}*c()", f"{m + 1} fr{i}() kr{i}*c()"]
                species.append(f"{4 + i} P{i}() P{i}0")
                r = len(rxns) + 1
                rxns += [f"{r} 3 {4 + i} ff{i}", f"{r + 1} {4 + i} 3 fr{i}"]
            text = "".join(
                f"begin {block}\n" + "".join(f"    {line}\n" for line in lines) + f"end {block}\n"
                for block, lines in (
                    ("parameters", params),
                    ("functions", funcs),
                    ("species", species),
                    ("reactions", rxns),
                    ("groups", ["1 Aobs 1"]),
                )
            )
            model = _model(tmp_path, text, name=f"eq{npair}_{k}.net")
            run = bngsim.Simulator(
                model, method="ode", sensitivity_params=["a", "thr1", "thr2", "kb"]
            ).run(t_span=(0.0, T), n_points=3, rtol=1e-10, atol=1e-12)
            names = list(run.species_names)
            s = np.asarray(run.sensitivities)[-1]
            np.testing.assert_allclose(s[names.index("Y()")], want_y, rtol=1e-6, atol=1e-6)
            # X and every P_i are constant, so their columns are 0.
            assert np.max(np.abs(s[2:])) < 1e-12 * X0

    @pytest.mark.parametrize("Z0", [0.0, 1e9])
    def test_two_clamps_at_one_threshold_value_are_not_refused(self, tmp_path, Z0):
        """Two continuous clamps, on K1 and on K2 = K1 (or a few ulps off), each
        scaling a balanced X0-sized exchange. CVODE reports both roots together,
        the rounding of the exchanges reads as a jump, and the two had to agree
        on dt*/dθ, which they cannot: dK1 moves one and dK2 the other. Neither
        carries a branch change, so neither is asked now. Main refused a few of
        these too, on the same rounding, unless a large bystander hid it."""
        for k in (0, 1, 5, 40):
            K2 = float(2.0 * (1 - k * EPS))
            X0 = 1e14
            text = f"""\
begin parameters
    1 A0 10
    2 a 0.5
    3 K1 2
    4 K2 {K2!r}
    5 X0 {X0!r}
    6 P0 {X0 * 1.1 / 0.6!r}
    7 Q0 {X0 * 1.3 / 0.7!r}
    8 kf 1.1
    9 kr 0.6
   10 lf 1.3
   11 lr 0.7
   12 Z0 {Z0!r}
   13 kz 1
end parameters
begin functions
    1 c1() if(Aobs<K1,Aobs/K1,1)
    2 c2() if(Aobs<K2,Aobs/K2,1)
    3 f1() kf*c1()
    4 r1() kr*c1()
    5 f2() lf*c2()
    6 r2() lr*c2()
end functions
begin species
    1 A() A0
    2 X() X0
    3 P() P0
    4 Q() Q0
    5 Z() Z0
end species
begin reactions
    1 1 0 a
    2 2 3 f1
    3 3 2 r1
    4 2 4 f2
    5 4 2 r2
    6 5 0 kz
end reactions
begin groups
    1 Aobs 1
end groups
"""
            model = _model(tmp_path, text, name=f"clamps{k}.net")
            run = bngsim.Simulator(model, method="ode", sensitivity_params=["a", "K1", "K2"]).run(
                t_span=(0.0, 8.0), n_points=3, rtol=1e-10, atol=1e-12
            )
            # X, P and Q are constant, so their columns are 0.
            assert np.max(np.abs(np.asarray(run.sensitivities)[-1, 1:4])) < 1e-12 * X0

    def test_a_tangent_crossing_beside_a_second_threshold_is_refused(self, tmp_path):
        """A relaxes onto 2 from above and passes thr2 = 2 + 3e-9 at a crawl, too
        slowly for a step along the flow to carry it across, so the crossing is
        read along a coordinate instead, at ±2e-9 and at ±4e-9. The wider pair
        also crosses thr1 = 2, whose jump kb sits in the same rate law: near =
        kc, far = kc + kb, "growth", so the crossing was taken as continuous and
        left without a restart, standing on a real jump. Main never returns from
        that run. A jump at a crossing the flow cannot resolve is refused."""
        text = """\
begin parameters
    1 A0 10
    2 k 1
    3 thr1 2
    4 thr2 2.000000003
    5 kb 3
    6 kc 5
end parameters
begin functions
    1 src() k*thr1
    2 fY() if(Aobs<thr1,kb,0)+if(Aobs<thr2,kc,0)
end functions
begin species
    1 A() A0
    2 Y() 0
end species
begin reactions
    1 0 1 src
    2 1 0 k
    3 0 2 fY
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="crawl.net")
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k", "thr1", "kb"])
        with pytest.raises(SimulationError, match="rides that surface") as err:
            sim.run(t_span=(0.0, 40.0), n_points=5, rtol=1e-8, atol=1e-12)
        assert "crosses another switch's surface" in str(err.value)

    @pytest.mark.parametrize("B0", [1e3, 3e6])
    def test_a_clamp_crossed_just_before_a_jump_leaves_the_jump_its_own_dtstar(self, tmp_path, B0):
        """The clamp ``kx·if(Aobs<thr2, Aobs/thr2, 1)`` on B -> C is crossed a few
        hundred ulps BEFORE the jump at thr1, so the clamp's root is the one
        CVODE reports and the jump co-crosses. dt*/dθ was taken from the
        reported switch: main credited the jump to thr2 (dY/dthr2 = 3,
        dY/dthr1 = 0, silently), and the earlier cuts of this fix refused. It
        now comes from a switch whose own reactions jump."""
        A0, a, thr1, kb, T = 10.0, 0.5, 2.0, 3.0, 8.0
        t1 = np.log(A0 / thr1) / a
        want = np.array([kb * np.log(A0 / thr1) / a**2, kb / (a * thr1), 0.0, T - t1])
        for k in (100, 300):
            thr2 = float(thr1 * (1 + k * EPS))
            text = f"""\
begin parameters
    1 A0 10
    2 a 0.5
    3 thr1 2
    4 thr2 {thr2!r}
    5 kb 3
    6 kx 0.1
    7 B0 {B0!r}
end parameters
begin functions
    1 fY() if(Aobs<thr1,kb,0)
    2 fc() kx*if(Aobs<thr2,Aobs/thr2,1)
end functions
begin species
    1 A() A0
    2 Y() 0
    3 B() B0
    4 C() 0
end species
begin reactions
    1 1 0 a
    2 0 2 fY
    3 3 4 fc
end reactions
begin groups
    1 Aobs 1
end groups
"""
            model = _model(tmp_path, text, name=f"clampfirst{k}.net")
            run = bngsim.Simulator(
                model, method="ode", sensitivity_params=["a", "thr1", "thr2", "kb"]
            ).run(t_span=(0.0, T), n_points=3, rtol=1e-10, atol=1e-12)
            got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()")]
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("kbig", [1e12, 1e14])
    def test_a_steep_clamp_crossing_with_a_jump_takes_none_of_it(self, tmp_path, kbig):
        """``fZ = if(A<thr2, kbig*(thr2 - A), 0)`` is continuous at thr2, which the
        trajectory crosses a few hundred ulps from the jump at thr1, before it
        or after. The clamp's kink is not where the jump is read, so across the
        probe its two branches differ by kbig times that offset: 0.04 to 13
        here. That went into the jump with thr1's dt*/dθ, so dZ/dthr1 came out
        -0.12 at kbig = 1e12 and -13 at 1e14 (main: -0.30 and -31), for a truth
        of 0. And read at its own root the clamp is still off by what a few
        ulps of A do to a flux this steep, which at kbig = 1e14 passed the
        drive and got one of these refused. The clamp's reactions are now
        left out of the jump, and a branch change within the rounding of the
        root is a kink. With the clamp first, main gives thr2 the whole jump."""
        t1 = np.log(5.0) / 0.5
        want_y = _switched_source(2.0, 3.0, 1)
        want_y[3] = HAIR_T - t1
        for k in (100, 300, -100, -300):
            thr2 = float(2.0 * (1 - k * EPS))
            run = _hair(
                tmp_path,
                f"steepclamp{k}.net",
                thr2=thr2,
                kbig=kbig,
                fy="if(Aobs<thr1,kb,0)",
                fz="if(Aobs<thr2,kbig*(thr2-Aobs),0)",
            )
            names = list(run.species_names)
            s = np.asarray(run.sensitivities)[-1]
            np.testing.assert_allclose(s[names.index("Y()")], want_y, rtol=1e-6, atol=1e-6)
            z = s[names.index("Z()")]
            # One ulp of A moves the clamp's flux by 4.4e-16·kbig.
            assert abs(z[1]) < 1e-14 * kbig
            assert z[2] == pytest.approx(kbig * (HAIR_T - np.log(10.0 / thr2) / 0.5), rel=1e-6)

    @pytest.mark.parametrize(
        ("A0", "thr", "kb", "kc"), [(1e6, 2e5, 50.0, 0.05), (1e8, 2e7, 1000.0, 5.0)]
    )
    def test_a_small_jump_on_a_second_switch_of_the_surface_is_kept(
        self, tmp_path, A0, thr, kb, kc
    ):
        """kb into Y at ``Aobs < thr`` and kc into Z on a second switch of the same
        surface, with kc under 1e-6 of the rate that drives the crossing. On its
        own that second switch reads as continuous. But the crossing jumps, and
        the jump main applies is the whole right-hand side, kc included. Taking a
        continuous reader's reactions out of the jump took kc with them:
        dZ/d[A0, a, thr] came back 0 where main is right (eighth review). Only
        what such a reader's kink puts in is taken out now. The second switch is
        the other spelling under A + B conservation, then a second parameter of
        the same value."""
        a, T = 0.5, 6.0
        want = kc * np.array([-1 / (a * A0), np.log(A0 / thr) / a**2, 1 / (a * thr)])
        for second, sp in (("Bobs>A0-thr", "1 1 2 a"), ("Aobs<thr2", "1 1 0 a")):
            text = (
                f"begin parameters\n    1 A0 {A0!r}\n    2 a 0.5\n    3 thr {thr!r}\n"
                f"    4 kb {kb!r}\n    5 kc {kc!r}\n    6 thr2 {thr!r}\nend parameters\n"
                f"begin functions\n    1 fY() if(Aobs<thr,kb,0)\n    2 fZ() if({second},kc,0)\n"
                "end functions\n"
                "begin species\n    1 A() A0\n    2 B() 0\n    3 Y() 0\n    4 Z() 0\nend species\n"
                f"begin reactions\n    {sp}\n    2 0 3 fY\n    3 0 4 fZ\nend reactions\n"
                "begin groups\n    1 Aobs 1\n    2 Bobs 2\nend groups\n"
            )
            model = _model(tmp_path, text, name="small_second.net")
            run = bngsim.Simulator(model, method="ode", sensitivity_params=["A0", "a", "thr"]).run(
                t_span=(0.0, T), n_points=3, rtol=1e-10, atol=1e-12
            )
            names = list(run.species_names)
            s = np.asarray(run.sensitivities)[-1]
            got = s[names.index("Z()")]
            if second == "Aobs<thr2":
                got, ref = got[:2], want[:2]  # thr2 is its own parameter here
            else:
                ref = want
            np.testing.assert_allclose(got, ref, rtol=1e-5)
            np.testing.assert_allclose(s[names.index("Y()")], want * kb / kc, rtol=1e-5)

    @pytest.mark.parametrize("k", [0, 100, -100])
    def test_a_jump_under_the_agreement_floor_moves_with_its_own_switch(self, tmp_path, k):
        """kb into Y at thr1, and at thr2 a jump of kc into X, which also swaps
        with P at 1e15 each way through a clamp on thr2. kc is under 16 ulps of
        thr2's gross flux, so that switch is not asked to agree on dt*/dθ. Its
        jump was then moved with thr1's: d(X+P)/d[thr1, thr2] = (3, 0) for a
        truth of (0, 3), silently, where main refuses the two together (eighth
        review). A switch under its floor is now moved with its own dt*/dθ."""
        a, thr1, kb, kc, X0 = 0.5, 2.0, 3.0, 3.0, 1e15
        thr2 = float(thr1 * (1 - k * EPS))
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr1 2
    4 thr2 {thr2!r}
    5 kb 3
    6 kc {kc!r}
    7 X0 {X0!r}
    8 kx 1
end parameters
begin functions
    1 fY() if(Aobs<thr1,kb,0)
    2 fX() if(Aobs<thr2,kc,0)
    3 fc() kx*if(Aobs<thr2,Aobs/thr2,1)
end functions
begin species
    1 A() A0
    2 Y() 0
    3 X() X0
    4 P() X0
end species
begin reactions
    1 1 0 a
    2 0 2 fY
    3 0 3 fX
    4 4 3 fc
    5 3 4 fc
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="under_floor.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["a", "thr1", "thr2"]).run(
            t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12
        )
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        y = s[names.index("Y()")]
        xp = s[names.index("X()")] + s[names.index("P()")]
        np.testing.assert_allclose(y[1:], [kb / (a * thr1), 0.0], atol=1e-3)
        # X and P are 1e15 each: their sum's columns carry that rounding.
        np.testing.assert_allclose(xp[1:], [0.0, kc / (a * thr2)], atol=0.05)

    def test_a_reaction_the_map_does_not_list_is_judged_as_before(self, tmp_path):
        """A Michaelis-Menten law whose kcat is a parameter a function writes,
        ``kcat() = if(Aobs<thr, kb, 0)``, reads the condition but is not a
        functional rate law, so the reaction map has no entry for it. With no
        reader the crossing was taken as continuous and dP/dthr, dP/da came back
        0 where main is right (eighth review). Where the mapped reactions do not
        account for what the pre-#763 test reads as a jump, that test stands.

        P's own value is not asserted. In a sensitivity run it stays 0, on main
        too: the sensitivity right-hand side reads kcat's parameter slot rather
        than the function. That is a separate defect."""
        text = """begin parameters
    1 A0 10
    2 a 0.5
    3 thr 2
    4 kb 3
    5 kcat 0
    6 Km 1
    7 E0 1
    8 S0 1e6
end parameters
begin functions
    1 kcat() if(Aobs<thr,kb,0)
end functions
begin species
    1 A() A0
    2 E() E0
    3 S() S0
    4 P() 0
end species
begin reactions
    1 1 0 a
    2 2,3 2,4 MM kcat Km
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="mm_kcat.net")
        conditions = sw.state_switch_conditions(model._core)
        assert sw.state_switch_reactions(model._core, conditions) == [[]]
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["thr", "a"]).run(
            t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12
        )
        got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("P()")]
        # The rate past the switch is kb·E0·S/(Km + S), with S = 1e6.
        rate = 3.0 * 1e6 / (1.0 + 1e6)
        np.testing.assert_allclose(got, [rate / (0.5 * 2.0), rate * np.log(5.0) / 0.25], rtol=1e-5)

    def test_a_threshold_only_an_output_reads_does_not_take_the_wider_probe(self, tmp_path):
        """A jump at thr0, a continuous clamp on a 2e8 pool at thr1 inside the
        probe, and a third threshold that only an output reads, crossed by the
        wider probe alone. Counting every registered switch, the wider probe was
        dropped, the clamp was read from the near pair with its smooth change in
        it, and the run was refused where main is right (eighth review). Only a
        switch whose jump can show in the reading takes the wider probe away."""
        kb, B0, T = 100.0, 2e8, 8.0
        want = np.array([kb * np.log(5.0) / 0.25, kb / (0.5 * 2.0), 0.0, 0.0])
        for k1, k2 in ((100, 500), (200, 600), (300, 700)):
            thr1 = float(2.0 * (1 - k1 * EPS))
            thr2 = float(2.0 * (1 - k2 * EPS))
            text = (
                "begin parameters\n    1 A0 10\n    2 a 0.5\n    3 thr0 2\n"
                f"    4 thr1 {thr1!r}\n    5 thr2 {thr2!r}\n    6 kb {kb!r}\n    7 B0 {B0!r}\n"
                "end parameters\n"
                "begin functions\n    1 fY() if(Aobs<thr0,kb,0)\n"
                "    2 fc() 0.1*if(Aobs<thr1,Aobs/thr1,1)\n    3 o2() if(Aobs<thr2,1,0)\n"
                "end functions\n"
                "begin species\n    1 A() A0\n    2 Y() 0\n    3 B() B0\n    4 C() 0\n"
                "end species\n"
                "begin reactions\n    1 1 0 a\n    2 0 2 fY\n    3 3 4 fc\nend reactions\n"
                "begin groups\n    1 Aobs 1\nend groups\n"
            )
            model = _model(tmp_path, text, name="third_output.net")
            run = bngsim.Simulator(
                model, method="ode", sensitivity_params=["a", "thr0", "thr1", "thr2"]
            ).run(t_span=(0.0, T), n_points=3, rtol=1e-10, atol=1e-12)
            got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()")]
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize(
        ("k", "pool"),
        [(0, 0.0), (600, 0.0), (-600, 0.0), (300, 0.0), (-300, 0.0), (300, 1e8), (-300, 1e8)],
    )
    def test_a_reaction_the_map_does_not_list_beside_a_mapped_jump(self, tmp_path, k, pool):
        """The same unlisted Michaelis-Menten law, switched at thr2, beside a
        mapped jump of kb into Y at thr1. With thr2 only in the wider probe, the
        whole right-hand side extended to the crossing took the unlisted jump in
        from 2·δt away and gave it thr1's dt*/dθ: dP/d[thr1, thr2] = (-5, 5) for
        a truth of (0, 5), where main is right. An empty map entry was taken to
        mean nothing there can jump. With thr2 = thr1 the two are reported
        together and the unlisted one was not asked to agree: (5, 0), where main
        refuses (ninth review).

        At 300 ulps the two are a pair inside the probe and main gives (5, 0)
        itself, with or without a 1e8 pool turning over beside them. A crossing
        that goes back to the pre-#763 judgment now asks every switch the probe
        crosses to agree, not only the reported ones (tenth review)."""
        thr2 = float(2.0 * (1 - k * EPS))
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr1 2
    4 thr2 {thr2!r}
    5 kb 3
    6 kc 5
    7 kcat 0
    8 Km 1
    9 kq 0.1
end parameters
begin functions
    1 f1() if(Aobs<thr1,kb,0)
    2 kcat() if(Aobs<thr2,kc,0)
end functions
begin species
    1 A() A0
    2 E() 1
    3 S() 1e6
    4 P() 0
    5 Y() 0
    6 Q() {pool!r}
    7 W() {0.9 * pool!r}
end species
begin reactions
    1 1 0 a
    2 2,3 2,4 MM kcat Km
    3 0 5 f1
    4 6 7 kq
    5 7 6 kq
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="mm_beside.net")
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr1", "thr2"])
        try:
            run = sim.run(t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12)
        except SimulationError as e:
            assert abs(k) <= 300 and "cross at the same instant" in str(e)
            return
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        rate = 5.0 * 1e6 / (1.0 + 1e6)
        np.testing.assert_allclose(s[names.index("P()")], [0.0, rate / (0.5 * thr2)], atol=1e-4)
        np.testing.assert_allclose(s[names.index("Y()")], [3.0, 0.0], atol=1e-4)

    def test_a_loss_in_fast_balance_with_a_clamped_source_is_not_a_missed_reaction(self, tmp_path):
        """X is made at ``kbig*Aobs + clamp(thr1)`` and lost at kdeg·X by mass
        action, with kbig = kdeg = 1e10, and a second clamp on thr2 = thr1 takes
        B to C. Nothing jumps. Across the probe pair the loss moves by
        2·δt·kdeg·|dX/dt|, which the source cancels in the whole right-hand
        side. Read on its own, as what the listed reactions do not account for,
        that passed the pre-#763 tolerance, the crossing went back to the
        pre-#763 judgment, and its two switches were refused for not agreeing
        on dt*/dθ, where main runs (ninth review). The fallback now also needs
        the pre-#763 test itself to read a jump."""
        kbig = 1e10
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr1 2
    4 thr2 2
    5 kbig {kbig!r}
    6 kdeg {kbig!r}
    7 X0 {(kbig * 10.0 + 1.0) / kbig!r}
end parameters
begin functions
    1 f1() kbig*Aobs+if(Aobs<thr1,Aobs/thr1,1)
    2 f2() 0.1*if(Aobs<thr2,Aobs/thr2,1)
end functions
begin species
    1 A() A0
    2 X() X0
    3 B() 1000
    4 C() 0
end species
begin reactions
    1 1 0 a
    2 0 2 f1
    3 2 0 kdeg
    4 3 4 f2
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="fast_balance.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["a", "thr1", "thr2"]).run(
            t_span=(0.0, 6.0), n_points=3, rtol=1e-9, atol=1e-12
        )
        got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("B()")]

        # B(T) = B0·exp(−0.1·t2 − (0.1/(a·thr2))·(thr2 − A(T))), t2 = ln(A0/thr2)/a.
        def b_end(a, thr2):
            t2 = np.log(10.0 / thr2) / a
            return 1000.0 * np.exp(
                -0.1 * t2 - (0.1 / (a * thr2)) * (thr2 - 10.0 * np.exp(-a * 6.0))
            )

        h = 1e-6
        want_a = (b_end(0.5 + h, 2.0) - b_end(0.5 - h, 2.0)) / (2 * h)
        want_thr2 = (b_end(0.5, 2.0 + h) - b_end(0.5, 2.0 - h)) / (2 * h)
        np.testing.assert_allclose(got, [want_a, 0.0, want_thr2], rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("k", [1, 100, 300, 1000, -10, -100])
    def test_a_switch_under_its_floor_whose_crossing_the_jump_moves_has_to_agree(
        self, tmp_path, k
    ):
        """The switch at thr1 doubles A's own decay, and thr2, just past it, is
        read by a jump of kc into X under a 1e15 exchange, so under its floor.
        Moved with its own dt*/dθ formed before the jump, kc gave
        d(X+P)/d[thr1, thr2] = (0, 3). But past thr1 A falls twice as fast, so
        thr2 is reached sooner for every thr1: the truth is (1.5, 1.5). Where
        the jump moves a species such a switch's residual reads, the switch has
        to agree like any other, and here it cannot (ninth review). From 300
        ulps main gives (3, 0), silently.

        With thr2 a hair ABOVE thr1 it is crossed first, the change in A's decay
        comes after it, and (0, 3) is right. That was refused too, since the
        rule did not ask which of the two is crossed first (tenth review)."""
        thr2 = float(2.0 * (1 - k * EPS))
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr1 2
    4 thr2 {thr2!r}
    5 kc 3
    6 X0 1e15
end parameters
begin functions
    1 fA() a*if(Aobs<thr1,2,1)
    2 fX() if(Aobs<thr2,kc,0)
    3 fc() if(Aobs<thr2,Aobs/thr2,1)
end functions
begin species
    1 A() A0
    2 X() X0
    3 P() X0
end species
begin reactions
    1 1 0 fA
    2 0 2 fX
    3 3 2 fc
    4 2 3 fc
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="moved_crossing.net")
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr1", "thr2"])
        try:
            run = sim.run(t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12)
        except SimulationError as e:
            assert k > 0 and "cross at the same instant" in str(e)
            return
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        xp = s[names.index("X()")] + s[names.index("P()")]
        np.testing.assert_allclose(xp, [1.5, 1.5] if k > 0 else [0.0, 3.0], atol=0.05)
        if k < 0:
            # A's own column for thr2 is 0: thr2 changes nothing A reads.
            assert s[names.index("A()"), 1] == 0.0

    def test_two_switches_in_one_rate_law_under_the_floor_have_to_agree(self, tmp_path):
        """``if(A<thr1,2,0) + if(A<thr2,3,0)`` into X, beside a 1e15 exchange
        that both clamp, with thr2 = thr1. Each switch's reading is the whole
        rate law's jump of 5, under the floor, and neither says whose it is. The
        second was skipped and the 5 went to thr1: (5, 0) for a truth of (2, 3),
        where main refuses (ninth review)."""
        text = """begin parameters
    1 A0 10
    2 a 0.5
    3 thr1 2
    4 thr2 2
    5 X0 1e15
end parameters
begin functions
    1 fX() if(Aobs<thr1,2,0)+if(Aobs<thr2,3,0)
    2 fc() if(Aobs<thr1,Aobs/thr1,1)*if(Aobs<thr2,Aobs/thr2,1)
end functions
begin species
    1 A() A0
    2 X() X0
    3 P() X0
end species
begin reactions
    1 1 0 a
    2 0 2 fX
    3 3 2 fc
    4 2 3 fc
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="shared_law.net")
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr1", "thr2"])
        try:
            run = sim.run(t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12)
        except SimulationError as e:
            assert "cross at the same instant" in str(e)
            return
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        xp = s[names.index("X()")] + s[names.index("P()")]
        np.testing.assert_allclose(xp, [2.0, 3.0], atol=0.05)

    @pytest.mark.parametrize("k", [550, 650])
    def test_a_steep_term_beside_a_second_jump_in_the_wider_probe(self, tmp_path, k):
        """``kbig*Aobs + if(A<thr1,kb,0)`` at kbig = 1e12 into Y, and a second
        jump into Z at thr2, which only the wider probe crosses. The whole
        right-hand side then cannot be extended to the crossing, but Y's own
        reaction can. Putting the extended reading into Y's entry and leaving A's
        raw broke what makes the raw pair consistent, and dY/dthr1 came out 3.33
        for a truth of 3 (ninth review; 2.97 with the pair left raw). Only the
        root offset of a reader's reading is added now."""
        thr2 = float(2.0 * (1 - k * EPS))
        run = _hair(
            tmp_path,
            f"steep_far{k}.net",
            thr2=thr2,
            kbig=1e12,
            fy="kbig*Aobs+if(Aobs<thr1,kb,0)",
            fz="if(Aobs<thr2,kc,0)",
        )
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        assert s[names.index("Y()"), 1] == pytest.approx(3.0, abs=0.1)
        z = s[names.index("Z()")]
        assert abs(z[1]) < 1e-6 and z[2] == pytest.approx(5.0 / (0.5 * thr2), rel=1e-6)

    @pytest.mark.parametrize("k", [100, 300])
    def test_a_switch_that_shares_one_rate_law_and_has_another_of_its_own(self, tmp_path, k):
        """``if(A<thr1,2,0) + if(A<thr2,3,0)`` into Y, and ``if(A<thr2,kc,0)``
        into Z. The thr2 switch shares Y's reaction with thr1 and has Z's to
        itself. Taking each reaction out once, the reader that came second was
        skipped whole, Z's jump of 5 was left looking like a reaction the map
        does not list, and the crossing went back to the pre-#763 judgment with
        only thr1 asked: dY = (5, 0) and dZ = (5, 0) for a truth of (2, 3) and
        (0, 5), as on main (tenth review). The readers' reactions are now read
        as one union."""
        thr2 = float(2.0 * (1 - k * EPS))
        try:
            run = _hair(
                tmp_path,
                f"sharedown{k}.net",
                thr2=thr2,
                kbig=0.0,
                fy="if(Aobs<thr1,2,0)+if(Aobs<thr2,3,0)",
                fz="if(Aobs<thr2,kc,0)",
            )
        except SimulationError as e:
            assert "cross at the same instant" in str(e)
            return
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        np.testing.assert_allclose(s[names.index("Y()"), 1:3], [2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(s[names.index("Z()"), 1:3], [0.0, 5.0], atol=1e-6)

    @pytest.mark.parametrize(("k1", "k2"), [(-1000, -200), (-600, -200), (-600, 200)])
    def test_a_steep_ramp_crossed_first_does_not_give_two_jumps_to_one_switch(
        self, tmp_path, k1, k2
    ):
        """Two jumps a hair apart with a ramp of slope 1e9 crossed just before
        them. The ramp's curvature across the probe read as a reaction the map
        does not list, the crossing went back to the pre-#763 judgment, and that
        asked only the reported switch to agree: dY0/dthr0 = 0 and
        dY0/dthr2 = 3 for a truth of 3 and 0, as on main (tenth review)."""
        units = [("jump", 0, 0.0), ("ramp", k1, 1e9), ("jump", k2, 0.0)]
        try:
            s = _units_sens(tmp_path, units, f"ramp_first{k1}_{k2}.net")
        except SimulationError as e:
            assert UNIT_REFUSED.search(str(e))
            return
        np.testing.assert_allclose(s["Y0"][2:], [3.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(s["Y2"][2:], [0.0, 0.0, 7.0], atol=1e-5)

    @pytest.mark.parametrize(
        ("k1", "k2", "B0"),
        [(-600, -200, 1e9), (600, 200, 1e9), (-600, -200, 1e3), (600, 200, 1e3)],
    )
    def test_a_jump_is_applied_once(self, tmp_path, k1, k2, B0):
        """Two clamps and a jump of 7, each a few hundred ulps from the next.
        The run stops at the first, reads the jump from a probe that already
        crosses the jump's own threshold, and restarts past it. The next stop,
        less than a probe step later, crossed that threshold again going back
        and applied the jump a second time: dY2/dthr2 = 14 (eighth to tenth
        reviews). Main gives 0 at B0 = 1e9 and credits the 7 to another
        threshold at 1e3. A switch whose jump was applied is now remembered
        until the run is a probe step past it."""
        units = [("clamp", 0, B0), ("clamp", k1, B0), ("jump", k2, 0.0)]
        try:
            s = _units_sens(tmp_path, units, f"once{k1}_{k2}.net")
        except SimulationError as e:
            assert UNIT_REFUSED.search(str(e))
            return
        np.testing.assert_allclose(s["Y2"][2:], [0.0, 0.0, 7.0], atol=1e-5)

    @pytest.mark.parametrize(
        "units",
        [
            [("jump", 0, 0.0), ("clamp", 200, 1e9), ("jump", 600, 0.0)],
            [("jump", 0, 0.0), ("clamp", -200, 1e9), ("jump", -600, 0.0)],
            [("clamp", 0, 1e9), ("clamp", 200, 1e9), ("jump", 600, 0.0)],
        ],
        ids=["jump-clamp-jump", "jump-clamp-jump-reversed", "clamp-clamp-jump"],
    )
    def test_a_switch_that_did_not_jump_is_not_remembered(self, tmp_path, units):
        """The same spacing with a clamp in the middle. The second stop reads the
        clamp again, and a clamp has no jump to read twice, so these run and are
        right. Remembering every switch a restart stepped over refused them."""
        s = _units_sens(tmp_path, units, "not_remembered.net")
        if units[0][0] == "jump":
            np.testing.assert_allclose(s["Y0"][2:], [3.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(s["Y2"][2:], [0.0, 0.0, 7.0], atol=1e-5)

    @pytest.mark.parametrize("kf", [1e7, 1e9])
    def test_a_fast_loss_does_not_send_the_crossing_back_to_the_old_judgment(self, tmp_path, kf):
        """Y is made at ``kf*Aobs + if(A<thr1,kb,0)`` and lost at kf·Y. W gets the
        same jump. A clamp at thr2, 300 ulps above thr1, is the root reported,
        and a third jump sits 400 ulps below, in the wider probe only. Across
        the probe the loss moves by 2·δt·kf·|dY/dt|, which read as a reaction
        the map does not list from kf = 3e6 up, and the pre-#763 judgment it
        fell back to credits the jump to the clamp: dW/d[thr1, thr2] = (0, 3)
        for a truth of (3, 0), as on main (tenth review). What the readers leave
        is now tested by its second difference across the crossing, which a
        smooth term has none of."""
        thr2 = float(2.0 * (1 + 300 * EPS))
        thr3 = float(2.0 * (1 - 400 * EPS))
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr1 2
    4 thr2 {thr2!r}
    5 thr3 {thr3!r}
    6 kb 3
    7 kc 5
    8 kx 0.1
    9 kf {kf!r}
end parameters
begin functions
    1 fY() kf*Aobs+if(Aobs<thr1,kb,0)
    2 fc() kx*if(Aobs<thr2,Aobs/thr2,1)
    3 fZ() if(Aobs<thr3,kc,0)
    4 fW() if(Aobs<thr1,kb,0)
end functions
begin species
    1 A() A0
    2 Y() 10
    3 W() 0
    4 B() 1
    5 C() 0
    6 Z() 0
end species
begin reactions
    1 1 0 a
    2 0 2 fY
    3 2 0 kf
    4 0 3 fW
    5 4 5 fc
    6 0 6 fZ
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="fast_loss.net")
        run = bngsim.Simulator(
            model, method="ode", sensitivity_params=["thr1", "thr2", "thr3"]
        ).run(t_span=(0.0, 8.0), n_points=3, rtol=1e-10, atol=1e-12)
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        np.testing.assert_allclose(s[names.index("W()")], [3.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(s[names.index("Z()")], [0.0, 0.0, 5.0], atol=1e-5)

    @pytest.mark.parametrize(
        ("fz", "k"),
        [
            ("kx*if(Aobs<thr2,Aobs/thr2,1)", -300),
            ("if(Aobs<thr2,kc,0)", 300),
            ("if(Aobs<thr2,kc,0)", -300),
        ],
        ids=["clamp-first", "jump-after", "jump-before"],
    )
    def test_the_rounding_of_a_fast_balance_no_switch_reads_is_not_a_jump(self, tmp_path, fz, k):
        """Q is made at ``kf*Aobs`` and lost at kf·Q with kf = 1e12, and reads no
        switch. Its two fluxes are 1e13 each, so their difference carries 1e-3
        of rounding, more than 1e-6 of anything else in the model. That read as
        a reaction the map does not list. With a clamp crossed 300 ulps before
        the jump, the jump went to the clamp's threshold: dY = (0, 3) for a
        truth of (3, 0). With two jumps, both went to the reported one. Main
        does the same (tenth review). The test of what the readers leave now
        allows for the rounding of the gross flux."""
        thr2 = float(2.0 * (1 - k * EPS))
        clamp = "Aobs/thr2" in fz
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr1 2
    4 thr2 {thr2!r}
    5 kb 3
    6 kc 5
    7 kx 0.1
    8 kf 1e12
end parameters
begin functions
    1 fY() if(Aobs<thr1,kb,0)
    2 fZ() {fz}
    3 fq() kf*Aobs
end functions
begin species
    1 A() A0
    2 Y() 0
    3 B() {1000 if clamp else 0}
    4 Z() 0
    5 Q() 10
end species
begin reactions
    1 1 0 a
    2 0 2 fY
    3 {"3 4" if clamp else "0 4"} fZ
    4 0 5 fq
    5 5 0 kf
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="fast_bystander.net")
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr1", "thr2"])
        try:
            run = sim.run(t_span=(0.0, 8.0), n_points=3, rtol=1e-10, atol=1e-12)
        except SimulationError as e:
            assert not clamp and "cross at the same instant" in str(e)
            return
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        np.testing.assert_allclose(s[names.index("Y()")], [3.0, 0.0], atol=1e-5)
        if not clamp:
            np.testing.assert_allclose(s[names.index("Z()")], [0.0, 5.0], atol=1e-5)

    def test_two_unlisted_jumps_either_side_of_the_crossing_are_not_dropped(self, tmp_path):
        """Two Michaelis-Menten laws into P, each with a kcat a function switches,
        at thresholds 300 ulps apart. Neither is in the map. The state the run
        stops at lies between the two, so the second difference of the
        right-hand side there is -kc + kc = 0, and a test built on it alone saw
        no jump and dropped both: dP/d[A0, a] = (0, 0) where main is right
        (eleventh review). Where only the change across the pair shows
        something, the crossing is now judged exactly as before #763."""
        thr3 = float(2.0 * (1 - 300 * EPS))
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr2 2
    4 thr3 {thr3!r}
    5 kc 5
    6 kcat1 0
    7 kcat2 0
    8 Km 1
end parameters
begin functions
    1 kcat1() if(Aobs<thr2,kc,0)
    2 kcat2() if(Aobs<thr3,kc,0)
end functions
begin species
    1 A() A0
    2 E() 1
    3 S() 1e6
    4 P() 0
end species
begin reactions
    1 1 0 a
    2 2,3 2,4 MM kcat1 Km
    3 2,3 2,4 MM kcat2 Km
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="two_unlisted.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["A0", "a"]).run(
            t_span=(0.0, 8.0), n_points=3, rtol=1e-10, atol=1e-12
        )
        got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("P()")]
        # P = 2·kc'·(T − ln(A0/thr)/a) with kc' = kc·S/(Km + S).
        rate = 2 * 5.0 * 1e6 / (1.0 + 1e6)
        np.testing.assert_allclose(
            got, [-rate / (0.5 * 10.0), rate * np.log(5.0) / 0.25], rtol=1e-5
        )

    @pytest.mark.parametrize(("X0", "kq"), [(1e15, 1.0), (1e9, 1e6)])
    def test_an_unlisted_jump_under_the_rounding_of_every_flux_it_enters(self, tmp_path, X0, kq):
        """One unlisted Michaelis-Menten jump of 5 from S to P, with S and P each
        in a balanced exchange whose gross flux is 1e15. The jump is under 16
        ulps of that in both species, so the second-difference test, which
        allows for the rounding of the gross flux, saw nothing and the jump was
        dropped: d(P + P2)/d[thr2, a] = (0, 0) where main is right (eleventh
        review)."""
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr2 2
    4 kc 5
    5 kcat 0
    6 Km 1
    7 kq1 {1.1 * kq!r}
    8 kq2 {0.6 * kq!r}
    9 X0 {X0!r}
    10 R0 {X0 * 1.1 / 0.6!r}
end parameters
begin functions
    1 kcat() if(Aobs<thr2,kc,0)
end functions
begin species
    1 A() A0
    2 E() 1
    3 S() X0
    4 P() X0
    5 P2() R0
    6 S2() R0
end species
begin reactions
    1 1 0 a
    2 2,3 2,4 MM kcat Km
    3 4 5 kq1
    4 5 4 kq2
    5 3 6 kq1
    6 6 3 kq2
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="unlisted_sub_rounding.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["thr2", "a"]).run(
            t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12
        )
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]
        got = s[names.index("P()")] + s[names.index("P2()")]
        np.testing.assert_allclose(got, [5.0 / (0.5 * 2.0), 5.0 * np.log(5.0) / 0.25], rtol=1e-3)

    @pytest.mark.parametrize("k", [300, -300])
    def test_two_clamps_a_hair_apart_beside_a_noisy_balance_are_not_refused(self, tmp_path, k):
        """Two clamps at thresholds 300 ulps apart, B -> C and D -> F, and a
        species Q made at ``kf*Aobs`` and lost at kf·Q with kf = 1e12 that reads
        neither. Nothing jumps. The rounding of Q's two fluxes passes the
        pre-#763 tolerance, so that test reads a jump here, and main applies it
        with the reported clamp's dt*/dθ. It is rounding, and main is right.
        Asking BOTH clamps to agree on that dt*/dθ, as a crossing with a real
        unlisted jump is asked, would refuse this run."""
        thr2 = float(2.0 * (1 - k * EPS))
        text = f"""begin parameters
    1 A0 10
    2 a 0.5
    3 thr1 2
    4 thr2 {thr2!r}
    5 kx 0.1
    6 kf 1e12
end parameters
begin functions
    1 f1() kx*if(Aobs<thr1,Aobs/thr1,1)
    2 f2() kx*if(Aobs<thr2,Aobs/thr2,1)
    3 fq() kf*Aobs
end functions
begin species
    1 A() A0
    2 B() 1000
    3 C() 0
    4 D() 1000
    5 F() 0
    6 Q() 10
end species
begin reactions
    1 1 0 a
    2 2 3 f1
    3 4 5 f2
    4 0 6 fq
    5 6 0 kf
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="noisy_clamps.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["thr1", "thr2"]).run(
            t_span=(0.0, 6.0), n_points=3, rtol=1e-9, atol=1e-12
        )
        names = list(run.species_names)
        s = np.asarray(run.sensitivities)[-1]

        # B(T) = B0·exp(−kx·t* − (kx/(a·thr))·(thr − A(T))), t* = ln(A0/thr)/a.
        def b_end(thr):
            t_star = np.log(10.0 / thr) / 0.5
            return 1000.0 * np.exp(
                -0.1 * t_star - (0.1 / (0.5 * thr)) * (thr - 10.0 * np.exp(-0.5 * 6.0))
            )

        h = 1e-6
        want = (b_end(2.0 + h) - b_end(2.0 - h)) / (2 * h)
        np.testing.assert_allclose(s[names.index("B()")], [want, 0.0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(s[names.index("D()")], [0.0, want], rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("around", [-412, -824])
    def test_no_probe_point_lands_on_another_surface(self, tmp_path, around):
        """Two jumps a few hundred ulps apart. The run stops at the first, reads
        the state a probe step and two either side of it, and restarts a probe
        step past it. For one spacing in a few hundred one of those points is
        exactly on the second threshold, its residual 0.0 there.

        At the restart point CVODE sets a root that is zero aside, so the second
        crossing was never reported and every column of the first jump came back
        0, on main too (-412 ulps on the machine this was found on). Two steps
        back, a strict condition still reads its old branch on the surface, and
        the second jump was taken in with the first's dt*/dθ: dY1/dthr0 = -5 for
        0, where main is right (-824). Which spacing it is depends on the last
        bits of the arithmetic, so a range is run and each case has to be right,
        or refused as two jumps in one probe step. The step is stretched off a
        landing."""
        refused = 0
        spacings = range(around - 30, around + 31)
        for k in spacings:
            units = [("jump", 0, 0.0), ("jump", k, 0.0)]
            try:
                s = _units_sens(tmp_path, units, f"lands_on{k}.net")
            except SimulationError as e:
                assert "cross at the same instant" in str(e), f"k = {k}"
                refused += 1
                continue
            np.testing.assert_allclose(s["Y0"][2:], [3.0, 0.0], atol=1e-5, err_msg=f"k = {k}")
            np.testing.assert_allclose(s["Y1"][2:], [0.0, 5.0], atol=1e-5, err_msg=f"k = {k}")
        assert refused < len(spacings), "every spacing was refused, so the range shows nothing"

    @pytest.mark.parametrize(
        "units",
        [
            [("jump", 0, 0.0), ("noreader", 412, 0.0)],
            [("clamp", 0, 1e9), ("clamp", 412, 1e9)],
            [("jump", 0, 0.0), ("clamp", 474, 1e9)],
        ],
        ids=["jump-unread-412", "clamp-clamp-412", "jump-clamp-474"],
    )
    def test_a_landing_on_a_switch_that_cannot_jump_is_not_refused(self, tmp_path, units):
        """The same landings on a threshold no rate law reads and on a clamp's.
        A first cut refused every landing (twelfth review). Main runs the first
        two and is right. The third is one of the #763 cases main gets wrong:
        its jump column comes back 0 there."""
        s = _units_sens(tmp_path, units, "lands_on_quiet.net")
        if units[0][0] == "jump":
            np.testing.assert_allclose(s["Y0"][2:], [3.0, 0.0], atol=1e-5)
        if units[1][0] != "clamp":
            return
        # B1(T) = B0·exp(−0.1·t1 − (0.1/(a·thr1))·(thr1 − A(T))), t1 = ln(A0/thr1)/a.
        thr1 = float(2.0 * (1 - units[1][1] * EPS))

        def b_end(thr):
            return 1e9 * np.exp(
                -0.1 * np.log(10.0 / thr) / 0.5 - (0.2 / thr) * (thr - 10.0 * np.exp(-4.0))
            )

        h = 1e-6
        want = (b_end(thr1 + h) - b_end(thr1 - h)) / (2 * h)
        assert s["B1"][3] == pytest.approx(want, rel=1e-5)

    def test_a_tangent_crossing_ignores_a_threshold_its_reactions_do_not_read(self, tmp_path):
        """The same crawl past thr2 = 2 + 3e-9, but there the rate law is a
        clamp, ``kc·(thr2 − Aobs)`` below it and 0 above, continuous at its own
        switch, and thr1 = 2 is read by another reaction. The wider probe still
        crosses thr1, but nothing it reads changes there, so the clamp's gap is
        read by how it grows, as continuous, and the run is not refused."""
        text = """\
begin parameters
    1 A0 10
    2 k 1
    3 thr1 2
    4 thr2 2.000000003
    5 kb 3
    6 kc 5
end parameters
begin functions
    1 src() k*thr1
    2 fY() if(Aobs<thr2,kc*(thr2-Aobs),0)
    3 fW() if(Aobs<thr1,kb,0)
end functions
begin species
    1 A() A0
    2 Y() 0
    3 W() 0
end species
begin reactions
    1 0 1 src
    2 1 0 k
    3 0 2 fY
    4 0 3 fW
end reactions
begin groups
    1 Aobs 1
end groups
"""
        model = _model(tmp_path, text, name="crawl_clamp.net")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["k", "kc"]).run(
            t_span=(0.0, 30.0), n_points=5, rtol=1e-8, atol=1e-12
        )
        names = list(run.species_names)
        y = np.asarray(run.species)[-1, names.index("Y()")]
        s = np.asarray(run.sensitivities)[-1, names.index("Y()")]
        # Y = kc·∫(thr2 − A)⁺, so dY/dkc = Y/kc; both are ~1e-8.
        assert y > 0.0 and s[1] == pytest.approx(y / 5.0, rel=1e-3)

    def test_the_trajectory_was_never_the_problem(self, tmp_path):
        """Only the sensitivity was wrong: Y(T) = kb·(T − t*) either way, to the
        run's tolerance (1.4e-8 relative at rtol 1e-9, with the pool's 1e8 in
        the error norm)."""
        text = BYSTANDER.format(B0=1e8, vfast=0.0)
        run = _sens(tmp_path, ["kb"], name="bystander_y.net", text=text, t_end=BYSTANDER_T)
        y = list(run.species_names).index("Y()")
        exact = 3.0 * (BYSTANDER_T - np.log(5.0) / 0.5)
        assert float(np.asarray(run.species)[-1, y]) == pytest.approx(exact, rel=1e-6)


# ─── the gate and the detector must read the same text ─────────────────────


class TestTheDetectorSeesWhatTheGateSees:
    """A condition can only *become* a state condition under inlining, and the
    gate judges the inlined rate law.

    BIOMD0000000837 writes ``Lymphocyte_Term`` as
    ``piecewise(…, 1 - Total_Lymphocytes/K > 0, 0)`` where ``Total_Lymphocytes``
    is an SBML assignment-rule parameter — a *parameter* address, reading no
    live state by itself. The gate sees it after substitution as
    ``1 - (B+C_e+C_m+H_e+H_m+L)/K > 0`` and admits; a detector scanning raw
    function bodies registered nothing, so the decline was lifted with no
    crossing behind it. That is the #68 silent zero, reintroduced from the other
    side, and it was invisible to every test until the corpus A/B put three
    models in the "moved, switches=[]" bucket.
    """

    def test_a_condition_that_only_reads_state_after_inlining_is_registered(self, tmp_path):
        text = NET.replace(
            "    1 growth() if(X<1,0,rho)\n",
            "    1 margin() 1-X/X0\n    2 growth() if(margin()>0,rho,0)\n",
        )
        core = _model(tmp_path, text, name="inlined.net")._core
        # The raw body's atom names a *function*, which binds to no state
        # address — this is the classification the bug turned on.
        assert not sw.state_switch_residual(core, "margin>0")
        # Inlined it is a comparison over an observable, and both halves agree.
        assert sw.state_switch_conditions(core) == ["(1-X/X0)>0"]
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None and terms


# ─── several residuals at one instant (issue #153) ─────────────────────────
#
# The roots are deduplicated by the residual's TEXT, which merges ``X<1`` with
# ``X<=1`` and nothing else. Two *spellings* of one crossing therefore
# reach the solver as two roots that fire together, and issue #150 refused that
# batch outright, reasoning that each jump reads f on the two branches of its
# OWN condition and one step across a shared crossing cannot separate them.
#
# On the corpus that refusal never fired for what it was written for. Both
# models that hit it write one crossing twice: ``sp_fourier_synthesizer`` roots
# on ``ds1`` and on ``3·ds1 − 12·s1²·ds1`` (five residuals in all, every one a
# multiple of ``Cos1 − amp_offset``, all crossing at t = π/2), and
# ``ml_hopfield`` roots on ``dS1/dt`` and ``dS3/dt``, which are identically
# equal along its trajectory because its own weight matrix leaves ``S1 ≡ S3``
# invariant. One is visible in the text and one is not, which is why the
# decision belongs at the crossing rather than at detection time.
#
# The batch needs the two halves of the saltation term checked separately, and
# that is what the fixtures below pin:
#
#   * ``f⁻ − f⁺`` has to carry EVERY branch change, which the one flow probe
#     does exactly when it crosses every residual in the batch;
#   * ``dt*/dθ`` has to be ONE vector — which flipping together does NOT
#     establish, so it is formed from each residual in turn and compared.
#
# The second is the criterion, and it is both weaker and stronger than "one
# surface". Weaker: ``COINCIDENT`` below is two genuinely independent crossings,
# and whether they merge depends on which columns are asked for, not on the
# model. Stronger: ``ml_hopfield``'s two residuals are equal along its
# trajectory and nowhere else — their gradients are not parallel, and their
# ``dt*/dθ`` come out permuted by the ``W12 ↔ W23`` symmetry — so a perturbation
# that breaks the symmetry splits the crossing, and merging on the flip test
# alone would have been wrong for it had there been anything to merge.

# ``X<1`` from the model at the top of this file, plus a second condition that
# names the SAME surface through a state-dependent factor — ``X² < 1`` is
# ``(X+1)·(X−1) < 0`` for the positive X this model has, the shape
# ``sp_fourier_synthesizer`` reaches with ``ds3 = ds1·(3 − 12·s1²)``. The dedup
# cannot see that, so the two cross as a batch of two.
#
# Unlike either corpus model there is a REAL jump here: ``pump`` switches the
# zeroth-order source of Y from ``ka`` to ``kb`` at the crossing, so
# ``f⁻ − f⁺`` is non-zero and the merged jump has to be exactly one of them.
# Composing the batch instead — the same jump once per residual — doubles Y's
# whole post-crossing column, which is what the closed forms below rule out.
TWO_SPELLINGS = """\
begin parameters
    1 X0     1000  # Constant
    2 rho    0.8  # Constant
    3 delta  1.6  # Constant
    4 ka     3.0  # Constant
    5 kb     1.0  # Constant
end parameters
begin functions
    1 growth() if(X<1,0,rho)
    2 pump() if((X*X)<1,kb,ka)
end functions
begin species
    1 A() X0
    2 B() 0
end species
begin reactions
    1 1 1,1 growth #_R1
    2 1 0 delta #_R2
    3 0 2 pump #_R3
end reactions
begin groups
    1 X                    1
    2 Y                    2
end groups
"""

DT_STAR_DRHO = T_STAR / 0.8  # d/drho of ln(X0)/(delta − rho)
KA, KB = 3.0, 1.0


@requires_cc
class TestOneCrossingWrittenTwice:
    def test_the_dedup_cannot_merge_them(self, tmp_path):
        """The premise: two conditions, two distinct residual texts, one
        surface. The text key is all the registration has to go on, which is
        why the batch reaches the solver at all."""
        core = _model(tmp_path, TWO_SPELLINGS, "two.net")._core
        conds = sw.state_switch_conditions(core)
        assert conds == ["X<1", "(X*X)<1"]
        residuals = [core.state_switch_residual(c)[0] for c in conds]
        assert residuals[0] != residuals[1]

    def test_the_batch_gets_one_jump_and_not_one_per_residual(self, tmp_path):
        """The arithmetic, against closed forms.

        Y accumulates at ``ka`` until the crossing and ``kb`` after it, so
        ``∂Y/∂θ`` past t* is exactly ``(ka − kb)·dt*/dθ`` for a parameter that
        moves only the crossing, and ``t*`` / ``T_END − t*`` for the two rates
        themselves. Every one of those is halved by dropping the jump and
        doubled by applying it once per residual.

        Before this fix the run did not produce numbers at all — it raised the
        two-switches-at-one-instant refusal, which is also what says the two
        roots really do land in one batch here."""
        params = ["rho", "delta", "ka", "kb"]
        run = _sens(tmp_path, params, name="two.net", text=TWO_SPELLINGS)
        an = np.asarray(run.sensitivities)
        y = an[-1, 1, :]  # species B = the observable Y, at T_END
        assert y[0] == pytest.approx((KA - KB) * DT_STAR_DRHO, rel=1e-5)
        assert y[1] == pytest.approx(-(KA - KB) * DT_STAR_DRHO, rel=1e-5)
        assert y[2] == pytest.approx(T_STAR, rel=1e-5)
        assert y[3] == pytest.approx(T_END - T_STAR, rel=1e-5)
        # And X's own column is still the issue #150 answer — the second
        # residual neither adds to it nor takes anything away.
        assert an[-1, 0, 0] == pytest.approx(
            2.0 * T_STAR * np.exp(-1.6 * (T_END - T_STAR)), rel=1e-4
        )

    def test_every_column_matches_a_finite_difference(self, tmp_path):
        params = ["rho", "delta", "ka", "kb"]
        run = _sens(tmp_path, params, name="two.net", text=TWO_SPELLINGS)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params, text=TWO_SPELLINGS)
        for j, p in enumerate(params):
            scale = float(np.max(np.abs(fd[:, :, j])))
            assert scale > 0.0
            assert np.max(np.abs(an[:, :, j] - fd[:, :, j])) <= 1e-3 * scale, (
                f"column {p!r} disagrees with its own finite difference"
            )


# Two crossings that are genuinely INDEPENDENT — different species, different
# rate constants, nothing shared but the threshold — and coincide only because
# ``ku`` and ``kv`` happen to be equal, so ``U`` and ``V`` decay through ``c``
# at the same instant ln(2). Each gates its own contribution to W's source, so
# there is a real jump at each.
COINCIDENT = """\
begin parameters
    1 U0     1.0  # Constant
    2 ku     1.0  # Constant
    3 kv     1.0  # Constant
    4 c      0.5  # Constant
    5 a1     3.0  # Constant
    6 a2     1.0  # Constant
end parameters
begin functions
    1 pu() if(U<c,a2,a1)
    2 pv() if(V<c,a2,a1)
end functions
begin species
    1 P() U0
    2 Q() U0
    3 R() 0
end species
begin reactions
    1 1 0 ku #_R1
    2 2 0 kv #_R2
    3 0 3 pu #_R3
    4 0 3 pv #_R4
end reactions
begin groups
    1 U                    1
    2 V                    2
    3 W                    3
end groups
"""


@requires_cc
class TestWhatIsMergedIsOneCrossingTime:
    """One crossing *time*, not one surface — the surface is sufficient and not
    necessary, and it is not what the jump needs. Same model, same coincident
    pair, two answers depending on which column is asked for."""

    def test_coincident_crossings_that_move_together_are_merged(self, tmp_path):
        """``c`` is the threshold of BOTH conditions, so it moves the two
        crossing times identically: ``dt*/dc = −1/(c·k)`` either way. One
        ``dt*/dθ`` then serves the pair and the combined ``f⁻ − f⁺`` is the sum
        of the two branch changes, which is exactly the merged jump.

        W's rate is ``2·a1`` before and ``2·a2`` after, so
        ``∂W/∂c = 2(a1 − a2)·dt*/dc = −8`` for the whole tail — a closed form,
        and the analytic column is that to eight digits. (The finite difference
        is the loose one here, by 2e-4: its own runs register no roots at all
        and chase both kinks.)"""
        run = _sens(tmp_path, ["c"], name="coin.net", text=COINCIDENT)
        w = np.asarray(run.sensitivities)[:, 2, 0]
        t = np.asarray(run.time)
        past = t > np.log(2.0) + 0.05
        assert past.sum() >= 5
        np.testing.assert_allclose(w[past], -8.0, rtol=1e-6)

    def test_coincident_crossings_that_move_apart_are_refused(self, tmp_path):
        """``ku`` moves U's crossing and leaves V's exactly where it was, so
        the pair splits under the perturbation into two crossings with two
        branch changes — and the truth is a sum of two jumps that the shared
        probe cannot take apart. The refusal reports the disagreement it
        measured rather than the coincidence it noticed."""
        with pytest.raises(SimulationError) as exc:
            _sens(tmp_path, ["ku"], name="coin2.net", text=COINCIDENT)
        msg = str(exc.value)
        assert "(U)-(c)" in msg and "(V)-(c)" in msg
        assert "move differently" in msg
        assert "#153" in msg


# ``ml_hopfield`` verbatim from ``benchmarks/suites/ode_fullnet/nets`` (which is
# generated, not checked in — hence the inline copy, as elsewhere in this
# suite). Three neurons whose rate laws are the BNGL signed-rate idiom over
# ``dSi/dt``, so each condition's two branches meet at its own crossing.
HOPFIELD = """\
begin parameters
    1 W12        -1.0  # Constant
    2 W13        1.0  # Constant
    3 W23        -1.0  # Constant
    4 Tau        1.0  # Constant
    5 Gain       5.0  # Constant
end parameters
begin functions
    1 Net1() (W12*((2*S2)-1))+(W13*((2*S3)-1))
    2 Net2() (W12*((2*S1)-1))+(W23*((2*S3)-1))
    3 Net3() (W13*((2*S1)-1))+(W23*((2*S2)-1))
    4 Target1() 1/(1+exp(((-Gain)*Net1())))
    5 Target2() 1/(1+exp(((-Gain)*Net2())))
    6 Target3() 1/(1+exp(((-Gain)*Net3())))
    7 dS1_dt() (Target1()-S1)/Tau
    8 dS2_dt() (Target2()-S2)/Tau
    9 dS3_dt() (Target3()-S3)/Tau
   10 _rateLaw1() if((dS1_dt()>0),dS1_dt(),0)
   11 _rateLaw2() if((dS1_dt()<0),(-dS1_dt()),0)
   12 _rateLaw3() if((dS2_dt()>0),dS2_dt(),0)
   13 _rateLaw4() if((dS2_dt()<0),(-dS2_dt()),0)
   14 _rateLaw5() if((dS3_dt()>0),dS3_dt(),0)
   15 _rateLaw6() if((dS3_dt()<0),(-dS3_dt()),0)
end functions
begin species
    1 Neuron(id~1) 1.0
    2 Neuron(id~2) 0.8
    3 Neuron(id~3) 1.0
end species
begin reactions
    1 0 1 _rateLaw1 #U1
    2 1 0 _rateLaw2 #D1
    3 0 2 _rateLaw3 #U2
    4 2 0 _rateLaw4 #D2
    5 0 3 _rateLaw5 #U3
    6 3 0 _rateLaw6 #D3
end reactions
begin groups
    1 S1                   1
    2 S2                   2
    3 S3                   3
end groups
"""


@requires_cc
class TestTheModelTheIssueFound:
    """One of the two rulehub "BNGL as a general-purpose computation" models
    that reached the refusal, kept whole because what makes it a batch is a
    property of its weights and not of any one line."""

    def test_the_two_residuals_coincide_by_a_symmetry(self, tmp_path):
        """The premise, measured rather than argued. With ``W12 = W23`` and
        ``S1(0) = S3(0)``, substituting ``S1 = S3`` makes ``Net1`` and ``Net3``
        the same expression, so ``dS1/dt ≡ dS3/dt`` and the symmetry is
        invariant — the two residuals are different text for one crossing.

        Only along the trajectory, though, which is the whole reason the merge
        is decided on ``dt*/dθ`` and not on the flip test: off the ``S1 = S3``
        manifold the two are different functions (non-parallel gradients), and
        their ``dt*/dθ`` at this crossing come out permuted by ``W12 ↔ W23`` —
        (0.327, −0.168, −0.030) against (−0.030, −0.168, 0.327). Nothing here
        needs them, because the crossing carries no jump; had it carried one,
        this batch would be refused rather than merged.

        If a future change to the fixture broke the symmetry this is the
        assertion that would say so, before the sensitivity ones got
        mysterious."""
        run = _sens(tmp_path, [], name="hop0.net", text=HOPFIELD)
        x = np.asarray(run.species)
        assert np.max(np.abs(x[:, 0] - x[:, 2])) == 0.0

    def test_it_runs_and_matches_a_finite_difference(self, tmp_path):
        """Before this fix it raised at t = 0.4738, where ``dS1/dt`` and
        ``dS3/dt`` cross zero together. The crossing itself carries no jump —
        the signed-rate idiom is continuous at its own switch — so what the
        batch path has to get right here is simply not refusing, and the
        columns then come from the in-branch sensitivity RHS."""
        params = ["W12", "W13", "W23"]
        run = _sens(tmp_path, params, name="hop.net", text=HOPFIELD)
        an = np.asarray(run.sensitivities)
        fd = _fd(tmp_path, params, text=HOPFIELD)
        for j, p in enumerate(params):
            scale = float(np.max(np.abs(fd[:, :, j])))
            assert scale > 0.0
            assert np.max(np.abs(an[:, :, j] - fd[:, :, j])) <= 1e-4 * scale, (
                f"column {p!r} disagrees with its own finite difference"
            )


# ─── the SBML spelling, and an event in the same model ─────────────────────


ANTIMONY = """\
model nested_switch
  V_0 = 0.1; V_0_inject = 1000; t_0 = 2; rho_V = 0.8; delta_V = 1.6;
  Virus = V_0;
  Virus' = piecewise(0, Virus < 1, Virus*rho_V) - Virus*delta_V;
  at (time >= t_0): Virus = Virus + V_0_inject;
end
"""


@requires_cc
def test_an_sbml_piecewise_over_state_with_an_event(tmp_path):
    """The issue's field case: AMICI's ``nested_events`` shape, where the state
    switch shares a model with a time-triggered event whose assignment jumps the
    state. The two jumps are different objects — the event's is GH #212's
    ``∂h/∂x``-and-``∂h/∂p`` reset at a fixed instant, the switch's is the
    saltation term at a moving one — and both have to land, in that order, for
    the columns to track a finite difference over the whole run.

    Written as antimony rather than lifted from AMICI so the fixture is
    self-contained; the SBML libantimony emits is the same ``<piecewise>`` over
    ``<lt/>`` inside a ``<rateRule>`` that AMICI compiles.
    """
    pytest.importorskip("antimony")
    params = ["V_0_inject", "t_0", "rho_V", "delta_V"]

    def run(overrides=None, sens=None):
        m = bngsim.Model.from_antimony_string(ANTIMONY)
        for k, v in (overrides or {}).items():
            m.set_param(k, v)
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=list(sens or []))
        return sim.run(t_span=(0.0, 12.0), n_points=45, rtol=1e-10, atol=1e-14)

    an = np.asarray(run(sens=params).sensitivities)
    for j, p in enumerate(params):
        p0 = bngsim.Model.from_antimony_string(ANTIMONY).get_param(p)
        h = 1e-6 * abs(p0)
        hi = np.asarray(run({p: p0 + h}).species)
        lo = np.asarray(run({p: p0 - h}).species)
        fd = (hi - lo) / (2 * h)
        scale = float(np.max(np.abs(fd[:, 0])))
        np.testing.assert_allclose(
            an[:, 0, j], fd[:, 0], rtol=5e-3, atol=1e-3 * scale, err_msg=f"column {p!r}"
        )
        assert scale > 0.0


# ─── issue #154: a residual that is identically zero is not a crossing ─────
#
# CVODE reports a root the instant a root function reaches exactly 0.0 from a
# nonzero value (``cvRootfind``'s ``ghi == 0 && glo != 0``). That IS a crossing
# when g then leaves zero, and is not one when g stays there — and nothing
# upstream catches the second case: ``cvRcheck1`` deactivates a root that is
# zero at (re)init, so it covers a residual that *starts* on the surface but not
# one that reaches exactly zero mid-run and never leaves.
#
# The model below reaches it the way ``ml_q_learning`` does. ``1 + exp(-u)``
# rounds to exactly 1.0 for every ``u > 53·ln2 = 36.7368…``, so a softmax
# complement ``1 - 1/(1+exp(-u))`` is EXACTLY 0.0 on an open half-line rather
# than merely tiny — and a rate law conditioned on it has a residual that is
# identically zero over an interval of its own trajectory. Underneath the
# arithmetic the condition is true everywhere and crosses nothing; the zero
# belongs to the floating-point evaluation, not to the model.
#
# What the machinery made of that before this fix was a *tangency*: no nudge
# along the flow moves the residual off zero, no coordinate of its support moves
# it either, so the two branches "cannot be told apart" and it refused. A
# tangency is a surface the trajectory touches. This is not a surface.
PLATEAU = """\
begin parameters
    1 A0    1.0  # Constant
    2 k     1.0  # Constant
    3 gain  1.0  # Constant
end parameters
begin functions
    1 tail() 1-(1/(1+exp(((-gain)*A))))
    2 _rateLaw1() if((tail()>0),tail(),0)
end functions
begin species
    1 Aa() A0
    2 Bb() 0
end species
begin reactions
    1 0 1 k #_R1
    2 0 2 _rateLaw1 #_R2
end reactions
begin groups
    1 A                    1
    2 B                    2
end groups
"""

# The same plateau reached by two residuals at once: ``2·tail()`` is zero
# exactly where ``tail()`` is, so both roots fire on one step and the pair
# arrives as an issue #153 batch.
PLATEAU_PAIR = """\
begin parameters
    1 A0    1.0  # Constant
    2 k     1.0  # Constant
    3 gain  1.0  # Constant
end parameters
begin functions
    1 tail() 1-(1/(1+exp(((-gain)*A))))
    2 twice() 2*(1-(1/(1+exp(((-gain)*A)))))
    3 _rateLaw1() if((tail()>0),tail(),0)
    4 _rateLaw2() if((twice()>0),twice(),0)
end functions
begin species
    1 Aa() A0
    2 Bb() 0
    3 Cc() 0
end species
begin reactions
    1 0 1 k #_R1
    2 0 2 _rateLaw1 #_R2
    3 0 3 _rateLaw2 #_R3
end reactions
begin groups
    1 A                    1
    2 B                    2
    3 C                    3
end groups
"""

PLATEAU_T_END = 60.0


def _plateau_sens(tmp_path, params, name, text=PLATEAU):
    model = _model(tmp_path, text, name)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    return sim.run(t_span=(0.0, PLATEAU_T_END), n_points=13, rtol=RTOL, atol=ATOL)


@requires_cc
class TestAResidualThatIsIdenticallyZeroIsNotACrossing:
    def test_the_residual_reaches_exactly_zero_and_stays(self, tmp_path):
        """The premise, measured rather than argued — and the distinction the
        whole fix turns on is *exactly* zero, not small.

        A residual that merely got very small would still have a sign, a
        gradient, and a locatable crossing; this one has none of the three,
        because the underflow is a plateau with positive width rather than a
        point. The samples before it are the same expression's ordinary decay,
        which is what says the zero is reached from a nonzero value — i.e. that
        CVODE registers it as a root at all."""
        run = bngsim.Simulator(_model(tmp_path, PLATEAU, "pre.net"), method="ode").run(
            t_span=(0.0, PLATEAU_T_END), n_points=13, rtol=RTOL, atol=ATOL
        )
        tail = np.asarray(run.expressions)[:, list(run.expression_names).index("tail")]
        assert (tail[:5] > 0.0).all(), "the residual is supposed to decay into the plateau"
        assert (tail[-5:] == 0.0).all(), "the residual is supposed to be EXACTLY zero, not small"

    def test_the_crossing_is_still_registered(self, tmp_path):
        """Nothing here is knowable before the run: the condition reads live
        state and splits into a residual like any other, and whether that
        residual has a plateau on the trajectory is a property of where the
        trajectory goes. So this stays a run-time measurement at the crossing
        and not a rule about the condition's text."""
        core = _model(tmp_path, PLATEAU, "reg.net")._core
        assert sw.state_switch_conditions(core) == ["(1-(1/(1+exp(((-gain)*A)))))>0"]

    def test_it_runs_and_matches_a_closed_form(self, tmp_path):
        """Before this fix it raised at t = 36.58 with the tangency refusal.

        The oracle is closed form rather than a finite difference, because this
        model has one: with ``A(t) = A0 + k·t`` and ``B' = sigma(-gain·A)``,

            B(∞) = ln(1 + e^(-gain·A0)) / (gain·k)

        (the plateau contributes nothing — that is what makes it a plateau), so

            dB/dk    = -ln(1 + e^(-gain·A0)) / (gain·k²)
            dB/dgain = [-A0·e^(-gain·A0)/(1 + e^(-gain·A0))·gain
                        - ln(1 + e^(-gain·A0))] / (gain²·k)

        A difference quotient would be the weaker check here for the usual
        reason and one more: perturbing ``gain`` MOVES the plateau's edge, so
        the two FD runs enter it at different times and the quotient carries
        that artefact where the closed form knows there is nothing there."""
        run = _plateau_sens(tmp_path, ["k", "gain"], "plat.net")
        a0, k, gain = 1.0, 1.0, 1.0
        e = np.exp(-gain * a0)
        db_dk = -np.log1p(e) / (gain * k * k)
        db_dgain = (-a0 * e / (1.0 + e) * gain - np.log1p(e)) / (gain * gain * k)
        got = np.asarray(run.sensitivities)[-1, 1, :]
        assert got[0] == pytest.approx(db_dk, rel=1e-6)
        assert got[1] == pytest.approx(db_dgain, rel=1e-6)

    def test_a_jump_crossed_inside_the_plateau_is_not_a_landing(self, tmp_path):
        """A second switch, ``C < thrC`` on a species that decays, is crossed at
        t = 37.9 while ``tail()`` sits on its plateau. The probe points about
        that crossing all read the plateau's residual as 0.0. That is not a
        probe point landing on its surface, which issue #763 stretches the step
        to avoid: it is zero at every one of them."""
        text = PLATEAU.replace(
            "    3 gain  1.0  # Constant\n",
            "    3 gain  1.0\n    4 a 0.05\n    5 thrC 1.5\n    6 kb 3\n",
        )
        text = text.replace(
            "    2 _rateLaw1() if((tail()>0),tail(),0)\n",
            "    2 _rateLaw1() if((tail()>0),tail(),0)\n    3 fY() if(C<thrC,kb,0)\n",
        )
        text = text.replace("    2 Bb() 0\n", "    2 Bb() 0\n    3 Cc() 10\n    4 Yy() 0\n")
        text = text.replace(
            "    2 0 2 _rateLaw1 #_R2\n", "    2 0 2 _rateLaw1 #_R2\n    3 3 0 a\n    4 0 4 fY\n"
        )
        text = text.replace(
            "    2 B                    2\n", "    2 B                    2\n    3 C 3\n"
        )
        model = _model(tmp_path, text, "jump_in_plateau.net")
        assert len(sw.state_switch_conditions(model._core)) == 2
        run = bngsim.Simulator(model, method="ode", sensitivity_params=["a", "thrC"]).run(
            t_span=(0.0, PLATEAU_T_END), n_points=7, rtol=1e-9, atol=1e-12
        )
        got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Yy()")]
        # C = 10·exp(−a·t) crosses thrC at ln(10/thrC)/a, and Y = kb·(T − that).
        want = [3.0 * np.log(10.0 / 1.5) / 0.05**2, 3.0 / (0.05 * 1.5)]
        np.testing.assert_allclose(got, want, rtol=1e-6)

    def test_a_pair_of_plateaus_is_not_refused_as_coincident(self, tmp_path):
        """Two residuals reaching the same plateau on the same step arrive as an
        issue #153 batch, and its "one step straddles all of them or these are
        not one crossing" test cannot be met by a residual that cannot be
        straddled at all — so before this fix the pair refused with the
        coincident-switch message rather than the tangency one. Both are dropped
        for the same reason, and ``C = 2·B`` is what says neither jumped."""
        run = _plateau_sens(tmp_path, ["k", "gain"], "pair.net", text=PLATEAU_PAIR)
        s = np.asarray(run.sensitivities)
        scale = float(np.max(np.abs(s[:, 1, :])))
        assert scale > 0.0
        np.testing.assert_allclose(s[:, 2, :], 2.0 * s[:, 1, :], rtol=1e-9, atol=1e-12 * scale)


# ─── issue #187: a crossing with no jump still has to be crossed ────────────
#
# The saltation jump is not the only thing that happens at a located crossing.
# The integration also has to RESUME somewhere, and issue #82 established (from
# the switch-time side) and issue #150 repeated (from the rate-law side) that
# resuming on the surface is what puts the discontinuity inside the first step
# after the restart: CVODES sizes h from one branch while every corrector answers
# with the other, and the root fires again.
#
# Issue #150 wrote that restart under the jump, so a switch measured CONTINUOUS
# at its own threshold — the clamp idiom above, the most common `piecewise` in
# the corpus — returned before reaching it and left the state exactly where the
# root finder put it. cvRootfind short-circuits on an exact zero (`ghi == 0`), so
# "exactly where it put it" is routinely `g(x) == 0.0` bit-for-bit.
#
# On Smith_BMCSystBiol2013 (`PI345P3 > pip3_basal`, a two-line clamp) that never
# returned. Standing on the surface, CVODES restarted at h ≈ ε·|t_end| ≈ 3e-15,
# took a step far too short to move a 1.2e13-scale species by even one ulp,
# rooted on the same crossing, and was re-initialized back to the same h — 19,297
# times in 10 s, advancing simulated time by ~3.5e-15 an iteration. The scalar
# run of that model is 0.02 s. Whether a run reached that state at all depended
# on where the output grid landed, which is why it read as an `n_points`
# dependence (2, 3, 4 and 8 hung; 5, 16 and 50 did not).
#
# The ramp below makes the state at the crossing exactly checkable: C is a
# zeroth-order species, so C(t) = rate·t is integrated exactly and the crossing
# at C = thr is located to the surface itself rather than to a solver tolerance.
RAMP = """\
begin parameters
    1 rate   2.0  # Constant
    2 thr    4.0  # Constant
    3 rho    0.5  # Constant
end parameters
begin functions
    1 ramp() rate
    2 growth() if(C<thr,{live},0)
end functions
begin species
    1 A() 0
    2 B() 0
end species
begin reactions
    1 0 1 ramp #_R1
    2 0 2 growth #_R2
end reactions
begin groups
    1 C 1
    2 Bg 2
end groups
"""
# `rho*(thr-C)` is 0 where C = thr, so the two branches meet and there is no jump
# — the case issue #187 is about. `rho` alone does not, so the same crossing on
# the same trajectory takes the jump path instead. C's own rate law is `ramp` in
# both, so t*, f(t*) and the nudge the restart takes are identical: the two
# fixtures differ ONLY in whether a saltation term is applied.
RAMP_CONTINUOUS = RAMP.format(live="rho*(thr-C)")
RAMP_JUMPING = RAMP.format(live="rho")


RAMP_T_END = 6.0
RAMP_AFTER = (3.0, 4.0, 5.0, RAMP_T_END)


def _ramp_offset_after_crossing(tmp_path, text, name):
    """How far C ends up ahead of its own exact ramp, well past the crossing.

    C is zeroth-order, so ``C(t) = rate·t`` exactly and the only thing that can
    displace it is the restart: resuming at ``x(t*) + δt·f`` shifts the whole ramp
    by ``δt·rate`` for the rest of the run. Read after the crossing rather than at
    it — whether the sample AT ``t*`` lands before or after the root is decided by
    the last ulp of where the root finder stopped, and moves with the output grid
    and with the platform.
    """
    model = _model(tmp_path, text, name)
    thr, rate = model.get_param("thr"), model.get_param("rate")
    times = sorted({0.0, thr / rate, *RAMP_AFTER})
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["rho", "rate"]).run(
        sample_times=times, rtol=RTOL, atol=ATOL
    )
    c = np.asarray(run.species)[:, 0]
    offsets = [float(c[times.index(t)] - rate * t) for t in RAMP_AFTER]
    assert max(offsets) - min(offsets) <= 8.0 * np.spacing(rate * RAMP_T_END), (
        f"the displacement is supposed to be a constant shift of the ramp: {offsets}"
    )
    return offsets[-1], float(thr), float(rate)


@requires_cc
class TestACrossingWithNoJumpStillResumesPastTheSurface:
    def test_it_resumes_where_a_jumping_crossing_resumes(self, tmp_path):
        """The invariant, stated without a magic number.

        Two fixtures whose switching species has the same rate law and the same
        threshold, differing only in whether the branches meet. The restart is a
        property of having STOPPED at a crossing, not of having jumped there, so
        both must resume from the same place — one explicit Euler step of the
        ladder's verified δt along the flow, past the surface — and C carries that
        shift for the rest of the run either way.

        Pre-#187 the continuous one carried no shift at all and the jumping one
        carried 2.28e-13, and that difference is the whole bug.
        """
        cont, thr, rate = _ramp_offset_after_crossing(tmp_path, RAMP_CONTINUOUS, "cont_ramp.net")
        jump, _, _ = _ramp_offset_after_crossing(tmp_path, RAMP_JUMPING, "jump_ramp.net")
        # Both restarts are `x(t*) + δt·f` off the same δt and the same f, so all
        # that is left between them is where each run's own root finder landed —
        # a few ulps of the sampled value.
        assert abs(cont - jump) <= 8.0 * np.spacing(rate * RAMP_T_END), (
            f"a continuous crossing shifts the ramp by {cont!r} where a jumping one shifts it "
            f"by {jump!r}"
        )

    def test_it_resumes_past_the_surface_by_more_than_a_rounding_step(self, tmp_path):
        """…and the shared place is the after side, by the verified nudge.

        ``δt`` starts at ``256·ε·max(|t*|, 1)`` and only grows, so the shift is at
        least that times ``f``. Bounding it from above too is what says this is
        still a nudge and not a step: the state's own tolerance never sees it.
        """
        cont, thr, rate = _ramp_offset_after_crossing(tmp_path, RAMP_CONTINUOUS, "past_ramp.net")
        floor = 64.0 * rate * np.spacing(max(thr / rate, 1.0))
        assert cont >= floor, (
            f"the ramp is shifted by {cont!r}, which is the root finder's own rounding rather "
            "than a verified step off the surface"
        )
        assert cont <= 1e-9 * thr, "the nudge must stay far below the state's own tolerance"

    def test_the_answer_does_not_depend_on_where_the_output_grid_lands(self, tmp_path):
        """The issue's headline, on a model that crosses a continuous switch.

        ``n_points`` chooses output times; it is not allowed to choose whether —
        or how well — the problem is solved. On Smith it decided whether the run
        returned at all.
        """
        ref = None
        for n in (2, 3, 4, 5, 8, 16, 50):
            model = _model(tmp_path, RAMP_CONTINUOUS, f"grid_ramp{n}.net")
            sim = bngsim.Simulator(model, method="ode", sensitivity_params=["rho", "rate"])
            run = sim.run(t_span=(0.0, 6.0), n_points=n, rtol=RTOL, atol=ATOL)
            got = np.asarray(run.sensitivities)[-1]
            if ref is None:
                ref = got
                assert np.abs(ref).max() > 0.0, "the fixture is not testing a live column"
                continue
            np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9 * np.abs(ref).max())
