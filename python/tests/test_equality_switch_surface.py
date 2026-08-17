"""Issue #381 — an equality names the surface its own true-set is bounded by.

``MODEL2003190004``'s forward-sensitivity solve stalled outright::

    CVODE made no progress while integrating to the next output point

at t ≈ 2.82673, for 33 of its 43 shared parameters. The plain ODE run is fine and
AMICI produces an oracle, so the failure is the sensitivity path's alone.

## The chain

The model gates its synthesis rate on ``APC <= 0.2``, but spells it as an
``<or/>`` of ``<eq/>`` and ``<lt/>`` over one pair of operands::

    <assignmentRule variable="ks">
      <piecewise>
        <piece> 0.5 <apply><or/>
          <apply><eq/> <ci>APC</ci> <cn>0.2</cn> </apply>
          <apply><lt/> <ci>APC</ci> <cn>0.2</cn> </apply>
        </apply> </piece>
        <otherwise> 0 </otherwise>
      </piecewise>
    </assignmentRule>

:func:`bngsim._switch_sensitivity._split_logical_atoms` hands a disjunction over
as its two atoms, and the issue #68 gate requires *every* atom's crossing to be
compensated. ``APC<0.2`` was: issue #150 roots on ``APC − 0.2`` and jumps the
saltation term there. ``APC==0.2`` was not — ``NetworkModel::state_switch``
refused an equality, on the grounds that a continuous trajectory satisfies one
only on a measure-zero set.

That refusal is right about the geometry and wrong about the consequence. It
declined the analytic sensitivity RHS for the **whole model**, and CVODES'
difference quotient then answered every column — probing ``f`` at ``y + σ·s``,
which just past a crossing lands on the other branch. The stall was at exactly
the crossing the ``<`` half had already earned a root for.

## What changed

``state_switch`` now builds ``(lhs)-(rhs)`` for ``==``, ``!=`` and ExprTk's
single ``=`` as well. The two callers of that splitter want different things and
now differ on exactly that operator:

* an **event trigger** (issue #144) fires on a rising edge, and ``x == c`` has
  none a root finder can straddle — still refused, pinned below and in
  ``test_event_sensitivity.py``;
* a **rate-law switch** (issue #150) needs the surface its branch can change
  across, and for ``x == c`` that is the ``x − c = 0`` that ``x < c`` names.

Whether the branch really changes there is measured at the root rather than
assumed: the branch-gap probe in ``apply_state_switch_sensitivity_jump`` returns
with no jump when the two sides of the surface evaluate the same, which is what a
*lone* equality always does.

## What this locks

1. the witness returns a finite tensor rather than stalling, and no condition in
   it declines the analytic sensitivity RHS;
2. the redundant spelling costs one root, not two — both atoms resolve to the
   same residual, so the detector registers the single crossing ``APC <= 0.2``
   would have given;
3. a lone equality changes no answer: the branch it selects is never live over an
   interval, so the columns match the model without it to a finite difference's
   own accuracy;
4. the event path still refuses an equality, asked of the very same model.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg
from bngsim import _switch_sensitivity as sw

pytest.importorskip("sympy")

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"

# The witness, and the manifest horizon / tolerances issue #381 reproduced at.
_WITNESS = "MODEL2003190004"
_T_SPAN = (0.0, 100.0)
_N_POINTS = 101
_RTOL, _ATOL = 1e-9, 1e-12

# The atoms of the witness's one condition, and the crossing they both name.
_EQ_ATOM = "APC==0.2"
_LT_ATOM = "APC<0.2"


def _path() -> Path:
    return _MODELS_DIR / _WITNESS / f"{_WITNESS}.xml"


def _load():
    if not _path().is_file():
        pytest.skip(f"rr_parity corpus model {_WITNESS} not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Model.from_sbml(str(_path()))


# ─── the witness ───────────────────────────────────────────────────────────


class TestTheWitness:
    def test_the_stalled_solve_now_returns(self):
        """The issue's own reproducer, reduced to one of the 33 failing columns.

        ``V2c`` reads ``APC`` through the rate law the switch gates, so it is one
        of the columns whose difference quotient walked into the crossing. Asked
        through ``run()`` rather than ``compute_all_sensitivities``, whose
        column-by-column retry would rescue a transient failure and hide this
        one (issue #401).
        """
        m = _load()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run = bngsim.Simulator(m, method="ode", sensitivity_params=["V2c"]).run(
                _T_SPAN, _N_POINTS, rtol=_RTOL, atol=_ATOL
            )
        s = np.asarray(run.sensitivities)
        assert s.shape[0] == _N_POINTS and s.shape[2] == 1
        assert np.all(np.isfinite(s))
        assert np.any(s != 0.0), "a column that is identically zero is the silent failure"

    def test_no_condition_in_the_witness_declines_the_analytic_rhs(self):
        """The fix at the gate rather than at the symptom. Before #381 this
        returned the issue #68 message naming ``APC==0.2``; the model then ran
        every column on the difference quotient, and that is what stalled."""
        core = _load()._core
        ctx = core.functional_jacobian_context()
        scope = sw.switch_condition_scope(core, ctx)
        func_map = dict(ctx["function_map"])
        from bngsim._jacobian import _inline_functions

        for body in func_map.values():
            flat = _inline_functions(body, func_map) or body
            assert sw.uncompensated_condition_reason(flat, scope) is None

    def test_the_analytic_sensitivity_rhs_is_emitted(self):
        _src, has_sens = cg.generate_combined_from_model(_load())
        assert has_sens is True

    def test_the_two_atoms_are_one_crossing(self):
        """The redundant spelling costs one root, not two.

        ``state_switch_conditions`` deduplicates by *residual*, so the ``<or/>``
        of ``<eq/>`` and ``<lt/>`` registers the single crossing the ``<=``
        spelling would have. Two coincident roots on one surface is what the
        solver refuses as an ambiguous simultaneous pair (issue #153), so this is
        not cosmetic."""
        core = _load()._core
        eq_residual, why = core.state_switch_residual(_EQ_ATOM)
        assert eq_residual and not why
        assert eq_residual == core.state_switch_residual(_LT_ATOM)[0]
        # ...and the root is the `<` half's. The equality is admitted at the
        # gate, never registered by the detector — see
        # TestALoneEqualityMustNotEarnARootOfItsOwn.
        assert sw.state_switch_conditions(core) == [_LT_ATOM]


# ─── the two callers, on one model ─────────────────────────────────────────


def test_the_event_path_still_refuses_the_equality_the_switch_path_admits():
    """The one operator the two callers now disagree about, asked of one model.

    An event needs a *rising edge* to fire and to differentiate ``dt*/dp`` at, and
    ``Sobs == 50`` has none a root finder can straddle — which is what the
    original refusal was about, and it is still right there. A rate-law switch
    needs the *surface* its branch can change across, and that surface exists.
    Same text, same evaluator, two answers.
    """
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("k", 0.1)
    s = b.add_species("S", 100.0)
    b.add_reaction([s], [], "elementary", "k")
    b.add_observable("Sobs", [(s, 1.0)])
    b.add_event("evt", "Sobs == 50", [(s, "2.0")])
    core = b.build()

    residual, why = core.state_switch_residual("Sobs == 50")
    assert residual and not why, "the rate-law switch path must see the surface"

    reason = core.event_sensitivity_unsupported_reason(["k"]) or ""
    assert "equality" in reason, (
        f"an equality trigger must not resolve to a rising edge; got {reason!r}"
    )


# ─── the operand that is itself a comparison ───────────────────────────────


class TestABooleanDifferenceIsNotASurface:
    """The hole this change would otherwise have opened.

    ``(I>thresh) != (sigma>100)`` is the SBML ``<xor/>`` idiom, and the splitter
    finds its operator at depth 0 — so the residual comes out
    ``(I>thresh) - (sigma>100)``, a difference of two BOOLEANS. That is a step,
    not a surface: its gradient is zero wherever it is defined, so there is
    nothing for CVODE to bracket and nothing for ``dt*/dθ``'s denominator
    (``∂g/∂y·f``) to be. Admitting it would be the silent zero the issue #68 gate
    exists to stop — the gate's admission IS the promise that issue #150 located
    the crossing and will jump it.

    The ``<`` spelling was never safe either; it was only ever saved by an
    accident downstream, sympy declining to parse ``Lt(Gt(…), Gt(…))`` — while
    the root was registered regardless. Both spellings are refused at the
    recognizer now, which is the one place that can answer for both callers.
    """

    @pytest.mark.parametrize("op", ["!=", "<", "==", ">="])
    def test_a_comparison_of_comparisons_is_refused(self, tmp_path, op):
        core = _growth(
            tmp_path, f"if((I>thresh){op}(beta>1.0), 0, beta*I)", f"b{ord(op[0])}.net"
        )._core
        residual, why = core.state_switch_residual(f"(I>thresh){op}(beta>1.0)")
        assert not residual
        assert "itself a comparison" in why, why

    def test_it_registers_no_root_and_declines_the_gate(self, tmp_path):
        """Both halves, so "refused" cannot mean "refused by the gate but rooted
        anyway" — which is what the ``<`` spelling did before."""
        core = _growth(tmp_path, "if((I>thresh)!=(beta>1.0), 0, beta*I)", "bd.net")._core
        assert sw.state_switch_conditions(core) == []
        _terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is not None

    def test_a_comparison_merely_CONTAINING_one_is_still_a_surface(self, tmp_path):
        """The line is drawn at the operand's OWN outermost operator, and it has
        to be: seven corpus residuals are arithmetic over an inner ``if()`` whose
        head is a comparison — ``(… if(PO2AMB>80, 80, PO2AMB) …) < 0`` — and
        those are surfaces the trajectory really does cross. Refusing by "names a
        comparison anywhere" would take MODEL0911270005, issue #382's witness,
        with it."""
        core = _growth(
            tmp_path, "if((if(I>thresh, I, thresh) - 2.0) < 0.0, 0, beta*I)", "inner.net"
        )._core
        residual, why = core.state_switch_residual("(if(I>thresh, I, thresh) - 2.0)<0.0")
        assert residual and not why


# ─── a lone equality changes no answer ─────────────────────────────────────


_GROWTH = """\
begin parameters
    1 I0      1.0  # Constant
    2 beta    0.2  # Constant
    3 thresh  4.0  # Constant
end parameters
begin functions
    1 rate() BODY
end functions
begin species
    1 person(state~I) I0
end species
begin reactions
    1 0 1 rate #_R1
end reactions
begin groups
    1 I 1
end groups
"""


def _growth(tmp_path, body: str, name: str):
    net = tmp_path / name
    net.write_text(_GROWTH.replace("BODY", body))
    return bngsim.Model.from_net(net)


def _run(model, params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = bngsim.Simulator(model, method="ode", sensitivity_params=list(params)).run(
            t_span=(0.0, 10.0), n_points=51, rtol=1e-11, atol=1e-13
        )
    return np.asarray(run.species), np.asarray(run.sensitivities)


class TestALoneEqualityMustNotEarnARootOfItsOwn:
    """``if(I==thresh, 0, beta*I)`` is ``beta*I`` at every time but one — and a
    root of its own is what would make that false.

    The equality's true-set has empty interior, so the branch it selects is never
    live over an interval and the exact trajectory is the unconditional one. A
    CVODE root on ``I − thresh`` breaks exactly that. The root finder does not
    step *over* the surface, it stops the integrator *on* it — and there the
    equality is true, the zero branch is live, and ``I`` never leaves. The
    measure-zero set stops being measure-zero because the solver was told to land
    in it.

    That is not hypothetical. With the equality registered, ``ubuntu-latest``
    returned ``I(t)`` climbing normally to exactly 4.0 = ``thresh`` and then
    holding 4.0 for the rest of the run, against 7.389 unconditional; macOS
    landed a few ulps off the surface and did not latch. A platform-split
    trajectory is the worst possible form for this to take.

    So the equality is admitted at the GATE — there is no branch interval, so
    there is nothing to compensate — and skipped by the DETECTOR, which is the
    one place a root gets registered. The redundant spelling in
    MODEL2003190004 still gets its root, because the ``<`` half earns it
    (:class:`TestTheWitness`).
    """

    BODY = "if(I==thresh, 0, beta*I)"
    PLAIN = "beta*I"

    def test_no_root_is_registered_for_it(self, tmp_path):
        """The structural half, which holds on every platform — unlike the
        numeric half below, which only catches the latch when the root finder
        happens to land bit-exactly on the surface."""
        core = _growth(tmp_path, self.BODY, "eq.net")._core
        assert sw.state_switch_conditions(core) == []

    def test_the_gate_still_admits_it(self, tmp_path):
        """Admitted and rooted are different things, and this pair is the only
        place that says so out loud. Losing the admission would put the model
        back on the difference quotient — the #381 regression — and gaining the
        root would latch it."""
        core = _growth(tmp_path, self.BODY, "eq.net")._core
        scope = sw.switch_condition_scope(core)
        assert sw.uncompensated_condition_reason(self.BODY, scope) is None

    def test_the_condition_does_not_decline_the_analytic_rhs(self, tmp_path):
        core = _growth(tmp_path, self.BODY, "eq.net")._core
        terms, reason = cg._functional_dfdp_terms(core, core.codegen_data())
        assert reason is None and terms

    def test_the_answer_is_the_unconditional_one(self, tmp_path):
        """The numbers, species and sensitivities alike. Two different ``.so``s
        and two different root sets, so not bit-identical — but the difference is
        integrator noise, not a jump."""
        x_eq, s_eq = _run(_growth(tmp_path, self.BODY, "eq.net"), ["beta", "I0"])
        x_plain, s_plain = _run(_growth(tmp_path, self.PLAIN, "plain.net"), ["beta", "I0"])
        # Named separately because it is the latch's own signature: `I` grows
        # through thresh=4.0 at t ≈ 6.9 and must keep going. Holding 4.0 to the
        # end of the window is what a root on the surface produced.
        assert x_eq[-1, 0] > 4.0 + 1.0, f"trajectory latched at {x_eq[-1, 0]}"
        assert np.allclose(x_eq, x_plain, rtol=1e-8, atol=1e-10)
        assert np.allclose(s_eq, s_plain, rtol=1e-6, atol=1e-9)
        # and the columns are not trivially zero on either side
        assert np.abs(s_plain).max() > 1.0

    def test_the_beta_column_matches_a_central_finite_difference(self, tmp_path):
        """The oracle that does not share the analytic path's assumptions.

        ``d/dbeta`` of ``I(t) = I0·exp(beta·t)`` is ``I0·t·exp(beta·t)``, and a
        central difference of the *trajectory* reproduces it without ever
        differentiating the ``Piecewise``. If admitting the equality had injected
        a spurious saltation jump at the crossing, it would show here."""
        _x, s = _run(_growth(tmp_path, self.BODY, "eq.net"), ["beta"])
        h = 1e-6
        up = _growth(tmp_path, self.BODY, "up.net")
        up.set_param("beta", 0.2 + h)
        dn = _growth(tmp_path, self.BODY, "dn.net")
        dn.set_param("beta", 0.2 - h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_up = np.asarray(
                bngsim.Simulator(up, method="ode")
                .run(t_span=(0.0, 10.0), n_points=51, rtol=1e-12, atol=1e-14)
                .species
            )
            x_dn = np.asarray(
                bngsim.Simulator(dn, method="ode")
                .run(t_span=(0.0, 10.0), n_points=51, rtol=1e-12, atol=1e-14)
                .species
            )
        fd = (x_up - x_dn) / (2 * h)
        assert np.allclose(s[:, :, 0], fd, rtol=1e-5, atol=1e-6)
