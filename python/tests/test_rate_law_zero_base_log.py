"""GH #333 — a rate law's own value at a zero logarithm base.

#310 and #317 take the limit of ``base^exp · ln(base)`` at ``base == 0`` on the
way out of the two *derivative* emitters. The rate law's own value never passed
through them: ExprTk evaluates it straight from the model's text, and the codegen
C emitter prints that same text. So ``vmax*Atot^n*ln(Atot)`` answered ``NaN`` for
``f`` at ``Atot = 0`` — CVODE failing at the first call with ``flag=-9`` — while
its own ``∂f/∂n`` answered the limit. One floating-point value away, at
``Atot = 1e-30``, the identical model ran to completion.

The rewrite lands on a function's **evaluation** expression, never on its
declared one, and that separation is the design rather than an implementation
detail. Writing the limit as a run-time branch puts a state-dependent condition
(``Atot == 0``) into the rate law, which the forward-sensitivity path correctly
reads as a state switch whose crossing time moves and refuses (the #52 / #150
machinery). Overwriting the declared law was measured doing exactly that: it
bought a finite value at one point and lost the analytic sensitivity RHS for the
whole run. Keeping the declared law smooth lets the derivative emitters go on
differentiating it and apply their own guard to the result, while only the two
value consumers read the branch.

Those two value consumers are the point of ``TestBothRhsEmitters``. bngsim has
*two* RHS emitters — one that builds C from a loaded model and one that builds it
from the ``.net`` file — and a guard on either alone would leave them disagreeing
about the same rate law at the same point.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("sympy")

from bngsim._codegen import generate_rhs_c, generate_rhs_from_model  # noqa: E402
from bngsim._jacobian import guard_rate_law_text  # noqa: E402

# A is produced from B and never consumed, so it leaves zero immediately and
# rises monotonically — the logarithm's argument never goes negative, where the
# NaN would be real rather than removable. Reaction 2 is the log-bearing term,
# evaluated at Atot = 0 on the very first RHS call, and it only produces C, so
# nothing feeds back into the logarithm's own species.
LOG_RATE_LAW_NET = """\
begin parameters
    1 n      3.0  # Constant
    2 vmax   1.5  # Constant
    3 kdeg   0.4  # Constant
end parameters
begin functions
    1 basal()   kdeg*Btot
    2 logterm() vmax*Atot^n*ln(Atot)
end functions
begin species
    1 A() 0.0
    2 B() 4.0
    3 C() 1.0
end species
begin reactions
    1 2 1 basal   #_R1
    2 0 3 logterm #_R2
end reactions
begin groups
    1 Atot                 1
    2 Btot                 2
    3 Ctot                 3
end groups
"""


@pytest.fixture
def log_net(tmp_path):
    net = tmp_path / "logterm.net"
    net.write_text(LOG_RATE_LAW_NET)
    return net


# ─── the text rewrite ──────────────────────────────────────────────────────


class TestTheRewrite:
    def test_a_log_rate_law_gains_the_branch(self):
        guarded = guard_rate_law_text("vmax*Atot^n*ln(Atot)")
        assert guarded is not None
        assert "if(" in guarded
        assert "Atot == 0" in guarded

    def test_a_rate_law_without_a_logarithm_is_left_alone(self):
        """The substring gate, which is what keeps this off the derivation
        budget: 97.9% of corpus rate laws mention no logarithm and must not
        reach sympy at all."""
        assert guard_rate_law_text("vmax*Atot^n/(KM^n + Atot^n)") is None
        assert guard_rate_law_text("kdeg*Btot") is None

    def test_a_logarithm_needing_no_guard_is_left_alone(self):
        """A logarithm with nothing to pair against is not a removable
        singularity, so it buys no branch."""
        assert guard_rate_law_text("vmax*ln(KM)") is None
        assert guard_rate_law_text("vmax*log10(Atot)") is None

    def test_the_rewrite_keeps_the_instrumented_spelling_of_the_logarithm(self):
        """``ln`` and ``log`` are the same function to ExprTk and not to bngsim:
        ``ln`` is a registered adapter carrying the ``NonFiniteWarningSet``
        (issue #42's follow-up), ``log`` is ExprTk's uninstrumented built-in.

        Every rate law this rewrites was declared with ``ln``, so emitting
        ``log`` traded the model's own non-finite diagnostic away for free — and
        that warning is the only thing that names the rate law when a solve later
        dies on a bare CVODE flag. The C emitters translate either spelling to
        C's ``log``, so this costs the compiled path nothing.
        """
        guarded = guard_rate_law_text("vmax*Atot^n*ln(Atot)")
        assert guarded is not None
        assert "ln(" in guarded
        assert "log(" not in guarded

    def test_an_existing_conditional_is_left_alone(self):
        """Two reasons, and either is sufficient. It makes the pass idempotent —
        re-running it over text this rewrite produced cannot wrap the guard
        twice. And it declines to re-spell a conditional the *modeller* wrote,
        which matters because GH #335 means a hand-written ``==`` does not
        survive the parser intact."""
        already = guard_rate_law_text("vmax*Atot^n*ln(Atot)")
        assert already is not None
        assert guard_rate_law_text(already) is None
        assert guard_rate_law_text("if(t>1,vmax*Atot^n*ln(Atot),0)") is None


# ─── the model ─────────────────────────────────────────────────────────────


class TestTheModel:
    def test_the_declared_law_is_untouched_and_the_evaluated_one_is_guarded(self, log_net):
        """The whole design in one assertion. The declared text is what gets
        differentiated and must stay smooth; the evaluated text carries the
        branch."""
        model = bngsim.Model.from_net(log_net)

        declared = dict(
            zip(model._core.function_names, model._core.function_expressions, strict=False)
        )
        evaluated = dict(
            zip(model._core.function_names, model._core.function_eval_expressions, strict=False)
        )

        assert declared["logterm"] == "vmax*Atot^n*ln(Atot)"
        assert "if(" not in declared["logterm"]
        assert "if(" in evaluated["logterm"]
        # A function that needs no guard carries no evaluation expression at all.
        assert evaluated["basal"] == ""

    def test_the_rate_law_is_the_limit_at_a_zero_base(self, log_net):
        """``lim_{S→0+} vmax·S^n·ln(S) = 0``. Before the fix this was the NaN
        that killed the first RHS call."""
        model = bngsim.Model.from_net(log_net)
        at_zero = model._core._eval_functions(0.0, [0.0, 4.0, 1.0])
        assert at_zero["logterm"] == 0.0

    def test_a_negative_base_keeps_its_nan(self, log_net):
        """The limit exists only from above. A negative concentration makes
        ``ln`` genuinely undefined, and #310's contract is that such a NaN is
        the honest answer and stays one."""
        model = bngsim.Model.from_net(log_net)
        with np.errstate(divide="ignore", invalid="ignore"):
            below = model._core._eval_functions(0.0, [-1e-9, 4.0, 1.0])
        assert np.isnan(below["logterm"])

    def test_a_negative_base_still_reports_itself(self, log_net, capfd):
        """...and it must still say so. A solve that wanders below zero fails
        with a bare CVODE flag naming no species and no rate law, so this warning
        is the only thread back to the cause — which is exactly why the rewrite
        may not quietly move the call off the instrumented ``ln`` adapter."""
        model = bngsim.Model.from_net(log_net)
        capfd.readouterr()  # drop anything the load itself printed
        with np.errstate(divide="ignore", invalid="ignore"):
            model._core._eval_functions(0.0, [-1e-9, 4.0, 1.0])
        assert "ln(-1e-09)" in capfd.readouterr().err


# ─── both RHS emitters ─────────────────────────────────────────────────────


class TestBothRhsEmitters:
    """bngsim builds the RHS twice — from a loaded model, and from the ``.net``
    file — and a guard on one only would leave two emitters disagreeing about
    the same rate law at the same point. The rewrite is shared
    (``guard_rate_law_text``), not reimplemented at each site.
    """

    def test_the_model_based_emitter_guards_the_function_body(self, log_net):
        c = generate_rhs_from_model(bngsim.Model.from_net(log_net))
        body = [ln for ln in c.splitlines() if "logterm" in ln and "func[" in ln]
        assert body, "no function body emitted for logterm"
        assert "== 0.0" in body[0] and "?" in body[0]

    def test_the_net_file_emitter_guards_the_function_body(self, log_net):
        c = generate_rhs_c(str(log_net))
        body = [ln for ln in c.splitlines() if "func_logterm" in ln and "=" in ln]
        assert body, "no function body emitted for logterm"
        assert "== 0.0" in body[0] and "?" in body[0]

    def test_the_two_engines_integrate_to_the_same_trajectory(self, log_net):
        """The fix is only worth having if it lands identically on both paths:
        an interpreted run and a compiled run of a model that starts at the
        singularity must not disagree about where it goes."""
        runs = []
        for codegen in (False, True):
            model = bngsim.Model.from_net(log_net)
            result = bngsim.Simulator(model, method="ode", codegen=codegen).run(
                t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12
            )
            runs.append(np.asarray(result.species))
        assert np.all(np.isfinite(runs[0]))
        np.testing.assert_allclose(runs[0], runs[1], rtol=1e-8, atol=1e-10)


# ─── end to end ────────────────────────────────────────────────────────────


class TestTheSolve:
    def test_the_solve_completes_from_an_exactly_zero_species(self, log_net):
        """The issue's reproducer. Before the fix: ``CVODE integration failed at
        the first call``, because ``ln(0)`` is ``-inf`` and ``0·(-inf)`` is NaN.
        """
        model = bngsim.Model.from_net(log_net)
        result = bngsim.Simulator(model, method="ode").run(
            t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12
        )
        species = np.asarray(result.species)
        assert np.all(np.isfinite(species))
        assert species[0, 0] == 0.0  # A really did start at the singularity
        assert species[-1, 0] > 0.0  # ...and the run moved off it

    def test_the_zero_start_agrees_with_the_limit_from_above(self, log_net, tmp_path):
        """The sharper statement of the same thing: an initial condition of
        ``1e-30`` rather than ``0.0`` always worked, so a fix is only credible if
        starting *at* zero now lands on what approaching zero always gave."""
        near = tmp_path / "near.net"
        near.write_text(LOG_RATE_LAW_NET.replace("1 A() 0.0", "1 A() 1e-30"))

        def final(path):
            return np.asarray(
                bngsim.Simulator(bngsim.Model.from_net(path), method="ode")
                .run(t_span=(0, 4), n_points=5, rtol=1e-11, atol=1e-13)
                .species
            )[-1]

        np.testing.assert_allclose(final(log_net), final(near), rtol=1e-6, atol=1e-9)
