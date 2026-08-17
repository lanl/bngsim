"""A forward-sensitivity tensor that comes back non-finite is refused (issue #384).

CVODES can return ``CV_SUCCESS`` with NaN in the sensitivity vectors, and the
reason is structural rather than unlucky: the error test is a comparison, and
every comparison against NaN is false, so the machinery whose job is to reject
the value cannot reject it once it has arrived. The run then reports a clean
solve and hands back a poisoned gradient.

**How this module is split, and why (issue #389).** The guard is deterministic;
every witness that provokes it for real is not. The first version of this module
pinned the whole guard to ``BIOMD0000000480``, whose blow-up is a floating-point
knife edge — 0 non-finite cells at tolerance-floor tau 9e-4, 1517 at 1e-3, 0 at
1.05e-3, 33661 at 1.2e-3 — and on macOS x86_64 that edge is not crossed at *any*
setting in that sweep, so all three tests failed there with ``DID NOT RAISE``.
The remaining witnesses are no better as a foundation: the 14 models of issue
#388 fail structurally rather than by arithmetic luck, but each is a defect
someone intends to remove, so pinning the guard to one couples an infrastructure
test to that defect's lifetime.

So:

* :class:`TestTheRefusal` hands the guard a tensor directly, through
  ``_bngsim_core._refuse_nonfinite_sensitivities``. It is the test that should
  never need re-sourcing — no model, no solve, no arithmetic.
* :class:`TestTheRealModels` keeps real witnesses as supplementary regressions,
  and each **skips rather than fails** when its model comes back finite. That is
  the correct report both on a host where the knife edge is not crossed and on
  the day issue #388 is fixed.

:class:`TestTheKindOfNonFinite` covers issue #394 — the message counts ``inf``
and ``nan`` apart, because a diverging derivative and an arithmetic accident are
different failures wanting different advice. It also pins the measurement that
keeps the census from being read as a verdict: the two models #394 identified as
having no defect at all would reach the guard as **nan**, not as ``inf``. Since
GH #395 they no longer reach it at all — their value is caught in the
sensitivity RHS, at the point of production — so that one test asks for the
pre-#395 path by name. What it measures is the arithmetic, which #395 moved the
interception of rather than changed.
"""

import os
import re
from functools import cache
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import _refuse_nonfinite_sensitivities as refuse

# ─── The mechanism, on a tensor of our own ───────────────────────────────────

TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
COLS = ["k_on", "k_off", "k_cat"]
N_SPECIES = 4


def _clean():
    """A finite (n_times, n_species, n_cols) tensor, shaped like a real one."""
    return np.zeros((len(TIMES), N_SPECIES, len(COLS)))


def _refused(tensor, times=TIMES, cols=COLS, axis="parameter column"):
    """The message the guard refuses `tensor` with. Fails if it accepts it."""
    with pytest.raises(RuntimeError, match="non-finite") as exc:
        refuse(tensor, times, cols, axis)
    return str(exc.value)


class TestTheRefusal:
    """The guard itself: deterministic, and independent of every model."""

    def test_a_finite_tensor_is_handed_back(self):
        """The control. A guard that refuses everything would pass every test
        below it, so the negative case is asserted first."""
        assert refuse(_clean(), TIMES, COLS) is None

    def test_a_nonfinite_tensor_is_refused_rather_than_returned(self):
        """The defect was a silent NaN: a run carrying this used to succeed."""
        t = _clean()
        t[3, 2, 1] = np.nan
        _refused(t)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
    def test_an_infinity_is_refused_as_well_as_a_nan(self, bad):
        """``Result.gradient`` reduces the tensor, so an infinity poisons a
        scalar an optimizer will step on exactly as a NaN does. NaN is the case
        that motivated the guard (#384) only because it is the one CVODES cannot
        reject; the guard is written on finiteness, not on NaN, and that is
        asserted so a future rewrite cannot narrow it to NaN unnoticed."""
        t = _clean()
        t[1, 0, 0] = bad
        _refused(t)

    def test_the_message_localizes_the_failure(self):
        """A caller has to be able to act on it: which column, how much, and
        when. The column is what to drop to get a usable run out of the same
        model; the output point is where to start a bisection."""
        t = _clean()
        t[2, 0, 1] = np.nan
        t[2, 3, 1] = np.nan
        msg = _refused(t)
        assert "'k_off'" in msg
        assert "'k_on'" not in msg  # only the columns actually implicated
        assert "2 nan cell(s)" in msg  # the count AT that point, not the tensor's total
        assert f"t=0.5 (index 2 of {len(TIMES)})" in msg
        assert "parameter column" in msg

    def test_the_first_affected_point_is_the_one_reported(self):
        """A blow-up spreads: by the last output point most of the tensor can be
        non-finite, and reporting that tells a caller nothing. The bisection
        starts at the *first* point that went bad."""
        t = _clean()
        t[3, 1, 0] = np.nan  # later, and wider
        t[3, 2, 0] = np.nan
        t[1, 0, 2] = np.nan  # earlier, and narrower
        msg = _refused(t)
        assert "index 1 of 5" in msg
        assert "1 nan cell(s)" in msg
        assert "'k_cat'" in msg
        assert "'k_on'" not in msg

    def test_every_column_implicated_at_that_point_is_named(self):
        """Columns fail together (BIOMD0000000480 took all 41 rows of one column
        at once; #388's models take several columns at the first output point),
        and naming one of them would send a caller back for another run."""
        t = _clean()
        t[0, 0, 0] = np.nan
        t[0, 1, 2] = np.inf
        msg = _refused(t)
        assert "'k_on'" in msg
        assert "'k_cat'" in msg
        assert "2 cell(s) (1 inf, 1 nan)" in msg

    def test_a_wide_failure_is_summarized_rather_than_dumped(self):
        """A budget-capped parameter list runs to hundreds of columns, and a
        message that names all of them is one no terminal will show."""
        cols = [f"p{i}" for i in range(20)]
        t = np.zeros((len(TIMES), 2, len(cols)))
        t[0, 0, :] = np.nan
        msg = _refused(t, cols=cols)
        assert msg.count("'p") == 6
        assert "… (14 more)" in msg

    def test_the_message_carries_both_remedies(self):
        """The two things a caller can do next, and the counter that explains
        why the solver looked clean. Asserted because a diagnostic that stops
        naming them is a diagnostic that stops being actionable."""
        t = _clean()
        t[0, 0, 0] = np.nan
        msg = _refused(t)
        assert "n_sens_err_test_fails" in msg
        assert "BNGSIM_SENS_ERROR_FLOOR=0" in msg

    def test_the_floor_is_offered_as_one_cause_and_not_as_the_cause(self):
        """Issue #388 measured 14 corpus models that stay non-finite with the
        issue #177 floor switched off. Advertising it as *the* remedy sends
        exactly those callers down a dead end, so the message hedges it."""
        t = _clean()
        t[0, 0, 0] = np.nan
        msg = _refused(t)
        assert "on some models" in msg

    def test_the_initial_condition_axis_is_named_as_itself(self):
        """dY/dY(0) and dY/dp are different tensors with different column
        vocabularies — a caller told 'parameter column S1' would go looking for
        a parameter that does not exist."""
        t = _clean()
        t[0, 0, 0] = np.nan
        msg = _refused(t, cols=["S1", "S2", "S3"], axis="initial-condition column")
        assert "initial-condition column 'S1'" in msg

    def test_an_unnamed_column_still_gets_an_index(self):
        """Names are supplied by the caller and a short list is a bug, not a
        reason to lose the diagnostic."""
        t = _clean()
        t[0, 0, 2] = np.nan
        assert "'2'" in _refused(t, cols=["k_on"])


class TestTheKindOfNonFinite:
    """inf and nan are counted apart, because they mean different things (#394).

    An ``inf`` is a derivative that diverges — the model sitting on a branch
    point of its own, a square root or a fractional power at zero — and there is
    no finite number for bngsim to return. A ``nan`` is an arithmetic accident
    over a derivative that usually exists, and every one traced so far has been
    a bngsim defect (#310, #317, #333, #351, #391).

    The census does NOT settle which of the two a given failure is, and this
    class pins that it does not claim to: see
    :meth:`test_the_nan_reading_is_hedged_because_a_divergence_arrives_as_nan`.
    """

    def test_a_homogeneous_point_is_named_by_its_kind(self):
        """ "3 nan cell(s)" answers in the first clause the question the old
        message needed a trace to answer."""
        t = _clean()
        t[1, 0, 0] = t[1, 1, 0] = t[1, 2, 0] = np.nan
        assert "3 nan cell(s)" in _refused(t)
        t = _clean()
        t[1, 0, 0] = t[1, 1, 0] = np.inf
        assert "2 inf cell(s)" in _refused(t)

    def test_a_mixed_point_is_broken_down(self):
        """A mixed point is the one that needs the split spelled out: half of it
        points at a divergence and half at an accident."""
        t = _clean()
        t[0, 0, 0] = np.inf
        t[0, 1, 0] = -np.inf
        t[0, 2, 1] = np.nan
        msg = _refused(t)
        assert "3 cell(s) (2 inf, 1 nan)" in msg

    def test_a_negative_infinity_counts_as_an_infinity(self):
        """-inf is the same statement about the derivative as +inf, and a census
        that called it a nan would invert the advice."""
        t = _clean()
        t[0, 0, 0] = -np.inf
        assert "1 inf cell(s)" in _refused(t)

    def test_only_the_nan_case_offers_the_tolerance_knobs(self):
        """A tighter atol and ``BNGSIM_SENS_ERROR_FLOOR=0`` move where a blow-up
        happens. Neither moves a derivative that is unbounded, so offering them
        on a pure-inf failure is advice that cannot work."""
        t = _clean()
        t[0, 0, 0] = np.inf
        msg = _refused(t)
        assert "BNGSIM_SENS_ERROR_FLOOR=0" not in msg
        assert "tighter atol" not in msg
        # ...but the counter that explains the clean solve applies to both.
        assert "n_sens_err_test_fails" in msg

    def test_each_reading_appears_only_when_its_kind_is_present(self):
        """Both paragraphs on every failure would be two paragraphs of which one
        is always wrong."""
        t = _clean()
        t[0, 0, 0] = np.inf
        inf_only = _refused(t)
        t = _clean()
        t[0, 0, 0] = np.nan
        nan_only = _refused(t)
        assert "diverges rather than an arithmetic accident" in inf_only
        assert "diverges rather than an arithmetic accident" not in nan_only
        assert "arithmetic accident — 0*inf" in nan_only
        assert "arithmetic accident — 0*inf" not in inf_only

    def test_an_inf_that_arrives_later_is_still_reported(self):
        """The first affected point is where the bisection starts, but "a
        derivative diverges somewhere in this run" changes the reading of the
        whole failure — and a nan-only first point would otherwise bury it."""
        t = _clean()
        t[1, 0, 0] = np.nan
        t[3, 2, 2] = np.inf
        msg = _refused(t)
        assert "1 nan cell(s) at the first affected output point t=0.25 (index 1 of 5)" in msg
        assert "first inf cell is later, at t=0.75 (index 3)" in msg
        assert "'k_cat'" in msg  # the column that diverges, named too

    def test_no_later_line_when_the_first_point_already_has_the_inf(self):
        """Saying "later" about the point just named would be noise."""
        t = _clean()
        t[1, 0, 0] = np.inf
        assert "later" not in _refused(t)

    def test_the_nan_reading_is_hedged_because_a_divergence_arrives_as_nan(self):
        """The measurement that stops the census from being read as a verdict.

        #394 filed this expecting the split to separate the two causes: its two
        examples are models whose derivative is genuinely unbounded, and the
        expectation was that they would show up as ``inf``. Measured against
        this branch, they do not — ``BIOMD0000000632`` (``∂√Gy/∂Gy`` at
        ``Gy = 0``) and ``MODEL2403070001`` (``(t − Tmeal)^-0.6`` at
        ``t = Tmeal``) both reach the scan as **nan**, because the inf is
        annihilated before the output.

        So "nan" cannot be reported as "therefore a bngsim defect". The message
        names the arithmetic that arrived and then hands the reader the check
        that does discriminate — whether the column's parameter sits at a
        singular value of its own. That hedge is the load-bearing part of #394's
        fix, and this pins it.
        """
        t = _clean()
        t[0, 0, 0] = np.nan
        msg = _refused(t)
        assert "genuinely diverges also arrives as nan" in msg
        assert "singular value of its own" in msg


# ─── Real models, kept as supplementary regressions ──────────────────────────

RUN = {"rtol": 1e-9, "atol": 1e-12, "max_steps": 100000}


def _model_path(name):
    p = Path(__file__).resolve().parents[2] / "benchmarks" / "sbml_events" / f"{name}.xml"
    if not p.exists():  # pragma: no cover - the corpus ships with the repo
        pytest.skip(f"{name} not available")
    return str(p)


def _targets(m):
    """The parameter list amici_parity negotiates: budget-capped, no
    function-backed slots and no compartment sizes (both are refused by name,
    which would mask what this module is testing)."""
    cap = max(1, 20000 // max(len(m.species_names), 1))
    skip = (m._internal_param_names() & set(m.function_names)) | set(m.compartment_size_params)
    return [n for n in m.primary_param_names if n not in skip][:cap]


@cache
def _outcome(name, t_span, n_points, floor=None, recover=None):
    """Run `name` and return the refusal message, or None if it came back finite.

    ``floor`` sets ``BNGSIM_SENS_ERROR_FLOOR`` (issue #177) and ``recover`` sets
    ``BNGSIM_SENS_NONFINITE_RECOVER`` (GH #395). Cached: these are the slow part
    of the module, deterministic, and several tests ask about the same run.
    """
    env = {"BNGSIM_SENS_ERROR_FLOOR": floor, "BNGSIM_SENS_NONFINITE_RECOVER": recover}
    env = {k: v for k, v in env.items() if v is not None}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        m = bngsim.Model.from_sbml(_model_path(name))
        sim = bngsim.Simulator(
            m, method="ode", sensitivity_params=_targets(m), sensitivity_method="staggered"
        )
        r = sim.run(t_span=t_span, n_points=n_points, **RUN)
    except bngsim.SimulationError as e:
        if "non-finite" not in str(e):
            raise
        return str(e)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert np.isfinite(np.asarray(r.sensitivities)).all()
    return None


def _caught_in_the_rhs(msg):
    """True when GH #395's recoverable return caught the value in the callback,
    before any tensor existed for this scan to look at."""
    return "sensitivity RHS returned a non-finite value" in msg


def _refusal_or_skip(name, t_span, n_points):
    msg = _outcome(name, t_span, n_points)
    if msg is None:
        pytest.skip(
            f"{name} returns a finite sensitivity tensor on this host — nothing "
            f"for the guard to refuse. Expected either where the arithmetic does "
            f"not reproduce the blow-up (issue #389) or once the model's own "
            f"defect is fixed (issue #388); the guard itself is covered by "
            f"TestTheRefusal."
        )
    return msg


class TestTheRealModels:
    """Witnesses, not the mechanism. Each skips when its model comes back finite.

    Kept because ``TestTheRefusal`` cannot see the one thing that matters most
    about the guard in production: that a completed ``run()`` actually passes
    its tensor through it, on the shared cold/warm exit, before handing the
    result to the caller.

    **GH #395 split these witnesses into two kinds, and which kind a model is
    now says where its non-finite value was caught.** The sensitivity RHS hands
    CVODES the recoverable code when it computes one, so a model on the analytic
    path never gets far enough for the tensor to exist — it fails inside the
    solve, naming the callback and the time. Only a model on CVODES' internal
    difference quotient, where bngsim's callback is not in the loop, still
    reaches this scan. Both refuse; they refuse in different words, and asserting
    the wrong words would pin the defect rather than the contract.
    """

    # Issue #388: fails at the first output point after t_start, with the issue
    # #177 floor on or off. That is a derivative that was never defined rather
    # than one that drifted, which is why it reproduces where #480 does not.
    # Post-GH #395 both are caught in the RHS instead of in this scan.
    STRUCTURAL = ("BIOMD0000000829", (70.0, 160.0), 201, "n_1")
    STRUCTURAL_2 = ("BIOMD0000000632", (0.0, 10.0), 201, "Gy")

    # Issue #384's own witness: `parameter_63` tracks AMICI to six significant
    # figures for 963 of 1001 output points and then all 41 of its rows go NaN
    # together while `n_err_test_fails` reads 0. n_points is load-bearing —
    # CVODES steps to each output point, so 201 or 101 points walk a different
    # trajectory and both come back finite.
    KNIFE_EDGE = ("BIOMD0000000480", (0.0, 10.0), 1001, "parameter_63")

    @pytest.mark.parametrize("case", [STRUCTURAL, STRUCTURAL_2, KNIFE_EDGE], ids=lambda c: c[0])
    def test_the_run_refuses_rather_than_returning_the_tensor(self, case):
        """The wiring: a real solve refuses rather than handing the tensor back.

        The exact crossing time, cell count and column count are deliberately
        NOT asserted. An earlier version of this module pinned macOS's t=9.64
        and failed on both Linux legs while the refusal fired identically. What
        is portable is the SHAPE of the diagnostic, and after GH #395 there are
        two shapes — see the class docstring — so which one is asserted follows
        from where the value was caught, not from the model.
        """
        name, t_span, n_points, column = case
        msg = _refusal_or_skip(name, t_span, n_points)
        if _caught_in_the_rhs(msg):
            # Caught at the point of production: no tensor exists yet, so there
            # is no column or output index to name. What it must localize
            # instead is the callback and the time.
            assert re.search(r"t=[\d.eE+-]+", msg)
            return
        assert f"'{column}'" in msg
        # The census names a kind on every failure — homogeneous or mixed. WHICH
        # kind is not asserted here: it is a property of the model, and this test
        # is about the diagnostic's shape.
        assert re.search(r"\d+ (nan|inf) cell\(s\)|\d+ cell\(s\) \(\d+ inf, \d+ nan\)", msg)
        assert re.search(rf"output point t=[\d.eE+-]+ \(index \d+ of {n_points}\)", msg)
        assert "n_sens_err_test_fails" in msg

    @pytest.mark.parametrize("case", [STRUCTURAL, STRUCTURAL_2], ids=lambda c: c[0])
    def test_a_divergent_derivative_would_reach_the_tensor_as_nan(self, case):
        """The measurement behind the hedged nan reading (issue #394).

        ``BIOMD0000000632`` ships ``Gy = 0``, so ``∂√Gy/∂Gy`` is an honest
        ``+inf``; ``BIOMD0000000829``'s ``n_1`` column is the same shape. #394
        expected the tensor to show an ``inf`` for exactly that reason, and the
        message's advice would follow from it. It does not: what would reach the
        tensor is **nan**, because the inf is annihilated on the way there. That
        is why the nan reading is hedged rather than reported as a verdict.

        Run with GH #395's interception switched off, and deliberately so. #395
        now catches these in the sensitivity RHS callback, at the point of
        production — which is better, and means no tensor exists for the scan to
        census. But the claim under test is about the ARITHMETIC, not about
        where the value is intercepted: an inf that annihilates to nan does so
        whether or not something catches it first, and the scan is still the
        backstop for every column CVODES difference-quotients itself, where
        bngsim's callback is not in the loop. ``BNGSIM_SENS_NONFINITE_RECOVER=0``
        restores the pre-#395 path exactly, so it is the honest way to keep
        observing a quantity that is otherwise no longer reachable.
        """
        name, t_span, n_points, _ = case
        if _outcome(name, t_span, n_points) is None:
            pytest.skip(
                f"{name} returns a finite sensitivity tensor on this host — nothing "
                f"for the guard to refuse (see _refusal_or_skip)."
            )
        msg = _outcome(name, t_span, n_points, recover="0")
        assert msg is not None and not _caught_in_the_rhs(msg)
        assert re.search(r"\d+ nan cell\(s\)", msg)
        assert "inf cell(s)" not in msg
        # The tolerance knobs are nan-only advice, so a nan failure carries them.
        assert "BNGSIM_SENS_ERROR_FLOOR=0" in msg

    def test_the_remedy_the_message_advertises_works_on_the_knife_edge(self):
        """The diagnostic names ``BNGSIM_SENS_ERROR_FLOOR=0``. On #480 that is
        the whole story — the issue #177 floor's relaxation is what admits the
        blow-up — so if it stops being true the message is worse than none.

        It does NOT hold for the structural failures: issue #388 measured 14 of
        14 still non-finite with the floor off, which is why the message hedges
        (see ``test_the_floor_is_offered_as_one_cause_and_not_as_the_cause``).
        """
        name, t_span, n_points, _ = self.KNIFE_EDGE
        _refusal_or_skip(name, t_span, n_points)
        assert _outcome(name, t_span, n_points, floor="0") is None

    def test_the_floor_does_not_clear_a_structural_failure(self):
        """The other half of the same claim, and the measurement that separates
        the two kinds of witness (issue #388). Without it, a reader could take
        the test above as saying the floor explains every case."""
        name, t_span, n_points, column = self.STRUCTURAL
        _refusal_or_skip(name, t_span, n_points)
        msg = _outcome(name, t_span, n_points, floor="0")
        assert msg is not None
        assert _caught_in_the_rhs(msg) or f"'{column}'" in msg


class TestTheSensitivityCounters:
    """``n_err_test_fails`` counts the STATE solve. Reading it as "the solve was
    clean" is how a poisoned column got called healthy, so CVODES' separate
    sensitivity counters are reported separately (issue #384)."""

    T_SPAN = (0.0, 10.0)
    N_POINTS = 1001

    def _stats(self, **env):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            m = bngsim.Model.from_sbml(_model_path("BIOMD0000000480"))
            sim = bngsim.Simulator(
                m, method="ode", sensitivity_params=_targets(m), sensitivity_method="staggered"
            )
            return sim.run(t_span=self.T_SPAN, n_points=self.N_POINTS, **RUN).solver_stats
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_the_state_counter_does_not_see_the_sensitivity_rejections(self):
        """The two are different quantities, and this model is where that shows.

        The relation is asserted rather than either absolute value: how many
        steps a stiff 41x75 coupled solve rejects is platform arithmetic, and a
        sibling test in this file has already been bitten once by pinning a
        number that only held on one OS. What is portable is that the
        sensitivity solve rejects a great deal that the state solve never sees.
        """
        stats = self._stats(BNGSIM_SENS_ERROR_FLOOR="0")
        assert stats["n_sens_err_test_fails"] > 0, (
            "if this ever reads 0, the counter is being read off the state "
            "solve again — which is the whole defect"
        )
        assert stats["n_sens_err_test_fails"] > stats["n_err_test_fails"]

    def test_both_counters_are_present_and_zero_without_sensitivities(self):
        m = bngsim.Model.from_sbml(_model_path("BIOMD0000000480"))
        r = bngsim.Simulator(m, method="ode").run(t_span=self.T_SPAN, n_points=11, **RUN)
        assert r.solver_stats["n_sens_err_test_fails"] == 0
        assert r.solver_stats["n_sens_nonlin_conv_fails"] == 0
