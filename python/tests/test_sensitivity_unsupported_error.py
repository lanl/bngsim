"""``SensitivityUnsupportedError`` — a declared refusal, distinguishable by type.

bngsim has two constructs it *declares* it cannot differentiate and raises on
rather than answer wrongly: an event whose crossing time moves in a way
``dt*/dp`` cannot be computed for (GH #205), and a rate law codegen cannot
differentiate to closed form (GH #214). Both raised a bare ``ValueError``, which
made a documented capability gap indistinguishable from a bug unless the caller
matched on message text — and those messages are long prose that cites GH issue
numbers, so any such match is one rewording away from silently breaking.

What is asserted here, in the order the tests read:

  * **The type contract** — ``BngsimError`` (so ``except BngsimError`` catches
    it) *and* ``ValueError`` (so every pre-existing handler still does).
  * **Both live raise sites** actually produce it, reached through the public
    API rather than by calling the private helper: this is the half a message
    match could not protect, and the half that decides whether a parity sweep
    scores these models as UNSUPPORTED or EXCEPTION.
  * **The boundary** — an environment failure (no compiler, no JIT) is NOT this
    exception. "The machine lacks a backend" is a fixable local problem, not a
    property of the model; typing it as a declared refusal would tell a parity
    sweep to stop counting a broken toolchain.
"""

import bngsim
import pytest
from bngsim._bngsim_core import ModelBuilder
from bngsim._exceptions import BngsimError, SensitivityUnsupportedError


def _decay(with_event=None):
    """``dS/dt = -k·S``, ``S(0) = 100``; optionally one event."""
    b = ModelBuilder()
    b.add_parameter("k", 0.1)
    s = b.add_species("S", 100.0)
    b.add_reaction([s], [], "elementary", "k")
    if with_event is not None:
        trigger, assign, kwargs = with_event
        b.add_event("evt", trigger, [(s, assign)], **kwargs)
    return bngsim.Model(_core=b.build())


# --------------------------------------------------------------------------- #
# Type contract
# --------------------------------------------------------------------------- #
class TestTypeContract:
    def test_is_a_bngsim_error(self):
        assert issubclass(SensitivityUnsupportedError, BngsimError)
        assert issubclass(SensitivityUnsupportedError, RuntimeError)

    def test_is_also_a_value_error(self):
        """The back-compat half. Both sites raised a plain ``ValueError`` before
        this class existed, so any caller — including this repo's own event and
        codegen tests — that writes ``except ValueError`` must keep catching it.
        Dropping ``ValueError`` from the bases would be a silent API break that
        only shows up as an escaped exception in someone else's fitting loop."""
        assert issubclass(SensitivityUnsupportedError, ValueError)
        with pytest.raises(ValueError):
            raise SensitivityUnsupportedError("x")

    def test_is_exported(self):
        assert bngsim.SensitivityUnsupportedError is SensitivityUnsupportedError
        assert "SensitivityUnsupportedError" in bngsim.__all__

    def test_is_not_confusable_with_the_ssa_refusal(self):
        """Its peer, not its subclass: a caller that falls back to a
        derivative-free optimizer on one must not swallow the other."""
        assert not issubclass(SensitivityUnsupportedError, bngsim.SsaValidationError)
        assert not issubclass(bngsim.SsaValidationError, SensitivityUnsupportedError)


# --------------------------------------------------------------------------- #
# Raise site 1 — a non-differentiable event crossing time (GH #205 / issue #144)
# --------------------------------------------------------------------------- #
class TestEventRefusalIsTyped:
    """The `BIOMD0000000342` shape: an event bngsim will not differentiate.

    Reached through ``Simulator.run()``, not through the private support check,
    because the contract under test is what a *caller* sees.
    """

    def test_a_delayed_event_raises_the_typed_refusal(self):
        m = _decay(with_event=("time() >= 5", "2.0", {"delay": 1.0}))
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(SensitivityUnsupportedError, match="delay"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_a_trigger_with_no_single_crossing_surface_raises_the_typed_refusal(self):
        m = _decay(with_event=("S < 50 && time() >= 1", "2.0", {}))
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(SensitivityUnsupportedError, match="combines conditions"):
            sim.run(t_span=(0, 10), n_points=11)

    def test_the_refusal_still_names_the_regime_and_the_issue(self):
        """Typing the exception is what parity suites match on; the prose is what
        a human reads. Keep both — a typed exception with a gutted message would
        pass every by-type assertion and tell the user nothing."""
        m = _decay(with_event=("time() >= 5", "2.0", {"delay": 1.0}))
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k"])
        with pytest.raises(SensitivityUnsupportedError) as ei:
            sim.run(t_span=(0, 10), n_points=11)
        msg = str(ei.value)
        assert "Output sensitivities are not supported" in msg
        assert "#205" in msg

    def test_a_supported_event_model_is_untouched(self):
        """The guard against over-typing: a fixed-time event is differentiable
        (GH #212) and must still run, not raise a 'declared refusal'."""
        m = _decay(with_event=("time() >= 5", "2.0", {}))
        r = bngsim.Simulator(m, method="ode", sensitivity_params=["k"]).run(
            t_span=(0, 10), n_points=11
        )
        assert r.sensitivities is not None


# --------------------------------------------------------------------------- #
# Raise site 2 — codegen cannot differentiate the rate laws (GH #214)
# --------------------------------------------------------------------------- #
class TestCodegenDeclineIsTyped:
    """Codegen inspected the model and DECLINED — nothing failed, there was just
    nothing differentiable to compile.

    A ``None`` return from the model-path entry point is the signal, and
    ``last_codegen_error() is None`` is what separates it from the failure case
    below: both return the same sentinel.
    """

    @staticmethod
    def _sensitivity_run(monkeypatch, *, decline: bool, jit: bool):
        import bngsim._codegen as cg

        if decline:
            # Model the DECLINE precisely: the sentinel, and no recorded cause.
            def _decline(*_a, **_k):
                cg._record_codegen_error(None)
                return None

            monkeypatch.setattr(cg, "prepare_model_codegen_source", _decline)
            monkeypatch.setattr(cg, "prepare_model_codegen", _decline)
        if jit:
            monkeypatch.setenv("BNGSIM_CODEGEN_JIT", "mir")
        else:
            monkeypatch.delenv("BNGSIM_CODEGEN_JIT", raising=False)
        return bngsim.Simulator(_decay(), method="ode", sensitivity_params=["k"])

    @pytest.mark.parametrize("jit", [False, True], ids=["compiled", "mir-jit"])
    def test_a_declined_codegen_raises_the_typed_refusal(self, monkeypatch, jit):
        """Both backends carry their own raise. They are separate statements over
        separate locals (``auto_so`` vs ``auto_src``), so typing one and missing
        the other would make the verdict depend on whether a C compiler happened
        to be present on the sweep machine."""
        with pytest.raises(SensitivityUnsupportedError, match="closed form"):
            self._sensitivity_run(monkeypatch, decline=True, jit=jit)

    def test_it_is_catchable_as_a_value_error(self, monkeypatch):
        with pytest.raises(ValueError):
            self._sensitivity_run(monkeypatch, decline=True, jit=False)

    def test_a_differentiable_model_is_untouched(self, monkeypatch):
        """Over-typing guard, as above: ordinary mass action still builds."""
        sim = self._sensitivity_run(monkeypatch, decline=False, jit=False)
        r = sim.run(t_span=(0, 10), n_points=11)
        assert r.sensitivities is not None


# --------------------------------------------------------------------------- #
# The boundary: an environment failure is NOT a declared refusal
# --------------------------------------------------------------------------- #
class TestSwallowedBuildFailureIsNotARefusal:
    """The `BIOMD0000000608` shape, and the reason this class needed a boundary.

    The model-path ``prepare_*`` entry points catch every exception and return
    ``None`` — the SAME sentinel a decline returns. So a compile timeout on a
    huge translation unit arrived at the refusal site indistinguishable from
    "your rate laws are not differentiable", and got reported as exactly that.

    `BIOMD0000000608` is real: 66.6 MB of C generated fine (so the rate laws
    *were* differentiated), then blew a 600 s compile budget. Before
    ``last_codegen_error()`` the refusal asserted a cause it had never checked;
    typing that refusal as UNSUPPORTED would have gone further and made the
    resource failure **non-scoring** in a parity sweep, which is the opposite of
    what a bucket for declared gaps is for.
    """

    @staticmethod
    def _timeout_at_compile(monkeypatch, *, jit: bool):
        """Fail inside the real ``prepare_model_codegen*``, so the swallow-and-
        return-None path runs for real rather than being stubbed around."""
        import bngsim._codegen as cg

        def _timed_out(*_a, **_k):
            raise RuntimeError("Codegen compilation timed out after 600 s (66.6 MB C source)")

        if jit:
            monkeypatch.setenv("BNGSIM_CODEGEN_JIT", "mir")
            monkeypatch.setattr(cg, "generate_combined_from_model", _timed_out)
        else:
            monkeypatch.delenv("BNGSIM_CODEGEN_JIT", raising=False)
            # The on-disk .so cache short-circuits before compile_rhs, and this
            # decay model is small enough that a previous test already populated
            # it — miss it deliberately so the compile step actually runs.
            monkeypatch.setattr(cg, "get_cached_so", lambda *_a, **_k: None)
            monkeypatch.setattr(cg, "compile_rhs", _timed_out)
        return bngsim.Simulator(_decay(), method="ode", sensitivity_params=["k"])

    @pytest.mark.parametrize("jit", [False, True], ids=["compiled", "mir-jit"])
    def test_a_swallowed_build_failure_is_not_a_declared_refusal(self, monkeypatch, jit):
        with pytest.raises(RuntimeError) as ei:
            self._timeout_at_compile(monkeypatch, jit=jit)
        assert not isinstance(ei.value, SensitivityUnsupportedError)

    def test_the_message_names_the_real_cause_not_differentiability(self, monkeypatch):
        """The half a user reads. Telling someone to rewrite a smooth rate law
        when the fix is `BNGSIM_CODEGEN_TIMEOUT` or more cores costs them the
        afternoon."""
        with pytest.raises(RuntimeError) as ei:
            self._timeout_at_compile(monkeypatch, jit=False)
        msg = str(ei.value)
        assert "timed out" in msg
        assert "BUILD failure" in msg
        assert "closed form" not in msg

    def test_the_recorded_cause_is_chained_not_just_quoted(self, monkeypatch):
        """``raise ... from cause`` keeps the original traceback reachable; the
        swallow threw it away, which is why this went undiagnosed."""
        with pytest.raises(RuntimeError) as ei:
            self._timeout_at_compile(monkeypatch, jit=False)
        assert isinstance(ei.value.__cause__, RuntimeError)
        assert "timed out" in str(ei.value.__cause__)

    def test_a_stale_cause_cannot_leak_into_the_next_build(self, monkeypatch):
        """``last_codegen_error`` is a thread-local. If a failed build left it set,
        the NEXT model's clean decline would be misreported as that failure — so
        every model-path entry point clears it on the way in."""
        import bngsim._codegen as cg

        with pytest.raises(RuntimeError):
            self._timeout_at_compile(monkeypatch, jit=False)
        assert cg.last_codegen_error() is not None
        monkeypatch.undo()
        bngsim.Simulator(_decay(), method="ode", sensitivity_params=["k"])
        assert cg.last_codegen_error() is None


# --------------------------------------------------------------------------- #
# The boundary: an environment failure is NOT a declared refusal
# --------------------------------------------------------------------------- #
class TestBackendFailureIsNotARefusal:
    def test_a_codegen_backend_failure_stays_a_plain_runtime_error(self, monkeypatch):
        """ "No C compiler and no JIT" is a broken machine, not an undifferentiable
        model. Typing it as a declared refusal would move a whole sweep's worth of
        toolchain failures into UNSUPPORTED — non-scoring — and a parity report
        would show a green wall for a build that never ran.
        """
        import bngsim._codegen as cg

        def _boom(*_a, **_k):
            raise OSError("no C compiler")

        monkeypatch.delenv("BNGSIM_CODEGEN_JIT", raising=False)
        monkeypatch.setattr(cg, "prepare_model_codegen", _boom)
        with pytest.raises(RuntimeError) as ei:
            bngsim.Simulator(_decay(), method="ode", sensitivity_params=["k"])
        assert not isinstance(ei.value, SensitivityUnsupportedError)
        assert "Failed to build" in str(ei.value)
