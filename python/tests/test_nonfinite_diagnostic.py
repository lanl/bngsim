"""GH #336 — a rate law evaluated outside its domain, and one that only looked
like it was.

Two halves, and the second is only worth having because of what the first turned
out to be.

**The compiled path had no non-finite instrumentation.** The interpreted ExprTk
evaluator has carried some since issue #42: every registered adapter checks its
own return value and prints ``'ln(-1e-09)' returned nan`` once per distinct
argument. The codegen path carries none — the emitted C calls libm's ``log``
directly — and that is the path a forward-sensitivity run *forces*, so the engine
where a domain error was least diagnosable was the one it was most likely to
happen on. What the user got was ``CVODE integration failed at t=... with
flag=-4``, naming no time, no species and no rate law.

Instrumenting the emitted C would put a finiteness test on the hot path of every
rate law of every model. The callbacks already scan their output for
non-finiteness (the GH #135 nonnegative-clamp retries), so instead the first such
scan that trips with no clamp left to try keeps ``(t, y)`` as a *witness*.
Nothing is printed then — a step CVODE recovers from must stay silent. Only if
the integration fails is that state replayed through the interpreted evaluator,
which does carry the instrumentation, and what it finds appended to the
exception.

**And the issue's own reproducer was not a domain error at all.** It was GH
#151's native SymPy-free differentiator re-introducing the NaN that #310 / #317
exist to remove. That path recognizes ``^`` and ``ln`` and emits closed-form
derivatives directly, reaching neither ``sympy_to_exprtk`` nor ``sympy_to_c`` —
the only two places the zero-base guard is applied. So ``vmax·S^n·ln(S)``
differentiated to ``(n·S^(n-1))·ln(S) + S^n·(1/S)``, both halves ``0·∞`` at
``S = 0``, and that expression *is* the Jacobian entry — which is why the
sensitivity run failed on its first call, at the exact zero, for every parameter
including ones the logarithm does not touch. A state-dependent logarithm now
defers to SymPy, and the guard applies.
"""

from __future__ import annotations

import ctypes
import hashlib

import bngsim
import numpy as np
import pytest

pytest.importorskip("sympy")

from bngsim import _codegen as cg  # noqa: E402

# The issue's reproducer verbatim. A is produced and never consumed, so it leaves
# zero immediately and rises monotonically — the logarithm's argument never
# reaches the negative half-line where its NaN would be real. `kdeg` appears only
# in `basal()`, which has no logarithm, and is here to pin that the failure was
# not the exponent column.
LOG_SENS_NET = """\
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

# A rate law that genuinely leaves its domain, and cannot be talked out of it.
# A rises linearly past 1, so `1 - Atot` goes negative and `ln` of it is a real
# NaN — not a removable singularity, and not something the nonnegative clamp can
# reach, because A itself never goes below zero. This is the model the diagnostic
# is *for*.
OUT_OF_DOMAIN_NET = """\
begin parameters
    1 k    1.0  # Constant
end parameters
begin functions
    1 logterm() ln(1 - Atot)
end functions
begin species
    1 A() 0.0
    2 C() 0.0
end species
begin reactions
    1 0 1 k        #_R1
    2 0 2 logterm  #_R2
end reactions
begin groups
    1 Atot 1
    2 Ctot 2
end groups
"""


# GH #353 — a non-finite STATE, not a rate law out of its domain. A rate-rule
# species whose initial condition is already non-finite makes every law that
# reads it non-finite before a step is taken; the diagnostic's below-zero scan is
# blind to it (``nan < 0.0`` is false), so the corrupt state went unnamed and the
# innocent rate law took the blame. ``{ic}``/``{kval}`` pick the flavour: ``inf``
# survives the nonnegative clamp directly; ``nan`` is clamped to 0, so a ``nan``
# rate constant keeps the clamped RHS non-finite (``nan*0 = nan``) and forces the
# witness to capture the unclamped ``nan`` state — the shape of the #353 model,
# whose NaN compartment size had made an IA-derived parameter ``nan`` too.
NONFINITE_STATE_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="nonfinite_state">
    <listOfCompartments>
      <compartment id="C" size="1" spatialDimensions="3" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="{ic}"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="{kval}" constant="true"/></listOfParameters>
    <listOfRules>
      <rateRule variable="A"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>A</ci></apply></math></rateRule>
    </listOfRules>
  </model>
</sbml>"""


@pytest.fixture
def log_sens_net(tmp_path):
    net = tmp_path / "logsens.net"
    net.write_text(LOG_SENS_NET)
    return net


@pytest.fixture
def out_of_domain_net(tmp_path):
    net = tmp_path / "outofdomain.net"
    net.write_text(OUT_OF_DOMAIN_NET)
    return net


def _run(path, **kwargs):
    model = bngsim.Model.from_net(path)
    sim = bngsim.Simulator(model, method="ode", **kwargs)
    return sim.run(t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12)


# ─── the guard the native differentiator was skipping ──────────────────────


class TestTheReproducer:
    """The issue as filed: a forward-sensitivity run from a species sitting
    exactly at the logarithm's singularity."""

    @pytest.mark.parametrize("param", ["n", "vmax", "kdeg"])
    def test_every_column_integrates_from_the_exact_zero(self, log_sens_net, param):
        """All three failed with ``flag=-4`` at ``t=0``, and ``kdeg`` is the one
        that named the cause: it appears only in ``basal()``, which contains no
        logarithm, so the NaN could not have been in its ``∂f/∂p`` column. It was
        in ``J``, shared by every column."""
        result = _run(log_sens_net, sensitivity_params=[param])
        sens = np.asarray(result.sensitivities)
        assert np.all(np.isfinite(sens))

    def test_the_zero_start_lands_where_approaching_zero_lands(self, log_sens_net, tmp_path):
        """One floating-point value away the run always worked, so a fix is only
        credible if starting *at* the singularity now gives the same answer as
        starting beside it."""
        near = tmp_path / "logsens_eps.net"
        near.write_text(LOG_SENS_NET.replace("1 A() 0.0", "1 A() 1e-30"))

        at_zero = np.asarray(_run(log_sens_net, sensitivity_params=["n"]).sensitivities)[-1]
        beside_it = np.asarray(_run(near, sensitivity_params=["n"]).sensitivities)[-1]
        np.testing.assert_allclose(at_zero, beside_it, rtol=1e-6, atol=1e-9)


class TestTheEmittedDerivative:
    """Solver-free: the compiled sensitivity RHS called directly at the
    singularity. A solve reaches this through step selection, which makes it a
    poor place to pin the contract — the emitted arithmetic is the contract."""

    @staticmethod
    def _compiled(net, tmp_path, monkeypatch):
        model = bngsim.Model.from_net(net)
        bngsim.Simulator(model, method="ode", sensitivity_params=["n"])
        src, has_sens = cg.generate_combined_from_model(model)
        assert has_sens, "model did not get an analytic sensitivity RHS"
        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
        so = cg.compile_rhs(src, hashlib.sha256(src.encode()).hexdigest()[:16])
        return model, src, ctypes.CDLL(str(so))

    def test_the_jacobian_entry_carries_the_guard(self, log_sens_net, tmp_path, monkeypatch):
        """``∂(vmax·S^n·ln S)/∂S`` reaches the emitted C as the guarded
        ``S^(n-1)`` form. The unguarded native spelling is recognisable on sight
        by the ``1.0/`` it divides the base out with — the product rule's
        ``d(ln S) = 1/S`` — which is ``inf`` where the ``S^n`` beside it is 0."""
        _model, src, _lib = self._compiled(log_sens_net, tmp_path, monkeypatch)
        # The same entry is emitted twice — once for the standalone Jacobian and
        # once for the `J·yS` half of the sensitivity RHS — and both come from
        # the one shared builder. Assert over every copy: a guard on one of them
        # only would leave the two disagreeing at the same point.
        entries = [ln for ln in src.splitlines() if "double dj =" in ln and "log(" in ln]
        assert entries, "no log-bearing Jacobian entry emitted"
        for entry in entries:
            assert "1.0/" not in entry
            assert "?" in entry and "== 0.0" in entry

    def test_the_sensitivity_rhs_is_finite_at_the_singularity(
        self, log_sens_net, tmp_path, monkeypatch
    ):
        """Both halves of ``ySdot = J·yS + ∂f/∂p`` at ``Atot = 0``: ``yS = 0``
        isolates ``∂f/∂p`` (#317's ``ln(S)^2`` shape), ``yS = 1`` adds the ``J·yS``
        term that made every other column NaN too."""
        model, _src, lib = self._compiled(log_sens_net, tmp_path, monkeypatch)
        data = model._core.codegen_data()
        n_sp = len(data["species"])
        params = [q["value"] for q in data["parameters"]]

        dp = ctypes.POINTER(ctypes.c_double)

        class _SensUserData(ctypes.Structure):
            _fields_ = [
                ("param_values", dp),
                ("plist", ctypes.POINTER(ctypes.c_int)),
                ("n_sens", ctypes.c_int),
            ]

        lib.bngsim_codegen_sens_rhs.restype = ctypes.c_int
        lib.bngsim_codegen_sens_rhs.argtypes = [
            ctypes.c_int,
            ctypes.c_double,
            dp,
            dp,
            ctypes.c_int,
            dp,
            dp,
            ctypes.c_void_p,
            dp,
            dp,
        ]
        pbuf = (ctypes.c_double * len(params))(*params)
        plist = (ctypes.c_int * 1)(0)  # column 0 = the exponent `n`
        ud = _SensUserData(param_values=ctypes.cast(pbuf, dp), plist=plist, n_sens=1)

        y = (ctypes.c_double * n_sp)(0.0, 4.0, 1.0)  # A exactly at the singularity
        for seed in (0.0, 1.0):
            ydot = (ctypes.c_double * n_sp)()
            yS = (ctypes.c_double * n_sp)(*([seed] * n_sp))
            ySdot = (ctypes.c_double * n_sp)()
            tmp1 = (ctypes.c_double * n_sp)()
            tmp2 = (ctypes.c_double * n_sp)()
            rc = lib.bngsim_codegen_sens_rhs(
                1, 0.0, y, ydot, 0, yS, ySdot, ctypes.byref(ud), tmp1, tmp2
            )
            assert rc == 0
            assert np.all(np.isfinite(np.asarray(ySdot))), f"yS={seed}: {list(ySdot)}"

    def test_a_negative_argument_keeps_its_nan(self, log_sens_net, tmp_path, monkeypatch):
        """The guard removes a *removable* singularity and nothing else. Below
        zero the logarithm is genuinely undefined and #310's contract is that the
        NaN is the honest answer — which is what the diagnostic below exists to
        explain rather than to paper over."""
        model, _src, lib = self._compiled(log_sens_net, tmp_path, monkeypatch)
        data = model._core.codegen_data()
        n_sp = len(data["species"])
        params = [q["value"] for q in data["parameters"]]

        dp = ctypes.POINTER(ctypes.c_double)

        class _UserData(ctypes.Structure):
            _fields_ = [
                ("param_values", dp),
                ("tfun_ctx", ctypes.c_void_p),
                ("tfun_eval", ctypes.c_void_p),
            ]

        lib.bngsim_codegen_rhs.restype = ctypes.c_int
        lib.bngsim_codegen_rhs.argtypes = [ctypes.c_double, dp, dp, ctypes.c_void_p]
        pbuf = (ctypes.c_double * len(params))(*params)
        ud = _UserData(param_values=ctypes.cast(pbuf, dp), tfun_ctx=None, tfun_eval=None)

        ydot = (ctypes.c_double * n_sp)()
        y = (ctypes.c_double * n_sp)(-1e-12, 4.0, 1.0)
        assert lib.bngsim_codegen_rhs(0.0, y, ydot, ctypes.byref(ud)) == 0
        assert not np.all(np.isfinite(np.asarray(ydot)))


# ─── the diagnostic ────────────────────────────────────────────────────────


class TestTheFailureMessage:
    """What a genuinely out-of-domain model reports. Both engines, because the
    compiled one is the whole point and the interpreted one is the control."""

    @pytest.mark.parametrize("codegen", [True, False], ids=["compiled", "interpreted"])
    def test_the_failure_names_the_rate_law(self, out_of_domain_net, codegen):
        with pytest.raises(bngsim.SimulationError) as excinfo:
            _run(out_of_domain_net, codegen=codegen)
        message = str(excinfo.value)
        assert "logterm" in message
        assert "ln(1 - Atot)" in message
        # ...and says which side of the run it came from, so the reader knows
        # whether to look at the right-hand side or the sensitivity columns.
        assert "returned a non-finite value at t=" in message

    def test_no_numeric_flag_is_reported_bare(self, out_of_domain_net):
        """``flag=-4`` is not a thing anyone remembers, and looking it up means
        finding the right SUNDIALS header. Every numeric code carries the name
        SUNDIALS itself gives it. (Stated as "every occurrence" rather than
        "this run says CV_CONV_FAILURE" because which failure a collapsing step
        lands on is a step-sequence detail, and the contract is not.)"""
        import re

        with pytest.raises(bngsim.SimulationError) as excinfo:
            _run(out_of_domain_net, codegen=False)
        message = str(excinfo.value)
        for match in re.finditer(r"flag=(-?\d+)", message):
            assert message[match.end() :].startswith(" (CV_"), message

    def test_a_healthy_run_says_nothing(self, log_sens_net, capfd):
        """The witness is captured on the exceptional path and reported only on
        failure, so a run that completes — including one whose clamp retry
        rescued a transiently non-finite step — must not gain a single line of
        output."""
        capfd.readouterr()  # drop anything the load itself printed
        result = _run(log_sens_net, sensitivity_params=["n"])
        assert np.all(np.isfinite(np.asarray(result.sensitivities)))
        captured = capfd.readouterr()
        assert "non-finite" not in captured.err
        assert "non-finite" not in captured.out

    def test_the_compiled_path_borrows_the_interpreted_instrumentation(
        self, out_of_domain_net, capfd
    ):
        """The mechanism, asserted rather than described: replaying the witness
        state through the interpreted evaluator is what makes issue #42's
        ``bngsim: warning: 'ln(...)'`` appear for a run that never touched the
        interpreted right-hand side.

        ``jacobian="analytical"`` so the run is a single integration attempt —
        under the default ``"auto"`` a GH #176 finite-difference retry follows
        the failure and would leave its own warnings on stderr, which would make
        the count prove nothing about where this one came from. Exactly one, and
        the replay evaluates each function exactly once.

        The *value* is deliberately not asserted. Which state the witness holds
        is a step-sequence detail and it differs by platform: where `Atot` has
        gone past 1 the argument is negative and `ln` answers ``nan``, and where
        a step lands on `Atot == 1` exactly it answers ``-inf``. Both are
        non-finite, both name the same rate law, and pinning one of them pins the
        step sequence rather than the diagnostic."""
        capfd.readouterr()
        with pytest.raises(bngsim.SimulationError):
            _run(out_of_domain_net, codegen=True, jacobian="analytical")
        assert capfd.readouterr().err.count("bngsim: warning: 'ln(") == 1


class TestNonFiniteStateNamesTheSpecies:
    """GH #353 — when the *state* is non-finite, name the species, not just the
    rate law. The below-zero scan cannot see a ``nan`` (``nan < 0.0`` is false),
    so a corrupt initial condition left the user staring at a law that only
    answered ``nan`` because its inputs already had."""

    @pytest.mark.parametrize(
        "ic, kval, shown",
        [("INF", "0.1", "inf"), ("NaN", "NaN", "nan")],
        ids=["inf", "nan"],
    )
    def test_the_failure_names_the_non_finite_species(self, ic, kval, shown):
        src = NONFINITE_STATE_SBML.format(ic=ic, kval=kval)
        model = bngsim.Model.from_sbml_string(src)
        with pytest.raises(bngsim.SimulationError) as excinfo:
            bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=3)
        message = str(excinfo.value)
        assert "Non-finite species there: A = " + shown in message

    def test_it_points_at_the_state_not_the_law_domain(self):
        """The closing sentence must not send the reader to constrain a species or
        rewrite a law — the state was already non-finite before the law ran. The
        below-zero-only domain advice is what issue #353 called misdirection."""
        src = NONFINITE_STATE_SBML.format(ic="INF", kval="0.1")
        model = bngsim.Model.from_sbml_string(src)
        with pytest.raises(bngsim.SimulationError) as excinfo:
            bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=3)
        message = str(excinfo.value)
        assert "symptom, not the cause" in message
        assert "check the initial condition" in message
        # The finite-but-out-of-domain advice ("a logarithm, a sqrt ...") belongs
        # to a different failure and must not fire here.
        assert "outside its" not in message
