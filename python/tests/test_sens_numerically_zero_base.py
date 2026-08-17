"""GH #392 — the zero-base exponent guard tests ``base == 0``, and the solver
hands it a base a few ulps the wrong side of zero.

Differentiating a Hill/power law w.r.t. its **exponent** produces
``d/dn base^n = base^n·ln(base)``. At ``base == 0`` that is ``0·(-inf)`` = NaN
in floating point even though the limit exists and is ``0`` for every
``exp > 0``, so ``_guard_exponent_log_at_zero`` (GH #310/#317) rewrites the
sub-product to a ``Piecewise`` whose condition is ``Eq(base, 0)``.

The solver does not hand it a base of exactly zero.

``BIOMD0000000833``'s ``S35`` is ``0.0`` at ``t = 0`` and non-negative at every
one of 2001 output points, and CVODES' predictor puts it at ``-3.75e-36``
between two of them — twenty-four orders below the run's own ``atol``.
``pow(x, 4.0)`` is finite there, so the *value* path never notices;
``log(x)`` is not, and ``x == 0.0`` is false, so the branch never fires and the
whole exponent column is NaN.

**The fix is a retry, not a wider condition.** Telling "numerically zero" from
"negative" needs a scale, and the emitter has none at build time — carrying one
into the emitted text would make the compiled artifact tolerance-dependent, and
widening the condition to ``base <= 0`` would answer a confident ``0`` for a
base that is *genuinely* negative, where ``∂/∂n base^n`` does not exist at all.
So the sensitivity RHS retries at a state whose sub-``atol`` negative components
are snapped to zero (``clamp_state_numerically_zero``) — the GH #135 conditional
clamp the value RHS and the analytical Jacobian already have, applied where the
run's own tolerance is known. The emitted arithmetic is untouched, and a
component negative by *more* than ``atol`` is left alone so its NaN still reaches
the GH #384/#386 refusal.

The fixture below is that distinction with nothing else in it. ``A`` is produced
and consumed by nothing, so its declared initial condition **is** its value at
every call — the singularity sits at a number in the file rather than at a state
the predictor may or may not visit, which is what makes each row deterministic
on every host and in every column position.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# `hill()` is `vmax*Atot^n/(Kd + Atot^n)` — the shape whose ∂/∂n carries
# `Atot^n·ln(Atot)`. `clear()` is log-free and its column is here to show the
# rest of the tensor is unaffected. B is the only species that moves.
HILL_NET = """\
begin parameters
    1 n      4.0  # Constant
    2 Kd     1.0  # Constant
    3 vmax   1.0  # Constant
    4 kdeg   0.3  # Constant
end parameters
begin functions
    1 hill()  vmax*Atot^n/(Kd + Atot^n)
    2 clear() kdeg
end functions
begin species
    1 A() {a0}
    2 B() 1.0
end species
begin reactions
    1 0 2 hill  #_R1
    2 2 0 clear #_R2
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""

ATOL = 1e-12
RUN = {"t_span": (0, 5), "n_points": 6, "rtol": 1e-9, "atol": ATOL}

# Twenty-four orders below `atol`, the way BIOMD0000000833's predictor overshoot
# is; and a base that is negative because the model says so, the way
# BIOMD0000000374 carries `V_membrane = -61`.
NUMERICALLY_ZERO = "-1e-30"
GENUINELY_NEGATIVE = "-61.0"

# Every position of the exponent column, because before GH #395 that decided the
# outcome and this file must not be able to pass by landing in a lucky one.
POSITIONS = [
    pytest.param(["n", "vmax", "kdeg"], id="first"),
    pytest.param(["vmax", "n", "kdeg"], id="middle"),
    pytest.param(["vmax", "kdeg", "n"], id="last"),
]


def _net(tmp_path, a0):
    path = tmp_path / f"hill_{a0}.net"
    path.write_text(HILL_NET.format(a0=a0))
    return path


def _sens(net, params):
    """``("raised", message)`` or ``("returned", tensor)``.

    A fresh ``Model`` per call: a simulator advances the model's species state,
    and a second run over a carried-over state is refused for its own (GH #81)
    reasons, which would mask what this file is testing.
    """
    model = bngsim.Model.from_net(net)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
    try:
        return "returned", np.asarray(sim.run(**RUN).sensitivities)
    except Exception as exc:  # noqa: BLE001 - the outcome class is what is under test
        return "raised", str(exc)


class TestTheFixtureIsTestingWhatItClaims:
    """Three preconditions. Without them the rows below could pass for reasons
    that have nothing to do with GH #392."""

    def test_the_value_path_is_clean_at_the_negative_base(self, tmp_path):
        """The defect is a sensitivity that is non-finite while the state is
        not. ``pow(-1e-30, 4.0)`` is finite, so the trajectory never notices."""
        model = bngsim.Model.from_net(_net(tmp_path, NUMERICALLY_ZERO))
        species = np.asarray(bngsim.Simulator(model, method="ode").run(**RUN).species)
        assert np.all(np.isfinite(species))

    def test_the_base_holds_its_declared_value_for_the_whole_run(self, tmp_path):
        """No reaction touches ``A``, so every callback sees ``-1e-30`` — the
        singularity is at a declared number, not at a predictor excursion."""
        model = bngsim.Model.from_net(_net(tmp_path, NUMERICALLY_ZERO))
        species = np.asarray(bngsim.Simulator(model, method="ode").run(**RUN).species)
        assert np.all(species[:, 0] == float(NUMERICALLY_ZERO))

    def test_the_run_uses_the_analytic_sensitivity_rhs(self, tmp_path):
        """CVODES' difference quotient never evaluates the emitted derivative —
        it perturbs parameters and calls the *value* RHS, which is finite here —
        so it cannot reach this defect, and a run that fell back to it would
        make every row below vacuous."""
        import ctypes

        model = bngsim.Model.from_net(_net(tmp_path, NUMERICALLY_ZERO))
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["n"])
        so = sim._codegen_so_path or getattr(model, "_codegen_so_path", "")
        assert so, "no compiled artifact: the sensitivity run would use CVODES' DQ"
        assert hasattr(ctypes.CDLL(str(so)), "bngsim_codegen_sens_rhs")


class TestABaseThatIsNumericallyZero:
    """``-1e-30`` against ``atol = 1e-12``: eighteen orders inside the accuracy
    the run asked for. The answer is the one-sided limit."""

    @pytest.mark.parametrize("params", POSITIONS)
    def test_the_exponent_column_is_returned(self, tmp_path, params):
        kind, payload = _sens(_net(tmp_path, NUMERICALLY_ZERO), params)
        assert kind == "returned", payload
        assert np.all(np.isfinite(payload))

    @pytest.mark.parametrize("params", POSITIONS)
    def test_it_is_the_limit_and_not_merely_finite(self, tmp_path, params):
        """``lim_{A→0+} ∂/∂n [vmax·A^n/(Kd + A^n)]`` is ``0`` for ``n > 0``, and
        that is what the guard's own ``Eq(base, 0)`` branch returns. Pinned
        against the run at ``A(0) = 0.0`` *exactly* — the state the emitter was
        written for — so this asserts the two agree rather than restating a
        constant."""
        _, near = _sens(_net(tmp_path, NUMERICALLY_ZERO), params)
        _, exact = _sens(_net(tmp_path, "0.0"), params)
        col = params.index("n")
        assert np.array_equal(near[:, :, col], exact[:, :, col])
        # ...and name the limit, so a reader does not have to run the second
        # model to find out what the two agreed on.
        assert not np.any(near[:, :, col])

    @pytest.mark.parametrize("params", POSITIONS)
    def test_the_other_columns_are_untouched(self, tmp_path, params):
        """The retry is conditional: it is reached only from a non-finite
        ``ySdot``, so a column whose RHS was already finite must come back
        bit-identical to the same column of a run whose base is the same
        distance from zero on the *positive* side. ``n = 4``, so
        ``pow(±1e-30, 4)`` is one number and those columns are the same
        arithmetic — any difference would be the retry reaching a column it has
        no business in."""
        _, minus = _sens(_net(tmp_path, NUMERICALLY_ZERO), params)
        _, plus = _sens(_net(tmp_path, "1e-30"), params)
        for name in ("vmax", "kdeg"):
            col = params.index(name)
            assert np.array_equal(minus[:, :, col], plus[:, :, col]), name

    @pytest.mark.parametrize("params", POSITIONS)
    def test_the_hatch_restores_the_failure(self, tmp_path, params, monkeypatch):
        """The A/B, so this file cannot pass vacuously if the retry is reverted.

        The tolerance is read once per run onto the user data rather than into a
        function-local static, which is what makes this ``setenv`` bite in a
        process that has already solved something."""
        monkeypatch.setenv("BNGSIM_SENS_CLAMP_NUMERIC_ZERO", "0")
        kind, detail = _sens(_net(tmp_path, NUMERICALLY_ZERO), params)
        assert kind == "raised"
        assert "sensitivity RHS returned a non-finite value" in detail


class TestABaseThatIsGenuinelyNegative:
    """``-61.0``: thirteen orders the *other* side of ``atol``. ``∂/∂n base^n``
    does not exist there, and the retry must not manufacture one.

    This is the row that makes the fix a retry rather than a widened condition.
    ``base <= 0`` in the emitter would return ``0`` here — a confident wrong
    number, which is exactly what GH #384/#386 took out of this code path.
    """

    @pytest.mark.parametrize("params", POSITIONS)
    def test_it_is_still_refused(self, tmp_path, params):
        kind, detail = _sens(_net(tmp_path, GENUINELY_NEGATIVE), params)
        assert kind == "raised"
        assert "sensitivity RHS returned a non-finite value" in detail

    def test_and_the_refusal_names_the_state(self, tmp_path):
        """A refusal that did not say which species and what value would leave a
        reader unable to tell this case from the rescued one."""
        _, detail = _sens(_net(tmp_path, GENUINELY_NEGATIVE), ["n", "vmax", "kdeg"])
        assert "A()" in detail and "-61" in detail
