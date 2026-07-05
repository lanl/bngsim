"""Oracle-anchored unit tests of the C++ core, built through the direct
``ModelBuilder`` API with **no loader in the loop**.

Why this module exists
----------------------
The SBML loader (`_sbml_loader.py`) is the most-edited file in the engine, and
several of its fixes lean on a specific *core* behaviour being correct (a
clamped species read frozen in a rate law; a rate rule lowered to a functional
synthesis reaction; an observable/function refreshed live before each rate
evaluation; per-species volume scaling). An audit of that fix history (see
`dev/notes/rr_parity_triage.md` and issues #74/#75) asked whether any loader fix
*masks* a core bug — i.e. compensates in Python for a defect that would still be
wrong for a model reaching the same core path through `.net` or the direct API.

Cross-checking two loaders against each other (the `sbml_roundtrip` harness)
only catches *asymmetric* distortion; a core bug both loaders feed identically
slips through. The defence against that is here: build each core primitive
directly via `ModelBuilder` (bypassing every loader) and pin it to a
**closed-form oracle**. If one of these regresses, the bug is in the core, not
in a loader's interpretation — and no loader workaround can make it pass.

Each "primitive" test names the loader fix whose correctness it underwrites.
The "bedrock" tests establish that the bare CVODE integrator is trustworthy
before the primitives lean on it.
"""

from __future__ import annotations

import numpy as np
import pytest
from bngsim._bngsim_core import CvodeSimulator, ModelBuilder, TimeSpec

# Tight, loader-free integration so the comparison is against the analytical
# oracle, not the default solver tolerance.
_RTOL = 1e-10
_ATOL = 1e-12


def _run_ode(builder: ModelBuilder, t_end: float, n_points: int):
    """Build, integrate at tight tol, return (time, species[n_t, n_sp], names)."""
    model = builder.build()
    sim = CvodeSimulator(model)
    sim.set_tolerances(_RTOL, _ATOL)
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = n_points
    res = sim.run(ts)
    return np.asarray(res.time), np.asarray(res.species_data), list(model.species_names)


# --------------------------------------------------------------------------- #
# Bedrock: the bare integrator is correct on textbook mass-action kinetics.
# --------------------------------------------------------------------------- #
class TestBedrockMassAction:
    def test_first_order_decay(self):
        """A -> ∅, rate k[A]: A(t) = A0 e^{-k t}."""
        b = ModelBuilder()
        b.add_parameter("k", 0.1)
        a = b.add_species("A", 100.0)
        b.add_reaction([a], [], "elementary", "k")

        t, y, _ = _run_ode(b, 10.0, 11)
        np.testing.assert_allclose(y[:, 0], 100.0 * np.exp(-0.1 * t), rtol=1e-6)

    def test_reversible_equilibrium_and_conservation(self):
        """A ⇌ B (kf=2, kr=0.5), A0=10, B0=0.

        Conserved: A+B ≡ 10. Relaxation: A(t) = A_eq + (A0-A_eq) e^{-(kf+kr)t}
        with A_eq = kr/(kf+kr)·10 = 2, B_eq = 8.
        """
        b = ModelBuilder()
        b.add_parameter("kf", 2.0)
        b.add_parameter("kr", 0.5)
        a = b.add_species("A", 10.0)
        bb = b.add_species("B", 0.0)
        b.add_reaction([a], [bb], "elementary", "kf")
        b.add_reaction([bb], [a], "elementary", "kr")

        t, y, _ = _run_ode(b, 5.0, 11)
        a_eq = 2.0
        np.testing.assert_allclose(y[:, 0], a_eq + (10.0 - a_eq) * np.exp(-2.5 * t), rtol=1e-6)
        # Total mass is conserved to integrator precision the whole way.
        np.testing.assert_allclose(y[:, 0] + y[:, 1], 10.0, rtol=1e-9, atol=1e-9)

    def test_bimolecular_second_order(self):
        """A + B -> C with A0=B0=5, rate k[A][B]: A(t) = A0/(1 + A0 k t)."""
        b = ModelBuilder()
        b.add_parameter("k", 0.1)
        a = b.add_species("A", 5.0)
        bb = b.add_species("B", 5.0)
        c = b.add_species("C", 0.0)
        b.add_reaction([a, bb], [c], "elementary", "k")

        t, y, _ = _run_ode(b, 10.0, 11)
        a_oracle = 5.0 / (1.0 + 5.0 * 0.1 * t)
        np.testing.assert_allclose(y[:, 0], a_oracle, rtol=1e-6)
        np.testing.assert_allclose(y[:, 2], 5.0 - a_oracle, rtol=1e-6)  # C = A0 - A


# --------------------------------------------------------------------------- #
# Primitives the loader masking-suspects lean on.
# --------------------------------------------------------------------------- #
class TestClampedSpeciesIsReadFrozen:
    """Pins the core semantics commit 4158575c / e1b15765 rely on: a `fixed`
    (`$`-clamped) species is held at its IC AND is read at that frozen value
    when it appears inside a mass-action rate law.

    The loader's `fixed`-species handling is only correct *because* the core
    reads a clamped species frozen in rate laws. If the core instead integrated
    it, the SBML loader's `boundaryCondition`/`constant` mapping would be wrong.
    """

    def test_fixed_reactant_drives_linear_growth(self):
        """A fixed at 2, B(0)=0, reaction A -> B with rate k[A].

        A is a clamped pool, so [A] ≡ 2 in the rate law and B accumulates at the
        constant rate k·A0: B(t) = k·A0·t. (If A were integrated, A would drain
        and B would saturate — the discriminating wrong answer.)
        """
        b = ModelBuilder()
        b.add_parameter("k", 0.5)
        a = b.add_species("A", 2.0, fixed=True)
        bb = b.add_species("B", 0.0)
        b.add_reaction([a], [bb], "elementary", "k")

        t, y, names = _run_ode(b, 10.0, 11)
        a_idx, b_idx = names.index("A"), names.index("B")
        np.testing.assert_allclose(y[:, a_idx], 2.0, rtol=0, atol=1e-12)  # frozen
        np.testing.assert_allclose(y[:, b_idx], 0.5 * 2.0 * t, rtol=1e-6)  # linear


class TestFunctionalSynthesisReaction:
    """Pins the rate-rule lowering commit e1b15765 emits: an SBML rate rule
    ``dX/dt = f`` becomes ``add_reaction([], [X], "functional", <fn>,
    apply_species_factor=False)`` — an empty-reactant synthesis whose rate law
    *is* the full signed RHS. A boundary species with a rate rule integrates
    through this path, not by being clamped.
    """

    def test_signed_rhs_integrates_as_ode(self):
        """dS/dt = -k·S via a functional synthesis reaction: S(t) = S0 e^{-k t}.

        The rate law is a *function* (the loader always wraps the RHS in
        add_function), here ``rhs = -k*S_obs`` with S_obs an observable on S.
        apply_species_factor=False means the rate law is the entire dS/dt, with
        no extra reactant-population multiplier.
        """
        b = ModelBuilder()
        b.add_parameter("k", 0.3)
        s = b.add_species("S", 100.0)
        b.add_observable("S_obs", [(s, 1.0)])
        b.add_function("rhs", "-k*S_obs")
        b.add_reaction([], [s], "functional", "rhs", apply_species_factor=False)

        t, y, _ = _run_ode(b, 5.0, 11)
        np.testing.assert_allclose(y[:, 0], 100.0 * np.exp(-0.3 * t), rtol=1e-6)


class TestObservableRefreshedLiveInRateLaw:
    """Pins the core property commit 4158575c relies on: an observable/function
    referenced in a functional rate law is recomputed from live state on every
    RHS evaluation, not frozen at its initial value. This is what makes routing
    an AssignmentRule-target species through the Functional path correct.
    """

    def test_function_of_observable_uses_live_value(self):
        """A -> B with functional rate = A_obs (an observable tracking A).

        dA/dt = -A_obs = -[A](t)  ⇒  A(t) = A0 e^{-t}. If A_obs were read frozen
        at its initial 100, dA/dt would be a constant -100 and A would cross
        zero into negative territory near t≈1 — a qualitatively different, and
        obviously wrong, trajectory. Asserting the exponential (and positivity)
        is therefore a direct test that the value is refreshed.
        """
        b = ModelBuilder()
        a = b.add_species("A", 100.0)
        bb = b.add_species("B", 0.0)
        b.add_observable("A_obs", [(a, 1.0)])
        b.add_function("frate", "A_obs")
        b.add_reaction([a], [bb], "functional", "frate", apply_species_factor=False)

        t, y, _ = _run_ode(b, 3.0, 7)
        np.testing.assert_allclose(y[:, 0], 100.0 * np.exp(-t), rtol=1e-6)
        assert np.all(y[:, 0] > 0.0)  # frozen-value bug would drive A negative
        # Conservation: every A lost becomes B.
        np.testing.assert_allclose(y[:, 0] + y[:, 1], 100.0, rtol=1e-9, atol=1e-9)


class TestPerSpeciesVolumeScaling:
    """Pins the cross-compartment ODE accumulator the hOSU fixes (ce9719f8 /
    c064bb51) route through: with ``per_species_volume_scaling=True``,
    ``compute_derivs`` divides a reaction's contribution to each affected
    species by *that species'* ``volume_factor`` (rather than applying one
    scalar rate). This is how an hOSU=true V≠1 reaction restores the amount law.
    """

    def test_each_derivative_divided_by_own_volume_factor(self):
        """A -> B, rate k[A], with V_A=2, V_B=4 and per_species_volume_scaling.

        Reaction rate r = k·[A]. The A-derivative is -r/V_A and the B-derivative
        is +r/V_B. At t=0 (A=10, k=1): dA/dt = -10/2 = -5, dB/dt = +10/4 = +2.5.
        A scalar-rate (non-per-species) accumulator would give ∓r identically,
        so the asymmetric 5 : 2.5 split is the discriminator.
        """
        b = ModelBuilder()
        b.add_parameter("k", 1.0)
        a = b.add_species("A", 10.0, volume_factor=2.0)
        bb = b.add_species("B", 0.0, volume_factor=4.0)
        b.add_reaction([a], [bb], "elementary", "k", per_species_volume_scaling=True)

        # Finite-difference the initial derivatives off a short, tight solve.
        t, y, names = _run_ode(b, 1e-3, 2)
        a_idx, b_idx = names.index("A"), names.index("B")
        dA = (y[1, a_idx] - y[0, a_idx]) / (t[1] - t[0])
        dB = (y[1, b_idx] - y[0, b_idx]) / (t[1] - t[0])
        assert dA == pytest.approx(-5.0, rel=1e-3)
        assert dB == pytest.approx(2.5, rel=1e-3)

    def test_volume_scaled_synthesis_constant_accumulation(self):
        """∅ -> X, constant rate k, V_X=4, per_species_volume_scaling: a clean
        single-species check that dX/dt = k/V_X. X(t) = (k/V_X)·t.
        """
        b = ModelBuilder()
        b.add_parameter("k", 2.0)
        x = b.add_species("X", 0.0, volume_factor=4.0)
        b.add_reaction([], [x], "elementary", "k", per_species_volume_scaling=True)

        t, y, _ = _run_ode(b, 10.0, 11)
        np.testing.assert_allclose(y[:, 0], (2.0 / 4.0) * t, rtol=1e-6)
