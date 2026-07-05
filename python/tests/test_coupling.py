"""bngsim.coupling — the hardened state-exchange layer (GH #102 Stage 1).

Covers the Stage 1 deliverables: bulk count/concentration/amount converters,
shared-species name↔index addressing, the discrete rounding policy with leak
accounting, conservation/no-leak checks, the cell-division divider, the
volume-coupling helpers, and the ``make_subset_model`` ODE-subset-as-model
helper. The headline acceptance — a two-subset operator split reproducing a
reference trajectory within tolerance — is the ``TestOperatorSplit`` class.

Oracles, not tautologies: converters are pinned by round-trip identity and the
``storage == amount`` invariant for ``.net`` models; the divider by *exact*
integer conservation; ``make_subset_model`` by integrating identically to the
model it reconstructs; and the operator split by reproducing the monolithic ODE
(exactly for ODE+ODE in the dt→0 limit, and in the mean for ODE+SSA, since a
purely first-order network's SSA mean solves the ODE).
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim import (
    ConservationError,
    ConservationLedger,
    CouplingMap,
    DiscreteExchange,
    Divider,
    ReactionKernel,
    UnitConverter,
    get_compartment_volume,
    make_subset_model,
    moiety_total,
    round_to_counts,
    set_compartment_volume,
)
from bngsim._bngsim_core import ModelBuilder

# ─── Model fixtures ──────────────────────────────────────────────────────────


def _linear_chain(
    rates: list[float],
    init0: float = 100.0,
    *,
    volume_factors: list[float] | None = None,
    observables: bool = False,
) -> bngsim.Model:
    """S0 → S1 → ... → S_n first-order transfer chain (conserves total count).

    ``len(rates)`` reactions over ``len(rates)+1`` species; only S0 is seeded.
    Optionally per-species ``volume_factor`` (SBML-like V_c) and a Total
    observable.
    """
    n = len(rates) + 1
    vf = volume_factors if volume_factors is not None else [1.0] * n
    b = ModelBuilder()
    sp = [b.add_species(f"S{i}", init0 if i == 0 else 0.0, False, vf[i]) for i in range(n)]
    for i, k in enumerate(rates):
        b.add_parameter(f"k{i}", k)
        b.add_reaction([sp[i]], [sp[i + 1]], "elementary", f"k{i}")
    if observables:
        b.add_observable("Total", [(i, 1.0) for i in range(n)])
    return bngsim.Model(_core=b.build())


# ─── UnitConverter ───────────────────────────────────────────────────────────


class TestUnitConverter:
    def test_net_model_storage_equals_amount(self):
        # .net / V=1 ⇒ volume_factor == 1 ⇒ storage == amount == count.
        m = _linear_chain([0.3, 0.5, 0.7])
        uc = UnitConverter.from_model(m)
        np.testing.assert_array_equal(uc.volume_factors, np.ones(4))
        s = np.array([0.4, 1.6, 2.5, 10.0])
        np.testing.assert_array_equal(uc.to_amounts(s), s)
        np.testing.assert_array_equal(uc.from_amounts(s), s)

    def test_amount_round_trip_with_nonuniform_vc(self):
        # SBML-like: storage = amount / V_c, so amount = storage * V_c.
        m = _linear_chain([0.3, 0.5], volume_factors=[2.0, 5.0, 0.5])
        uc = UnitConverter.from_model(m)
        np.testing.assert_array_equal(uc.volume_factors, [2.0, 5.0, 0.5])
        storage = np.array([3.0, 4.0, 8.0])
        np.testing.assert_allclose(uc.to_amounts(storage), [6.0, 20.0, 4.0])
        np.testing.assert_allclose(uc.from_amounts(uc.to_amounts(storage)), storage)

    def test_counts_round_half_away_from_zero(self):
        uc = UnitConverter([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(
            uc.to_counts(np.array([0.4, 1.5, 2.5, 3.49])), [0.0, 2.0, 3.0, 3.0]
        )

    def test_concentration_default_volume_is_storage(self):
        # concentration at the load volume V_c == storage (the at-load conc).
        m = _linear_chain([0.3, 0.5], volume_factors=[2.0, 5.0, 0.5])
        uc = UnitConverter.from_model(m)
        storage = np.array([3.0, 4.0, 8.0])
        np.testing.assert_allclose(uc.to_concentrations(storage), storage)

    @pytest.mark.parametrize("volume", [2.0, [2.0, 4.0, 8.0]])
    def test_concentration_round_trip_with_live_volume(self, volume):
        m = _linear_chain([0.3, 0.5], volume_factors=[2.0, 5.0, 0.5])
        uc = UnitConverter.from_model(m)
        storage = np.array([3.0, 4.0, 8.0])
        conc = uc.to_concentrations(storage, volume=volume)
        # amount = storage * V_c is volume-invariant; conc = amount / volume.
        np.testing.assert_allclose(conc, uc.to_amounts(storage) / np.asarray(volume, float))
        np.testing.assert_allclose(uc.from_concentrations(conc, volume=volume), storage)

    def test_from_model_gathers_names_in_state_order(self):
        m = _linear_chain([0.3, 0.5, 0.7])
        uc = UnitConverter.from_model(m)
        assert uc.names == m.species_names
        assert uc.n_species == m.n_species == len(uc)

    def test_from_kernel_alias(self):
        k = ReactionKernel(_linear_chain([0.3, 0.5]))
        uc = UnitConverter.from_kernel(k)
        assert uc.n_species == 3

    def test_for_species_subsets(self):
        m = _linear_chain([0.3, 0.5], volume_factors=[2.0, 5.0, 0.5])
        uc = UnitConverter.from_model(m).for_species(["S2", "S0"])
        np.testing.assert_array_equal(uc.volume_factors, [0.5, 2.0])

    def test_rejects_bad_volume_factors_and_lengths(self):
        with pytest.raises(ValueError):
            UnitConverter([1.0, 0.0])  # non-positive V_c
        with pytest.raises(ValueError):
            UnitConverter([1.0, -1.0])
        uc = UnitConverter([1.0, 1.0])
        with pytest.raises(ValueError):
            uc.to_amounts(np.array([1.0, 2.0, 3.0]))  # wrong length
        with pytest.raises(ValueError):
            uc.to_concentrations(np.array([1.0, 2.0]), volume=[1.0, 2.0, 3.0])


# ─── CouplingMap ─────────────────────────────────────────────────────────────


class TestCouplingMap:
    def test_gather_scatter_round_trip(self):
        cm = CouplingMap(["A", "B", "C", "D"], ["C", "A"])
        state = np.array([10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(cm.gather(state), [30.0, 10.0])
        out = cm.scatter(state, [7.0, 8.0])
        np.testing.assert_array_equal(out, [8.0, 20.0, 7.0, 40.0])
        # gather is the left-inverse of scatter on the shared slots.
        np.testing.assert_array_equal(cm.gather(out), [7.0, 8.0])

    def test_scatter_copy_semantics(self):
        cm = CouplingMap(["A", "B"], ["A"])
        state = np.array([1.0, 2.0])
        cm.scatter(state, [9.0])  # default copy=True
        np.testing.assert_array_equal(state, [1.0, 2.0])  # untouched
        same = cm.scatter(state, [9.0], copy=False)
        assert same is state and state[0] == 9.0

    def test_read_write_against_live_model(self):
        m = _linear_chain([0.3, 0.5, 0.7])  # S0..S3, S0=100
        cm = CouplingMap.from_model(m, ["S3", "S0"])
        np.testing.assert_array_equal(cm.read(m), [0.0, 100.0])
        cm.write(m, [5.0, 60.0])
        np.testing.assert_array_equal(m.get_concentration("S0"), 60.0)
        np.testing.assert_array_equal(m.get_concentration("S3"), 5.0)

    def test_cross_ordering_addressing(self):
        # Two subsets share species by NAME at different indices; a common
        # shared order lets the orchestrator move state between them.
        a = CouplingMap(["X", "Y", "Z"], ["Y", "Z"])
        b = CouplingMap(["Z", "W", "Y"], ["Y", "Z"])
        state_a = np.array([1.0, 2.0, 3.0])  # X,Y,Z
        shared = a.gather(state_a)  # [Y, Z] = [2, 3]
        state_b = b.scatter(np.zeros(3), shared)  # Z,W,Y
        np.testing.assert_array_equal(state_b, [3.0, 0.0, 2.0])

    def test_rejects_bad_names(self):
        with pytest.raises(KeyError):
            CouplingMap(["A", "B"], ["C"])
        with pytest.raises(ValueError):
            CouplingMap(["A", "A"], ["A"])  # duplicate in all_names
        with pytest.raises(ValueError):
            CouplingMap(["A", "B"], ["A", "A"])  # duplicate shared


# ─── Discrete rounding policy + leak accounting ──────────────────────────────


class TestRounding:
    def test_policies(self):
        a = np.array([0.2, 0.5, 1.5, 2.9, -0.5, -1.5])
        np.testing.assert_array_equal(
            round_to_counts(a, "nearest"), [0.0, 1.0, 2.0, 3.0, -1.0, -2.0]
        )
        np.testing.assert_array_equal(round_to_counts(a, "floor"), np.floor(a))
        np.testing.assert_array_equal(round_to_counts(a, "ceil"), np.ceil(a))

    def test_stochastic_is_integer_and_unbiased(self):
        rng = np.random.default_rng(0)
        a = np.full(20000, 0.3)
        counts = round_to_counts(a, "stochastic", rng=rng)
        assert set(np.unique(counts)) <= {0.0, 1.0}
        assert counts.mean() == pytest.approx(0.3, abs=0.01)  # E[round] = frac

    def test_stochastic_requires_rng(self):
        with pytest.raises(ValueError):
            round_to_counts([0.3], "stochastic")

    def test_dithering_conserves_on_average(self):
        # Feeding a sub-integer amount repeatedly: dithered counts track the
        # running continuous total to within one molecule; plain rounding loses
        # the fraction every step and drifts far below.
        dx = DiscreteExchange(1, policy="nearest", dither=True)
        plain = DiscreteExchange(1, policy="nearest", dither=False)
        N = 100
        amt = 0.3
        dith_total = sum(float(dx.discretize([amt])[0]) for _ in range(N))
        plain_total = sum(float(plain.discretize([amt])[0]) for _ in range(N))
        assert abs(dith_total - N * amt) <= 1.0
        assert abs(dx.leak) <= 1.0  # leak == -carry.sum(), bounded
        assert plain_total == 0.0  # every 0.3 rounds to 0 — silent leak
        assert plain.leak == pytest.approx(-N * amt)  # surfaced

    def test_nonneg_clamps_but_keeps_debt_in_carry(self):
        dx = DiscreteExchange(1, policy="nearest", dither=True, nonneg=True)
        # Build a negative carry, then feed 0 — count must clamp at 0.
        dx._carry[:] = -0.8  # noqa: SLF001 — exercise the clamp directly
        counts = dx.discretize([0.0])
        assert counts[0] == 0.0
        assert dx.carry[0] == pytest.approx(-0.8)  # debt retained

    def test_last_residual_tracks_rounding_error(self):
        dx = DiscreteExchange(3, policy="nearest", dither=False)
        counts = dx.discretize([1.4, 2.6, 3.0])
        np.testing.assert_allclose(dx.last_residual, counts - np.array([1.4, 2.6, 3.0]))

    def test_reset_clears_state(self):
        dx = DiscreteExchange(2)
        dx.discretize([0.6, 0.6])
        dx.reset()
        assert dx.leak == 0.0
        np.testing.assert_array_equal(dx.carry, [0.0, 0.0])


# ─── Conservation / no-leak ──────────────────────────────────────────────────


class TestConservation:
    def test_moiety_total_weighted_and_default(self):
        s = np.array([1.0, 2.0, 3.0])
        assert moiety_total(s) == 6.0
        assert moiety_total(s, weights=[1.0, 0.0, 2.0]) == 7.0

    def test_ledger_tracks_drift_and_baseline(self):
        led = ConservationLedger(atol=1e-9, name="N")
        led.record(np.array([10.0, 0.0]))  # baseline 10
        led.record(np.array([6.0, 4.0]))  # total 10, no drift
        assert led.baseline == 10.0
        assert led.max_abs_drift == pytest.approx(0.0)
        ok, drift = led.check(np.array([6.0, 4.5]))  # total 10.5
        assert not ok and drift == pytest.approx(0.5)
        assert led.max_abs_drift == pytest.approx(0.5)

    def test_assert_conserved_raises_past_tolerance(self):
        led = ConservationLedger(atol=1e-6)
        led.record(np.array([100.0]))
        led.assert_conserved(np.array([100.0 + 1e-9]))  # within tol — no raise
        with pytest.raises(ConservationError):
            led.assert_conserved(np.array([101.0]))


# ─── Divider ─────────────────────────────────────────────────────────────────


class TestDivider:
    @pytest.mark.parametrize("method", ["binomial", "multinomial", "deterministic"])
    @pytest.mark.parametrize("n_daughters", [2, 3, 5])
    def test_partition_conserves_exactly(self, method, n_daughters):
        rng = np.random.default_rng(7)
        d = Divider(method=method, rng=rng)
        parent = np.array([1000.0, 37.0, 0.0, 5.0, 1.0])
        for _ in range(50):
            daughters = d.divide(parent, n_daughters)
            assert len(daughters) == n_daughters
            total = np.sum(daughters, axis=0)
            np.testing.assert_array_equal(total, parent)  # exact integer conservation
            for day in daughters:
                assert np.all(day >= 0) and np.allclose(day, np.rint(day))

    def test_partition_mask_copies_shared_environment(self):
        d = Divider(method="deterministic")
        parent = np.array([10.0, 100.0])
        mask = np.array([True, False])  # S1 is shared environment
        d1, d2 = d.divide(parent, 2, partition_mask=mask)
        assert d1[0] + d2[0] == 10.0  # partitioned
        assert d1[1] == 100.0 and d2[1] == 100.0  # copied, not split

    def test_deterministic_even_split(self):
        d = Divider(method="deterministic")
        d1, d2, d3 = d.divide(np.array([7.0, 6.0, 2.0]), 3)
        # 7 → 3,2,2 ; 6 → 2,2,2 ; 2 → 1,1,0 (largest-remainder)
        np.testing.assert_array_equal(d1, [3.0, 2.0, 1.0])
        np.testing.assert_array_equal(d2, [2.0, 2.0, 1.0])
        np.testing.assert_array_equal(d3, [2.0, 2.0, 0.0])

    def test_binomial_split_is_unbiased(self):
        rng = np.random.default_rng(1)
        d = Divider(method="binomial", rng=rng)
        firsts = [d.divide(np.array([1000.0]), 2)[0][0] for _ in range(400)]
        assert np.mean(firsts) == pytest.approx(500.0, abs=10.0)

    def test_rejects_non_integer_and_negative(self):
        d = Divider(method="deterministic")
        with pytest.raises(ValueError):
            d.divide(np.array([1.5, 2.0]), 2)
        with pytest.raises(ValueError):
            d.divide(np.array([-1.0, 2.0]), 2)

    def test_stochastic_requires_rng(self):
        with pytest.raises(ValueError):
            Divider(method="binomial").divide(np.array([10.0]), 2)


# ─── Volume coupling ─────────────────────────────────────────────────────────


class TestVolumeCoupling:
    def test_get_set_compartment_volume(self):
        b = ModelBuilder()
        b.add_parameter("V", 2.0)
        b.add_species("A", 5.0)
        m = bngsim.Model(_core=b.build())
        assert get_compartment_volume(m, "V") == 2.0
        set_compartment_volume(m, "V", 3.5)
        assert get_compartment_volume(m, "V") == 3.5

    def test_rejects_nonpositive_volume(self):
        b = ModelBuilder()
        b.add_parameter("V", 1.0)
        b.add_species("A", 1.0)
        m = bngsim.Model(_core=b.build())
        with pytest.raises(ValueError):
            set_compartment_volume(m, "V", 0.0)


# ─── make_subset_model ───────────────────────────────────────────────────────


class TestMakeSubsetModel:
    def test_reconstruction_integrates_identically(self):
        # The strongest oracle: a full reconstruction (keep every reaction, fix
        # nothing) must integrate bit-for-bit like the model it copies.
        m = _linear_chain([0.3, 0.5, 0.7, 0.9], observables=True)
        rebuilt = make_subset_model(m)
        assert rebuilt.species_names == m.species_names
        assert rebuilt.n_reactions == m.n_reactions
        ref = bngsim.Simulator(m).run(t_span=(0.0, 20.0), n_points=11)
        got = bngsim.Simulator(rebuilt).run(t_span=(0.0, 20.0), n_points=11)
        np.testing.assert_allclose(got.species, ref.species, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(got.observables, ref.observables, rtol=1e-9, atol=1e-12)

    def test_keep_reactions_subsets_and_fixes(self):
        m = _linear_chain([0.3, 0.5, 0.7])  # 3 reactions, S0..S3
        sub = make_subset_model(m, keep_reactions=[0], fixed_species=["S2", "S3"])
        assert sub.n_reactions == 1
        flags = {s["name"]: s["fixed"] for s in sub._core.codegen_data()["species"]}
        assert flags == {"S0": False, "S1": False, "S2": True, "S3": True}

    def test_fixed_subset_holds_boundary_species_constant(self):
        # With S0→S1 kept and S2,S3 fixed, integrating only changes S0,S1; the
        # fixed boundary species stay put.
        m = _linear_chain([0.3, 0.5, 0.7])
        sub = make_subset_model(m, keep_reactions=[0])
        sub.set_state(np.array([100.0, 0.0, 50.0, 25.0]))
        # S2,S3 are not fixed here but no kept reaction touches them ⇒ constant.
        final = bngsim.Simulator(sub).run(t_span=(0.0, 50.0), n_points=2).species[-1]
        assert final[2] == pytest.approx(50.0) and final[3] == pytest.approx(25.0)
        assert final[0] + final[1] == pytest.approx(100.0)  # S0→S1 conserves

    def test_functional_rate_law_reconstructs(self):
        b = ModelBuilder()
        a = b.add_species("A", 10.0)
        c = b.add_species("C", 0.0)
        b.add_parameter("k", 0.4)
        b.add_observable("Atot", [(a, 1.0)])
        b.add_function("rate", "k*Atot")  # functions reference observables, not raw species
        b.add_reaction([a], [c], "functional", "rate", 1.0, False)
        m = bngsim.Model(_core=b.build())
        rebuilt = make_subset_model(m)
        ref = bngsim.Simulator(m).run(t_span=(0.0, 5.0), n_points=6).species
        got = bngsim.Simulator(rebuilt).run(t_span=(0.0, 5.0), n_points=6).species
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)

    def test_rate_rule_flag_survives_reconstruction(self):
        # GH #81: a rate rule compiles to a Functional `[] → [X]` reaction flagged
        # is_rate_rule_ode, which the SSA loop integrates deterministically rather
        # than firing as a stochastic channel. make_subset_model must forward that
        # flag through codegen_data — otherwise the rebuilt subset would silently
        # fire it as a birth/death channel again (the pre-fix #81 bug). Oracle: the
        # rate-rule target k stays identical across replicates (std ≈ 0, integrated
        # as an ODE) and the dependent decay reaches its correct value rather than
        # the frozen-k value.
        # This is the only test in the module that needs the optional `antimony`
        # dependency (GH #153); it is excluded from the cibuildwheel test env, so
        # guard here rather than skipping the whole module.
        pytest.importorskip("antimony")
        m = bngsim.Model.from_antimony_string(
            "model tv_decay\n"
            "  compartment C = 1;\n"
            "  species A in C = 1000;\n"
            "  k = 0.0; c = 0.05;\n"
            "  k' = c;\n"
            "  J1: A => ; k*A;\n"
            "end\n"
        )
        src_flags = [r.get("is_rate_rule_ode") for r in m._core.codegen_data()["reactions"]]
        assert any(src_flags), "source rate-rule reaction not flagged is_rate_rule_ode"
        sub = make_subset_model(m)
        assert [
            r.get("is_rate_rule_ode") for r in sub._core.codegen_data()["reactions"]
        ] == src_flags

        reps = 64
        res = bngsim.Simulator(sub, method="ssa").run_batch(
            t_span=(0.0, 10.0), n_points=11, params=[{} for _ in range(reps)], seed=11
        )
        arr = np.stack([np.asarray(r.species) for r in res], axis=0)
        names = list(sub.species_names)
        ki, ai = names.index("k"), names.index("A")
        t = np.linspace(0.0, 10.0, 11)
        # k integrated as an ODE ⇒ deterministic across replicates, equals c·t.
        assert arr[:, :, ki].std(0).max() < 1e-9
        np.testing.assert_allclose(arr[:, :, ki].mean(0), 0.05 * t, atol=1e-9)
        # Dependent decay reaches ≈82 (correct), not ≈625 (frozen-k channel bug).
        assert arr[:, -1, ai].mean() < 0.25 * 1000.0

    @staticmethod
    def _mm_model(*, with_decay: bool = False) -> bngsim.Model:
        # E + S → E + P with MM(kcat, Km); optionally P → ∅ (elementary decay).
        b = ModelBuilder()
        b.add_parameter("kcat", 1.0)
        b.add_parameter("Km", 50.0)
        e = b.add_species("E", 10.0)
        s = b.add_species("S", 100.0)
        p = b.add_species("P", 0.0)
        b.add_reaction([e, s], [e, p], "mm", "kcat,Km")
        if with_decay:
            b.add_parameter("kdeg", 0.2)
            b.add_reaction([p], [], "elementary", "kdeg")
        b.add_observable("Stot", [(s, 1.0)])
        return bngsim.Model(_core=b.build())

    def test_michaelis_menten_reconstructs_identically(self):
        # GH #103: the MM rate law reconstructs and integrates bit-for-bit.
        m = self._mm_model(with_decay=True)
        rebuilt = make_subset_model(m)
        assert rebuilt._core.codegen_data()["reactions"][0]["type"] == "mm"
        ref = bngsim.Simulator(m).run(t_span=(0.0, 30.0), n_points=16).species
        got = bngsim.Simulator(rebuilt).run(t_span=(0.0, 30.0), n_points=16).species
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)

    def test_michaelis_menten_subset_preserves_enzyme_substrate(self):
        # Keep only the MM reaction (drop the decay). The reconstructed reaction
        # must keep E as enzyme (catalyst, unchanged) and S as substrate (S+P
        # conserved as E turns S into P).
        m = self._mm_model(with_decay=True)
        sub = make_subset_model(m, keep_reactions=[0])
        assert sub.n_reactions == 1
        final = bngsim.Simulator(sub).run(t_span=(0.0, 50.0), n_points=2).species[-1]
        names = sub.species_names
        e, s, p = (final[names.index(n)] for n in ("E", "S", "P"))
        assert e == pytest.approx(10.0)  # enzyme is catalytic — unchanged
        assert s + p == pytest.approx(100.0)  # substrate→product conserves the pool
        assert s < 100.0 and p > 0.0  # the MM reaction actually ran

    def test_rejects_unreconstructable_features(self):
        # An event cannot round-trip through codegen_data.
        b = ModelBuilder()
        a = b.add_species("A", 10.0)
        b.add_parameter("k", 1.0)
        b.add_reaction([a], [], "elementary", "k")
        b.add_event("e", "time() > 1", [(a, "0.0")])
        m = bngsim.Model(_core=b.build())
        with pytest.raises(NotImplementedError, match="events"):
            make_subset_model(m)


# ─── Acceptance: two-subset operator split ───────────────────────────────────


def _strang_step(ka, kb, full_state, dt, *, converter=None, discrete=None, seed=None):
    """One Strang step ½A · B · ½A over a shared full-state vector.

    ``ka`` advances the ODE subset, ``kb`` the other operator. Both kernels hold
    the full species set in the same order, so the exchange is the whole vector;
    when ``converter``/``discrete`` are given the B-subset hand-off goes through
    count space (storage → amounts → integer counts → storage), exercising the
    units + rounding layer. Returns the post-step full state.
    """

    def run_a(state, h):
        ka.set_state(state)
        return ka.advance(h)

    def run_b(state, h):
        if converter is not None and discrete is not None:
            counts = discrete.discretize(converter.to_amounts(state))
            state = converter.from_counts(counts)
        kb.set_state(state)
        return kb.advance(h, seed=seed)

    return run_a(run_b(run_a(full_state, dt / 2), dt), dt / 2)


class TestOperatorSplit:
    """Headline acceptance: a bngsim+bngsim operator split over shared species
    reproduces the monolithic reference within tolerance."""

    def test_ode_ode_split_reproduces_monolithic(self):
        # Partition a chain's reactions into two ODE operators and Strang-split.
        # As dt→0 the split converges to the monolithic ODE at second order; the
        # total count is conserved across every exchange to machine precision.
        rates = [0.3, 0.6, 0.45, 0.8, 0.5]
        full = _linear_chain(rates, init0=100.0)
        T = 12.0

        ref = bngsim.Simulator(full).run(t_span=(0.0, T), n_points=2).species[-1]

        even = make_subset_model(full, keep_reactions=[0, 2, 4])
        odd = make_subset_model(full, keep_reactions=[1, 3])
        ka, kb = ReactionKernel(even), ReactionKernel(odd)

        def split_error(n_steps):
            ka.reset()
            kb.reset()  # subset models are reused → reset to the initial state
            led = ConservationLedger(atol=1e-7)
            state = ka.get_state()
            led.record(state)
            for _ in range(n_steps):
                state = _strang_step(ka, kb, state, T / n_steps)
                led.record(state)
            assert led.max_abs_drift < 1e-7  # total count conserved across exchange
            return np.max(np.abs(state - ref))

        coarse = split_error(24)
        fine = split_error(48)
        assert coarse / fine == pytest.approx(4.0, abs=0.6)  # Strang is O(dt²)
        assert fine < 0.1  # and already close to the monolithic at this resolution

    def test_ode_ssa_split_mean_matches_monolithic(self):
        # A purely first-order network's SSA mean solves the ODE exactly (linear
        # propensities ⇒ no moment closure error), so an ODE+SSA operator split,
        # averaged over seeds, must reproduce the monolithic ODE — within Monte
        # Carlo noise. This is the hybrid-at-scale acceptance in miniature.
        rates = [0.5, 0.3, 0.7, 0.4]
        init = 5000.0
        full = _linear_chain(rates, init0=init)
        T = 6.0
        n_steps = 12

        ref = bngsim.Simulator(full).run(t_span=(0.0, T), n_points=2).species[-1]

        ode_part = make_subset_model(full, keep_reactions=[0, 2])
        ssa_part = make_subset_model(full, keep_reactions=[1, 3])
        uc = UnitConverter.from_model(ssa_part)  # V_c = 1 ⇒ storage == counts
        ka = ReactionKernel(ode_part, method="ode")
        kb = ReactionKernel(ssa_part, method="ssa")

        n_reps = 48
        finals = np.zeros((n_reps, full.n_species))
        for r in range(n_reps):
            ka.reset()
            kb.reset()  # reused subset models → reset to the seeded initial state
            dx = DiscreteExchange(full.n_species, policy="nearest", dither=True)
            state = ka.get_state()
            for step in range(n_steps):
                state = _strang_step(
                    ka, kb, state, T / n_steps, converter=uc, discrete=dx, seed=r * 1000 + step
                )
            finals[r] = state

        mean_final = finals.mean(axis=0)
        # Total molecule count is conserved exactly by every SSA fire and by the
        # dithered hand-off to within a molecule, so the ensemble total tracks
        # the seeded count.
        assert mean_final.sum() == pytest.approx(init, rel=5e-3)
        # The mean reproduces the deterministic reference within MC tolerance
        # (measured max |Δ| ≈ 0.0033·init over 48 reps).
        np.testing.assert_allclose(mean_final, ref, atol=0.012 * init)
