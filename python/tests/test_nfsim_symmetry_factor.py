"""Reaction center symmetry regression tests for the vendored NFsim engine.

GH #195. BNG2.pl emits ``symmetry_factor`` on a ``<ReactionRule>`` whenever the
reactant pattern has a non-trivial automorphism (``RxnRule.pm`` computes
``MultScale = 1/automorphisms/context-permutations`` and applies it as the
statistical factor to every generated reaction), independently of rate law type.

NFsim used to honor it only for constant (``Ele``) rates. ``ReactionClass``'s
constructor scaled its own ``baseRate`` *argument*, which shadows the member the
argument had already been copied into, so the correction was discarded; ``Ele``
recovered it only because ``NFinput`` follows the constructor with
``setBaseRate()``, which applies the factor itself. Every other rate law is
constructed with ``baseRate=1`` and never calls ``setBaseRate``, so a symmetric
rule fired at ``1/symmetry_factor`` times its intended rate -- 2x for a
homodimer.

``symmetry_factor_rate_laws.xml`` puts five dimer pools of ``X0`` copies through
five rate laws that all encode the same intended per-dimer rate ``mu``, so every
pool must follow the same exponential ``X0*exp(-mu*t)``.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

import bngsim
import pytest
from bngsim import NfsimSession


def _has_nfsim() -> bool:
    return getattr(bngsim, "HAS_NFSIM", False)


pytestmark = pytest.mark.skipif(
    not _has_nfsim(),
    reason="NFsim not compiled in",
)

# Must match tests/data/nfsim/symmetry_factor_rate_laws.bngl.
X0 = 4000.0
MU = 1.0e-3
T_END = 1000.0
SEEDS = (1, 2, 3)

#: Survivor count a correctly divided-out symmetry factor produces.
EXACT = X0 * math.exp(-MU * T_END)  # 1471.5
#: Survivor count a dropped symmetry factor produces (the rule runs at 2*mu).
DOUBLE_RATE = X0 * math.exp(-2.0 * MU * T_END)  # 541.3

# Per-seed scatter is ~sqrt(X0*p*(1-p)) ~= 30 counts, so a 3-seed mean has
# sigma ~= 18. A +/-170 band is ~9 sigma wide and still leaves DOUBLE_RATE
# 50 sigma outside it -- the two hypotheses are never confusable.
TOLERANCE = 170.0


def _mean_observables(
    xml: Path, *, seeds: tuple[int, ...] = SEEDS, t_end: float = T_END
) -> dict[str, float]:
    runs: dict[str, list[float]] = {}
    for seed in seeds:
        with NfsimSession(str(xml), molecule_limit=1_000_000) as nf:
            nf.initialize(seed=seed)
            nf.simulate(0.0, t_end, 2)
            for name, value in zip(
                nf.get_observable_names(), nf.get_observable_values(), strict=False
            ):
                runs.setdefault(name, []).append(value)
    return {name: statistics.fmean(values) for name, values in runs.items()}


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "tests" / "data"


@pytest.fixture(scope="module")
def symmetry_observables() -> dict[str, float]:
    """Final observables averaged over ``SEEDS`` (one NFsim sweep for the module)."""
    return _mean_observables(_data_dir() / "nfsim" / "symmetry_factor_rate_laws.xml")


@pytest.fixture(scope="module")
def saturated_mm_observables() -> dict[str, float]:
    """Final observables for the above-linear-regime Michaelis-Menten pair."""
    return _mean_observables(
        _data_dir() / "nfsim" / "symmetry_factor_mm_saturated.xml",
        seeds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        t_end=2000.0,
    )


class TestSymmetryFactorHonoredByEveryRateLaw:
    """Each symmetric pool must decay at ``mu``, not ``2*mu`` (#195)."""

    @pytest.mark.parametrize(
        ("observable", "rate_law"),
        [
            ("Sym_fn", "global function (FunctionalRxnClass)"),
            ("Sym_dor", "local function (DORRxnClass)"),
            ("Sym_mm", "Michaelis-Menten (MMRxnClass)"),
        ],
    )
    def test_symmetric_pool_decays_at_the_intended_rate(
        self, symmetry_observables: dict[str, float], observable: str, rate_law: str
    ) -> None:
        got = symmetry_observables[observable]
        assert got == pytest.approx(EXACT, abs=TOLERANCE), (
            f"{observable} ({rate_law}) ended at {got:.1f}; expected ~{EXACT:.1f}. "
            f"{DOUBLE_RATE:.1f} means the symmetry factor was dropped and the rule "
            "fired at twice its intended rate (#195)."
        )

    @pytest.mark.parametrize(
        ("observable", "rate_law"),
        [
            ("Sym_k", "symmetric, constant rate (BasicRxnClass)"),
            ("Asym_fn", "asymmetric, global function"),
            ("Asym_mm", "asymmetric, Michaelis-Menten"),
        ],
    )
    def test_already_correct_paths_are_unchanged(
        self, symmetry_observables: dict[str, float], observable: str, rate_law: str
    ) -> None:
        # These three were correct before the fix. They are the guard against
        # applying the factor twice, or applying it to an asymmetric rule.
        got = symmetry_observables[observable]
        assert got == pytest.approx(EXACT, abs=TOLERANCE), (
            f"{observable} ({rate_law}) ended at {got:.1f}; expected ~{EXACT:.1f}. "
            "This path did not need correcting, so a shift here means the symmetry "
            "factor is now being applied where it should not be (#195)."
        )

    def test_symmetric_and_asymmetric_agree_pairwise(
        self, symmetry_observables: dict[str, float]
    ) -> None:
        # The sharpest form of the claim: a rule's decay must not depend on
        # whether its reactant pattern happens to be symmetric. Same rate law,
        # same intended rate, so the paired pools must land together.
        obs = symmetry_observables
        assert obs["Sym_fn"] == pytest.approx(obs["Asym_fn"], abs=2 * TOLERANCE), (
            f"symmetric ({obs['Sym_fn']:.1f}) and asymmetric ({obs['Asym_fn']:.1f}) "
            "global-function rules disagree (#195)"
        )
        assert obs["Sym_mm"] == pytest.approx(obs["Asym_mm"], abs=2 * TOLERANCE), (
            f"symmetric ({obs['Sym_mm']:.1f}) and asymmetric ({obs['Asym_mm']:.1f}) "
            "Michaelis-Menten rules disagree (#195)"
        )


class TestMichaelisMentenAboveTheLinearRegime:
    """The MM factor goes on the substrate count, not the propensity (#195).

    ``symmetry_factor_rate_laws.xml`` keeps its MM pair far below saturation,
    where the two placements coincide, so it cannot tell them apart::

        sigma * MM(2N, E)  ==  MM(sigma * 2N, E)      (linear regime only)

    This fixture sits at ``X0/Km == 0.4``, where they separate. The symmetry
    factor corrects a *match multiplicity* -- a symmetric substrate pattern
    matches each complex twice, so the law is handed ``2N`` when only ``N``
    complexes exist -- and MM is not linear in that count.
    """

    def test_symmetric_and_asymmetric_decay_together(
        self, saturated_mm_observables: dict[str, float]
    ) -> None:
        # No closed form here; the pairing is the oracle. The two rules differ
        # only in whether the substrate dimer's halves are the same molecule
        # type, so they must land together. Measured, 10 seeds, t=2000:
        #
        #   no factor at all             sym  153.0   asym 754.8
        #   factor on the propensity     sym  977.0   asym 746.0
        #   factor on the substrate      sym  758.0   asym 756.3
        #
        # Per-seed scatter is ~25 counts, so a 10-seed mean has sigma ~8. An
        # 80-count band is ~10 sigma wide and still leaves the propensity
        # placement's ~230 gap far outside.
        obs = saturated_mm_observables
        sym, asym = obs["Sym_mm"], obs["Asym_mm"]
        assert sym == pytest.approx(asym, abs=80.0), (
            f"saturated MM: symmetric ended at {sym:.1f}, asymmetric at {asym:.1f}. "
            f"A gap near +230 means the symmetry factor is being applied to the "
            f"finished propensity instead of to the substrate count, which is only "
            f"exact below saturation (#195)."
        )

    def test_the_pair_actually_saturates(self, saturated_mm_observables: dict[str, float]) -> None:
        # Guard the fixture itself: if a future parameter edit drifted this model
        # back into the linear regime, the test above would keep passing while
        # having stopped discriminating between the two placements.
        obs = saturated_mm_observables
        # Linear-regime decay for these parameters would leave ~1471 (as in the
        # companion fixture); saturated decay leaves ~750.
        assert obs["Asym_mm"] < 1100.0, (
            f"the MM control ended at {obs['Asym_mm']:.1f}, too close to the linear-"
            "regime value -- this fixture no longer probes the nonlinear range and "
            "cannot distinguish the two placements of the symmetry factor"
        )


# ── TotalRate: the other half of the 2x2 (GH #426) ────────────────────────────
#
# Everything above is the non-TotalRate row: a rule that states a per-instance
# rate needs the symmetry factor, and #195 made every rate law honor it. This is
# the TotalRate row, where the same factor must NOT be applied -- it corrects a
# match count, and TotalRate states the whole propensity outright, so there is
# no count to correct.
#
# Must match tests/data/nfsim/symmetry_factor_total_rate.bngl.
TR_X0 = 4000.0
TR_KT = 1.0
TR_T_END = 1000.0

#: Survivors when the symmetry factor is correctly left off: kt*T firings
#: consuming two molecules each (and 2*kt*T firings consuming one, for the
#: asymmetric control) -- both pools land here.
TR_EXACT = TR_X0 - 2.0 * TR_KT * TR_T_END  # 2000.0
#: Survivors when a 0.5 is wrongly applied: half the firings, half the loss.
TR_HALVED = TR_X0 - TR_KT * TR_T_END  # 3000.0

# Firing counts are Poisson, so the symmetric pool's per-seed scatter is
# 2*sqrt(1000) ~= 63 counts and a 3-seed mean has sigma ~= 37. A +/-200 band is
# ~5.5 sigma wide and still leaves TR_HALVED 22 sigma outside it.
TR_TOLERANCE = 200.0


@pytest.fixture(scope="module")
def total_rate_observables() -> dict[str, float]:
    """Final observables averaged over ``SEEDS`` for the TotalRate fixture."""
    return _mean_observables(
        _data_dir() / "nfsim" / "symmetry_factor_total_rate.xml",
        t_end=TR_T_END,
    )


class TestTotalRateIgnoresTheSymmetryFactor:
    """A TotalRate rule fires at the stated rate, symmetric or not (#426).

    BNG2.pl forces every TotalRate rate law into a Function even when it is a
    bare constant (``RateLaw.pm``: ``my $force_fcn = $totalRate ? 1 : 0;``) and
    rejects TotalRate on Sat/MM/Hill, Arrhenius, and local functions. So
    ``FunctionalRxnClass`` is the entire reachable surface -- and it is also the
    class #195 taught to scale by ``baseRate``, which for that class carries the
    symmetry factor and nothing else. That is how the two features collided:
    every BNG-generated TotalRate rule with a symmetric reaction center ran at
    half the rate its model asked for.

    Measured against released NFsim v1.14.3 on the issue's reproducer (a
    homodimer stating k=0.02 over t=1000 from 400 monomers, so 360 survivors):
    stock 360.3, bngsim before the fix 383.4, after 360.3.
    """

    def test_symmetric_rule_fires_at_the_stated_total_rate(
        self, total_rate_observables: dict[str, float]
    ) -> None:
        got = total_rate_observables["Tsym_free"]
        assert got == pytest.approx(TR_EXACT, abs=TR_TOLERANCE), (
            f"Tsym_free ended at {got:.1f}; expected ~{TR_EXACT:.1f}. "
            f"{TR_HALVED:.1f} means the reaction center symmetry factor was applied "
            "to a rule that states its own total rate, halving the propensity "
            "(#426)."
        )

    def test_asymmetric_control_is_unchanged(
        self, total_rate_observables: dict[str, float]
    ) -> None:
        # symmetry_factor="1" here, so no placement of the factor can move this
        # pool. It fails only if TotalRate handling broke in some other way --
        # which is what separates "the fix worked" from "TotalRate went to zero".
        got = total_rate_observables["Tasym_free"]
        assert got == pytest.approx(TR_EXACT, abs=TR_TOLERANCE), (
            f"Tasym_free ended at {got:.1f}; expected ~{TR_EXACT:.1f}. This rule "
            "has no reaction center symmetry, so a shift here is a change in "
            "TotalRate handling itself, not in the symmetry factor (#426)."
        )

    def test_symmetric_and_asymmetric_agree(
        self, total_rate_observables: dict[str, float]
    ) -> None:
        # The sharpest form of the claim, and the one that needs no expected
        # value: under TotalRate a rule's propensity is whatever it says it is,
        # so it must not depend on whether its reactant pattern is symmetric.
        obs = total_rate_observables
        sym, asym = obs["Tsym_free"], obs["Tasym_free"]
        assert sym == pytest.approx(asym, abs=2 * TR_TOLERANCE), (
            f"symmetric ({sym:.1f}) and asymmetric ({asym:.1f}) TotalRate rules "
            "disagree; both state a total rate that consumes the observed species "
            "at the same rate, so they must land together (#426)"
        )
