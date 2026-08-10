"""Pure-context reactant counting for the vendored NFsim engine.

GH #281, and distinct from #195 (``test_nfsim_symmetry_factor.py``). There the
symmetry factor *was* emitted by BNG and NFsim discarded it. Here BNG emits
``symmetry_factor="1"`` and there is nothing to discard -- NFsim over-counts a
pattern that BNG never flagged.

BioNetGen gives a reactant pattern the rule does **not** transform one reaction
instance per matching *complex*, however many molecules inside that complex
match it. The reason is that every embedding of such a pattern yields the
identical reaction -- same reactants, same products, same transformation -- so
there is only one reaction to count. NFsim enumerates matches per molecule, so
it reports one per matching molecule instead.

The over-count is therefore **not** a pattern-symmetry effect, even though the
symmetric cases are the ones that make it obvious. ``Sub_scaffold`` below is the
sharp case: a single-molecule pattern against a complex holding two
*distinguishable* copies, so there is no symmetry anywhere in it, and it was
still counted twice.

Where the pattern *is* transformed none of this applies: binding one of a
homodimer's two sites really is two reactions, and BNG emits the factor of two
for it. ``Bind_sym`` guards that direction.

The oracle throughout is BNG's own network expansion simulated with bngsim's
ODE engine, which is a genuinely independent path to the same model.
"""

from __future__ import annotations

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

# Must match tests/data/nfsim/context_symmetry.bngl.
T_END = 2000.0
SEEDS = tuple(range(1, 11))

#: Endpoint of the BNG-generated network under bngsim's ODE engine, for every
#: pool that shares the intended per-substrate rate mu = kcat*E0 = 2.5e-4.
#: X0*exp(-mu*t) = 4000*exp(-0.5) closes the loop analytically.
EXACT = 2426.1
#: Same oracle for the Michaelis-Menten pair, which has no closed form here.
EXACT_MM = 2344.8

# Per-seed scatter is ~30 counts, so a 10-seed mean has sigma ~10. A +/-100 band
# is ~10 sigma wide and still leaves every over-counted value (~1471 at 2x, ~893
# at 3x) more than 900 counts outside it.
TOLERANCE = 100.0


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "tests" / "data"


@pytest.fixture(scope="module")
def context_observables() -> dict[str, float]:
    """Final observables averaged over ``SEEDS`` (one NFsim sweep for the module)."""
    xml = _data_dir() / "nfsim" / "context_symmetry.xml"
    runs: dict[str, list[float]] = {}
    for seed in SEEDS:
        with NfsimSession(str(xml), molecule_limit=1_000_000) as nf:
            nf.initialize(seed=seed)
            nf.simulate(0.0, T_END, 2)
            for name, value in zip(
                nf.get_observable_names(), nf.get_observable_values(), strict=False
            ):
                runs.setdefault(name, []).append(value)
    return {name: statistics.fmean(values) for name, values in runs.items()}


class TestContextIsCountedOncePerComplex:
    """A catalyst the rule does not transform must not set the rule's rate (#281)."""

    @pytest.mark.parametrize(
        ("observable", "shape"),
        [
            ("Dim_sym", "homodimer catalyst, constant rate (BasicRxnClass)"),
            ("Fn_sym", "homodimer catalyst, global function (FunctionalRxnClass)"),
            ("Dor_sym", "homodimer catalyst, local function (DORRxnClass)"),
            ("Mm_sym", "symmetric enzyme, Michaelis-Menten (MMRxnClass)"),
            ("Sub_subunit", "single-subunit pattern Ux(d!+) against a homodimer"),
            ("Sub_scaffold", "single-molecule pattern, two distinguishable copies"),
        ],
    )
    def test_two_matching_molecules_in_one_complex_count_once(
        self, context_observables: dict[str, float], observable: str, shape: str
    ) -> None:
        expected = EXACT_MM if observable == "Mm_sym" else EXACT
        got = context_observables[observable]
        assert got == pytest.approx(expected, abs=TOLERANCE), (
            f"{observable} ({shape}) ended at {got:.1f}; expected ~{expected:.1f}. "
            "A value near 1471 (1290 for the MM pool) means the two matching "
            "molecules in each catalyst complex were counted as two reaction "
            "instances instead of one (#281)."
        )

    def test_the_symmetry_free_case_is_covered(
        self, context_observables: dict[str, float]
    ) -> None:
        # The load-bearing case, called out separately because it rules out every
        # fix keyed on the pattern's own automorphisms. Sub_scaffold's pattern is a
        # single molecule, and its catalyst holds two copies at *distinguishable*
        # scaffold positions, so neither the pattern nor the species has any
        # symmetry to divide out -- and BNG still counts one instance per complex.
        got = context_observables["Sub_scaffold"]
        assert got == pytest.approx(EXACT, abs=TOLERANCE), (
            f"Sub_scaffold ended at {got:.1f}; expected ~{EXACT:.1f}. ~1471 means "
            "counting is still keyed on symmetry rather than on complex identity "
            "(#281)."
        )

    def test_a_three_subunit_context_is_corrected_threefold(
        self, context_observables: dict[str, float]
    ) -> None:
        # The over-count is N-fold in the number of matching molecules per complex,
        # not 2-fold. A fix hardcoded to a factor of two passes every pair above and
        # fails only here.
        got = context_observables["Ring_sym"]
        assert got == pytest.approx(EXACT, abs=TOLERANCE), (
            f"Ring_sym ended at {got:.1f}; expected ~{EXACT:.1f}. ~893 means the "
            "homotrimer ring catalyst was counted three times per complex; ~1471 "
            "means only a factor of two was divided out (#281)."
        )

    @pytest.mark.parametrize(
        ("multi", "single", "rate_law"),
        [
            ("Dim_sym", "Dim_asym", "constant rate"),
            ("Ring_sym", "Ring_asym", "constant rate, trimer ring"),
            ("Fn_sym", "Fn_asym", "global function"),
            ("Dor_sym", "Dor_asym", "local function"),
            ("Mm_sym", "Mm_asym", "Michaelis-Menten"),
        ],
    )
    def test_multi_and_single_subunit_catalysts_agree_pairwise(
        self,
        context_observables: dict[str, float],
        multi: str,
        single: str,
        rate_law: str,
    ) -> None:
        # The sharpest form of the claim, and the one that needs no oracle at all:
        # inside a single run, two catalysts present at the same complex count and
        # carrying the same rate constant must give the same rate.
        obs = context_observables
        many, one = obs[multi], obs[single]
        assert many == pytest.approx(one, abs=2 * TOLERANCE), (
            f"{rate_law}: multi-subunit catalyst ended at {many:.1f}, single-subunit "
            f"control at {one:.1f}. Same rate constant, same number of catalyst "
            "complexes, so the gap is the pure-context over-count (#281)."
        )


class TestTransformedPatternsAreLeftAlone:
    """The correction must not reach a pattern BNG has already accounted for."""

    def test_a_symmetric_binding_partner_keeps_both_of_its_sites(
        self, context_observables: dict[str, float]
    ) -> None:
        # Bind_sym's catalyst dimer IS transformed -- one half of it binds -- so its
        # two halves are two genuinely distinct reactive sites and two distinct
        # reactions. BNG says so explicitly: the generated network gives this rule
        # `2*kb` where the heterodimer control gets `kb`. Those two matches per
        # complex are correct and must survive.
        #
        # This is the case that makes "which reactants are pure context" delicate:
        # NFsim marks the second partner of a binding with an EMPTY transform, the
        # same type TransformationSet::finalize() uses for the placeholder it puts
        # on an untransformed reactant, so deciding it from the transformation types
        # after the fact misclassifies exactly this rule. Measured with that
        # misclassification in place, this pool lands at 314 -- on top of its own
        # control, its 2x erased.
        got = context_observables["Bind_sym"]
        assert got == pytest.approx(217.6, abs=25.0), (
            f"Bind_sym ended at {got:.1f}; expected ~217.6. ~314 means per-complex "
            "counting was applied to a reactant the rule transforms, dividing out a "
            "factor BNG had deliberately put in the rate (#281)."
        )

    def test_the_binding_control_is_unchanged(self, context_observables: dict[str, float]) -> None:
        # Guards the pairing above: the heterodimer control has one reactive site,
        # so nothing in #281 may move it.
        got = context_observables["Bind_asym"]
        assert got == pytest.approx(318.9, abs=30.0), (
            f"Bind_asym ended at {got:.1f}; expected ~318.9. This rule has one "
            "reactive site and one matching molecule per complex, so a shift here "
            "means the correction is reaching rules it has no business touching."
        )

    def test_the_binding_pair_still_separates(self, context_observables: dict[str, float]) -> None:
        # Guard the fixture: the two assertions above only discriminate while the
        # two-site arm actually runs faster than its control. If a future parameter
        # edit flattened that separation, both would keep passing while having
        # stopped testing anything.
        obs = context_observables
        assert obs["Bind_asym"] - obs["Bind_sym"] > 50.0, (
            f"Bind_sym ({obs['Bind_sym']:.1f}) and Bind_asym ({obs['Bind_asym']:.1f}) "
            "have converged -- this pair no longer distinguishes a transformed "
            "pattern's real multiplicity from a context over-count"
        )
