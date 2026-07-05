"""Product-molecularity regression tests for the vendored NFsim engine.

These guard the multi-bond ring-opening fix (internal#57) without
re-breaking the single-bond cyclic-dissociation behavior that the same
``checkMolecularity`` path is responsible for (internal#54/#55).

``checkMolecularity`` enforces product-side molecularity for unimolecular
unbinding rules under complex bookkeeping: ``A.B -> A + B`` may only fire when
the deletion actually separates the match into different connected components.
The fix makes that test consider *all* the bonds a rule deletes at once instead
of one bond in isolation.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim import NfsimSession


def _has_nfsim() -> bool:
    return getattr(bngsim, "HAS_NFSIM", False)


pytestmark = pytest.mark.skipif(
    not _has_nfsim(),
    reason="bngsim compiled without NFsim support",
)


def _final_observables(
    xml: Path, *, seed: int, t_end: float, n_pts: int = 2, block_same_complex_binding: bool = True
) -> dict[str, float]:
    with NfsimSession(
        str(xml),
        molecule_limit=1_000_000,
        block_same_complex_binding=block_same_complex_binding,
    ) as nf:
        nf.initialize(seed=seed)
        nf.simulate(0.0, t_end, n_pts)
        return dict(zip(nf.get_observable_names(), nf.get_observable_values(), strict=False))


class TestMultiBondMolecularity:
    """A symmetric two-bond ring-opening dissociation must fire (#57)."""

    def test_two_bond_ring_dissociates(self, data_dir: Path) -> None:
        # ``ring2_homodimer.xml`` seeds 197 copies of the two-bond ring
        #   M(h!1,f!2).M(h!2,f!1)
        # whose only reaction is the reverse homodimerization
        #   M(h!1,f!2).M(h!2,f!1) -> M(h,f) + M(h,f)
        # deleting *both* ring bonds at once. The network ODE relaxes to
        # ~63 free monomers (KD-set equilibrium). Before the fix the per-bond
        # molecularity check refused the dissociation entirely and the ring was
        # a permanent trap (monomers stuck at 0); afterwards it dissociates.
        xml = data_dir / "nfsim" / "ring2_homodimer.xml"
        obs = _final_observables(xml, seed=1, t_end=200_000.0)

        # Conservation: 197 dimers => 394 monomer-equivalents of M.
        assert obs["Mtot"] == pytest.approx(394.0)
        # The ring must actually open. The decisive contrast with the bug is
        # 0 (trapped) vs the ~63 analytical equilibrium; a generous band keeps
        # the assertion robust to single-seed stochastic scatter.
        assert obs["monomers"] >= 30.0, (
            f"two-bond ring failed to dissociate (monomers={obs['monomers']}); "
            "the multi-bond molecularity fix has regressed"
        )
        assert obs["monomers"] <= 110.0


class TestSingleBondMolecularityStillBlocked:
    """Single-bond dissociation inside a cycle must stay blocked (#54/#55)."""

    def test_single_bond_ring_break_blocked(self, data_dir: Path) -> None:
        # ``ring_singlebond.xml`` seeds 100 copies of the size-2 ring
        #   L(r!1,r!2).R(l!2,l!3).L(r!3,r!4).R(l!4,l!1)
        # with only a two-product dissociation rule ``L(r!1).R(l!1) -> L(r) + R(l)``.
        # Breaking a single L-R bond inside the ring leaves the partners
        # connected through the rest of the cycle, so the products do not
        # separate: the network generator drops the reaction and the ODE keeps
        # Ring2 == 100. The fix must preserve this (zero-variance across seeds).
        xml = data_dir / "nfsim" / "ring_singlebond.xml"
        for seed in (1, 2, 3):
            obs = _final_observables(xml, seed=seed, t_end=5.0)
            assert obs["Ring2"] == pytest.approx(100.0), (
                f"single-bond ring break fired (Ring2={obs['Ring2']}, seed={seed}); "
                "molecularity enforcement has regressed (#54/#55)"
            )


class TestSpeciesObservableComplexTracking:
    """Species observables count correctly even when same-complex binding is
    allowed, i.e. ``block_same_complex_binding=False`` (#57, phenomenon 2)."""

    def test_species_count_correct_with_bscb_off(self, data_dir: Path) -> None:
        # ``ring2_homodimer.xml`` declares the Species observable ``dimers``
        # (the two-bond ring). Species observables are tallied by iterating
        # complexes, which requires complex tracking. Before the fix, turning
        # off same-complex binding also disabled complex tracking, so ``dimers``
        # reported a wildly inflated count (thousands, far above the 197 rings
        # that physically exist) even though the molecules were fine. The count
        # must now be physical and track the binding-blocked mode (this model
        # has no same-complex binding, so the two modes should agree).
        xml = data_dir / "nfsim" / "ring2_homodimer.xml"
        on = _final_observables(xml, seed=1, t_end=200_000.0, block_same_complex_binding=True)
        off = _final_observables(xml, seed=1, t_end=200_000.0, block_same_complex_binding=False)

        assert on["Mtot"] == pytest.approx(394.0)
        assert off["Mtot"] == pytest.approx(394.0)
        # 197 rings seeded; the count can never physically exceed that. The
        # decisive contrast with the bug is <=197 vs the thousands it reported.
        assert off["dimers"] <= 197.0, (
            f"Species observable inflated under bscb=False (dimers={off['dimers']}); "
            "complex tracking for Species observables has regressed"
        )
        # And it should match the binding-blocked count within stochastic scatter.
        assert off["dimers"] == pytest.approx(on["dimers"], abs=20)
