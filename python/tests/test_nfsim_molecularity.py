"""Product-molecularity regression tests for the vendored NFsim engine.

These guard the multi-bond ring-opening fix (internal#57) without
re-breaking the single-bond cyclic-dissociation behavior that the same
``checkMolecularity`` path is responsible for (internal#54/#55).

``checkMolecularity`` enforces product-side molecularity: ``A.B -> A + B`` may
only fire when the deletion actually separates the match into different
connected components. The test considers *all* the bonds a rule deletes at once
instead of one bond in isolation, and the bonds it adds. Like BioNetGen, it is
made for every bond deleted between two product patterns, whatever the number
of reactant patterns and whatever ``block_same_complex_binding`` says; a bond
deleted inside one product pattern claims no separation.
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
    reason="NFsim not compiled in",
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


BSCB = pytest.mark.parametrize("bscb", [True, False], ids=["bscb_on", "bscb_off"])


class TestProductMolecularityWhateverTheFlag:
    """The product-side test holds with ``block_same_complex_binding`` off.

    The flag governs which molecules a bimolecular rule may bind. NFsim also
    gated the product-side ``+`` test on it, so with the flag off
    ``L(r!1).R(l!1) -> L(r) + R(l)`` opened every ring.
    """

    def test_single_bond_ring_break_blocked_with_bscb_off(self, data_dir: Path) -> None:
        # Same fixture as TestSingleBondMolecularityStillBlocked: the network
        # keeps Ring2 == 100. With the flag off main broke all 100 rings.
        xml = data_dir / "nfsim" / "ring_singlebond.xml"
        for seed in (1, 2, 3):
            obs = _final_observables(xml, seed=seed, t_end=5.0, block_same_complex_binding=False)
            assert obs["Ring2"] == pytest.approx(100.0), (
                f"single-bond ring break fired with block_same_complex_binding=False "
                f"(Ring2={obs['Ring2']}, seed={seed})"
            )

    @BSCB
    def test_long_way_round_blocked(self, data_dir: Path, bscb: bool) -> None:
        # 20 ten-member rings L.R x 5 with only L(r!1).R(l!1) -> L(r) + R(l):
        # a broken bond's ends stay connected the long way round, eight more
        # molecules, so BNG2.pl generates no reaction and Lfree stays 0. The
        # search that shows it must go past its first budget of 8 molecules.
        # Main opened the rings with the flag off. Control: bscb_on passes on
        # main.
        xml = data_dir / "nfsim" / "ring_large_singlebond.xml"
        for seed in (1, 2):
            obs = _final_observables(xml, seed=seed, t_end=5.0, block_same_complex_binding=bscb)
            assert obs["Lfree"] == pytest.approx(0.0), (
                f"ten-member ring opened (Lfree={obs['Lfree']}, seed={seed}, bscb={bscb})"
            )

    @BSCB
    def test_conserved_total_with_same_complex_binding(self, data_dir: Path, bscb: bool) -> None:
        # 30 A(l,r) bind into chains (rings, with the flag off) and break again.
        # Nothing makes or removes an A, so Atot == 30 at every sample. With the
        # flag off, a binding whose two reactants sat in one complex beyond each
        # other's traversal limit listed their shared neighbours twice and added
        # them to the Molecules observables twice: main read 84-98. Control:
        # bscb_on passes on main (no two reactants share a complex).
        xml = data_dir / "nfsim" / "ring_close_open.xml"
        for seed in (1, 2, 3):
            with NfsimSession(str(xml), block_same_complex_binding=bscb) as nf:
                nf.initialize(seed=seed)
                res = nf.simulate(0.0, 40.0, 81)
                col = list(res.observable_names).index("Atot")
                atot = [row[col] for row in res.observables]
            assert min(atot) == pytest.approx(30.0) and max(atot) == pytest.approx(30.0), (
                f"conserved Atot left 30 (range {min(atot)}..{max(atot)}, "
                f"seed={seed}, bscb={bscb})"
            )


class TestProductMolecularityBimolecular:
    """A bimolecular rule's product-side ``+`` is tested too."""

    @BSCB
    def test_bimolecular_split_on_a_ring_blocked(self, data_dir: Path, bscb: bool) -> None:
        # L(r!1,c).R(l!1) + C(x) -> L(r,c!2).C(x!2) + R(l) on 100 rings: L and R
        # stay connected through the ring, so BNG2.pl generates no reaction and
        # Cfree stays 100. Main fired it and bound every C (Cfree 0).
        xml = data_dir / "nfsim" / "ring_bimolecular_split.xml"
        for seed in (1, 2):
            obs = _final_observables(xml, seed=seed, t_end=5.0, block_same_complex_binding=bscb)
            assert obs["Cfree"] == pytest.approx(100.0), (
                f"bimolecular split fired on a ring bond (Cfree={obs['Cfree']}, "
                f"seed={seed}, bscb={bscb})"
            )
            assert obs["Ring2"] == pytest.approx(100.0)

    def test_split_rejoined_by_the_added_bond_blocked(self, data_dir: Path) -> None:
        # With the flag off, A(b!1).B(a!1) + C(x) -> A(b!2).C(x!2) + B(a) may
        # match the C bonded to B in the same complex. The bond the rule adds
        # then joins A back to B through C, so the products do not separate and
        # B must stay bound (Bfree 0). Main freed it (Bfree 1).
        xml = data_dir / "nfsim" / "split_rejoined_by_added_bond.xml"
        for seed in (1, 2, 3):
            obs = _final_observables(xml, seed=seed, t_end=20.0, block_same_complex_binding=False)
            assert obs["Bfree"] == pytest.approx(0.0), (
                f"split rejoined by the added bond fired (Bfree={obs['Bfree']}, seed={seed})"
            )

    def test_split_rejoined_control_bscb_on(self, data_dir: Path) -> None:
        """Control: with the flag on, the only C is in the same complex, so the
        rule never matches (main passes too)."""
        xml = data_dir / "nfsim" / "split_rejoined_by_added_bond.xml"
        obs = _final_observables(xml, seed=1, t_end=20.0, block_same_complex_binding=True)
        assert obs["Bfree"] == pytest.approx(0.0)


class TestBondInsideOneProductPattern:
    """A bond deleted inside one product pattern claims no separation."""

    @BSCB
    def test_intra_pattern_bond_does_not_block(self, data_dir: Path, bscb: bool) -> None:
        # A(x!1,y!2,z!3).B(x!1,y!2).C(a!3) -> A(x,y!2,z).B(x,y!2) + C(a), k = 1:
        # the x bond is deleted inside the first product pattern (A and B stay
        # bonded through y) and only the z bond separates C. BNG2.pl fires it at
        # rate k, so Cfree(1) = 100 (1 - e^-1) = 63.2 (binomial sd 4.8; sd 2.2 for
        # the mean of 5 seeds). Main tested the x bond too, found A and B still
        # connected, and never fired with the flag on (Cfree 0). Control:
        # bscb_off passes on main, which made no product-side test with the flag
        # off; it guards against a fix that tests every deleted bond.
        xml = data_dir / "nfsim" / "intra_pattern_bond_split.xml"
        cfree = [
            _final_observables(xml, seed=seed, t_end=1.0, block_same_complex_binding=bscb)["Cfree"]
            for seed in range(1, 6)
        ]
        mean = sum(cfree) / len(cfree)
        assert mean == pytest.approx(63.2, abs=11.0), (
            f"Cfree mean {mean} (seeds {cfree}, bscb={bscb})"
        )


class TestEveryPairOfProductPatterns:
    """No two molecules the rule acts on may end up connected across product
    patterns, and a molecule the rule deletes takes its bonds with it."""

    @BSCB
    def test_three_products_on_a_ring_blocked(self, data_dir: Path, bscb: bool) -> None:
        # A(b!1,c!2).B(a!1).C(a!2) -> A(b,c) + B(a) + C(a) on 100 rings where B
        # and C are also bonded: the products never separate, so BNG2.pl reacts
        # only the 100 linear complexes and Ring stays 100. Testing only the ends
        # of each deleted bond (A-B, A-C) let main break every ring.
        xml = data_dir / "nfsim" / "three_product_ring.xml"
        for seed in (1, 2):
            obs = _final_observables(xml, seed=seed, t_end=5.0, block_same_complex_binding=bscb)
            assert obs["Ring"] == pytest.approx(100.0), (
                f"three-product split fired on a ring (Ring={obs['Ring']}, "
                f"seed={seed}, bscb={bscb})"
            )
            assert obs["Afree"] >= 95.0  # the linear complexes do react

    @BSCB
    def test_deleted_molecule_takes_its_bonds(self, data_dir: Path, bscb: bool) -> None:
        # A(b!1,x!2).B(a!1).X(a!2) -> A(b,x) + B(a) DeleteMolecules on 100 rings
        # A.B.X: BioNetGen writes no DeleteBond for X's bonds, which go with X,
        # and once X is gone A and B are apart, so the network fires the rule at
        # k = 1: Afree(1) = 63.2 (sd 2.2 for the mean of 5 seeds). Main never
        # fired it with the flag on (it tested the A-B bond and found A and B
        # connected through X). Control: bscb_off passes on main, which made no
        # test with the flag off.
        xml = data_dir / "nfsim" / "deleted_bridge_ring.xml"
        afree = [
            _final_observables(xml, seed=seed, t_end=1.0, block_same_complex_binding=bscb)["Afree"]
            for seed in range(1, 6)
        ]
        mean = sum(afree) / len(afree)
        assert mean == pytest.approx(63.2, abs=11.0), (
            f"Afree mean {mean} (seeds {afree}, bscb={bscb})"
        )

    def test_deleted_molecule_bimolecular_control(self, data_dir: Path) -> None:
        """Control: main fires this default-path rule, and BNG2.pl's network
        agrees. A search that walked through the deleted X refused it (Afree 0).
        Rings and C(y~0) fall together, 100 / (1 + 100 k t): Afree(5) = 83.3."""
        xml = data_dir / "nfsim" / "deleted_bridge_ring_bimolecular.xml"
        for seed in (1, 2):
            obs = _final_observables(xml, seed=seed, t_end=5.0)
            assert obs["Afree"] == pytest.approx(83.3, abs=12.0), (
                f"Afree {obs['Afree']} (seed={seed})"
            )

    def test_created_molecule_on_a_bridge_blocked(self, data_dir: Path) -> None:
        # With the flag off, A(b!1,c).B(a!1) + D(x) -> A(b,c!3).X(a!3,d!4).D(x!4)
        # + B(a) may match the D bonded to B in the same complex. D stays bonded
        # to B, so the products do not separate and the rule must not fire
        # (Xtot 0). Main fired it (Xtot 1).
        xml = data_dir / "nfsim" / "created_bridge_same_complex.xml"
        for seed in (1, 2, 3):
            obs = _final_observables(xml, seed=seed, t_end=20.0, block_same_complex_binding=False)
            assert obs["Xtot"] == pytest.approx(0.0), (
                f"split onto a created molecule fired (Xtot={obs['Xtot']}, seed={seed})"
            )

    def test_untouched_reactant_in_the_same_complex_blocked(self, data_dir: Path) -> None:
        # With the flag off, A(b!1).B(a!1) + C(x) -> A(b) + B(a) + C(x) may match
        # the C bonded to B. The rule never touches C, but its product C stays
        # bonded to product B, so the rule must not fire (Afree 0). Main fired it
        # (Afree 1).
        xml = data_dir / "nfsim" / "untouched_reactant_same_complex.xml"
        for seed in (1, 2, 3):
            obs = _final_observables(xml, seed=seed, t_end=20.0, block_same_complex_binding=False)
            assert obs["Afree"] == pytest.approx(0.0), (
                f"split beside an untouched reactant fired (Afree={obs['Afree']}, seed={seed})"
            )
