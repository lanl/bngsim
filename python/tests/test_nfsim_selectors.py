"""Regression tests for NFsim reaction-rule selectors (internal#60).

Before the vendored NFsim was bumped to absorb selector enforcement
(base_commit 43f635dd, RuleWorld/nfsim PR #23), a selector-bearing BNGL ran
to a *silently incorrect* trajectory through the NFsim backend because the
vendored source had no selector code at all. These tests pin the post-bump
contract with an analytical oracle so a future stale-vendor refresh that
drops selector handling fails loudly instead of going silently wrong.

Oracle (from RuleWorld/nfsim#63):

    include_reactants(1, A(p~P))   # only phosphorylated A may bind B
    k_phos = 0                     # nothing ever gets phosphorylated
    => AB_total MUST stay exactly 0 if the selector is enforced
    => AB_total grows (A+B bind freely) if the selector is ignored

The ungated control (no selector) binds freely, proving the gated == 0 is the
selector doing its job rather than the model being trivially unable to bind.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest


def _has_nfsim() -> bool:
    return getattr(bngsim, "HAS_NFSIM", False)


pytestmark = pytest.mark.skipif(
    not _has_nfsim(),
    reason="bngsim compiled without NFsim support",
)


def _nfsim_data(data_dir: Path) -> Path:
    return data_dir / "nfsim"


def _run_ab_total(
    xml_path: Path, *, t_end: float = 20.0, n_points: int = 21, seed: int = 7
) -> np.ndarray:
    """Run an NFsim XML and return the AB_total observable trajectory."""
    with bngsim.NfsimSession(str(xml_path)) as nf:
        nf.initialize(seed=seed)
        result = nf.simulate(0.0, t_end, n_points=n_points)
    names = list(result.observable_names)
    obs = np.asarray(result.observables)
    return obs[:, names.index("AB_total")]


class TestReactantSelectors:
    """include_/exclude_reactants must gate the binding rule."""

    @pytest.mark.parametrize(
        "fixture",
        ["include_reactants_gate", "exclude_reactants_gate"],
    )
    def test_gated_binding_stays_zero(self, data_dir: Path, fixture: str):
        """With k_phos=0 no A qualifies, so an enforced selector keeps AB_total==0.

        A regression that drops selector enforcement lets A(s,p~U) bind B(s)
        freely and AB_total grows away from 0 — this assertion catches it.
        """
        ab = _run_ab_total(_nfsim_data(data_dir) / f"{fixture}.xml")
        assert np.all(ab == 0.0), (
            f"{fixture}: selector not enforced — AB_total reached {ab.max()} "
            f"(expected 0 for all time). Vendored NFsim may have lost selector "
            f"handling (see internal#60 / RuleWorld/nfsim#63)."
        )

    def test_ungated_control_binds(self, data_dir: Path):
        """Same kinetics without the selector: A binds B freely, AB_total > 0.

        This is the control proving the gated cases' AB_total==0 is the
        selector working, not the model being incapable of binding.
        """
        ab = _run_ab_total(_nfsim_data(data_dir) / "include_reactants_ungated.xml")
        assert ab.max() > 0.0, (
            "ungated control produced no binding — fixture is degenerate, so "
            "the gated == 0 oracle would be meaningless."
        )


class TestSelectorSessionLifetime:
    """Two selector-bearing sessions in one process must not double-free.

    The selector cleanup in TransformationSet::~TransformationSet deleted the
    readPattern-generated TemplateMolecules, which MoleculeType already frees on
    System teardown (system.cpp:169). That double-free crashed (SIGABRT/SIGSEGV)
    intermittently on a single selector session and deterministically (12/12)
    once two selector-bearing sessions shared a process — which is exactly what
    PyBNF fitting does. Fixed in the vendored NFsim by not deleting them here.

    A regression re-introducing the delete crashes the interpreter on the second
    iteration, failing this test (and the suite) loudly. See internal#60.
    """

    def test_two_selector_sessions_same_process(self, data_dir: Path):
        nfsim = _nfsim_data(data_dir)
        for fixture in ("include_reactants_gate", "exclude_reactants_gate"):
            ab = _run_ab_total(nfsim / f"{fixture}.xml")
            assert np.all(ab == 0.0)


class TestProductSelectors:
    """include_/exclude_products are not yet enforced; the vendor hard-aborts.

    internal#60 item 2 decided to keep the current behavior and pin it.
    The vendored NFsim (NFinput.cpp:1793) refuses to load a products-side
    selector rather than running silently-incorrect. This test locks that in:
    a silent change (warning-only, or real enforcement) must update this test
    deliberately rather than slip through.
    """

    def test_products_selector_hard_aborts_on_load(self, data_dir: Path):
        xml = _nfsim_data(data_dir) / "include_products_warn.xml"
        with pytest.raises(bngsim.SimulationError) as exc:
            session = bngsim.NfsimSession(str(xml))
            session.initialize(seed=1)
        assert "not yet enforced in NFsim" in str(exc.value), (
            "products-side selector contract changed: expected a hard load "
            "abort citing 'not yet enforced in NFsim' (internal#60 item 2)."
        )
