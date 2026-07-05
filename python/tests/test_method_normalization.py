"""Tests for network-free method normalization and dispatch."""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._simulator import (
    _AVAILABLE_NF_METHODS,
    _NF_METHOD_ALIASES,
    _UNAVAILABLE_NF_METHODS,
    normalize_method,
)


def _has_nfsim() -> bool:
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return bool(HAS_NFSIM)
    except ImportError:
        return False


def _has_rulemonkey() -> bool:
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


@pytest.fixture
def dummy_model(simple_decay_net: Path) -> bngsim.Model:
    """Minimal model for Simulator construction on XML-backed NF paths."""
    return bngsim.Model.from_net(str(simple_decay_net))


class TestNormalizeMethod:
    def test_ode_passthrough(self):
        canonical, dispatch = normalize_method("ode")
        assert canonical == "ode"
        assert dispatch == "ode"

    def test_ssa_passthrough(self):
        canonical, dispatch = normalize_method("ssa")
        assert canonical == "ssa"
        assert dispatch == "ssa"

    def test_psa_passthrough(self):
        canonical, dispatch = normalize_method("psa")
        assert canonical == "psa"
        assert dispatch == "psa"

    def test_nf_to_nf_reject(self):
        canonical, dispatch = normalize_method("nf")
        assert canonical == "nf_reject"
        assert dispatch == "nfsim"

    def test_nfsim_alias(self):
        canonical, dispatch = normalize_method("nfsim")
        assert canonical == "nf_reject"
        assert dispatch == "nfsim"

    def test_case_insensitive(self):
        canonical, dispatch = normalize_method(" NF ")
        assert canonical == "nf_reject"
        assert dispatch == "nfsim"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            normalize_method("bogus")

    def test_nf_fixed_is_recognized_but_unavailable(self):
        with pytest.raises(ValueError, match="nf_fixed"):
            normalize_method("nf_fixed")

    def test_dynstoc_alias_is_unavailable(self):
        with pytest.raises(ValueError, match="nf_fixed"):
            normalize_method("dynstoc")


class TestRuleMonkeyNetworkFreeMethods:
    def test_nf_exact_dispatches_to_rulemonkey_when_available(self):
        if not _has_rulemonkey():
            pytest.skip("RuleMonkey not compiled in")
        canonical, dispatch = normalize_method("nf_exact")
        assert canonical == "nf_exact"
        assert dispatch == "rulemonkey"

    def test_rulemonkey_alias_dispatches_when_available(self):
        if not _has_rulemonkey():
            pytest.skip("RuleMonkey not compiled in")
        canonical, dispatch = normalize_method("rulemonkey")
        assert canonical == "nf_exact"
        assert dispatch == "rulemonkey"

    def test_rm_alias_dispatches_when_available(self):
        if not _has_rulemonkey():
            pytest.skip("RuleMonkey not compiled in")
        canonical, dispatch = normalize_method("rm")
        assert canonical == "nf_exact"
        assert dispatch == "rulemonkey"

    def test_nf_exact_is_unavailable_without_rulemonkey(self):
        if _has_rulemonkey():
            pytest.skip("RuleMonkey compiled in")
        with pytest.raises(
            ValueError,
            match="nf_exact.*unavailable in this environment",
        ):
            normalize_method("nf_exact")

    def test_rulemonkey_alias_is_unavailable_without_rulemonkey(self):
        if _has_rulemonkey():
            pytest.skip("RuleMonkey compiled in")
        with pytest.raises(ValueError, match="nf_exact"):
            normalize_method("rulemonkey")

    def test_simulator_nf_exact_raises_without_rulemonkey(self, dummy_model, nfsim_xml):
        if _has_rulemonkey():
            pytest.skip("RuleMonkey compiled in")
        with pytest.raises(ValueError, match="nf_exact"):
            bngsim.Simulator(
                dummy_model,
                method="nf_exact",
                xml_path=str(nfsim_xml),
            )


class TestSimulatorMethodDispatch:
    def test_ode_dispatch(self, simple_decay_net):
        model = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(model, method="ode")
        assert sim.method == "ode"
        assert sim.requested_method == "ode"

    def test_ssa_dispatch(self, simple_decay_net):
        model = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(model, method="ssa")
        assert sim.method == "ssa"
        assert sim.requested_method == "ssa"

    @pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
    def test_nfsim_requested_method_preserved(self, dummy_model, nfsim_xml):
        sim = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_xml),
        )
        assert sim.method == "nfsim"
        assert sim.requested_method == "nfsim"

    @pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
    def test_nf_requested_method_preserved(self, dummy_model, nfsim_xml):
        sim = bngsim.Simulator(
            dummy_model,
            method="nf",
            xml_path=str(nfsim_xml),
        )
        assert sim.method == "nfsim"
        assert sim.requested_method == "nf"

    @pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
    def test_nf_reject_requested_method_preserved(self, dummy_model, nfsim_xml):
        sim = bngsim.Simulator(
            dummy_model,
            method="nf_reject",
            xml_path=str(nfsim_xml),
        )
        assert sim.method == "nfsim"
        assert sim.requested_method == "nf_reject"

    @pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
    def test_nf_requires_xml_path(self, simple_decay_net):
        model = bngsim.Model.from_net(str(simple_decay_net))
        with pytest.raises(ValueError, match="xml_path"):
            bngsim.Simulator(model, method="nf")

    @pytest.mark.skipif(not _has_rulemonkey(), reason="RuleMonkey not compiled in")
    def test_rulemonkey_requested_method_preserved(self, dummy_model, nfsim_xml):
        sim = bngsim.Simulator(
            dummy_model,
            method="rm",
            xml_path=str(nfsim_xml),
        )
        assert sim.method == "rulemonkey"
        assert sim.requested_method == "rm"


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
class TestNfParity:
    def test_nf_vs_nf_reject_identical_output(self, dummy_model, nfsim_xml):
        sim_nf = bngsim.Simulator(
            dummy_model,
            method="nf",
            xml_path=str(nfsim_xml),
        )
        sim_reject = bngsim.Simulator(
            dummy_model,
            method="nf_reject",
            xml_path=str(nfsim_xml),
        )

        r1 = sim_nf.run(t_span=(0.0, 1.0), n_points=11, seed=42)
        r2 = sim_reject.run(t_span=(0.0, 1.0), n_points=11, seed=42)

        np.testing.assert_array_equal(np.asarray(r1.observables), np.asarray(r2.observables))
        np.testing.assert_array_equal(r1.time, r2.time)
        assert r1.observable_names == r2.observable_names

    def test_nfsim_alias_vs_nf_reject_identical(self, dummy_model, nfsim_xml):
        sim_alias = bngsim.Simulator(
            dummy_model,
            method="nfsim",
            xml_path=str(nfsim_xml),
        )
        sim_reject = bngsim.Simulator(
            dummy_model,
            method="nf_reject",
            xml_path=str(nfsim_xml),
        )

        r1 = sim_alias.run(t_span=(0.0, 1.0), n_points=11, seed=42)
        r2 = sim_reject.run(t_span=(0.0, 1.0), n_points=11, seed=42)

        np.testing.assert_array_equal(np.asarray(r1.observables), np.asarray(r2.observables))


class TestAliasMapIntegrity:
    def test_all_aliases_resolve_to_canonical(self):
        canonical_set = {"nf_reject", "nf_exact", "nf_fixed"}
        for _alias, target in _NF_METHOD_ALIASES.items():
            assert target in canonical_set

    def test_available_and_unavailable_disjoint(self):
        overlap = _AVAILABLE_NF_METHODS & set(_UNAVAILABLE_NF_METHODS.keys())
        assert not overlap

    def test_all_canonical_covered(self):
        canonical_set = {"nf_reject", "nf_exact", "nf_fixed"}
        covered = _AVAILABLE_NF_METHODS | set(_UNAVAILABLE_NF_METHODS.keys())
        assert canonical_set == covered
