"""Tests for the capability introspection surface (#13).

Covers:

- ``bngsim.HAS_LIBSBML`` and ``bngsim.HAS_ANTIMONY`` simple bool flags
  (matching the existing ``HAS_NFSIM`` / ``HAS_RULEMONKEY`` pattern).
- ``bngsim.capabilities()`` aggregator: schema, stable feature names,
  consistency with module-level flags, and ``missing`` explanations
  that distinguish compiled-backend gaps from missing optional Python
  dependencies.
"""

from __future__ import annotations

import bngsim
import pytest

EXPECTED_FEATURE_KEYS = frozenset(
    {
        "nfsim",
        "rulemonkey",
        "klu",
        "libsbml",
        "antimony",
        "vivarium",
        "sbml_import",
        "sbml_ssa",
        "sbml_psa",
        "antimony_import",
        "codegen",
        "output_sensitivities",
    }
)


class TestModuleFlags:
    def test_has_libsbml_is_bool(self):
        assert isinstance(bngsim.HAS_LIBSBML, bool)

    def test_has_antimony_is_bool(self):
        assert isinstance(bngsim.HAS_ANTIMONY, bool)

    def test_has_vivarium_is_bool(self):
        assert isinstance(bngsim.HAS_VIVARIUM, bool)

    def test_has_nfsim_still_present(self):
        assert isinstance(bngsim.HAS_NFSIM, bool)

    def test_has_rulemonkey_still_present(self):
        assert isinstance(bngsim.HAS_RULEMONKEY, bool)

    def test_has_klu_is_bool(self):
        assert isinstance(bngsim.HAS_KLU, bool)


class TestCapabilitiesSchema:
    def test_top_level_keys(self):
        caps = bngsim.capabilities()
        assert set(caps) == {"version", "features", "missing"}

    def test_version_matches_module(self):
        assert bngsim.capabilities()["version"] == bngsim.__version__

    def test_features_keys_are_stable(self):
        caps = bngsim.capabilities()
        assert set(caps["features"]) == EXPECTED_FEATURE_KEYS

    def test_features_all_bool(self):
        caps = bngsim.capabilities()
        for name, value in caps["features"].items():
            assert isinstance(value, bool), f"features[{name!r}] not bool: {value!r}"

    def test_missing_values_are_strings(self):
        caps = bngsim.capabilities()
        for name, msg in caps["missing"].items():
            assert isinstance(msg, str) and msg, f"missing[{name!r}] empty/non-str"


class TestCapabilitiesConsistency:
    """Aggregator output must agree with the module-level flags."""

    def test_nfsim_matches_module_flag(self):
        assert bngsim.capabilities()["features"]["nfsim"] == bngsim.HAS_NFSIM

    def test_rulemonkey_matches_module_flag(self):
        assert bngsim.capabilities()["features"]["rulemonkey"] == bngsim.HAS_RULEMONKEY

    def test_klu_matches_module_flag(self):
        assert bngsim.capabilities()["features"]["klu"] == bngsim.HAS_KLU

    def test_libsbml_matches_module_flag(self):
        assert bngsim.capabilities()["features"]["libsbml"] == bngsim.HAS_LIBSBML

    def test_antimony_matches_module_flag(self):
        assert bngsim.capabilities()["features"]["antimony"] == bngsim.HAS_ANTIMONY

    def test_vivarium_matches_module_flag(self):
        assert bngsim.capabilities()["features"]["vivarium"] == bngsim.HAS_VIVARIUM

    def test_sbml_features_track_libsbml(self):
        caps = bngsim.capabilities()
        assert caps["features"]["sbml_import"] == bngsim.HAS_LIBSBML
        assert caps["features"]["sbml_ssa"] == bngsim.HAS_LIBSBML
        assert caps["features"]["sbml_psa"] == bngsim.HAS_LIBSBML

    def test_antimony_import_requires_both(self):
        caps = bngsim.capabilities()
        assert caps["features"]["antimony_import"] == (bngsim.HAS_ANTIMONY and bngsim.HAS_LIBSBML)

    def test_codegen_always_present(self):
        assert bngsim.capabilities()["features"]["codegen"] is True

    def test_output_sensitivities_always_present(self):
        # The output-sensitivity tensor handshake (GH #207) is unconditional,
        # like codegen — fitting frontends gate their gradient path on it.
        assert bngsim.capabilities()["features"]["output_sensitivities"] is True

    def test_missing_subset_of_unavailable_features(self):
        caps = bngsim.capabilities()
        unavailable = {n for n, v in caps["features"].items() if not v}
        # codegen and output_sensitivities are always True, so they should
        # never appear in `missing`.
        assert "codegen" not in caps["missing"]
        assert "output_sensitivities" not in caps["missing"]
        # Every key in `missing` corresponds to an unavailable feature.
        assert set(caps["missing"]).issubset(unavailable)

    def test_no_missing_entries_when_everything_available(self):
        caps = bngsim.capabilities()
        if all(caps["features"].values()):
            assert caps["missing"] == {}


class TestCapabilitiesMissingExplanations:
    """When a feature is unavailable, ``missing`` must distinguish a
    compiled-backend gap (rebuild flag) from a missing optional Python
    dependency (pip install)."""

    def test_missing_compiled_backend_message(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_NFSIM", False)
        monkeypatch.setattr(bngsim, "HAS_RULEMONKEY", False)
        caps = bngsim.capabilities()
        # Message must name the backend, the vendored source path, and
        # the CMake flag — and must convey that the source default is to
        # build it (so `=OFF` is the real signal, not `=ON`).
        for backend, flag, vendor_path in [
            ("nfsim", "BNGSIM_BUILD_NFSIM", "third_party/nfsim/"),
            ("rulemonkey", "BNGSIM_BUILD_RULEMONKEY", "third_party/rulemonkey/"),
        ]:
            msg = caps["missing"][backend]
            assert "not present in this install" in msg
            assert vendor_path in msg
            assert f"-D{flag}=OFF" in msg
            assert "wheel" in msg

    def test_missing_klu_explanation(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_KLU", False)
        caps = bngsim.capabilities()
        msg = caps["missing"]["klu"]
        # Names the cause (no KLU / dense O(N³)) and a concrete rebuild path.
        assert "KLU" in msg
        assert "dense" in msg
        assert "suitesparse" in msg.lower()
        assert caps["features"]["klu"] is False

    def test_missing_libsbml_explanation(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_LIBSBML", False)
        caps = bngsim.capabilities()
        for key in ("libsbml", "sbml_import", "sbml_ssa", "sbml_psa"):
            assert "python-libsbml" in caps["missing"][key]
            assert "optional dependency" in caps["missing"][key]
        assert caps["features"]["libsbml"] is False
        assert caps["features"]["sbml_ssa"] is False

    def test_missing_antimony_explanation(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_ANTIMONY", False)
        caps = bngsim.capabilities()
        assert "antimony" in caps["missing"]["antimony"]
        assert "optional dependency" in caps["missing"]["antimony"]

    def test_missing_vivarium_explanation(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_VIVARIUM", False)
        caps = bngsim.capabilities()
        assert "vivarium-core" in caps["missing"]["vivarium"]
        assert "optional dependency" in caps["missing"]["vivarium"]
        assert caps["features"]["vivarium"] is False

    def test_antimony_import_needs_both(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_ANTIMONY", False)
        monkeypatch.setattr(bngsim, "HAS_LIBSBML", True)
        caps = bngsim.capabilities()
        assert caps["features"]["antimony_import"] is False
        assert "antimony" in caps["missing"]["antimony_import"]
        assert "python-libsbml" not in caps["missing"]["antimony_import"]

    def test_antimony_import_needs_libsbml(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_ANTIMONY", True)
        monkeypatch.setattr(bngsim, "HAS_LIBSBML", False)
        caps = bngsim.capabilities()
        assert caps["features"]["antimony_import"] is False
        # antimony_import gets a libsbml-only message; antimony itself
        # is reported as available.
        assert "python-libsbml" in caps["missing"]["antimony_import"]
        assert "antimony" not in caps["missing"]["antimony_import"].split()[1:]

    def test_antimony_import_needs_both_when_both_missing(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_ANTIMONY", False)
        monkeypatch.setattr(bngsim, "HAS_LIBSBML", False)
        caps = bngsim.capabilities()
        msg = caps["missing"]["antimony_import"]
        assert "antimony" in msg
        assert "python-libsbml" in msg

    def test_full_install_no_missing(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_NFSIM", True)
        monkeypatch.setattr(bngsim, "HAS_RULEMONKEY", True)
        monkeypatch.setattr(bngsim, "HAS_KLU", True)
        monkeypatch.setattr(bngsim, "HAS_LIBSBML", True)
        monkeypatch.setattr(bngsim, "HAS_ANTIMONY", True)
        monkeypatch.setattr(bngsim, "HAS_VIVARIUM", True)
        caps = bngsim.capabilities()
        assert caps["missing"] == {}
        assert all(caps["features"].values())


class TestPublicSurface:
    """The new API must be reachable via the public namespace and `__all__`."""

    @pytest.mark.parametrize(
        "name", ["HAS_LIBSBML", "HAS_ANTIMONY", "HAS_VIVARIUM", "HAS_KLU", "capabilities"]
    )
    def test_in_all(self, name):
        assert name in bngsim.__all__

    @pytest.mark.parametrize(
        "name", ["HAS_LIBSBML", "HAS_ANTIMONY", "HAS_VIVARIUM", "HAS_KLU", "capabilities"]
    )
    def test_attribute_present(self, name):
        assert hasattr(bngsim, name)
