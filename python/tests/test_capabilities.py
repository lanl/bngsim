"""Tests for the capability introspection surface (#13, #431).

Covers:

- ``bngsim.HAS_LIBSBML`` and ``bngsim.HAS_ANTIMONY`` simple bool flags
  (matching the existing ``HAS_NFSIM`` / ``HAS_RULEMONKEY`` pattern).
- ``bngsim.capabilities()`` aggregator: schema, stable feature names,
  consistency with module-level flags, and ``missing`` explanations
  that distinguish compiled-backend gaps from missing optional Python
  dependencies.
- The **behaviour keys** and the **build identity** added by #431: the four
  keys that report what this build computes rather than what it was compiled
  with, and the ``build`` block that says which build it is. What those keys
  claim is measured in ``test_behaviour_capability_keys.py``; what is asserted
  here is the reporting contract around them.
"""

from __future__ import annotations

import bngsim
import pytest

EXPECTED_FEATURE_KEYS = frozenset(
    {
        "nfsim",
        "rulemonkey",
        "klu",
        "lapack_dense",
        "mir",
        "libsbml",
        "antimony",
        "vivarium",
        "bngl",
        "sbml_import",
        "sbml_ssa",
        "sbml_psa",
        "antimony_import",
        "codegen",
        "output_sensitivities",
        "effective_ic_sensitivity",
        # Behaviour keys (#431).
        "event_sensitivities",
        "cross_compartment_sensitivities",
        "per_species_atol",
        "tracking_atol",
    }
)

# The subset of the above that #431 added, with the probe each one reads. A
# consumer gates a gradient path on these, so every one of them must be
# published on every build — a key that is absent means "too old to have been
# asked", which is a different answer from False and must not be reachable here.
BEHAVIOUR_FEATURE_PROBES = {
    "event_sensitivities": "_event_sensitivities_available",
    "cross_compartment_sensitivities": "_cross_compartment_sensitivities_available",
    "per_species_atol": "_per_species_atol_available",
    "tracking_atol": "_tracking_atol_available",
}


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

    def test_has_lapack_dense_is_bool(self):
        # GH #269: the flag existed only as bngsim._bngsim_core.HAS_LAPACK_DENSE,
        # which is why test_lapack_dense_solver.py had to getattr into the
        # private extension module to gate on it.
        assert isinstance(bngsim.HAS_LAPACK_DENSE, bool)

    def test_has_lapack_dense_matches_extension_flag(self):
        from bngsim import _bngsim_core as _core

        assert bool(getattr(_core, "HAS_LAPACK_DENSE", False)) == bngsim.HAS_LAPACK_DENSE


class TestCapabilitiesSchema:
    def test_top_level_keys(self):
        caps = bngsim.capabilities()
        assert set(caps) == {"version", "features", "missing", "build"}

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

    def test_lapack_dense_matches_module_flag(self):
        assert bngsim.capabilities()["features"]["lapack_dense"] == bngsim.HAS_LAPACK_DENSE

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

    def test_every_unavailable_feature_is_explained(self):
        """The other direction, which nothing asserted: an unavailable feature
        must SAY why.

        ``mir`` was False on every default build and had no entry at all, so a
        caller doing the documented thing — read ``features[name]``, and on
        False print ``missing[name]`` — got a KeyError instead of an
        explanation. The rule is symmetric, so test it symmetrically.
        """
        caps = bngsim.capabilities()
        unavailable = {n for n, v in caps["features"].items() if not v}
        assert set(caps["missing"]) == unavailable

    def test_mir_off_is_explained_as_a_default_not_a_breakage(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_MIR", False)
        msg = bngsim.capabilities()["missing"]["mir"]
        # Names the CMake flag that turns it on, says it is off by default, and
        # says nothing is broken by its absence.
        assert "-DBNGSIM_ENABLE_MIR=ON" in msg
        assert "OFF by default" in msg
        assert "Nothing needs it" in msg

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

    def test_missing_lapack_dense_explanation(self, monkeypatch):
        monkeypatch.setattr(bngsim, "HAS_LAPACK_DENSE", False)
        caps = bngsim.capabilities()
        msg = caps["missing"]["lapack_dense"]
        # Names the env var it makes inert, a concrete way to get a backend, and
        # — the part that distinguishes it from klu — says results do not change.
        assert "BNGSIM_LAPACK_DENSE" in msg
        assert "lapack" in msg.lower()
        assert "correctness" in msg
        assert caps["features"]["lapack_dense"] is False

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
        # HAS_MIR is a COMPILED flag (the MIR micro-JIT is an OFF-by-default
        # prototype absent from shipped wheels — GH #78), so it reflects the local
        # build. This is a build-independent LOGIC test of capabilities() — "if
        # every optional feature were present, nothing is missing" — so pin it too;
        # otherwise the test spuriously fails on any default (non-MIR) build.
        monkeypatch.setattr(bngsim, "HAS_MIR", True)
        # Same for the BLAS dense backend (GH #269): a compiled flag that is
        # False on a Linux/Windows wheel and True on macOS, so pin it rather
        # than let the local build decide whether this logic test passes.
        monkeypatch.setattr(bngsim, "HAS_LAPACK_DENSE", True)
        # ...and BNGL (GH #162), for the same reason one step further out: it is
        # not a flag at all but a probe for an external Perl toolchain, so on a
        # machine without BNG2.pl this build-independent logic test would
        # otherwise fail on an environment fact it is not about.
        monkeypatch.setattr(bngsim, "_bngl_available", lambda: True)
        # ...and the four behaviour keys (#431), for the same reason again: one
        # of them reads an environment switch and two read the compiled core, so
        # on a hatched environment or an out-of-date extension this
        # build-independent logic test would otherwise fail on a fact about the
        # machine rather than about capabilities().
        for probe in BEHAVIOUR_FEATURE_PROBES.values():
            monkeypatch.setattr(bngsim, probe, lambda: True)
        caps = bngsim.capabilities()
        assert caps["missing"] == {}
        assert all(caps["features"].values())


class TestBehaviourFeatureKeys:
    """#431: keys that report what this build COMPUTES.

    The measurements that say whether each key is telling the truth live in
    ``test_behaviour_capability_keys.py``. These tests are about the reporting
    contract: the keys are always published, they are real booleans, and a
    ``False`` comes with an explanation a consumer can print.
    """

    @pytest.mark.parametrize("name", sorted(BEHAVIOUR_FEATURE_PROBES))
    def test_published_on_every_build(self, name):
        """Published, not merely truthy. An absent key means "too old to have
        been asked" — a third state a consumer has to handle differently — so
        the value of a behaviour key is that it is always there."""
        caps = bngsim.capabilities()
        assert name in caps["features"]
        assert isinstance(caps["features"][name], bool)

    @pytest.mark.parametrize("name", sorted(BEHAVIOUR_FEATURE_PROBES))
    def test_true_on_this_build(self, name):
        """Every one of these is True on a current build with no A/B hatch set.

        This is the test that fails if a probe is silently rewired — for
        instance to a core binding that a later refactor renames. The behaviour
        would still be there and the key would start saying it is not, which is
        the safe direction to be wrong in but is still wrong.
        """
        assert bngsim.capabilities()["features"][name] is True

    @pytest.mark.parametrize("name, probe", sorted(BEHAVIOUR_FEATURE_PROBES.items()))
    def test_false_is_an_answer_with_a_reason(self, monkeypatch, name, probe):
        monkeypatch.setattr(bngsim, probe, lambda: False)
        caps = bngsim.capabilities()
        assert caps["features"][name] is False
        msg = caps["missing"][name]
        assert isinstance(msg, str) and len(msg) > 40
        # Every one of these names the issue it came from, so a reader can find
        # out what changed rather than only that something did.
        assert "GH #" in msg

    def test_event_sensitivities_reason_names_the_silent_failure(self, monkeypatch):
        monkeypatch.setattr(bngsim, "_event_sensitivities_available", lambda: False)
        msg = bngsim.capabilities()["missing"]["event_sensitivities"]
        # The point of the key is that the pre-fix behaviour is a wrong number
        # and not a refusal, so the message has to say that outright.
        assert "GH #144" in msg and "#146" in msg
        assert "wrong" in msg
        assert "rebuild_editable" in msg

    def test_cross_compartment_reason_names_the_environment_switch(self, monkeypatch):
        monkeypatch.setattr(bngsim, "_cross_compartment_sensitivities_available", lambda: False)
        msg = bngsim.capabilities()["missing"]["cross_compartment_sensitivities"]
        # The only way this one can be False, so the message says which variable
        # to unset rather than sending the reader to a rebuild.
        assert "BNGSIM_NO_FUNCTIONAL_SENS_RHS=1" in msg
        assert "rebuild_editable" not in msg
        # ...and it must not claim a wrong answer: the fallback is correct and
        # slow, which is a different thing to tell a user.
        assert "correct" in msg

    @pytest.mark.parametrize(
        "probe, member",
        [
            ("_event_sensitivities_available", "events_with_runtime_event_time_sens"),
            ("_per_species_atol_available", "atol_vec"),
            ("_tracking_atol_available", "atol_track_decades"),
        ],
    )
    def test_the_core_probes_read_the_compiled_extension(self, monkeypatch, probe, member):
        """Three of the four ask the loaded ``_bngsim_core`` for a binding its
        fix added, rather than trusting this Python file.

        That is the whole point: in a source checkout the extension is built
        separately and does not rebuild on import (GH #23), so the Python half
        can be current while the compiled half is not. Simulated here by hiding
        the binding — the probe must go False, not stay True because the Python
        layer looks new.
        """
        from bngsim import _bngsim_core as core

        owner = core.NetworkModel if member.startswith("events") else core.SolverOptions
        assert hasattr(owner, member), "probe target moved; update the probe and this test"
        monkeypatch.delattr(owner, member, raising=True)
        assert getattr(bngsim, probe)() is False


class TestBuildIdentity:
    """#431: ``caps["build"]`` — which build this is.

    Two installs can report the same ``version`` and be different builds,
    because bngsim bumps ``__version__`` at the start of a release cycle. The
    commit the extension was built from is the only thing in the public API
    that separates them, and it was readable only from a private module.
    """

    def test_shape(self):
        build = bngsim.capabilities()["build"]
        assert set(build) == {"commit", "stale"}
        assert build["commit"] is None or isinstance(build["commit"], str)
        assert isinstance(build["stale"], bool)

    def test_commit_matches_the_extension_stamp(self):
        from bngsim import _bngsim_core as core

        stamped = getattr(core, "__build_commit__", None)
        commit = bngsim.capabilities()["build"]["commit"]
        if stamped and stamped != "unknown":
            assert commit == stamped
        else:
            # An sdist build has no commit to name, and saying so is the honest
            # answer — not an empty string a consumer would log as a build id.
            assert commit is None

    def test_stale_agrees_with_the_import_time_guard(self):
        from bngsim import _build_provenance as prov

        assert bngsim.capabilities()["build"]["stale"] == prov.is_stale()

    def test_this_checkout_is_not_stale(self):
        """A green suite against a stale extension is a verdict about old code,
        so the preflight in conftest already refuses to run here. That makes
        this assertion free, and it pins the reporting to the same answer."""
        assert bngsim.capabilities()["build"]["stale"] is False

    def test_the_build_check_opt_out_is_honoured(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_NO_BUILD_CHECK", "1")
        build = bngsim.capabilities()["build"]
        assert build["stale"] is False
        # The commit still resolves: the opt-out disables the mtime comparison,
        # not the identity of the build.
        from bngsim import _bngsim_core as core

        stamped = getattr(core, "__build_commit__", None)
        if stamped and stamped != "unknown":
            assert build["commit"] == stamped

    def test_a_provenance_failure_does_not_take_capabilities_down(self, monkeypatch):
        """``capabilities()`` is called at setup by consumers that have not
        started their real work yet. A provenance read that raises must cost
        them the build block, not the call."""
        from bngsim import _build_provenance as prov

        def boom(*_a, **_k):
            raise OSError("no such thing")

        monkeypatch.setattr(prov, "gather", boom)
        caps = bngsim.capabilities()
        assert caps["build"] == {"commit": None, "stale": False}
        assert caps["features"]  # the rest of the report is unaffected


class TestPublicSurface:
    """The new API must be reachable via the public namespace and `__all__`."""

    @pytest.mark.parametrize(
        "name",
        [
            "HAS_LIBSBML",
            "HAS_ANTIMONY",
            "HAS_VIVARIUM",
            "HAS_KLU",
            "HAS_LAPACK_DENSE",
            "capabilities",
        ],
    )
    def test_in_all(self, name):
        assert name in bngsim.__all__

    @pytest.mark.parametrize(
        "name",
        [
            "HAS_LIBSBML",
            "HAS_ANTIMONY",
            "HAS_VIVARIUM",
            "HAS_KLU",
            "HAS_LAPACK_DENSE",
            "capabilities",
        ],
    )
    def test_attribute_present(self, name):
        assert hasattr(bngsim, name)
