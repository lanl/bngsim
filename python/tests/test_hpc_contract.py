"""GH #203 — HPC-facing scheduler-free evaluation contract.

bngsim is a clean, *stateless* single-evaluation kernel + optional local batch
helper; the frontend (PyBNF) owns the scheduler. These tests pin the cluster-safe
contract:

1. Artifact-cache + temp-dir controls (``BNGSIM_CODEGEN_CACHE_DIR``).
2. ``run_batch`` reuses one compiled artifact across many independent evaluations
   (statelessness / concurrency stress) and now yields per-row output
   sensitivities for a ``sensitivity_params``-configured Simulator.
3. Deterministic batch row ordering, independent of worker count.
4. Serialization round-trip of ``EvaluationSpec`` + ``resolve_outputs`` metadata
   + ``Result.summary()`` (checkpoint/restart).

No scheduler/MPI/Slurm code is added to bngsim — that is, deliberately, the
frontend's job.
"""

import json
import os
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _codegen

# Honor BNGSIM_TEST_DATA so this module works under run_tests.sh (which copies
# tests to a temp dir, breaking __file__-relative resolution).
_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def _reversible_net() -> str:
    p = DATA_DIR / "two_species_reversible.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _expr_chain_net() -> str:
    # A->B->C with observables + global functions scaled/ratio/combo/tdep;
    # expression output sensitivities (GH #198) need the compiled evaluator.
    p = DATA_DIR / "expr_sens_chain.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


# Expression model is moderately stiff; mirror tolerances proven in
# test_parallel_sensitivity.py / test_expression_output_sensitivities.py.
_EXPR_RUN = dict(rtol=1e-11, atol=1e-13, max_steps=10**6)
_SENS_TOL = dict(atol=1e-6, rtol=1e-5)


# ── Deliverable 1: artifact-cache + temp-dir controls ─────────────


class TestCacheDirControl:
    """BNGSIM_CODEGEN_CACHE_DIR redirects the compiled-artifact cache."""

    def test_default_cache_dir_honors_env(self, monkeypatch, tmp_path):
        target = tmp_path / "scratch" / "codegen"
        monkeypatch.setenv("BNGSIM_CODEGEN_CACHE_DIR", str(target))
        assert _codegen._default_cache_dir() == target

    def test_default_cache_dir_expanduser(self, monkeypatch):
        monkeypatch.setenv("BNGSIM_CODEGEN_CACHE_DIR", "~/somewhere/codegen")
        assert _codegen._default_cache_dir() == Path.home() / "somewhere" / "codegen"

    def test_unset_env_falls_back_to_home(self, monkeypatch):
        monkeypatch.delenv("BNGSIM_CODEGEN_CACHE_DIR", raising=False)
        assert _codegen._default_cache_dir() == Path.home() / ".cache" / "bngsim" / "codegen"

    def test_compiled_artifact_lands_in_redirected_dir(self, monkeypatch, tmp_path):
        """A codegen build writes its .so into the redirected cache, not ~/.cache."""
        cache = tmp_path / "cache"
        monkeypatch.setattr(_codegen, "CACHE_DIR", cache)
        sim = bngsim.Simulator(
            bngsim.Model.from_net(_expr_chain_net()), method="ode", codegen=True
        )
        # Construction compiled the .so into the redirected dir.
        assert sim._codegen_so_path
        so = Path(sim._codegen_so_path)
        assert so.parent == cache, f"{so} not under redirected cache {cache}"
        assert so.exists()
        # And it actually runs from there.
        sim.run(t_span=(0, 10), n_points=11)


# ── Deliverable 2: batch reuses one artifact; statelessness ───────


class TestBatchArtifactReuseStateless:
    """Many independent evaluations reuse one compiled artifact, no cross-talk."""

    def test_plain_batch_carries_codegen_artifact(self):
        """A codegen Simulator's run_batch reuses the one shared .so per row
        (previously the per-row path ran interpreted, reusing nothing)."""
        sim = bngsim.Simulator(
            bngsim.Model.from_net(_expr_chain_net()), method="ode", codegen=True
        )
        assert sim._codegen_so_path  # one artifact, prepared once
        so_before = sim._codegen_so_path

        theta = [{"k1": 0.05 + 0.01 * i} for i in range(12)]
        rows = sim.run_batch(
            t_span=(0, 10), n_points=11, params=theta, num_processors=4, **_EXPR_RUN
        )

        # The artifact is unchanged (shared, not re-created per row).
        assert sim._codegen_so_path == so_before
        assert Path(so_before).exists()

        # Each row equals an independent single run for that θ — no shared
        # mutable state leaked across the concurrent clones.
        for i, th in enumerate(theta):
            ref_sim = bngsim.Simulator(
                bngsim.Model.from_net(_expr_chain_net()), method="ode", codegen=True
            )
            ref_sim._model.set_params(th)
            ref_sim._model.reset()
            ref = ref_sim.run(t_span=(0, 10), n_points=11, **_EXPR_RUN)
            np.testing.assert_allclose(
                rows[i].species, ref.species, err_msg=f"row {i} species cross-talk", **_SENS_TOL
            )

    def test_concurrency_stress_many_workers(self):
        """Heavier fan-out (more rows than workers) stays correct and ordered."""
        net = _reversible_net()
        sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
        theta = [{"kf": 0.001 * (i + 1), "kr": 0.1} for i in range(24)]
        rows = sim.run_batch(t_span=(0, 10), n_points=21, params=theta, num_processors=8)
        assert len(rows) == len(theta)
        for i, th in enumerate(theta):
            ref = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
            ref._model.set_params(th)
            ref._model.reset()
            r = ref.run(t_span=(0, 10), n_points=21)
            np.testing.assert_allclose(rows[i].species, r.species, atol=1e-9, rtol=1e-7)


# ── Deliverable 2: per-row output sensitivities in a batch ────────


class TestBatchSensitivities:
    """A sensitivity_params Simulator yields per-row output sensitivities."""

    def test_species_sensitivities_match_single_shot(self):
        net = _reversible_net()
        params = ["kf", "kr"]
        theta = [{"kf": 0.001, "kr": 0.1}, {"kf": 0.002, "kr": 0.2}, {"kf": 0.003, "kr": 0.05}]

        sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode", sensitivity_params=params)
        rows = sim.run_batch(t_span=(0, 10), n_points=11, params=theta, num_processors=3)

        assert all(r.has_sensitivities for r in rows)
        for i, th in enumerate(theta):
            ref_sim = bngsim.Simulator(
                bngsim.Model.from_net(net), method="ode", sensitivity_params=params
            )
            ref_sim._model.set_params(th)
            ref_sim._model.reset()
            ref = ref_sim.run(t_span=(0, 10), n_points=11)
            np.testing.assert_allclose(
                rows[i].sensitivities, ref.sensitivities, err_msg=f"row {i} sens", **_SENS_TOL
            )

    def test_expression_output_sensitivities_in_batch(self):
        """Expression sensitivities (needs the compiled output-sens evaluator)
        populate in a batch and match single-shot — confirming the shared
        codegen artifact carries the GH #198 output-sens ABI per row."""
        net = _expr_chain_net()
        params = ["k1", "k2", "scale", "eps"]
        theta = [{"k1": 0.1}, {"k1": 0.15}, {"k1": 0.2}]

        sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode", sensitivity_params=params)
        rows = sim.run_batch(
            t_span=(0, 10), n_points=11, params=theta, num_processors=3, **_EXPR_RUN
        )

        for i, th in enumerate(theta):
            ref_sim = bngsim.Simulator(
                bngsim.Model.from_net(net), method="ode", sensitivity_params=params
            )
            ref_sim._model.set_params(th)
            ref_sim._model.reset()
            ref = ref_sim.run(t_span=(0, 10), n_points=11, **_EXPR_RUN)
            assert rows[i]._expression_sensitivities.size > 0
            np.testing.assert_allclose(
                rows[i]._expression_sensitivities,
                ref._expression_sensitivities,
                err_msg=f"row {i} expression sens",
                **_SENS_TOL,
            )
            np.testing.assert_allclose(
                rows[i]._observable_sensitivities,
                ref._observable_sensitivities,
                err_msg=f"row {i} observable sens",
                **_SENS_TOL,
            )

    def test_named_output_sensitivities_via_selectors(self):
        """Batch rows answer output_sensitivities() selectors (the PyBNF seam)."""
        net = _reversible_net()
        sim = bngsim.Simulator(
            bngsim.Model.from_net(net), method="ode", sensitivity_params=["kf", "kr"]
        )
        rows = sim.run_batch(
            t_span=(0, 10), n_points=11, params=[{"kf": 0.001}, {"kf": 0.002}], num_processors=2
        )
        for r in rows:
            sens = r.output_sensitivities(["observable:A_free"])
            assert sens.shape == (11, 1, 2)


# ── Deliverable 2: event sensitivities in a batch (GH #205 → #212) ─────────


class TestEventBatchSensitivity:
    """Sensitivity batches on event models: fixed-time events now run (GH #212
    Phase 1), still-unsupported subclasses keep raising (GH #205)."""

    def _event_sim(self, trigger="time() >= 1000", assign="A"):
        from bngsim._bngsim_core import ModelBuilder

        b = ModelBuilder()
        b.add_parameter("k_prod", 5.0)
        b.add_parameter("k_deg", 0.5)
        s = b.add_species("A", 0.0)
        b.add_reaction([], [s], "elementary", "k_prod")
        b.add_reaction([s], [], "elementary", "k_deg")
        b.add_event("evt", trigger, [(s, assign)])
        m = bngsim.Model(b.build())
        assert m._core.n_events == 1
        return bngsim.Simulator(m, method="ode", sensitivity_params=["k_prod", "k_deg"])

    def test_sensitivity_batch_fixed_time_event_runs(self):
        # Fixed-time event (never fires in window) is Phase-1-safe: the batch
        # runs instead of raising, and propagates sensitivities through it.
        sim = self._event_sim()
        rows = sim.run_batch(t_span=(0, 5), n_points=6, params=[{"k_prod": 5.0}, {"k_prod": 6.0}])
        assert len(rows) == 2

    def test_sensitivity_batch_state_dependent_event_raises(self):
        # A state-dependent trigger is still unsupported for sensitivities.
        sim = self._event_sim(trigger="A > 1000", assign="0")
        with pytest.raises(ValueError, match=r"state-dependent|205"):
            sim.run_batch(t_span=(0, 5), n_points=6, params=[{"k_prod": 5.0}, {"k_prod": 6.0}])

    def test_non_sensitivity_batch_on_event_model_is_fine(self):
        """No sensitivities requested → event model batches normally."""
        from bngsim._bngsim_core import ModelBuilder

        b = ModelBuilder()
        b.add_parameter("k_prod", 5.0)
        b.add_parameter("k_deg", 0.5)
        s = b.add_species("A", 0.0)
        b.add_reaction([], [s], "elementary", "k_prod")
        b.add_reaction([s], [], "elementary", "k_deg")
        b.add_event("noop", "time() >= 1000", [(s, "A")])
        sim = bngsim.Simulator(bngsim.Model(b.build()), method="ode")
        rows = sim.run_batch(t_span=(0, 5), n_points=6, params=[{"k_prod": 5.0}, {"k_prod": 6.0}])
        assert len(rows) == 2


# ── Deliverable 3: deterministic batch row ordering ───────────────


class TestDeterministicOrdering:
    """Row order tracks the input θ order, independent of worker count."""

    def test_order_independent_of_worker_count(self):
        net = _reversible_net()
        theta = [{"kf": 0.001 * (i + 1), "kr": 0.1} for i in range(10)]
        sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")

        serial = sim.run_batch(t_span=(0, 10), n_points=11, params=theta, num_processors=1)
        parallel = sim.run_batch(t_span=(0, 10), n_points=11, params=theta, num_processors=4)

        assert len(serial) == len(parallel) == len(theta)
        for i in range(len(theta)):
            np.testing.assert_array_equal(
                serial[i].species, parallel[i].species, err_msg=f"row {i} order/det mismatch"
            )

    def test_squeeze_preserves_order(self):
        net = _reversible_net()
        theta = [{"kf": 0.001 * (i + 1)} for i in range(5)]
        sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
        squeezed = sim.run_batch(
            t_span=(0, 10), n_points=11, params=theta, num_processors=2, squeeze=True
        )
        listed = sim.run_batch(t_span=(0, 10), n_points=11, params=theta, num_processors=2)
        assert squeezed.species.shape[0] == len(theta)
        for i in range(len(theta)):
            np.testing.assert_array_equal(squeezed.species[i], listed[i].species)


# ── Deliverable 3: EvaluationSpec serialization round-trip ────────


class TestEvaluationSpecSerialization:
    """The serializable single-evaluation kernel (checkpoint/restart)."""

    def _spec(self):
        return bngsim.EvaluationSpec(
            model_source=_reversible_net(),
            model_format="net",
            t_span=(0.0, 10.0),
            n_points=11,
            params={"kf": 0.002, "kr": 0.05},
            sensitivity_params=("kf", "kr"),
            outputs=("species:A", "observable:Atot"),
        )

    def test_json_round_trip_equal(self):
        spec = self._spec()
        restored = bngsim.EvaluationSpec.from_json(spec.to_json())
        assert restored == spec

    def test_dict_round_trip_equal(self):
        spec = self._spec()
        assert bngsim.EvaluationSpec.from_dict(spec.to_dict()) == spec

    def test_json_is_byte_stable(self):
        """Two specs with the same fields (params in different insertion order)
        serialize to identical JSON — usable as a content key."""
        a = bngsim.EvaluationSpec(model_source="m.net", params={"b": 2.0, "a": 1.0})
        b = bngsim.EvaluationSpec(model_source="m.net", params={"a": 1.0, "b": 2.0})
        assert a.to_json() == b.to_json()

    def test_post_init_normalizes_containers(self):
        spec = bngsim.EvaluationSpec(
            model_source="m.net", t_span=[0, 5], sensitivity_params=["k"], outputs=["species:A"]
        )
        assert isinstance(spec.t_span, tuple) and spec.t_span == (0.0, 5.0)
        assert isinstance(spec.sensitivity_params, tuple)
        assert isinstance(spec.outputs, tuple)
        assert isinstance(spec.params, dict)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="Unknown EvaluationSpec field"):
            bngsim.EvaluationSpec.from_dict({"model_source": "m.net", "bogus": 1})

    def test_bad_format_rejected(self):
        with pytest.raises(ValueError, match="Unknown model_format"):
            bngsim.EvaluationSpec(model_source="m.net", model_format="nope")

    def test_with_params_returns_new_spec(self):
        spec = self._spec()
        replaced = spec.with_params({"kf": 9.9})
        assert replaced.params == {"kf": 9.9}
        assert spec.params == {"kf": 0.002, "kr": 0.05}  # original untouched
        merged = spec.with_params({"kf": 9.9}, merge=True)
        assert merged.params == {"kf": 9.9, "kr": 0.05}

    def test_evaluate_reproduces_direct_run(self):
        """A round-tripped spec evaluates to the same arrays as a direct run."""
        spec = self._spec()
        restored = bngsim.EvaluationSpec.from_json(spec.to_json())
        r_spec = restored.evaluate()

        ref_sim = bngsim.Simulator(
            bngsim.Model.from_net(_reversible_net()),
            method="ode",
            sensitivity_params=["kf", "kr"],
        )
        ref_sim._model.set_params({"kf": 0.002, "kr": 0.05})
        ref_sim._model.reset()
        ref = ref_sim.run(t_span=(0.0, 10.0), n_points=11)

        np.testing.assert_array_equal(r_spec.species, ref.species)
        np.testing.assert_array_equal(r_spec.sensitivities, ref.sensitivities)

    def test_evaluate_is_deterministic(self):
        spec = self._spec()
        r1, r2 = spec.evaluate(), spec.evaluate()
        np.testing.assert_array_equal(r1.species, r2.species)
        np.testing.assert_array_equal(r1.sensitivities, r2.sensitivities)

    def test_sha256_match_and_mismatch(self):
        net = _reversible_net()
        good = bngsim.EvaluationSpec(model_source=net, model_format="net")
        digest = good.compute_source_sha256()
        ok = bngsim.EvaluationSpec(model_source=net, model_format="net", model_sha256=digest)
        ok.build_model()  # no raise
        bad = bngsim.EvaluationSpec(model_source=net, model_format="net", model_sha256="0" * 64)
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            bad.build_model()


# ── Deliverable 3: output metadata + Result.summary round-trip ────


class TestOutputMetadataAndSummary:
    """resolve_outputs metadata and Result.summary() are JSON-serializable."""

    def _result(self):
        sim = bngsim.Simulator(
            bngsim.Model.from_net(_reversible_net()), method="ode", sensitivity_params=["kf", "kr"]
        )
        return sim.run(t_span=(0, 10), n_points=11)

    def test_resolve_outputs_metadata_json_round_trips(self):
        r = self._result()
        meta = r.resolve_outputs(["species:A()", "observable:A_free"])
        blob = json.dumps(meta)
        assert json.loads(blob) == meta

    def test_summary_is_json_serializable(self):
        r = self._result()
        s = r.summary()
        # Round-trips through JSON unchanged (all built-in types).
        assert json.loads(json.dumps(s)) == s

    def test_summary_reports_shapes_and_sensitivity_flags(self):
        r = self._result()
        s = r.summary()
        assert s["is_batch"] is False
        assert s["n_sims"] is None
        assert s["n_times"] == 11
        assert s["t_start"] == 0.0 and s["t_end"] == 10.0
        assert s["has_sensitivities"] is True
        assert s["sensitivity_params"] == ["kf", "kr"]
        assert s["shapes"]["species"] == list(r.species.shape)

    def test_summary_on_batch_result(self):
        sim = bngsim.Simulator(bngsim.Model.from_net(_reversible_net()), method="ode")
        batch = sim.run_batch(
            t_span=(0, 10), n_points=11, params=[{"kf": 0.001}, {"kf": 0.002}], squeeze=True
        )
        s = batch.summary()
        assert s["is_batch"] is True
        assert s["n_sims"] == 2
        assert json.loads(json.dumps(s)) == s
