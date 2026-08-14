"""Tests for parallel chunked sensitivity and gradient API (Session 28).

Verifies:
1. compute_all_sensitivities() — parallel chunked CVODES forward sensitivity
2. Result.gradient() — parameter gradient from sensitivity tensor + loss function
3. Correctness: parallel results match serial all-at-once CVODES
4. FIM from parallel matches FIM from serial
"""

import os
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import SimulationError

# Honor BNGSIM_TEST_DATA so this module works under run_tests.sh, which copies
# tests to a temp dir (breaking __file__-relative resolution).
_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def _get_simple_decay_net() -> str:
    p = DATA_DIR / "simple_decay.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_reversible_net() -> str:
    p = DATA_DIR / "two_species_reversible.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _get_expr_chain_net() -> str:
    # A --k1--> B --k2--> C with observables A_obs, BC2, Total and global
    # functions scaled/ratio/combo/tdep — expression output sensitivities (GH
    # #198) need the compiled codegen evaluator, so this drives the codegen path.
    p = DATA_DIR / "expr_sens_chain.net"
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


# ── compute_all_sensitivities: shape and correctness ─────────────


class TestComputeAllSensitivitiesShape:
    """Shape and basic properties of the stitched tensor."""

    def test_single_param_shape(self):
        """1 param, chunk_size=1 → 1 chunk, shape correct."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode")
        result = sim.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=11,
            params=["k1"],
            chunk_size=1,
            n_workers=1,
        )
        assert result.has_sensitivities
        assert result.sensitivities.shape == (11, 2, 1)
        assert result.sensitivity_params == ["k1"]

    def test_two_param_shape(self):
        """2 params, chunk_size=2 → 1 chunk."""
        model = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(model, method="ode")
        result = sim.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=11,
            params=["kf", "kr"],
            chunk_size=2,
            n_workers=1,
        )
        ns = model.n_species
        assert result.sensitivities.shape == (11, ns, 2)
        assert result.sensitivity_params == ["kf", "kr"]

    def test_two_param_chunked(self):
        """2 params, chunk_size=1 → 2 chunks stitched."""
        model = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(model, method="ode")
        result = sim.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=11,
            params=["kf", "kr"],
            chunk_size=1,
            n_workers=1,
        )
        ns = model.n_species
        assert result.sensitivities.shape == (11, ns, 2)
        assert result.sensitivity_params == ["kf", "kr"]

    def test_all_params_default(self):
        """params=None → all model params."""
        model = bngsim.Model.from_net(_get_reversible_net())
        n_params = model.n_parameters
        sim = bngsim.Simulator(model, method="ode")
        result = sim.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=11,
            chunk_size=2,
            n_workers=1,
        )
        ns = model.n_species
        assert result.sensitivities.shape == (11, ns, n_params)
        assert len(result.sensitivity_params) == n_params

    def test_species_preserved(self):
        """ODE species trajectory should be identical to plain run."""
        net_path = _get_simple_decay_net()
        model1 = bngsim.Model.from_net(net_path)
        sim1 = bngsim.Simulator(model1, method="ode")
        r1 = sim1.run(t_span=(0, 10), n_points=101)

        model2 = bngsim.Model.from_net(net_path)
        sim2 = bngsim.Simulator(model2, method="ode")
        r2 = sim2.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=101,
            params=["k1"],
            chunk_size=1,
            n_workers=1,
        )
        np.testing.assert_allclose(r1.species, r2.species, atol=1e-10)


class TestParallelMatchesSerial:
    """Parallel chunked results should match serial all-at-once."""

    def test_serial_vs_parallel(self):
        """chunk_size=1 n_workers=2 matches chunk_size=2 n_workers=1.

        Note: CVODES internal FD sensitivity approximation produces
        slightly different results when computing 2 params together
        vs 1 at a time, because the adaptive step controller sees
        different error estimates. Max diff ~3e-8 is expected.
        """
        net_path = _get_reversible_net()

        model1 = bngsim.Model.from_net(net_path)
        sim1 = bngsim.Simulator(model1, method="ode")
        r_serial = sim1.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=51,
            params=["kf", "kr"],
            chunk_size=2,
            n_workers=1,
        )

        model2 = bngsim.Model.from_net(net_path)
        sim2 = bngsim.Simulator(model2, method="ode")
        r_parallel = sim2.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=51,
            params=["kf", "kr"],
            chunk_size=1,
            n_workers=2,
        )

        np.testing.assert_allclose(
            r_serial.sensitivities,
            r_parallel.sensitivities,
            atol=1e-6,
            rtol=1e-5,
            err_msg="Parallel != serial sensitivities",
        )

    def test_matches_direct_cvodes(self):
        """compute_all_sensitivities matches direct CVODES run.

        Note: chunk_size=1 runs each param separately, while direct
        CVODES runs both together. CVODES internal FD produces
        slightly different values (~3e-8) due to step-size coupling.
        """
        net_path = _get_reversible_net()

        # Direct CVODES (Session 26 API)
        m1 = bngsim.Model.from_net(net_path)
        sim1 = bngsim.Simulator(m1, method="ode", sensitivity_params=["kf", "kr"])
        r1 = sim1.run(t_span=(0, 10), n_points=51)

        # Parallel chunked
        m2 = bngsim.Model.from_net(net_path)
        sim2 = bngsim.Simulator(m2, method="ode")
        r2 = sim2.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=51,
            params=["kf", "kr"],
            chunk_size=1,
            n_workers=2,
        )

        np.testing.assert_allclose(
            r1.sensitivities,
            r2.sensitivities,
            atol=1e-6,
            rtol=1e-5,
            err_msg="Parallel != direct CVODES sensitivities",
        )

    def test_analytical_accuracy(self):
        """Parallel sensitivity matches analytical dA/dk1."""
        net_path = _get_simple_decay_net()
        model = bngsim.Model.from_net(net_path)
        sim = bngsim.Simulator(model, method="ode")
        result = sim.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=101,
            params=["k1"],
            chunk_size=1,
            n_workers=1,
        )

        k1 = 0.1
        A0 = 100.0
        t = result.time
        analytical = -t * A0 * np.exp(-k1 * t)
        computed = result.sensitivities[:, 0, 0]

        max_err = np.max(np.abs(computed - analytical))
        assert max_err < 1.0, f"Max error {max_err} too large"


class TestParallelWithMultipleWorkers:
    """Test with actual parallel threads (n_workers > 1)."""

    def test_parallel_2_workers(self):
        """2 workers, 2 chunks."""
        model = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(model, method="ode")
        result = sim.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=11,
            params=["kf", "kr"],
            chunk_size=1,
            n_workers=2,
        )
        assert result.has_sensitivities
        ns = model.n_species
        assert result.sensitivities.shape == (11, ns, 2)

        # Should be nonzero at t>0
        assert np.any(np.abs(result.sensitivities[-1]) > 1e-10)


# ── FIM from parallel ────────────────────────────────────────────


class TestFIMFromParallel:
    """FIM from parallel matches FIM from serial."""

    def test_fim_matches_serial(self):
        """FIM from chunked parallel == FIM from direct CVODES."""
        net_path = _get_reversible_net()

        # Direct
        m1 = bngsim.Model.from_net(net_path)
        sim1 = bngsim.Simulator(m1, method="ode", sensitivity_params=["kf", "kr"])
        r1 = sim1.run(t_span=(0, 10), n_points=51)
        fim1 = r1.fisher_information(sigma=1.0)

        # Parallel
        m2 = bngsim.Model.from_net(net_path)
        sim2 = bngsim.Simulator(m2, method="ode")
        r2 = sim2.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=51,
            params=["kf", "kr"],
            chunk_size=1,
            n_workers=2,
        )
        fim2 = r2.fisher_information(sigma=1.0)

        np.testing.assert_allclose(
            fim1,
            fim2,
            atol=1e-6,
            err_msg="FIM parallel != FIM serial",
        )


# ── Result.gradient() ────────────────────────────────────────────


class TestGradient:
    """Tests for Result.gradient(loss_fn)."""

    def test_gradient_shape(self):
        """Gradient should have shape (n_params,)."""
        model = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(
            model,
            method="ode",
            sensitivity_params=["kf", "kr"],
        )
        result = sim.run(t_span=(0, 10), n_points=51)

        # Zero loss → zero gradient
        def zero_loss(species, time):
            return np.zeros_like(species)

        grad = result.gradient(zero_loss)
        assert grad.shape == (2,)
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)

    def test_gradient_nonzero(self):
        """Non-trivial loss should give nonzero gradient."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(
            model,
            method="ode",
            sensitivity_params=["k1"],
        )
        result = sim.run(t_span=(0, 10), n_points=101)

        # SSE loss: L = Σ (species - target)^2
        # dL/dY = 2 * (species - target)
        target = np.zeros_like(result.species)  # target = 0

        def sse_loss(species, time):
            return 2 * (species - target)

        grad = result.gradient(sse_loss)
        assert grad.shape == (1,)
        assert grad[0] != 0.0, "Gradient should be nonzero"

    def test_gradient_analytical_simple_decay(self):
        """Verify gradient for simple decay against analytical formula.

        Model: dA/dt = -k1*A, A(0)=100, k1=0.1
        Loss: L = Σ_t A(t)^2
        dL/dA = 2*A(t), dL/dB = 0
        ∇_{k1} L = Σ_t dA/dk1 * 2*A(t) + dB/dk1 * 0
                 = Σ_t (-t * A0 * exp(-k1*t)) * 2 * (A0 * exp(-k1*t))
                 = -2 * A0^2 * Σ_t t * exp(-2*k1*t)
        """
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(
            model,
            method="ode",
            sensitivity_params=["k1"],
        )
        result = sim.run(t_span=(0, 10), n_points=101)

        def loss_fn(species, time):
            # dL/dY where L = sum(A^2) over time
            # dL/dA_t = 2*A_t, dL/dB_t = 0
            grad = np.zeros_like(species)
            grad[:, 0] = 2 * species[:, 0]  # species 0 = A
            return grad

        grad = result.gradient(loss_fn)

        # Analytical expected gradient
        k1 = 0.1
        A0 = 100.0
        t = result.time
        # dA/dk1 = -t * A0 * exp(-k1*t)
        # dL/dA = 2 * A0 * exp(-k1*t)
        # grad = sum( dA/dk1 * dL/dA ) = sum(-2 * A0^2 * t * exp(-2*k1*t))
        expected = np.sum(-2 * A0**2 * t * np.exp(-2 * k1 * t))

        # Allow some tolerance (CVODES FD vs analytical)
        np.testing.assert_allclose(
            grad[0],
            expected,
            rtol=0.01,
            err_msg="Gradient analytical check failed",
        )

    def test_gradient_from_parallel(self):
        """Gradient from parallel compute_all_sensitivities matches
        gradient from direct CVODES run."""
        net_path = _get_reversible_net()

        # Direct
        m1 = bngsim.Model.from_net(net_path)
        sim1 = bngsim.Simulator(m1, method="ode", sensitivity_params=["kf", "kr"])
        r1 = sim1.run(t_span=(0, 10), n_points=51)

        # Parallel
        m2 = bngsim.Model.from_net(net_path)
        sim2 = bngsim.Simulator(m2, method="ode")
        r2 = sim2.compute_all_sensitivities(
            t_span=(0, 10),
            n_points=51,
            params=["kf", "kr"],
            chunk_size=1,
            n_workers=2,
        )

        def loss_fn(species, time):
            return 2 * species  # dL/dY for L = sum(Y^2)

        g1 = r1.gradient(loss_fn)
        g2 = r2.gradient(loss_fn)

        np.testing.assert_allclose(
            g1,
            g2,
            atol=1e-6,
            err_msg="Gradient parallel != gradient serial",
        )


class TestGradientErrors:
    """Error handling for gradient()."""

    def test_no_sensitivity_raises(self):
        """Should raise if no sensitivity data."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, 10), n_points=11)

        with pytest.raises(ValueError, match="No sensitivity data"):
            result.gradient(lambda sp, t: np.zeros_like(sp))

    def test_wrong_return_shape_raises(self):
        """Should raise if loss_fn returns wrong shape."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(
            model,
            method="ode",
            sensitivity_params=["k1"],
        )
        result = sim.run(t_span=(0, 10), n_points=11)

        with pytest.raises(ValueError, match="must return shape"):
            result.gradient(lambda sp, t: np.zeros(5))

    def test_not_callable_raises(self):
        """Should raise TypeError if loss_fn is not callable."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(
            model,
            method="ode",
            sensitivity_params=["k1"],
        )
        result = sim.run(t_span=(0, 10), n_points=11)

        with pytest.raises(TypeError, match="must be callable"):
            result.gradient("not_a_function")


# ── compute_all_sensitivities error handling ─────────────────────


class TestComputeAllSensitivitiesErrors:
    """Error handling for compute_all_sensitivities."""

    def test_ssa_raises(self):
        """Should raise for non-ODE methods."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ssa")

        with pytest.raises(ValueError, match="only supported.*ode"):
            sim.compute_all_sensitivities(t_span=(0, 10), n_points=11)

    def test_unknown_param_raises(self):
        """Should raise for unknown parameter names."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode")

        with pytest.raises(ValueError, match="Unknown parameter"):
            sim.compute_all_sensitivities(
                t_span=(0, 10),
                n_points=11,
                params=["nonexistent"],
            )

    def test_invalid_chunk_size_raises(self):
        """chunk_size < 1 should raise."""
        model = bngsim.Model.from_net(_get_simple_decay_net())
        sim = bngsim.Simulator(model, method="ode")

        with pytest.raises(ValueError, match="chunk_size"):
            sim.compute_all_sensitivities(
                t_span=(0, 10),
                n_points=11,
                chunk_size=0,
            )


# ── GH #204: observable/expression output-sensitivity stitching ──
#
# compute_all_sensitivities must produce IDENTICAL observable AND expression
# output-sensitivity tensors to a single-shot Simulator(sensitivity_params=...)
# run. Observables (GH #197 runtime chain rule) need no codegen; expressions
# (GH #198) require the compiled output-sensitivity evaluator, which each
# parallel chunk only inherits if codegen was enabled for the workflow.
#
# CVODES internal FD couples step-size selection across the parameters solved
# together, so chunk_size=1 (one param per solve) differs from an all-at-once
# solve by ~1e-8 — exactly the tolerance the species tests above already use.

_OUT_SENS_TOL = dict(atol=1e-6, rtol=1e-5)


class TestObservableOutputSensitivityStitching:
    """Observable block stitches with no codegen (GH #197 is runtime)."""

    def test_observable_block_matches_single_shot(self):
        """compute_all_sensitivities(chunk_size=1) == single-shot for the
        observable output-sensitivity block, on a model with no expressions
        (the block itself is a runtime chain rule — it needs no codegen)."""
        net = _get_reversible_net()
        params = ["kf", "kr"]
        t_span, n_points = (0, 10), 11

        m1 = bngsim.Model.from_net(net)
        single = bngsim.Simulator(m1, method="ode", sensitivity_params=params)
        r_single = single.run(t_span=t_span, n_points=n_points)

        m2 = bngsim.Model.from_net(net)
        chunked = bngsim.Simulator(m2, method="ode")
        r_chunk = chunked.compute_all_sensitivities(
            t_span=t_span, n_points=n_points, params=params, chunk_size=1, n_workers=1
        )

        # Single-shot actually has the observable block (the test is meaningful).
        assert r_single._observable_sensitivities.size > 0
        np.testing.assert_allclose(
            r_chunk._observable_sensitivities,
            r_single._observable_sensitivities,
            err_msg="chunked observable sensitivities != single-shot",
            **_OUT_SENS_TOL,
        )

    def test_no_expression_model_still_gets_sensitivity_codegen(self):
        """A model with no global functions emits no *expression* block, but it
        still gets the sensitivity codegen the chunks integrate (GH #62).

        The observable block itself is a runtime chain rule and needs no codegen;
        the state-sensitivity RHS underneath it does. This assertion used to read
        ``not sim._codegen_so_path`` — the ``n_functions`` gate that made the
        chunked path finite-difference the sensitivity RHS interpreted."""
        m = bngsim.Model.from_net(_get_reversible_net())
        assert m._core.n_functions == 0
        sim = bngsim.Simulator(m, method="ode")
        r = sim.compute_all_sensitivities(
            t_span=(0, 10), n_points=11, params=["kf", "kr"], chunk_size=1, n_workers=1
        )
        assert sim._codegen_so_path or sim._codegen_c_source
        # No expression outputs to emit, so that block stays legitimately empty.
        assert r._expression_sensitivities.size == 0


class TestExpressionOutputSensitivityStitching:
    """Expression block stitches under codegen (GH #198, the #204 fix)."""

    @pytest.fixture(autouse=True)
    def _force_codegen(self, monkeypatch):
        """Expression output sensitivities require the compiled ``.so``; force
        codegen on for every test here. monkeypatch restores the environment
        afterwards so the threshold never leaks into threshold-sensitive tests."""
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
        monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)

    # expr_sens_chain is moderately stiff; mirror the tolerances proven out in
    # test_expression_output_sensitivities.py.
    _RUN = dict(rtol=1e-11, atol=1e-13, max_steps=10**6)
    _PARAMS = ["k1", "k2", "scale", "eps"]
    _T_SPAN = (0.0, 10.0)
    _N_POINTS = 11

    def _single_shot(self, net, params):
        sim = bngsim.Simulator(
            bngsim.Model.from_net(net), method="ode", sensitivity_params=list(params)
        )
        return sim.run(t_span=self._T_SPAN, n_points=self._N_POINTS, **self._RUN)

    def test_expression_block_matches_single_shot(self):
        """The DoD: chunked expression sensitivities == single-shot.

        Without the GH #204 fix the chunk path runs interpreted and this block
        comes back (0, 0, 0); with it, codegen is auto-enabled and the tensors
        match the single-shot run."""
        net = _get_expr_chain_net()
        r_single = self._single_shot(net, self._PARAMS)

        chunked = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
        r_chunk = chunked.compute_all_sensitivities(
            t_span=self._T_SPAN,
            n_points=self._N_POINTS,
            params=self._PARAMS,
            chunk_size=1,
            n_workers=1,
            **self._RUN,
        )

        # The fix actually populated the block (not silently empty).
        assert r_single._expression_sensitivities.size > 0
        assert r_chunk._expression_sensitivities.shape == (
            self._N_POINTS,
            bngsim.Model.from_net(net)._core.n_functions,
            len(self._PARAMS),
        )
        np.testing.assert_allclose(
            r_chunk._expression_sensitivities,
            r_single._expression_sensitivities,
            err_msg="chunked expression sensitivities != single-shot",
            **_OUT_SENS_TOL,
        )
        # The observable block must still match too (no regression).
        np.testing.assert_allclose(
            r_chunk._observable_sensitivities,
            r_single._observable_sensitivities,
            err_msg="chunked observable sensitivities != single-shot",
            **_OUT_SENS_TOL,
        )

    def test_parallel_workers_match_single_shot(self):
        """Same equivalence with real parallel threads (n_workers > 1)."""
        net = _get_expr_chain_net()
        r_single = self._single_shot(net, self._PARAMS)

        chunked = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
        r_chunk = chunked.compute_all_sensitivities(
            t_span=self._T_SPAN,
            n_points=self._N_POINTS,
            params=self._PARAMS,
            chunk_size=1,
            n_workers=4,
            **self._RUN,
        )
        np.testing.assert_allclose(
            r_chunk._expression_sensitivities,
            r_single._expression_sensitivities,
            err_msg="parallel expression sensitivities != single-shot",
            **_OUT_SENS_TOL,
        )

    def test_parameter_order_preserved_across_chunks(self):
        """Each stitched parameter column must land in the REQUESTED order.

        Uses a deliberately non-model order and chunk_size=1 (one param per
        chunk, >2 chunks), then checks every column against an independent
        single-parameter ground-truth run — a reordering bug would mismatch."""
        net = _get_expr_chain_net()
        order = ["scale", "eps", "k2", "k1"]  # not the model's parameter order

        # Independent ground truth: one single-param single-shot run per param.
        truth = {p: self._single_shot(net, [p])._expression_sensitivities[:, :, 0] for p in order}

        chunked = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
        r = chunked.compute_all_sensitivities(
            t_span=self._T_SPAN,
            n_points=self._N_POINTS,
            params=order,
            chunk_size=1,
            n_workers=1,
            **self._RUN,
        )

        assert r.sensitivity_params == order
        for i, p in enumerate(order):
            np.testing.assert_allclose(
                r._expression_sensitivities[:, :, i],
                truth[p],
                err_msg=f"column {i} is not parameter '{p}' (order not preserved)",
                **_OUT_SENS_TOL,
            )

    def test_inconsistent_output_block_raises(self):
        """A block present in some chunks but missing in others is a real bug,
        not 'nobody computed it' — the stitch must raise loudly (mirroring the
        species-path error), naming the offending block."""
        net = _get_expr_chain_net()
        chunk_a = self._single_shot(net, ["k1"])
        chunk_b = self._single_shot(net, ["k2"])

        # Precondition: both chunks genuinely carry the expression block.
        assert chunk_a._expression_sensitivities.size > 0
        assert chunk_b._expression_sensitivities.size > 0

        # Construct the inconsistency: drop the block from the second chunk only.
        chunk_b._expression_sensitivities = np.empty((0, 0, 0))

        with pytest.raises(SimulationError, match=r"Inconsistent.*_expression_sensitivities"):
            bngsim.Simulator._stitch_sensitivity_results([chunk_a, chunk_b], ["k1", "k2"])

    def test_all_chunks_empty_block_is_legitimate(self):
        """When EVERY chunk lacks an output block (interpreted run / no
        expressions), the stitch leaves it empty rather than raising."""
        net = _get_expr_chain_net()
        chunk_a = self._single_shot(net, ["k1"])
        chunk_b = self._single_shot(net, ["k2"])

        # Drop the expression block from BOTH chunks — legitimately empty.
        chunk_a._expression_sensitivities = np.empty((0, 0, 0))
        chunk_b._expression_sensitivities = np.empty((0, 0, 0))

        stitched = bngsim.Simulator._stitch_sensitivity_results([chunk_a, chunk_b], ["k1", "k2"])
        assert stitched._expression_sensitivities.size == 0
        # The species block still stitched along the param axis (sanity).
        assert stitched.sensitivities.shape[2] == 2


# ── GH #62: function-free models get the SAME sensitivity codegen ────────────
#
# compute_all_sensitivities used to attach the sensitivity codegen only when
# ``model.n_functions > 0``. A function-free model therefore ran every chunk on
# the interpreted RHS, with CVODES finite-differencing the whole sensitivity RHS
# — the exact path GH #214 retired for the single-shot solve. Two consequences,
# and a test for each, both against the single-shot run as ground truth:
#
#   * cost — the FD sensitivity RHS costs extra evaluations AND (through its
#     noise) extra steps, so the chunked call ran up to 49x slower than the
#     identical single-shot solve on large function-free networks (fceri_fyn
#     768 s vs 15.7 s), which made parameter sharding a pessimization exactly
#     where it should help;
#   * robustness — that ~sqrt(eps) noise cannot support tight tolerances, so the
#     error test micro-steps: even the 3-species fixture below takes ~24k steps
#     at rtol=1e-10 and trips the CVODES step cap on the old path.
#
# Cost is asserted through the solver's own RHS-evaluation count rather than
# wall-clock, so the test is deterministic: with ``chunk_size == len(params)``
# and ``n_workers=1`` the chunked call is one chunk over the same parameters as
# the coupled call, i.e. exactly the same work.


class TestFunctionFreeSensitivityCodegen:
    """The chunked path must not fall back to the interpreted sensitivity RHS."""

    _PARAMS = ["kf", "kr"]
    # Tight enough that the retired FD sensitivity path visibly fails.
    _RUN = dict(rtol=1e-10, atol=1e-12)
    _T_SPAN = (0.0, 10.0)
    _N_POINTS = 11

    @pytest.fixture(autouse=True)
    def _allow_codegen(self, monkeypatch):
        monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)

    def _single_shot(self):
        m = bngsim.Model.from_net(_get_reversible_net())
        assert m._core.n_functions == 0, "fixture must be function-free to test the gate"
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=list(self._PARAMS))
        return sim, sim.run(t_span=self._T_SPAN, n_points=self._N_POINTS, **self._RUN)

    def _chunked(self):
        m = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(m, method="ode")
        return sim, sim.compute_all_sensitivities(
            t_span=self._T_SPAN,
            n_points=self._N_POINTS,
            params=list(self._PARAMS),
            chunk_size=len(self._PARAMS),
            n_workers=1,
            **self._RUN,
        )

    def test_codegen_attached_like_single_shot(self):
        """Both entry points end up with a codegen sensitivity RHS attached."""
        single, _ = self._single_shot()
        chunked, _ = self._chunked()
        assert single._codegen_so_path or single._codegen_c_source
        assert chunked._codegen_so_path or chunked._codegen_c_source

    def test_matches_single_shot_at_tight_tolerance(self):
        """Switching the chunks onto the analytical sensitivity RHS must not
        move the answer: same tensor as the coupled solve. (This one also passes
        pre-fix on a model this small — it guards the new path, while
        ``test_costs_the_same_as_single_shot`` is what locks the fix.)"""
        _, r_single = self._single_shot()
        _, r_chunk = self._chunked()
        np.testing.assert_allclose(
            r_chunk.sensitivities,
            r_single.sensitivities,
            rtol=1e-6,
            atol=1e-9,
            err_msg="chunked sensitivities != single-shot (interpreted FD fallback?)",
        )

    def test_costs_the_same_as_single_shot(self):
        """One chunk over the same parameters must not cost more RHS
        evaluations than the coupled solve. Under the old gate CVODES
        finite-differenced the sensitivity RHS and burned ~100x as many."""
        _, r_single = self._single_shot()
        _, r_chunk = self._chunked()
        single_evals = r_single.solver_stats["n_rhs_evals"]
        chunk_evals = r_chunk.solver_stats["n_rhs_evals"]
        assert chunk_evals <= 3 * single_evals, (
            f"chunked run used {chunk_evals} RHS evaluations vs {single_evals} "
            "for the identical coupled solve — the chunks are not using the "
            "codegen sensitivity RHS"
        )

    def test_refuses_without_codegen_like_single_shot(self, monkeypatch):
        """GH #214's refusal now reaches function-free models too: with codegen
        unavailable both entry points raise instead of silently returning
        finite-difference sensitivities."""
        monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")
        m = bngsim.Model.from_net(_get_reversible_net())
        with pytest.raises(ValueError, match="requires code generation"):
            bngsim.Simulator(m, method="ode", sensitivity_params=list(self._PARAMS))

        m2 = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(m2, method="ode")
        with pytest.raises(ValueError, match="requires code generation"):
            sim.compute_all_sensitivities(
                t_span=self._T_SPAN,
                n_points=self._N_POINTS,
                params=list(self._PARAMS),
                chunk_size=1,
                n_workers=1,
            )


# ── GH #243: a chunk failure is about the grouping until proven otherwise ────
class TestChunkFailureIsRetriedPerColumn:
    """``chunk_size`` decided whether the *answer existed*, not just how fast.

    A chunk is ``chunk_size`` sensitivity columns sharing one CVODES error test,
    so the grouping is part of what succeeded or failed. On BIOMD0000000044 the
    same 23 columns raised at ``chunk_size`` 1, 2 and 4 and returned at 3 — and
    the bare ``CVODE integration failed`` message read as a statement about the
    model, which is the one thing it was not. Now every failed chunk is retried
    one column at a time before anything is reported, which answers the question
    the raw failure could not: unresolvable column, or unlucky grouping?
    """

    _PARAMS = ["kf", "kr"]
    _RUN = dict(t_span=(0.0, 10.0), n_points=11)

    def _sim(self):
        return bngsim.Simulator(bngsim.Model.from_net(_get_reversible_net()), method="ode")

    @staticmethod
    def _fail_when(monkeypatch, predicate):
        """Make ``_run_sensitivity_chunk`` raise for the chunks ``predicate`` picks."""
        real = bngsim.Simulator._run_sensitivity_chunk
        seen: list[list[str]] = []

        def fake(self, sens_params, *args, **kwargs):
            seen.append(list(sens_params))
            if predicate(list(sens_params)):
                raise SimulationError(
                    f"Sensitivity chunk failed (params={list(sens_params)}): "
                    "CVODE integration failed at t=2.000000 with flag=-4"
                )
            return real(self, sens_params, *args, **kwargs)

        monkeypatch.setattr(bngsim.Simulator, "_run_sensitivity_chunk", fake)
        return seen

    def test_group_only_failure_is_recovered(self, monkeypatch):
        """Every column integrates alone → the grouping was the problem, and the
        call returns the full tensor built from the single-column solves."""
        seen = self._fail_when(monkeypatch, lambda p: len(p) > 1)
        with pytest.warns(UserWarning, match="integrated on its own"):
            got = self._sim().compute_all_sensitivities(
                **self._RUN, params=list(self._PARAMS), chunk_size=2, n_workers=1
            )

        assert list(got.sensitivity_params) == self._PARAMS
        assert got.sensitivities.shape[2] == 2
        assert np.isfinite(got.sensitivities).all()
        # The grouped attempt came first, then one solo run per column.
        assert seen == [["kf", "kr"], ["kf"], ["kr"]]

        # ...and the recovered tensor is the chunk_size=1 tensor, column for column.
        ref = self._sim().compute_all_sensitivities(
            **self._RUN, params=list(self._PARAMS), chunk_size=1, n_workers=1
        )
        np.testing.assert_allclose(got.sensitivities, ref.sensitivities, rtol=1e-9, atol=1e-12)

    def test_recovery_preserves_column_order(self, monkeypatch):
        """The spliced solo results must land in parameter order, not run order."""
        self._fail_when(monkeypatch, lambda p: len(p) > 1)
        with pytest.warns(UserWarning, match="integrated on its own"):
            got = self._sim().compute_all_sensitivities(
                **self._RUN, params=["kr", "kf"], chunk_size=2, n_workers=1
            )
        ref = self._sim().compute_all_sensitivities(
            **self._RUN, params=["kr", "kf"], chunk_size=1, n_workers=1
        )
        assert list(got.sensitivity_params) == ["kr", "kf"]
        np.testing.assert_allclose(got.sensitivities, ref.sensitivities, rtol=1e-9, atol=1e-12)

    def test_unresolvable_column_is_named_with_the_survivors(self, monkeypatch):
        """One column fails alone → raise, but say which column, which are fine,
        that the scope is the chunk, and what to re-run."""
        self._fail_when(monkeypatch, lambda p: "kr" in p)
        with pytest.raises(SimulationError) as exc:
            self._sim().compute_all_sensitivities(
                **self._RUN, params=list(self._PARAMS), chunk_size=2, n_workers=1
            )
        msg = str(exc.value)
        assert "['kr'] also fail on their own" in msg
        assert "['kf'] integrated fine alone" in msg
        assert "issue #243" in msg
        # A copy-pasteable next step that keeps the columns that do work.
        assert "params=['kf']" in msg
        # The CVODE message is still there, and still the cause.
        assert "flag=-4" in msg
        assert isinstance(exc.value.__cause__, SimulationError)

    def test_single_column_chunk_skips_the_retry(self, monkeypatch):
        """``chunk_size=1`` already ran the column alone — there is no grouping
        to blame and no second run to make."""
        seen = self._fail_when(monkeypatch, lambda p: "kr" in p)
        with pytest.raises(SimulationError, match=r"\['kr'\] also fail on their own"):
            self._sim().compute_all_sensitivities(
                **self._RUN, params=list(self._PARAMS), chunk_size=1, n_workers=1
            )
        assert seen.count(["kr"]) == 1, "the failing column must not be re-run"

    def test_wide_chunk_reports_that_it_did_not_diagnose(self, monkeypatch):
        """The retry costs one solve per column of a *failing* chunk, so past a
        couple of dozen columns it is named rather than run — and the message
        must not then claim to know which column is at fault."""
        monkeypatch.setattr(bngsim.Simulator, "_MAX_COLUMNS_FOR_SOLO_RETRY", 1)
        seen = self._fail_when(monkeypatch, lambda p: len(p) > 1)
        with pytest.raises(SimulationError) as exc:
            self._sim().compute_all_sensitivities(
                **self._RUN, params=list(self._PARAMS), chunk_size=2, n_workers=1
            )
        msg = str(exc.value)
        assert "were NOT retried individually" in msg
        assert "chunk_size=1" in msg
        assert seen == [["kf", "kr"]], "no solo runs should have been attempted"

    def test_parallel_path_recovers_too(self, monkeypatch):
        """Same treatment when the chunks ran on a ThreadPoolExecutor: only the
        failed chunk is retried, and the chunks that were fine are untouched."""
        params = ["k1", "k2", "scale", "eps"]
        self._fail_when(monkeypatch, lambda p: p == ["k1", "k2"])
        sim = bngsim.Simulator(bngsim.Model.from_net(_get_expr_chain_net()), method="ode")
        with pytest.warns(UserWarning, match="integrated on its own"):
            got = sim.compute_all_sensitivities(
                **self._RUN, params=params, chunk_size=2, n_workers=2
            )
        assert list(got.sensitivity_params) == params
        assert got.sensitivities.shape[2] == 4
        assert np.isfinite(got.sensitivities).all()


class TestNonFiniteSensitivityChunkIsAFailure:
    """A NaN column is a failure the solver forgot to report (GH #243).

    On BIOMD0000000044 at ``chunk_size=3`` the ``_lp_v7_n`` column integrates to
    NaN from the first output point on while CVODES returns success — the
    columns sharing its chunk pass the error test on its behalf. That tensor is
    not a partial answer: :meth:`Result.gradient` contracts the whole parameter
    axis, so one dead column NaNs every entry of the gradient. Left unchecked it
    made the *silent* outcome the one you get at some chunk sizes and the loud
    one at others.
    """

    def _chunk_result(self):
        m = bngsim.Model.from_net(_get_reversible_net())
        sim = bngsim.Simulator(m, method="ode")
        r = sim.compute_all_sensitivities(
            t_span=(0.0, 10.0), n_points=11, params=["kf", "kr"], chunk_size=2, n_workers=1
        )
        assert np.isfinite(r.sensitivities).all()
        return r

    def test_clean_chunk_passes(self):
        r = self._chunk_result()
        bngsim.Simulator._raise_if_nonfinite_sensitivities(r, ["kf", "kr"])

    def test_nan_column_raises_and_is_named(self):
        r = self._chunk_result()
        r._sensitivities = np.array(r.sensitivities, copy=True)
        r._sensitivities[1:, :, 1] = np.nan
        with pytest.raises(SimulationError) as exc:
            bngsim.Simulator._raise_if_nonfinite_sensitivities(r, ["kf", "kr"])
        assert "['kr']" in str(exc.value)
        assert "non-finite" in str(exc.value)

    def test_infinite_column_raises_too(self):
        r = self._chunk_result()
        r._sensitivities = np.array(r.sensitivities, copy=True)
        r._sensitivities[-1, 0, 0] = np.inf
        with pytest.raises(SimulationError, match=r"\['kf'\]"):
            bngsim.Simulator._raise_if_nonfinite_sensitivities(r, ["kf", "kr"])

    def test_deliberately_refused_ar_row_is_not_a_failure(self):
        """GH #221 NaNs an assignment-rule row on purpose and records it in
        ``ar_sensitivity_refused``; that row must not be read as a blow-up."""
        r = self._chunk_result()
        r._sensitivities = np.array(r.sensitivities, copy=True)
        refused_species = r.species_names[0]
        r._sensitivities[:, 0, :] = np.nan
        r._ar_sens_refused = frozenset({refused_species})
        bngsim.Simulator._raise_if_nonfinite_sensitivities(r, ["kf", "kr"])

        # ...but a NaN OUTSIDE the refused row is still a failure.
        r._sensitivities[2, 1, 0] = np.nan
        with pytest.raises(SimulationError, match=r"\['kf'\]"):
            bngsim.Simulator._raise_if_nonfinite_sensitivities(r, ["kf", "kr"])


_BIOMD44 = (
    Path(__file__).resolve().parent.parent.parent
    / "parity_checks"
    / "rr_parity"
    / "models"
    / "BIOMD0000000044"
    / "BIOMD0000000044_url.xml"
)


@pytest.mark.skipif(not _BIOMD44.exists(), reason="BIOMD0000000044 SBML not present")
def test_biomd44_resolves_every_column_at_every_chunk_size():
    """The model that filed GH #243, end to end.

    Three verdicts, in order. ``chunk_size`` 1, 2 and 4 raised a bare CVODE
    message while 3 returned a tensor whose ``_lp_v7_n`` column was silently
    all-NaN — the grouping decided whether an answer existed. #243 made the
    verdict grouping-independent: every chunk size then reported the same
    marginal column rather than one of them handing back a dead column dressed
    as an answer.

    GH #310 removed the cause. ``_lp_v7_n`` is a **Hill exponent**, and the
    column was unresolvable only because ``∂/∂n base^n = base^n·ln(base)`` was
    ``NaN`` wherever the base was zero. With that guarded at its limit the
    column integrates like any other, so what this pins now is the third
    verdict: all 18 columns resolve, at every chunk size, to the same numbers —
    and the formerly-marginal one is checked against finite differences, since
    "it no longer raises" is not by itself evidence that it is right.

    18, not 23: issue #329 refuses the five slots a model **function** owns
    (``vin``, ``v2``, ``v3``, ``v7``, ``_vd_v7_unified`` — this model's rate
    laws, bound to their own parameter rows). The engine rewrites those from the
    function's expression before every derivative evaluation, so they were
    columns of exact zeros padding a claim about columns that resolve. They are
    dropped here the same way the unwritable compartment sizes already were,
    and for the same reason: not a coordinate any write can move.
    """
    m0 = bngsim.Model.load(str(_BIOMD44))
    refused = set(m0.unwritable_compartment_size_params)
    refused |= m0._internal_param_names() & set(m0.function_names)
    names = [p for p in m0.param_names if p not in refused]
    assert "_lp_v7_n" in names and len(names) == 18

    run = dict(params=names, n_workers=1, rtol=1e-10, atol=1e-12)
    reference = None
    for chunk_size in (1, 2, 3, 4):
        sim = bngsim.Simulator(bngsim.Model.load(str(_BIOMD44)), method="ode")
        got = sim.compute_all_sensitivities((0.0, 10.0), 6, chunk_size=chunk_size, **run)
        sens = np.asarray(got.sensitivities)
        assert list(got.sensitivity_params) == names
        assert np.isfinite(sens).all(), f"chunk_size={chunk_size}"
        if reference is None:
            reference = (sens, list(got.sensitivity_params))
        else:
            assert list(got.sensitivity_params) == reference[1]
            # Not bit-for-bit: the columns in a chunk share one CVODES error
            # test, so the grouping still moves the adaptive step sequence. What
            # must not depend on it any more is the answer.
            np.testing.assert_allclose(sens, reference[0], rtol=1e-3, atol=1e-5)

    # The exponent column against central finite differences of the states.
    sens, params = reference
    column = sens[:, :, params.index("_lp_v7_n")]
    assert np.abs(column).max() > 1.0

    nominal = m0.get_param("_lp_v7_n")
    step = 1e-6 * max(1.0, abs(nominal))
    legs = []
    for offset in (+step, -step):
        perturbed = bngsim.Model.load(str(_BIOMD44))
        perturbed.set_param("_lp_v7_n", nominal + offset)
        leg = bngsim.Simulator(perturbed, method="ode").run(
            t_span=(0.0, 10.0), n_points=6, rtol=1e-12, atol=1e-14
        )
        legs.append(np.asarray(leg.species))
    fd = (legs[0] - legs[1]) / (2 * step)

    # Loose on the FD side only. This law is sharp in ``n``, so the step cannot be
    # opened up to damp the noise the way a milder one could: each leg is a solver
    # output carrying its own integration error, and the quotient divides that by
    # 2h. Two orders of margin over the observed agreement, which is all the claim
    # needs — a NaN or a wrong-by-orders column fails this by a mile.
    np.testing.assert_allclose(column, fd, rtol=1e-2, atol=1e-4)
