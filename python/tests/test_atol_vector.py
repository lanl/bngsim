"""Per-species absolute tolerance — issue #196.

``Simulator.run`` took a scalar ``atol`` and the core set it with
``CVodeSStolerances``. For a model whose species span decades that is one
number asked to mean two incompatible things: the tolerance the smallest
species needs makes the model unintegrable, and the tolerance the model can
integrate at leaves the smallest species unresolved. CVODE's answer is
``CVodeSVtolerances``, a per-species absolute tolerance vector — which bngsim
already used one axis over, on the sensitivity columns.

``wide_dynamic_range.net`` is the minimum reproducer of the half that
reproduces from a model alone (the issue is explicit that the *unintegrable*
half needs a full pre-equilibration protocol): two decoupled decays nine
decades apart in magnitude and three in rate, where the default scalar
``atol=1e-8`` sits an order of magnitude ABOVE the small species for the whole
run. That species is then not error-controlled at all, and comes back
**negative** where the analytical answer is 6.7e-12.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._atol import derive_atol, normalize_atol_vector

# Fast/small vs slow/large — see the module docstring and the .net header.
_T_END = 0.5
_N_POINTS = 6
_S0, _KS = 1.0e-9, 10.0
_B0, _KB = 1.0e01, 0.01


@pytest.fixture
def wide_net(data_dir: Path) -> Path:
    """Path to wide_dynamic_range.net."""
    return data_dir / "wide_dynamic_range.net"


@pytest.fixture
def wide_sim(wide_net: Path):
    """Simulator over the wide-dynamic-range model, plus its species index."""
    model = bngsim.Model.from_net(str(wide_net))
    return bngsim.Simulator(model, method="ode"), model


def _exact(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _S0 * np.exp(-_KS * times), _B0 * np.exp(-_KB * times)


def _rel_err(sim, model, **run_kwargs) -> tuple[float, float, int]:
    """Run and return (relative error in S, relative error in B, n_steps)."""
    model.reset()
    result = sim.run(t_span=(0.0, _T_END), n_points=_N_POINTS, **run_kwargs)
    names = model.species_names
    s = result.species[:, names.index("S()")]
    b = result.species[:, names.index("B()")]
    exact_s, exact_b = _exact(result.time)
    return (
        float(np.max(np.abs(s - exact_s) / exact_s)),
        float(np.max(np.abs(b - exact_b) / exact_b)),
        int(result.solver_stats.get("n_steps", 0)),
    )


# ─── The defect the issue is about ────────────────────────────────────────────


def test_scalar_atol_cannot_resolve_the_small_species(wide_sim):
    """The premise: no scalar atol is a tolerance for both ends of this model."""
    sim, model = wide_sim

    # Loose enough to integrate cheaply — and S is unresolved, by a wide margin.
    err_s_loose, err_b_loose, _ = _rel_err(sim, model, atol=1e-8)
    assert err_s_loose > 0.1, "expected the small species to be badly wrong at atol=1e-8"
    assert err_b_loose < 1e-6, "the large species is fine at atol=1e-8 — that is the conflict"

    # Tight enough for S — and now every species pays for it.
    _, _, n_steps_loose = _rel_err(sim, model, atol=1e-8)
    err_s_tight, _, n_steps_tight = _rel_err(sim, model, atol=1e-17)
    assert err_s_tight < 1e-3
    assert n_steps_tight > n_steps_loose


def test_per_species_atol_resolves_both_ends(wide_sim):
    """The fix: one vector states both tolerances, and both are met."""
    sim, model = wide_sim
    n = model.n_species
    atol = [1e-17] + [1e-8] * (n - 1)

    err_s, err_b, _ = _rel_err(sim, model, atol=atol)
    assert err_s < 1e-3, "S is error-controlled against its own magnitude now"
    assert err_b < 1e-6, "...and B is unharmed"


def test_small_species_goes_negative_under_a_scalar_atol(wide_sim):
    """Concretely: unresolved does not mean 'slightly off'."""
    sim, model = wide_sim
    idx = model.species_names.index("S()")

    model.reset()
    scalar = sim.run(t_span=(0.0, _T_END), n_points=_N_POINTS, atol=1e-8)
    assert scalar.species[-1, idx] < 0.0

    model.reset()
    vector = sim.run(
        t_span=(0.0, _T_END),
        n_points=_N_POINTS,
        atol=[1e-17] + [1e-8] * (model.n_species - 1),
    )
    assert vector.species[-1, idx] == pytest.approx(_S0 * np.exp(-_KS * _T_END), rel=1e-3)


# ─── The vector actually reaches CVODE ────────────────────────────────────────


def test_constant_vector_reproduces_the_scalar(wide_sim):
    """A constant vector is the same tolerance, and behaves like it.

    Not asserted bit-for-bit: ``cvEwtSetSS`` scales then adds a constant while
    ``cvEwtSetSV`` takes one fused ``N_VLinearSum``, and only the second is
    FMA-contractable — a ~1 ulp difference in the error weights.
    """
    sim, model = wide_sim
    err_s_scalar, err_b_scalar, n_scalar = _rel_err(sim, model, atol=1e-17)
    err_s_vec, err_b_vec, n_vec = _rel_err(sim, model, atol=[1e-17] * model.n_species)

    assert err_s_vec == pytest.approx(err_s_scalar, rel=0.05)
    assert err_b_vec == pytest.approx(err_b_scalar, rel=0.05)
    assert abs(n_vec - n_scalar) <= 0.1 * n_scalar


def test_tolerance_change_invalidates_the_warm_cvode_cache(wide_sim):
    """The warm path reuses CVODE memory across runs; tolerances are part of it.

    ``CVodeReInit`` does not touch tolerances, so a fingerprint that ignored the
    vector would silently integrate the second run at the first run's
    tolerances. Alternating proves each run gets its own.
    """
    sim, model = wide_sim
    vec = [1e-17] + [1e-8] * (model.n_species - 1)
    tighter = [1e-20] + [1e-8] * (model.n_species - 1)

    scalar_first = _rel_err(sim, model, atol=1e-8)
    vector_first = _rel_err(sim, model, atol=vec)
    scalar_again = _rel_err(sim, model, atol=1e-8)
    vector_again = _rel_err(sim, model, atol=vec)
    tighter_run = _rel_err(sim, model, atol=tighter)

    assert scalar_again == scalar_first
    assert vector_again == vector_first
    assert vector_first[2] != scalar_first[2]
    # A different vector is a different configuration, not a cache hit.
    assert tighter_run[2] != vector_first[2]


def test_sensitivity_atol_is_built_from_the_per_species_vector(wide_net):
    """The sensitivity columns inherit the state axis's per-species tolerance.

    Discriminating by construction: passing a vector leaves ``opts.atol`` at the
    Simulator's scalar (1e-8), so a sensitivity setup that still read the scalar
    would hold ``atolS`` to 1e-8 while the state axis ran at 1e-17. It does not
    — a constant 1e-17 vector reproduces the scalar-1e-17 sensitivity, and both
    are orders of magnitude better than scalar 1e-8.
    """
    model = bngsim.Model.from_net(str(wide_net))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kS"])
    idx = model.species_names.index("S()")

    def sens_err(**kwargs) -> float:
        model.reset()
        result = sim.run(t_span=(0.0, _T_END), n_points=_N_POINTS, **kwargs)
        got = result.sensitivities[:, idx, 0]
        # d/dkS [S0 exp(-kS t)] = -t S0 exp(-kS t)
        exact = -result.time * _S0 * np.exp(-_KS * result.time)
        return float(np.max(np.abs(got - exact) / np.abs(exact[1:]).min()))

    loose = sens_err(atol=1e-8)
    scalar_tight = sens_err(atol=1e-17)
    vector_tight = sens_err(atol=[1e-17] * model.n_species)

    assert scalar_tight < 1e-3 < loose
    assert vector_tight == pytest.approx(scalar_tight, rel=0.5)


# ─── The length/value contract ────────────────────────────────────────────────


def test_wrong_length_is_rejected_not_broadcast(wide_sim):
    sim, model = wide_sim
    with pytest.raises(ValueError, match=r"2 entries but the model has 4 species"):
        sim.run(t_span=(0.0, _T_END), n_points=_N_POINTS, atol=[1e-8, 1e-8])
    with pytest.raises(ValueError, match=r"broadcast or truncated"):
        sim.run(t_span=(0.0, _T_END), n_points=_N_POINTS, atol=[1e-8] * (model.n_species + 1))


def test_length_error_names_the_ordering(wide_sim):
    """The message has to say what the ordering was supposed to be."""
    sim, model = wide_sim
    with pytest.raises(ValueError, match=r"Model\.species_names"):
        sim.run(t_span=(0.0, _T_END), n_points=_N_POINTS, atol=[1e-8])


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_unusable_entries_are_rejected(wide_sim, bad):
    sim, model = wide_sim
    atol = [1e-8] * model.n_species
    atol[1] = bad
    with pytest.raises(ValueError, match=r"entry 1 \('B\(\)'\)"):
        sim.run(t_span=(0.0, _T_END), n_points=_N_POINTS, atol=atol)


def test_two_dimensional_atol_is_rejected(wide_sim):
    sim, model = wide_sim
    with pytest.raises(ValueError, match=r"must be 1-D"):
        sim.run(t_span=(0.0, _T_END), n_points=_N_POINTS, atol=np.ones((2, 2)))


def test_unknown_token_is_rejected(wide_sim):
    sim, _ = wide_sim
    with pytest.raises(ValueError, match=r"unknown atol token 'tight'"):
        sim.run(t_span=(0.0, _T_END), n_points=_N_POINTS, atol="tight")


def test_numpy_array_is_accepted(wide_sim):
    sim, model = wide_sim
    atol = np.full(model.n_species, 1e-17)
    err_s, _, _ = _rel_err(sim, model, atol=atol)
    assert err_s < 1e-3


# ─── atol="auto" ──────────────────────────────────────────────────────────────


def test_auto_atol_follows_the_documented_rule(wide_sim):
    """``rtol * max(|y_i|, floor)``, floor = the smallest positive magnitude."""
    sim, model = wide_sim
    model.reset()
    got = sim.auto_atol()

    y = np.abs(model.get_state())
    floor = y[y > 0].min()
    assert got == pytest.approx(1e-8 * np.maximum(y, floor))
    assert len(got) == model.n_species
    # The species that has a magnitude gets its own; the zero-initialized ones
    # get the smallest magnitude the model exhibits.
    names = model.species_names
    assert got[names.index("B()")] == pytest.approx(1e-8 * _B0)
    assert got[names.index("S()")] == pytest.approx(1e-8 * _S0)
    assert got[names.index("Sd()")] == pytest.approx(1e-8 * _S0)


def test_auto_atol_honours_rtol_and_floor(wide_sim):
    sim, model = wide_sim
    model.reset()
    assert sim.auto_atol(rtol=1e-6) == pytest.approx(100.0 * sim.auto_atol(rtol=1e-8))
    floored = sim.auto_atol(floor=1.0)
    assert floored.min() == pytest.approx(1e-8)


def test_auto_run_matches_the_vector_it_derives(wide_sim):
    sim, model = wide_sim
    model.reset()
    derived = list(sim.auto_atol())

    err_auto = _rel_err(sim, model, atol="auto")
    err_explicit = _rel_err(sim, model, atol=derived)
    assert err_auto == err_explicit


def test_auto_resolves_the_small_species(wide_sim):
    """The point of the derivation, not just its arithmetic."""
    sim, model = wide_sim
    err_s, err_b, _ = _rel_err(sim, model, atol="auto")
    assert err_s < 1e-3
    assert err_b < 1e-6


# ─── set_tolerances ───────────────────────────────────────────────────────────


def test_set_tolerances_takes_a_vector(wide_sim):
    sim, model = wide_sim
    sim.set_tolerances(1e-8, [1e-17] + [1e-8] * (model.n_species - 1))
    err_s, _, _ = _rel_err(sim, model)
    assert err_s < 1e-3


def test_set_tolerances_takes_auto(wide_sim):
    sim, model = wide_sim
    model.reset()
    sim.set_tolerances(1e-8, "auto")
    err_s, _, _ = _rel_err(sim, model)
    assert err_s < 1e-3


def test_a_scalar_clears_a_previously_set_vector(wide_sim):
    """Otherwise the call appears to set the tolerance and changes nothing."""
    sim, model = wide_sim
    sim.set_tolerances(1e-8, [1e-17] * model.n_species)
    assert _rel_err(sim, model)[0] < 1e-3

    sim.set_tolerances(1e-8, 1e-8)
    assert _rel_err(sim, model)[0] > 0.1


def test_run_argument_overrides_the_configured_vector(wide_sim):
    sim, model = wide_sim
    sim.set_tolerances(1e-8, [1e-17] * model.n_species)
    assert _rel_err(sim, model, atol=1e-8)[0] > 0.1


def test_set_tolerances_rejects_a_wrong_length_vector(wide_sim):
    sim, _ = wide_sim
    with pytest.raises(ValueError, match=r"set_tolerances\(atol=\.\.\.\)"):
        sim.set_tolerances(1e-8, [1e-8, 1e-8])


# ─── Every other entry point that takes atol ──────────────────────────────────


def test_run_batch_takes_a_vector(wide_sim):
    sim, model = wide_sim
    model.reset()
    atol = [1e-17] + [1e-8] * (model.n_species - 1)
    results = sim.run_batch(
        (0.0, _T_END),
        _N_POINTS,
        params=[{"kS": 10.0}, {"kS": 20.0}],
        atol=atol,
    )
    idx = model.species_names.index("S()")
    for result, k in zip(results, (10.0, 20.0), strict=True):
        assert result.species[-1, idx] == pytest.approx(_S0 * np.exp(-k * _T_END), rel=1e-3)


def test_run_batch_rejects_a_wrong_length_vector(wide_sim):
    sim, _ = wide_sim
    with pytest.raises(ValueError, match=r"run_batch\(atol=\.\.\.\)"):
        sim.run_batch((0.0, _T_END), _N_POINTS, params=[{"kS": 10.0}], atol=[1e-8])


def test_parameter_scan_takes_a_vector(wide_sim):
    sim, model = wide_sim
    model.reset()
    results = sim.parameter_scan(
        "kS",
        par_scan_vals=[10.0, 20.0],
        t_span=(0.0, _T_END),
        n_points=_N_POINTS,
        atol=[1e-17] + [1e-8] * (model.n_species - 1),
    )
    idx = model.species_names.index("S()")
    for result, k in zip(results, (10.0, 20.0), strict=True):
        assert result.species[-1, idx] == pytest.approx(_S0 * np.exp(-k * _T_END), rel=1e-3)


def test_parameter_scan_freezes_auto_at_the_invocation_state(wide_sim):
    """Every point of a scan is held to the same tolerance.

    ``kS`` does not move any initial condition, so a per-point derivation would
    give the same vector here anyway — what this pins is that the scan and an
    explicit vector of the invocation-state derivation agree, i.e. that the
    resolution happens once, up front.
    """
    sim, model = wide_sim
    model.reset()
    frozen = list(sim.auto_atol())

    auto = sim.parameter_scan(
        "kS", par_scan_vals=[10.0, 20.0], t_span=(0.0, _T_END), n_points=_N_POINTS, atol="auto"
    )
    model.reset()
    explicit = sim.parameter_scan(
        "kS", par_scan_vals=[10.0, 20.0], t_span=(0.0, _T_END), n_points=_N_POINTS, atol=frozen
    )
    for a, b in zip(auto, explicit, strict=True):
        assert np.array_equal(a.species, b.species)


def test_compute_all_sensitivities_takes_a_vector(wide_net):
    model = bngsim.Model.from_net(str(wide_net))
    sim = bngsim.Simulator(model, method="ode")
    result = sim.compute_all_sensitivities(
        t_span=(0.0, _T_END),
        n_points=_N_POINTS,
        params=["kS", "kB"],
        atol=[1e-17] * model.n_species,
    )
    idx = model.species_names.index("S()")
    col = result.sensitivity_params.index("kS")
    exact = -result.time * _S0 * np.exp(-_KS * result.time)
    got = result.sensitivities[:, idx, col]
    assert np.allclose(got[1:], exact[1:], rtol=1e-3)


def test_steady_state_takes_a_vector(wide_net):
    model = bngsim.Model.from_net(str(wide_net))
    sim = bngsim.Simulator(model, method="ode")
    scalar = sim.steady_state(atol=1e-10)
    model.reset()
    vector = sim.steady_state(atol=[1e-10] * model.n_species)
    assert vector.converged == scalar.converged
    assert np.allclose(vector.concentrations, scalar.concentrations, rtol=1e-6, atol=1e-12)


def test_steady_state_rejects_a_wrong_length_vector(wide_net):
    model = bngsim.Model.from_net(str(wide_net))
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(ValueError, match=r"steady_state\(atol=\.\.\.\)"):
        sim.steady_state(atol=[1e-8, 1e-8])


def test_steady_state_batch_takes_a_vector(wide_net):
    model = bngsim.Model.from_net(str(wide_net))
    sim = bngsim.Simulator(model, method="ode")
    results = sim.steady_state_batch([{"kS": 10.0}, {"kS": 20.0}], atol=[1e-10] * model.n_species)
    assert len(results) == 2
    assert all(r.converged for r in results)


def test_run_until_takes_a_vector(wide_sim):
    sim, model = wide_sim
    model.reset()
    result = sim.run_until(_T_END, n_points=_N_POINTS, atol=[1e-17] * model.n_species)
    idx = model.species_names.index("S()")
    assert result.species[-1, idx] == pytest.approx(_S0 * np.exp(-_KS * _T_END), rel=1e-3)


# ─── The core options object ──────────────────────────────────────────────────


def test_solver_options_exposes_atol_vec():
    from bngsim._bngsim_core import SolverOptions

    opts = SolverOptions()
    assert list(opts.atol_vec) == []
    opts.atol_vec = [1e-9, 1e-10]
    assert list(opts.atol_vec) == [1e-9, 1e-10]


def test_core_rejects_a_wrong_length_vector(wide_net):
    """The C++ guard holds independently of the Python one."""
    from bngsim._bngsim_core import CvodeSimulator, SolverOptions, TimeSpec

    model = bngsim.Model.from_net(str(wide_net))
    core_sim = CvodeSimulator(model._core)
    times = TimeSpec()
    times.t_start, times.t_end, times.n_points = 0.0, _T_END, _N_POINTS
    opts = SolverOptions()
    opts.atol_vec = [1e-8, 1e-8]
    with pytest.raises(ValueError, match=r"2 entries but the model has 4 species"):
        core_sim.run(times, opts)


def test_core_set_tolerances_overloads(wide_net):
    from bngsim._bngsim_core import CvodeSimulator

    model = bngsim.Model.from_net(str(wide_net))
    core_sim = CvodeSimulator(model._core)
    core_sim.set_tolerances(1e-8, 1e-8)  # scalar overload
    core_sim.set_tolerances(1e-8, [1e-8] * model.n_species)  # vector overload
    with pytest.raises(ValueError, match=r"set_tolerances\(\)"):
        core_sim.set_tolerances(1e-8, [1e-8])


# ─── EvaluationSpec / SED-ML ──────────────────────────────────────────────────


def test_evaluation_spec_round_trips_a_vector(wide_net):
    """A cluster worker has to receive the same tolerance the driver meant."""
    spec = bngsim.EvaluationSpec(
        model_source=str(wide_net),
        model_format="net",
        t_span=(0.0, _T_END),
        n_points=_N_POINTS,
        atol=[1e-17, 1e-8, 1e-8, 1e-8],
    )
    assert spec.atol == (1e-17, 1e-8, 1e-8, 1e-8)  # canonicalized, hashable
    assert bngsim.EvaluationSpec.from_json(spec.to_json()) == spec

    result = spec.evaluate()
    idx = result.species_names.index("S()")
    assert result.species[-1, idx] == pytest.approx(_S0 * np.exp(-_KS * _T_END), rel=1e-3)


def test_evaluation_spec_carries_auto(wide_net):
    spec = bngsim.EvaluationSpec(
        model_source=str(wide_net),
        model_format="net",
        t_span=(0.0, _T_END),
        n_points=_N_POINTS,
        atol="auto",
    )
    assert bngsim.EvaluationSpec.from_json(spec.to_json()).atol == "auto"
    result = spec.evaluate()
    idx = result.species_names.index("S()")
    assert result.species[-1, idx] == pytest.approx(_S0 * np.exp(-_KS * _T_END), rel=1e-3)


@pytest.mark.parametrize("atol", [[1e-17, 1e-8, 1e-8, 1e-8], "auto"])
def test_sedml_export_refuses_a_non_scalar_atol(wide_net, atol):
    """KISAO:0000211 is one number; writing one would describe a different run."""
    from bngsim.convert import write_sedml

    spec = bngsim.EvaluationSpec(
        model_source=str(wide_net), model_format="net", t_span=(0.0, _T_END), atol=atol
    )
    with pytest.raises(bngsim.ConversionError, match=r"per-species absolute tolerance"):
        write_sedml(spec)


# ─── The derivation and validation helpers ────────────────────────────────────


def test_derive_atol_rule():
    got = derive_atol([2.0, 0.0, 4.0], 1e-6)
    # floor = 2.0 (smallest positive), so the zero entry rides at 2.0's scale.
    assert got == pytest.approx([2e-6, 2e-6, 4e-6])


def test_derive_atol_uses_magnitude_not_sign():
    assert derive_atol([-3.0], 1e-6) == pytest.approx([3e-6])


def test_derive_atol_all_zero_state_falls_back_to_one():
    assert derive_atol([0.0, 0.0], 1e-8) == pytest.approx([1e-8, 1e-8])


def test_derive_atol_explicit_floor():
    assert derive_atol([1e-3, 10.0], 1e-8, floor=1.0) == pytest.approx([1e-8, 1e-7])


@pytest.mark.parametrize("floor", [0.0, -1.0, float("nan")])
def test_derive_atol_rejects_an_unusable_floor(floor):
    with pytest.raises(ValueError, match=r"floor must be finite and > 0"):
        derive_atol([1.0], 1e-8, floor=floor)


@pytest.mark.parametrize("rtol", [0.0, -1e-8, float("inf")])
def test_derive_atol_rejects_an_unusable_rtol(rtol):
    with pytest.raises(ValueError, match=r"rtol must be finite and > 0"):
        derive_atol([1.0], rtol)


def test_derive_atol_rejects_a_non_finite_state():
    with pytest.raises(ValueError, match=r"NaN or inf"):
        derive_atol([1.0, float("nan")], 1e-8)


def test_normalize_atol_vector_returns_plain_floats():
    got = normalize_atol_vector(np.array([1e-8, 1e-9]), 2)
    assert got == [1e-8, 1e-9]
    assert all(type(v) is float for v in got)


def test_normalize_atol_vector_allows_zero():
    """CVODE takes abstol == 0 (pure relative control); only negatives are out."""
    assert normalize_atol_vector([0.0, 1e-8], 2) == [0.0, 1e-8]


def test_normalize_atol_vector_names_the_species_it_can():
    with pytest.raises(ValueError, match=r"\[a, b, c, \.\.\.\]"):
        normalize_atol_vector([1e-8], 4, ["a", "b", "c", "d"])
