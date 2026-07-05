"""bngsim.vivarium — the optional Vivarium process shell (GH #102 Stage 0).

Covers the thin vivarium-core ``Process`` wrapper over the reaction kernel:
ports_schema / next_update / calculate_timestep, construction from a model or a
pre-built kernel, and the headline parity invariant — driving a model through a
Vivarium ``Engine`` reproduces the same model advanced step-wise through the
bare :class:`bngsim.ReactionKernel` (a single-process composite is just the
kernel, so they must agree exactly).

Skipped entirely when vivarium-core is not installed.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

pytest.importorskip("vivarium", reason="vivarium-core not installed (optional extra)")

from bngsim.vivarium import BngsimProcess  # noqa: E402
from vivarium.core.engine import Engine  # noqa: E402


def _chain(n: int = 5, *, observables: bool = True):
    b = ModelBuilder()
    sp = []
    for i in range(n):
        b.add_parameter(f"k{i}", 0.1 * (i + 1))
        sp.append(b.add_species(f"S{i}", 100.0 if i == 0 else 0.0))
    for i in range(n - 1):
        b.add_reaction([sp[i]], [sp[i + 1]], "elementary", f"k{i}")
    if observables:
        b.add_observable("Total", [(i, 1.0) for i in range(n)])
    b.set_compute_conservation_laws(False)
    return b.build()


def _model(n: int = 5, **kw) -> bngsim.Model:
    return bngsim.Model(_core=_chain(n, **kw))


def _run_engine(proc: BngsimProcess, total_time: float) -> dict:
    engine = Engine(
        processes={"bngsim": proc},
        topology={"bngsim": {"species": ("species",), "observables": ("observables",)}},
        initial_state=proc.initial_state(),
    )
    engine.update(total_time)
    return engine.emitter.get_timeseries()


class TestAvailability:
    def test_has_vivarium_flag_true(self):
        # This suite only runs when vivarium is importable, so the flag must agree.
        assert bngsim.HAS_VIVARIUM is True
        assert bngsim.capabilities()["features"]["vivarium"] is True


class TestConstruction:
    def test_from_model(self):
        proc = BngsimProcess({"model": _model(), "time_step": 2.0})
        assert isinstance(proc.kernel, bngsim.ReactionKernel)
        assert isinstance(proc.simulator, bngsim.Simulator)

    def test_from_kernel(self):
        kernel = bngsim.ReactionKernel(_model())
        proc = BngsimProcess({"kernel": kernel, "time_step": 1.0})
        assert proc.kernel is kernel

    def test_requires_model_or_kernel(self):
        with pytest.raises(ValueError):
            BngsimProcess({"time_step": 1.0})

    def test_rejects_non_model(self):
        with pytest.raises(TypeError):
            BngsimProcess({"model": object()})

    def test_rejects_non_kernel(self):
        with pytest.raises(TypeError):
            BngsimProcess({"kernel": object()})


class TestPortsSchema:
    def test_species_port_accumulates_with_initial_defaults(self):
        proc = BngsimProcess({"model": _model(5)})
        schema = proc.ports_schema()
        assert set(schema) == {"species", "observables"}
        sp = schema["species"]
        assert list(sp) == [f"S{i}" for i in range(5)]
        assert sp["S0"] == {"_default": 100.0, "_updater": "accumulate", "_emit": True}
        assert sp["S1"]["_default"] == 0.0
        # Deltas compose additively under operator splitting → accumulate.
        assert all(v["_updater"] == "accumulate" for v in sp.values())

    def test_observables_port_is_set_updater(self):
        proc = BngsimProcess({"model": _model(5, observables=True)})
        obs = proc.ports_schema()["observables"]
        assert list(obs) == ["Total"]
        assert obs["Total"]["_updater"] == "set"
        assert obs["Total"]["_default"] == pytest.approx(100.0)

    def test_no_observables_yields_empty_port(self):
        proc = BngsimProcess({"model": _model(4, observables=False)})
        assert proc.ports_schema()["observables"] == {}

    def test_initial_state_matches_defaults(self):
        proc = BngsimProcess({"model": _model(5)})
        init = proc.initial_state()
        assert init["species"]["S0"] == 100.0
        assert init["observables"]["Total"] == pytest.approx(100.0)


class TestHooks:
    def test_calculate_timestep_returns_configured(self):
        proc = BngsimProcess({"model": _model(), "time_step": 0.25})
        assert proc.calculate_timestep(None) == 0.25
        assert proc.calculate_timestep({"species": {}}) == 0.25

    def test_next_update_returns_species_deltas(self):
        proc = BngsimProcess({"model": _model(5), "time_step": 5.0})
        states = proc.initial_state()
        update = proc.next_update(5.0, states)
        assert set(update) == {"species", "observables"}
        # S0 (the only seeded species) must drain → negative delta.
        assert update["species"]["S0"] < 0.0
        # Pure transfer chain conserves mass: deltas sum to ~0.
        assert sum(update["species"].values()) == pytest.approx(0.0, abs=1e-6)

    def test_next_update_reads_store_as_source_of_truth(self):
        # An externally-modified store value must be picked up via set_state.
        proc = BngsimProcess({"model": _model(4, observables=False), "time_step": 1.0})
        states = {"species": {"S0": 0.0, "S1": 0.0, "S2": 0.0, "S3": 50.0}}
        update = proc.next_update(1.0, states)
        # Only S3 has mass; S3→(end) drains, downstream none, so S3 delta <= 0
        # and total change is ~0 (conservation).
        assert sum(update["species"].values()) == pytest.approx(0.0, abs=1e-6)
        assert update["species"]["S3"] <= 0.0


class TestEngineParity:
    """A single bngsim process in a composite is just the kernel — the Engine
    trajectory must match a bare ReactionKernel advance loop exactly."""

    def test_engine_matches_kernel_loop(self):
        dt, T, n = 2.0, 20.0, 5
        proc = BngsimProcess({"model": _model(n), "time_step": dt})
        ts = _run_engine(proc, T)
        viv_final = np.array([ts["species"][f"S{i}"][-1] for i in range(n)])

        kernel = bngsim.ReactionKernel(_model(n))
        for _ in range(int(round(T / dt))):
            kernel.advance(dt)

        np.testing.assert_array_equal(viv_final, kernel.get_state())

    def test_engine_observables_match_kernel(self):
        dt, T, n = 2.0, 20.0, 5
        proc = BngsimProcess({"model": _model(n, observables=True), "time_step": dt})
        ts = _run_engine(proc, T)

        kernel = bngsim.ReactionKernel(_model(n, observables=True))
        for _ in range(int(round(T / dt))):
            kernel.advance(dt)

        assert ts["observables"]["Total"][-1] == pytest.approx(kernel.observables()[0])
