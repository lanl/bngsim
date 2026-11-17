"""method="nfsim" refuses a total propensity that is not finite (issues #1818, #1788).

NFsim keeps its total propensity as a running sum, so one non-finite propensity
makes it NaN for the rest of the run. NFsim then takes the system to have
nothing left to fire, and every later row repeats the state it stopped in. That
was returned as a trajectory, with no error.

Two models reach it:

* ``rate_divides_by_emptied_observable`` (RuleWorld/nfsim#115): ``kgap*(1+9*Xp/Xtot)``
  with one X. Every state the model reaches has a finite rate, but NFsim reads
  the rate law while a firing has X out of the observables, where it is 0/0.
  When the vendored NFsim carries the upstream fix this model runs to the end,
  and its tests here are to assert that instead.
* ``nan_rate_law`` (#1788): ``B -> 0`` at ``sqrt(k)``, k = -1, beside ``A -> 0``.
  The NaN is the model's own, which ssa and ode refuse.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import NfsimSession


def _has_nfsim() -> bool:
    return getattr(bngsim, "HAS_NFSIM", False)


pytestmark = pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")

MID_FIRING = "rate_divides_by_emptied_observable.xml"
NAN_LAW = "nan_rate_law.xml"


def _xml(data_dir: Path, name: str) -> str:
    return str(data_dir / "nfsim" / name)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_rate_law_read_mid_firing_is_refused_not_frozen(data_dir: Path, seed: int) -> None:
    # On main each of these seeds returned 501 rows whose last change came at
    # t = 1, 54, 14, 19 and 4 of 500.
    with NfsimSession(_xml(data_dir, MID_FIRING)) as nf:
        nf.initialize(seed=seed)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan") as err:
            nf.simulate(0.0, 500.0, 501)
    # Each propensity is finite again once the firing is done: the message says
    # where the NaN came from, and names no reaction.
    assert "Every reaction's propensity is finite now" in str(err.value)
    assert "RuleWorld/nfsim#115" in str(err.value)


def test_the_models_own_nan_rate_law_is_refused_and_named(data_dir: Path) -> None:
    # On main: A = 100 and B = 10 at every sample, for E[A(1)] = 36.8.
    with NfsimSession(_xml(data_dir, NAN_LAW)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan") as err:
            nf.simulate(0.0, 1.0, 11)
    assert "Non-finite propensity in 1 reaction(s)" in str(err.value)


def test_step_to_refuses_it_too(data_dir: Path) -> None:
    with NfsimSession(_xml(data_dir, NAN_LAW)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan"):
            nf.step_to(1.0)


def test_a_session_that_stopped_refuses_to_go_on_until_initialized(data_dir: Path) -> None:
    # NFsim fired nothing from the point its total went non-finite, so the
    # state is the one it stopped in. Repairing the rate does not make the
    # span between run: the session refuses until it is started again.
    with NfsimSession(_xml(data_dir, NAN_LAW)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan"):
            nf.step_to(1.0)
        nf.set_param("k", 1.0)
        with pytest.raises(
            bngsim.SimulationError, match=r"stopped when step_to\(\) found .* at t = 1 in"
        ):
            nf.simulate(1.0, 2.0, 2)
        with pytest.raises(bngsim.SimulationError, match="initialize"):
            nf.step_to(2.0)
        nf.initialize(seed=1)
        result = nf.simulate(0.0, 1.0, 11)
    # k = 1 now: A and B both decay, and nothing was refused.
    assert result.n_times == 11
    assert np.asarray(result.observables["A"])[-1] < 100


def test_a_session_that_stopped_in_simulate_refuses_step_to(data_dir: Path) -> None:
    with NfsimSession(_xml(data_dir, MID_FIRING)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan"):
            nf.simulate(0.0, 500.0, 501)
        with pytest.raises(bngsim.SimulationError, match=r"stopped when simulate\(\) found"):
            nf.step_to(600.0)


def test_a_restore_does_not_lift_the_stop(data_dir: Path) -> None:
    # restore_concentrations() puts the molecules back and leaves NFsim's
    # clock where the stop left it; only initialize() starts again.
    with NfsimSession(_xml(data_dir, MID_FIRING)) as nf:
        nf.initialize(seed=1)
        nf.save_concentrations("start")
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan"):
            nf.simulate(0.0, 500.0, 501)
        nf.restore_concentrations("start")
        with pytest.raises(bngsim.SimulationError, match="this session stopped"):
            nf.simulate(0.0, 100.0, 101)


def test_a_total_that_is_not_finite_from_the_start_is_refused_at_once(data_dir: Path) -> None:
    # The #1788 model's total is NaN in its initial state: a one-point segment
    # or run is refused too, at its only sample.
    with NfsimSession(_xml(data_dir, NAN_LAW)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match=r"by the sample at t = 0\b"):
            nf.simulate(0.0, 0.0, 1)
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    times = TimeSpec()
    times.t_start, times.t_end, times.n_points = 0.0, 0.0, 1
    with pytest.raises(RuntimeError, match=r"by the sample at t = 0\b"):
        NfsimSimulator(_xml(data_dir, NAN_LAW)).run(times, 1)


@pytest.mark.parametrize("t_start", [0.0, 100.0])
def test_the_time_named_is_the_samples_own(data_dir: Path, t_start: float) -> None:
    # Seed 1's lone X first fires before the sample one time unit in. The
    # message gives that sample's time in the caller's labels, which NFsim's
    # own clock (0 at t_start) is not.
    expected = rf"by the sample at t = {int(t_start) + 1}\b"
    with NfsimSession(_xml(data_dir, MID_FIRING)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match=expected):
            nf.simulate(t_start, t_start + 500.0, 501)
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    times = TimeSpec()
    times.t_start, times.t_end, times.n_points = t_start, t_start + 500.0, 501
    with pytest.raises(RuntimeError, match=expected):
        NfsimSimulator(_xml(data_dir, MID_FIRING)).run(times, 1)


def test_the_one_shot_run_refuses_it_too(data_dir: Path) -> None:
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    times = TimeSpec()
    times.t_start, times.t_end, times.n_points = 0.0, 500.0, 501
    with pytest.raises(RuntimeError, match="NfsimSimulator::run: NFsim's total propensity is nan"):
        NfsimSimulator(_xml(data_dir, MID_FIRING)).run(times, 1)
    times.t_end, times.n_points = 1.0, 11
    with pytest.raises(RuntimeError, match=r"Non-finite propensity in 1 reaction\(s\)"):
        NfsimSimulator(_xml(data_dir, NAN_LAW)).run(times, 1)


def test_the_same_model_with_no_x_firing_runs_to_the_end(data_dir: Path) -> None:
    # The control: with kp = 0 the lone X never fires, the rate law is never
    # read in between, and the run is as it was. RT relaxes to 8*kex/(kex+kgap)
    # = 4 (boost() is 1 with X unphosphorylated).
    finals = []
    for seed in range(1, 21):
        with NfsimSession(_xml(data_dir, MID_FIRING)) as nf:
            nf.set_param("kp", 0.0)
            nf.initialize(seed=seed)
            result = nf.simulate(0.0, 500.0, 501)
        rt = np.asarray(result.observables["RT"])
        assert len(rt) == 501
        assert np.any(rt[400:] != rt[400])  # still changing at the end
        finals.append(rt[100:].mean())
    assert np.mean(finals) == pytest.approx(4.0, abs=0.15)


def test_a_finite_run_is_not_refused(data_dir: Path) -> None:
    # A model with no such rate law runs as before: the check raises nothing.
    with NfsimSession(_xml(data_dir, "ring2_homodimer.xml"), molecule_limit=1_000_000) as nf:
        nf.initialize(seed=7)
        nf.simulate(0.0, 5.0, 6)
        values = nf.get_observable_values()
    assert len(values) > 0 and all(np.isfinite(v) for v in values)


def test_a_refusal_at_a_segments_first_sample_does_not_stop_the_session(data_dir: Path) -> None:
    # Nothing has run when the first sample is checked, so the state is the
    # true state: once the rate is repaired the session goes on.
    with NfsimSession(_xml(data_dir, NAN_LAW)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match=r"by the sample at t = 0\b"):
            nf.simulate(0.0, 1.0, 11)
        nf.set_param("k", 1.0)
        result = nf.simulate(0.0, 1.0, 11)
    assert result.n_times == 11
    assert np.asarray(result.observables["B"])[-1] < 10


def test_finite_propensities_whose_sum_overflows_are_said_to(data_dir: Path) -> None:
    with NfsimSession(_xml(data_dir, "propensity_sum_overflows.xml")) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="total propensity is inf") as err:
            nf.simulate(0.0, 1.0, 2)
    assert "their sum overflowed" in str(err.value)
