"""method="nfsim" and a total propensity that is not finite (issues #1818, #1788).

NFsim keeps its total propensity as a running sum, so one non-finite propensity
makes it NaN for the rest of the run. NFsim then takes the system to have
nothing left to fire, and every later row repeats the state it stopped in. That
was returned as a trajectory, with no error.

Two things are tested here.

* The vendored NFsim no longer makes such a NaN itself (RuleWorld/nfsim#117,
  carried). It used to read a rate law while a firing had its molecules out of
  the observables: with one X, ``kgap*(1+9*Xp/Xtot)`` was 0/0 there, though
  every state the model reaches is finite, and the run froze at the first X
  firing. ``rate_divides_by_emptied_observable`` is #1818's model; the four
  beside it reach the same fault by other paths.
* A NaN that is the model's own is refused, not returned as a frozen run:
  ``nan_rate_law`` (#1788: ``B -> 0`` at ``sqrt(k)``, k = -1, NaN from the
  start, which ssa and ode refuse) and ``rate_nan_in_reached_state``
  (``R(g~T) -> R(g~D)`` at ``kgap/RT``: NaN once the last R(g~T) is gone).
"""

from __future__ import annotations

import re
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import NfsimSession


def _has_nfsim() -> bool:
    return getattr(bngsim, "HAS_NFSIM", False)


pytestmark = pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")

ISSUE_1818 = "rate_divides_by_emptied_observable"
NAN_LAW = "nan_rate_law.xml"
NAN_REACHED = "rate_nan_in_reached_state.xml"


def _xml(data_dir: Path, name: str) -> str:
    return str(data_dir / "nfsim" / (name if name.endswith(".xml") else name + ".xml"))


def _still_changing(result, after: int = 400) -> bool:
    cols = [np.asarray(result.observables[name]) for name in result.observable_names]
    return any(np.any(c[after:] != c[after]) for c in cols)


# ─── The vendored NFsim computes no rate mid-firing ───────────────────────────


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_issue_1818s_model_runs_to_the_end(data_dir: Path, seed: int) -> None:
    # Before the carry each of these seeds froze: the last change came at
    # t = 1, 54, 14, 19 and 4 of 500.
    with NfsimSession(_xml(data_dir, ISSUE_1818)) as nf:
        nf.initialize(seed=seed)
        result = nf.simulate(0.0, 500.0, 501)
    assert result.n_times == 501
    assert _still_changing(result)


def _time_average(run, observable: str) -> tuple[float, float]:
    """Mean over 40 seeds of *observable* averaged over t = 100..500, and its
    standard error."""
    vals = [np.asarray(run(seed).observables[observable])[100:].mean() for seed in range(1, 41)]
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def _nfsim_run(data_dir: Path, name: str, connectivity: bool):
    def run(seed: int):
        with NfsimSession(_xml(data_dir, name), connectivity=connectivity) as nf:
            nf.initialize(seed=seed)
            return nf.simulate(0.0, 500.0, 501)

    return run


def _ssa_run(data_dir: Path, name: str):
    model = bngsim.Model.from_net(str(data_dir / "nfsim" / (name + ".net")))

    def run(seed: int):
        model.reset()
        return bngsim.Simulator(model, method="ssa").run(t_span=(0, 500), n_points=501, seed=seed)

    return run


@pytest.mark.parametrize("connectivity", [False, True])
@pytest.mark.parametrize(
    "name,observable",
    [
        # #1818's model. X is in each state half the time, so the closed form is
        # 8*(1/(1+10) + 1/(1+1))/2 = 2.364; frozen at the first X firing, NFsim
        # gave about 4 (RT with X unphosphorylated).
        (ISSUE_1818, "RT"),
        ("rate_divides_and_firing_deletes", "RT"),  # the firing deletes a molecule too
        ("rate_negative_mid_firing", "RT"),  # 1+9*(Xtot-Xp): negative only in between
        # A synthesis at kmake*Xp/Xtot: its rate changes only when Xp does, and
        # nothing else recomputes it, so a held update that was dropped shows.
        ("rate_only_the_held_update_refreshes", "Ytot"),
    ],
)
def test_a_rate_that_reads_an_observable_agrees_with_ssa_on_the_network(
    data_dir: Path, name: str, observable: str, connectivity: bool
) -> None:
    nf_mean, nf_se = _time_average(_nfsim_run(data_dir, name, connectivity), observable)
    ssa_mean, ssa_se = _time_average(_ssa_run(data_dir, name), observable)
    assert abs(nf_mean - ssa_mean) < 5 * float(np.hypot(nf_se, ssa_se)), (nf_mean, ssa_mean)
    if name == ISSUE_1818:
        assert nf_mean == pytest.approx(2.364, abs=0.3), (nf_mean, nf_se)


@pytest.mark.parametrize("connectivity", [False, True])
@pytest.mark.parametrize(
    "name,closed_form",
    [
        # X(s~P) -> X(s~0) at kd/Xp TotalRate is the rule at rate kd with one X,
        # which is #1818's model: 2.364. Network ssa refuses kd/Xp at Xp = 0, so
        # these two are held to the closed form.
        ("total_rate_divides_by_emptied_observable", 2.364),
        # With two reactants, R(g~T) + X(s~P) at total rate kd: X leaves P at kd
        # only while an R(g~T) exists, which is nearly always.
        ("total_rate_divides_two_reactants", None),
    ],
)
def test_a_total_rate_that_reads_an_observable_runs_to_the_end(
    data_dir: Path, name: str, closed_form: float | None, connectivity: bool
) -> None:
    run = _nfsim_run(data_dir, name, connectivity)
    for seed in range(1, 6):
        result = run(seed)
        assert result.n_times == 501
        assert _still_changing(result), (name, connectivity, seed)
    if closed_form is not None:
        mean, se = _time_average(run, "RT")
        assert mean == pytest.approx(closed_form, abs=0.3), (mean, se)


def test_a_rate_held_during_a_firing_is_applied_after_it(data_dir: Path) -> None:
    # A synthesis at kmake*Xp/Xtot: its rate changes only when Xp does, and
    # nothing else recomputes it. If the held update were dropped the selector
    # would keep a stale total and X would sit in its P state three quarters
    # of the time (measured upstream, 0.75 to 0.77), where kp = kd puts it
    # there half the time.
    in_p = []
    for seed in range(1, 21):
        with NfsimSession(_xml(data_dir, "rate_only_the_held_update_refreshes")) as nf:
            nf.initialize(seed=seed)
            result = nf.simulate(0.0, 500.0, 501)
        in_p.append(np.asarray(result.observables["Xp"]).mean())
    assert 0.38 < float(np.mean(in_p)) < 0.62, np.mean(in_p)


def test_a_firing_left_by_an_exception_leaves_no_rate_held(data_dir: Path) -> None:
    # bngsim turns NFsim's fatal exits into exceptions, and a session outlives
    # one. Here a synthesis hits the molecule limit partway through its firing.
    # The hold must not outlive it: with X then put in its unphosphorylated
    # state the synthesis rate, kmake*Xp/Xtot, is 0, and Y only decays. With
    # the hold left on, that update was held and never applied, and Y went on
    # being made at the old rate until the limit was hit again.
    name = "rate_only_the_held_update_refreshes"
    with NfsimSession(_xml(data_dir, name), molecule_limit=8) as nf:
        nf.set_param("kp", 0.0)
        nf.set_param("kd", 0.0)  # X does not switch on its own
        nf.initialize(seed=1)
        nf.set_species_count("X(s~0)", 0)
        nf.set_species_count("X(s~P)", 1)  # Xp = 1: Y is made at kmake
        with pytest.raises(bngsim.SimulationError, match="molecule limit"):
            nf.simulate(0.0, 200.0, 201)
        nf.set_species_count("X(s~P)", 0)
        nf.set_species_count("X(s~0)", 1)  # Xp = 0: nothing makes Y now
        result = nf.simulate(0.0, 300.0, 301)
    y = np.asarray(result.observables["Ytot"])
    assert y[0] > 0 and y[-1] == 0
    assert np.all(np.diff(y) <= 0), "Y was made though Xp = 0"


# ─── A NaN that is the model's own is refused ─────────────────────────────────


def test_a_rate_law_that_is_nan_from_the_start_is_refused_and_named(data_dir: Path) -> None:
    # #1788. Before: A = 100 and B = 10 at every sample, for E[A(1)] = 36.8.
    with NfsimSession(_xml(data_dir, NAN_LAW)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan") as err:
            nf.simulate(0.0, 1.0, 11)
        assert "Non-finite propensity in 1 reaction(s)" in str(err.value)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan"):
            nf.step_to(1.0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_rate_law_that_becomes_nan_mid_run_is_refused_not_frozen(
    data_dir: Path, seed: int
) -> None:
    # kgap/RT with three R(g~T) and no way back: NaN once the last is gone.
    with NfsimSession(_xml(data_dir, NAN_REACHED)) as nf:
        nf.initialize(seed=seed)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan") as err:
            nf.simulate(0.0, 200.0, 201)
    assert "Non-finite propensity in 1 reaction(s)" in str(err.value)
    assert "NFsim fires nothing once it is" in str(err.value)


def test_the_one_shot_run_refuses_both(data_dir: Path) -> None:
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    times = TimeSpec()
    times.t_start, times.t_end, times.n_points = 0.0, 200.0, 201
    with pytest.raises(RuntimeError, match="NfsimSimulator::run: NFsim's total propensity is nan"):
        NfsimSimulator(_xml(data_dir, NAN_REACHED)).run(times, 1)
    times.t_end, times.n_points = 1.0, 11
    with pytest.raises(RuntimeError, match=r"Non-finite propensity in 1 reaction\(s\)"):
        NfsimSimulator(_xml(data_dir, NAN_LAW)).run(times, 1)


def test_a_total_that_is_not_finite_from_the_start_is_refused_at_once(data_dir: Path) -> None:
    # A one-point segment or run is refused too, at its only sample.
    with NfsimSession(_xml(data_dir, NAN_LAW)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match=r"by the sample at t = 0\b"):
            nf.simulate(0.0, 0.0, 1)
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    times = TimeSpec()
    times.t_start, times.t_end, times.n_points = 0.0, 0.0, 1
    with pytest.raises(RuntimeError, match=r"by the sample at t = 0\b"):
        NfsimSimulator(_xml(data_dir, NAN_LAW)).run(times, 1)


def _sample_named(message: str) -> float:
    found = re.search(r"by the sample at t = ([0-9.eE+-]+?)\. ", message)
    assert found, message
    return float(found.group(1))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_time_named_is_the_samples_own(data_dir: Path, seed: int) -> None:
    # The message gives the sample's time in the caller's labels, which NFsim's
    # own clock (0 at t_start) is not: the same seed run from t_start = 100
    # names a time 100 later, through a session and through the one-shot run.
    from bngsim._bngsim_core import NfsimSimulator, TimeSpec

    named = []
    for t_start in (0.0, 100.0):
        with NfsimSession(_xml(data_dir, NAN_REACHED)) as nf:
            nf.initialize(seed=seed)
            with pytest.raises(bngsim.SimulationError) as err:
                nf.simulate(t_start, t_start + 200.0, 201)
        times = TimeSpec()
        times.t_start, times.t_end, times.n_points = t_start, t_start + 200.0, 201
        with pytest.raises(RuntimeError) as core_err:
            NfsimSimulator(_xml(data_dir, NAN_REACHED)).run(times, seed)
        assert _sample_named(str(core_err.value)) == _sample_named(str(err.value))
        named.append(_sample_named(str(err.value)))
    assert named[0] >= 1.0
    assert named[1] == named[0] + 100.0


def test_finite_propensities_whose_sum_overflows_are_said_to(data_dir: Path) -> None:
    with NfsimSession(_xml(data_dir, "propensity_sum_overflows.xml")) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="total propensity is inf") as err:
            nf.simulate(0.0, 1.0, 2)
    assert "their sum overflowed" in str(err.value)


# ─── The session after a refusal ──────────────────────────────────────────────


def test_a_session_that_stopped_refuses_to_go_on_until_initialized(data_dir: Path) -> None:
    # The total goes non-finite during the step and NFsim fires nothing from
    # there, so the state is not the state at t = 200. Changing a rate does not
    # make the span between run: the session refuses until it is started again.
    with NfsimSession(_xml(data_dir, NAN_REACHED)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan"):
            nf.step_to(200.0)
        nf.set_param("kgap", 2.0)
        with pytest.raises(
            bngsim.SimulationError, match=r"stopped when step_to\(\) found .* at t = 200 in"
        ):
            nf.simulate(200.0, 300.0, 2)
        with pytest.raises(bngsim.SimulationError, match="initialize"):
            nf.step_to(300.0)
        nf.initialize(seed=1)
        result = nf.simulate(0.0, 0.0, 1)  # the start again: three R(g~T), a finite total
    assert result.n_times == 1
    assert np.asarray(result.observables["RT"])[0] == 3


def test_a_session_that_stopped_in_simulate_refuses_step_to(data_dir: Path) -> None:
    with NfsimSession(_xml(data_dir, NAN_REACHED)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan"):
            nf.simulate(0.0, 200.0, 201)
        with pytest.raises(bngsim.SimulationError, match=r"stopped when simulate\(\) found"):
            nf.step_to(300.0)


def test_a_restore_does_not_lift_the_stop(data_dir: Path) -> None:
    # restore_concentrations() puts the molecules back and leaves NFsim's
    # clock where the stop left it; only initialize() starts again.
    with NfsimSession(_xml(data_dir, NAN_REACHED)) as nf:
        nf.initialize(seed=1)
        nf.save_concentrations("start")
        with pytest.raises(bngsim.SimulationError, match="total propensity is nan"):
            nf.simulate(0.0, 200.0, 201)
        nf.restore_concentrations("start")
        with pytest.raises(bngsim.SimulationError, match="this session stopped"):
            nf.simulate(0.0, 100.0, 101)


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


def test_step_to_before_any_step_does_not_stop_the_session(data_dir: Path) -> None:
    # As simulate() at its first sample: a total already non-finite when
    # step_to() is called is refused before NFsim steps, and a repair holds.
    with NfsimSession(_xml(data_dir, NAN_LAW)) as nf:
        nf.initialize(seed=1)
        with pytest.raises(bngsim.SimulationError, match=r"by the sample at t = 0\b"):
            nf.step_to(1.0)
        nf.set_param("k", 1.0)
        nf.step_to(1.0)
        result = nf.simulate(1.0, 2.0, 11)
    assert result.n_times == 11


@pytest.mark.parametrize("advance", ["simulate", "step_to"])
def test_a_total_left_stale_by_an_edit_is_summed_afresh(data_dir: Path, advance: str) -> None:
    # With the lone X taken out, kgap*(1+9*Xp/Xtot) is NaN in the state the
    # session is in, and is refused. With X put back every rate is finite again,
    # but NFsim's running sum, NaN - NaN + new, is not. The state is sound and
    # nothing has run from it, so the total is summed afresh and the session
    # goes on; it was refused before, naming no reaction.
    with NfsimSession(_xml(data_dir, ISSUE_1818)) as nf:
        nf.initialize(seed=1)
        nf.set_species_count("X(s~0)", 0)
        with pytest.raises(bngsim.SimulationError, match="Non-finite propensity in 1 reaction"):
            nf.simulate(0.0, 1.0, 2)
        nf.add_molecules("X", 1)
        if advance == "step_to":
            nf.step_to(1.0)
        result = nf.simulate(1.0, 101.0, 101)
    assert result.n_times == 101
    assert _still_changing(result, after=50)


# ─── Controls ─────────────────────────────────────────────────────────────────


def test_issue_1818s_model_with_no_x_firing_relaxes_to_its_closed_form(data_dir: Path) -> None:
    # With kp = 0 the lone X never fires, so Xp/Xtot stays 0/1 and the rate is
    # never read with X out of the observables. RT relaxes to 8*kex/(kex+kgap)
    # = 4 (boost() is 1 with X unphosphorylated).
    finals = []
    for seed in range(1, 21):
        with NfsimSession(_xml(data_dir, ISSUE_1818)) as nf:
            nf.set_param("kp", 0.0)
            nf.initialize(seed=seed)
            result = nf.simulate(0.0, 500.0, 501)
        rt = np.asarray(result.observables["RT"])
        assert len(rt) == 501
        finals.append(rt[100:].mean())
    assert np.mean(finals) == pytest.approx(4.0, abs=0.15)


def test_a_finite_run_is_not_refused(data_dir: Path) -> None:
    # A model with no such rate law runs as before: the check raises nothing.
    with NfsimSession(_xml(data_dir, "ring2_homodimer.xml"), molecule_limit=1_000_000) as nf:
        nf.initialize(seed=7)
        nf.simulate(0.0, 5.0, 6)
        values = nf.get_observable_values()
    assert len(values) > 0 and all(np.isfinite(v) for v in values)
