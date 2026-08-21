"""Issue #447 — an ordinary ODE run must not print SUNDIALS error lines.

``read_segment_counters`` in ``src/cvode_simulator.cpp`` collects the counters
CVODE has accumulated since its last (re-)initialization, once per solver
segment. Two of the counters it asked for belong to the forward sensitivity
solve, and CVODES answers a request for either of those on a run that has no
forward sensitivities by printing

    [ERROR][rank 0][.../cvodes_io.c:2322][CVodeGetSensNumNonlinSolvConvFails]
    Forward sensitivity analysis not activated.

to standard error and then returning a "no sensitivities here" flag. The flag
was ignored on purpose, because leaving the two counters at zero is the right
answer for such a run. The printed line reached the user anyway: two lines for
a plain run, and two more for every event fire, because each fire
re-initializes CVODE and so closes a segment. The 20 dose model below printed
42 of them, and a reader had no way to tell any of that from a real solver
failure.

The fix skips the two requests when the run has no forward sensitivities,
rather than switching off the SUNDIALS error handler, so a genuine solver error
still reaches the user. What the counters report is unchanged, which is what the
sensitivity runs here check: they still come back nonzero.
"""

from __future__ import annotations

from bngsim._bngsim_core import (
    CvodeSimulator,
    ModelBuilder,
    NetworkModel,
    SolverOptions,
    TimeSpec,
)

# The words SUNDIALS prints when asked for a sensitivity counter on a run that
# has none. Matching on this rather than on "[ERROR]" keeps the test about
# these two calls and not about every diagnostic the solver can emit.
NOT_ACTIVATED = "Forward sensitivity analysis not activated"

# One dose every 12 time units over 240, which is the schedule the issue was
# reported on. Each fire re-initializes CVODE and so closes a solver segment,
# and every closed segment used to cost two of the lines.
DOSE_SPACING = 12.0
N_DOSES = 20
T_END = 240.0


def _decay(n_doses: int) -> NetworkModel:
    """dS/dt = -k·S from S(0) = 100, plus a ``S := S + 50`` event per dose."""
    b = ModelBuilder()
    b.add_parameter("k", 0.05)
    s = b.add_species("S", 100.0)
    b.add_reaction([s], [], "elementary", "k")
    for i in range(n_doses):
        b.add_event(f"dose{i}", f"time() >= {(i + 1) * DOSE_SPACING}", [(s, "S + 50")])
    return b.build()


def _times(t_end: float = T_END) -> TimeSpec:
    ts = TimeSpec()
    ts.t_start = 0.0
    ts.t_end = t_end
    ts.n_points = 25
    return ts


def _sens_opts() -> SolverOptions:
    opts = SolverOptions()
    opts.set_sensitivity_params(["k"])
    return opts


def test_a_plain_run_with_no_events_says_nothing(capfd):
    """The shortest repro: no events, so one segment, so two lines before."""
    sim = CvodeSimulator(_decay(0))
    capfd.readouterr()  # drop anything building the model printed
    sim.run(_times(10.0))
    assert NOT_ACTIVATED not in capfd.readouterr().err


def test_a_run_that_re_initializes_says_nothing_either(capfd):
    """The issue's own case: 20 fires over 240 time units, 42 lines before."""
    sim = CvodeSimulator(_decay(N_DOSES))
    capfd.readouterr()
    result = sim.run(_times())
    assert NOT_ACTIVATED not in capfd.readouterr().err
    # The run really did re-initialize, so the count above is not vacuous.
    assert result.solver_stats.n_steps > 0


def test_a_sensitivity_run_still_reports_what_its_solve_rejected(capfd):
    """The other half: the two counters must still be read where they exist.

    A fix that never asks for them would pass the two tests above and quietly
    zero a diagnostic the library points users at (see the sensitivity failure
    message in src/result.cpp, which tells the reader to check
    ``n_sens_err_test_fails``). This model's sensitivity solve fails its error
    test 42 times, so the margin on "nonzero" is wide. The run is also quiet,
    because here the sensitivity analysis genuinely is switched on.
    """
    sim = CvodeSimulator(_decay(N_DOSES))
    capfd.readouterr()
    stats = sim.run(_times(), _sens_opts()).solver_stats
    assert NOT_ACTIVATED not in capfd.readouterr().err
    assert stats.n_sens_err_test_fails > 0


def test_a_plain_run_after_a_sensitivity_run_says_nothing(capfd):
    """Reusing a simulator must not leave the previous run's answer behind.

    Whether to ask for the counters is decided per run, not per simulator, so
    it is cleared where the rest of the per-run counter state is. Before the
    fix the second run here printed 42 lines.
    """
    sim = CvodeSimulator(_decay(N_DOSES))
    capfd.readouterr()
    with_sens = sim.run(_times(), _sens_opts()).solver_stats
    plain = sim.run(_times()).solver_stats
    assert NOT_ACTIVATED not in capfd.readouterr().err
    # The first run did compute sensitivities and the second did not, so the
    # zero below is the honest answer rather than a lost counter.
    assert with_sens.n_sens_err_test_fails > 0
    assert plain.n_sens_err_test_fails == 0
    assert plain.n_sens_nonlin_conv_fails == 0


def test_the_reuse_that_takes_the_fast_path_says_nothing_either(capfd):
    """The same reuse, for the other of the two integration paths.

    A model with no events and no sensitivities is eligible for the fast path,
    which keeps one set of CVODE memory alive across runs instead of building
    it fresh each time. That path clears the decision in its own place, so it
    needs its own check. Two plain runs follow the sensitivity run here, to
    cover both building that memory and re-entering it.
    """
    sim = CvodeSimulator(_decay(0))
    capfd.readouterr()
    sim.run(_times(10.0), _sens_opts())
    first = sim.run(_times(10.0)).solver_stats
    second = sim.run(_times(10.0)).solver_stats
    assert NOT_ACTIVATED not in capfd.readouterr().err
    assert first.n_steps > 0
    assert second.n_steps > 0
    assert first.n_sens_err_test_fails == 0
    assert second.n_sens_err_test_fails == 0
