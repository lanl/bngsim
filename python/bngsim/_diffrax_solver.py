"""bngsim._diffrax_solver — Diffrax ODE solver (pure JAX, no CVODE).

Runs the entire ODE solve in JAX/XLA — RHS, Jacobian (AD), and the
adaptive stepper are all JIT-compiled into a single fused computation.
No Python↔C++ boundary crossings during the solve.

Uses Kvaerno5 (5th-order implicit Runge-Kutta, L-stable) for stiff
biochemical systems, with PID step size control; next to a switch on the clock,
Kvaerno3 (see below).

The RHS, the parameter values and the initial state all come from the built
model (issue #803 step 4). They used to be re-read from the ``.net`` text by
codegen's private parser: a synthesis reaction's rate was multiplied by the last
species, a model without observables crashed, and an initial concentration
written as an expression was looked up as a parameter name.

A rate law that switches on the clock, ``if(time() > t*, ...)``, jumps there.
Stepped across, such a jump cost several times CVODE's error, and at a tight
tolerance the whole step budget (issue #865). The run is split at each such time
and restarted just past it, and the last stretch before a switch is solved with
Kvaerno3, whose stages stay within its step. A switch on the model's state,
whose times are not known before the run, is refused, as events are.

Optional dependency: ``pip install diffrax``.
"""

from __future__ import annotations

import logging

import numpy as np

from bngsim._jax_rhs import (
    _as_model,
    check_rhs_against_engine,
    generate_jax_rhs,
    jax_available,
)

logger = logging.getLogger("bngsim")

# ─── Availability check ────────────────────────────────────────────

_DIFFRAX_AVAILABLE: bool | None = None

# Kvaerno5 stretches before Kvaerno3 in a piece that ends at a switch: each takes
# four fifths of what is left (issue #865).
_KVAERNO5_STRETCHES = 3

# A stretch of a split run shorter than this, times the run's span, is narrow:
# a piece that short is one Kvaerno3 stretch, and a narrow stretch Kvaerno3 fails
# on is solved again with Tsit5 (issue #865).
_NARROW = 1e-3


def diffrax_available() -> bool:
    """Check if diffrax is importable (cached)."""
    global _DIFFRAX_AVAILABLE
    if _DIFFRAX_AVAILABLE is None:
        if not jax_available():
            _DIFFRAX_AVAILABLE = False
        else:
            try:
                import diffrax  # noqa: F401

                _DIFFRAX_AVAILABLE = True
            except ImportError:
                _DIFFRAX_AVAILABLE = False
    return _DIFFRAX_AVAILABLE


# ─── Diffrax solver ─────────────────────────────────────────────────


def _use_ode(message: str) -> ValueError:
    """A refusal that tells the caller to run with ``method='ode'``: a
    :class:`~bngsim.UsageError` (a ``ValueError``, as it was raised before it
    had a type; issue #1702)."""
    from bngsim._error_codes import use_method
    from bngsim._exceptions import UsageError

    return UsageError(message, code="usage.method_unsupported", remedies=[use_method("ode")])


def run_diffrax(
    model,
    param_dict: dict[str, float] | None = None,
    t_start: float = 0.0,
    t_end: float = 100.0,
    n_points: int = 101,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 100000,
) -> dict:
    """Run an ODE simulation using Diffrax (pure JAX).

    Parameters
    ----------
    model : Model or str
        The built model, or a ``.net`` path loaded with ``Model.from_net``.
    param_dict : dict[str, float], optional
        Parameter overrides, applied with ``set_param`` to a clone of the model,
        so derived parameters and parameter-valued initial conditions follow.
        Every other parameter keeps the model's value. Naming a derived
        parameter pins it at the value given. The primary parameters are applied
        first and the derived ones after, whatever the order of the keys, so a
        dict naming every parameter -- this function's contract before issue
        #803 -- runs on exactly those values.
    t_start, t_end : float
        Time interval.
    n_points : int
        Number of output time points.
    rtol, atol : float
        Solver tolerances.
    max_steps : int
        Maximum internal solver steps: of the run, or, for a run split at its
        switches, of each stretch of it one solver integrates. ``n_steps`` counts
        them all.

    Returns
    -------
    dict
        Keys: 'time' (n_points,), 'species' (n_points, n_sp),
        'species_names' list[str], 'n_steps' int.

    Raises
    ------
    ValueError
        If the model has events, or a rate law that switches on the model's
        state, neither of which this solver implements; if its switches on the
        clock cannot all be placed (see :func:`_switch_times`); or if it has a
        construct the JAX RHS does not implement or misdescribes (see
        :func:`bngsim._jax_rhs.check_rhs_against_engine`).
    """
    if not diffrax_available():
        from bngsim._error_codes import install
        from bngsim._exceptions import DependencyError

        raise DependencyError(
            "Diffrax is required for method='diffrax'. Install with: pip install diffrax",
            remedies=[install("diffrax")],
        )

    import diffrax
    import jax
    import jax.numpy as jnp

    # A clone, put back to its initial state: the run starts from the initial
    # state (get_initial_state below), and its switch times have to be of that
    # state too. A counter clock's crossing is placed from the counter's present
    # value, which a run on the model before this one has moved, and then no
    # switch was found and the run stepped across it (issue #865).
    model = _as_model(model).clone()
    model.reset()
    if param_dict:
        derived = {
            n for n, e in zip(model.param_names, model.param_is_expression, strict=True) if e
        }
        # Primaries first: a derived parameter set before one of its inputs would
        # be recomputed over (set_param re-derives), and a key order would decide.
        ordered = sorted(param_dict.items(), key=lambda kv: kv[0] in derived)
        for name, value in ordered:
            model.set_param(name, value)
    if model.n_events:
        # A compartment size that jumps keeps the amounts in it through an event
        # the loader adds (issue #1054), which is no event of the model's own.
        jumps = list(model._core.event_size_jump_compartments)
        n_own = sum(1 for comp in jumps if not comp)
        has = (
            f"this model has {n_own}"
            if n_own
            else "this model keeps the amounts in compartment "
            f"{', '.join(sorted(set(jumps)))} across a jump of its size with one"
        )
        raise _use_ode(
            f"run_diffrax does not implement events, and {has}; use Simulator(method='ode')."
        )
    # A step or bend call over state is one of these since issue #1068: a pulse
    # written as `max(0, 1 - abs(u)/w)` is stepped over where nothing roots it.
    # At this run's parameter values: a condition they decide, such as the
    # `.net` loader's Hill guard at h = 2, switches nothing (issue #1636).
    state_switches = model._state_switch_roots_at_run_values()
    if state_switches:
        raise _use_ode(
            "run_diffrax does not integrate a rate law that switches on the model's state, "
            "with a comparison or a step or bend call such as floor(), max() or abs(), "
            "whose switch times are not known before the run, and this model has "
            f"{list(state_switches)}; use Simulator(method='ode')."
        )
    switch_ts = _switch_times(model, float(t_start), float(t_end))
    from bngsim._exceptions import UsageError

    try:
        rhs = generate_jax_rhs(model)
    except UsageError as e:
        # The JAX RHS's refusals name jacobian='auto', which run_diffrax does
        # not take; here the way on is the engine's own integrator.
        if any(r.get("argument") == "jacobian" for r in e.remedies):
            raise _use_ode(str(e)) from None
        raise
    span = float(t_end) - float(t_start)
    check_rhs_against_engine(
        model, rhs, times=[float(t_start) + f * span for f in (0.0, 0.37, 0.81, 1.0)]
    )

    # Parameters in the model's order, derived ones at their evaluated values --
    # the array the RHS indexes -- and the model's own initial state.
    core = model._core
    params = jnp.array([core.get_param(n) for n in core.param_names], dtype=jnp.float64)
    y0 = jnp.array(np.asarray(core.get_initial_state()), dtype=jnp.float64)
    sp_names = list(model.species_names)

    # Output times
    ts = jnp.linspace(t_start, t_end, n_points)
    out_ts = np.asarray(ts)

    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    term = diffrax.ODETerm(lambda t, y, args: rhs(y, t, args))

    # Issue #865. The run is split at the times its rate laws switch at
    # (_switch_times): each piece ends at the float just before a switch, and the
    # next begins at the float just after it, so that no step is taken across the
    # jump (to within the few floats a crossing the engine scans is placed to).
    # Kvaerno5 reads one of its stages at 1.23 steps, past the end of the step,
    # and in a piece's last step that is past the switch: a step that ends just
    # before one read the other branch, and was turned down until the steps ran
    # out. That held for diffrax's own jump_ts as well. Kvaerno3's stages all lie
    # within its step (c <= 1), but it is of third order, and over a whole piece
    # of fast dynamics it took ten to thirty times main's steps. So in a piece
    # that ends at a switch, Kvaerno5 takes four fifths of what is left, where no
    # stage can reach past 1.23 * 4/5 of it, three times over, and Kvaerno3 the
    # last 0.8%; the last piece, and a run with no switch, are Kvaerno5's as
    # before. Reading the time clamped to the piece instead kept the branch, but
    # froze the rate laws' smooth dependence on the time in each last step, past
    # the tolerance, which the error estimate does not see. One compiled solve per
    # solver serves every piece: its bounds, start state and output times are
    # arguments, the output times padded with the piece's end, the last of which
    # is the state the next piece starts from.
    #
    # The implicit solvers' Newton iteration reads roundoff as divergence on a
    # step as short as some pieces are, and a switch 1e-6 after t_start made the
    # solve turn its steps down until they ran out; Kvaerno5 also failed or took
    # thousands of steps on pieces up to 1e-4 long. A piece shorter than
    # _NARROW of the run's span is one Kvaerno3 stretch, and a narrow stretch
    # Kvaerno3 fails on goes to Tsit5 (see below). A run with no switch keeps
    # main's controller exactly, and raises what it raised.
    def _compiled(solver, controller, throw):
        @jax.jit
        def _solve(a, b, y_a, seg_ts, params):
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=a,
                t1=b,
                dt0=jnp.minimum((b - a) / 100.0, 0.01),
                y0=y_a,
                args=params,
                saveat=diffrax.SaveAt(ts=seg_ts),
                stepsize_controller=controller,
                max_steps=max_steps,
                throw=throw,
            )
            # The least step is checked on the step proposed after one is
            # accepted. A solve whose last accepted step lands on b, as short as a
            # step clipped to b can be, comes back dt_min_reached with every
            # point saved; one that stopped short of b leaves inf in sol.ts.
            done = (sol.result == diffrax.RESULTS.successful) | (
                (sol.result == diffrax.RESULTS.dt_min_reached) & (sol.ts[-1] == b)
            )
            return sol.ys, sol.stats["num_steps"], done

        return _solve

    if switch_ts:
        span = abs(float(t_end) - float(t_start))
        # A stretch shorter than this is narrow. Relative to the run's span: at
        # t ~ 1e6 a piece 1.37 long is no narrower than at t ~ 1.
        narrow = _NARROW * span
        # The least step: 128 floats of the run's time, the step below which a
        # step is roundoff.
        dtmin = (
            128.0 * np.finfo(np.float64).eps * max(abs(float(t_start)), abs(float(t_end)), span)
        )

        # A solve that would need a shorter step gives up, and the run is refused.
        # Made to take the least step whatever its error estimate (force_dtmin),
        # Kvaerno5 crawled to max_steps on a short stretch, and Kvaerno3 brought a
        # stiff transient at t ~ 1e6, where the least step is 3e-8, back negative
        # with nothing said.
        controller = diffrax.PIDController(rtol=rtol, atol=atol, dtmin=dtmin, force_dtmin=False)
        solve_k5 = _compiled(diffrax.Kvaerno5(), controller, throw=False)
        solve_k3 = _compiled(diffrax.Kvaerno3(), controller, throw=False)
        # A stretch can be too short for the implicit solvers' Newton iteration,
        # which reads roundoff as divergence there (a switch 1e-6 after t_start,
        # or 1e-12 before t_end, ran their steps out). A narrow one that Kvaerno3
        # fails on is solved again with Tsit5, which has no Newton iteration and
        # whose stages also lie within its step. Kvaerno3 is tried first: Tsit5's
        # step is bound by stiffness, and a narrow stretch is narrow against the
        # span, so it can be long, and a stiff model took Tsit5 tens of thousands
        # of steps there, or more than max_steps, where main took a hundred.
        solve_explicitly = _compiled(diffrax.Tsit5(), controller, throw=False)
    else:
        solve_k5 = _compiled(diffrax.Kvaerno5(), stepsize_controller, throw=True)

    def solve_span(solve, a, b, y_a, out_idx):
        """Solve from *a* to *b* from *y_a*, filling the output times *out_idx*:
        the state at *b*, the steps taken, and whether the solve succeeded."""
        seg_ts = np.full(n_points + 1, b)
        seg_ts[: out_idx.size] = np.clip(out_ts[out_idx], min(a, b), max(a, b))
        ys, n, ok = solve(
            jnp.float64(a), jnp.float64(b), y_a, jnp.asarray(seg_ts, dtype=jnp.float64), params
        )
        ys = np.asarray(ys)
        species_out[out_idx] = ys[: out_idx.size]
        return jnp.asarray(ys[-1]), int(n), bool(ok)

    def stretch(a, b, y_a, out_idx, kvaerno5: bool):
        """One stretch of a split run: with Kvaerno5 where its stages cannot
        reach a switch (*kvaerno5*, and not narrow) and Kvaerno3 otherwise, each
        failure tried again with the next: a Kvaerno5 stretch with Kvaerno3,
        which carries one run up to a switch (a guard on the bound above), and a
        narrow stretch with Tsit5."""
        solves: tuple = ()
        if abs(b - a) < narrow:
            solves = (solve_k3, solve_explicitly)
        elif kvaerno5:
            solves = (solve_k5, solve_k3)
        else:
            solves = (solve_k3,)
        tried = 0
        for solve in solves:
            y_b, n, ok = solve_span(solve, a, b, y_a, out_idx)
            tried += n
            if ok:
                return y_b, tried
        raise _use_ode(
            f"run_diffrax could not integrate the run from {a!r} to {b!r}, next to a "
            f"switch, within max_steps={max_steps} and with no step shorter than {dtmin:.3g}; "
            "use Simulator(method='ode')."
        )

    # Piece i runs from just past switch i-1 to just short of switch i. A switch
    # is a group of crossings within the ODE path's one instant (_same_instant),
    # its lo the first and its hi the last; the piece before it ends short of lo
    # and the one after begins past hi, so no piece is turned backward or to
    # nothing by two crossings a float apart.
    n_pieces = len(switch_ts) + 1
    species_out = np.empty((n_points, len(sp_names)), dtype=np.float64)
    y_a = y0
    n_steps = 0
    for i in range(n_pieces):
        a = float(t_start) if i == 0 else float(np.nextafter(switch_ts[i - 1][1], np.inf))
        last = i == n_pieces - 1
        b = float(t_end) if last else float(np.nextafter(switch_ts[i][0], -np.inf))
        # A piece reports the output times up to the end of its switch, the first
        # also t_start. One on a switch is the piece's before it, read at the float
        # just before: the state does not jump there, only its derivative. A run
        # with no switch is one piece, forward or backward, and reports them all.
        if n_pieces == 1:
            mine = np.arange(n_points)
        else:
            after = switch_ts[i - 1][1] if i else -np.inf
            upto = float(t_end) if last else switch_ts[i][1]
            mine = np.flatnonzero((out_ts > after) & (out_ts <= upto))
        if i > 0:
            # A counter clock reaches its threshold at the switch only to within
            # the integration's error, and a few floats short the next piece read
            # the branch that had just ended: a dozen rejected steps, and an
            # accepted one on the wrong branch. It is put just past its threshold,
            # as the core puts it before CVODE restarts (issue #82). (A counter
            # that runs ahead reads the next branch before the piece ends instead,
            # which this cannot repair.)
            y_np = np.array(y_a, dtype=np.float64)
            for idx, threshold in switch_ts[i - 1][2]:
                y_np[idx] = max(y_np[idx], float(np.nextafter(threshold, np.inf)))
            y_a = jnp.asarray(y_np)
        if n_pieces == 1:
            y_a, n, _ = solve_span(solve_k5, a, b, y_a, mine)
            n_steps += n
        elif abs(b - a) < narrow:
            # A piece this short (a switch 1e-6 after t_start, a pulse 2e-8
            # wide) took Kvaerno5 thousands of steps, or failed it; it is one
            # stretch, Kvaerno3's (or Tsit5's).
            y_a, n = stretch(a, b, y_a, mine, kvaerno5=False)
            n_steps += n
        elif last:
            y_a, n = stretch(a, b, y_a, mine, kvaerno5=True)
            n_steps += n
        else:
            lo = a
            for _ in range(_KVAERNO5_STRETCHES):
                m = lo + 0.8 * (b - lo)
                here = mine[(out_ts[mine] <= m) & ((out_ts[mine] > lo) | (lo == a))]
                y_a, n = stretch(lo, m, y_a, here, kvaerno5=True)
                n_steps += n
                lo = m
            rest = mine[(out_ts[mine] > lo) | (lo == a)]
            y_a, n = stretch(lo, b, y_a, rest, kvaerno5=False)
            n_steps += n

    return {
        "time": np.asarray(ts),
        "species": species_out,
        "species_names": sp_names,
        "n_steps": n_steps,
    }


def _switch_times(model, t_start: float, t_end: float) -> list[tuple[float, float, tuple]]:
    """The instants in ``(t_start, t_end)`` at which a rate law of *model* switches
    on the clock, which run_diffrax splits the run at (issue #865), as
    ``(lo, hi, counters)``: crossings within one instant of each other
    (``_same_instant``) are one switch from ``lo`` to ``hi``, and *counters* are the
    counter clocks that cross there, as ``(species index, threshold)``.

    A condition on the clock, ``if(time() > 1.37, 5, 0.1)``, is a jump in the
    right-hand side, and the step-size controller stepped across it: several
    times CVODE's error at ``rtol=1e-6``, and at ``rtol=1e-8`` it spent its whole
    step budget there and failed. The crossings are constants of the run, the
    ones CVODE stops on (:func:`bngsim._switch_sensitivity.crossings_with_periodic`).
    These are the crossings themselves, a periodic condition's included, not the
    stops CVODE places a little past one. A condition only a reporting function
    reads changes an output, not the trajectory, which is all run_diffrax returns,
    and is left out (as :func:`~bngsim._switch_sensitivity.conditions_a_rate_law_reads`
    tells them apart, by whether a rate law's text holds the condition's): a
    split there cost a model whose rate laws never switch hundreds of steps, and
    changed its answer.

    A condition a rate law reads that switches more often than a run's budget of
    stops has none placed, and one that cannot be placed, or a failure to place
    them, would leave the run stepping across them: each raises ``ValueError``
    rather than returning the trajectory #865 did. So does a run backward in time
    across a switch on the clock, or on a model with a switch on a counter clock.
    """
    conditions = model.time_discontinuity_conditions()
    if conditions:
        conditions = _conditions_the_run_switches_on(model, conditions)
    if not conditions:
        return []
    from bngsim._switch_sensitivity import (
        _PLAIN_INSTANT_FLOOR,
        _rewrite_counter_clock,
        _same_instant,
        crossings_with_periodic,
        switch_condition_scope,
    )

    # A condition on two counters is one no resolver can place, and the
    # resolvers pass it over without a word: the run then stepped across it,
    # which is the issue again.
    scope = switch_condition_scope(model._core)
    rewrites = {c: _rewrite_counter_clock(model._core, c, scope, t_start) for c in conditions}
    unplaced = [c for c, rewrite in rewrites.items() if rewrite is None]
    if unplaced:
        raise _use_ode(
            f"run_diffrax cannot place the times these conditions switch at: {unplaced}; "
            "use Simulator(method='ode')."
        )
    backward = t_end < t_start
    if backward and any(rewrite is not None and rewrite[1] >= 0 for rewrite in rewrites.values()):
        # A counter's crossing is placed from its value at t_start, forward.
        raise _use_ode(
            "run_diffrax cannot run backward in time on a model that switches on a "
            "counter clock, whose crossings are placed forward; use "
            "Simulator(method='ode') or run forward."
        )
    over_budget: list = []
    try:
        crossings, _stops, periodic_at = crossings_with_periodic(
            model._core,
            min(t_start, t_end),
            max(t_start, t_end),
            conditions,
            touches=getattr(model, "_event_disc_conditions", frozenset()),
            over_budget=over_budget,
        )
    except Exception as e:
        raise _use_ode(
            f"run_diffrax could not place the times this model's rate laws switch at ({e}); "
            "use Simulator(method='ode')."
        ) from e
    if over_budget:
        raise _use_ode(
            "run_diffrax cannot stop at every switch of this model: "
            f"{len(over_budget)} condition(s) switch more often than a run may stop, or "
            "too irregularly to count; use Simulator(method='ode')."
        )
    lo_t, hi_t = min(t_start, t_end), max(t_start, t_end)
    found = sorted((float(c.time), c) for c in (*crossings, *periodic_at) if lo_t < c.time < hi_t)
    if backward:
        # Over literal time alone, the crossings are the same either way; a run
        # backward over one is refused, and one with none in it is main's.
        if found:
            raise _use_ode(
                "run_diffrax cannot run backward in time across a switch on the clock, "
                f"and this run crosses {[t for t, _ in found]}; use "
                "Simulator(method='ode') or run forward."
            )
        return []
    groups: list[list] = []
    for t, c in found:
        if groups and _same_instant(groups[-1][1], t, _PLAIN_INSTANT_FLOOR):
            groups[-1][1] = t
        else:
            groups.append([t, t, set()])
        if c.clock_species_idx >= 0:
            groups[-1][2].add((int(c.clock_species_idx), float(c.threshold)))
    return [(lo, hi, tuple(sorted(counters))) for lo, hi, counters in groups]


def _conditions_the_run_switches_on(model, conditions) -> list:
    """Those of *conditions* a rate law reads, or a loader registered as the
    run's own, as the ODE path counts them (``Simulator._switches_the_run``)."""
    from bngsim._switch_sensitivity import conditions_a_rate_law_reads

    registered = set(getattr(model, "_time_disc_conditions", None) or ())
    asked = [c for c in conditions if c not in registered]
    try:
        read = conditions_a_rate_law_reads(model._core, asked)
    except Exception as e:  # pragma: no cover - defensive: then all of them count
        logger.debug("which conditions a rate law reads: not read (%s)", e)
        read = frozenset(asked)
    return [c for c in conditions if c in registered or c in read]
