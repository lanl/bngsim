"""SSA/PSA with rates that move between firings: exact in time (issues #719,
#751, #753).

A rate that reads time, or a rate-rule target, used to be held constant over
sub-steps of (t_end − t_start)/1000, so the answer depended on the horizon: a
synthesis at 5·(1 + sin t) sampled once per 5 time units over (0, 5000)
reported E[N(995)] = 2707 against an exact 5002. Rate-rule targets were
integrated by forward Euler on that grid (x' = −30x over (0, 100) went to
1e33), and a model with rate rules and no firing reaction advanced them only
at output times.

The loop now integrates each dynamic propensity over adaptive panels and fires
at the root of the integrated hazard (the time-change theorem), with rate
rules on an L-stable Rosenbrock method. Oracles are closed forms: Poisson
means of integrated rates, and first moments of linear systems, which equal
the ODE.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")


def _ant(text):
    return bngsim.Model.from_antimony_string(text)


def _col(r, name):
    return np.asarray(r.species)[..., list(r.species_names).index(name)]


def _within(got, want, reps, z=4.5):
    """Poisson (or sum-of-Poisson) mean: SE √(want/reps)."""
    se = np.sqrt(max(want, 1e-12) / reps)
    assert abs(got - want) <= z * se, (got, want, se)


def _near(samples, want, z=4.5):
    """The sample mean against `want`, by the samples' own standard error (a
    PSA count is far from Poisson: each firing moves m_r molecules)."""
    samples = np.asarray(samples, dtype=float)
    se = samples.std(ddof=1) / np.sqrt(len(samples))
    assert abs(samples.mean() - want) <= z * se + 1e-12, (samples.mean(), want, se)


SINE = "species N = 0; J: => N; 5*(1 + sin(time));"


def _sine_mean(t):
    return 5.0 * (t + 1.0 - np.cos(t))


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_periodic_rate_over_a_long_horizon(method):
    """E[N(t)] = 5·(t + 1 − cos t). The old sub-step, 5 time units at
    T = 5000, sampled the sine once per cycle."""
    kw = {"poplevel": 100} if method == "psa" else {}
    reps = 40
    r = bngsim.Simulator(_ant(SINE), method=method, **kw).run_replicates(
        reps, t_span=(0, 5000), n_points=5001, seed=3, squeeze=True
    )
    n = _col(r, "N")
    for t in (50.0, 995.0, 4321.0):
        _near(n[:, int(t)], _sine_mean(t))


def test_the_answer_does_not_depend_on_the_horizon():
    """Two horizons, independent seeds: the counts at t = 50 agree with each
    other and with the exact mean. The frozen sub-step was (t_end − t_start)/
    1000, so the answer at t = 50 used to move with t_end."""
    out = []
    for t_end, seed in ((100, 11), (5000, 12)):
        r = bngsim.Simulator(_ant(SINE), method="ssa").run_replicates(
            100, t_span=(0, t_end), n_points=t_end // 50 + 1, seed=seed, squeeze=True
        )
        out.append(_col(r, "N")[:, 1])
    for o in out:
        _near(o, _sine_mean(50.0))
    se = np.sqrt(out[0].var(ddof=1) / 100 + out[1].var(ddof=1) / 100)
    assert abs(out[0].mean() - out[1].mean()) <= 4.5 * se


def test_a_step_in_the_rate():
    """``piecewise(10, t > 5, 0)``: E[N(20)] = 150."""
    reps = 200
    r = bngsim.Simulator(_ant("species N = 0; J: => N; piecewise(10, time > 5, 0);"), method="ssa")
    r = r.run_replicates(reps, t_span=(0, 20), n_points=3, seed=9, squeeze=True)
    _within(_col(r, "N")[:, -1].mean(), 150.0, reps)


NET_PULSE = """begin parameters
    1 k 1000
end parameters
begin functions
    1 pulse() if(time()>5 && time()<5.001, k, 0)
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 pulse
end reactions
begin groups
    1 Atot 1
end groups
"""


@pytest.mark.parametrize(("method", "kw"), [("ssa", {}), ("psa", {"poplevel": 100})])
def test_a_pulse_narrower_than_a_step_is_not_stepped_over(tmp_path, method, kw):
    """1000 for a millisecond at t = 5 in a run to 100: E[A] = 1. The loop's
    step does not see a pulse that falls between its nodes; the pulse's edges
    are breakpoints, and main reported A = 0 in every replicate."""
    p = tmp_path / "pulse.net"
    p.write_text(NET_PULSE)
    reps = 400
    r = bngsim.Simulator(bngsim.Model.from_net(str(p)), method=method, **kw).run_replicates(
        reps, t_span=(0, 100), n_points=2, seed=3, squeeze=True
    )
    _within(np.asarray(r.observables)[:, -1, 0].mean(), 1.0, reps)


def test_a_rate_rule_drives_a_propensity():
    """X' = 1 from 0 and ``0 -> S`` at X: E[S(4)] = ∫₀⁴ t dt = 8."""
    reps = 400
    m = _ant("species S = 0; X = 0; X' = 1; J: => S; X;")
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 4), n_points=5, seed=5, squeeze=True
    )
    _within(_col(r, "S")[:, -1].mean(), 8.0, reps)
    np.testing.assert_allclose(_col(r, "X")[0], [0, 1, 2, 3, 4], rtol=1e-6, atol=1e-9)


def test_a_rate_rule_with_nothing_firing_is_integrated():
    """#751: x' = −x with no reaction that can fire. x(t) = e^−t."""
    m = _ant("species z = 0; x = 1; x' = -x; E: at (time >= 8): z = 1;")
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 10), n_points=11, seed=1)
    t = np.asarray(r.time)
    np.testing.assert_allclose(_col(r, "x"), np.exp(-t), rtol=2e-3)
    assert _col(r, "z")[-1] == 1.0


@pytest.mark.parametrize("t_end", [100, 1000])
def test_a_stiff_rate_rule(t_end):
    """#753: x' = −30x beside a slow decay. Forward Euler on a step of
    t_end/1000 diverged (|x| to 1e33 at t_end = 100); x stays at 0."""
    m = _ant("species A = 10; J: A => ; 0.1*A; x = 1; x' = -30*x;")
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, t_end), n_points=11, seed=1)
    assert np.max(np.abs(_col(r, "x")[1:])) < 1e-8


def test_an_event_on_a_rate_rule_target_fires_at_the_crossing():
    """y' = 1, ``at (y > 0.5): b = time``: b = 0.5."""
    m = _ant(
        "y = 0; y' = 1; species b = 0; species a = 0; E2: at (y > 0.5): b = time; J: => a; 0.1;"
    )
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 3), n_points=4, seed=1)
    assert _col(r, "b")[-1] == pytest.approx(0.5, abs=1e-9)


# ── linear systems: the mean is the ODE ──────────────────────────────────────


def _mean_vs_ode(text, name, t_end, reps, method="ssa", **kw):
    m = _ant(text)
    ode = bngsim.Simulator(_ant(text), method="ode").run(t_span=(0, t_end), n_points=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.SsaRoundingWarning)
        r = bngsim.Simulator(m, method=method, **kw).run_replicates(
            reps, t_span=(0, t_end), n_points=5, seed=17, squeeze=True
        )
    got = _col(r, name)
    want = _col(ode, name)
    for i in range(1, 5):
        se = got[:, i].std(ddof=1) / np.sqrt(reps)
        assert abs(got[:, i].mean() - want[i]) <= 4.5 * se + 1e-9, (
            i,
            got[:, i].mean(),
            want[i],
            se,
        )


def test_forcing_beside_a_fast_decay():
    """``0 -> M`` at 50·(1 + sin t) and ``M -> 0`` at 0.5·M. The forcing reads
    only time, so a decay firing keeps the loop's panel; the mean is the
    ODE's."""
    _mean_vs_ode("species M = 0; J1: => M; 50*(1 + sin(time)); J2: M => ; 0.5*M;", "M", 20.0, 200)


def test_a_time_dependent_rate_that_reads_the_state():
    """``M -> 2M`` at c(t)·M: each firing moves what the rate reads, so the
    panel is rebuilt. Linear, so the mean is the ODE's."""
    _mean_vs_ode(
        "species M = 20; J1: M => 2 M; 0.1*(1 + sin(time))*M; J2: M => ; 0.15*M;", "M", 10.0, 300
    )


def test_psa_with_a_dynamic_reaction():
    _mean_vs_ode(
        "species M = 0; J1: => M; 500*(1 + sin(time)); J2: M => ; 0.5*M;",
        "M",
        10.0,
        100,
        method="psa",
        poplevel=50,
    )


def test_the_integrated_propensity_of_a_dynamic_reaction():
    """Reaction stats (#616) bank ∫a dt for a dynamic reaction from the loop's
    quadrature: 5·(t + 1 − cos t) at every row, whatever fired."""
    m = _ant("species N = 0; species M = 0; J1: => N; 5*(1 + sin(time)); J2: => M; 3;")
    sim = bngsim.Simulator(m, method="ssa", reaction_stats=True)
    r = sim.run(t_span=(0, 30), n_points=7, seed=2)
    integ = np.asarray(r.reaction_propensity_integrals)
    t = np.asarray(r.time)
    np.testing.assert_allclose(integ[:, 0], _sine_mean(t) - _sine_mean(0.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(integ[:, 1], 3.0 * t, rtol=1e-12)


def test_carry_with_a_slow_forcing():
    """A forcing that barely moves runs long panels, so a firing that wrongly
    kept its panel would show. The decay's firings keep it (the forcing reads
    only time), and the mean is the ODE's."""
    _mean_vs_ode(
        "species M = 0; J1: => M; 50*(1 + 0.001*sin(time)); J2: M => ; 0.5*M;", "M", 20.0, 200
    )


def test_no_carry_with_a_slow_forcing_that_reads_the_state():
    _mean_vs_ode(
        "species M = 20; J1: M => 2 M; 0.1*(1 + 0.001*sin(time))*M; J2: M => ; 0.15*M;",
        "M",
        10.0,
        300,
    )


TFUN = """begin parameters
    1 k0 0
end parameters
begin functions
    1 drive() tfun([XS],[YS],time,method=>"step")
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 drive
end reactions
begin groups
    1 Atot 1
end groups
"""


@pytest.mark.parametrize(
    ("xs", "ys", "area"),
    [("0,50,50.01,100", "0,1000,0,0", 10.0), ("0,37.3,42.3,100", "0,100,0,0", 500.0)],
    ids=["narrow", "wide"],
)
def test_a_table_function_pulse(tmp_path, xs, ys, area):
    """A time-indexed table's knots are breakpoints, and no panel outgrows
    (t_end − t_start)/1000: over a stretch where the rate reads 0 at every node
    the step had nothing to measure and grew 5× a panel, so both pulses were
    stepped over and every replicate reported 0."""
    p = tmp_path / "tfun.net"
    p.write_text(TFUN.replace("XS", xs).replace("YS", ys))
    reps = 200
    r = bngsim.Simulator(bngsim.Model.from_net(str(p)), method="ssa").run_replicates(
        reps, t_span=(0, 100), n_points=2, seed=3, squeeze=True
    )
    _within(np.asarray(r.observables)[:, -1, 0].mean(), area, reps)


def test_a_smooth_bump_no_breakpoint_covers():
    """``10·exp(−(t−50)²)``, area 10·√π, and 0 to double precision over most
    of the run."""
    reps = 200
    m = _ant("species N = 0; J: => N; 10*exp(-((time - 50)^2));")
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 100), n_points=2, seed=3, squeeze=True
    )
    _within(_col(r, "N")[:, -1].mean(), 10 * np.sqrt(np.pi), reps)


def test_a_rate_periodic_in_the_node_spacing():
    """Period 0.025 over (0, 100): at the step cap of 0.1, equally spaced
    nodes are whole periods apart and read the rate as constant (E[N] was
    twice the truth). The Lobatto nodes are not commensurate with any period."""
    reps = 200
    m = _ant("species N = 0; J: => N; 1 + sin(2*pi*time/0.025 + pi/2);")
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        reps, t_span=(0, 100), n_points=2, seed=3, squeeze=True
    )
    _within(_col(r, "N")[:, -1].mean(), 100.0, reps)


@pytest.mark.parametrize(
    "rate", ["0.001", "0.001*(1 + 0.001*sin(time))"], ids=["discrete-loop", "continuous-loop"]
)
def test_an_event_true_only_inside_a_window(rate):
    """``time > 37.3 && time < 37.5`` is false at both ends of its window, which
    are breakpoints; a trigger looked at only at those ends never fired."""
    txt = (
        f"species c = 0; species N = 0; J: => N; {rate};"
        " E: at (time > 37.3 && time < 37.5): c = c + 1;"
    )
    for seed in range(5):
        r = bngsim.Simulator(_ant(txt), method="ssa").run(t_span=(0, 100), n_points=2, seed=seed)
        assert _col(r, "c")[-1] == 1.0, seed


@pytest.mark.parametrize(
    "extra",
    ["species D = 0;", "species N = 0; J: => N; 0.001*(1 + sin(time));"],
    ids=["discrete-loop", "continuous-loop"],
)
def test_a_periodic_trigger_rearms(extra):
    """``sin(time) > 0.9`` rises 8 times in (0, 50). With nothing firing between
    rises its fall was never seen, so it fired once."""
    txt = f"species c = 0; {extra} E: at (sin(time) > 0.9): c = c + 1;"
    r = bngsim.Simulator(_ant(txt), method="ssa").run(t_span=(0, 50), n_points=2, seed=1)
    assert _col(r, "c")[-1] == 8.0


@pytest.mark.parametrize("t0", [0.0, 1e4, 1e6])
def test_a_rate_rule_that_stops_itself(t0):
    """``x' = piecewise(-1, x > 0, 0)`` reaches 0 at t0 + 3 and stays. A step
    whose midpoint stage fell past 0 read a slope of 0 and left x where it was;
    its error test still passed, so the run crawled forward a few ns at a time
    for ever (and, near a large t0, threw at the minimum step)."""
    m = _ant("species A = 10; J: A => ; 1e-6*A; x = 3; x' = piecewise(-1, x > 0, 0);")
    r = bngsim.Simulator(m, method="ssa").run(
        t_span=(t0, t0 + 10), n_points=11, seed=1, timeout=60
    )
    x = _col(r, "x")
    np.testing.assert_allclose(x[:3], [3.0, 2.0, 1.0], atol=1e-8)
    assert np.all(np.abs(x[3:]) < 1e-8)


def test_breakpoints_follow_run_until_legs(tmp_path):
    p = tmp_path / "pulse.net"
    p.write_text(NET_PULSE)
    reps = 300
    a = []
    for k in range(reps):
        m = bngsim.Model.from_net(str(p))
        sim = bngsim.Simulator(m, method="ssa")
        for t in (2.0, 4.99, 100.0):
            sim.run_until(t, n_points=2, seed=1000 + 3 * k + int(t))
        a.append(np.asarray(m.get_state())[0])
    _within(np.mean(a), 1.0, reps)


def test_breakpoints_follow_each_run_batch_row(tmp_path):
    """Each row's ``t_on`` places its own pulse."""
    p = tmp_path / "pulse_param.net"
    p.write_text(
        NET_PULSE.replace("    1 k 1000\n", "    1 k 1000\n    2 t_on 5\n").replace(
            "time()>5 && time()<5.001", "time()>t_on && time()<t_on+0.001"
        )
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(p)), method="ssa")
    rows = [{"t_on": v} for v in (37.123, 88.8) for _ in range(150)]
    res = sim.run_batch(t_span=(0, 100), n_points=2, params=rows, seed=4)
    a = np.array([np.asarray(r.observables)[-1, 0] for r in res])
    _within(a[:150].mean(), 1.0, 150)
    _within(a[150:].mean(), 1.0, 150)


def test_a_time_function_that_feeds_no_rate_changes_nothing(tmp_path):
    """A function of time read by nothing but the output leaves every
    propensity constant between firings: the run is the discrete loop's,
    replicate for replicate."""
    base = (
        "begin parameters\n    1 k 1\nend parameters\n"
        "begin species\n    1 A() 50\n    2 B() 0\nend species\n"
        "begin reactions\n    1 1 2 k\n    2 2 1 k\nend reactions\n"
        "begin groups\n    1 Btot 2\nend groups\n"
    )
    plain = tmp_path / "plain.net"
    plain.write_text(base)
    clock = tmp_path / "clock.net"
    clock.write_text(base + "begin functions\n    1 clock() 2*time()\nend functions\n")
    rs = []
    for path in (plain, clock):
        r = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ssa").run(
            t_span=(0, 20), n_points=11, seed=5
        )
        rs.append(np.asarray(r.species))
        assert r.solver_stats["n_steps"] > 0
    np.testing.assert_array_equal(rs[0], rs[1])


AR_COMP = "compartment C; C := 1 + 0.5*time; species A in C = 100; k = 0.3; J: A => ; {law};"


@pytest.mark.parametrize("law", ["C*k*A", "k*A"])
def test_an_assignment_rule_compartment_is_refused(law):
    """SSA does not follow a compartment an assignment rule sizes: a count is
    stored over the load-time size, so the concentration a law reads is stale
    (A(4) came out 3x the ODE's), and the run is refused rather than wrong."""
    m = _ant(AR_COMP.format(law=law))
    assert "assignment_rule_compartment" in [i.code for i in m.validate_for_ssa()]
    with pytest.raises(bngsim.SsaValidationError, match="assignment rule"):
        bngsim.Simulator(m, method="ssa").run(t_span=(0, 1), n_points=2, seed=1)


def test_amounts_in_an_assignment_rule_compartment_still_run():
    txt = AR_COMP.format(law="k*A").replace("species A", "substanceOnly species A")
    m = _ant(txt)
    assert m.validate_for_ssa() == []
    ode = bngsim.Simulator(_ant(txt), method="ode").run(t_span=(0, 4), n_points=2)
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        300, t_span=(0, 4), n_points=2, seed=3, squeeze=True
    )
    _near(_col(r, "A")[:, -1], _col(ode, "A")[-1])


# ── what a rate reads, through definitions ──────────────────────────────────


def _net_text(functions, reactions, species="    1 A() 0\n", groups="    1 Atot 1\n"):
    return (
        "begin parameters\n    1 k 1\nend parameters\n"
        f"begin functions\n{functions}end functions\n"
        f"begin species\n{species}end species\n"
        f"begin reactions\n{reactions}end reactions\n"
        f"begin groups\n{groups}end groups\n"
    )


@pytest.mark.parametrize(
    ("functions", "want"),
    [
        pytest.param(
            "    1 p() time()\n"
            '    2 drive() tfun([0,5,10,100],[0,0,100,100],p,method=>"linear")\n',
            250.0,
            id="table-indexed-by-a-clock-function",
        ),
        pytest.param("    1 drive() if(time > 5, 10, 0)\n", 50.0, id="bare-time"),
    ],
)
def test_a_rate_that_reads_the_clock_out_of_sight(tmp_path, functions, want):
    """A table's index is not in the text of its call, and ExprTk calls
    ``time`` without parentheses. Both rates were taken as constant and frozen
    at their t = 0 value of 0."""
    p = tmp_path / "m.net"
    p.write_text(_net_text(functions, "    1 0 1 drive\n"))
    reps = 200
    r = bngsim.Simulator(bngsim.Model.from_net(str(p)), method="ssa").run_replicates(
        reps, t_span=(0, 10), n_points=2, seed=3, squeeze=True
    )
    _within(np.asarray(r.observables)[:, -1, 0].mean(), want, reps)


def test_a_firing_that_moves_a_table_index_rebuilds_the_panel(tmp_path):
    """``g = tfun(…, q)`` with ``q() = Btot``: B's firings move what the dynamic
    rate reads, though no text names B. Kept, the panel's stale hazard set the
    firing times while fresh propensities chose the reaction, and the
    unrelated ``0 -> D`` at 1e4 over-fired (z = 31)."""
    p = tmp_path / "m.net"
    p.write_text(
        "begin parameters\n    1 k 2000\n    2 kd 10000\nend parameters\n"
        "begin functions\n    1 q() Btot\n"
        '    2 g() tfun([0,100],[0,100000],q,method=>"linear")\n'
        "    3 f() g()*(1+1e-9*time())\nend functions\n"
        "begin species\n    1 B() 100\n    2 C() 0\n    3 D() 0\nend species\n"
        "begin reactions\n    1 1 0 k\n    2 0 2 f\n    3 0 3 kd\nend reactions\n"
        "begin groups\n    1 Btot 1\n    2 Ctot 2\n    3 Dtot 3\nend groups\n"
    )
    reps = 60
    r = bngsim.Simulator(bngsim.Model.from_net(str(p)), method="ssa").run_replicates(
        reps, t_span=(0, 10), n_points=2, seed=3, squeeze=True
    )
    _within(np.asarray(r.observables)[:, -1, 2].mean(), 1e5, reps)


@pytest.mark.parametrize(
    "text",
    [
        "x = 0; x' = 0.01*(1 - x); species N = 0; J: => N; 0.001*x;",
        "x = 1; x' = 0.5*x*(1 - x/100); species N = 0; J: => N; 0.001*x;",
        "x = 0; x' = 1e6*(1 - x); species N = 0; J: => N; 0.1*x;",
    ],
    ids=["slow", "logistic", "stiff"],
)
def test_a_rate_rule_at_its_steady_state_runs(text):
    """Near a steady state a step moves y by less than an ulp, and y not moving
    is right; the rule that shrinks a step stuck at a switch took it for one and
    the run was refused."""
    m = _ant(text)
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 1e4), n_points=3, seed=1, timeout=60)
    x = _col(r, "x")
    assert abs(x[-1] - (100.0 if "x/100" in text else 1.0)) < 1e-6


@pytest.mark.parametrize("rule", ["C := 2", "p = 2; C := p"])
def test_an_assignment_rule_compartment_that_holds_still_runs(rule):
    txt = f"compartment C = 2; {rule}; species A in C = 100; k = 0.3; J: A => ; C*k*A;"
    m = _ant(txt)
    assert m.validate_for_ssa() == []
    ode = bngsim.Simulator(_ant(txt), method="ode").run(t_span=(0, 4), n_points=2)
    r = bngsim.Simulator(m, method="ssa").run_replicates(
        300, t_span=(0, 4), n_points=2, seed=3, squeeze=True
    )
    _near(_col(r, "A")[:, -1], _col(ode, "A")[-1])


@pytest.mark.parametrize(
    ("text", "code"),
    [
        pytest.param(
            "compartment C = 1; C := 1 + 0.5*time; compartment D = 1; species S in C = 10;"
            " species A in D = 0; q := S; k = 1; J: => A; k*q*D;",
            "assignment_rule_compartment",
            id="through-an-assignment-rule",
        ),
        pytest.param(
            "compartment C = 1; compartment D = 1; species S in C = 10; species A in D = 0;"
            " k = 1; J: => A; k*S*D; E: at (time > 2): C = 2;",
            "variable_compartment_read",
            id="event-resized-read-from-outside",
        ),
    ],
)
def test_a_concentration_read_in_a_moving_compartment_is_refused(text, code):
    """Both read [S] at C's load-time size: z = +50 and +25 against the ODE."""
    m = _ant(text)
    assert code in [i.code for i in m.validate_for_ssa()]


def test_a_rate_on_a_rate_rule_clock_keeps_its_breakpoints():
    """``T' = 1`` is a clock no text calls ``time``: its pulse's edges still
    have to bound the panels, or a 1 ms pulse is stepped over (x = 0)."""
    txt = (
        "species A = 0; T = 0; T' = 1; J: => A; piecewise(10000, (T > 5.0003) && (T < 5.0013), 0);"
    )
    reps = 200
    r = bngsim.Simulator(_ant(txt), method="ssa").run_replicates(
        reps, t_span=(0, 10), n_points=2, seed=3, squeeze=True
    )
    _within(_col(r, "A")[:, -1].mean(), 10.0, reps)


@pytest.mark.parametrize("expr", ["2time()", "2time", "0.2time()*10"])
def test_implicit_multiplication_by_the_clock(tmp_path, expr):
    """ExprTk reads ``2time()`` as 2*time(); a scan that took ``2time`` for a
    number froze the rate at 0 (∫₀¹⁰ 2t dt = 100)."""
    p = tmp_path / "m.net"
    p.write_text(_net_text(f"    1 f() {expr}\n", "    1 0 1 f\n"))
    reps = 200
    r = bngsim.Simulator(bngsim.Model.from_net(str(p)), method="ssa").run_replicates(
        reps, t_span=(0, 10), n_points=2, seed=3, squeeze=True
    )
    _within(np.asarray(r.observables)[:, -1, 0].mean(), 100.0, reps)


@pytest.mark.parametrize("trigger", ["time > 5", "2time() > 10"])
def test_a_clock_trigger_fires_with_nothing_to_fire(trigger):
    """The discrete loop looks only at triggers that read the clock; one it
    took for a state trigger never fired."""
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    a = b.add_species("A", 0.0)
    s = b.add_species("B", 0.0)
    b.add_observable("Aobs", [(a, 1.0)])
    b.add_reaction([s], [], "elementary", "k")
    b.add_event("E", trigger, [(a, "100")])
    m = bngsim.Model(_core=b.build())
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 10), n_points=2, seed=1)
    assert np.asarray(r.species)[-1, 0] == 100.0


def test_a_clamp_beside_a_moving_rule():
    """``x' = piecewise(-1, x > 0, 0)`` beside ``z' = 1``: the stuck-step test
    has to look at x alone; z moving hid it and the run crawled for minutes."""
    m = _ant(
        "species A = 10; J: A => ; 1e-6*A; x = 3; z = 0; x' = piecewise(-1, x > 0, 0); z' = 1"
    )
    r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 6), n_points=2, seed=1, timeout=30)
    assert abs(_col(r, "x")[-1]) < 1e-8
    assert _col(r, "z")[-1] == pytest.approx(6.0, rel=1e-9)


def test_a_fast_forcing_at_a_large_clock():
    """At t = 1e12 a panel of a few hundred ulps of t is a legitimate step for a
    forcing of period 10; the no-headway guard counts only panels at the floor."""
    t0, h = 1e12, 2e4
    m = _ant(f"species A = 0; J: => A; 1 + sin(2*pi*(time - {t0})/10);")
    r = bngsim.Simulator(m, method="ssa").run(t_span=(t0, t0 + h), n_points=2, seed=1, timeout=60)
    assert abs(_col(r, "A")[-1] - h) < 5 * np.sqrt(h)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "compartment C = 1; C := 1 + 0.5*time; species S in C = 10; x = 0; x' = S;"
            " species A = 0; J: => A; 0.01;",
            id="rate-rule",
        ),
        pytest.param(
            "compartment C = 1; C := 1 + 0.5*time; species S in C = 10; species A = 0;"
            " E: at S < 6: A = 100;",
            id="trigger",
        ),
        pytest.param(
            "compartment C = 1; compartment D = 1; E: at time > 2: C = 2;"
            " species X in C = 50; species S in C = 10; species P in D = 0; k = 0.01;"
            " J: X => P; k*X*S*C;",
            id="cross-compartment-modifier",
        ),
    ],
)
def test_more_stale_concentration_reads_are_refused(text):
    m = _ant(text)
    assert "variable_compartment_read" in [i.code for i in m.validate_for_ssa()]


def test_a_compartment_ruled_by_a_constant_species_runs():
    txt = (
        "compartment C = 2; species P = 2; const P; C := P; species B in C = 50;"
        " species A in C = 0; k = 0.5; J: B => A; k*B*C;"
    )
    m = _ant(txt)
    assert m.validate_for_ssa() == []
