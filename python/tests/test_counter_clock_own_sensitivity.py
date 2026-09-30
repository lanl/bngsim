"""Issue #725 — a counter-clock crossing moves with the clock's own sensitivity.

A unit-rate counter c(t) = c(t_start) + (t − t_start) crosses a threshold θ at
t* = t_start + θ − c(t_start), so ∂t*/∂p = ∂θ/∂p − s_clock(t*). The switch-time
jump used only ∂θ/∂p, so the columns for the counter's seed, its rate constant
and its initial-condition axis came back exactly 0, with no warning, where the
closed form is O(1). The threshold columns were right.

Closed forms for C(0) = c0 = 0.5, rate rc = 1, X' = k after t ≥ sigma and
Y' = k2 while t < tau: t*_X = (sigma − c0)/rc = 2.5, t*_Y = (tau − c0)/rc = 3.5,
X(5) = k·(5 − t*_X) and Y(5) = k2·t*_Y, so

    dX/d[c0, rc, sigma, tau] = [ k/rc,  k·t*_X/rc,  −k/rc,  0 ] = [2, 5, −2, 0]
    dY/d[c0, rc, sigma, tau] = [−k2/rc, −k2·t*_Y/rc, 0, k2/rc] = [−1, −3.5, 0, 1]

and along C's own IC axis, dX/dC(0) = 2 and dY/dC(0) = −1.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

NET = """\
begin parameters
    1 c0     0.5
    2 rc     1
    3 sigma  3
    4 k      2
    5 tau    4
    6 k2     1
end parameters
begin functions
    1 rate_X() if(t>=sigma,k,0)
    2 rate_Y() if(t<tau,k2,0)
end functions
begin species
    1 C() c0
    2 X() 0
    3 Y() 0
end species
begin reactions
    1 0 1 rc
    2 0 2 rate_X
    3 0 3 rate_Y
end reactions
begin groups
    1 t                    1
end groups
"""


def _model(tmp_path, text=NET):
    p = tmp_path / "clock.net"
    p.write_text(text)
    return bngsim.Model.from_net(p)


def _final(run, species):
    names = list(run.species_names)
    return names.index(species)


def test_the_clocks_seed_and_rate_columns_match_the_closed_form(tmp_path):
    run = bngsim.Simulator(
        _model(tmp_path), method="ode", sensitivity_params=["c0", "rc", "sigma", "tau"]
    ).run(sample_times=[0, 1, 2, 3, 4, 5], rtol=1e-10, atol=1e-12)
    s = np.asarray(run.sensitivities)[-1]
    np.testing.assert_allclose(s[_final(run, "X()")], [2, 5, -2, 0], atol=1e-7)
    np.testing.assert_allclose(s[_final(run, "Y()")], [-1, -3.5, 0, 1], atol=1e-7)


def test_the_clocks_own_ic_axis_is_jumped(tmp_path):
    run = bngsim.Simulator(_model(tmp_path), method="ode", sensitivity_ic=["C()"]).run(
        sample_times=[0, 1, 2, 3, 4, 5], rtol=1e-10, atol=1e-12
    )
    s = np.asarray(run.sensitivities_ic)[-1]
    assert s[_final(run, "X()"), 0] == pytest.approx(2.0, abs=1e-7)
    assert s[_final(run, "Y()"), 0] == pytest.approx(-1.0, abs=1e-7)


def test_a_counter_moved_off_its_initial_value(tmp_path):
    """C set to 1.5 before the run: c0 no longer reaches the clock (its IC seed
    is retired, issue #113), and the crossing is at t = 1.5, so
    dX/d[c0, rc, sigma] = [0, k·1.5/rc, −k/rc] = [0, 3, −2]. The clock's own
    sensitivity is read from the run, not assumed from the IC."""
    model = _model(tmp_path)
    model.set_concentration("C()", 1.5)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["c0", "rc", "sigma"]).run(
        sample_times=[0, 1, 2, 3, 4, 5], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(run.sensitivities)[-1, _final(run, "X()")], [0, 3, -2], atol=1e-7
    )


def test_compute_all_default_columns_include_the_counter_rate(tmp_path):
    """BNG2.pl writes ``0 -> counter() 1`` with a synthesized ``_rateLaw1``;
    compute_all_sensitivities asks for it by default. It moves the crossing."""
    text = NET.replace("    2 rc     1\n", "    2 _rateLaw1 1\n").replace(
        "    1 0 1 rc\n", "    1 0 1 _rateLaw1\n"
    )
    result = bngsim.Simulator(_model(tmp_path, text), method="ode").compute_all_sensitivities(
        t_span=(0.0, 5.0), n_points=6, params=["_rateLaw1"], rtol=1e-10, atol=1e-12
    )
    x = list(result.species_names).index("X()")
    assert float(np.asarray(result.sensitivities)[-1, x, 0]) == pytest.approx(5.0, abs=1e-6)


# Two switches on the same counter at the same instant: X at t >= s1, Z at t >= s2,
# s1 = s2 = 3, so both cross at t* = 2.5. The core reads a crossing's jump by
# nudging the clock, which flips both, so each is isolated by its own threshold
# parameter (issue #375) — and a crossing only the clock moves must be kept inside
# such a group too, or its columns come back 0.
COINCIDENT = """\
begin parameters
    1 c0     0.5
    2 rc     1
    3 s1     3
    4 s2     3
    5 k      2
    6 kz     5
end parameters
begin functions
    1 rate_X() if(t>=s1,k,0)
    2 rate_Z() if(t>=s2,kz,0)
end functions
begin species
    1 C() c0
    2 X() 0
    3 Z() 0
end species
begin reactions
    1 0 1 rc
    2 0 2 rate_X
    3 0 3 rate_Z
end reactions
begin groups
    1 t                    1
end groups
"""
# dX/d[c0, rc] = [k/rc, k·t*/rc] = [2, 5]; dZ/d[c0, rc] = [kz/rc, kz·t*/rc] = [5, 12.5].


@pytest.mark.parametrize(
    "params,want_x,want_z",
    [
        (["c0"], [2.0], [5.0]),
        (["rc"], [5.0], [12.5]),
        (["c0", "s1"], [2.0, -2.0], [5.0, 0.0]),
    ],
    ids=["seed", "rate", "seed-and-one-threshold"],
)
def test_coinciding_crossings_on_a_counter_each_move_with_it(tmp_path, params, want_x, want_z):
    run = bngsim.Simulator(
        _model(tmp_path, COINCIDENT), method="ode", sensitivity_params=params
    ).run(sample_times=[0, 1, 2, 3, 4, 5], rtol=1e-10, atol=1e-12)
    s = np.asarray(run.sensitivities)[-1]
    np.testing.assert_allclose(s[_final(run, "X()")], want_x, atol=1e-7)
    np.testing.assert_allclose(s[_final(run, "Z()")], want_z, atol=1e-7)


def test_coinciding_crossings_along_the_clocks_ic_axis(tmp_path):
    run = bngsim.Simulator(_model(tmp_path, COINCIDENT), method="ode", sensitivity_ic=["C()"]).run(
        sample_times=[0, 1, 2, 3, 4, 5], rtol=1e-10, atol=1e-12
    )
    s = np.asarray(run.sensitivities_ic)[-1]
    assert s[_final(run, "X()"), 0] == pytest.approx(2.0, abs=1e-7)
    assert s[_final(run, "Z()"), 0] == pytest.approx(5.0, abs=1e-7)


@pytest.mark.parametrize("chunk", [1, 2, 10])
def test_compute_all_does_not_depend_on_the_chunk_size(tmp_path, chunk):
    """Which chunk a column lands in must not decide its answer (#243)."""
    sim = bngsim.Simulator(_model(tmp_path, COINCIDENT), method="ode")
    result = sim.compute_all_sensitivities(
        t_span=(0.0, 5.0),
        n_points=6,
        params=["c0", "rc", "s1", "s2"],
        chunk_size=chunk,
        rtol=1e-10,
        atol=1e-12,
    )
    s = np.asarray(result.sensitivities)[-1]
    names = list(result.species_names)
    np.testing.assert_allclose(s[names.index("X()")], [2, 5, -2, 0], atol=1e-6)
    np.testing.assert_allclose(s[names.index("Z()")], [5, 12.5, 0, -5], atol=1e-6)


def test_run_batch_matches_run_on_an_ic_only_request(tmp_path):
    sim = bngsim.Simulator(_model(tmp_path), method="ode", sensitivity_ic=["C()"])
    single = sim.run(t_span=(0.0, 5.0), n_points=6, rtol=1e-10, atol=1e-12)
    batch = sim.run_batch(
        t_span=(0.0, 5.0), n_points=6, params=[{"k": 2.0}], rtol=1e-10, atol=1e-12
    )[0]
    x = _final(single, "X()")
    assert np.asarray(single.sensitivities_ic)[-1, x, 0] == pytest.approx(2.0, abs=1e-7)
    assert np.asarray(batch.sensitivities_ic)[-1, x, 0] == pytest.approx(2.0, abs=1e-7)
