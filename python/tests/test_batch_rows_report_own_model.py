"""Every result reports through the model it ran on (issues #697, #698, #743).

The report remaps (an assignment-rule species' column and sensitivity row, a
variable-volume species' live concentration, the volume factors behind
``as_roadrunner`` amounts) read compartment sizes. They used to read them from
the Simulator's own model, and the volume factors from a cache filled by the
first run:

- a ``run_batch`` row runs on a clone that may have written a size, and was
  reported through the parent's (#743);
- ``run_batch(squeeze=True)`` stamped the stacked tensor, where every remap
  skips the 3-D layout, so the rows came back unmapped (#698);
- after a size write, a reused Simulator kept the first run's sizes (#697);
- a squeezed ``parameter_scan`` over a size took its AR redirect from the
  parent, so its output sensitivities were off by C_row/C_parent.

Oracles are closed forms: dS/dt = -k·[S] in compartment C, so [S] = e^(-k t/C)
for [S](0) = 1.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

K = 0.5
AR = (
    "compartment C = 2; species S in C = 1; substanceOnly species T in C; k = 0.5;"
    " R1: S -> ; k*S; T := 3*S*C"
)
T = np.array([0.0, 0.5, 1.0])


def _col(r, name):
    return np.asarray(r.species)[..., list(r.species_names).index(name)]


def _conc(C):
    return np.exp(-K * T / C)


@pytest.mark.parametrize("num_processors", [None, 2])
def test_batch_rows_writing_a_compartment(num_processors):
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(AR), sensitivity_params=["k"])
    sizes = (4.0, 8.0)
    rows = sim.run_batch(
        t_span=(0, 1), n_points=3, params=[{"C": c} for c in sizes], num_processors=num_processors
    )
    for C, r in zip(sizes, rows, strict=False):
        np.testing.assert_allclose(_col(r, "T"), 3 * _conc(C), rtol=1e-5)
        np.testing.assert_allclose(r.as_roadrunner(selections=["S"])["S"], C * _conc(C), rtol=1e-5)
        dT = -3 * (T / C) * _conc(C)
        np.testing.assert_allclose(
            np.ravel(r.output_sensitivities("species:T")), dT, rtol=1e-4, atol=1e-9
        )
        np.testing.assert_allclose(
            r.sensitivities[:, list(r.species_names).index("T"), 0], dT, rtol=1e-4, atol=1e-9
        )


SQUEEZE = """
compartment C = 2; compartment V = 1; V' = 0.5
species A in C = 10; species X in C; X := 2*A
substanceOnly species T in C; T := 3*A*C
substanceOnly species Y in V = 2
k = 0.5
J1: A => ; k*A*C
J2: Y => ; 0.1*Y
"""


def test_a_squeezed_batch_is_its_rows_stacked():
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(SQUEEZE), sensitivity_params=["k"])
    P = [{"k": 0.5}, {"k": 1.0}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = sim.run_batch(t_span=(0, 2), n_points=3, params=P)
        sq = sim.run_batch(t_span=(0, 2), n_points=3, params=P, squeeze=True)
    for i, (p, r) in enumerate(zip(P, rows, strict=False)):
        t = r.time
        A = 10 * np.exp(-p["k"] * t)  # [A]' = -k[A]·C / C
        np.testing.assert_allclose(_col(r, "X"), 2 * A, rtol=1e-5)
        np.testing.assert_allclose(_col(r, "T"), 3 * A, rtol=1e-5)  # amount / C
        np.testing.assert_allclose(_col(r, "Y"), 2 * np.exp(-0.1 * t) / (1 + 0.5 * t), rtol=1e-4)
        np.testing.assert_array_equal(np.asarray(sq.species)[i], np.asarray(r.species))
        np.testing.assert_array_equal(sq.sensitivities[i], r.sensitivities)
    np.testing.assert_allclose(
        np.asarray(sq.output_sensitivities("species:X"))[0].ravel(),
        np.ravel(rows[0].output_sensitivities("species:X")),
    )


def test_a_squeezed_ssa_batch_maps_its_rules():
    text = "compartment C = 2; species A in C = 10; species X in C; X := 2*A;" + (
        " substanceOnly species T in C; T := 3*A*C; k = 0.5; J1: A => ; k*A*C"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ssa")
    P = [{"k": 0.5}, {"k": 1.0}]
    rows = sim.run_batch(t_span=(0, 2), n_points=3, params=P, seed=1)
    sq = sim.run_batch(t_span=(0, 2), n_points=3, params=P, seed=1, squeeze=True)
    for i, r in enumerate(rows):
        np.testing.assert_array_equal(np.asarray(sq.species)[i], np.asarray(r.species))
        A = _col(r, "A")
        np.testing.assert_allclose(_col(r, "X"), 2 * A)
        np.testing.assert_allclose(_col(r, "T"), 3 * A)


STALE = """
compartment C = 2; substanceOnly species S in C = 200; k = 0.1; R1: S -> ; k*S
"""


def test_amounts_follow_a_compartment_write():
    """``S`` is an amount: 200·e^(-k t) at any C."""
    exact = 200 * np.exp(-0.1 * np.array([0.0, 5.0]))
    m = bngsim.Model.from_antimony_string(STALE)
    sim = bngsim.Simulator(m)
    sim.run(t_span=(0, 5), n_points=2)
    m.set_param("C", 4.0)
    m.reset()
    r = sim.run(t_span=(0, 5), n_points=2)
    np.testing.assert_allclose(r.as_roadrunner(selections=["S"])["S"], exact, rtol=1e-5)
    # Each point starts from the state at invocation: the declared amount.
    m.reset()
    for r in sim.parameter_scan("C", [2.0, 4.0, 8.0], t_span=(0, 5), n_points=2):
        np.testing.assert_allclose(r.as_roadrunner(selections=["S"])["S"], exact, rtol=1e-5)
    for r in sim.run_batch(t_span=(0, 5), n_points=2, params=[{"C": c} for c in (2.0, 4.0, 8.0)]):
        np.testing.assert_allclose(r.as_roadrunner(selections=["S"])["S"], exact, rtol=1e-5)
    # ...and a plain run after the scan, at the model's own size.
    r = sim.run(t_span=(0, 5), n_points=2)
    np.testing.assert_allclose(r.as_roadrunner(selections=["S"])["S"], exact, rtol=1e-5)


def test_a_squeezed_scan_over_a_compartment_reports_each_rows_sensitivity():
    def scan(squeeze):
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(AR), sensitivity_params=["k"])
        sim.run(t_span=(0, 1), n_points=2)  # the scan carries this state's dx/dk
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sim.parameter_scan("C", [4.0, 8.0], t_span=(0, 1), n_points=3, squeeze=squeeze)

    rows, sq = scan(False), scan(True)
    got = np.asarray(sq.output_sensitivities("species:T")).reshape(2, 3)
    want = np.array([np.ravel(r.output_sensitivities("species:T")) for r in rows])
    np.testing.assert_allclose(got, want, rtol=1e-12)
