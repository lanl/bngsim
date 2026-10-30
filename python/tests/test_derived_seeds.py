"""Issue #1365: rows of a batch, an ensemble or a scan get derived seeds.

Row *i* was seeded ``base + i``, so an ensemble from seed 1 was the one from
seed 0 shifted by a row: 49 of 50 trajectories bit-identical, and error bars or
between-ensemble comparisons built from consecutive bases looked far tighter
than they are. Each row's seed is now derived from ``(base, i)`` by numpy's
``SeedSequence``: nearby bases are unrelated, one call's seeds are distinct, and
``Result.seed`` still reruns its row alone.

A birth-death network, 0 -> A at k1 = 20, A -> 0 at k2 = 1, from A = 0.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest
from bngsim._seed import derived_seeds

NET = (
    "begin parameters\n 1 k1 20\n 2 k2 1\nend parameters\n"
    "begin species\n 1 A() 0\nend species\n"
    "begin reactions\n 1 0 1 k1\n 2 1 0 k2\nend reactions\n"
)
RUN = dict(t_span=(0, 10), n_points=11)


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "bd.net"
    path.write_text(NET)
    return str(path)


def _sim(net):
    return bngsim.Simulator(bngsim.Model.from_net(net), method="ssa")


def _paths(results) -> set[bytes]:
    return {np.asarray(r.species)[:, 0].tobytes() for r in results}


def test_ensembles_from_neighbouring_bases_share_no_trajectory(net):
    """On main, seed=1's 50 replicates were 49 of seed=0's."""
    r0 = _sim(net).run_replicates(50, seed=0, **RUN)
    r1 = _sim(net).run_replicates(50, seed=1, **RUN)
    assert len(_paths(r0)) == 50
    assert _paths(r0).isdisjoint(_paths(r1))


def test_batches_from_neighbouring_bases_share_no_trajectory(net):
    b0 = _sim(net).run_batch(params=[{}] * 20, seed=0, **RUN)
    b1 = _sim(net).run_batch(params=[{}] * 20, seed=1, **RUN)
    assert _paths(b0).isdisjoint(_paths(b1))


def test_scans_from_neighbouring_bases_share_no_trajectory(net):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s0 = _sim(net).parameter_scan("k1", par_scan_vals=[20.0] * 8, seed=0, **RUN)
        s1 = _sim(net).parameter_scan("k1", par_scan_vals=[20.0] * 8, seed=1, **RUN)
    assert _paths(s0).isdisjoint(_paths(s1))


def test_a_replicate_reruns_alone_from_its_seed(net):
    reps = _sim(net).run_replicates(10, seed=5, **RUN)
    assert [r.seed for r in reps] == derived_seeds(5, 10)
    one = _sim(net).run(seed=reps[7].seed, **RUN)
    np.testing.assert_array_equal(np.asarray(one.species), np.asarray(reps[7].species))


def test_threads_draw_the_seeds_the_sequential_path_does(net):
    seq = _sim(net).run_replicates(12, seed=9, **RUN)
    par = _sim(net).run_replicates(12, seed=9, num_processors=4, **RUN)
    assert [r.seed for r in par] == [r.seed for r in seq]
    for a, b in zip(seq, par, strict=True):
        np.testing.assert_array_equal(np.asarray(a.species), np.asarray(b.species))


def test_the_seeds_of_one_call_are_distinct_and_fit_the_backends():
    """200000 31-bit draws repeat about nine times; a repeat would be a
    duplicated trajectory, which base + i never gave."""
    seeds = derived_seeds(5, 200_000)
    assert len(set(seeds)) == len(seeds)
    assert min(seeds) >= 0 and max(seeds) < 2**31


def test_a_rows_seed_does_not_depend_on_how_many_rows_there_are():
    assert derived_seeds(3, 10) == derived_seeds(3, 1000)[:10]
    assert derived_seeds(3, 0) == []


def test_bases_next_to_each_other_share_no_seed():
    assert set(derived_seeds(0, 1000)).isdisjoint(derived_seeds(1, 1000))
