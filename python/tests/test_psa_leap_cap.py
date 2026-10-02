"""A PSA leap never takes more of a species than it holds.

The leap of a reaction is floor(n_min / N_c) fires' worth, n_min its smallest
population, and a fire moves each species by its stoichiometry times that. When
a stoichiometry exceeds N_c, that is more than n_min: `3A ->` at N_c = 2 with
A = 4 leapt by 2 and took 6, and the count went negative (warned). The leap is
now also at most floor(n / |stoichiometry|) for such a species. With N_c at
least every stoichiometry, which is every usual setting, nothing changes: the
leap is run_network's.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")


@pytest.mark.parametrize(
    ("text", "name", "conserved"),
    [
        ("species A = 4000; k = 1e-10; J: 3A => ; k*A*A*A;", "A", None),
        (
            "species A = 4000; species B = 0; k = 1e-10; J: 3A => B; k*A*A*A;",
            "A",
            {"A": 1, "B": 3},
        ),
        (
            "species A = 300; species B = 0; k = 1e-6; J: 4A => B; k*A*A*A*A;",
            "A",
            {"A": 1, "B": 4},
        ),
    ],
)
def test_a_leap_with_a_stoichiometry_above_poplevel_never_overdraws(text, name, conserved):
    model = bngsim.Model.from_antimony_string(text)
    names = list(model.species_names)
    for seed in range(60):
        model.reset()
        r = bngsim.Simulator(model, method="psa", poplevel=2.0).run(
            t_span=(0, 50), n_points=51, seed=seed, timeout=20
        )
        x = np.asarray(r.species)
        assert x[:, names.index(name)].min() >= 0
        if conserved:
            total = sum(w * x[:, names.index(s)] for s, w in conserved.items())
            assert np.all(total == total[0])


def test_a_reactant_that_is_also_a_product_is_no_negative_crossing():
    """`3A -> A + B` takes three A and gives one back. The firing subtracted
    every reactant before adding any product, and A's dip through zero between
    the two was counted as a crossing: a warning in 36 of 40 runs at
    poplevel 1.5 whose A never went below 0."""
    import warnings

    model = bngsim.Model.from_antimony_string(
        "species A = 3000; species B = 0; k = 1e-7; J: 3A => A + B; k*A*A*A;"
    )
    for seed in range(20):
        model.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("error", bngsim.SsaBoundaryWarning)
            r = bngsim.Simulator(model, method="psa", poplevel=1.5).run(
                t_span=(0, 50), n_points=11, seed=seed, timeout=20
            )
        assert np.asarray(r.species)[:, 0].min() >= 0


@pytest.mark.parametrize("bad", [1.0, 0.5, float("nan")])
def test_a_poplevel_not_above_one_is_refused(bad):
    """NaN passed the `poplevel <= 1` check and ran as exact SSA."""
    model = bngsim.Model.from_antimony_string("species A = 100; J: A => ; 0.1*A;")
    with pytest.raises(ValueError, match="poplevel"):
        bngsim.Simulator(model, method="psa", poplevel=bad)
