"""Issue #1368: Result.species reports amount/V_live in a resized compartment.

bngsim stores a species as ``amount/V_static``. Where a compartment's size
moves, by an event or a rate rule, the report passes rescale that to the live
concentration. Under SSA they did so only for the ``[X]`` selector of
``as_roadrunner``, and ``Result.species`` kept the stored value. For a
concentration species in a compartment an event doubles, the SSA mean read
twice the ODE value from the event on, at z of about 90, with no warning. So
did ODE for an amount species (``hasOnlySubstanceUnits``) after an event. Of
the eight cases (event or rate rule, hOSU or not, ODE or SSA), five were stale.

``Result.species`` is now the live concentration in all eight, equal to
``[X]``; ``Result.state`` keeps the stored value, whose ``V_static`` multiple is
the molecule count; and the bare ``X`` selector is still the amount.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

T = np.arange(4.0)


def _model(hosu: bool, resize: str):
    species = "substanceOnly species X in c; X = 100;" if hosu else "species X in c; X = 50;"
    change = "c' = 1;" if resize == "rate rule" else "E1: at time > 1: c = 4;"
    return bngsim.Model.from_antimony_string(
        f"compartment c = 2;\n{species}\nk2 = 0.5;\nR2: X -> ; k2*X;\n{change}\n"
    )


def _run(hosu, resize, method, seed=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(_model(hosu, resize), method=method)
        return sim.run(t_span=(0, 3), n_points=4, seed=seed)


def _stored(r):
    cols = [list(r.state_names).index(n) for n in r.species_names]
    return np.asarray(r.state)[:, cols]


CASES = [
    pytest.param(h, z, m, id=f"{z}-{'hosu' if h else 'conc'}-{m}")
    for z in ("event", "rate rule")
    for h in (False, True)
    for m in ("ode", "ssa")
]


@pytest.mark.parametrize(("hosu", "resize", "method"), CASES)
def test_species_is_the_live_concentration(hosu, resize, method):
    """Result.species equals [X] = amount/V_live; the bare X selector is the
    amount; Result.state is the stored value, amount/V_static except for a
    concentration species under ODE, whose dilution is integrated and which is
    stored live. On main, five of the eight cases reported the stored value in
    Result.species."""
    r = _run(hosu, resize, method, seed=3)
    species = np.asarray(r.species)[:, 0]
    conc = r.as_roadrunner(["[X]"])[:, 0]
    amount = r.as_roadrunner(["X"])[:, 0]
    v_live = 2.0 + T if resize == "rate rule" else np.where(T > 1, 4.0, 2.0)
    np.testing.assert_allclose(species, conc, rtol=1e-12)
    np.testing.assert_allclose(species * v_live, amount, rtol=1e-9)
    if hosu or method == "ssa":
        np.testing.assert_allclose(_stored(r)[:, 0] * 2.0, amount, rtol=1e-9)
    else:
        np.testing.assert_allclose(_stored(r)[:, 0], species, rtol=1e-12)
    if method == "ssa":
        np.testing.assert_allclose(amount, np.round(amount), atol=1e-9)


def test_the_ssa_mean_tracks_the_ode_across_an_event_resize():
    """The issue's model: [X] = 50 in c = 2, decaying at 0.5 per amount; the
    event at t = 1 doubles c, so [X] halves. On main the SSA mean read 34.4 and
    30.4 at t = 2, 3 against 17.2 and 15.2."""
    ode = np.asarray(_run(False, "event", "ode").species)[:, 0]
    ssa = np.array(
        [np.asarray(_run(False, "event", "ssa", seed=s + 1).species)[:, 0] for s in range(200)]
    )
    mean, se = ssa.mean(0), ssa.std(0, ddof=1) / np.sqrt(len(ssa))
    analytic = np.where(T <= 1, 50 * np.exp(-0.25 * T), 25 * np.exp(-0.25 - 0.125 * (T - 1)))
    np.testing.assert_allclose(ode, analytic, rtol=1e-6)
    z = np.abs(mean[1:] - ode[1:]) / se[1:]
    assert (z < 5.0).all(), z


def test_an_event_resized_amount_species_differentiates_its_reported_value():
    """ODE, hOSU = true, an event to c = cnew: the reported [X] is amount/cnew
    after it, and its sensitivity rows are those of that value (the quotient
    rule over the compartment's observable row), against a central difference."""
    text = (
        "compartment c = 2;\nsubstanceOnly species X in c; X = 100;\n"
        "k2 = 0.5; cnew = 4;\nR2: X -> ; k2*X;\nE1: at time > 1.05: c = cnew;\n"
    )
    nominal = {"k2": 0.5, "cnew": 4.0}

    def solve(params=None, sens=True):
        model = bngsim.Model.from_antimony_string(text)
        if params:
            model.set_params(params)
        kw = {"sensitivity_params": ["k2", "cnew"]} if sens else {}
        sim = bngsim.Simulator(model, method="ode", **kw)
        sim.set_tolerances(rtol=1e-11, atol=1e-13)
        return sim.run(t_span=(0, 3), n_points=4)

    r = solve()
    analytic = np.asarray(r.sensitivities)[:, 0, :]
    fd = []
    for name in ("k2", "cnew"):
        h = 1e-5 * nominal[name]
        up, down = dict(nominal), dict(nominal)
        up[name] += h
        down[name] -= h
        lu = np.asarray(solve(up, sens=False).species)[:, 0]
        ld = np.asarray(solve(down, sens=False).species)[:, 0]
        fd.append((lu - ld) / (2 * h))
    np.testing.assert_allclose(analytic, np.array(fd).T, rtol=1e-6, atol=1e-9)
    assert analytic[2, 1] < 0.0  # a larger cnew dilutes [X]
