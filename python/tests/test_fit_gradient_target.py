"""Issue #1363: a fit gradient compares data with what ``target`` names.

``Result.sse_gradient`` documented observable data, ``(n_times, n_obs)``, but
always scored it against the species, through the species sensitivities. With
as many observables as species, the shapes agree, and it returned a loss of 277
for 8.5 and a gradient of the wrong sign in both parameters, with no warning.
``target="observables"`` now compares with :attr:`Result.observables` through
:attr:`Result.sensitivities_observables`, in all three fit gradients; the
default stays ``"species"``.

A + B <-> AB at kon, koff, from A = 10, B = 5; observables Atot = A + AB,
Btot = B + AB, Cplx = AB, as many as there are species.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

NET = """begin parameters
 1 kon 0.5
 2 koff 0.3
end parameters
begin species
 1 A(b) 10
 2 B(a) 5
 3 A(b!1).B(a!1) 0
end species
begin reactions
 1 1,2 3 kon
 2 3 1,2 koff
end reactions
begin groups
 1 Atot 1,3
 2 Btot 2,3
 3 Cplx 3
end groups
"""
NOMINAL = {"kon": 0.5, "koff": 0.3}


def _run(path, params=None, sens=True):
    model = bngsim.Model.from_net(str(path))
    if params:
        model.set_params(params)
    kw = {"sensitivity_params": ["kon", "koff"]} if sens else {}
    sim = bngsim.Simulator(model, method="ode", **kw)
    sim.set_tolerances(rtol=1e-11, atol=1e-13)
    return sim.run(t_span=(0, 5), n_points=6)


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "bind.net"
    path.write_text(NET)
    return path


def _central_difference(net, loss_of_observables):
    """d loss / d(kon, koff) by central differences over re-runs."""
    out = []
    for name in ("kon", "koff"):
        h = 1e-4 * NOMINAL[name]
        up, down = dict(NOMINAL), dict(NOMINAL)
        up[name] += h
        down[name] -= h
        lu = loss_of_observables(np.asarray(_run(net, up, sens=False).observables))
        ld = loss_of_observables(np.asarray(_run(net, down, sens=False).observables))
        out.append((lu - ld) / (2 * h))
    return np.array(out)


def test_observable_data_is_scored_against_the_observables(net):
    """The issue's numbers: SSE 8.507 and a gradient of (-4.19, +5.92). On
    main, as species: 277.1 and (+93.7, -132.3)."""
    r = _run(net)
    obs = np.asarray(r.observables)
    assert obs.shape == np.asarray(r.species).shape
    data = 1.1 * obs
    loss, grad = r.sse_gradient(data, target="observables")
    assert loss == pytest.approx(float(np.sum((obs - data) ** 2)), rel=1e-14)
    fd = _central_difference(net, lambda o: float(np.sum((o - data) ** 2)))
    np.testing.assert_allclose(grad, fd, rtol=1e-6)
    assert grad[0] < 0 < grad[1]


@pytest.mark.parametrize("kind", ["chi2", "nll"])
def test_the_weighted_gradients_take_the_same_target(net, kind):
    r = _run(net)
    obs = np.asarray(r.observables)
    data = 1.1 * obs
    sigma = np.array([0.5, 1.0, 2.0])
    if kind == "chi2":
        loss, grad = r.chi2_gradient(data, sigma, target="observables")
        expected = float(np.sum(((obs - data) / sigma) ** 2))
        fd = _central_difference(net, lambda o: float(np.sum(((o - data) / sigma) ** 2)))
    else:
        loss, grad = r.neg_log_likelihood_gradient(data, sigma, target="observables")
        expected = 0.5 * float(
            np.sum(((obs - data) / sigma) ** 2)
            + obs.shape[0] * np.sum(np.log(2 * np.pi * sigma**2))
        )
        fd = _central_difference(net, lambda o: 0.5 * float(np.sum(((o - data) / sigma) ** 2)))
    assert loss == pytest.approx(expected, rel=1e-13)
    np.testing.assert_allclose(grad, fd, rtol=1e-6)


def test_a_subset_of_observables(net):
    r = _run(net)
    obs = np.asarray(r.observables)
    data = 1.1 * obs
    loss, grad = r.sse_gradient(data[:, [2]], target="observables", observable_indices=[2])
    assert loss == pytest.approx(float(np.sum((obs[:, 2] - data[:, 2]) ** 2)), rel=1e-14)
    fd = _central_difference(net, lambda o: float(np.sum((o[:, 2] - data[:, 2]) ** 2)))
    np.testing.assert_allclose(grad, fd, rtol=1e-6)


def test_the_default_target_is_the_species_as_before(net):
    """Control: without target, data is species data, through the species
    sensitivities, to the bit."""
    r = _run(net)
    species = np.asarray(r.species)
    data = 1.1 * species
    loss, grad = r.sse_gradient(data)
    assert loss == float(np.sum((species - data) ** 2))
    np.testing.assert_array_equal(
        grad, np.einsum("tsi,ts->i", r.sensitivities, 2.0 * (species - data))
    )
    assert r.sse_gradient(data, target="species")[1].tolist() == grad.tolist()


def test_observable_shaped_data_against_species_names_the_target(tmp_path):
    """With a fourth observable the shapes differ, and the species error says
    what the data's shape is."""
    path = tmp_path / "four.net"
    path.write_text(NET.replace(" 3 Cplx 3\n", " 3 Cplx 3\n 4 All 1,2,3\n"))
    r = _run(path)
    data = np.asarray(r.observables)
    with pytest.raises(ValueError, match=r"pass target='observables'"):
        r.sse_gradient(data)
    loss, _ = r.sse_gradient(data, target="observables")
    assert loss == 0.0


def test_the_arguments_are_checked(net):
    r = _run(net)
    data = np.asarray(r.observables)
    with pytest.raises(ValueError, match=r"target must be 'species' or 'observables'"):
        r.sse_gradient(data, target="obs")
    with pytest.raises(ValueError, match=r"species_indices selects species"):
        r.sse_gradient(data[:, [0]], target="observables", species_indices=[0])
    with pytest.raises(ValueError, match=r"observable_indices selects observables"):
        r.sse_gradient(data[:, [0]], observable_indices=[0])


def test_no_observable_sensitivities_is_refused(tmp_path):
    path = tmp_path / "none.net"
    path.write_text(NET.split("begin groups")[0])
    r = _run(path)
    assert not r.has_sensitivities_observables
    with pytest.raises(ValueError, match=r"No observable sensitivities"):
        r.sse_gradient(np.zeros((6, 0)), target="observables")


def test_a_per_column_sigma_counts_its_constant_at_every_time(net):
    """Found with this issue: the constant ln(2 pi sigma^2) of a per-column
    sigma was summed once per column, not once per data point, so the loss
    came out 8.05 for 21.84 (the gradient does not read it)."""
    r = _run(net)
    species = np.asarray(r.species)
    data = 1.1 * species
    sigma = np.array([0.5, 1.0, 2.0])
    var = np.broadcast_to(sigma**2, species.shape)
    expected = 0.5 * float(np.sum((species - data) ** 2 / var) + np.sum(np.log(2 * np.pi * var)))
    per_point = np.broadcast_to(sigma, species.shape)
    assert r.neg_log_likelihood_gradient(data, sigma)[0] == pytest.approx(expected, rel=1e-13)
    assert r.neg_log_likelihood_gradient(data, sigma)[0] == pytest.approx(
        r.neg_log_likelihood_gradient(data, per_point)[0], rel=1e-13
    )


def test_an_unknown_observable_row_the_loss_does_not_weight_is_left_out(net):
    """An observable that reads a refused species (GH #221) has a NaN row. A
    row the data matches exactly weighs nothing, and is left out, as a refused
    species row is; one the loss weighs keeps its NaN. The row is made NaN by
    hand here: the model refuses nothing."""
    r = _run(net)
    obs = np.asarray(r.observables)
    data = 1.1 * obs
    data[:, 0] = obs[:, 0]
    want = r.sse_gradient(data[:, [1, 2]], target="observables", observable_indices=[1, 2])[1]
    r._observable_sensitivities = r._observable_sensitivities.copy()
    r._observable_sensitivities[:, 0, :] = np.nan
    np.testing.assert_allclose(r.sse_gradient(data, target="observables")[1], want, rtol=1e-14)
    assert np.isnan(r.sse_gradient(1.1 * obs, target="observables")[1]).all()
