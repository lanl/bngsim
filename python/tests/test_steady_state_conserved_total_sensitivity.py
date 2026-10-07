"""dY_ss/dp for a parameter that sets the initial amount of a conserved species
(issue #704).

With a conservation law the steady state keeps one thing of where the run
started, the conserved total: ``A <-> B`` settles at ``A* = kr·T/(kf + kr)``
with ``T = A(0) + B(0)``. A parameter that sets an initial amount moves ``T``,
and so the steady state. The reduced sensitivity solve held ``T`` fixed: the
column of ``A0`` came back ``[0, 0]`` for ``[1/3, 2/3]``, and where ``A0`` is in
a rate law too it came back the fixed-total partial, ``[0.1875, -0.1875]`` for
``[0.9375, 0.0625]``, with the wrong sign on B. Nothing was logged.

Expected values are closed forms, or central differences of steady states
solved again at a moved parameter, which recompute the totals from x(0) and go
nowhere near the sensitivity solve.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

METHODS = pytest.mark.parametrize("method", ["newton", "integration"])


def _net(tmp_path, name, params, species, reactions, groups=()):
    text = (
        "begin parameters\n"
        + "".join(f"  {i + 1} {k} {v}\n" for i, (k, v) in enumerate(params))
        + "end parameters\nbegin species\n"
        + "".join(f"  {i + 1} {s}\n" for i, s in enumerate(species))
        + "end species\nbegin reactions\n"
        + "".join(f"  {i + 1} {r}\n" for i, r in enumerate(reactions))
        + "end reactions\n"
    )
    if groups:
        text += "begin groups\n" + "".join(f"  {i + 1} {g}\n" for i, g in enumerate(groups))
        text += "end groups\n"
    path = tmp_path / f"{name}.net"
    path.write_text(text)
    return str(path)


def _isomerization(tmp_path, a0="A0", extra=(), groups=()):
    """A <-> B at kf = 1, kr = 0.5, from A = A0 = 3 and B = 0."""
    return _net(
        tmp_path,
        "ab",
        [("A0", 3), ("kf", 1), ("kr", 0.5), *extra],
        [f"A() {a0}", "B() 0"],
        ["1 2 kf", "2 1 kr"],
        groups,
    )


def _columns(path, params, method, **kw):
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode")
    out = sim.steady_state(sensitivity_params=params, method=method, tol=1e-12, **kw)
    assert out.converged
    return np.asarray(out.sensitivity), out


def _resolved(load, name, method, rel=1e-4):
    """d(steady state)/d``name`` by central differences of steady states solved
    again, at two steps and extrapolated."""

    def at(value):
        model = load()
        model.set_param(name, value)
        out = bngsim.Simulator(model, method="ode").steady_state(method=method, tol=1e-13)
        assert out.converged
        return np.asarray(out.concentrations)

    p0 = float(load()._core.get_param(name))

    def d(h):
        return (at(p0 + h) - at(p0 - h)) / (2 * h)

    h = rel * abs(p0)
    return (4 * d(h / 2) - d(h)) / 3


@METHODS
def test_a_parameter_that_sets_a_conserved_amount(tmp_path, method):
    """A* = kr·A0/(kf + kr): dA*/dA0 = 1/3 and dB*/dA0 = 2/3. The column was
    [0, 0]. The kf column was right, and is: -+kr·A0/(kf + kr)² = -+2/3."""
    got, _ = _columns(_isomerization(tmp_path), ["A0", "kf"], method)
    np.testing.assert_allclose(got, [[1 / 3, -2 / 3], [2 / 3, 2 / 3]], rtol=1e-7, atol=1e-9)


@METHODS
def test_each_species_of_a_total_has_its_own_seed(tmp_path, method):
    """A <-> B from A = A0 and B = B0: the total moves with either, whichever
    of the two species the law is solved for."""
    path = _net(
        tmp_path,
        "ab2",
        [("A0", 3), ("B0", 1.5), ("kf", 1), ("kr", 0.5)],
        ["A() A0", "B() B0"],
        ["1 2 kf", "2 1 kr"],
    )
    got, _ = _columns(path, ["A0", "B0"], method)
    np.testing.assert_allclose(got, [[1 / 3, 1 / 3], [2 / 3, 2 / 3]], rtol=1e-7, atol=1e-9)


@METHODS
def test_a_parameter_that_is_an_initial_amount_and_a_rate_constant(tmp_path, method):
    """A0 seeds A and is the B -> A rate constant: A* = A0²/(kf + A0), so
    dA*/dA0 = 15/16 and dB*/dA0 = 1/16 at A0 = 3, kf = 1. With the total held
    fixed the column was [0.1875, -0.1875]."""
    path = _net(
        tmp_path, "mixed", [("A0", 3), ("kf", 1)], ["A() A0", "B() 0"], ["1 2 kf", "2 1 A0"]
    )
    got, _ = _columns(path, ["A0", "kf"], method)
    np.testing.assert_allclose(got, [[15 / 16, -9 / 16], [1 / 16, 9 / 16]], rtol=1e-7, atol=1e-9)


@METHODS
def test_a_catalyst_set_by_a_parameter(tmp_path, method):
    """A -> A + X at kp and X -> at kd: A is conserved on its own, and
    X* = kp·A0/kd. dA*/dA0 = 1 and dX*/dA0 = kp/kd = 2 came back 0 and 0."""
    path = _net(
        tmp_path,
        "catalyst",
        [("A0", 3), ("kp", 1), ("kd", 0.5)],
        ["A() A0", "X() 0"],
        ["1 1,2 kp", "2 0 kd"],
    )
    got, _ = _columns(path, ["A0", "kp"], method)
    np.testing.assert_allclose(got, [[1.0, 0.0], [2.0, 6.0]], rtol=1e-7, atol=1e-9)


@METHODS
def test_an_initial_amount_set_through_a_derived_parameter(tmp_path, method):
    """A starts at Atot = 2·A0 + 1: the seed goes through the derived parameter,
    dA*/dA0 = 2/3 and dB*/dA0 = 4/3."""
    path = _isomerization(tmp_path, a0="Atot", extra=[("Atot", "2*A0+1")])
    got, _ = _columns(path, ["A0"], method)
    np.testing.assert_allclose(got[:, 0], [2 / 3, 4 / 3], rtol=1e-7, atol=1e-9)


@METHODS
def test_two_totals_moved_by_two_parameters(tmp_path, method):
    """A + B <-> AB with A + AB and B + AB conserved, each set by its own
    parameter. Against steady states solved again."""
    path = _net(
        tmp_path,
        "bind",
        [("A0", 3), ("B0", 2), ("kon", 1.5), ("koff", 0.7)],
        ["A() A0", "B() B0", "AB() 0"],
        ["1,2 3 kon", "3 1,2 koff"],
    )
    got, _ = _columns(path, ["A0", "B0", "kon"], method)
    for j, name in enumerate(["A0", "B0", "kon"]):
        want = _resolved(lambda: bngsim.Model.from_net(path), name, method)
        np.testing.assert_allclose(got[:, j], want, rtol=2e-6, atol=1e-8, err_msg=name)
    # The totals themselves: d(A + AB)/dA0 = 1, d(B + AB)/dA0 = 0, and so on.
    np.testing.assert_allclose(got[0] + got[2], [1, 0, 0], atol=1e-8)
    np.testing.assert_allclose(got[1] + got[2], [0, 1, 0], atol=1e-8)


@METHODS
def test_a_total_that_feeds_an_enzyme_cycle(tmp_path, method):
    """E + S <-> ES -> E + S2 -> S: total enzyme E0 and total substrate S0, with
    both in more than one species. Against steady states solved again."""
    path = _net(
        tmp_path,
        "cycle",
        [("E0", 1.2), ("S0", 4), ("k1", 2), ("k2", 0.5), ("k3", 1.1), ("k4", 0.8)],
        ["E() E0", "S() S0", "ES() 0", "P() 0"],
        ["1,2 3 k1", "3 1,2 k2", "3 1,4 k3", "4 2 k4"],
    )
    got, _ = _columns(path, ["E0", "S0", "k3"], method)
    for j, name in enumerate(["E0", "S0", "k3"]):
        want = _resolved(lambda: bngsim.Model.from_net(path), name, method)
        np.testing.assert_allclose(got[:, j], want, rtol=2e-6, atol=1e-8, err_msg=name)


def test_the_observable_and_expression_rows_follow(tmp_path):
    """The output sensitivities are projected from the species columns."""
    path = _isomerization(tmp_path, groups=["Atot 1", "Btot 2", "Both 1,2"])
    _, out = _columns(path, ["A0"], "newton")
    got = out.output_sensitivities(["observable:Atot", "observable:Btot", "observable:Both"])
    np.testing.assert_allclose(got[:, 0], [1 / 3, 2 / 3, 1.0], rtol=1e-7)


def test_a_parameter_named_twice_has_both_columns(tmp_path):
    got, _ = _columns(_isomerization(tmp_path), ["A0", "kf", "A0"], "newton")
    np.testing.assert_allclose(got[:, 0], [1 / 3, 2 / 3], rtol=1e-7)
    np.testing.assert_allclose(got[:, 2], [1 / 3, 2 / 3], rtol=1e-7)


@METHODS
def test_an_sbml_initial_assignment_from_a_parameter(method):
    """The same model with its initial amount set by an initialAssignment."""
    text = (
        "species A, B; A0 = 3; kf = 1; kr = 0.5; A = A0; B = 0\n"
        "J1: A -> B; kf*A\nJ2: B -> A; kr*B\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    out = sim.steady_state(sensitivity_params=["A0", "kf"], method=method, tol=1e-12)
    names = list(out.species_names)
    got = np.asarray(out.sensitivity)[[names.index("A"), names.index("B")]]
    np.testing.assert_allclose(got, [[1 / 3, -2 / 3], [2 / 3, 2 / 3]], rtol=1e-7, atol=1e-9)


@METHODS
def test_with_no_conservation_law_the_start_is_forgotten(tmp_path, method):
    """Control. -> A at kp, A -> at kd, from A = A0: A* = kp/kd whatever A0 is."""
    path = _net(
        tmp_path, "open", [("A0", 3), ("kp", 2), ("kd", 0.5)], ["A() A0"], ["0 1 kp", "1 0 kd"]
    )
    got, _ = _columns(path, ["A0", "kp"], method)
    np.testing.assert_allclose(got, [[0.0, 2.0]], rtol=1e-7, atol=1e-9)


@METHODS
def test_a_rate_constant_in_a_model_with_a_total(tmp_path, method):
    """Control. A total no requested parameter moves is held, as it was."""
    got, _ = _columns(_isomerization(tmp_path), ["kf", "kr"], method)
    np.testing.assert_allclose(got, [[-2 / 3, 4 / 3], [2 / 3, -4 / 3]], rtol=1e-7, atol=1e-9)


def test_a_species_moved_off_its_initial_condition_is_not_seeded(tmp_path):
    """Control. A is set to 5 by hand, so A0 no longer reaches the state the
    solve starts from and the steady state does not move with it (issue #113's
    rule, which the time course follows too)."""
    model = bngsim.Model.from_net(_isomerization(tmp_path))
    model.set_concentration("A()", 5.0)
    assert model.effective_ic_sensitivity(["A0"]) == {}
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["A0", "kf"], method="newton", tol=1e-12
    )
    # T = 5: A* = 5/3, dA*/dkf = -kr·T/(kf + kr)² = -10/9.
    np.testing.assert_allclose(out.concentrations, [5 / 3, 10 / 3], rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(out.sensitivity), [[0.0, -10 / 9], [0.0, 10 / 9]], rtol=1e-7, atol=1e-9
    )


def test_a_declared_seed_for_a_state_set_by_hand(tmp_path):
    """A is set to 5 by hand and declared to move at 2 per unit of A0: the
    total does, and dA*/dA0 = 2/3, dB*/dA0 = 4/3."""
    model = bngsim.Model.from_net(_isomerization(tmp_path))
    model.set_concentration("A()", 5.0)
    model.declare_ic_sensitivity({"A()": {"A0": 2.0}})
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["A0"], method="newton", tol=1e-12
    )
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [2 / 3, 4 / 3], rtol=1e-7)


def test_a_declared_zero_is_kept(tmp_path):
    """Control. A species still at its initial condition, declared not to move
    with any parameter: the core is told so, and does not fall back on the
    loader's record that A starts at A0."""
    model = bngsim.Model.from_net(_isomerization(tmp_path))
    model.declare_ic_sensitivity({"A()": {}})
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["A0", "kf"], method="newton", tol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(out.sensitivity), [[0.0, -2 / 3], [0.0, 2 / 3]], rtol=1e-7, atol=1e-12
    )


def test_the_models_state_is_left_where_it_was(tmp_path):
    """The seed is read before the model is moved to the steady state, and the
    model is put back (issue #705): a second solve gives the same columns."""
    sim = bngsim.Simulator(bngsim.Model.from_net(_isomerization(tmp_path)), method="ode")
    first = np.asarray(sim.steady_state(sensitivity_params=["A0"], tol=1e-12).sensitivity)
    second = np.asarray(sim.steady_state(sensitivity_params=["A0"], tol=1e-12).sensitivity)
    np.testing.assert_allclose(first[:, 0], [1 / 3, 2 / 3], rtol=1e-7)
    np.testing.assert_array_equal(first, second)


def test_the_core_seeds_a_direct_reference_with_nothing_handed_to_it(tmp_path):
    """Asked with no seeds at all, the core takes the loader's own record of
    which parameter a species starts at, as a time course does, for a species
    still at its initial condition."""
    from bngsim._bngsim_core import SteadyStateOptions, find_steady_state

    def solve(model):
        opts = SteadyStateOptions()
        opts.method = "newton"
        opts.tol = 1e-12
        opts.sensitivity_params = ["A0"]
        out = find_steady_state(model._core, opts)
        return np.asarray(out.sensitivity_data).reshape(2, 1)[:, 0]

    np.testing.assert_allclose(
        solve(bngsim.Model.from_net(_isomerization(tmp_path))), [1 / 3, 2 / 3], rtol=1e-7
    )
    moved = bngsim.Model.from_net(_isomerization(tmp_path))
    moved.set_concentration("A()", 5.0)
    np.testing.assert_allclose(solve(moved), [0.0, 0.0], atol=1e-12)


def test_the_steady_state_agrees_with_a_long_time_course(tmp_path):
    """Against the forward sensitivities of a run to t = 200, which integrate
    the seed and share nothing with the reduced solve."""
    path = _net(
        tmp_path,
        "bind2",
        [("A0", 3), ("B0", 2), ("kon", 1.5), ("koff", 0.7)],
        ["A() A0", "B() B0", "AB() 0"],
        ["1,2 3 kon", "3 1,2 koff"],
    )
    got, _ = _columns(path, ["A0", "B0"], "newton")
    run = bngsim.Simulator(
        bngsim.Model.from_net(path), method="ode", sensitivity_params=["A0", "B0"]
    ).run(t_span=(0.0, 200.0), n_points=3, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(got, np.asarray(run.sensitivities)[-1], rtol=1e-6, atol=1e-8)


COMPARTMENT = (
    "compartment C; C = 2; species A in C, B in C; A = 3; B = 0; kf = 1; kr = 0.5\n"
    "J1: A -> B; kf*A\nJ2: B -> A; kr*B\n"
)


def test_a_compartment_size_in_a_model_with_a_total_is_refused():
    """What is conserved is an amount, and the solve holds a total of
    concentrations: the column of the size came back 0 where the steady state
    moves with it, and with the total differentiated it is right in some
    models and far off in others. It is refused."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(COMPARTMENT), method="ode")
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"compartment size.*#704"):
        sim.steady_state(sensitivity_params=["C", "kf"])


def test_a_rate_constant_beside_a_compartment_runs():
    """Control. The same model in a column that is no size."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(COMPARTMENT), method="ode")
    out = sim.steady_state(sensitivity_params=["kf"], tol=1e-12)
    names = list(out.species_names)
    got = np.asarray(out.sensitivity)[[names.index("A"), names.index("B")], 0]
    np.testing.assert_allclose(got, [-2 / 3, 2 / 3], rtol=1e-7)


def test_a_compartment_size_with_no_conservation_law_is_not_refused():
    """Control. -> A at kp, A -> at kd: nothing is conserved, and the size's
    column is what it was."""
    text = (
        "compartment C; C = 2; species A in C; A = 3; kp = 1; kd = 0.5\n"
        "J1: -> A; kp\nJ2: A -> ; kd*A\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    out = sim.steady_state(sensitivity_params=["C", "kd"], tol=1e-12)
    assert out.converged


# ─── What the first review found ────────────────────────────────────────────


def _michaelis(tmp_path):
    """S + E <-> ES -> E + P: P is a pure sink, and is in the law S + ES + P."""
    return _net(
        tmp_path,
        "mm",
        [("S0", 4), ("E0", 1.2), ("kf", 1), ("kr", 0.5), ("kc", 0.3)],
        ["S() S0", "E() E0", "ES() 0", "P() 0"],
        ["1,2 3 kf", "3 1,2 kr", "3 2,4 kc"],
    )


@METHODS
def test_a_total_with_a_species_the_mask_leaves_out_is_held(tmp_path, method):
    """With P masked out, as the documented ``mask=~is_pure_sink()`` has it,
    P ends at S0 and S and ES at 0 whatever S0 is: the total is what the sink
    takes, and the S0 column is 0 as it was (an earlier cut of this branch
    gave the total to the law's dependent, [0.4, -0.6, 0.6]). The enzyme's
    total has no sink in it and moves E, which came back 0."""
    model = bngsim.Model.from_net(_michaelis(tmp_path))
    mask = ~np.asarray(model.is_pure_sink())
    assert list(mask) == [True, True, True, False]
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["S0", "E0"], method=method, tol=1e-12, mask=mask
    )
    got = np.asarray(out.sensitivity)
    np.testing.assert_allclose(got[:3, 0], [0.0, 0.0, 0.0], atol=1e-9)
    assert np.isnan(got[3]).all()
    np.testing.assert_allclose(got[:3, 1], [0.0, 1.0, 0.0], atol=1e-9)


def test_a_drain_through_a_masked_sink_is_held(tmp_path):
    """Control. A <-> B -> P with P masked out: A* = B* = 0 whatever A0 is."""
    path = _net(
        tmp_path,
        "drain",
        [("A0", 3), ("kf", 1), ("kr", 0.5), ("kp", 0.7)],
        ["A() A0", "B() 0", "P() 0"],
        ["1 2 kf", "2 1 kr", "2 3 kp"],
    )
    model = bngsim.Model.from_net(path)
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["A0"], tol=1e-12, mask=~np.asarray(model.is_pure_sink())
    )
    got = np.asarray(out.sensitivity)[:, 0]
    np.testing.assert_allclose(got[:2], [0.0, 0.0], atol=1e-9)
    assert np.isnan(got[2])


def _saved(tmp_path, set_after=None):
    model = bngsim.Model.from_net(_isomerization(tmp_path))
    model.set_concentration("A()", 5.0)
    model.save_concentrations()
    if set_after is not None:
        model.set_param("A0", set_after)
    return model


@pytest.mark.parametrize("set_after", [None, 7.0])
def test_a_saved_baseline_is_set_by_no_parameter(tmp_path, set_after):
    """``save_concentrations()`` makes the state its own baseline: A(0) is 5
    whatever A0 is, and ``set_param`` no longer moves it. The steady state does
    not move with A0."""
    model = _saved(tmp_path, set_after)
    assert model.effective_ic_sensitivity(["A0"]) == {}
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["A0", "kf"], tol=1e-12
    )
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 1], [-10 / 9, 10 / 9], rtol=1e-7)


def test_a_time_course_from_a_saved_baseline_is_seeded_by_no_parameter(tmp_path):
    """The same rule for the seeding a time course starts from, which the
    steady state takes its own from: dY/dA0 came back [1/3, 2/3] at t = 400
    for an A(0) that A0 does not set."""
    model = _saved(tmp_path)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["A0", "kf"])
    run = sim.run(t_span=(0.0, 400.0), n_points=3, rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1]
    np.testing.assert_allclose(got[:, 0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(got[:, 1], [-10 / 9, 10 / 9], rtol=1e-6)


def test_the_core_seeds_nothing_from_a_saved_baseline(tmp_path):
    """Control. Asked with no seeds at all, the core follows the same rule."""
    from bngsim._bngsim_core import SteadyStateOptions, find_steady_state

    opts = SteadyStateOptions()
    opts.method = "newton"
    opts.tol = 1e-12
    opts.sensitivity_params = ["A0"]
    out = find_steady_state(_saved(tmp_path)._core, opts)
    np.testing.assert_allclose(np.asarray(out.sensitivity_data).ravel(), [0.0, 0.0], atol=1e-12)


def test_the_core_seed_is_over_the_volume_of_an_amount_valued_species():
    """An amount-declared species in a compartment of size 2 starts at A0/2:
    the core's own seed was 1, and the column [1/3, 2/3] for [1/6, 1/3]."""
    from bngsim._bngsim_core import SteadyStateOptions, find_steady_state

    text = (
        "compartment C; C = 2; substanceOnly species A in C, B in C; A0 = 3; A = A0; B = 0;"
        " kf = 1; kr = 0.5\nJ1: A -> B; kf*A\nJ2: B -> A; kr*B\n"
    )
    model = bngsim.Model.from_antimony_string(text)
    opts = SteadyStateOptions()
    opts.method = "newton"
    opts.tol = 1e-12
    opts.sensitivity_params = ["A0"]
    out = find_steady_state(model._core, opts)
    np.testing.assert_allclose(np.asarray(out.sensitivity_data).ravel(), [1 / 6, 1 / 3], rtol=1e-7)


def test_a_state_a_run_advanced_is_refused_for_the_parameter_that_set_it(tmp_path):
    """After ``run()`` to t = 0.7 the total is still A0, and the state no
    longer says so: the seed is retired with the initial condition it
    described, and dY/dA0 came back [0, 0] for [1/3, 2/3]. A time course
    refuses sensitivities on such a state (GH #210). A rate constant's column
    does not go through the total and is what it is from a fresh start, and
    ``reset()`` brings the other back."""
    model = bngsim.Model.from_net(_isomerization(tmp_path))
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0.0, 0.7), n_points=3)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"carried-over.*#704"):
        sim.steady_state(sensitivity_params=["A0", "kf"], tol=1e-12)
    out = sim.steady_state(sensitivity_params=["kf"], tol=1e-12)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [-2 / 3, 2 / 3], rtol=1e-7)
    model.reset()
    out = sim.steady_state(sensitivity_params=["A0"], tol=1e-12)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [1 / 3, 2 / 3], rtol=1e-7)


def test_an_advanced_state_is_refused_for_a_declared_seed_too(tmp_path):
    """A seed the caller declared for a conserved species is of the state it
    was declared on."""
    model = bngsim.Model.from_net(_isomerization(tmp_path, a0="3"))
    model.declare_ic_sensitivity({"A()": {"kr": 2.0}})
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0.0, 0.7), n_points=3)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"carried-over.*#704"):
        sim.steady_state(sensitivity_params=["kr"], tol=1e-12)


def test_an_advanced_state_with_no_conservation_law_is_not_refused(tmp_path):
    """Control. With nothing conserved the steady state keeps nothing of the
    start, advanced or not."""
    path = _net(
        tmp_path,
        "open",
        [("A0", 3), ("kp", 1), ("kd", 0.5)],
        ["A() A0"],
        ["0 1 kp", "1 0 kd"],
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode")
    sim.run(t_span=(0.0, 0.7), n_points=3)
    out = sim.steady_state(sensitivity_params=["A0", "kp"], tol=1e-12)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[0], [0.0, 2.0], rtol=1e-7, atol=1e-12)


TWO_COMPARTMENTS = (
    "compartment C1, C2; C1 = 1; C2 = {c2}; species A in C1, B in C2; A0 = 3; A = A0; B = 0;"
    " kf = 1; kr = 0.5\nJ1: A -> B; kf*A\nJ2: B -> A; kr*B\n"
)


@pytest.mark.parametrize("params", [["A0"], ["kf"], ["A0", "kf"]])
def test_a_law_across_compartments_of_different_size_is_refused(params):
    """A in a compartment of size 1 and B in one of size 3: what is conserved
    is A + 3·B, and the law is found as A + B (issue #758). Every column of the
    reduced solve is off: dA*/dkf = -0.286 for -0.367, and dA*/dA0 = 1/3 for
    1/7 with the total differentiated."""
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(TWO_COMPARTMENTS.format(c2=3)), method="ode"
    )
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"different size.*#758"):
        sim.steady_state(sensitivity_params=params, tol=1e-12)


def test_a_law_across_compartments_of_one_size_runs():
    """Control for the refusal. With both at size 1 the law found is the one
    that holds."""
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(TWO_COMPARTMENTS.format(c2=1)), method="ode"
    )
    out = sim.steady_state(sensitivity_params=["A0", "kf"], tol=1e-12)
    names = list(out.species_names)
    got = np.asarray(out.sensitivity)[[names.index("A"), names.index("B")]]
    np.testing.assert_allclose(got[:, 0], [1 / 3, 2 / 3], rtol=1e-7)
    np.testing.assert_allclose(got[:, 1], [-2 / 3, 2 / 3], rtol=1e-7)


def test_a_fixed_species_set_by_a_parameter_moves_what_reads_it():
    """``$A`` at A0 makes X at kp·A and X decays at kd: X* = kp·A0/kd, and
    Y := 2·X. A fixed species is no unknown and is in no law, so its seed
    never reached the solve: dY/dA0 came back [0, 0, 0] for [1, 2, 4]."""
    text = (
        "species $A, X, $Y; A0 = 3; A = A0; X = 0; kp = 1; kd = 0.5; Y := 2*X\n"
        "J1: -> X; kp*A\nJ2: X -> ; kd*X\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    out = sim.steady_state(sensitivity_params=["A0", "kp"], tol=1e-12)
    names = list(out.species_names)
    got = np.asarray(out.sensitivity)[[names.index(n) for n in ("A", "X", "Y")]]
    np.testing.assert_allclose(got[:, 0], [1.0, 2.0, 4.0], rtol=1e-7)
    np.testing.assert_allclose(got[:, 1], [0.0, 6.0, 12.0], rtol=1e-7, atol=1e-12)


def _fixed_source(tmp_path):
    """$A at A0 makes X at kp; X decays at kd and leaves a count in P."""
    return _net(
        tmp_path,
        "fixed",
        [("A0", 3), ("kp", 1), ("kd", 0.5), ("ks", 0.2)],
        ["$A() A0", "X() 0", "P() 0"],
        ["1 1,2 kp", "2 0 kd", "2 2,3 ks"],
    )


def test_a_fixed_species_in_a_net_model_with_a_mask(tmp_path):
    """The same in a ``.net`` model, on the masked path: [0, 0, nan] for
    [1, 2, nan]."""
    model = bngsim.Model.from_net(_fixed_source(tmp_path))
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["A0", "kp"], tol=1e-12, mask=~np.asarray(model.is_pure_sink())
    )
    got = np.asarray(out.sensitivity)
    np.testing.assert_allclose(got[:2, 0], [1.0, 2.0], rtol=1e-7)
    np.testing.assert_allclose(got[:2, 1], [0.0, 6.0], rtol=1e-7, atol=1e-12)
    assert np.isnan(got[2]).all()


def test_a_fixed_species_set_by_hand_is_set_by_no_parameter(tmp_path):
    """Control. Moved off its initial condition, the fixed species is where it
    was put whatever A0 is (issue #113)."""
    model = bngsim.Model.from_net(_fixed_source(tmp_path))
    model.set_concentration("A()", 5.0)
    out = bngsim.Simulator(model, method="ode").steady_state(
        sensitivity_params=["A0", "kp"], tol=1e-12, mask=~np.asarray(model.is_pure_sink())
    )
    got = np.asarray(out.sensitivity)
    np.testing.assert_allclose(got[:2, 0], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(got[:2, 1], [0.0, 10.0], rtol=1e-7, atol=1e-12)


def _a_pair_and_a_pool(tmp_path):
    """A <-> B started by A0, and X made and lost with nothing conserved,
    started by X0."""
    return _net(
        tmp_path,
        "pool",
        [("A0", 3), ("X0", 2), ("kf", 1), ("kr", 0.5), ("kp", 1), ("kd", 0.5)],
        ["A() A0", "B() 0", "X() X0"],
        ["1 2 kf", "2 1 kr", "0 3 kp", "3 0 kd"],
    )


def test_on_an_advanced_state_a_parameter_that_sets_no_conserved_amount_runs(tmp_path):
    """X0 sets a species that is in no law: the steady state keeps nothing of
    it, on an advanced state as on a fresh one, and it is not refused where A0
    is."""
    sim = bngsim.Simulator(bngsim.Model.from_net(_a_pair_and_a_pool(tmp_path)), method="ode")
    sim.run(t_span=(0.0, 0.7), n_points=3)
    out = sim.steady_state(sensitivity_params=["X0", "kp"], tol=1e-12)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[2], [0.0, 2.0], rtol=1e-7, atol=1e-12)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"carried-over.*#704"):
        sim.steady_state(sensitivity_params=["A0"], tol=1e-12)
