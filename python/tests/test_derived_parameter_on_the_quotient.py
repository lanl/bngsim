"""A derived parameter's column on the difference-quotient path (issue #707).

``kd = 2*k0`` and a rate law the analytic sensitivity right-hand side declines,
``kd*max(1, 0*Atot)*Atot``, so CVODES takes each column by its internal difference
quotient: it moves one parameter, reads the right-hand side, and puts it back.

bngsim mirrors the moved value into the model and re-derives the derived
parameters, so that a moved primary reaches what is built from it. A moved
*derived* parameter was re-derived with the rest: kd went back to 2·k0 at the
unmoved k0 before the right-hand side was read, the moved right-hand side was
the nominal one, and the column for kd came back exactly 0, with no warning.

The moved parameter is held now, and what reads it is re-derived from it.

A column for a derived parameter is taken on its own terms, as an independent
axis (issue #475): A = A0·e^(−kd·t), so dA/dkd = −A0·t·e^(−kd·t), and through
the chain dA/dk0 = 2·dA/dkd.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 k0 0.5
    2 kd 2*k0
    3 ke 3*kd
    4 A0 10
    5 ks 4
end parameters
begin functions
    1 rf() {law}
end functions
begin species
    1 A() {ic}
end species
begin reactions
    1 1 0 rf{source}
end reactions
begin groups
    1 Atot 1
end groups
"""
T = np.linspace(0.0, 2.0, 5)
# A rate constant of kd, written so that the analytic path declines it. Not
# as ``kd*abs(Atot)/Atot``, a sign written as a quotient, which a run on the
# quotient is refused for (issue #938).
DECLINED = "kd*max(1,0*Atot)"
ANALYTIC = "kd"


def _model(tmp_path, law=None, ic="A0", source=""):
    path = tmp_path / "m.net"
    path.write_text(NET.format(law=DECLINED if law is None else law, ic=ic, source=source))
    return bngsim.Model.from_net(path)


def _columns(tmp_path, law, params, ic="A0", source=""):
    path = tmp_path / "m.net"
    path.write_text(NET.format(law=law, ic=ic, source=source))
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=params)
    run = sim.run(sample_times=list(T), rtol=1e-10, atol=1e-12)
    return sim, np.asarray(run.sensitivities)[:, 0, :]


def _dA_dk(k):
    return -10.0 * T * np.exp(-k * T)


@pytest.mark.parametrize("params", [["kd"], ["kd", "k0"], ["k0", "kd"]], ids="-".join)
def test_a_derived_parameters_column_on_the_quotient(tmp_path, params):
    """The column for kd was exactly 0 in every order it was asked for in."""
    sim, got = _columns(tmp_path, DECLINED, params)
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(got[:, params.index("kd")], _dA_dk(1.0), rtol=2e-5, atol=1e-7)
    if "k0" in params:
        np.testing.assert_allclose(
            got[:, params.index("k0")], 2 * _dA_dk(1.0), rtol=2e-5, atol=1e-7
        )


def test_what_reads_the_moved_parameter_moves_with_it(tmp_path):
    """``ke = 3*kd`` in the rate law: A = A0·e^(−ke·t). On its own terms kd
    carries ke with it, so dA/dkd = 3·dA/dke, and both were 0."""
    _sim, got = _columns(tmp_path, DECLINED.replace("kd", "ke"), ["kd", "ke"])
    np.testing.assert_allclose(got[:, 1], _dA_dk(3.0), rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(got[:, 0], 3 * _dA_dk(3.0), rtol=2e-5, atol=1e-7)


def test_a_primarys_column_on_the_quotient(tmp_path):
    """Control. k0 has no expression to go back to: dA/dk0 = 2·dA/dkd."""
    sim, got = _columns(tmp_path, DECLINED, ["k0"])
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(got[:, 0], 2 * _dA_dk(1.0), rtol=2e-5, atol=1e-7)


def test_the_same_column_on_the_analytic_path(tmp_path):
    """Control. ``rf() = kd`` is differentiated, and the column was right."""
    sim, got = _columns(tmp_path, ANALYTIC, ["kd", "k0"])
    assert sim.has_analytic_sens_rhs
    np.testing.assert_allclose(got[:, 0], _dA_dk(1.0), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got[:, 1], 2 * _dA_dk(1.0), rtol=1e-7, atol=1e-9)


def test_the_models_parameters_are_as_they_were_after_the_run(tmp_path):
    """Control (issue #690). The held value does not outlive the probe."""
    path = tmp_path / "m.net"
    path.write_text(NET.format(law=DECLINED, ic="A0", source=""))
    model = bngsim.Model.from_net(path)
    bngsim.Simulator(model, method="ode", sensitivity_params=["kd", "k0"]).run(
        sample_times=list(T), rtol=1e-10, atol=1e-12
    )
    assert [model.get_param(n) for n in ("k0", "kd", "ke")] == [0.5, 1.0, 3.0]


def test_an_sbml_parameter_set_by_an_initial_assignment():
    """``kd = 2*k0`` as an initial assignment, with ``kd*abs(A)`` as the
    kinetic law: the same 0."""
    text = "species A; A = 10; k0 = 0.5; kd = 2*k0\nJ1: A -> ; kd*abs(A)\n"
    model = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kd", "k0"])
    run = sim.run(sample_times=list(T), rtol=1e-10, atol=1e-12)
    assert not sim.has_analytic_sens_rhs
    got = np.asarray(run.sensitivities)[:, 0, :]
    np.testing.assert_allclose(got[:, 0], _dA_dk(1.0), rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(got[:, 1], 2 * _dA_dk(1.0), rtol=2e-5, atol=1e-7)


def test_compute_all_sensitivities_asked_for_the_derived_parameter(tmp_path):
    """The same column through ``compute_all_sensitivities(params=["kd"])``,
    which is how its #203 notice says to ask for a derived parameter's own
    column."""
    path = tmp_path / "m.net"
    path.write_text(NET.format(law=DECLINED, ic="A0", source=""))
    result = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").compute_all_sensitivities(
        (0.0, 2.0), 5, params=["kd"], n_workers=1, rtol=1e-10, atol=1e-12
    )
    got = np.asarray(result.sensitivities)[:, 0, 0]
    np.testing.assert_allclose(got, _dA_dk(1.0), rtol=2e-5, atol=1e-7)


def test_a_function_that_reads_the_derived_parameter(tmp_path):
    """``Yobs() = kd*Atot``: dYobs/dkd = A + kd·dA/dkd. The second term was
    missing, so the column was A, 3.68 for 0 at t = 1."""
    path = tmp_path / "m.net"
    path.write_text(
        NET.format(law=DECLINED, ic="A0", source="").replace(
            "end functions", "    2 Yobs() kd*Atot\nend functions"
        )
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=["kd"])
    run = sim.run(sample_times=list(T), rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities_expressions)[:, list(run.expression_names).index("Yobs"), 0]
    np.testing.assert_allclose(got, 10.0 * np.exp(-T) + _dA_dk(1.0), rtol=2e-5, atol=1e-6)


def test_an_initial_condition_that_is_the_derived_parameter(tmp_path):
    """``A(0) = kd``: A = kd·e^(−kd·t), so dA/dkd = (1 − kd·t)·e^(−kd·t). The
    seed at t = 0 was right and the rate's part was missing: e^(−kd·t), 0.368
    for 0 at t = 1."""
    _sim, got = _columns(tmp_path, DECLINED, ["kd", "k0"], ic="kd")
    want = (1.0 - T) * np.exp(-T)
    np.testing.assert_allclose(got[:, 0], want, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(got[:, 1], 2 * want, rtol=2e-5, atol=1e-7)


def test_the_steady_state_of_a_run_that_is_carried_to_one(tmp_path):
    """``0 -> A`` at ks beside the decay: A* = ks/kd, dA*/dkd = −ks/kd² = −4
    and dA*/dk0 = −8. ``run(steady_state=True)`` returned 0 and −8."""
    model = _model(tmp_path, source="\n    2 0 1 ks")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kd", "k0"])
    run = sim.run(
        t_span=(0, 200),
        n_points=3,
        steady_state=True,
        steady_state_tol=1e-12,
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.asarray(run.species)[-1, 0] == pytest.approx(4.0, rel=1e-9)
    np.testing.assert_allclose(np.asarray(run.sensitivities)[-1, 0], [-4.0, -8.0], rtol=1e-5)


def test_every_row_of_a_batch(tmp_path):
    """``run_batch`` over two initial amounts: each row's column for kd was 0."""
    sim = bngsim.Simulator(_model(tmp_path), method="ode", sensitivity_params=["kd"])
    rows = sim.run_batch(
        (0.0, 2.0), 5, params=[{"A0": 10.0}, {"A0": 20.0}], rtol=1e-10, atol=1e-12
    )
    for row, scale in zip(rows, (1.0, 2.0), strict=True):
        got = np.asarray(row.sensitivities)[:, 0, 0]
        np.testing.assert_allclose(got, scale * _dA_dk(1.0), rtol=2e-5, atol=1e-7)


EVENT = "species A; A = 10; k0 = 0.5; kd = 2*k0\nJ1: A -> ; kd*abs(A)\n"


def _event_column(event, times):
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(EVENT + event), method="ode", sensitivity_params=["kd"]
    )
    run = sim.run(sample_times=times, rtol=1e-10, atol=1e-12)
    assert not sim.has_analytic_sens_rhs
    return np.asarray(run.sensitivities)[:, 0, 0]


def test_an_event_assignment_that_reads_the_derived_parameter():
    """``A = A + kd`` at t = 1: A = (10·e^(−kd) + kd)·e^(−kd·(t − 1)) from
    there. The column came back 0 before the event and 0.607 for −3.044 at
    t = 1.5: the assignment's own ∂/∂kd, and nothing of the rate."""
    got = _event_column("E1: at (time > 1): A = A + kd\n", [0.0, 0.5, 1.5, 2.5])
    a1 = 10.0 * np.exp(-1.0)
    want = [0.0, _dA_dk(1.0)[1]] + [
        (1.0 - a1) * np.exp(-(t - 1.0)) - (t - 1.0) * (a1 + 1.0) * np.exp(-(t - 1.0))
        for t in (1.5, 2.5)
    ]
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-7)


def test_an_event_trigger_that_reads_the_derived_parameter():
    """``at (A < 2*kd): A = A + 1``. A reaches 2·kd at t* = ln(5/kd)/kd and
    is 2·kd + 1 after it, so with dt*/dkd = −(1 + ln(5/kd))/kd²,
    dA/dkd = e^(−kd·(t − t*))·(2 + (2·kd + 1)·(kd·dt*/dkd − (t − t*))). The
    run ends before the trigger fires a second time. The column came back 0
    up to the event and −0.677 for −4.74 after it."""
    t_star = float(np.log(5.0))
    dt_star = -(1.0 + np.log(5.0))
    t_end = 2.0
    got = _event_column("E1: at (A < 2*kd): A = A + 1\n", [0.0, 0.5, 1.5, t_end])
    after = np.exp(-(t_end - t_star)) * (2.0 + 3.0 * (dt_star - (t_end - t_star)))
    want = [0.0, -5.0 * np.exp(-0.5), -15.0 * np.exp(-1.5), after]
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-7)
