"""A derived parameter's column on the difference-quotient path (issue #707).

``kd = 2*k0`` and a rate law the analytic sensitivity right-hand side declines,
``kd*abs(Atot)``, so CVODES takes each column by its internal difference
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
end parameters
begin functions
    1 rf() {law}
end functions
begin species
    1 A() A0
end species
begin reactions
    1 1 0 rf
end reactions
begin groups
    1 Atot 1
end groups
"""
T = np.linspace(0.0, 2.0, 5)
DECLINED = "kd*abs(Atot)/Atot"  # the reaction's rate is rf·A = kd·|A|
ANALYTIC = "kd"


def _columns(tmp_path, law, params):
    path = tmp_path / "m.net"
    path.write_text(NET.format(law=law))
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
    path.write_text(NET.format(law=DECLINED))
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
    path.write_text(NET.format(law=DECLINED))
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
        NET.format(law=DECLINED).replace("end functions", "    2 Yobs() kd*Atot\nend functions")
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=["kd"])
    run = sim.run(sample_times=list(T), rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities_expressions)[:, list(run.expression_names).index("Yobs"), 0]
    np.testing.assert_allclose(got, 10.0 * np.exp(-T) + _dA_dk(1.0), rtol=2e-5, atol=1e-6)
