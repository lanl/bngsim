"""The example models match their exact solutions, in every format (#1706).

``bngsim.examples`` ships small models with exact solutions so that a user, or
an AI agent, can check BNGsim on their own machine. That only works if the
solutions are right and every file is the same model, so:

* every format of every model matches the exact solution, as an ODE run;
* the exact solution's default parameters are the values in the files;
* the stochastic methods match the exact mean where it is also the
  stochastic mean (first-order models), and NFsim and RuleMonkey agree;
* Schlogl's model has the three steady states, with their stability.
"""

from __future__ import annotations

import shutil

import bngsim
import numpy as np
import pytest
from bngsim.convert._bng2 import find_bng2

E = bngsim.examples

_HAS_BNG2 = find_bng2() is not None and shutil.which("perl") is not None


def _runnable(name: str, fmt: str):
    marks = []
    if fmt == "antimony":
        marks.append(
            pytest.mark.skipif(not bngsim.HAS_ANTIMONY, reason="antimony is not installed")
        )
    if fmt == "bngl":
        marks.append(pytest.mark.skipif(not _HAS_BNG2, reason="BNG2.pl / perl not available"))
    return pytest.param(name, fmt, id=f"{name}-{fmt}", marks=marks)


_TIME_COURSES = [
    _runnable(name, fmt)
    for name in E.names()
    if name != "schlogl"
    for fmt in E.formats(name)
    if fmt != "bng-xml"
]


@pytest.mark.parametrize(("name", "fmt"), _TIME_COURSES)
def test_every_format_matches_the_exact_solution(name, fmt):
    model = E.load(name, fmt)
    result = bngsim.Simulator(model).run(t_span=(0, 10), n_points=101, rtol=1e-10, atol=1e-10)
    exact = E.exact(name, result.time)
    for obs, values in exact.items():
        np.testing.assert_allclose(result.observables[obs], values, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("name", "fmt"),
    [_runnable(name, fmt) for name in E.names() for fmt in E.formats(name) if fmt != "bng-xml"],
)
def test_the_exact_solution_has_the_files_parameters(name, fmt):
    model = E.load(name, fmt)
    from bngsim.examples._exact import DEFAULTS

    for param, value in DEFAULTS[name].items():
        if param in model.param_names:
            assert model.get_param(param) == pytest.approx(value), param
    # Initial amounts: A0 and B0 are the first points of A and B.
    first = bngsim.Simulator(model).run(t_span=(0, 1), n_points=2).observables
    for param, obs in (("A0", "A"), ("B0", "B")):
        if param in DEFAULTS[name]:
            assert first[obs][0] == pytest.approx(DEFAULTS[name][param]), param


def test_every_model_has_a_python_and_an_sbml_form():
    for name in E.names():
        assert {"python", "sbml"} <= set(E.formats(name)), name
        assert E.describe(name)


def test_schlogl_has_its_three_steady_states():
    exact = E.steady_states("schlogl")
    assert [round(s["X"], 6) for s in exact] == [100, 250, 400]
    assert [s["stable"] for s in exact] == [True, False, True]
    sim = bngsim.Simulator(E.load("schlogl"))
    assert sim.steady_state()["X"] == pytest.approx(100, rel=1e-6)  # from X = 50
    roots = sim.steady_states()
    found = sorted(zip(roots.states[:, 0], roots.stability, strict=True))
    np.testing.assert_allclose([x for x, _ in found], [100, 250, 400], rtol=1e-6)
    assert [s for _, s in found] == ["stable", "unstable", "stable"]


@pytest.mark.parametrize("name", ["binding", "decay", "chain", "birth_death"])
def test_the_exact_steady_state_is_where_a_long_run_ends(name):
    # The Python form names each species as its observable.
    ss = bngsim.Simulator(E.load(name, "python")).steady_state()
    for obs, value in E.steady_states(name)[0].items():
        if obs != "stable":
            assert ss[obs] == pytest.approx(value, rel=1e-5, abs=1e-6), obs


def test_the_ssa_mean_of_decay_is_the_exact_solution():
    reps = bngsim.Simulator(E.load("decay"), method="ssa").run_replicates(
        400, t_span=(0, 4), n_points=5, seed=1, squeeze=True
    )
    a = reps.observables[:, :, 0]
    se = a.std(axis=0, ddof=1) / np.sqrt(len(a))
    exact = E.exact("decay", reps.time)["A"]
    assert np.all(np.abs(a.mean(axis=0) - exact) <= 4 * se + 1e-12)


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_birth_death_keeps_its_exact_mean(method):
    kw = {"poplevel": 1000.0} if method == "psa" else {}
    reps = bngsim.Simulator(E.load("birth_death"), method=method, **kw).run_replicates(
        100, t_span=(0, 10), n_points=2, seed=2, squeeze=True
    )
    a = reps.observables[:, -1, 0]
    exact = E.exact("birth_death", [10.0])["A"][0]
    assert abs(a.mean() - exact) <= 4 * a.std(ddof=1) / np.sqrt(len(a))
    if method == "ssa":  # started at 0, the count is Poisson
        assert a.var(ddof=1) == pytest.approx(exact, rel=0.4)


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not compiled in")
@pytest.mark.skipif(not bngsim.HAS_RULEMONKEY, reason="RuleMonkey not compiled in")
def test_nfsim_and_rulemonkey_agree_on_binding():
    xml = str(E.path("binding", "bng-xml"))
    finals = {}
    for method in ("nf", "rm"):
        sim = bngsim.Simulator(E.load("binding", "net"), method=method, xml_path=xml)
        finals[method] = np.array(
            [sim.run(t_span=(0, 10), n_points=2, seed=s).observables["C"][-1] for s in range(60)]
        )
    nf, rm = finals["nf"], finals["rm"]
    se = np.sqrt(nf.var(ddof=1) / len(nf) + rm.var(ddof=1) / len(rm))
    assert abs(nf.mean() - rm.mean()) <= 4 * se
    # The stochastic mean of a bimolecular reaction is near, not at, the ODE.
    assert nf.mean() == pytest.approx(E.exact("binding", [10.0])["C"][0], rel=0.05)


def test_exact_takes_parameters_by_name():
    t = np.array([0.0, 1.0])
    assert E.exact("decay", t, k=1.0)["A"][1] == pytest.approx(100 * np.exp(-1))
    with pytest.raises(ValueError, match="no parameter kk"):
        E.exact("decay", t, kk=1.0)


def test_the_errors_say_what_to_do():
    with pytest.raises(ValueError, match="the examples are"):
        E.load("nope")
    with pytest.raises(ValueError, match="comes in"):
        E.path("bolus", "net")
    with pytest.raises(ValueError, match="xml_path"):
        E.load("binding", "bng-xml")
    with pytest.raises(ValueError, match="steady_states"):
        E.exact("schlogl", [0.0])
