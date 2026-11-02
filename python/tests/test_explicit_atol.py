"""Issue #1420: a scalar atol the caller chose is a ceiling too, and ExactAtol opts out.

A scalar ``atol`` is one number asked to be a tolerance for every species. A
SED-ML file's ``atol=1e-9`` at ``rtol=1e-6`` held BIOMD0000000457's species_42,
near 1.6e-7, to 13% at t = 5. So an explicit scalar is read as the default is:
each species takes ``min(atol, rtol * max(|x_i|, floor))``, never looser than the
number given. ``bngsim.ExactAtol(value)`` holds every species to the number
itself, as a scalar was read before; a vector is used exactly as given.
"""

from __future__ import annotations

import json

import bngsim
import numpy as np
import pytest
from bngsim._eval_spec import EvaluationSpec
from bngsim.convert import read_sedml, write_sedml

BINDING = """begin parameters
 1 kf 1e6
 2 kr 1e-3
end parameters
begin species
 1 A() {a}
 2 B() {b}
 3 C() 0
end species
begin reactions
 1 1,2 3 kf
 2 3 1,2 kr
end reactions
"""

SPAN = {"t_span": (0, 5000), "n_points": 51}


def _sim(tmp_path, a=1e-9, b=2e-9):
    path = tmp_path / "binding.net"
    path.write_text(BINDING.format(a=a, b=b))
    return bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")


def _off_by(result, reference):
    x, ref = np.asarray(result.species), np.asarray(reference.species)
    return float((np.abs(x - ref).max(axis=0) / np.abs(ref).max(axis=0)).max())


def test_an_explicit_scalar_is_a_ceiling(tmp_path):
    """At SED-ML's usual rtol 1e-6 and atol 1e-9, A at 1e-9 and B at 2e-9 take
    1e-6 of themselves, and C, at 0, of the floor."""
    own = _sim(tmp_path)._run_atol(1e-9, 1e-6)
    assert own.vector == pytest.approx([1e-15, 2e-15, 1e-15], rel=1e-15)


def test_the_scalar_still_reaches_what_reads_a_scalar(tmp_path):
    """steady_state_tol falls back to the run's scalar atol, as BNG2.pl's
    early stop does: read as a ceiling, atol=1e-4 is still the cutoff, not
    the Simulator's own 1e-8."""
    sim = _sim(tmp_path)
    resolved = sim._resolve_atol(sim._run_atol(1e-4, 1e-8), 1e-8, where="test")
    assert resolved.scalar == 1e-4 and resolved.vector is not None


def test_an_explicit_scalar_resolves_species_below_it(tmp_path):
    """Held to 1e-9 exactly, species that start at 1e-9 are 29% off; read as a
    ceiling, the run gives the reference to 1e-4."""
    reference = _sim(tmp_path).run(rtol=1e-12, atol=1e-22, **SPAN)
    assert _off_by(_sim(tmp_path).run(rtol=1e-6, atol=1e-9, **SPAN), reference) < 1e-4
    exact = _sim(tmp_path).run(rtol=1e-6, atol=bngsim.ExactAtol(1e-9), **SPAN)
    assert _off_by(exact, reference) > 0.1


def _antimony(src, **kw):
    pytest.importorskip("antimony")
    return bngsim.Simulator(bngsim.Model.from_antimony_string(src), method="ode", **kw)


# BIOMD0000000857 (Larbat 2016), each species' production and consumption summed
# into one law. Ephe is made at about 9.4 and consumed at 9.8·Ephe/(1e-6 + Ephe),
# which holds it near 1e-5 until sucrose makes more than 9.8 can take.
LARBAT = """
sucr = 0.7; EtrioseP = 8.06; Ephe = 1e-5; N = 5; starch = 0; Next = 500;
Enitrate = 0.075; trioseP = 1.1; Estarch = 1e-4;
F: trioseP -> sucr; EtrioseP*trioseP;
P: sucr -> ; sucr*Ephe*0.2/(0.2 + N) + 8*sucr*N/(1 + N) + 0.4*sucr;
S: starch -> sucr; 0.1*starch*Estarch;
T: -> EtrioseP; 1 - sucr*EtrioseP/(1e-5 + EtrioseP);
E: -> Ephe; 10*sucr*1000/(1000 + N) - 9.8*Ephe/(1e-6 + Ephe);
U: Next -> N; 0.2*Next*Enitrate;
V: N -> ; 1.5*N;
W: -> Enitrate; 0.5 - 0.1*N*Enitrate/(1e-6 + Enitrate);
X: trioseP -> starch; 10*trioseP;
Y: -> trioseP; 10.1*N/(1e-11 + N);
Z: -> Estarch; 9.8 - 10*sucr*Estarch/(1e-4 + Estarch);
"""


def test_a_species_below_the_floor_keeps_a_scale_of_its_own():
    """Ephe, at 1e-5 beside Next at 500, is under the floor of 1e-6 of the
    largest, 5e-4, but 1e-6 of it is far above what double precision resolves
    beside 500. It keeps its own 1e-11 and Estarch its 1e-10; starch, at 0,
    takes 1e-6 of the floor."""
    sim = _antimony(LARBAT)
    own = dict(zip(sim.model.species_names, sim._run_atol(1e-9, 1e-6).vector, strict=True))
    assert own["Ephe"] == pytest.approx(1e-11, rel=1e-15)
    assert own["Estarch"] == pytest.approx(1e-10, rel=1e-15)
    assert own["starch"] == pytest.approx(5e-10, rel=1e-15)


def test_a_species_held_to_its_own_scale_integrates_on_every_grid():
    """At SED-ML's rtol 1e-6 and atol 1e-9, held as though at the floor, Ephe
    took 5e-10 for its 1e-11, and CVODE stopped with CV_ERR_FAILURE on 8 of
    these 40 grids: it left Ephe 1.4e-6 off the level its consumption holds
    it at, then failed the next step's error test seven times (#1439). The
    analytical Jacobian alone, as the finite-difference retry hides some."""
    for n in range(101, 141):
        sim = _antimony(LARBAT, jacobian="analytical")
        sim.run(t_span=(0, 200), n_points=n, rtol=1e-6, atol=1e-9)


def test_a_scalar_no_species_is_below_stays_the_scalar(tmp_path):
    """Molecule counts: every species takes the number, on the scalar's path."""
    sim = _sim(tmp_path, a=1000, b=2000)
    assert sim._run_atol(1e-8, 1e-8) == 1e-8
    chosen = sim.run(atol=1e-8, **SPAN)
    sim.model.reset()
    exact = sim.run(atol=bngsim.ExactAtol(1e-8), **SPAN)
    assert np.array_equal(np.asarray(chosen.species), np.asarray(exact.species))


@pytest.mark.parametrize("how", ["run", "set_tolerances", "vector"])
def test_exact_atol_holds_every_species_to_the_number(tmp_path, how):
    """The old reading, for a caller who wants it: #1361's binding model ends
    at a negative A again, as it did under the scalar."""
    sim = _sim(tmp_path)
    if how == "run":
        result = sim.run(atol=bngsim.ExactAtol(1e-8), **SPAN)
    elif how == "set_tolerances":
        sim.set_tolerances(rtol=1e-8, atol=bngsim.ExactAtol(1e-8))
        result = sim.run(**SPAN)
    else:
        result = sim.run(atol=[1e-8] * 3, **SPAN)
    assert np.asarray(result.species)[-1, 0] < 0.0


# The binding model with its reverse rate behind clamp(0.5, A, big): bngsim has
# no analytic df/dp for a clamp whose bound is a parameter (see
# test_state_crossing_on_fallback_sensitivity.py), so CVODES' difference
# quotient carries every sensitivity column.
DECLINED = """begin parameters
 1 kf 1e6
 2 kr 1e-3
 3 big 1e300
end parameters
begin functions
 1 fR() kr*clamp(0.5,Aobs,big)
end functions
begin species
 1 A() 1e-9
 2 B() 2e-9
 3 C() 0
end species
begin reactions
 1 1,2 3 kf
 2 3 1,2 fR
end reactions
begin groups
 1 Aobs 1
end groups
"""


def test_difference_quotient_sensitivities_keep_the_scalar(tmp_path):
    """The quotient sizes its increment from the state's error weights, so a
    species held tighter is perturbed by less and carries more roundoff. A run
    whose sensitivities fall back there keeps the scalar, given or default; a
    plain run of the same model reads it as a ceiling."""
    path = tmp_path / "declined.net"
    path.write_text(DECLINED)
    model = bngsim.Model.from_net(str(path))
    dq = bngsim.Simulator(model, method="ode", sensitivity_params=["kr"])
    assert not dq.has_analytic_sens_rhs
    assert dq._run_atol(1e-8, 1e-8) == 1e-8
    assert dq._run_atol(None, 1e-8) is None
    plain = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    assert plain._run_atol(1e-8, 1e-8).vector is not None


def test_set_tolerances_with_a_float_clears_exactness(tmp_path):
    sim = _sim(tmp_path)
    sim.set_tolerances(rtol=1e-8, atol=bngsim.ExactAtol(1e-8))
    assert sim._run_atol(None, 1e-8) is None
    sim.set_tolerances(rtol=1e-8, atol=1e-8)
    assert sim._run_atol(None, 1e-8).vector == pytest.approx([1e-17, 2e-17, 1e-17], rel=1e-15)


@pytest.mark.parametrize("value", [-1e-9, float("nan"), float("inf")])
def test_exact_atol_refuses_an_unusable_value(value):
    with pytest.raises(ValueError, match=r"finite and >= 0"):
        bngsim.ExactAtol(value)


def test_a_scan_freezes_a_scalar_once_for_every_point(tmp_path):
    """A scalar resolved for the scan reaches each point as an ExactAtol: as a
    float, each point's run() would read it as a ceiling again, from that
    point's own start."""
    sim = _sim(tmp_path, a=1000, b=2000)
    frozen = sim._freeze_atol(sim._run_atol(1e-8, 1e-8), 1e-8, where="test")
    assert frozen == bngsim.ExactAtol(1e-8)


def test_every_ode_call_takes_exact_atol(tmp_path):
    reference = _sim(tmp_path).run(**SPAN)
    exact = bngsim.ExactAtol(1e-8)
    batch = _sim(tmp_path).run_batch(params=[{"kr": 1e-3}], atol=exact, **SPAN)[0]
    scan = _sim(tmp_path).parameter_scan("kr", [1e-3], atol=exact, **SPAN)[0]
    for result in (batch, scan):
        assert np.asarray(result.species)[-1, 0] < 0.0  # the scalar's answer
        assert _off_by(result, reference) > 0.1


def test_steady_state_takes_exact_atol(tmp_path):
    """steady_state holds a scalar as given either way; ExactAtol says so."""
    exact = _sim(tmp_path).steady_state(atol=bngsim.ExactAtol(1e-20))
    given = _sim(tmp_path).steady_state(atol=1e-20)
    assert np.array_equal(np.asarray(exact.concentrations), np.asarray(given.concentrations))


def test_an_evaluation_spec_round_trips_exact_atol():
    """It goes out as {"exact": value}, so a spec held exactly is not the same
    content key as one read as a ceiling."""
    kw = {"model_source": "m.xml", "model_format": "sbml", "t_span": (0, 1), "n_points": 3}
    exact = EvaluationSpec(atol=bngsim.ExactAtol(1e-9), **kw)
    assert exact.to_dict()["atol"] == {"exact": 1e-9}
    assert EvaluationSpec.from_dict(json.loads(exact.to_json())) == exact
    assert exact.to_json() != EvaluationSpec(atol=1e-9, **kw).to_json()
    with pytest.raises(ValueError, match=r"Unknown atol mapping"):
        EvaluationSpec.from_dict({**exact.to_dict(), "atol": {"value": 1e-9}})


def test_read_sedml_takes_the_files_atol_exactly_on_request():
    kw = {"model_source": "m.xml", "model_format": "sbml", "t_span": (0, 1), "n_points": 3}
    text = write_sedml(EvaluationSpec(rtol=1e-6, atol=bngsim.ExactAtol(1e-9), **kw))
    assert read_sedml(text).atol == 1e-9  # SED-ML says one number, not "exactly"
    assert read_sedml(text, exact_atol=True).atol == bngsim.ExactAtol(1e-9)
