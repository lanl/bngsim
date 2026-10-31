"""Issue #1420: where no caller chose an atol, each species takes its own.

The scalar ``1e-8`` is BNG2.pl's, for species held as molecule counts. Below it,
a species of a model in molar units has no error control but the tolerance. At
bngsim's defaults, 62 of 1,251 BioModels were off their reference by over 1% of
a species' peak, 21 of them with no warning. The default is now, per species,
``min(1e-8, rtol * max(|x_i(0)|, floor))``, with ``floor`` the smallest start of a
species the solver integrates, but not under 1e-6 of the largest. That cut the 62
to 14, broke none, and cost 17% of the corpus's run time. A model of counts,
whose species start at 0 or at least 1, takes the scalar, on the scalar's path.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

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


def _antimony(src):
    pytest.importorskip("antimony")
    return bngsim.Simulator(bngsim.Model.from_antimony_string(src), method="ode")


def _off_by(x, reference):
    x, ref = np.asarray(x), np.asarray(reference)
    return float((np.abs(x - ref).max(axis=0) / np.abs(ref).max(axis=0)).max())


def test_each_species_is_held_to_rtol_of_its_start(tmp_path):
    """A at 1e-9 and B at 2e-9 take 1e-8 of themselves; C, at 0, the floor:
    the smallest start, 1e-9, which is above 1e-6 of the largest."""
    assert _sim(tmp_path)._run_atol(None, 1e-8) == pytest.approx([1e-17, 2e-17, 1e-17], rel=1e-15)


def test_a_model_of_counts_takes_the_scalar_bit_for_bit(tmp_path):
    sim = _sim(tmp_path, a=1000, b=2000)
    assert sim._run_atol(None, 1e-8) is None
    default = sim.run(**SPAN)
    sim.model.reset()
    chosen = sim.run(atol=1e-8, **SPAN)
    assert np.array_equal(np.asarray(default.species), np.asarray(chosen.species))
    assert default.solver_stats["n_steps"] == chosen.solver_stats["n_steps"]


def test_it_is_never_looser_than_the_scalar():
    """X at 1e3 would take 1e-5 of rtol alone; the scalar holds it to 1e-8."""
    sim = _antimony("X = 1e3; Y = 1e-3; k = 0.1; R: X -> Y; k*X; D: Y -> ; k*Y;")
    assert sim._run_atol(None, 1e-8) == pytest.approx([1e-8, 1e-11], rel=1e-15)


def test_the_floor_is_a_millionth_of_the_largest():
    """MODEL2003200002 declares species at 1e-31 beside one at 10, and at their
    own scale its run fails. Held as though at 1e-5, they take 1e-13; so does a
    species at 0."""
    sim = _antimony("X = 10; Y = 1e-31; Z = 0; k = 0.1; R: X -> Y; k*X; S: Y -> Z; k*Y;")
    assert sim._run_atol(None, 1e-8) == pytest.approx([1e-8, 1e-13, 1e-13], rel=1e-15)


def test_with_every_integrated_species_at_0_a_fixed_one_gives_the_floor():
    """BIOMD0000000495's shape: every species the solver integrates starts at 0
    beside $-fixed ones. It took the scalar and was 9.4% off; the fixed
    species' scale holds X to 5e-13, 1e-5 of the 5e-8 it settles at, and the
    run is off by about that instead of the scalar's 13%."""
    src = "$L = 5e-5; X = 0; k = 1e-4; R: L -> X; k*L; D: X -> ; 0.1*X;"
    assert _antimony(src)._run_atol(None, 1e-8) == pytest.approx([5e-13, 5e-13], rel=1e-15)
    span = {"t_span": (0, 100), "n_points": 21}
    reference = _antimony(src).run(rtol=1e-12, atol=1e-22, **span)
    assert _off_by(_antimony(src).run(**span).species, reference.species) < 1e-4
    assert _off_by(_antimony(src).run(atol=1e-8, **span).species, reference.species) > 1e-2


def test_with_every_species_at_0_it_is_the_scalar():
    assert _antimony("X = 0; P: -> X; 1e-12; D: X -> ; 1e-3*X;")._run_atol(None, 1e-8) is None


def test_a_run_until_step_keeps_the_default_its_trajectory_started_with(tmp_path):
    """A model of counts leaves whole numbers after its first step, and the
    default derived there is a vector. Derived again at each step, a coupling
    loop's answer depended on how it was cut into steps, and the warm path lost
    its count of dense factorizations; each step keeps the start's scalar."""
    default, chosen = _sim(tmp_path, a=1000, b=2000), _sim(tmp_path, a=1000, b=2000)
    for t in (1.0, 2.0, 3.0):
        kept = default.run_until(t, n_points=3)
        given = chosen.run_until(t, n_points=3, atol=1e-8)
        assert np.array_equal(np.asarray(kept.species), np.asarray(given.species))
        assert default._run_atol(None, 1e-8) is not None  # where the step starts


def test_an_atol_the_caller_chose_is_left_alone(tmp_path):
    sim = _sim(tmp_path)
    assert sim._run_atol(1e-8, 1e-8) == 1e-8
    assert sim._run_atol("auto", 1e-8) == "auto"
    sim.set_tolerances(rtol=1e-8, atol=1e-8)
    assert sim._run_atol(None, 1e-8) is None


def test_an_rtol_of_0_keeps_the_scalar(tmp_path):
    """Pure absolute control is CVODE's to run; rtol 0 scales nothing."""
    assert _sim(tmp_path)._run_atol(None, 0.0) is None


@pytest.mark.parametrize(
    "call", ["run", "run_batch", "parameter_scan", "compute_all_sensitivities"]
)
def test_every_ode_call_integrates_at_it(tmp_path, call):
    """#1361's binding model ended at a negative A under the scalar, 2.2 of its
    peak off; each call that integrates it at the default now gives the
    reference."""
    reference = _sim(tmp_path).run(rtol=1e-12, atol=1e-22, **SPAN)
    sim = _sim(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if call == "run":
            result = sim.run(**SPAN)
        elif call == "run_batch":
            result = sim.run_batch(params=[{"kr": 1e-3}], **SPAN)[0]
        elif call == "parameter_scan":
            result = sim.parameter_scan("kr", [1e-3], **SPAN)[0]
        else:
            result = sim.compute_all_sensitivities(**SPAN)
    assert _off_by(result.species, reference.species) < 1e-5
    assert not [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]
