"""Issue #1361: an ODE run says when the default atol is above species that matter.

The default ``atol`` was the scalar ``1e-8``, BNG2.pl's, for species held as
molecule counts. A species that stays below it all run has no error control but
the tolerance: the issue's binding model, species near 1e-9 M, returned A =
-8.7e-10 for 4.1e-10 at t = 5000, and BIOMD0000000002 (ligand at 1e-5 M,
complexes at 3e-10 to 7e-9 M) errors of 1e-2, both with no word. ``run()`` raises
:class:`bngsim.AtolScaleWarning`, naming the species, where the caller gave no
atol and a species stays below its tolerance while reaching at least 1e-6 of the
run's largest species.

Since issue #1420 the default takes each species' scale from the start
(``test_default_atol.py``), and both of those models are right and quiet. What
is left to warn of is a species the start gives no scale: here, a model whose
species all start at 0 and settle near 1e-9.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

NET = """begin parameters
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


# A made at 1e-12 per second from nothing and lost at 1e-3, half of it into B:
# A settles at 1e-9 and B at 5e-10, from a start where every species is 0.
FROM_ZERO = """begin parameters
 1 kp 1e-12
 2 kd 1e-3
end parameters
begin species
 1 A() 0
 2 B() 0
end species
begin reactions
 1 0 1 kp
 2 1 0 kd
 3 1 1,2 kd
 4 2 0 kd
end reactions
"""


def _sim(tmp_path, a=1e-9, b=2e-9):
    path = tmp_path / "binding.net"
    path.write_text(NET.format(a=a, b=b))
    return bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")


def _from_zero(tmp_path):
    path = tmp_path / "from_zero.net"
    path.write_text(FROM_ZERO)
    return bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")


def _off_by(result, reference):
    x, ref = np.asarray(result.species), np.asarray(reference.species)
    return float((np.abs(x - ref).max(axis=0) / np.abs(ref).max(axis=0)).max())


def _warned(sim, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(t_span=(0, 5000), n_points=51, **kw)
    return result, [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]


def test_a_model_whose_species_all_start_at_0_is_warned(tmp_path):
    """No species starts with a scale, so the default is the scalar, and A and
    B, under it all run, are a quarter off; the warning names both."""
    reference = _from_zero(tmp_path).run(t_span=(0, 5000), n_points=51, rtol=1e-12, atol=1e-22)
    result, caught = _warned(_from_zero(tmp_path))
    assert _off_by(result, reference) > 0.1
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "atol=1e-08" in message and "'A()', 'B()'" in message
    assert "atol='tracking'" in message


def test_the_remedy_it_names_is_right_and_quiet(tmp_path):
    """Control: atol='tracking' gives the reference to 1e-5 of the peak, with no warning."""
    reference = _from_zero(tmp_path).run(t_span=(0, 5000), n_points=51, rtol=1e-12, atol=1e-22)
    result, caught = _warned(_from_zero(tmp_path), atol="tracking")
    assert caught == []
    assert _off_by(result, reference) < 1e-5


def test_the_issues_molar_model_is_right_and_quiet_at_the_default(tmp_path):
    """#1361's binding model: under the scalar it ended at a negative A, 2.2
    of its peak off. Its species' own tolerances (#1420) give the reference."""
    reference = _sim(tmp_path).run(t_span=(0, 5000), n_points=51, rtol=1e-12, atol=1e-22)
    result, caught = _warned(_sim(tmp_path))
    assert caught == []
    assert _off_by(result, reference) < 1e-5
    scalar, _ = _warned(_sim(tmp_path), atol=1e-8)
    assert np.asarray(scalar.species)[-1, 0] < 0.0


@pytest.mark.parametrize("how", ["run", "set_tolerances"])
def test_an_atol_the_caller_chose_is_not_second_guessed(tmp_path, how):
    sim = _from_zero(tmp_path)
    if how == "run":
        _, caught = _warned(sim, atol=1e-8)
    else:
        sim.set_tolerances(atol=1e-8)
        _, caught = _warned(sim)
    assert caught == []


def test_a_count_model_is_not_warned(tmp_path):
    """Control: species of a thousand molecules, with C starting at 0."""
    _, caught = _warned(_sim(tmp_path, a=1000, b=2000))
    assert caught == []


def test_a_species_at_the_noise_of_a_count_model_is_not_warned(tmp_path):
    """Control: a species that stays below atol, but at under 1e-6 of the
    largest species, is rounding in a model of counts, not a scale."""
    path = tmp_path / "noise.net"
    path.write_text(
        "begin parameters\n 1 k 1\n 2 kn 1e-15\nend parameters\n"
        "begin species\n 1 A() 1000\n 2 N() 0\nend species\n"
        "begin reactions\n 1 1 0 k\n 2 0 2 kn\nend reactions\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    result, caught = _warned(sim)
    assert 0.0 < np.abs(np.asarray(result.species)[:, 1]).max() < 1e-8
    assert caught == []


def test_a_mixed_scale_biomodel_is_right_and_quiet_at_the_default():
    """BIOMD0000000002: L at 1e-5 is above the scalar; complexes at 3e-10 to
    7e-9 are below it and are part of the model, and under it were 2.2e-2 of
    their peak off, with 'BL' warned. Each species' own tolerance gives 2e-7."""
    pytest.importorskip("libsbml")

    def run(**kw):
        sim = bngsim.Simulator(
            bngsim.Model.from_sbml(str(DATA / "BIOMD0000000002.xml")), method="ode"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sim.run(t_span=(0, 500), n_points=51, **kw)
        return result, [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]

    reference, _ = run(rtol=1e-12, atol=1e-22)
    result, hits = run()
    assert hits == []
    ref, x = np.asarray(reference.species), np.asarray(result.species)
    peak = np.abs(ref).max(axis=0)
    keep = peak >= 1e-6 * peak.max()
    assert (np.abs(x - ref).max(axis=0)[keep] / peak[keep]).max() < 1e-5


def test_ssa_is_not_warned():
    """SSA has no atol: a concentration of 1e-9 in a compartment of 1e10 is
    ten molecules, run exactly, and below 1e-8 all run."""
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(
        "compartment c = 1e10;\nspecies X in c; X = 1e-9;\nk = 0.1;\nR: X -> ; k*X;\n"
    )
    sim = bngsim.Simulator(model, method="ssa")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(t_span=(0, 1), n_points=3, seed=1)
    assert 0.0 < np.abs(np.asarray(result.species)).max() < 1e-8
    assert not [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]


@pytest.mark.parametrize("call", ["parameter_scan", "run_batch", "compute_all_sensitivities"])
def test_a_call_of_many_runs_is_warned_once(tmp_path, call):
    """A scan froze the default into the float 1e-8 before run() saw it, and a
    batch or a sensitivity tensor never asked: each returned the issue's
    negative A with no word. One warning per call, at the caller's line."""
    sim = _from_zero(tmp_path)
    span = {"t_span": (0, 5000), "n_points": 51}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if call == "parameter_scan":
            sim.parameter_scan("kd", [1e-3, 2e-3], **span)
        elif call == "run_batch":
            sim.run_batch(params=[{"kd": 1e-3}, {"kd": 2e-3}], **span)
        else:
            sim.compute_all_sensitivities(**span)
    hits = [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]
    assert len(hits) == 1
    assert "'A()'" in str(hits[0].message)
    assert hits[0].filename == __file__


def test_a_species_the_solver_does_not_integrate_is_not_named():
    """A $-fixed species and an assignment-rule target below atol are exact
    (or as good as their rule's inputs): no tolerance makes them wrong. At rtol
    1e-4 the default gives each the scalar, 1e-8, which each stays below: it
    takes no part in the floor, which is the integrated X's 1e-3."""
    pytest.importorskip("antimony")
    for src, name in (
        ("$L = 5e-9; X = 1e-3; k = 1; R: L -> X; k*L; D: X -> ; 0.1*X;", "L"),
        ("species X = 1e-3; species Y; Y := 5e-6*X; D: X -> ; 0.1*X;", "Y"),
    ):
        sim = bngsim.Simulator(bngsim.Model.from_antimony_string(src), method="ode")
        assert sim._run_atol(None, 1e-4) is None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sim.run(t_span=(0, 100), n_points=21, rtol=1e-4)
        peaks = np.abs(np.asarray(result.species)).max(axis=0)
        # In the band a species the solver integrates is named for.
        peak = peaks[result.species_names.index(name)]
        assert 1e-6 * peaks.max() <= peak < 1e-8
        assert not [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]


def test_a_column_with_no_finite_value_adds_no_warning():
    """np.nanmax warned 'All-NaN slice' on a reported column that is never
    finite (an assignment rule at 1/0), and raised under -W error."""
    pytest.importorskip("antimony")
    src = "species X = 1e-3; species Y; z = 0; Y := 1/z; D: X -> ; 0.1*X;"
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(src), method="ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(t_span=(0, 10), n_points=3)
    assert not np.isfinite(np.asarray(result.species)[:, 1]).any()
    assert caught == []


def test_the_steps_of_a_loop_are_warned_once(tmp_path):
    """The text carries nothing that moves from run to run, so the default
    filter shows it once at the caller's line; peaks in the text made 20
    run_until steps print it 11 times, from a line of bngsim's own."""
    sim = _from_zero(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter("default")
        for step in range(1, 21):
            sim.run_until(250.0 * step, n_points=3)
    hits = [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]
    assert len(hits) == 1
    assert hits[0].filename == __file__


def test_a_kernel_observables_probe_is_not_warned():
    """ReactionKernel.observables() reads row 0 of a throwaway ODE run, on an
    SSA kernel too; it warned about rows nothing reads."""
    pytest.importorskip("antimony")
    src = "compartment c = 1e10; species X in c; X = 1e-9; k = 0.1; R: X -> ; k*X;"
    for method in ("ode", "ssa"):
        kernel = bngsim.ReactionKernel(bngsim.Model.from_antimony_string(src), method=method)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            observables = kernel.observables()
        assert np.allclose(observables, [1e-9])
        assert not [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]
