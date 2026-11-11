"""``Result.to_cdat`` writes the species the Result reports (issue #1505).

``Simulator._stamp`` corrects species columns in Python after the solve: a
linear assignment-rule species gets the rule's live value, and a species in a
compartment an event resizes is rescaled to amount over the live volume
(#1368). The C++ core's copy is never updated, and ``to_cdat`` wrote it, so the
``.cdat`` disagreed with ``Result.species``, with ``to_csv(kind="species")``
(documented as matching ``.cdat``), and with the ``.cdat`` the same Result
wrote after ``save``/``Result.load``. ``S := 2*A`` read 20 at every sample, and
an SSA resize c: 2 -> 5 left [X] at 10 for 4.

``to_cdat`` now hands ``Result.species`` to the core's own writer, so the file
has the core's layout. A Result whose species were not corrected writes the
bytes it wrote before, and a loaded Result writes the bytes the run wrote.

Oracles: the closed forms of each model, ``Result.species`` and
``to_csv(kind="species")``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import warnings

import bngsim
import numpy as np
import pytest

from .test_sbml_event_param_not_reported import SBML_EVENT_PARAM  # GH #71: P is not reported

RULE = """
  compartment c = 1; species A in c = 10; species S in c
  k = 0.5
  R1: A => ; k*A
  S := 2*A
"""
# Substance-only Y, Y' = -0.2 Y in amount: the amount is conserved through the
# resize, so [Y] = 10 e^(-0.2 t) / c(t) (libRoadRunner 2.9.2 agrees).
RESIZE_ODE = """
  compartment c = 1; substanceOnly species Y in c = 10
  R1: Y => ; 0.2*Y
  E1: at time >= 2: c = 4
"""
# X = 10 in c = 2 is 20 molecules no reaction touches, so [X] = 20 / c(t).
RESIZE_SSA = """
  compartment c = 2; species X in c = 10; species W in c = 3
  R1: W => ; 0.7*W
  E1: at time > 0.75: c = 5
"""


def _sim(ant, method="ode"):
    return bngsim.Simulator(bngsim.Model.from_antimony_string(ant), method=method)


def _run(ant, method="ode", **kw):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _sim(ant, method).run(**kw)


def _column(path, name, sep=None):
    with open(path) as f:
        header = f.readline().lstrip("#").replace(",", " ").split()
    return np.loadtxt(path, ndmin=2, delimiter=sep, skiprows=1)[:, header.index(name)]


CASES = [
    pytest.param(
        RULE,
        "ode",
        "S",
        lambda t: 20 * np.exp(-0.5 * t),
        {"t_span": (0, 4), "n_points": 5, "rtol": 1e-10, "atol": 1e-12},
        id="S := 2*A, ODE",
    ),
    pytest.param(
        RESIZE_ODE,
        "ode",
        "Y",
        lambda t: 10 * np.exp(-0.2 * t) / np.where(t >= 2, 4.0, 1.0),
        {"t_span": (0, 4.5), "n_points": 4, "rtol": 1e-10, "atol": 1e-12},
        id="event resize c: 1 -> 4, ODE",
    ),
    pytest.param(
        RESIZE_SSA,
        "ssa",
        "X",
        lambda t: 20 / np.where(t > 0.75, 5.0, 2.0),
        {"t_span": (0, 2), "n_points": 5, "seed": 11},
        id="event resize c: 2 -> 5, SSA",
    ),
]


@pytest.mark.parametrize(("ant", "method", "name", "oracle", "kw"), CASES)
def test_the_cdat_column_is_the_reported_species(tmp_path, ant, method, name, oracle, kw):
    """The issue's three cases: the .cdat column against the closed form,
    Result.species and to_csv(kind="species"). Main wrote the core's stale
    copy: 20 at every sample, amount/V_static after the resize."""
    r = _run(ant, method, **kw)
    r.to_cdat(tmp_path / "a.cdat")
    r.to_csv(tmp_path / "a.csv", kind="species")
    got = _column(tmp_path / "a.cdat", name)
    np.testing.assert_allclose(got, oracle(np.asarray(r.time)), rtol=1e-6)
    np.testing.assert_allclose(got, r.species[:, r.species_names.index(name)], rtol=1e-11)
    np.testing.assert_allclose(got, _column(tmp_path / "a.csv", name, sep=","), rtol=1e-11)


@pytest.mark.parametrize(("ant", "method", "name", "oracle", "kw"), CASES)
def test_a_loaded_result_writes_the_bytes_the_run_wrote(tmp_path, ant, method, name, oracle, kw):
    """A Result without a core used another writer, with another layout
    (`# time ...`, 20-wide fields). One writer now: the .cdat a run writes and
    the one its saved and reloaded copy writes are the same bytes."""
    pytest.importorskip("h5py")
    r = _run(ant, method, **kw)
    r.to_cdat(tmp_path / "a.cdat")
    r.save(tmp_path / "a.h5")
    bngsim.Result.load(tmp_path / "a.h5").to_cdat(tmp_path / "b.cdat")
    assert (tmp_path / "a.cdat").read_bytes() == (tmp_path / "b.cdat").read_bytes()


def test_an_uncorrected_result_writes_the_bytes_it_wrote_before(tmp_path):
    """Where nothing is corrected, the file is the core's own, byte for byte:
    the same writer, handed the same numbers."""
    r = _run("species A = 10; species B = 0; R1: A -> B; 0.3*A\n", t_span=(0, 5), n_points=11)
    r.to_cdat(tmp_path / "new.cdat")
    r._core.to_cdat(str(tmp_path / "core.cdat"))
    assert (tmp_path / "new.cdat").read_bytes() == (tmp_path / "core.cdat").read_bytes()
    assert (tmp_path / "new.cdat").read_text().splitlines()[0].startswith("#          time")


def test_a_projected_result_writes_the_core_s_bytes(tmp_path):
    """GH #71: the core's writer projects its stored species to the reported
    subset (an event-mutated parameter P is a species slot, not reported), and
    Result.species holds that subset. The two files are the same bytes."""
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = bngsim.Simulator(bngsim.Model.from_sbml_string(SBML_EVENT_PARAM), method="ode").run(
            t_span=(0, 1.0), n_points=6, rtol=1e-10, atol=1e-12
        )
    r.to_cdat(tmp_path / "new.cdat")
    r._core.to_cdat(str(tmp_path / "core.cdat"))
    assert "P" not in (tmp_path / "new.cdat").read_text().splitlines()[0].split()
    assert (tmp_path / "new.cdat").read_bytes() == (tmp_path / "core.cdat").read_bytes()


def test_every_route_with_a_core_writes_the_reported_species(tmp_path):
    """run_batch rows, run_replicates rows, parameter_scan rows and a copy of
    a Result all share a core, and all wrote its stale copy."""
    t = {"t_span": (0, 4), "n_points": 5}
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = _sim(RULE).run_batch(**t, params=[{}])
        scan = _sim(RULE).parameter_scan("k", [0.5], **t)
        reps = _sim(RULE, "ssa").run_replicates(n_replicates=2, **t, seed=3)
        one = _sim(RULE).run(**t)
    rows = list(batch) + list(getattr(scan, "results", scan)) + list(reps) + [copy.copy(one)]
    for i, r in enumerate(rows):
        r.to_cdat(tmp_path / f"{i}.cdat")
        np.testing.assert_allclose(
            _column(tmp_path / f"{i}.cdat", "S"),
            r.species[:, r.species_names.index("S")],
            rtol=1e-11,
        )


def test_a_squeezed_result_is_refused_by_name(tmp_path):
    """A squeezed run_batch Result holds one trajectory per row, and a .cdat
    file holds one. It failed inside string formatting, with a TypeError that
    named neither; it is refused, saying so."""
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _sim(RULE).run_batch(
            t_span=(0, 2), n_points=3, params=[{"k": 0.1}, {"k": 0.2}], squeeze=True
        )
    with pytest.raises(ValueError, match=r"to_cdat writes one trajectory; this Result holds 2"):
        r.to_cdat(tmp_path / "s.cdat")
