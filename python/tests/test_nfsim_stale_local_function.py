"""GH #1062: NFsim refuses a function with an argument that reads a changing observable outside it.

NFsim re-evaluates a function with an argument only on the complexes a reaction
produces (RuleWorld/nfsim#104). In ``fA(x) = k*(1+AC(x))*(1+0.01*MAB)`` the
``AC(x)`` term follows the molecule, but the ``MAB`` term keeps, on every
molecule off the firing complex, the value it had at that molecule's last event.
On ``tests/data/nfsim/local_function_outside_observable.bngl``, AB at t = 10 was
50.1 against network SSA 55.6 (100 seeds, z = -10), with no error. A rate law
``fB(x)*g()`` whose global ``g()`` reads the observable goes stale the same way.
So does a read through a table function's counter, ``tfun(..., MAB)``. Until a
fix is vendored, NfsimSession and Simulator(method='nf') refuse such a read by
name.

A read stays allowed when its observable can never change: every pattern is
inert (no rule names its types, and their seed complexes hold only types no rule
names) or a conserved total (``X()`` with no compartment that no rule adds,
deletes, or can delete with a whole species). Only functions a rate law uses are
checked. The check errs toward refusing; the generated network (ode / ssa)
always runs.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ModelError
from bngsim._network_free_checks import stale_local_function_reads

pytestmark = pytest.mark.skipif(
    not getattr(bngsim, "HAS_NFSIM", False),
    reason="NFsim not compiled in",
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"

REFUSED = [
    ("local_function_outside_observable.xml", r"'fA' reads 'MAB'"),
    ("local_function_outside_species_observable.xml", r"'fA' reads 'SpAB'"),
    # Src is named in no rule, but A(d)+A(d) merges the complexes that hold it.
    ("local_function_species_merge.xml", r"'fB' reads 'SpSrc'"),
    # fB(x)*g(): BNG2.pl writes _rateLaw1(x), which calls g(), which reads MAB.
    (
        "rate_law_local_times_global.xml",
        r"'_rateLaw1' \(the rate law of rule '_R1'\) reads 'MAB'",
    ),
    # fB(x)*tfun(..., MAB): the table's counter, kept only in ctrName.
    ("rate_law_table_function_counter.xml", r"'_rateLaw1' .*reads 'MAB'"),
    # AEC = A()@EC: no A is made or destroyed, but transport moves A out of EC.
    ("local_function_compartment_total.xml", r"'fA' reads 'AEC'"),
    # fB(x)*tfun(..., time): a table indexed by the clock.
    ("rate_law_time_table.xml", r"'_rateLaw1' .*reads 'time'"),
    # Xtot = X(): no rule adds or deletes an X, but $X is topped back up.
    ("local_function_fixed_species_total.xml", r"'fB' reads 'Xtot'"),
]

ALLOWED = [
    # Src has no components and no rule names it.
    "local_function_constant_outside_observable.xml",
    # Src has a component, but its only seed complex holds Src alone.
    "local_function_bondable_outside_observable.xml",
    # Atot = A(): no rule adds or deletes an A.
    "local_function_conserved_total.xml",
    # The same, plus C() -> 0: C never shares a complex with A.
    "local_function_conserved_total_with_degradation.xml",
    # fA(x) reads MAB, but no rule uses fA.
    "unused_local_function_reads_observable.xml",
    # The rate is a global g() reading MAB: re-evaluated whenever MAB changes.
    "global_function_reads_observable.xml",
    # Both read Obs_Src = Src(), which no rule names, outside the argument.
    "context_symmetry.xml",
    "symmetry_factor_rate_laws.xml",
]


@pytest.mark.parametrize("fixture, says", REFUSED)
def test_the_session_refuses_a_stale_outside_read(fixture, says):
    with pytest.raises(ModelError, match=rf"{says}.*RuleWorld/nfsim#104.*#1062"):
        bngsim.NfsimSession(str(DATA / fixture))


@pytest.mark.parametrize("fixture, says", REFUSED)
def test_the_simulator_refuses_a_stale_outside_read(fixture, says, simple_decay_net):
    m = bngsim.Model.from_net(str(simple_decay_net))
    with pytest.raises(ModelError, match=says):
        bngsim.Simulator(m, method="nf", xml_path=str(DATA / fixture))


@pytest.mark.parametrize("fixture", ALLOWED)
def test_an_outside_read_of_a_constant_observable_runs(fixture):
    xml = DATA / fixture
    assert stale_local_function_reads(xml) == []
    with bngsim.NfsimSession(str(xml)) as nf:
        nf.initialize(seed=1)
        nf.simulate(0, 1, n_points=2)


def test_an_allowed_conserved_total_read_agrees_with_the_network():
    # Allowed must mean right: NFsim against the generated network under SSA.
    xml = DATA / "local_function_conserved_total.xml"
    net = DATA / "local_function_conserved_total.net"
    n = 60
    nf = []
    for seed in range(1, n + 1):
        with bngsim.NfsimSession(str(xml)) as s:
            s.initialize(seed=seed)
            nf.append(np.asarray(s.simulate(0, 10, n_points=3).observables)[-1, 0])
    ssa = [
        # A fresh Model each time: run() writes its final state back into it.
        np.asarray(
            bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ssa")
            .run(t_span=(0, 10), n_points=3, seed=seed)
            .observables
        )[-1, 0]
        for seed in range(1, n + 1)
    ]
    a, b = np.array(nf), np.array(ssa)
    z = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
    assert abs(z) < 4.5, (a.mean(), b.mean(), z)


def test_a_read_through_the_argument_is_not_an_outside_read(tmp_path):
    # The same function with the MAB term dropped reads only AC(x).
    src = (DATA / "local_function_outside_observable.xml").read_text()
    old = "<Expression> (k*(1+AC(x)))*(1+(0.01*MAB)) </Expression>"
    assert src.count(old) == 1
    xml = tmp_path / "argument_only.xml"
    xml.write_text(src.replace(old, "<Expression> k*(1+AC(x)) </Expression>"))
    assert stale_local_function_reads(xml) == []
    bngsim.NfsimSession(str(xml)).destroy()


@pytest.mark.parametrize("ctr", ["Time", " time", "TIME()"])
def test_a_table_read_as_the_clock_however_it_is_spelled_is_refused(tmp_path, ctr):
    # NFsim trims and lowercases a table's counter and takes time or t as the
    # clock before any symbol. A step table, which NFsim reads exactly at its
    # breakpoints (#892), read in fB(x)*tfun(..., Time) goes stale the same way:
    # AX at t = 12 was 19.1 on nf against 69.1 on rm.
    src = (DATA / "rate_law_time_table.xml").read_text()
    old = 'ctrName="time" xData="0,10" yData="0.1,5" method="linear"'
    assert src.count(old) == 1
    xml = tmp_path / "clock_spelling.xml"
    xml.write_text(src.replace(old, f'ctrName="{ctr}" xData="0,10" yData="0.1,5" method="step"'))
    assert stale_local_function_reads(xml) == [("_rateLaw1", ctr.strip())]
    with pytest.raises(ModelError, match=r"'_rateLaw1' .*reads.*#1062"):
        bngsim.NfsimSession(str(xml))


@pytest.mark.parametrize(
    "edit",
    [
        lambda nf: nf.add_molecules("A", 10),
        lambda nf: nf.add_species("A(b,c)", 10),
        lambda nf: nf.remove_species("A(b,c)", 10),
        lambda nf: nf.set_species_count("A(b,c)", 50),
    ],
)
def test_a_session_edit_that_would_change_an_exempt_count_is_refused(edit):
    # Atot = A() runs only because no rule can change it; an edit could.
    with bngsim.NfsimSession(str(DATA / "local_function_conserved_total.xml")) as nf:
        nf.initialize(seed=1)
        nf.simulate(0, 1, n_points=2)
        with pytest.raises(ModelError, match=r"count of 'A': 'fA' reads 'Atot'.*#1062"):
            edit(nf)


def test_a_session_edit_of_another_type_runs():
    with bngsim.NfsimSession(str(DATA / "local_function_conserved_total.xml")) as nf:
        nf.initialize(seed=1)
        nf.add_molecules("B", 10)
        nf.simulate(0, 1, n_points=2)
