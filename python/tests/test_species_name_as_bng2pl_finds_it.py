"""A species name written another way than the network writes it finds the
species BNG2.pl's ``setConcentration`` finds (issue #1833; the first case of #1480).

BNG2.pl reads the first argument of ``setConcentration`` and ``addConcentration``
as one complete species and sets the species of the network that is the same
graph: components or molecules in another order, other bond labels, a symmetric
dimer bonded on the other of two equal sites, a compartment as ``@c:A()``,
``@c::A()`` or ``A()@c``, a bare ``C`` for a molecule with no components. It
aborts on anything that is no complete species of the network. bngsim's
``Model.set_concentration`` and ``get_concentration`` took only the network's
exact name (PyBNF replays a model's actions through the two), and the action
replay took a one-molecule species written another way but no complex, which
counted an isomorphic complex as a dropped change.

Oracle: BNG2.pl 2.9.3. ``ROWS`` holds, for each spelling, the species its
``setConcentration`` set, or ``None`` where it aborted, each from a run of its
own on the model beside it (``generate_network``, ``setConcentration(<spelling>,
777)``, ``writeNET``); the networks are the ones it wrote. Where BNG2.pl is
installed, a test runs it again on the same models. A second, independent
oracle for the matching itself: every pairing of molecules and components tried
in turn, on small random species; and for a large branched species, the code of
its tree of molecules.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim import ModelError
from bngsim._bngl_loader import _species_name, dropped_state_changes
from bngsim._species_graph import SpeciesIndex, parse, same_species
from bngsim.convert._bng2 import find_bng2
from bngsim.convert._protocol import Experiment, ProtocolSpec, StateChange

BNGL = {
    "plain": """begin model
begin parameters
  kf 0.001
  kr 0.1
  k 0.1
  C 5
end parameters
begin molecule types
  A(x,y,s~U~P)
  B(a)
  C()
  D(d,d)
  E()
  G(g)
  S(a,b)
  T(s,u)
  U(t)
end molecule types
begin seed species
  A(x,y,s~U) 100
  B(a) 100
  C() 100
  D(d,d) 100
  E() 100
  G(g) 100
  S(a,b) 100
  T(s,u) 100
  U(t) 100
end seed species
begin observables
  Molecules Atot A()
  Molecules E A()
  Molecules G G()
  Molecules Gobs G()
  Molecules Cobs C()
end observables
begin reaction rules
  A(x) + B(a) <-> A(x!1).B(a!1) kf, kr
  A(s~U) -> A(s~P) k
  D(d) + D(d) <-> D(d!1).D(d!1) kf, kr
  S(b) + T(s) <-> S(b!1).T(s!1) kf, kr
  T(u) + U(t) <-> T(u!1).U(t!1) kf, kr
end reaction rules
end model
generate_network({overwrite=>1,max_stoich=>{D=>2}})
""",
    "compartments": """begin model
begin parameters
  k 0.001
end parameters
begin compartments
  EC 3 1
  PM 2 1 EC
  CP 3 1 PM
end compartments
begin molecule types
  L(r)
  R(l)
  B()
end molecule types
begin seed species
  @EC:L(r) 100
  @PM:R(l) 100
  @EC:B() 50
  @CP:B() 20
end seed species
begin observables
  Molecules Ltot L()
end observables
begin reaction rules
  L(r) + R(l) <-> L(r!1).R(l!1) k, k
end reaction rules
end model
generate_network({overwrite=>1})
""",
    "two bonds between two sites": """begin model
begin parameters
  k 0.001
end parameters
begin molecule types
  W(x,y)
  E(e)
  F(f)
end molecule types
begin seed species
  W(x!1!2,y!1!2) 50
  E(e!1!2).F(f!1!2) 60
  W(x,y) 10
end seed species
begin observables
  Molecules Wtot W()
end observables
begin reaction rules
  W(x,y) -> 0 k
end reaction rules
end model
generate_network({overwrite=>1})
""",
}

# The networks BNG2.pl 2.9.3 wrote for them (parameters, species, reactions, groups).
NET = {
    "plain": """begin parameters
    1 kf    0.001  # Constant
    2 kr    0.1  # Constant
    3 k     0.1  # Constant
    4 C     5  # Constant
end parameters
begin species
    1 A(s~U,x,y) 100
    2 B(a) 100
    3 C() 100
    4 D(d,d) 100
    5 E() 100
    6 G(g) 100
    7 S(a,b) 100
    8 T(s,u) 100
    9 U(t) 100
   10 A(s~U,x!1,y).B(a!1) 0
   11 A(s~P,x,y) 0
   12 D(d!1,d).D(d!1,d) 0
   13 S(a,b!1).T(s!1,u) 0
   14 T(s,u!1).U(t!1) 0
   15 A(s~P,x!1,y).B(a!1) 0
   16 S(a,b!1).T(s!1,u!2).U(t!2) 0
end species
begin reactions
    1 1,2 10 kf #_R1
    2 1 11 k #_R2
    3 4,4 12 2*kf #_R3
    4 7,8 13 kf #_R4
    5 8,9 14 kf #_R5
    6 2,11 15 kf #_R1
    7 10 1,2 kr #_reverse__R1
    8 10 15 k #_R2
    9 12 4,4 kr #_reverse__R3
   10 7,14 16 kf #_R4
   11 13 7,8 kr #_reverse__R4
   12 9,13 16 kf #_R5
   13 14 8,9 kr #_reverse__R5
   14 15 2,11 kr #_reverse__R1
   15 16 7,14 kr #_reverse__R4
   16 16 9,13 kr #_reverse__R5
end reactions
begin groups
    1 Atot                 1,10,11,15
    2 E                    1,10,11,15
    3 G                    6
    4 Gobs                 6
    5 Cobs                 3
end groups
""",
    "compartments": """begin parameters
    1 k     0.001  # Constant
end parameters
begin species
    1 @EC::L(r) 100
    2 @PM::R(l) 100
    3 @EC::B() 50
    4 @CP::B() 20
    5 @PM::L(r!1)@EC.R(l!1) 0
end species
begin reactions
    1 1,2 5 1*k #_R1 unit_conversion=1/1
    2 5 1,2 k #_reverse__R1
end reactions
begin groups
    1 Ltot                 1,5
end groups
""",
    "two bonds between two sites": """begin parameters
    1 k     0.001  # Constant
end parameters
begin species
    1 W(x!1!2,y!1!2) 50
    2 E(e!1!2).F(f!1!2) 60
    3 W(x,y) 10
end species
begin reactions
    1 3 0 k #_R1
end reactions
begin groups
    1 Wtot                 1,3
end groups
""",
}

# (spelling, the species BNG2.pl 2.9.3 set, or None where it aborted)
ROWS = {
    "plain": [
        ("A(s~U,x,y)", "A(s~U,x,y)"),
        ("A(y,x,s~U)", "A(s~U,x,y)"),
        ("A(y,s~U,x)", "A(s~U,x,y)"),
        ("A(x,y,s~P)", "A(s~P,x,y)"),
        ("$B(a)", "B(a)"),
        ("B(a)", "B(a)"),
        ("C", "C()"),
        ("C()", "C()"),
        ("E", "E()"),
        ("E()", "E()"),
        ("G", None),
        ("A", None),
        ("A()", None),
        ("B", None),
        ("Gobs", None),
        ("Atot", None),
        ("Cobs", None),
        ("c", None),
        ("F()", None),
        ("A(x,s~U)", None),
        ("A(x,y,s)", None),
        ("A(x,y,s~Q)", None),
        ("A(x,y,s~U,z)", None),
        ("A(s~P)", None),
        ("A(x,y,s~?)", None),
        ("A(s~U,x!+,y)", None),
        ("A(s~U,x!?,y)", None),
        ("A(s~U,x!1,y).B(a!1)", "A(s~U,x!1,y).B(a!1)"),
        ("B(a!1).A(s~U,x!1,y)", "A(s~U,x!1,y).B(a!1)"),
        ("A(s~U,x!7,y).B(a!7)", "A(s~U,x!1,y).B(a!1)"),
        ("B(a!3).A(y,x!3,s~U)", "A(s~U,x!1,y).B(a!1)"),
        ("B(a!1).A(x!1,y,s~P)", "A(s~P,x!1,y).B(a!1)"),
        ("A(s~U,x,y).B(a)", None),
        ("A(s~U,x!1,y)", None),
        ("A(s~U,x!1,y).B(a!2)", None),
        ("A(s~U,x!1,y!1)", None),
        ("A(s~P,x!1,y).A(s~P,x!1,y)", None),
        ("D(d!1,d).D(d!1,d)", "D(d!1,d).D(d!1,d)"),
        ("D(d,d!1).D(d!1,d)", "D(d!1,d).D(d!1,d)"),
        ("D(d,d!2).D(d,d!2)", "D(d!1,d).D(d!1,d)"),
        ("D(d!1,d!2).D(d!1,d!2)", None),
        ("D(d,d)", "D(d,d)"),
        ("S(a,b!1).T(s!1,u!2).U(t!2)", "S(a,b!1).T(s!1,u!2).U(t!2)"),
        ("U(t!2).T(u!2,s!1).S(b!1,a)", "S(a,b!1).T(s!1,u!2).U(t!2)"),
        ("T(s!9,u!8).U(t!8).S(a,b!9)", "S(a,b!1).T(s!1,u!2).U(t!2)"),
        ("S(a,b!1).T(s!1,u!1).U(t!1)", None),
        ("S(a,b!1).T(s!2,u!1).U(t!2)", None),
        ("S(a,b!1).T(s!1,u)", "S(a,b!1).T(s!1,u)"),
        ("T(u,s!1).S(b!1,a)", "S(a,b!1).T(s!1,u)"),
        ("T(s,u!1).U(t!1)", "T(s,u!1).U(t!1)"),
        ("U(t!5).T(u!5,s)", "T(s,u!1).U(t!1)"),
        ("", None),
        ("()", None),
        ("A(", None),
        ("A(x,y,s~U", "A(s~U,x,y)"),
        ("A(x,,y,s~U)", "A(s~U,x,y)"),
        ("A(x,y,s~U).", "A(s~U,x,y)"),
        (".A(x,y,s~U)", "A(s~U,x,y)"),
        ("C()()", "C()"),
        ("C().C()", None),
        (" B(a)", "B(a)"),
        ("B(a) ", None),
        ("B (a)", None),
        ("B( a)", None),
        ("A(s~U, x,y)", None),
        ("A(s~U,x!1,y). B(a!1)", None),
        ("S(a,b!1).T(s!1,u!1)", None),
        ("A(x,y,s~U~P)", None),
        ("A(x,y,s~P~U)", None),
        ("A(s~U,y,x!1).B(a!1)", "A(s~U,x!1,y).B(a!1)"),
        ("A(x!1,y,s~U).B(a!1)", "A(s~U,x!1,y).B(a!1)"),
        ("A(x!1~U,y,s~U).B(a!1)", None),
        ("B(a!1).A(s~U,x!1,y).C()", None),
    ],
    "compartments": [
        ("@EC::L(r)", "@EC::L(r)"),
        ("@EC:L(r)", "@EC::L(r)"),
        ("L(r)@EC", "@EC::L(r)"),
        ("$@EC::L(r)", None),
        ("@EC::$L(r)", "@EC::L(r)"),
        ("L(r)", None),
        ("@PM:L(r)", None),
        ("@CP:L(r)", None),
        ("@PM::L(r!1).R(l!1)", None),
        ("@PM:R(l!1).L(r!1)", None),
        ("L(r!1)@EC.R(l!1)@PM", "@PM::L(r!1)@EC.R(l!1)"),
        ("R(l!4)@PM.L(r!4)@EC", "@PM::L(r!1)@EC.R(l!1)"),
        ("@PM::L(r!1)@EC.R(l!1)", "@PM::L(r!1)@EC.R(l!1)"),
        ("@PM:R(l!2).L(r!2)@EC", "@PM::L(r!1)@EC.R(l!1)"),
        ("L(r!1).R(l!1)", None),
        ("@EC:L(r!1).R(l!1)", None),
        ("L(r!1)@PM.R(l!1)@EC", None),
        ("@EC:B", "@EC::B()"),
        ("@EC:B()", "@EC::B()"),
        ("B()@CP", "@CP::B()"),
        ("B@CP", "@CP::B()"),
        ("@CP::B()", "@CP::B()"),
        ("B", None),
        ("B()", None),
        ("@PM:B()", None),
        ("@XX:B()", None),
        ("@EC:$L(r)", "@EC::L(r)"),
        ("$L(r)@EC", "@EC::L(r)"),
        ("@EC::$B()", "@EC::B()"),
        ("$B()@CP", "@CP::B()"),
        ("@CP:B()@EC", None),
        ("@EC:B()@CP", None),
        ("@EC:B()@EC", "@EC::B()"),
        ("@EC:L(r!1).R(l!1)@PM", None),
        ("@PM:L(r!1)@EC.R(l!1)@PM", "@PM::L(r!1)@EC.R(l!1)"),
        ("@EC::L(r)@EC", "@EC::L(r)"),
        ("@PM:L(r)@EC", None),
        ("@PM:R(l!1).L(r!1)@EC", "@PM::L(r!1)@EC.R(l!1)"),
        ("@EC:R(l!1)@PM.L(r!1)", None),
        ("@ EC:L(r)", None),
        ("L(r)@ EC", None),
        ("L(r) @EC", None),
        ("@EC :L(r)", None),
        ("@EC: L(r)", "@EC::L(r)"),
    ],
    "two bonds between two sites": [
        ("W(x!1!2,y!1!2)", "W(x!1!2,y!1!2)"),
        ("W(y!1!2,x!1!2)", "W(x!1!2,y!1!2)"),
        ("W(y!7!8,x!8!7)", "W(x!1!2,y!1!2)"),
        ("W(x!2!1,y!1!2)", "W(x!1!2,y!1!2)"),
        ("F(f!1!2).E(e!1!2)", "E(e!1!2).F(f!1!2)"),
        ("E(e!3!4).F(f!4!3)", "E(e!1!2).F(f!1!2)"),
        ("E(e!1).F(f!1)", None),
        ("W(x!1,y!1)", None),
        ("W(x!1!1,y)", None),
        ("W(x!1!2,y!1)", None),
        ("W(y,x)", "W(x,y)"),
    ],
}

# Where bngsim reads a name otherwise than BNG2.pl, by choice. Neither list can
# name a wrong species: the looser ones spell one species only, and the
# stricter ones are refused.
LOOSER = {
    # A space after or inside the name, with the species each one spells.
    # BNG2.pl aborts on these and takes one before the name; the action replay
    # took such names in a one-molecule name before this, and keeps doing so.
    "B(a) ": "B(a)",
    "B (a)": "B(a)",
    "B( a)": "B(a)",
    "A(s~U, x,y)": "A(s~U,x,y)",
    "A(s~U,x!1,y). B(a!1)": "A(s~U,x!1,y).B(a!1)",
    "L(r) @EC": "@EC::L(r)",
    # A fixed species' $ before the compartment: Model drops a $ in either place.
    "$@EC::L(r)": "@EC::L(r)",
}
STRICTER = {
    # Malformed names that BNG2.pl's parser lets through. (A doubled comma it
    # lets through as well, and so does bngsim: the replay always took it.)
    "A(x,y,s~U",
    "A(x,y,s~U).",
    ".A(x,y,s~U)",
    "C()()",
}
SETS = [(m, s, want) for m, rows in ROWS.items() for s, want in rows if want and s not in STRICTER]
REFUSED = [(m, s) for m, rows in ROWS.items() for s, want in rows if not want and s not in LOOSER]


def _model(tmp_path: Path, which: str) -> bngsim.Model:
    net = tmp_path / f"{which}.net"
    net.write_text(NET[which], encoding="utf-8")
    return bngsim.Model.from_net(str(net))


def _holding(model, value: float) -> list[str]:
    return [n for n in model.species_names if model.get_concentration(n) == value]


def test_the_table_is_the_one_the_tests_read():
    # Every spelling is in exactly one of the three groups, and the two lists
    # of chosen differences are the rows where bngsim and BNG2.pl part.
    every = {s for rows in ROWS.values() for s, _ in rows}
    assert every >= set(LOOSER) and every >= STRICTER
    assert {s for rows in ROWS.values() for s, want in rows if want} >= STRICTER
    assert {s for rows in ROWS.values() for s, want in rows if not want} >= set(LOOSER)
    assert len(SETS) + len(REFUSED) + len(LOOSER) + len(STRICTER) == sum(map(len, ROWS.values()))
    # BNG2.pl only ever sets a species of the network it wrote.
    for which, rows in ROWS.items():
        names = set(_species_block(NET[which]))
        assert {want for _, want in rows if want} <= names


def _species_block(net: str) -> list[str]:
    block = net[net.index("begin species") : net.index("end species")].splitlines()[1:]
    return [line.split()[1] for line in block]


@pytest.mark.parametrize(("which", "spelling", "want"), SETS, ids=[s for _, s, _ in SETS])
def test_a_spelling_sets_and_reads_the_species_bng2pl_sets(tmp_path, which, spelling, want):
    model = _model(tmp_path, which)
    before = {n: model.get_concentration(n) for n in model.species_names}
    model.set_concentration(spelling, 777.0)
    # That species and no other: no species holds 777 as loaded.
    assert _holding(model, 777.0) == [want]
    assert {n: v for n, v in before.items() if n != want} == {
        n: model.get_concentration(n) for n in model.species_names if n != want
    }
    assert model.get_concentration(spelling) == 777.0
    assert _species_name(model, spelling) == want


@pytest.mark.parametrize(("which", "spelling"), REFUSED, ids=[repr(s) for _, s in REFUSED])
def test_a_spelling_bng2pl_aborts_on_is_refused(tmp_path, which, spelling):
    model = _model(tmp_path, which)
    before = [model.get_concentration(n) for n in model.species_names]
    with pytest.raises(ModelError, match="not found in model"):
        model.set_concentration(spelling, 777.0)
    with pytest.raises(ModelError, match="not found in model"):
        model.get_concentration(spelling)
    assert _species_name(model, spelling) is None
    assert [model.get_concentration(n) for n in model.species_names] == before


@pytest.mark.parametrize("spelling", sorted(LOOSER))
def test_a_space_or_a_fixed_mark_bng2pl_aborts_on_still_names_one_species(tmp_path, spelling):
    (which,) = [m for m, rows in ROWS.items() if spelling in dict(rows)]
    model = _model(tmp_path, which)
    model.set_concentration(spelling, 777.0)
    assert _holding(model, 777.0) == [LOOSER[spelling]]
    assert _species_name(model, spelling) == LOOSER[spelling]


@pytest.mark.parametrize("spelling", sorted(STRICTER))
def test_a_malformed_name_bng2pl_lets_through_is_refused(tmp_path, spelling):
    assert parse(spelling) is None
    with pytest.raises(ModelError, match="not found in model"):
        _model(tmp_path, "plain").set_concentration(spelling, 777.0)


# --- a bare name, and the names of observables and parameters ---------------


def test_a_bare_name_is_a_molecule_with_no_components(tmp_path):
    # E() has no components, so `E` is the species E(). The model also has an
    # observable E, which counts A: BNG2.pl does not consult it, and the
    # species holding an A keep the amounts loaded.
    model = _model(tmp_path, "plain")
    assert "E" in model.observable_names and "E()" in model.species_names
    with_a = {n: model.get_concentration(n) for n in model.species_names if n.startswith("A(")}
    model.set_concentration("E", 7.0)
    assert model.get_concentration("E()") == 7.0 == model.get_concentration("E")
    assert with_a == {n: model.get_concentration(n) for n in with_a}
    # C is a parameter's name as well: the species C() all the same.
    assert "C" in model.param_names
    model.set_concentration("C", 8.0)
    assert model.get_concentration("C()") == 8.0 and model.get_param("C") == 5.0


@pytest.mark.parametrize(
    ("name", "says"),
    [
        # BNG2.pl: "Component(s) g missing from molecule G()". G is an
        # observable too, and the message says which species it counts.
        ("G", "'G' is an observable"),
        ("G", "It counts one species, 'G(g)'"),
        # "Component(s) a missing from molecule B()".
        ("B", "Model.species_names lists all 16."),
        # "Molecule c() does not match any declared molecule types": the
        # species that differs only in case is still pointed to.
        ("c", "The species 'C()' is close"),
    ],
)
def test_a_name_that_spells_no_species_keeps_the_message_it_had(tmp_path, name, says):
    # Issue #1700's messages, for every bare name BNG2.pl refuses.
    model = _model(tmp_path, "plain")
    with pytest.raises(ModelError) as raised:
        model.set_concentration(name, 1.0)
    assert f"Species '{name}' not found in model" in str(raised.value)
    assert says in str(raised.value)


def test_an_observables_name_is_no_species(tmp_path):
    model = _model(tmp_path, "plain")
    for name in ("Gobs", "Atot", "Cobs"):
        assert name in model.observable_names
        with pytest.raises(ModelError, match="not found in model"):
            model.get_concentration(name)


# --- the action replay (#1480, first case) ----------------------------------

RUN = Experiment(kind="simulate", method="ode", t_span=(0.0, 1e-9), n_points=2)


def _change(kind: str, target: str, value: float) -> StateChange:
    return StateChange(kind=kind, target=target, value=float(value))


def test_the_replay_sets_and_adds_to_a_complex_written_another_way(tmp_path):
    model = _model(tmp_path, "plain")
    steps = (
        _change("set_concentration", "B(a!3).A(y,x!3,s~U)", 40),
        _change("add_concentration", "A(s~U,x!7,y).B(a!7)", 2),
        _change("set_concentration", "D(d,d!2).D(d,d!2)", 9),
        RUN,
    )
    out = bngsim.run_protocol(model, ProtocolSpec(steps=steps), rtol=1e-10, atol=1e-12)
    start = dict(zip(out[0].result.species_names, out[0].result.species[0], strict=True))
    assert start["A(s~U,x!1,y).B(a!1)"] == 42.0
    assert start["D(d!1,d).D(d!1,d)"] == 9.0


def test_an_isomorphic_complex_set_to_its_own_amount_is_no_dropped_change(tmp_path):
    # #1480's case: the network holds D(d!1,d).D(d!1,d) at 0, and the action
    # writes the same species with another label on the other site.
    model = _model(tmp_path, "plain")
    text = 'generate_network({overwrite=>1})\nsetConcentration("%s",%s)\nsimulate({})\n'
    assert dropped_state_changes(model, text % ("D(d,d!2).D(d,d!2)", 0)) == []
    (named,) = dropped_state_changes(model, text % ("D(d,d!2).D(d,d!2)", 3))
    assert "D(d,d!2).D(d,d!2)" in named
    # A species the network has not still counts as a change.
    assert len(dropped_state_changes(model, text % ("D(d!1,d!2).D(d!1,d!2)", 0))) == 1


@pytest.mark.parametrize("spelling", sorted(STRICTER))
def test_the_replay_refuses_a_malformed_name_without_saying_bng2pl_does(tmp_path, spelling):
    # BNG2.pl 2.9.3 lets each of these through (the table's rows); the
    # replay's refusal says what bngsim found and claims nothing of BNG2.pl.
    model = _model(tmp_path, "plain")
    steps = (_change("set_concentration", spelling, 40), RUN)
    with pytest.raises(bngsim.ConversionError) as caught:
        bngsim.run_protocol(model, ProtocolSpec(steps=steps))
    assert "the network has no species" in str(caught.value)
    assert "BNG2.pl" not in str(caught.value)


# --- the index: built late, one per model, and never two answers ------------


def test_the_index_is_built_at_the_first_name_written_another_way(tmp_path):
    model = _model(tmp_path, "plain")
    model.set_concentration("A(s~U,x,y)", 5.0)
    model.set_concentration("$B(a)", 6.0)
    assert model._species_index is None  # exact names never need it
    model.set_concentration("A(y,x,s~U)", 7.0)
    assert model._species_index is not None
    copy = model.clone()
    assert copy._species_index is None
    assert copy.get_concentration("A(y,s~U,x)") == 7.0
    copy.set_concentration("A(y,x,s~U)", 8.0)
    assert model.get_concentration("A(s~U,x,y)") == 7.0


def test_two_names_of_one_species_in_a_network_are_not_chosen_between():
    index = SpeciesIndex(["A(x,y)", "A(y,x)", "B()"])
    assert index.find("A( y, x )") is None
    assert index.find("B") == "B()"
    assert index.find("") is None  # not what the name before it found
    assert index.find("B") == "B()"  # the second time from memory
    assert index.find("not a species (") is None


HAND_WRITTEN = """\
begin parameters
    1 k 1.0
end parameters
begin compartments
    EC 3 1
    PM 2 1 EC
    CP 3 1 PM
end compartments
begin species
    1 L(r)@EC 5
    2 %s 7
    3 L(r!1)@EC.R(l!1)@PM 3
    4 A(b!1)@CP.C(a!1)@CP 2
end species
begin reactions
    1 1 0 k
end reactions
"""
# (spelling, the species bngsim finds, whether BNG2.pl 2.9.3 sets one). BNG2.pl
# read the network with readFile, one run for each spelling, and set the same
# species wherever bngsim finds one.
HAND_ROWS = [
    # The compartment every molecule of the species is in, before the name.
    ("@EC:L(r)", "L(r)@EC", True),
    ("@EC::L(r)", "L(r)@EC", True),
    ("@CP:B", "B()@CP", True),
    ("@CP:B()", "B()@CP", True),
    # A complex with every molecule in one compartment, the same way.
    ("@CP:A(b!1).C(a!1)", "A(b!1)@CP.C(a!1)@CP", True),
    ("@CP::C(a!3)@CP.A(b!3)", "A(b!1)@CP.C(a!1)@CP", True),
    ("@EC:A(b!1).C(a!1)", None, False),
    # No prefix: found as any species written another way.
    ("R(l!7)@PM.L(r!7)@EC", "L(r!1)@EC.R(l!1)@PM", True),
    # Another compartment than the molecule's.
    ("@CP:L(r)", None, False),
    ("@EC:B()", None, False),
    ("@EC:B()@CP", None, False),
    # Molecules in two compartments. BNG2.pl, which has the compartments'
    # dimensions, takes the surface's prefix (@PM) and aborts on @EC. The
    # index has no dimensions and takes neither, as the replay did before.
    ("@EC:L(r!1).R(l!1)@PM", None, False),
    ("@PM:R(l!1).L(r!1)@EC", None, True),
    ("@PM:L(r!1)@EC.R(l!1)@PM", None, True),
]


def _hand_written(tmp_path: Path, second: str = "B()@CP") -> Path:
    net = tmp_path / "hand.net"
    net.write_text(HAND_WRITTEN % second, encoding="utf-8")
    return net


@pytest.mark.parametrize(("spelling", "want"), [row[:2] for row in HAND_ROWS])
def test_a_prefix_on_a_network_that_writes_compartments_on_molecules_only(
    tmp_path, spelling, want
):
    # A network written by hand, with no @c:: prefix before the names (BNG2.pl
    # writes one before every species of a compartmental network). The action
    # replay took the four prefixed one-molecule names before this, by its
    # molecule key; a compartment that is not the molecules' never named one.
    model = bngsim.Model.from_net(str(_hand_written(tmp_path)))
    assert _species_name(model, spelling) == want
    if want is None:
        with pytest.raises(ModelError, match="not found in model"):
            model.set_concentration(spelling, 777.0)
        assert _holding(model, 777.0) == []
    else:
        model.set_concentration(spelling, 777.0)
        assert _holding(model, 777.0) == [want]


def test_a_prefix_the_networks_own_contradicts_finds_nothing(tmp_path):
    # A name that states one compartment before it and another on its
    # molecule (no network BNG2.pl writes, and one it does not read): the
    # prefix the name states is the species' own, and the molecule's is not.
    model = bngsim.Model.from_net(str(_hand_written(tmp_path, "@EC::B()@CP")))
    assert "@EC::B()@CP" in model.species_names
    assert _species_name(model, "@CP:B()") is None
    with pytest.raises(ModelError, match="not found in model"):
        model.set_concentration("@CP:B()", 777.0)
    assert _holding(model, 777.0) == []


def test_names_that_are_no_bngl_species_do_not_stop_the_index():
    # An SBML model names its species by id; some names are no BNGL at all.
    # The last is alike enough to be looked at beside the one asked for, and
    # is no species (a wildcard).
    index = SpeciesIndex(["S1", "glucose-6-P", "[x]", "A(b!1).B(a!1)", "A(b!+).B(a!1)"])
    assert index.find("B(a!4).A(b!4)") == "A(b!1).B(a!1)"
    # A name with no parentheses is an id, not a molecule: S1() is not S1.
    assert index.find("S1()") is None and index.find(" S1") is None
    assert index.find("glucose") is None


@pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML loading requires libsbml")
def test_an_sbml_species_is_found_by_its_exact_id_only():
    sbml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">'
        '<model id="m"><listOfCompartments><compartment id="c" size="1" constant="true"/>'
        "</listOfCompartments><listOfSpecies>"
        '<species id="S1" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"'
        ' boundaryCondition="false" constant="false"/>'
        "</listOfSpecies></model></sbml>"
    )
    model = bngsim.Model.from_sbml_string(sbml)
    model.set_concentration("S1", 3.0)
    for other in ("S1()", "@c:S1", " S1", "S1 "):
        with pytest.raises(ModelError, match="not found in model"):
            model.set_concentration(other, 4.0)
        with pytest.raises(ModelError, match="not found in model"):
            model.get_concentration(other)
    assert model.get_concentration("S1") == 3.0
    # The replay keeps its own rule for such a model: the SBML bngsim writes
    # for a network names A() "A", and an archive's protocol still says A().
    assert _species_name(model, "S1()") == "S1"
    assert _species_name(model, "S1(x)") is None


def test_a_write_by_another_spelling_is_booked_under_the_networks_name(tmp_path):
    # set_concentration supersedes what the species' initial condition was
    # declared to depend on. The declaration is kept under the network's name,
    # so a write by another spelling has to drop it there. Oracle: the same
    # write by the network's own name, which main dropped it for.
    declared = {"A(s~U,x,y)": {"kf": 2.0}, "B(a)": {"kf": 3.0}}
    for spelling in ("A(s~U,x,y)", "A(y,x,s~U)"):
        model = _model(tmp_path, "plain")
        model.declare_ic_sensitivity(declared)
        assert model._declared_ic_sens == declared
        model.set_concentration(spelling, 5.0)
        assert model._declared_ic_sens == {"B(a)": {"kf": 3.0}}, spelling


def test_a_doubled_comma_is_read_as_bng2pl_and_the_replay_always_read_it(tmp_path):
    model = _model(tmp_path, "plain")
    assert dict(ROWS["plain"])["A(x,,y,s~U)"] == "A(s~U,x,y)"
    model.set_concentration("A(x,,y,s~U)", 777.0)
    assert _holding(model, 777.0) == ["A(s~U,x,y)"]
    assert _species_name(model, "A(y,,x,s~U,)") == "A(s~U,x,y)"


# --- the reading and the matching by themselves -----------------------------


@pytest.mark.parametrize(
    "text",
    [
        "S(a,b!1).T(s!1,u!1)",  # a bond with three ends
        "A(x!1,y)",  # a bond with one
        "A(x,y,s~U~P)",  # two states
        "W(x!1!1,y)",  # a site bonded to itself
        "@ EC:L(r)",  # a space inside the compartment before the name
        "@EC :L(r)",
        "A(x!+,y)",
        "A(x!?,y)",
        "A(x,y,s~?)",
        "A(x,y",
        "A(x,y))",
        "A(x,y).",
        ".A(x,y)",
        "A B",
        "1A()",
        "",
    ],
)
def test_what_is_no_complete_species_is_not_read(text):
    assert parse(text) is None


def test_a_bond_inside_one_molecule_and_two_bonds_between_two_sites_are_read():
    # Species all the same: BNG2.pl writes both in a network.
    one, two = parse("A(x!1,y!1)"), parse("W(x!1!2,y!1!2)")
    assert one is not None and two is not None
    # Two sites and a molecule, and a node for each bond.
    assert len(one.labels) == 4 and len(two.labels) == 5


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("B()", "C()"),  # the molecule's name
        ("A(x)", "A(y)"),  # the component's
        ("A(s~U)", "A(s~P)"),  # its state
        ("A(s~U)", "A(s)"),
        ("@EC:B()", "@CP:B()"),  # the compartment
        ("B()", "@EC:B()"),
        ("A(x!1).B(a!1)@EC", "A(x!1)@EC.B(a!1)"),  # and whose it is
        ("A(x,y!1).B(a!1)", "A(x!1,y).B(a!1)"),  # which site is bonded
        ("A(x,x)", "A(x)"),  # how many sites
        ("W(x!1!2,y!1!2)", "W(x!1,y!1)"),  # two bonds between two sites, and one
        ("E(e!1!2).F(f!1!2)", "E(e!1).F(f!1)"),
    ],
)
def test_two_names_that_differ_in_one_thing_are_two_species(one, other):
    a, b = parse(one), parse(other)
    assert a is not None and b is not None
    assert not same_species(a, b) and not same_species(b, a)
    assert same_species(a, a) and same_species(b, b)


# --- the matching against every pairing, on small random species ------------

MOLECULES = {"A": ("x", "x", "s"), "B": ("a", "b"), "C": ()}
STATES = {"s": ("U", "P")}


def _random_species(rng: random.Random):
    """Molecules as (name, compartment, [(component, state, bonds)]), with
    each bond on two components, and now and then two bonds on the same two."""
    species = []
    for _ in range(rng.randint(1, 3)):
        name = rng.choice(sorted(MOLECULES))
        components = [[c, rng.choice(STATES.get(c, ("",))), []] for c in MOLECULES[name]]
        species.append([name, rng.choice(("", "", "EC", "PM")), components])
    sites = [c for _, _, components in species for c in components]
    rng.shuffle(sites)
    for label in range(1, rng.randint(0, len(sites) // 2) + 1):
        one, two = sites.pop(), sites.pop()
        for bond in (label, label + 50)[: 2 if rng.random() < 0.15 else 1]:
            one[2].append(bond)
            two[2].append(bond)
    return species


def _write(species, rng: random.Random | None = None) -> str:
    """The species as BNGL; with *rng*, molecules and components in a random
    order and the bonds under random labels."""
    species = [[n, c, [list(x) for x in comps]] for n, c, comps in species]
    relabel: dict[int, int] = {}
    if rng is not None:
        rng.shuffle(species)
        labels = sorted({b for _, _, comps in species for _, _, bonds in comps for b in bonds})
        relabel = dict(zip(labels, rng.sample(range(10, 99), len(labels)), strict=True))
        for _, _, comps in species:
            rng.shuffle(comps)
    parts = []
    for name, compartment, comps in species:
        inside = ",".join(
            c + (f"~{s}" if s else "") + "".join(f"!{relabel.get(b, b)}" for b in bonds)
            for c, s, bonds in comps
        )
        parts.append(f"{name}({inside})" + (f"@{compartment}" if compartment else ""))
    return ".".join(parts)


def _same_by_every_pairing(a, b) -> bool:
    """Whether some order of *b*'s molecules, and of the components in each,
    makes it *a* with the bonds renamed: every order tried."""
    if sorted((n, c) for n, c, _ in a) != sorted((n, c) for n, c, _ in b):
        return False

    def bonds_of(species):
        ends: dict[int, list[tuple[int, int]]] = {}
        for i, (_, _, comps) in enumerate(species):
            for j, (_, _, bonds) in enumerate(comps):
                for bond in bonds:
                    ends.setdefault(bond, []).append((i, j))
        return sorted(tuple(sorted(pair)) for pair in ends.values())

    want = bonds_of(a)
    for order in itertools.permutations(b):
        if [(n, c) for n, c, _ in order] != [(n, c) for n, c, _ in a]:
            continue
        choices = [itertools.permutations(comps) for _, _, comps in order]
        for inner in itertools.product(*choices):
            candidate = [
                [n, c, list(comps)] for (n, c, _), comps in zip(order, inner, strict=True)
            ]
            if [[(x, s) for x, s, _ in comps] for _, _, comps in candidate] != [
                [(x, s) for x, s, _ in comps] for _, _, comps in a
            ]:
                continue
            if bonds_of(candidate) == want:
                return True
    return False


def test_the_matching_agrees_with_trying_every_pairing():
    rng = random.Random(1833)
    same = different = 0
    for _ in range(400):
        a = _random_species(rng)
        # Half the time another spelling of it, half the time another species.
        b = a if rng.random() < 0.5 else _random_species(rng)
        text_a, text_b = _write(a), _write(b, rng)
        graph_a, graph_b = parse(text_a), parse(text_b)
        assert graph_a is not None and graph_b is not None, (text_a, text_b)
        expected = _same_by_every_pairing(a, b)
        assert same_species(graph_a, graph_b) == expected, (text_a, text_b)
        assert same_species(graph_b, graph_a) == expected, (text_a, text_b)
        # And through the index, which first sorts names by what the text shares.
        assert SpeciesIndex([text_a]).find(text_b) == (text_a if expected else None)
        same += expected
        different += not expected
    # Both answers are well represented, so neither is trivially right.
    assert same > 100 and different > 100


def _six(edges) -> str:
    """Six molecules M(s,s,s), bonded along *edges*, pairs of molecules."""
    sites: list[list[str]] = [[] for _ in range(6)]
    for label, (i, j) in enumerate(edges, 1):
        sites[i].append(f"s!{label}")
        sites[j].append(f"s!{label}")
    return ".".join(f"M({','.join(s)})" for s in sites)


def test_two_species_alike_in_every_count_but_bonded_otherwise_are_told_apart():
    # Six molecules of three equal sites each, every site bonded: a prism (two
    # triangles joined corner to corner) and the graph that joins each of
    # three molecules to each of the other three. Every molecule and every
    # component looks the same in both, and only the bonding differs: the
    # prism has triangles, the other has none.
    prism = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
    k33 = [(i, j) for i in (0, 1, 2) for j in (3, 4, 5)]
    a, b = parse(_six(prism)), parse(_six(k33))
    assert a is not None and b is not None and a.key == b.key
    assert not same_species(a, b) and not same_species(b, a)
    # Each is itself under another numbering of its molecules and bonds.
    turn = [3, 5, 1, 0, 4, 2]
    for edges, graph in ((prism, a), (k33, b)):
        again = [(turn[i], turn[j]) for i, j in reversed(edges)]
        assert same_species(graph, parse(_six(again)))
    index = SpeciesIndex([_six(prism)])
    assert index.find(_six([(turn[i], turn[j]) for i, j in prism])) == _six(prism)
    assert index.find(_six(k33)) is None


def test_one_ring_and_two_rings_of_the_same_parts_are_told_apart():
    # Six two-site molecules in one ring, and in two rings of three: the same
    # labels, the same degrees and the same neighborhoods at any depth. Only
    # the bonding tells the two apart, and two rings are no one complex.
    ring = "Q(p!1,p!2).Q(p!2,p!3).Q(p!3,p!4).Q(p!4,p!5).Q(p!5,p!6).Q(p!6,p!1)"
    two = "Q(p!1,p!2).Q(p!2,p!3).Q(p!3,p!1).Q(p!4,p!5).Q(p!5,p!6).Q(p!6,p!4)"
    again = "Q(p!4,p!5).Q(p!1,p!2).Q(p!6,p!1).Q(p!2,p!3).Q(p!5,p!6).Q(p!3,p!4)"
    a, b = parse(ring), parse(two)
    assert a is not None and b is not None and a.key == b.key
    assert not same_species(a, b) and not same_species(b, a)
    assert same_species(a, parse(again))
    assert SpeciesIndex([ring]).find(two) is None


def test_a_species_of_hundreds_of_molecules_is_matched():
    # A chain of 400 two-site molecules, its molecules in another order with
    # the sites swapped and the bonds renumbered; and the ring that closes it,
    # which has two more bond ends and is another species. (The search keeps
    # its own stack: 1,200 nodes are deeper than a function may call itself.)
    n = 400
    chain = [f"M(l{f'!{i}' if i else ''},r{f'!{i + 1}' if i < n - 1 else ''})" for i in range(n)]
    ring = [f"M(l!{i or n},r!{i + 1})" for i in range(n)]
    rng = random.Random(400)

    def again(molecules: list[str]) -> str:
        out = []
        for m in rng.sample(molecules, len(molecules)):
            left, right = m[2:-1].split(",")
            out.append(f"M({right},{left})")
        text = ".".join(out)
        return re.sub(r"!(\d+)", lambda b: f"!{int(b.group(1)) + 5000}", text)

    for molecules in (chain, ring):
        a, b = parse(".".join(molecules)), parse(again(molecules))
        assert a is not None and b is not None
        # A node for each molecule, each of its two sites, and each bond.
        assert len(a.labels) == 3 * n + (n if molecules is ring else n - 1)
        assert same_species(a, b) and same_species(b, a)
    assert not same_species(parse(".".join(chain)), parse(".".join(ring)))
    # A chain with one molecule's sites the other way round is another species
    # with the same counts of everything: an l bonded to an l.
    turned = list(chain)
    turned[200] = "M(l!201,r!200)"
    c = parse(".".join(turned))
    assert c is not None and c.key == parse(".".join(chain)).key
    assert not same_species(parse(".".join(chain)), c)


def _tree(n: int, rng: random.Random) -> list[tuple[str, list[int]]]:
    """A random tree of *n* molecules, L(r,r,r) and R(l,l) in turn along every
    branch, as each molecule's kind and the molecules it is bonded to."""
    tree: list[tuple[str, list[int]]] = [("L", [])]
    while len(tree) < n:
        at = rng.randrange(len(tree))
        kind, bonded = tree[at]
        if len(bonded) < (3 if kind == "L" else 2):
            bonded.append(len(tree))
            tree.append(("R" if kind == "L" else "L", [at]))
    return tree


def _tree_name(tree: list[tuple[str, list[int]]], rng: random.Random) -> str:
    """One spelling of *tree*: its molecules in a random order, each bond on
    any of the molecule's sites, under a random label."""
    order = rng.sample(range(len(tree)), len(tree))
    bonds = sorted({(min(i, j), max(i, j)) for i, (_, bonded) in enumerate(tree) for j in bonded})
    label = dict(zip(bonds, rng.sample(range(1, 10 * len(tree)), len(bonds)), strict=True))
    molecules = []
    for i in order:
        kind, bonded = tree[i]
        site = "r" if kind == "L" else "l"
        sites = [f"{site}!{label[min(i, j), max(i, j)]}" for j in bonded]
        sites += [site] * ((3 if kind == "L" else 2) - len(bonded))
        rng.shuffle(sites)
        molecules.append(f"{kind}({','.join(sites)})")
    return ".".join(molecules)


def _tree_code(tree: list[tuple[str, list[int]]]) -> str:
    """A code two trees share exactly when they are one tree (Aho, Hopcroft
    and Ullman): a molecule's code is its kind and its branches' codes in
    order, and the tree's is that of its centre, the smaller of two."""
    degree = [len(bonded) for _, bonded in tree]
    outer = [i for i, d in enumerate(degree) if d <= 1]
    left = len(tree)
    while left > 2:  # strip the leaves until the centre is left
        left -= len(outer)
        inner = []
        for i in outer:
            for j in tree[i][1]:
                degree[j] -= 1
                if degree[j] == 1:
                    inner.append(j)
        outer = inner

    def code(i: int, parent: int | None) -> str:
        return f"{tree[i][0]}({''.join(sorted(code(j, i) for j in tree[i][1] if j != parent))})"

    return min(code(centre, None) for centre in outer)


def test_a_branched_species_whose_sites_share_a_name_is_matched():
    # A tree of 300 molecules: at every fork the branches hang on sites of one
    # name, so a search that places a node wherever its label fits can take a
    # wrong branch at each of them. Oracle for the species told apart: the
    # tree code, which reads neither a site nor a label.
    rng = random.Random(300)
    for _ in range(3):
        tree = _tree(300, rng)
        a, b = parse(_tree_name(tree, rng)), parse(_tree_name(tree, rng))
        assert a is not None and b is not None
        assert same_species(a, b) and same_species(b, a)
        # One leaf moved to another molecule: the same molecules, sites and
        # bonds in every count, and another tree unless the code says not.
        told_apart = 0
        for _ in range(4):
            moved = [(kind, list(bonded)) for kind, bonded in tree]
            leaf = rng.choice([i for i, (_, bonded) in enumerate(moved) if len(bonded) == 1 and i])
            (old,) = moved[leaf][1]
            new = rng.choice(
                [
                    i
                    for i, (kind, bonded) in enumerate(moved)
                    if kind == moved[old][0]
                    and i != old
                    and len(bonded) < (3 if kind == "L" else 2)
                ]
            )
            moved[old][1].remove(leaf)
            moved[new][1].append(leaf)
            moved[leaf][1][0] = new
            c = parse(_tree_name(moved, rng))
            assert c is not None and c.key == a.key
            same = _tree_code(moved) == _tree_code(tree)
            assert same_species(a, c) == same and same_species(c, b) == same
            told_apart += not same
        assert told_apart  # the oracle is not always "the same"


def _scrambled(name: str, rng: random.Random) -> str:
    """*name*, a network's own, with its molecules and each one's components
    in another order, its bonds under other labels, and its compartment
    prefix with one colon."""
    prefix, body = re.fullmatch(r"(?:@(\w+)::)?(.*)", name).groups()
    labels = sorted(set(re.findall(r"!(\w+)", body)))
    new = dict(zip(labels, rng.sample(range(100, 200), len(labels)), strict=True))
    molecules = []
    for molecule in body.split("."):
        mol, inside, own = re.fullmatch(r"(\w+)\((.*)\)(@\w+)?", molecule).groups()
        components = inside.split(",") if inside else []
        rng.shuffle(components)
        components = [re.sub(r"!(\w+)", lambda b: f"!{new[b.group(1)]}", c) for c in components]
        molecules.append(f"{mol}({','.join(components)}){own or ''}")
    rng.shuffle(molecules)
    return (f"@{prefix}:" if prefix else "") + ".".join(molecules)


def test_every_species_of_a_real_network_is_found_by_a_scrambled_spelling():
    # Compartments, complexes across compartments, components with two bonds.
    # Oracle: the network's own name, since a network holds each species once.
    net = Path(__file__).resolve().parents[2] / "benchmarks/models/net/ode"
    names = bngsim.Model.from_net(str(net / "Motivating_example_cBNGL.net")).species_names
    assert any("!1!2" in n for n in names) and any(n.startswith("@") for n in names)
    index = SpeciesIndex(names)
    rng = random.Random(1833)
    for name in names:
        assert index.find(name) == name
        assert index.find(_scrambled(name, rng)) == name, name


# The five connected graphs of eight vertices with three edges at each, found
# by trying every renumbering; the triangles and four-cycles each has are
# beside it, and no two have the same counts.
CUBIC = {
    (1, 3): [
        (0, 2),
        (0, 3),
        (0, 7),
        (1, 2),
        (1, 4),
        (1, 5),
        (2, 7),
        (3, 4),
        (3, 5),
        (4, 6),
        (5, 6),
        (6, 7),
    ],
    (2, 2): [
        (0, 4),
        (0, 6),
        (0, 7),
        (1, 2),
        (1, 4),
        (1, 7),
        (2, 3),
        (2, 5),
        (3, 5),
        (3, 6),
        (4, 5),
        (6, 7),
    ],
    (4, 2): [
        (0, 2),
        (0, 4),
        (0, 6),
        (1, 3),
        (1, 4),
        (1, 7),
        (2, 5),
        (2, 6),
        (3, 4),
        (3, 7),
        (5, 6),
        (5, 7),
    ],
    (0, 4): [
        (0, 1),
        (0, 4),
        (0, 5),
        (1, 6),
        (1, 7),
        (2, 3),
        (2, 4),
        (2, 7),
        (3, 5),
        (3, 6),
        (4, 6),
        (5, 7),
    ],
    (0, 6): [
        (0, 5),
        (0, 6),
        (0, 7),
        (1, 2),
        (1, 6),
        (1, 7),
        (2, 3),
        (2, 4),
        (3, 5),
        (3, 7),
        (4, 5),
        (4, 6),
    ],
}


def _cycles(edges) -> tuple[int, int]:
    """How many triangles and four-cycles a graph on eight vertices has."""
    joined = {i: set() for i in range(8)}
    for i, j in edges:
        joined[i].add(j)
        joined[j].add(i)
    triangles = sum(
        b in joined[a] and c in joined[a] and c in joined[b]
        for a, b, c in itertools.combinations(range(8), 3)
    )
    squares = sum(
        a < min(b, c, d)
        and b < d
        and b in joined[a]
        and c in joined[b]
        and d in joined[c]
        # each four-cycle once: from its least vertex, in one direction
        and a in joined[d]
        for a, b, c, d in itertools.permutations(range(8), 4)
    )
    return triangles, squares


def _eight(edges) -> str:
    """Eight molecules M(s,s,s), bonded along *edges*."""
    sites: list[list[str]] = [[] for _ in range(8)]
    for label, (i, j) in enumerate(edges, 1):
        sites[i].append(f"s!{label}")
        sites[j].append(f"s!{label}")
    return ".".join(f"M({','.join(s)})" for s in sites)


def test_eight_molecules_bonded_five_ways_are_five_species():
    # Every molecule and component is alike in all five, so only the bonds
    # tell the five apart. Oracle: the counts of triangles and four-cycles,
    # which no renumbering changes and which differ between any two of the five.
    for counts, edges in CUBIC.items():
        assert _cycles(edges) == counts
    rng = random.Random(8)
    graphs = {counts: parse(_eight(edges)) for counts, edges in CUBIC.items()}
    assert len({g.key for g in graphs.values()}) == 1  # alike in every count
    for (one, a), (two, edges) in itertools.product(graphs.items(), CUBIC.items()):
        for _ in range(4):
            number = rng.sample(range(8), 8)
            again = [(number[i], number[j]) for i, j in edges]
            rng.shuffle(again)
            assert same_species(a, parse(_eight(again))) == (one == two), (one, two)


# --- BNG2.pl itself, where it is installed ----------------------------------

# BNG2.pl as bngsim finds it (BNGPATH, or an installed BioNetGen).
_FOUND = find_bng2()
BNG2 = Path(_FOUND).resolve() if _FOUND else None
HAS_BNG2 = BNG2 is not None and BNG2.is_file() and bool(shutil.which("perl"))


def _bng2pl(tmp_path: Path, text: str) -> subprocess.CompletedProcess:
    (tmp_path / "m.bngl").write_text(text, encoding="utf-8")
    return subprocess.run(
        ["perl", str(BNG2), "m.bngl"],
        cwd=tmp_path,
        # BNG2.pl finds run_network through BNGPATH, from that directory too.
        env={**os.environ, "BNGPATH": str(BNG2.parent)},
        capture_output=True,
        text=True,
        timeout=300,
    )


def _amounts(net: Path) -> dict[str, str]:
    text = net.read_text(encoding="utf-8")
    block = text[text.index("begin species") : text.index("end species")].splitlines()[1:]
    return {line.split()[1]: line.split()[2] for line in block}


@pytest.mark.skipif(not HAS_BNG2, reason="BNG2.pl not found")
@pytest.mark.parametrize("which", list(BNGL))
def test_bng2pl_sets_what_the_table_says_it_sets(tmp_path, which):
    # Every spelling the table says BNG2.pl takes, in one run: each set to a
    # number of its own, and the network written after each.
    taken = [(s, want) for s, want in ROWS[which] if want]
    actions = "".join(
        f'setConcentration("{s}", {1000 + i})\nwriteNET({{prefix=>"after{i}"}})\n'
        for i, (s, _) in enumerate(taken)
    )
    done = _bng2pl(tmp_path, BNGL[which] + actions)
    assert done.returncode == 0, done.stdout[-2000:]
    assert list(_amounts(tmp_path / "m.net")) == _species_block(NET[which])
    for i, (spelling, want) in enumerate(taken):
        after = _amounts(tmp_path / f"after{i}.net")
        assert after[want] == str(1000 + i), spelling


ABORTS = [(m, s) for m, rows in ROWS.items() for s, want in rows if not want]


@pytest.mark.skipif(not HAS_BNG2, reason="BNG2.pl not found")
@pytest.mark.parametrize(("which", "spelling"), ABORTS, ids=[repr(s) for _, s in ABORTS])
def test_bng2pl_aborts_where_the_table_says_it_aborts(tmp_path, which, spelling):
    # One run of BNG2.pl for each: an abort ends the run. The network it wrote
    # before the action is the table's, so the abort is the action's own, and
    # nothing is written after it.
    action = f'setConcentration("{spelling}", 777)\nwriteNET({{prefix=>"after"}})\n'
    done = _bng2pl(tmp_path, BNGL[which] + action)
    assert done.returncode != 0 and "ABORT" in done.stdout + done.stderr
    assert list(_amounts(tmp_path / "m.net")) == _species_block(NET[which])
    assert not (tmp_path / "after.net").exists()


@pytest.mark.skipif(not HAS_BNG2, reason="BNG2.pl not found")
@pytest.mark.parametrize(("spelling", "want", "taken"), HAND_ROWS)
def test_bng2pl_on_the_network_written_by_hand(tmp_path, spelling, want, taken):
    # BNG2.pl writes the network back with a prefix before every name.
    prefixed = {
        "L(r)@EC": "@EC::L(r)",
        "B()@CP": "@CP::B()",
        "L(r!1)@EC.R(l!1)@PM": "@PM::L(r!1)@EC.R(l!1)",
        "A(b!1)@CP.C(a!1)@CP": "@CP::A(b!1).C(a!1)",
    }
    _hand_written(tmp_path)
    (tmp_path / "run.bngl").write_text(
        f'readFile({{file=>"hand.net"}})\nsetConcentration("{spelling}", 777)\n'
        'writeNET({prefix=>"after"})\n',
        encoding="utf-8",
    )
    done = subprocess.run(
        ["perl", str(BNG2), "run.bngl"],
        cwd=tmp_path,
        env={**os.environ, "BNGPATH": str(BNG2.parent)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    if not taken:
        assert done.returncode != 0 and "ABORT" in done.stdout + done.stderr
        assert not (tmp_path / "after.net").exists()
        return
    assert done.returncode == 0, done.stdout[-2000:]
    after = _amounts(tmp_path / "after.net")
    (holds,) = [name for name, amount in after.items() if amount == "777"]
    assert sorted(after) == sorted(prefixed.values())
    # Where bngsim finds a species it is the one BNG2.pl set.
    assert want is None or prefixed[want] == holds


# --- #1480's own case, end to end through run_bngl ---------------------------

AB_MODEL = """\
begin model
begin parameters
  kf 1
  kr 0.5
end parameters
begin molecule types
  A(b)
  B(a)
end molecule types
begin seed species
  A(b) 10
  B(a) 8
end seed species
begin observables
  Species AB A(b!1).B(a!1)
end observables
begin reaction rules
  A(b) + B(a) <-> A(b!1).B(a!1) kf, kr
end reaction rules
end model
generate_network({overwrite=>1})
"""


@pytest.mark.skipif(not HAS_BNG2, reason="BNG2.pl not found")
@pytest.mark.parametrize(
    "action",
    [
        'setConcentration("B(a!1).A(b!1)", 4)',
        'setConcentration("A(b!2).B(a!2)", 4)',
        'addConcentration("B(a!1).A(b!1)", 4)',
        'setConcentration("A(b!1).B(a!1)", 4)',
    ],
)
def test_run_bngl_applies_an_action_on_a_complex_written_another_way(tmp_path, action):
    # The repro on #1480: these stopped the run ("the network has no species"),
    # while BNG2.pl 2.9.3 applies each to the network's A(b!1).B(a!1); the last
    # is the network's own spelling, the control. Oracle, closed form: from
    # A = 10, B = 8 with AB set to 4, A + AB = 14 and B + AB = 12, so
    # dAB/dt = (AB - 10.5)(AB - 16) and AB(t) = (16 - 10.5 E)/(1 - E) with
    # E = (24/13) exp(5.5 t). Tolerance 1e-6 relative: the run's own error at
    # its default tolerances.
    path = tmp_path / "m.bngl"
    path.write_text(AB_MODEL + action + '\nsimulate({method=>"ode",t_end=>0.2,n_steps=>2})\n')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = bngsim.run_bngl(str(path), cache=False)
    e = 24 / 13 * math.exp(5.5 * 0.2)
    exact = (16 - 10.5 * e) / (1 - e)
    assert float(out[0].result.observables["AB"][-1]) == pytest.approx(exact, rel=1e-6)
