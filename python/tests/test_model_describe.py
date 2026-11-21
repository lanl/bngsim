"""``Model.describe()``: what a model is, in one call (issue #1705).

Each fact the description states is checked against a source other than the
description: the file's bytes for its hash, the first row of a run for the
initial values, ``Result.as_roadrunner`` for an amount, and the Simulator's own
refusal for each method verdict, raised from the same check (``ode_refusals``,
``validate_for_ssa``) or, for the two SSA refusals made in C++, from the engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import bngsim
import pytest

# ─── Models ──────────────────────────────────────────────────────────────────

# S + E -> P + E by Michaelis-Menten, P -> S at 2*kf, a fixed $C, a derived
# parameter, and an observable over two species with a coefficient.
NET = (
    "begin parameters\n"
    " 1 kcat 1.0\n 2 Km 0.5\n 3 kf 2\n 4 kr 2*kf\n"
    "end parameters\n"
    "begin species\n 1 S() 10\n 2 E() 1\n 3 P() 0\n 4 $C() 5\nend species\n"
    "begin reactions\n 1 1,2 2,3 MM kcat Km\n 2 3 1 2*kf\n 3 4 3 kr\nend reactions\n"
    "begin groups\n 1 St 1,2*3\n 2 Ctot 4\nend groups\n"
)

_SBML_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">\n'
    ' <model id="m">\n'
)
_SBML_TAIL = " </model>\n</sbml>\n"


def _ci(name: str) -> str:
    return f"<ci>{name}</ci>"


def _math(body: str) -> str:
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML">{body}</math>'


def _reaction(rid: str, reactant: str, product: str, law: str, *, stoich: str = "1") -> str:
    return (
        f'   <reaction id="{rid}" reversible="false">\n'
        f'    <listOfReactants><speciesReference species="{reactant}" stoichiometry="{stoich}" '
        'constant="true"/></listOfReactants>\n'
        f'    <listOfProducts><speciesReference species="{product}" stoichiometry="1" '
        'constant="true"/></listOfProducts>\n'
        f"    <kineticLaw>{_math(law)}</kineticLaw>\n"
        "   </reaction>\n"
    )


# A in a compartment of size 2 by concentration, B in one of size 0.5 by amount
# (initial amount 3, so a concentration of 6), C a boundary species.
TWO_COMPARTMENTS = (
    _SBML_HEAD + "  <listOfCompartments>\n"
    '   <compartment id="cell" size="2" constant="true" spatialDimensions="3"/>\n'
    '   <compartment id="nuc" size="0.5" constant="true" spatialDimensions="3"/>\n'
    "  </listOfCompartments>\n"
    "  <listOfSpecies>\n"
    '   <species id="A" compartment="cell" initialConcentration="10" '
    'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>\n'
    '   <species id="B" compartment="nuc" initialAmount="3" '
    'hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>\n'
    '   <species id="C" compartment="cell" initialConcentration="1" '
    'hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>\n'
    "  </listOfSpecies>\n"
    "  <listOfParameters>\n"
    '   <parameter id="k1" value="1" constant="true"/>\n'
    '   <parameter id="k2" value="0.1" constant="true"/>\n'
    "  </listOfParameters>\n"
    "  <listOfReactions>\n"
    + _reaction("R1", "A", "B", f"<apply><times/>{_ci('k1')}{_ci('A')}{_ci('cell')}</apply>")
    + _reaction("R2", "B", "A", f"<apply><times/>{_ci('k2')}{_ci('B')}{_ci('C')}</apply>")
    + "  </listOfReactions>\n"
    + _SBML_TAIL
)


def _one_compartment(species: str, reactions: str, extra: str = "") -> str:
    return (
        _SBML_HEAD + "  <listOfCompartments>\n"
        '   <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>\n'
        "  </listOfCompartments>\n"
        f"  <listOfSpecies>\n{species}  </listOfSpecies>\n"
        "  <listOfParameters>\n"
        '   <parameter id="k" value="0.1" constant="true"/>\n'
        "  </listOfParameters>\n"
        f"  <listOfReactions>\n{reactions}  </listOfReactions>\n"
        f"{extra}" + _SBML_TAIL
    )


def _species_xml(sid: str, amount: float) -> str:
    return (
        f'   <species id="{sid}" compartment="cell" initialAmount="{amount}" '
        'hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>\n'
    )


_AB = _species_xml("A", 100) + _species_xml("B", 0)
_DECAY = _reaction("J", "A", "B", f"<apply><times/>{_ci('k')}{_ci('A')}</apply>")

# A 1.5-fold reactant: SSA fires whole molecules, so it refuses (and ODE runs).
NON_INTEGER_STOICH = _one_compartment(
    _AB, _reaction("J", "A", "B", f"<apply><times/>{_ci('k')}{_ci('A')}</apply>", stoich="1.5")
)

# A rate that reads rateOf(A): ODE runs it; SSA refuses it in C++ (GH #106).
_RATE_OF = (
    '<apply><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/rateOf">'
    f"rateOf</csymbol>{_ci('A')}</apply>"
)
READS_RATE_OF = _one_compartment(
    _AB + _species_xml("C", 0),
    _DECAY
    + _reaction(
        "J2", "B", "C", f"<apply><times/>{_ci('k')}<apply><minus/>{_RATE_OF}</apply></apply>"
    ),
)

# An event at t >= 2 that sets B to 7, and a rate that reads the clock.
WITH_EVENT = _one_compartment(
    _AB,
    _reaction(
        "J",
        "A",
        "B",
        f'<apply><times/>{_ci("k")}{_ci("A")}<csymbol encoding="text" '
        'definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol></apply>',
    ),
    "  <listOfEvents>\n"
    '   <event id="bump" useValuesFromTriggerTime="true">\n'
    '    <trigger initialValue="false" persistent="true">'
    + _math(
        '<apply><geq/><csymbol encoding="text" '
        'definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>2</cn></apply>'
    )
    + "</trigger>\n"
    '    <listOfEventAssignments><eventAssignment variable="B">'
    + _math("<cn>7</cn>")
    + "</eventAssignment></listOfEventAssignments>\n"
    "   </event>\n"
    "  </listOfEvents>\n",
)

# fast="true" is an SBML L2 attribute.
FAST_REACTION = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="fast_rxn">
    <listOfCompartments><compartment id="cell" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialAmount="10"/>
      <species id="B" compartment="cell" initialAmount="0"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.1"/></listOfParameters>
    <listOfReactions>
      <reaction id="rfast" reversible="false" fast="true">
        <listOfReactants><speciesReference species="A" stoichiometry="1"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


def _net(tmp_path: Path, text: str = NET, name: str = "m.net") -> tuple[bngsim.Model, Path]:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return bngsim.Model.from_net(str(path)), path


def _chain(tmp_path: Path, n: int) -> bngsim.Model:
    """S1 -> S2 -> ... -> Sn, and one observable over every species."""
    species = "".join(f" {i} S{i}() {10 if i == 1 else 0}\n" for i in range(1, n + 1))
    reactions = "".join(f" {i} {i} {i + 1} k\n" for i in range(1, n))
    text = (
        "begin parameters\n 1 k 1\nend parameters\n"
        f"begin species\n{species}end species\n"
        f"begin reactions\n{reactions}end reactions\n"
        f"begin groups\n 1 Total {','.join(str(i) for i in range(1, n + 1))}\nend groups\n"
    )
    return _net(tmp_path, text, "chain.net")[0]


def _first_row(model: bngsim.Model):
    result = bngsim.Simulator(model.clone(), method="ode").run(t_span=(0, 1e-9), n_points=2)
    return result


# ─── Shape ───────────────────────────────────────────────────────────────────


def test_it_is_a_dict_json_writes_and_print_reads(tmp_path):
    model, _ = _net(tmp_path)
    d = model.describe()
    assert isinstance(d, dict)
    assert json.loads(json.dumps(d)) == d
    text = str(d)
    assert text.splitlines()[0].startswith("Model from net: ")
    assert "Methods:" in text and "Reactions (3):" in text
    assert "Species values are as the net file gives them; it states no units." in text
    # And nothing else about it is unlike a dict: repr is a dict's.
    assert repr(d) == repr(dict(d))


def test_the_schema_version_and_keys(tmp_path):
    model, _ = _net(tmp_path)
    d = model.describe()
    assert d["schema_version"] == 1
    for key in (
        "source",
        "counts",
        "next_run_starts_from",
        "species_values",
        "methods",
        "time_dependence",
        "jacobian",
        "parameters",
        "species",
        "observables",
        "functions",
        "reactions",
        "events",
    ):
        assert key in d, key
    assert set(d["methods"]) == {
        "ode",
        "ssa",
        "psa",
        "nf",
        "rm",
        "forward_sensitivities",
        "steady_state",
    }


# ─── A .net model ────────────────────────────────────────────────────────────


def test_a_net_model_says_where_it_came_from(tmp_path):
    model, path = _net(tmp_path)
    src = model.describe()["source"]
    assert src == {
        "format": "net",
        "path": os.path.abspath(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_a_net_model_counts_and_lists(tmp_path):
    model, _ = _net(tmp_path)
    d = model.describe()
    assert d["counts"] == {
        "species": 4,
        "reactions": 3,
        "observables": 2,
        "parameters": 4,
        "functions": 0,
        "events": 0,
        "table_functions": 0,
    }
    # A .net file states no units and has no compartments: absent, not guessed.
    assert d["species_values"] is None
    assert "compartments" not in d
    assert all("quantity" not in s and "compartment" not in s for s in d["species"])

    params = {p["name"]: p for p in d["parameters"]}
    assert params["kf"] == {"name": "kf", "value": 2.0, "kind": "primary"}
    assert params["kr"] == {"name": "kr", "value": 4.0, "kind": "derived", "expression": "2*kf"}

    row = _first_row(model)
    species = {s["name"]: s for s in d["species"]}
    assert [s["name"] for s in d["species"]] == list(row.species_names)
    for i, name in enumerate(row.species_names):
        assert species[name]["initial"] == row.species[0][i]
    assert species["C()"]["fixed"] is True
    assert species["S()"]["fixed"] is False

    assert d["observables"] == [
        {"name": "St", "reads": "S() + 2*P()"},
        {"name": "Ctot", "reads": "C()"},
    ]
    assert d["reactions"] == [
        {
            "reactants": ["S()", "E()"],
            "products": ["E()", "P()"],
            "equation": "S() + E() -> E() + P()",
            "rate_law": "michaelis_menten",
            "rate": "kcat, Km",
        },
        {
            "reactants": ["P()"],
            "products": ["S()"],
            "equation": "P() -> S()",
            "rate_law": "elementary",
            "rate": "kf",
            "multiplier": 2.0,
        },
        {
            "reactants": ["C()"],
            "products": ["P()"],
            "equation": "C() -> P()",
            "rate_law": "elementary",
            "rate": "kr",
        },
    ]


def test_a_repeated_reactant_is_counted_in_the_equation(tmp_path):
    text = NET.replace(" 3 4 3 kr\n", " 3 3,3 1 kr\n")
    model, _ = _net(tmp_path, text)
    third = model.describe()["reactions"][2]
    assert third["reactants"] == ["P()"]
    assert third["equation"] == "2*P() -> S()"


def test_a_derived_parameter_set_to_a_number_says_so(tmp_path):
    """set_param pins a derived parameter to the number (issue #188): it no
    longer follows its expression, which stays on record."""
    model, _ = _net(tmp_path)
    model.set_param("kr", 9.0)
    kr = next(p for p in model.describe()["parameters"] if p["name"] == "kr")
    assert kr == {
        "name": "kr",
        "value": 9.0,
        "kind": "derived",
        "expression": "2*kf",
        "overridden": True,
    }


# ─── An SBML model ───────────────────────────────────────────────────────────


def test_an_sbml_file_says_where_it_came_from(tmp_path):
    path = tmp_path / "two.xml"
    path.write_text(TWO_COMPARTMENTS, encoding="utf-8")
    src = bngsim.Model.from_sbml(str(path)).describe()["source"]
    assert src == {
        "format": "sbml",
        "path": os.path.abspath(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    # Model.load routes there too.
    assert bngsim.Model.load(path).describe()["source"] == src


def test_an_sbml_string_is_hashed_as_utf8():
    src = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS).describe()["source"]
    assert src == {
        "format": "sbml",
        "path": None,
        "sha256": hashlib.sha256(TWO_COMPARTMENTS.encode("utf-8")).hexdigest(),
    }


def test_compartments_and_what_each_species_is():
    model = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS)
    d = model.describe()
    assert d["species_values"] == "concentration"
    assert d["counts"]["compartments"] == 2
    assert "Species values are concentrations (amount / compartment size)" in str(d)
    assert d["compartments"] == [
        {"name": "cell", "size": 2.0, "size_declared": True, "size_set_by": []},
        {"name": "nuc", "size": 0.5, "size_declared": True, "size_set_by": []},
    ]
    species = {s["name"]: s for s in d["species"]}
    assert species["A"] == {
        "name": "A",
        "initial": 10.0,
        "fixed": False,
        "compartment": "cell",
        "quantity": "concentration",
    }
    assert species["B"]["compartment"] == "nuc"
    assert species["B"]["quantity"] == "amount"
    assert species["C"]["fixed"] is True

    # The numbers are what a run reports: the species column holds the
    # concentration, 3 / 0.5 for B, and the amount is its bare-id selector.
    row = _first_row(model)
    for i, name in enumerate(row.species_names):
        assert species[name]["initial"] == pytest.approx(row.species[0][i], rel=1e-12)
    assert species["B"]["initial"] == pytest.approx(6.0)
    amount = row.as_roadrunner(["B"])[0, 0]
    assert species["B"]["initial_amount"] == pytest.approx(amount, rel=1e-12)
    assert species["B"]["initial_amount"] == pytest.approx(3.0)

    params = {p["name"]: p["kind"] for p in d["parameters"]}
    assert params == {
        "cell": "compartment_size",
        "nuc": "compartment_size",
        "k1": "primary",
        "k2": "primary",
    }


def test_compartment_sizes_set_at_load_are_recorded():
    model = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS, compartment_sizes={"cell": 4.0})
    d = model.describe()
    assert d["source"]["compartment_sizes"] == {"cell": 4.0}
    assert d["compartments"][0] == {
        "name": "cell",
        "size": 4.0,
        "size_declared": True,
        "size_set_by": [],
    }


def test_compartment_sizes_set_at_a_file_load_are_recorded(tmp_path):
    path = tmp_path / "two.xml"
    path.write_text(TWO_COMPARTMENTS, encoding="utf-8")
    d = bngsim.Model.from_sbml(str(path), compartment_sizes={"nuc": 0.25}).describe()
    assert d["source"]["compartment_sizes"] == {"nuc": 0.25}
    assert d["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert d["compartments"][1]["size"] == 0.25
    assert "compartment sizes set at load: nuc=0.25" in str(d)


def test_a_compartment_with_no_size_says_so():
    text = TWO_COMPARTMENTS.replace(' size="0.5"', "")
    d = bngsim.Model.from_sbml_string(text).describe()
    nuc = next(c for c in d["compartments"] if c["name"] == "nuc")
    assert nuc == {"name": "nuc", "size": 1.0, "size_declared": False, "size_set_by": []}
    assert "(not given in the file; 1 used)" in str(d)


def test_events_and_the_clock():
    d = bngsim.Model.from_sbml_string(WITH_EVENT).describe()
    assert d["counts"]["events"] == 1
    (event,) = d["events"]
    assert event["id"] == "bump"
    assert event["assigns"] == ["B"]
    assert event["trigger"] == "(time()>=2.0)"
    assert "delay" not in event
    assert d["time_dependence"]["rate_laws_read_clock"] is True
    assert d["time_dependence"]["events"] == 1
    species = {s["name"]: s for s in d["species"]}
    assert species["B"]["set_by"] == ["event"]
    assert "set_by" not in species["A"]


def test_a_model_whose_rates_do_not_read_the_clock(tmp_path):
    t = _net(tmp_path)[0].describe()["time_dependence"]
    assert t == {
        "rate_laws_read_clock": False,
        "functions_reading_clock": [],
        "state_reads_clock": False,
        "time_tables": [],
        "events": 0,
    }


# ─── Methods: each verdict against the Simulator's own ───────────────────────


def test_a_clean_model_admits_ode_ssa_and_psa(tmp_path):
    model, _ = _net(tmp_path)
    m = model.describe()["methods"]
    for method in ("ode", "ssa", "psa"):
        assert m[method]["status"] == "ok", method
        assert m[method]["reasons"] == []
    # And the runs agree.
    bngsim.Simulator(model.clone(), method="ode").run(t_span=(0, 1), n_points=2)
    bngsim.Simulator(model.clone(), method="ssa").run(t_span=(0, 1), n_points=2, seed=1)


def test_a_fast_reaction_refuses_ode_with_the_simulators_message(monkeypatch):
    monkeypatch.delenv("BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS", raising=False)
    model = bngsim.Model.from_sbml_string(FAST_REACTION)
    ode = model.describe()["methods"]["ode"]
    assert ode["status"] == "refused"
    assert [r["code"] for r in ode["reasons"]] == ["fast_reaction"]
    with pytest.raises(bngsim.ModelError) as ei:
        bngsim.Simulator(model, method="ode")
    assert str(ei.value) == ode["reasons"][0]["message"]
    # SSA refuses it too, by validate_for_ssa's code.
    assert "fast_reaction" in [r["code"] for r in model.describe()["methods"]["ssa"]["reasons"]]


def test_the_fast_reaction_override_is_read_as_the_simulator_reads_it(monkeypatch):
    monkeypatch.setenv("BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS", "1")
    model = bngsim.Model.from_sbml_string(FAST_REACTION)
    assert model.describe()["methods"]["ode"]["status"] == "ok"
    bngsim.Simulator(model, method="ode")  # does not raise


def test_a_ring_of_rates_refuses_ode():
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.enable_rateof()
    x = b.add_species("X", 0.0)
    y = b.add_species("Y", 0.0)
    b.add_function("fX", "0.5*rate_of__Y + 1")
    b.add_function("fY", "0.5*rate_of__X + 1")
    b.add_reaction([], [x], "functional", "fX")
    b.add_reaction([], [y], "functional", "fY")
    model = bngsim.Model(_core=b.build())
    ode = model.describe()["methods"]["ode"]
    assert ode["status"] == "refused"
    assert [r["code"] for r in ode["reasons"]] == ["rateof_ring"]
    with pytest.raises(bngsim.ModelError) as ei:
        bngsim.Simulator(model, method="ode")
    assert str(ei.value) == ode["reasons"][0]["message"]


def test_non_integer_stoichiometry_refuses_ssa_and_psa_not_ode():
    model = bngsim.Model.from_sbml_string(NON_INTEGER_STOICH)
    m = model.describe()["methods"]
    assert m["ode"]["status"] == "ok"
    for method in ("ssa", "psa"):
        assert m[method]["status"] == "refused"
        assert [r["code"] for r in m[method]["reasons"]] == ["non_integer_stoichiometry"]
        assert m[method]["reasons"][0]["count"] == 1
        assert m[method]["reasons"][0]["locations"] == ["reaction:J"]
        with pytest.raises(bngsim.SsaValidationError) as ei:
            bngsim.Simulator(
                model.clone(), method=method, poplevel=10 if method == "psa" else None
            )
        assert "non_integer_stoichiometry" in str(ei.value)


def test_rate_of_refuses_ssa_as_the_engine_does():
    """The engine raises it in C++, at the run: validate_for_ssa does not see it."""
    model = bngsim.Model.from_sbml_string(READS_RATE_OF)
    assert not [i for i in model.validate_for_ssa() if i.severity == "error"]
    m = model.describe()["methods"]
    assert m["ode"]["status"] == "ok"
    for method in ("ssa", "psa"):
        assert m[method]["status"] == "refused"
        (reason,) = m[method]["reasons"]
        assert reason["code"] == "rateof"
        sim = bngsim.Simulator(
            model.clone(), method=method, poplevel=10 if method == "psa" else None
        )
        with pytest.raises(Exception) as ei:
            sim.run(t_span=(0, 1), n_points=2, seed=1)
        assert reason["message"] in str(ei.value)


def test_a_model_with_no_species_refuses_ssa_as_the_engine_does():
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    model = bngsim.Model(_core=b.build())
    ssa = model.describe()["methods"]["ssa"]
    assert ssa["status"] == "refused"
    (reason,) = ssa["reasons"]
    assert reason["code"] == "no_species"
    with pytest.raises(Exception) as ei:
        bngsim.Simulator(model, method="ssa").run(t_span=(0, 1), n_points=2, seed=1)
    assert reason["message"] in str(ei.value)


def test_the_network_free_methods_need_a_bng_xml_file(tmp_path):
    for model in (_net(tmp_path)[0], bngsim.Model.from_sbml_string(TWO_COMPARTMENTS)):
        m = model.describe()["methods"]
        for method in ("nf", "rm"):
            assert m[method]["status"] in ("needs_input", "unavailable")
            if m[method]["status"] == "needs_input":
                assert [r["code"] for r in m[method]["reasons"]] == ["needs_bng_xml"]
                assert "xml_path=" in m[method]["reasons"][0]["message"]
                # The Simulator asks for the same thing.
                with pytest.raises(ValueError, match="xml_path"):
                    bngsim.Simulator(model.clone(), method=method)


def test_forward_sensitivities_with_codegen_switched_off(tmp_path, monkeypatch):
    model, _ = _net(tmp_path)
    monkeypatch.setenv("BNGSIM_NO_CODEGEN", "1")
    fs = model.describe()["methods"]["forward_sensitivities"]
    assert fs["status"] == "unavailable"
    assert [r["code"] for r in fs["reasons"]] == ["codegen_disabled"]
    with pytest.raises(ValueError, match="BNGSIM_NO_CODEGEN"):
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf"])
        sim.run(t_span=(0, 1), n_points=2)


def test_forward_sensitivities_with_no_compiler(tmp_path, monkeypatch):
    from bngsim import _codegen_toolchain

    model, _ = _net(tmp_path)
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
    monkeypatch.delenv("BNGSIM_CODEGEN_JIT", raising=False)
    monkeypatch.setenv("CC", str(tmp_path / "no-such-compiler"))
    monkeypatch.setattr(_codegen_toolchain, "_verdicts", {})
    monkeypatch.setattr(_codegen_toolchain, "_warned", set())
    fs = model.describe()["methods"]["forward_sensitivities"]
    assert fs["status"] == "unavailable"
    assert [r["code"] for r in fs["reasons"]] == ["no_compiler"]
    assert "no-such-compiler" in fs["reasons"][0]["message"]


def test_forward_sensitivities_with_a_compiler_found_broken(tmp_path, monkeypatch):
    """A toolchain a probe found unable to build is reported with its verdict."""
    from bngsim import _codegen_toolchain

    model, _ = _net(tmp_path)
    monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)
    monkeypatch.delenv("BNGSIM_CODEGEN_JIT", raising=False)
    monkeypatch.delenv("CC", raising=False)
    from bngsim._codegen import _find_c_compiler

    compiler = _find_c_compiler()
    if _codegen_toolchain.shutil.which(compiler[0]) is None:
        pytest.skip("no C compiler: test needs a C compiler on PATH")
    verdict = "cc cannot link a shared library (test)."
    key = _codegen_toolchain.toolchain_key(compiler)
    monkeypatch.setitem(_codegen_toolchain._verdicts, key, (verdict, 0.0))
    # describe() reads the kept verdict, even one past its time, and probes nothing.
    monkeypatch.setattr(
        _codegen_toolchain, "_probe", lambda *a, **k: pytest.fail("describe() probed")
    )
    fs = model.describe()["methods"]["forward_sensitivities"]
    assert fs["status"] == "unavailable"
    assert [r["code"] for r in fs["reasons"]] == ["no_compiler"]
    assert fs["reasons"][0]["message"].startswith(verdict)
    assert "backend" not in fs


def test_forward_sensitivities_name_the_parameters_folded_at_load():
    # A compartment sized by an initialAssignment over p: p is read once, at load.
    text = TWO_COMPARTMENTS.replace(
        "  </listOfParameters>\n",
        '   <parameter id="p" value="1" constant="true"/>\n  </listOfParameters>\n'
        "  <listOfInitialAssignments>\n"
        '   <initialAssignment symbol="cell">'
        + _math(f"<apply><times/><cn>2</cn>{_ci('p')}</apply>")
        + "</initialAssignment>\n"
        "  </listOfInitialAssignments>\n",
    )
    model = bngsim.Model.from_sbml_string(text)
    assert model.frozen_params == ["p"]
    fs = model.describe()["methods"]["forward_sensitivities"]
    assert fs["refused_parameters"] == {"p": f"folded at load into {model._frozen_params['p']}"}
    p = next(e for e in model.describe()["parameters"] if e["name"] == "p")
    assert p["frozen"] == model._frozen_params["p"]
    # And a sensitivity run refuses it.
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="'p'"):
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["p"])
        sim.run(t_span=(0, 1), n_points=2)


# ─── State, size and the model left alone ────────────────────────────────────


def test_where_the_next_run_starts(tmp_path):
    model, _ = _net(tmp_path)
    assert model.describe()["next_run_starts_from"] == "initial"
    bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
    assert model.describe()["next_run_starts_from"] == "live"
    assert "which is not its initial state" in str(model.describe())
    model.reset()
    assert model.describe()["next_run_starts_from"] == "initial"
    model.set_concentration("S()", 3.0)
    assert model.describe()["next_run_starts_from"] == "live"


def test_describe_changes_nothing(tmp_path):
    model = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS)
    state = model.get_state().copy()
    params = {n: model.get_param(n) for n in model.param_names}
    status = model.analytical_jacobian_status
    model.describe(limit=None)
    str(model.describe())
    assert (model.get_state() == state).all()
    assert {n: model.get_param(n) for n in model.param_names} == params
    # It reports the Jacobian's status; it does not derive it.
    assert model.analytical_jacobian_status == status == "pending"
    assert model.describe()["jacobian"] == {"analytical": "pending"}


def test_a_long_model_is_cut_and_says_how_much(tmp_path):
    model = _chain(tmp_path, 120)
    d = model.describe(limit=10)
    assert len(d["species"]) == 10
    assert [s["name"] for s in d["species"]] == [f"S{i}()" for i in range(1, 11)]
    assert len(d["reactions"]) == 10
    assert d["omitted"] == {"species": 110, "reactions": 109}
    assert d["counts"]["species"] == 120
    (total,) = d["observables"]
    assert total["reads"].endswith("+ ... (110 more terms)")
    assert total["reads"].count("+") == 10
    text = str(d)
    assert "Species (120):" in text
    assert "... 110 more not shown (describe(limit=None) lists all)" in text

    full = model.describe(limit=None)
    assert len(full["species"]) == 120 and len(full["reactions"]) == 119
    assert "omitted" not in full
    assert "more terms" not in full["observables"][0]["reads"]
    # The default is 50.
    assert model.describe()["omitted"] == {"species": 70, "reactions": 69}


@pytest.mark.parametrize("bad", [-1, 1.5, True, "10"])
def test_a_limit_must_be_a_count(tmp_path, bad):
    model, _ = _net(tmp_path)
    with pytest.raises(ValueError, match="limit"):
        model.describe(limit=bad)


def test_a_clone_keeps_its_parents_record(tmp_path):
    model = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS)
    a, b = model.describe(), model.clone().describe()
    assert b["source"] == a["source"]
    assert b["compartments"] == a["compartments"]
    assert b["species"] == a["species"]


def test_a_model_built_in_python_has_no_source():
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    x = b.add_species("X", 2.0)
    b.add_reaction([x], [], "elementary", "k")
    d = bngsim.Model(_core=b.build()).describe()
    assert d["source"] == {"format": None, "path": None, "sha256": None}
    assert str(d).splitlines()[0] == "Model built in Python (no source file recorded)"
    assert "Species values carry no stated units." in str(d)


# ─── The other loaders ───────────────────────────────────────────────────────


def test_an_antimony_model_says_so(tmp_path):
    pytest.importorskip("antimony")
    text = "compartment c = 2; species A in c = 10; J: A -> ; k*A; k = 0.5\n"
    path = tmp_path / "m.ant"
    path.write_text(text, encoding="utf-8")
    from_file = bngsim.Model.from_antimony(str(path)).describe()
    assert from_file["source"] == {
        "format": "antimony",
        "path": os.path.abspath(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    from_text = bngsim.Model.from_antimony_string(text).describe()
    assert from_text["source"] == {
        "format": "antimony",
        "path": None,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    assert from_text["compartments"] == [
        {"name": "c", "size": 2.0, "size_declared": True, "size_set_by": []},
    ]
    assert from_text["species"][0]["compartment"] == "c"


def test_a_bngl_model_names_the_file_and_its_network(tmp_path):
    from bngsim._bngpath import resolve_bng

    bng = resolve_bng()
    if not bng.ok:
        pytest.skip(f"BNG2.pl unavailable: {bng.why_not()}")
    path = tmp_path / "m.bngl"
    path.write_text(
        "begin model\nbegin parameters\n k 1\nend parameters\n"
        "begin molecule types\n A()\nend molecule types\n"
        "begin seed species\n A() 10\nend seed species\n"
        "begin reaction rules\n A() -> 0 k\nend reaction rules\nend model\n"
        "generate_network({overwrite=>1})\n",
        encoding="utf-8",
    )
    d = bngsim.Model.from_bngl(str(path), cache=False).describe()
    assert d["source"]["format"] == "bngl"
    assert d["source"]["path"] == os.path.abspath(path)
    assert d["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert d["source"]["network"].endswith(".net")
    assert "writeXML()" in d["methods"]["nf"]["reasons"][0]["message"] or (
        d["methods"]["nf"]["status"] == "unavailable"
    )


# ─── What sets a value ───────────────────────────────────────────────────────

# c sized by an assignment rule over p, v by a rate rule; R set by a rate rule;
# q, a parameter an event sets, held as a state with no column; a delayed event.
RULES_AND_EVENTS = """
compartment c = 1, v = 2;
v' = 0.1;
species A in c = 10, B in v = 1, R in c = 0;
R' = k;
k = 0.5; q = 1; p = 3;
c := 1 + p;
J1: A -> B; k*A;
E1: at (time > 1), t0 = false: q = 5;
E2: at 0.5 after (time > 2): A = 7;
"""


def test_what_sets_a_compartment_or_a_species():
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(RULES_AND_EVENTS)
    d = model.describe()
    assert d["compartments"] == [
        {"name": "c", "size": 4.0, "size_declared": True, "size_set_by": ["assignment_rule"]},
        {"name": "v", "size": 2.0, "size_declared": True, "size_set_by": ["rate_rule"]},
    ]
    species = {s["name"]: s for s in d["species"]}
    assert list(species) == ["A", "B", "R"]  # the file's three
    assert species["A"]["set_by"] == ["event"]
    assert "set_by" not in species["B"]
    assert species["R"]["set_by"] == ["rate_rule"]
    # The rate-rule compartment and the event-set parameter are states, not
    # species of the file; only q has no column.
    assert d["other_states"] == [
        {"name": "v", "initial": 2.0, "holds": "compartment_size", "set_by": ["rate_rule"]},
        {
            "name": "q",
            "initial": 1.0,
            "holds": "parameter",
            "set_by": ["event"],
            "reported": False,
        },
    ]
    assert d["counts"]["species"] == 3 and d["counts"]["other_states"] == 2
    row = _first_row(model)
    columns = list(row.species_names)
    assert columns == ["A", "B", "R", "v"]
    # The file's parameters only (q is a state), and the file's functions only:
    # c's assignment rule and J1's rate, not the loader's _rhs_ and _dil_.
    assert [p["name"] for p in d["parameters"]] == ["k", "p"]
    assert [f["name"] for f in d["functions"]] == ["c", "J1"]
    assert d["reactions"] == [
        {
            "id": "J1",
            "reactants": ["A"],
            "products": ["B"],
            "equation": "A -> B",
            "rate_law": "kinetic_law",
            "rate": "k * A",
            "reversible": True,
        }
    ]

    events = {e["id"]: e for e in d["events"]}
    assert events["E1"] == {"id": "E1", "trigger": "(time()>1)", "assigns": ["q"]}
    assert events["E2"] == {"id": "E2", "trigger": "(time()>2)", "assigns": ["A"], "delay": 0.5}
    text = str(d)
    assert "c  size 4, set by assignment rule" in text
    assert "q  1  (a parameter held as a state; no column in Result.species; set by event)" in text
    assert "E2: when (time()>2), after 0.5 sets A" in text


def test_an_observable_with_negative_coefficients():
    from bngsim._bngsim_core import ModelBuilder

    b = ModelBuilder()
    x = b.add_species("X", 1.0)
    y = b.add_species("Y", 2.0)
    b.add_observable("diff", [(x, 1.0), (y, -2.0)])
    b.add_observable("neg", [(y, -1.0), (x, 0.5)])
    obs = {o["name"]: o["reads"] for o in bngsim.Model(_core=b.build()).describe()["observables"]}
    assert obs == {"diff": "X - 2*Y", "neg": "-Y + 0.5*X"}


def test_an_unavailable_network_free_method_says_so(tmp_path, monkeypatch):
    from bngsim import _simulator

    message = "RuleMonkey is not present in this bngsim install (test)."
    monkeypatch.setitem(_simulator._UNAVAILABLE_NF_METHODS, "nf_exact", message)
    model, path = _net(tmp_path)
    rm = model.describe()["methods"]["rm"]
    assert rm == {
        "status": "unavailable",
        "reasons": [{"code": "not_in_this_install", "message": message}],
    }
    with pytest.raises(ValueError, match="RuleMonkey is not present"):
        bngsim.Simulator(model, method="rm", xml_path=str(path))


# ─── The file as written, not the model's channels ───────────────────────────


def test_an_sbml_files_reactions_as_written():
    d = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS).describe()
    assert d["counts"]["reactions"] == 2
    assert d["reactions"][0] == {
        "id": "R1",
        "reactants": ["A"],
        "products": ["B"],
        "equation": "A -> B",
        "rate_law": "kinetic_law",
        "rate": "k1 * A * cell",
        "reversible": False,
    }
    # A non-unit stoichiometry is in the equation.
    r = bngsim.Model.from_sbml_string(NON_INTEGER_STOICH).describe()["reactions"][0]
    assert r["equation"] == "1.5*A -> B"


def test_a_reversible_law_lists_the_files_reaction_and_parameters_only():
    """The loader splits a reversible mass-action law into two channels with
    rate parameters of its own; describe() reports the file's one reaction and
    its two parameters."""
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(
        "A = 10; B = 0; J: A -> B; kf*A - kr*B; kf = 1; kr = 0.5"
    )
    assert model.n_reactions == 2  # the channels
    d = model.describe()
    assert d["counts"]["reactions"] == 1
    assert [r["id"] for r in d["reactions"]] == ["J"]
    # Antimony's default_compartment is a compartment of the file.
    assert [p["name"] for p in d["parameters"]] == ["default_compartment", "kf", "kr"]
    assert any(n.startswith("_") for n in model.param_names)  # the loader's own


def test_a_compartment_both_rate_ruled_and_resized_by_an_event():
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(
        "compartment v = 2; v' = 0.1; species B in v = 1; E: at (time > 1): v = 3;"
    )
    (v,) = model.describe()["compartments"]
    assert v["size_set_by"] == ["rate_rule", "event"]


def test_a_compartment_size_written_after_load():
    model = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS)
    model.set_param("nuc", 1.0)
    d = model.describe()
    nuc = next(c for c in d["compartments"] if c["name"] == "nuc")
    assert nuc["size"] == 1.0
    b = next(s for s in d["species"] if s["name"] == "B")
    assert b["initial"] == pytest.approx(_first_row(model).species[0][1], rel=1e-12)
    assert b["initial_amount"] == pytest.approx(3.0)


def test_a_non_finite_size_is_not_declared():
    d = bngsim.Model.from_sbml_string(
        TWO_COMPARTMENTS.replace('size="0.5"', 'size="NaN"')
    ).describe()
    nuc = next(c for c in d["compartments"] if c["name"] == "nuc")
    assert nuc["size"] == 1.0 and nuc["size_declared"] is False


def test_a_renamed_compartment_keeps_the_files_id():
    """``default`` is an ExprTk word: the model calls it ``_ant_default``."""
    text = TWO_COMPARTMENTS.replace('"cell"', '"default"').replace(
        "<ci>cell</ci>", "<ci>default</ci>"
    )
    d = bngsim.Model.from_sbml_string(text).describe()
    first = d["compartments"][0]
    assert first["name"] == "_ant_default" and first["id"] == "default"
    assert "(the file's default)" in str(d)
    # A rate reads it by the model's name too.
    assert d["reactions"][0]["rate"] == "k1 * A * _ant_default"


def test_many_ssa_refusals_of_one_kind_are_grouped():
    two = _one_compartment(
        _AB,
        _reaction("J1", "A", "B", f"<apply><times/>{_ci('k')}{_ci('A')}</apply>", stoich="1.5")
        + _reaction("J2", "A", "B", f"<apply><times/>{_ci('k')}{_ci('A')}</apply>", stoich="2.5"),
    )
    model = bngsim.Model.from_sbml_string(two)
    (reason,) = model.describe()["methods"]["ssa"]["reasons"]
    assert reason["code"] == "non_integer_stoichiometry"
    assert reason["count"] == 2
    assert reason["locations"] == ["reaction:J1", "reaction:J2"]
    (cut,) = model.describe(limit=1)["methods"]["ssa"]["reasons"]
    assert cut["locations"] == ["reaction:J1"] and cut["locations_omitted"] == 1
    assert "(2 like it)" in str(model.describe())


def test_a_delayed_event_refuses_forward_sensitivities_as_the_run_does():
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(
        "species A = 10; J: A -> ; k*A; k = 1; k2 = 3; J2: -> A; k2; "
        "E: at 0.5 after (time > 1): A = 10;"
    )
    fs = model.describe()["methods"]["forward_sensitivities"]
    if fs["status"] == "unavailable":
        pytest.skip("no C compiler: the install cannot run forward sensitivities")
    assert fs["status"] == "refused"
    (reason,) = fs["reasons"]
    assert reason["code"] == "event"
    for params in (["k"], ["k2"]):
        with pytest.raises(bngsim.SensitivityUnsupportedError) as ei:
            sim = bngsim.Simulator(model.clone(), method="ode", sensitivity_params=params)
            sim.run(t_span=(0, 3), n_points=4)
        assert reason["message"] in str(ei.value)


def test_forward_sensitivities_are_never_ok(tmp_path):
    """Whether the rate laws differentiate is decided when a run builds its
    code: a floor() in a rate law loads, describes as undecided, and the run
    refuses it."""
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(
        "species A = 10; J: A -> ; k*A*floor(A/p + 1); k = 1; p = 2"
    )
    fs = model.describe()["methods"]["forward_sensitivities"]
    if fs["status"] == "unavailable":
        pytest.skip("no C compiler: the install cannot run forward sensitivities")
    assert fs["status"] == "undecided"
    assert fs["reasons"] == [] and "decided when a" in fs["note"]
    with pytest.raises(Exception, match="analytic sensitivity right-hand side"):
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
        sim.run(t_span=(0, 1), n_points=2)


def test_antimony_bytes_load_and_are_hashed_as_given():
    pytest.importorskip("antimony")
    text = b"species A = 1; J: A -> ; k*A; k = 1"
    src = bngsim.Model.from_antimony_string(text).describe()["source"]
    assert src["format"] == "antimony"
    assert src["sha256"] == hashlib.sha256(text).hexdigest()


def test_an_omex_model_names_its_archive_entry(tmp_path):
    from bngsim.convert import read_omex, write_omex

    out = write_omex(tmp_path / "m.omex", sbml=TWO_COMPARTMENTS)
    archive = read_omex(out, extract_dir=tmp_path / "x")
    src = archive.load_model().describe()["source"]
    assert src["format"] == "sbml"
    assert src["archive_entry"] == "./model.xml"
    assert src["sha256"] == hashlib.sha256(TWO_COMPARTMENTS.encode("utf-8")).hexdigest()


def test_model_load_records_every_suffix(tmp_path):
    _, net = _net(tmp_path)
    assert bngsim.Model.load(net).describe()["source"]["format"] == "net"
    pytest.importorskip("antimony")
    ant = tmp_path / "m.ant"
    ant.write_text("species A = 1; J: A -> ; k*A; k = 1\n", encoding="utf-8")
    assert bngsim.Model.load(ant).describe()["source"]["format"] == "antimony"


def test_limit_zero_lists_nothing_and_counts_everything(tmp_path):
    model, _ = _net(tmp_path)
    d = model.describe(limit=0)
    for key in ("parameters", "species", "observables", "reactions"):
        assert d[key] == []
        assert d["omitted"][key] == d["counts"][key] > 0


def test_a_long_expression_is_clipped_while_limit_is_set(tmp_path):
    from bngsim._bngsim_core import ModelBuilder
    from bngsim._describe import TEXT_CAP

    b = ModelBuilder()
    x = b.add_species("X", 1.0)
    b.add_parameter("k", 1.0)
    long = " + ".join(["k"] * 200)
    b.add_function("f", long)
    b.add_reaction([x], [], "functional", "f")
    model = bngsim.Model(_core=b.build())
    (f,) = model.describe()["functions"]
    assert f["expression"] == f"{long[:TEXT_CAP]} ... ({len(long) - TEXT_CAP} more characters)"
    assert model.describe(limit=None)["functions"][0]["expression"] == long


def test_describe_is_cheap_on_a_large_network(tmp_path):
    """It used to find the conservation laws for a count: gigabytes on a large
    network (issue #1705 review). It reads nothing that scales like that now."""
    n, m = 3000, 9000
    species = "".join(f" {i} S{i}() {10 if i == 1 else 0}\n" for i in range(1, n + 1))
    reactions = "".join(f" {j} {1 + (7 * j) % n} {1 + (13 * j) % n} k\n" for j in range(1, m + 1))
    text = (
        "begin parameters\n 1 k 1\nend parameters\n"
        f"begin species\n{species}end species\n"
        f"begin reactions\n{reactions}end reactions\n"
    )
    model, _ = _net(tmp_path, text, "big.net")

    def forbidden(self):
        pytest.fail("describe() found the conservation laws")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bngsim.Model, "conservation_laws", property(forbidden))
        d = model.describe()
    assert d["counts"]["species"] == n and d["counts"]["reactions"] == m
    assert "conservation_laws" not in d["counts"]


def test_a_file_gone_before_it_is_hashed_records_no_hash(tmp_path):
    from bngsim._model import _record_source

    model, path = _net(tmp_path)
    path.unlink()
    _record_source(model, "net", path=path)
    assert model.describe()["source"] == {
        "format": "net",
        "path": os.path.abspath(path),
        "sha256": None,
    }


def test_a_stoichiometry_an_assignment_sets_is_named_not_numbered():
    """A speciesReference id an initialAssignment sets holds a number the file
    does not: the equation names the id."""
    text = NON_INTEGER_STOICH.replace(
        '<speciesReference species="A" stoichiometry="1.5" constant="true"/>',
        '<speciesReference id="nA" species="A" stoichiometry="1.5" constant="true"/>',
    ).replace(
        "  </listOfParameters>\n",
        '   <parameter id="m" value="2" constant="true"/>\n  </listOfParameters>\n'
        "  <listOfInitialAssignments>\n"
        '   <initialAssignment symbol="nA">' + _math(_ci("m")) + "</initialAssignment>\n"
        "  </listOfInitialAssignments>\n",
    )
    assert 'id="nA"' in text and 'symbol="nA"' in text
    r = bngsim.Model.from_sbml_string(text).describe()["reactions"][0]
    assert r["equation"] == "(nA)*A -> B"


def test_a_local_parameter_reads_as_the_models_name_for_it():
    """A kinetic law's local k shadows the global k: the rate names the
    parameter the run reads, the model's _lp_R1_k, not the global."""
    text = TWO_COMPARTMENTS.replace(
        '   <parameter id="k1" value="1" constant="true"/>\n',
        '   <parameter id="k1" value="1" constant="true"/>\n'
        '   <parameter id="k" value="5" constant="true"/>\n',
    ).replace(
        "<kineticLaw>"
        + _math(f"<apply><times/>{_ci('k1')}{_ci('A')}{_ci('cell')}</apply>")
        + "</kineticLaw>",
        "<kineticLaw>"
        + _math(f"<apply><times/>{_ci('k')}{_ci('A')}</apply>")
        + '<listOfLocalParameters><localParameter id="k" value="0.1"/></listOfLocalParameters>'
        + "</kineticLaw>",
    )
    assert 'localParameter id="k"' in text
    model = bngsim.Model.from_sbml_string(text)
    d = model.describe()
    assert d["reactions"][0]["rate"] == "_lp_R1_k * A"
    params = {p["name"]: p["value"] for p in d["parameters"]}
    assert params["k"] == 5.0 and params["_lp_R1_k"] == 0.1
    # And it is the rate the run reads. A (in cell, of size 2) loses R1 = k*A
    # and gains R2 = k2*B*C, amount rates, so [A]' = (-k*10 + 0.1*3*1) / 2:
    # -0.35 with the local k = 0.1, where the global k = 5 would give -2.35.
    assert model.rhs(model.get_state())[0] == pytest.approx((-0.1 * 10 + 0.1 * 3 * 1) / 2)


def test_function_definitions_a_rate_calls_are_listed():
    text = TWO_COMPARTMENTS.replace(
        "  <listOfCompartments>\n",
        "  <listOfFunctionDefinitions>\n"
        '   <functionDefinition id="twice">'
        + _math(
            "<lambda><bvar><ci>x</ci></bvar><apply><times/><cn>2</cn><ci>x</ci></apply></lambda>"
        )
        + "</functionDefinition>\n"
        "  </listOfFunctionDefinitions>\n"
        "  <listOfCompartments>\n",
    ).replace(
        f"<apply><times/>{_ci('k2')}{_ci('B')}{_ci('C')}</apply>",
        f"<apply><ci>twice</ci><apply><times/>{_ci('k2')}{_ci('B')}{_ci('C')}</apply></apply>",
    )
    d = bngsim.Model.from_sbml_string(text).describe()
    assert d["reactions"][1]["rate"] == "twice(k2 * B * C)"
    assert d["function_definitions"] == [{"name": "twice", "definition": "lambda(x, 2 * x)"}]
    assert d["counts"]["function_definitions"] == 1
    assert "twice = lambda(x, 2 * x)" in str(d)


def test_an_sbml_files_reactions_are_cut_at_the_limit():
    d = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS).describe(limit=1)
    assert [r["id"] for r in d["reactions"]] == ["R1"]
    assert d["counts"]["reactions"] == 2 and d["omitted"]["reactions"] == 1


def _with_local(local_id: str, law: str, extra: str = "") -> str:
    """One reaction A -> B whose law reads a local parameter ``local_id`` of
    0.1, beside a global ``k = 5``; ``extra`` goes before the compartments."""
    return (
        _SBML_HEAD + extra + "  <listOfCompartments>\n"
        '   <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>\n'
        "  </listOfCompartments>\n"
        f"  <listOfSpecies>\n{_AB}  </listOfSpecies>\n"
        '  <listOfParameters><parameter id="k" value="5" constant="true"/></listOfParameters>\n'
        "  <listOfReactions>\n"
        '   <reaction id="R1" reversible="false">\n'
        '    <listOfReactants><speciesReference species="A" stoichiometry="1" '
        'constant="true"/></listOfReactants>\n'
        '    <listOfProducts><speciesReference species="B" stoichiometry="1" '
        'constant="true"/></listOfProducts>\n'
        f"    <kineticLaw>{_math(law)}<listOfLocalParameters>"
        f'<localParameter id="{local_id}" value="0.1"/></listOfLocalParameters></kineticLaw>\n'
        "   </reaction>\n"
        "  </listOfReactions>\n" + _SBML_TAIL
    )


@pytest.mark.parametrize(
    "word", ["Pi", "pi", "INF", "infinity", "NaN", "notanumber", "time", "True", "avogadro"]
)
def test_a_local_named_like_a_parser_constant_is_renamed(word):
    """libsbml's formula parser reads these words as constants, in any case: the
    law is parsed with the ids to rename declared, so each is read as the id."""
    text = _with_local(word, f"<apply><times/>{_ci(word)}{_ci('A')}</apply>")
    model = bngsim.Model.from_sbml_string(text)
    assert f"_lp_R1_{word}" in model.param_names
    assert model.describe()["reactions"][0]["rate"] == f"_lp_R1_{word} * A"
    # The run reads the local: 0.1 * 100.
    assert model.rhs(model.get_state())[0] == pytest.approx(-10.0)


def test_a_renamed_species_named_like_a_parser_constant():
    """BIOMD0000000201 has a species Nan, the model's _ant_Nan."""
    text = _one_compartment(
        _species_xml("Nan", 100) + _species_xml("B", 0),
        _reaction("J", "Nan", "B", f"<apply><times/>{_ci('k')}{_ci('Nan')}</apply>"),
    )
    d = bngsim.Model.from_sbml_string(text).describe()
    assert d["species"][0]["name"] == "_ant_Nan"
    assert d["reactions"][0]["rate"] == "k * _ant_Nan"


def test_a_law_the_parser_cannot_read_back_is_renamed_by_token():
    """A function definition named log with three arguments writes a law the L3
    parser cannot read (MODEL2003050002); its local is still renamed."""
    definition = (
        "  <listOfFunctionDefinitions>\n"
        '   <functionDefinition id="log">'
        + _math(
            "<lambda><bvar><ci>a</ci></bvar><bvar><ci>b</ci></bvar><bvar><ci>c</ci></bvar>"
            "<apply><times/><ci>a</ci><ci>b</ci><ci>c</ci></apply></lambda>"
        )
        + "</functionDefinition>\n  </listOfFunctionDefinitions>\n"
    )
    law = f"<apply><ci>log</ci>{_ci('kl')}{_ci('A')}{_ci('kl')}</apply>"
    model = bngsim.Model.from_sbml_string(_with_local("kl", law, definition))
    assert model.describe()["reactions"][0]["rate"] == "log(_lp_R1_kl, A, _lp_R1_kl)"
    # A local that shares the function's id: the call keeps its name.
    law = f"<apply><ci>log</ci>{_ci('log')}{_ci('A')}{_ci('log')}</apply>"
    model = bngsim.Model.from_sbml_string(_with_local("log", law, definition))
    assert model.describe()["reactions"][0]["rate"] == "log(_lp_R1_log, A, _lp_R1_log)"


_CLOCK = (
    '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
)


def _ast_renamed(text: str, names: dict[str, str]) -> str:
    """The first reaction's law renamed on its own math by libsbml, which tells
    the clock and pi from ids of the same spelling: the oracle."""
    import libsbml

    doc = libsbml.readSBMLFromString(text)
    math = doc.getModel().getReaction(0).getKineticLaw().getMath().deepCopy()
    for old, new in names.items():
        math.renameSIdRefs(old, new)
    return libsbml.formulaToL3String(math)


def test_the_clock_beside_a_parameter_named_time():
    """A model with a parameter ``time`` (the model's ``_ant_time``): a law
    that reads the clock still reads the clock (issue #1705 review)."""
    text = _with_local("kl", f"<apply><times/>{_CLOCK}{_ci('A')}</apply>").replace(
        '<parameter id="k" value="5" constant="true"/>',
        '<parameter id="k" value="5" constant="true"/><parameter id="time" value="0.1"/>',
    )
    model = bngsim.Model.from_sbml_string(text)
    assert "_ant_time" in model.param_names
    rate = model.describe()["reactions"][0]["rate"]
    assert rate == _ast_renamed(text, {"time": "_ant_time"})
    assert "_ant_time" not in rate
    # The run reads the clock: the rate is 0 at t = 0.
    assert model.rhs(model.get_state(), t=0.0)[0] == 0.0


def test_the_clock_beside_a_local_named_time():
    law = f"<apply><times/>{_CLOCK}{_ci('time')}{_ci('A')}</apply>"
    text = _with_local("time", law)
    rate = bngsim.Model.from_sbml_string(text).describe()["reactions"][0]["rate"]
    assert rate == _ast_renamed(text, {"time": "_lp_R1_time"})
    assert rate.count("_lp_R1_time") == 1


def test_pi_beside_a_local_named_pi():
    law = f"<apply><times/><pi/>{_ci('pi')}{_ci('A')}</apply>"
    text = _with_local("pi", law)
    model = bngsim.Model.from_sbml_string(text)
    rate = model.describe()["reactions"][0]["rate"]
    assert rate == _ast_renamed(text, {"pi": "_lp_R1_pi"}) == "pi * _lp_R1_pi * A"
    assert model.rhs(model.get_state())[0] == pytest.approx(-math.pi * 0.1 * 100)


@pytest.mark.parametrize("word", ["Time", "Avogadro", "Infinity", "NotANumber"])
def test_a_global_the_model_keeps_spelled_like_a_parser_word(word):
    """A global the model does not rename, spelled like a word of the L3 parser
    in another case, stays itself in a law that has a local (issue #1705
    review): parsed back, the text would read the word."""
    law = f"<apply><times/>{_ci(word)}{_ci('kl')}{_ci('A')}</apply>"
    text = _with_local("kl", law).replace(
        '<parameter id="k" value="5" constant="true"/>',
        f'<parameter id="k" value="5" constant="true"/><parameter id="{word}" value="0.1"/>',
    )
    model = bngsim.Model.from_sbml_string(text)
    assert word in model.param_names
    rate = model.describe()["reactions"][0]["rate"]
    assert rate == _ast_renamed(text, {"kl": "_lp_R1_kl"}) == f"{word} * _lp_R1_kl * A"


# ─── Compartment sizes and sensitivities (issue #1845) ───────────────────────


def test_a_d_dv_column_matches_a_difference_of_rebuilds():
    """A compartment size's sensitivity column is computed, and it is right: it
    matches a central difference across models rebuilt at V +/- h, the oracle
    Simulator._raise_if_compartment_size_params names (issue #1845, whose
    docstring said the column was refused)."""
    model = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS)
    assert "cell" in model.compartment_size_params
    assert "cell" not in model.describe()["methods"]["forward_sensitivities"].get(
        "refused_parameters", {}
    )
    run = {"t_span": (0, 1), "n_points": 3, "rtol": 1e-12, "atol": 1e-14}
    try:
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["cell"])
    except bngsim.CodegenToolchainError:
        pytest.skip("no C compiler: a sensitivity run compiles the model")
    column = sim.run(**run).sensitivities[-1, :, 0]

    def end(size):
        m = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS, compartment_sizes={"cell": size})
        return bngsim.Simulator(m, method="ode", codegen=False).run(**run).species[-1]

    h = 1e-5
    difference = (end(2 + h) - end(2 - h)) / (2 * h)
    assert column == pytest.approx(difference, rel=1e-7, abs=1e-10)
    assert abs(column[1]) > 1  # B moves with the size: the column is not a zero


def test_unwritable_compartment_sizes_are_refused_parameters():
    """Two compartments of one size share a mass-action scalar, so set_param
    cannot write either, and a sensitivity run refuses both: describe() says so."""
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(
        "compartment c1 = 2, c2 = 2; species A in c1 = 10, B in c2 = 0; J: A -> B; k*A*B; k = 0.1"
    )
    assert model.unwritable_compartment_size_params == ["c1", "c2"]
    refused = model.describe()["methods"]["forward_sensitivities"]["refused_parameters"]
    assert set(refused) == {"c1", "c2"}
    assert all("set_param cannot write" in why for why in refused.values())
    for name in ("c1", "c2"):
        with pytest.raises(
            ValueError, match="Forward sensitivity is not supported for compartment"
        ):
            sim = bngsim.Simulator(model.clone(), method="ode", sensitivity_params=[name])
            sim.run(t_span=(0, 1), n_points=2)


# ─── Steady state (issue #1844) ──────────────────────────────────────────────

_STEADY_STATE_REFUSED = {
    "event": lambda: bngsim.Model.from_sbml_string(WITH_EVENT),
    "reads_clock": lambda: bngsim.Model.from_antimony_string(
        "species A = 1; J: -> A; k*time; k = 0.1; J2: A -> ; kd*A; kd = 1"
    ),
    "size_jump": lambda: bngsim.Model.from_antimony_string(
        "compartment c; c := piecewise(1, time < 5, 2); species A in c = 1; "
        "J: -> A; k; J2: A -> ; kd*A; k = 1; kd = 0.1"
    ),
}


@pytest.mark.parametrize("code", sorted(_STEADY_STATE_REFUSED))
def test_a_steady_state_refusal_is_the_calls_own(code):
    """describe()'s steady-state verdict is read from the check steady_state()
    and steady_state_batch() raise from: the same code, the same words."""
    if code != "event":
        pytest.importorskip("antimony")
    model = _STEADY_STATE_REFUSED[code]()
    verdict = model.describe()["methods"]["steady_state"]
    assert verdict["status"] == "refused"
    (reason,) = verdict["reasons"]
    assert reason["code"] == code
    with pytest.raises(bngsim.SimulationError) as ei:
        bngsim.Simulator(model.clone(), method="ode").steady_state()
    assert str(ei.value) == reason["message"]
    with pytest.raises(bngsim.SimulationError) as ei:
        bngsim.Simulator(model.clone(), method="ode").steady_state_batch([{}])
    assert str(ei.value) == reason["message"].replace("steady_state()", "steady_state_batch()")


def test_a_model_with_a_steady_state_says_ok():
    model = bngsim.Model.from_sbml_string(TWO_COMPARTMENTS)
    verdict = model.describe()["methods"]["steady_state"]
    assert verdict["status"] == "ok" and verdict["reasons"] == []
    assert "conservation law" in verdict["note"]  # what the verdict does not cover
    assert bngsim.Simulator(model, method="ode").steady_state().converged
    methods = str(model.describe()).split("Methods:")[1].split("Analytical")[0]
    (line,) = [ln for ln in methods.splitlines() if "steady state" in ln]
    assert line.endswith(": ok")


def test_ok_is_not_a_promise_of_convergence(tmp_path):
    """The .net model's fixed source C feeds P without end, so it has no steady
    state: nothing refuses the solve, which runs and does not converge, as the
    note says it may."""
    model, _ = _net(tmp_path)
    verdict = model.describe()["methods"]["steady_state"]
    assert verdict["status"] == "ok" and "converge" in verdict["note"]
    assert not bngsim.Simulator(model, method="ode").steady_state().converged
