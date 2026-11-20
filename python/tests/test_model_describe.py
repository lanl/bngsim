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
    assert set(d["methods"]) == {"ode", "ssa", "psa", "nf", "rm", "forward_sensitivities"}


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
        "conservation_laws": model.conservation_laws["n_laws"],
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
            "rate_law": "michaelis_menten",
            "rate": "kcat, Km",
        },
        {
            "reactants": ["P()"],
            "products": ["S()"],
            "rate_law": "elementary",
            "rate": "kf",
            "multiplier": 2.0,
        },
        {"reactants": ["C()"], "products": ["P()"], "rate_law": "elementary", "rate": "kr"},
    ]


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
        assert m[method]["reasons"][0]["location"]
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
    assert fs["refused_parameters"] == {"p": model._frozen_params["p"]}
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
