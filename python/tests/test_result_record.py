"""``Result.record`` (issue #1703): what a result's numbers are and how its run went.

A Result held the arrays of a run and little about the run: nothing said what
kind of quantity a column is, which method ran, the tolerances as applied, which
code ran, or what file the model came from. These tests hold the record to the
run it describes, and where the record makes a claim about the numbers (that a
stochastic SBML column is molecules over the size, that an observable of a
species declared in amounts is that amount), they check the claim against the
numbers themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ExactAtol, Model, Result, Simulator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _codegen_backend import selected_codegen_backend  # noqa: E402

# A in C (size 2) declared by concentration, N in C declared in amounts
# (hasOnlySubstanceUnits), with Level 3 units on the model and the compartment.
SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="units" timeUnits="second" substanceUnits="mole" volumeUnits="litre"
         extentUnits="mole">
    <listOfUnitDefinitions>
      <unitDefinition id="per_second">
        <listOfUnits><unit kind="second" exponent="-1" scale="0" multiplier="1"/></listOfUnits>
      </unitDefinition>
    </listOfUnitDefinitions>
    <listOfCompartments>
      <compartment id="C" size="2" spatialDimensions="3" units="litre" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="10" substanceUnits="mole"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="N" compartment="C" initialAmount="4" substanceUnits="item"
               hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.5" units="per_second" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="J1" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>C</ci><ci>k</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="J2" reversible="false">
        <listOfReactants>
          <speciesReference species="N" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>N</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

# A -> B at kf, B -> A at kr, in BioNetGen's .net format.
NET = """\
begin parameters
    1 kf 0.5
    2 kr 0.25
    3 A0 100
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")


@pytest.fixture
def net_path(tmp_path: Path) -> Path:
    path = tmp_path / "ab.net"
    path.write_text(NET, encoding="utf-8")
    return path


def _sbml() -> Model:
    return Model.from_sbml_string(SBML)


# ─── What the columns are ──────────────────────────────────────────────


def test_an_sbml_run_says_each_column_is_a_concentration_in_its_compartment() -> None:
    """The issue's example: A at 10 in a compartment of size 2 reads 10. The
    record says it is an amount over a size of 2, and the declared units."""
    r = Simulator(_sbml(), method="ode").run(t_span=(0, 1), n_points=3)
    cols = r.record["columns"]
    assert cols["species"] == [
        {"name": "A", "quantity": "concentration", "compartment": "C", "substance_units": "mole"},
        {"name": "N", "quantity": "concentration", "compartment": "C", "substance_units": "item"},
    ]
    assert cols["compartments"] == {"C": {"size": 2.0, "size_varies": False, "units": "litre"}}
    assert r.record["time"] == {"units": "second"}
    # N is declared in amounts: its observable is the amount, the species
    # column that amount over the size.
    assert cols["observables"] == [
        {"name": "A", "quantity": "concentration", "compartment": "C"},
        {"name": "N", "quantity": "amount", "compartment": "C"},
    ]
    np.testing.assert_allclose(r.observables["N"], r.species["N"] * 2.0, rtol=1e-12)
    np.testing.assert_allclose(r.observables["A"], r.species["A"], rtol=0)


def test_a_stochastic_sbml_column_is_molecules_over_the_size_and_says_so() -> None:
    """Under SSA the same column reads 10, 9.5, ...: molecule counts divided by
    the size. The record says concentration in C of size 2, and the column
    times that size is whole molecules; the amount-declared observable is the
    count itself."""
    r = Simulator(_sbml(), method="ssa").run(t_span=(0, 2), n_points=21, seed=1)
    rec = r.record
    assert rec["run"]["method"] == "ssa"
    sizes = {
        c["name"]: rec["columns"]["compartments"][c["compartment"]]["size"]
        for c in rec["columns"]["species"]
    }
    assert sizes == {"A": 2.0, "N": 2.0}
    for name, size in sizes.items():
        molecules = r.species[name] * size
        np.testing.assert_array_equal(molecules, np.round(molecules))
    assert np.any(r.species["A"] * 2.0 % 2.0 == 1.0), "a column of whole molecules over 2"
    obs = {o["name"]: o["quantity"] for o in rec["columns"]["observables"]}
    assert obs == {"A": "concentration", "N": "count"}
    np.testing.assert_array_equal(r.observables["N"], np.round(r.observables["N"]))


def test_a_network_is_counts_under_a_stochastic_method_and_unstated_under_ode(
    net_path: Path,
) -> None:
    model = Model.from_net(net_path)
    ode = Simulator(model).run(t_span=(0, 1), n_points=3, start="initial")
    ssa = Simulator(model, method="ssa").run(t_span=(0, 1), n_points=3, seed=3, start="initial")
    for column in (*ode.record["columns"]["species"], *ode.record["columns"]["observables"]):
        assert column["quantity"] is None and column["compartment"] is None
    for column in (*ssa.record["columns"]["species"], *ssa.record["columns"]["observables"]):
        assert column["quantity"] == "count"
    np.testing.assert_array_equal(ssa.species, np.round(ssa.species))
    assert ode.record["columns"]["compartments"] == {}
    assert ode.record["time"] == {"units": None}


def test_a_compartment_a_rule_moves_has_no_one_size() -> None:
    sbml = SBML.replace(
        '<compartment id="C" size="2" spatialDimensions="3" units="litre" constant="true"/>',
        '<compartment id="C" size="2" spatialDimensions="3" units="litre" constant="false"/>',
    ).replace(
        "</listOfReactions>",
        '</listOfReactions><listOfRules><rateRule variable="C"><math '
        'xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.1</cn></math></rateRule>'
        "</listOfRules>",
    )
    r = Simulator(Model.from_sbml_string(sbml)).run(t_span=(0, 1), n_points=3)
    assert r.record["columns"]["compartments"]["C"]["size_varies"] is True
    assert r.record["columns"]["compartments"]["C"]["size"] is None


# ─── The run ───────────────────────────────────────────────────────────


def test_the_run_part_says_what_was_asked_and_what_the_solver_was_given(
    net_path: Path,
) -> None:
    model = Model.from_net(net_path)
    model.set_param("kf", 0.7)
    sim = Simulator(model, method="ode", codegen=False)
    r = sim.run(sample_times=[0, 0.5, 2, 4], rtol=1e-9, atol=1e-10, max_steps=5000)
    run = r.record["run"]
    assert run["method"] == run["requested_method"] == "ode"
    assert (run["t_start"], run["t_end"], run["n_points"]) == (0.0, 4.0, 4)
    assert run["sample_times"] == [0.0, 0.5, 2.0, 4.0]
    assert run["rtol"] == 1e-9 and run["max_steps"] == 5000
    assert run["atol"]["given"] == 1e-10
    assert run["parameters"] == {"kf": 0.7, "kr": 0.25, "A0": 100.0}
    assert run["started_from"] == "initial" and run["start"] is None
    assert run["start_state"] == [100.0, 0.0]
    assert run["start_state_sha256"] == r.start_state_sha256
    assert run["seed"] is None and run["wall_time_s"] > 0
    backend = r.record["backend"]
    assert backend["rhs"] == "interpreted"
    assert backend["jacobian"] == "analytical" and backend["jacobian_retried"] is False
    assert backend["linear_solver"] == "dense"
    assert backend["sensitivity_rhs"] is None and backend["propensities"] is None


def test_the_tolerances_are_those_the_solver_was_given(net_path: Path) -> None:
    """The default atol is a ceiling scaled per species from the start (issue
    #1420), so the value passed is not the value used: B starts at 0 and A at
    1e-6, below the scale of the scalar. The record holds the vector."""
    model = Model.from_net(net_path)
    model.set_param("A0", 1e-6)
    sim = Simulator(model, method="ode")
    r = sim.run(t_span=(0, 1), n_points=3, start="initial")
    atol = r.record["run"]["atol"]
    assert atol["given"] is None and atol["scalar"] == 1e-8
    assert atol["per_species"] is not None and len(atol["per_species"]) == 2
    assert atol["per_species"][0] < 1e-8
    tracking = sim.run(t_span=(0, 1), n_points=3, start="initial", atol="tracking")
    assert tracking.record["run"]["atol"]["given"] == "tracking"
    assert tracking.record["run"]["atol"]["tracking_decades"] > 0


def test_the_record_holds_what_the_run_needs_to_be_made_again(net_path: Path) -> None:
    """Every input of a run, read back from its record alone and given to a
    fresh model from the recorded file, makes the same numbers bit for bit: a
    run from a live state after a set_param, at the tolerances as applied."""
    model = Model.from_net(net_path)
    sim = Simulator(model, method="ode")
    model.set_param("kf", 0.9)
    sim.run(t_span=(0, 3), n_points=4)
    first = sim.run(t_span=(3, 8), n_points=11, start="live", rtol=1e-9)
    rec = json.loads(json.dumps(first.record))
    assert rec["run"]["started_from"] == "live"

    path = Path(rec["model"]["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["model"]["sha256"]
    fresh = Model.load(path)
    fresh.set_params(rec["run"]["parameters"])
    fresh.set_state(np.array(rec["run"]["start_state"]))
    atol = rec["run"]["atol"]
    again = Simulator(fresh, method=rec["run"]["method"]).run(
        t_span=(rec["run"]["t_start"], rec["run"]["t_end"]),
        n_points=rec["run"]["n_points"],
        rtol=rec["run"]["rtol"],
        atol=atol["per_species"] or ExactAtol(atol["scalar"]),
        max_steps=rec["run"]["max_steps"],
        start="live",
    )
    np.testing.assert_array_equal(again.species, first.species)
    np.testing.assert_array_equal(again.observables, first.observables)


def test_a_stochastic_run_records_its_seed_and_propensities(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path), method="ssa").run(t_span=(0, 1), n_points=3)
    run, backend = r.record["run"], r.record["backend"]
    assert run["seed"] == r.seed is not None
    assert run["rtol"] is None and run["atol"] is None and run["max_steps"] is None
    assert backend["propensities"] == r.ssa_diagnostics["propensity_backend"]
    assert backend["rhs"] is None and backend["jacobian"] is None


# ─── Which code ran ────────────────────────────────────────────────────

# S empties at t = 2, where d/dS sqrt(S) is infinite: the analytical attempt
# fails and the run is made again on a difference quotient (GH #176, #867;
# test_nonfinite_jacobian_after_clamp.py's model).
SINK = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sink">
    <listOfCompartments>
      <compartment id="V" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="V" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="R1" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><apply><root/><ci>S</ci></apply></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_a_jacobian_retry_is_on_the_result() -> None:
    """The issue's "not checked": a run whose analytical-Jacobian attempt fails
    and whose retry succeeds left no trace on its Result."""
    sim = Simulator(Model.from_sbml_string(SINK), method="ode")
    if sim.jacobian_strategy != "analytical":
        pytest.skip("the model has no analytical Jacobian on this build")
    r = sim.run(t_span=(0, 40), n_points=5)
    backend = r.record["backend"]
    assert backend["jacobian_requested"] == "auto"
    assert backend["jacobian"] == "difference_quotient"
    assert backend["jacobian_retried"] is True
    # The next run goes straight to the difference quotient: no retry of its own.
    again = sim.run(t_span=(0, 40), n_points=5, start="initial")
    assert again.record["backend"]["jacobian"] == "difference_quotient"
    assert again.record["backend"]["jacobian_retried"] is False


def test_the_backend_names_the_jacobian_and_linear_solver_asked_for(net_path: Path) -> None:
    fd = Simulator(Model.from_net(net_path), jacobian="fd").run(t_span=(0, 1), n_points=2)
    assert fd.record["backend"]["jacobian"] == "difference_quotient"
    assert fd.record["backend"]["jacobian_requested"] == "fd"
    if not bngsim.capabilities()["features"].get("klu"):
        pytest.skip("this install has no KLU")
    sparse = Simulator(Model.from_net(net_path), force_sparse_linear_solver=True)
    r = sparse.run(t_span=(0, 1), n_points=2)
    assert r.record["backend"]["linear_solver"] == "klu"


@pytest.mark.skipif(_CC is None, reason="needs a C compiler for codegen")
def test_a_compiled_run_names_its_backend(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path), codegen=True)
    r = sim.run(t_span=(0, 1), n_points=2)
    assert r.record["backend"]["rhs"] == selected_codegen_backend() == sim.codegen_backend


@pytest.mark.skipif(_CC is None, reason="needs a C compiler for sensitivities")
def test_a_sensitivity_run_names_its_sensitivity_rhs(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path), sensitivity_params=["kf"])
    r = sim.run(t_span=(0, 1), n_points=3)
    assert r.record["run"]["sensitivity_params"] == ["kf"]
    assert r.record["backend"]["sensitivity_rhs"] == (
        "analytic" if sim.has_analytic_sens_rhs else "difference_quotient"
    )


# ─── The model and the build ───────────────────────────────────────────


def test_the_model_and_software_parts(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 1), n_points=2)
    assert r.record["model"] == {
        "format": "net",
        "path": str(net_path.resolve()),
        "sha256": hashlib.sha256(net_path.read_bytes()).hexdigest(),
    }
    assert r.record["software"] == {
        "bngsim": bngsim.__version__,
        "build_commit": bngsim.capabilities()["build"]["commit"],
    }
    assert r.record["schema_version"] == 1


# ─── Where the record goes ─────────────────────────────────────────────


def test_summary_carries_the_record_without_the_start_vector(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 1), n_points=2)
    s = json.loads(json.dumps(r.summary()))
    want = json.loads(json.dumps(r.record))
    del want["run"]["start_state"]
    assert s["record"] == want


def test_each_read_is_a_new_dict(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 1), n_points=2)
    r.record["run"]["parameters"]["kf"] = -1.0
    r.record["columns"]["species"].clear()
    assert r.record["run"]["parameters"]["kf"] == 0.5
    assert len(r.record["columns"]["species"]) == 2


def test_save_and_load_keep_the_record(tmp_path: Path, net_path: Path) -> None:
    pytest.importorskip("h5py")
    model = Model.from_net(net_path)
    sim = Simulator(model)
    sim.run(t_span=(0, 1), n_points=2)
    with pytest.warns(bngsim.LiveStartWarning):
        r = sim.run(t_span=(0, 1), n_points=3)
    assert r.notices
    r.save(tmp_path / "r.h5")
    back = Result.load(tmp_path / "r.h5")
    assert back.record == r.record
    assert back.notices == r.notices
    # And again, from the loaded copy.
    back.save(tmp_path / "again.h5")
    assert Result.load(tmp_path / "again.h5").record == r.record


def test_a_file_written_before_the_record_has_none(tmp_path: Path, net_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 1), n_points=2)
    r.save(tmp_path / "r.h5")
    with h5py.File(tmp_path / "r.h5", "a") as f:
        del f["record"], f["start_state"]
    back = Result.load(tmp_path / "r.h5")
    rec = back.record
    for part in ("software", "model", "run", "backend", "time", "columns"):
        assert rec[part] is None
    assert rec["notices"] == [] and back.start_state_sha256 == r.start_state_sha256


def test_a_stop_condition_cut_keeps_the_record(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path))
    sim.add_stop_condition("Atot < 60", label="half")
    with pytest.raises(bngsim.StopConditionMet) as stop:
        sim.run(t_span=(0, 10), n_points=101)
    cut = stop.value.result
    assert cut.stop_condition == "half"
    assert cut.record["run"]["t_end"] == 10.0  # the run asked, not the cut
    assert cut.record["run"]["parameters"]["kf"] == 0.5


def test_run_until_records_its_step(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path))
    sim.run_until(1.0)
    r = sim.run_until(2.0, n_points=3)
    assert r.record["run"]["t_start"] == 1.0 and r.record["run"]["t_end"] == 2.0
    assert r.record["run"]["started_from"] == "live"


# ─── Stacks ────────────────────────────────────────────────────────────


def test_a_batch_stack_keeps_what_its_rows_agree_on(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path))
    rows = sim.run_batch(t_span=(0, 1), n_points=3, params=[{"kf": 0.1}, {"kf": 0.2}])
    assert [r.record["run"]["parameters"]["kf"] for r in rows] == [0.1, 0.2]
    stack = sim.run_batch(
        t_span=(0, 1), n_points=3, params=[{"kf": 0.1}, {"kf": 0.2}], squeeze=True
    )
    run = stack.record["run"]
    assert run["parameters"] == {"kf": None, "kr": 0.25, "A0": 100.0}
    assert run["method"] == "ode" and run["n_points"] == 3
    assert run["wall_time_s"] is None
    assert stack.record["backend"] == rows[0].record["backend"]
    assert stack.record["columns"] == rows[0].record["columns"]


def test_a_batch_row_that_writes_a_size_has_its_own(tmp_path: Path) -> None:
    sim = Simulator(_sbml())
    rows = sim.run_batch(t_span=(0, 1), n_points=2, params=[{"C": 2.0}, {"C": 4.0}])
    assert [r.record["columns"]["compartments"]["C"]["size"] for r in rows] == [2.0, 4.0]
    stack = Result.squeeze(rows)
    assert stack.record["columns"]["compartments"]["C"]["size"] is None


def test_replicates_record_each_seed_and_the_stack_none(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path), method="ssa")
    rows = sim.run_replicates(3, t_span=(0, 1), n_points=2, seed=7)
    seeds = [r.record["run"]["seed"] for r in rows]
    assert seeds == [r.seed for r in rows] and len(set(seeds)) == 3
    stack = Result.squeeze(rows)
    assert stack.record["run"]["seed"] is None
    assert stack.record["run"]["started_from"] == "initial"
    assert stack.record["run"]["start_state"] == rows[0].record["run"]["start_state"]


def test_a_scan_point_records_its_value(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path))
    points = sim.parameter_scan("kf", [0.1, 0.3], t_span=(0, 1), n_points=2)
    assert [p.record["run"]["parameters"]["kf"] for p in points] == [0.1, 0.3]
