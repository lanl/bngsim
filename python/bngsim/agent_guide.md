# BNGsim agent guide

BNGsim simulates biochemical reaction networks in Python. It loads BioNetGen
(`.net`, `.bngl`), SBML and Antimony models, runs them as ODEs or
stochastically, in process, and returns NumPy arrays. This page is for an AI
agent, or anyone, using BNGsim for the first time. It gives rules and recipes,
not a tutorial.

This is the guide for the installed version. `bngsim.agent_guide()` returns it
and `python -m bngsim guide` prints it, with no network. Every call has a full
docstring (`help(bngsim.Simulator.run)`). The user guides are at
<https://bngsim.readthedocs.io>, with an index for agents at
<https://bngsim.readthedocs.io/llms.txt>.

Every Python block below runs as written, in order, in an empty directory,
after `pip install bngsim`. The sensitivities also need a C compiler (see
"What needs what"). The test suite runs the blocks this way. The calls on this
page keep their names across releases: if a later release renames one, the old
name raises an error that gives the new one.

## The calls

A model is a `Model`, a `Simulator` runs it, and a run returns a `Result`.
This block writes a small network, `A <-> B`, for the rest of the page to use.

```python
from pathlib import Path

import numpy as np

import bngsim

# A <-> B with forward rate kf and reverse rate kr, in BioNetGen's .net format.
Path("model.net").write_text(
    """\
begin parameters
    1 kf 0.5
    2 kr 0.25
end parameters
begin species
    1 A() 100
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
begin groups
    1 A 1
    2 B 2
end groups
""",
    encoding="utf-8",
)

model = bngsim.Model.load("model.net")  # by suffix: .net .bngl .xml .sbml .ant
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0, 10), n_points=101)

print(result.time.shape)            # (101,)
print(result.observables.shape)     # (101, 2): time by observable
print(result.observable_names)      # ['A', 'B']
print(result.species_names)         # ['A()', 'B()']
a = result.observables["A"]         # one column, by name
print(round(float(a[-1]), 2))       # 33.37

d = model.describe()                # what the model is; print(d) is a report
print(d["methods"]["ssa"]["status"])  # ok
```

| to | call |
|---|---|
| load a file by its suffix | `bngsim.Model.load(path)` |
| what the model is, and which methods it admits | `print(model.describe())`; `model.describe()["methods"]["ssa"]` gives a status and the reasons; lists past 50 entries are cut, and `model.describe(limit=None)` keeps all |
| load one format | `Model.from_net(path)`, `Model.from_bngl(path)`, `Model.from_sbml(path)`, `Model.from_antimony(path)` |
| load text | `Model.from_sbml_string(text)`, `Model.from_antimony_string(text)`; BNGL only from a file |
| load a BioModels entry | `Model.from_biomodels("BIOMD0000000010")`: fetched once, then cached |
| choose the method | `bngsim.Simulator(model, method="ode")`; `"ssa"`, `"psa"`; network-free `"nf"` (NFsim) and `"rm"` (RuleMonkey) also take `xml_path=`; `Simulator.METHODS` lists all |
| run | `sim.run(t_span=(t0, t1), n_points=n)` or `sim.run(sample_times=[...])` |
| read a run | `result.time`, `result.species`, `result.observables`, `result.expressions`; each takes a name: `result.observables["A"]` |
| what a run's numbers are and how it went | `result.record`: a JSON-ready dict of each column's quantity and compartment, the method, the tolerances as applied, the start, the code that ran and the model file; `result.notices`, every warning of the run |
| read names | `model.param_names`, `model.species_names`, `model.observable_names` |
| a table | `result.dataframe` (a property; needs pandas) |
| set a parameter | `model.set_param("kf", 1.0)`, `model.set_params({"kf": 1.0, "kr": 0.5})`; read with `model.get_param("kf")` |
| set a species | `model.set_concentration("A()", 50.0)`; read with `model.get_concentration("A()")`; all at once `model.get_state()`, `model.set_state(array)` |
| back to the initial conditions | `model.reset()` (species only; parameters keep their values) |
| scan a parameter | `sim.parameter_scan("kf", [0.1, 1.0], t_span=..., n_points=...)`, or `par_min=`, `par_max=`, `n_scan_pts=`, `log_scale=True` |
| steady state | `sim.steady_state()`: `ss.converged`, `ss["A()"]`, `ss.concentrations` |
| sensitivities | `bngsim.Simulator(model, sensitivity_params=["kf"])`, then `result.sensitivities` |
| stochastic replicates | `bngsim.Simulator(model, method="ssa").run_replicates(n, t_span=..., n_points=..., seed=...)` |
| a BNGL file's own actions | `bngsim.run_bngl(path)` |
| save a result | `result.save("run.h5")`, `bngsim.Result.load("run.h5")` (needs h5py) |

The next block changes a parameter and runs again, then scans a parameter,
finds the steady state, computes sensitivities and runs stochastic replicates:

```python
# Change a parameter and run again from the initial conditions.
model.set_param("kf", 1.0)
fast = sim.run(t_span=(0, 10), n_points=101, start="initial")
model.set_param("kf", 0.5)

# A scan: one Result per value. Each point starts from the model's state at the
# call, so reset it first to start them all from the initial conditions.
model.reset()
scan = sim.parameter_scan("kf", [0.1, 0.5, 1.0], t_span=(0, 10), n_points=11)
print([round(float(r.observables["A"][-1]), 2) for r in scan])  # [72.29, 33.37, 20.0]

# Steady state: A = 100 * kr / (kf + kr). It leaves the model's state alone.
ss = sim.steady_state()
print(ss.converged, round(float(ss["A()"]), 4))  # True 33.3333

# Forward sensitivities, d(species)/d(parameter): (n_times, n_species, n_params).
sens = bngsim.Simulator(model, sensitivity_params=["kf", "kr"])
s = sens.run(t_span=(0, 10), n_points=11, start="initial")
print(s.sensitivities.shape)  # (11, 2, 2)

# Stochastic replicates, stacked: (n_replicates, n_times, n_observables).
# They start from the model's state, which the run above left at t = 10.
model.reset()
ssa = bngsim.Simulator(model, method="ssa")
reps = ssa.run_replicates(50, t_span=(0, 10), n_points=11, seed=1, squeeze=True)
print(reps.observables.shape)  # (50, 11, 2)
```

## Rules that otherwise give wrong answers

1. **A run continues from where the last one ended.** The state lives in the
   `Model`, and each run writes its end state back to it. A second
   `sim.run(t_span=(0, 10), ...)` starts from the first run's end state with
   the clock back at `t = 0`, and BNGsim warns (`LiveStartWarning`). To start
   over, pass `start="initial"` or call `model.reset()` first. To continue on
   purpose (equilibrate, then stimulate), pass `start="live"`.
   `result.started_from` says which it was. `run_batch` starts each point
   from the initial conditions. `parameter_scan` and `run_replicates` start
   each point from the model's state at the call, so call `model.reset()`
   first.
2. **`n_points` counts the output points, the first included.**
   `t_span=(0, 10), n_points=11` gives `0, 1, ..., 10`. BioNetGen's
   `n_steps=10`, RoadRunner's `steps=10` and basico's `intervals=10` are
   `n_points=11`.
3. **`sample_times` sets the interval, not only the grid.** The run starts at
   the first sample time and ends at the last: `sample_times=[5, 10]` never
   integrates `[0, 5]`. Put the start time first when the history matters.
4. **Species names and observable names differ.** In a `.net` or BNGL model
   a species is named by its pattern, `A()`, and an observable by its
   `groups` name, `A`. `result.species["A()"]`, `result.observables["A"]`,
   `model.set_concentration("A()", ...)`. Read the names before guessing them.
5. **A species column is a concentration, in a stochastic run too.** For an
   SBML or Antimony model, `result.species` holds each species' amount divided
   by its compartment's size. That is so whether the species was declared as
   an amount or a concentration. A stochastic run reports the same quantity,
   not molecule counts: in a compartment of size 2, an SSA column moves in
   steps of 0.5. `result.as_roadrunner(["time", "A"])` gives amounts. A `.net`
   model has no compartments, so its columns hold the numbers in the file,
   which a stochastic run treats as molecule counts. `result.record["columns"]`
   says which, column by column, with each compartment's size.
6. **A stochastic run without a seed differs every time.** Pass `seed=n` to
   reproduce one. `result.seed` is the seed a run used, drawn or given. In a
   `run_replicates` or `run_batch` list, each row's `seed` reruns that row.
7. **Set the primary parameters.** Set the ones in
   `model.primary_param_names`. A parameter defined by an expression of others
   follows them. If you set such a parameter to a value other than its
   expression's, it keeps that value from then on, whatever its inputs do.
8. **A run that fails raises an error.** `run()` returns only runs that
   finished, so there is no `success` flag to check. `bngsim.SimulationError`
   means the solver failed. `bngsim.StopConditionMet` carries the result a stop
   condition cut short. Read warnings; don't silence them. `result.notices`
   keeps every warning of the run, whatever the warning filters showed.

```python
model.reset()
first = sim.run(t_span=(0, 5), n_points=6)
second = sim.run(t_span=(5, 10), n_points=6)  # continues: starts at t = 5
assert first.started_from == "initial" and second.started_from == "live"
assert np.isclose(second.observables["A"][0], first.observables["A"][-1])

assert len(sim.run(t_span=(0, 10), n_points=11, start="initial").time) == 11

r1 = ssa.run(t_span=(0, 10), n_points=11, start="initial")  # no seed: one is drawn
r2 = ssa.run(t_span=(0, 10), n_points=11, seed=r1.seed, start="initial")
assert (r1.species == r2.species).all()  # the same seed, the same trajectory
```

## When BNGsim refuses

BNGsim refuses a model or a call it cannot answer correctly, rather than
return a plausible wrong number. The error names the construct: an SBML
symbol with no value (`bngsim.UnderSpecifiedModelError`), a rate law a
stochastic method cannot take (`bngsim.SsaValidationError`), a sensitivity
it cannot compute (`bngsim.SensitivityUnsupportedError`), or an SBML package
or construct it does not support (`bngsim.ModelError`). All derive from
`bngsim.BngsimError`. To check before a run, `model.describe()["methods"]`
gives each method's status and the reasons, from the same checks the runs
make.

Report that **BNGsim cannot simulate the model as written**, and name the
construct the error names. Some refusals mention an environment variable
(`BNGSIM_ALLOW_UNSET_PARAMS`, `BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS` and
others) that lets the run go ahead anyway. Setting it is not a fix. It brings
back the approximation the refusal is there to prevent, and the numbers it
gives must not be reported as the model's. Change the model (give the symbol
a value, or choose the ODE method), or report the refusal.

## What needs what

`bngsim.capabilities()` says what this install can do, and why not when it
can't:

```python
caps = bngsim.capabilities()
print(caps["version"])
for feature, why in caps["missing"].items():
    print(f"{feature}: {why}")
```

| to | needs | capability |
|---|---|---|
| `.net`, SBML, ODE, SSA, PSA, steady state | nothing beyond `pip install bngsim` | |
| BNGL (`.bngl`, `from_bngl`, `run_bngl`) | BioNetGen's `BNG2.pl` and Perl: `pip install "bngsim[bngl]"`, or `BNGPATH` set to a BioNetGen install | `bngl` |
| Antimony | `pip install "bngsim[antimony]"` | `antimony` |
| `result.dataframe` | `pip install "bngsim[pandas]"` | |
| `result.save`, `Result.load` (HDF5) | `pip install "bngsim[hdf5]"` | |
| forward sensitivities | a C compiler: `cc` on macOS and Linux, `cl` or `gcc` on Windows, or `$CC`. BNGsim compiles the model's derivatives; without one, `bngsim.CodegenToolchainError`. `capabilities()` does not check for it | |
| network-free methods `nf`, `rm` | the engines are in the wheels; each run needs `xml_path=`, the model's BioNetGen XML, which BNG2.pl's `writeXML()` action writes from the `.bngl` | `nfsim`, `rulemonkey` |

## If you know another tool

This table gives what a user of another tool writes, and the BNGsim call to
write instead. The first column is one of two things. It is either the other
tool's habit tried on a BNGsim object (`bngsim.loada` for Tellurium's
`te.loada`, `sim.simulate` for RoadRunner's `r.simulate`), which raises an
error that says the same thing, or a call on the other tool's own objects, as
that tool spells it.

<!-- translation-table: generated from bngsim._foreign_names; do not edit -->

**libRoadRunner**

| you write | in BNGsim | note |
|---|---|---|
| `bngsim.RoadRunner` | `bngsim.Simulator(bngsim.Model.load(path))` | A Model holds the network and its parameters; a Simulator runs it. |
| `model.getFloatingSpeciesIds` | `model.species_names` |  |
| `model.getGlobalParameterIds` | `model.param_names` | model.primary_param_names leaves out the ones the model derives from others. |
| `model.getValue` | `model.get_param(name)` | model.get_concentration(name) reads a species. |
| `model.setValue` | `model.set_param(name, value)` | model.set_concentration(name, value) sets a species. |
| `model.resetAll` | `model.reset()` | It resets the species only; parameters keep the values set since the model was loaded. Set them back with model.set_params({...}). |
| `model.getInfo` | `model.describe()` | print(model.describe()) is the readable form. |
| `model.getCompartmentIds` | `model.describe(limit=None).get("compartments", [])` | Each entry is a dict, its name under "name". An SBML or Antimony model has compartments; a .net or BNGL file has none, and the list is empty. |
| `model.getReactionIds` | `model.describe(limit=None)["reactions"]` | Each entry is a dict; for an SBML or Antimony model, the file's reactions as written, each with its "id". The channels BNGsim runs can be more or fewer: n_reactions on the model counts those. |
| `model.getNumReactions` | `model.describe()["counts"]["reactions"]` | The model's reactions as its file has them, as RoadRunner counts them. n_reactions on the model counts the channels BNGsim runs, which can be more or fewer. |
| `sim.simulate` | `sim.run(t_span=(t0, t1), n_points=n)` | BNGsim has no simulate(). n_points counts the output points, the first included: BioNetGen's n_steps=10 and RoadRunner's simulate(0, 10, 11) are both n_points=11. |
| `sim.reset` | `sim.model.reset()` | Resetting the species is the Model's: it returns them to their initial conditions, or to the state an unlabeled model.save_concentrations() saved. |
| `sim.resetAll` | `sim.model.reset()` | It resets the species only; parameters keep the values set since the model was loaded. Set them back with sim.model.set_params({...}). |
| `sim.steadyState` | `sim.steady_state()` |  |
| `sim.getInfo` | `sim.model.describe()` | print(sim.model.describe()) is the readable form. |
| `sim.getCompartmentIds` | `sim.model.describe(limit=None).get("compartments", [])` | Each entry is a dict, its name under "name". An SBML or Antimony model has compartments; a .net or BNGL file has none, and the list is empty. |
| `sim.getReactionIds` | `sim.model.describe(limit=None)["reactions"]` | Each entry is a dict; for an SBML or Antimony model, the file's reactions as written, each with its "id". The channels BNGsim runs can be more or fewer: n_reactions on the model counts those. |
| `sim.getNumReactions` | `sim.model.describe()["counts"]["reactions"]` | The model's reactions as its file has them, as RoadRunner counts them. n_reactions on the model counts the channels BNGsim runs, which can be more or fewer. |
| `sim.getSteadyStateValues` | `sim.steady_state().concentrations` | steady_state() returns a SteadyStateResult; ss[name] reads one species. |
| `sim.gillespie` | `bngsim.Simulator(model, method="ssa").run(t_span=(t0, t1), n_points=n)` | The method is chosen when the Simulator is built; Simulator.METHODS lists them. |
| `sim.setIntegrator` | `bngsim.Simulator(model, method="ssa")` | The method is chosen when the Simulator is built; Simulator.METHODS lists them. |
| `sim.integrator` | `sim.method` | Choose the method with bngsim.Simulator(model, method=...), and the tolerances with sim.set_tolerances(rtol=..., atol=...). |
| `sim.setSeed` | `sim.run(seed=n)` | Each run takes its own seed; result.seed reports the one used. |
| `sim.getFloatingSpeciesIds` | `sim.model.species_names` |  |
| `sim.getGlobalParameterIds` | `sim.model.param_names` |  |
| `sim.getValue` | `sim.model.get_param(name)` | sim.model.get_concentration(name) reads a species. |
| `sim.setValue` | `sim.model.set_param(name, value)` | sim.model.set_concentration(name, value) sets a species. |
| `sim.timeCourseSelections` | `result.observables[name]` | A run records every species, observable and expression; select from its Result: result.species[name], result.observables[name], result.expressions[name]. |
| `result.colnames` | `result.species_names` | result.observable_names names the observable columns, and result.as_roadrunner().colnames gives RoadRunner's. |
| `sim.run(end=...)` | `t_span=(t_start, end)` | The interval is one (t_start, t_end) tuple. |
| `sim.run(points=...)` | `n_points=points` | Both count the output points, the first included. |
| `sim.run(steps=...)` | `n_points=steps + 1` | RoadRunner's steps counts the intervals between output points; n_points counts the points, the first included. |
| `sim.run(integrator=...)` | `bngsim.Simulator(model, method=...)` | The method is chosen when the Simulator is built; Simulator.METHODS lists them. |
| `sim.run(selections=...)` | `result.observables[name]` | run() records every species, observable and expression; select from its Result: result.species[name], result.observables[name], result.expressions[name]. |
| `model.k1 = value` | `model.set_param("k1", value)` | libRoadRunner sets a parameter as an attribute; on a BNGsim Model the write fails with Python's own AttributeError. A species: model.set_concentration(name, value). |
| `model.k1` | `model.get_param("k1")` | A species: model.get_concentration(name). |
| `model["k1"] = value` | `model.set_param("k1", value)` | A species: model.set_concentration(name, value). |
| `model["S1"]` | `model.get_concentration("S1")` | A parameter: model.get_param(name). |
| `sim.k1 = value` | `sim.model.set_param("k1", value)` | libRoadRunner's one object is both the model and its simulator; in BNGsim, parameters and species are the Model's, and sim.model is it. |

**Tellurium**

| you write | in BNGsim | note |
|---|---|---|
| `bngsim.loada` | `bngsim.Model.from_antimony_string(text)` |  |
| `bngsim.loadAntimonyModel` | `bngsim.Model.from_antimony_string(text)` | bngsim.Model.from_antimony(path) reads a file. |
| `bngsim.loadSBMLModel` | `bngsim.Model.from_sbml(path)` | bngsim.Model.from_sbml_string(text) reads SBML text. |
| `result.plot` | `result.dataframe.plot(x="time")` | BNGsim draws no plots; this needs pandas and matplotlib. |

**AMICI**

| you write | in BNGsim | note |
|---|---|---|
| `amici.sim.sundials.run_simulation(model, solver)` | `sim.run(sample_times=times)` | AMICI before 1.0: amici.runAmiciSimulation. There is no import or compile step: bngsim.Model.load(path) reads the SBML, and a Simulator runs it. AMICI integrates from t0; a BNGsim run starts at the first sample time, so put t0 first. |
| `model.set_timepoints(times)` | `sim.run(sample_times=times)` | AMICI before 1.0: setTimepoints. The run starts at the first sample time, so put the start time first. |
| `model.set_free_parameter_by_name(name, value)` | `model.set_param(name, value)` | AMICI before 1.0: setParameterByName. |
| `solver.set_sensitivity_order(SensitivityOrder.first)` | `bngsim.Simulator(model, sensitivity_params=[...])` | AMICI before 1.0: setSensitivityOrder. result.sensitivities is (n_times, n_species, n_params); AMICI's rdata.sx is (n_times, n_params, n_species). |
| `rdata.x` | `result.species` |  |

**COPASI (basico)**

| you write | in BNGsim | note |
|---|---|---|
| `basico.load_model(path)` | `bngsim.Model.load(path)` |  |
| `basico.run_time_course(duration=d, intervals=n)` | `sim.run(t_span=(0, d), n_points=n + 1)` | intervals counts the steps between output points. basico starts each time course from the initial values; a BNGsim run continues from the last one unless start="initial". |
| `basico.run_time_course(method="stochastic")` | `bngsim.Simulator(model, method="ssa")` | The method is chosen when the Simulator is built; Simulator.METHODS lists them. |
| `basico.set_parameters(name, initial_value=v)` | `model.set_param(name, v)` |  |
| `basico.set_species(name, initial_concentration=c)` | `model.set_concentration(name, c)` |  |
| `basico.run_steadystate()` | `sim.steady_state()` |  |

**BNGL actions**

| you write | in BNGsim | note |
|---|---|---|
| `bngsim.simulate` | `bngsim.Simulator(bngsim.Model.load(path)).run(t_span=(t0, t1), n_points=n)` | A Model holds the network; a Simulator runs it. n_points counts the output points, the first included: BioNetGen's n_steps=10 and RoadRunner's simulate(0, 10, 11) are both n_points=11. |
| `model.setParameter` | `model.set_param(name, value)` |  |
| `model.setConcentration` | `model.set_concentration(name, value)` |  |
| `model.saveConcentrations` | `model.save_concentrations()` |  |
| `model.resetConcentrations` | `model.reset()` | model.restore_concentrations(label) restores a state saved with model.save_concentrations(label). |
| `sim.run(t_end=...)` | `t_span=(t_start, t_end)` | The interval is one (t_start, t_end) tuple. |
| `sim.run(t_start=...)` | `t_span=(t_start, t_end)` | The interval is one (t_start, t_end) tuple. |
| `sim.run(n_steps=...)` | `n_points=n_steps + 1` | BioNetGen's n_steps counts the intervals between output points; n_points counts the points, the first included, so n_steps=10 is n_points=11. |
| `simulate({method=>"ode", t_end=>T, n_steps=>N})` | `bngsim.Simulator(model, method="ode").run(t_span=(0, T), n_points=N + 1)` | bngsim.run_bngl(path) runs the actions a .bngl file holds, as written. |
| `simulate({..., continue=>1})` | `sim.run(t_span=(t_end, t_next), n_points=n)` | Every run goes on from the model's state; a t_span that starts where the last one ended continues its clock too. |
| `parameter_scan({parameter=>"k", par_min=>a, par_max=>b, n_scan_pts=>n, log_scale=>1, t_end=>T, n_steps=>N})` | `sim.parameter_scan("k", par_min=a, par_max=b, n_scan_pts=n, log_scale=True, t_span=(0, T), n_points=N + 1)` | It returns one Result per point and writes no files. Without t_span and n_points it runs (0, 100) at 101 points. |
| `generate_network({overwrite=>1})` | `bngsim.Model.from_bngl(path)` | It runs BNG2.pl's generate_network and loads the network; a repeated load of an unchanged file reads a cache. |

**PySB**

| you write | in BNGsim | note |
|---|---|---|
| `ScipyOdeSimulator(model, tspan=t).run()` | `bngsim.Simulator(model).run(sample_times=t)` | Export the PySB model first: pysb.export.export(model, 'bngl') returns BNGL text, which bngsim.Model.from_bngl(path) reads from a file. |
| `ScipyOdeSimulator(model).run(param_values={...})` | `model.set_params({name: value, ...})` | Set the parameters on the Model, then run. PySB's values hold for that run only; the Model keeps these. |
| `BngSimulator(model).run(method="ssa", n_runs=n)` | `bngsim.Simulator(model, method="ssa").run_replicates(n, t_span=(t0, t1), n_points=m)` |  |
| `res.observables["obs"]` | `result.observables["obs"]` | A numpy column. |

**SciPy**

| you write | in BNGsim | note |
|---|---|---|
| `result.t` | `result.time` |  |
| `result.y` | `result.species` | Its shape is (n_times, n_species), the transpose of SciPy's sol.y: result.species.T is SciPy's layout. |
| `result.success` | — | run() raises bngsim.SimulationError when a run fails, so a Result it returns is a run that finished. A stop condition that ends a run early raises bngsim.StopConditionMet, which carries the cut Result. |
| `result.status` | — | run() raises bngsim.SimulationError when a run fails, so a Result it returns is a run that finished. A stop condition that ends a run early raises bngsim.StopConditionMet, which carries the cut Result. |
| `result.message` | — | run() raises bngsim.SimulationError when a run fails, so a Result it returns is a run that finished. A stop condition that ends a run early raises bngsim.StopConditionMet, which carries the cut Result. |
| `sim.run(t_eval=...)` | `sample_times=t_eval` | SciPy integrates from t_span[0]; a BNGsim run starts at the first sample time, so put the start time first. |
| `sim.run(method=...)` | `bngsim.Simulator(model, method=...)` | The method is chosen when the Simulator is built; Simulator.METHODS lists them. |
| `sim.run(y0=...)` | `model.set_concentration(name, value)` | A run starts from the model's state: set it before run(), all of it at once with model.set_state(values). |

**pandas**

| you write | in BNGsim | note |
|---|---|---|
| `result.to_dataframe` | `result.dataframe` | It is a property, read without parentheses: time and one column per observable. It needs pandas. |
| `result.to_pandas` | `result.dataframe` | It is a property, read without parentheses: time and one column per observable. It needs pandas. |
| `result.as_dataframe` | `result.dataframe` | It is a property, read without parentheses: time and one column per observable. It needs pandas. |
| `result.to_df` | `result.dataframe` | It is a property, read without parentheses: time and one column per observable. It needs pandas. |
| `result.df` | `result.dataframe` | It is a property, read without parentheses: time and one column per observable. It needs pandas. |
| `result.to_numpy` | `result.species` | result.time, result.species, result.observables and result.expressions are numpy arrays. |
| `result.A` | `result.observables["A"]` | A species: result.species["A"]; an expression: result.expressions["A"]. |

<!-- end of translation-table -->

## Example models, and models built in Python

`bngsim.examples` holds small models that ship with BNGsim, each with its
exact solution. Use one to try BNGsim without a model file of your own, or to
check that BNGsim works on this machine. `bngsim.examples.formats(name)` lists
the formats each comes in. `bngsim.ModelBuilder` builds a model in Python.

```python
print(bngsim.examples.names())  # ['binding', 'decay', 'chain', 'birth_death', 'bolus', 'schlogl']
example = bngsim.examples.load("binding")  # A + B <-> C
res = bngsim.Simulator(example).run(t_span=(0, 10), n_points=101)
exact = bngsim.examples.exact("binding", res.time)
print(np.allclose(res.observables["C"], exact["C"], rtol=1e-4))  # True

b = bngsim.ModelBuilder()
b.add_parameter("k", 0.5)
b.add_species("A", 100)  # each species is also an observable
b.add_reaction(["A"], [], rate="k")  # rate: mass action; law=: the whole rate
built = b.build()
print(bngsim.Simulator(built).run(t_span=(0, 2), n_points=3).observables["A"][-1].round(2))  # 36.79
```

## Check a result before reporting it

- **A conservation law holds.** In this model `A + B` stays 100. A network
  with a conserved total should conserve it to within the tolerance.
- **The answer does not move with the tolerance.** Run again with `rtol` and
  `atol` ten or a hundred times smaller. A number that moves was not resolved.
- **A closed form or a steady state agrees**, where one is known.
- **A stochastic answer comes from an ensemble.** One trajectory is one draw.
  Report a mean or a distribution over replicates, with the number of
  replicates and the seed.
- **The run started where it was meant to:** `result.started_from`.
- **What the numbers are and how they were made:** `result.record` says it
  from the result, not from memory of the script; `result.notices` lists what
  the run warned of.

```python
model.reset()
r = sim.run(t_span=(0, 10), n_points=101)
total = r.observables["A"] + r.observables["B"]
assert np.allclose(total, 100.0, rtol=1e-6)

tight = sim.run(t_span=(0, 10), n_points=101, rtol=1e-10, atol=1e-12, start="initial")
assert np.allclose(r.observables, tight.observables, rtol=1e-5, atol=1e-6)

kf, kr = 0.5, 0.25
closed = 100 * (kr + kf * np.exp(-(kf + kr) * r.time)) / (kf + kr)
assert np.allclose(r.observables["A"], closed, rtol=1e-5)

mean_a = reps.observables[:, -1, 0].mean()  # 50 SSA replicates at t = 10
assert abs(mean_a - closed[-1]) < 5 * np.sqrt(closed[-1])  # within the noise

rec = r.record  # json.dumps(rec) writes it
print(rec["run"]["method"], rec["run"]["rtol"], rec["backend"]["jacobian"])  # ode 1e-08 analytical
print(r.notices)  # []
```

## Report a suspected bug

Report it at <https://github.com/lanl/bngsim/issues>, with what a maintainer
needs to run it again:

- `bngsim.capabilities()["version"]` and `["build"]["commit"]`, the Python
  version and the operating system;
- the model file, or the smallest one that shows the problem;
- the calls, as code that runs, with every `seed`, and `result.record`;
- what came back, and what was expected and why: a closed form, another
  simulator's answer (name it and its version), or a conservation law.

A refusal (above) is not a bug unless the model is valid and BNGsim names the
wrong construct.
