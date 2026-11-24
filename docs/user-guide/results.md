---
summary: "The Result object: arrays by name, pandas and xarray views, saving to HDF5, exporting BioNetGen .gdat and .cdat text files, and per-reaction statistics from SSA runs."
---

# Working with results

A run returns a `Result`. Its arrays are NumPy arrays that also take a name.

The examples on this page read one run of a small model, with sensitivities
to its two rate constants: A is made at rate `k1` and decays at rate `k2`.

```python
from pathlib import Path

import bngsim

# 0 -> A -> 0, in BioNetGen's network (.net) format.
Path("model.net").write_text(
    """\
begin parameters
    1 k1 10
    2 k2 0.5
end parameters
begin species
    1 A() 0
end species
begin reactions
    1 0 1 k1
    2 1 0 k2
end reactions
begin groups
    1 A_total 1
end groups
""",
    encoding="utf-8",
)

model = bngsim.Model.load("model.net")
sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1", "k2"])
result = sim.run(t_span=(0, 20), n_points=21)
```

## Read a result by name

```python
# One observable, as a 1D array
a_total = result.observables["A_total"]

# Species by name too, in the column position
a = result.species["A()"]           # the column, as result.species[:, index]
a_end = result.species[-1, "A()"]   # its last value

# A pandas DataFrame
df = result.dataframe  # requires: pip install bngsim[pandas]
print(df.head())

# Labeled xarray views, as AMICI has (requires: pip install xarray)
result.xr.species          # DataArray, dims (time, state)
result.xr.observables      # DataArray, dims (time, observable)
result.xr.sensitivities    # DataArray, dims (time, state, parameter)
result.xr.observables.sel(observable="A_total")
result.xr.sensitivities.sel(parameter="k1", state="A()")

# One xarray.Dataset with every field on a shared time axis
ds = result.to_xarray()
ds.to_netcdf("result.nc")
```

Notes:

- `result.dataframe` is a property, not a method.
- The xarray dimension names follow AMICI (`state`, not `species`), so code
  written for `rdata.xr.x.sel(state=...)` works on a BNGsim result.
- `custom_attrs` and the stochastic `seed` (when set) go into `ds.attrs`.
- `sensitivities` and `sensitivities_ic` exist only when the run had
  `sensitivity_params=` or `sensitivity_ic=`. Asking for them otherwise raises
  `AttributeError`.

## Save and load results (HDF5)

HDF5 keeps everything in a result.

```python
# Requires: pip install bngsim[hdf5]
result.save("results.h5")
loaded = bngsim.Result.load("results.h5")

# Your own metadata
result.custom_attrs["experiment"] = "dose_response_2026"
result.save("results_with_meta.h5")
```

The file keeps each species' compartment size and, where a size changes, the
volume factor at each sample. `as_roadrunner` needs these to report amounts.
A file in the older format 2 does not have them. On such a file,
`as_roadrunner` refuses an amount selector (`"S"`) and warns on a
concentration one (`"[S]"`). Multiply `"[S]"` by the size yourself, or run the
model again and save the new result.

The file also keeps `ic_sensitivity_seed`, `ssa_diagnostics` and
`psa_diagnostics` (for a stacked batch, each run's, as
`psa_diagnostics_per_run`). An older file without them loads with the defaults
a result built from arrays has: the seed `None` (not recorded), SSA counts of 0
with the propensity backend `"unknown"`, and PSA inactive.

## Export to text files

```python
# BioNetGen's format ('#' header, fixed-width columns)
result.to_gdat("output.gdat")   # observables
result.to_cdat("output.cdat")   # species

# Plain CSV or TSV (no '#', named columns)
result.to_csv("output.csv")                       # observables, comma
result.to_csv("output.tsv", sep="\t")             # observables, tab
result.to_csv("species.csv", kind="species")      # species

# With pandas
result.dataframe.to_csv("output.csv", index=False)
```

Use `to_csv` for RoadRunner or Tellurium users, or any tool that expects plain
delimited text. Its first column is `time` (unless `include_time=False`),
and the rest are the observable or species names, so `pandas.read_csv` or
`numpy.loadtxt` reads it back as is.

The text writers work on a single-run `Result` from any method (ODE, SSA, PSA,
NFsim, RuleMonkey) and on results loaded from HDF5. They do not keep
expressions, sensitivities, solver statistics, custom attributes or the seed.
Use [HDF5](#save-and-load-results-hdf5) to keep everything.

| Writer | Columns | Header | Delimiter | Keeps |
|---|---|---|---|---|
| `to_gdat(path)` | `time`, observables | `#`-prefixed, space-padded | fixed-width spaces | observables only |
| `to_cdat(path)` | `time`, species | `#`-prefixed, space-padded | fixed-width spaces | species only |
| `to_csv(path, kind=..., sep=...)` | `time`, observables *or* species | plain `time,A,B,...` | your choice | one block at a time |
| `result.dataframe.to_csv(path)` | `time`, observables | pandas | pandas | observables only |
| `result.save(path)` (HDF5) | n/a | n/a | n/a | everything: time, species, observables, expressions, sensitivities, solver statistics, SSA and PSA diagnostics, seed, custom attributes |

The text writers do not write a stacked batch result, such as
`run_batch(squeeze=True)` returns. Call the writer on each `Result` of the list
(`squeeze=False`), or save the batch to HDF5.

## Per-reaction statistics from an SSA run

An exact SSA run can record, at every output time, each reaction's cumulative
firing count and integrated propensity. These are what a likelihood-ratio
(Girsanov) gradient of an ensemble needs. It is off by default, and turning it
on does not change the trajectory.

```python
model.reset()  # back to the initial counts: the ODE run above moved the model
sim = bngsim.Simulator(model, method="ssa", reaction_stats=True)
batch = sim.run_replicates(200, t_span=(0, 20), n_points=21, seed=1, squeeze=True)

batch.reaction_firing_counts.shape         # (200, 21, n_reactions): N_r(t) per replicate
batch.reaction_propensity_integrals.shape  # (200, 21, n_reactions): ∫₀ᵗ a_r ds
batch.reaction_labels                      # ['R1 (0 -> A())', 'R2 (A() -> 0)']

# Scores for named rate constants, and the ensemble gradient
# d E[observable] / d log10(k) with its standard error.
from bngsim import girsanov

w = girsanov.scores(model, batch, ["k1", "k2"], log10=True)   # (200, 21, 2)
grad, stderr = girsanov.gradient(batch.observables, w)         # (21, n_observables, 2)
```

To check the statistics, use `N_r(t) − ∫₀ᵗ a_r ds`. Its mean across
replicates is zero, so its sample mean should be within a few standard errors
of zero.

`scores` takes mass-action rate laws only. It refuses a model with a
functional or Michaelis-Menten reaction.
