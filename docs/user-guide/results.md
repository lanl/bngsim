# Working with results

## Named observable access

```python
# Access by name (returns 1D array for that observable)
a_total = result.observables["A_total"]

# Access as pandas DataFrame
df = result.dataframe  # requires: pip install bngsim[pandas]
print(df.head())

# AMICI-style labeled xarray access (requires: pip install xarray)
result.xr.species          # DataArray, dims (time, state)
result.xr.observables      # DataArray, dims (time, observable)
result.xr.sensitivities    # DataArray, dims (time, state, parameter)
result.xr.observables.sel(observable="A_total")
result.xr.sensitivities.sel(parameter="k1", state="A")

# One-shot xarray.Dataset bundling every field (shared time coord)
ds = result.to_xarray()
ds.to_netcdf("result.nc")  # archive via xarray's writer
```

Dimension names follow AMICI's convention (`state` rather than `species`)
so code written against `rdata.xr.x.sel(state=...)` works against a
bngsim Result. `custom_attrs` and the stochastic `seed` (when set)
propagate to `ds.attrs`. `sensitivities` / `sensitivities_ic` appear
only when the simulator was run with `sensitivity_params=` /
`sensitivity_ic=`; requesting them on a plain Result raises a clear
`AttributeError`.

## Save and load results (HDF5)

```python
# Requires: pip install bngsim[hdf5]
result.save("results.h5")
loaded = bngsim.Result.load("results.h5")

# Custom metadata
result.custom_attrs["experiment"] = "dose_response_2026"
result.save("results_with_meta.h5")
```

## Export results to text files

bngsim provides three text-export methods on `Result`; pick by audience.

```python
# BNG-native (#-prefixed header, fixed-width, space-padded)
result.to_gdat("output.gdat")   # observable time courses
result.to_cdat("output.cdat")   # species concentrations

# Plain CSV/TSV (no '#' prefix, named columns, choose delimiter)
result.to_csv("output.csv")                       # observables, comma
result.to_csv("output.tsv", sep="\t")             # observables, tab
result.to_csv("species.csv", kind="species")      # species block

# pandas one-liner (requires pandas)
result.dataframe.to_csv("output.csv", index=False)
```

`to_csv` is the natural entry point for SBML/RoadRunner/Tellurium consumers
who expect plain delimited text without BNG-specific formatting. The first
column is `time` (unless `include_time=False`); the remaining columns carry
the observable or species names from the in-memory `Result`, so the file
can be loaded back with `pandas.read_csv` or `numpy.loadtxt` without
extra parsing.

**What each writer exports — and what it loses.** All three text writers
work on any single-sim `Result` from any backend (ODE / SSA / PSA / NFsim /
RuleMonkey) and on results loaded back from HDF5. None of them archives
expressions, sensitivities, solver stats, custom attrs, or the stochastic
seed — use [HDF5](#save-and-load-results-hdf5) for a lossless capture.

| Writer | Columns | Header | Delimiter | Captures |
|---|---|---|---|---|
| `to_gdat(path)` | `time`, observables | `#`-prefixed, space-padded | fixed-width spaces | observables only |
| `to_cdat(path)` | `time`, species | `#`-prefixed, space-padded | fixed-width spaces | species only |
| `to_csv(path, kind=..., sep=...)` | `time`, observables *or* species | plain `time,A,B,...` | caller-chosen char | one block at a time |
| `result.dataframe.to_csv(path)` | `time`, observables | pandas | pandas | observables only |
| `result.save(path)` (HDF5) | n/a | n/a | n/a | **everything** — time/species/observables/expressions/sensitivities/solver stats/seed/custom attrs |

Batch (3-D) results from `run_batch(squeeze=True)` are intentionally not
written to text files by these methods; iterate the per-replicate
`Result` list (the `squeeze=False` shape) and call the writer on each one,
or use HDF5 for the assembled batch.
