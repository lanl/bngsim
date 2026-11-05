![bngsim](docs/bngsim-coyote-logo/bngsim-readme-card-sienna.svg)

# bngsim

**Embeddable simulation engine for biochemical reaction networks.**

`bngsim` is a high-performance C++ simulation kernel with Python bindings that replaces
BioNetGen's subprocess-based `run_network` driver. The library also includes two network-free simulators. It loads BioNetGen `.net` and `.xml` files, runs deterministic
and stochastic simulations in-process, and returns results as NumPy arrays —
no file I/O, no subprocess spawning, no Perl dependency.

> **Status: alpha (0.x).** bngsim is checked every night against BioNetGen,
> libRoadRunner, AMICI and the SBML test suite, but it is early software, and
> bugs that give a wrong answer without an error are still open. They carry the
> [`correctness concern`](https://github.com/lanl/bngsim/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22correctness%20concern%22)
> label. Before you rely on a result for a new model, check it against a second
> engine or a tighter tolerance, and please report what you find.

## Highlights

- **Fast** — in-process execution with the GIL released during simulation; thread-parallel batch sweeps
- **No Perl at runtime** — `import bngsim`; no `BNGPATH`, no Perl (loading `.bngl` aside). A C compiler (`cc`) is recommended: models of 256 or more species compile with it and fall back to the interpreter without it. Forward sensitivities need a codegen backend: `cc`, or the opt-in MIR JIT in a source build ([compiled RHS](docs/user-guide/codegen.md))
- **Modern SUNDIALS** — v7.x CVODE/CVODES with re-entrant `SUNContext`
- **Multi-format** — loads BioNetGen `.net` and `.xml`, Antimony (`.ant`), and SBML (`.xml`) models [SBML Test Suite results](https://sbml.bioquant.uni-heidelberg.de/Submission/Details/72), plus rule-based BNGL (`.bngl`) via BNG2.pl with `bngsim[bngl]`
- **Rich results** — NumPy arrays, named observable access, pandas/xarray, HDF5 save/load
- **Standards interchange** — `.net`/cBNGL ⇄ SBML and SED-ML + OMEX packaging, each conversion checked by a round-trip validation ladder (a check, not a proof; see [interchange](docs/user-guide/interchange.md))
- **Gradient-ready** — CVODES forward sensitivities, Fisher information
- **Cross-checked** — in the nightly sweep 589 of 592 BNGL models agree with `run_network` to 1e-4 (ODE), and results are also checked against libRoadRunner, AMICI and the SBML semantic test suite

## Installation

```bash
pip install bngsim
```

Prebuilt wheels ship for common platforms, with the sparse (KLU) solver bundled.
See the [installation guide](docs/installation.md) for source builds, the KLU
dependency, and optional extras.

## Quickstart

```python
import bngsim

# Load a model (BioNetGen .net/.xml, Antimony, or SBML) and run an ODE simulation
model = bngsim.Model.from_net("model.net")
sim = bngsim.Simulator(model, method="ode")
result = sim.run(t_span=(0.0, 100.0), n_points=1001)

result.time               # (1001,) NumPy array of time points
result.observables["A"]   # trajectory of observable "A"
result.dataframe          # pandas DataFrame of time + all observables
```

Other formats load through the sibling constructors — `Model.from_sbml`,
`Model.from_antimony`, and their `*_string` variants.

Stochastic (`method="ssa"` / `"psa"`), network-free (NFsim / RuleMonkey), sensitivity
analysis, steady-state, and events are all covered in the [quickstart](docs/quickstart.md)
and [user guide](docs/index.md).

## Documentation

Full documentation is hosted at **[bngsim.readthedocs.io](https://bngsim.readthedocs.io)**
and lives in [`docs/`](docs/):

- [Installation](docs/installation.md) · [Quickstart](docs/quickstart.md)
- User guide — [loading models](docs/user-guide/loading-models.md),
  [simulation](docs/user-guide/simulation.md),
  [network-free](docs/user-guide/network-free.md),
  [results](docs/user-guide/results.md),
  [events](docs/user-guide/events.md),
  [sensitivities](docs/user-guide/sensitivities.md),
  [steady state](docs/user-guide/steady-state.md),
  [compiled RHS and its artifact cache](docs/user-guide/codegen.md),
  [SBML/SED-ML/OMEX interchange](docs/user-guide/interchange.md)
- Reference — [API](docs/reference/api.md) · [expression language](docs/reference/expressions.md)
- [Architecture](docs/about/architecture.md) · [benchmarks & validation](docs/about/benchmarks.md)

The package ships console scripts for model conversion (`bngsim-sbml2net`,
`bngsim-net2sbml`, `bngsim-sbml2bngl`, `bngsim-validate-conversion`, `bngsim-omex`;
see [interchange](docs/user-guide/interchange.md)) and `bngsim-cache` (equivalently
`python -m bngsim.cache`), which reports on and prunes the compiled-artifact cache
that codegen fills, which nothing empties on its own. See
[managing the artifact cache](docs/user-guide/codegen.md#managing-the-artifact-cache).

## Benchmarks & validation

`bngsim` beats `run_network` on every SSA/PSA model measured (geometric-mean speedup
~8.7× SSA), agrees with libRoadRunner to ~1e-4 on 1,237 of the 1,323 BioModels ODE
models, and scores 1577 `Match` on the SBML semantic test suite with **zero**
wrong-but-plausible answers.
The numbers, methodology, and reproducible harnesses are in
[benchmarks & validation](docs/about/benchmarks.md), with the committed cross-engine
snapshots under [`parity_checks/`](parity_checks/). To re-run the suites and reproduce the
numbers — the pinned-fetch model, corpus provenance, tool versions, and a no-download smoke
path — see [`REPRODUCING.md`](REPRODUCING.md). The supported-construct matrix is in
[`SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md).

Those results are re-checked every night: a [scheduled workflow](.github/workflows/nightly-parity.yml)
runs the parity sweeps and alerts on any change in correctness or solver work
([`parity_checks/nightly/`](parity_checks/nightly/README.md)). To check whether a change of
yours slowed bngsim down, `benchmarks/perf_ab.py` times two builds against each other on your
machine (see [Checking performance and parity](CONTRIBUTING.md#checking-performance-and-parity)).

## Contributing

Build-from-source, test, and CI instructions are in [`CONTRIBUTING.md`](CONTRIBUTING.md)
and the [development docs](docs/development/building.md). Release history is in
[`CHANGELOG.md`](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE) for the full text and the Triad/LANL copyright notice.

Third-party code and model/test-data corpora that BNGsim redistributes are
listed, with their licenses, in [NOTICE](NOTICE).

© 2026. Triad National Security, LLC. All rights reserved. This is a Los Alamos National
Laboratory open-source release; LANL software release reference **O5098**.

## Citation

If you use `bngsim` in your research, please cite:

> Preprint coming soon!

## Acknowledgments

BNGsim was developed and validated against RoadRunner, AMICI, COPASI, BioNetGen,
and the SBML/DSMTS test suites. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for
citations.
