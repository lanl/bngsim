# `parity_checks/` — bngsim engine-correctness suites

Proves bngsim is a reliable functional replacement for RoadRunner, the legacy
BioNetGen stack, and AMICI. This directory validates the **engines** (via thin
scripts ≈ near-direct engine calls) against non-bngsim references, and publishes
**golden reference data** that downstream consumers (PyBioNetGen, PyBNF)
regenerate through their own bridges. Correctness is kept strictly separate from
benchmarking (`benchmarks/`).

## Suites

| suite | input | methods | reference engine |
|---|---|---|---|
| [`rr_parity`](rr_parity/) | SBML | ODE, SSA | libRoadRunner |
| [`bng_parity`](bng_parity/) | BNGL / .net | ODE, SSA, NF | legacy BNG stack (BNG2.pl / run_network / NFsim) |
| [`amici_parity`](amici_parity/) | SBML | sensitivities / gradients | AMICI |
| [`_core`](_core/) | — | — | shared contracts only, **no models** |

Each suite is self-contained: its own vendored `models/`, a spec manifest, an
adapter, a results report, and golden references. BNGL/.net model bytes are
vendored (real files, not symlinks); the exception is rr_parity's BioModels
inputs — both the ~270 MB SBML and the ~49 MB curated SED-ML are gitignored and
re-placed locally by `materialize.py` from their sources (a working checkout is
expected to hold local copies of both). See each suite's README.

## Corpus provenance

Fetched/vendored corpora carry a **provenance manifest** — one receipt per model
recording its pinned upstream version, `sha256`, and license — so a re-fetch is
verifiable byte-for-byte. The contract is [`manifest_schema.md`](manifest_schema.md)
(machine-checkable [`manifest.schema.json`](manifest.schema.json), JSON Schema
draft 2020-12; it structurally rejects unpinned `@main`/`latest` sources). See
e.g. `rr_parity/temp_biomodels_manifest.json`.

## Workflow — running a check and reading the result

Every suite has the same shape: a **runner** executes bngsim and the reference
engine on every model in a committed **manifest** (`jobs.json` / `ode_jobs.json`),
compares them with the shared `_core.differ` oracle, and writes a **report** (JSON);
a **matrix generator** renders the report to a single-page HTML table.

```
# bng_parity (BNGL vs the legacy BNG stack) — needs $BNGPATH (BNG2.pl + run_network + NFsim)
export BNGPATH=/path/to/BioNetGen-2.9.3
python bng_parity/bng_ode_run.py --workers 4                   # -> runs/report_ode.json
python bng_parity/generate_bng_matrix.py runs/report_ode.json  # -> runs/bng_matrix_ode.html

# rr_parity (SBML vs RoadRunner) — pure Python, no external engine
python rr_parity/rr_run.py --workers 8                         # -> runs/report_ode.json
python rr_parity/generate_matrix.py
```

**Reading the matrix** — one row per model: **PASSED** (bngsim matched the
reference within tolerance), **FAILED** (a real, scoring divergence), **KNOWN REF /
KNOWN ARTIFACT** (a DIFF adjudicated as a reference-engine bug or a benign artifact,
*not* a bngsim defect — non-scoring; the row comment carries the evidence), or
**TRIAGED / REFUSED** (one engine errored, or the model couldn't be run).

### Parity vs golden — two different questions

- **Parity** (the matrix): *"does bngsim match another engine — RoadRunner / legacy
  BNG / AMICI — right now?"* Requires that engine installed.
- **Golden** (`*/golden/`, emitted by `parity_golden.py` / `rr_golden.py`): a
  fingerprint of **bngsim's own output** at a known-good version. It is *not* a
  cross-engine comparison — it lets a **downstream consumer** (PyBNF, PyBioNetGen)
  verify *it* drives bngsim correctly by regenerating the golden through its own
  bridge with **no** RR/BNG/AMICI install, and it is a regression tripwire (if
  bngsim's output changes, the golden changes).

## Settled principles

- **Layering.** Here = engine correctness (vs RR / legacy / AMICI). Consumers =
  bridge+engine end-to-end, checked by **regenerating our golden references** —
  so a consumer needs only a pinned bngsim, no RR/BNG/AMICI install.
- **Three files, separate.** `_core` pins them: the **manifest** (spec —
  committed, stable, one job per line), the **report** (results — regenerated
  per run, version-stamped), and the **golden** references (checksum +
  fingerprint; full trajectories for a representative subset only).
- **One outcome taxonomy** (`_core.Outcome`) across all suites + consumers.
- **Cross-engine = numeric tolerance**; consumer regeneration = byte-identical
  within a pinned (version, platform, seed) cell, fingerprint as fallback.
- **Pin versions** everywhere; every report stamps observed engine versions.
- **Fitting/sampling parity is PyBNF's, not ours** — `__FREE` fitting templates
  are not parity corpus.

## Reproduce-before-trust (issue #125)

These suites and the `dev/investigations/*` oracles import the compiled
`_bngsim_core` to decide *"is bngsim correct?"* and commit the verdict into the
harness as a disposition. But the editable install loads a **separately-built**
`.so` that does **not** auto-rebuild (`editable.rebuild = false`, #23). So a
forgotten rebuild means the oracle adjudicates **old C++** while you read the new
C++ in your editor — exactly the stale-binary near-miss that produced a false
"warm-start rounding bug" verdict on #118. Before any local finding flips a
harness disposition:

1. **Print the binary identity.** Every oracle calls
   `bngsim._build_provenance.print_identity()` at startup; confirm the banner
   says `fresh` (not `STALE`) and that `built=<commit>` matches the source you
   are reasoning about. A stale binary aborts the pytest suite by default.
2. **Rebuild from the committed state.** `python scripts/rebuild_editable.py`
   (the only supported editable rebuild; it reconfigures + rebuilds
   `_bngsim_core` + reinstalls). The guard refuses to declare *fresh* until the
   loaded `.so` is newer than every `src/**`, `include/**`, and `third_party/**`
   source.
3. **Re-run the committed artifact on the fresh core.** A finding only lands once
   the *committed* oracle (not a throwaway local edit) reproduces it on an
   in-sync binary. If it doesn't reproduce, suspect the binary before the model.

Escape hatches exist for the rare false positive — `BNGSIM_ALLOW_STALE_CORE=1`
(proceed with a warning) and `BNGSIM_NO_BUILD_CHECK=1` (skip the guard) — but a
disposition must never be committed from a run that used one.

## Status

- `_core` — contracts locked; the single verdict kernel (`differ`) for all suites.
- `rr_parity` — operational: ODE (`rr_run.py`) + SSA (`ssa_screen.py`, with its own
  z-gate + third-oracle attribution) + golden. bngsim matches RoadRunner across the
  BioModels corpus (0 bngsim-attributable diffs).
- `bng_parity` — operational: ODE/SSA/NF timing+parity matrix (`bng_ode_run.py` /
  `bng_stoch_run.py` → `generate_bng_matrix.py`) + golden (`parity_golden.py`). The
  legacy `parity_core_run.py` / `parity_run.py` correctness suite and `parity_diff.py`'s
  duplicate verdict were retired in **GH #69** (single-source on `_core.differ`; the
  matrix is now the adjudication home).
- `amici_parity` — operational (ODE): `amici_run.py` → `generate_amici_matrix.py`.
  AMICI-side limitations are catalogued in `amici_parity/AMICI_KNOWN_ISSUES.md`.
