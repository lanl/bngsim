# `biomodels` suite

Benchmarks BNGsim against libRoadRunner and AMICI on the BioModels SBML
corpus. The events tag splits the output: events-tagged models feed the
SBML-events figure (Fig S2), the rest feed the BioModels figure (Fig S1).

> **Distinct from `benchmarks/biomodels_ssa/`.** This directory is the **SBML** BioModels
> corpus for the ODE-parity suite (BNGsim vs RoadRunner/AMICI). The separate
> `benchmarks/biomodels_ssa/` builds the **SSA `.net` / Antimony** pool that feeds the
> `bench_1013_4engine` harness — different corpus, different benchmark family.

## Pipeline

| Stage | Script | Output |
|-------|--------|--------|
| 0 — fetch | `fetch.py` | SBML files in git-ignored `data/sbml_downloads/` (pinned) |
| — pin | `generate_provenance_manifest.py` | `biomodels_provenance.json` (committed) |
| 1 — filter | `filter.py` | `manifest.csv` (committed) |
| 2 — run | `run.py` | `results/coverage.csv` + `figS1`/`figS2` splits |

```sh
python fetch.py            # reconstruct the corpus at its PINNED versions (default)
python filter.py           # partition keep/drop, refresh manifest.csv
python run.py --effort low # coverage + accuracy sweep (cheap subset)
```

## Reproducibility model — no large files in git

The raw SBML corpus (~1,650 files, ~273 MB) is **never committed**. It
lives in a git-ignored `data/sbml_downloads/` directory, reconstructed
locally:

- `fetch.py` — **Stage 0**: reconstructs the corpus at its **pinned
  versions** by default — reads `biomodels_provenance.json` and downloads
  each model via `model/download/<id>.<version>?filename=<member>`, then
  verifies the bytes against the recorded sha256. Deterministic even after
  BioModels revises a model. `--upstream-latest` instead crawls the latest
  of every ODE model (refresh/extend; unpinned, may drift). Idempotent: a
  valid, hash-matching `<id>.xml` on disk is kept. Vendored from the `ssys`
  project's `step1_fetch.py`.
- `biomodels_provenance.json` — **committed**. One provenance receipt per
  `keep` model (schema: `../../../parity_checks/manifest_schema.md`), pinning
  the exact BioModels version + OMEX member whose sha256 matches our tested
  bytes. 1323 models: 1274 pin to the latest version, 49 to the version whose
  archived member (`_url.xml` / `_urn.xml` / author-named) reproduces the
  tested bytes. License: 1011 curated `BIOMD*` are `CC0-1.0`, 312 non-curated
  `MODEL*` are `unknown` upstream. This manifest also covers rr_parity's 332
  `biomodels_dir` SBML (all inside this keep-set). Regenerate with
  `generate_provenance_manifest.py` (metadata-API driven; cached in
  git-ignored `data/`).
- `filter.py` — **Stage 1**: crawls `data/sbml_downloads/`,
  deterministically partitions into keep / drop, tags structural
  features, writes `manifest.csv`.
- `manifest.csv` — **committed**. The reproducible good/bad partition:
  `model_id, verdict, reason, …, tags, sha256`. Re-running `filter.py`
  on the same corpus regenerates it identically; the SHA-256 column
  lets a reviewer confirm byte-identical inputs.

## Stage 1 — filter criteria

`filter.py` drops a model only if it is **un-simulable as an ODE
system** — never on SBML-validation severity (which is
libSBML-version-sensitive):

| reason | meaning |
|--------|---------|
| `no_model` | libSBML produced no Model element |
| `missing_kinetic_law` | a reaction has no `kineticLaw` |
| `zero_dynamics` | no reactions and no rate rules |
| `negative_population` | a species' initial amount/concentration is negative |
| `undefined_symbol` | a math expression references an identifier defined in no namespace |

Events, delays, algebraic rules, and function definitions are **tagged,
not dropped** — BNGsim handles them.

`undefined_symbol` is a **curated manual exclusion** (`MANUAL_DROP` in
`filter.py`), not auto-detected: libSBML validates these models clean, but the
referenced identifier exists in no namespace so **both** bngsim **and**
RoadRunner reject them — reference-engine-confirmed broken, not a bngsim bug.
Current members (surfaced by the `rr_parity` ODE EXCEPTION triage, 2026-06-02;
per-model evidence in `parity_checks/rr_parity/dev/notes/rr_parity_triage.md`):
`MODEL1208280001` (`rateLaw1`), `MODEL1101180000` and `MODEL5974712823`
(`size_subVolume`); plus the 7 from issue #89 — `MODEL0403888565`,
`MODEL0403928902`, `MODEL0403954746`, `MODEL0403988150`, `MODEL0404023805`
(`min(max, …)` with `max` declared nowhere — a corrupt
`min(max(0,meanAct),ymax)` export) and `MODEL1405070000`, `MODEL2402030002`
(simulation time written as a bare `<ci>time</ci>` with no csymbol and no
declared `time` symbol). A model bngsim rejects but RoadRunner *simulates* is a
bngsim bug, not a drop — it stays in the corpus and gets a tracking issue.

## Stage 2 — the runner

`run.py` takes every `keep`-verdict model and, for each, loads and
simulates it under **BNGsim**, **libRoadRunner** and **AMICI**, each in
its own subprocess with a wall-clock timeout — a crash, hang or runaway
integration in one engine on one model cannot take down the sweep.

- **Coverage**: did the engine load the model? did it simulate to a
  finite trajectory? Coverage is kept distinct from the Stage-1
  partition: a `keep` model no engine can simulate is a coverage miss,
  not a filter error.
- **Accuracy**: libRoadRunner fixes the time horizon (an adaptive
  steady-state search) and is the reference; BNGsim and AMICI are
  scored by peak-normalised max relative error against it, under
  matched solver tolerances.
- `--effort {low,medium,high}` runs a size-tiered subset (cumulative).
- `--engines` selects which engines to run (default all three). The
  full sweep is dominated by AMICI's per-model C++ compilation, so
  `--engines roadrunner,bngsim` gives a fast full-corpus pass and AMICI
  can be filled in later — the CSV schema is stable, AMICI columns just
  stay blank until then.
