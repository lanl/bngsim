# Reproducing bngsim's validation & benchmark numbers

`bngsim` ships its correctness suites ([`parity_checks/`](parity_checks/)) and benchmarks
([`benchmarks/`](benchmarks/)) so a third party can **reproduce the published numbers
exactly**. This is the top-level index: the reproducibility model, where each corpus comes
from, the external tool versions, the environment wiring, a step-by-step path, and a
**hermetic smoke path** that runs the pipelines end-to-end in seconds with no large
download.

> Scope: this covers the *data + tooling* to re-run the checks. The engine itself is
> pinned by the package version (`bngsim.__version__`) and the git commit. Results in this
> tree were generated with bngsim `0.11.31`.

---

## 1. The reproducibility model — pinned fetch + sha256

The model corpora are **not committed to git** (hundreds of MB, undiffable). Instead each
fetched corpus is **pinned to an exact upstream version and sha256-verified on fetch**, so
a re-fetch reproduces the identical bytes our results were generated against — no Zenodo /
DOI needed. What lives in git is textual and diffable: the **fetch scripts**, the
**provenance manifests** (one sha256 receipt per model, conforming to
[`parity_checks/manifest.schema.json`](parity_checks/manifest.schema.json)), committed
**overrides / repairs**, and a small **smoke subset** (§6).

Each `fetch.py` has two modes:

- **default (pinned, reproducible)** — checks out the pinned commit / downloads the pinned
  version, then verifies every file's sha256 against the committed manifest.
- **`--upstream-latest`** — refresh/extend against the moving upstream (unpinned; warns,
  and the manifest verification flags any drift).

---

## 2. Corpora — where each comes from, and how to fetch it

| Corpus | Suite | Upstream (pinned) | Fetch | Manifest |
|---|---|---|---|---|
| BioModels SED-ML + SBML | `parity_checks/rr_parity` | `sys-bio/temp-biomodels` @ `6a09daf4` | `rr_parity/fetch_sedml.py` → `materialize.py` | `rr_parity/temp_biomodels_manifest.json` |
| BioModels SBML (Stage-2) | `benchmarks/suites/biomodels` | BioModels REST, snapshot `2026-07-03` (per-model version pin) | `benchmarks/suites/biomodels/fetch.py` | `biomodels/biomodels_provenance.json` |
| BNGL corpus (895) | `parity_checks/bng_parity` | RuleWorld/RuleHub `479d6d62`, richardposner/RuleMonkey `0f701125`, wshlavacek/BNGL-Models `0df6cbd7` | `bng_parity/vendor_corpus.py` (committed) | `bng_parity/manifest.json` |
| SBML semantic suite | `harness/sbml_test_suite` | `sbmlteam/sbml-test-suite` @ `473e119d` (`3.4.0-36`) | `harness/sbml_test_suite/fetch_semantic_suite.py` | `harness/sbml_test_suite/SUITE_PIN.json` |
| DSMTS stochastic (39) | `harness/sbml_test_suite/dsmts` | same commit, **committed** (hermetic) | — | `dsmts/dsmts_manifest.json` |

The **BNGL corpus is committed** (real bytes under `bng_parity/models/`); `vendor_corpus.py`
re-vendors it from the pinned source repos and records provenance. The two BioModels stacks
and the SBML semantic suite are gitignored and reconstructed by their `fetch.py`.

---

## 3. External tool versions (references / oracles)

bngsim is validated *against* these; none is needed to run bngsim itself.

| Tool | Version / pin | Used by | Where pinned |
|---|---|---|---|
| libRoadRunner | 2.9.2 | `rr_parity`, `biomodels` | `pip install bngsim[roadrunner]` |
| BioNetGen (BNG2.pl / run_network / NFsim) | 2.9.3 | `bng_parity`, benchmarks | `$BNGPATH` |
| AMICI | `v1.0.1-12-g667b17b6b` (`667b17b6`, BSD-3) | `amici_parity`, `biomodels` | `parity_checks/amici_parity/AMICI_PIN.json` |
| SBML Test Suite | `473e119d` (`3.4.0-36`, LGPL-2.1) | `harness/sbml_test_suite` | `harness/sbml_test_suite/SUITE_PIN.json` |
| RuleMonkey | vendored `third_party/rulemonkey` | `bng_parity` (network-free) | in-tree |

Python 3.10–3.13.

---

## 4. Environment wiring

Tools and corpus locations resolve from env vars (never hard-coded paths). Each default
points into the suite's own tree, so with the corpora fetched in place no env vars are
strictly required.

| Var | Points at | Consumed by |
|---|---|---|
| `BNGPATH` | BioNetGen root (has `BNG2.pl`, `bin/run_network`) | bng_parity, benchmarks |
| `RUN_NETWORK` / `NFSIM` | those binaries, if outside `$BNGPATH` | bng_parity |
| `BIOMODELS_SBML_DIR` | BioModels SBML tree | rr_parity `materialize.py`, biomodels |
| `BIOMODELS_SEDML_DIR` | temp-biomodels `final/` | rr_parity `materialize.py` |
| `SBML_TEST_SUITE_DIR` | `<checkout>/cases/semantic` | harness/sbml_test_suite |
| `RULEHUB_DIR` / `RULEMONKEY_DIR` / `BNGL_MODELS_DIR` | existing source checkouts at the pin (skip re-fetch) | bng_parity `vendor_corpus.py` |

---

## 5. Step-by-step

```bash
# 0. install bngsim + the reference engine
pip install -e 'bngsim[roadrunner]'                     # bngsim + libRoadRunner

# 1a. rr_parity (SBML vs RoadRunner)
python parity_checks/rr_parity/fetch_sedml.py           # pinned temp-biomodels, sha256-verified
python benchmarks/suites/biomodels/fetch.py             # pinned BioModels SBML
python parity_checks/rr_parity/materialize.py           # lay out models/<id>/
python parity_checks/rr_parity/rr_run.py                # → runs/report_ode.json

# 1b. bng_parity (BNGL vs legacy BNG) — corpus already committed
python parity_checks/bng_parity/build_jobs.py           # manifest.json → jobs.json
#   (drive the sweep per parity_checks/bng_parity/README.md)

# 1c. SBML semantic suite
python harness/sbml_test_suite/fetch_semantic_suite.py  # pinned checkout
```

---

## 6. Hermetic smoke path (no download)

To confirm the two fetched pipelines run end-to-end **without** the ~300 MB / ~270 MB
fetch, a tiny provenance-pinned subset (6 small BioModels) is committed beside each suite
and exercised by `parity_checks/tests/test_smoke_corpus.py`:

```bash
pip install -e 'bngsim[roadrunner]'
pytest parity_checks/tests/test_smoke_corpus.py -v
```

It asserts two things: (a) every committed smoke byte's sha256 matches its manifest pin
(the provenance guarantee), and (b) both real runners go green on the subset
(rr_parity PASS; biomodels load + simulate + accuracy vs RoadRunner). Run either subset
directly:

```bash
python parity_checks/rr_parity/rr_run.py --jobs smoke/ode_jobs_smoke.json
python benchmarks/suites/biomodels/run.py \
    --manifest smoke/manifest_smoke.csv --sbml-dir smoke/sbml --engines roadrunner,bngsim
```

See each suite's `smoke/README.md` for the subset and its provenance.
