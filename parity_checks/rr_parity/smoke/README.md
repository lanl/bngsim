# rr_parity smoke subset (hermetic)

A tiny, provenance-pinned slice of the rr_parity ODE corpus, committed to git so
the bngsim-vs-RoadRunner pipeline runs end-to-end **without** the ~300 MB
`fetch_sedml.py` + `materialize.py` reconstruction. Exercised by
`parity_checks/tests/test_smoke_corpus.py`.

- `models/<id>/<id>_url.xml` — BioModels SBML (temp-biomodels origin)
- `models/<id>/<id>_url.sedml` — the curated SED-ML protocol
- `ode_jobs_smoke.json` — the 6 job specs, lifted from `ode_jobs.json` with each
  `model` repointed to `smoke/models/…`

Run just this subset:

```bash
python rr_run.py --jobs smoke/ode_jobs_smoke.json --out runs/report_smoke.json
```

**Provenance.** Every byte here is sha256-verified against
`temp_biomodels_manifest.json` (upstream pin: `sys-bio/temp-biomodels`
@`6a09daf46af1bb89e4857436b623a7b8720863ad`) — the same manifest a full
`fetch_sedml.py` checks. The smoke test asserts this on every run, so a silent
edit or an upstream drift fails loudly.

**The models** (all small, verified-green through the real pipeline):
BIOMD1 (events), BIOMD3, BIOMD4, BIOMD5 (`figure_sedml` horizon tier), BIOMD6,
BIOMD10.

**Regenerate** (only if the corpus/pin changes): copy the six models' SBML+SED-ML
out of the materialized `models/<id>/` tree into `smoke/models/<id>/`, keep
`ode_jobs_smoke.json`'s specs in sync with `ode_jobs.json`, and re-run the smoke
test — its sha256 gate confirms the new bytes match the manifest.
