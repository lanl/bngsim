# BioModels suite smoke subset (hermetic)

A tiny, provenance-pinned slice of the BioModels Stage-2 corpus, committed to git
so the runner goes end-to-end **without** the ~270 MB `fetch.py` download.
Exercised by `parity_checks/tests/test_smoke_corpus.py`.

- `sbml/<id>.xml` — the pinned SBML (BioModels REST, exact tested version)
- `manifest_smoke.csv` — the 6 `keep` rows, same schema as `manifest.csv`

Run just this subset (skip the slow AMICI per-model compile):

```bash
python run.py --manifest smoke/manifest_smoke.csv --sbml-dir smoke/sbml \
    --engines roadrunner,bngsim --results-dir results/smoke
```

**Provenance.** Each `sbml/<id>.xml` is the version-pinned byte stream recorded in
`biomodels_provenance.json` (and `manifest_smoke.csv`'s `sha256` column) — i.e.
what `fetch.py` (the default pinned fetch) reproduces and sha256-verifies. The
committed copies here are the pinned bytes, **not** whatever a stale local
`data/sbml_downloads/` may hold; the smoke test asserts the match every run.

**The models** (all small, verified-green through the real pipeline): BIOMD1
(events → Fig S2 split), BIOMD3, BIOMD4, BIOMD5, BIOMD6, BIOMD10.

**Regenerate** (only if the corpus/pin changes): fetch the six ids' pinned bytes
via `fetch.py`'s `fetch_pinned` into `smoke/sbml/`, refresh the six `keep` rows in
`manifest_smoke.csv` from `manifest.csv`, and re-run the smoke test.
