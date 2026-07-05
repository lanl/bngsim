# Committed SBML overrides

These are canonical BioModels SBML files for models whose **upstream source is bad**
and would otherwise reconstruct *wrong* on a fresh `materialize.py` rebuild. Unlike
the rest of the corpus (gitignored, ~270 MB, re-fetched from BioModels), these 15
small files (~1 MB) are committed so the corpus is correct on any machine with no
network dependency.

`materialize.py` copies `data/sbml_overrides/<model_id>/<file>.xml` to
`models/<model_id>/<file>.xml` **last**, so it wins over whatever the normal source
placed.

## Why each is here (discovered 2026-06-14; all 15 were `BAD_TEST` before repair)

- **BIOMD0000000883** — its `sbml_origin` was `temp-biomodels`, and that SED-ML
  mirror (`sys-bio/temp-biomodels`, `final/BIOMD0000000883/Giani2019.xml`) ships a
  **COPASI** export, not SBML. The manifest entry has since been repointed to
  `sbml_origin: biomodels_dir`; this override is the no-network guarantee.
- **14 others** (`BIOMD0000001102`, `BIOMD0000001103`, and twelve `MODEL*`) — their
  BioModels download in `sbml_downloads/` was a **0-byte file** (a failed fetch that
  `fetch.py` then skipped forever as "already present"). `fetch.py` is now hardened
  to treat empty/invalid files as un-fetched, validate downloads, and fall back to
  the BioModels REST download endpoint; this override is the no-network guarantee.

All 15 are the BioModels `main` SBML file, validated to parse with libSBML.
Regenerate from a corrected `models/` tree if BioModels updates a model.
