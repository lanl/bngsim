# rr_parity data directory

## Contents

- `sedml_source/` - Curated SED-ML protocols from sys-bio/temp-biomodels (gitignored, ~49 MB)

## Setup

Run `fetch_sedml.py` from the parent directory to download the SED-ML corpus:

```bash
# from the rr_parity suite directory (the parent of this data/ folder)
cd parity_checks/rr_parity
python fetch_sedml.py
```

This downloads the curated simulation protocols from the temp-biomodels paper (PMC12677764)
into `data/sedml_source/final/`.
