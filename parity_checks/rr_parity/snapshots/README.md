# Parity benchmark snapshots

Released results of the `rr_parity` ODE parity sweep (BNGsim vs libRoadRunner over the
BioModels ODE corpus), committed here as **gzipped report JSON** so the benchmark ships
in the repository without bloating history with the ~9 MB rendered HTML. The JSON is the
source of truth; the interactive HTML matrix regenerates from it deterministically.

## What's here

`report_ode_<version>.json.gz` — the gzipped `runs/report_ode.json` produced by the
sweep at that tagged release.

## Viewing a snapshot

```bash
gunzip -c report_ode_<version>.json.gz > ../runs/report_ode.json
cd .. && python generate_matrix.py        # -> runs/parity_matrix.html (open in a browser)
```

## Convention

- A snapshot is committed **only at a tagged / public release**, not on every sweep, to
  keep git history small (the corpus and per-run reports otherwise stay gitignored under
  `runs/` and `models/`).
- To cut one on the release commit:

  ```bash
  python rr_run.py --workers 5                                   # full sweep -> runs/report_ode.json
  gzip -c runs/report_ode.json > snapshots/report_ode_<version>.json.gz
  ```

- Only the JSON is committed; the rendered `parity_matrix.html` is derived and stays
  gitignored.
