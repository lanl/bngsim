# `_core` — shared parity contracts

Machinery only; **no models**. Every suite and every downstream consumer that
regenerates our golden references imports from here so reports stay comparable.

## The contracts

| module | what it pins |
|---|---|
| `taxonomy.py` | the one outcome set: `PASS / DIFF / EXCEPTION / TIMEOUT / UNSUPPORTED / SKIP` (`Outcome`), plus `CLEAN`/`FAILING` sets and `tally()`. |
| `schema.py` | the **spec** (`Job`, `Oracle`, `Override`) and **results** (`JobResult`) and **golden** (`Golden`) records, with compact JSON I/O (`write_manifest`/`read_manifest`, `write_report`/`read_report`, `write_golden`/`read_golden`). |
| `oracles.py` | cross-engine comparison **metrics** (single scalars): `max_rel_err` (ODE), `mean_zscore` (SSA/NF), `ensemble_stats`. Registry: `oracles.METRICS`. |
| `differ.py` | the cross-engine **pass/fail protocol** the suites gate on (lifted from bng_parity's tuned `parity_diff.py`): `deterministic_verdict` (combined abs+rel per-cell tol + fail-fraction budget + hard ceilings) and `ensemble_verdict` (K-sigma frac-pass). Use this for the verdict; `oracles` for a headline scalar. |
| `fingerprint.py` | golden tooling: `checksum` (byte-identity within a pinned cell), `fingerprint` (cross-platform numeric fallback), `fingerprint_max_rel`. |
| `versions.py` | best-effort version probes (`bngsim`, `roadrunner`, `amici`, `libsbml`, legacy `bng`) and `stamp(reference_engine)` for report provenance. |

## Three-file separation (settled principle)

- **manifest** = the spec. Committed, stable, one job per line. Unit of work is
  `model × method × reference_engine`. Carries `params` (suite-specific run
  config) and `overrides` (each with a mandatory `reason`).
- **report** = one run's results. Regenerated every run, never hand-edited.
  Stamps observed engine versions per job.
- **golden** = per-job `checksum` + numeric `fingerprint`; full trajectories
  only for a representative subset. What consumers regenerate through their own
  bridge (so they need only a pinned bngsim, no RR/BNG/AMICI install).

## Two comparison modes

- **Cross-engine, here**: always a numeric `tol` (engines never go byte-identical).
- **Consumer regeneration**: byte-identical `checksum` within a pinned
  `(version, platform, seed)` cell; `fingerprint_max_rel` is the cross-platform
  fallback when checksums legitimately differ (BLAS/rounding).
