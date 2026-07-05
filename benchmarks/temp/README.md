# benchmarks/temp

Scratch space for benchmark scripts that need to survive (be tracked in git so
they don't get lost across machines/worktrees) but aren't wired into the
`suites/` framework. Move things out to a real suite when they earn it.

## GH #84 — BLAS dense linear solver (Accelerate/LAPACK dgetrf)

All resolve the corpus from `RR_PARITY_DIR` (the rr_parity SBML models), falling
back to `../../parity_checks/rr_parity` relative to this dir. Run with an
interpreter that has `bngsim` importable.

    RR_PARITY_DIR=.../bngsim/parity_checks/rr_parity python bench_large_dense.py

- `bench_large_dense.py` — built-in dense LU vs BLAS dgetrf, A/B per model
  (median-of-N, with factorization/solve counts), one subprocess per
  (model, backend) with a hard timeout. This is the script that established the
  win tracks factorization count, not N/density. Used to tune the adaptive gate
  (#132).
- `bench_lapack_gate.py` — same A/B over the feasibility-classified small/medium
  corpus (the N≤786 set).
- `scan_dense.py`, `measure_model.py` — corpus scans: per-model dense-vs-sparse
  walls and setup/solve counts.
- `lu_bench.c`, `lu_split.c`, `lu_verify.c` — standalone microbenchmarks of the
  built-in vs LAPACK GETRF/GETRS kernels (factorization-only speedup,
  back-solve comparison, and bit-level correctness). Build with
  `cc -O3 lu_*.c -framework Accelerate -o lu_*` on macOS.

See `dev/notes/gh84_lapack_dense_findings.md` for the data and conclusions.
