# Stashed: Kozer-2013 EGFR network-free fitting jobs

This directory holds the **network-free** half of the original single-model
Kozer-EGFR fitting benchmark, moved here out of the active harness when the
benchmark was generalized to the multi-model mmc3 sweep (2026-05-17).

It is a self-contained, runnable snapshot — not dead code. The
`run_jobs.py` / `consistency_check.py` / `emit_latex_table.py` harness no
longer references it; nothing here is on the hot path.

## What's here

- `bngl/nf.bngl`            — Kozer EGFR, in-process NFsim action block.
- `bngl/nf_subprocess.bngl` — same model, `param=>"-bscb"` + composite
  functions inlined for the standalone NFsim `.scan` writer.
- `bngl/rm.bngl`            — Kozer EGFR, in-process RuleMonkey action block.
- `conf/nfsim_subprocess.conf`   — standalone NFsim app (`bngl_backend=bionetgen`).
- `conf/nfsim_bngsim.conf`       — in-process vendored NFsim (`bngl_backend=bngsim`).
- `conf/rulemonkey_bngsim.conf`  — in-process RuleMonkey (`bngl_backend=bngsim`).
- `conf/_smoke/`           — reduced-size smoke/calibration confs.
- `data/`                  — `doseresponse.exp`, `timecourse.exp` (copies; the
  ODE pair in `../../kozer_egfr/data/` keeps the originals).

## Why it was stashed, not deleted

The network-free comparison (NFsim app vs. in-process BNGsim/RuleMonkey) is
still wanted for the paper, but:

- In-process NFsim is blocked — the bngsim-vendored NFsim carries an
  incomplete `connectivity-determinism` patch that shifts its oligomer
  distribution (internal#44, 2026-05-17).
- The new benchmark is multi-model (mmc3 sweep). A single hand-curated
  Kozer network-free job does not fit that structure cleanly.

When issue #44's NFsim vendoring lands, this snapshot is the starting point
for re-adding a network-free row.

## Running it standalone

The conf paths are relative (`../bngl/...`, `../data/...`), so from this
directory:

    BNGPATH=~/Simulations/BioNetGen-2.9.3 \
      ~/Code/PyBNF/.venv/bin/pybnf -c conf/nfsim_subprocess.conf -o
