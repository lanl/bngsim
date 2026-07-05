# `sbml_roundtrip` suite

Round-trips the SSA model corpus through BioNetGen's SBML writer and
checks that BNGsim's SBML loader can read the result. This is the
generation half of the SBML-SSA parity story; the distributional
trajectory checks that consume these SBMLs live in
`bngsim/harness/run_ssa_roundtrip.py` (BNGsim `from_net` vs `from_sbml`)
and `bngsim/harness/run_rr_ssa_trajectory_parity.py` (BNGsim vs
libRoadRunner).

## What `run.py` does

For each `../../models/bngl/ssa/*.bngl` (12 models):

1. Writes a wrapper `.bngl` that reads the source with
   `skip_actions=>1` (so the source's `simulate(...)` action does not
   run), expands the network, and calls `writeSBML()`.
2. Invokes `BNG2.pl` on the wrapper in a temp dir.
3. Copies `<name>_sbml.xml` here as `<name>.xml`, prepending a
   traceability comment (BNG version + source `.bngl`).
4. Runs a loader smoke test (`Model.from_sbml` per file) and writes
   `load_check.json`.

Generation is idempotent — an `<name>.xml` is rebuilt only when older
than its source `.bngl` or `BNG2.pl`. Pass `--force` to rebuild all.

```sh
python run.py                  # generate + load-check the full corpus
python run.py --no-load-check  # generate only
python run.py --no-generate    # load-check an existing corpus only
python run.py --force          # ignore idempotency, regenerate all
```

`BNG2.pl` is located via `--bng-path` (default
`~/Simulations/BioNetGen-2.9.3/BNG2.pl`).

## Artifacts

The generated `*.xml` and `load_check.json` are **regenerated
artifacts, not tracked fixtures** — every file is byte-derivable from
the `.bngl` source plus `BNG2.pl`, so `.gitignore` excludes them all.
Run `run.py` to (re)materialise the corpus before any downstream
harness check.
