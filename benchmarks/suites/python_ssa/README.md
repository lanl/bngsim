# `python_ssa` suite

Pure-Python SSA workflow comparison — BNGsim's in-process exact-SSA
(Gillespie) engine versus **gillespy2**, a pure-Python SSA package
(paper Table S6, SSA half).

Promoted from the SSA half of
`harness/comparison/bench_pythonic_workflows.py` (the ODE half is the
`python_ode` suite).

## A deliberately small suite

gillespy2's pure-Python SSA is only feasible on small networks, so the
corpus is the single 4-species `simple_system` model. The larger SSA
corpus is exercised by the `ssa` suite (BNGsim versus `run_network`).
`--effort` is accepted for cross-suite uniformity but every tier runs
the same single job.

## Paths

- **ModelBuilder** — the model is defined entirely in Python (BNGsim's
  `ModelBuilder` API, a hand-built gillespy2 `Model`), no files.
- **.net reader** — the model is parsed from a BNG `.net` file by
  BNGsim's universal `_net_reader`; the parsed structure is handed to
  both engines. (BNGL species names carry pattern syntax that gillespy2
  rejects, so the gillespy2 model uses safe synthetic `S<index>` names —
  the SSA dynamics are name-agnostic.)

## Gates

| Gate | Check |
|------|-------|
| correctness | An ensemble of replicate trajectories is simulated by each engine and the two ensemble means are compared cell-by-cell with a two-sample *z*-test (`_netbench.zscore_gate`, tol 6.0). Both engines run exact SSA, so the means must agree within stochastic error — the same exact-vs-exact test the `ssa` suite uses. |
| timing | Warmup + timed-run median wall time of a single trajectory; reported only for a path that passed correctness. |

```sh
python run.py                     # both gates
python run.py --mode correctness  # z-test only
python run.py --mode timing       # timing only
python run.py --replicates 40     # larger correctness ensemble
```

`simple_system.net` is vendored at `../../models/net/ode/`. Results are
written to the git-ignored `results/` (`python_ssa_results.json` +
`.md`).
