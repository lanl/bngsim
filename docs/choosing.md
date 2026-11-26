---
summary: "When to use BNGsim and when to use another tool, and tips on compilers, solvers, sensitivities, steady states, SSA vs. PSA, NFsim and RuleMonkey, and JAX."
---

# When to use BNGsim

| Task | BNGsim | Alternatives |
|---|---|---|
| Simulate an SBML or Antimony model | Yes (Antimony needs `bngsim[antimony]`) | libRoadRunner, Tellurium, COPASI, AMICI (ODEs only) |
| Simulate a BNGL model, network-based or network-free | Yes (needs `bngsim[bngl]` and Perl) | BioNetGen |
| Run many simulations (scans, batches, replicates) | Yes, on parallel threads | COPASI, AMICI |
| Forward sensitivity analysis | Yes (needs a C compiler) | AMICI, libRoadRunner |
| Adjoint sensitivity analysis | No | AMICI |
| Parameterization, uncertainty quantification, model checking | Yes, through PyBNF | COPASI, AMICI with pyPESTO |
| Models with delays, algebraic rules, or fast reactions | No (BNGsim refuses these models) | Other tools |
| Graphical user interface | No | COPASI |
| ODEs written as a Python function | No | SciPy (`solve_ivp`), Diffrax |

Information about other tools comes from our inspection of their source code in
July and August 2026 (libRoadRunner 2.9.2, AMICI 1.0.1, COPASI 4.46, BioNetGen
2.9.3).

## Tips

- **Compiler.** If a C compiler is available, BNGsim uses it automatically.
  Without one, sensitivity analysis needs the MIR JIT, which is not yet in the
  published wheels (build with `-DBNGSIM_ENABLE_MIR=ON`, run with
  `BNGSIM_CODEGEN_JIT=mir`).
- **Linear solver.** BNGsim picks a dense or sparse (KLU) solver for each
  model. To override, pass `force_dense_linear_solver=True` or
  `force_sparse_linear_solver=True` to `Simulator`.
- **Many parameters.** `sim.compute_all_sensitivities(..., n_workers=8)`
  divides the parameters among threads. BNGsim does not do adjoint sensitivity
  analysis, which gives only the gradient of an objective function and has
  numerical stability problems.
- **Steady states.** Use `sim.steady_state()` instead of a long simulation. It
  tells you whether a steady state was reached (`ss.converged`). With
  `method="newton"`, BNGsim simulates first and then refines the result with
  KINSOL's Newton solver. Simulating first avoids convergence to an unstable
  steady state. `sim.steady_states()` searches for the other steady states,
  stable and unstable, by running Newton from many starts. It can miss one.
- **SSA vs. PSA.** Use PSA (`method="psa"`) when you care about means, not
  variances. It is much faster for models with large populations. Use SSA
  (`method="ssa"`) when fluctuations matter.
- **NFsim and RuleMonkey.** BNGsim includes two independent network-free
  simulators: RuleMonkey (`method="rm"`), first published in 2010 and recently
  rewritten, and NFsim (`method="nf"`). Network-free results are otherwise hard
  to check, so compare the two.
- **BNGsim's NFsim vs. standalone NFsim.** BNGsim's NFsim has bug fixes, and it
  refuses models that standalone NFsim simulates incorrectly without warning.
  It also skips process launch and model parsing on each run.
  `nfsim_v1143_compat=True` reproduces NFsim 1.14.3 trajectories.
- **JAX.** To differentiate through a simulation in a JAX workflow, use
  `bngsim.jax.differentiable_solve` (`bngsim[jax]`).
