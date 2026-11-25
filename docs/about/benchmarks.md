---
summary: "Where correctness and efficiency results are, how BNGsim is checked against other simulators, and which SBML constructs it refuses or deliberately treats differently."
---

# Benchmarks & validation

[bngsim-benchmarks](https://github.com/lanl/bngsim-benchmarks) publishes
[correctness and timing results](https://github.com/lanl/bngsim-benchmarks/tree/main/published)
for test suites of 10 published models, one for each BNGsim feature and input
format, and for sweeps over large model collections (BNGsim vs libRoadRunner,
BNGsim vs AMICI with sensitivities, and RuleMonkey vs NFsim). Each result names
the BNGsim version it was measured on.

## How BNGsim is checked

Every night, BNGsim's results are compared with BioNetGen's `run_network`
(BNGL models), libRoadRunner (BioModels ODE models), AMICI and finite
differences (sensitivities), and the
[SBML semantic test suite](https://github.com/sbmlteam/sbml-test-suite). The
reference engines run live once a week; on other nights, BNGsim is compared with
their recorded results. A regression raises an alert. The suites are in
[`parity_checks/`](https://github.com/lanl/bngsim/tree/HEAD/parity_checks), and
[`REPRODUCING.md`](https://github.com/lanl/bngsim/blob/HEAD/REPRODUCING.md)
says how to run them.

The SBML test suite is graded with a port of the official test runner's
grading, in
[`benchmarks/suites/sbml_test_suite/testrunner/`](https://github.com/lanl/bngsim/tree/HEAD/benchmarks/suites/sbml_test_suite/testrunner/),
with a committed list of unsupported tags that a unit test pins to
[`bngsim._sbml_unsupported`](https://github.com/lanl/bngsim/blob/HEAD/python/bngsim/_sbml_unsupported.py).

## PSA on SBML / Antimony models

`Simulator(model, method="psa", poplevel=N_c)` accepts SBML or Antimony models
loaded via `Model.from_sbml(...)` / `Model.from_antimony(...)` — the same
dispatch as `.net`, sharing the `validate_for_ssa` gate at construct time
(`reversible_non_mass_action`, `non_integer_stoichiometry`, etc. raise
`SsaValidationError` before the simulator is built). PSA semantics for `N_c`
are identical to the `.net` path; choose per [Lin, Feng, Hlavacek
(2019)](https://doi.org/10.1063/1.5096774) — accuracy degrades when `N_c` is
too small relative to per-species populations (e.g. `erk_activation` requires
`N_c ≳ 100` to track the network reference).

## SBML: what BNGsim refuses, and why

**Unsupported constructs (refused at load — never silently approximated).** BNGsim is
an ODE/SSA kinetic engine with no differential-algebraic, delay, or fast-equilibrium
solver. Where such a construct appears in the equations that define the system, BNGsim
fails closed with a `ModelError` naming it — rather than integrate a *different*
mathematical system and return a confident-but-wrong trajectory (opt into the legacy
silent approximation with `BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1`):

| Construct (tag) | Why unsupported | Trivial form still simulated |
|-----------------|-----------------|------------------------------|
| `AlgebraicRule` (non-empty) | DAE constraint — no algebraic/DAE solver | an empty `<algebraicRule/>` (no constraint) |
| `csymbol delay` / `delay(x, τ≠0)` | delay-differential equation — no DDE solver | `delay(x, 0)` = identity |
| `FastReaction` (`fast="true"`) under ODE | fast-equilibrium constraint — no constraint solver | — |

A whole **SBML Level 3 package** declared `required="true"` is refused the same way
and for the same reason: `required="true"` means the package changes the model's
mathematical meaning, so reading only the core layer of a `multi`, `qual` or `spatial`
document integrates a different system than the one on disk. `comp` (flattened) and
`distrib` (random draws refused individually by the math translator) are the two
BNGsim accounts for; a presentation-only package declares `required="false"` and loads
untouched. No SBML Test Suite case is affected — the suite declares only
`comp:required="true"` and `fbc:required="false"`.

The suite's flux-balance test type (`fbc` / `FluxBalanceSteadyState`, 34 cases) is
likewise declared unsupported — it is a steady-state constraint problem, not a
time-course, and BNGsim is a time-course ODE/SSA engine. All of the above are recorded in
the committed manifest, so the official grading reports them as `Unsupported` (an honest
capability boundary) rather than as failures.

BNGsim supports, fully or in part, some tags that other simulators declare
unsupported, such as `VolumeConcentrationRates` (compartment volumes that
change in time) and `AssignedVariableStoichiometry`.

**Intentional deviations (simulated, but deliberately differ).** Three cases where
BNGsim runs the model but does not match the suite's reference on purpose:

- **Avogadro constant (3 cases: 00960, 00961, 01323).** BNGsim uses the SI-exact
  Avogadro number `Nₐ = 6.02214076e23` (fixed by the 2019 redefinition of the SI
  mole) everywhere the `avogadro` csymbol appears. The suite's analytic reference
  predates that redefinition and uses the older measured value, so BNGsim differs in
  the last few digits. We match the current SI definition on purpose.

**Reproducible random event ordering.** When several events fire at the same instant
with equal (or unset) priority, SBML L3v2 §4.11.6 chooses their order at random. BNGsim
implements this with a **per-run seed** (`SolverOptions.event_seed`, exposed as
`Simulator(method="ode").run(seed=…)`): equal-priority ties are broken by a
`std::mt19937_64`, so the `RandomEventExecution` suite family (00962, 01588, 01590,
01591, 01599, 01605, 01627 and the T0/no-delay variants) passes, while a *fixed default
seed* keeps every run reproducible out of the box — the hard requirement for the
parameter-fitting workloads BNGsim is built for (PyBNF). The RNG is consumed **only** at
a genuine equal-priority tie, so any model without such simultaneity is byte-identical
regardless of the seed, and an event-free ODE run stays seed-less (`Result.seed is
None`). Pass an explicit `seed` for an independent event-ordering realization.

All other suite gaps are tracked as ordinary compliance work.
