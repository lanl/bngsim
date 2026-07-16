# Changelog

All notable changes to bngsim will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows the pre-1.0 SemVer convention `0.MAJOR.MINOR`:
**MAJOR** bumps for behavioral or API breaks; **MINOR** bumps for additive
changes and bug fixes that don't change observable behavior.

Since #31, `pyproject.toml` is the single source of truth for the version
string; every other anchor (Python `__version__`, the C extension
`__version__`, the `Result` HDF5 attr, and `project(... VERSION ...)`
in `CMakeLists.txt`) is derived from it.

## [Unreleased]

### Added
- **ODE Jacobian characterization harness**
  (`parity_checks/bng_parity/jacobian_characterization.py`): characterizes each
  ODE model in the `bng_parity` corpus by structural Jacobian density
  (`nnz/N^2`) and stiffness ratio — `max|Re lambda| / min_{!=0}|Re lambda|` on the
  conservation-reduced Jacobian, computed from BNGsim's own native analytical
  Jacobian (no autodiff) — and, in `--analyze` mode, partitions the corpus into
  sparse-stiff / dense-stiff / non-stiff regimes. For models with `N > 300` the
  stiffness ratio is sampled at up to `DENSE_TIME_SAMPLES` log-spaced trajectory
  points (default 64; `--dense-time-samples` to override) rather than the former
  3, and each result now reports both `stiffness_ratio_max` (trajectory peak) and
  `stiffness_ratio_median` (sustained). The old 3-point sampling under-resolved
  the peak for large networks — e.g. the `N=1281` `fceri_fyn` peak rose from
  `1.4e7` to `1.0e8` under dense resampling.
- **SBML BioModels counterpart** of the characterization harness
  (`parity_checks/rr_parity/jacobian_characterization_sbml.py`): applies the same
  density and stiffness metrics and regime classification to the `rr_parity`
  SBML corpus, loading each model via `Model.from_sbml` (no network-generation
  step) and reusing the `bng_parity` metric helpers — including the dense
  trajectory time sampling above — so both corpora are characterized by identical
  code and report the same `stiffness_ratio_max` / `stiffness_ratio_median`
  fields.

### Fixed
- **PSA now scales zeroth-order synthesis reactions and bounds every reaction's
  leap by its products as well as its reactants (issue #14).** The partial-scaling
  leap factor `iScaling = max(1, ⌊N_min/N_c⌋)` previously took `N_min` over
  *reactant* species only. Synthesis reactions (`∅ → A`) have no reactants, so
  `N_min` defaulted to 0 and they were never scaled — the source channel dominated
  the step budget in source-driven models even when the product was abundant.
  Separately, a reaction with a large reactant but a small product was
  over-scaled: the leap was bounded by the (large) reactant and dumped a coarse
  jump into a currently-small product. `N_min` is now the minimum population over
  the **union of reactants and products**, matching BioNetGen `run_network`'s
  default heterogeneous adaptive scaling (`rxn_rate_scaled`, `pScaleChecker=true`):
  reactants bound depletion, products bound overshoot of a small produced species.
  For synthesis the product population governs — the reaction is scaled once the
  product is large and runs as exact SSA while it is small. This intentionally
  departs from `run_network`, which scales synthesis by a flat `N_c` regardless of
  the product. The SSA dependency graph gained a PSA-only product-population
  dependency so a reaction is re-evaluated when a product it makes changes its leap
  factor; the exact-SSA path is unchanged. The scaling of nonlinear rate laws
  (MichaelisMenten / Sat / Hill) still differs from `run_network` and is tracked
  separately in issue #16.
- **`Model.from_net` no longer fails with `stoi: no conversion` on a `.net`
  containing a `reactions_text` block (issue #13).** BNG2.pl emits an optional
  `begin reactions_text ... end reactions_text` block (when the corresponding
  print option is set) that restates the numeric `reactions` block in
  human-readable pattern form (`1 A(b) -> B(a) k1`). The loader's block dispatch
  matched `"begin reactions"` as a *substring* of `"begin reactions_text"`, so
  those pattern lines were fed to the numeric reaction parser, where
  `std::stoi("A(b)")` threw. The loader now recognizes and skips the
  `reactions_text` block (checked before `begin reactions`, since that string is
  a prefix substring), as it already does for other optional blocks — the
  numeric `reactions` block remains authoritative and the network is unchanged.
  Two parity-corpus models that previously loaded only after the block was
  stripped by hand (`ComplexDegradation` N=6, `BaruaBCR_2012` N=1122) now load
  directly.

## [0.11.35] - 2026-07-14

### Added
- **Steady-state forward sensitivities at the observable / expression level
  (issue #12).** `Simulator.steady_state(sensitivity_params=[...])` now returns a
  `SteadyStateResult` that also exposes `output_sensitivities(selectors,
  axis="parameter")`, plus `observable_names`, `expression_names`,
  `sensitivities_observables`, and `sensitivities_expressions` — mirroring
  `Result.output_sensitivities` on a CVODE run. These project the exact species
  `dY_ss/dp` onto the model's observables (exact linear group map) and global
  functions (finite-difference total derivative: state chain **plus** the
  function's explicit `∂func/∂p`), so a gradient consumer reads
  `∂(observable)/∂θ` / `∂(expression)/∂θ` directly instead of re-deriving the
  output Jacobian. Validated against the CVODES forward-sensitivity `run()` at
  steady state. The `ic` axis is structurally zero for a stable steady state
  (`∂x*/∂x(0) = 0`) and raises a directed error. Unblocks a scored, gradient-
  differentiable KINSOL steady-state dose-response scan in PyBNF
  (lanl/PyBNF#478).

## [0.11.34] - 2026-07-12

### Added
- **True multi-slot named saved-concentration states for the NFsim backend
  (issue #11).** `NfsimSession.save_concentrations(label=...)` /
  `restore_concentrations(label=...)` now hold each named state in its own
  in-session snapshot, so multiple named NFsim states coexist and round-trip
  faithfully — a later `save_concentrations("other")` no longer clobbers an
  earlier one, matching the network-based `Model`. Adds the
  `NfsimSession.saved_concentration_labels` property. This replaces the previous
  single-slot-with-label shim, which held only one snapshot and raised when a
  differently-named state was requested. Named and default (unlabeled) slots are
  independent; unlabeled `restore_concentrations()` still rewinds the default
  slot and requires a prior unlabeled `save_concentrations()` (NFsim has no
  seed-reset). Implemented in C++ via a per-label `NFcore::SystemSnapshot` map
  (no vendored-NFsim changes).

## [0.11.33] - 2026-07-05

### Changed
- **Self-sufficient sdist install:** builds and bundles SuiteSparse/KLU from
  source when no system SuiteSparse is present (GH #209), so `pip install` from
  an sdist always gets the sparse solver instead of failing or degrading to
  dense. Intel-mac wheels build in CI again (retired `macos-13` →
  `macos-15-intel`) and publish via Trusted Publishing. No library API changes
  since 0.11.32.

## [0.11.32] - 2026-07-05

### Added
- **PyPI Trusted-Publishing release workflow** (`.github/workflows/release.yml`):
  builds Linux (manylinux x86_64), macOS (arm64 + Intel), and Windows wheels via
  cibuildwheel plus an sdist, and publishes via PyPI Trusted Publishing (OIDC, no
  tokens). `workflow_dispatch` targets TestPyPI (rehearsal) or PyPI; a `v*` tag
  publishes to PyPI. No library API changes since 0.11.31.

## [0.11.31] - 2026-07-04

First public release of bngsim, as a Los Alamos National Laboratory open-source
release (LANL software release reference **O5098**). No changes to library
behavior or API since 0.11.30.

### Added
- Third-party `NOTICE` listing every redistributed component — vendored code
  (NFsim, RuleMonkey, MIR, ExprTk), the bundled SUNDIALS solver, and the vendored
  model/test-data corpora — with its license terms, and `ACKNOWLEDGMENTS.md`
  citing the reference simulators, standards, and libraries BNGsim builds on and
  validates against.
- Per-model provenance tables for the curated benchmark model corpora
  (`benchmarks/models/README.md`).

### Changed
- **License changed from BSD-3-Clause to MIT** for the LANL-developed portion of
  bngsim, carrying the Triad National Security, LLC / U.S. Government copyright
  notice (produced under U.S. Government contract 89233218CNA000001). Acknowledged
  by NNSA for open-source release; LANL software release reference **O5098**.
  Vendored third-party components retain their own licenses. Updated `LICENSE`,
  `pyproject.toml` (metadata + classifier), and `README.md`.

## [0.11.30] - 2026-07-01

### Added
- **Codegen RHS + analytical Jacobian for cross-compartment variable-volume
  reactions (GH #171, Parts 2–3).** Completes #171: after Part 1 (0.11.29) gave
  these models an interpreted analytical Jacobian, the compiled-C codegen path now
  supports them too, so a large cross-compartment variable-volume model gets a
  compiled RHS + Jacobian (`.so`) instead of declining to the interpreted engine.
  - **RHS** (`generate_rhs_from_model`): the `NotImplementedError` decline for
    `ode_live_volume_idx0 ≥ 0` is removed. The per-species accumulation scatter
    emits `rate / y[live_idx]` (falling back to the static `volume_factor` when
    `y[live_idx] ≤ 0`) for a live-volume row and keeps `rate * inv_vf` for a
    static row — a bit-exact mirror of `compute_derivs_core`'s `species_divisor`.
  - **Jacobian** (`generate_jacobian_from_model`): the per-species existing
    columns defer the volume divide to a runtime `/ y[live_idx]`, and a new
    `−func/y[live_idx]²` column is emitted per reaction (`func = func[fidx]`,
    the reaction's bound rate value) — mirroring the interpreted `volume_terms`
    scatter, in fill order (species → volume → observable) so the explicit-factor
    (#172) case's cancelling columns accumulate identically.
  - Verified the compiled Jacobian equals the interpreted `_dense_analytical_
    jacobian` (the FD-self-checked oracle) **bit-for-bit** across spread-out
    states on all four #144/#172 case models, and an end-to-end codegen ODE run
    reproduces the interpreted trajectory exactly. Optimization only — no
    behavioral or SBML-suite-score change.
  - Byte-identical codegen output for every non-varvol model: the live divide and
    the new column are gated on `per_species_volume_scaling` + a live index, the
    `inv_vf` table is emitted only when a static row actually reads it (an
    all-live reaction like `_C4_BOTH_RR` never does, which would otherwise leave
    an unused-static `-Werror` failure), and the per-observable path is unchanged.

## [0.11.29] - 2026-07-01

### Added
- **Analytical Jacobian for cross-compartment variable-volume reactions (GH #171,
  Part 1).** These reactions — an `A + B => P` whose reactants live in different
  compartments, at least one of which resizes at runtime (a rate rule or an
  event) — have always *simulated* correctly, but GH #144 left them on a slower
  finite-difference Jacobian + interpreted RHS because the per-species
  accumulation divide by the LIVE compartment volume happens *outside* the rate
  function, where SymPy could not see it. The interpreted analytical Jacobian now
  covers them: each existing column swaps the baked-in `1/V_static` for the
  runtime `1/V_live = 1/conc[ode_live_volume_idx0]`, and a new column carries
  `∂(dSᵢ/dt)/∂V_live = −(varvol RHS of i)/V_live` (read from the reaction's bound
  rate parameter, exactly as the per-observable path reads `func`). The live
  index is stored per affected row — one reaction can mix live and static rows
  (`A,P` in a growing `cell`, `B` in a static `dish`) and even two distinct live
  columns (`A,P → cell`, `B → dish`). CVODE now integrates these models with the
  analytical Jacobian (stiffer-stable, no per-step FD sweep) instead of falling
  back. **This is an optimization: the ODE/SSA solution, correctness, and SBML
  Test Suite score are all unchanged** — verified by asserting the analytical
  Jacobian attaches (`analytical_jacobian_complete is True`, the all-or-nothing
  self-check gate) on all four #144/#172 case models, an independent finite-
  difference cross-check of the assembled Jacobian (including the new column and
  the `_C5` explicit-factor case whose `∂/∂V_live` cancels to exactly zero), and
  a trajectory match against the FD path. Codegen RHS/Jacobian for these models
  (Parts 2–3) remain a separate follow-up increment; codegen still declines them
  (keyed on `ode_live_volume_idx0 ≥ 0`, independent of the completeness flag).

### Changed
- `build_jac_sparsity` (`src/model_builder.cpp`) now takes the species list and
  adds the `(row i, col V_live)` structural nonzero for a
  `per_species_volume_scaling` reaction's live-volume affected rows. Guarded on
  the varvol flag + a live divisor, so the emitted CSC is byte-identical for
  every static-volume / `.net` model (no codegen-parity or suite-score impact).
- `FunctionalJacobianData::SpeciesTerm::affected` (`include/bngsim/types.hpp`) is
  now a small struct `{csc, coeff, live_idx, static_divisor}` — the volume divide
  is deferred from load time to the runtime scatter so a variable-volume row uses
  the live volume. For a non-varvol row the divisor is `1.0` (an exact no-op,
  byte-identical to the pre-#171 folded coefficient).

### Added (developer)
- `BNGSIM_JAC_NO_SELFCHECK=1` trusts the assembled analytical Jacobian and skips
  the load-time FD self-validation. Debug lever to isolate a self-check
  false-fail from a genuine symbolic↔engine mismatch; never set in production.

## [0.11.28] - 2026-07-01

### Added
- **bngsim now has an authoritative-equivalent SBML Test Suite score backed by a
  committed unsupported-tags manifest (GH #241).** The community-standard
  yardstick is the *official* SBML Test Suite runner, whose grading differs from
  our in-repo fair harness in one key way: it classifies a case a tool declares
  it cannot handle as `Unsupported` (an honest capability boundary) rather than
  as a failure. bngsim now declares that boundary in one shipped place —
  `bngsim._sbml_unsupported`, the single source of truth for the constructs it
  refuses under ODE (`delay()` → DDE, non-empty `AlgebraicRule` → DAE,
  `fast="true"` → fast-equilibrium) and the flux-balance (`fbc`) test type it
  does not attempt — and the loader/simulator refusal messages take their labels
  from it, so the strings the guard emits and the tags the manifest declares can
  never drift. `benchmarks/suites/sbml_test_suite/testrunner/` adds the pieces
  that feed the official runner: a fail-closed `bngsim_wrapper.py` (+ shell
  shim), the committed `bngsim-unsupported-tags.txt` manifest (regenerated from
  the SSOT by `gen_manifest.py`), and `score.py` — a faithful local port of the
  runner's compare (`CompareResultSets`, `requireAllColumns`), outcome enum
  (`getResultTypeInternal`), and tag matching (`TestCase.matches`/`prefixMatch`)
  that drives bngsim through the *same* shared load+integrate+resolve code as the
  fair harness. On SBML Test Suite v3.3.0 (1823 cases) it reports **1577 `Match`
  (1577/1789 in-scope TimeCourse = 88.1%), 242 `Unsupported`, 3 `NoMatch`, 1
  `CannotSolve`, 0 `Error`** — reconciling with the fair harness's 1578 to a
  single, fully-explained case (`01244`, a no-op empty `<algebraicRule/>` that
  bngsim solves correctly but the `AlgebraicRule` tag honestly excludes; the
  suite has no finer sub-tag). The 3 `NoMatch` are the documented SI-exact
  Avogadro deviation (`00960`/`00961`/`01323`); `CSymbolAvogadro` and
  `RandomEventExecution` are deliberately *not* declared. Dev-only: the wheel
  packages only `python/bngsim`, and only the SSOT module ships.

### Changed
- The shared SBML-suite grading kernel (`_grading.py`) factors its per-variable
  resolve + amount-conversion into `resolve_columns()` / `_resolve_graded()`, and
  the bngsim adapter (`_engines.py`) factors its load+integrate into
  `build_bngsim_series()`, so the fair harness and the new test-runner wrapper
  produce delivered results through byte-identical code. No change to any
  engine's graded verdict.

## [0.11.27] - 2026-07-01

### Fixed
- **A `delay()` inside an L2 `<stoichiometryMath>` is now refused at load
  instead of being silently zero-delayed into a wrong trajectory (SBML semantic
  suite case `01481`).** The unsupported-construct gate (GH #113) scans every
  math container that feeds the integrated system for a non-trivial `delay()` —
  rate/assignment rules, reaction kinetic laws, initial assignments, and (since
  GH #240) all event math — and refuses the model as a DDE bngsim has no solver
  for. It did not, however, reach a `speciesReference`'s `<stoichiometryMath>`,
  which libSBML exposes only via the reaction's reactant/product lists. So a
  delay there slipped the gate: the AST handler returned the zero-delay value
  (`delay(A, 1) → A`) and bngsim integrated a *different* system, returning a
  confident but wrong result (`01481` max error `≈66`) where libRoadRunner
  correctly refuses (`Unable to support delay differential equations`). The gate
  now also scans every reactant/product `<stoichiometryMath>`, naming the
  offending `reaction:…:stoichiometryMath:<species>` location; `delay(x, 0)`
  (exactly `x`) still loads, and the `BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1`
  opt-out still restores the legacy silent approximation. This is a distinct
  location from GH #240 (event math); no effect on any model without a delay in
  its stoichiometry math (62 `<stoichiometryMath>` suite cases still pass; the
  bngsim+RR sweep pass count is unchanged at 1578/1789, with `01481` moving from
  a wrong answer to an honest load refusal).

## [0.11.26] - 2026-07-01

### Fixed
- **A delayed, `useValuesFromTriggerTime="true"` event that resizes a species'
  compartment now conserves the species' *execution-time* amount, not its stale
  *trigger-time* amount (GH #248; kitchen-sink suite case `01000`).** When such
  an event fires, bngsim injects a per-species concentration rescale
  (`[S] ← [S]·V_old/V_new`, GH #74) so the species' amount is preserved across
  the discontinuous volume change. That rescale is a physical consequence of the
  resize, which happens at the event's *execution* time — but for a delayed
  `useValuesFromTriggerTime=true` event, the injected rescale was being frozen at
  *trigger* time along with the user assignments. A species whose amount changes
  between trigger and execution (e.g. one produced/consumed by a reaction during
  the delay window) was then rescaled from its stale trigger-time amount,
  corrupting it by `V_old/V_new` evaluated at the wrong volume. In `01000` this
  drove the product species `S2` to a `value_mismatch` (max error `0.968`) that
  masked an otherwise-correct trajectory. The injected rescale (the only
  `ode_only` event assignment) is now always evaluated against pre-fire state at
  execution time, exactly reproducing the already-correct
  `useValuesFromTriggerTime=false` apply path. No effect on non-resize events,
  immediate (delay-0) events, or events that do not touch a compartment.
- **Event-chatter guard is honored by the same-instant cascade again — deep
  ODE-with-clamp models no longer stall (GH #95 regression from the #242 event
  rework; `BIOMD0000000711`).** The Zeno chatter guard marks a non-negativity
  clamp *dormant* after it re-fires `CHATTER_LIMIT` times in the sub-tolerance
  noise floor, and `root_fn` drops its trigger root so CVODE steps over the
  floor. The GH #242 cascade rework began firing *immediate* delay-0 risers
  through `process_firing_batch` (it previously queued only delayed ones), and
  those two cascade paths did not consult `event_dormant` — so a dormant clamp
  was re-fired on every root batch, defeating the suppression and stalling the
  solve (millions of same-instant fires). Both cascade paths now skip
  chatter-dormant events, matching `root_fn`. Non-dormant delay-0 cascades (GH
  #242/#233) are unaffected.
- **Deep state-dependent floor/ceiling scan no longer overflows the recursion
  limit (GH #111 contract; GH #244 scanner).** The floor/ceiling discontinuity
  detector added for GH #244 walked the MathML AST recursively, so a
  BNG-roundtripped observable — a left-leaning `<plus/>` chain hundreds–thousands
  of operands deep — overflowed Python's frame limit and surfaced as
  `ModelError: maximum recursion depth exceeded` on load. Converted to the same
  explicit-stack iterative walk every other AST scanner here already uses;
  visit order (function body before siblings) is preserved, so the emitted
  discontinuity roots are byte-identical.
- **Non-finite initialAssignments (`INF` / `-INF` / `NaN`) now land as the IEEE
  value, and the suite grader compares those sentinels exactly (GH #247; SBML
  suite 00950, 00951, 01811, 01813).** A `<notanumber/>` initialAssignment was
  silently dropped: the section-0 fixpoint loop decided "did this fold move the
  value?" with `abs(new - old) > 1e-30`, and `abs(NaN - old)` is `NaN`, so the
  target kept its stale raw value (case 00950's `R` came out `0.0` instead of
  `NaN`; `<infinity/>` survived only because `abs(inf - 0)` is `inf`). The
  change-detector is now NaN-aware. Independently, the shared suite grader
  (`_grading.compare_series`) treated `INF` / `-INF` / `NaN` expected cells as
  numbers under the `|a-e| <= atol + rtol*|e|` rule, which can never hold for a
  non-finite target, so a *correct* engine value was graded as a mismatch;
  non-finite expected cells are now graded by an exact IEEE match (same NaN-ness,
  same-signed infinity). Both fixes are value-based, so ids merely *spelled*
  `INF` / `NaN` whose data is finite (01811/01813) are unaffected. +4 SBML
  semantic suite cases.
- **L3 species-reference ids directly targeted by rules/events are now live
  stoichiometry symbols (GH #237; SBML suite 00972, 01583).** A
  `<speciesReference id="...">` targeted by an assignment rule, rate rule, or
  event assignment now resolves as a first-class live symbol rather than a baked
  load-time coefficient. Direct rate-rule targets are promoted with the
  reference's declared/initial-assigned stoichiometry as their initial value;
  event-targeted ids use the same hidden-state + observable path as event-driven
  parameters. The reaction emission keeps the id symbolic via the existing
  per-species Functional variable-stoichiometry path, so event priority/order
  changes the ODE coefficient exactly when SBML says it should. +2 SBML semantic
  suite cases.

## [0.11.25] - 2026-06-30

### Added
- **`SolverOptions.event_seed` — seed for random tie-breaking among simultaneous
  equal-priority events (GH #242).** SBML L3v2 §4.11.6 requires that when several
  events fire at the same instant with equal priority, one is chosen at random each
  round. The ODE path now does this via a per-run `std::mt19937_64` seeded from
  `event_seed`, exposed on `Simulator(method="ode").run(seed=...)` and stamped on
  `Result.seed` for models with events. The RNG is consumed ONLY at a genuine
  equal-priority tie, so every model without such ties is byte-identical regardless
  of the seed; a model with ties is fully reproducible for a fixed seed (a fixed
  default keeps a no-arg run reproducible out of the box — the PyBNF-fitting
  requirement). An event-free ODE run stays seed-less (`Result.seed is None`).

### Fixed
- **Same-instant event cascade + honest `RandomEventExecution` (GH #242, #233; SBML
  suite 00978 + 00962/01588/01590/01591/01599/01605/01627, +8).** The event-firing
  drain (`process_firing_batch`) is now the SBML L3v2 §4.11.6 simultaneous-event
  algorithm: a dynamic multiset of execution instances rather than a fixed batch.
  After each immediate (delay-0) fire it re-checks every trigger — a rising edge
  enqueues a new instance (the same-instant cascade the CVODE root finder cannot
  see, since it is a discrete jump), a falling edge cancels the not-done
  non-persistent instances of that event — draining highest-priority-first with a
  seed-keyed random pick among equal-priority ties. This lands 00978 exactly
  (`x=5, y=1, z=3`) and makes the `RandomEventExecution` family correct rather than
  accidentally passing: those tests' divergence monitors previously never fired
  because the missing cascade suppressed them; the cascade activates the monitors
  and the random selection keeps the competing counters balanced (reproducible per
  seed). A `CASCADE_LIMIT` backstops an algebraic loop.
- **Delayed competing non-persistent events now mutually exclude (GH #242; SBML
  suite 01590).** When two delayed non-persistent events compete (each disabling the
  other), the delayed-apply path applied *all* due events at once, so both fired
  every round (`Q == R`, the divergence never grew). Due events are now applied one
  at a time in queue order (preserving the trigger-time random pick), re-cancelling
  any non-persistent pending event whose trigger the assignment just lapsed and
  firing any same-instant delay-0 event it triggers — so exactly one competitor
  increments per round and a delayed apply can drive a delay-0 monitor.

## [0.11.24] - 2026-06-30

### Fixed
- **Simultaneous events now order by their real-valued `<priority>` (GH #233; SBML
  suite 01714, 01533).** SBML L3v2 §4.11.3 priorities are arbitrary real numbers,
  but a constant-folded priority was stored in an integer field — truncating a
  fractional priority (`2.5 → 2`) so two distinct priorities collapsed into a tie
  and their events fired in declaration order instead of priority order (the
  higher-priority event should fire first; when both assign the same variable, the
  one firing *last* wins). A non-integer constant priority is now routed through the
  double-valued `priority_expr` path — which the event dispatcher already evaluates
  and compares as a double — exactly at trigger time. Integer priorities keep the
  fast int field, so every existing model, including the deterministically-ordered
  equal-priority `RandomEventExecution` cases, is byte-identical. Loader-only.
- **An event whose `<trigger>` has no `<math>` — or no `<trigger>` element at all —
  is a no-op that never fires (GH #233; SBML suite 01238, 01239).** Such a trigger
  has no condition that can transition false→true (§4.11.2), so the event never
  runs. The §10 event build already skipped it, but the event-target pre-scan still
  queued the event's assignment target for promotion to event-driven species state,
  stranding the (non-constant) parameter — its column was dropped from the output
  (`var_missing`). The pre-scan now also skips a no-math/absent-trigger event,
  leaving its targets as plain parameters reported at their IC (matching
  RoadRunner). A target assigned by some other event with a real trigger is still
  promoted by that pass. Loader-only.
- Net: +4 SBML Test Suite cases (1510 → 1514).

## [0.11.23] - 2026-06-30

### Added
- **A reaction's `id` used as a rate symbol in an `initialAssignment` now folds to the
  reaction's initial rate (GH #239; SBML suite 01224 / 01233 / 01300 + comp-flattened
  01345).** In SBML L3 a reaction's id is a first-class symbol whose value is that
  reaction's current rate — the kinetic-law extent, in substance/time (NOT ÷V), analogous
  to a species id denoting its amount. bngsim previously left such a symbol unbound, so an
  `initialAssignment` like `p1 = J0` or `p1 = addone(J0)` silently kept the target's
  declared value (`value_mismatch`). Each reaction id is now seeded into the numeric
  context as its initial kinetic-law value (local parameters shadow the global context)
  before the initialAssignment convergence loop, mirroring the species-reference
  stoichiometry seeding; the `<ci>` leaf then resolves it — including inside a function
  definition argument and inside a `comp`-flattened submodel. Loader-only, pure Python.
  Only the *initial* rate is bound (initialAssignment-only); a live runtime reaction-rate
  binding remains out of scope. +4 SBML Test Suite cases (1506 → 1510).

### Documentation
- README's SBML section now also **declares the capability boundary** — the constructs
  bngsim refuses at load rather than approximate (`AlgebraicRule` DAE, non-zero `delay`
  DDE, `fast="true"` fast-equilibrium) — alongside the intentional Avogadro /
  `RandomEventExecution` deviations. Notes that bngsim's unsupported set is narrower than
  some engines' (it supports `VolumeConcentrationRates` and `AssignedVariableStoichiometry`,
  commonly declared unsupported elsewhere).

## [0.11.22] - 2026-06-30

### Fixed
- **A no-math `<algebraicRule/>` is a no-op, not a refused DAE (SBML suite 01244).**
  An `AlgebraicRule` with no MathML states `0 = ∅` — it imposes no constraint, so it
  is not a differential-algebraic system. bngsim's GH #113 unsupported-construct gate
  refused *every* `AlgebraicRule`, including this empty one, declining a model it can
  trivially simulate (a lone variable parameter that just holds its initial value).
  The gate now skips a no-math algebraic rule (mirroring the existing no-math
  event-assignment / rate-rule handling); a non-empty algebraic rule is still a real
  DAE constraint and is refused. +1 SBML Test Suite case (1505 → 1506).

### Documentation
- README now records the SBML semantic test suite's intentional non-conformances:
  the 3 Avogadro cases (bngsim uses the SI-exact `Nₐ`) and the 7 `RandomEventExecution`
  cases (bngsim orders simultaneous equal-priority events deterministically for
  run-to-run reproducibility, rather than the spec-recommended random order).

## [0.11.21] - 2026-06-30

### Fixed
- **`rateOf` csymbol in an initialAssignment folds to the initial `dX/dt`
  (GH #231 sub-cluster 1).** An initialAssignment such as `p2 = rateOf(S1)` takes
  the value of `X`'s time-derivative at the initial state — for a rate-ruled `X`
  the rule's RHS, and for a reaction-driven species the signed
  `Σ net_stoich·kineticLaw` in the species' reporting units. bngsim folds
  initialAssignments numerically at load with no RHS context, so a `rateOf`
  csymbol there was non-foldable and the target silently kept its declared value.
  The section-0 fold now resolves `rateOf(<symbol>)` from the model at the live
  t=0 state (a resolver threaded through `_eval_ast_numeric`, active only on the
  IA/AR fold path so every other fold is byte-identical). SBML Test Suite cases
  01250 / 01251 / 01252 / 01254 now pass.
- **`rateOf` in an event trigger fires deterministically (GH #231 sub-cluster 2).**
  A trigger gated on `rateOf(X)` is bracketed by the CVODE root finder, which
  already refreshes the live derivative buffer during root-finding — but the
  main-loop *rising-edge confirmation* (and the post-fire / delayed-event trigger
  re-checks) read the buffer after only refreshing observables/functions, so they
  saw a stale derivative left by the last RHS probe. The event then fired only at
  some solver tolerances and was silently missed at others. The confirmation and
  cascade re-checks now refresh `current_derivs` first (no-op for non-`rateOf`
  models). SBML Test Suite cases 01261 / 01293 now pass.
- **`rateOf` of a `hasOnlySubstanceUnits=true` species in a variable-volume
  compartment reports the amount rate (GH #231, 01463).** Extends sub-cluster 3
  to a hOSU species whose compartment is driven by a rate rule or an event: the
  integrator stores `amount/V_static` (rescaling to live volume separately), so
  the rateOf buffer is already `d(amount)/dt / V_static` and the existing
  `×volume_factor` scaling recovers the amount rate with no `conc·V̇` correction.
  The loader now flags these species too (only AR-compartment hOSU species, whose
  volume is a rule function rather than an integrator state, stay excluded). SBML
  Test Suite case 01463 now passes.
- Net effect: +7 SBML Test Suite cases (1498 → 1505), 0 regressions.

## [0.11.20] - 2026-06-30

### Fixed
- **`rateOf` of a `hasOnlySubstanceUnits=true` species reports the amount rate
  (GH #231 sub-cluster 3).** A hOSU=true species's symbol denotes its *amount*,
  so the SBML `rateOf` csymbol of it is `d(amount)/dt`. bngsim stores
  concentration and the rateOf accessor buffer holds `d(conc)/dt`, so for such a
  species `rateOf` was off by the compartment volume `V`. For a **constant-volume**
  compartment `d(amount)/dt = V·d(conc)/dt` exactly (the `conc·V̇` chain-rule term
  vanishes), so the loader now flags these species (`Species::report_rateof_amount`)
  and the engine's `refresh_rateof_derivs` — and the codegen rateOf map, in
  lock-step — scale the published rateOf buffer by `volume_factor`. SBML Test Suite
  cases 01455 / 01457 now pass; +2 suite cases (1496 → 1498), 0 regressions.
  Variable-volume hOSU species are excluded (the static volume_factor is not the
  live `V` and the `conc·V̇` term is unhandled) — 01463 is deferred. No change for
  `.net`, `V=1`, or `hasOnlySubstanceUnits=false` species.

## [0.11.19] - 2026-06-30

### Added
- **Time-varying species-reference stoichiometry (GH #237 Phase 2).** An SBML L2
  `<stoichiometryMath>` whose value changes over the simulation — it reads
  `time`, a species, or a rate-rule-/assignment-/event-driven parameter — makes a
  reaction's net stoichiometric coefficient time-varying, so `dS/dt` must use the
  LIVE coefficient instead of the frozen load-time value Phase 1 (§6b) bakes. The
  affected reference is now emitted as its own per-species Functional reaction
  whose rate is `law · stoich_expr` — exactly the SBML extent law
  `stoich_{s,r}·v_r` with the coefficient kept symbolic — so the engine reads it
  fresh every RHS evaluation with no kernel change (the per-species accumulator
  already existed for non-integer stoichiometry). The reaction is forced off the
  Elementary/mass-action path (which would bake the coefficient) onto §9. SBML
  Test Suite cases 00973 / 00989 / 00990 / 00992 / 00994 / 01632 / 01634 / 01743
  / 01745 / 01749 / 01751 now pass, and 00991 / 01581 / 01585 / 01587 (also event-
  coupled, delay events fixed in 0.11.17) unblock; +15 suite cases. SSA refuses a
  variable coefficient (`variable_stoichiometry`, consistent with the non-integer
  gate). Not yet covered: an L3 `<speciesReference>` id directly targeted by a
  rule/event (the id itself becomes a state variable) — 00972 / 01583, which are
  also event-coupled.

### Fixed
- **Duplicate / signed `<speciesReference>` multiset preserves catalysts.**
  Refines the 0.11.18 duplicate-reference fix: instead of collapsing the emission
  multiset through the aggregated `net` dict (which dropped a net-0 catalyst — a
  species on both sides — from the loaded reaction's topology), each reference is
  sign-routed (a negative reactant coefficient is emitted as a product, and vice
  versa). This sums signed/duplicate references correctly AND stays byte-identical
  for ordinary reactions and catalysts, fixing a net2sbml roundtrip-validation
  regression (e.g. `mm_tqssa.net`, whose enzyme is a reactant-and-product
  catalyst).

## [0.11.18] - 2026-06-30

### Fixed
- **Duplicate / signed `<speciesReference>` entries now sum to the net
  stoichiometry (GH #238).** A reaction may list the same species in several
  `<speciesReference>` entries — same side and/or both sides — each carrying a
  SIGNED stoichiometry; SBML L3 §4.11.3 defines the net coefficient as
  Σproducts − Σreactants per species. The §9 Functional emission built the
  reactant/product multisets by extending them once per reference, which
  silently dropped a negative coefficient (`[idx] * int(-1) == []`) and never
  aggregated duplicates — so a reactant listed `+1` and `-1` (net 0) was emitted
  as a net `-1` decay. The multisets are now derived from the already-correct
  aggregated `net` dict. SBML Test Suite cases 01422 / 01426 / 01427 / 01432 /
  01433 ("Multiple species references to the same species") now pass, along with
  01561 / 01562 (uncommon MathML assigned to stoichiometries, whose resolved
  coefficients were also being dropped when negative); +7 suite cases, no
  regressions.

## [0.11.17] - 2026-06-30

### Fixed
- **Events triggered by another event's assignment now fire (GH #233).** An event
  assignment is a discrete state jump; when it pushes another (or the same)
  event's trigger from false to true, SBML L3 §3.4 requires the newly-true event
  to fire. The CVODE root finder only detects zero-crossings during continuous
  integration, never an assignment-induced jump, so such re-triggers were silently
  dropped: the event chain froze after one round. The cold event path now
  re-checks every trigger after each assignment batch (at a root fire and at a
  delayed apply) and schedules the delayed events that just rose, against the
  pre-assignment `trigger_was_true` baseline (falling edges still re-arm). This
  sustains the self-perpetuating persistent-delayed-event chains in the suite.
  SBML semantic suite: 1471→1474 (+3) — 01754, 01758, 01759 (EventT0Firing /
  EventIsPersistent: two persistent delayed events that re-trigger each other,
  one of which would cancel the other were it not persistent); 0 regressions over
  the full 1789-case bngsim-only sweep. T0 firing, persistence and delay mechanics
  were already correct; only the assignment-induced re-trigger was missing.

  Same-instant (delay-0) assignment-induced re-triggers are intentionally *not*
  cascaded: matching the `RandomEventExecution` references (00952/00964–00966 —
  competing same-priority non-persistent events) requires random event selection,
  which a deterministic ODE engine does not implement; firing them deterministically
  makes one competitor out-run the others and trips the tests' divergence monitors.
  No deterministic suite case needs immediate event-triggered-events.

### Fixed
- **No-math `<eventAssignment>` no longer drops its target parameter (GH #233).**
  SBML L3v2 §4.11.5 permits an event assignment to omit its `<math>` child; such
  an assignment never writes its target, so the variable keeps its value (a
  no-op). The loader's event-target pre-scan promoted *every* assignment target
  to event-driven species state regardless of whether it carried math, but the
  emit pass skips no-math assignments — so the promotion never completed and the
  symbol was stranded, dropping the (non-constant) parameter from the output
  entirely (`var_missing`). The pre-scan now ignores no-math assignments, leaving
  such a target as a plain parameter reported at its constant value (matching
  RoadRunner). A target assigned *with* math by another event is still promoted
  by that event's pass; a target assigned only by no-math assignments stays a
  constant parameter (its `_leaf_deriv` becomes the correct `0` in the §8c
  time-varying-volume dilution chain rule, rather than bailing the whole `V̇`).
  SBML semantic suite: 1465→1471 (+6, all `var_missing`→pass) — 01237, 01243,
  01600, 01601, 01602, 01603 (NoMathML × delayed/no-delay × trigger-time/
  assignment-time × lone/mixed-with-real assignment); 0 regressions over the full
  1789-case bngsim-only sweep.

### Added
- **Time-varying compartment volume: dilution term in concentration rates
  (GH #234).** A concentration-valued (`hasOnlySubstanceUnits=false`) species `S`
  of amount `A = [S]·V` in a compartment whose volume `V(t)` varies obeys
  `d[S]/dt = (1/V)·dA/dt − [S]·(V̇/V)`. bngsim stores the concentration, so the
  **dilution term** `−[S]·V̇/V` — the concentration change driven by the volume
  itself moving — must be integrated. Section 8b already emitted it for *rate-rule*
  compartments; this release extends the same physics to the two cases the SBML
  semantic suite exercises that bngsim was dropping:

  - **Assignment-rule compartments with a time-varying RHS.** `C := p1·p2` with a
    rate rule `p2' = r` is just as time-varying as a rate-ruled compartment, but
    `C` carries no rate rule, so no `V̇` was available. The new section 8c derives
    `V̇` as the *chain-rule time derivative of the assignment-rule RHS*
    (`V̇ = Σ_k (∂g/∂x_k)·ẋ_k`) symbolically via sympy — following `time()`,
    rate-ruled parameters, and nested assignment targets, and bailing (emitting
    nothing, exactly as before) on any reference whose time-derivative is unknown
    (a reaction/rule-driven species, an event-promoted parameter, an unparseable
    RHS). A static-RHS AR compartment (`V̇ ≡ 0`) and every non-AR model stay
    byte-identical. The bare-id *amount* selector now reads `V_live` from the AR
    compartment's expression column (`_varvol_ar_amount_map`) so it reports
    `[S]·V_live(t)`, not the stale `[S]·V_static`.
  - **`constant=true` species in a variable-volume compartment.** Such a species'
    *amount* is immutable, but its *concentration* still dilutes as the volume
    moves. It is now un-fixed — so the dilution term takes effect — whenever it
    carries no reaction flux (boundary, or not a reactant/product anywhere), which
    keeps the original `#86` caveat honest: un-fixing never admits flux RoadRunner
    does not apply. The un-fix gate also extends from rate-rule compartments to
    assignment-rule compartments.

  Measured on the SBML semantic test suite (fair four-engine harness, GH #225),
  this captures the full assignment-rule-compartment dilution cluster (00310-00318)
  plus the constant-species dilution cases (01117/01118): bngsim pass count
  **1454 → 1465 (+11)**, with **zero regressions** — verified both on the
  1789-case suite and by a structural scan of the rr / amici / events / BioModels
  parity corpora (1594 models), where the only species whose fixed/un-fixed status
  changes belongs to a single model that simulates byte-identically. The remaining
  recoverable cases in this cluster are distinct root causes (event-resize volume
  assignments, `rateOf` reporting, a stochastic event-priority test) tracked
  separately.

## [0.11.14] - 2026-06-30

### Added
- **SBML `comp` hierarchical models: flatten submodels at load (GH #230).**
  bngsim has no native interpreter for the SBML Level 3 `comp` package
  (ModelDefinition / Submodel / ReplacedElement / Port / SubmodelOutput /
  SBaseRef), so every composed model used to fail downstream as `var_missing` —
  submodel-scoped variables could not be resolved. The SBML loader now detects a
  document that actually composes submodels (`_doc_uses_comp`) and runs libSBML's
  `CompFlatteningConverter` to inline every submodel — renaming scoped ids,
  applying the ReplacedElement / ReplacedBy / SBaseRef substitutions — before the
  existing flat-model pipeline runs, unchanged. This is the same ingestion path
  RoadRunner takes. The flattener is permissive (`abortIfUnflattenable="none"`,
  `stripUnflattenablePackages=True`, validation off) and raises a clear
  `RuntimeError` only when the converter cannot inline at all, rather than
  passing a still-composed document through to a misleading `var_missing`.
  External `comp` references resolve relative to the SBML file's own directory;
  string loads inline in-document ModelDefinitions only. The gate means non-comp
  models never touch the converter.

  Measured on the SBML semantic test suite (fair four-engine harness, GH #225),
  the `comp` cluster goes from **bngsim failing 114** to passing **112 / 123** —
  ahead of RoadRunner (94 / 123) — with **zero remaining cases where RoadRunner
  passes and bngsim fails**. The 11 residual failures are all cases RoadRunner
  also fails: 9 are bngsim's deliberate "cannot faithfully simulate" refusal on
  a construct the *flattened* model contains (a capability gap unrelated to
  comp), and 2 (01345 value_mismatch, 01360 var_missing) RoadRunner misses too.
  Full-suite bngsim pass count: 1351 → 1454 (+103), no regressions elsewhere
  (the change is gated and only fires on composed documents).

## [0.11.13] - 2026-06-30

### Fixed
- **rateOf csymbol: exact per-row recording + local-parameter / no-math rate
  rules (GH #231).** Three independent rateOf gaps the SBML semantic suite hits:
  - a `rate_of__<species>` accessor reads `model.current_derivs`, which is only
    refreshed as a side effect of an RHS evaluation. When a rule/observable
    function reading rateOf was *recorded*, the value was the last internal
    integration step's derivative — and at t=0, before any step, a stale (zero)
    buffer. `fill_row` (no-event path) and the event-loop recorder now probe
    `dx/dt` at the exact recorded `(t, y)` first, so every row is exact,
    including the initial one (01255/01256/01257/01402/01405/01408/01822, and a
    previously var-missing 01236);
  - `rateOf` of a kinetic-law `<localParameter>` is `0` — a local parameter is
    constant by SBML definition and has no species index, so the old code either
    failed to compile an unbound accessor or silently bound to a same-named
    global (01459/01460);
  - a rate rule with no `<math>` means `dvar/dt = 0`; `var` is now still promoted
    to a (zero-RHS) species so a `rateOf(var)` accessor binds, instead of leaving
    it a parameter with no `rate_of__var` to reference (01461).

  Zero regressions (full before/after per-case diff); SBML semantic suite
  1340 → 1351 in-scope passes. Still uncovered in this cluster: rateOf folded in
  an initialAssignment (needs the t=0 derivative at load — 01250–01254), rateOf
  in an event trigger (01261/01293), and the volume scaling of rateOf for an
  `hasOnlySubstanceUnits` species (01455/01457/01463).

## [0.11.12] - 2026-06-30

### Fixed
- **Uncommon MathML, the `avogadro` csymbol, and L3v2 operators (GH #231).**
  bngsim's two MathML translators — the t=0 numeric folder (`_eval_ast_numeric`,
  used for initialAssignment / assignmentRule / stoichiometryMath) and the
  runtime ExprTk emitter (`_ast_to_exprtk_recursive`) — had diverged on several
  constructs the SBML semantic suite exercises:
  - the `avogadro` csymbol was handled by the runtime ExprTk emitter but not the
    numeric folder, so an initialAssignment reading it silently kept its default;
    both now fold to it. bngsim keeps the current SI-exact value `6.02214076e23`
    for both the csymbol and the BNG `_NA` built-in (a physical-correctness
    choice). The SBML semantic suite predates the 2019 SI redefinition and bakes
    the older `6.02214179e23` into its references, so the three cases that test
    the avogadro magnitude to full precision (00960/00961/01323) sit 1.7e-7 off
    and remain a documented known limitation rather than passes;
  - L3v2 `quotient` / `rem` / `max` / `min` / `implies` and `xor` were absent
    from the numeric folder, so an initialAssignment using them silently kept its
    default (01113/01115/01272–01276);
  - `factorial` emitted `exp(lgamma(...))`, but ExprTk has no `lgamma` — so every
    factorial model *load-failed*. It now emits `tgamma` (a newly registered C++
    function backed by `std::tgamma`; native codegen emits the C library
    `tgamma`), and the numeric folder uses `math.gamma` (00957/00958/01486);
  - reciprocal / inverse-hyperbolic trig (`sec`, `csc`, `cot`, `sech`, `csch`,
    `coth`, `arcsinh`, `arccosh`, `arctanh`, `arcsec`, `arccsc`, `arccot`,
    `arcsech`, `arccsch`, `arccoth`) were missing from the numeric folder, so an
    initialAssignment / stoichiometryMath fold over them defaulted to 0 (00958;
    also unblocks the folding in 01561/01562, which still need variable-stoich
    application — GH #237 — to pass);
  - a zero-argument `<plus/>` / `<times/>` / `<and/>` / `<or/>` emitted an empty
    `()` and failed to compile; they now fold to their MathML identity element
    (01530/01531/01564).

  The avogadro fold is suppressed inside event delay/priority expressions
  (`avogadro_value=None`) so it stays on the dynamic-expression path (regression
  guard: suite case 01662). SBML semantic suite 1317 → 1340 in-scope passes,
  zero regressions (full before/after per-case diff).

## [0.11.11] - 2026-06-30

### Fixed
- **Named species-reference stoichiometry as a symbol — Phase 1 (GH #237).** An
  SBML `<speciesReference>` may carry an `id`; that id is a first-class symbol
  whose value is the reactant/product stoichiometry and may be read in a kinetic
  law / rule / initial assignment (testTag `SpeciesReferenceInMath`). bngsim
  bakes the stoichiometry into the reaction coefficient and never registered the
  id, so a rate law like `S1_stoich * k` was an undefined symbol → ExprTk compile
  failure (`load_fail`). Each *static* stoich id is now registered as a constant
  parameter equal to its resolved coefficient, and seeded into the
  initial-assignment numeric context, so it resolves everywhere a global
  parameter would. SBML SId uniqueness guarantees no clash; a local parameter
  that shadows the id stays scoped to its kinetic law. Variable stoichiometry
  (the id targeted by an assignment/rate rule or non-constant `stoichiometryMath`)
  is deferred to Phase 2 — those are skipped here rather than frozen to a
  constant, since the ODE kernel does not yet track a time-varying coefficient.

On the SBML semantic test suite this lifts bngsim from 1295 → 1317 in-scope
passes, zero regressions.

## [0.11.10] - 2026-06-30

### Fixed
- **Simulate algebraic-only SBML models (no ODE state) — CSymbol `time` in
  assignment/initial expressions (GH #229).** A model whose only dynamics are
  assignment rules / functions of the SBML `time` csymbol (plus constants) has no
  species to integrate, yet still defines a trajectory over the requested grid
  (RoadRunner integrates these; the SBML semantic suite grades them against an
  analytical reference). bngsim previously refused them outright with
  `Cannot simulate: model has no species`; the CVODE simulator now evaluates the
  observables + functions once per output row when `n_species == 0`, with no
  integrator state — exactly the value RoadRunner reports. The time csymbol was
  always bound correctly inside the expression layer; the gap was that nothing
  drove it. Models with no species but with *events* are still refused loud (the
  trigger crossing needs an integrator state to anchor); discontinuity triggers
  (piecewise / comparison rules) need no special handling — per-grid evaluation
  already yields the correct piecewise value.
- **N-ary relational MathML in expressions.** The ExprTk translator emitted only
  the first binary pair of a 3+-argument relational (`gt(2,1,2)` became `(2>1)`,
  dropping `(1>2)`); it now expands the MathML chained-pairwise semantics
  (`(x1 op x2) and (x2 op x3) and …`; `neq` ⇒ all pairwise distinct), matching
  the t=0 constant folder. Previously masked because such targets were read as
  folded constants; surfaced once their live trajectory became readable.

The SBML-suite benchmark harness now reads bngsim function/expression columns
when resolving an output variable (an assignment-rule *parameter* whose rule is
not a linear species sum is a bngsim function, e.g. `p2 := 1 + time`) and no
longer short-circuits no-species models to constants. On the SBML semantic test
suite this lifts bngsim from 1236 → 1295 in-scope passes (CSymbolTime cluster
116 → 148), zero regressions.

## [0.11.9] - 2026-06-30

### Added
- **SBML `conversionFactor` support (GH #232).** `Model.from_sbml` now honors a
  model-level and/or species-level `conversionFactor` — the constant parameter
  that scales how a species' amount changes per unit reaction extent
  (`d(amount_i)/dt = cf_i · Σ_r stoich_{i,r} · rate_r`) without changing how
  species appear in rate laws. Realized by scaling the reaction rate per cf: a
  uniform-cf reaction folds the factor into its statistical factor / functional
  `stat_factor`; a reaction whose changed species carry *different* factors is
  split per cf-group (each group emitted with the shared rate and its own
  factor), and is forced off the Elementary path onto the Functional path so the
  split applies. Correct for both ODE and SSA on uniform-cf reactions; mixed-cf
  is ODE-correct. Only cross-compartment *mixed*-cf reactions remain refused
  (loudly). Previously the factor was silently dropped, integrating affected
  models at the wrong rate. On the SBML semantic test suite this takes the
  `ConversionFactors` cluster from 3/81 to 51/81 passing (the rest fail on
  unrelated features); models with no conversionFactor are byte-identical
  (verified: 0 regressions across the 1789-case suite and the BioModels corpus).

## [0.11.8] - 2026-06-29

### Added

- **`convert`: BNGL-action disposition is now a single, documented four-tier
  contract (GH #226).** `parse_bngl_protocol` sorts every action into one of four
  tiers: **execute** (`simulate*`/`parameter_scan`/`set*` → recovered into the
  `ProtocolSpec`), **drop** (pure build/IO directives — recorded in
  `ProtocolSpec.dropped`, no warning, both modes), **flag-lossy** (recognized
  actions BNGsim does not execute but whose omission *changes the results* —
  recorded in the new `ProtocolSpec.lossy` channel; `strict` refuses, best-effort
  warns), and **unknown** (`strict` refuses, best-effort warns + drops). The
  contract is documented at the top of `convert/_protocol.py`. New
  `ProtocolSpec.lossy` field (serialized in `to_dict`/`from_json`, concatenated by
  `combine_protocols`, carried by the `.bngl` writer) lets a consumer tell a clean
  drop from a fidelity risk.

### Fixed

- **`convert`: `writeBNGL` / `setOutputDir` no longer hard-error under `strict`
  (GH #226 consistency bug).** They are pure-IO tooling siblings of the
  already-recognized `writeNET`/`writeSBML`/`writeModel`/`readFile` and now drop
  cleanly like them — previously a model with `writeBNGL()` raised
  `ConversionError` in strict mode while the same model with `writeNET()` converted
  fine, an arbitrary asymmetry.

### Changed

- **`convert`: fidelity-affecting BNGL actions now refuse under `strict` instead of
  silently dropping (GH #226).** `setVolume` (rescales a compartment volume →
  concentrations/rates), `substanceUnits` (concentration-vs-number output
  semantics), a result-changing `setOption` (e.g. `NumberPerQuantityUnit`, which
  scales bimolecular rate constants; `SpeciesLabel` is cosmetic and still drops),
  and `readModel`/`readNetwork`/`readSBML` (load a different model mid-protocol,
  unrepresentable in one `ProtocolSpec`) move from the silent build/IO drop set to
  the lossy channel: `strict=True` now raises (pass `strict=False` /
  `--allow-lossy` to record them as a best-effort lossy conversion). This makes a
  fidelity-affecting action impossible to lose without a signal.
- **`convert`: `quit()` truncates the recovered protocol (GH #226).** In BNG2.pl
  `quit()` halts action processing, so anything after it never runs;
  `parse_bngl_protocol` now stops parsing at `quit` (recording it in `dropped`)
  rather than replaying trailing `simulate`/`set*` actions the reference engine
  would have skipped.

## [0.11.7] - 2026-06-28

### Added

- **`convert`: `net_to_omex` records conversion provenance in the archive
  (`provenance=True`, default).** Two complementary files document how the archive
  was produced and that it is faithful: a COMBINE-standard **`metadata.rdf`**
  (`dcterms` creator = `bngsim <version>`, creation date, description — the channel
  BioModels/COMBINE tools read), and a human/machine-readable
  **`bngsim-conversion.json`** carrying the full **faithfulness verdict** — gate
  level and per-level L0–L4 result, `ok` / `rhs_faithful` / `max_rhs_delta`, any
  dropped/lossy notes, source→target, and counts. This makes the *verified-faithful*
  claim auditable by anyone who opens the archive (and identifies the producing
  bngsim version, e.g. if a conversion bug is later found). The `created` timestamp
  is injectable (`created="…"`) for byte-reproducible archives — verified
  byte-identical on repeated runs with a fixed `created`. `bngsim-omex pack` gains
  `--no-provenance`. New `build_metadata_rdf` helper + `_FMT_JSON`; `.json`/RDF
  entries classify as non-model so the reader still dispatches the SBML master.

### Added

- **`convert`: `net_to_omex` bundles the original `.net` and rule-based `.bngl` into
  the COMBINE archive for provenance (`include_source=True`, default).** A published
  OMEX now carries the modeller's *actual formulation* — the rule-based `.bngl` (as a
  `source` entry) and the flattened `.net` (a secondary, non-master model entry) —
  not just its SBML projection. The SBML stays the `master` curated entry, so this is
  non-breaking for SBML-only consumers, and [BioModels accepts COMBINE archives with
  such supporting files](https://www.ebi.ac.uk/biomodels/model/submission-guidelines-and-agreement)
  (SBML gets full curation; the bundled BNGL/`.net` ride along as authoritative
  source). Motivated by BioNetGen models being deposited as SBML-only, discarding the
  rule-based source. `bngsim-omex pack` gains `--no-source`; pass `include_source=False`
  for the lean SBML + SED-ML archive. New `_FMT_BNGL` format URI; `.bngl` entries
  classify as `source` (provenance, not a dispatchable model — `from_net` cannot read
  rule-based BNGL), fixing a latent misclassification of any `bngl`-substring URI as a
  loadable `.net`.

## [0.11.5] - 2026-06-28

### Added

- **`convert`: `sbml_to_bngl(validate="bng2")` — an on-demand BNG2.pl round-trip
  faithfulness gate for cBNGL (GH #224).** Previously cBNGL faithfulness rested on
  the capability check (a *predictor*); the actual BNG2.pl round-trip lived only in
  the test suite, so every silent over-acceptance (rateOf, `!=`, `inf`) was found
  only by a manual sweep. The gate is now a first-class option: it flattens the
  emitted `.bngl` via `BNG2.pl generate_network`, reloads through `Model.from_net`,
  and compares the ODE right-hand side to the source's — setting
  `ConversionReport.rhs_faithful` / `max_rhs_delta`, raising `ConversionError` on a
  divergence under `strict` (warning otherwise), and raising a clear error if
  BNG2.pl is unavailable / times out / the `.bngl` does not build (faithfulness
  unprovable, never silently skipped). Needs `BNG2.pl` on `$BNGPATH` or `PATH`. CLI:
  `bngsim-sbml2bngl --gate`. The round-trip machinery is the new shared
  `convert/_bng2.py`, reused by the production gate, the test suite, and the corpus
  sweep so all three measure faithfulness identically. Default (`validate=None`) is
  unchanged — no external tool required.

### Fixed

- **`convert`: the cBNGL faithfulness oracle now probes the ODE RHS at several t>0
  instants, not just t=0 (GH #224).** BNG2.pl rewrites `>=`/`<=` against numeric
  literals to `>`/`<`, so a time-pulse "on" at exactly t=0 in the source reads
  "off" in BNG — a measure-zero boundary that made trajectory-faithful pulse models
  (e.g. MODEL1112110002, BIOMD0000000527, MODEL0848507209) read as full RHS
  mismatches under a t=0-only probe. Sampling generic t>0 instants both eliminates
  that false positive and genuinely exercises time-dependent forcing (a single
  probe time could miss a pulse-train mismatch). The flat-`.net` L2 gate was already
  unaffected (it probes t=0 **and** t=1.0, and the `.net`/ExprTk path has no `>=`→`>`
  rewrite).

## [0.11.4] - 2026-06-28

### Added

- **`convert`: cBNGL writer recovers reactant-cross-compartment transport (GH #224
  deliverable 1b extension).** A reaction whose *reactants* sit in different
  compartment volumes (not just reactant-here / products-elsewhere) was previously
  refused fail-loud ("reactants in more than one compartment"). It is now recovered:
  the per-species signed-flux split (`_expand_functional_reactions`) already emits
  one `0 -> Molᵢ()` flux per affected species at rate `(sᵢ/Vᵢ)·P`, dividing by each
  species' *own* volume — so it needs no single-reactant-compartment assumption and
  carries an arbitrary spread of reactant/product compartments. The capability check
  now refuses only cross-compartment reactions the split genuinely cannot carry
  (non-functional / applied-reactant-factor / non-unit statistical factor —
  `_handled_transport` is the gate). **+20 corpus models recovered, all BNG2.pl
  round-trip RHS-faithful** (delta ≤ 1e-8; e.g. BIOMD0000000075, BIOMD0000000161).
  Combined SBML→(c)BNGL/net faithful over the ODE-verified rr_parity corpus:
  **1186/1237 ≈ 95.9%** (was 94.3%).

### Fixed

- **`convert`: cBNGL writer refuses two BNG2.pl-dialect constructs fail-loud instead
  of silently emitting an unbuildable `.bngl` (GH #224 audit).** Both passed the
  capability check (`ok=True`) yet BNG2.pl produced no network — a silent
  over-acceptance. The `.net`/ExprTk channel carries both, so these are cBNGL-only
  refusals:
  - the **not-equal operator `!=`** in a function — BNG2.pl's parser reads the bare
    `!` as logical-not and aborts (3 corpus models, e.g. MODEL1006230049);
  - **non-finite parameter values** (±inf / nan — e.g. FBA flux-bound `_lp_*`
    parameters) — BNG2.pl has no such literal and reads it as an undefined parameter
    (4 corpus models, e.g. MODEL1703150000).

### Notes

- **GH #224 residual triage (BNG2.pl round-trip over all 869 cBNGL-accepted corpus
  models): 856 faithful, 2 benign, 11 oracle-unavailable, now 0 silent.** The 2
  `rhs_mismatch` (BIOMD0000000527, MODEL0848507209) — and the previously-flagged
  MODEL1112110002 — are the documented measure-zero `time()>=0`→`>` boundary
  artifact: BNG2.pl rewrites `>=`/`<=` against numeric literals to `>`/`<`, so a
  pulse "on" at exactly t=0 in the source is "off" in BNG; **RHS is exact at every
  t>0** (delta=0), wrong only at the instant t=0. Trajectory-faithful, not a defect.
  The 11 oracle-unavailable are 4 BNG2.pl `generate_network` timeouts (large
  combinatorial networks) and the 7 now-refused dialect models above. The cBNGL
  faithfulness oracle is the BNG2.pl round-trip by design (no in-tree cBNGL reader);
  the sweep harness is `parity_checks/rr_parity/cbngl_bng2_validate.py`.

## [0.11.3] - 2026-06-28

### Fixed

- **`convert`: `rateOf` csymbols in functions are now refused fail-loud by both
  the `.net` and cBNGL writers, and best-effort emission no longer crashes (GH
  #224 re-sweep).** An SBML `rateOf` csymbol (`<csymbol ...rateOf>`) is wired to a
  reaction's derivative at run time; the loader names it `rate_of__<species>`, and
  it is not a species/parameter/observable — so neither the flat `.net` text nor
  cBNGL can carry it (the reloaded function fails to compile, `Undefined symbol:
  'rate_of__x8'`). Previously the `.net` path **crashed** on the L2 reload and the
  cBNGL path **silently accepted** the unrepresentable model (no in-tree gate),
  violating the no-fabricated-faithful-artifact principle. A shared
  `_rateof_refs(data)` detector now flags such functions as `lossy` in both
  `capability_report` and `bngl_capability_report` (so `strict=True` raises a
  clean `ConversionError`); the `strict=False` `.net` reload is guarded to record
  `rhs_faithful=False` for an already-lossy model rather than propagate the loader
  error. Found auditing the GH #224 converter corpus re-sweep (2 models:
  BIOMD0000000696, Iarosz2015/BIOMD0000000775).

### Notes

- **GH #224 converter re-sweep over the ODE-verified rr_parity corpus (1237 models
  bngsim↔RoadRunner agree on):** combined faithful (`.net`-faithful **or**
  cBNGL-accepted) = **1166/1237 ≈ 94.3%**, exceeding the epic's ~90% projection.
  Flat `.net` alone now reaches **90.7%** (the GH #223 flux-expansion lift), and
  cBNGL recovers **44** static-compartment / transport models `.net` cannot carry.
  Audit of the 71 refused-by-both: all are genuine multi-blocker models
  (reactant-cross-compartment reactions, amount-valued non-unit species,
  live/time-varying volumes, state-triggered/pulse events) — **no over-refusals**;
  the hypothesized MM→elementary over-refusal does not occur in this subset. The
  re-sweep harness is `parity_checks/rr_parity/convert_sweep.py`.

## [0.11.2] - 2026-06-28

### Added

- **`convert`: OMEX/SED-ML multi-experiment & multi-file protocols compose into a
  `.bngl` actions block (GH #222).** The reverse path (`omex_to_net`) previously
  under-consumed the protocol channel: an archive with several SED-ML *files* used
  only the master/first, multiple experiments *within* a SED-ML were parsed then
  dropped (only the primary horizon drove the gate), and no protocol was ever
  emitted — the round trip was asymmetric with `net_to_omex`, which carries the
  whole multi-experiment protocol forward. Now:
  - `OmexArchive.load_full_protocol()` composes **every** experiment from **every**
    SED-ML file in the archive into one ordered `ProtocolSpec`; `omex_to_net` warns
    when more than one SED-ML file is present (the master no longer silently wins).
  - `convert.combine_protocols(specs)` concatenates independent protocols in order,
    inserting a `resetConcentrations` boundary between distinct files (they are not
    continuations); a single spec is returned verbatim, so the common one-file
    round trip stays exact.
  - `omex_to_net` emits the composed protocol as a **`.bngl` actions block**
    alongside the `.net` (`<stem>.bngl` by default; `actions_out=`/`write_actions=`
    to redirect or suppress), reusing the existing `write_bngl_protocol` writer.
    The path is reported as `ConversionReport.bngl_out`. The gate's L3 horizon
    still uses the representative (first-deterministic) experiment — gating every
    horizon is a deliberate non-goal (risks a stiff-grid hang for no added signal).
  - `bngsim-omex to-net` gains `--actions-out` / `--no-actions`; `unpack` notes when
    an archive carries more than one SED-ML file.

## [0.11.1] - 2026-06-28

### Fixed

- **`.net` writer: reactant-independent functional laws were silently zeroed (the
  GH #223 "25 undiagnosed" silent losses).** An `apply_species_factor=False`
  functional reaction was emitted with a reactant-division guard
  `if(r>1e-300, P/r, 0)` (the `.net` reader re-multiplies a functional rate by the
  reactant amounts). When a reactant was zero at the initial state this zeroed the
  whole propensity — but `P` is reactant-*independent* (a constant influx) or
  saturable/Hill or a reversible flux, so the source RHS is nonzero there. The
  network then mis-fired (e.g. a species fed only by a constant influx stayed
  pinned at zero). This was the **same defect 1b fixed for cBNGL**, never applied
  to the older `.net` writer. Such reactions are now rewritten as per-species
  signed-flux reactions (`0 -> Xᵢ` with rate `sᵢ·P`, carrying `P` whole) — the
  flux-expansion machinery is now shared between the `.net` and cBNGL writers
  (`convert/_net_writer.py`). A full rr_parity sweep diagnosed **all 25** as this
  one root cause; the fix converts them (plus 17 assignment-rule models with the
  same latent guard) faithfully: cap-clean `.net` faithful rose **1162 → 1204**,
  with **zero** remaining silent losses and **zero** regressions.
  - The rewrite fires **only** for the reactions the guard would actually break —
    detected by `_guard_unsafe_reactions`, which evaluates each candidate's rate
    function at the initial state and flags it only when a reactant is zero there
    while the propensity is non-negligible. Every other reaction keeps its
    original topology, so structural round-trip identity is preserved for the
    ~1160 unaffected models (only 43 gain a flux reaction). A miss cannot ship
    silently — the default `"L2"` RHS self-check still measures any residual loss.
  - `ConversionReport.ok` under the `"L2"` path now treats the RHS self-check as
    the authoritative reaction-level gate (with species conservation), so a model
    the writer legitimately flux-expands is `ok` when RHS-faithful even though its
    reaction count/topology changed. The counts-only `"L1"` mode is unchanged.

## [0.11.0] - 2026-06-28

### Changed

- **`sbml_to_net` default gate is now `validate="L2"` — a direct ODE-RHS identity
  self-check (GH #223).** Previously the default `validate="L1"` checked only
  structural counts/topology, so a network whose forcing the flat `.net` cannot
  carry (assignment-rule / time-dependent constructs the writer froze to
  constants) shipped **confidently wrong with nothing flagged**. A full corpus
  sweep found ~42 such models that pass the L1 counts but diverge in RHS. The new
  default reloads the emitted `.net` and confirms it reproduces the source ODE
  right-hand side (the project's faithfulness measure) at the initial and a
  nonlinear-probe state — no integration, no `BNG2.pl`. Under `strict=True` (the
  default) a divergence now **raises** `ConversionError` (escape hatch:
  `strict=False` / `--allow-lossy`, which emits the best-effort network and
  records `ConversionReport.rhs_faithful=False`); the counts-only behavior remains
  available as `validate="L1"`. This is a behavioral break: a strict conversion of
  one of those ~42 models now refuses instead of silently shipping. `omex_to_net`
  already ran the full L0–L4 gate and is unchanged.
  - A structural predictor was rejected as unreliable: `is_const`/function-shadow
    flags do not separate the lossy models from the ~125 assignment-rule and ~166
    `time`-csymbol models that convert **faithfully** (their rules fold into live
    `.net` functions). Measuring the RHS identity is precise where a heuristic
    over-flags. New `ConversionReport.rhs_faithful` field; `bngsim-sbml2net`
    gains `--gate L2` (the new default; `--gate L1` keeps the counts-only check).

### Fixed

- **cBNGL writer: catalytic identical-reactant reactions were rate-doubled (GH
  #224, `BIOMD0000000233`).** `_bng_stat_factor` returned `∏(mⱼ!)` as BNG's
  symmetry divisor, but BNG only treats identical reactant molecules as
  interchangeable when the reaction acts on them the same way. For a catalytic
  transform like `X + X -> X + Y` BNG preserves one `X` (mapped to the product
  `X`) and consumes the other, so the factor is `1!·1! = 1`, **not** `2!` — the
  `×2` pre-compensation then doubled the emitted rate. Corrected to
  `∏ⱼ pⱼ!·(mⱼ−pⱼ)!` with `pⱼ = min(reactant, product mult)`, verified against
  `BNG2.pl` 2.9.3 across 1-, 2- and 3-body reactant groups; `BIOMD0000000233` now
  round-trips RHS-exact. The change only affects identical-reactant reactions
  whose species also appear in the products (the broken class) — distinct-reactant
  reactions are unchanged. The other four corpus round-trip mismatches were
  confirmed benign: all are the measure-zero `time()>=0` → `>` boundary artifact
  (wrong only at `t=0`, trajectory-faithful).

## [0.10.18] - 2026-06-27

### Added

- **Cross-compartment transport: cBNGL recovery (GH #224, deliverable 1b).** A
  transport reaction (reactant in one volume, products elsewhere — the
  `per_species_volume_scaling` per-species `1/Vᵢ` asymmetry) was refused by 1a
  because a single flat unit-volume `.net` reaction can only carry a symmetric
  `±P`. `sbml_to_bngl`/`write_bngl` now recover it by rewriting every `asf=False`
  functional reaction into per-species **signed-flux** reactions — one `0 -> Molᵢ()`
  per affected species with rate `(sᵢ/Vᵢ)·P` — so `BNG2.pl generate_network` →
  `Model.from_net` reproduces the source ODE RHS exactly.
  - The zero-reactant (pure-flux) form is the correctness key: it also fixes
    **reversible** transport (`Vin·k·Ain − Vout·k·A`) and reactant-independent
    functional influx (`kabs·MD`), which the prior reactant-guard emission silently
    zeroed whenever the reactant was momentarily zero (e.g. at t=0). This was a
    latent 1a bug for within-compartment functional laws too, now fixed.
  - Validated over the rr_parity ODE corpus: **36/37** clean-convert transport
    models round-trip RHS-faithful (the one exception, BIOMD490, is now refused —
    `ceil`, see below). The named #224 transport-heavy model BIOMD600 converts.
  - Still refused fail-loud: **reactant-cross-compartment** reactions (reactants
    span >1 volume — the split assumes a single reactant compartment) and exotic
    transport rate kinds (elementary, applied reactant factor with reactants,
    non-unit statistical factor).

### Fixed

- **cBNGL writer dialect gaps surfaced by 1b.** (1) The π symbol `_pi` is folded
  to its numeric literal — BNG2.pl reserves the name but does not define it and
  aborts at network generation whether it is left bare or declared as a parameter.
  (2) Functions using `ceil`/`floor` (no BNG primitive) are refused fail-loud
  rather than emitting a `.bngl` BNG silently misparses and aborts on.
  - Note: BNG2.pl rewrites `>=`/`<=` against any operand to `>`/`<` in function
    conditions, so a `time()>=T` stimulus boundary differs from the source only at
    the measure-zero instant `t=T` (the trajectory is faithful; an RHS probe
    exactly at `t=T` differs).

## [0.10.17] - 2026-06-27

### Added

- **SBML events → BNGL actions (GH #224, phase 2).** `sbml_to_bngl` now recovers
  **fixed-time** events instead of refusing every event model: the source SBML is
  re-read (the loaded model exposes only an event *count*), each event classified,
  and the tractable ones emitted as a trailing `begin actions` block —
  `simulate` phases with `setConcentration` state changes at each fire time,
  targets translated to the compartment-qualified `@comp:Mol()` pattern.
  - A **fixed-time** event is a rising time threshold (`time >= T`) with a
    constant `T`, constant delay, and constant assignment values. **State-triggered**
    events (the trigger depends on a species / state variable) and non-schedulable
    shapes (pulse windows like `(time>50)&&(time<=300)`, expression thresholds,
    non-constant trigger times/delays/values) have no actions form and are refused
    fail-loud, each with a plain-English reason naming the offending event.
  - New `write_bngl_protocol(ProtocolSpec) → str` — the BNGL-actions serializer
    mirroring `parse_bngl_protocol` (round-trips exactly); exported from
    `bngsim.convert`. New `bngsim.convert._events` carries the SBML event-walk /
    classifier (`sbml_events_to_protocol`). `sbml_to_bngl` gains `t_span`/`n_points`
    for the actions horizon and carries the recovered `ProtocolSpec` on the report.
  - Validated over the rr_parity ODE corpus: 44 clean fixed-time event models
    convert; the emitted actions are valid executable BNGL (BNG2.pl builds the
    network, applies each `setConcentration`, and runs to the horizon).

## [0.10.16] - 2026-06-27

### Added

- **`bngsim-sbml2bngl` CLI (GH #224).** A console entry point for the SBML→cBNGL
  writer, mirroring `bngsim-sbml2net`/`bngsim-net2sbml`: `bngsim-sbml2bngl
  model.xml [-o out.bngl] [--allow-lossy] [-q]`. Writes the `begin model` …
  `end model` block (append a `generate_network` action to run it through
  BNG2.pl), exits non-zero when the capability check refuses an out-of-scope
  construct, and `--allow-lossy` downgrades that to a best-effort emit. There is
  no `--gate` (no in-tree cBNGL reader yet — the round-trip gate lives in the
  test suite, against BNG2.pl).

## [0.10.15] - 2026-06-27

### Added

- **SBML→cBNGL writer: recover static compartments (GH #224, deliverable 1).**
  `sbml_to_bngl` / `write_bngl` (in `bngsim.convert`) serialize a loaded model to
  **compartmental BNGL** — a `begin compartments` block (one `name 3 V` per
  distinct static `volume_factor`) plus compartment-qualified species, reactions,
  parameters, observables and functions. A non-unit-volume model that the flat,
  unit-volume `.net` channel refuses now round-trips RHS-exact through
  `BNG2.pl generate_network` → `Model.from_net` (verified faithful over the
  rr_parity ODE corpus: 22/22 clean within-compartment static-volume models).
  - The reaction rates are reused from the `.net` writer's token logic and
    pre-scaled by `V^(n-1)·∏(mⱼ!)` so BNG2.pl's `unit_conversion` and symmetry
    bake cancel exactly, leaving the source propensity after the flat readback.
  - `bngl_capability_report` refuses fail-loud (under `strict`) the out-of-scope
    classes, each with a plain-English reason: cross-compartment / transport
    reactions (the per-species `1/V` asymmetry — deliverable 1b), events (→ phase 2
    actions), live / time-varying compartment volumes, Michaelis–Menten kinetics,
    amount-valued non-unit species, and assignment-rule report species
    (`_ar_report_map`, GH #205/#223). Function bodies are normalized for BNG's
    dialect (`and`/`or` → `&&`/`||`, natural `log` → `ln`, `log2`/`log1p` → `ln`
    identities). Events→actions and the cross-compartment set are future
    deliverables of the #224 epic.

## [0.10.14] - 2026-06-27

### Added

- **The SBML→``.net`` direction is now symmetric with ``.net``→SBML (GH #211).**
  The forward container path (``net_to_sbml``/``net_to_omex``) consumes a source
  ``.bngl`` to carry the real protocol and drive the gate's L3 horizon; the
  reverse now does the mirror with the SED-ML sidecar:
  - ``sbml_to_net(sedml=…)`` parses the SED-ML time course (via
    ``read_sedml_protocol``) and runs the ``validate="full"`` gate's L3 over the
    model's *own* horizon instead of the blanket ``t=0..100`` grid (avoids the
    stiff-model hang and exercises the trajectory the modeller actually ran). The
    parsed ``ProtocolSpec`` is attached to the report. ``bngsim-sbml2net --sedml``
    adds the flag and auto-detects a sibling ``<stem>.sedml`` — the exact analog
    of ``bngsim-net2sbml --bngl``.
  - ``omex_to_net(omex, out, gate="full")`` — the reverse of ``net_to_omex``:
    reads a COMBINE archive's master SBML + SED-ML, converts the SBML to a
    ``.net``, and uses the carried protocol for the L3 horizon. CLI verb
    ``bngsim-omex to-net``. Defaults to the full L0–L4 gate, so the unpack ships a
    verified-faithful verdict just like ``pack``.

### Fixed

- **``net2sbml`` now round-trips time-gated forcing functions (GH #211).** ExprTk
  infix boolean operators ``and``/``or`` (ubiquitous in ``if((time()>=a) and
  (time()<=b), …)`` dosing/stimulus functions) are keyword operators that
  ``libsbml.parseL3Formula`` rejects — it accepts only ``&&``/``||``. The
  ExprTk→MathML normalizer now rewrites them (word-boundary-anchored, so
  ``ligand``/``factor`` are untouched and ``nand``/``nor`` are left to fail loud).
  Over the rr_parity corpus this fixed **~70 models** whose L2 round-trip the
  conversion previously refused with "could not parse expression".
- **``net2sbml`` no longer emits a document with duplicate parameter ids on a
  best-effort volume-scaled conversion.** A ``.net`` synthesized from a
  cross-compartment volume-scaled model can carry the same ``_vd_…`` helper
  function many times over; the writer keyed them by name and emitted each as its
  own SBML parameter, producing duplicate ``SId``s that bngsim's own ``from_sbml``
  then rejected. ``write_sbml`` now collapses byte-identical same-named functions
  to one and guards display-name uniqueness.

### Changed

- **``capability_report`` flags live/assignment-rule compartment-volume report
  corrections as lossy.** A species whose reported concentration is corrected at
  run time for a live or assignment-rule volume (the SBML loader's ``_varvol_*``
  maps) carries a Python-side report transform that the ``.net`` graph cannot
  store, so a reloaded network mis-scales it. ``sbml_to_net(strict=True)`` now
  refuses such a model rather than silently shipping a wrong network. (The
  authoritative faithfulness guard remains the L0–L4 ``validate="full"`` gate,
  which catches the silent assignment-rule/forcing losses a structural heuristic
  cannot predict without false-flagging the many models that convert faithfully.)

## [0.10.13] - 2026-06-27

### Changed

- **A fabricated default protocol is no longer passed off as the modeller's
  (GH #211 fidelity).** When no real simulation protocol is available — no
  ``.bngl`` companion, or a ``.bngl`` with no ``simulate`` action —
  ``net_to_omex`` and ``bngsim-net2sbml --sidecar`` still bundle a runnable
  default uniform time course (``t=0..100``, ODE, every observable), but they now
  (a) emit a :class:`bngsim.ConversionWarning` saying the protocol was
  synthesized and is **not** the modeller's, and (b) mark the SED-ML with a
  ``<bngsim:synthesizedDefault>`` annotation and a distinguishing report name so
  a consumer can never mistake the placeholder for a real protocol. ``read_sedml``
  / ``read_sedml_protocol`` surface that marker as a warning on read.
  ``write_sedml`` gains a ``synthesized_default`` flag. Previously the default was
  emitted silently and unlabeled (inherited from the #218/#219 sidecar/OMEX
  fallback) — on the bng_parity corpus that was ~134 of 663 models (130
  no-``simulate``-action ``.bngl`` + 4 no-``.bngl``) shipping an unmarked
  fabricated protocol. A real ``.bngl`` protocol is still carried verbatim with no
  warning or marker.

## [0.10.12] - 2026-06-27

### Fixed

- **L4 symbolic equivalence no longer reports a misleading ``not-equal`` for
  floating-point round-off / catastrophic cancellation (GH #211/#217).** The
  BNGL-float → MathML → back conversion round-trip reassociates arithmetic and
  leaves residuals ``sympy.simplify`` cannot crush to 0 even when the RHS is
  identical to machine precision (the L2 numeric gate, RHS identity at ~1e-15,
  passed on every such model). Over the full bng_parity corpus (663 models) the
  old equality test produced **261 false ``not-equal``, 0 genuine differences**.
  ``_symbolic_verdict`` now adjudicates a non-zero residual Δ in two stages:
  (1) a cheap coefficient screen — if every coefficient is below ``1e-9`` of the
  RHS's own coefficient scale it is forgiven as ``equal`` (noted "up to
  floating-point round-off"); (2) otherwise the residual is **evaluated
  numerically** at sample states (robust to the term cancellation that makes a
  per-coefficient magnitude a poor proxy — e.g. a real ``5.79e+77`` residual
  coefficient that nets to ~1e-16). Only a residual that evaluates *non-zero* is
  ``not-equal`` (now meaning "numerically confirmed different"); a residual that
  evaluates ~0 but resists symbolic reduction is honestly ``inconclusive``. Over
  the corpus this turns the 261 false ``not-equal`` into **592 equal / 67
  inconclusive / 0 not-equal**. L4 remains best-effort and non-gating.

## [0.10.11] - 2026-06-27

### Added

- **The converter recovers and carries the real BNGL simulation protocol — a
  faithful, runnable consumer deliverable (GH #211, Option 3).** A BioNetGen
  ``.net`` discards the ``simulate`` protocol at network-generation time (it
  lived in the ``.bngl``), so the converter previously could only synthesize a
  default. It now optionally takes the source ``.bngl`` and:
  - **Parses the whole actions block** into a new ordered, JSON-serializable
    ``bngsim.convert.ProtocolSpec`` IR (``Experiment`` | ``StateChange`` steps)
    via ``parse_bngl_protocol`` — a hand-rolled, shippable parser (no BNG2.pl, no
    ``parity_checks`` dependency) covering ``simulate``/``simulate_ode``/``_ssa``/
    ``_nf``, ``parameter_scan``, ``setParameter``/``setConcentration``/
    ``resetConcentrations``/``saveConcentrations``, hash (``{k=>v}``) and
    positional args (incl. literal arithmetic like ``t_end=>3600*5``), ``\``
    continuations, comments, and bare-vs-``begin actions`` forms. Build/IO
    directives (``generate_network``, ``writeSBML``, …) are dropped and recorded;
    unknown actions fail loud under ``strict`` / warn under best-effort. Parses
    the full benchmark ``.bngl`` corpus (103/103).
  - **Drives the ``"full"`` gate's L3 at the model's own horizon.**
    ``net_to_sbml(bngl=…)`` (and ``net_to_omex``) simulate L3 over the protocol's
    real, integrable time range instead of the blanket ``t_end=100`` — avoiding
    the stiff-model hang and exercising the trajectory the modeller actually ran
    (ODE-vs-ODE regardless of the protocol method; L2 already proves RHS identity
    so equivalence holds at every operating point). CLI ``--bngl`` (+ sibling
    ``<stem>.bngl`` auto-detect).
  - **Carries the whole protocol into SED-ML / OMEX.** New
    ``write_sedml_protocol`` / ``read_sedml_protocol`` emit a uniform time course
    + task per experiment, a ``repeatedTask`` + range + ``setValue`` per
    ``parameter_scan``, and ``changeAttribute``-derived models for accumulated
    ``set*`` overrides — multi-experiment and multi-stage, beyond the single
    course of the #218 sidecar. A verbatim ``ProtocolSpec`` ``<annotation>`` makes
    the round-trip exact (the #218 fidelity-via-annotation pattern); foreign
    SED-ML reconstructs best-effort from the standard elements.
  - **``net_to_omex(bngl=…, gate="full")``** packages the converted SBML + the
    real multi-experiment SED-ML, gated on the L0–L4 ladder by default (the OMEX
    is the faithful-deliverable container; exit 1 / ``ConversionReport.ok`` False
    on a hard-gate failure). CLI ``bngsim-omex pack --bngl``/``--gate``.

## [0.10.10] - 2026-06-27

### Added

- **The converters can now gate themselves on the full L0–L4 ladder — "convert
  *and prove faithful*" (GH #211, #217).** `net_to_sbml` / `sbml_to_net` (and
  their CLIs `bngsim-net2sbml` / `bngsim-sbml2net`) gain `validate="full"`, which
  runs the complete conversion-validation framework (L0 syntactic validity, L1
  structural equivalence, L2 round-trip identity, L3 numerical equivalence as
  hard gates; L4 symbolic equivalence best-effort, non-gating) on the artifact
  the converter just produced and fails loud — `ConversionReport.ok` is False
  (CLI exit 1) — when a hard gate fails. Previously the default `validate="L1"`
  ran only the lightweight structural + ODE-RHS round-trip, so conversions were
  not gated on L0/L3/L4; the full ladder lived in `validate_conversion`, which
  the converter never called. The verdict is attached as the new
  `ConversionReport.validation` field. `validate="L1"` remains the default
  (fast, non-breaking); the L3 simulation grid for `"full"` is tunable via the
  new `t_span`/`n_points` arguments (CLI `--t-end`/`--n-points`).
- **CLI `--gate {none,L1,full}`** on `bngsim-net2sbml` and `bngsim-sbml2net`
  selects the validation gate (default `L1`). The pre-#217 `--no-validate` flag
  is retained as a hidden alias for `--gate none`.
- **`bngsim.convert.grade_conversion(direction, source_model, target_text, …)`** —
  the shared core that grades an *already-converted* model pair at L0–L4 without
  re-running the forward conversion. `validate_conversion` (path-driven) and the
  converters' `validate="full"` gate both delegate to it, so wiring the gate into
  the converter does not convert twice (only a cheap re-serialize of text the
  converter already produced).

## [0.10.9] - 2026-06-27

### Fixed

- **net2sbml: a `pi`-spelled model symbol no longer collapses to the π constant
  (silent wrong RHS).** libsbml's `parseL3Formula` reads the bareword `pi` —
  *case-insensitively*, so `pi`/`pI`/`PI`/`Pi` too — as the π constant. A model
  with a function or parameter so named (e.g. `pI = c·V0·Inorm`) therefore
  serialized that symbol as `<pi/>` = 3.14159…, a silently-wrong rate law that
  passed structural checks but diverged numerically. In BNGL/ExprTk π is spelled
  `_pi`, so the writer now routes genuine π through a sentinel and reverts any
  other π-constant the parser produces back to a name reference (applied to both
  function/observable expressions and kinetic laws). Surfaced by net2sbml
  round-tripping a bng_parity model whose `N`-production rate is a `pI` function;
  with this fix the full bng_parity ODE corpus (591 networks) converts **and**
  L1-validates 591/591.

## [0.10.8] - 2026-06-27

### Added

- **net2sbml translates `log2` / `log1p` (GH #216 follow-up).** These ExprTk
  functions have no MathML primitive but reduce to `ln` exactly
  (`log2(x) = ln(x)/ln(2)`, `log1p(x) = ln(1+x)`), so the SBML writer now
  rewrites them (paren-matched, nesting-safe) instead of refusing the
  conversion. Surfaced by a parity sweep of the bng_parity ODE corpus, where 3
  models used `log2` in a function and previously failed to convert.

### Fixed

- **SBML loader: lazy piecewise constant-folding (real-valued powers).** The
  load-time expression folder (`_eval_ast_numeric`) evaluated *both* arms of a
  `piecewise` eagerly, even the un-taken one. For the common real-cube-root
  idiom `if(x>0, x^(1/3), -(abs(x))^(1/3))` the un-taken `x^(1/3)` arm is a
  *complex* under Python's `**` when `x<0` (the C ODE engine yields a real NaN),
  which then leaked into a downstream comparison and crashed `from_sbml` with
  `TypeError: '>' not supported between 'complex' and 'float'`. Piecewise is now
  folded lazily — test each condition first, evaluate only the taken branch —
  matching the runtime engine; the `pow`/`root` operators additionally treat a
  complex result as non-foldable. Surfaced by net2sbml round-tripping a
  bng_parity model with a guarded cube root.

## [0.10.7] - 2026-06-27

### Added

- **OMEX / COMBINE archive packaging (GH #219, converter epic #211).** A COMBINE
  archive (`.omex`) is the standard zip container that bundles a model (SBML),
  its simulation protocol (SED-ML), and a `manifest.xml` listing every entry's
  *format URI* — it is how these artifacts travel (BioModels distributes models
  this way). The converter now has the packaging layer on top of the existing
  network (SBML, #216) and protocol (SED-ML, #218) channels:
  - `bngsim.convert.read_omex` unzips an archive, parses `manifest.xml`, and
    returns an `OmexArchive` whose `load_model()` / `load_protocol()` accessors
    dispatch the master model and SED-ML entries to the bngsim readers — turning
    a `.omex` into a runnable network + protocol. Versioned COMBINE format URIs
    (`…/sbml.level-3.version-2`) dispatch by substring; the SED-ML protocol is
    pointed at the archive's extracted model so the returned `EvaluationSpec` is
    directly runnable. Archives with no SED-ML fall back to a default uniform
    time course over the model's observables.
  - `bngsim.convert.write_omex` bundles SBML/`.net` + SED-ML (+ optional RDF
    metadata) plus a generated `manifest.xml` into a `.omex` zip, with correct
    COMBINE format URIs and a master-model marker.
  - `bngsim.convert.net_to_omex` packages a `.net` end-to-end: convert to SBML,
    derive a SED-ML protocol, and bundle both into one archive.
  - `bngsim-omex pack`/`unpack` CLI.

  Implementation is `zipfile` + a hand-rolled manifest reader/writer over the
  stdlib XML tools (no `python-libcombine` dependency); extraction refuses
  zip-slip paths. New module `convert/_omex.py`; tests in `test_omex.py`.

## [0.10.6] - 2026-06-27

### Added

- **SED-ML sidecar protocol channel (GH #218, converter epic #211).** SBML and
  `.net` carry structure+math only — neither encodes a *simulation protocol*
  (start/end time, number of points, outputs, solver, tolerances). SED-ML is the
  COMBINE-standard sidecar that supplies it, so the converter now has a protocol
  channel alongside the network channel (#215/#216). The bridge type is
  `bngsim.EvaluationSpec` (whose `.evaluate()` is a runnable job):
  - `bngsim.convert.read_sedml(source, …)` — parse a SED-ML uniform time course
    → `EvaluationSpec`, recovering the time grid, output selectors, and
    solver+tolerances (KiSAO terms). Namespace-agnostic (SED-ML L1 V1–V4 parse);
    `model_source`/`model_format` overrides let an `sbml2net` `.net` run under
    the SED-ML protocol.
  - `bngsim.convert.write_sedml(spec, out_path=None, …)` — emit a SED-ML (L1V3)
    document from a spec. `ode`→CVODE / `ssa`→Gillespie KiSAO algorithms;
    rel/abs tolerance and max-steps as algorithm parameters. bngsim output
    selectors (`observable:`/`expression:`/`species:`) are carried verbatim in
    each data generator's `name` so a bngsim→SED-ML→bngsim round-trip is exact,
    with a standards-compliant `variable` (time `symbol` / species `target`)
    also emitted for interop. `numberOfPoints` = `n_points - 1` (SED-ML counts
    steps after the start), preserved across the round-trip.
  - `bngsim.convert.default_protocol(model, …)` — a sensible default spec (every
    observable as an output, species fallback) for the *no-sidecar* case.
  - Hand-rolled over stdlib `xml.etree` — **no `python-libsedml` runtime
    dependency** added.
- **`bngsim-net2sbml --sidecar`.** Also emits a SED-ML protocol sidecar
  (`<stem>.sedml`, a uniform time course reporting every observable) next to the
  converted SBML, since SBML carries no protocol of its own.

## [0.10.5] - 2026-06-27

### Added

- **Conversion-validation framework: L0–L4 (GH #217, converter epic #211).** A
  single entry point, `bngsim.convert.validate_conversion(source, …)`, grades a
  format conversion (SBML⇄`.net`) at five escalating levels and returns a
  structured `ConversionValidationReport` artifact (`.ok`, `.summary()`,
  `.to_dict()`). The acceptance bar is **L0–L3 as hard gates plus best-effort,
  non-gating L4**, run for **both** directions (the source suffix selects
  `sbml2net` vs `net2sbml`):
  - **L0 — syntactic validity.** The converted output passes the *target*
    format's own validator: libsbml consistency checks (gating on error/fatal
    diagnostics) for SBML; the `.net` reader accepting a non-empty network.
  - **L1 — structural equivalence.** Species/reaction counts and the dynamic
    reaction reactant→product topology match. Parameter/observable counts are
    *recorded but not gated* — the two loaders label these differently by
    convention, so a strict count match across formats would flag a benign
    difference, not a defect.
  - **L2 — round-trip identity.** `X → Y → X` reproduces the source model graph
    (counts, dynamic topology, and the ODE right-hand side) under a loader-level
    canonical normalization that is robust to benign reordering/relabelling. The
    RHS check is the substantive identity gate.
  - **L3 — numerical equivalence.** Source and conversion are simulated on a
    shared time grid and compared species-trajectory-by-species-trajectory under
    a **scale-aware** per-cell tolerance — a self-contained port of the #214
    parity verdict (the shipped converter carries no parity-suite dependency).
    This is what catches a *lossy* conversion that L0/L1 pass: e.g. an
    `--allow-lossy` SBML→`.net` of a non-unit-volume model is syntactically and
    structurally valid yet numerically wrong, and L2/L3 flag it.
  - **L4 — symbolic/algebraic equivalence** (best-effort, never blocks). The
    per-dynamic-species ODE RHS is reconstructed symbolically from each
    representation (parameters and fixed/boundary species folded to constants,
    observables inlined to species sums, functions translated through the
    Jacobian's ExprTk→sympy bridge) and compared with `sympy.simplify`. Reports
    **equal / not-equal / inconclusive**; it punts to *inconclusive* on
    Michaelis–Menten/volume-scaled/table/piecewise/transcendental kinetics it
    cannot reconstruct faithfully.
- **`bngsim-validate-conversion` CLI.** Wraps `validate_conversion` with
  `--direction`, `--levels`, `--allow-lossy`, `--t-end`/`--n-points` (L3 grid),
  `-o/--out-dir`, and `--json`. Exits non-zero when any hard gate fails, so
  scripts/CI can gate on a conversion.

## [0.10.4] - 2026-06-27

### Changed

- **net2sbml: constant assignment-rule compartment volumes now convert (GH #216
  follow-up).** The 0.10.3 boundary refused *every* assignment-rule compartment
  (`_varvol_ar_conc_map`) as potentially time-varying. Most are in fact constant
  — the PBPK pattern `organ_volume = bodyweight · fraction` built from constant
  parameters — so their report-time rescale is a no-op and they round-trip as
  plain static compartments. `sbml_capability_report` now classifies each AR
  compartment's volume expression: it is refused only when the expression
  *transitively* references a species, observable, or the time csymbol.
  Recovers BIOMD1027/1028/1029/1039 (verified RHS- and initial-state-exact);
  BIOMD856 (`tV = mV + dV`, a cell mass that grows over time) stays correctly
  refused. The constancy check resolves functions before their constant shadow
  parameters, so a time-varying `cell = 1 + 0.1·time` is not masked by its shadow.

## [0.10.3] - 2026-06-27

### Added

- **Volume-faithful kinetics for net2sbml (GH #216 follow-up).** `write_sbml`
  now scales each kinetic law by its reaction's compartment volume, so static
  non-unit-volume models — previously refused under `strict` — round-trip
  exactly (ODE-RHS equivalent to the source). The rule keys off the engine's
  storage convention: a `per_species_volume_scaling` reaction already divides
  each species's accumulation by its own volume, so its propensity *is* the SBML
  extent rate `L` (this carries genuinely cross-compartment reactions
  faithfully); a uniform-propensity reaction emits `L = propensity · V_c`, where
  `V_c` is the shared volume of the reaction's dynamic species. Verified
  RHS-exact across 23 non-unit-volume BioModels spanning two-compartment
  kinetics, cross-compartment reactions, amount-valued species, and up to 7
  distinct static volumes.

### Changed

- **net2sbml capability boundary narrowed to *time-varying* volumes.** The
  blanket "non-unit compartment volume" refusal is replaced by a precise one:
  only volumes that *move in time* (rate-rule-driven, event-resized, or
  assignment-rule compartments) are refused under `strict`, because a static
  SBML document cannot carry a moving volume — a species whose reported
  concentration tracks it would round-trip mis-reported. Detected via the
  loader's report-time rescale maps on the `Model`. A uniform-propensity
  reaction whose dynamic species span more than one volume (no benchmark model
  exercises this) is likewise refused, fail-loud.

## [0.10.2] - 2026-06-27

### Added

- **`.net`→SBML network exporter (GH #216), the reverse of sbml2net.** New
  `bngsim.convert.net_to_sbml(net_path, out_path=None, *, validate="L1",
  strict=True)` and a `bngsim-net2sbml` console entry point. Loads the source
  with the existing `.net` reader, then serializes the in-memory network to
  SBML Level 3 Version 2 via the new `bngsim.convert.write_sbml` writer (built
  on libsbml — symmetric with how `from_sbml` reads). Scope is the **network
  channel** only (species, reactions, parameters, observables, functions,
  compartments).
  - **Faithful half.** SBML can carry amount/concentration semantics the plain
    `.net` text cannot, so net2sbml reconstructs them: each species is emitted
    with `initialConcentration` or (for `hasOnlySubstanceUnits` species)
    `initialAmount`, and compartments are reconstructed per distinct volume.
    Unit-volume models — including amount-valued species — round-trip exactly
    (verified by ODE-RHS equivalence). Michaelis–Menten reactions are emitted
    as their explicit tQSSA closed form, and a BNG zero-arg observable call
    (`Atot()`) collapses to the bareword.
  - **Expression translation.** Function/observable bodies (engine ExprTk) are
    translated to MathML: `if(c,a,b)`→`piecewise`, `time()`→the SBML time
    csymbol, `_pi`→`pi`, and ExprTk `log` (natural log)→SBML `ln` (SBML `log`
    is base-10), then parsed with `libsbml.parseL3Formula`.
  - **Capability boundary (fail-loud, never silently wrong).** `<event>`
    elements are dropped with a `ConversionWarning`. Constructs net2sbml v1
    cannot carry faithfully — non-unit compartment volumes (the cross-
    compartment volume-factor inversion is future work), live (time-varying)
    volumes, and `tfun` table-function calls (no SBML/MathML form) — raise
    `ConversionError` under the default `strict=True`, naming the construct;
    `strict=False` (`--allow-lossy`) downgrades to a `ConversionWarning` and
    emits a best-effort document.
  - **Round-trip validation.** `validate="L1"` reloads the emitted SBML and
    runs `validate_roundtrip` — species/reaction counts, reaction topology over
    dynamic (non-fixed) species, and an ODE right-hand-side numerical check.
    Observable/parameter counts are recorded but not gated, since `from_net`
    honors explicit `begin groups` while `from_sbml` auto-reports species. This
    seeds the #217 (#211c) L2 round-trip-identity gate.

### Changed

- `bngsim.convert.ConversionReport` is now direction-neutral: the produced text
  is `output_text` (with `net_text`/`sbml_text` aliases), plus an optional
  `max_rhs_delta`. The SBML→`.net` surface is unchanged.

## [0.10.1] - 2026-06-26

### Added

- **SBML→`.net` network converter, productized (GH #215).** New
  `bngsim.convert.sbml_to_net(sbml_path, out_path=None, *, validate="L1",
  strict=True)` and a `bngsim-sbml2net` console entry point (also runnable as
  `python -m bngsim.convert`). Parses the source with the existing libsbml
  loader — so the full range of SBML semantics is honored (initial
  amount/concentration, `hasOnlySubstanceUnits`, compartment volumes,
  assignment/rate rules, initial assignments, local kinetic-law parameters,
  function definitions, multi-compartment reactions, MathML→ExprTk) — then
  serializes the in-memory network to BioNetGen `.net` text via the new
  `bngsim.convert.write_net` writer. Scope is the **network channel** only
  (species, reactions, parameters, observables, functions).
  - **Capability boundary (fail-loud, never silently wrong).** `<event>`
    elements are dropped with a `ConversionWarning` (they belong to a
    simulation-protocol sidecar, not the network). Constructs the plain `.net`
    text format cannot carry faithfully — amount-valued species in a volume≠1
    compartment, cross-compartment reactions needing per-species volume scaling,
    live (time-varying) compartment volumes, Michaelis–Menten rate-law types —
    raise `ConversionError` under the default `strict=True`, with a clear note
    naming the construct; `strict=False` (`--allow-lossy`) downgrades to a
    warning and emits a best-effort network.
  - **L1 structural validation.** `bngsim.convert.validate_structural_l1`
    compares the source model and its reloaded `.net` by counts and per-reaction
    reactant/product topology, returning a structured report (seeds the #211c
    validation framework). The converter's `ConversionReport` carries the
    counts, dropped/lossy annotations, and the L1 verdict.
- New exceptions `bngsim.ConversionError` and `bngsim.ConversionWarning`.

## [0.10.0] - 2026-06-26

### Changed (breaking)

- **Forward sensitivity now requires code generation (GH #214 follow-up).** The
  interpreted finite-difference sensitivity path was retired. Without a codegen
  sensitivity RHS, CVODES finite-differences the *entire* sensitivity RHS
  (`∂f/∂y·s + ∂f/∂p`); that ~sqrt(eps) noise cannot support tight tolerances, so
  the error test silently micro-steps to a halt (the preequilibration model hung
  at rtol=1e-11 — ~92M steps). bngsim now builds an analytical codegen sensitivity
  RHS for **every** sensitivity run and **raises** rather than degrading:
    * `codegen=False` or `BNGSIM_NO_CODEGEN` together with `sensitivity_params` /
      `sensitivity_ic` → raises (the two are contradictory);
    * no codegen backend (no C compiler *and* no MIR JIT) → raises;
    * a model whose rate laws cannot be differentiated to closed form (a
      non-smooth construct such as `min`/`max`/`abs`/`floor`, `rateOf()` inside a
      rate law, or an unparseable expression) → raises with a cause-specific
      message.
  The old `n_species·(n_params+1)` size gate (GH #198) was removed for sensitivity
  workflows — it rested on the false premise that the interpreted path is
  "numerically identical" (true for the state RHS `f(x)`, false for the
  sensitivity RHS). The analytical RHS builds via cc, or the in-process MIR JIT
  where no compiler exists (`BNGSIM_CODEGEN_JIT=mir`), so requiring it does **not**
  require a system compiler. This also resolves the GH #198-introduced hang in
  small tight-tolerance sensitivity runs (e.g. pre-equilibration).

## [0.9.70] - 2026-06-26

### Fixed

- **Scale-aware forward-sensitivity error control (GH #214).** bngsim integrates
  concentrations (`amount / V_compartment`), so for the sub-picoliter
  compartments of real cell-biology models `1/V` reaches ~1e11–1e14 and inflates
  both the state and its sensitivities by that factor (Smith2013: `|s|` ~ 1e18 vs
  AMICI's amount-based ~1e10). The previous `CVodeSensEEtolerances` gives every
  sensitivity column a single scalar absolute floor (`atol / pbar`); against a
  1e18-magnitude sensitivity that floor sits ~30 orders below the variable, so
  the CVODES error test demanded sub-machine-eps relative accuracy and the step
  collapsed across a large discontinuity — Smith2013's full forward-sensitivity
  run died at the t=2880 insulin restimulation (CVODE `flag=-3`), the run AMICI
  completes in ~4 s. The fix sets a per-`(state × parameter)` absolute floor
  proportional to each sensitivity's own magnitude scale, `abstolS[iS][i] =
  atol · scale[i] / pbar[iS]` with `scale[i] = max(|y_i(0)|, 1)`, via
  `CVodeSensSVtolerances`. This is the non-dimensionalizing move: error control
  becomes relative-per-component regardless of the unit system. For a well-scaled
  model (every state ≤ 1) it reduces **exactly** to the old `atol / pbar` floor,
  so well-scaled models are byte-identical (190 sensitivity tests + AMICI
  species/observable/expression cross-validation unchanged); only large-magnitude
  states get a proportionally relaxed, reachable floor. Smith2013 now integrates
  the coupled state+sensitivity system through all three events to t=3000.

## [0.9.69] - 2026-06-26

### Added

- **Forward sensitivity through fixed-time events (GH #212, Phase 1).** The
  blanket GH #205 refusal of output sensitivities on any model with events is
  lifted for the fixed-time / persistent / no-delay subclass (`g = time − T`,
  the dosing/stimulation pattern, e.g. Smith2013's `geq(time, const)`). At each
  such event the integrator now jumps the forward-sensitivity vectors by
  `s⁺ = J_h·s⁻ + ∂h/∂p` and calls `CVodeSensReInit`, instead of letting the
  columns go silently stale across the discontinuity. The assignment-value
  derivatives `∂h/∂x` and `∂h/∂p` are obtained by central finite-difference of
  the event-assignment expressions at the pre-event state (pruned to the
  variables each expression references, so a constant/parameter reset costs
  O(1)); the jump is applied at the runtime event-fire site, covering both the
  interpreted and the code-generated sensitivity RHS. Validated against bngsim's
  own central-difference across the event (constant reset, additive bolus, and
  parameter-valued reset all agree to ~1e-6). On the real Smith2013 model the
  coupled solve runs through the first event (t=15) with finite `dx/dp`; a
  full-horizon run to t=3000 currently fails at the t=2880 event due to a
  units / error-control artifact (verified: bngsim integrates concentrations and
  AMICI amounts — the dynamics are identical to ≤5e-7, but bngsim's `1/V`-inflated
  ~1e18 sensitivities trip the CVODES error test; `CVodeSetSensErrCon(false)`
  completes the run, and AMICI completes it in amount units in 4.3s), tracked as
  GH #214. The jump math itself is unaffected.

  The remaining event subclasses keep raising, now with a precise per-event
  reason: state-dependent triggers and parameter-valued trigger times (Phase 2,
  `∂t*/∂p ≠ 0`), and delays / non-persistent triggers (Phase 3). Classification
  is delegated to the core (`NetworkModel.event_sensitivity_unsupported_reason`),
  which inspects each trigger's referenced variables (new
  `ExpressionEvaluator::referenced_variable_addresses`) to decide whether the
  trigger is fixed-time and whether its crossing time depends on a requested
  sensitivity parameter. Discontinuity triggers (forcing pulses) are unaffected.

## [0.9.68] - 2026-06-26

### Changed

- **Sensitivity auto-codegen is now size-gated (was unconditional).**
  `Simulator`'s forward-sensitivity auto-codegen decision triggers on the
  coupled-system "effective RHS dimension" `n_species * (n_params + 1)` against
  `BNGSIM_CODEGEN_THRESHOLD` (256), instead of compiling for *every* sensitivity
  run regardless of size. A few-parameter solve on a large network now codegens
  — previously a many-species/few-parameter model could be left on the slow
  interpreted path — while a tiny coupled system stays interpreted, where the
  compile cost cannot amortize. The gate is bypassed when codegen is required
  for correctness, not just speed: IC sensitivities (no model parameter for
  CVODES to perturb) and expression (function) output sensitivities (GH #198,
  produced only by the compiled output-sensitivity ABI). Sensitivity *values*
  are unchanged — the interpreted and codegen RHS agree within solver tolerance;
  this only changes which path runs. The `forward_sens` benchmark's auto mode
  mirrors the library decision (new `S10_BNG_CODEGEN_MIN_EFFDIM` knob), replacing
  its old `n_params >= 30` heuristic that could starve such models.

## [0.9.67] - 2026-06-26

### Added

- **`output_sensitivities` capability flag (GH #207).** `capabilities()['features']`
  now advertises `output_sensitivities` (always `True`, like `codegen`) — the
  handshake a gradient-based fitting frontend gates its path on before consuming
  the `(n_times, n_outputs, n_param)` tensor from `Result.output_sensitivities()`.
  This is the bngsim half of the PyBNF gradient-integration contract (#207, part
  of the #194 epic); the consumer-side `BNGSIM_HAS_OUTPUT_SENS` flag and gradient
  optimizer live in PyBNF. Feature key is stable and never appears in `missing`.

## [0.9.66] - 2026-06-26

### Added

- **HPC-facing scheduler-free evaluation contract (GH #203).** Formalizes
  bngsim's role as a clean, *stateless* single-evaluation kernel + optional local
  batch helper — the frontend (PyBNF) owns the scheduler (multistart / bootstrap /
  profile / Slurm / MPI) and the objective/noise/loss layer; bngsim exposes the
  raw output + sensitivity *primitives*, never a pre-baked loss. Three additions:
  - **`BNGSIM_CODEGEN_CACHE_DIR`** relocates the content-addressed compiled-artifact
    cache (default `~/.cache/bngsim/codegen`). Point it at node-local scratch, or at
    a directory of artifacts pre-warmed on a login node so worker jobs reuse one
    `.so` instead of recompiling. Resolved once at import (`export` it before
    launching `python`); the compile-to-temp-then-atomic-`os.replace` flow keeps the
    cache concurrency-safe at any location.
  - **`bngsim.EvaluationSpec`** — a frozen, JSON-serializable record of one
    evaluation (model source + optional SHA-256 integrity guard, θ vector, time
    grid, sensitivity set, solver options, output selectors) with
    `to_dict`/`from_dict`/`to_json`/`from_json` (byte-stable), `with_params`,
    `build_model`/`build_simulator`, and a deterministic `evaluate()` for
    checkpoint/restart and cluster fan-out.
  - **`Result.summary()`** — a compact, JSON-serializable description (shapes,
    output names, sensitivity-availability flags, solver stats, seed) for cheap
    indexing/logging without re-reading the full HDF5 payload.

### Changed

- **`Simulator.run_batch` now reuses the one compiled artifact per row and honors
  a `sensitivity_params`-configured Simulator (GH #203).** The per-row batch path
  previously built a fresh `SolverOptions` that carried neither the Simulator's
  codegen `.so`/source nor its sensitivity configuration — so a batch over a
  codegen model ran *interpreted* (reusing no artifact) and a Simulator built with
  `sensitivity_params` silently produced **no** sensitivities in a batch, unlike
  single-shot `run()`. `run_batch` now mirrors `run()`'s ODE option-building:
  every row reuses the one shared read-only `.so`, and a sensitivity-configured
  Simulator yields the full per-row output-sensitivity tensor (species,
  observable, and expression blocks) with deterministic, input-order rows. As with
  every other sensitivity entry point, a sensitivity request on a model with
  `n_events > 0` hard-raises (GH #205), now checked once up front for the batch.

## [0.9.65] - 2026-06-25

### Fixed

- **`compute_all_sensitivities` now emits expression output sensitivities even
  when a plain-RHS codegen artifact was already attached at construction (GH
  #205 follow-up).** The 0.9.64 fix marked `_want_output_sens` before generating
  codegen, but `_auto_codegen_for_sensitivity` no-ops when a codegen `.so`/JIT
  source is already present — so an SBML / builder model that auto-codegened at
  construction (species above `BNGSIM_CODEGEN_THRESHOLD`, an explicit
  `codegen=True`, or an inherited `.so`) carried a *plain* RHS evaluator built
  without the GH #198 output-sensitivity ABI, which then shadowed the sensitivity
  codegen and left the expression block (and the nonlinear-AR `species:`
  redirect) empty. `compute_all_sensitivities` now clears that plain artifact and
  regenerates with output sensitivities when the model has functions and the
  attached codegen predates the sensitivity request (the result is a superset;
  the `.so` cache keeps a repeat cheap), restoring the prior artifact if
  regeneration produces nothing. A sim built with `sensitivity_params` already
  carries output-sens codegen and skips the regeneration (no needless
  large-model rebuild).

## [0.9.64] - 2026-06-25

### Changed

- **Output sensitivities now hard-*raise* (not warn) for models with events
  (GH #205).** Events `CVodeReInit` the integrator state discontinuously, but
  the CVODES forward-sensitivity vectors are never reinitialised (there is no
  `CVodeSensReInit` in the core), so the sensitivity columns go silently stale
  at and after an event fires. bngsim now refuses outright — raising a clear
  `ValueError` — whenever output sensitivities are requested
  (`sensitivity_params` / `sensitivity_ic`, including the `carry_sensitivities`
  pre-equilibration path) on a model with `n_events > 0`, on every sensitivity
  entry point (`Simulator.run`, `steady_state`, `compute_all_sensitivities`).
  This **upgrades the narrow carry-over warning shipped in 0.9.61 (GH #210) to a
  unified raise** — a deliberate behavioral change. The trigger is `n_events > 0`
  *only*: discontinuity triggers (forcing pulses / piecewise-time dosing
  schedules) break the integrator step but do not jump state, so sensitivities
  through them stay valid and are unaffected.

### Added

- **`species:<name>` output sensitivities for SBML AssignmentRule-target species
  follow the assignment expression (GH #205).** An AR-target species is emitted
  `fixed` (its ODE derivative zeroed) and the value path overwrites its column
  from the rule's live value — an *observable* for a linear-on-species rule
  (GH #197) or a *function/expression* otherwise (GH #198). The raw integrated
  forward-sensitivity `yS` for that frozen slot is therefore meaningless (~0), so
  `Result.output_sensitivities("species:<ar>")` now redirects through the same
  `_ar_report_map` the value path uses and returns the sensitivity of the rule's
  observable/expression instead. The raw integrated-state tensor stays available
  as the low-level `Result.sensitivities_species`. AR species whose reported
  value also carries a time-varying volume rescale (variable-volume compartment,
  GH #85/#87) are refused with a clear error rather than returned subtly wrong.

### Fixed

- **`compute_all_sensitivities` now emits expression output sensitivities for
  SBML / builder models (GH #205).** The model-based codegen path
  (`from_sbml` / `from_builder`) only appended the GH #198 output-sensitivity
  evaluator when `_want_output_sens` was set, which the constructor does for
  `sensitivity_params` runs but the `compute_all_sensitivities` entry point (built
  without them) did not — so expression (and nonlinear-AR `species:`) output
  sensitivities came back empty there for SBML models, even though the `.net`
  path already worked (it emits the evaluator unconditionally).
  `compute_all_sensitivities` now marks the flag before generating codegen.

## [0.9.63] - 2026-06-25

### Fixed

- **`compute_all_sensitivities` now stitches observable *and* expression
  output-sensitivity blocks identically to a single-shot run (GH #204).** The
  parallel chunk path reuses the simulator's codegen `.so`/JIT source but never
  triggered codegen itself, so a simulator built without `sensitivity_params`
  (the normal `compute_all_sensitivities` entry point) ran its chunks
  interpreted and every chunk's **expression** output-sensitivity block (GH
  #198, which requires the compiled output-sensitivity evaluator) came back
  empty — the stitch then silently returned `(0, 0, 0)`. The constructor's
  sensitivity auto-codegen logic is refactored into a reusable
  `_auto_codegen_for_sensitivity` helper that `compute_all_sensitivities` now
  invokes when the model carries global-function (expression) outputs, so the
  chunked expression and observable tensors match the single-shot
  `Simulator(sensitivity_params=...).run()` tensors exactly. Species- and
  observable-only models (no expressions) stay on the interpreted path
  unchanged — observable sensitivities (GH #197) need no codegen.
- **The output-block stitch is loud on inconsistency.** Stitching previously
  collapsed an output-sensitivity block to `(0, 0, 0)` whenever *any* chunk's
  block was empty, conflating "no chunk computed it" (legitimately empty) with
  "some chunks have it, some don't" (a real bug). It now raises a
  `SimulationError` naming the offending block when chunks disagree, mirroring
  the species-path error, while still treating an all-empty block as
  legitimately empty.

## [0.9.62] - 2026-06-25

### Added

- **Tensor-only Fisher Information / model identifiability over named-output
  selectors (GH #202).** `Result.fisher_information` is generalized from a
  species-only FIM to one built over **named outputs**: pass
  `outputs=[...]` with any mix of `species:`/`observable:`/`expression:`
  selectors and the FIM `Σₜ (∂Y/∂θ)ᵀ Σ⁻¹ (∂Y/∂θ)` is contracted over those
  output columns of the output-sensitivity tensor (GH #197/#198) instead of
  raw species. The optional σ scaling is kept (scalar, or per-output 1-D
  array); there is **no measurement data, no residuals, no objective** — this
  is sensitivity analysis of the *model* (sloppiness / practical
  identifiability), independent of any fit. `outputs=None` (the default) keeps
  the original species-only behaviour bit-for-bit; `axis="ic"` builds the FIM
  over differentiated initial conditions. A new **`Result.identifiability`**
  returns an **`IdentifiabilityReport`** (exported at the package top level):
  the FIM's eigenvalues/eigenvectors, numerical rank, condition number,
  per-direction identifiability flags (small-eigenvalue / "sloppy" directions),
  and the Cramér–Rao bound `FIM⁻¹` — clearly labelled as a lower bound and an
  identifiability aid only, **not** a data/noise-weighted fit covariance. A
  rank-deficient FIM warns and returns NaN for the inverse rather than emitting
  a garbage one; a configurable `rtol` controls the eigenvalue cutoff that
  flags practically non-identifiable directions. Batch results are refused (the
  FIM is per single simulation). The data/noise-weighted diagnostics that ride
  the *residual* Jacobian (Gauss–Newton Hessian, fit covariance/correlation,
  parameter-std, objective HVP) intentionally remain PyBNF's, not bngsim's.

## [0.9.61] - 2026-06-25

### Added

- **Pre-equilibration / steady-state output sensitivities via two-phase
  carry-over IC seeding (GH #210).** A pre-equilibration protocol (ADR-0052,
  PyBNF #440) equilibrates to steady state under a pre-condition (unmeasured),
  then perturbs and measures — running the *same* persistent `Simulator` across
  two `run()` calls with **no reset between them**, so the equilibration steady
  state `x_ss(θ)` is the measurement phase's initial condition. The measurement
  phase's forward-sensitivity seed must therefore be the steady-state
  sensitivity `dx_ss/dθ`, not the fresh-start zero. `Simulator.run()` gains an
  opt-in **`carry_sensitivities=True`**: a sensitivity run captures its final
  `dx/dθ` matrix onto the model, and the next carried-over run seeds `yS(0)`
  from it (so `∂x(0)/∂θ = dx_ss/dθ` rather than 0), making
  `output_sensitivities()` correct across the boundary. This is the forward-
  sensitivity carry-over the issue scopes — capture the CVODES sensitivities at
  the phase boundary — and works for the observable/expression output blocks too
  (they chain-rule the seeded species sensitivities). Validated against central
  finite differences taken over the *full* two-phase run (matches to ~1e-9). The
  carried seed is a model-level state alongside the species concentrations:
  `clone()` copies it; `reset()`/`save_concentrations()` clear it; a
  non-sensitivity run or a `set_concentration()`/`set_state()` (a fresh literal
  IC) drops it. New read-only introspection on the core model:
  `ic_state_dirty`, `has_pending_sensitivity_seed`, `pending_sensitivity_seed()`,
  `pending_sensitivity_seed_param_names`.
- **No silent wrong derivatives across a pre-equilibration boundary (GH #210).**
  Requesting output sensitivities on a carried-over species state *without*
  `carry_sensitivities=True` now **raises** (fresh seeding would silently assume
  `∂x(0)/∂θ = 0`), as does `carry_sensitivities=True` with no matching seed from
  a prior phase (e.g. the equilibration phase was not run with the same
  `sensitivity_params`, or a `reset()` — an SBML/RoadRunner-style every-action
  reset — wiped the carry-over). Initial-condition (`sensitivity_ic`) axis
  sensitivities across a carry-over boundary are refused (the carried state is no
  longer the model's IC), and combining `carry_sensitivities` with events warns
  (event-time sensitivity discontinuities are tracked by GH #205). A fresh single
  sensitivity run is unaffected. Scope matches ADR-0052: steady-state (`-inf`)
  equilibration and absolute (`=`) pre-condition perturbations; finite-time
  pre-equilibration is deferred.
- **Expression / global-function output sensitivities via a codegen evaluator
  (GH #198).** Global functions are nonlinear in their inputs, so `d func/dθ`
  needs the full chain rule over the *same* expression graph the values use —
  emitted as compiled C (`bngsim_codegen_output_sens`) into the same `.so` as the
  RHS/sens-RHS, so value and derivative never diverge. At each output row of the
  cold (CVODES sensitivity) path the evaluator folds the per-column state
  sensitivities into
  `d func/dθ = Σ_i ∂f/∂x_i·dx_i/dθ + Σ_j ∂f/∂obs_j·dobs_j/dθ + Σ_k ∂f/∂p_k·dp_k/dθ
  + Σ_m ∂f/∂f_m·df_m/dθ` (the parameter term is the Kronecker-δ plus the
  derived-parameter chain from GH #15; the `time()` term drops). Both axes are
  populated — `Result.sensitivities_expressions` (parameter) and
  `sensitivities_expressions_ic` (initial condition) — and `expression:`
  selectors now resolve through `Result.output_sensitivities(...)`. The
  per-expression partials reuse the analytical-Jacobian sympy machinery
  (`bngsim._jacobian`, no inlining), and the recorded block is filtered to the
  user-facing functions (auto-generated `_rateLawN` columns dropped, mirroring
  the value columns). The evaluator is emitted only for sensitivity runs (its
  build-time differentiation is gated behind `sensitivity_params`/`sensitivity_ic`
  and folded into the codegen cache key), so non-sensitivity builds are
  unaffected. Validated against central finite differences across every
  dependency kind (parameter, observable, species, earlier function, time,
  derived parameter) and the IC axis.
- **Unsupported expression sensitivities fail loudly (GH #198).** Comparisons,
  logical operators, `if(...)`, `abs`, `min`/`max`, `floor`/`ceil`,
  `round`/`rint` cannot be differentiated correctly, and table functions are not
  differentiated at all, so requesting the output sensitivity of any of these
  raises a targeted, actionable error naming the cause (a function transitively
  depending on an unsupported one is rejected too). An `expression:` selector on
  an interpreted run (no codegen) raises an actionable "requires codegen" error
  rather than returning silently-empty data.
- **Observable output sensitivities, computed at runtime via the linear chain
  rule (GH #197).** BNGL observables are linear in species
  (`obs_j = Σ_i c_ji·x_i`), so `d obs_j/dθ = Σ_i c_ji·dx_i/dθ` is now computed in
  C++ at each output time from the CVODES species sensitivities already
  extracted by the cold ODE path — **no codegen required**. Both axes are
  populated: parameter (`Result.sensitivities_observables`, shared
  `sensitivity_params` axis) and initial-condition
  (`sensitivities_observables_ic`, shared `sensitivity_ic_species` axis). The
  coefficient `c_ji` folds the observable-group factor and, for amount-valued
  species, the same volume scaling `update_observables()` applies. A new
  `Result.output_sensitivities(selectors, *, axis="parameter"|"ic")` exposes the
  result through the typed `observable:`/`species:` selectors from GH #195
  (`expression:` selectors were deferred to the codegen stage, since implemented
  in GH #198 above), and empty-observable models yield empty `(0, 0, 0)` blocks.
  Parameter-chunked `compute_all_sensitivities` stitches the observable block
  along the parameter axis. Validated against central finite differences.
- **Storage + Python API for observable & expression output sensitivities
  (GH #196).** `Result` now carries four optional sensitivity blocks alongside
  the existing species blocks — `d observable/dθ`, `d expression/dθ`, and their
  initial-condition variants — surfaced as `Result.sensitivities_observables`,
  `sensitivities_expressions`, `sensitivities_observables_ic`,
  `sensitivities_expressions_ic` (plus `has_*` predicates), with
  `sensitivities_species` added as an alias for the species-only
  `sensitivities`. The parameter / IC-species axes are shared with the species
  blocks (same `sensitivity_params` / `sensitivity_ic_species`). The blocks are
  empty `(0, 0, 0)` until a later stage computes them: this change is storage
  and plumbing only — HDF5 save/load (format_version bumped to 2, additive and
  backward-compatible), xarray dims/coords (`(time, observable, parameter)` etc.,
  xarray not required), and batch `Result.squeeze` stacking all flow the new
  blocks through. The species-only `sensitivity_data` pybind property and
  `Result.sensitivities` are unchanged.

## [0.9.60] - 2026-06-25

### Fixed

- **SuiteSparse/KLU is now discovered portably on Linux/HPC/conda — no more
  silent dense-only builds (GH #209).** The build's KLU probe was a hardcoded
  `foreach(/opt/homebrew /usr/local /usr)` requiring
  `<prefix>/include/suitesparse/klu.h`, so it found Homebrew on macOS but missed
  conda prefixes (`$CONDA_PREFIX`) and HPC Spack/Lmod module trees entirely. A
  miss force-disabled KLU (`set(ENABLE_KLU OFF … FORCE)`, un-overridable) and only
  emitted a `message(WARNING)` lost in `pip` output — so a from-source Linux/HPC
  install silently shipped **dense-only**, and the CVODE Newton solve factorized
  the full N×N Jacobian at O(N³) for every model regardless of sparsity. On a
  genome-scale network (74,795 species, Jacobian 99.997% zeros, dense storage
  ≈ 45 GB) that was the difference between ~1 minute and ~80 minutes. KLU
  discovery is now a portable `find_path`/`find_library` that honors
  `CMAKE_PREFIX_PATH`, `$CONDA_PREFIX`, `KLU_ROOT`/`SUITESPARSE_ROOT`, and an
  explicit `-DKLU_INCLUDE_DIR`/`-DKLU_LIBRARY_DIR` (none force-overridden), while
  still finding the historical system prefixes — so macOS (brew), vanilla Linux
  (`/usr`), conda, and HPC modules all resolve with no CMake edits. The
  `BNGSIM_USE_SYSTEM_SUNDIALS` branch now *verifies* the system SUNDIALS actually
  provides the KLU solver (via target existence) instead of assuming it.

### Added

- **`-DBNGSIM_REQUIRE_KLU=ON` (default OFF) turns a missed KLU discovery into a
  `FATAL_ERROR`** with an actionable fix-it message (install recipe per platform
  + the prefix/`*_ROOT` hints), so an HPC deploy or CI build can never silently
  produce a dense-only artifact (GH #209). The Linux CI build sets it and asserts
  `has_klu` is True.
- **`bngsim.HAS_KLU` flag and `capabilities()["features"]["klu"]`** expose whether
  the sparse KLU solver was compiled in, so downstream tools (PyBNF,
  PyBioNetGen) can detect a dense-only install programmatically rather than
  discovering it on a multi-hour run; `capabilities()["missing"]["klu"]` carries
  the rebuild recipe when absent (GH #209).
- **One-time `bngsim.DenseSolverFallbackWarning` at `run()`** when a large ODE
  model (`n_species ≥ 2000`) is about to run on the dense solver *only because*
  this install lacks KLU — not when the user asked for
  `force_dense_linear_solver` or `jacobian="jax"`. It names the cause and the
  fix; a KLU-enabled install never emits it (GH #209).

## [0.9.59] - 2026-06-24

### Fixed

- **Build-time analytical-Jacobian derivation budget now scales with model size
  (GH #187).** The #95 budget that cuts off pathological symbolic derivations was
  a fixed 20 s wall-clock cap. On a genome-scale model that cap silently expired
  mid-derivation and dropped to a finite-difference (FD) Jacobian — which at tens
  of thousands of species needs ~`n_species` RHS evaluations per Newton step and
  is effectively non-terminating, not merely wasteful (measured on GS-SPARCED,
  74,795 species: derivation ~33 s, cut off at the 20 s default on a slightly
  slower/busier node). The budget is now keyed on species count: it stays at the
  20 s base below ~4,000 species (so the #95 small-model losers, ≤ 295 species,
  are unaffected and still fall back to FD), scales up linearly past that, and
  becomes **unbounded at ≥ 20,000 species** where FD is not a viable solver path —
  there the analytical Jacobian is mandatory and always derived to completion.
  This removes the machine-dependent silent cliff. When the budget *does* expire
  on a large model (e.g. an explicit finite `BNGSIM_JAC_DERIV_BUDGET_S` on a
  genome-scale model), the fallback is now logged at **WARNING** with the exact
  workaround instead of degrading silently at INFO. An explicit
  `BNGSIM_JAC_DERIV_BUDGET_S` (seconds, or `inf`/`none`/`0` for unbounded) still
  overrides the size policy.

## [0.9.58] - 2026-06-24

### Added

- **`result.ssa_diagnostics["propensity_backend"]`** reports how an SSA run
  evaluated propensities: `"cc"` / `"mir"` (the compiled structure-specialized
  vector driving the recompute-all loop) or `"interpreted"` (per-reaction
  `compute_propensity` + Fenwick). Recorded by the engine per run (GH #190).

### Fixed

- **rr_parity SSA matrix now reports the real propensity backend.** The matrix
  hardcoded bngsim's SSA backend as "ExprTk (no codegen)", which has been stale
  since codegen became the default (0.9.55–0.9.57) — fast cc-codegen per-replicate
  timings were mislabeled as interpreted. It now reads the engine's
  `propensity_backend` (e.g. *Native C propensity vector (cc-compiled .so) —
  recompute-all*), adds a **Codegen (cc)** cell to bngsim's per-model load tier
  showing the one-time structure-spec compile (cold), and corrects the stale
  "no codegen on the SSA path" descriptions. The cross-engine RoadRunner load
  timing also populates on arm64 via the instrumented build (0.9.57).
- **rr_parity matrix cost plots: adjustable linear y-axis cap** (50/100/250/500 ms
  or auto-99th-pct; default 100 ms) instead of a fixed 99th-percentile top that a
  few slow models pushed to ~500 ms, making the bulk of fast models unreadable.

## [0.9.57] - 2026-06-24

### Added

- **Michaelis–Menten propensities are now codegen'd for SSA (GH #190).** The
  structure-specialized propensity emitter
  (`emit_ssa_propensity_source_structure`) gained a MichaelisMenten branch
  emitting the tQSSA rate `kcat·stat·sFree·E/(Km+sFree)` (with
  `kcat`/`Km` read from the runtime `params[]` array, mirroring
  `compute_rxn_rate`'s SSA MM path bit-for-bit). Previously only Elementary
  mass-action was codegen'd, so any model with an MM reaction fell back to the
  interpreted propensity loop; now mass-action **+ MM** models compile a
  propensity `.so` (`n_unsupported==0`) and reach the RoadRunner-parity
  recompute-all path. Bit-identical to the interpreted realization on the MM
  test model (maxabs 0; ensemble means match across 20 seeds). Re-screened
  against RoadRunner (318 models): **0 regressions, 0 improvements**.

  Hill/saturation rate laws are *not* covered: BNG rewrites them to Functional
  expressions over observables, which stay on the interpreted path (codegen'ing
  arbitrary Functional propensities — with per-step observable updates — is a
  separate, larger effort).

## [0.9.56] - 2026-06-24

### Changed

- **SSA propensity codegen is now structure-specialized (GH #190).** The compiled
  propensity vector reads each reaction's rate constant from a runtime `params[]`
  argument (`emit_ssa_propensity_source_structure`:
  `bngsim_ssa_propensities(const double* x, const double* p, double* a)`) instead
  of baking it as a literal; only the structural `stat·svf` factor is baked. The
  `.so` cache key therefore depends only on model **structure** — it compiles
  **once per model** and is reused across every parameter point (a fit) and
  replicate (an ensemble). This replaces (and removes) the prior
  value-specialized codegen, which recompiled ~100 ms per distinct parameter set
  and keyed the on-disk cache by value — so a fit recompiled every evaluation and
  spawned thousands of single-use `.so` files. Measured: varying a kinetic
  parameter across 8 points now compiles once (92 ms) then 7 cache hits (0.16 ms)
  while outputs correctly track the live parameters. End-to-end ensembles still
  reach/beat RoadRunner (BIOMD0000000030 0.8×, BIOMD0000000431 1.1×,
  BIOMD0000000430 1.3×) — ~10% behind value-spec per call, the right trade for
  eliminating per-point recompiles.

  Structure-spec is bit-identical to the interpreted incremental path on the
  high-activity suite models (maxabs 0); the compiler-less MIR fallback is
  bit-identical to the explicit MIR recompute-all. The cross-engine
  `ssa_baseline.json` was re-screened against RoadRunner (318 models): **0
  regressions, 0 improvements**. Opt out with `codegen=False` /
  `BNGSIM_SSA_NO_CODEGEN=1`; `BNGSIM_SSA_PROP_CC` / `BNGSIM_SSA_PROP_JIT` /
  `BNGSIM_SSA_RECOMPUTE_ALL` remain as in-process / ablation overrides.

## [0.9.55] - 2026-06-24

### Performance

- **Exact SSA reaches RoadRunner parity on cheap-step models — by default, no MIR
  (GH #190).** For an eligible model (pure mass-action exact SSA, no events,
  reaction count ≤ 64) the Python layer now compiles the value-specialized
  propensity vector (`emit_ssa_propensity_source`, rate constants baked as
  literals) to a content-cached `.so` through the same `cc -O3` codegen path the
  ODE RHS uses (`_codegen.prepare_ssa_propensity_lib`), and hands it to the C++
  `SsaSimulator`. The simulator then takes a RoadRunner-style **recompute-all +
  flat-scan** loop: one native call refills the whole propensity vector each step,
  a single contiguous pass sums the total and selects the reaction, and the
  dependency-graph affected-set lookup, the per-affected Fenwick update, and the
  per-step `set_current_time` are all skipped. Measured vs RoadRunner's Gillespie
  (median of 7): BIOMD0000000431 1.5× → **1.0× (parity)**, BIOMD0000000030 1.2× →
  **0.7× (faster than RoadRunner)**, BIOMD0000000430 1.8× → 1.2×; neutral by
  nr≈60, where the O(nr) full recompute meets the incremental cost (hence the size
  gate). The `.so` is cached on disk, so an SSA *ensemble* compiles once and
  reuses it across replicates. The experiment that chose `cc` over the MIR JIT
  (end-to-end identical — the propensity fill is memory-bound in the SSA loop) is
  in `dev/notes/gh190_cc_vs_mir_kernel.py`; this needs no MIR build and is reached
  by stock wheels.

  This **changes the SSA realization** for eligible models (the JIT'd propensity
  vector differs from the per-reaction `compute_propensity` by a few ULP, and the
  flat index-order sum/scan replaces the Fenwick tree). The cross-engine
  `ssa_baseline.json` was re-screened against RoadRunner (318 models): **0
  regressions**, PASS 201 → 205, `not_expected=0`. Opt out with `codegen=False` or
  `BNGSIM_SSA_NO_CODEGEN=1` (interpreted Fenwick path, the prior realization).
  `BNGSIM_SSA_PROP_CC` / `BNGSIM_SSA_PROP_JIT` / `BNGSIM_SSA_RECOMPUTE_ALL` remain
  available as in-process / ablation overrides.

## [0.9.54] - 2026-06-23

### Performance

- **SSA affected-reaction sets are precomputed once, ~24–29% faster exact SSA
  (GH #190).** After a reaction fires, the set of reactions whose propensity must
  be refreshed is a pure function of topology — but it was re-derived every step
  with a `std::sort` + `std::unique` + vector inserts inside the dependency
  graph. Profiling the per-step gap to RoadRunner's Gillespie traced ~30 ns/step
  (the largest single component of the remaining gap) to exactly this. The set is
  now derived once at dependency-graph construction and read as a const reference
  in the hot loop — zero per-step allocation, sort, or dedup. Bit-identical (same
  reactions, same order); cuts ~24–29% of exact-SSA wall time across the
  high-activity suite models (e.g. BIOMD0000000431 202 → 153 ms; BIOMD0000000940
  now matches RoadRunner without the JIT propensity path).
- **Opt-in flat reaction selection (GH #190).** `BNGSIM_SSA_SELECT=flat` swaps the
  Fenwick tree for a flat cumulative-propensity array (O(N) contiguous scan, the
  RoadRunner direct-method structure); `=auto` does so size-adaptively for
  `nr <= 64`. A microbench (`dev/notes/gh190_select_microbench.cpp`) shows it is
  1.4–1.9× faster on the *isolated* selection workload for small reaction counts,
  but once the affected-set precompute above lands, selection is a small fraction
  of per-step cost and the win washes out end-to-end — so it stays opt-in (default
  remains the Fenwick tree). Validated bit-identical and green across all SSA tests.

## [0.9.53] - 2026-06-22

### Changed

- **Re-vendored RuleMonkey 3.4.0 (`775a933`) → 3.5.0 (`fbdde54`), adding a
  stateful `simulate(const TimeSpec&)` session overload; the GH #184 RuleMonkey
  `sample_times` path now routes through it instead of a bngsim-side
  workaround.** 0.9.52 honored explicit session `sample_times` for RuleMonkey by
  stitching N back-to-back uniform `simulate(.., 2)` segments in the wrapper —
  correct and bit-identical, but a workaround for a gap in RuleMonkey's own
  session API (the stateless `run(TimeSpec)` honored `TimeSpec::sample_times`;
  the stateful `simulate` did not). That gap is now filled upstream: the engine
  exposes `simulate(const TimeSpec&)`, the stateful counterpart of
  `run(TimeSpec)`, recording at exactly `sample_times` in a single `run_ssa`
  pass (RuleMonkey #16). The bngsim wrapper is back to a thin `TimeSpec`
  pass-through. Behavior is unchanged — `test_session_sample_times` still shows
  the explicit-time result bit-identical to the uniform grid at shared instants
  under the same seed, and event_count preserved — but the implementation is
  single-pass and lives where it belongs. RuleMonkey is compiled into
  `_bngsim_core`, so the re-vendor required a rebuild; ExprTk/bngsim_expr
  vendoring is byte-identical (drift guard clean, pin `1a1d49da`). NFsim's
  `sample_times` path is unchanged (bngsim-owned `stepTo` loop).

## [0.9.52] - 2026-06-22

### Added

- **`sample_times` on the stateful network-free session API —
  `NfsimSession.simulate` and `RuleMonkeySession.simulate` (GH #184).**
  Both sessions now accept `simulate(..., sample_times=[t0, t1, …])` and return a
  `Result` whose time axis equals the requested instants exactly (sorted
  ascending, treated as absolute) instead of a uniform grid — continuing from the
  live session state, so it works mid-protocol (after `set_param` / `add_species`
  / a prior segment), not just from a fresh session. This gives the *stateful*
  session the same explicit-time capability the stateless `run(TimeSpec)` /
  `Simulator.run(sample_times=...)` path already had, without re-seeding or
  resetting state. NFsim drives its existing `stepTo` loop at the requested
  absolute times; RuleMonkey threads a `TimeSpec` into the session engine's
  `run_ssa`, which already records at arbitrary sorted sample times in a single
  SSA pass. The contract is single-sourced (`bngsim._sample_times`) so it is
  byte-identical across both backends — `nf` and `rm` stay interchangeable.
  Sampling does not perturb the SSA stream: observable values at instants shared
  with the uniform grid are bit-identical under the same seed. Validation requires
  ≥2 finite points; on `NfsimSession`, `sample_times` and `relative_time` are
  mutually exclusive (sample_times are absolute). The uniform-grid path is
  unchanged (byte-identical). Unblocks PyBNF's new-era `experiment:` / `data:`
  surface for `method: nf` / RuleMonkey under `bngl_backend = bngsim`
  (lanl/PyBNF#427): a network-free fit now outputs at the data's time points.

## [0.9.51] - 2026-06-21

### Changed

- **Re-vendored RuleMonkey 3.3.0 (`dd3539e3`) → 3.4.0 (`775a933`), adding
  `FunctionProduct` (NFsim DOR2) rate-law support (GH #178, RuleMonkey#19).**
  RuleMonkey now runs network-free rules whose rate is
  `RateLaw type="FunctionProduct"` — the per-instance product of two
  per-reactant local-function factors, each evaluated in the context of a
  different tagged reactant (what BNG2.pl emits for
  `%x:A(..) + %y:B(..) -> ... FunctionProduct("f1(x)", "f2(y)")`). RuleMonkey
  previously refused these at Tier-0, which broke `method=>"nf"` ↔
  `method=>"rm"` interchange for the network-free corpus models that use the
  idiom (e.g. `BLBR_immobilization_simple`, `BSA_v9`/`v10`,
  `immob_equiv_lig_sites`). The propensity is realized as `S1·S2`, matching
  NFsim's `DOR2RxnClass`; validated upstream against NFsim 2.9.3 and re-checked
  here on the issue reproducer (rm-vs-NFsim ensemble: maxAbsDiff 0.4, RMSE 0.1,
  within SSA noise). RuleMonkey is compiled into `_bngsim_core`, so the feature
  required a re-vendor + rebuild; the standalone `bngsim_expr`/ExprTk vendoring
  remains byte-identical to this tree (drift guard clean, pin `1a1d49da`).
  The GH #175 golden is unchanged — `method=>"nf"` models route to NFsim, not
  RuleMonkey — so no golden regen. Payoff: full nf↔rm swappability, and
  `parity_diff.revalidate_against_rulemonkey` (the RM cross-check oracle) can
  now adjudicate FunctionProduct `nf` models that previously hard-errored on rm.

## [0.9.50] - 2026-06-21

### Fixed

- **`jacobian="auto"` (the default) now falls back to the finite-difference
  Jacobian when the analytical Jacobian de-stabilizes CVODE on a rate law that
  is discontinuous in a state variable (GH #176).**
  `l-type-calcium-channel-dynamics` failed CVODE under the default analytical
  Jacobian (`flag=-3` at t≈25) while the FD Jacobian and legacy `run_network`
  integrated it cleanly. The root cause is *not* a wrong or non-finite Jacobian
  term — the analytical Jacobian is mathematically correct. The model's
  functional rate law `v_rec = if((-70+V)<-20, 0.5, 0.05)` is a genuine value
  discontinuity (step 0.5→0.05) in the state `Voltage_Level`, which
  asymptotically approaches the threshold 50 exactly at t≈25 (`50-V ≈ 1.2e-9` at
  the failure point). The exact derivative of a step is 0, so the analytical
  Jacobian cannot warn CVODE's implicit corrector about the impending jump: the
  BDF predictor overshoots the discontinuity, the corrector meets an
  unanticipated jump, the local error test fails repeatedly and the step
  collapses to `hmin`. The finite-difference Jacobian instead straddles the step
  and supplies a regularizing slope — which is exactly why FD/`run_network`
  succeed. No smooth Jacobian can represent a state discontinuity, and a
  derivation-time gate that declined every state-dependent `if()` would also
  reject the GH #168 per-capita guard idiom `if(X>1e-300, expr/X, 0)` (whose
  rate-law jump is absorbed by the mass-action factor `X→0`, so its analytical
  Jacobian is correct and wanted). So the fix lives at the `Simulator`:
  `jacobian="auto"` honours the meaning of "auto" — it tries the analytical
  Jacobian and, on a solver failure, transparently retries once with the FD
  Jacobian (identical to an explicit FD run, `diff = 0.0`). An explicit
  `jacobian="analytical"` is *not* second-guessed and surfaces the failure. The
  fallback is memoized per-`Simulator` (no wasted attempt on repeated runs) and
  reported by `jacobian_strategy` (`"fd"` after a fallback). Validated against
  the BNG2.pl `run_network` FD oracle: `max_rel_err = 4.9e-13` over the t∈[0,70]
  grid spanning the discontinuity region. Runtime-only Python change; the
  compiled-codegen Jacobian path (derivative baked into the `.so`) is excluded.
  With this fix `l-type-calcium-channel-dynamics` rejoins the `bng_parity`
  golden under the default config (893 → 894/895).

## [0.9.49] - 2026-06-20

### Fixed

- **Analytical Jacobian no longer goes NaN at a transiently-negative
  fractional-power concentration, completing the GH #135 dose-region fix
  (BIOMD0000000994/995/996).** The first GH #135 ship (0.9.41) clamped only the
  ODE *RHS* on a non-finite result, on the premise that the offending overshoot is
  a BDF predictor excursion the RHS sees while the *reused* analytical Jacobian
  sits at the last accepted, nonnegative state. That held for the original
  early-time failure (t≈1.9) but not at the t≈180 ligand-wash/injection dose events
  in the three COPASI TGF-β/Smad models, where CVODE re-evaluates the analytical
  Jacobian *at* the predictor's slightly-negative state: `d/dx (k·x^3.98) =
  k·3.98·x^2.98` is NaN for any negative base — exactly as the RHS `k·x^3.98` is —
  poisoning the Newton iteration matrix and walling the corrector at `|h|=hmin`
  (`CV_CONV_FAILURE`, flag=-4) even though the RHS clamp kept the RHS finite. The
  three models therefore still failed — now at t≈180, not t≈1.9. The analytical and
  codegen dense+sparse Jacobian callbacks now re-fill on the nonnegative-clamped
  state, but — exactly like the RHS clamp — ONLY when the unclamped fill produces a
  non-finite entry, so every finite-Jacobian model (mass-action / polynomial /
  Michaelis–Menten) is byte-identical and the all-Elementary parity solves an
  always-on clamp would have perturbed are untouched. All three models now match
  RoadRunner bit-for-bit (`max_rel_err = 0`) over the full 1400-unit horizon; the
  1323-model ODE parity sweep shows them moving EXCEPTION→PASS with no other
  scoring change.

## [0.9.48] - 2026-06-20

### Added

- **Bare amount-law (`k*A`, hOSU=true, no compartment factor) variable-volume
  reactions now run under SSA (#170).** A mass-action reaction over an hOSU=true
  (amount-valued) species in a rate-rule or event-resized compartment, with no
  compartment appearing as an explicit law factor — e.g. `A -> ; k*A` — was
  refused under SSA with `varvol_non_mass_action`, even though the engine reads
  `A` as its amount (= molecule count), making the propensity `k*n_A` *provably
  volume-independent*: `d(n_A)/dt = -k*n_A` is plain exponential decay with no
  live-volume term. This was a safe over-refusal (it recommended `method="ode"`),
  but an inconsistency with #144: `cell*k*A` (hOSU=true *with* a compartment
  factor, volume-*dependent*) already ran under SSA while `k*A` (volume-
  *independent*) did not. The Elementary varvol gate now admits an all-hOSU=true
  single-compartment monomial with no surviving compartment power (`p == 0`),
  tagging it with live-volume exponent `0` (run uncorrected) — the Elementary-path
  twin of #144 case 1's Functional `cell*k*H`. Validated for unimolecular and
  bimolecular shapes against the closed form `n_A(t) = n_A0*exp(-k*t)` and the
  independent Extrande sampler (Tier 1 event + Tier 2 rate rule). Out-of-scope
  shapes stay refused: a surviving compartment power, a mixed/hOSU=false factor
  (still carries a stale `V_static`), and cross-compartment laws — so the general
  hOSU=true volume handling that #131 finding 4 routes to the Functional path is
  not re-opened.

## [0.9.47] - 2026-06-20

### Fixed

- **Explicit-compartment-factor cross-compartment variable-volume ODE was silently
  wrong (#172).** A cross-compartment mass-action monomial carrying an explicit
  variable-volume compartment factor (`cell*k*A*B`, where `cell` is a rate-rule or
  event-resized compartment and the reactants span two or more compartments) ran
  under `method="ode"` but returned a wrong trajectory — the per-species storage
  divide used the load-time `V_static` instead of the live `V_live(t)`, drifting
  ~60% from RoadRunner and COPASI (which agree), with only a debug-level warning.
  This was the deliberately-deferred "widen later" shape from #144 case 4 (the
  cross-compartment analogue of #130's single-compartment p≠1 fix). It now routes to
  the same per-species ÷`V_live` path as the bare law and matches RR+COPASI to
  ~1e-8.

### Added

- **Explicit-compartment-factor cross-compartment variable-volume reactions now run
  under SSA (#172).** The same shape was correctly refused under SSA with
  `varvol_non_mass_action`; the gate is now lifted. For an all-hOSU=false monomial
  the ODE and SSA corrections are *independent of the explicit compartment-factor
  power*: the factor lives in `base_func`, and both the per-species ÷`V_live` divide
  and the SSA propensity correction `∏_c (V_c,static / V_c,live)^{m_c}` (where `m_c`
  is the count of hOSU=false reactant species-factors in variable-volume compartment
  `c`) fold it automatically, because the live `V_c` cancels between the true rate
  and the base propensity. No engine change was needed — the #144 case-4
  `Reaction.ssa_live_volume_terms` / `Species.ode_live_volume_idx0` machinery already
  keys on species factors only; the fix is a one-line relaxation of the loader's
  classifier gate (drop the `not compartment_factors` requirement). Validated against
  the independent Extrande sampler (z<5, Tier 2 rate rule) and cross-checked vs
  RoadRunner + COPASI (10/10 in
  `dev/investigations/xcheck_144_xcompartment_varvol.py`). hOSU=true cross-compartment
  mixes and reversible laws stay refused.

## [0.9.46] - 2026-06-19

### Added

- **Cross-compartment variable-volume reactions now run under SSA (#144, case 4 —
  the last of the four #144 gates).** A bare-law mass-action monomial whose
  reactants span two or more compartments, at least one of which changes size at
  runtime (`A + B => P` with `A` in a rate-rule/event-resized compartment and `B`
  in another), was refused under SSA with `varvol_non_mass_action` because the
  single scalar live-volume correction (cases 1–3) cannot hold a *product* over
  compartments. It now simulates. Each variable-volume compartment `c` contributes
  its own propensity factor `(V_c,static / V_c,live)^{m_c}`, where `m_c` is the
  count of hOSU=false reactant law-factors in `c`. This is the only #144 case that
  needed an engine change: the scalar `Reaction.ssa_live_volume_*` field is joined
  by a per-compartment vector `Reaction.ssa_live_volume_terms`, applied as a product
  in `compute_rxn_rate` (empty by default ⇒ `.net`/static-V/single-compartment
  reactions byte-identical). Validated against the independent Extrande sampler
  (z<5, Tier 1 event + Tier 2 rate rule, one *and* both compartments variable).
  Irreversible only; explicit-compartment-factor laws (`cell*k*A*B`), hOSU=true
  cross-compartment mixes, and reversible laws stay refused.

### Fixed

- **Cross-compartment variable-volume ODE (the loader's long-standing "may be off by
  the volume ratio" warning).** A reaction spanning a variable-volume compartment
  divided each species's storage derivative by its *static* `volume_factor`, but an
  hOSU=false species in a variable-volume compartment stores live concentration
  (`amount/V_live`), so the post-resize derivative was wrong by `V_live/V_static`
  (~30% in the cross-check models). The per-species accumulation in `compute_derivs`
  now divides by the *live* compartment volume for such species
  (`Species.ode_live_volume_idx0`), reproducing the exact SBML semantics. Confirmed
  to ~1e-8 against RoadRunner + COPASI for the bare-law shapes
  (`dev/investigations/xcheck_144_xcompartment_varvol.py`; the explicit-compartment-
  factor shape remains the documented "widen later" case). The analytical Jacobian
  self-check detects the live-volume term and falls back to finite differences for
  these (small, niche) models; the codegen RHS declines them cleanly. Full
  analytical-Jacobian support is tracked separately.

## [0.9.45] - 2026-06-19

### Added

- **hOSU=true (amount-valued) and p≠1 variable-volume reactions now run under SSA
  (#144, cases 1 & 2).** A single-compartment, irreversible mass-action *monomial*
  in a variable-volume compartment that the loader routes to the Functional path —
  either because it carries an hOSU=true (amount-valued) species as a law factor
  (case 1, e.g. `cell*k*H`; routed there by #131 finding 4) or because the
  compartment power doesn't cancel (case 2, e.g. the bare `k*A*B`, p=0; routed
  there by #130) — was refused under SSA with `varvol_non_mass_action`. It now
  simulates. The Functional emission already carries the live volume (a live-symbol
  divide for laws with an hOSU=false factor, a numeric-`V_static` divide for an
  all-hOSU=true law), so the exact SSA propensity needs only a scalar live-volume
  correction `(V_static/V_live)^(n_f − 1)` for the live-symbol divide (`n_f` =
  count of hOSU=false law factors) or `0` for the numeric divide — independent of
  the compartment power. No engine change: the existing `ssa_live_volume_*`
  correction (which the runtime already applies to Functional reactions) is now
  tagged onto these reactions by the loader. Validated against the exact closed
  form (case 1 is a linear death process, mean `H0·exp(−k·∫V dt)`) and the
  independent Extrande sampler, Tier 1 (event resize) and Tier 2 (rate rule)
  (`python/tests/test_ssa_variable_volume.py`). Still refused: cross-compartment
  reactions (#144 case 4), reversible non-mass-action laws
  (`reversible_non_mass_action` — the forward-minus-reverse SSA hazard), and
  genuine non-mass-action kinetics (MM/Hill). With #144 case 3 (0.9.44), the
  variable-volume SSA subset now spans synthesis, hOSU=true, and p≠1 monomials.

## [0.9.44] - 2026-06-19

### Added

- **Zeroth-order synthesis now runs under variable-volume SSA (#144, case 3).**
  A synthesis reaction `∅ → P` written in the BNG `compartment*k` convention
  (the cancelling `p == 1` form) in a variable-volume compartment — event-resized
  (Tier 1) or rate-rule driven (Tier 2) — previously failed loud with
  `varvol_non_mass_action` and required `method="ode"`. It now simulates under
  SSA. The live-volume propensity correction `(V_static/V_live)^(n_h − p)`
  already in the engine extends to the `n_h = 0` (no reactant) case with exponent
  `−1`, so the amount/time propensity is `k·V_live(t)` — synthesis speeds up as
  the compartment grows, exactly as the variable-volume ODE (already cross-checked
  against RoadRunner + COPASI in #131) does. No engine change was needed; the gate
  in the SBML loader was the only blocker. Validated against both the exact
  time-inhomogeneous Poisson mean `k·∫V dt` and the independent Extrande sampler,
  Tier 1 and Tier 2 (`python/tests/test_ssa_variable_volume.py`). This is the
  first of the four gated cases in #144; hOSU=true reactants (case 1), `p ≠ 1`
  laws (case 2), and cross-compartment reactions (case 4) remain refused, as does
  a bare `k` (`p = 0`) synthesis.

## [0.9.43] - 2026-06-19

### Changed

- **RuleMonkey (`nf_exact`) now honors `sample_times` in-engine (#169, upstream
  RuleMonkey #16).** The interim workaround shipped in 0.9.42 drove RuleMonkey's
  session API (`step_to` + `get_observable_values`/`get_function_values`) once per
  output segment to keep the vendored engine pristine. Because each segment
  re-entered the SSA loop and rebased the running propensity sum, the recorded
  trajectory was a reproducible, statistically unbiased, but *different*
  floating-point realization than the uniform grid, and it could not report
  `solver_stats.n_steps` (the session API exposes no event counter). Upstream
  RuleMonkey 3.3.0 adds an explicit `TimeSpec::sample_times` field honored
  directly inside `run_ssa`, so the bngsim wrapper now routes the explicit-times
  path through the same single, non-invasive `run()` as the uniform grid and
  retires the ~90-line `run_with_sample_times` session-stepping helper. Net
  effect for `nf_exact` + `sample_times`:
  - Output is now **bit-identical** to the uniform-grid run at any instants the
    two schedules share for a fixed seed — matching the NFsim (`nf_reject`)
    backend, which already had this property.
  - `solver_stats.n_steps` (the SSA `event_count`) is now reported, where the
    workaround left it at `0`.
  - The recorded *times* are unchanged (exactly the requested instants); only the
    exact stochastic realization at a fixed seed shifts to the now-canonical
    bit-identical one. `ode`/`ssa`/`psa`/`nf_reject` are unaffected.

### Dependencies

- **Vendored RuleMonkey 3.3.0** (`dd3539e`, via `scripts/vendor_rulemonkey.py`):
  upstream #16 adds `TimeSpec::sample_times` (honored by `Engine::run_ssa`), and
  #18 refreshes RuleMonkey's standalone `bngsim_expr` evaluator pin to the
  current bngsim tree (adding `expr_compat.hpp`). The `third_party/` tree remains
  excluded from the vendor export — inside a bngsim build RuleMonkey links the
  host `bngsim::expression` target.

## [0.9.42] - 2026-06-19

### Fixed

- **Network-free backends (NFsim / RuleMonkey) now honor `sample_times` (#169).**
  `Simulator.run(sample_times=[...])` was silently dropped on the network-free
  path: both `nf_reject` (NFsim) and `nf_exact` (RuleMonkey) emitted a uniform
  `t_start..t_end` grid sized by `len(sample_times)` instead of recording at the
  requested instants — requesting `[0, 2, 5, 9]` returned output at `[0, 3, 6, 9]`.
  This blocked PyBNF's new-era config (ADR-0028), which fits each backend at the
  experimental data's exact independent-variable points (BNGL `sample_times` for
  BNG2.pl/bngsim, `simulate(times=…)` for RoadRunner). Both backends now record
  observable/global-function output at exactly the (sorted) requested times,
  matching BNG2.pl's `simulate_nf` sample_times branch. `ode`/`ssa`/`psa` already
  honored `sample_times` and are unchanged; `pla` remains out of scope.
  - NFsim records explicit times by stepping its single live `System` inside one
    `run()` call, so sampling is non-invasive: the trajectory at a fixed seed is
    independent of which instants are requested (bit-identical to the uniform
    grid at the matching instants).
  - RuleMonkey drives the upstream session API (`step_to` + `get_observable_values`/
    `get_function_values`) per segment, keeping the vendored `third_party/rulemonkey`
    tree pristine (its vendoring policy forbids a local carry queue). The result is
    reproducible for a fixed seed and statistically unbiased, but — because each
    segment re-enters the SSA loop and rebases the propensity sum — is a different
    floating-point realization than the uniform grid, not bit-identical. The
    explicit-times path does not report `solver_stats.n_steps` (the upstream
    session API exposes no event counter).

## [0.9.41] - 2026-06-19

### Fixed

- **ODE solve no longer fails on a non-integer power of a transiently-negative
  concentration (#135).** BIOMD0000000994/995/996 (and any model with a
  non-integer Hill exponent, e.g. `conc^3.98`) failed `CVODE flag=-4`
  (`CV_CONV_FAILURE`) under the default analytical-Jacobian config. The
  closed-form Jacobian is *correct* (it matches SymPy, a Richardson-gated finite
  difference, and a hand calculation) — but being exact, it lets CVODE step
  confidently enough that the BDF predictor pushes a zero-pinned fast species
  slightly negative, where `pow(conc, 3.98)` is NaN (NaN for *any* negative base,
  so step reduction alone never escapes it); the NaN RHS then poisons the Newton
  solve. The finite-difference Jacobian survived only by luck of less-aggressive
  stepping. The RHS callbacks now retry on a nonnegative-clamped copy of the state
  — but ONLY after the unclamped RHS comes back non-finite. Concentrations are
  physically `>= 0`, so the clamp is the correct boundary value (the way
  RoadRunner keeps such a variable cleanly positive). The conditional gate is
  essential: it keeps a model whose RHS is *finite* at a transiently-negative
  concentration byte-identical — a mass-action law like `-k·conc` self-corrects
  toward 0, and clamping it unconditionally would freeze the species slightly
  negative and make the solve chatter. A recoverable-error backstop covers any
  residual non-finite (e.g. `inf` from `1/conc`). Pure integrator-callback change;
  the generated C source and the `.so` cache key are unaffected
  (`_CODEGEN_VERSION` unchanged).

## [0.9.40] - 2026-06-19

### Fixed

- **Codegen parallel-compile memory cap now applies on macOS (#168 follow-up).**
  `_available_memory_bytes()` read only Linux interfaces (`/proc/meminfo`, cgroup
  files) and returned `None` on macOS, so the sharded compile (`_resolve_codegen_jobs`)
  fell back to the CPU cap with no memory bound. A cold genome-scale codegen
  (tens of thousands of species → hundreds of shard units, one `cc -c` per core)
  under memory pressure could then overcommit and be killed mid-compile. macOS now
  derives available RAM from `vm_stat` (free + inactive + speculative + purgeable
  pages) — conservative by construction, so it under-subscribes RAM rather than
  risking an OOM. Under normal memory the job count is unchanged (still CPU-capped)
  and builds stay byte-identical; the cap only engages under pressure. Pure
  job-scheduling change — the generated C source and the `.so` cache key are
  unaffected (`_CODEGEN_VERSION` unchanged).

## [0.9.39] - 2026-06-19

### Fixed

- **Analytical-Jacobian FD self-check no longer rejects correct Jacobians for
  amount-scaled models (#168).** The self-validation gate
  (`NetworkModel::set_functional_jacobian`) used a finite-difference perturbation
  step floored at `1.0` (`h = 1e-5·max(|y[j]|, 1.0)`). For amount-scaled models —
  species ≈ 1e-12, as produced when an SBML compartmental model (concentration ×
  a tiny compartment volume) is converted to `.net` — that floor makes the step
  ~1e7× the species value, which leaves the linear regime and drives a nonnegative
  species across converter-emitted division guards `if(X > 1e-300, expr/X, 0)`, so
  the central difference reads exactly half the true slope and the **correct**
  analytical Jacobian is discarded → finite-difference fallback → stiff
  integration fails (`CVODE flag -4`). The step is now **scale-relative**
  (`h = 1e-5·|y[j]|`) on both the dense and sparse-sampled paths, and a zero
  species (no valid central difference) is **skipped, never rejected**. For
  `|y[j]| ≥ 1` the step is byte-identical to the old one, so O(1)-scale models and
  genuine-mismatch detection are unchanged. Runtime-only change to the attach
  gate; no codegen source changed (`_CODEGEN_VERSION` unchanged).

## [0.9.38] - 2026-06-19

### Added

- **Compiled CSC sparse analytical Jacobian for large sparse/KLU models (#162).**
  `generate_jacobian_from_model` now emits `bngsim_codegen_jac_sparse` — a C mirror
  of `NetworkModel::fill_sparse_analytical_jacobian` that fills the `nnz`-length CSC
  value array — for KLU-routed models, and the dense `bngsim_codegen_jac` otherwise.
  The sparse KLU CVODE callback uses it so per-step Jacobian setup is compiled (no
  ExprTk evaluation); the solve itself is unchanged. The `.net` codegen path
  (`generate_combined_c`/`prepare_codegen`) now also **appends** the compiled
  Jacobian, so the genome-scale SBML→`sbml2net`→`.net` workflow gets a compiled
  per-step Jacobian instead of the interpreted fallback. Declines cleanly to the
  interpreted Jacobian for any un-emittable derivative or CSC-pattern mismatch.

- **Compiled output evaluator on the `.net` codegen path (#163).** The compiled
  observable/function recorder (`bngsim_codegen_outputs`, #136) now appends onto the
  `.net` RHS, so `Model.from_net(...)` + `Simulator(codegen=True)` fills the per-row
  observable and function buffers with one compiled call instead of re-walking the
  ExprTk trees for every observable/function at every output row. Independent of the
  Jacobian gate — emitted for every `jacobian` strategy (`fd`/`jax` record
  observables too). Declines cleanly (interpreted recorder) for `rateOf` /
  no-observable / embedded-tfun models.

- **Sparse + sampled analytical-Jacobian self-validation, unblocking very large
  models (#151).** `set_functional_jacobian`'s self-check cross-validated the
  assembled analytical Jacobian against finite differences by allocating a dense
  `n×n` matrix and differencing every column — O(n²) memory and O(n) RHS evals.
  That is fine for the modest models it was validated on but infeasible at scale:
  a genome-scale model (tens of thousands of species) would need a multi-gigabyte
  dense Jacobian and ~10⁵ RHS evals, so the analytical Jacobian could never attach
  (OOM). Above a crossover
  (`BNGSIM_JAC_SELFCHECK_DENSE_MAX`, default 4096) the check now validates
  **sparsely**: it assembles into the `nnz`-length CSC value array via the new
  `NetworkModel::fill_sparse_analytical_jacobian` (no dense buffer), checks
  **every** analytical entry for finiteness, and reliability-gates FD on a
  deterministic **sample** of columns (`BNGSIM_JAC_SELFCHECK_SAMPLE`, default 256,
  offset per probe for wider coverage), each sampled column compared across all
  rows so a wrong value or a missing structural entry is still caught. Models at
  or below the crossover keep the exhaustive dense check unchanged. The CVODE
  sparse Jacobian callback (`cvode_analytical_jac`) now delegates its value
  accumulation to `fill_sparse_analytical_jacobian`, so the dense fill, the sparse
  integration callback, and the self-check share one assembly. Net: a genome-scale
  model with tens of thousands of saturable Functional reactions attaches a
  **complete** analytical Jacobian in seconds using well under a gigabyte and
  integrates with it; a forced-sparse self-check on small models is byte-identical
  to the dense path.

- **Numerically stable quotient derivative in the native saturable path (#151).**
  The closed-form quotient rule now emits `da/b − (a/b)·(db/b)` instead of the
  algebraically equivalent `(da·b − a·db)/b²`. The naive numerator forms the
  product `da·b` (≈ result × b²), which overflows to `inf` for a saturable term
  `u/(1 + u)` whose `u` is astronomically large at an extreme state — e.g. a Hill
  term in `concentration / volume` with a very small compartment volume — yielding
  `inf − inf = nan` and failing the self-check even though the true derivative is a
  finite small number. The stable form keeps every intermediate at result scale.

- **Native closed-form analytical Jacobian for the saturable rate-law family —
  no SymPy (#151).** The #76 analytical Jacobian differentiates Functional rate
  laws with SymPy (`differentiate_rate_law` → `sympy_to_exprtk`/`sympy_to_c`),
  which is correct but slow: a model with many saturable Functional reactions can
  blow the per-build derivation budget (#95), and because
  `NetworkModel::analytical_jacobian_complete()` is all-or-nothing, a single
  un-derived reaction discards the whole model's closed-form Jacobian. Saturable
  kinetics are a small fixed algebraic family — Hill terms `S^h/(K^h + S^h)`,
  rational/saturation terms `k/(K+S)` (the legacy `Sat`/`Hill` `.net` tokens,
  #48), basal + regulated production `(k0 + k1·Φ)·P`, and products (AND) /
  shared-denominator sums (OR) of Hill terms over several regulators, all built
  from `+ - * / ^` over species and parameters — so their derivatives have simple
  closed forms. A new pure-Python module `bngsim/_saturable_jacobian.py` tokenizes
  and parses the (function-inlined) ExprTk rate law into a tiny arithmetic AST,
  differentiates it in closed form (sum/product/quotient/power/chain rule), and
  emits the derivative directly as ExprTk **or** C — reusing the same
  power-emission idioms as the SymPy emitters. `bngsim/_jacobian.py` and the
  codegen Jacobian (`generate_jacobian_from_model`) try this native path first and
  fall back to SymPy only for expressions outside the family, so a model whose
  Functional rate laws are entirely saturable obtains a **complete** analytical
  Jacobian with **zero SymPy invocations** (even with SymPy uninstalled) and no
  derivation-budget pressure. The native engine returns `None` (never a wrong
  derivative) for anything outside the family — `if(...)`, comparisons, logical
  operators, un-whitelisted functions, un-inlined or unknown symbols, or
  keyword-named identifiers — so the path is a strict speedup where it engages and
  byte-identical to before where it does not. The existing in-C++ FD
  self-validation gate (`set_functional_jacobian`) is unchanged and still guards
  correctness. Validated against finite differences to ≈1e-8 relative across every
  form in the scope (single- and multi-regulator, product and shared-denominator,
  basal+regulated, constant scalar factors), in both the interpreted (ExprTk) and
  codegen (C) paths; the native dense analytical Jacobian matches the SymPy one to
  ≈1e-14 on a real SBML model (`BIOMD0000000003`, 5 functional reactions). New
  tests in `python/tests/test_jacobian_native.py`.

### Performance

- **Sharded the codegen driver so genome-scale models compile within budget
  (#165).** The chunked RHS reaction bodies already compiled as parallel translation
  units (#160), but the analytical Jacobian scatter, the output evaluator (#163), and
  the obs[]/func[] recompute stayed in a single non-sharded **driver** translation
  unit — at genome scale (~113k reactions / ~75k species / ~18k functions) that was a
  ~38 MB serial `cc -O2` that blew the 600 s compile budget and fell back to the
  interpreted path. These are now split into NOINLINE units (via the new
  `_shard_value_lines`) compiled in parallel; the genome-scale driver drops 37.7 MB →
  0.2 MB and the cold compile goes from >600 s (timeout) to ~33 s. A follow-up drops
  132k dead `P_<name>` macros that were duplicated into every unit (scratch
  644 MB → 48 MB). Chunked output stays bit-identical to the flat path.

- **Parallel sharded codegen compile (#160).** Large chunked sources are split into
  independent NOINLINE translation units and compiled with an allocation-aware,
  memory-bounded pool of `cc -c`, then linked into the `.so`. The partition is
  job-count-independent and the link order fixed, so the result is byte-identical
  regardless of how many compilers run; a 1-core allocation (or
  `BNGSIM_CODEGEN_JOBS=1`) takes the unchanged serial path.

- **De-quadratic codegen source generation (#161).** `generate_rhs_c` /
  `generate_sens_rhs_c` and the model/SBML emitters rebuilt their identifier lookups
  per reaction and per function body; the rebuilds are hoisted out of the per-call
  path (built once via `_build_ident_lookup`). On the real 113k-reaction model,
  source generation drops from ~11 min to ~1.4 s with byte-identical output.

- **De-quadratic `from_sbml` / `from_net` model build (#164).** Two stacked O(n²)
  bugs — per-id libSBML lookups in the Python interpret phase, and a C++
  build-Jacobian-sparsity all-species fallback that densified the sparsity via
  transitive function-dependency resolution — made the genome-scale loader take
  minutes/OOM. With O(1) id lookups and proper transitive dep resolution,
  `from_sbml` goes from ~8 min/OOM to ~14 s.

### Fixed

- **Codegen timeout/abort no longer orphans `clang -cc1` (#166).** A compiler driver
  execs its backend (`clang -cc1`) as a separate process, so `subprocess`'s own
  timeout/kill only signaled the driver — the backend was reparented to PID 1 and
  kept pegging a core. `_run_compile` now launches each compile in its own process
  group and tears the whole group down (`killpg`) on timeout or abort.

## [0.9.37] - 2026-06-16

### Fixed

- **`.net` Michaelis-Menten codegen no longer emits a silent zero-rate RHS.**
  The lightweight `.net` parser kept only the first token of a reaction rate law,
  so BNG's whitespace form `MM kcat Km` became just `MM`, fell through as an
  unknown elementary parameter, and generated `0.0` for that reaction in the C
  RHS. The parser now preserves the full multi-token rate-law field, and the
  classifier accepts both `MM kcat Km` and `MM(kcat,Km)`. The codegen cache
  version is bumped again so stale cached `.so` files cannot keep serving the
  bad zero-rate RHS.

- **SSA no longer silently mis-simulates a rate rule that feeds a reaction
  propensity (#81).** An SBML rate rule `dX/dt = f` makes `X` a deterministic
  continuous quantity; bngsim compiles it into the Functional reaction
  `[] → [X]`, which the ODE path integrates as `+f`. Under *exact* SSA that
  synthetic reaction was fired as an ordinary stochastic birth/death **channel**,
  so a species/parameter target moved by integer ±1 at Poisson times instead of
  evolving smoothly — and when `X` then fed another reaction's kinetic law (the
  construct exact SSA cannot represent), the dependent propensity was corrupted.
  A time-varying decay `dk/dt = c` driving `A → ∅` at rate `k·A` reported
  `A(10) ≈ 625` against the analytical `≈ 82` (`z ≈ 24`): in most replicates `k`
  was still `0` because its birth reaction had fired ~Poisson(0.5) times. It was
  also a performance trap — a large `|f|` floods the propensity sum and the
  sampler crawls. The SSA/PSA loop now **excludes** rate-rule reactions from the
  stochastic selection and integrates their targets **deterministically**
  (forward Euler at the existing time-dependent sub-step granularity,
  `dt_max = horizon/1000`), exactly as the CVODE path accumulates `+f`; the
  propensities that read a target then follow its continuous trajectory. Targets
  are recorded at the exact output time (no sub-step lag), so a purely
  time-driven target shows zero cross-replicate jitter. Validated against
  closed-form means for the linear/affine cases and against an independent
  hand-rolled **Extrande** sampler (`python/tests/_extrande_reference.py`) for
  nonlinear propensities where the SSA mean differs from the ODE mean — neither
  RoadRunner `gillespie` nor COPASI's stochastic Time-Course accepts rate rules,
  so neither is usable as a reference here. The ODE path is untouched (the
  `is_rate_rule_ode` flag is read only by the SSA/PSA loop), and every
  non-rate-rule model records byte-identically. The `make_subset_model`
  operator-split helper forwards the flag, so a reconstructed subset keeps
  integrating a rate-rule target deterministically rather than re-firing it as a
  channel. Variable-volume compartments
  (a rate rule or event on a *compartment*) remain gated under SSA
  (`compartment_rate_rule` / `compartment_event_resize`): those additionally
  require a live per-reaction volume factor and have no SSA-runnable corpus model
  today; the deterministic-integration machinery added here is their foundation.

- **The SBML loader no longer silently approximates constructs it cannot
  faithfully simulate under ODE (#113).** `delay()`, `AlgebraicRule`, and
  `fast="true"` reactions were dropped with no warning and no error, producing a
  confident finite trajectory for a *different* mathematical system: `delay(x, τ)`
  was rewritten to `x` (the zero-delay ODE — bngsim has no DDE integrator), an
  `AlgebraicRule` DAE constraint was ignored by every rule loop, and a fast
  reaction integrated as an ordinary one. This contradicted the loader's
  "never a silent pass" contract; RoadRunner refuses all three at construction.
  bngsim now refuses too, mirroring the #94 unset-parameter gate. `delay()` and
  `AlgebraicRule` raise a `ModelError` at load (unsupported under every method);
  `fast="true"` stays a loadable SSA issue (the existing `validate_for_ssa` /
  `strict_ssa` override contract is untouched) and raises a `ModelError` at
  `Simulator` construction under `method="ode"`. The error names the construct
  and the offending element. `delay(x, 0)` is exactly `x`, so the zero-delay
  carve-out still loads, and a `delay()` buried in an *uncalled* funcDef does not
  trip the gate (only constructs that feed the integrated system count). Set
  `BNGSIM_ALLOW_UNSUPPORTED_CONSTRUCTS=1` to restore the legacy
  silent-approximation behavior for deliberate triage (e.g. bngsim↔RoadRunner
  comparison). In the `rr_parity` ODE sweep the 13 affected models move from
  `REFERENCE_FAILED` (bngsim ran, no oracle) to the auto-derived `BAD_TEST`
  (both engines refuse) — no longer a false asymmetry.

- **Periodic `floor()`/modulo dosing schedules no longer step over their dose
  pulses (#88).** A chemo/drug schedule encoded as a piecewise that switches on
  `floor()`/modulo time arithmetic (e.g. `exposure` active during a 0.0625-day
  window each day, MODEL1708310001 / Claret2009) puts narrow periodic
  discontinuities in the ODE RHS. The #72 root machinery only catches inequalities
  that compare the `time` csymbol *directly* against a constant; these edges flow
  through intermediate assignment-rule parameters and a single boolean root for a
  periodic pulse is non-monotonic, so the adaptive integrator stepped clean over
  the windows — on an exponentially growing state the missed dose-decay compounded
  (bngsim read y(100)=1603 / RoadRunner 1570 at the sweep tol, vs the exact
  segmented answer 953.07). The SBML loader now detects a time-dependent
  `floor`/`ceil`/`rem` feeding the ODE RHS, numerically measures the narrowest
  dose-window width, and stores a recommended integrator step bound on the model
  (`Model._periodic_disc_max_step`); `Simulator.run` applies it so no step can
  span a pulse. bngsim is then tol-stable at the segmented oracle (953.07) for
  either Jacobian. Models without a time-dependent floor/modulo in the RHS are
  byte-identical (no bound). A new `max_step` keyword on `Simulator.run` /
  `run_batch` overrides the auto bound (or bounds any model); `max_step<=0`
  disables it.

### Added

- **Large-model codegen chunking — hours-to-minutes compile for huge reaction
  networks.** A flat code-generated RHS over *N* reactions is one enormous basic
  block, and the C optimizer's per-function passes are superlinear in function
  size, so a ~100k-reaction model could take **hours** to compile at `-O1`/`-O2`
  (a synthetic mass-action RHS scales ≈ O(N^2.5) at `-O1`: 95 s at 20k reactions,
  521 s at 40k); the previous fallback dropped huge sources to `-O0`, dodging the
  cliff but shipping an unoptimized RHS the integrator then calls millions of
  times. At/above `BNGSIM_CODEGEN_CHUNK` reactions (default **2000**) BNGsim now
  splits the RHS — both the `.net` and model-based emitters — and the analytical
  sensitivity `bngsim_jac_vec` into many small `noinline` helper functions
  (`BNGSIM_CODEGEN_CHUNK_SIZE`, default 256, reactions per block). This caps
  basic-block size so compile time is ≈ linear, and `compile_rhs` then compiles
  the chunked source at `-O2` at any size (≈ minutes for 100k reactions; measured
  a real 6,000-reaction model at 39.8 s flat `-O1` → 22.6 s chunked `-O2`). The
  split preserves reaction order, so every `ydot`/`Jv_out` accumulation order is
  unchanged and the chunked `.so` is **bit-identical** to the flat one; below the
  threshold the emitted C is **byte-identical** to prior versions (no cache
  churn). The codegen cache version is bumped to invalidate stale `.net` `.so`s.
  `BNGSIM_CODEGEN_CHUNK=off` restores the flat emission.

- **`Simulator.run(..., max_step=...)` / `run_batch(..., max_step=...)`** — an
  explicit upper bound (time units) on a single internal ODE integrator step,
  overriding the per-model periodic-dosing bound above (#88).

- **Opt-in BLAS dense linear solver (#84).** A custom direct `SUNLinearSolver`
  factors the dense ODE Jacobian with Accelerate/LAPACK `dgetrf` (using its own
  LAPACK-ABI-matched pivots, so KLU's 64-bit indices are untouched — no
  `SUNDIALS_INDEX_SIZE` flip)
  while keeping SUNDIALS' built-in triangular back-solve. It is correct
  (bit-equivalent to the built-in LU including pivoting; `rr_parity` DIFF 0 with
  it forced on across the corpus) and gives a real end-to-end speedup on
  **factorization-bound** large dense models (e.g. 2.27× on a 1,265-species,
  11-factorization model). But the speedup tracks the *factorization count*, not
  `N` or density — a model that factorizes only a handful of times (even a
  fully-dense 5,000-species one) sees no benefit — and that count is a runtime
  property no static gate can target. So it ships **off by default** and engages
  only via `BNGSIM_LAPACK_DENSE=1`; the default dense path is unchanged (built-in
  LU, zero regression). An adaptive factorization-count gate that auto-enables it
  on factorization-bound runs is tracked in #132.
  `Result.solver_stats["linear_solver"]` reports which dense backend ran
  (0 = built-in, 1 = KLU, 2 = BLAS), and `bngsim._bngsim_core.HAS_LAPACK_DENSE`
  reports whether the build links a backend. See
  `dev/notes/gh84_lapack_dense_findings.md`.

### Changed

- **One vendored `exprtk.hpp` instead of two (#126, GH #49 Phase B).** The
  vendored NFsim tree no longer carries its own byte-identical copy of
  `exprtk.hpp`; `src/NFfunction/exprtk` is pruned from the vendor export
  (`vendor_nfsim.py` `PRUNED_PATHS`), and the NFsim `mu::Parser` shim's
  `#include "exprtk.hpp"` now resolves against the single host snapshot in
  `third_party/exprtk` (supplied to the NFsim build targets from the parent
  `CMakeLists.txt`, no new vendor carry). This completes the de-duplication ADR-005
  started — #49 collapsed the duplicated *logic*, this collapses the duplicated
  *file*. The byte-drift guard `test_exprtk_reserved_consistency.py` is retired:
  the two copies it pinned in lockstep are now one. No behavioral change
  (full suite green; an ExprTk refresh now touches exactly one place).

## [0.9.35] - 2026-06-06

### Added

- **The SBML `rateOf` csymbol (instantaneous species derivative dx/dt) is now
  evaluated (#106).** `rateOf(x)` asks the integrator for `dx/dt` at the current
  instant — a value the expression parser cannot fold. bngsim previously
  rendered it as `0` (official csymbol, libsbml type 323) or `NaN` (the COPASI
  `functionDefinition`/`<notanumber/>` idiom that #92 stopped crashing on), so
  any trigger or rate law reading it was silently wrong. The loader now
  normalizes **both** encodings to a per-species accessor, and `compute_derivs`
  refreshes a live derivative buffer via a one-pass *probe* before the real RHS
  (and the CVODE event root function / t=0 trigger init refresh it before
  evaluating triggers). One probe is exact for the SBML-supported acyclic case:
  every `rateOf` argument is a species whose derivative is independent of the
  values that consume it. Works in event triggers, rate rules, and assignment
  rules, under both the interpreted and codegen ODE paths; the Jacobian falls
  back to finite differences for `rateOf`-bearing reactions. Validated against
  libRoadRunner on all four BioModels SBML-corpus models that use `rateOf`
  (e.g. MODEL1910030001's event now fires at t≈92.5). `rateOf` is rejected
  under SSA (no defined instantaneous derivative in a stochastic trajectory).
  Models without `rateOf` are byte-identical.

- **Time-dependent piecewise discontinuities are now resolved by the ODE
  integrator (#72).** A piecewise assignment rule / rate law that switches on
  the SBML `time` csymbol (drug-dosing windows, scheduled stimuli) puts a
  discontinuity in the RHS that CVODE, with its default interpolated output,
  could step clean over — silently dropping a narrow pulse. At load time the
  SBML loader now extracts every `time` inequality inside such a piecewise and
  registers it as a CVODE root ("discontinuity trigger", reusing the event
  root-finding path), so the integrator stops exactly at each pulse edge and
  cannot step over it. Models with no time-dependent piecewise register zero
  triggers and integrate bit-for-bit as before. New
  `NetworkModel.n_discontinuity_triggers` /
  `ModelBuilder.add_discontinuity_trigger`.

- **Analytical Jacobian for Functional / Michaelis–Menten rate laws, on by
  default (#76).** Previously only all-Elementary (mass-action) networks got an
  analytical Jacobian; every model with a Functional rate law fell back to
  CVODE's finite-difference Jacobian (O(N) extra RHS evals per Jacobian build).
  At load time bngsim now symbolically differentiates each Functional rate law
  (`bngsim._jacobian`, sympy) with respect to the observables it depends on,
  chain-rules through each observable's species group, and registers the
  derivative expressions in the C++ ExprTk evaluator; the dense/sparse Jacobian
  callbacks evaluate them and scatter by net stoichiometry — entirely in C++ at
  run time (the integration loop never calls Python). Reaches the **interpreted**
  engine (the SBML path), not just codegen.

  Every attach is guarded by an in-C++ FD self-validation gate that compares the
  assembled analytical Jacobian against **reliability-gated** finite differences
  (two-step Richardson convergence + catastrophic-cancellation detection + a
  non-finite-entry guard) at the initial state and several spread-out probe
  states; on any trustworthy mismatch the model silently keeps the
  finite-difference Jacobian. Validated across the full BioModels SBML corpus
  (1597 models): **1186 functional models attach, zero wrong attaches, zero
  needless fall-backs** — every non-attach is a warranted bail (singular-at-init
  derivative, un-inlinable construct, or a genuine symbolic divergence).
  All-Elementary models are **byte-identical** (no Functional reactions ⇒ the
  symbolic path is skipped). Set `BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0` to force
  the finite-difference Jacobian.

- **Compiled-C analytical Jacobian for codegen models (#76, Task 4).** When a
  model is codegen-compiled, the `.so` now also carries `bngsim_codegen_jac` — a
  C mirror of `NetworkModel::fill_dense_analytical_jacobian` emitted by
  `bngsim._codegen.generate_jacobian_from_model`. The CVODE **dense** Jacobian
  dispatch prefers it over the interpreted (ExprTk) `cvode_analytical_dense_jac`
  whenever codegen is active and the symbol resolved, keeping the interpreted
  path as the fallback. With this, a codegen dense ODE run is **fully compiled**
  — RHS *and* state Jacobian — so the integration loop no longer touches the
  ExprTk evaluator (previously a codegen run still fell back to the interpreted
  analytical Jacobian). The emitted C reproduces all four contribution blocks
  **scatter-for-scatter** — Elementary
  closed form and Michaelis–Menten (tQSSA) closed form from the C++ scatter plan
  (`codegen_jacobian_plan`, rows pre-resolved), and Functional per-species (SBML
  chain rule) + per-observable (.net product rule) reconstructed from
  `functional_jacobian_context()` via the shared sympy core
  (`bngsim._jacobian.sympy_to_c`) — plus the fixed-species row zeroing. It is
  emitted **only** when the interpreted analytical Jacobian is itself complete
  (so the compiled scatter always matches the FD-self-checked interpreted
  assembly) and the model takes the dense path; any un-emittable derivative
  declines the whole compiled Jacobian (interpreted/FD fallback — never wrong C).
  Verified bit-identical against `_dense_analytical_jacobian` across all four
  blocks (`test_codegen_jacobian.py`), and per-eval ~6× cheaper to compute. The
  end-to-end wall-clock win is **modest (~1.0–1.05×)** on the large functional
  SBML corpus measured (`benchmarks/suites/jacobian/bench_codegen_jac.py`):
  there the run is dominated by the dense LU factorization and RHS, and CVODE
  reuses each Jacobian across many steps, so the Jacobian eval is a small slice
  (Amdahl) — same regime the analytical-vs-FD benchmark already documents. The
  primary value is the fully-compiled loop and the foundation it lays; a
  jac-eval-bound model (frequent rebuilds, cheap LU, expensive functional rate
  laws) benefits more. The sparse-CSC (KLU) compiled Jacobian remains a
  follow-up. Set `BNGSIM_NO_CODEGEN_JAC=1` to force the interpreted Jacobian (A/B
  the feature); `BNGSIM_JAC_DEBUG=1` prints which dense Jacobian the dispatch
  selected.

- **Michaelis–Menten (tQSSA) closed-form analytical Jacobian (#76 follow-up).**
  MM reactions (`rate = kcat·E·sFree/(Km+sFree)`,
  `sFree = ½((S-Km-E)+√((S-Km-E)²+4·Km·S))`) previously cleared the analytical-
  availability flag, so any model containing one fell back to CVODE's finite-
  difference Jacobian. The engine now emits `∂rate/∂E` and `∂rate/∂S` in closed
  form — the chain rule through `sFree` done analytically, matching the engine's
  own rate law exactly — and scatters them by net stoichiometry in the dense and
  sparse Jacobian callbacks (`include/bngsim/mm_jacobian.hpp`, one source of
  truth). Where the RHS clamps `sFree` to 0 the derivative is 0 (the flat region).
  No sympy is involved: the derivative is hand-derived and validated against both
  central finite differences and generic sympy differentiation of the tQSSA rate
  law (≤1e-10 relative). Like the Elementary closed form it carries no per-step
  self-check (it is the exact derivative of the engine's rate law). MM models now
  integrate with an exact analytical Jacobian instead of FD.

- **`.net` per-observable analytical Jacobian + mass-action product rule (#76
  follow-up).** Rule-based `.net` Functional reactions have the form
  `rate = func(observables)·∏reactants`; the C++ analytical-Jacobian path
  previously rejected these (`per_observable`) terms and routed the whole model
  to finite differences. It now scatters the full column derivative
  `∂rate/∂x_j = (∂func/∂x_j)·∏R + func·∂(∏R)/∂x_j`, chain-ruling
  `∂func/∂x_j = Σ_k (∂func/∂obs_k)·(∂obs_k/∂x_j)` through each observable's
  species group and reusing the Elementary product-rule machinery for the
  species factor. The scatter is one source of truth
  (`include/bngsim/functional_jac_scatter.hpp`) shared by the dense and sparse
  CVODE callbacks; `func` is read from the reaction's bound rate parameter so the
  RHS and the Jacobian use an identical value. The Python symbolic core and the
  input wire contract were already in place — this is the C++ acceptance +
  scatter, still guarded by the FD self-check. On a reduced EGFR network
  (`egfr_net_red.net`: 40 species, 123 reactions, 16 per-observable Functional
  reactions) the analytical Jacobian now attaches (was FD) and runs ~1.15× faster
  per step with an identical trajectory (peak-relative ~1e-15). This is the path
  that wins decisively on large, dense rule-based functional networks, where
  colored finite differences degrade toward O(ns).

- `SolverStats.n_nonlin_conv_fails` — CVODE's nonlinear (Newton) convergence
  failure count, now stored in `Result.solver_stats` (#76 follow-up). The value
  was already queried from CVODE (`CVodeGetNumNonlinSolvConvFails`) but discarded
  before reaching Python. It is the most direct robustness signal for the
  Jacobian — an inexact Jacobian makes Newton give up more often, forcing step
  cuts — so the analytical-vs-FD benchmark can quantify convergence robustness,
  not only per-step cost.

### Fixed

- **ODE event chattering on a non-negativity clamp no longer stalls the
  integrator (#95).** A "keep X ≥ 0" event (`trigger: X < 0`, `assignment:
  X := 0`) re-fires on every floating-point sign flip once X decays far below
  `atol`, and each firing forces a `CVodeReInit` that resets BDF to order-1 tiny
  steps — Zeno behavior that crawled the solver. `BIOMD0000000711` (Hancioglu2007
  influenza model, 11 species) was the cleanest case: bngsim took >300s where
  RoadRunner — which keeps the variable cleanly positive down to ~1e-58, so its
  event never fires — finishes in ~0.1s. The CVODE event loop now detects an
  event re-firing with **both** negligible time advance **and** a sub-tolerance
  state change, and after a run of such fires suppresses that event's trigger
  root so the integrator steps over the noise floor (re-arming if the assigned
  species climb back above the floor). The dual criterion leaves genuine
  recurring events untouched. `BIOMD0000000711` goes TIMEOUT→PASS
  (`max_rel_err=0`) in ~0.5s; a re-run of all 214 event-bearing rr_parity ODE
  models shows that model as the **only** changed verdict, with zero metric drift
  on the 204 passing in both builds. Regression test
  `test_sbml_event_chatter_biomd711.py`.

- **The #76 analytical-Jacobian derivation is now time-budgeted, so a large model
  no longer hangs the build (#95, ODE "timeout" half).** `attach_functional_jacobian`
  symbolically differentiates every Functional rate law with sympy at
  `Model.from_sbml`/`from_net` time. On a handful of large BioModels that
  derivation runs tens of seconds to over a minute while the ODE solve is already
  sub-second under a finite-difference Jacobian — and because the rr_parity harness
  times build+solve against one wall cap, the slow *build* read as an ODE
  *timeout*. Profiling showed the cost is the build-time sympy derivation, not the
  dense linear solve (#84) it was filed under: `BIOMD0000000496` derives in ~41s
  but solves in 0.25s; `BIOMD0000000628`'s 18-char rate laws each inline to ~21kB
  and derive in ~75s but solve in 0.1s. In every measured case the analytical and
  finite-difference solves are identical to within solver noise, so the derivation
  bought nothing. The derivation now runs under a wall budget
  (`BNGSIM_JAC_DERIV_BUDGET_S`, default `20.0`s; `0`/`inf`/`none` disables it),
  checked both between reactions and inside the per-observable differentiation loop
  so overshoot is bounded to a single rate law: a model that derives under budget
  keeps the analytical Jacobian, one that exceeds it logs the fallback (reactions
  processed + elapsed) and integrates on the finite-difference Jacobian — the
  pre-#76 behavior, byte-identical results. `BIOMD0000000496`/`628` build collapses
  47x / >15x with an unchanged trajectory; models that derive quickly are
  unaffected and keep the analytical Jacobian. The 20 s default was set by
  classifying every slow-deriving rr_parity model (analytical vs finite-difference
  solve at the 1e-9/1e-12 parity tolerance): the analytical Jacobian is a pure
  speedup that FD reproduces for all of them **except** `BIOMD0000000457` (a stiff
  model whose FD solve fails at that tolerance), which derives in ~12 s — so the
  budget sits in the [~12 s, ~41 s] gap between it and the fastest derivation loser,
  keeping `457` on analytical while still cutting the 40–75 s pathologies. The
  previously-silent `try/except: pass` around the `.net` attach call (`_model.py`)
  now logs at debug. Regression test `test_sbml_jacobian_budget_biomd496.py`.

- **The SBML math translator now fails closed on an unsupported construct
  instead of silently emitting `0` (#97).** The live MathML→ExprTk translator's
  fallback logged a warning and returned `"0"` for any AST node it did not
  recognise — a silent wrong RHS: the model loaded fine and mis-simulated, with
  no load error. (The `rateOf` csymbol fixed in #106 was one instance: type 323
  hit this same fallback and became `0`.) The fallback now raises `ModelError`
  naming the libsbml AST type number, its symbolic name, and the offending
  construct's infix form (e.g. `normal(0, 1.5)`), with a targeted hint for the
  `distrib` package. A blast-radius survey before the change confirmed this is a
  loud reject, not a regression: **0** of 1597 BioModels (rr_parity) and **0** of
  2042 benchmark SBML models reach the fallback — every deterministic operator
  is already handled. The only constructs that now fail closed are the SBML
  `distrib` package random-draw csymbols (`normal`/`uniform`/`poisson`/… —
  libsbml AST types 500–511, exercised by the SBML Test Suite's `distrib`
  cases): stochastic draws with no deterministic translation, which previously
  loaded and silently computed with `0`. Regressions in
  `test_sbml_unsupported_math.py`.

- **Removed the dead second MathML→ExprTk translator (#97).** `_sbml_loader.py`
  carried two string translators; only `_ast_to_exprtk_recursive` (via
  `_ast_to_exprtk_with_funcdefs`) was reachable. The unused `_ast_to_exprtk` /
  `_piecewise_to_exprtk` pair had already drifted — `min`/`max`/`quotient`/`rem`/
  `implies` and the full inverse-trig/hyperbolic set existed only in the live
  copy — so a maintainer editing the dead one would have had no effect. Both are
  deleted; a guard test keeps them from creeping back. No behavior change.

- **Codegen now translates ExprTk `max`/`min` to C `fmax`/`fmin`.** The
  model-based codegen RHS emitted `max(...)`/`min(...)` verbatim, but `<math.h>`
  has no `max`/`min`, so any model with a `max`/`min` in a rate law or function
  failed to compile under `codegen=True` (e.g. `BIOMD0000000696`). They now map
  to the binary `fmax`/`fmin` builtins; the loader already emits nested binary
  forms for n-ary `max`/`min`, so this covers both. Both names are
  ExprTk-reserved, so they can never collide with a user model symbol.

- **A non-finite (`nan`/`inf`) real literal in an expression now compiles
  (#92).** The MathML→ExprTk translator rendered every `AST_REAL` with
  `repr()`, so a NaN constant became the bare token `nan` — which ExprTk has
  no symbol for (`init_builtins` registers neither `nan` nor `inf`, and the
  `<n>#nan` lexer form does not survive embedding in a parenthesised
  subexpression), failing the load with `ERR239 - Undefined symbol: 'nan'`.
  `MODEL1910030001` hit this: COPASI exports the SBML `rateOf` csymbol as a
  `functionDefinition` whose body is `<notanumber/>`, and inlining it into the
  `control_of_chi_rise` event trigger produced `nan*c > 0`. A non-finite value
  is now emitted as pure arithmetic that constant-folds to the same IEEE double
  (`(0.0/0.0)` for NaN, `(1.0/0.0)` / `(-1.0/0.0)` for ±inf), so the literal
  compiles and propagates correctly: the NaN trigger comparison is permanently
  false (NaN compares false), the inf comparisons behave as IEEE dictates. The
  model now loads and integrates to RoadRunner parity (max rel ~3e-5) up to the
  point RR's event fires. **Known gap:** RR evaluates `rateOf(species)` as that
  species' instantaneous derivative (not the NaN stub), so RR fires
  `control_of_chi_rise` at t≈92.5 while bngsim's permanently-false trigger does
  not — full parity for this model needs `rateOf`-csymbol support, tracked
  separately. Regressions:
  `test_sbml_loader_followup.py::test_gh92_nan_literal_in_event_trigger` and
  `::test_gh92_real_literal_renders_nonfinite_as_arithmetic`.
- **A reaction id referenced inside another reaction's kineticLaw now resolves
  to that reaction's rate (#91).** SBML L3 lets a `<ci>` name a reaction id,
  where it evaluates to the reaction's rate of progress (its kineticLaw value).
  The loader already pre-registered such reaction ids as ExprTk functions when a
  *rule* (rate/assignment) referenced one, but not when *another reaction's
  kineticLaw* did — so `MODEL2306170002`, whose `r0` law reads
  `… * ((r9b − r10a) + r6n_c)` over three mass-action reactions, failed to load
  with `ERR239 - Undefined symbol: 'r9b'`. The pre-scan that collects
  reaction-id references now walks every reaction's kineticLaw in addition to
  the rules (a reaction's own id is excluded — a law referencing its own rate is
  a self-referential fixed point, not a resolvable symbol), so the referenced
  reactions get an `add_function(rid, …)` and the referencing law compiles. The
  model now loads and integrates to RoadRunner parity (38 species, max abs diff
  8e-5 at atol/rtol 1e-10). Regression:
  `test_sbml_loader_followup.py::test_gh91_kinetic_law_references_reaction_id`
  (a reaction firing at `2·r1` into a fresh species, checked against the exact
  invariant `C == 2·B`).
- **Mass-action reactions in an assignment-rule compartment now use the live
  volume (#98, 0.9.34).** The Elementary counterpart of #87. A mass-action
  (Elementary-classified) kinetic law whose compartment factor is an
  assignment-rule (variable-volume) compartment — e.g. the leading `tC` of a
  COPASI-style `tC · (kf·A·B − …)` law where `tC := mC + …` — folded
  `comp_volumes[tC]` (the load-time *static* volume) into the reaction's scalar
  rate `sf`. A scalar rate cannot carry the live `tC` symbol, so as the
  compartment grew the integrated amount was wrong by `V_static/V_live(t)`
  (a bare `tC·k` synthesis gave `amount = V_static·k·t`, missing the volume
  growth entirely; RoadRunner matches the closed form). `_classify_mass_action_ast`
  now refuses mass-action classification when a compartment factor is an
  assignment-rule compartment (mirroring the existing guard for AR-driven
  *species* factors), routing the reaction to the Functional path where #87's
  live-aware divide (live symbol inside the law, numeric `V_static` storage
  divide) integrates it exactly. Unsurfaced in the corpus — no BioModels keep-set
  model has a mass-action reaction factoring an assignment-rule compartment (the
  9 assignment-rule-compartment models stay `max_rel_err=0`, unchanged) — so the
  change only affects models that previously computed the wrong trajectory.
  Regression: `test_sbml_assignment_rule_compartment.py::test_massaction_law_in_ar_compartment_uses_live_volume`
  (closed-form amount in a linearly growing compartment, with an explicit
  Elementary-bake regression guard).
- **Assignment-rule-driven variable-volume compartments now integrate and report
  correctly (#87, 0.9.33).** A compartment whose size is set by an *assignment
  rule* — e.g. `tV := mV + dV` in BIOMD0000000856 (Heldt2018 budding-yeast
  cell-cycle oscillator) — is variable-volume, but bngsim recognised only
  rate-rule (#86) and event-resized (#74) compartments as such; an
  assignment-rule compartment was treated as constant at its load-time size. The
  symbol itself was live (its rule function writes `mV+dV` into the same-named
  parameter each RHS), but for an amount-valued (`hasOnlySubstanceUnits=true`)
  species stored as `amount/V_static`, two paths used the *live* compartment
  symbol `V(t)` where they should have used the load-time numeric `V_static`: the
  Functional storage-conversion divide and the event-assignment target divide.
  Dividing the amount-rate by `V_live(t)` throttled every reaction by
  `V_static/V_live(t)` as the compartment grew — for #856 the SBF→CLN cascade
  never ignited, `CLN/tV` never reached `StartThr`, no cell-cycle event ever
  fired, and the published limit cycle collapsed to a flat monotone line while
  RoadRunner and COPASI (third-oracle confirmed) both oscillate. A third bug was
  reporting: the integrated amount is correct, but the reported concentration
  must be `amount/V_live(t)`, and bngsim reported the stale `amount/V_static`. A
  new report pass (`_apply_varvol_ar_conc_map`) rescales it, reading `V_live(t)`
  from the compartment's own assignment-rule *expression* column (an
  assignment-rule compartment has no ODE state, so — unlike #85's rate-rule map —
  the live volume is not a promoted-species column). With all three fixed, #856
  reproduces the RoadRunner/COPASI limit cycle to `max_rel_err=0`. Scoped to
  amount-valued species in assignment-rule compartments: static, rate-rule, and
  event-resized compartments are byte-identical (the divide is numeric == the
  symbol value when the compartment is static), and all 9 BioModels keep-set
  models with an assignment-rule compartment PASS rr_parity (MODEL1606100000's
  #85/#86 rate-rule case unchanged). Known limitation (tracked as #98): a
  *mass-action* (Elementary-classified) reaction in an assignment-rule
  compartment still bakes the static volume into its scalar rate; no corpus model
  exercises it (all 9 PASS), so the fix is deliberately confined to the Functional
  path #856 uses.
  Regression: `python/tests/test_sbml_assignment_rule_compartment.py`
  (closed-form amount + concentration oracles in a linearly growing compartment,
  plus a static no-op control).
- **Two SBML identifier collisions with the ExprTk namespace now load (#90,
  0.9.32).** Same family as the closed reserved-word work (#24 `t`/`time`, #18
  `const`/`true`/`false`), extended to two classes the BioModels corpus
  surfaced. **(a) Reserved math-builtin** — a model symbol named after an ExprTk
  builtin function (`log`, `sin`, `exp`, …). MODEL1812040006 is a COPASI export
  with a parameter literally named `log` whose assignment rule is `log = ln(V)`;
  the `<ln/>` builtin renders to ExprTk `log(...)` and a single flat namespace
  could not hold both the variable `log` and the builtin (the C++ evaluator
  raised the #64 "declared model symbol … used as a function call" error). The
  SBML loader's `_safe_name` now renames any model symbol whose name is an ExprTk
  builtin — sourced from the core's `reserved_names()["functions"]` so the set
  cannot drift from the symbol table that rejects the collision — to `_ant_<name>`,
  leaving the builtin call distinct. **(b) `u_`-prefix builtin-constant** —
  bngsim registers its `_X` constants (Planck `_h`, Avogadro `_NA`, …) under the
  ExprTk key `u_X` (ExprTk rejects a leading `_`), so a user parameter named
  literally `u_h` aliased Planck's slot and failed to register
  (BIOMD0000000950, Chitnis2012). `expression.cpp` now reserves the `u_X`
  constant keys, so such a parameter takes the transparent `r_<name>` mangling
  path like any other reserved-word collision; its Python-facing name and value
  are unchanged. Both fixes are additive: the rename only fires on a name that
  previously failed to load (no corpus species is named after a math builtin, so
  the species-name parity alignment is untouched), and the `u_X` reservation
  only mangles names that previously could not register. Regressions:
  `python/tests/test_sbml_exprtk_identifier_collisions.py` (load + value + ODE
  oracle for each class, parametrized over the builtin and the seven constant
  keys) and `tests/test_bngsim.cpp::test_u_constant_key_collision` (a user `u_h`
  and the built-in `_h` coexist).
- **Concentration-valued species in a rate-rule (continuously variable-volume)
  compartment now carry the dilution term `−[S]·V̇/V` (#86, 0.9.30).** For a
  `hasOnlySubstanceUnits=false` species `S` of amount `A = [S]·V` in a
  compartment whose volume `V(t)` is driven by a rate rule, the concentration
  ODE is `d[S]/dt = (1/V)·dA/dt − [S]·V̇/V`. bngsim emitted only the reaction
  term (the `_vd_<rid>_varvol` ÷V_live path, #74); the **dilution term** — the
  concentration change caused by the volume itself moving — was missing, so the
  species was integrated as if its compartment were static and both its
  concentration and its implied amount diverged from RoadRunner (the repro was
  247% high at t=10). The loader now emits the dilution term as an additive
  Functional reaction for every such species, including pure-dilution
  (reaction-free) species, whose closed form is `[S](t) = A/V(t)`. Boundary
  (`boundaryCondition=true`) species in a rate-rule compartment are un-fixed so
  the dilution term integrates (their amount is conserved → concentration
  dilutes, matching RoadRunner); boundary species are excluded from every
  reaction's reactant/product lists, so the dilution term is their sole
  derivative. The bare-id (amount) selector in `Result.as_roadrunner` now
  recovers `conc·V_live` for these species via a new `_varvol_amount_map`
  (the hOSU=false counterpart of #85's concentration-rescale map — the
  concentration column is already correct and is *not* rescaled). Amount-valued
  (`hasOnlySubstanceUnits=true`) species are untouched (#85 handles their
  report-time rescale); rate-rule- and assignment-rule-target species are
  excluded (their own rule defines the derivative); `constant=true` species are
  out of scope for now. Gated on (hOSU=false ∧ rate-rule compartment), a
  combination no BioModels corpus model exhibits — so every static / unit-volume
  / corpus model is byte-identical (full 1597-model load sweep: every
  `_varvol_amount_map` empty). The new term stays inside the analytical
  Functional Jacobian (#76, no finite-difference fallback). Regressions:
  `python/tests/test_sbml_variable_volume_dilution.py` (closed-form floating +
  boundary dilution, shrinking and state-dependent compartments, the issue's
  Michaelis–Menten repro vs a SciPy + RoadRunner oracle, amount conservation,
  and scope guards).
- **initialAssignment now tracks an assignment-rule-target dependency that
  carries a stale raw `value=` (#73, MODEL1606100000 Talemi2016 yeast osmo).**
  COPASI exports duplicate a quantity as both a raw-valued parameter and an
  assignment-rule target, then point another parameter's initialAssignment at it
  (`ModelValue_19 := cin0`, where `cin0 := ModelValue_18 - Metabolite_1` is an
  assignment rule whose stored `value=` is stale). The loader's guarded
  IA-evaluation loop read `cin0`'s stale raw value, and the post-loop AR override
  that finalizes `cin0` did not re-propagate into `ModelValue_19` — leaving
  `ModelValue_19` ≠ `cin0` despite the SBML equating them, which flipped the sign
  of the boundary species `Osmin`. The loader now re-runs initialAssignments (and
  the AR override) to convergence after the AR override, so each IA target tracks
  its assignment-rule dependency (per SBML, an assignment rule holds at t=0).
  Additive and gated on the presence of initialAssignments; models without this
  cross-dependency are byte-identical (full 1597-model rr_parity ODE re-sweep:
  zero PASS→DIFF, the one affected model improved). Third-oracle confirmed: at
  the disputed value RoadRunner **and** COPASI agree with the post-fix bngsim
  (the issue's original "RoadRunner blows up / bngsim bounded" premise was stale —
  all three engines agree the osmotic volume runs away; see
  `dev/notes/rr_parity_triage.md`). Regression:
  `python/tests/test_assignment_rule_init.py::TestInitialAssignmentTracksAssignmentRuleTarget`.
- **Chemotherapy / dosing-pulse models no longer escape because the integrator
  skipped narrow infusions (#72).** BIOMD0000000879 (Rodrigues2019
  chemoimmunotherapy of CLL) delivers 7 chemo infusions each only 0.125 t-units
  wide via a piecewise-in-`time` assignment rule. bngsim's CVODE stepped over 6
  of the 7 (delivering λ·∫Q = 1080 = one dose vs the correct 7560), so the
  cancer `N` escaped to carrying capacity (N→9.6e11, k=1e12) instead of the
  immune-controlled branch. With the discontinuity-trigger roots above all 7
  doses are delivered and `N` decays to ~2.5 — matching a segmented SciPy oracle
  and libRoadRunner on a fine output grid. (The residual rr_parity gate DIFF at
  the coarse sweep grid is a RoadRunner grid/tol sensitivity — observed only at
  the coarse default settings, where RR's value diverges and then converges to
  bngsim's once the grid is refined or tolerance tightened; not traced to RR's
  source — allow-listed as a known artifact, not a bngsim defect. See
  `dev/notes/rr_parity_triage.md`.)

- **An SBML parameter / compartment mutated only by an event no longer leaks
  into the trajectory output as a species column (#71).** The engine applies
  event assignments by writing species slots, so a `parameter` or `compartment`
  that an event changes must be *promoted to a species* to carry per-trajectory
  state. It is not a floating species, though, and RoadRunner does not emit it as
  a trajectory column — but bngsim was appending it to the species output anyway
  (MODEL1108260014 surfaced `parameter_1` and `compartment_1` as spurious
  bngsim-only columns, 84 vs RoadRunner's 82). This is an output-correctness bug
  on its own (`Result.species` / `species_names` and `.cdat` carried entities
  that are not floating species), and it perturbed the `rr_check` peak-relative
  screen (`dev/investigations/`), whose **global-peak** significance denominator
  was inflated by `compartment_1`'s value (5 vs the true common scale ~0.01),
  mislabeling the model. (The shared `rr_parity` differ was unaffected — it
  scores only the *common* species intersection, which never contained these
  bn-only columns, so this change is verdict-neutral for that sweep.) Fix: a new
  `Species::reported` flag (default `true`) marks a promotion as
  internal-state-only; the SBML loader sets it `false` for event-promoted
  parameters/compartments (rate-rule-promoted parameters stay reported — they are
  genuine ODE variables RoadRunner reports too). The promotion keeps its ODE
  slot, RHS/Jacobian participation, and a same-named observable (so referencing
  expressions resolve the live value); only the *output projection* drops it —
  `Result.species` / `species_names`, the `.cdat` export, and the per-species
  volume-factor list all project to the reported subset, mirroring the existing
  `public_observable_indices` filtering. MODEL1108260014 now reports 82 columns
  matching RoadRunner's species set exactly; its earlier-suspected `species_57`
  divergence is not present in the current build (both engines hold it at 9e-6,
  agreeing to ~4e-18 — an independently-fixed dynamics concern, orthogonal to this
  output-only change). **Byte-identical for `.net` models, ordinary SBML, and
  every all-reported model** — the projection is wired only when a model has an
  unreported species, so the output column set and ordering are unchanged
  everywhere else (the full BioModels ODE `rr_parity` sweep is unchanged: same 5
  pre-existing DIFFs, none of them event-promotion models). Regression tests:
  `python/tests/test_sbml_event_param_not_reported.py` (parameter case + `.cdat`
  export) and the compartment case in
  `python/tests/test_sbml_compartment_resize_event.py` (the resized compartment
  is now read from observables). Bumps version to 0.9.26.

- **Functions are now evaluated in topological (dependency) order, not
  declaration order — fixes a latent path-dependent RHS for SBML models with
  non-topological assignment rules (#76).** `evaluate_functions()` writes each
  function's value into its bound parameter and later functions read those
  parameters; when a function referenced one declared *after* it, a single
  declaration-order pass read the referenced function's STALE bound value (left
  from the previous RHS evaluation). Since `compute_derivs()` runs one pass per
  RHS evaluation, the RHS became path-dependent rather than a pure function of
  `(t, y)`, silently corrupting integration (e.g. MODEL8684444027 failed CVODE
  at t=0.5; BIOMD0000000268 at t=20 — both now integrate and match RoadRunner to
  ~1e-6). `ModelBuilder` topologically sorts `var_param_bindings` (Kahn, seeding
  ready nodes in ascending index) so one pass converges; any residual cycle
  (malformed input) falls back to declaration order. **`.net` models are
  byte-identical** — BNG `run_network` emits functions in dependency order, so 0
  of 2272 `.net` function blocks forward-reference and the sort is a no-op there;
  only non-topological SBML changes (now correct). Regression test:
  `python/tests/test_topological_function_eval.py`.

- **Event that changes a compartment's size now rescales species
  concentrations and divides Functional rates by the live volume (#74).**
  When an SBML event assignment resizes a compartment, the contained species'
  *amounts* are preserved and their *concentrations* recomputed. bngsim stores
  concentrations, so it previously left them unchanged — silently multiplying
  every amount by the volume ratio (BIOMD0000000338's `dilution_event` tripled
  `compartment_1` and bngsim held `Pk` at ~450 where RoadRunner correctly drops
  it to 150 = 450/3). Two coupled fixes in `_sbml_loader`:
  (1) the loader injects a per-species rescale assignment `s := s·V_old/V_new`
  into the resizing event for every non-hOSU, non-AssignmentRule-target species
  in that compartment — evaluated against pre-fire state thanks to the engine's
  simultaneous event semantics (hOSU species are skipped: the engine already
  reads them as amount = `stored × V_c`, a load-time constant, so the amount is
  preserved with no rescale); and (2) a Functional reaction whose species live
  in a variable-volume compartment now divides its rate by the *live* compartment
  symbol (the `common_vs == 1.0` emission previously reused the raw
  `compartment·f(...)` law, leaking the post-resize volume into `d[conc]/dt` and
  running the cascade 3× too fast — which also made BIOMD0000000338 unintegrable
  at the parity tol). Mass action is unaffected (the compartment cancels
  analytically) and `V_c≠1` variable compartments already divided by the live
  symbol. A compartment resized by an event is now rejected under SSA
  (`compartment_event_resize`): a discrete event resize is tractable for SSA in
  principle (preserve counts, recompute propensities), but bngsim's SSA engine
  bakes each reaction's volume factor at load and the ODE-side resize handling
  would corrupt molecule counts, so it is rejected rather than run wrong (the
  continuous rate-rule-on-compartment case stays rejected for the harder
  reason). BIOMD0000000338 now matches
  RoadRunner at the sweep tol (`max_rel_err=0` in `rr_parity`), as do synthetic
  mass-action / Functional resize models against closed-form and SciPy oracles.
  Byte-identical for static / constant-volume / hOSU-only models (`variable_comps`
  is empty, so no rescale is injected and no divide is added). Bumps version to
  0.9.18.

- **Rate rule on an hOSU=true species in a V≠1 compartment (#75 follow-up).**
  A `rateRule` on a `hasOnlySubstanceUnits=true` species defines `d(amount)/dt`,
  and its RHS reads that species as an amount. The step-2 observable-shadow
  change made such a species read as its amount everywhere, but the rate-rule
  lowering (a Functional synthesis reaction) was not dividing the storage
  accumulation by the compartment volume, so a linear decay came out at rate
  `k·V_c` instead of `k` — a factor-of-V error. The lowered reaction is now
  marked `per_species_volume_scaling` when the target is amount-valued, which
  divides the ODE accumulation by `V_c(target)` while leaving the SSA propensity
  in amount/time. Verified against libRoadRunner on a synthetic model and on
  BIOMD0000000353 (a rate rule on an hOSU species in a 3.5e-13 L compartment,
  previously off by ~10⁹, now matching RoadRunner to 2.5e-6). Byte-identical for
  hOSU=false targets and V_c=1. Bumps version to 0.9.17.

### Changed

- **Codegen ODE path: cache the compiled `.so` across `run()`s (#77).** The
  CVODE simulator previously `dlopen`/`dlsym`/`dlclose`'d the codegen `.so` on
  *every* `run()`, so the codegen path carried a flat fixed per-run overhead
  (~0.5 ms on the dev machine) with no integration compute to amortize it — it
  lost to the ExprTk bytecode path on short-horizon / small models where fixed
  setup dominates. The library handle and the resolved `bngsim_codegen_rhs` /
  `bngsim_codegen_sens_rhs` / `bngsim_codegen_jac` function pointers are now
  cached on `CvodeSimulator::Impl`, keyed by `.so` path: repeated `run()`s on
  the same simulator reuse the already-mapped library, and only a changed path
  triggers a one-time reload (the cached library stays open for the simulator's
  lifetime). The `CodegenUserDataForSO` struct handed to the `.so` is likewise
  built once per `run()` rather than reconstructed on every RHS / Jacobian
  callback. With both, the codegen `floor` (a 3-output-point run, isolating
  fixed setup) now ties or beats the ExprTk floor at every model size measured
  (2–40 species; was 1.5–6× slower). No numerical change — repeated-run output
  is byte-identical to the interpreted path and the full suite stays green
  (pytest 1141 + 3 skipped, C++ 4/4). Bumps version to 0.9.25.
- **hOSU amount restoration is now a single core capability (#75, step 2).**
  Step 2 folds the two remaining loader-side hOSU paths — the Functional
  `_wrap_hosu_amounts` / `_amt_<rid>` AST rewrite and the linear-AR
  `_hosu_amount_factor` observable-weight reweighting — into the same
  `Species::amount_valued` flag, so a `hasOnlySubstanceUnits=true` symbol reads
  as its amount (`stored × volume_factor`) at *every* evaluation site, not just
  the mass-action species factor. The mechanism: `NetworkModel::update_observables`
  multiplies an `amount_valued` species's contribution by `volume_factor`, and
  because the loader registers a same-named observable that shadows each species
  variable in the ExprTk evaluator (observables are bound before species, and
  `define_variable` skips the duplicate), every kinetic-law / observable-sum /
  assignment-rule reference to such a species now resolves to its amount with no
  per-emission AST rewrite. The loader keeps only one residual concern: an event
  assignment whose *target* is itself an hOSU=true V≠1 species writes a stored
  concentration slot, so the assigned amount is divided by `V_c(target)`
  (`_divide_by_target_vc`); for an AssignmentRule target the same `÷V_c` is
  applied at report time (`Simulator._apply_ar_report_map` gained a per-target
  `vdiv`). The codegen C emitter mirrors all of this: observable coefficients
  fold in `V_c`, and an Elementary/Functional rate carries the per-reaction
  `amount_factor = ∏ V_c^mult` over amount_valued reactants (RHS *and* analytical
  sensitivity RHS); `_CODEGEN_VERSION` 9 → 10. This **closes the latent
  multi-compartment hOSU Functional-under-SSA gap** — the
  `non_mass_action_volumetric_species` SSA validation error is removed, and such
  models now simulate amount-correctly under both ODE and SSA. Byte-identical for
  `.net`, V=1 SBML, and every hOSU=false species (the folded factors are all
  `1.0`); the whole-corpus gates stay green (pytest 1080+, ssa-roundtrip 7/7
  byte-identical, DSMTS 38/39 strict @ N=10000, C++ 49/49 + 3/3). Bumps version
  to 0.9.16.
- **hOSU amount restoration is now a single core capability (#75, step 1).**
  A `hasOnlySubstanceUnits=true` species's symbol denotes an *amount*, not the
  stored concentration. This invariant was previously re-implemented in three
  independent places in the SBML loader (the mass-action `hosu_numerator`
  product, the Functional `_wrap_hosu_amounts` AST rewrite, and the linear-AR
  observable-weight reweighting) — the configuration that caused the #70-after
  -#30 regression. The mass-action (Elementary) path is now collapsed into a
  single engine flag `Species::amount_valued`: when set, `compute_rxn_rate`
  (ODE + SSA species factor) and the analytical Jacobian read the species as
  `stored × volume_factor`. The loader simply *selects* the flag per species
  (`add_species(..., amount_valued=hasOnlySubstanceUnits)`) instead of pre-baking
  `×V_c` into the scalar rate. ODE results are byte-identical (the same `∏ V_c`
  reaches the accumulation via the species factor instead of the rate constant);
  SSA is byte-identical for first-order hOSU and uses the physically-correct
  population falling factorial for higher order. Default `amount_valued=false`
  leaves all `.net`, V=1 SBML, and hOSU=false behavior unchanged. The Functional
  and observable-weight paths are unchanged in this step (step 2 folds them in).
  Bumps version to 0.9.15.

### Tests

- **Rate rule on an hOSU=true V≠1 species (#75 follow-up regression guard).**
  `test_sbml_assignment_rule_species_ode.py` pins `dR/dt=-kdeg·R` on an hOSU
  V=2 species to the analytical amount-law decay (rate `kdeg`, not `kdeg·V`),
  cross-checked against RoadRunner. `test_sbml_ssa_cross_compartment.py` adds a
  multi-compartment hOSU=true Functional reaction under SSA (the headline
  "still-uncovered surface" the refactor was meant to close).
- **hOSU Functional/observable amount restoration (#75, step 2).** A
  hOSU=true V≠1 non-mass-action (Functional) reaction — formerly the
  `non_mass_action_volumetric_species` SSA error — now validates clean and
  simulates amount-correctly: `test_sbml_ssa_validation.py` gained an
  ODE/SSA amount-conservation test and an SSA-ensemble-mean-converges-to-the
  -amount-correct-ODE test (a saturating law over a V=4 hOSU species, where
  reading concentration instead of amount would mis-scale the propensity ~2×).
- **hOSU under codegen (#75, step 2).** `test_model_codegen_sensitivity.py`
  gained `TestModelCodegenHosuAmountFactor`: the emitted C carries the
  Elementary `amount_factor` and V_c-folded observable coefficients; the
  codegen RHS trajectory matches the analytical amount-law oracle and the
  ExprTk engine; a Functional law reads its hOSU species through the
  observable shadow under codegen; and the analytical sensitivity RHS matches
  an external finite-difference reference.
- **Oracle-anchored core unit tests via the direct `ModelBuilder` API**
  (`python/tests/test_core_direct_oracles.py`). Build each core primitive that
  the SBML loader's fixes depend on — `fixed`-clamped species read frozen in a
  rate law, rate-rule lowering to a functional synthesis reaction, live
  observable/function refresh in a rate law, per-species volume scaling — with
  **no loader in the loop**, and pin each to a closed-form analytical oracle.
  These defend against a loader fix silently masking a core defect (the loader
  audit behind #74/#75): a regression here is in the core, and no loader
  workaround can make it pass. Plus bedrock integrator anchors (first-order
  decay, reversible equilibrium + conservation, second-order kinetics). 8 tests.

## [0.9.30] - 2026-06-01

### Fixed

- **Variable-volume dilution term for hOSU=false species in rate-rule
  compartments (#86).** For a concentration-valued (hasOnlySubstanceUnits=false)
  species `S` of amount `A = [S]·V` in a compartment whose volume `V(t)` is driven
  by a rate rule, the concentration ODE is `d[S]/dt = (1/V)·dA/dt − [S]·V̇/V`.
  bngsim emitted only the reaction term (the #74 `_vd_<rid>_varvol` divide by
  `V_live`); the dilution term `−[S]·V̇/V` — the concentration change caused by the
  volume itself moving — was missing, so a species in a growing/shrinking
  compartment was integrated as if its compartment were static (both `[S]` and the
  implied amount diverged from RoadRunner; the Michaelis–Menten repro was 247% high
  at t=10). The loader (section 8b) now emits the dilution term as an additive
  Functional reaction `−1·S·V̇_C/C` for every hOSU=false species in a rate-rule
  compartment, covering reaction-driven and pure-dilution (reaction-free) species
  alike (closed form `[S](t)=A/V(t)`). Boundary species are un-fixed so the term
  integrates (amount conserved → concentration dilutes). A new `_varvol_amount_map`
  makes the bare-id amount selector report `conc·V_live`. **Zero corpus impact by
  construction** — no corpus model has a hOSU=false species in a rate-rule
  compartment, so every `_varvol_amount_map` is empty and the path is byte-identical.

## [0.9.29] - 2026-06-01

### Fixed

- **Variable-volume species concentration reported at `amount/V_live(t)` (#85).**
  A species in a rate-rule-driven (variable-volume) compartment reported
  concentration as `amount/V_static` (the compartment size at load) instead of
  `amount/V_live(t)`; MODEL1606100000's Vos-compartment species (Glyin, Hog1,
  Hog1PP, Slt2, Slt2P, Osmin) read ~2e5× too large at the end of the run. The
  integrated *amounts* were already correct (Functional rates divide by the live
  compartment symbol — the #74 `_vd_<rid>_varvol` path), so only the
  amount→concentration reporting kept the stale static volume. Report-time fix (no
  dynamics change): the loader records a `_varvol_conc_map`; `Simulator._stamp`
  rescales the reported concentration by `V_static/V_live(t)` (reading `V_live`
  from the compartment's own promoted-species column), and the `as_roadrunner`
  bare-id selector recovers the amount as `conc·V_live`. Scoped to amount-valued
  (hOSU=true) species, where `stored·V_static == amount` holds and the rescale is
  exact. Empty map ⇒ byte-identical for static, event-resized, unit-volume, and
  .net models.
- **Rate-rule compartment with no `initialAssignment` seeded its promoted species
  to 0** (entangled bug surfaced by the same models). The step-8 promotion path
  looked up `getParameter` only (None for a compartment), so the live volume was
  `g·t` instead of `V0+g·t` and the live-volume divide blew up — the cause of
  bngsim CVODE failures on the other variable-volume models. Promotion now reads
  the resolved compartment size.

## [0.9.28] - 2026-06-01

### Fixed

- **`initialAssignment` tracks an assignment-rule-target dependency with a stale
  `value=` (#73).** MODEL1606100000 (Talemi2016 yeast osmo) loaded
  `ModelValue_19 = +322026` while its own `cin0 = −4807974`, despite the SBML's
  `ModelValue_19 := cin0` — flipping the sign of the boundary species `Osmin`.
  COPASI exports duplicate a quantity as both a raw-valued parameter and an
  assignment-rule target (raw `value=` stored stale), then point another
  parameter's `initialAssignment` at it; the loader's guarded IA-evaluation loop
  read the stale raw value. The loader now re-runs initialAssignments (and the AR
  override) to convergence after the AR override, so each IA target tracks its
  assignment-rule dependency (per SBML an assignment rule holds at t=0). Additive
  and gated on the presence of initialAssignments; models without the
  cross-dependency are byte-identical. Full 1597-model rr_parity ODE re-sweep: zero
  PASS→DIFF, only MODEL1606100000 changed.

## [0.9.27] - 2026-06-01

### Fixed

- **Time-dependent piecewise discontinuities resolved in the ODE integrator
  (#72).** A piecewise assignment rule / rate law that switches on the SBML `time`
  csymbol (drug-dosing windows, scheduled stimuli) puts a discontinuity in the
  RHS. bngsim's CVODE integrates with interpolated output and only did
  root-finding for SBML events, so its adaptive steps could jump clean over a
  narrow pulse. BIOMD0000000879 (Rodrigues2019 chemoimmunotherapy of CLL) has 7
  chemo infusions each only 0.125 t-units wide; bngsim delivered just the t=0 dose
  (`λ·∫Q = 1080` vs the correct 7560), so the cancer escaped to carrying capacity
  (`N→9.6e11`) instead of the immune-controlled branch (~2.5, confirmed by a
  segmented SciPy oracle and RoadRunner). The loader now extracts every `time`
  inequality inside a piecewise rate/assignment expression and registers it as a
  CVODE root, reusing the event root-finding path, so the integrator stops at each
  pulse edge and cannot step over it. Models with no time-dependent piecewise
  register zero triggers and integrate bit-for-bit as before.

## [0.9.26] - 2026-05-31

### Fixed

- **Event-promoted SBML parameter/compartment no longer emitted as a trajectory
  column (#71).** A parameter or compartment mutated only by an event must be
  promoted to a species (the engine applies event assignments by writing species
  slots), but it is not a floating species and RoadRunner does not emit it as a
  trajectory column. bngsim appended it anyway — MODEL1108260014 showed
  `parameter_1` + `compartment_1` as spurious bn-only columns (84 vs RR's 82). New
  `Species::reported` flag (default true): the loader marks event-promoted
  parameters/compartments `reported=false` (rate-rule-promoted parameters stay
  reported — RoadRunner reports those too); the promotion keeps its ODE slot,
  RHS/Jacobian participation, and a same-named observable, but the output layer
  projects it out of `Result.species`/`species_names` and the `.cdat` export.
  Wired only when a model has an unreported species ⇒ .net and all-reported models
  byte-identical; the common-species rr_parity differ is verdict-neutral.

## [0.9.25] - 2026-05-31

### Performance

- **Codegen `.so` cached across `run()`s (#77).** The CVODE simulator
  dlopen/dlsym/dlclose'd the codegen library on every `run()`, adding a flat
  ~0.5 ms fixed per-run overhead with no integration compute to amortize it — so
  the codegen path lost to the ExprTk bytecode path on short-horizon / small
  models where fixed setup dominates. The library handle and resolved
  `bngsim_codegen_rhs` / `_sens_rhs` / `_jac` pointers are now cached on the
  simulator (keyed by `.so` path); repeated runs reuse the mapped library (only a
  changed path triggers a one-time reload), and the user-data struct handed to the
  `.so` is built once per `run()` rather than per callback. The codegen floor now
  ties or beats the ExprTk floor at every measured size (2–40 species; was 1.5–6×
  slower). No numerical change — repeated-run output is byte-identical.

## [0.9.24] - 2026-05-31

### Added

- **Compiled-C analytical Jacobian for codegen models (#76, Task 4).** The codegen
  `.so` now emits `bngsim_codegen_jac`, a C mirror of
  `NetworkModel::fill_dense_analytical_jacobian`; when codegen is active the CVODE
  dense Jacobian dispatch prefers it over the interpreted ExprTk
  `cvode_analytical_dense_jac` (kept as fallback), so a codegen dense ODE run is
  fully compiled — RHS + Jacobian, no ExprTk in the loop.
  `generate_jacobian_from_model` emits all four contribution blocks (Elementary +
  MM closed-form from `codegen_jacobian_plan`; Functional per-species chain rule +
  per-observable product rule reconstructed from `functional_jacobian_context()`
  via the shared sympy core), and declines to the interpreted/FD fallback when the
  analytical Jacobian is incomplete, the model takes the sparse path, or any
  derivative is un-emittable — never wrong C. `_CODEGEN_VERSION` 10→11.

## [0.9.23] - 2026-05-31

### Added

- **Michaelis–Menten (tQSSA) closed-form analytical Jacobian (#76).** MM reactions
  (`rate = kcat·stat·sFree·E/(Km+sFree)`) previously cleared the analytical
  Jacobian for the whole network, forcing CVODE's finite-difference Jacobian. The
  engine now emits `∂rate/∂E` and `∂rate/∂S` in closed form — the chain rule
  through the tQSSA free-substrate root done analytically — via
  `mm_tqssa_derivatives()`, one source of truth shared by the dense and sparse
  callbacks (in the RHS-clamped region `sFree≤0` the derivative is 0). No sympy at
  runtime: the derivative is hand-derived and exact, so like the Elementary closed
  form it carries no per-step self-check. Validation: `mm_tqssa.net` analytical==FD
  to 1.3e-14; 5 corpus MM models pass bng_parity with MM analytical on (0 DIFF).

## [0.9.22] - 2026-05-31

### Added

- **`.net` per-observable analytical Jacobian + mass-action product rule (#76).**
  Rule-based `.net` Functional reactions have `rate = func(observables)·∏reactants`
  (`apply_species_factor=true`); the analytical path previously rejected these and
  routed the whole model to finite differences. It now scatters the full column
  derivative `∂rate/∂x_j = (∂func/∂x_j)·∏R + func·∂(∏R)/∂x_j`, chain-ruling each
  `∂func/∂obs_k` (from the Python symbolic core) through the observable's species
  group and reusing the Elementary product-rule machinery for the mass-action
  species factor. One shared scatter implementation
  (`include/bngsim/functional_jac_scatter.hpp`) serves both the dense path and the
  sparse CVODE callback. Validation: 17/17 functional `.net` models give
  analytical==FD trajectories (≤~1e-6 peak-rel, all self-check-validated); 19
  functional ODE corpus models pass bng_parity (0 DIFF).

## [0.9.21] - 2026-05-31

### Added

- **CVODE nonlinear-convergence-failure counter exposed + analytical-vs-FD
  Jacobian benchmark (#76).** `SolverStats.n_nonlin_conv_fails` (CVODE's
  `CVodeGetNumNonlinSolvConvFails`, previously queried but discarded) is now
  threaded through to Python (`Result.solver_stats`), making the Jacobian's
  robustness benefit — fewer Newton convergence failures on stiff models —
  measurable rather than just per-step cost. New `benchmarks/suites/jacobian` suite
  (per-step timing, correctness, attachment probe, tolerance-convergence, corpus
  robustness sweep). Honest finding: the per-step win is modest (~1.0–1.2×, geomean
  1.05×) on SBML functional models because the large-model FD baseline is already
  colored finite differences (O(n_colors)); the robustness win concentrates on the
  stiffest models, and the decisive speedups are gated on the `.net`
  per-observable follow-up.

## [0.9.20] - 2026-05-31

### Added

- **Analytical Jacobian for Functional / Michaelis–Menten rate laws, on by default
  (#76).** Lifts bngsim's analytical Jacobian from all-Elementary networks to
  models with Functional rate laws, reaching the interpreted (SBML) engine — not
  just codegen — and removing the O(N)-extra-RHS-evals-per-step finite-difference
  Jacobian for every Functional model that attaches. At load time
  `bngsim._jacobian` (sympy) differentiates each Functional rate law w.r.t. the
  observables it depends on, chain-rules through each observable's species group
  into a per-species ExprTk derivative string, and registers them via
  `NetworkModel::set_functional_jacobian`; the integration loop stays pure C++
  (GIL released, never calls Python). The dense/sparse callbacks evaluate the
  registered derivatives at the live state and scatter by net stoichiometry,
  alongside the existing closed-form mass-action contributions.
- **In-C++ FD self-validation gate.** At load, the assembled analytical Jacobian is
  cross-checked against reliability-gated central differences of the engine's own
  RHS (two-step Richardson convergence + catastrophic-cancellation detection, so a
  correct Jacobian on a stiff model is not false-failed by FD noise; plus a
  non-finite-entry guard that rejects a NaN/inf entry at a finite-RHS state). On
  any trustworthy mismatch the model silently keeps the finite-difference Jacobian
  — it never ships a wrong Jacobian.

## [0.9.19] - 2026-05-31

### Fixed

- **Functions evaluated in topological order — latent path-dependent RHS (#76).**
  `evaluate_functions()` wrote each function's value into its bound parameter in
  declaration order; when a function referenced one declared *after* it, the single
  pass read the referenced function's **stale** bound value (left from the previous
  RHS evaluation). Because `compute_derivs()` runs one pass per RHS evaluation, the
  RHS became path-dependent — not a pure function of `(t, y)` — silently corrupting
  integration for any SBML model whose assignment-rule declaration order is not a
  topological (dependency) order. `ModelBuilder` now topologically sorts the
  bindings (Kahn, seeding ready nodes in ascending index so an already-ordered
  model is unchanged); SBML rule graphs are acyclic, any residual cycle falls back
  to declaration order. `.net` byte-identical (BNG emits functions in dependency
  order, 0 of 2272 blocks forward-reference). MODEL8684444027 (failed CVODE at
  t=0.5) and BIOMD0000000268 (failed at t=20) now integrate and match RoadRunner to
  ~1e-6.

## [0.9.18] - 2026-05-30

### Fixed

- **Compartment-resize event preserves species amounts (#74).** An SBML event that
  changes a compartment's size must preserve the contained species' amounts and
  recompute their concentrations; bngsim stores concentrations, so it left them
  unchanged (silently scaling amounts by the volume ratio) — BIOMD0000000338's
  `dilution_event` tripled `compartment_1` and bngsim held `Pk` at ~450 where
  RoadRunner correctly drops it to 150 = 450/3. Two coupled loader fixes: (1)
  inject a per-species rescale `s := s·V_old/V_new` into the resizing event for
  every non-hOSU, non-AR-target species in that compartment (pre-fire-evaluated via
  simultaneous-event semantics; hOSU species are skipped — the engine already reads
  them as `amount = stored·V_c`); (2) a Functional reaction whose species live in a
  variable-volume compartment now divides its rate by the **live** compartment
  symbol — the `common_vs==1.0` emission previously reused the raw
  `compartment·f(...)` law, leaking the post-resize volume into `d[conc]/dt` and
  running the cascade 3× too fast (which is what actually made the stiff BIOMD338
  unintegrable at the parity tol). Mass action is unaffected (the compartment
  cancels analytically). A compartment resized by an event is now SSA-rejected.

## [0.9.17] - 2026-05-30

### Fixed

- **Rate-rule accumulation divided by `V_c` for hOSU targets (#75 follow-up).** A
  `rateRule` on a hasOnlySubstanceUnits=true species in a non-unit-volume
  compartment came out wrong by a factor of the compartment volume: the #75 step-2
  observable-shadow change made the species read as its amount everywhere, but the
  rate-rule lowering (a Functional synthesis reaction) was not dividing the storage
  accumulation by `V_c(target)`, so `dR/dt = −k·R` decayed at rate `k·V_c` instead
  of `k`. The lowered synthesis reaction is now marked `per_species_volume_scaling`
  when the target is amount-valued (dividing the ODE accumulation by `V_c(target)`
  while leaving the SSA propensity in amount/time). Byte-identical for hOSU=false
  targets and `V_c=1`. BIOMD0000000353 (a rate rule on an hOSU species in a
  3.5e-13 L compartment, previously off by ~1e9) now matches RoadRunner to 2.5e-6.

## [0.9.16] - 2026-05-30

### Changed

- **hOSU amount restoration folded into the `Species::amount_valued` engine flag —
  step 2 (#75).** The two remaining loader-side hOSU amount-restoration paths now
  route through the engine flag (step 1 covered the mass-action species factor), so
  a hasOnlySubstanceUnits=true symbol reads as its amount (`stored × volume_factor`)
  at every evaluation site via the same-named-observable shadow:
  `NetworkModel::update_observables` applies the volume factor for amount-valued
  species (the single read site). Removes the `_wrap_hosu_amounts` AST rewrite, the
  `_amt_<rid>` law, the hOSU amount-factor observable weights, and the
  `non_mass_action_volumetric_species` SSA gate; linear-AR observable entries are
  now the literal rule coefficients. Byte-identical for .net / V=1 / hOSU=false.
  `_CODEGEN_VERSION` 9→10.

## [0.9.15] - 2026-05-30

### Changed

- **hOSU amount restoration collapsed into one engine capability — step 1 (#75).**
  A hasOnlySubstanceUnits=true species's symbol denotes an *amount*, not the stored
  concentration; this invariant had been re-implemented in three independent loader
  sites (the configuration behind the #70-after-#30 regression). Step 1 collapses
  the mass-action (Elementary) path into a single engine flag
  `Species::amount_valued`: when set, the species participates in a reaction's
  species factor by its amount (`stored × volume_factor`) rather than the stored
  concentration, wired through every Elementary rate-evaluation path (ODE + SSA,
  the latter sharing it via `compute_propensity`) and the analytical Jacobian
  (`ReactionTerms::amount_factor`, folded into `k_sf` so J stays consistent with
  the amount-reading RHS). The loader now selects the flag
  (`add_species amount_valued=hasOnlySubstanceUnits`) instead of folding the
  per-reaction `V_c` product into the scalar rate. Defaults (`amount_valued` false,
  `volume_factor` 1.0) keep .net models, V=1 SBML, and every hOSU=false species
  byte-identical.

## [0.9.14] - 2026-05-29

### Changed

- **`rr_parity` ODE adapter now compares boundary species even when RoadRunner
  exposes no floating species.** `_rr_common.rr_ode` previously only *rewrote*
  boundary-species columns already present in RR's default `timeCourseSelections`;
  for a model with 0 floating species whose only dynamic species are boundary
  (BIOMD0000000567 — `A` constant, `B` an assignmentRule target, both
  `boundaryCondition=true`, so RR's default selection is just `['time']`) the
  boundary species were dropped, leaving the species sets disjoint. It now
  appends any boundary species missing from the default as `[id]` concentration.
  BIOMD0000000567 was a spurious `disjoint` DIFF; it is now a genuine numeric
  compare and PASSes (bngsim, RoadRunner, and the closed-form assignment rule all
  agree). Suite-only change; no engine behavior change.

### Triage

- **3 remaining `rr_parity` ODE DIFFs dispositioned** (`dev/notes/rr_parity_triage.md`):
  - BIOMD0000000567 → PASS via the adapter fix above.
  - MODEL0912940495 (Demir1999 sinoatrial-node pacemaker) →
    `overrides.py` KNOWN_ARTIFACT: oscillator phase drift. The gating variables
    drift in spike timing (worst `d_L`, relmax 0.354) but the amplitude ranges
    are engine-identical (both find the same limit cycle); same class as
    MODEL0406553884. No engine attributable.
  - BIOMD0000000338 (Wajima2009 coagulation) → **honest DIFF, NOT allow-listed**,
    filed as #74. A `dilution_event` (`compartment_1 *= 3`) exposes a real bngsim
    bug: bngsim preserves species *concentration* across the compartment-size
    change (tripling amounts) instead of preserving *amounts* (SBML semantics);
    RoadRunner is correct (`Pk` 450 → 150 = 450/3). Allow-listing would mask the
    defect.

## [0.9.13] - 2026-05-29

### Changed

- **Shared parity differ (`parity_checks/_core/differ.py`) now applies a
  per-species peak-relative significance gate.** `deterministic_verdict` judged
  the relative divergence per cell, so a species that peaked at the file scale
  then decayed to a near-zero tail and disagreed on that tail was flagged even
  though it is below the model's dynamic range. The verdict now mirrors the
  per-species gate the rr_parity triage trusts
  (`dev/investigations/rr_check.py`): each column is normalized by its own
  peak-over-time (`reld_peak = |a-b| / col_peak`), and a new `SIGNIF_FLOOR =
  1e-3` marks columns carrying at least that fraction of the file peak. A column
  is a genuine ("real") divergence only if it both diverges past
  `HARD_REL_CEILING` at its peak scale **and** is within the dynamic range;
  failing cells in any non-real column are forgiven before the fail-fraction
  budget. The change only ever *loosens* the verdict, so it cannot introduce a
  PASS→DIFF regression. Invariants preserved: a one-side-non-finite cell (a
  blow-up on one engine) and an absolute-ceiling breach are never forgiven by
  the gate; both-NaN is a zero-diff pass; the fail-fraction budget and absolute
  ceiling are retained (the budget now forgives soft cells within a real
  column). Differ-only — no bngsim engine behavior changes.
- **rr_parity ODE sweep: 1344→1403 PASS, 65→6 DIFF, 0 PASS→DIFF.** The gate
  closes 54 dynamic-range "artifact" DIFFs the 2026-05-29 peak-relative
  re-screen flagged; the remaining 11 were then triaged to closure:
  - **4 tolerance artifacts** (BIOMD374/876/951, MODEL0913003363) — both engines
    diverge at the shared sweep tol (1e-9/1e-12) but converge at 1e-10/1e-16 —
    resolved with per-model `TOL_OVERRIDES` (the divergence is a property of the
    ill-conditioned IVP at the loose default, applied identically to both
    engines) → PASS.
  - **BIOMD375** (stiff, unconverged: bngsim hits the CVODE step cap, residual
    1.58% peak-relative — below the gate) held non-PASS by a `TOL_OVERRIDES`
    entry (1e-11/1e-16 → RoadRunner raises → EXCEPTION), since the bare gate
    would silently pass an unconverged solve.
  - **6 remaining DIFFs are genuine divergences, left honestly flagged (not
    allow-listed):** BIOMD567 (disjoint — RR exposes 0 floating species),
    BIOMD338 (28 real-column divergence), MODEL0912940495 (real divergence +
    stiffness), and three the 2026-05-29 re-screen had mislabeled "artifact" but
    investigation showed are real: BIOMD879 (bngsim `N`→9.6e11 vs RoadRunner
    `N`→119, an instability/bifurcation split — not oscillator phase-drift),
    MODEL1606100000 (RoadRunner `Slt2Signal` blows up to 1.6e7 while bngsim stays
    bounded ~0.48), and MODEL1108260014 (bngsim emits `compartment_1`/`parameter_1`
    as trajectory columns, inflating rr_check's global-peak denominator, plus a
    minor real `species_57` divergence). The gate is *more* correct than the
    re-screen here — it keeps real divergences flagged rather than masking them.

### Known / follow-up

- MODEL1108260014 surfaced a likely loader quirk: bngsim emits a compartment
  (`compartment_1`) and a parameter (`parameter_1`) as trajectory columns. Worth
  a separate look (it perturbs the parity common-species alignment).

### Note

- The bng_parity suite gates on its own `parity_diff.py` copy, so its 895/895
  golden is unaffected by this `_core/differ.py` change (the unification is the
  separate #69 migration).

## [0.9.12] - 2026-05-29

### Fixed

- **Codegen ODE RHS now honors ExprTk's double-division semantics for rational
  constants.** The SBML loader auto-enables the C-codegen RHS at ≥256 species,
  and the ExprTk→C translator passed numeric literals through verbatim — so a
  rate law's `(1/2)` (which ExprTk evaluates as `0.5`) compiled to C integer
  division `1/2 == 0`, silently zeroing any rate carrying a rational constant.
  Surfaced by the `rr_parity` SBML suite on MODEL1112100000 (1012-species
  WUSCHEL model): every `Wus_*` synthesis used a `Sigma` sigmoid whose leading
  `(1/2)` codegen'd to `0`, so all `Wus` species froze at their initial value
  under the codegen RHS while the ExprTk RHS and RoadRunner grew them. The fix
  float-ifies integer literals (`1` → `1.0`) before identifier substitution
  introduces array subscripts, so `p[0]`/`y[5]` indices stay integer and
  scientific notation (`2.5e-3`) survives. `_CODEGEN_VERSION` 8 → 9.
- **`initialAssignment` evaluation reads `hasOnlySubstanceUnits=true` species as
  amounts.** The IA/`assignmentRule`-at-t=0 evaluation context seeded every
  species symbol with its *concentration*, but a hOSU=true species's MathML
  symbol denotes its *amount*. An IA that divides a referenced hOSU species by
  its compartment (the SBML idiom for amount→concentration) therefore read
  `conc/V` instead of `amount/V` — off by `1/V`, a ~1e10 blow-up at tiny `V`.
  Surfaced on BIOMD0000000547 (12 hOSU species, V∈[5e-14, 5e-11]): `species_14`
  loaded at 7.27e6 vs RoadRunner's 3.16e-4 (the value the SBML itself records as
  the curated `Metabolite_6`). The fix seeds hOSU species with their amount;
  V=1 / hOSU=false models are byte-identical. Species carrying neither an
  initial concentration nor amount stay absent from the context until their IA
  resolves (so a rule like `(GE1/Gss)^GPRG` never evaluates `0^-2.79` on a
  placeholder before `GE1`'s IA fires — MODEL1112110004).

This release closes the last of the 28 REAL bngsim-vs-RoadRunner ODE
divergences from the 2026-05-29 `rr_parity` triage. Full ODE corpus sweep:
**1284→1341 PASS, 97→65 DIFF, 40→12 TIMEOUT, 0 PASS→DIFF regressions** (35
DIFF→PASS numeric fixes plus 24 TIMEOUT→PASS from the codegen rational fix
unfreezing large WUSCHEL-family models); 7 third-oracle-attributed non-bngsim
divergences (1 RoadRunner bug, 1 invalid SBML, 4 long-horizon oscillator phase
drifts, 1 unstable IVP) are allow-listed as `KNOWN_ARTIFACT` in
`parity_checks/rr_parity/overrides.py`. See
`dev/notes/rr_parity_diff_resolution.md`.

## [0.9.11] - 2026-05-28

### Fixed

- **`hasOnlySubstanceUnits=true` species in *multi-compartment* /
  cross-compartment reactions now integrate the amount law** (#70). The v0.9.10
  fix covered only the single-compartment Elementary path; a reaction whose
  species span two compartments with different `V_c` fails mass-action
  classification (and the reversible splitter), so it is emitted as a
  Functional. The §9 emission collapsed every hOSU=true species's volume factor
  to `1.0`, so an all-hOSU cross-compartment reaction took the unified path with
  `common_vs=1` — no `/V_c` divide — and read hOSU V≠1 species as concentration
  where the literal SBML MathML wants an amount. Surfaced by the `rr_parity`
  SBML suite on BIOMD0000000019 (Schoeberl EGFR; all 100 species hOSU=true,
  `c3=4.3e-6`): `x13` was 1.0 vs RoadRunner's 2.36e5 (off by exactly
  `1/V_c3`), `x15` 729 vs 1.70e8. The fix (1) makes the Functional emission's
  per-species volume factor always `V_c` (so cross-compartment reactions divide
  each species by its own `V_c` and single-compartment hOSU V≠1 ones divide
  once), (2) rewrites the kinetic law so each hOSU=true V≠1 species reference
  `s → s·V_c(s)` restores the amount, and (3) applies the same restoration to
  linear `AssignmentRule → observable` weights (fixing `EGF_EGFR_act`, a
  cross-compartment hOSU species sum). hOSU=false, V=1, and single-compartment
  cases stay byte-identical. Full ODE corpus parity sweep:
  1300→1307 PASS / 109→102 DIFF, 0 regressions (7 DIFF→PASS, all the same hOSU
  multi-compartment class); DSMTS 38/39 and the SSA roundtrip unchanged. SSA on
  these reactions remains gated by `non_mass_action_volumetric_species` (the fix
  is ODE-path only). Regression tests: `test_hosu_true_cross_compartment_*` in
  `python/tests/test_sbml_assignment_rule_species_ode.py`.

## [0.9.10] - 2026-05-24

### Fixed

- **`hasOnlySubstanceUnits=true` species in a mass-action law now integrate the
  amount law in a V≠1 compartment** (#30, ODE-level follow-up; the deferred
  "latent Phase-2.7" case). An hOSU=true species's kinetic-law symbol is its
  *amount*, but bngsim stores every species as `amount/V` (concentration) and
  fed that stored value into the Elementary reactant factor, accumulating ±rate
  with no compartment factor. The per-reaction rate was therefore off by
  `∏_{i hOSU in law} V_i / V_X` (= `V^(order−1)` for an all-hOSU single-V
  reaction: `×V` bimolecular, `1` unimolecular, `/V` synthesis). For
  MODEL1102210001 (all 119 species hOSU=true in a V=1e-12 compartment) the
  bimolecular inflation was ~1e12 and Egfr emptied while libRoadRunner held
  flat. The mass-action classifier (`_classify_mass_action_ast`) now applies the
  `/V_X` storage conversion uniformly (regardless of hOSU) and restores each
  hOSU reactant's amount via a `∏_{i hOSU in law} V_c(i)^mult_i` numerator, so
  `sf = numeric_const · kl_volume_product · ∏_{i hOSU} V_c(i)^mult / V_X`. This
  also fixes the matching SSA propensity: `ssa_volume_factor` stays `V_c`, and
  `sf · V_c` nets to the `∏amount_i` conversion, so the propensity is
  `k·∏amount_i` for both hOSU and non-hOSU reactants with no separate change.
  Closes the lone remaining ODE-level bngsim-vs-libRoadRunner divergence from
  the #30 BioModels work. **Invariant:** `V_c=1` and `hOSU=false` reactions are
  byte-for-byte unchanged (verified against the DSMTS strict suite 38/39, the
  `.net` SSA roundtrip 7/7, and the V≠1 `hOSU=false` corpus); the only behavior
  delta is hOSU=true V≠1. The non-mass-action Functional path (`hOSU` refs not
  yet amount-substituted) remains a separately-gated latent case.

## [0.9.9] - 2026-05-24

### Fixed

- **AssignmentRule-target species in a kinetic law now read their live rule
  value, not a frozen initial value** (#30, ODE-level follow-up). A reaction
  whose rate references an `AssignmentRule`-target species — typically as a
  modifier, e.g. BIOMD0000000104's `reaction_1 = k2·species_1·species_3` with
  `species_3 = species_5 − species_2` — was classified as mass-action and folded
  that species into the Elementary reactant factor, which reads the species's
  *frozen* `conc[]` slot (assignment-rule targets are emitted `fixed`). The
  reaction therefore saw `species_3 ≡ species_3(0)` for all time and the
  downstream cascade froze. The mass-action classifier now refuses any law whose
  species leaf is an assignment-rule target, routing the reaction to the
  Functional path where the name binds to the rule's live value.
- **`Result.species` reports AssignmentRule-target species at their live rule
  value** instead of the frozen `fixed` slot (BIOMD0000000016 `Pt = ΣP_i`,
  BIOMD0000000199 `FeIII_t`, BIOMD0000000312 `S = floor(time/tau)`). The value
  is taken from the same-named observable (linear-on-species rules) or
  expression (everything else), matching libRoadRunner, which re-evaluates the
  rule each step. Affects only the *reporting* of these species; the dynamics of
  models whose reactions don't read them were already correct.

These resolve 11 of the 12 ODE-level bngsim-vs-libRoadRunner divergences from
the #30 BioModels cross-engine screen; 7 of the original 12 turned out to be a
near-zero-stiff-value artifact of the screen's relative-error metric, not bugs.
The lone genuine remainder, MODEL1102210001 (all-`hasOnlySubstanceUnits=true`
species in a V=1e-12 compartment), is the latent Phase-2.7 amount-vs-storage
case and is deferred — see `dev/notes/SBML_VS_ROADRUNNER.md`.

## [0.9.8] - 2026-05-24

### Fixed

- **SSA now refreshes time-dependent rate laws instead of freezing them at
  t=0** (#30, RC#2). A reaction whose rate reads an assignment rule that
  depends on `time()` — e.g. a synthesis flux gated by
  `Mpl = A·exp(c·t) − B·exp(d·t)` (BIOMD0000001040 / BIOMD0000001026) — was
  frozen at its t=0 value under SSA, because the direct method holds
  propensities constant between fires and the dependency graph only refreshes a
  propensity when one of its *species* changes. With every t=0 propensity zero
  (`Mpl(0)=0`), the loop wedged in its `a0==0` fast-forward and the trajectory
  flat-lined at the initial state. The SSA now detects time-dependence (probing
  whether any function value moves when only time advances) and switches to
  **piecewise-constant sub-stepping**: each step is capped at
  `dt_max = (t_end − t_start)/1000` and all functions + propensities are
  re-evaluated at the new time, the same way the ODE RHS refreshes assignment
  rules on every call. Discard-and-resample at the cap is exact for the
  piecewise-constant rate by exponential memorylessness. Models with no
  time-dependent functions keep the original O(k log N) dependency-graph fast
  path unchanged. The SSA mean now tracks the ODE (and libRoadRunner's ODE)
  within stochastic tolerance; in the process this surfaced that RR's own
  `gillespie` lags the rule by one output interval (roadrunner#1317), so the
  cross-engine screen still flags these models — but as an RR defect, not a
  bngsim one. See `dev/notes/SBML_VS_ROADRUNNER.md` (RC#2).

## [0.9.7] - 2026-05-24

### Fixed

- **NFsim/RuleMonkey models that declare a parameter / observable / function
  named like an ExprTk built-in (e.g. `frac`, `min`) now run** (#64). ExprTk
  reserves a large built-in name set (`frac`, `min`, `max`, `sum`, `avg`,
  `mod`, `root`, `hypot`, `erf`, …) and rejects `add_variable()` for any of
  them; muParser — what legacy BNG2.pl→NFsim used — reserved far fewer names,
  so a model such as `V1988a_endemic_infection` with a parameter `frac` used in
  `frac * infection_force()` ran upstream but regressed in bngsim's ExprTk
  shim. The symbol silently never registered, so operator-form use compiled to
  `ERR029` (a hard failure, surfaced clearly by #63) while call-form use
  (`frac(x)`) silently resolved to the built-in and ignored the declared
  symbol. The vendored `nfsim_funcparser.h` `mu::Parser` now registers
  colliding model symbols under an internal `r_<name>` key and rewrites only
  those references in compiled expressions, leaving genuine built-in calls
  (`frac(x)` in a model that never declared `frac`) untouched — the direct
  NFsim analogue of the ODE engine's reserved-word handling in
  `expression.cpp` (#27). A declared symbol that is *also* used in call form is
  genuinely ambiguous in a single flat namespace and now raises a clear,
  deterministic load-time error rather than silently picking the built-in. The
  reserved set is read directly from the vendored `exprtk.hpp`, so an ExprTk
  snapshot bump (e.g. RuleWorld/nfsim#80) cannot drift the mangling
  assumptions. RuleMonkey already inherited this behavior via the shared
  `ExprTkEvaluator`; this closes the NFsim gap. New NFsim carry patch
  `bngsim/carry-reserved-symbol-remap`.
- **Reserved-symbol call-form ambiguity now errors identically on both ExprTk
  engines** (#64 follow-up). The ODE/RuleMonkey `ExprTkEvaluator` previously
  rewrote a declared-and-called reserved symbol (`frac(x)` where `frac` is also
  a declared parameter) to `r_frac(x)` and let ExprTk emit a cryptic "not a
  function" error; it now raises the same clear, deterministic message the
  NFsim `mu::Parser` shim does. Behavior is unchanged for the legitimate cases
  (operator-form `frac * x`, and unshadowed built-in `frac(x)`).
- **`mratio()` now works in NFsim function / rate-law expressions** (#64
  follow-up). `mratio(a,b,z)` — the confluent hypergeometric ratio
  `M(a+1,b+1,z)/M(a,b,z)`, a BNGL built-in (BNG2.pl `Expression.pm`; see #42) —
  is registered by the ODE/SSA engine and RuleMonkey but was missing from the
  NFsim ExprTk shim, so a model using it ran on the network engines yet failed
  under NFsim with an unknown-function error. The modified-Lentz
  continued-fraction implementation is ported verbatim from `expression.cpp`
  into the vendored `nfsim_funcparser.h`, and `mratio` is added to the shim's
  reserved-alias set. New NFsim carry patch `bngsim/carry-mratio-builtin`.

### Internal

- New `test_exprtk_reserved_consistency.py` drift guard asserts the two
  vendored ExprTk snapshots (`third_party/exprtk` for the ODE/RuleMonkey engine
  and `third_party/nfsim/.../exprtk` for NFsim) reserve exactly the same
  `reserved_words[]` / `reserved_symbols[]`. Independently bumping one (e.g. via
  RuleWorld/nfsim#80) without the other would make the two engines disagree
  about which model-symbol names collide with a built-in; this test fails on
  that drift.

## [0.9.6] - 2026-05-24

### Fixed

- **NFsim `initialize()`/`run()` setup failures now surface NFsim's underlying
  diagnostic instead of a bare `Quitting`** (#63). When the NFsim backend aborts
  *after* the BNG XML parses successfully — during `prepareForSimulation()` — the
  vendored core prints the real reason to `cout`/`cerr` and then throws a bare
  `"Quitting"`. `prepare_system()` ran that under a `StreamSuppressor` whose
  captured output was discarded on the exception, so the only thing reaching the
  Python `SimulationError` was `"NFsim initialization failed: Quitting"` — and
  process-level fd redirection couldn't recover it either, because the suppressor
  swaps C++ stream `rdbuf`s rather than touching fds 1/2. `prepare_system()` now
  wraps setup in a try/catch and appends `summarize_nfsim_log(...)` on failure,
  mirroring the XML-load path (`create_system()`). Callers now see e.g. the
  `ExprTk compilation failed for '…': ERR029 …` line that pinpoints the cause.
  Shared by both the one-shot `run()` and the session `initialize()` paths.

## [0.9.5] - 2026-05-24

### Fixed

- **Internal network-rewrite observables no longer leak into `.gdat` output or
  the `Result` observable API** (#61). When a `.net` reaction uses a legacy
  functional/saturation rate law (`Sat`, `Hill`), the loader rewrites it to an
  explicit Functional rate law and synthesizes a per-reactant single-species
  observable (`__bngsim_net_rewrite_obs_r<N>_<p>`) so the rewritten expression
  can reference the reactant count. These scaffolding observables were being
  emitted as extra trailing columns that `run_network`/BNG2.pl never produce,
  breaking `.gdat` column-set parity on any model with a `Sat`/`Hill` rule (the
  count of leaked columns tracked the number of such rules). The simulated
  values were always correct — `.cdat` and the user observable columns matched
  the subprocess stack — so this was purely an output-schema leak, not a
  numerics bug. The observables still live in the model (the rate law needs
  them) but are now filtered from every user-facing surface — `to_gdat`,
  `Result.observable_names`, `Result.observable_data`, and `Result.n_observables`
  — via the reserved `__bngsim_` namespace, mirroring how the auto-generated
  `_rateLawN` function columns are filtered. Found while screening RuleHub
  candidate models for the PyBioNetGen parity suite.

## [0.9.4] - 2026-05-23

### Fixed

- **NFsim reactant-selector teardown double-free** (#60, follow-up to #34).
  `TransformationSet::~TransformationSet()` deleted the `readPattern`-generated
  `TemplateMolecules` held in each `ReactantFilter`'s `parsedTemplates`, but
  `MoleculeType` already frees them on `System` teardown — a second free of the
  same objects. Intermittent on a single selector session, deterministic (12/12
  SIGABRT/SIGSEGV) once two selector-bearing simulations share a process; would
  crash any library consumer running many sessions per process (e.g. parameter
  fitting) on a selector model. Landed as vendor carry patch 0019
  (`bngsim/carry-selector-teardown-uaf`, candidate for upstream
  [RuleWorld/nfsim#82](https://github.com/RuleWorld/nfsim) PR #23); re-vendored
  at source `d8dc7d60`, base `43f635dd` unchanged.

### Added

- **Reactant-selector regression fixtures with an analytical oracle** (#60).
  `include_reactants(1,A(p~P))` / `exclude_reactants(1,A(p~U))` with `k_phos=0`
  means no `A` ever qualifies, so `AB_total` stays exactly `0` iff the selector
  is enforced, while the ungated control binds freely (`AB_total>0`) — proving
  the gated zero is the selector working rather than a brittle snapshot.
  `tests/data/nfsim/{include,exclude}_reactants_*` plus
  `test_nfsim_selectors.py`; a stale-vendor refresh that drops selector handling
  now fails loudly. The products-side selector contract keeps NFsim's existing
  hard abort (silently-incorrect-result guard), pinned by a test.

## [0.9.3] - 2026-05-23

### Fixed

- **Codegen C-compile timeout is configurable and no longer aborts large
  models at 60 s** (#37). `compile_rhs` previously hard-coded
  `subprocess.run(..., timeout=60)`; multi-MB flat RHS sources from large
  reaction networks (e.g. Kozer-EGFR at 913 species / 11,918 reactions →
  4.6 MB C) take minutes to compile at `-O3`, so the compile was killed and
  the caller silently fell back to the slower interpreted ODE RHS. Now:
  - default timeout raised 60 → 600 s, overridable via
    `BNGSIM_CODEGEN_TIMEOUT` (seconds; `0` disables it);
  - sources over ~1 MB compile at `-O1` instead of `-O3` (negligible runtime
    cost for a single flat arithmetic function), overridable via
    `BNGSIM_CODEGEN_OPT` (integer level `0`–`3`, or `high`/`low`);
  - a compile timeout now raises a `RuntimeError` naming
    `BNGSIM_CODEGEN_TIMEOUT` instead of surfacing as a generic failure.

### Changed

- **Codegen `.so` installation is now atomic.** `compile_rhs` compiles to a
  process-unique temp path and `os.replace()`s it into the hash-named cache
  file, so concurrent Dask workers racing to compile the same `model_hash`
  can no longer read a half-written `.c` or load a partially-linked `.so`.

## [0.9.2] - 2026-05-23

### Fixed

- **RuleMonkey exact-species methods are now component-order-insensitive**
  (resolves the 0.9.1 "Known issue"). Vendored RuleMonkey bumped 3.2.0
  (`13e9f63`) → 3.2.1 (`0f70112`), pulling in richardposner/RuleMonkey#13
  (the engine fix, PR #14) plus #15 (re-sync of RuleMonkey's standalone
  `bngsim_expr` copy to BNGsim's #42 Mratio rewrite, required to clear the
  `vendor_rulemonkey.py` ExprTk-drift guard). The engine now canonicalizes
  species-pattern component order on the match path, so
  `RuleMonkeySession.get_species_count` / `remove_species` /
  `set_species_count` resolve a non-canonical-order pattern (`X(p~0,y)`) to
  the same species as the canonical order (`X(y,p~0)`) — matching NFsim and
  removing the `set_species_count` overshoot. Regression tests in
  `test_rulemonkey.py::TestRuleMonkeySessionPatternOrder` (formerly a strict
  `xfail`).

## [0.9.1] - 2026-05-23

### Added

- **`RuleMonkeySession` full session-API parity with `NfsimSession`**
  (issue #38). The vendored RuleMonkey 3.2.0 engine already implemented these
  (RuleMonkey#9); this release binds them through to Python:
  - **`save_species(path)`** — writes the live pool to a BNG-format
    `.species` file (graph-isomorphism dedup, `readNFspecies`-compatible),
    the exact peer of `NfsimSession.save_species`. This is the load-bearing
    one for PyBioNetGen: its NF backend writeback already probes
    `getattr(session, "save_species", None)`, so multi-segment
    `method=>"rm"` runs now get `get_final_state` continuation across
    `saveConcentrations`/`resetConcentrations` segments with no PyBioNetGen
    change.
  - **`get_species_count` / `add_species` / `remove_species` /
    `set_species_count`** — exact, fully-specified, connected BNGL
    species-pattern queries and mutations. `set_species_count` is probed via
    `getattr` by PyBioNetGen's `_apply_nfsim_concentration_changes`.
  - **`evaluate(expr, overrides=None)`** — evaluates a BNG expression against
    the live session (parameters, observables, global functions, clock `t`).
    Unlike `NfsimSession.evaluate`, this requires an **initialized** session
    (the RM engine resolves against the live pool).
  - **`save_state(path)` / `load_state(path)`** — binary in-process session
    snapshot/restore (RuleMonkey is *ahead* of NFsim here; NFsim has no
    equivalent). `seed` reads back `None` after `load_state` (not recoverable
    from a snapshot). For BNG2.pl-driven hosts, prefer `save_species`.

### Known issues

- RuleMonkey's exact-species pattern matcher is **component-order-sensitive**,
  unlike NFsim's (which canonicalizes): only RuleMonkey's own canonical
  component order matches a live species; a semantically identical but
  differently-ordered pattern (e.g. `X(p~0,y)` vs `X(y,p~0)`) silently
  returns 0. Because the `add`/`set` path *does* canonicalize while the
  `get`/match path does not, `set_species_count` with a non-canonical order
  diffs against a wrong baseline and lands on the wrong final count. This is
  an upstream RuleMonkey engine concern (the bngsim binding forwards
  faithfully); tracked by an xfail in `test_rulemonkey.py`
  (`TestRuleMonkeySessionPatternOrder`). Hosts should pass species patterns
  in RuleMonkey's canonical component order until the engine canonicalizes on
  the match path.

## [0.9.0] - 2026-05-23

### Changed

- **`.gdat`/`.scan` function-column output is now method-independent**
  (issue #58). bngsim emits one convention for every simulation method
  (ode/ssa/psa/nf/rm):
  - **Function headers are always bare** — the BNG2.pl `()` suffix is never
    written (`kf_BSA`, not `kf_BSA()`). This **reverses the per-method `()`
    behaviour #53 introduced in 0.7.0** for the NFsim path. `Result.to_gdat`
    headers, `Result.gdat_expression_names`, and the C-extension
    `gdat_expression_names` are all bare now; `gdat_expression_names` is
    consequently identical to `expression_names` and is retained only as an
    intent-revealing alias for consumers assembling file headers.
  - **Auto-generated `_rateLaw<digits>` columns are omitted by default** and
    opted in with the new writer flag `print_rate_laws` (complementing the
    existing `print_functions`). `print_functions=True` appends the
    user-named functions; `print_rate_laws=True` additionally appends the
    synthetic rate-law columns (still bare). Both default `False`.
  - Because there is a single `Result::to_gdat`, the `.gdat` header schema is
    now byte-identical across methods for a given model. (bngsim does not
    write `.scan` itself — consumers assemble `.scan` headers from
    `gdat_expression_names`.)

### Added

- **`Result.raw_expression_names` / `raw_expressions` / `raw_n_expressions`**
  on the Python `Result` (issue #58). These recover the internal
  `_rateLaw<digits>` rate-law columns that the default `expression_*` view
  filters out — the actual #58 fix is "keep all function data in memory and
  filter only at the writer, by flag." The raw columns are available on a
  freshly-simulated `Result`; a `Result` loaded from HDF5 (or assembled from
  raw arrays) carries only the filtered, bare set, so `raw_expression_*`
  there equals `expression_*`.

### Migration

- Tooling that depended on the NFsim `.gdat` `()` decoration introduced in
  0.7.0 (#53), or that filtered `_rateLaw` columns from bngsim's write path
  itself, can drop that code: bngsim now writes bare headers and omits
  `_rateLaw` columns by default for every method. To get the synthetic
  rate-law columns, pass `print_rate_laws=True` to `to_gdat` (or read
  `raw_expression_*`).

## [0.8.0] - 2026-05-23

### Fixed

- **NFsim over-assembly on multivalent ring-closure models** (issue #57).
  `TransformationSet::checkMolecularity` enforced product-side molecularity
  for unimolecular unbinding rules by testing each deleted bond in isolation,
  which wrongly blocked dissociations that delete several bonds at once to
  open a cyclic complex (e.g. a symmetric two-bond homodimer splitting into
  two monomers). The reverse reaction never fired, the complex became a
  kinetic trap, and the system over-assembled relative to the network ODE.
  The check now excludes the full set of bonds the rule deletes before
  testing connectivity. Single-bond ring dissociations that genuinely violate
  product molecularity (issues #54/#55) remain blocked. **Behavioral change:**
  NFsim trajectories for affected models (notably ones with `<->`
  ring-closure rules) shift toward the network-ODE result.
- **Inflated `Species` observable counts when same-complex binding is
  allowed** (issue #57). `System.useComplex` (complex tracking) was derived
  solely from `block_same_complex_binding`; `Species`-typed observables are
  tallied by iterating complexes, so with `block_same_complex_binding=False`
  they were counted without complex tracking and reported wildly inflated
  values. The loader now enables complex tracking whenever the model declares
  a `Species` observable, independently of the binding policy, so
  `block_same_complex_binding=False` counts correctly and matches BNG2.pl's
  `complex=>1` semantics. **Behavioral change:** `Species` observable values
  under `block_same_complex_binding=False`.

## [0.7.0] - 2026-05-22

### Added

- **In-process NFsim save/restore concentrations** (issue #52).
  `NfsimSession` gains `save_concentrations()`,
  `restore_concentrations()`, and `has_saved_concentrations()`, mirroring
  the BNG `saveConcentrations()` / `resetConcentrations()` actions. This
  snapshots the live agent population (counts, component states, and
  bonds) and rewinds to it without touching disk — useful for
  equilibrate → snapshot → perturb → restore workflows. (For
  out-of-process state threading, `save_species()` is still the right
  tool.) Exposed in C++ as `NfsimSimulator::save_concentrations` /
  `restore_concentrations` / `has_saved_concentrations`.

  Restoring previously **segfaulted** the host process inside NFsim's
  `SystemSnapshot::restore()`, which is why the wrapper had not been
  landed. Root cause was a use-after-free / double-free in
  `System::destroyAllMolecules()`: `MoleculeType::removeAllMolecules()`
  `delete`d `Molecule` objects still owned by the `MoleculeList` object
  pool (dangling on the next `genDefaultMolecule()`, double free in
  `~MoleculeList()`), and `destroyAllMolecules()` deleted the `Complex`
  objects the recycled pool molecules still referenced. The bug is in
  upstream `RuleWorld/nfsim` `origin/master` (commit `66d3cc57`); bngsim
  is the first library-style consumer to exercise the code path. Fixed
  via a new NFsim vendor carry
  (`bngsim/carry-reset-concentrations-uaf`, patch `0016`) and a
  candidate to push upstream.

- **BNG2.pl `.gdat`/`.scan` function-column parity** (issue #53). Two
  additive surfaces let consumers (e.g. PyBioNetGen) emit files that
  diff byte-identically against BNG2.pl subprocess output:
  - `Result.to_gdat(path, print_functions=True)` (and the C++
    `Result::to_gdat(path, print_functions)`) appends user-named
    function columns after the observables, using BNG2.pl's `()`
    header convention (e.g. `kf_BSA()`). Default `False` keeps the
    observables-only output, matching BNG2.pl when `print_functions=>1`
    is not set.
  - `Result.gdat_expression_names` (and `ResultCore.gdat_expression_names`)
    returns the same columns as `expression_names` but with the `()`
    suffix, for assembling a BNG-native header line where bngsim does
    not own the writer (e.g. a consumer-built `.scan`).
  - `ResultCore.raw_expression_names` / `raw_expression_data` /
    `raw_n_expressions` expose the unfiltered function columns
    (including internal `_rateLawN` values) for debugging.
  The `()` convention is intentionally *not* baked into the in-memory
  `expression_names`: those stay bare because downstream consumers use
  them as column keys for constraint/experimental-data matching (a
  BNG2.pl `.gdat` is read back with the `()` stripped). `.cdat` is
  untouched — it carries species only, never functions.

- **`Simulator.run(steady_state=True)` early termination** (issue #47).
  Adds BNG2.pl `simulate({steady_state=>1})` parity (i.e. `run_network
  -c`): after recording each output point, the CVODE simulator computes
  `||f(t,y)||_2 / n_species` and stops integrating once it falls below
  the tolerance, returning a `Result` truncated to only the rows
  actually integrated. Previously `run()` always integrated the full
  `t_span`, so a `steady_state=>1` model read as a row-count DIFF in
  PyBioNetGen corpus sweeps even though the trajectories matched to
  ~1e-9. New kwargs: `steady_state: bool = False` and
  `steady_state_tol: float | None = None` (when unset, the cutoff is
  the integrator `atol`, matching BNG2.pl which reuses its integration
  atol as the dx/dt criterion). `Result.solver_stats["steady_state_reached"]`
  reports whether the criterion fired before `t_end`. The flag is
  ODE-only; passing it with `method != "ode"` raises `ValueError`.
  Implemented in `SolverOptions::steady_state`/`steady_state_tol`,
  `SolverStats::steady_state_reached`/`steady_state_residual`, the new
  `Result::truncate()`, and the CVODE output loop. Where the kept rows
  overlap a full run they are byte-identical (the early stop only drops
  trailing rows; it does not perturb the integration).

- **`steady_state` / `steady_state_tol` on `Simulator.run_batch`.** The
  BNG2.pl `run_network -c` early-stop (above) is now available on the
  batch/scan path: each parameter point integrates until
  `||f(t,y)||_2 / n_species` drops below `steady_state_tol` (defaulting to
  `atol`) and its `Result` is truncated to the rows actually integrated.
  ODE-only; `steady_state=True` with `method != "ode"` raises `ValueError`.
  `run_batch(steady_state=True, squeeze=True)` is rejected because each
  point may truncate to a different row count (the trajectories cannot be
  stacked into one 3-D array). This is the per-point parity default that
  PyBNF's `parameter_scan` dispatch routes `steady_state=>1` to.

- **`"kinsol"` alias for the Newton steady-state method.**
  `sim.steady_state(method="kinsol")` and `steady_state_batch(method="kinsol")`
  are accepted as aliases for `"newton"`; `ss.method_used` always echoes
  the canonical `"newton"`.

- **Diagnostic warning when a custom expression function returns
  `nan`/`inf`** (issue #42 follow-up). Every custom function registered
  with the ExprTk evaluator — both bngsim's built-ins (`mratio`, `ln`,
  `rint`, `sign`) and anything passed to `define_function()` — now
  routes its return value through a per-evaluator
  `NonFiniteWarningSet`. The first non-finite return for a given
  `(function name, argument bit-pattern)` tuple prints a one-line
  warning to `stderr` naming the function and the offending arguments;
  subsequent calls with the same tuple stay silent, so a long-running
  ODE that repeatedly evaluates the same bad input prints once rather
  than once-per-step. This was added in response to #42, where the
  symptom of mratio's overflow was silent `nan` propagation through
  every downstream parameter and `print_functions=>1` column — without
  such a diagnostic, the next adapter-shaped bug would slip past in
  the same way. `time()` is intentionally not wired (its value comes
  from the simulator, not a computation that can go non-finite at
  this layer).

### Changed

- **Auto-generated `_rateLawN` functions are filtered from the Result
  expression columns** (issue #53). `Result.expression_names` /
  `expression_data` / `n_expressions` (and the `ResultCore` properties
  consumers read) now drop the synthetic `_rateLaw1`, `_rateLaw2`, …
  rate-law functions that BNG2.pl emits into BNG-XML for
  run_network/NFsim but filters out of its own `.gdat`/`.scan` output.
  This applies uniformly across ODE/SSA/PSA/NFsim/RuleMonkey. Names
  stay bare (no `()`); the filtering is at the pybind boundary so the
  underlying C++ `Result` still records every function. The unfiltered
  columns remain available via the new `raw_expression_*` accessors.

- **BREAKING: steady-state solver standardized on the BNG2.pl parity
  criterion and `method="newton"` default.** Three coupled changes to
  `find_steady_state` / `Simulator.steady_state` / `steady_state_batch`:
  1. **One convergence rule.** Every integrate-to-steady-state path now
     uses `||f(y)||_2 / n_species < tol` — the same `run_network -c`
     criterion as `run(steady_state=True)`. The old integration Tier-1
     `max|f(y)|` (L∞) norm and its geometric time-horizon (`t = 10 → 100
     → 1000 → max_time`) were removed; the integration path now marches
     one CVODE step at a time, capped at `max_time`. `SteadyStateResult.residual`
     is now `||f||_2/n` rather than `max|f|`.
  2. **`method="auto"` removed.** `SteadyStateOptions.method` and the
     Python `method=` argument accept only `"newton"`, `"integration"`,
     and `"kinsol"` (alias for `"newton"`); passing `"auto"` (or any
     other value) raises. `"newton"` already means try-Newton-then-
     parity-fallback, so `"auto"` was a redundant synonym with the wrong
     fallback criterion.
  3. **Default is now `"newton"`** (was `"auto"`). On non-convergence
     `"newton"` falls back EXPLICITLY to the parity integration path, so
     the default result always honors the `||f||_2/n` criterion. Callers
     that passed `method="auto"` should drop the argument or pass
     `method="newton"`.

### Fixed

- **`.net` loader recognizes the `$` clamp marker after a
  `@<compartment>::` prefix** (issue #41). BNG2.pl writes cBNGL
  fixed-concentration species as `@CP::$Sink()`, putting the `$`
  marker *after* the compartment prefix. The C++ loader
  (`src/net_file_loader.cpp`) and both Python helpers
  (`_codegen._parse_species_line`, `_net_reader._parse_species`)
  detected the marker only at position 0, so compartmentalized clamps
  fell through to the free-state path: their derivatives were never
  zeroed and the species drifted (e.g. `@C::$ATP()` / `@C::$ADP()` in
  `catalysis.bngl` showed ~0.36 %/h monotonic drift; `@CP::$Sink()`
  integrated the full inflow flux). The new shared helper
  `_strip_fixed_marker` (and the equivalent C++ inline) detects `$`
  at position 0 *or* immediately after an `@<compartment>::` prefix,
  strips just the marker, and preserves the compartment prefix in the
  stored species name. Verified against BNG2.pl `run_network` to
  better than 1e-10 absolute on the issue's minimal `@CP::X() ->
  @CP::$Sink()` repro.

- **`mratio()` no longer overflows to `nan` for large `|z|`** (issue
  #42). `src/expression.cpp`'s `mratio_impl()` evaluated the
  Kummer-function ratio `M(a+1,b+1,z)/M(a,b,z)` by directly summing
  the two power series and dividing. For BNG2.pl-supported parameter
  ranges with large negative-integer `a` and large `|z|` — e.g.
  `test_Mratio_1.bngl`'s `a=-1000, b=9001, z=-10000` — the
  intermediate partial sums peaked at ~1.5e308 and overflowed
  `double` to `inf`, so the ratio became `inf/inf = nan`. The failure
  silently propagated through `U1_U0` / `U2_U1` / `C_mean` / `C_sdev`
  parameters and into any `print_functions=>1` `.gdat` columns
  derived from them. Replaced with a direct port of BNG2.pl
  `Perl2/Expression.pm` `sub Mratio` (Fortran by W. S. Hlavacek 2018,
  Perl by L. A. Harris 2019), which uses Gauss's continued fraction
  evaluated by the modified-Lentz method [Lentz 1976; Thompson &
  Barnett 1986]. Lentz keeps each per-step ratio `Δ_j = C_j·D_j ≈ 1`,
  so partial values stay `O(1)` and the same parameter range now
  converges in a few hundred iterations. The new implementation
  reproduces BNG2.pl's `test_Mratio_1_ode.gdat` to every printed
  digit (`mratio(-1000, 9001, -10000) ≈ 0.46128328365229`,
  `C_mean ≈ 487.5199603907`, `C_sdev ≈ 15.60309027589`). A safety
  iteration cap raises `std::runtime_error` on non-convergence
  instead of hanging; BNG's reference has no such cap but in
  practice the supported range converges well below it.

## [0.6.0] - 2026-05-20

### Added

- **NFsim global/composite functions surface as `Result` expression
  columns.** A BNGL `begin functions` block (e.g.
  `pre1_dose() = alpha1_pre*Clusters()/f`) was previously invisible on
  the in-process NFsim backend: `NfsimSimulator.run()` and the
  `NfsimSession.simulate()` path reported observables only, so any model
  output defined as a global function went missing from the `Result`.
  NFsim already evaluates these functions internally for rate laws; the
  embedded `System` just exposes no public enumerator for them. bngsim
  now reads the function names from the BNG XML `<ListOfFunctions>`
  block and resolves each against the `System` — `GlobalFunction`s
  (observable/parameter expressions) via `FuncFactory::Eval`, and
  `CompositeFunction`s (function-of-function) via `evaluateOn` — writing
  the values into the `Result` expression columns at every output step.
  Composite functions with molecule-scoped (local) dependencies have no
  scalar value and are skipped. This is a bngsim-side change with no
  vendored-NFsim edit; standalone NFsim's own `-ogf` writer walks only
  global functions, so resolving composites here is a strict superset.

- **RuleMonkey (`nf_exact`) global functions surface as `Result`
  expression columns.** The counterpart of the NFsim change above for
  the in-process RuleMonkey backend. Vendored RuleMonkey refreshed to
  3.2.0, whose `rulemonkey::Result` now reports BNGL `begin functions`
  globals alongside the observables (`function_names` / `function_data`,
  parallel to `observable_names` / `observable_data`).
  `convert_rulemonkey_result()` copies those columns into the bngsim
  `Result` expression columns at every output step. Local
  (molecule-scoped) functions have no scalar value and are omitted by
  RuleMonkey from `function_names`. Models whose fitted outputs are
  defined as global functions (e.g. the Kozer-EGFR `Clusters()`,
  `pre1_dose()`) are now usable on the RuleMonkey backend.

- **`Result.has_simulation_data` classification property.** Batch
  harnesses processing mixed BNGL corpora (some files complete, some
  work-in-progress with parameters-only) need a programmatic signal to
  distinguish "ran and produced data" from "ran and produced nothing
  meaningful" without resorting to exception handling. NFsim, like the
  BNG2.pl subprocess, succeeds vacuously on a BNGL file whose `simulate`
  action references an empty model body — bngsim returns a `Result` with
  `n_times > 0` but `n_species == n_observables == n_expressions == 0`,
  mirroring BNG2.pl's `.gdat` with only the `# time` header. The new
  property is `True` iff at least one of species/observables/expressions
  has nonzero column count, so harnesses can route empty results to an
  `empty_simulation` bucket instead of flagging them as parity failures
  against subprocess output that is equally empty. Closes #40 B-2.

- **`NfsimSession.save_species(path)` for file-based state writeback.**
  Binds NFsim's `System::saveSpecies` so PyBioNetGen-style hosts can
  thread state across multi-segment NF protocols using the same
  `.species` artifact BNG2.pl emits under
  `simulate({method=>"nf", get_final_state=>1, ...})`. The file lists
  one species pattern per line with its integer count; bonded
  multi-molecule complexes are encoded with BNGL `!N` notation,
  matching NFsim CLI's standalone output. The underlying NFsim call is
  wrapped in `StreamSuppressor` so the "saving list of final molecular
  species..." progress line stays off stderr. The in-process snapshot
  path remains blocked on an upstream NFsim segfault
  (internal#52); `save_species` + file replay is the supported
  way to thread NF state across bngsim segments today. Closes #40 B-3b.

- **`NfsimSession.simulate(..., relative_time=False)` opt-in for
  BNG2.pl time-axis parity.** BNG2.pl's `simulate({method=>"nf", ...})`
  action reports timepoints as elapsed time since `t_start` (per its
  runtime warning "NFsim timepoints are reported as time elapsed since
  `t_start=$t_start`"), a different convention than every other bngsim
  backend. Pass `relative_time=True` on a per-call basis to opt into
  that labelling: the returned `Result.time` starts at `0.0` and ends
  at `t_end - t_start`. Default `False` preserves the existing
  absolute-time stamps so no existing caller is affected. The internal
  NFsim clock and `session_logical_time` are unaffected — multi-segment
  threading still advances physical time across mixed-flag segments —
  so the flag is purely a labelling toggle. Closes #40 t_start.

### Changed

- **`src/expression.cpp` is now built as a standalone `bngsim::expression`
  CMake target** (#39). Previously a plain entry in `BNGSIM_SOURCES`,
  `bngsim::ExprTkEvaluator` is now its own static target, declared before
  `add_subdirectory(third_party/rulemonkey)`; the main `bngsim` library links
  it. This lets the vendored RuleMonkey — which since RuleMonkey#6 (the
  ExprTk swap) reuses `bngsim::ExprTkEvaluator` — detect the host target via
  `if(TARGET bngsim::expression)` and link it instead of compiling a
  duplicate copy of `exprtk.hpp` + the evaluator (a duplicate-symbol / ODR
  hazard). No behavior change; the in-tree build still produces exactly one
  copy of the expression symbols. `vendor_rulemonkey.py` already excludes
  RuleMonkey's `third_party/` from the vendored copy — now documented as
  deliberate so it is not "helpfully" re-added.

- **BioModels benchmark corpus (Pool B) migrated to manifest-driven fetch** (#17).
  Removed 963 vendored `.ant` files from git (repo is ~45K lines lighter).
  Added `biomodels_ant_pool_manifest.json` with 1013 BioModels IDs. Fresh
  clones now require one-time pool materialization (~10-30 min):
  ```bash
  python bngsim/benchmarks/convert_sbml_to_ant.py
  # or use --ensure-pool flag in harness scripts
  ```
  Benchmark scope increased from 963 to 1013 models by default. New env var:
  `BENCH_AUTO_ENSURE_POOL=1` for automatic fetch. **Breaking:** First-time
  setup requires network access to EBI BioModels. **Note:** Fetch/convert
  pipeline requires Python 3.10+ (uses union types `int | None`).

- **Benchmark suite restructured into `benchmarks/suites/<name>/`**
  (Phase 1-6 reorg). Each paper-coordinated benchmark now lives in its
  own `suites/<name>/` directory with a shared `_emit.py` LaTeX-fragment
  layer, `paper_role` / `--audience` filtering for selective table
  rendering, and a top-level `run_all.py` orchestrator. Model corpora
  moved under `benchmarks/models/` — `net/{ode,ssa}/` for `.net`
  benchmarks, `antimony/ssys/` for the 117 hand-crafted Antimony models.
  All scripts now resolve tool and corpus paths from env vars
  (`BNG2_PL`, `RUN_NETWORK`, `NFSIM`, `BNG_MODELS2`, `PYBNF_EXAMPLES`,
  `RULEHUB_DIR`, `RULEBENDER_WS`) rather than hardcoded absolute paths,
  so the benchmark layer is portable across machines.

### Fixed

- **NFsim `stepTo` over-binding** (closes PyBNF#391). `System::stepTo`
  sampled an inter-event waiting time and, when it overshot the
  stopping time, discarded the sample and re-sampled on the next call.
  The re-sample was drawn from `current_time`, which is still before
  the previous stopping time, so the next event was biased earlier and
  could fall inside the already-elapsed window — reactions
  systematically over-fired, accumulating with each output step. On an
  irreversible A+B→AB binding model this gave a bound mean of ~55 vs
  the exact master-equation value 50.08 (~10% bias). `stepTo` now
  caches the overshooting waiting time and consumes it on the next
  call; `invalidateStepToCache()`, previously a no-op, clears the cache
  and is called after every parameter/population mutation and from
  `equilibrate()`. The draw uses `random_open()` (as `System::sim()`
  does) so a cached `delta_t` can never be 0 or infinite. Carried in
  the NFsim vendoring queue so future refreshes preserve it.

- **Accept legacy `Sat` and `Hill` `.net` rate laws.** `Sat` and `Hill`
  rate-law tokens (removed as native types in v0.2.0) used to raise at
  `Model.from_net()` load time with a rewrite suggestion. The `.net`
  loader now rewrites supported cases into synthetic `Functional` rate
  laws and emits a `UserWarning` with migration guidance (e.g. the
  `k/(K+S)`, *not* `k*S` distinction for Sat) so legacy BNG-generated
  `.net` files load directly without manual hand-translation. A new
  `NetworkModel.load_warnings()` accessor exposes the same messages to
  consumers that don't subscribe to the `warnings` stream.

- **Round fractional SSA initial populations.** Models with fractional
  initial species amounts (typically introduced by an SBML/Antimony
  loader's volume-factor multiplication on a concentration that
  doesn't land on an integer count) previously left the SSA setup path
  to consume the fractional value as-is, biasing initial reaction
  propensities. `validate_for_ssa` now emits a
  `non_integer_initial_population` warning per offending species, and
  the C++ SSA setup path rounds the storage value to the nearest
  integer at simulation start. `Simulator(model, method="ssa")` no
  longer raises on fractional initial counts; the rounding is reported
  through `validate_for_ssa` so consumers (PyBNF, batch harnesses) can
  surface it to users.

- **ODE solver no longer aborts on CVODE's recoverable `CV_TOO_MUCH_WORK`
  return.** The CVODE integration loop in `cvode_simulator.cpp` advanced
  to each output point with `CVode(..., CV_NORMAL)` and threw on any
  negative return flag. `CV_TOO_MUCH_WORK` (flag `-1`) is the one
  negative flag that is *recoverable*: it means CVODE used its per-call
  step budget (`max_steps`, default 20000) without reaching the target,
  but the integrator state is intact and a further `CVode()` call simply
  continues. Treating it as fatal made `Simulator.run(method="ode")` fail
  on bounded, integrable models that `run_network` handles — e.g. the
  Lotka-Volterra benchmark, whose fast rate constants need far more than
  20000 steps to cross a sparse output interval. The loop now re-calls
  `CVode()` on `CV_TOO_MUCH_WORK` (so `max_steps` is a per-call batch
  size, not a hard ceiling), re-checking the wall-clock `timeout` budget
  between batches; genuinely fatal flags still raise. Closes #50.

## [0.5.5] - 2026-05-12

### Fixed

- **Analytic Jacobian for Python-keyword-named primaries and `if(...)`
  derived params (closes #27).** The `_compute_derived_param_jacobian`
  pass that powers codegen forward sensitivity used to give up — return
  `None` — on two derived-parameter shapes:
  1. expressions referencing a primary literally named with a Python
     keyword (`lambda`, `if`, `class`, `for`, ...), because sympy's
     `parse_expr` tokenizer chokes on the keyword;
  2. expressions of the form `if(c, t, f)`, because sympy has no
     built-in BNGL `if` and `parse_expr` raised `SyntaxError` on the
     trailing comma.
  The pre-fix `None` outcome from #26 meant the codegen sensitivity
  RHS silently treated `∂p_d/∂primary` as zero for these shapes; CVODES
  forward sensitivity fell back to internal finite differences (slow
  but correct), and the analytic chain-rule path through the derived
  param was lost.
  This release adds two preprocessing passes to
  `_compute_derived_param_jacobian` before `parse_expr`:
  - BNGL `if(c, t, f)` is rewritten to sympy
    `Piecewise((t, c), (f, True))` with balanced-paren and
    top-level-comma parsing (whole-word match on `if`, so identifiers
    like `if_thresh` are untouched); nested `if`s are recursed into.
    Sympy then differentiates the conditional analytically — the
    boundary delta follows sympy's standard `Piecewise` convention.
  - Primary parameter names that happen to be Python keywords are
    aliased to `_BNG_KW_<name>` placeholders before `parse_expr` and
    round-tripped back to `p[idx]` after `sp.ccode`. The alias never
    appears in emitted C, and the keyword itself never appears either
    (which would be a C-side syntax error anyway).
  Either pass is independent of the other; both are no-ops on
  expressions that don't trip the corresponding shape, so byte-for-byte
  codegen output is unchanged for the rest of the corpus. The three
  motivating PyBioNetGen corpus models now get an analytic Jacobian:
  `ode/scaling_example.bngl` (`_rateLaw1 = lambda*(1-phi)`),
  `ode/4var_model.bngl` (`lambda`-named primary + `T0 = if(...)`),
  and `ode/4var_model_with_FDC.bngl` (`T0 = if(...)`).
  End-to-end check: forward sensitivity for `sensitivity_params=["lambda"]`
  on a `scaling_example`-shaped `.net` now matches CVODES internal FD to
  `relerr < 1e-3` over the whole trajectory.
  `_CODEGEN_VERSION` bumped 7 → 8 to invalidate v7-vintage cached `.so`
  files for the (small set of) models whose generated dfdp switch now
  carries extra `derived_terms`. pytest 948 passed, 3 skipped.

## [0.5.4] - 2026-05-12

### Fixed

- **Large-SBML codegen scales poorly on >1000-reaction models (closes #25).**
  `Model.from_sbml_string` was taking ~35 minutes on MODEL1009150002
  (1604 species, 1855 reactions) and ~10 minutes on MODEL1007060000
  (1025 species, 1126 reactions). The loader itself was already fast
  (<1 s); 100% of the observed time was spent in the auto-triggered
  C codegen pass (`_codegen.prepare_model_codegen`, fires at
  ≥256 species), specifically in `_translate_expr_to_c` and the
  `.net`-path twin `_translate_expr`. Both ran one `re.sub` per
  parameter / species / observable / function name; on a 5000-parameter
  / 1000-function model that was ~7000 full-expression regex passes per
  expression × ~1000 expressions, with the `re` module's compile cache
  thrashing on top. Replaced with a single tokenizing regex
  (`[A-Za-z_]\w*(\s*\(\s*\))?`) plus a unified priority lookup dict
  (`func > obs > species > param > builtin`). The per-reaction
  `func_names.index(fname)` linear search was also lifted to a dict
  lookup, and the per-name C-reference maps are now built once in
  `generate_rhs_from_model` and reused across every function-body
  translation rather than rebuilt on each `_expr_to_c` call.
  Measured speedups on a 2018-era Mac Mini:

  | Model                  | Before  | After  | Speedup |
  | ---------------------- | ------- | ------ | ------- |
  | MODEL1009150002 (1855 rxn) | 2079 s | 11.4 s | ~180×   |
  | MODEL1007060000 (1126 rxn) | 594 s  | 4.1 s  | ~145×   |

  bngsim is now faster than libroadrunner 2.9.2 by 4–13× across the
  medium-to-large model range (giant: 12.5 s vs 53.8 s). C output is
  byte-identical for unchanged input on MODEL1007060000 (codegen cache
  hash preserved); `_CODEGEN_VERSION` bumped 6 → 7 defensively so
  v6-vintage cached `.so` files are not silently reused. The codegen
  path is CVODE-ODE-only; SSA simulation correctness is unaffected.
  pytest 944 passed, 3 skipped.

## [0.5.3] - 2026-05-11

### Fixed

- **Wrapper-form `tfun(...)` in BNGL function bodies (closes #33).**
  Functions of the form `f() = (tfun('drive.tfun') + 5) / k_scale` —
  i.e., a `tfun(...)` call wrapped in arithmetic — previously failed two
  ways: the `.net` interpreter silently overwrote the whole function
  expression with `tfun_<name>()` (dropping all wrapper math; numeric
  output wrong with no warning), and the C codegen path emitted invalid
  C with a raw `tfun('...',time)` token that `cc` rejected. The loader
  now scans each function expression for embedded `tfun(...)` calls
  left-to-right, registers each as a synthetic anonymous table function
  named `<func>__tfun<k>`, and substitutes `tfun_<synth>()` for just the
  call substring — so wrapper math survives untouched into ExprTk.
  Whole-body `tfun(...)` keeps the legacy naming convention, so
  `table_function_names == ["cumNcases"]`-style assertions still hold.
  `TableFunction::from_file` grew an optional `header_name` parameter
  (plumbed through `NetworkModel::add_table_function` and
  `ModelBuilder::add_table_function_spec`) so the synthetic-named table
  still validates against the original BNG function name in the `.tfun`
  column-2 header. The Python codegen mirrors the C++ change: each
  embedded `tfun(...)` is replaced with a unique placeholder, the
  surrounding expression is translated normally, then placeholders are
  post-substituted with `data->tfun_eval(tf_id, idx, ctx)` callbacks.
  `_CODEGEN_VERSION` bumped 5 → 6 to invalidate cached `.so` files.
  Regression fixture promoted to `tests/data/{wrap_single.net,
  drive.tfun}`; four new tests in `TestTfunWrapperForm` cover interp
  numeric (`Xtot(t=2) ≈ 1.45`), codegen end-to-end numeric, the emitted
  C shape, and the synthetic-naming convention. Related: surfaced an
  upstream gap in BioNetGen's own `run_network` (silent-wrong-answer
  and loud-error modes for wrapper-form), filed as
  [RuleWorld/bionetgen#314](https://github.com/RuleWorld/bionetgen/issues/314).

- **`.tfun` header + index canonicalization to match BioNetGen (closes
  #35).** bngsim's tfun index resolution and `.tfun` header validation
  diverged from `bng2/Perl2/TfunReader.pm` in two ways: only lowercase
  `time`/`t` were accepted (BNG matches `/^(time|t)$/i`), and a trailing
  `()` was only tolerated on the time-index column (BNG strips it from
  both columns regardless of index kind). So a `.tfun` written with
  `# Time  cumNcases()` or `# drug_conc()  response()` — both legal
  per BNG — failed to load. The two file-local `is_time_index*` statics
  in `src/table_function.cpp` and `src/model.cpp` are now consolidated
  into shared inline helpers in `include/bngsim/table_function.hpp`:
  `strip_paren_suffix` strips a trailing `()`, and `is_time_index`
  lowercases and matches against `time`/`t` after stripping. Applied at
  all six divergence sites: column-1 and column-2 header validation in
  the `.tfun` reader, and the parameter / observable map lookups in
  `register_table_function_` and `table_function_specs`. Two new
  fixtures (`tfun_uppercase_time.net` + `cumNcases_uppercase.tfun`;
  `tfun_paren_param.net` + `dose_response_paren.tfun`) and four new
  tests in `TestTfunIndexCanonicalization` verify the new variants load
  and produce trajectories indistinguishable from the canonical
  lowercase fixtures (`rel=1e-9`).

## [0.5.2] - 2026-05-11

### Added

- **`Result.xr` + `Result.to_xarray()` — xarray accessors (AMICI-style).**
  New optional surface for downstream code accustomed to AMICI's
  `rdata.xr.x` / `.y` / `.sx` labeled-array ergonomics. Two entry
  points, both gated on the optional `xarray` dependency:
  - **`result.xr`** — lazy per-field accessor; each attribute access
    builds a fresh `xarray.DataArray` with labeled coords.
    `result.xr.species` has dims `(time, state)`,
    `result.xr.observables` has `(time, observable)`,
    `result.xr.expressions` has `(time, expression)`,
    `result.xr.sensitivities` has `(time, state, parameter)`,
    `result.xr.sensitivities_ic` has `(time, state, ic_state)`.
    Slicing reads naturally: `result.xr.sensitivities.sel(parameter="k1", state="A")`.
  - **`result.to_xarray()`** — one-shot constructor returning an
    `xarray.Dataset` with `species` / `observables` / `expressions`
    (and `sensitivities` / `sensitivities_ic` when present) as data
    vars sharing a `time` coord. `custom_attrs` is mirrored onto
    `ds.attrs`; the stochastic `seed` (when set) is written as
    `ds.attrs["seed"]`. Enables `ds.to_netcdf(...)` for users who
    prefer xarray's archive format.
  Dimension naming follows AMICI's convention (`state` rather than
  `species`) so the same code that slices `rdata.xr.x.sel(state=...)`
  works against a bngsim Result. Both raise `AttributeError` /
  `ImportError` with actionable messages when xarray is missing or
  the requested block is empty (e.g. requesting `sensitivities` on a
  Result run without `sensitivity_params`). `to_xarray()` rejects 3-D
  batch results with `RuntimeError` directing callers to iterate
  replicates.
  Same data-only scope as the other in-memory views: solver stats
  and the C++ `_core` are not part of the xarray surface; HDF5
  (`Result.save`) remains the lossless archive. README's "In-memory
  access" section gains an xarray subsection. New `TestResultXarray`
  class in `python/tests/test_new_features.py` (9 tests) covers
  per-field dims/coords, named selection round-trips against the raw
  ndarrays, sensitivity layout, missing/unknown-field error paths,
  Dataset shape, seed + custom-attr propagation, HDF5-loaded
  results, and the batch-3D rejection.

## [0.5.1] - 2026-05-10

### Added

- **`Result.to_csv(...)` — plain delimited-text export (closes #11).**
  New writer on `Result` for SBML/RoadRunner/Tellurium-style output:
  a plain header row (no `#` prefix) followed by data rows, with a
  caller-chosen single-character delimiter (`","` by default; pass
  `sep="\t"` for TSV). The first column is `time` (unless
  `include_time=False`); the remaining columns carry the in-memory
  observable or species names verbatim, so the file loads with
  `pandas.read_csv` or `numpy.loadtxt` without extra parsing. The
  `kind` keyword selects the `"observables"` block (default,
  `.gdat`-equivalent) or the `"species"` block (`.cdat`-equivalent);
  `header=False` omits the column-name row for appending to existing
  files. Works on every backend (ODE / SSA / PSA / NFsim / RuleMonkey)
  and on results round-tripped through HDF5. Batch (3-D) results are
  rejected with a clear `ValueError` directing callers to iterate the
  per-replicate list. Same scope as `to_gdat` / `to_cdat`: text
  trajectories only — expressions, sensitivities, solver stats,
  custom attrs, and the stochastic seed survive only through
  `Result.save(...)` (HDF5). Added a README "Export results to text
  files" section that documents the full text-export matrix and what
  each writer loses. New `TestResultToCsv` class in
  `python/tests/test_new_features.py` (11 tests) covers CSV/TSV
  round-trips via `numpy.loadtxt` and `pandas.read_csv`,
  `kind="species"` columns, `include_time=False`, `header=False`,
  raw-constructed and HDF5-loaded results, and the invalid-kind /
  multi-char-separator / 3-D-batch rejection paths.

## [0.5.0] - 2026-05-10

### Added

- **Wall-clock timeout / cancellation for `Simulator.run` (closes #9).**
  New `timeout: float | None = None` keyword argument on
  `Simulator.run(...)` and `Simulator.run_batch(...)`. When a positive
  budget is supplied, the simulator raises `bngsim.SimulationTimeout`
  (a new typed exception, sibling of `SimulationError` under
  `BngsimError`) once elapsed wall-clock time exceeds the limit. The
  exception carries `timeout` (configured limit) and `elapsed` (actual
  wall-clock at trip-time) attributes so consumers like PyBNF's
  `wall_time_sim` can classify wall-clock terminations distinctly from
  solver/convergence failures. Supported on ODE (CVODE), SSA, PSA, and
  NFsim; the RuleMonkey backend's vendored sampler is opaque from the
  bngsim wrapper and currently rejects positive timeouts with
  `NotImplementedError`. The check is performed on a steady-clock
  snapshot taken at the top of each simulator's main loop and is
  GIL-free, so concurrent threads remain responsive. Partial results
  are not currently salvaged; the timeout exception's
  `partial_result` field is reserved for a future iteration. New
  `python/tests/test_timeout.py` (11 tests) covers exception hierarchy,
  argument validation, per-backend firing behavior, and the
  RuleMonkey-rejection path. Low-level callers can also set
  `SolverOptions.timeout_seconds` directly when working with
  `bngsim._bngsim_core.CvodeSimulator`.
- **`NfsimSession.simulate(...)` honors `timeout` (closes #32, followup
  to #9).** The session-API entry point used by PyBNF's
  `BngsimNfModel.execute` now accepts `timeout: float | None = None`
  with the same normalization rules as `Simulator.run` (None / 0 / non-
  positive disables the budget, negative raises `ValueError`). The C++
  `NfsimSimulator::simulate` checks the `WallClockBudget` before each
  `stepTo()` output point — the same granularity as the stateless
  `run()` path, since NFsim's sampler is opaque inside one stepTo().
  On overrun the call raises `bngsim.SimulationTimeout`; the session
  clock is left at its segment-entry value and the live NFsim System
  has advanced an indeterminate distance into the segment, so the only
  safe follow-up is `destroy_session()` / context-manager exit. The
  RuleMonkey session simulate() and step_to() paths are deliberately
  unchanged for the same opaque-sampler reason as `Simulator.run` on
  RuleMonkey. New tests in `python/tests/test_timeout.py` cover
  mid-segment firing, clean post-timeout destroy, generous-budget
  completion, negative-timeout rejection, and the
  `timeout=None`/`timeout=0` no-op contract.
- **RuleMonkey backend now honors `timeout` everywhere (closes the last
  gap in the wall-clock timeout surface).** With upstream
  [richardposner/RuleMonkey#3](https://github.com/richardposner/RuleMonkey/issues/3)
  landing a cooperative-cancellation hook (commit
  [`70fcac2`](https://github.com/richardposner/RuleMonkey/commit/70fcac2),
  vendored at `6d7f240` HEAD), bngsim now passes a
  `WallClockBudget`-backed `rulemonkey::CancelCallback` into every RM
  call site and translates the upstream `rulemonkey::Cancelled` into the
  typed `bngsim::TimeoutError` that the existing pybind11 translator
  maps to `bngsim.SimulationTimeout`. The Python `Simulator.run(
  method="nf_exact", timeout=...)` no longer raises
  `NotImplementedError`; instead it raises `SimulationTimeout` once the
  budget trips (granularity = upstream's 1024-event stride).
  `RuleMonkeySession.simulate(..., timeout=)` and
  `RuleMonkeySession.step_to(time, timeout=)` gain the same kwarg with
  matching normalization. Post-timeout the live RM session is at the
  last completed SSA event; subsequent state is undefined for bngsim
  purposes, so callers should `destroy_session()` before reuse. The
  previously deferred `NotImplementedError` path in `Simulator.run` /
  `.run_batch` is removed. Vendored RM advances from `97a08e0` →
  `6d7f240`; diff = the cancellation hook plus a docs follow-up, no
  other behavioral changes. New tests in `python/tests/test_timeout.py`
  cover stateless and session firing, step_to firing, clean post-
  timeout destroy, generous-budget completion, and the negative /
  None / 0 normalization rules.
- **Capability introspection surface for optional features (closes #13).**
  New module-level flags `bngsim.HAS_LIBSBML` and `bngsim.HAS_ANTIMONY`
  match the existing `HAS_NFSIM` / `HAS_RULEMONKEY` pattern, plus a new
  `bngsim.capabilities()` aggregator that returns a stable structured
  dict `{"version", "features", "missing"}`. `features` contains the
  same keys regardless of build (`nfsim`, `rulemonkey`, `libsbml`,
  `antimony`, `sbml_import`, `sbml_ssa`, `sbml_psa`, `antimony_import`,
  `codegen`); `missing[name]` distinguishes a compiled-backend gap
  ("NFsim/RuleMonkey backend not present in this install" — vendored at
  `third_party/<x>/` and built by default, so the install was either
  configured `-DBNGSIM_BUILD_<X>=OFF` or comes from a wheel that
  excludes the backend) from a missing optional Python dependency
  ("optional dependency `'python-libsbml'` not installed"). PyBNF (and
  other downstream tools) can probe with a single call instead of
  try/except probing every loader. New `python/tests/test_capabilities.py`
  (31 tests) covers schema, consistency with module-level flags,
  missing-explanation distinction, and public-`__all__` exposure. README
  adds a "Capability introspection" section under Optional dependencies.

### Changed

- **BREAKING: stochastic seed default is now `None` (fresh draw) instead
  of `42` (closes #10).** `Simulator.run`, `Simulator.run_batch`,
  `Simulator.run_until`, `NfsimSession.initialize`, and
  `RuleMonkeySession.initialize` all change their `seed` parameter
  from `seed: int = 42` to `seed: int | None = None`. When the caller
  omits `seed=` (or passes `None`), bngsim now draws a fresh 31-bit
  seed from system entropy on each call so two consecutive calls
  produce independent stochastic trajectories — fixing the
  surprising-silent-reuse behavior #10 calls out. Explicit
  `seed=N` continues to pass `N` straight through to the backend
  verbatim, so any caller that wants reproducibility just keeps doing
  what they were doing. The actual integer used is exposed on
  `Result.seed` (None for ODE results) and on `session.seed` for
  stateful sessions; `Result.save()` / `Result.load()` round-trip the
  seed via an HDF5 attribute. For `run_batch`, `seed=` is the base
  seed; per-sim seeds are `base_seed + i` and stamped on each
  `Result.seed`. New `python/tests/test_seed_semantics.py` (28 tests)
  pins the contract; README adds a "Seed semantics for stochastic
  methods" section under SSA/PSA. The reproducibility unit is
  **same starting model state + same `seed=N`**: the C++ SSA/PSA
  backends already construct a fresh `std::mt19937_64(seed)` on every
  `.run()` call (`src/ssa_simulator.cpp`), so passing the same seed
  always seeds identically; what persists across `.run()` calls on the
  same Simulator is the *model state*, which is what makes
  multi-segment SSA protocols (`simulate(...); simulate({continue=>1,
  ...})`) work. Use `model.reset()` (or any explicit
  `set_concentrations`) to return to initial state before re-running
  for trajectory reproduction.

  Migration: any caller that relied on the implicit `seed=42` for
  reproducibility should pass `seed=42` explicitly. Callers that want
  fresh trajectories on each call (PyBNF fitting/smoothing workflows,
  most ad-hoc usage) get the new behavior automatically. Warrants a
  MAJOR bump (0.4 → 0.5) per the CHANGELOG SemVer policy.

- **Backend-unavailable error messages reflect that vendored NFsim and
  RuleMonkey are built by default.** Previously six "Rebuild with
  -DBNGSIM_BUILD_<X>=ON" messages (in `_nfsim_session.py`,
  `_rulemonkey_session.py`, and `_simulator.py`) implied the default
  was OFF, but `BNGSIM_BUILD_NFSIM` and `BNGSIM_BUILD_RULEMONKEY` both
  default to `ON` in `CMakeLists.txt`. Reworded to: "<X> backend not
  present in this install. The vendored backend at `third_party/<x>/`
  is built by default; this install was either configured with
  `-DBNGSIM_BUILD_<X>=OFF` or installed from a wheel that excludes
  <X>." This guides users toward the actual fix (reinstall a build
  that includes the backend) rather than implying they have to enable
  a flag that's already on.

### Fixed

- **NfsimSession species API on multi-state symmetric components (closes #21).**
  `set_species_count`, `add_species`, `remove_species`, and
  `get_species_count` previously raised `unknown component '<name>' on
  MoleculeType '<X>'` for any pattern where two or more components shared
  the same name (e.g. BLBR-style `L(r~u,r~u)` against
  `L(r~u~c~g, r~u~c~g)`). NFsim's XML loader renames duplicate
  `<ComponentType id="r">` entries to internal `r1`/`r2` and surfaces them
  as a symmetric equivalency class; the resolver now recognizes class-
  original bare names and routes each parsed component into a class
  bucket. Per-class state/bond constraints are matched as a sorted
  multiset, so `get_species_count("L(r~u,r~c)")` returns the same total as
  `L(r~c,r~u)` instead of double-counting or missing one ordering. Direct
  `r1`/`r2` disambiguation continues to work. New
  `TestNfsimSessionSymmetricSites` covers homogeneous, heterogeneous, and
  stateless symmetric cases against fresh `sym_state_sites.xml` /
  `sym_stateless_sites.xml` fixtures.

### Changed

- **Single source of truth for the version string (closes #31).**
  `pyproject.toml` is now the only file with a literal `"X.Y.Z"`.
  `bngsim.__version__` reads from `importlib.metadata` (with a
  `pyproject.toml` fallback for source-tree imports), the C extension
  receives the version via a `BNGSIM_VERSION_STR` compile define set by
  CMake, `CMakeLists.txt` regex-parses `pyproject.toml` before
  `project()`, and `Result.save()` reads `bngsim._version.__version__`
  at write time. New `python/tests/test_version_consistency.py` (8
  tests) guards against re-introducing a literal in any of the four
  derived anchors. The release procedure in `dev/notes/RELEASING.md`
  collapses from "edit five files" to "edit `pyproject.toml`, run
  tests."

## [0.4.1] — 2026-05-10

### Added

- **Stochastic PSA on SBML models (closes #8).** `Simulator(model,
  method="psa", poplevel=N_c)` now accepts SBML/Antimony models loaded via
  `Model.from_sbml(...)` / `Model.from_antimony(...)`, sharing the
  `validate_for_ssa` gate with SSA at construct time (PSA-options check
  fires first, then SSA validation — same dispatch path as `.net`). New
  `bngsim/python/tests/test_sbml_psa.py` locks the user-facing contract:
  `poplevel` validation on SBML, end-to-end smoke, shared-validation
  errors (`reversible_non_mass_action`, `non_integer_stoichiometry`),
  and a statistical mass-action parity test against the analytical
  isomer mean (5·SE band, n_reps=200). Harness extension in
  `bench_psa_vs_runnetwork.py` adds an additive third arm
  (`bngsim_sbml_time` + `sbml_vs_net_ratio`) when a model entry in
  `suite_psa.json` carries a cross-referenced `sbml_file`. PyBNF wiring
  follows in a separate PR against `lanl/pybnf`.

## [0.4.0] — 2026-05-10

### Added

- **Stochastic SBML support (closes #7).** SBML / Antimony models now
  simulate under `Simulator(model, method="ssa")` end-to-end, with the
  same Gillespie semantics the BNGL backend uses for `.net` models.
  Round-tripped through bngsim's `.net` emitter (7/7 reference SBMLs
  byte-identical at N=200) and benchmarked against DSMTS at N=10000
  sb=1000 (38/39 strict pass, the one remainder per DSMTS README "no
  action" guidance for `00039`). Phases that landed:
  - **Phase 2 / 2.5**: per-species `volume_factor` and per-reaction
    `ssa_volume_factor` fix V≠1 single- and cross-compartment
    propensities; new `Reaction::per_species_volume_scaling` flag for
    cross-compartment kineticLaws (defaults `false`, off by default).
  - **Phase 3**: load-time `validate_for_ssa(model) -> [SsaIssue]` and
    construction-time `SsaValidationError` raised by
    `Simulator(method="ssa")` for `reversible_non_mass_action`,
    `non_integer_stoichiometry`, `assignment_rule_on_reactant`,
    `non_mass_action_volumetric_species`, `compartment_rate_rule`,
    `fast_reaction`. Loader stays permissive (always loads); the gate
    fires only at SSA-simulator construction.
  - **Phase 4**: `Result.as_roadrunner()` returns a roadrunner-compatible
    `NamedArray` so PyBNF can swap libroadrunner for bngsim under SSA
    without changing downstream consumers.
  - **Phase 5a**: per-species emission honors the kineticLaw / SBML
    reactant-list multiplicity reconciliation under SSA (new
    `Reaction::apply_species_factor` flag).
  - **Phase 5b**: SBML L3 events fire correctly under SSA via a new
    `EventNoDelay` channel (delays remain ODE-only / Phase 5c
    deferred). DSMTS event subset 36/37 strict at N=10000.
  - **Phase 6**: PyBNF's bngsim SBML/Antimony bridge can route SSA
    workflows through bngsim end-to-end (PyBNF-side gates lifted in
    `~/Code/PyBNF` master `f474e53`).
  - **Phase 7**: SBML loader recognizes the COPASI/Antimony reversible
    kineticLaw shape (`[wrapper *] (kf*A - kr*B)`, including
    `compartment * (...)` wrapping) and emits two Elementary SSA
    channels per SBML reaction. Models like `abc.xml` now run under
    SSA directly without manual splitting.
- **Loader robustness on the BioModels corpus**: A1 hoists the
  rate-rule + event-promotion pre-scan above compartment registration
  so models like BIOMD338/339 no longer hit the `compartment_1`
  ExprTk double-registration; A2 pre-scans rules for reaction-id
  references so kineticLaw expressions like
  `Ap' = r77` (BIOMD542) compile; A3 raises a clear actionable error
  on reactions declared without `<kineticLaw>` (MODEL0568648427 et al.)
  rather than emit a Functional reaction referencing a nonexistent
  symbol.
- **`Simulator(strict_ssa=False)` override (B1).** Optional opt-in
  that downgrades overridable `validate_for_ssa` errors
  (`reversible_non_mass_action`,
  `assignment_rule_on_reactant`,
  `non_mass_action_volumetric_species`,
  `compartment_rate_rule`) to warnings, matching libroadrunner's
  "warn and run" UX. `non_integer_stoichiometry` and `fast_reaction`
  remain non-overridable (true correctness violations under SSA).
  Default stays `strict_ssa=True`.
- **Selective `logger.warning` on referenced-but-unset parameters**
  (C2). The loader walks every kineticLaw / rule / event /
  initialAssignment, builds the set of names actually referenced,
  and warns only on parameters that are both `!isSetValue()` AND in
  the referenced set. Unused unset-value parameters
  (extension-package placeholders, doc-only declarations) stay
  silent. Workaround for stricter checks (load through libroadrunner
  first) documented in `bngsim/dev/notes/SBML_VS_ROADRUNNER.md`.
- **`bngsim/dev/notes/SBML_VS_ROADRUNNER.md`** (C1) covers the
  load-time + SSA-time differences between bngsim and libroadrunner
  side by side, the issue-code table, and the `strict_ssa` override
  semantics — the canonical reference for users hitting load- or
  SSA-gate divergence between the two backends.

### Fixed

- **Accept BNGL `obs()` zero-arg Observable references in rate laws
  (closes #28).** BNGL's grammar
  (`bionetgen/bng2/Perl2/Expression.pm:870-927`) accepts an Observable
  as a zero-arg call (`obs()`) anywhere a bareword `obs` is valid;
  BNG2.pl preserves the user's syntax verbatim when emitting the
  `.net` file. ExprTk's grammar would parse `obs()` as the implicit
  multiplication `obs * ()` and rejected the empty parens with
  ERR248, breaking any `.net` that referenced an observable in
  zero-arg-call form (e.g., `proliferation.bngl` in the parity
  corpus). `ExprTkEvaluator::compile()` now tracks names registered
  via `define_variable` / `add_remapped_constant` and strips
  `name()` → `name` for any matched scalar before identifier
  remapping. Function names — built-ins (`sin`, `time`, `mratio`, …)
  and user-registered `Func0/1/2/3` — go through `add_function` and
  are not in the scalar set, so their parens are preserved. The
  1-arg LocalFunction form `obs(s)` is BNG2.pl's responsibility and
  is fully expanded into per-instance constants during
  `generate_network`; it never reaches bngsim.
- **Mangle bngsim-only function aliases to match the BNGL contract.**
  `init_builtins()` registers `ln`, `rint`, `sign`, `mratio`, and
  `time` as ExprTk function aliases. The first four come from
  BNG2.pl's `%functions` and are also rejected by BNG2.pl's parser
  as parameter names — so they never reach bngsim from
  `generate_network`. But `sign` is bngsim-only: BNG2.pl accepts it
  as a user parameter and emits the `.net` cleanly, then bngsim
  aborted at load with a confusing "name 'sign' is already
  registered" error because `sign` was not on ExprTk's
  `reserved_symbols[]` list (so no mangling) yet WAS in bngsim's
  symbol table as a function (so `add_variable` rejected). All five
  names are now in `is_exprtk_reserved()`'s mangling set so user
  parameters with those names register under the mangled key
  `r_<name>`.
- **`NfsimSession.set_param` re-evaluates `<Species concentration="X">`
  before `initialize()` (closes #29).** NFsim's XML loader resolves
  `<Species concentration="X">` and `<RateConstant value="_rateLawN"/>`
  against its parameter map at parse time, then bakes the agent
  population at `prepareForSimulation`. The previous fix for #20 only
  ran after that, so pre-init `set_param` calls landed in the parameter
  map but the agent count was already locked in at the XML-time value —
  models like `scaling_example.bngl` that drop `S0` from 3.31e8 to 1
  via `setParameter` still tried to allocate 3.31e8 agents. The fix
  rewrites a temp XML with override-resolved `<Parameter value="...">`
  values via the existing ExprTk evaluator and points NFsim at that
  copy, so the agent population is correct on first parse — matching
  BNG2.pl's behavior when `setParameter` runs between `generate_network`
  and a `method=>"nf"` simulate. Post-init `set_param` continues to
  refresh reaction rates only (live agent counts unchanged; use
  `add_species`/`remove_species` to mutate the live population).
- **Free `t` as a model identifier (closes #24).** The ExprTk evaluator no
  longer registers `t` as an alias for the simulation-time function — only
  `time()` is reserved, matching BNG2.pl's muParser convention. BNGL models
  that define a parameter, observable, or species literally named `t` (such
  as the `Molecules t counter()` event-counter pattern in the corpus —
  `ATG_model_v12.bngl`, `SIR_v4.bngl`, `SIR_v5.bngl`, ≈43 models in the
  PyBioNetGen-via-bngsim parity sweep) now load and simulate. `time()`
  remains the supported way to read simulation time inside expressions.
- **Mangle ExprTk reserved-word names on registration (closes #18).**
  ExprTk's `add_variable()` rejects names matching its
  `reserved_words[]` / `reserved_symbols[]` tables (`const`, `true`,
  `false`, `sin`, `if`, …). BNGL parameters with these literal names —
  e.g., the `const 1` toggle in the Harmon 2017 model — round-trip
  through BNG2.pl but blocked `Model.from_net()` with "ExprTk: failed
  to register variable '<name>'". `define_variable()` now registers
  reserved names under a mangled `r_<name>` key, a per-evaluator map
  records the rewrite, and `remap_expression()` rewrites references
  in compiled expressions. The map records only names that were
  actually mangled, so built-in tokens (`sin`, `if`, `time`) used in
  user expressions pass through unchanged. The
  duplicate-registration error message also calls out the
  reserved-word case explicitly instead of blaming "case sensitivity".
  This is the foundational mangling machinery the obs() and
  sign-collision fixes above build on.

### Added

- **`NfsimSession.set_param` propagates to dependent parameters via ExprTk**
  (closes #20). The vendored NFsim XML loader records only the precomputed
  `value=` of each `<Parameter>` and discards the `expr=` attribute, so the
  old `set_param` was a flat write to NFsim's `paramMap` — every parameter
  whose BNG XML expression transitively referenced the overridden name
  (e.g., `LT = LT_conc_M*NA*V_sim`, `_rateLawN = kf*(1-use_excess)`) stayed
  pinned at its XML-time value, and downstream tooling had to reimplement
  the BNGL math grammar in Python to scan one parameter. `nfsim_simulator.cpp`
  now parses every `<Parameter id expr value>` from the XML once into a
  dependency-ordered table and re-evaluates the whole table through bngsim's
  ExprTk evaluator on each `set_param`/`clear_param_overrides` call. New
  values are pushed to NFsim's `paramMap` and `updateSystemWithNewParameters`
  cascades through global / composite / local functions and reaction base
  rates. Both pre-init and post-init writes propagate; the post-init silent
  drop is also fixed (the value lands and observation-time `get_parameter`
  reads the new namespace). New `NfsimSession.evaluate(expr, overrides=...)`
  exposes the same evaluator for downstream tools (PyBioNetGen bridge,
  PyBNF) to replace hand-rolled AST walkers — overrides layer on top of
  the simulator's persistent `set_param` state for one-shot probes.

- **Vendored RuleMonkey and exposed exact network-free simulation** as
  `method="nf_exact"` / `method="rulemonkey"` / `method="rm"` alongside the
  existing NFsim backend. BNGsim now builds RuleMonkey in-process from
  `third_party/rulemonkey`, exposes `RuleMonkeySession`, and documents the
  refresh workflow in `bngsim/scripts/RULEMONKEY_VENDORING.md`.

- **Expose `-bscb` and `-utl` on `NfsimSession` and `Simulator`
  (closes #19).** New `block_same_complex_binding: bool = True` and
  `traversal_limit: int | None = None` kwargs on
  `NfsimSession.__init__` and `Simulator(method="nf").__init__`.
  BNGL models commonly request `-bscb -utl N` for correctness on
  aggregation/ring-formation rules (BLBR sweep models); previously
  `NfsimSimulator` hardcoded `blockSameComplexBinding=true` and
  ignored `-utl` entirely, so the PyBioNetGen bridge had nowhere to
  plumb these through. bngsim keeps `-bscb` ON by default
  (deliberately differs from NFsim CLI's off-by-default —
  same-complex blocking is required for correctness on BLBR-style
  models).

- **Forward IC sensitivity API.** New `sensitivity_ic=[species]`
  kwarg on `Simulator`, with `Result.sensitivities_ic` and
  `Result.sensitivity_ic_species` accessors. CVODES seeding extended
  to handle species-IC columns alongside parameter columns; IC
  entries use a sentinel `plist[iS] = n_params` so the codegen
  `bngsim_dfdp` default arm produces `dfdp = 0` and the variational
  ODE collapses to `ds/dt = J*s`. Auto-codegen trigger now also
  fires for IC-only workflows. Becker showcase migrated from
  centered-FD on `init_Epo` / `init_EpoR_rel` to analytic chain rule
  (Epo/EpoR ICs + Bmax + ant_kon + scale_effective): identical χ² in
  ~3.0× less wall time, multistart Jacobian assembly drops from 152
  to 0 `sim_plain` calls. Companion SBML loader fix: detect
  `<initialAssignment>` elements whose math AST is a single bare
  `<ci>` referencing a model parameter and register the
  species/param pair so CVODES seeds `yS_p[species_idx] = 1`.
  Pre-fix, every D2D-style `init_X = 100; species X = init_X` SBML
  model silently returned `dY/d(init_X) = 0` for every `t > 0`.
  Compound IAs like `2 * init_X` are deliberately not handled — use
  `sensitivity_ic=[species_name]` and chain-rule analytically.


- **Analytical sensitivity RHS for SBML/Antimony mass-action models**
  (closes #16). The SBML loader (`bngsim/python/bngsim/_sbml_loader.py`)
  used to walk every kinetic law into a Functional reaction, so
  `prepare_model_codegen` always bailed to RHS-only and CVODES used
  internal FD for sensitivity (paying ~N+1× the per-step cost).
  `_classify_mass_action` now walks the kinetic-law AST as a flat
  product (`AST_TIMES`/`AST_POWER` only) and emits a single Elementary
  reaction when it matches `[c *] [V *] k * x_1^{m_1} * ...` — `c`
  numeric (folded into stat_factor), `V` an optional compartment-volume
  factor, `k` exactly one parameter (or a synthesized derived parameter
  for products of constant params, see below). Per-species multiplicity
  reconciliation (`P_count = c - b + a`, where `a/b/c` are kinetic-law
  / SBML-reactant / SBML-product multiplicities) covers canonical
  patterns where a species appears in the rate but not in
  `<listOfReactants>`: SIR-style infection (`S -> I; beta*S*I`),
  enzyme catalysis (`E + S -> E + P; k*E*S`). Multi-compartment
  reactions accepted when `V_s_factor` (= compartment volume for
  concentration species, 1 for `hasOnlySubstanceUnits` species) is
  uniform across involved species — covers BioModels-style models
  with nominal compartments at V=1 (e.g. `medium`/`cellsurface`/`cell`).

- **Synthesis of derived rate constants from products of constant
  parameters** in the SBML loader (e.g. `kt * Bmax * cell` → derived
  `_rateLaw_<rid>` with `expression = "kt * Bmax"`). Each such derived
  parameter is added via `builder.add_parameter(..., is_expression=True)`
  so the codegen sensitivity RHS chain-rules through it correctly when
  any constituent primary parameter is a sensitivity target.

- **`is_const` and `expression` per-parameter fields** on
  `NetworkModel.codegen_data()` (closes #15). The C++ binding now
  surfaces both, parallel to the .net path's
  `# ConstantExpression` handling, so `generate_sens_from_model` can
  compute `∂p_d/∂primary` via sympy and emit the chain-rule
  contributions in `bngsim_dfdp` for any model whose rate constants
  are derived expressions of primary parameters.

### Fixed

- **CVODES FD on RHS-only codegen** now correctly mirrors `sens_p` into
  the codegen parameter buffer before forwarding to the .so. Previous
  behavior: every sensitivity column came back identically zero for
  any model whose reactions weren't all Elementary
  (cvode_simulator.cpp:635 vs :924 buffer divergence). Surfaced when
  the auto-trigger experiment landed and went hot on Antimony models
  via the now-closed #16.

- **Force editable rebuild on import (closes #23).**
  scikit-build-core's editable hook defaulted to `rebuild=False`, so
  `pip install -e .` / `uv ... --with-editable` loaded a cached `.so`
  after C++ source edits and silently missed new pybind11 bindings.
  Setting `editable.rebuild = true` makes the meta-path finder
  re-invoke cmake on import when sources are dirty. Pairs with a
  CMakeCache-survival fix: `uv`'s default `pip install -e` runs
  configure inside a temporary build-isolation venv that is deleted
  after install, leaving CMakeCache pointing at phantom
  `Python_EXECUTABLE` / `pybind11_DIR` paths. The configure now drops
  stale cache entries when their referenced paths no longer exist
  and re-resolves pybind11 via a candidate list
  (`BNGSIM_PYTHON_EXECUTABLE`, `$VIRTUAL_ENV`, project-local
  `.venv/`, `PATH`).

- **Swallow `tokenize.TokenError` in `_compute_derived_param_jacobian`
  (closes #26).** `tokenize.TokenError` inherits from `Exception`,
  not `SyntaxError`, so the existing
  `except (SyntaxError, TypeError, ValueError, sp.SympifyError)`
  clause let it leak out whenever a derived-param expression
  referenced a parameter named with a Python keyword (most importantly
  `lambda` — sympy's tokenizer enters lambda-expression grammar on
  `lambda *…`). The leak propagated through `generate_rhs_c` /
  `prepare_codegen`, hit the bridge's broad `except Exception`, and
  the model silently fell back to the slow interpreted ODE RHS
  (`scaling_example.bngl` exceeded the 180 s parity-sweep timeout).
  The except clause is widened to `Exception`, returning `None` for
  the affected derived param — the same outcome already produced for
  `if(c, t, f)`-style expressions; codegen continues for the rest of
  the model.

### Changed

- **`Simulator(model, sensitivity_params=[...])` auto-enables codegen**
  regardless of the species threshold the SBML loader uses for plain
  RHS. Sensitivity evaluates the RHS N+1× per step, so codegen pays off
  even on tiny models. The `codegen` kwarg becomes tri-state
  (`bool | None = None`): `None` is auto, `True` is manual `.net` path,
  `False` is explicit opt-out. `from_net` models route to
  `prepare_codegen(net_path)` so the chain rule lands via the .net
  path; everyone else goes through `prepare_model_codegen(model)`.

- **`generate_combined_from_model`** emits combined RHS + analytical
  sensitivity RHS for any all-Elementary model built via
  `ModelBuilder` directly, `from_antimony`, or `from_sbml` — parallel
  to the .net path's `generate_combined_c`. Bumped
  `_CODEGEN_VERSION` 4 → 5; cached `.so` files in
  `~/.cache/bngsim/codegen/` invalidate automatically on first call.

## [0.3.0] — 2026-05-04

### Fixed

- **Forward-sensitivity codegen now propagates the chain rule through
  any derived rate-constant parameter expression**, not just numeric-prefixed
  products. The previous `_parse_param_product` fast-path returned `None`
  for quotients (`5/MEK`, `0.3/TCR`), products of quotients with parens
  (`((kp9/km9)*(kp10/km10))/((kp11/km11)*(kp12/km12))`), and any
  non-`a*b*c`-shaped expression — so codegen sensitivities for the
  *primary* parameters those derived params depend on were silently
  wrong (BNGsim returned the wrong value, not zero, because the direct
  contribution still landed). Fix: replaced `_parse_param_product` with
  sympy-backed `_compute_derived_param_jacobian` (`sympy.parse_expr` +
  `sympy.diff` + `sympy.ccode` + a primary-name → `p[idx]` rewrite).
  Affects any model whose `.net` declares a `# ConstantExpression`
  parameter that is referenced as a reaction rate constant — most
  visibly `tcr_signaling`'s `m1 = 5/MEK`. Verified against
  `bngsim/python/tests/test_codegen_sensitivity.py::TestDerivedQuotientChainRule`
  (new fixture `tests/data/derived_quotient.net`) and the S10 forward
  sensitivity bench (`tcr_signaling` flips from `sens_max ≈ 0.57` to
  `sens=PASS`).

### Changed (cache invalidation)

- Bumped `bngsim._codegen._CODEGEN_VERSION` to `"2"` and mixed it into
  `compute_model_hash` so cached `.so` files in
  `~/.cache/bngsim/codegen/` invalidate automatically on first call
  after upgrade. No user action required; the next codegen call
  re-emits the updated C and recompiles.

### Dependencies

- Added `sympy>=1.10` as a hard runtime dependency.
  Required by `_compute_derived_param_jacobian` to differentiate
  arbitrary derived-parameter expressions and emit the C source for
  `∂p_d/∂primary`. Sympy was previously a transitive dep via AMICI in
  the bench environment but was not declared in `bngsim`'s own deps;
  making it explicit prevents the codegen path from silently
  degrading to "treat derived param as independent" (the pre-fix
  bug) when bngsim is installed into an environment without AMICI.

### S10 forward-sensitivity bench (`bngsim/harness/comparison/bench_forward_sensitivity.py`)

- **IC parameter linkage recovery on both sides of the BNGsim ↔ AMICI
  comparison.** Two complementary asymmetries broke `∂y(0)/∂p` on
  `egfr_path`'s row: `setConcentration` actions in the .bngl preserved
  `species → param` links in the BNG2.pl-emitted .net (BNGsim saw
  them) but the bench's action stripper dropped them when generating
  SBML (AMICI lost them); and `begin seed species` parameter-named
  ICs survived in SBML via libSBML's `<initialAssignment>` but
  BNG2.pl substituted the post-equilibration *literal* into the .net
  (BNGsim lost them). Fix: parse the .bngl seed-species block and the
  .net species/parameter blocks; build a unified
  `{species → IC parameter}` link map; for each pair, inject a
  matching `<initialAssignment>` into the SBML before AMICI compiles
  it (via `libsbml`), and rewrite a temp .net so BNGsim's species
  block expresses the link literally and the parameter block pins
  values to the post-equilibration literal. After `Model.from_net`,
  `set_param` restores nominal parameter values for derived-param
  re-evaluation. New JSON diagnostic field
  `model.ic_link_mismatches` lists any AMICI-only links the bench
  could not recover (empty for all four target models). Verified
  with `dev/repro_sbml_setconcentration_loss.py` (4/4 probes
  `BNG ≈ AMI ≈ 1.0` at `t=0`); the `egfr_path` row flips from
  `sens_max ≈ 3.3e-2` to `sens=PASS` on both `simultaneous` and
  `staggered`.

- **Windowed denominator for the symmetric-relerr xval normalization.**
  Cells that cross zero in an otherwise non-zero sensitivity
  trajectory (e.g. `SHP2_base_model`'s
  `R(DD!1,Y1~P,Y2~P).R(DD!1,Y1~U,Y2~P) / kkin_Y1` near `t≈7.04`) sat
  at CVODES' default-tolerance forward-sensitivity noise floor on
  both engines (~1e-6 absolute), but the previous scalar atol floor
  produced a spurious headline relerr of ~0.30. Fix: the per-cell
  denom now inherits a fraction of its trajectory's neighbouring
  peak — `denom[t,sp,p] = max(|sa|, |sb|, ATOL_REL_WIN ×
  max|sa[t-w:t+w+1, sp, p], sb[t-w:t+w+1, sp, p]|, abs_floor)` —
  with `S9_XVAL_SENS_ATOL_REL_WIN=1e-2` and `S9_XVAL_SENS_WIN_RADIUS=5`
  by default. The absolute floor `S9_XVAL_SENS_ATOL` was bumped
  from `1e-9` to `1e-6` to match the noise floor itself. The
  `thresholds` block in the bench JSON now records both the new
  windowed parameters and the global atol floor for auditability.
  Verified with `dev/repro_zero_crossing_noise_floor.py`
  (worst-cell relerr 0.30 → 5.7e-3); the `SHP2_base_model` row
  flips from `sens_max ≈ 4.7e-2` to `sens=PASS`.

- **Final acceptance.** With this release in place,
  `S10_BNG_CODEGEN_MODE=always bngsim/harness/comparison/bench_forward_sensitivity.py
  --no-sharded` reports `traj=PASS sens=PASS sens_norm=PASS` for all
  four target models (`egfr_path`, `tcr_signaling`,
  `Scaff_22_ground`, `SHP2_base_model`) on both `simultaneous` and
  `staggered` correctors — 8/8 XVAL outcomes. Full diagnosis
  in `dev/report-residual-fwd-sens-bugs.md`.

## [0.2.2] — 2026-05-03

### Tooling (internal)

- Added `python/bngsim/_bngsim_core.pyi` type stubs for the pybind11
  C++ extension. Generated with `pybind11-stubgen` and hand-tightened:
  the JAX Jacobian callback (`SolverOptions.set_jax_jac_fn`) now has a
  precise `Callable[[float, NDArray[float64]], NDArray[float64]] | None`
  signature, and bare `dict` returns (`codegen_data`, `conservation_laws`,
  `set_params`, `to_dict`, `reserved_names`) are typed as
  `dict[str, Any]` / `dict[str, float]` / `dict[str, list[str]]` /
  `dict[str, int]` as appropriate. The stub covers the full public
  surface used by the wrappers (`SolverStats`, `SolverOptions`,
  `TimeSpec`, `ResultCore`, `NetworkModel`, `CvodeSimulator`,
  `SsaSimulator`, `NfsimSimulator`, `ModelBuilder`,
  `SteadyStateOptions`, `SteadyStateResultCore`, plus
  `find_steady_state` and `reserved_names`).
- Re-enabled the `mirrors-mypy v1.13.0` pre-commit hook (scoped to
  `bngsim/python/bngsim/*.py`, with `--ignore-missing-imports
  --follow-imports=silent` and `numpy` as an `additional_dependencies`
  entry so the isolated env sees numpy stubs). Added `[tool.mypy]` to
  `bngsim/pyproject.toml` (Python 3.10 baseline,
  `files = ["python/bngsim"]`).
- Tightened wrapper code so mypy is clean against the stubs:
  `Model.__init__` types `_core` as `NetworkModel`; `Result.__init__`
  types `core` as `ResultCore | None`; `Simulator._sim` is annotated
  as `Any` because the backend is runtime-dispatched between
  `CvodeSimulator`, `SsaSimulator`, and `NfsimSimulator` (a true
  tagged union that mypy cannot narrow on a string key without
  `TypeGuard` plumbing). Added missing `list[str]` / `dict[str, Any]`
  / `dict[int, int]` annotations on previously inferred-empty
  containers in `_codegen.py`, `_net_reader.py`, and
  `_sbml_loader.py`. Corrected
  `_codegen.generate_sens_rhs_c`'s declared return type from `str`
  to `str | None` to match its `return None` fallback path.
- All hooks (`ruff`, `ruff-format`, `clang-format`, `mypy`, hygiene)
  pass on `pre-commit run --all-files`. Full test suite passes
  (563/563).

## [0.2.1] — 2026-05-03

### Fixed

- **`test_cvodes_sensitivity::test_results_similar`** false-positive
  failure (max diff ~232.5). The test reused one `Model` across two
  `Simulator` runs; the first run writes the final-time species state
  back to the model (BNG-style writeback in `cvode_simulator.cpp`,
  intentional and used by multi-action sequences), so the second run
  started from `t = t_end` rather than the original ICs. Both CVODES
  staggered and simultaneous methods are correct; with separate `Model`
  instances they agree to ~2e-5 on `simple_decay`. The test now uses
  two independent models and tightens the threshold from `< 1.0` to
  `< 1e-3`.

- **`harness/comparison/bench_ode_scipy_diffrax.py::run_scipy_bngsim_rhs`**
  referenced an undefined `_compute_rhs_fd` in dead code from an
  abandoned implementation path. The function had switched to a JAX
  RHS at line 103 but left the original `rhs` closure (and an unused
  `Simulator` instance, `n_sp` binding, and explanatory comments)
  dangling. Removed the dead code and corrected the docstring to
  describe what the engine actually does (scipy BDF + bngsim-derived
  JAX RHS evaluated with numpy arrays).

### Tooling (internal)

- Added repository-level `pre-commit` configuration
  (`.pre-commit-config.yaml`) wiring up:
  - **pre-commit stage**: ruff lint+fix and ruff-format on
    `bngsim/`, clang-format on `bngsim/{src,include}/*.{c,h,cpp,hpp,…}`,
    plus standard hygiene hooks (yaml, EOF, whitespace, merge
    conflicts, large files, mixed line endings, debug statements,
    private-key detection).
  - **pre-push stage**: `pytest -q python/tests` from `bngsim/`.
  - Configuration deliberately omits mypy (pybind11 extension
    `_bngsim_core` lacks `.pyi` stubs, which produces ~77 false-
    positive `"object" has no attribute …` errors against
    `model._core`) and clang-tidy (too slow without a configured
    `compile_commands.json`); both are intended for CI.
- Added `bngsim/.clang-format` (LLVM base, 4-space indent, 100-col
  limit) matching the `python-cpp-template` reference.
- Applied `ruff --fix --unsafe-fixes` and `ruff format` across the
  `bngsim/` tree (116 files, ≈4.7k lines net delta). Pure cosmetic
  / lint cleanup — no behavioral changes; full test suite passes
  (563/563).
- Added repository-level `.git-blame-ignore-revs` recording the
  reformat commit so `git blame` skips formatting-only changes.
- Cleaned up the deferred ~100 lint findings in
  `bngsim/{benchmarks,harness}/`: 31× UP031 (printf → f-string in
  `gen_metapop.py`, generator output verified byte-identical),
  28× B023 late-binding closures (bound loop vars as default args;
  none were live bugs), 1× F821 (libsbml type annotation behind
  TYPE_CHECKING), 2× B904, 2× F401 noqa for availability checks,
  6× E402 noqa for after-banner imports, 1× SIM115, 1× E741. Long
  lines in benchmark/harness scripts are exempted from E501 via
  `[tool.ruff.lint.per-file-ignores]` in `bngsim/pyproject.toml`
  (LaTeX captions, multi-line doc strings); notebooks excluded via
  `extend-exclude`. Pre-commit ruff lint scope widened to the full
  `^bngsim/.*\.py$`. Hygiene hooks (EOF, whitespace, mixed-line-
  ending) restricted to source-code extensions and explicitly
  excluded from `bngsim/third_party/` (vendored NFsim) and
  `bngsim/dev/` (informal notes).
- Fixed pre-existing `SyntaxError` in
  `harness/comparison/bench_pythonic_workflows.py`: `global _WARMUP,
  _RUNS` was declared after the names were already read in argparse
  defaults, so the script could not be imported or executed at all.
- Applied first `clang-format` pass across `bngsim/{src,include}`
  (27 files, line wrapping + canonical pointer style + alphabetical
  include ordering). Verified by rebuilding bngsim from these
  sources and re-running the full test suite (563/563 pass).

## [0.2.0] — 2026-05-03

### Fixed

- **CVODES forward sensitivity for derived rate-constant parameters**
  (internal#2). BNG2.pl
  encodes compound BNGL rate laws (e.g., `chi_r1*kon_CSH2`) as derived
  `_rateLaw{N}` `ConstantExpression` parameters. Both the codegen
  analytical sensitivity RHS and the CVODES-internal-FD path treated
  these as independent rate constants, dropping the chain rule through
  the expression. On `SHP2_base_model` this presented as **sign-flipped
  sensitivities** for receptor-substrate complex states with respect
  to `kon_NSH2`/`kon_CSH2`. The S10 cross-validation `sens_max_re`
  was exactly 2.0 (equal magnitude, opposite sign vs AMICI). The fix
  expands product-of-parameters expressions in the codegen `df/dp` and
  re-evaluates `is_expression` parameters in `cvode_rhs` after every
  CVODES `sens_p` sync. After the fix, `sens_max_re` for SHP2 drops
  to ~1.0 and the remaining disagreement is the same Pattern-B
  initial-condition-parameter case other models exhibit.

### Changed

- **`bngsim.jax.differentiable_solve` defaults to primary parameters**
  (breaking). The `params` argument is now sized to
  `model.primary_param_names` (excludes derived `ConstantExpression`
  parameters) and gradients reflect the chain rule through derived
  expressions. Pass `flat=True` to keep the legacy independent-vector
  behavior where every parameter is a separate coordinate. Models with
  no derived parameters (`simple_decay`, `two_species_reversible`,
  `fixed_species`, …) are unaffected because primary == all.

### Added

- `Model.param_is_expression` — list[bool] parallel to `param_names`,
  flagging derived `ConstantExpression` parameters.
- `Model.primary_param_names` — convenience accessor returning only
  non-derived parameter names. Recommended input to external optimizers.
- Regression tests `TestDerivedRateConstantSens` (sensitivity correctness
  for derived rate constants) and `TestPrimaryParamsDefault` /
  `TestFlatLegacyMode` (JAX bridge primaries-only and `flat=True` paths).
- `tests/data/derived_rate_const.net` — minimal synthetic model
  reproducing the issue #2 pattern.

## [0.1.0]

Initial development version. No public release.
