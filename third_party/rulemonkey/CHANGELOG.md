# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.5.0] — 2026-06-22

### Added

- **`simulate(const TimeSpec&)` on the stateful session API.** The session now
  has an explicit-`TimeSpec` overload alongside
  `simulate(t_start, t_end, n_points)`, so a live, continuing session can
  record at exactly `TimeSpec::sample_times` (the arbitrary, possibly
  non-uniform output instants honored by `run_ssa` since issue #16) instead of
  only a uniform grid. It is the stateful counterpart of `run(TimeSpec)`: it
  continues from the current session state (no re-seed / reset) and routes
  through the same `run_ssa` path. The segment must start at the current
  session time (`sample_times.front()`, or `t_start` when `sample_times` is
  empty, must equal `current_time()`); a disagreeing start throws, mirroring
  the convenience overload. This lets in-process hosts (e.g. BNGsim) sample a
  network-free protocol at a dataset's exact time points mid-run. Covered by
  `session_sample_times_test` (bit-identical to a uniform session run at the
  shared instants, mid-protocol continuation, start-alignment validation).

## [3.4.0] — 2026-06-20

### Added

- **`FunctionProduct` rate laws (NFsim's DOR2).** RuleMonkey now runs
  rules whose rate is `RateLaw type="FunctionProduct"` — the per-instance
  product of two per-reactant local-function factors, each evaluated in
  the context of a different tagged reactant (what BNG2 emits for
  `%x:A(..) + %y:B(..) -> ... FunctionProduct("f1(x)", "f2(y)")`). RM
  previously refused these at Tier-0, which broke `method=>"nf"` ↔
  `method=>"rm"` interchange for the ~6–8% of network-free corpus models
  that use the idiom. The propensity is realized as `S1·S2`, where
  `S1 = Σ_a w_a·f1(a)` over reactant-A matches and `S2 = Σ_b w_b·f2(b)`
  over reactant-B matches — the local-rate analogue of an ordinary
  bimolecular rule's `a_eff·b_eff·k`. Each reactant draw is weighted by
  its own factor, and same-molecule pairs reject as null events (exact for
  distinct-type reactants, statistically exact otherwise), matching
  NFsim's `DOR2RxnClass`. Each factor honors per-molecule or complex-wide
  scope independently. Validated against NFsim 2.9.3 (300–400-rep
  ensembles) on the issue reproducer, a pure-binding isolation, and a
  discriminating methylation+unbinding model — all within SSA noise
  (maxAbsDiff < 1.1, RMSE < 0.5). Adds the `nf_function_product`
  feature-coverage model (NFsim-only reference).
  Closes [#19](https://github.com/richardposner/RuleMonkey/issues/19).

### Fixed

- **Local observables were read globally during initial propensity
  setup.** `local_obs_indices` (the set of observable slots a local
  function localizes) was populated *after* the first per-rule propensity
  rescan in `init_rule_states`, so at `t=0` a local-function rule's
  observables were evaluated over the whole system instead of the tagged
  molecule's scope (e.g. a free monomer's local count read as the
  system-wide count), giving a wildly wrong initial propensity until the
  first incremental update partially corrected it. Surfaced by the
  `FunctionProduct` transient; the fix (computing `local_obs_indices`
  before the rescan loop) also hardens the single-local-function path.

## [3.3.0] — 2026-06-19

### Added

- **Explicit output times: `TimeSpec::sample_times`.** `run()` (and any
  path through `Engine::run_ssa`) can now record output at an explicit,
  possibly non-uniform list of instants instead of the uniform
  `t_start..t_end / n_points` grid. Set the new
  `std::vector<double> TimeSpec::sample_times` to a sorted-ascending
  vector and it overrides `n_points`; `Result::time` then echoes the
  requested list, one row per instant. This is the network-free analogue
  of BNG2.pl's `simulate_nf` `sample_times` branch — the motivating use is
  recording at an experimental dataset's time points so an embedder
  (PyBNF via [bngsim](https://github.com/lanl/bngsim)) can fit
  against them directly, without the per-segment `step_to` workaround that
  rebases the propensity sum and drops `event_count`. Sampling is
  non-invasive: output times never draw from the RNG or perturb reaction
  selection, so a run with explicit `sample_times` is bit-identical to the
  uniform-grid run at any shared instants and `event_count` is unchanged.
  `t_end` still bounds the SSA loop; times at/after `t_end` (or below
  `t_start`) are recorded at the final/initial state, and an out-of-order
  list throws `std::runtime_error`.
  Closes [#16](https://github.com/richardposner/RuleMonkey/issues/16).

## [3.2.1] — 2026-05-23

### Fixed

- **Exact-species pattern methods are now component-order-insensitive.**
  The `#9` session methods `get_species_count` / `remove_species` /
  `set_species_count` matched a species only when its components were
  written in RuleMonkey's own canonical order; a semantically identical
  pattern with the components swapped (e.g. `X(p~0,y)` for the canonical
  `X(y,p~0)`) silently matched nothing and returned 0. The `add_species`
  path, by contrast, already placed components by molecule-type
  declaration index, so it *did* canonicalize — leaving
  `set_species_count` with a non-canonical pattern diffing against a
  wrong baseline of 0 and overshooting its target. The fix routes the
  match path through the same declaration-order placement
  (`pattern_to_complex_graph` now orders each molecule's components by
  `comp_type_index` and remaps bond endpoints accordingly), matching
  `extract_complex` and NFsim's order-insensitive exact-species matcher.
  Closes [#13](https://github.com/richardposner/RuleMonkey/issues/13);
  follow-up to
  [#9](https://github.com/richardposner/RuleMonkey/issues/9).

## [3.2.0] — 2026-05-18

### Added

- **Parameter sweeps: `parameter_scan` and `bifurcate`.**
  `RuleMonkeySimulator` gains two methods — `parameter_scan(ScanSpec,
  seed)` and `bifurcate(ScanSpec, seed)` — the RuleMonkey equivalents of
  BioNetGen's `parameter_scan` and `bifurcate` actions. A sweep runs the
  model at each value of one parameter (an explicit value list, or a
  linear / geometric `min`/`max`/`n_points` range) and records the
  *endpoint* observable and global-function values, matching BNG's
  extraction of the last `.gdat` row per run. `parameter_scan` with
  `reset_conc=false` and `bifurcate` carry molecular state over between
  points; `bifurcate` runs the forward and backward sweeps as one
  continuous trajectory so a bistable model surfaces hysteresis. New
  `ScanSpec` / `ScanResult` / `BifurcateResult` types in `types.hpp`. A
  new `rm_scan` command-line tool exposes both modes and writes the
  result in tab-separated `.scan` format on stdout (function columns
  gated behind `--print-functions`, mirroring
  [#7](https://github.com/richardposner/RuleMonkey/issues/7); see
  [`docs/scan_format.md`](docs/scan_format.md)). Closes
  [#8](https://github.com/richardposner/RuleMonkey/issues/8). SSA
  trajectories and existing output are unchanged; the header-only ABI
  change means consumers must rebuild against the new headers.

- **Global-function values in the public API.** `rulemonkey::Result`
  now carries `function_names` and `function_data` alongside
  `observable_names` / `observable_data`, populated at every output time
  point; `function_data` is column-major (`function_data[fn_idx][t_idx]`)
  and parallel to `observable_data`. `RuleMonkeySimulator` gains
  `function_names()` (XML declaration order, captured at construction)
  and `get_function_values()` (live-session readback, mirroring
  `get_observable_values()`). These expose the BNGL `begin functions`
  entries — the derived quantities models commonly use as their
  measured/fitted outputs (e.g. `Clusters() = monomer + dimer + …`) —
  which the engine already evaluates internally for rate laws. Only
  *global* (non-local) functions are surfaced; local functions evaluate
  per-molecule and have no single global value, so `function_names` may
  be shorter than the model's full `begin functions` block. The API
  surface is unconditional. Closes
  [#7](https://github.com/richardposner/RuleMonkey/issues/7). SSA
  trajectories and observable output are unchanged; the header-only ABI
  change means consumers must rebuild against the new headers.

- **`rm_driver --print-functions`.** A new opt-in flag that appends the
  model's global-function values as trailing `.gdat` columns (after the
  observables). Off by default, mirroring BNGL's `print_functions=>1`:
  the default `.gdat` stays observables-only and byte-identical to what
  earlier RM versions emitted. The flag governs only `rm_driver`'s text
  output — the in-process `Result` API exposes the values regardless.

- **`tests/cpp/function_values_test.cpp`** — regression test for the new
  function surface: `function_names()` declaration order, the
  column-major shape of `Result::function_data`, per-sample algebraic
  consistency with the observables each function derives from (covering
  nested function-of-function settle order), live `get_function_values()`
  readback against `get_observable_values()`, the no-session throw, and
  the empty-not-absent function surface of a model with no functions.

- **Cooperative cancellation hook on `run()` / `simulate()` / `step_to()`.**
  Each of the three public entry points now accepts an optional
  `rulemonkey::CancelCallback` (a `std::function<bool()>`) that the SSA
  event loop polls roughly every 1024 events; returning `false` raises
  `rulemonkey::Cancelled` (a `std::runtime_error` subclass) at a safe
  between-event point.  Empty callbacks disable polling and pay no
  per-event overhead.  This unblocks the BNGsim `timeout` kwarg for the
  RuleMonkey backend (closes
  [#3](https://github.com/richardposner/RuleMonkey/issues/3)); the prior
  workaround of wrapping each evaluation in a subprocess can now go
  away.  Source-compatible — existing callers see only the defaulted
  parameter — but mangled-name ABI changes, so consumers must rebuild
  against the new headers.

- **`tests/cpp/cancellation_test.cpp`** — regression test for the four
  behavioral contracts the new hook adds: pre-cancelled callback throws
  on entry, `Cancelled` inherits `std::runtime_error`, mid-session
  `simulate()` cancellation leaves the session live with
  `current_time()` strictly inside the requested window and is
  recoverable via `destroy_session()` + re-`initialize()`, and an
  always-true callback produces a bit-identical trajectory to the
  no-callback path.

- **Species enumeration, canonical complex labeling, and `.species`
  output.** RuleMonkey can now enumerate the distinct chemical species
  in the live pool by graph isomorphism. A new DIY canonical-labeling
  core (`cpp/rulemonkey/canonical.{hpp,cpp}` — 1-WL color refinement
  plus individualization–refinement for symmetric residue such as rings
  and homo-oligomers; no nauty/bliss, preserving the cleanroom property)
  assigns each complex a canonical normalized-BNGL label.
  `RuleMonkeySimulator` gains `enumerate_species()` (returns `SpeciesRow`
  records — a new type in `types.hpp`), `write_species_file(path)`
  (BNG-format `.species` output, live species only, NFsim `-ss` parity —
  see [`docs/species_format.md`](docs/species_format.md)),
  `species_count(canonical_species)`, and `total_complex_count()`. A new
  `rm_driver --species <path>` flag writes the `.species` file from the
  command line. A cached-incremental labeling mode (per-complex cached
  label with dirty-bit invalidation in the structural mutators) is built
  and validated by a Debug/ASan-build invariant — cached label equals a
  from-scratch recompute, gated by the `RULEMONKEY_CANONICAL_CACHE_SELFCHECK`
  compile definition — awaiting its downstream consumer. Closes
  [#9](https://github.com/richardposner/RuleMonkey/issues/9) §2. New
  ctest cases `canonical_test`, `species_enumeration_test`. Header-only
  ABI change — consumers must rebuild against the new headers.

- **Session API: live expression evaluation and pattern-keyed species
  methods.** On an active session, `RuleMonkeySimulator` gains
  `evaluate_expression(expr, extra)` — compiles and evaluates an
  arbitrary BNGL expression against the live session (parameters,
  observables, global functions, and `time()`/`t`; an optional `extra`
  map shadows those names on clash) — and four pattern-keyed species
  methods, `get_species_count` / `add_species` / `remove_species` /
  `set_species_count`, each taking a BNGL species-pattern string. A new
  runtime BNGL species-pattern parser (`cpp/rulemonkey/pattern_parser.{hpp,cpp}`)
  backs the latter four: it accepts exact, fully-specified, connected
  species (every component listed, stateful components with a concrete
  `~state`, numeric bonds) and rejects partial patterns (`!+` / `!?` /
  omitted components). `get_species_count` canonicalizes the parsed
  species and reuses the `species_count` lookup above;
  `add_`/`remove_`/`set_` resync all rule propensities after the
  structural change. Closes
  [#9](https://github.com/richardposner/RuleMonkey/issues/9) §1 (and,
  with §2 above and §3 — which needed no work — issue #9 in full). New
  ctest cases `evaluate_expression_test`, `pattern_parser_test`,
  `species_methods_test`. Header-only ABI change — consumers must
  rebuild against the new headers.

### Changed

- **Expression evaluator: hand-rolled parser replaced with ExprTk.** The
  BNGL rate-law / function / parameter math evaluator (`expr_eval`) is now
  [ExprTk](https://www.partow.net/programming/exprtk/), via the
  `bngsim::ExprTkEvaluator` wrapper RuleMonkey shares with its BNGsim
  integration host. All four expression consumers — global functions,
  rate-law ASTs, the simulator parameter cascade, and local functions —
  moved at once; the hand-rolled recursive-descent parser and `AstNode`
  tree-walker are gone. Expression evaluation is ~16–30% faster per call
  on function-rate models (no effect on mass-action); SSA trajectories
  are bit-identical to 3.1.x. Closes
  [#6](https://github.com/richardposner/RuleMonkey/issues/6). No public
  API or header change. Build note: ExprTk is vendored under
  `third_party/` and compiled only in a standalone build — a CMake gate
  (`if(TARGET bngsim::expression)`) links the host's copy inside a BNGsim
  build instead. `scripts/vendor_exprtk.py --check` guards the vendored
  copy against drift from its pinned BNGsim commit.

- **CMake vendoring defaults.** The minimum CMake version is now 3.20.
  `RULEMONKEY_BUILD_TESTS` and `RULEMONKEY_BUILD_CLI` default to
  `PROJECT_IS_TOP_LEVEL`, `RULEMONKEY_WARNINGS_AS_ERRORS` defaults off
  when RuleMonkey is added as a subdirectory, and tests no longer depend
  on `CMAKE_SOURCE_DIR`.

- **Local-function rate laws: redundant per-molecule observable
  re-evaluation eliminated.** On models with local-function rate laws,
  `evaluate_local_rate` recomputed each rule's local observables from
  scratch (`count_embeddings_*`) for every affected molecule on every
  event — up to ~75% of wall time on local-function-heavy models.
  `evaluate_observable_on` now routes tracked `Molecules`-type
  observables through the per-molecule `obs_mol_contrib` table that the
  species-observable incremental machinery already maintains and
  refreshes before the propensity recompute each event: per-molecule
  scope becomes a table read, complex-wide scope a sum over the complex
  — no embedding counts. A from-scratch recompute remains as a
  bounds-checked fallback, and a Debug/ASan-build invariant (gated by
  the `RULEMONKEY_LOCAL_OBS_SELFCHECK` compile definition) cross-checks
  the fast path against it. Wall-time reductions: `isingspin_localfcn`
  71%, `ANx` 20%, `AN` 16%, `t3` 9%. Closes
  [#10](https://github.com/richardposner/RuleMonkey/issues/10). No
  public API or header change; SSA trajectories are bit-identical.

## [3.1.2] — 2026-05-02

### Added

- **`docs/internals.md`** — engine-internals reading guide for
  contributors about to modify `cpp/rulemonkey/engine.cpp`.  Covers
  the SSA event loop, the three pattern-matching layers
  (`count_embeddings_single`, `count_multi_mol_fast`,
  `count_2mol_1bond_fc`), complex tracking on bind/unbind, propensity
  computation and `incremental_update`, the 2-mol/1-bond fast-path
  specialization, `fire_rule`'s OpType switch, and the five
  `select_reactants` paths.  Cites engine.cpp line ranges as anchors.

- **"Adding a new profile" recipe in `engine_profile.hpp`.** Five
  mechanical steps to wire a gate, struct, member, increment site,
  and report function for a new hot path.  Existing per-profile gate
  comments and field-level documentation were already strong; the
  missing piece was a contributor recipe.

- **`tests/cpp/error_paths_test.cpp`** — pins down that the
  documented public-API error surfaces throw `std::runtime_error`
  (not `std::exception`, not silent failure) for: missing XML file,
  malformed XML, unknown `set_param` name, and the four mutators
  that reject calls while a session is active (`set_param`,
  `clear_param_overrides`, `set_molecule_limit`,
  `set_block_same_complex_binding`).  Previously these paths
  existed in `simulator.cpp` but were only exercised indirectly by
  the corpus parity tests.

- **`harness/perf_diff.py`** — diffs per-model wall-time between two
  `feature_coverage_report.md` files.  Sorts by absolute `Δ%`;
  flags ±15% as `SLOWER` / `FASTER`; marks `NEW` / `GONE` for
  models present on only one side.  Companion `.github/workflows/perf-diff.yml`
  runs the full feature_coverage benchmark on both PR base and HEAD
  on the same runner (controls hardware variance) and uploads the
  diff as an artifact.  Not a hard gate — shared GitHub runners are
  noisy enough that single-model deltas of 30%+ come from
  neighbour-VM contention rather than real regressions.

### Changed

- **`-Werror` is on by default** for the in-tree build, gated by
  `RULEMONKEY_WARNINGS_AS_ERRORS=ON`.  Default ON so a stray warning
  shows up on the developer's machine before it lands in CI.
  Downstream consumers building RM as a subdirectory or against an
  installed package can opt out with
  `-DRULEMONKEY_WARNINGS_AS_ERRORS=OFF` if their toolchain flags
  things ours does not.  Verified clean against AppleClang 17;
  CI exercises Linux clang and gcc.

- **CI `asan` job is now a Linux + macOS matrix.** Same code, same
  compiler family (clang), but different stdlib (libstdc++ vs
  libc++) and different sanitizer-runtime image — exactly the
  divergence that hides UB on one platform and reveals it on the
  other.  The CI step sets
  `UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1` and
  `ASAN_OPTIONS=detect_leaks=0` to keep diagnostic output uniform
  across platforms.

## [3.1.1] — 2026-04-30

### Added

- **Schema fingerprint on `save_state` / `load_state`.** The pool
  serialization keys molecule and component instances by integer
  indices into `Model::molecule_types` and `MoleculeType::components`,
  so a state file written against one XML can be read against a
  structurally different XML without runtime errors but with every
  index referring to a different schema slot — silently corrupt
  trajectories.  `save_state` now embeds an FNV-1a 64-bit hash over
  the canonical schema text (molecule type names, ordered component
  names, ordered allowed states); `load_state` recomputes the hash
  and throws on mismatch with both digests in the error message.
  Parameter values, rate constants, and seed species do NOT
  participate, so a checkpoint can still resume with mutated
  `set_param` overrides.  State file marker bumped `RM_STATE_V1` →
  `RM_STATE_V2`; V1 files are refused explicitly with a "re-save
  with this build" message.

- **`tests/cpp/save_load_test.cpp`** — new ctest suite covering
  (a) round-trip equivalence of split-run continuation vs an
  uninterrupted run, (b) fingerprint-mismatch refusal across two
  structurally different XMLs, (c) explicit V1 marker rejection.

- **Regression coverage in `tests/cpp/set_param_test.cpp`** and a new
  `tests/cpp/derived_param_model.xml` fixture — two-layer derived-
  parameter chains (`A_tot = A_base * A_factor`,
  `kp = kp_base * kp_mult`) plus get_parameter-coherence and
  unknown-name-rejection cases. Each new assertion was verified to
  fail on the unfixed engine via stash-and-rerun.

### Changed

- **`set_param` validates parameter names against the loaded XML.**
  Typo'd names previously leaked into `param_overrides` as silent
  no-ops; they now throw `std::runtime_error("Unknown parameter
  '...'")`. Mirrors `get_parameter`'s existing throw on unknown
  names.

- **`expr_eval` builtin dispatch returns `std::optional<double>`.**
  `eval_builtin` previously used `quiet_NaN()` as the
  no-signature-matched sentinel, which the caller couldn't
  distinguish from a legitimate NaN result on out-of-domain input
  (`pow(-2, 0.5)`, `acos(2)`, `log10` / `log2` / `atan2` of
  out-of-domain input).  An ad-hoc allowlist on `{log, ln, sqrt}`
  preserved their NaN results; every other builtin's NaN fell
  through to variable lookup and surfaced as a confusing
  "unknown function 'pow'" error.  Switched the return type to
  `std::optional<double>`: a signature match returns the math
  result (NaN included), no match returns `nullopt`.  The caller
  now throws "wrong arity for builtin 'X'" rather than falling
  through, eliminating the NaN-as-sentinel pattern.

- **`CountMultiProfile` and `CmmFcProfile` are per-Engine.**
  These two profile structs (whose call sites are static free
  functions with no Engine pointer in scope) were file-scope mutable
  globals — first plain `inline`, then briefly `inline thread_local`
  as a one-keyword race fix.  TLS eliminated the concurrent-write
  race but preserved a separate cross-Engine accumulation issue:
  Engine B's report on the same thread would include Engine A's
  counts.  Now `CountMultiProfile* cm_prof` / `CmmFcProfile* fc_prof`
  are threaded through `count_multi_mol_fast`,
  `count_multi_mol_fast_generic`, and `count_2mol_1bond_fc`; both
  structs live as `Engine::Impl` members alongside the other six
  profile structs.  `report_count_multi(p)` and
  `report_cmm_fc(q, cm)` take their data by const reference.  Result:
  per-Engine reports show only that Engine's contributions even
  under BNGsim's ThreadPoolExecutor + PyBNF integration target where
  multiple `RuleMonkeySimulator` instances run concurrently in one
  process.  Default-build runtime cost: zero (every increment is
  inside `if constexpr (k*Profile)` and dead-strips).
  Dev-build runtime cost: one extra pointer in the call signatures.
  Output format unchanged.

- **Parameter forward-reference resolution iterates to fixed point.**
  `load_model` previously did a single retry pass after the initial
  resolution pass, which handled at most one level of forward
  reference.  BNG2 emits parameters in dependency order so this
  never bit in practice, but arbitrary XML declaring
  `P3 = 2*P2; P2 = P1; P1 = 1` in that order would have left `P3`
  stale.  Now iterates until either every value is stable or the cap
  (param-count + 4) is hit.

- **CONTRACT comment on `*_expr` fields in `model.hpp`.**  Each
  symbolic-source field (`rate_expr`, `mm_kcat_expr`, `mm_Km_expr`,
  `concentration_expr`) now carries an inline note saying it must be
  re-resolved in `RuleMonkeySimulator::Impl::apply_overrides`.
  Defensive against future parser extensions that add a 5th
  parameter-derived field and forget to wire it into the override
  cascade — exactly the regression that the 3.1.x `apply_overrides`
  fix was designed to close.

- **`asan` preset disables `RULEMONKEY_INSTALL`; CMakeLists refuses
  `RULEMONKEY_INSTALL=ON` with `RULEMONKEY_ENABLE_ASAN=ON`.**  The asan
  flags are PRIVATE compile / PUBLIC link on the `rulemonkey` target,
  so an asan-instrumented installed archive would propagate
  `-fsanitize=address,undefined` to every downstream
  `find_package(RuleMonkey)` consumer's final link line — failing
  outright unless the consumer also enables asan, and producing a
  binary that runs against the wrong runtime even when it links.
  Asan is dev-only; the supported `asan` preset now auto-disables
  install, and the build refuses the dangerous combo with an
  actionable error message pointing at `-DRULEMONKEY_INSTALL=OFF`.

- **Tier-0 refusal error strings in `scan_unsupported` no longer say
  "RM v1".**  The Tier-0 refusals for multi-mol / duplicate-type
  Fixed species emitted "RM v1 only supports …" / "RM v1 allows at
  most …" — a reader could mistake "v1" for RuleMonkey 1.x rather
  than the scope of the Fixed-species feature.  Tightened to "RM
  currently …" in the user-visible strings, the supporting comments
  in `simulator.cpp` and `model.hpp`, and the `edg_fixed_competition`
  test fixture comment (the docs-side change landed in the prior
  cycle).

- **`docs/model_semantics.md`**: new "Parameter overrides" section
  documents the cascade behavior and the unknown-name rejection;
  the parameter-resolution paragraph now correctly distinguishes
  parse-time fixed-point iteration from `apply_overrides`'s
  single-pass cascade.

- **Cross-references throughout the repo**: dropped wrong-direction
  pointer from `harness/benchmark_full.py` at "docs/FAILING_MODELS.md"
  (the canonical tier list is the inline `SMOKE_MODELS` /
  `GUARD_MODELS` Python lists), repointed three references to the
  out-of-tree `compute_noise_floor.py` at the artifact
  (`tests/reference/nfsim/noise_floor.tsv`) and its `PROVENANCE.md`,
  fixed a `docs/sprint_basicmodels_failures.md` path that lived at
  `dev/`, and replaced a `harness/dev/` pointer in
  `docs/timing_comparison.md` with a pointer at the actual
  implementation (`init_incremental_observables` /
  `flush_species_incr_observables` in `cpp/rulemonkey/engine.cpp`).

### Removed

- **All references to the `nfsim-rm` development repository.**  That
  repo is being archived; this repo supersedes it.  Removed
  hardcoded `~/Code/nfsim-rm/build/NFsim` defaults from three
  harness scripts (`benchmark_feature_coverage.py`,
  `benchmark_rm_vs_nfsim_timing.py`, `generate_basicmodels_refs.py`)
  — `NFSIM_BIN` must now be set explicitly when regenerating
  references.  Stripped `nfsim-rm` cross-references from
  `harness/benchmark_full.py`, `tests/reference/basicmodels/PROVENANCE.md`,
  and `tests/reference/nfsim/PROVENANCE.md`.  The "Regen tooling"
  section of the latter is rewritten to honestly state that the
  regeneration scripts are not currently in this repo's tree, and
  that `mean.tsv` / `std.tsv` / `tint.tsv` / `noise_floor.tsv` are
  treated as frozen artifacts.

- **Dead pytest config in `pyproject.toml`.** The
  `[tool.pytest.ini_options]` block declared
  `testpaths = ["tests", "harness"]`, but `tests/` is C++-only and
  `harness/` holds benchmark/research scripts — no `test_*.py`,
  `*_test.py`, or `conftest.py` exists anywhere.  Running `pytest`
  from the repo root collected zero tests, which is a misleading
  signal for an external reader.  Removed the config block and the
  unused `pytest>=8.0` dev dependency; the suite is C++ ctest plus
  harness-driven Python scripts.

- **Dead helper functions and locals in `simulator.cpp`.**
  Three unused free functions in the XML-parser anonymous namespace
  (`need_child`, `has_attr`, `any_rule_has_child`) and two unused
  local variables (`rp_start_0`, `rp_start_1` inside the
  same-components detection) had been triggering compiler
  `-Wunused-function` / `-Wunused-variable` warnings on every clean
  build.  Removed; the library now compiles warning-free under both
  the `release` and `asan` presets.

### Fixed

- **`save_state` / `load_state` had no XML-mismatch guard.** The
  public docstring promised "the model XML must match the one used
  to save the state" but nothing enforced it; loading state from a
  structurally different XML produced silently corrupt trajectories
  rather than an error.  See "Schema fingerprint" under Added above
  for the fix.

- **Null-deref window in `parse_pattern` species-bonds fallback.**
  `<Species>` parsing fell back to `find_child(*mol_list,
  "ListOfBonds")` when no top-level `<ListOfBonds>` existed, but
  dereferenced `*mol_list` unconditionally — a degenerate
  `<Species>` without `<ListOfMolecules>` would null-deref.  BNG2
  doesn't emit such species, but hand-crafted XML or a future
  emitter could trip it.  Guarded with `if (!bl && mol_list)`.

- **Confusing arity-mismatch errors from `expr_eval` builtins.**
  See "expr_eval builtin dispatch" under Changed above.

- **`get_parameter` returned the parsed-at-load value between
  `set_param` and the next `run()` / `initialize()`.** The
  override map only synced into `model.parameters` inside
  `apply_overrides()`, which only ran on session start. Embedders
  querying overrides via `get_parameter()` between runs saw stale
  values. `set_param` and `clear_param_overrides` now invoke a
  light `sync_parameters()` so the public read is coherent
  immediately.

- **Derived parameter expressions (`B = 2*A`) did not cascade
  through `set_param` overrides.** Parameter values were resolved
  once at XML parse time and the override map only splatted the
  literal name's value, leaving downstream derivations frozen at
  their parsed numeric. Captured the symbolic expression for each
  declared parameter at parse time (`Model::parameter_exprs`) and
  re-cascade in declaration order inside `sync_parameters()` so
  set_param on a base parameter propagates to every derived
  parameter that references it. Override on a derived parameter
  still wins (skips the expression for that name).

### Fixed (docs)

- **CHANGELOG 3.0.0 model count off by one.** The 3.0.0 entry said
  "51 BNGL feature-coverage models"; `git ls-tree v3.0.0 --
  tests/models/feature_coverage/` counts 52 `.bngl` files.  Corrected.

- **README pointer to the find_package / add_subdirectory snippet.**
  The "Embedding (C++ API)" section pointed readers at
  `examples/CMakeLists.txt` for the CMake consumption snippet, but
  that file is two lines (`add_executable` + `target_link_libraries`).
  The actual snippets live in the doc-comment header of
  `examples/embed.cpp`.  Updated the pointer.

- **`docs/model_semantics.md` "RM v1" wording.**  The Tier-0 refusals
  table for multi-molecule and duplicate Fixed species said "RM v1
  only supports …" / "RM v1 allows at most …".  RM is at 3.x; a
  reader could mistake "v1" for RuleMonkey 1.x rather than the v1
  scope of the Fixed-species feature itself.  Wording tightened to
  "RM currently …" (this cycle extended the same tightening to the
  user-visible error strings — see Changed above).

## [3.1.0] — 2026-04-29

### Fixed

- **Self-binding propensity under bscb under-counted by `(N-1)/N`**
  (`9f25bba`). `compute_propensity` was subtracting within-molecule
  pair contributions (`extra_eff`) at propensity time AND
  `select_reactants` was rejecting `mol_a == mol_b` at sample time —
  the same self-pairs were removed twice. Hetero-binding was
  unaffected (`extra_eff` is naturally zero across distinct molecule
  types); the bias only appeared on `A+A` shapes (both symmetric
  `A(c)+A(c)` and asymmetric same-type `A(l)+A(r)`). Hidden under
  the standard 20-rep 5σ benchmark noise floor; emerged at 100+
  reps. RM-vs-NFsim avg-z dropped from ~−1.0 to ~−0.2 across the
  affected models post-fix; vs deterministic ODE on
  `combo_addbond_connected`, RM moved from z=−3.21 to z=−1.64 (now
  better than NFsim's z=−2.02). Surfaced by the new `edg_*` stress
  suite at 500 reps. Same bias also affected the existing
  `combo_addbond_connected` corpus model.

- **MM(kcat, Km) rate law was silently zero-rate** (`6421017`).
  The XML loader parsed `RateLawType::MM` and stored `mm_kcat` /
  `mm_Km` on the rule, but the engine never read those fields —
  every MM rule ran at zero propensity. Confirmed live: an MM rule
  on a 100-substrate / 5-enzyme system left S unchanged at 100
  forever in RM, while NFsim depleted S to ~10 by t=30. Now uses
  NFsim's QSS formula `sFree = 0.5·((S−Km−E) + √((S−Km−E)² + 4·Km·S))`,
  `a = kcat·sFree·E/(Km+sFree)` (mirrors `MMRxnClass::update_a` in
  the NFsim source). Verified at 50 reps each: RM and NFsim agree
  within stochastic noise.

- **TFUN sentinel substitution + .tfun file resolution**
  (`cb03071`). Two bugs in RM's TFUN code path, both surfaced by
  the new `ft_tfun.bngl` test:
  - Sentinel name mismatch: RM looked for `__TFUN_VAL__` (single
    underscore) but BNG2 2.9.3 emits `__TFUN__VAL__` (double
    underscore). The dev branch `fix-tfun-has-tfuns-reset` reverts
    to the single-underscore form for the new lowercase `tfun()`
    syntax. RM now accepts both, longer match first.
  - File-resolution path: RM only searched relative to the XML
    directory; the harness lays out XML in `tests/.../xml/` with
    the `.tfun` next to the source BNGL one level up. RM now falls
    back to `<xml_dir>/..` so author-side and harness-side
    layouts both work.

- **Multi-mol Molecules observable counts on palindromic patterns
  with symmetric components** (`5d90724`). The BFS in
  `count_multi_molecule_embeddings` committed to the first valid
  partner embedding consistent with the walked bond and silently
  `count_multi_molecule_embeddings` committed to the first valid
  partner embedding consistent with the walked bond and silently
  dropped sym-equivalent alternatives. On a 5-mol palindromic
  observable like `B(c!1).R(b!1,a!2).A(r!2,r!3).R(b!4,a!3).B(c!4)`
  with a sym partner reached via a non-sym bond endpoint (the
  basicmodels v07 / r07 shape), this under-counted by exactly the
  partner's symmetry factor. Replaced the BFS body with a recursive
  enumerator that branches over every partner embedding consistent
  with the walked bond. Untouched: rule rates (different code path),
  `Species` observables (deduped by complex anyway), single-mol
  observables, and the 2-mol-1-bond fast path.

- **Multi-mol unimolecular rule rates over-counted on hosts with
  symmetric components** (`7472a07`). `count_multi_mol_fast` (the
  generic path; the 2-mol-1-bond specialization is gated off for
  this shape) called `count_embeddings_single` for the seed without
  `reacting_local`, so injective embeddings differing only in
  non-reacting sym slots were never deduped. On the basicmodels
  v02 / r02 unbind rule `X(y!1, p~0).Y(x!1) -> X(y, p~0) + Y(x)`
  this fired at 2× the BNG2-strict rate. Threaded `reacting_local`
  through `count_multi_mol_fast` → `count_multi_mol_fast_generic` →
  the seed-side `count_embeddings_single` call. Phos remains
  correct (mult=2): StateChange targets the matched p, so the two
  sym embeddings produce different keys under dedup and stay
  distinct.

- **Compile-time embedding correction was double-applied for
  multi-mol patterns after the seed-dedup fix** (`3423b0d`). The
  previous fix subsumed the work `compute_embedding_correction_multimol`
  was doing, so applying both halved the rate when the pattern had
  sym non-reacting components (the basicmodels v18 / r18 shape).
  Set `embedding_correction_a/_b = 1.0` for multi-mol (mirroring
  single-mol, which always did this). Removed
  `compute_embedding_correction[_multimol]` and `_impl` —
  ~110 lines of dead code.

### Added

- **`edg_*` stress-test suite — 20 honest probes designed to break
  RM** (`38d2087`, `6617a84`, `563495b`). Targets feature
  combinations not covered by the existing `ft_*` / `combo_*`
  suite: state-increment ladders, synthesis-into-pre-bonded
  complexes, time-explicit rates, pattern-level local functions,
  ternary embeddings, branched aggregates, multi-Fixed competition,
  zero-rate edge cases, and several `A+A` self-binding shapes that
  were hiding the propensity bug above. 18 models use BNG2 ODE as
  the deterministic verdict reference; the two polymer-style models
  (`edg_pattern_local_fcn`, `edg_branched_polymer`) use 100-rep
  NFsim refs since their network generation is intractable. Per-
  model rationale and 500-rep verification table at
  `tests/models/feature_coverage/EDG_RATIONALE.md`.

- **Tier-0 refusal for Sat / Hill / FunctionProduct rate laws**
  (`13bb424`). The rule loader recognised only `Ele`, `Function`,
  and `MM`; everything else fell through to the default Ele type
  with `rate_value=0.0`, so rules using `Sat()` / `Hill()` /
  `FunctionProduct()` silently produced zero propensity and never
  fired. RM now refuses loudly. Each error names the offending
  rule id and gives per-type guidance: Sat → "use MM instead"
  (mirroring NFsim's own policy at `NFinput.cpp:2459`); Hill →
  "use generate_network + ODE"; FunctionProduct → "rewrite as a
  single Function".

- **Feature-coverage tests for MM and TFUN** (`6421017`,
  `cb03071`).  `ft_mm_ratelaw.bngl` exercises `MM(kcat,Km)`
  against NFsim; `ft_tfun.bngl` exercises the new lowercase
  `tfun()` syntax against BNG dev-branch ODE (set
  `BNG2=$HOME/Code/bionetgen/bng2/BNG2.pl` to regenerate). Both
  code paths now have regression coverage; both exposed real RM
  bugs while being authored.

- **`examples/embed.cpp` + `examples/CMakeLists.txt`**
  (`a797b81`).  Minimal compilable C++ embedding example showing
  both stateless `run()` and stateful session usage. Off by
  default; opt in via `cmake -DRULEMONKEY_BUILD_EXAMPLES=ON`.

- Three feature_coverage regression tests pinning the sym-K shapes
  fixed above:
  - `ft_multimol_sym_obs.bngl` — 5-mol palindromic observable with
    sym partner (the r07 shape).
  - `ft_multimol_unimol_unbind_sym.bngl` — multi-mol unimolecular
    unbind on a host with sym components, where operations don't
    differentiate the embeddings (the r02 shape).
  - `ft_multimol_pattern_sym_nonreacting.bngl` — multi-mol pattern
    with sym non-reacting components on the seed (the r18 shape,
    catches the embedding-correction × dedup double-apply).
  Each was verified to fail pre-fix and pass post-fix via stash-
  and-rerun.

### Changed

- **README**: replaced the bare "Public API" snippet with a longer
  "Embedding (C++ API)" section that pairs stateless and session
  usage and points at the new example and public headers
  (`a797b81`).

- **`harness/benchmark_feature_coverage.py`**: added
  `_copy_aux_files()` to stage `*.tfun` next to the BNGL when
  invoking BNG2 / NFsim from a tempdir; `_run_one_nfsim_rep` now
  runs with `cwd=tmpdir` so NFsim's relative-path lookups
  resolve. New `RM_ONLY` set (currently empty) for any future
  model that has no third-party reference.

- **Two existing models cleaned up post-`edg_*` benchmark**
  (`6617a84`): `edg_fixed_competition` dropped a bogus
  `conserved Total_S = 90` invariant (S is consumed by catalysis,
  not conserved); `edg_ring_break_constraint` corrected
  `conserved A_total` from 30 → 40 (miscounted seed: 10×2-mer +
  5×4-mer = 40 A's); `edg_deep_param_chain` magnitudes bumped so
  the default 5-rep ODE comparison is stable. Harness routes
  `edg_time_dependent_rate` and `edg_deep_param_chain` through
  ODE verdict (NFsim rejects `time()` and function-of-function
  chains).

- `tests/reference/basicmodels/PROVENANCE.md` rewritten to lead
  with what the suite *is* (29 imported tests, source, reference
  flow) and treat the seven upstream NFsim tests not carried over
  as a clearly-labeled appendix grouped by reason.
  `harness/basicmodels.py` and
  `harness/generate_basicmodels_refs.py` docstrings tightened to
  a one-line origin pointer.

### Removed

- **Dead `extra_eff` machinery** (`d061f18`). After the self-
  binding propensity fix above, `compute_extra()`,
  `PerMolRuleData::extra`, `RuleState::total_extra`, and
  `extra_eff` in `compute_propensity` became unused — within-mol
  pair removal is now handled at sample time by
  `select_reactants`, not at propensity time. 35 lines removed
  from `engine.cpp`.

- **Four upstream NFsim regression tests dropped from the
  basicmodels suite** (`85feae1`, `9fb2efb`). All four tested
  NFsim-specific behaviors that don't apply to RuleMonkey:
  - `r33` and `r35` pinned NFsim issues #22 / #21 ("occupied-
    site bond error") and #14 (RHS `.` between products that
    NFsim splits anyway). On the BNGL these tests carry,
    BNG2.pl's `generate_network` produces the chemistry-correct
    behavior (bound by free-B count for r33; zero reactions for
    r35) and RuleMonkey matches BNG2.pl. The NFsim references
    captured the historic NFsim quirks, which by design diverge
    from BNGL strict.
  - `r31` is a crash regression test with no `begin observables`
    block (the author's own comment: *"validation harness will
    run NFsim on this XML and ensure it doesn't crash"*).
  - `r34` includes a `begin observables` block but the author
    deliberately commented out the only line in it.

  After the removals the suite is clean **29 PASS / 0 FAIL /
  0 NO_MATCH**. Joins the pre-existing r27 / r28 / r36 in the
  PROVENANCE appendix.

### Verification

End-to-end benchmark state on 2026-04-29 (post-fix):

| Suite                                       | Result |
|---------------------------------------------|---|
| `feature_coverage` (77 models, --reps 5)    | 77 PASS / 0 FAIL |
| `benchmark_full --tier full` (71 corpus)    | 71 PASS / 0 FAIL |
| `nfsim_basicmodels`         (29 models)     | 29 PASS / 0 FAIL |

177 / 177 models PASS RM-vs-NFsim z-score (or RM-vs-ODE rel-err
for `NFSIM_UNRELIABLE` models) post-fix.

## [3.0.0] — 2026-04-26

First release of the cleanroom C++17 rewrite. Not source-, ABI-, or
CLI-compatible with RuleMonkey 2.0.25.

### Added
- Cleanroom C++17 simulation engine (`librulemonkey`) with no external
  dependencies beyond the standard library — including an in-house XML
  parser, so neither TinyXML nor any other third-party XML library is
  required.
- Public C++ API at `include/rulemonkey/{simulator,types}.hpp` exposing
  `rulemonkey::RuleMonkeySimulator` for in-process embedding (init / run /
  step_to / simulate / add_molecules / set_param / save_state / load_state).
- `rm_driver` batch CLI emitting `.gdat`-format trajectories.
- Test corpora under `tests/models/`:
  - `feature_coverage/` — 52 BNGL feature-coverage models with invariants
    and golden values.
  - `corpus/` — 71 real-world rule-based models for efficiency and
    correctness benchmarking.
  - `nfsim_basicmodels/` — 36 models from the NFsim parity suite.
- 100-replicate ensemble reference trajectories under
  `tests/reference/nfsim/` (regenerated 2026-04-10) — 69 of 71 models
  from a gold-standard NFsim build, with `toy_jim` and `rm_tlbr_rings`
  replaced by hand-rolled Gillespie SSA where NFsim was confirmed to
  produce incorrect output. SSA scripts under `harness/ssa/`. See
  `tests/reference/nfsim/PROVENANCE.md` for the full provenance and
  per-model exceptions table.
- Python harness scripts under `harness/` for end-to-end benchmarking
  and validation.
- CMake (≥3.25) build with Ninja generator and configurable presets;
  smoke test wired into CTest.
- GitHub Actions CI for Linux and macOS.

### Changed
- License: **GPLv3 → MIT.** Cleanroom code reuses no legacy source.
- Build system: **autotools → CMake** with Ninja.
- Engine language: **C → C++17.**
- Default semantics: strict BNGL `block_same_complex_binding` is now on
  by default; pass `-no-bscb` for compatibility with NFsim runs that
  omitted `-bscb`.

### Removed
- Legacy C implementation (RuleMonkey 2.0.25). Preserved in the parent
  fork's git history (`richardposner/RuleMonkey`).
- Vendored `dSFMT-1.3` and `nauty22`. The cleanroom uses
  `std::mersenne_twister_engine` for RNG and **does not currently
  exploit graph canonical labeling** for complex/species identification.
  Re-introducing canonical labeling is a candidate optimization for a
  future minor release.

### Known limitations
- No Python bindings yet. Python access is via the `rm_driver`
  subprocess. Native bindings (`pybind11` + `scikit-build-core`) are
  planned for 3.1.0 alongside the BNGsim integration.
- Compartments are refused at Tier 0 (exit code 2). Volume scaling and
  surface chemistry remain open work.
- Arrhenius rate laws and a small set of unsupported BNGL constructs
  cause hard refusals; pass `--ignore-unsupported` to demote to
  warnings for testing.

### Lineage
The legacy implementation, RuleMonkey 2.0.25, was introduced in:

> Colvin J, Monine MI, Gutenkunst RN, Hlavacek WS, Von Hoff DD, Posner
> RG. *RuleMonkey: software for stochastic simulation of rule-based
> models.* BMC Bioinformatics 11:404 (2010). PMID: 20673321.

[3.2.0]: https://github.com/wshlavacek/RuleMonkey/releases/tag/v3.2.0
[3.1.2]: https://github.com/wshlavacek/RuleMonkey/releases/tag/v3.1.2
[3.1.1]: https://github.com/wshlavacek/RuleMonkey/releases/tag/v3.1.1
[3.1.0]: https://github.com/wshlavacek/RuleMonkey/releases/tag/v3.1.0
[3.0.0]: https://github.com/wshlavacek/RuleMonkey/releases/tag/v3.0.0
