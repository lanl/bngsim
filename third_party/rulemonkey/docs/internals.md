# RuleMonkey internals

Reading guide for the SSA engine. Targets a reader who is about to modify
`cpp/rulemonkey/engine.cpp` and wants to know where load-bearing code lives
and why it is shaped the way it is. All line numbers are relative to
`engine.cpp` at the time of writing — they will drift; treat them as anchors,
grep for the function name to confirm.

For BNGL semantics and what is supported / refused, see `model_semantics.md`.
For the dev-time profiler, see the preamble in `cpp/rulemonkey/engine_profile.hpp`.

## SSA event loop

Public entry: `Engine::run` (lines 7099–7103) → `Impl::run_ssa`
(lines 6501–7079). One iteration of the loop:

1. Recompute total propensity if a baseline-flush is due (line 6577).
   Propensity is otherwise maintained by delta-updates from
   `incremental_update` to absorb floating-point drift over long runs.
2. Sample a Δt via exponential draw on `total_propensity` (line 6605).
3. Record observables for any sample times the step crosses (line 6609).
4. Sample a rule index, weighted by per-rule propensity (lines 6625–6661).
5. `select_reactants` — pick the actual molecules and embeddings to fire
   on (line 6665).
6. `fire_rule` — apply the rule's ops, mark affected molecules
   (line 6912).
7. Update species-observable counts incrementally (line 6978).
8. `incremental_update` — recompute propensity for rules whose
   counts could have changed (line 6984).

Per-phase chrono (`timing_sample`, `timing_fire`, `timing_obs`,
`timing_update`, `timing_record`, lines 6987–6990) is always-on and cheap;
the deeper profile structs in `engine_profile.hpp` are dev-time only.

## Pattern matching

The engine matches BNGL patterns onto the live molecule pool by
enumerating injective mappings that respect type, internal-state, and
bond constraints. Three layers:

- `count_embeddings_single` (lines 685–925) — single-molecule patterns.
  Deduplicates mappings that differ only in non-reacting components
  (`reacting_local` parameter), matching NFsim's convention so propensity
  is invariant under reordering of inert sites.
- `count_multi_mol_fast` dispatcher (lines 2032–2059) — checks
  `FastMatchSlot.enabled`. If on, dispatches to the 2-mol/1-bond fast
  path; otherwise falls through to the generic body.
- `count_multi_mol_fast_generic` (lines 1759+) — multi-molecule patterns.
  Enumerates seed embeddings via `count_embeddings_single`, then BFS over
  the *pattern* adjacency graph (`PatternAdj`, line 1761) — each pattern
  bond contributes a partner-side `count_embeddings_single` call. The
  recursion is over pattern molecules, not pool molecules; termination is
  guaranteed because the pattern graph is finite and visited.

### The 2-mol/1-bond fast path

`count_2mol_1bond_fc` (lines 1463–1697) is taken when `build_fastmatch_slot`
(lines 1257–1287) finds a pattern with: exactly 2 molecules, both with bond
degree 1, no self-bonds, and both sides fully constrained. The path skips
seed-side enumeration entirely — it iterates pre-computed seed bond
candidates and traces the existing pool bond to the partner instead of
running BFS. Reduces a frequent shape (`A(b!1).B(a!1)`) from
O(N_seed_embs · partner-BFS) to O(seed_bond_candidates).

The invariant gate `kFastMatchInvariant` (in `engine_profile.hpp`) runs
generic and fast in parallel and asserts equality — flip it on while
refactoring either path, off before benchmarking.

## Complex tracking

Each molecule belongs to a complex (a connected component of the
bond graph). `AgentPool` maintains complex membership and a per-complex
cycle-bond count; the cycle count is what lets `compute_propensity`
classify same-complex bond candidates without re-walking the graph.

- `add_bond` (lines 214–230) — sets bidirectional partners; if endpoints
  are in different complexes, calls `merge_complexes`; if same,
  increments the cycle count (the new bond closes a cycle).
- `remove_bond` (lines 233–249) — clears partners, then
  `split_complex_if_needed`.
- `split_complex_if_needed` (lines 458–631) — singleton complexes
  short-circuit (lines 474–497). Otherwise BFS from `mol_a` within the
  old complex (lines 516–537), counting half-edges en route. If `mol_b`
  is reached, the removed bond was a cycle edge — decrement cycle count.
  If not, it was a tree edge — partition into two pieces and redistribute
  the cycle count using `edges − vertices + 1` per piece (lines 586–630).
  The BFS visited-set is reused for the partition; no second walk.

## Propensity

`compute_propensity` (lines 2268–2347) implements three rate-law shapes:

- Unimolecular: `(a_total / ca) · rate · symmetry`.
- Bimolecular heterodimer: `a_eff · b_eff · rate · symmetry`.
- Bimolecular homodimer same-components: `(ao·b_eff + ab·bo +
  (ab² − ab_sq) / 2) · rate` — the `ab_sq` deflation removes self-pair
  null events, which is what the homodimer rate test pins down.
- Michaelis–Menten quasi-steady-state: lines 2284–2298.

`incremental_update` (lines 4488–4870+) is the hot path. After
`fire_rule` marks affected molecules, this function: (1) optionally
expands affected molecules to whole complexes if any local-rate rules
exist (lines 4516–4534), (2) maps affected types to candidate rule
indices (lines 4549–4578), (3) for each candidate, recomputes
embedding totals via the pattern-matching layer above. Per-rule deltas
feed `set_rule_propensity` and a running `total_propensity` sum;
periodic baseline flushes (loop top, line 6577) absorb FP drift.

There is no per-rule embedding cache across events — each affected
rule recomputes from scratch. The cache that *does* exist is the
`RuleState` struct (lines 2201+), which stores within-event derived
values so a single tick doesn't recompute symmetry factors and
correction denominators twice.

## `select_reactants` — five paths

Lines 4983–5350+. The paths are exclusive and labeled in
`SrProfile`:

- `kPathZero` — zero-order synthesis, no seed.
- `kPathUniSingle` — unimolecular, single-molecule pattern. Sample a
  molecule weighted by embedding count, then uniformly sample one
  embedding.
- `kPathUniMultiFm` — unimolecular, multi-molecule pattern, fast-path
  active. Sample seed molecule, resolve partner via
  `select_2mol_1bond_fc_match` (no BFS).
- `kPathUniMultiGen` — same shape, generic path. Sample seed, then BFS
  via `select_multi_mol_unimolecular`.
- `kPathBimol` — bimolecular. Sample two seed molecules weighted by
  embedding counts; enumerate embeddings on each side; uniformly sample
  one embedding pair. Same-components rules retry up to 64 times for
  distinct seeds (the `homodimer_rate_test` is the regression gate on
  this loop).

## `fire_rule`

Lines 6071–6320+. Switches on `OpType` (definitions in `model.hpp`):

- `StateChange` (6161) — flip a component's internal state.
- `DeleteBond` (6179) — `pool.remove_bond` → potential complex split.
- `AddBond` (6197) — `pool.add_bond` → potential complex merge.
- `AddMolecule` (6234) — `pool.add_molecule`, register in
  `product_mol_to_actual` so subsequent `AddBond` ops in the same rule
  can target it.
- `DeleteMolecule` (6253) — mark inactive, queue for compaction.

`bond_changed` (line 6075) is the flag that gates expanding the
affected set to whole complexes after firing — local-rate rules need
the expansion; pure StateChange rules do not.

## Schema fingerprint

`compute_schema_fingerprint` (lines 61+) is FNV-1a over molecule type
names, ordered component names, and ordered allowed states.
`save_state` writes the fingerprint; `load_state` rejects on mismatch.
Parameter values and rate constants are deliberately *not* in the
fingerprint — those legally vary between save and load (e.g., resuming
a checkpoint with new `set_param` overrides). The caveat is documented
on `Simulator::save_state` in `include/rulemonkey/simulator.hpp`.

## Where to start when something is wrong

- Trajectory diverges from NFsim → bisect the BNGL (see
  memory/feedback note "bisect before reading"), then look at
  `compute_propensity` for the rule shape that diverges.
- Wall-time regression on a feature_coverage model → enable
  `RM_DEV_PROFILES`, run that model standalone, compare profile output
  against the previous commit. The hot phases are usually `incremental_update`
  or `count_multi_mol_fast_generic`.
- Crash under ASan → almost always `AgentPool` index reuse after
  `DeleteMolecule`. Check that the affected-molecules set was not
  populated with the deleted mid before compaction.
