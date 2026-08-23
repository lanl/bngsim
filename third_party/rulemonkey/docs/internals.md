# RuleMonkey internals

Reading guide for the SSA engine. Targets a reader who is about to modify
`cpp/rulemonkey/engine.cpp` and wants to know where load-bearing code lives
and why it is shaped the way it is. All line numbers are relative to
`engine.cpp` at the time of writing — they will drift; treat them as anchors,
grep for the function name to confirm.

For BNGL semantics and what is supported / refused, see `model_semantics.md`.
For the dev-time profiler, see the preamble in `cpp/rulemonkey/engine_profile.hpp`.

## Session build

`Engine::initialize` (near the bottom of `engine.cpp`) runs four steps,
and their order is load-bearing:

1. `init_species` — build the pool from the seed species.
2. `init_incremental_observables` — classify each observable, build the
   tracker's per-molecule contrib table and per-complex pass flags, and
   settle `obs_values` for every tracked observable out of those tables
   (`seed_tracked_obs_values`).
3. `compute_observables(skip_tracked)` — the from-scratch walk, now only
   for the observables the tracker does not keep.
4. `init_rule_states` — needs `obs_values` settled, because a Function
   rate law reads it.

Step 2 comes before step 3 on purpose. Building the tracker's tables
costs exactly the per-molecule embedding counts a full observable walk
costs, so running the walk first and the tracker second counted every
tracked observable twice — half of session build on a large pool
(issue #65). It reads only the model and the pool, so it does not mind
running before the other two.

Both step 2's seeding and the per-event delta path go through
`tracked_obs_contrib`, which owns the dispatch: a structurally
unconstrained pattern (`T()`) contributes one per molecule of its type
with no matching at all, a 2-molecule/1-bond pattern takes the
`count_2mol_1bond_fc` specialization, and the rest fall to the generic
BFS. Keep them on the shared routine — seeding used to write its own
call and silently missed both shortcuts.

`kLocalObsTrackInvariant` (Debug and ASan) proves the seeded values
against a from-scratch walk at init, in addition to its original job on
`evaluate_observable_on`.

### Step 1 — the seed walk

`init_species` runs in two passes. The first resolves each seed species
into a `SeedTemplate`: the molecule types to create, the (component
slot, state index) pairs to pin, and the bond endpoints, all of them
already mapped through `build_comp_mapping` from the species XML's
component order into the molecule type's declaration order. The second
stamps out the copies. Nothing in a template depends on which copy is
being made, so resolving one per copy — which is what this used to do —
rebuilt a name-keyed `unordered_map` per molecule per copy, and was
about a sixth of session build on a million-molecule pool (issue #67).
A change here that leaves copy 1 right and copies 2..N wrong is the
failure mode; `seed_build_test` is the gate.

The first pass also totals the seed: molecules, components, and
molecules per type. `AgentPool::reserve_for_seed` takes those and
reserves every arena the walk grows, which otherwise doubles its way
through the build — `molecules_` most expensively, since each element
carries a vector. It is a capacity hint and nothing more: a total that
is short, or that does not fit the `int` the pool addresses molecules
with, costs only the growth it failed to pre-empt.

### Step 4 — the per-rule tables

`init_rule_states` settles each rule's shape (which reactant slots it
has, whether its rate law is local, whether either slot is pure context)
and then hands it to `rescan_all_molecules_for_rule`, which sizes that
rule's per-molecule tables by the molecule arena and fills them.

Those tables are **two** vectors, both indexed by molecule id and kept
exactly the same length:

- `mol_data` (`PerMolRuleData`, 12 bytes) — the two slots' embedding
  counts and the P1 cache flag. Every rule with a reactant pattern
  molecule to seed on carries one.
- `mol_aux` (`PerMolRuleAux`, 64 bytes) — the shared-component split,
  the four local-rate fields, and the pure-context complex ids. Only
  allocated where `RuleState::needs_mol_aux` says the rule's shape reads
  one; left empty otherwise.

The row width is what a rule's table costs, once per molecule in the
pool, in zeroing at every session build and in residency for the whole
run. Before the split every rule paid the full 80-byte row, so a
unimolecular non-local rule with no pure-context slot — which reads two
of the eleven fields — cost 373 MB and about 0.1 s per session build on
a 4.7e6-molecule pool for a table it wrote a handful of entries of
(issue #71). Two things follow for anyone editing this:

- Every site that grows one table must grow the other, which is what
  `grow_mol_tables` is for. A `mol_aux` shorter than `mol_data` is an
  out-of-bounds read on the next event, not a wrong number.
- `rule_needs_mol_aux` is the single definition of which shapes get the
  wide row. Reading a `mol_aux` field from a path a shape outside that
  predicate can reach is the other way to break this.

The rescan fills `mol_data` with the P1 cache flag already set, because
a full rescan is what validates every row — including the zero-valued
defaults for molecules of types the rule cannot seed on. The default in
`PerMolRuleData` itself stays false, and the on-demand growth path takes
that default: those rows are for molecules born since the rescan, which
genuinely have not been computed.

Two rules never get either table: an n-ary rule (three or more reactant
patterns), whose counts live on `NaryState`, and a rule with no reactant
pattern molecule to seed on (issue #68). Every indexed read of
`mol_data` bounds-checks against its size or grows it first, so an empty
table gives the same "no entry for this molecule" answer a zeroed one
does.

`rule_table_shape_test` is the gate: one rule of each shape over
molecules made after the session build sized the tables, each arm priced
against an analytic reference.

#### What the tables hold

The row width is settled; the **sizing** is not. Both tables are
`assign(pool.molecule_count(), ...)` — the whole molecule arena, per rule
— and so is the Fenwick sampler a slot gets once its type population
passes `FENWICK_THRESHOLD`, and so are an n-ary rule's per-slot `counts`
(plus `raw` / `cx_of` on its context slots). None of that is a function
of the population the rule can actually see: a rule whose seed type holds
one molecule out of four million is charged for four million rows.

`RuleTableBytes` / `rule_table_bytes` account for what that comes to, and
the `[RM tables]` block prints it per rule and per model at the end of a
run, alongside the existing `[RM timing]` and `[RM per-rule]` blocks but
under its own `RM_PRINT_TABLES=1` gate — `reach` below walks each rule's
seed-type populations, and `RM_PRINT_TIMING` is what the wall-clock
harnesses set on every replicate:

```
[RM tables] arena=8030 rules=26 tabled=26 bytes=14582704 (13.9 MB) mol=12783760 sampler=1798944 reach_bytes=12994704
  RR1 (_R1): rows=8030 reach=7530 mol=610280 sampler=128496
  ...
```

Three things about the numbers.

They count rows *written* (`size()`), not rows allocated (`capacity()`).
Everything in `[0, size)` has been written by the `assign` that sized the
table or the `resize` that grew it, so it is resident; the tail up to
`capacity()` is address space the geometric growth of `resize(mid + 1)`
reserved and nothing has touched. Counting capacity put several corpus
models over 100% of their own peak RSS, which is the tell. So this is a
floor on what the tables cost, which is the right side to err on for an
argument that they are too big.

`reach_bytes` is the same total with every table cut to the highest live
molecule id its rule can index — one past the top of its seed types'
populations. Issue #71 names sizing the tables that way as the obvious
shortcut and warns it does not generalize; `reach_bytes` is what makes
that warning a number.

Read it as a bound on what a table *needs*, not on what sizing it that
way *delivers*. Building the tables at `reach` is safe — every read
bounds-checks or grows first — but it does not hold, because
`incremental_update` hands every rule every molecule an event touched and
grows that rule's table to cover it whatever its type, so each table
converges on the arena regardless. A molecule of a type neither slot
seeds on scores zero on both counts forever; its row exists only because
the loop made one. Measured: sizing by reach alone keeps 83.7% of the
bytes over the sweep against the 59.2% `reach_bytes` implies, and
`CaMKII_holo` keeps all of a table its rules reach 8.5% of.

The per-complex tallies (`PerCxTally`) are deliberately not counted: they
are sized by matching complexes rather than by the arena, so they are not
part of the product this measures.

`harness/rule_table_footprint.py` sweeps the three corpora with it,
pairing each model's table footprint against the peak RSS of its own
`rm_driver` process. What that sweep found, over all 189 models: on the
26 whose arena reaches 10 000 molecules the median share of peak RSS
held in these tables is 48.6%, and six are over 90% — `tcr_gen27ind33`
is 158 rules over 155 471 molecules and holds 887 MB of a 956 MB peak.
The unit cost is 44 bytes per (rule holding a table x arena row) at the
median, and a quarter of the total is Fenwick samplers. `rule_table_footprint_test` is the gate on the
accounting itself, and pins each rule shape's row width exactly: a field
added to either per-molecule struct is a byte per molecule per rule of
resident memory for the whole run, which is the change issue #71 says has
to be made deliberately.

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

### Pool side tables

Three `AgentPool` reads sit on the per-event path, so all three are O(1)
side tables rather than walks, and all three have an ordering or
bookkeeping contract a future edit can break silently:

- `type_mol_index_[type]` — the live molecules of one type. Removal is a
  swap-with-back using `type_mol_pos_[mol_id]`, **not** an order-preserving
  erase: the erase is O(population of the type) and every deletion pays it,
  which is enough on its own to make per-event cost grow with system size
  (issue #62). The list is therefore in no particular order. Nothing may
  depend on its order — `molecules_of_type` consumers scan it whole or
  sample from it by weight, and the Fenwick samplers key on molecule id,
  not on position.
- `active_mol_count_` — the live molecule count, maintained by
  `add_molecule` / `delete_molecule`. `active_molecule_count()` is read
  every event when a molecule limit is set, so it must never go back to
  scanning for `.active`.
- `cycle_bond_count_` — see above.

`kPoolIndexSelfCheck` (Debug and ASan builds, gated from CMakeLists.txt)
proves the first two on every deletion: the recorded position against the
type list's own contents, and the tally against `molecules_ − free_mol_ids_`.
Both checks are O(1).

### Who marks a complex dirty

`cxs_dirty_` is the canonical-label cache's invalidation set, and
`cached_label_of` treats an id that is *absent* from the label cache as
dirty too. Complex ids come from `next_complex_id_++` and are never
recycled, so a complex that was just born cannot have a cache entry —
which makes marking it dirty information-free. `add_molecule` therefore
does not mark, and at seed time that is one hash insert saved per
molecule (issue #67). Every mutator that edits an *existing* complex —
`set_state`, `add_bond`, `remove_bond`, `merge_complexes`,
`split_complex_if_needed` — still must, and missing one there is the bug
`kCanonicalCacheSelfCheck` exists to catch.

`cx_edits_` is not the same question and does not follow the same rule.
Its reader has to see each edit rather than ask whether one is
outstanding, so a birth is appended to it even though it is not marked
dirty.

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
- Per-event cost that grows with population → look for a walk over the
  whole pool on the event path, not for an algorithmic change in
  matching. Run the model at two scales, divide wall time by
  `events=` from `RM_PRINT_TIMING=1`, and compare: a per-event cost
  that tracks system size is a scan somewhere. The phase timers name
  which of `fire_rule` / `incr_update` / `record_at` it lives in.
  Sampling the process (`sample`, `perf`) then names the function
  directly — this is how the pool side tables above were found.
