# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.10.0] — 2026-08-21

### Added

- **What a rule's per-molecule tables actually hold, measured across the
  three corpora (issue #71).** #72 narrowed the per-molecule row from 80
  bytes to 12 and left #71's other two directions — sizing the tables, and
  the Fenwick samplers, by what the rule can see rather than by the molecule
  arena — alone, for the reason the issue gives: both need a stable dense
  per-type slot in the pool, which is a design question with a known failure
  shape (#62 made type-index removal a swap-with-back, and #64 had to fix
  what that broke), and the case for paying for it is residency on real
  models rather than on a synthetic scale-up. That case was asserted, not
  measured. It is measured now, and it is a good deal stronger than the
  issue supposed.

  `rule_table_bytes` is the accounting: what each rule holds across
  `mol_data`, `mol_aux`, an n-ary rule's per-slot count tables and every
  Fenwick tree. The `[RM tables]` stderr block prints it per rule and per
  model at the end of a run, under its own `RM_PRINT_TABLES=1` gate rather
  than `RM_PRINT_TIMING`'s — the block walks each rule's seed-type
  populations, and `RM_PRINT_TIMING` is what the wall-clock harnesses set on
  every replicate, so a memory question would otherwise put its cost on a
  timing measurement. Rows *written*, not rows allocated: counting
  `capacity()` put several corpus models over 100% of their own peak RSS,
  the untouched tail of a geometrically grown vector being address space and
  not residency. So every number below is a floor.

  `harness/rule_table_footprint.py` sweeps all 189 models of
  `feature_coverage`, `corpus` and `nfsim_basicmodels` with it — one
  replicate each at the suite's canonical `(t_end, n_steps)` — pairing each
  model's footprint against the peak RSS of its own `rm_driver` process,
  taken from `os.wait4` so it is that run's number.

  Across all 189 the median share is 1.79%, which is the wrong number to
  read: 67 of them have arenas under 200 molecules, where the whole peak is
  a 3.2 MB process floor. Conditioned on size it inverts. Of the 26 models
  whose arena reaches 10 000 molecules the **median share is 48.6%**, and
  six are over 90%:

  | model | rules | arena | tables | peak RSS | share |
  |---|---:|---:|---:|---:|---:|
  | `tcr_gen27ind33` | 158 | 155 471 | 887.2 MB | 955.5 MB | **92.9%** |
  | `example4_fit` | 158 | 158 456 | 904.3 MB | 980.8 MB | 92.2% |
  | `tcr` | 158 | 67 042 | 387.2 MB | 425.7 MB | 91.0% |
  | `ensemble` | 232 | 101 231 | 780.8 MB | 948.9 MB | 82.3% |
  | `egfr_nf_iter5p12h10` | 26 | 302 009 | 294.9 MB | 387.6 MB | 76.1% |
  | `CaMKII_holo` | 77 | 10 893 | 28.1 MB | 54.7 MB | 51.5% |

  50 of the 189 are at or above 10% of peak and 15 at or above 50%. Per
  suite the medians are 14.9% (`corpus`), 2.3% (`nfsim_basicmodels`) and
  0.4% (`feature_coverage`), which tracks median arena size (4 500, 1 271,
  100). The unit cost is 44 bytes per (rule holding a table x arena row) at
  the median and 92 at the worst — 12 for the narrow row, 76 where the shape
  reads the wide half, plus 8 per row per Fenwick-sampled slot; samplers are
  24% of all the table bytes in the sweep. So the issue's "a dozen rules
  over a few million molecules spends gigabytes on tables that are mostly
  zeroes" is not the hypothetical it was labelled as: `ensemble` is 232
  rules over 101 231 molecules and spends 780 MB, and it is in the reference
  corpus.

  The block also reports `reach` — one past the highest live molecule id a
  rule can index, over its seed types — and `reach_bytes`, what the same
  tables would hold cut to it. That is the shortcut #71 names and warns does
  not generalize. Summed over the sweep it says the tables could be 59.2% of
  their size: half on the `tcr` family, 8.5% on `CaMKII_holo`, and 99.9% on
  `egfr_nf_iter5p12h10`, whose seed types span the arena.

  It is a bound on what a table needs, and **not** on what sizing it that
  way delivers, which was measured before this was written up. Building the
  tables at `reach` in the rescan is safe — every read bounds-checks or
  grows first, the audit #68 relied on — but it does not hold: the sweep
  re-run against it keeps 83.7% of the bytes, not 59.2%, and no peak RSS at
  all. `CaMKII_holo` keeps 100% of a table its rules can reach 8.5% of, in a
  0.3 s run. The reason is in `incremental_update`, which hands every rule
  every molecule an event touched and grows that rule's table to cover it
  **whatever its type**, so each table converges on the arena no matter what
  it was sized to. A molecule of a type neither slot seeds on scores zero on
  both counts forever; the row exists only because the loop made one.

  Sizing by reach *and* declining to materialize those rows does hold, tried
  as an experiment against the same sweep: `tcr` 387.2 -> 194.0 MB,
  `ensemble` 780.8 -> 491.5 MB, `CaMKII_holo` 28.1 -> 8.4 MB, `lat` 122.3 ->
  92.3 MB, and `egfr_nf_iter5p12h10` unmoved at 294.7 MB, which is what
  `reach` predicts for it. That is a change to the hot per-molecule loop
  with a hazard the naive form has — an id the pool has handed to a molecule
  of another type still owes its predecessor's count back to `a_total`, the
  failure shape #62 introduced and #64 fixed — so it is not in this PR.
  #71's directions 3 and 4 stay open with both numbers attached: what the
  tables cost, and what the cheap sizing does and does not buy.

  `rule_table_footprint_test` is the gate on the accounting, over three
  fixtures already in the tree: `pool_churn_model` for the narrow row, the
  wide row and a seedless rule that must hold no table at all (#68),
  `trimolecular_model` for an n-ary rule's per-slot count tables and the
  three Fenwick samplers its 400-molecule seed type earns, and
  `rule_table_shape_model` for one rule per wide-half group. Each rule's row
  width is pinned exactly, so a field added to either per-molecule struct
  fails it and says what it costs; the summary is checked against the
  per-rule detail it claims to sum. `docs/internals.md` gains the
  corresponding subsection under "Step 4 — the per-rule tables".

  Nothing outside the gated block changes: `feature_coverage` is 89/89
  against the vendored NFsim ensembles, and ctest is 47/47 release and ASan.

- **Load-time diagnostics for the two `MM(kcat,Km)` constructs where RM
  cannot reproduce BNG2 (issue #45).** BNG2.pl is the reference RM is written
  against, so both of these are divergences from it and both now say so at
  load. They are `Severity::Warn` rather than refusals because the constructs
  are idiomatic BNGL and refusing would put a large share of real MM models
  out of reach. What the warning buys is that the divergence is named at load
  instead of being discovered by diffing trajectories against BNG2.

  **A substrate pattern that can match more than one species.** BNG2's
  network expansion emits one MM reaction per matching *(substrate, enzyme)
  species pair*, each evaluating the law on that pair's own counts, while RM
  applies one law to the summed match counts. Measured: a two-species
  substrate in saturation runs at **2.00x** under BNG2 (product at t=40:
  783.9 against RM's 398.4), and a two-species **enzyme** with the enzyme in
  excess runs at **1.81x** (BNG2 82.11 against RM 45.26), the law being
  nonlinear in both counts. Both vanish where the law is linear. BNG2's own
  `checkSpeciesGraph` check covers the substrate only, so the enzyme axis is
  silent there; RM checks both slots. Matching BNG2 would need a live map
  from species canonical form to match count on each slot, maintained per
  event, which is the species-level bookkeeping a network-free engine exists
  to avoid, and it costs most on the very models that carry the construct. BNG2 warns
  about this at rule-read time via `checkSpeciesGraph(..., IsSpecies => 1)`
  but the warning dies in the XML, so RM recomputes the predicate: a pattern
  matches at most one species iff every molecule specifies every component
  its type declares, each with a definite state and a definite bond status.
  Validated against BNG2 2.9.3 on seven cases — complete patterns, a missing
  component, an unspecified state, `!+` and `!?` bonds, a stateless molecule
  type, a complete two-molecule complex — agreeing on all of them. This is
  not hypothetical: `CaMKII_holo` in the reference corpus has it, its
  `CaMKII(Y286~P)` substrate leaving six of seven components unspecified.

  `docs/model_semantics.md` gains the corresponding modelling note: MM
  reproduces BNG2 exactly when both reactant patterns pin one species each,
  and since MM exists to keep a generated network small and network-free
  never generates one, writing the mechanism out (`S + E <-> SE -> P + E`)
  is exact in both engines, needs no warning, and gets shared-enzyme
  competition right, which neither MM reading does.

  **A `symmetry_factor` that cannot be attributed.** It belongs to the
  reactant pattern the rule transforms (#37); when the rule transforms
  **both**, the scalar is a product of both patterns' factors and the XML is
  one number with no way to split it. RM applies it to the substrate, which
  reproduces BNG2 for the ordinary shape where the enzyme is a catalyst and
  anywhere the law is linear (the rate goes as `S*E` there), and runs up to
  2x fast against BNG2 in saturation if the symmetry was the enzyme slot's.
  Now named at load either way.

  The transformed/context derivation that #33 and #37 both read is now a
  single shared function (`reactant_pattern_transforms` in `model.hpp`)
  rather than one copy in the engine's `init_rule_states` and another in the
  loader; the engine's inline copy is replaced by a call to it. Behaviour is
  unchanged: all 88 feature-coverage models and all 71 nfsim-corpus models
  are byte-identical against the pre-change binary. `mm_diagnostics_test`
  pins both diagnostics on a fixture where each rule trips exactly one, with
  `mm_symmetry_model` (four MM rules, including #37's enzyme-slot arm whose
  factor *is* attributable) as the negative control.

- `tests/models/feature_coverage/ft_local_fcn_bimol.bngl` and
  `ft_local_fcn_bimol_sym.bngl`, plus the `local_fcn_bimol_test` ctest
  case, covering the two DOR1 shapes fixed below — a bimolecular rule
  with a local-function rate, and the same rule carrying BNG2's
  `symmetry_factor`. No existing corpus model used a local function on
  a bimolecular rule, which is why the gap survived three minor
  releases.

### Changed

- **1-WL color refinement runs as a worklist over the cells that can still
  split, cutting the representative election 2.6x on a 80-subunit catalyst
  and the `.species` sweep 1.7x (issue #56).** #53 left the election's cost
  almost entirely inside the refinement itself, recomputed from scratch on
  every edit of a tagged complex however small the edit was. Re-measured
  here by an early return placed between the two stages, the split is 0.04 s
  of graph building against 2.43 s of refinement on the 20-subunit probe:
  the bridge is 2% and the refinement is the rest.

  The textbook round recolors *every* vertex by (own color, sorted neighbor
  colors) and re-ranks the whole vertex set, which is `O(rounds × V log V)`
  — and a chain needs on the order of `O(n)` rounds for the colors to
  propagate in from its ends. Most of that work is provably empty. A round
  never merges two cells and never reorders them (the own color leads the
  signature, so a round splits each cell *internally* and orders the pieces
  among themselves), so a cell whose members all carry the same
  neighbor-color slice comes out whole, and it can only stop carrying the
  same slice when a cell holding one of their neighbors splits. A round
  therefore examines the cells marked by the previous round's splits —
  every cell, on the first round — and nothing else.

  This is a change of *which cells a round looks at*, not of the rounds. A
  cell's color while refining is now its start index in the partition array,
  which is order-preserving and leaves an untouched cell's color literally
  unchanged, so nothing is renumbered that was not split; one pass at the
  end recovers the dense `0..k-1` ranking. The output is the same coloring,
  which is what keeps every canonical label and every elected representative
  byte-for-byte what they were. Two small-case specializations ride along:
  degree 2 is the common case for a slice sort (a bonded-component vertex
  has exactly two edges by construction, and those vertices outnumber the
  molecules), and the two-member cell is what most of a refinement's
  trailing rounds are cutting.

  Election overhead against the same model with the election gated off,
  best of 5, `--preset release`, on #53's probe shape — a chain of
  `W(l,r,m~0~1)` carrying the tag on a pure-context slot, with a toggle rule
  editing the chain on essentially every event and `0*Mod(x)` keeping every
  instance priced the same. `t_end` is scaled as `400/n` so every column
  runs the same number of events:

  | chain | before | after | |
  |---|---:|---:|---:|
  | 5-subunit | 0.150 s | 0.140 s | 1.1x |
  | 10-subunit | 0.550 s | 0.340 s | 1.6x |
  | 20-subunit | 2.550 s | 1.430 s | 1.8x |
  | 40-subunit | 7.260 s | 3.420 s | 2.1x |
  | 80-subunit | 18.080 s | 6.900 s | **2.6x** |

  Which is the point: #53's fix scaled the wrong way (5.4x on a 5-subunit
  catalyst, 1.8x on a 20-subunit one) because it removed a per-call constant
  and left the per-vertex work alone. This one grows the other way, taking
  the measured exponent from `n^1.73` to `n^1.40`. The 5-subunit column
  barely moves and is expected not to: at 13 vertices there are three
  refinement rounds to skip work in.

  The `.species` batch sweep is the other consumer and the one with the
  large complexes. Output is byte-identical on `machine`, `ensemble`,
  `egfr_net`, `egfr_nf_iter5p12h10`, `bench_tlbr_yang2008`, `rm_tlbr_rings`
  and `tlbr`; the `tlbr` sweep (largest complex 2757 molecules) runs 356.4 s
  → 210.0 s, and the models whose complexes are small are unchanged inside
  noise. A differential harness over 10,032 generated complexes — random
  connected complexes up to 20 molecules, plus chains, rings, state-carrying
  chains and stars up to 129 molecules — puts `canonicalize().label`,
  `.mol_order`, `canonical_order()` and `canonical_order_fast()`
  byte-identical against the pre-change canonicalizer.

  `canonical_test` gains the two things that catch this going wrong. An
  under-refining worklist is silent — the individualization search still
  finds a correct canonical form, just a different one — so the assertion
  that catches it is `fast_path`, and the new `lr_chain` shapes are chains
  of `A(l,r)` up to 24 long, asymmetric (the reversal is not an
  automorphism, `l` and `r` being different names) but only one hop per
  round to prove it, plus 12-chains carrying per-molecule states. And the
  property test's fast-path / search / answered / declined counts are now
  pinned exactly rather than bounded, since that split is precisely what
  moves when refinement stops one round early. 37/37 ctest under release and
  under ASan+UBSan, 29/29 corpus guard tier, 88/88 feature coverage,
  `bng_oracle` green.

  Pinning those counts needed one adjacent fix, which CI found: the property
  generator drew through `std::uniform_int_distribution` and `std::shuffle`,
  and neither is portable. Same engine, same seed, three different samples —
  3927 complexes under libc++, 3919 under libstdc++, 3892 under the MSVC
  STL. The file already says a property test must be reproducible so a
  failure is re-runnable, and it was reproducible only across runs on one
  platform: a failure CI reports from Linux could not be re-run on a
  developer's mac. The generator draws through a hand-rolled `pick` and
  Fisher-Yates `shuffle_v` now, so every platform walks the same sample and
  the counts are a property of the canonicalizer rather than of the standard
  library. Building the new test against the pre-change canonicalizer gives
  the identical 3586 / 345 / 7788 / 74, which is what makes them a
  regression gate at all.

- **The canonical-representative election no longer builds a string, cutting
  its cost 5.4x where it is live (issue #53).** #52 prices a pure-context
  reactant slot carrying a per-molecule local function tag at the complex's
  canonically-first matching molecule. Getting that one permutation cost a
  `canonical::canonical_order` call per representative change, and that call
  went through the pure core built for the `.species` batch sweep: a
  `std::string` per molecule and per bonded component, interned through a
  `std::map`, on top of a fresh `ComplexGraph`, adjacency and refinement
  scratch every time — on the order of 50 heap allocations to answer with
  one integer.

  The canonicalizer now also takes integers. `RankedComplex` describes the
  same port graph with a dense rank in place of every name, `Workspace`
  holds the refinement buffers across calls, and `canonical_order_fast`
  answers from those with no allocation once warm. The ranks are not free
  integers — the refined colors, and so the ordering, depend on the *order*
  of the initial colors and not only on which vertices share one — so
  `rank_molecule_type_names` / `rank_component_names` / `rank_state_names`
  derive them from the names and are the single place that correspondence is
  written down. The three are not one rule: a color string writes a molecule
  type as `Type(` and a component as `name~`, and `~` outranks every
  character a BNGL identifier may hold while `(` is outranked by all of
  them, so a name that is a prefix of another sorts *before* it as a
  molecule type and *after* it as a component name.

  The engine builds a RankedComplex straight off the pool — `type_index`,
  `comp_type_index`, `state_index` and `bond_partner` are all already dense
  integers — into buffers the next election reuses. The rank tables are
  model-wide and built at the first rule that elects, so a model without the
  construct never builds them.

  Where refinement leaves two molecules genuinely interchangeable (a ring, a
  homo-oligomer) the canonical order is the one belonging to the
  lexicographically minimal *render*, so only the strings can pick it and
  the election falls back to the `ComplexGraph` path. That gate is
  deliberately weaker than a canonical leaf: individualization only ever
  splits color classes and refinement never reorders colors that already
  differ, so once the molecule colors are distinct no leaf the search could
  reach can reorder them. `canonical_order` uses the same relaxed gate now
  and skips the search entirely on a complex whose only residual symmetry is
  *within* a molecule — 277 of the 304 complexes the property test sends to
  the search.

  Measured on #53's probe (best of 5, `--preset release`), a tagged catalyst
  chain that a toggle rule edits on essentially every event, against the
  same model with the election gated off:

  | | 5-subunit catalyst | 20-subunit catalyst |
  |---|---:|---:|
  | no election | 0.079 s | 0.062 s |
  | before (`d9880f4`) | 0.286 s | 0.515 s |
  | after | 0.117 s | 0.316 s |
  | election overhead | 0.207 → 0.038 s (**5.4x**) | 0.453 → 0.254 s (1.8x) |

  What is left is the refinement itself. The engine → graph bridge is down
  from 23% of the added cost to 2–3%, and 97% of what remains is 1-WL
  refinement recomputed from scratch on every edit. The 20-subunit column is
  the honest one: allocation was never the whole cost on a large complex,
  and making the refinement itself cheaper — an incremental or cached order
  — is a different change with a different blast radius.

  The `.species` batch sweep, the canonicalizer's other consumer, got faster
  too, with unchanged output: both entry points now build the port graph
  into one shared reusable core with a flat edge list rather than a
  `vector<vector<int>>` per vertex, and refinement stops on a discrete
  partition instead of spending one more round proving it is stable.
  `.species` is byte-identical on `machine`, `ensemble`, `egfr_net`,
  `egfr_nf_iter5p12h10`, `bench_tlbr_yang2008`, `rm_tlbr_rings` and `tlbr`
  (whose largest complex is 2757 molecules); the `tlbr` sweep runs 282.8 s →
  243.8 s.

  `canonical_test` is what pins the two rankings together: every hand-built
  shape and every complex the property test generates also goes through
  `canonical_order_fast`, checked against `canonicalize().mol_order` — 7800
  answers and 54 declines — and the generator's alphabet gained the
  prefix-name and prefix-state shapes that are the only way the rankings
  could silently disagree. Dropping the `~` from `rank_component_names`
  fails 354 assertions, two of them in the deterministic hand-built cases.
  29/29 corpus guard tier, 88/88 feature coverage, 37/37 ctest under both
  release and ASan.

- **`ctest --preset asan` instruments the test executables too.** The
  sanitizer flags reached the library, the vendored expression object
  library and the two CLIs, but not `tests/`. libc++ emits container
  annotations only from instrumented code, so a test TU that hands a
  `std::vector` to the library to fill desynchronizes them, and ASan reports
  a container-overflow against an ordinary `resize` inside the library —
  the same failure mode the `rm_bngsim_expr` case already documents, one
  boundary further out. The flags are now applied directory-wide below the
  CLI targets, which also means the test harnesses' own code is sanitized
  for the first time. All 37 tests pass with it on.

- **The `TotalRate` keyword now warns at load, and is refused where RM and
  NFsim genuinely disagree.** BioNetGen does not implement TotalRate for network
  simulations — `RxnRule.pm` carries the TODO saying it is "currently
  implemented only for XML network-free output" — and `generate_network`
  duly writes the rate law into the `.net` as an ordinary rate constant, so
  the ODE integrates plain mass action and the observable crashes to zero
  where NFsim holds it flat. There is no BNG2 result to check a TotalRate
  model against.

  That leaves NFsim as the only implementation, and it disagrees with RM.
  NFsim expands a rule whose reactant pattern has interchangeable components
  into one independent reaction class per permutation (`_R1_sym1`,
  `_R1_sym2`, ...). For an ordinary rate law that is correct, since the
  matches partition across the classes and sum back; under TotalRate every
  class returns the *whole* total rate, so the rule runs at
  `rate x #{permutations whose reactant lists are all non-empty}`. Measured
  on `C(s) + D(t)`: **1.00x** with one free site, **2.02x** with two,
  **2.98x** with three, and 1.00x again when every C has the same slot
  pre-bound, since only one permutation is populated then. That factor counts
  NFsim's internal reaction classes rather than anything in the model — it is
  capped by the permutation count however many molecules exist, and steps down
  as classes empty. BNG2's network expansion writes the statistical factor per
  species and live (`3*_rateLaw1`, `2*_rateLaw1`, `_rateLaw1`), implying a
  third number again.

  All three disagree, so RM refuses **those** rules instead of silently
  picking a reading. The test is structural and deliberately conservative: a
  TotalRate rule is refused when any reactant pattern touches a component
  whose molecule type declares two or more of that name, which is what lets a
  pattern component land on more than one slot.

  Every other TotalRate rule warns and runs. RM reads the keyword the way
  BioNetGen documents it in `RateLaw.pm` ("If true, this ratelaw specifies the
  Total reaction rate") — the propensity is the rate law's value — and NFsim
  computes the same thing there, so the two agree. Measured on unimolecular,
  heterodimer, homodimer carrying `symmetry_factor="0.5"`, zero-order
  synthesis, and a rate law that is a function of an observable. The warning
  exists because nothing can check such a model against BioNetGen's own result.

  Refusing every TotalRate rule was considered and rejected once the blast
  radius was measured: it would take out six models from **NFsim's own
  validation suite** (`v21`–`v26`), `oscSystem` in RM's corpus suite, and
  around eleven curated RuleHub biology examples, every one of which RM runs
  in agreement with NFsim today (verified over 12 seeds each: `oscSystem`
  worst z = 1.04, `r21` z = 1.40). Neither the basicmodels suite nor
  `oscSystem` is reachable from `ctest` or the guard tier, so that breakage
  would not have surfaced in CI.

  `ft_total_rate` was rewritten along with this. The model it replaces fired
  four events over its whole run and could not tell TotalRate from anything
  else, and its header stated the semantics wrongly ("binding rate is exactly
  `k*[A]*[B]`" — TotalRate removes the counts entirely rather than merely
  suppressing site multiplicity). The new one fires 313 events across three
  arms whose closed forms separate TotalRate from mass action in both
  directions: two linear-decay arms where mass action is curved, and an
  observable-driven arm that is exponential where mass action would be
  hyperbolic. Every arm names its components distinctly, so it is the warned
  shape rather than the refused one, and it scores tz = 2.36 against a
  20-replicate NFsim ensemble.

### Fixed

- **The clang-tidy gate had been off, and said nothing about it.** Three
  independent faults, only one of which was visible:

  1. `.git/hooks/` was empty in a working clone — `pre-commit install` had
     never been run, so *no* hook fired on commit: not clang-tidy, not
     clang-format, not ruff, not the whitespace fixers. Nothing in the
     README said to install them.
  2. The hook was `language: system`, so it ran only for whoever happened
     to have an LLVM on `PATH`, and `scripts/clang-tidy-staged.sh` exited
     **0** when it found none. pre-commit prints nothing for a passing
     hook, so a machine with no clang-tidy reported a clean lint.
  3. Installing one by hand does not necessarily help: macOS SDK 26's
     libc++ headers use `__builtin_clzg` / `__builtin_ctzg`, which
     front-ends older than clang 19 lack, so clang-tidy 18 emits thousands
     of `clang-diagnostic-error`s out of `<algorithm>` and `<charconv>`
     and never reaches any RuleMonkey code.

  The hook now takes its binary from pre-commit — `language: python` with
  `additional_dependencies: ['clang-tidy==21.1.6']`, the same wheel family
  the clang-format hook already pins — so it cannot be missing or the
  wrong version, and no system LLVM is required. "Not found" is now a real
  fault and exits non-zero, and the script carries a front-end version
  floor that names the SDK mismatch instead of drowning in it. Both
  failure paths print `NOT LINTED` rather than a hint that scrolls past.
  The one remaining soft path is a missing
  `build/release/compile_commands.json`, which a fresh clone genuinely has
  until the first `cmake --preset release`; it too now says plainly that
  nothing was checked.

  With the gate actually running, clang-tidy 21.1.6 over all 44 C++
  translation units reports **seven** findings, all `misc-const-correctness`
  and none in code this repo has touched recently: six variables that can
  be declared `const` (three in `canonical_test`, one each in
  `seed_build_test` and `species_enumeration_test`, plus a `char*` pair in
  `simulator.cpp`) — fixed. The seventh is a false positive: the check
  wants `const char*` for `strtol`'s endptr, whose parameter is `char**`,
  so the suggested form does not compile; that site takes a targeted
  `NOLINT` with the reason, next to the `NOLINTBEGIN` it already carried
  for a different checker's false positive on the same lines.

  `pre-commit run --all-files` is now green on every hook. Verified
  end-to-end rather than by inspection: a planted `int* p = 0;` in
  `smoke_test.cpp` fails the hook on `modernize-use-nullptr`, and a
  clang-tidy 18 on `PATH` is refused by name instead of reporting the
  standard library as broken.

  The README gains a "Contributing setup" section, since the gate is only
  as real as the `pre-commit install` nobody was told to run. `.clang-tidy`
  loses a stale comment claiming `WarningsAsErrors` was intentionally
  empty — it has not been empty since the ratchet was flipped — and gains
  the front-end version floor.

  And a `clang_tidy` CI job now runs the gate somewhere it cannot be
  skipped, which is the durable half of this: a hook only runs in a clone
  where someone ran `pre-commit install`, and nothing notices when that has
  not happened. The job runs the *hook*, not its own clang-tidy invocation,
  so the version pin, the check list and the wrapper script are the same
  ones a developer gets locally rather than a second spelling that drifts.
  It configures without building — nothing in this tree is generated, so
  `cmake --preset release` alone produces the `compile_commands.json`
  clang-tidy reads — and is independent of the `build` job, since a broken
  build leg should not withhold lint feedback.

  The job runs only the clang-tidy hook. clang-format, ruff and the
  whitespace hooks sit in the same "runs only if installed" position and
  could be swept in by dropping the hook id from that step; that is left as
  a separate decision rather than taken here.

  It earned itself on the first run, with a finding no amount of local
  macOS linting could have produced: `bugprone-misplaced-widening-cast` on
  `static_cast<unsigned long long>(rej_sum + q.fc_total_matches)` in
  `engine_profile.hpp`. `uint64_t` is `unsigned long` on Linux, so the cast
  is a no-op applied after the addition and the check fires; on macOS the
  two types coincide and it does not. No defect either way — both operands
  are already 64-bit, so nothing can overflow ahead of the cast — but the
  widening now sits on an operand rather than the sum, which is what the
  check asks for and is honest about the `%llu` it feeds.

- **A rule's per-molecule table was sized by the whole molecule arena
  *and* by every field any rule shape could want, so a rule that reads two
  of eleven fields was charged for all of them (issue #71).** After #70,
  `init_rule_states` was a third of an `error.bngl` session build at
  `scale=100`, and five sixths of that third was allocating, zeroing and
  re-walking side tables — not matching. The tables cost resident memory on
  the same terms: 785 MB of a 1.69 GB peak on a model with two rules that
  build one.

  `PerMolRuleData` was 80 bytes: two embedding counts, a cache flag, a
  three-field shared-component split, four doubles of local-rate state, and
  two pure-context complex ids. It is now 12 bytes — the counts and the
  flag, which every rule reads — with the other nine fields moved to a
  `PerMolRuleAux` row in a second table a rule allocates only if its shape
  reads one. The predicate (`rule_needs_mol_aux`) is the union of what
  writes each group: two reactant slots for the shared-component split, a
  local rate law (including the FunctionProduct B side) for the four rate
  fields, a pure-context slot for the complex ids. A unimolecular,
  non-local rule with no pure-context slot — the shape this is about —
  allocates no second table at all, so its row goes 80 bytes to 12; a rule
  that does need one pays 76 across the two, and neither its build nor its
  event loop measurably moves.

  Both tables are indexed by mol_id and either the aux table is empty or it
  is *exactly* as long as the other, which is what `grow_mol_tables` is
  for: an aux table shorter than `mol_data` is an out-of-bounds read on the
  next event, not a wrong number. All 63 `mol_data` sites were audited onto
  one side or the other.

  The `cache_init` flag goes the same way — #71's cheapest item. It is now
  part of the value the rescan fills the table with, instead of a second
  full pass over the table writing one byte per 80-byte row afterwards. A
  full rescan is what validates every row, including the zero-valued
  defaults for molecules of types the rule cannot seed on, so the fill can
  carry the flag. The default in `PerMolRuleData` itself stays false, and
  the on-demand growth path still takes it: those rows are for molecules
  born since the rescan, which genuinely have not been computed.

  Measured on macOS arm64, release preset, `error.bngl` at `scale=100`
  (4 661 136 molecules, three rules, two of which build a table), `t_end=1`
  so the SSA loop is empty. Best of seven paired runs — the two binaries
  alternated so both saw the same machine, which had other work on it:

  | | before | after | |
  |---|---:|---:|---:|
  | `init_rule_states` | 0.385 s | 0.144 s | **2.7x** |
  | session build | 1.147 s | 0.886 s | 1.3x |
  | peak RSS | 1.74 GB | 1.10 GB | |

  Per rule, inside the phase, same seven paired runs:

  | | `delta()->0` | `I(N!1).I(N!1)->0` |
  |---|---:|---:|
  | seed-type population | 1.55e6 | 3.11e6 |
  | `mol_data` fill | 0.087 → 0.009 s | 0.081 → 0.009 s |
  | the embedding scan | 0.017 → 0.008 s | 0.051 → 0.032 s |
  | Fenwick init + fill | 0.031 → 0.031 s | 0.052 → 0.051 s |
  | `cache_init` tail pass | 0.032 s → gone | 0.021 s → gone |
  | total | 0.174 → 0.048 s | 0.213 → 0.099 s |

  The fill goes down by more than the 6.7x the row width alone predicts,
  and the embedding scan gets faster without being touched at all: writing
  one count into a 12-byte row touches a fraction of the cache lines an
  80-byte row does. The Fenwick column is flat, which is both expected —
  that code is untouched here — and a check on the harness. What is left of
  the phase is now mostly that Fenwick `init`, a 37 MB zeroing sized by the
  arena for a tree only ever updated at ids of one type. That is #71's
  fourth item, and it is left alone with the third for the reason the issue
  gives: both need a stable dense per-type slot in the pool, which is a
  design question and should be measured before it is attempted.

  Where it bites hardest is the shape #71 sharpened it to — the same model
  plus four molecule types seeded with one molecule each and one
  unimolecular rule apiece. Rules that between them can see four molecules:

  | | before | after | |
  |---|---:|---:|---:|
  | `init_rule_states` | 0.908 s | 0.206 s | **4.4x** |
  | session build | 1.682 s | 0.986 s | 1.7x |
  | peak RSS | 3.12 GB | 1.33 GB | |

  Nothing regresses for the shapes that do carry an aux row. On
  `A(b) + B(a) <-> A(b!1).B(a!1)` with 2e6 of each, peak RSS goes 1.40 GB
  to 1.06 GB — the forward rule saves 4 bytes a row, and the unbinding
  rule, which is unimolecular, saves 68 — and session build is 3.92 s
  against 3.99 s, best of three. If that 2% is real rather than noise it is
  the second `assign` on a row that only got 4 bytes narrower, which is
  what an aux-carrying rule trades for the arithmetic above. Three
  feature-coverage stress models run to a 20 000-unit horizon are unchanged
  within 1% (13.47/8.36/10.37 s against 13.02/8.44/10.41 s), which is the
  event-loop check: the aux row costs one predicate test and one more
  indexed table per updated molecule.

  `rule_table_shape_test` is the new gate: one rule of each aux group plus
  one of the no-aux shape in a single pool, every reacting molecule made by
  a maker rule so each arm is priced on rows created after the session
  build sized the tables. Structural assertions are exact; the four arm
  means are 4-sigma bands against analytic references (the CME steady state
  for the homodimer, plain exponential survivor counts for the rest), so
  each arm is priced rather than merely observed. Deleting the aux-table
  growth segfaults it. `docs/internals.md` gains a "Step 4 — the per-rule
  tables" section covering the two-table invariant.

  ctest 46/46 release and ASan; feature_coverage 89/89 and basicmodels
  29/29 against the vendored NFsim ensembles.

- **`init_species` re-resolved every seed species once per copy, and grew
  the pool by doubling while it did (issue #67).** After #65, session
  build was two thirds of an `error.bngl` replicate's setup and two
  thirds of *that* was `init_species` — populating the pool from the seed
  species, at a flat ~330 ns per molecule, before any matching,
  propensity or observable work happens. It is now ~110 ns per molecule:
  at `scale=100` (4.66e6 molecules) `init_species` takes 0.52 s where it
  took 1.55 s, session build 1.12 s against 2.05 s, a full `t_end=2400`
  replicate 8.6 s against 10.3 s, and peak RSS 1.93 GB against 2.23 GB —
  for a byte-identical `.gdat`.

  Three things, none of which changes what the pool ends up holding.

  **The per-copy work did not depend on the copy.** For each of the
  `count` copies of a seed species, the walk rebuilt the species-XML
  component index → molecule-type component index mapping — a
  `std::unordered_map<std::string, int>` keyed by component name, built
  and thrown away once per molecule — re-ran the state-name lookups
  through it, re-resolved the bond endpoints against it, and allocated
  two scratch vectors. All of that is a function of the `SpeciesInit` and
  the model alone. Each seed species is now resolved once into a
  `SeedTemplate` and the copies are stamped out from it; on
  `I(N!1).I(N!1)` at 1.55e6 copies, the old shape meant 3.1e6 name-keyed
  hash maps built and destroyed.

  **Nothing was reserved.** `molecules_`, `components_`, `comp_to_mol_`,
  the per-type indices, the complex map and the move side-channel all
  grew by doubling through the whole walk, from totals the first pass now
  knows before the second starts. `molecules_` is the expensive one —
  each element carries a `comp_ids` vector, so every reallocation moves
  4.7e6 of them. `AgentPool::reserve_for_seed` is a capacity hint and
  nothing more: a total that is short, or that does not fit the `int` the
  pool addresses molecules with, costs only the growth it failed to
  pre-empt. The complex map is sized by the complexes the seed *leaves*
  rather than the one-per-molecule the walk creates before its bonds
  merge them, since a hash map never gives a bucket array back.

  **A newborn complex marked itself dirty.** `add_molecule` inserted the
  complex it had just created into `cxs_dirty_`, the canonical-label
  cache's invalidation set — one hash insert per molecule, which left the
  set holding every complex in the model. It was information-free the
  whole time, by the set's own documented contract: complex ids come from
  a counter that never recycles, so a just-born complex cannot have a
  cache entry, and `cached_label_of` already treats an absent id as
  dirty. Every mutator that edits an *existing* complex still marks.
  `cx_edits_` is a different question — its reader has to see the birth
  rather than ask whether an edit is outstanding — so that one still gets
  the id.

  Verified byte-identical against a build of the parent commit: every
  `.gdat` over the 187 model XMLs in `tests/reference/nfsim`,
  `tests/models/feature_coverage` and `tests/cpp`, and every
  `--species` census over the 160 of them that produce one.

  `seed_build_test` pins the seed walk itself. Its model's seed species
  list their components out of the molecule type's declaration order —
  which BNG2's `writeXML` normalizes away, so the XML is hand-edited on
  top of BNG2's output and the test refuses to run if that ordering is
  ever regenerated out of it. Every count it makes is a hand-derived
  integer, and each of the three ways the hoist could go wrong (a bond
  endpoint resolved without the mapping, duplicate component names
  collapsed onto one slot, the states applied to the first copy only) was
  introduced on purpose and confirmed to fail it. Its last arm runs past
  the seed, where `kCanonicalCacheSelfCheck` (Debug and ASan) covers the
  dropped dirty mark; deleting a mark that *is* needed was likewise
  confirmed to abort there.

  What remains of the issue is its last direction: `complex_members_` is
  a hash map to a heap-allocated vector per complex, the overwhelming
  majority of them singletons, and it is now most of what `init_species`
  still spends. That one changes `AgentPool`'s representation rather than
  one loop — the pool is read from the matching layer, the propensity
  layer, the observable tracker, save/load and species enumeration — so
  it stays open.

- **The `basicmodels` parity suite could not run from a clean clone: its
  reference manifest gated on gitignored files (issue #63).**
  `MANIFEST.tsv` is bootstrapped by walking the live reference tree, and
  the machine that bootstrapped `tests/reference/basicmodels/` still held
  the `replicates/` scratch its ensembles had been aggregated from — so
  2900 of that manifest's 3018 rows named per-rep `.gdat` files that
  `.gitignore` (`tests/reference/*/replicates/`) keeps out of the
  repository. Verification hard-fails on a missing file, so
  `harness/basicmodels.py` refused to start anywhere those had not been
  generated locally, over files no verdict reads: the comparison opens the
  aggregated `ensemble/*.{mean,std,tint}.tsv`, which are vendored, and
  `PROVENANCE.md` describes `replicates/` as scratch in the same breath as
  writing it. Regenerating was not a way out either — that path re-runs
  NFsim 100 times per model against a locally patched NFsim build that is
  not vendored. The suite is the manual gate for 29 curated models and
  (like `oscSystem`) is not reachable from `ctest`, so an engine change
  that moves every trajectory had nothing to check itself against.

  The manifest now covers the vendored artifacts and only those, which is
  what the corpus manifest next to it already did: the tree walk prunes
  `replicates/` and OS noise (`.DS_Store`, `Thumbs.db`) on both write and
  verify, so one manifest serves a clean clone and a machine holding a
  freshly regenerated ensemble alike. The second case used to fail the
  other way — as `untracked file in reference tree` — which is why the
  fix has to be symmetric rather than a one-off row deletion. The
  basicmodels manifest is rewritten accordingly, dropping 2900 rows and
  keeping 118. Every hash it keeps is byte-identical to the one it
  carried, bar `PROVENANCE.md`'s — edited here to state the coverage rule
  next to the `replicates/` line that already called them gitignored — so
  no reference data changed. `harness/basicmodels.py` then runs all 29
  models green from a tree with no `replicates/` in it.

  Four smaller defects in the same helper — three on the path anyone
  hitting a manifest problem walks, the fourth caught by the new test on
  CI's Windows leg:

  - Rows reaching outside that coverage are now one diagnostic naming the
    count and the first path, rather than one `missing reference file`
    line per file. The failure in #63 printed 2900 of them and pushed the
    line saying what to do about it off the screen; the list is capped at
    20 with an `… and N more` tail for the same reason.
  - The abort's regeneration hint named `--generate-refs` /
    `--force-refs` regardless of who called it. `benchmark_full.py` (and so
    `basicmodels.py`) accepts neither — it takes `--no-verify-manifest
    --write-manifest` — so the one line telling you how to fix the tree
    named flags that would have errored. Each caller now supplies its own.
  - Manifest paths are written `/`-separated on every platform.
    `os.path.relpath` yields backslashes on Windows, and while the
    existence check tolerates either, the untracked-file check compares
    the two spellings directly: a manifest written on Windows would have
    reported every file in the tree as untracked. Latent until now
    because nothing in the build verified a manifest.
  - The `# root` header line was `os.path.relpath(ref_root)` — relative
    to the working directory rather than to the repository, which the
    format comment claimed and which is what the committed manifests
    actually hold. So the line moved with wherever the harness was
    invoked from, and on Windows, writing a manifest for a tree on
    another drive than the CWD raised `ValueError: path is on mount 'C:',
    start on mount 'D:'` and took the writer down with it. Now
    repo-relative, falling back to an absolute path for a tree outside
    the repository. The line is cosmetic — `read_manifest` skips comments
    — but nothing else in `write_manifest` could fail, so it was the
    whole failure.

  Windows checkouts could never have passed the gate at all, and this is
  the first change that would have noticed: Git for Windows converts text
  files to CRLF on checkout, so every `.tsv` and `.xml` under a reference
  root hashes differently there — CI's Windows leg reported 306 of 306
  `feature_coverage` files as drifted, against a tree that is pristine
  (the live hash it printed is exactly the CRLF rewrite of the committed
  bytes). A new `.gitattributes` marks the hashed trees `-text`, so their
  bytes survive a checkout unchanged everywhere. `-text` rather than
  `text eol=lf`: normalising on the way in as well would mean a reference
  regenerated on Windows — Python text mode writes CRLF — is committed as
  something other than what its author hashed, and the manifest would
  fault on the next fresh checkout.

  `ctest` gains `ref_manifest_test` (`tests/harness/test_ref_manifest.py`,
  stdlib-only, no `rm_driver`), which pins the coverage rule on a
  synthetic tree and verifies every committed `MANIFEST.tsv` against the
  tree beside it. On CI that checkout is a clean clone, so a manifest that
  reaches outside its coverage now fails the build on all three OS legs
  instead of surfacing when someone reaches for a parity suite.

- **A rule with no reactant pattern to seed on still got a per-molecule
  table, one 80-byte entry per molecule in the pool, that nothing ever
  read (issue #67).** `rescan_all_molecules_for_rule` sized and zeroed
  `rs.mol_data` before deciding whether the rule had a seed molecule at
  all, so a zero-order synthesis (`0 -> X()`) took the allocation and the
  373 MB of zeroing on a 4.7e6-molecule pool, then returned from the
  early-out below it without ever indexing the table. The n-ary early-out
  a few lines above already returned without building `mol_data`, on the
  same reasoning; this extends that to the seedless case, and clears the
  table rather than leaving it sized.

  Every indexed read of `mol_data` elsewhere either bounds-checks against
  its size or resizes first, so an empty table gives the same "no entry
  for this molecule" answer a zeroed one gave.

  The cost is per rule of that shape and per session build, so it shows
  up mainly as resident memory. On `error.bngl` at 4.7e6 molecules, whose
  one such rule is `0 -> tim()`:

  | | peak RSS | session build |
  |---|---:|---:|
  | before | 2.21 GB | 2.26–2.39 s |
  | after  | 1.88 GB | 2.16–2.26 s |

  and on the same model with four more zero-order rules added:

  | | peak RSS | session build |
  |---|---:|---:|
  | before | 3.46 GB | 2.67–2.72 s |
  | after  | 1.91 GB | 2.14–2.20 s |

  — i.e. the added rules now cost essentially nothing, where each one
  used to cost 373 MB and about 0.09 s.

  `pool_churn_test`'s synthesis arm is the regression guard: its
  `0 -> W()` rule is exactly this shape, and the test pins the population
  a molecule limit stops it at to an exact integer. That arm now also
  prices the rule — a ±6 sd band on its Poisson yield — rather than only
  observing that it fired, since a seedless rule's propensity has nothing
  to come from but its rate.

- **Session build counted every tracked observable twice, and seeded it the
  slow way both times (issue #65).** With the per-event costs from #62
  gone, `Engine::initialize()` was what a run spent most of its wall
  clock on: 14.0 s of a 22.0 s replicate of `error.bngl` at 4.7e6
  molecules, against 8.1 s for the SSA loop it was setting up. It is now
  2.7 s, and the replicate is 9.9 s.

  **The observable walk and the tracker's tables were computed
  independently.** `initialize()` ran `compute_observables()` — a
  from-scratch walk of every observable — and then
  `init_incremental_observables()`, which recomputed the same
  per-molecule embedding counts to seed `obs_mol_contrib`. On
  `Species Initiator_I2_SSA I(N!1).I(N!1)` over 3.1e6 `I` molecules that
  is 3.1e6 multi-molecule embedding counts, done twice. The second
  computation subsumes the first: the contribs are the walk's
  per-molecule terms and the per-complex pass flags are its per-complex
  verdicts. The tracker now runs first and settles `obs_values` out of
  its own tables (`seed_tracked_obs_values`), and the walk that follows
  skips every observable the tracker keeps.

  **Seeding took the generic matcher where the per-event path takes
  shortcuts.** `incremental_update_observables` computes a molecule's
  contribution through a dispatch — a structurally unconstrained pattern
  (`T()`) contributes one per molecule of its type with no matching at
  all, a 2-molecule/1-bond pattern goes through the
  `count_2mol_1bond_fc` specialization, and only what is left falls to
  the generic BFS. The seeding loop called the generic BFS
  unconditionally, including for patterns whose `FastMatchSlot` had been
  built a few lines above it in the same function. The generic path
  allocates per call — `bond_infos`, a vector-of-vectors adjacency, a
  `std::function` closure per seed embedding, and inside
  `count_embeddings_single` a `std::vector<bool>` and a vector-of-vectors
  of candidates — and the init profile was about half allocator traffic
  as a result. Both callers now share one `tracked_obs_contrib`, so
  seeding gets the shortcuts and the two can no longer answer
  differently.

  The Species per-complex seeding also deduped complexes with a fresh
  `std::unordered_set<int>` — the same thing #62 replaced in
  `evaluate_observable`, in the other copy of that walk. It now uses the
  same generation stamp.

  Measured on `error.bngl` at `scale=100`, `t_end=2400, n_steps=240`, for
  a byte-identical `.gdat`:

  | | session build | SSA loop | process wall |
  |---|---:|---:|---:|
  | before | 14.0 s | 8.1 s | 22.0 s |
  | after  |  2.7 s | 7.2 s |  9.9 s |

  Session build against system size, `initialize()` alone:

  | population (`I20`) | before | after |
  |-------------------:|-------:|------:|
  |             1.55e4 | 0.083 s | 0.024 s |
  |             1.55e5 | 0.862 s | 0.212 s |
  |             4.66e5 | 2.753 s | 0.676 s |
  |             9.32e5 | 7.221 s | 1.299 s |
  |             1.55e6 | 14.186 s | 2.369 s |

  Still linear in population, which it has to be — the tracker's tables
  are per-molecule — so this is the constant, not the exponent. The SSA
  loop comes out about 11% faster too, reproducibly across runs; the
  likely reason is that build no longer leaves the allocator holding
  hundreds of millions of freed small blocks, but that was not chased
  further.

  What the seeded values are now rests on an equality rather than on a
  second measurement, so it is checked both ways.
  `kLocalObsTrackInvariant` (Debug and ASan builds, the gate that already
  covers the contrib table's other fast-path read) re-derives every
  tracked observable from scratch at init and aborts on a mismatch;
  `init_obs_seed_test` pins the seeded values in Release, against
  hand-computed counts over the model's seed species and against a
  from-scratch walk of the identical initial pool.

- **A three-rule model could not finish one replicate in eleven hours,
  because per-event cost grew with the molecule population (issue #62).**
  `error.bngl` is three rules over 4.7e6 molecules, every rate law a
  constant, and NFsim's per-event cost on it is flat at 5–7 µs from
  1.5e4 molecules to 9.3e5. RuleMonkey's was 53 µs/event at the small
  end and 2798 µs/event at 100x the population — the two engines
  executing essentially the same number of events, so the growth was
  pure per-event cost. None of it was in the matching or propensity code
  the model's shape would point at (the issue's own triage note flags
  `percx_resum_rates`, which this model never reaches: its rate laws are
  all constants, so the local-rate path never engages). Three pool walks
  were responsible; each is now an O(1) side table.

  **Removing a molecule from its type index was O(population of that
  type).** `AgentPool::delete_molecule` erased the id from
  `type_mol_index_[type]` with an order-preserving `std::remove` — a
  linear scan over every live molecule of that type, plus the shift,
  charged on every deletion. It was 59% of a sampled profile of the SSA
  loop at 2.8e6 molecules, and it is not a niche path: any rule that degrades,
  dissociates into nothing, or changes a molecule's type deletes. The
  list's order carries no meaning — `molecules_of_type` consumers scan it
  whole or sample from it by weight, and the Fenwick samplers key on
  molecule id rather than on position — so removal is now a
  swap-with-back through a `type_mol_pos_` side table. On the reproducer
  at `t_end=60` this alone took the SSA loop from 32.6 s to 2.6 s.

  **The `-gml` molecule limit re-counted the whole pool on every event.**
  The limit check called `active_molecule_count()`, which scanned
  `molecules_` for `.active`. A cap that cannot possibly bind therefore
  cost exactly what a binding one would, per event, forever.
  `error.bngl` declares `gml=>2.147e9` — an eyeballed `INT_MAX`, i.e.
  "no limit" — over 4.7e6 molecules: measured, that cap cost 104.6 s
  against 32.3 s for the identical run with no cap at all, and it now
  costs nothing measurable (1.05 s against 1.23 s, inside run-to-run
  spread). The three models in a 210-model NFsim-parity sweep that could
  not produce a RuleMonkey replicate are exactly the three declaring
  `gml >= 1e9`. The count is now a tally maintained by `add_molecule` /
  `delete_molecule`.

  **A Species observable outside the incremental tracker rebuilt a hash
  set of every complex, at every sample.** The full walk deduped
  complexes by inserting each complex id into a fresh
  `std::unordered_set<int>` — one node allocated per complex, per
  observable, per sample time — and then copied each complex's member
  vector by value to walk it (`auto` where the accessor returns a
  reference). Species observables whose pattern is a bare `T()` are
  deliberately excluded from per-event tracking, because a model like
  `rm_tlbr_rings` declares 300 of them and tracking all 300 costs more
  than the walk; that trade assumed the walk was cheap, which it was not
  at 1.5e6 complexes. Dedupe is now a generation stamp on the member
  molecules — molecule ids are dense, so no hashing is needed — the
  member list is taken by reference, and a molecule with no bonded
  component skips the complex lookup entirely, since it is alone in its
  complex by definition. Sample-time cost on the reproducer fell 6.4x.

  Measured end to end on `error.bngl` at its own `scale=100`, on one
  machine, at the model's own `t_end=2400, n_steps=240` horizon: the SSA
  loop goes from 1223 s to 9.9 s, and the whole `rm_driver` invocation
  from 1239 s to 25.4 s, for a byte-identical `.gdat`. (The reporter saw
  more than eleven hours where this machine takes twenty minutes; at
  `t_end=60` the two agree to within a percent, so the difference is in
  the horizon or the host, not in the effect.) Per-event cost across
  system size, SSA-loop wall over `events=`, at `t_end=60` (the population
  column is the model's own `I20`, as in the issue's table; the pool
  holds three molecules per unit of it):

  | population (`I20`) | before (µs/event) | after (µs/event) |
  |-------------------:|------------------:|-----------------:|
  |             1.55e4 |                53 |             10.6 |
  |             1.55e5 |               243 |             17.2 |
  |             4.66e5 |               839 |             38.4 |
  |             9.32e5 |              1664 |             65.3 |
  |             1.55e6 |              2798 |             86.8 |

  The remaining growth at this horizon is the fixed per-run cost — first
  touch of the pool's pages, and seven whole-pool sample points — spread
  over few events; at the real horizon the same configuration runs at
  22.8 µs/event.

  Two O(1) checks guard the new tables on every deletion under
  `kPoolIndexSelfCheck` (compiled in for Debug and ASan, out of Release,
  like the canonical-cache and local-observable self-checks): the
  recorded type-index position against the type list's own contents, and
  the live tally against `molecules_` minus the free-id list.

- **A run stopped by a molecule limit reported its last sample rows one
  event behind the pool they described (issue #62).** The limit check sat
  before the observable refresh, and the breaching event is not rolled
  back, so the trailing sample rows — written after the SSA loop exits —
  carried pre-event values for every incrementally tracked observable
  while the untracked ones, which full-walk the live pool at sample time,
  carried post-event values. The same row disagreed with itself, and with
  `get_molecule_count()` on the session left behind. The check now runs
  after the refresh; the cost is one event's worth of propensity update
  that is then discarded.

- **A rate law built on `reactant_N()` had a propensity of zero, so the rule
  never fired and the run finished with no events at all (issue #59).** A
  BNGL model that wants the match count of a rule's Nth reactant pattern
  inside that rule's own rate law declares an empty function by that name
  and writes it into the rate expression:

  ```
  begin functions
     reactant_1()
     reactant_2()
     NucF()=if(Dimer<1,kNuc,0)
  end functions
  A(b) + A(b) -> A(b!1).A(b!1)   reactant_1()*reactant_2()*NucF()
  ```

  BNG2 emits both placeholders as ordinary functions with an empty
  `<Expression>`, so nothing in the XML says what they mean — the name is
  the whole convention, and NFsim reads it the same way. RM read them as
  what they literally are, a function with no body, i.e. the constant zero,
  so every rate law built on one evaluated to zero. The rule's propensity
  was zero, no reaction ever fired, and nothing was reported: the run
  finished instantly with the initial condition unchanged and an exit code
  of zero. The rate law is not a total rate, so the count the placeholder
  supplies is multiplied in on top of the ordinary mass-action count, which
  is what NFsim does and what RM now does.

  The engine resolves the placeholder against the rule being priced, using
  the same per-pattern match count that already multiplies the rate, and
  the load-time walk records per rule which counts its rate function reads
  so models that never mention the construct pay nothing.  A placeholder is
  not reported as a function: it has no value outside the rule that reads
  it, and NFsim drops an empty-bodied declaration rather than creating a
  function for one, so `function_names()`, `get_function_values()`,
  `Result::function_names` and `rm_driver --print-functions` carry no column
  named after one.  Rate laws that read a placeholder, `_rateLaw1` and the
  like, report the value the engine actually uses. Two shapes are
  refused at load rather than resolved to zero: a count of a reactant the
  rule does not have (`reactant_2()` on a one-reactant rule), and a count
  read from a rate law that is also a local, per-instance function.

  A third shape warns. A rate law that is not a total rate states the rate
  *per set of reactants*, so the propensity is that value times the reactant
  match counts, and `reactant_N()` supplies those same counts a second time
  inside the rate. A bimolecular rule written `reactant_1()*reactant_2()*k`
  is therefore priced at `(N1*N2)^2 * k`, not `N1*N2*k`, which is not what
  the BNGL source reads like. NFsim does the same thing, so RM is not
  diverging and does not refuse; the warning exists so that reading is named
  at load instead of found by wondering why a rate constant is off by orders
  of magnitude. `TotalRate` is the one shape where the construct means what
  it looks like — there the rate function states the whole propensity, so
  `reactant_1()*k` is exactly mass action at `k` — and the warning is silent
  there. Verified against NFsim on a `TotalRate` model: RM 35.8, NFsim 37.7
  over 20 seeds each, against the `100*exp(-1) = 36.8` the law predicts.

  Reported on `actin_branch_forFitToData.bngl`, whose only initially
  possible reaction is a nucleation rule of exactly this shape, so the whole
  model was inert. It now nucleates and grows a filament, and over twelve
  seeds its subunit count at `t = 10` averages 78.3 against NFsim's 73.9,
  agreeing inside the seed-to-seed spread. The regression test measures the
  propensity itself rather than just checking that something fires: running
  400 replicates of the reported reproducer to exactly the mean waiting time
  of the expected propensity, 62.3% of them have fired against the 63.2% the
  rate predicts.

- **A pure-context reactant pattern carrying a per-molecule local function
  tag was priced at the lowest-id matching molecule, so two identical
  complexes charged different rates (issue #52).** #33 gives a reactant
  pattern the rule does not transform one instance per matching *complex*.
  When that pattern also carries a per-molecule tag, the collapse has to
  decide *which* of the complex's matching molecules prices the survivor,
  and the candidates price differently. RM picked the lowest live molecule
  id — a fact about pool history, not about the complex — so the same
  species answered differently depending on how each copy was assembled.

  On the reproducer, a homodimer catalyst holding one modified and one
  unmodified subunit against 4000 substrate, RM ran **5.5x** fast against
  BNG2 and disagreed with **itself** by 5.7x across two seedings of the
  same chemical state: seeded pre-formed it converted 1.95 substrate by
  `t = 1` against BNG2's 2.00, but assembled during the run it converted
  11.09 — the half-and-half average of the two candidate prices, since the
  assembling rule modifies either subunit with equal probability.

  #33's commit recorded that BNG2 "cannot arbitrate — its network expansion
  has no local functions". That is not what BNG2 does. It expands the rule
  to a `ConstantExpression` obtained by evaluating the tagged observable at
  the **canonically-first** matching molecule of the species, which is a
  property of the species, so every copy of a species prices identically.
  Confirmed against BNG2 2.9.3 by generating the network with the observable
  reading each state in turn: flipping which state it counts flips the
  emitted constant, and both times it lands on the canonically-first subunit
  rather than on the smaller value — so this is neither a default to zero
  nor a minimum.

  RM now takes the representative from the complex's canonical form, via a
  new `canonical::canonical_order` — the ordering the canonical label is
  rendered in, which inherits the label's guarantee that isomorphic
  complexes order corresponding molecules identically. The representative is
  what both prices the complex and fires, so count, price and draw stay
  mutually consistent as #33 left them.

  **The canonicalizer stays off the hot path for everything else.** The
  election runs only where the choice is observable — a pure-context slot
  whose local factor is evaluated *per molecule* and is not the constant 1
  the DOR1 normalization puts on the untagged side — and, within that, only
  on complexes holding more than one match, since one candidate needs no
  election. A complex-wide tag evaluates its observable over the whole
  complex, so every molecule in it prices the same and the cheap lowest-id
  representative is kept. Every rule with no local function keeps it too.
  Measured on the three corpus models that carry per-complex tallies:
  `r08` -0.9%, `machine` +0.7%, `ensemble` -1.5% — all inside run-to-run
  noise, and none of them reaches `canonical.cpp` at all. Where the
  election *is* live the cost is real: a probe with a 5-subunit tagged
  catalyst that every event edits runs **4.3x** slow, three quarters of it
  inside the canonicalizer's allocations. That shape was previously wrong
  by 5.5x, so the trade is taken; making the canonical order allocation-free
  from the engine side is its own piece of work.

  Re-electing on the right events needed one new signal: a state change
  inside a complex reorders its canonical form without moving any molecule
  between complexes, so #33's complex-move side-channel never mentions it
  even though the molecule that should price the complex has changed. The
  pool now also logs edited complex ids, gated on some rule actually needing
  a canonical representative, so it is a predictable branch and nothing else
  for every other model.

  `tests/bng_oracle/models/context_dor_price.bngl` pins it against BNG2's own
  expansion: three columns of the same chemistry that BNG2 gives the
  identical rate — the dimer seeded pre-formed, the identical species
  assembled in-run, and one whose canonically-first molecule carries the
  *larger* value, which is the discriminator against "price at the cheapest
  instance". The suite scores **137 sigma** against the pre-fix engine and
  1.4 after.

- **`MM(kcat,Km)` at the `Km <= 0` boundary: a rate of 0 where the limit is
  `kcat*min(S,E)`, and a NaN propensity the sampler kept firing against
  (issue #46).** The MM branch guarded its division with
  `if (Km + sFree <= 0) return 0;`. That answers 0 where the limit is not 0,
  and does not answer at all where the expression is NaN.

  **`Km == 0` is a removable singularity, not a zero.** With `Km = 0`,
  `sFree = max(S-E, 0)`, so `S > E` already read `kcat*E` but `S <= E` read
  0/0 — and the limit there is `kcat*S`, since `sFree -> Km*S/(E-S)` as
  `Km -> 0+`. The law on the whole `Km = 0` line is `a = kcat*min(S,E)`:
  binding infinitely tight, so turnover is substrate-limited when the enzyme
  is in excess and enzyme-limited otherwise. RM returned 0 instead, which
  left the rule silently inert — and froze a model that *started* in the
  good branch: with `Km = 0, S0 = 200, E0 = 100` the run proceeded at
  exactly `kcat*E` until the substrate decayed to `S == E`, then stopped
  dead at `S = 100` forever, mass conserved, no warning.

  **The approach to that limit was wrong before it was reached.** Taking
  `sFree` as `0.5*(diff + sqrt(diff^2 + 4*Km*S))` cancels catastrophically
  for `diff < 0`, which is exactly the regime `Km*S << (E-S)^2`. On
  `S=100, E=200, kcat=1`, where the limit is 100, the old form read 100.093
  at `Km=1e-12`, **117.392** at `1e-14`, and 0 from about `1e-15` down — so
  a scan or a fit walking `Km` toward zero passed through a band where the
  propensity was silently wrong by an arbitrary amount before going inert.
  `sFree` is now taken in the conjugate form `2*Km*S/(q - diff)` when
  `diff < 0`, which is algebraically identical and adds like-signed
  quantities. Against 60-digit arithmetic over 171 `(S, E, Km)` points, the
  old form's worst relative error is **7.6e-2** (at `S=50, E=1000,
  Km=1e-12`) and the new form's is **3.7e-16**, i.e. under 2 ulp. BNG2 and
  NFsim both use the textbook form, so in that corner RM is now the most
  accurate of the three; wherever both forms are well-conditioned they agree
  to ~1e-15, and all 88 feature-coverage models are byte-identical.

  **`Km < 0` produced a NaN that the sampler drew events against.**
  `set_rule_propensity` clamped on `new_value < 0`, which is false for NaN,
  and `std::max` returned NaN unchanged — so `total_propensity` became NaN
  and the SSA kept going. Measured: an MM rule with `Km = -1` fired **80
  events** against a NaN propensity, exited 0, and wrote an
  ordinary-looking `.gdat`. Three changes: the MM branch reports `Km < 0` as
  a domain error rather than letting a half-defined expression through (with
  a negative `Km` the discriminant can go negative, and where it does not
  the rate is finite but meaningless); the clamp now tests `!(v >= 0)` so
  NaN takes it too, from **any** rate law, and names a negative MM `Km` when
  that is the cause; and a statically negative `Km` is refused at load as a
  Tier-0 Error, since `Km` resolves through the parameter cascade. A `Km`
  arriving later through `set_param` / `parameter_scan` cannot be caught at
  load and takes the clamp — which is the reachable path, since `Km` is a
  fitted parameter in RM's embedders.

  Blast radius: all 88 feature-coverage models byte-identical against the
  pre-fix binary at their own parameters, ctest 36/36, guard tier 29/29. The
  new `mm_km_domain_test` drives `ft_mm_ratelaw` entirely through
  `set_param` against closed-form oracles — with `Km = 0` and the enzyme in
  excess the law is `kcat*S`, so the substrate is an exact binomial death
  process `S(t) ~ Binomial(S0, exp(-kcat*t))` — and covers `Km = 0` on both
  sides of `S = E`, the cancellation band at `Km = 1e-14` and `1e-16`, and a
  negative `Km` arriving through an override. Against the pre-fix engine it
  fails 15 of its checks.

- **`MM(kcat,Km)` dropped BNG2's `symmetry_factor` — the last member of the
  set (issue #37).** `compute_propensity` applies the factor on every rate
  law it handles except the Michaelis-Menten branch, which returned
  `kcat*sFree*E/(Km+sFree)` computed off the raw substrate match count. A
  rule whose substrate pattern has a non-trivial reaction-center
  automorphism — an enzyme taking apart a symmetric dimer, where either
  molecule can seed the match — therefore ran fast: 2x deep in the linear
  regime, tapering to 1x deep in saturation. #36 fixed the same defect on
  the local-function and `FunctionProduct` paths and filed this one
  separately because, unlike those, it needed an oracle established rather
  than read off.

  The factor scales a reactant **count**, inside the law — `S = (match
  count) · symmetry_factor` — not the finished propensity. What it corrects
  is a match multiplicity: the pattern matches each complex twice, so the law
  was handed more of that reactant than exists. MM is not linear in either
  count, so scaling the propensity instead is exact only below saturation
  and dropping the factor is exact only inside it; scaling the count is
  exact everywhere and the three agree wherever the law is linear. That is
  also what BNG2 integrates: its network expansion builds one reaction per
  rule application and folds the factor into the reaction multiplicity (2
  maps x 0.5 = 1), leaving the rate law reading species counts. bngsim's
  patched NFsim (lanl/bngsim#195, fixed in #278) puts it in the same place.

  Which count: the one the rule **transforms**. BNG2 refuses `MM` on a rule
  whose two reactant patterns are isomorphic (*"Michaelis-Menton type
  ratelaw requires non-identical reactants"*), so no automorphism can carry
  one pattern onto the other and the reactant automorphism group is the
  direct product of the two patterns' own. A pure-context pattern
  contributes 1 and is already counted per complex (issue #33), so after the
  factor the law reads complex counts on both sides — exactly what BNG2's
  ODE integrates. For the canonical shape, where the enzyme is a catalyst
  and so context, that is the substrate: BNG2 emits `symmetry_factor="1"`
  when only the enzyme is symmetric and the rule leaves it alone. A rule
  that transforms its enzyme slot instead is off-spec for MM — the QSS
  derivation assumes a conserved enzyme — but BNG2 writes the XML for it and
  attaches the factor there, and RM now attributes the scalar rather than
  assuming the substrate; assuming it would have left that shape 2x fast in
  saturation. All three behaviours verified against BNG2 2.9.3 directly.

  Blast radius: none. Across all 208 model XMLs in the tree, `MM` appears in
  four models that predate this change — `ft_mm_ratelaw`, `CaMKII_holo` and
  `context_symmetry` (two rules) — and every one carries
  `symmetry_factor="1"`, so the new multiplier is exactly 1.0 and every
  existing trajectory is bit-identical. The full feature-coverage suite is
  unchanged.

  Nothing exercised it, which is why it was filed rather than fixed
  alongside #36: no model in any corpus paired a non-unit `symmetry_factor`
  with `MM`, so the fix had to bring its own coverage. Two models now do.
  `tests/cpp/mm_symmetry_model.bngl` drives the `mm_symmetry_test` ctest
  with four arms against a mean-field firing-count oracle — a symmetric
  dimer in saturation, a symmetric dimer in the linear regime, an asymmetric
  control with `symmetry_factor="1"`, and one whose symmetry sits in the
  transformed enzyme slot — because no single arm separates the candidate
  propensities: dropping the factor fails only the linear arm (16.6 firings
  against 9.1 expected), putting it on the propensity fails only the
  saturated one (5.2 against 9.9), and assuming the substrate fails only the
  enzyme-slot one (36.6 against 19.0, which is BNG2's own ODE for that arm).
  Feature-coverage model `ft_mm_ratelaw_sym` adds the ensemble
  comparison: an enzyme disassembling a recycled dimer pool, with a
  turnover counter so the discrepancy accumulates instead of being bounded
  by the pool. Verdicted against BNG2's ODE (turnovers at t=30: ODE 736,
  BNG2 SSA 731, RM 739, RM before the fix 988) and listed in
  `NFSIM_UNRELIABLE`, since the pinned NFsim release builds `MMRxnClass`
  with `baseRate=1` and reads 994.

- **A local rate reading a bare observable was rescanned every event even
  when that observable never moved (issue #40).** #38's fix gave such a
  rule a full O(N) rescan after **every** event. The rescan itself is
  necessary — a moving global changes every instance's rate at once with
  no molecule marked affected, so the affected-molecule delta path cannot
  see it — but the guard answered *"can this rule read a moving global?"*
  when the question that gates an O(N) rescan is *"did that global
  actually move?"*. A bare observable is typically a volume proxy, a total
  or a clock, so on most models it answers yes to the first and no to the
  second for the entire run, and every one of those rescans recomputes
  rates that cannot have changed. Correctness was never affected; this was
  cost only.

  RM now records **what** the local-function chain reads at global scope —
  the resolved observable slots, plus flags for `time` and for a
  dependency the loader cannot enumerate — and caches those values at each
  rescan. When they are all unchanged the rescan is skipped and the rule
  falls through to the ordinary delta path, which still owns whatever the
  event itself touched. The skip is exact, not an approximation: if
  nothing the rate reads globally has moved, no per-instance rate can have
  moved either. A rule reading `time` is unavoidably dirty every event and
  keeps the unconditional rescan.

  On bngsim's `context_symmetry` fixture — 15 rules, 4000 molecules per
  pool, two DOR pools whose local functions read a `Src()` count that
  nothing in the model touches — the full `t_end=2000` run goes from
  **35.7 s to 0.12 s**, which is exactly what the same model costs with
  that observable replaced by the literal `1`, and the trajectories are
  byte-identical across seeds. The whole difference was overhead.

  Blast radius: a byte-identity sweep of all 87 feature-coverage models
  against the pre-fix binary finds 86 identical and one changed —
  `ft_local_fcn_mixed_scope`, the only model whose bare observable moves.
  Its rule now takes the delta path on the events that do *not* move that
  observable, and incremental accumulation differs from a full re-sum in
  the last bits, so the SSA draws a different realization of the same
  process. Not a different distribution: over 80 seeds per build the two
  sit at Aoff 72.0 vs 72.7 and Boff 57.4 vs 56.6 — z of +0.6 and −0.7
  between them, both straddling the analytic 73.58 and 56.49, and the
  model's own `local_fcn_global_obs_test` oracle (4 SE over 40 reps) is
  unchanged. Every other model is bit-for-bit unchanged,
  as is the local-function hot path — isingspin_localfcn 0.07s → 0.07s, t3
  17.1 → 16.5, AN 6.74 → 6.73, min of 3.

  Nothing caught this because none of the 199 model XMLs across the three
  corpora references a bare observable inside a local function — the same
  coverage hole that hid #34, #36 and #38 — so no corpus model entered the
  rescan path at all and `perf_diff` saw nothing. That hole is now closed
  at a scale where the cost shows: feature-coverage model
  `ss_local_fcn_const_global` (22000 instances, ~12800 events, a bare
  observable that is constant for the whole run) and the
  `local_fcn_rescan_gate_test` ctest, which runs it against a twin with
  that observable folded into a parameter and requires the two to agree on
  the trajectory (byte-identical) **and** on the wall time (1.01x measured,
  213.8x before this fix, bound at 5x). A third arm moves the observable
  with `add_molecules` from outside the event loop — where the value cache
  is the only thing watching — and checks the doubled rate reaches every
  instance.

- **A bare (global) observable inside a local function was evaluated at
  local scope, so the rule silently never fired (issue #38).** Inside a
  local function, an observable applied to the tag — `Mod(x)` — is local,
  and one written bare — `Vol` — is the ordinary system-wide count. RM
  treated **every** observable a local function referenced as local and
  evaluated it at the tagged molecule. A bare observable is typically a
  system-wide quantity — a volume proxy, a total count, a clock — and so is
  essentially never present in the tagged molecule's own complex: it read
  0, the product went to 0, the propensity went to 0, and the rule never
  fired at all. No warning, mass conserved, trajectory flat — the same
  signature as #34, and not specific to it: a plain unimolecular rule
  `A(m~0)%x -> A(m~1)  flip(x)` with `flip(x) = kf*Obs_Src*A0(x)` held 500
  molecules at 500 where BNG2's ODE reaches 67.7.

  The XML cannot answer the question. BNG2 emits both kinds identically —
  `<Reference name="Cnt_Wz" type="Observable"/>` next to
  `<Reference name="Obs_Src" type="Observable"/>` — and the distinction
  survives only in the `<Expression>`: an observable is local iff it
  appears applied to one of that function's own `<Argument id=>` names.
  The loader now classifies from the expression, per function, so a chain
  of local functions is resolved against each callee's own parameter. BNG2
  states the intended scoping directly in the network it generates —
  `_R_local1() ((kc*Obs_Src)*1)`, the bare observable folded in as a global
  and the tagged one resolved per instance — and RM now matches it.

  Three things fall out of the corrected classification:

  - The **rate-dependent closure** stops excluding bare observables, so
    they are globally refreshed after each event instead of only at sample
    points. Both consumers of the local-observable set are fed from the one
    classification.
  - A local rate that reads a bare observable is now **rescanned after
    every event**. That observable moves for every instance at once with no
    molecule marked affected, so the affected-molecule delta path cannot
    see it; without the rescan every instance keeps the rate it had when
    the observable was last read. Marking is narrow — a local rate built
    from tagged observables and constants alone keeps the delta path, which
    is every model in all three corpora, so no existing model takes a
    per-event rescan.
  - RM holds one value slot per observable, so the union of the
    per-function local sets can only stand in for them while they agree.
    When one local function reads an observable that another localizes,
    each function is now given its own view of the observable slots
    immediately before it is evaluated. Models that do not mix the two
    scopings keep the single-override fast path unchanged.

  An observable written **both ways inside one function** (`O + O(x)`)
  wants two values from one slot and cannot be represented. RM resolves it
  at local scope — the reading that needs the machinery — and emits a
  load-time warning naming the function and observable, rather than
  mis-evaluating it in silence.

  Blast radius: a scan of all 199 model XMLs across the three corpora finds
  67 local functions and **zero** referencing a bare observable, `time`, or
  a global function — the same coverage hole that hid #34 and #36. So the
  change is inert for every existing model, and the new coverage is what
  exercises it: feature-coverage models `ft_local_fcn_global_obs`
  (unimolecular and bimolecular DOR1 arms, each with a subpopulation whose
  tagged observable is 0 so the two possible scope errors separate) and
  `ft_local_fcn_mixed_scope` (one observable at both scopes across two
  functions, plus a bare observable that moves mid-run), and the
  `local_fcn_global_obs_test` ctest against exact analytic oracles.

  The pinned NFsim 2.9.3 gets the classification right but never refreshes
  a bare observable: on `ft_local_fcn_mixed_scope` its initial slope is
  exactly `kb*Aoff(0)` and the whole run then proceeds at that t=0 value,
  giving 27.1 against the true 56.5. BNG2's ODE (56.49) and SSA (54.4)
  agree with each other and with RM, so that model joins the ODE-verdict
  set. This fix and #34 together unblock lanl/bngsim#301.

- **A bimolecular rule with a local-function rate fired once and then went
  inert (issue #34).** For a rule like
  `S(s~0) + E()%x -> S(s~1) + E()%x  lf(x)` — NFsim's DOR1, one tagged
  reactant carrying a per-instance factor — no branch of the propensity
  code applied. `recompute_rule_state` handled `FunctionProduct` (two
  tagged reactants) and `is_local` with molecularity ≤ 1, but a local
  `Function` on a two-reactant rule fell through to the mass-action path,
  which evaluated the local function against no molecule at all and left
  `rs.local_propensity_total` at zero while `has_local_rates` stayed true.
  The first `incremental_update` then read that never-populated accumulator
  as the rule's propensity. The rule fired once, off the load-time
  mass-action value, and never again.

  Nothing surfaced it: no clamp warning (the rate law is never negative —
  the documented clamp diagnostic was not involved), mass conserved, and
  the substrate observable simply flat. On the issue's reproducer RM
  converted 1 of 2000 substrate molecules over an interval where NFsim
  converts about 92%.

  The loader now normalizes such a rule to a `FunctionProduct` whose
  untagged factor is the constant 1, so the existing DOR2 propensity,
  incremental update and sampler cover it. Verified for all four shapes a
  single tag can take — on either reactant, in per-molecule or
  complex-wide observable scope, and on a single- or multi-molecule
  tagged pattern — against **BNG2's network → ODE**, with an exact
  binomial death-process oracle and NFsim both agreeing. The test model
  uses a heterodimer enzyme context on purpose, so the per-molecule vs
  per-complex context-multiplicity defect of lanl/bngsim#281 (RM issue
  #33) — where the pinned NFsim over-counts a symmetric *context* pattern
  N-fold, with or without any local function involved — cannot reach the
  result.

- **BNG2's `symmetry_factor` did not reach the local-function or
  `FunctionProduct` propensity paths (issue #36).** Those paths build the
  propensity directly from `Σ w·f(mol)` rather than through
  `compute_propensity`, which applies the factor for every other rate
  law — so a rule with a symmetric reaction center and a local-function
  rate ran at `1/symmetry_factor` times its intended rate, 2x for a
  homodimer. Found while cross-checking #34 against BNG2 rather than
  against NFsim alone.

  The pinned NFsim release has the identical defect for the identical
  reason (every rate law except the ones routing through `setBaseRate()`
  is constructed with `baseRate = 1` and never recovers the factor), so
  NFsim is not a valid oracle here; it is fixed upstream in bngsim's
  vendored NFsim, lanl/bngsim#195 → #278. BNG2's network expansion has
  always applied the factor. RM now matches BNG2's ODE exactly
  (`ODE_tz = 0.00`).

  No model in any of the three corpora pairs a non-unit `symmetry_factor`
  with a local function, so this is inert for every existing model — the
  new `ft_local_fcn_bimol_sym` is the only coverage of the combination,
  and it verdicts against BNG2's ODE. The untagged homodimer path is
  untouched: its `/2` in `compute_propensity` already is the 0.5, and
  `homodimer_rate_test` still pins it against the chemical master
  equation.

## [3.9.0] — 2026-08-07

### Added

- **Windows / MSVC is now a CI gate (issue #29).** The `build` job runs
  `ubuntu-latest`, `macos-latest` and `windows-latest`; the Windows leg
  configures, compiles with MSVC 14.51 and passes all 30 ctest cases in
  under two minutes. Before this, no part of RuleMonkey had ever been
  compiled with MSVC — the code's `_MSC_VER` guards were an assessment from
  reading the source, not a tested claim, and one of them was missing (see
  below).

  All three legs share the same Ninja `release` preset. A multi-config
  Visual Studio generator would leave `CMAKE_BUILD_TYPE` empty and silently
  change which self-check defines compile in, so the Windows leg instead
  puts the MSVC environment on `PATH` first — the hosted image does not
  expose `cl.exe` to a non-VS generator, and CMake would otherwise
  configure against whatever compiler it found (the Strawberry Perl gcc).

  Two MSVC-only compile flags come with it, applied directory-wide and
  gated on `MSVC` so the Clang/GNU `-Wall -Wextra -Werror` sets are
  untouched: `/bigobj`, without which the vendored exprtk translation unit
  stops at `fatal error C1128`, and `/utf-8`, which pins source decoding so
  RM's non-ASCII diagnostic strings do not depend on the build machine's
  active code page. The tree compiles clean at MSVC's default `/W3` — zero
  warnings.

  The `asan` job stays ubuntu/macos: `-fsanitize=address,undefined` is the
  GCC/Clang spelling, MSVC ships no UBSan, and `CMakeLists.txt` already
  refuses `RULEMONKEY_ENABLE_ASAN` outside Clang/GCC.

### Changed

- **CI workflow actions moved off Node 20 (issue #30).** `actions/checkout`,
  `actions/setup-python` and `actions/upload-artifact` are on v7. The two pins
  with no Node 24 version were retired rather than bumped:
  `seanmiddleditch/gha-setup-ninja` is gone entirely — it is archived upstream
  and its latest release is still Node 20, while all three runner images ship
  Ninja 1.13.2 on PATH — and the MSVC environment now comes from a `vswhere` +
  `vcvars64.bat` step rather than `ilammy/msvc-dev-cmd`, every available
  MSVC-environment action being Node 20 as well. A full run now carries no
  deprecation annotations on any job.

  Build pipeline only: no engine, API or build-system behaviour changes for a
  consumer of the library.

### Fixed

- **`__builtin_popcount` made the engine uncompilable with MSVC.**
  `build_nary_partitions`, which pre-expands the Möbius coefficients for a
  rule with three or more reactant patterns, reached for the GCC/Clang
  builtin with no `_MSC_VER` counterpart — unlike `highest_pow2` a few lines
  above it, which carries GCC/Clang, MSVC and generic branches — so `cl.exe`
  stopped at `engine.cpp` with `error C3861: '__builtin_popcount':
  identifier not found`. This was the only MSVC break in the tree.

  It is now a portable Kernighan loop rather than a third per-compiler
  branch: the code runs once per partition at rule setup over a mask of at
  most `kMaxNarySlots` bits, so there is nothing for an intrinsic to win,
  and MSVC's `__popcnt` would have carried its own caveats (x86-only, and it
  assumes a POPCNT-capable target).

## [3.8.1] — 2026-08-06

### Fixed

- **The ASan build left the vendored expression layer uninstrumented.**
  `RULEMONKEY_ENABLE_ASAN` applied `-fsanitize=address,undefined` to
  `rulemonkey`, `rm_driver` and `rm_scan`, but not to the `rm_bngsim_expr`
  object library, whose objects are folded into `librulemonkey.a` by
  `target_sources`. libc++ emits `std::vector` container annotations only
  from instrumented translation units, so `ExprTkEvaluator`'s
  `vector<string>` members — grown inside the uninstrumented
  `expression.cpp`, read from instrumented code — carried annotations the
  two sides disagreed about, and ASan aborted with `container-overflow` at
  `expression.cpp:817`. Every one of the 30 ctest binaries died there,
  inside `RuleMonkeySimulator`'s constructor, before reaching its first
  assertion.

  Whether it fires depends on how much of `<vector>` the toolchain's libc++
  annotates: the CI macOS runner never reported it, Xcode 26.5 does, and
  v3.7.0 reproduces it identically at the same line — so this is a latent
  hole in the sanitizer gate rather than a regression. The gate was passing
  by aborting before it could check anything. With the object library
  instrumented, all 30 tests pass clean under `-fsanitize=address,undefined`.

- **`benchmark_feature_coverage.py` aborted the run instead of skipping when
  `NFSIM_BIN` was unset.** Four models carry a BNG2 ODE reference and no
  NFsim one, because NFsim refuses them: `ft_tfun`, `ft_nested_functions`,
  `edg_time_dependent_rate`, `edg_deep_param_chain`. On a checkout without
  `NFSIM_BIN`, `generate_nfsim_reference` tried to regenerate the missing
  reference anyway, reaching `subprocess.run` with an empty `argv[0]` and
  taking the whole suite down with `PermissionError: [Errno 13] Permission
  denied: ''` — mid-model, after the ODE comparison for that model had
  already passed. The script's own docstring promises the missing-ref model
  "is skipped with a warning".

  It now returns early when no NFsim binary is configured, which is the
  skip path that was always intended. The full 82-model suite runs to
  completion on a bare checkout again (82 PASS), instead of stopping at the
  first ODE-only model. CI is unaffected: it sets `NFSIM_BIN` from the
  BioNetGen tarball, which is why the suite has never hit this there.

## [3.8.0] — 2026-08-06

### Added

- **Multi-molecule reactant patterns on n-ary rules (issue #26).** An n-ary
  rule now accepts the same reactant patterns a bimolecular rule does: a
  reactant may be a `.`-joined complex, not only a single molecule. The
  issue's reproducer — `A(s,d!1).D(d!1) + A(s) + A(s) -> P()` with
  `DeleteMolecules` — was refused at load before; it now simulates, and over
  30 seeds RM makes 109-122 P (mean 117.4) against NFsim's 110-124
  (mean 117.0) on the same XML.

  Each slot spans `[reactant_pattern_starts[i], starts[i+1])` and is counted
  by the same multi-molecule counter the bimolecular path uses, seeded on the
  pattern's first molecule; the sampler resolves the rest of the slot's
  molecules by walking its bonds, drawing among the seed embeddings that
  reach a whole match rather than among all of them — an `A(d,d)` bonded to a
  D and an E offers two embeddings of `A(d!1)` but only one extends to
  `A(d!1).D(d!1)`, and calling the other a null event would halve the rate.

  The propensity keeps the exact partition-lattice sum over *distinct seeds*,
  which for multi-molecule patterns is an upper bound on the true count: a
  molecule inside one slot's complex may be another slot's seed, or sit
  inside another slot's complex, and firing on such a draw would consume it
  twice. Those draws are rejected as null events, so the realized rate is the
  injective count `D_inj` exactly — `D · k · sf · (D_inj/D) = D_inj · k · sf`
  — the same inflated-propensity trick the bimolecular sampler already uses
  for same-molecule and same-complex draws. On the overlap fixture, dropping
  the rejection runs 25% hot (1124 events against the analytic 900).

  One n-ary shape stays refused, and the refusal is new: a `.`-joined
  reactant whose molecules are *not* bonded to each other (`A(s).D(d!+)`,
  meaning "both, anywhere in the same complex"). The sampler reaches a
  slot's non-seed molecules by following bonds, so it cannot place those —
  the count would be non-zero and every draw a null event, which is the
  silent inertness of #24 again. The engine gate and the load-time refusal
  are updated together, as before.

  Uni- and bimolecular trajectories are bit-identical, and so are n-ary
  rules of single-molecule patterns: the injectivity check is skipped
  outright for them and consumes no draws. All 180 corpus XMLs produce
  byte-identical trajectories against the previous build. A new
  feature-coverage model, `ft_nary_complex_reactant`, holds the NFsim parity
  in CI: the complex population turns over through a binding rule, so the
  per-slot counts are maintained incrementally rather than only at load.

- **N-ary reactant rules (issue #24).** A rule may now carry three or more
  `+`-separated reactants — `A + A + A -> P`, `A + B + C -> P` — when every
  reactant pattern is a single molecule and the rate law is elementary. Up
  to 6 patterns are supported. (The single-molecule restriction was lifted
  by #26, above.)

  These rules take a path of their own, separate from the two reactant
  slots that serve uni- and bimolecular rules. The propensity is mass
  action over tuples of *distinct* molecules, evaluated exactly by
  expanding the distinct-tuple sum over the partition lattice of the slot
  indices; reactants are drawn per slot by embedding count and retried
  until distinct, which reproduces precisely the distribution the
  propensity integrates, so no null events are spent on coincidences. BNG2
  emits `symmetry_factor = 1/n!` for `n` identical patterns, so the
  textbook forms fall out — `k·N(N−1)(N−2)/6` for `A + A + A`,
  `k·N_A·N_B·N_C` for `A + B + C`.

  Verified against the closed-form propensity on constant-population
  fixtures over 200 replicates: `A + A + A` (symmetry factor 1/6),
  `A + B + C` (1) and `A + A + B` (1/2) all land within 0.1% of exact. On
  the issue's reproducer RM now fires the trimolecular rule 127-130 times
  against NFsim's reported 126-129, and the four-body rule 98 against
  NFsim's 98-99. Uni- and bimolecular trajectories are bit-identical — the
  new path is inert below three patterns, and no model in the 185-XML
  corpus reaches it.

### Fixed

- **Bimolecular rules on a multi-molecule reactant fired slow.** The sampler
  draws the seed molecule weighted by `count_a`, which `count_multi_mol_fast`
  defines as the number of seed embeddings that reach a whole match (`c`). It
  then chose among *all* `S >= c` embeddings of the seed molecule alone and
  turned a choice that dead-ended into a null event, so the rule fired at
  `c/S` of its rate — silently, with mass conserved and nothing in the
  trajectory looking wrong.

  `S > c` whenever the seed molecule has more than one way to satisfy the
  pattern's own components but only some of them reach the rest of the
  pattern. On `A(d!1).D(d!1) + X()` with `A(d,d)` bonded to a D on one `d`
  and an E on the other, `A(d!1)` has two embeddings and one reaches the D:
  RM produced 51.7 P over 10 seeds where NFsim produced 98.8 and the closed
  form is 100. The sampler now filters to the embeddings that reach a whole
  match before drawing, which puts that model at 100.0. A pattern with a
  single seed embedding skips the filter — `c/S` is 0 or 1 there and no rate
  is lost — so the common one-bond multi-molecule rules pay nothing.

  The multi-molecule *unimolecular* selector always shuffled and took the
  first extending embedding, and the n-ary path added in #26 filters; this
  brings the bimolecular path in line with both.

- **A bimolecular rule could consume one molecule twice.** `mol_a != mol_b`
  separates the two *seeds* only. A molecule reached through one pattern's
  bonds can be the other pattern's match as well — `A(s,d!1).A(s,d!1) +
  A(s) -> P()` can draw the dimer's second A for the second slot — and
  firing on that consumed it twice; under `DeleteMolecules` it was deleted
  twice and mass balance broke. On a 5-dimer fixture, 33 of 60 seeds ended
  with more P than the A budget allows. The sampler now rejects a
  non-injective assignment as a null event, exactly as the n-ary path does,
  which is statistically exact because the propensity counts those draws.

  Reachable only with `-bscb` off: under the default the two seeds share a
  complex, so the same-complex check rejects the draw first. That is why no
  corpus model moved.

  Both fixes leave every trajectory in the 181-model corpus byte-identical —
  the filter is skipped where it cannot matter and neither check consumes a
  `uniform()` draw. New feature-coverage model `ft_multimol_deadend_seed`
  fails against its NFsim reference at tz=18.0 on the previous build and
  passes at tz=1.5 now; `multimol_rate_test` pins both in ctest.

- **Rules with three or more reactant patterns no longer fail silently
  (issue #24).** The engine carries exactly two reactant slots, and every
  consumer of `reactant_pattern_starts` treats slot B as the whole tail
  `[starts[1], molecules.size())`. A rule written `A + A + A -> P`
  therefore collapsed its second and third patterns into one bond-free
  slot-B pattern, which `count_multi_mol_fast_generic` can only satisfy
  from inside the seed molecule's own complex — so three free monomers
  scored zero embeddings, `b_total` stayed zero, and the rule's propensity
  was identically zero. The cutoff was exactly at three: two identical
  reactant patterns were, and remain, correct.

  Nothing reported the loss. The rule simply behaved as if it were absent
  while the rest of the model simulated normally, and because mass was
  still conserved the trajectory looked entirely valid — the reporter only
  found it by running the same XML through NFsim, which fires the rule
  ~126 times over the same horizon. In the 194-rule model where it
  surfaced, an unrelated first-order decay arm tracked its closed-form
  solution to within 1% while one molecule sat at its seeded value of 50 at
  every output time.

  The common shapes are now simulated (see Added above). The rest — a
  disconnected `.`-joined reactant pattern, a `Function` rate law, or more
  than 6 patterns — are refused at Tier 0, naming the rule and which of the
  three it hit, rather than going quietly inert. `--ignore-unsupported` runs
  them anyway, with the rule still inert.

  Because that refusal is decided from the raw XML while the engine decides
  from the parsed rule, the engine additionally emits its own `WARN` for any
  rule of 3+ reactant patterns it cannot represent. If the two checks ever
  disagree about a shape, the run stays loud instead of quietly producing a
  valid-looking trajectory — which was the entire failure mode here.

## [3.7.0] — 2026-07-29

### Fixed

- **`set_param` now reaches derived parameters, and through them the
  seed-species populations (issue #23).** BNG2's `writeXML` records every
  parameter twice — `<Parameter value="1806.6422">`, already collapsed to a
  number, and `<Parameter expr="((AT_nM*1e-9)*NA)*V_sim">`, the symbolic
  derivation. RuleMonkey read only `value`, deliberately, because that is
  what NFsim reads and BNG2 sometimes writes fewer digits there than `expr`
  carries. The trap was that the `set_param` override cascade *also*
  re-resolved `value`: a collapsed number re-resolves to itself, so an
  override on a base parameter could never move a parameter derived from
  it, and any `<Species concentration="LT">` seeded from that derivation
  stayed at its XML-time amount.

  The failure was silent and only visible against another engine. A
  dose-response scan written as one loaded model plus a `set_param` per
  point ran the XML's default dose at *every* point without error or
  warning — the reporter hit it through BNGsim's `RuleMonkeySession`, whose
  NFsim twin re-bakes the XML per point and therefore tracked the dose.

  The cascade now re-derives from `expr`, gated on the value actually
  moving under the active overrides: a parameter the override does not
  reach keeps its loaded `value` bit-for-bit. Without that gate, merely
  calling `set_param` on an unrelated parameter would re-round every
  derived quantity in the model to `expr` precision (`NA` alone differs in
  its 9th digit between the two attributes, and every bimolecular rate
  constant divides by it). Hand-authored XML with no `expr` attribute is
  unaffected. The no-override path is untouched, confirmed by the full
  three-corpus parity ladder.

- **Clearing an override now un-bakes the previous run.** `apply_overrides`
  mutates the parsed model in place, so once a run had baked overridden
  numbers into `RateLaw::rate_value` / `SpeciesInit::concentration`, a
  later `clear_param_overrides()` restored the parameter map but left those
  fields stale — `get_parameter()` reported the default while the engine
  kept simulating the cleared override. The existing regression test missed
  it because it never ran between the set and the clear. Seed amounts are
  now kept coherent with the parameter map between runs (the same contract
  `get_parameter()` already offered), and a latch drives one restoring pass
  over the baked rate constants when the last override is dropped.

### Added

- **Post-load control over seed-species amounts (issue #23).**
  `initial_species()` reports one row per `<Species>` — XML id, BNGL
  pattern, the `concentration` attribute verbatim, the amount the next run
  would seed under current overrides, and whether it is pinned.
  `get_initial_amount(key)` / `set_initial_amount(key, amount)` /
  `clear_initial_amount_overrides()` read and pin that amount, keyed by the
  BNGL pattern (`"L(r1,r2)"`) or the XML id (`"S1"`).

  `set_param` remains the right entry point when a parameter drives the
  amount — the derivation is the modeller's own. These cover what it
  cannot reach: a bare `<Species concentration="1000">` with no parameter
  behind it, and callers that would rather state the molecule count
  outright. A pin outranks the `concentration` expression (including a
  later `set_param`), a `Fixed="1"` species' clamp target follows it, and
  amounts truncate toward zero at instantiation like any seed amount.

  Together these close the upstream ask behind issue #23: a driver can
  refresh initial populations in place and walk a scan on one loaded
  model, instead of re-emitting and re-parsing an XML per point.

## [3.6.1] — 2026-07-10

### Changed

- **Re-pinned the vendored BNGsim expression layer to public `lanl/bngsim`
  (issue #11).** The ExprTk swap (issue #6) vendored four files under
  `third_party/` (`exprtk.hpp` + `bngsim_expr/{include,src}`), pinned by
  `third_party/bngsim_expr/VENDOR{,.json}`. That pin referenced a commit on
  the now-retired private `wshlavacek/PyBNF-Private` monorepo, where BNGsim
  lived under a `bngsim/` subdir. BNGsim is now published standalone at the
  public `lanl/bngsim`, whose history does not carry the old private commit,
  so the pin has been moved to `lanl/bngsim@5ce19a4`. Provenance in
  `VENDOR`/`VENDOR.json` now records the public remote and drops the stale
  `bngsim` subdir.

  Two of the four files advanced in the public tree and were refreshed to
  match the new pin: `expression.hpp`/`expression.cpp` gain a `tgamma`
  built-in (SBML factorial support) and a `referenced_variable_addresses`
  accessor (BNGsim forward-sensitivity work, GH #212). Neither is used by
  RuleMonkey, and the remainder is clang-format only — the change is a
  behavioral no-op for RM, confirmed by the full 80-model feature-coverage
  suite (80 PASS / 0 FAIL at `--reps 5`) and the 26-test unit suite.

### Added

- **CI drift-guard for the vendored BNGsim expression layer (issue #11).** A
  new `vendor_check` job in `.github/workflows/ci.yml` clones `lanl/bngsim`
  (public, no credential) with full history and runs
  `scripts/vendor_exprtk.py --check --bngsim-repo bngsim`, byte-comparing the
  vendored files against their pinned commit. Version skew between
  RuleMonkey's standalone copy and BNGsim's expression layer is an ODR
  violation, so drift now fails the build. This was blocked until BNGsim went
  public: the previous private source repo would have required a stored
  deploy key to clone.

## [3.6.0] — 2026-07-03

### Added

- **Energy-based BNGL (eBNGL) `Arrhenius` rate laws via load-time rule
  expansion (issue #20).** RM now natively runs 2-reactant binding energy
  rules — the `A(s1) + B(s2) <-> A(s1!1).B(s2!1)  Arrhenius(phi, Ea0)` form
  paired with a `begin energy patterns` block — matching NFsim's own eBNGL
  coverage (RuleWorld/nfsim commit `c4f1bb2`, Kutuva & Faeder). Previously
  any `RateLaw type="Arrhenius"` was a Tier-0 refusal.

  The implementation is a faithful port of NFsim's Sekar (2015, Ch. 3)
  expansion algorithm (`cpp/rulemonkey/energy_expand.{hpp,cpp}`): at model
  load each energy rule is expanded into a finite set of conventional rules
  with pre-computed rate constants, so the hot SSA loop is untouched. Only
  energy patterns overlapping the reaction-center bond contribute to ΔG
  (Sekar Corollary 3.3-43); patterns adding extra context gate `2^n` rule
  variants whose reactant templates carry the context as bound/free
  component constraints, with rates
  `k_fwd = exp(-(Ea0 + phi·ΔG)/RT)`, `k_rev = exp(-(Ea0 + (phi-1)·ΔG)/RT)`.
  Each direction is expanded independently from its own BNG2-emitted
  `ReactionRule`. Rates are stored symbolically, so `set_param` on an energy
  parameter (`Ea0`, `phi`, `RT`, or an energy-pattern `Gf`) re-resolves the
  baked rates on the next run — verified by `set_param_test`.

  Validated four ways: `energy_expand_test` pins the ported expansion to
  NFsim's own printed `k_fwd`/`k_rev` for the reference `v40` model and the
  cooperative scaffold; and two new feature-coverage models
  (`ft_energy_arrhenius`, `ft_energy_arrhenius_coop`) match BNG2's
  independent network expansion under both ODE (verdict) and SSA, and NFsim
  at steady state to within ~0.03 particles. BNG2, NFsim, and RM produce
  identical expanded rate constants.

  Scope (Phase 1, matching NFsim): 2-reactant heterodimer binding only.
  Shapes RM does not expand are refused as Tier-0 errors (recorded during
  expansion, where the fully-parsed rule is available) with a specific
  message: state-change energy rules, intramolecular ring-closure binding,
  >2-reactant rules, same-type homodimer binding, rules coupling binding to
  another operation, and rules carrying exclude/include constraints. Two
  NFsim-parity quirks (multi-context-bond OR-union, state-gated patterns) are
  reproduced faithfully and documented. See `docs/model_semantics.md` →
  "Energy-based BNGL (eBNGL)".

  The two eBNGL models are added to the harness `NFSIM_UNRELIABLE` set:
  NFsim's eBNGL path is seed-invariant (every `-seed` yields a byte-identical
  trajectory), so its multi-rep "ensemble" collapses to a single realization
  and is useless as a z-score reference; the verdict uses BNG2 ODE instead.

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

[Unreleased]: https://github.com/richardposner/RuleMonkey/compare/v3.10.0...HEAD
[3.10.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.10.0
[3.9.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.9.0
[3.8.1]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.8.1
[3.8.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.8.0
[3.7.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.7.0
[3.6.1]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.6.1
[3.6.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.6.0
[3.5.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.5.0
[3.4.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.4.0
[3.3.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.3.0
[3.2.1]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.2.1
[3.2.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.2.0
[3.1.2]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.1.2
[3.1.1]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.1.1
[3.1.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.1.0
[3.0.0]: https://github.com/richardposner/RuleMonkey/releases/tag/v3.0.0
