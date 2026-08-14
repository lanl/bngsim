# RuleMonkey BNGL Coverage

A reference for "will my BNGL model run on RuleMonkey?" RM consumes the
XML emitted by `BNG2.pl writeXML`; this document is keyed off that XML.

The full programmatic inventory lives in
`scan_unsupported()` in `cpp/rulemonkey/simulator.cpp` — every refusal and
warning below is emitted from that function and surfaced to embedders via
`RuleMonkeySimulator::unsupported_features()` (and to the CLI via
`rm_driver`'s startup banner).

The runtime severity model is two-level:

- **Error** — RM cannot honor BNGL semantics for the construct. The
  `rm_driver` CLI refuses to run such models with exit code 2 unless
  `--ignore-unsupported` is passed; embedders should inspect each
  feature's `Severity` themselves before deciding to call `run()`.
- **Warn** — best-effort run. The construct is parsed but partially
  honored (or ignored where ignoring is benign). Trajectories may differ
  from a fully-conformant simulator on edge cases.

## Supported

### Molecule and rule structure

- Multi-component molecule types with named components and enumerated
  per-component states (`A(s~U~P, b)`).
- Pattern matching with all four bond constraints: `Free` (`x`),
  `Bound` (`x!+` or `x!?`), `BoundTo` (`x!1` shared label),
  `Wildcard` (component omitted).
- Multi-molecule reactant patterns (`A(b!1).B(a!1)`), including
  multi-bond rings. The pattern is seeded on its first molecule and the
  rest are reached through its bonds; the sampler draws among the seed
  embeddings that reach a whole match, and rejects a draw in which two
  patterns (or two molecules of one pattern) land on the same molecule —
  see "Multi-molecule reactant patterns" below.
- Three or more reactant patterns (`A + A + A -> P`,
  `A(s,d!1).D(d!1) + A(s) + A(s) -> P`) under an elementary rate law —
  see "N-ary reactant rules" below.
- Symmetric components (`P(s,s)`) with the correct combinatorial
  weighting (the engine carries a "same components" flag and a
  symmetry factor).
- Reactant constraints: `ListOfExcludeReactants`, `ListOfIncludeReactants`,
  `ListOfExcludeProducts`, `ListOfIncludeProducts`. Multiple patterns
  inside a single list are OR'd; the engine evaluates all of them on each
  candidate match.

### Reaction operations

- `AddBond` (including ring-closing bonds and bonds to molecules added
  in the same rule).
- `DeleteBond`, with `ensureConnected="1"` honored as a
  product-molecularity check at fire time.
- `StateChange`, including state-increment / state-decrement (the BNGL
  `s~U~P~PP+` shorthand).
- `AddMolecule` (synthesis from nothing or from existing reactants).
- `DeleteMolecule`, both `DeleteMolecules="1"` (single-molecule) and
  the default whole-species delete.

### Rate laws

- **`Ele`** — elementary mass-action. Rate constants resolve from the
  parameter map at run time (`set_param` overrides take effect, including
  through derived-parameter chains — see "Parameter overrides" below).
- **`Function`** — arbitrary expression evaluated against a live
  variable map. Supports parameters, observables, the special variables
  `time` / `t`, and references to other functions. Local functions
  (per-molecule or per-pattern arguments) are supported in both
  per-molecule and complex-wide scopes, on unimolecular and bimolecular
  rules alike.
  - **One tagged reactant on a bimolecular rule (NFsim's DOR1).** A rule
    like `S(s~0) + E()%x -> S(s~1) + E()%x  lf(x)` applies the
    per-instance factor to the tagged reactant only; the untagged one
    contributes its plain embedding count. The propensity is therefore
    `(Σ_t w_t·f(t)) · (Σ_u w_u)` over the tagged and untagged matches —
    a `FunctionProduct` (below) whose untagged factor is the constant 1,
    which is how RM loads it. The tag may sit on either reactant and on
    a single- or multi-molecule pattern.
    When both reactant patterns are structurally identical
    (`A(b)%x + A(b) -> ...`), BNG2 emits `symmetry_factor="0.5"` and RM
    applies it, like it does on every other rate law. The propensity the
    local paths build from `Σ w·f(mol)` is the *ordered* distinct-pair
    sum, and the 0.5 converts it to unordered.
    Note that the **pinned NFsim release disagrees here, and is wrong**:
    it constructs every rate law except the `setBaseRate()` ones with
    `baseRate = 1` and never recovers the symmetry factor, so a symmetric
    rule runs 2x fast. That is fixed upstream in bngsim's vendored NFsim
    (lanl/bngsim#195, fixed in #278), whose patch names "global function,
    local function (DOR), function product, MM" as the affected set.
    `ft_local_fcn_bimol_sym` is verdicted against BNG2's ODE for that
    reason.
  - **A per-molecule tag on a reactant pattern the rule does not
    transform.** Such a pattern is *context*: every embedding of it yields
    the identical reaction, so BNG2's network expansion emits one instance
    per matching **complex** however many molecules inside that complex
    match, and RM counts it the same way. Where several molecules of one
    complex do match, the collapse then has to decide which of them the
    surviving instance is priced at — and with a per-molecule tag they
    price differently. RM
    prices at the **canonically-first** matching molecule of the complex,
    which is what BNG2's network expansion does: it evaluates the tagged
    observable once and emits the result as a constant, at the first
    matching molecule of the species' canonical form. The point of taking
    it from the canonical form rather than from, say, the oldest molecule
    is that every copy of a species then prices identically, however each
    copy was assembled. A **complex-wide** tag raises no such question:
    the observable is evaluated over the whole complex, so every molecule
    in it reads the same value. `context_dor_price` in the BNG2-oracle
    suite pins the three readings apart.
  - **Observable scope inside a local function.** An observable *applied
    to the function's own tag* — `Mod(x)` — is evaluated at the tagged
    molecule (per-molecule or complex-wide, per the tag). An observable
    written **bare** — `Vol` — is the ordinary system-wide count, exactly
    as it is anywhere else, and keeps its global value inside a local
    function. So `lf(x) = k*Vol*Mod(x)` scales a per-instance factor by a
    global one, which is how growth-dilution and Shea-Ackers style rate
    laws are written.
    BNG2's XML marks both kinds `<Reference type="Observable"/>` with
    nothing to distinguish them, so RM recovers the split by parsing the
    `<Expression>`: local iff applied to one of that function's own
    `<Argument id=>` names, resolved per function so a chain of local
    functions is classified against each callee's own parameter. BNG2's
    generated network states the same split — `_R_local1() ((k*Vol)*1)`,
    the bare observable folded in as a global and the tagged one resolved
    to its per-instance value.
    Two consequences worth naming. A rule whose local rate reads a bare
    observable is **rescanned whenever that observable moves**: it moves
    for every instance at once with no molecule marked affected, so the
    usual affected-molecule delta path cannot see the change. The rescan
    is gated on the value, not on the rule's shape — RM records which
    observables the local-function chain reads at global scope and skips
    the rescan on events where none of them changed, since a per-instance
    rate cannot move if nothing the rate reads moved. That matters because
    a bare observable is usually a volume proxy, a total or a clock and is
    constant for most or all of a run; a rule reading `time` has no value
    to compare and is rescanned unconditionally. And an
    observable used *both ways inside one function* (`O + O(x)`) cannot be
    represented — RM holds one value slot per observable — so RM resolves
    it at local scope and emits a load-time warning naming the function.
    The **pinned NFsim release gets the scope right but never refreshes**:
    its initial slope matches the true bare-observable value and then the
    whole run proceeds at that t=0 value. `ft_local_fcn_mixed_scope` is
    verdicted against BNG2's ODE for that reason.
  - **Global function of a local function is ill-defined.** A global
    (no-argument) function that references a *local* function has no
    molecule context at global scope, so its exposed value (`.gdat`
    column, `get_function_values()`) reflects an unspecified per-molecule
    slot — don't rely on it; NFsim treats the same construct as a model
    error.
  - **Negative-value clamp.** If the expression evaluates to a negative
    number at some point in the trajectory, RM clamps the resulting
    propensity to zero for that step — the reaction simply doesn't fire
    until the expression becomes non-negative again. The clamp lives
    in `set_rule_propensity` (`cpp/rulemonkey/engine.cpp`) and runs on
    every rate-update path (synthesis rules, normal updates, local-rate
    updates). On the first clamp per rule, RM emits one stderr line
    like `WARN: rule 'RR2' (_R2) propensity clamped to 0 — rate function
    'prod_rate' evaluated to -1 at t=0.851; further clamps on this rule
    are silent` so the authoring slip surfaces without spamming the log
    on oscillator-style trajectories.  NFsim's behavior on the same
    input is to refuse the model and exit with a "negative propensity"
    error. **Both behaviors are defensible workarounds for a model
    that's mis-authored for SSA semantics**: a single reaction whose
    rate flips sign with state is really two reactions (one for each
    direction), each guarded by `if(expr > 0, expr, 0)` (or the
    symmetric form for the other direction). If you're porting a
    continuous-ODE rate law verbatim into a BNGL rule, audit for this
    and split before either engine will give you physically meaningful
    trajectories.
- **`FunctionProduct`** — NFsim's DOR2: the rule rate is the product of
  two per-reactant local-function factors, each evaluated in the context
  of a different tagged reactant (`%x:A(..) + %y:B(..) -> ...
  FunctionProduct("f1(x)", "f2(y)")`, what BNG2 emits for an explicit
  product of two per-instance local functions). RM realizes the propensity
  as `S1·S2` where `S1 = Σ_a w_a·f1(a)` over reactant-A matches and
  `S2 = Σ_b w_b·f2(b)` over reactant-B matches — the local-rate analogue of
  an ordinary bimolecular rule's `a_eff·b_eff·k`. Each reactant draw is
  weighted by its own factor; same-molecule pairs are rejected by the
  sampler as null events (exact for distinct-type reactants, statistically
  exact otherwise), matching NFsim's `DOR2RxnClass`. Each factor honors
  per-molecule or complex-wide scope independently, like a single local
  `Function`.
- **`MM(kcat, Km)`** — Michaelis-Menten via NFsim's QSS formula:
  `sFree = 0.5·((S − Km − E) + √((S − Km − E)² + 4·Km·S))`,
  `a = kcat·sFree·E/(Km + sFree)`. Mirrors `MMRxnClass::update_a` in
  the NFsim source.
  BNG2's `symmetry_factor` scales one of the two counts **inside** the law
  — normally `S = (substrate match count) · symmetry_factor` — rather than
  the finished propensity. What it corrects is a match multiplicity: a
  pattern with a non-trivial reaction-center automorphism
  (`A(d!1).A(d!1)`) matches each complex more than once. MM is not linear
  in either count, so scaling the propensity instead is exact only below
  saturation; scaling the count is exact everywhere and reproduces BNG2's
  network expansion, which folds the factor into the reaction
  multiplicity and leaves the law reading species counts.
  The count it scales is the one the rule **transforms**. BNG2 refuses
  `MM` on a rule whose two reactant patterns are isomorphic, so the
  reactant automorphism group is the direct product of the two patterns'
  own, and a pure-context pattern contributes 1 (and is already counted
  once per matching complex rather than once per matching molecule). For
  the canonical shape, where the enzyme is a catalyst and so context, that
  is the substrate; a rule that transforms its enzyme slot instead is
  off-spec for MM but BNG2 writes it and attaches the factor there, and RM
  follows.
  If both patterns are transformed the scalar cannot be split and RM
  applies it to the substrate. The pinned NFsim release drops the factor
  here for the same reason it drops it on the local-function paths, so
  `ft_mm_ratelaw_sym` is verdicted against BNG2's ODE.
  **`Km` must be > 0.** `Km = 0` is a removable singularity, not a zero: as
  `Km → 0⁺` the law tends to `a = kcat·min(S, E)` (binding infinitely tight,
  so turnover is substrate-limited when the enzyme is in excess and
  enzyme-limited otherwise), and RM returns exactly that on the `Km = 0`
  line. `Km < 0` is refused at load, or clamped to zero propensity with a
  warning when it arrives through an override (issue #46). RM evaluates
  `sFree` in the cancellation-free form `2·Km·S/(q − diff)` when
  `diff = S − Km − E` is negative, which is where the textbook
  `0.5·(diff + q)` loses its significant digits — against 60-digit
  arithmetic the textbook form is off by up to 7.6e-2 relative (at
  `S=50, E=1000, Km=1e-12`) while RM's stays within 2 ulp. BNG2 and NFsim
  both use the textbook form, so RM is the more accurate of the three in
  that corner; everywhere both are well-conditioned they agree to ~1e-15.
  **When MM agrees with BNG2, and when to avoid it.** The law is defined
  over species-level pools, and BNG2's network expansion evaluates it once
  per matching *(substrate, enzyme) species pair*, on that pair's own counts.
  A network-free engine has no species pools and evaluates it once on the
  summed match counts. The two coincide exactly when **both reactant
  patterns pin exactly one species each** — every molecule specifying every
  component its type declares, each with a definite state and a definite
  bond status. Under that condition RM reproduces BNG2 across the whole
  parameter range (verified against BNG2 2.9.3 over twelve regimes spanning
  `E << S`, `E ~ S ~ Km`, `E >> S`, `Km → 0` and `Km → ∞`, agreeing to
  ≤5e-6). Outside it they diverge, by up to the number of matching substrate
  species (measured 2.00x for two species in saturation) and by up to 1.81x
  for a two-species enzyme with the enzyme in excess, since the law is
  nonlinear in both counts. Both divergences vanish where the law is linear.
  RM warns at load on either axis (issue #45). Warnings rather than
  refusals, because both constructs are idiomatic BNGL.

  Since MM exists to keep the *generated network* small and a network-free
  engine never generates one, MM buys nothing here. Writing the mechanism
  out — `S + E <-> SE` then `SE -> P + E` — costs one molecule type and two
  rules, is exact in both engines, needs no warning, and gets multi-substrate
  competition for a shared enzyme right, which neither MM reading does (BNG2's
  per-species expansion hands each substrate species its own full enzyme
  pool, so its total turnover can exceed `kcat·E_total`). Prefer the explicit
  mechanism unless you are reproducing an existing MM model.

  A **`symmetry_factor` on a rule that transforms both reactant patterns**
  cannot be attributed to a slot from the XML, the scalar being a product of
  both patterns' factors. RM applies it to the substrate, which reproduces
  BNG2 for the ordinary shape and anywhere the law is linear, and runs up to
  2x fast against BNG2 in saturation otherwise. Also warned at load.
- **`TFUN`** — table-function rate laws backed by external `.tfun`
  files. RM searches both the XML directory and one level up to handle
  both author-side and harness-side layouts. Counter sources may be
  `time`, a parameter, an observable, or another function.

### Observables

- `Molecules` (counts every matching molecule across all complexes).
- `Species` (counts complexes that match, treating each complex as one
  unit). Stoichiometric quantifiers `==N`, `>=N`, `<=N` on Species
  observables are supported (RM has a fast incremental path for these
  and substantially outperforms NFsim on count-relation Species
  observables — see `docs/timing_comparison.md`).
- Multi-pattern observables (BNGL `Molecules X A(),B()` with multiple
  patterns under one observable) — the engine sums embeddings across
  all listed patterns.

### Strict BNGL semantics

- `block_same_complex_binding` is **on by default** (matches NFsim's
  `-bscb` flag). Bimolecular rules only fire across distinct complexes;
  intramolecular ring-closure must be written as an explicit
  `A(x).B(y) -> A(x!1).B(y!1)` rule. Disable via
  `set_block_same_complex_binding(false)` or the CLI `-no-bscb` flag for
  parity with NFsim runs that omitted `-bscb`.

### Parameter overrides

- `set_param(name, value)` rejects names not declared in the loaded XML
  (typos throw rather than silently no-op).
- Overrides cascade through derived parameter expressions. If the BNGL
  declares `B = 2*A`, `set_param("A", x)` recomputes `B` to `2*x` for
  the next run AND for `get_parameter("B")` queries in between runs.
  Overriding `B` directly wins over the cascade — the expression for
  `B` is skipped, and parameters that derive from `B` see the override.
- **Which XML attribute the cascade reads.** BNG2's `writeXML` records
  every parameter twice: `value=` is the derivation already collapsed to
  a number, `expr=` is the derivation itself.

  ```xml
  <Parameter id="LT" type="Constant" value="1806.6422" expr="((AT_nM*1e-9)*NA)*V_sim"/>
  ```

  Load-time resolution reads `value`, because that is what NFsim reads
  and BNG2 does not always write the two to the same precision (`NA`
  comes out as `value="6.0221408e+23"` against `expr="6.02214076e23"`).
  The override cascade reads `expr`, because a collapsed number
  re-resolves to itself and so could never propagate an override.

  To keep those two sources from fighting, the cascade takes an
  expr-derived value only where it actually *moves* the parameter off
  its no-override baseline. A parameter the override does not reach
  keeps its loaded `value` bit-for-bit, so a run with an override on
  `X` is numerically identical to the un-overridden run everywhere `X`
  does not reach. Hand-authored XML that puts the expression directly
  in `value=` and omits `expr=` cascades off `value` as before.

  Consequence worth stating plainly: a derived parameter *is*
  overridable even though BNG2 emitted its `value` as a literal. That
  was not true before 3.7.0 — see the issue #23 entry in the changelog.
- Cascade order is the parameter declaration order in the XML, iterated
  to fixed point (capped at `parameter_count + 4` passes) both at load
  time and at `set_param` time.  An arbitrarily-deep chain of forward
  references resolves correctly even if the emitter does not deliver
  them in dependency order — `C = 2*B; B = 2*A; A = 1` declared in that
  order settles after a `set_param("A", x)`. Cycles are the only thing
  that won't resolve; they warn on stderr and leave stale values.
- `get_parameter(name)` reflects the current overrides + cascade
  immediately, without requiring a `run()` or `initialize()` call.
- `clear_param_overrides()` restores the parsed values, including in the
  rate constants and seed concentrations a prior `run()` baked into the
  parsed model — so it un-does an override that has already been
  simulated, not just the override map. (Before 3.7.0 the baked fields
  were left stale, so `get_parameter()` and the engine disagreed.)

### Initial state and live mutation

- Seed species declared via `ListOfSpecies` with `concentration="N"` or
  `concentration="<expression>"` — expression-backed concentrations
  re-resolve through `set_param` overrides, including through a chain of
  derived parameters (`concentration="LT"`, `LT = f(AT_nM, …)`).
- Amounts are real-valued in the XML and **truncated toward zero** when
  the engine instantiates molecules — NFsim parity, and several corpus
  models depend on it (`NL = 421.5498` seeds 421, not 422).
- `set_initial_amount(key, amount)` pins a seed amount directly, for the
  case no parameter drives it (a bare `concentration="1000"`) or where
  the caller would rather state the molecule count outright. Keyed by
  the BNGL `<Species name=>` pattern or the XML `<Species id=>`. A pin
  outranks the `concentration` expression — including a `set_param`
  applied afterwards — until `clear_initial_amount_overrides()`.
  `initial_species()` reports every seed species with the amount the
  next run would use under the current overrides.
- Single-molecule `Fixed` species (BNGL `$` prefix) — the engine
  replenishes them after each event so their count is held at the
  initial value (matching BNG2's ODE semantics). The clamp target
  follows parameter overrides and direct pins.
- `add_molecules(type_name, count)` for live perturbation between
  segments of a stateful session.

### Functional surface

- `set_param`, `clear_param_overrides`, `set_initial_amount`,
  `clear_initial_amount_overrides`, `set_molecule_limit`,
  `set_block_same_complex_binding` — applied at the next `run()` /
  `initialize()` (throw if a session is currently active).
- `get_parameter`, `get_initial_amount`, `initial_species` — read the
  configuration the next run would use; callable with or without a live
  session, since they describe the initial state rather than the pool.
- `save_state(path)` / `load_state(path)` — full pool, RNG, and
  bookkeeping snapshot. The XML used at `load_state` time must match
  the one used at `save_state` time.

## Tier-0 refusals (Error severity)

These are emitted as `Severity::Error` entries on
`unsupported_features()`. The CLI exits 2 by default; embedders should
treat at least one of these as a signal to refuse the model.

| Trigger | Why refused |
|---|---|
| `<ListOfCompartments>` non-empty | RM does not implement compartment volume scaling — bimolecular rate constants would be silently incorrect. |
| An `RateLaw type="Arrhenius"` rule that is **not** a 2-reactant binding rule | eBNGL energy rules are expanded at load time (see "Energy-based BNGL" below), but only for the 2-reactant binding case — matching NFsim's own coverage. State-change energy rules, intramolecular ring-closure binding, and >2-reactant rules would be silently dropped, so they are refused. (2-reactant binding Arrhenius rules, and a bare `<ListOfEnergyPatterns>` with `Function`-type rate laws, are both fully supported and are **not** triggers.) |
| A rule with three or more `<ReactantPattern>` children, where a pattern's `.`-joined molecules are not bonded to each other, the rate law is not `Ele`, or there are more than 6 patterns | Rules of 3-6 reactant patterns under an elementary rate law are simulated, each pattern a single molecule or a bonded complex (see "N-ary reactant rules" below). The shapes listed here fall outside that path and would drop onto the two-slot machinery, whose slot B merges patterns 2..n into one bond-free pattern that scores zero embeddings for free reactants — the rule would silently hold zero propensity and never fire, with mass still conserved so the trajectory looks valid (issues #24, #26). Rewrite as a sequence of at most bimolecular steps. |
| Any rule with `RateLaw type="Sat"` | Deprecated; rewrite as `MM(kcat, Km)`. |
| An `MM(kcat,Km)` rule whose `Km` resolves to a negative value | Outside the rate law's domain: the discriminant `(S−Km−E)² + 4·Km·S` can go negative, and where it does not the expression still yields a finite but meaningless rate (issue #46). A `Km` that arrives *after* load through `set_param` / `parameter_scan` cannot be caught here and is clamped to zero propensity with one warning per rule instead. |
| Any rule with `RateLaw type="Hill"` | Network-only; use `generate_network()` + ODE/SSA instead of network-free. |
| Any `<MoleculeType population="1">` | Hybrid particle-population SSA not implemented; would be silently treated as ordinary particles with diverging trajectories. |
| Multi-molecule or bonded `<Species Fixed="1">` | RM currently supports only single-molecule, unbonded Fixed species. |
| Two or more `<Species Fixed="1">` of the same `MoleculeType` | RM currently allows at most one Fixed species per molecule type to avoid matching-overlap ambiguity. |

The CLI's `--ignore-unsupported` flag downgrades these to runs-anyway
mode. Each error message includes the specific behavior change that
results — embedders inspecting the feature list can replicate that
decision per-feature.

## Best-effort warnings (Warn severity)

These are emitted as `Severity::Warn` and the run proceeds.

| Trigger | Effect |
|---|---|
| Any rule with `MoveConnected` operation | Requires compartments; emitted as a warning because RM ignores the operation entirely. |
| Any rule with a `priority` attribute | Honored as ordinary rule firing; the priority modifier is ignored. |

## Multi-molecule reactant patterns

A reactant pattern may be a `.`-joined complex — `A(b!1).B(a!1)` — on a
unimolecular, bimolecular, or n-ary rule. The pattern is *seeded* on its
first molecule, the rest are reached by walking its bonds out of that seed,
and the per-molecule count `c(m)` that weights the draw is the number of
embeddings whose seed lands on `m` and which reach a whole match.

Two things follow, and both are enforced in the sampler:

- **The embedding is drawn among the ones that reach a whole match**, not
  among all embeddings of the seed molecule alone. A seed can offer an
  embedding that goes nowhere: an `A(d,d)` bonded to a D on one `d` and an E
  on the other has two embeddings of `A(d!1)`, and only the one that lands
  on the D extends to `A(d!1).D(d!1)`. Treating the dead end as a null event
  instead would fire the rule slow by exactly the fraction that dead-ends —
  a factor of 2 in that example.
- **The whole assignment must be injective.** Rejecting same-molecule
  *seeds* is not enough: a molecule pulled in through one pattern's bonds
  can be another pattern's match as well (`A(s,d!1).A(s,d!1) + A(s)` can
  draw the dimer's second A for the second slot), and firing on that
  consumes one molecule twice — under `DeleteMolecules`, deleting it twice.
  Such a draw is rejected as a null event. The propensity counts it, so the
  rejection makes the realized rate the injective count exactly rather than
  approximately. Under `-bscb` these draws are already gone: the two seeds
  share a complex, so the same-complex check rejects them first.

## N-ary reactant rules

A rule may carry three or more `+`-separated reactants — `A + A + A -> P`,
`A + B + C -> P`, `A(s,d!1).D(d!1) + A(s) + A(s) -> P` — under an elementary
rate law. Each reactant pattern may be a single molecule or a `.`-joined
complex, as on a bimolecular rule, so long as its molecules are bonded to
each other. Up to 6 patterns are supported; shapes outside that are refused
at Tier 0 rather than run (issues #24, #26).

Such a rule takes a dedicated path in the engine, separate from the two-slot
machinery that serves uni- and bimolecular rules. Reactant pattern `i` is
*seeded* on its first molecule, and `c_i(m)` counts the embeddings of the
whole pattern that place that seed on molecule `m` — for a single-molecule
pattern, just its embeddings on `m`. The propensity is mass action over
tuples of *distinct* seed molecules,

```
a = k · symmetry_factor · Σ_{(m_0..m_{n-1}) all distinct} Π_i c_i(m_i)
```

evaluated exactly by expanding the distinct-tuple sum over the partition
lattice of the slot indices (see the `NaryState` comment in
`cpp/rulemonkey/engine.cpp`). BNG2 emits `symmetry_factor = 1/n!` for `n`
identical patterns, so the familiar closed forms fall out: with one embedding
per molecule, `A + A + A` gives `k·N(N−1)(N−2)/6` and `A + B + C` gives
`k·N_A·N_B·N_C`.

Reactants are drawn one slot at a time, weighted by embedding count, and
retried until the `n` seeds are distinct — which reproduces exactly the
distribution the propensity integrates, so for a rule of single-molecule
patterns no null events are spent on coincidences.

Within a slot, the embedding is drawn from the seed embeddings that actually
extend to a whole match — the ones `c_i(m)` counted, as described under
"Multi-molecule reactant patterns" above.

A multi-molecule pattern needs one more step. Seed distinctness does not
prevent a molecule pulled into one slot's complex from also being another
slot's seed, or from sitting inside another slot's complex — firing on such
a draw would consume one molecule twice. Those draws are counted by the
propensity above and rejected by the sampler as null events, so the realized
rate is the injective count `D_inj` exactly:

```
D · k · sf · (D_inj / D)  =  D_inj · k · sf
```

This is the same inflated-propensity/null-event trick the bimolecular
sampler uses for same-molecule and same-complex draws; the cost is wasted
SSA cycles in proportion to how often the patterns overlap, which is rare —
and rarer still under `-bscb`, which rejects same-complex seed pairs first.

### What is and is not covered

| Left-hand side | Handled? |
|---|---|
| `A + A + A`, `A + B + C`, `A + A + B` — 3 to 6 single-molecule patterns, elementary rate | **Yes**, simulated |
| A pattern that is a bonded complex, e.g. `A(s,d!1).D(d!1) + A(s) + A(s)` | **Yes**, simulated |
| 7 or more reactant patterns | No — Tier-0 refusal |
| A `.`-joined pattern whose molecules are not bonded to each other, e.g. `A(s).D(d!+) + A(s) + A(s)` | No — Tier-0 refusal |
| A `Function` (global or local) rate law on 3+ reactants | No — Tier-0 refusal |
| `MM(kcat,Km)` on 3+ reactants | Cannot arise — BNG2 refuses to write the XML |

A left-hand side in the "No" rows is refused at load: `rm_driver` exits 2
and names the rule and which of the three it hit.  `--ignore-unsupported`
runs the model anyway with that rule inert, and the engine then emits its
own `WARN` line for the same rule, so the inertness is stated on both
paths rather than inferred.

That second, engine-side warning is deliberate belt-and-braces.  The
load-time refusal decides from the raw XML, while the engine decides from
the parsed rule; if those two ever disagree about a shape, the run stays
loud instead of quietly producing a valid-looking trajectory.

Before this was implemented such a rule was *silently inert*: it scored zero
embeddings, held zero propensity, and never fired, while the rest of the
model simulated normally and mass stayed conserved.

## Energy-based BNGL (eBNGL)

RM supports energy rules written with the `Arrhenius(phi, Ea0)` rate law
for the **2-reactant binding** case, matching NFsim's own coverage
(RuleWorld/nfsim commit `c4f1bb2`). No runtime free-energy computation
happens; instead the model loader expands each energy rule at load time
into a finite set of conventional rules with pre-computed rate constants
(Sekar 2015, *Rule-based Modeling of Cell Signaling*, Ch. 3), which then
run through the ordinary SSA loop unchanged. The port lives in
`cpp/rulemonkey/energy_expand.{hpp,cpp}`.

For a binding rule `mt1(s1) + mt2(s2) <-> mt1(s1!1).mt2(s2!1)`:

- The energy patterns that overlap the reaction-center bond are the only
  ones whose match count changes when the bond forms, so only they
  contribute to ΔG (Sekar Corollary 3.3-43).
- Patterns that pin only the reaction center are "always" contributors;
  patterns that additionally require context (another bound site, or a
  third molecule) are "conditional" and gate on that context.
- The rule expands into `2^n` context variants (n = number of distinct
  context conditions), each a conventional binding or unbinding rule whose
  reactant templates carry the context as extra bound/free component
  constraints, and whose rate is
  `k_fwd = exp(-(Ea0 + phi·ΔG)/RT)` (binding) or
  `k_rev = exp(-(Ea0 + (phi-1)·ΔG)/RT)` (unbinding). Each direction is
  taken from its own BNG2-emitted `ReactionRule` (forward = AddBond,
  reverse = DeleteBond), so the forward and reverse halves are expanded
  independently.
- `RT` is read from the `RT` parameter (default 2.478); `phi`/`Ea0` are
  the two `Arrhenius` rate constants. Energy-pattern values come from
  the `<EnergyPattern expression="...">` attribute.

Each expanded rate is stored as a symbolic expression of the energy
parameters, so `set_param` on `Ea0`, `phi`, `RT`, or an energy-pattern
`Gf` **does** re-resolve the baked rates on the next run (same path as
ordinary `Ele` rate constants).

**Deferred** (refused at Tier 0, matching NFsim's binding-only coverage):
state-change energy rules, intramolecular ring-closure binding, rules with
more than two reactants, **same-type homodimer binding** (`A + A <-> A.A`;
the context reactant attribution and the molecularity-1 symmetry factor are
not handled for automorphic reactants), rules that **couple binding to
another operation** (state change, molecule add/delete), and rules carrying
**exclude/include constraints**. Energy patterns paired with `Function`-type
rate laws that inline the Boltzmann factors by hand (e.g.
`isingspin_localfcn`, `ft_energy_patterns`) are a separate, long-supported
path and need no expansion.

**NFsim-parity quirks** (faithfully reproduced, so RM matches NFsim even
where NFsim's expansion is incomplete relative to BNG2's network
generation): an energy pattern that requires *two or more* context bonds is
included in a variant's ΔG when *any* one of those bonds is present
(OR-union, not AND); and an energy pattern gated purely by a non-center
*internal-state* constraint (rather than a bond) contributes no ΔG term.
Both mirror NFsim exactly. RM's supported test models avoid these shapes;
if you need BNG2-network-exact behavior on such patterns, use
`generate_network()` + ODE/SSA.

## Embedder integration pattern

```cpp
rulemonkey::RuleMonkeySimulator sim("model.xml");

bool any_errors = false;
for (const auto& f : sim.unsupported_features()) {
  if (f.severity == rulemonkey::Severity::Error) {
    log_error("RM cannot honor BNGL feature {} ({}): {}",
              f.element, f.feature, /* your model id */);
    any_errors = true;
  } else {
    log_warn("RM warning: {} — {}", f.element, f.feature);
  }
}
if (any_errors) {
  // Decide per-feature whether to refuse or fall back to a
  // different backend; do NOT call sim.run() if you'd misinterpret
  // an Error-level construct.
  return;
}

auto result = sim.run({0.0, t_end, n_points}, seed);
```

### Recording at explicit output times

`TimeSpec` samples a uniform `t_start..t_end / n_points` grid by default.
To record at an explicit, possibly non-uniform list of instants instead
— typically an experimental dataset's time points, so an embedder can fit
against them directly — set `TimeSpec::sample_times` to a sorted-ascending
vector; it overrides the `n_points` grid:

```cpp
rulemonkey::TimeSpec ts;
ts.t_start = 0.0;
ts.t_end   = 120.0;            // still bounds the run
ts.sample_times = {0.0, 5.0, 15.0, 30.0, 60.0, 120.0};  // ascending
auto result = sim.run(ts, seed);   // result.time == sample_times
```

Sampling is non-invasive: output times never draw from the RNG or perturb
reaction selection, so the realization (and `result.event_count`) is
bit-identical to the uniform-grid run at any instants the two schedules
share. `t_end` still bounds the SSA loop, so set it to at least the largest
sample time; times at/after `t_end` (and any below `t_start`) are still
emitted, recorded at the final/initial state. An out-of-order list throws
`std::runtime_error`. This mirrors BNG2.pl's `simulate_nf` `sample_times`.

## Open work

Neither of the two issues this section used to defer to is open any more,
so each entry states its own tracking status rather than implying a live
ticket.

- Compartment volume scaling (cBNGL) — **untracked**.
  [#21](https://github.com/richardposner/RuleMonkey/issues/21) was closed
  as *not planned* (2026-07-13), so nothing schedules this; the gap itself
  is unchanged. Compartments stay a Tier-0 refusal, because a well-mixed
  reading of a compartmental model gets bimolecular rates silently wrong;
  `--ignore-unsupported` runs one anyway at volume = 1.
- Energy-based BNGL (eBNGL) beyond 2-reactant heterodimer binding —
  **untracked**. [#20](https://github.com/richardposner/RuleMonkey/issues/20)
  is closed as *completed*: it delivered the binding case in 3.6.0, and
  does not cover the rest. Still refused at Tier 0 are state-change energy
  rules, intramolecular ring-closure binding, >2-reactant energy rules,
  same-type homodimer binding, binding coupled to another operation, and
  rules carrying exclude/include constraints — the same scope NFsim's own
  eBNGL implements. See "Energy-based BNGL (eBNGL)" above.
- Hybrid particle-population SSA — open work, no scheduled implementation.
- Multi-molecule Fixed species — would require pattern-based
  re-instantiation; not currently implemented (refused at Tier 0).
- Pattern canonical labeling (nauty integration) — flagged in
  `CHANGELOG.md` as the leading candidate for the next major
  performance work; no impact on coverage, only on speed for models
  with many isomorphic complexes.
