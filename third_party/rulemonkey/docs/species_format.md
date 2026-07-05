# `.species` output format and species enumeration

RuleMonkey can enumerate the **live species** of a paused simulation —
group the molecules in the agent pool into connected complexes, collapse
graph-isomorphic complexes into one species, and report each as a
BNG-format pattern line with its instance count. The result is written
to a `.species` file readable by BioNetGen's `BNG2.pl` (its
`readNFspecies` reader). This document is the authoritative reference
for the format and the enumeration semantics.

This is RuleMonkey's answer to [issue #9 §2][i9]. It is the analogue of
NFsim's `-ss` species dump, but RuleMonkey deduplicates by *true* graph
isomorphism (see [Canonical species identity](#canonical-species-identity)).

## File shape

One or more `#` comment lines, then one data line per species:

```
# RuleMonkey generated species list
# <N> species, <M> complexes
<canonical pattern><SPACE><SPACE><count>
<canonical pattern><SPACE><SPACE><count>
...
```

- Every line beginning with `#` is a comment. `BNG2.pl readNFspecies`
  strips `#` to end-of-line, so the header is informational only.
- Each data line is a **canonical BNGL species pattern** followed by an
  **integer count** — the number of complex instances of that species
  currently in the pool. The separator written is two spaces; any run of
  whitespace is accepted on read.
- Data lines are sorted lexicographically by the species pattern, so the
  file is deterministic for a given pool state.
- The count is always a non-negative integer (network-free particles
  have unit population). Only **live** species appear — there are no
  zero-count rows for extinct seed species, matching NFsim `-ss`.

Example (`A(a) + A(a) <-> A(a!1).A(a!1)`, mid-run):

```
# RuleMonkey generated species list
# 2 species, 754 complexes
A(a!1).A(a!1)  246
A(a)  508
```

Molecules are conserved: `508·1 + 246·2 = 1000`, the seed `A_tot`.

## Canonical species identity

Two complexes are the **same species** iff their molecule graphs are
isomorphic (same molecule types, component names and states, and bond
topology — independent of the order molecules and components happen to
occupy in the pool). RuleMonkey assigns each complex a *canonical label*
— a normalized BNGL string that is byte-identical for all isomorphic
complexes and distinct for non-isomorphic ones — and uses it as both the
deduplication key and the emitted pattern text. Symmetric complexes
(rings, homo-oligomers) are resolved by individualization-refinement, so
the count for, say, a symmetric homodimer is exact rather than inflated
by molecule-ordering accidents.

The canonical string is a valid BNG-format pattern. RuleMonkey's
canonical format need not byte-match NFsim's internal label: `BNG2.pl`
re-canonicalizes every species it reads, and `readNFspecies` sums the
counts of any lines that turn out to be isomorphic. A parity check
against NFsim `-ss` output must therefore compare the two files as a
*multiset of canonicalized species*, not as text.

## How to produce a `.species` file

### `rm_driver` CLI

```
rm_driver <model.xml> <t_end> <n_steps> [seed] --species <path>
```

`--species` writes the final-state census to `<path>` after the run.
Passing it forces session mode (the pool must outlive the run), so it
composes with `--save-state` / `--load-state` / `--t-start`.

### Embedding API

`RuleMonkeySimulator` exposes the census to in-process callers on an
active session:

- `std::vector<SpeciesRow> enumerate_species() const` — the sorted
  `(species, count)` rows. Each `SpeciesRow` (see `types.hpp`) is a
  canonical pattern string and an instance count.
- `void write_species_file(const std::string& path) const` — writes the
  rows to `path` in the format above.
- `long species_count(const std::string& canonical_species) const` —
  the count for a single species, looked up by its canonical string.
- `long total_complex_count() const` — the total number of live
  complexes in the pool (the sum of every row's count).

All four require a live session (`initialize()` / `simulate()` first)
and throw `std::runtime_error` otherwise. Enumeration is a one-shot pool
walk, intended to be called while the simulation is paused (between
`simulate()` segments or after a run) — not per SSA event.

### Looking up a single species — `species_count`

`species_count` answers "how many of *this* species are live" without
the caller scanning the whole `enumerate_species()` vector. Its key is a
**canonical string RuleMonkey itself emitted** — a `SpeciesRow::species`
value or a data-line pattern from a written `.species` file:

```cpp
for (const auto& row : sim.enumerate_species())
  assert(sim.species_count(row.species) == row.count);
```

`species_count` does *not* parse or canonicalize its argument: a
hand-written pattern that is not byte-identical to a label RuleMonkey
would emit returns `0`, even when it denotes the same species. For
pattern-keyed lookup from an arbitrary hand-written BNGL species string,
use `get_species_count` — see [Pattern-keyed species API](#pattern-keyed-species-api-issue-9-1)
below — which parses and canonicalizes the pattern internally.

`species_count` is a *batch query*: each call is internally a full pool
walk, the same cost as `enumerate_species()`. To read many species,
call `enumerate_species()` once and index its rows rather than calling
`species_count` per species. `total_complex_count()` is cheaper than
either — it counts live complexes without canonicalizing them.

## Pattern-keyed species API (issue #9 §1)

The methods above key on a *canonical string RuleMonkey emitted*. For
embedders that need NFsim-style session control keyed on a **runtime
BNGL species-pattern string** — a string the caller writes by hand,
e.g. `"A(b!1).B(a!1)"` — `RuleMonkeySimulator` exposes four further
methods on an active session:

- `int  get_species_count(const std::string& pattern) const` — live
  count of the species denoted by `pattern`.
- `void add_species(const std::string& pattern, int count)` —
  instantiate `count` fresh copies of the species into the pool.
- `void remove_species(const std::string& pattern, int count)` —
  remove `count` live copies (throws if fewer exist).
- `void set_species_count(const std::string& pattern, int count)` —
  add or remove the difference so the live count becomes exactly
  `count`.

Each parses `pattern` against the loaded molecule types, so — unlike
`species_count` — the string need *not* be a canonical label RuleMonkey
emitted. All four require a live session and are paused-session calls:
none advances logical time or touches the SSA event loop.

### Accepted pattern grammar

The runtime parser accepts an **exact, fully-specified, connected
species** — and only that (issue #9 §1 design decision A). A pattern is
a `.`-separated list of molecules, each `Name(comp,comp,...)`, each
component `name[~state][!bond]`:

- every molecule name must resolve to a declared `MoleculeType`;
- every component of each molecule type must be listed exactly once —
  same-named ("symmetric") components are matched positionally by
  occurrence;
- every component with a state set must carry a concrete `~state` drawn
  from that set; stateless components must carry none;
- bonds are numeric labels `!N`; each `N` must appear on exactly two
  distinct components. The bond wildcards `!+` and `!?`, don't-care
  states, and omitted components are **rejected** — they describe a
  pattern *class*, not one instantiable species;
- the molecules must form a single connected complex.

A pattern that violates any of these throws `std::runtime_error` naming
the offending token. A canonical species string from `enumerate_species()`
or a `.species` file is always a valid input — so
`get_species_count(row.species)` agrees with `species_count(row.species)`
and with `row.count`.

### Evaluating expressions — `evaluate_expression`

Also part of issue #9 §1, `double evaluate_expression(const std::string&
expr, const std::unordered_map<std::string,double>& extra = {})`
compiles and evaluates an arbitrary BNGL expression against the active
session. It resolves every parameter, observable, global function, and
the `time()` / `t` clock against the current pool; the `extra` map
layers additional name=value bindings on top, shadowing model symbols on
a clash. It needs no pattern parser — the expression evaluator from
[issue #6] already compiles BNGL expression strings.

[i9]: https://github.com/richardposner/RuleMonkey/issues/9
[issue #6]: https://github.com/richardposner/RuleMonkey/issues/6
