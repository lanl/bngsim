# Troubleshooting / FAQ

The questions below come up often enough that they're worth pulling
into one place rather than scattering them across the changelog,
header comments, and the model-semantics doc.

## Loading the model

### "ERROR: Unsupported BNGL feature: …"

`rm_driver` refuses by default if the model uses a BNGL construct RM
cannot honor (compartments, Arrhenius rate laws, Sat / Hill rate laws,
`population` molecule types, multi-molecule Fixed species, …).  Each
refusal names the offending element and gives per-feature guidance.
(`FunctionProduct` rate laws are now implemented — see
[`model_semantics.md`](model_semantics.md).)

Two ways forward:

- **Fix the model.**  Most refusals come with a concrete suggestion
  (e.g. "rewrite Sat() as MM(kcat, Km)").  See
  [`model_semantics.md`](model_semantics.md) for the full table.
- **Run anyway.**  Pass `--ignore-unsupported` to the CLI; the
  features are demoted to warnings.  The trajectory will run but
  may diverge from BNGL semantics — each warning explains exactly
  how.

### "Cannot resolve value '...'"

The XML referenced a parameter or expression the loader couldn't
resolve.  Most common causes:

- Typo in the BNGL parameter name.
- Forward reference that the parser's iteration cap can't resolve
  (the cap is `parameter_count + 4` passes; cycles never resolve).
- A rate-law expression using a function the expression evaluator
  doesn't know.  Standard math built-ins (`exp`, `ln`, `sqrt`, `if`,
  `min` / `max`, `pow`, …) are provided by ExprTk; user-defined global
  functions need a matching `<Function>` block in the XML.

### "ExprTk compilation failed for expression: '...'"

A rate-law / function / parameter expression could not be compiled by
the ExprTk evaluator.  Common causes:

- A built-in math function called with the wrong arity (e.g. `pow(x)`
  with one arg instead of `pow(x, y)`).
- A reference to a name that is neither a parameter, an observable, nor
  a declared global function.
- A genuine syntax error in the BNGL expression.

The ExprTk parser message after the `—` pinpoints the offending token.

### Differs from NFsim by a small but reproducible amount

RuleMonkey defaults to **strict BNGL semantics**
(`block_same_complex_binding = true`, equivalent to NFsim's `-bscb`
flag).  NFsim's default is non-strict: bimolecular rules can fire
intra-complex, and DeleteBond on a ring bond does not check whether
the products are actually disconnected.

If you're comparing against an NFsim run that omitted `-bscb`, pass
`-no-bscb` to `rm_driver` (or `set_block_same_complex_binding(false)`
on the in-process API).  Otherwise the trajectories will diverge
on any model with same-complex binding shapes or ring bonds.

See `model_semantics.md` § "Strict BNGL semantics" for the precise
LHS-`+` and RHS-`+` rules RM enforces.

### Function rate law goes negative

If a function rate law in your BNGL can evaluate to a negative number
during simulation, the two engines diverge:

- **NFsim** prints `The function you provided for functional rxn …
  evaluates to a value less than zero!` and aborts the run.  This
  surfaces immediately when the negative branch is reached, which may
  be at `t=0` or partway through the trajectory.
- **RuleMonkey** clamps the propensity to zero for that step and keeps
  running — the affected reaction is treated as having no firing rate
  while the underlying expression is negative.  The clamp is in
  `set_rule_propensity` (`cpp/rulemonkey/engine.cpp`) and applies to
  every rate-update path.  On the first clamp per rule, RM prints one
  diagnostic line to stderr:
  ```
  WARN: rule '<rule-id>' (<rule-name>) propensity clamped to 0 — rate
  function '<fn-name>' evaluated to <value> at t=<time>;
  further clamps on this rule are silent
  ```
  Subsequent clamps on the same rule are silent.  This catches the
  authoring slip without spamming the log on a model that legitimately
  oscillates around the zero crossing.

This shows up when a continuous-ODE rate term has been folded into a
single BNGL rule.  For example:

```
prod_X() = p_3 * scale * (Obs_I - I_b)   # ODE term: dX/dt += this
0 -> X()  prod_X()                       # SSA: rate can be negative
```

In the ODE this term provides decay when `Obs_I < I_b`; in SSA the
"production" reaction has no decay channel and the negative value is
not physically meaningful.

To make the model run identically on both engines, **split into two
sign-guarded reactions**:

```
prod_X_pos() = if(Obs_I > I_b, p_3 * scale * (Obs_I - I_b), 0)
prod_X_neg() = if(Obs_I < I_b, p_3 * scale * (I_b - Obs_I), 0)
0 -> X()  prod_X_pos()
X() -> 0  prod_X_neg() / Obs_X   # or whatever turns the rate into a per-mol rate
```

(The per-molecule conversion for the decay branch is model-specific —
ODE→SSA conversion isn't mechanical when the underlying term mixes
production and decay.)

See also `model_semantics.md` § "Rate laws / Function" for the
engine-side rule.

## Parameter overrides

### `set_param` had no effect on my run

Three things to verify:

1. **Did you call it before `run()` / `initialize()`?**  Mutators
   throw if a session is active — call `destroy_session()` first.
2. **Is the parameter name spelled exactly as it appears in the
   BNGL?**  `set_param` rejects names not declared in the loaded
   XML; if it didn't throw, the name matched.
3. **Is the parameter referenced through the rate-law / observable
   AST, or baked into a `RateConstant` value?**  RM re-resolves the
   four parameter-derived numeric fields (`Ele rate_value`,
   `MM kcat / Km`, `SpeciesInit concentration`) inside
   `apply_overrides()` at the start of every `run()`.  A parameter
   that BNG2.pl emits as a literal (post-evaluation numeric in the
   XML, not a symbolic expression) cannot be overridden — that's a
   BNG2.pl emit choice, not an RM limitation.  Inspect the XML to
   confirm the field still carries the symbolic source.

`set_param` cascades through derived parameter expressions in
declaration order (see `model_semantics.md` § "Parameter
overrides"), so `set_param("A_base", x)` propagates to
`A_tot = A_base * A_factor` automatically.

### `get_parameter` returned the parsed default after I called `set_param`

Fixed in 3.1.1.  If you're seeing this, you're on an older build.

## Save / load

### "State file schema fingerprint mismatch"

The XML the simulator was constructed from is structurally different
from the one used at `save_state` time (different molecule types,
different component names, or different allowed states).  The pool
serialization is keyed by integer indices into the schema, so a
mismatched XML would silently produce corrupt trajectories — RM
refuses loudly instead.

Fix: load with the same XML that was used to save.  Parameter
values, rate constants, and seed concentrations may legally differ
between save and load (e.g. resuming a checkpoint with new
`set_param` overrides) — only the schema needs to match.

### "State file is RM_STATE_V1 (pre-fingerprint format)"

V1 state files do not embed a schema fingerprint, so loading them
risked the silent-corruption mode above.  3.1.1 refuses V1 files
explicitly.  Re-save with the current build to get a V2 file.

## Output

### Trajectory only has 6 significant figures

The CLI's `.gdat` output uses `std::cout`'s default float
precision.  See [`gdat_format.md`](gdat_format.md) § "Numeric
precision" for the rationale and the patch site if you need more
digits.

The in-process C++ API (`rulemonkey::Result`) returns full
double-precision values; the precision loss is purely a property of
the text output path.

## Building and testing

### `RULEMONKEY_INSTALL=ON` plus `RULEMONKEY_ENABLE_ASAN=ON` is refused

Intentional.  An asan-instrumented installed library would propagate
`-fsanitize=address,undefined` to every downstream
`find_package(RuleMonkey)` consumer's final link line, breaking
their builds.  Use `cmake --preset asan` (which auto-disables
install) for sanitiser runs, and the `release` preset for installed
artifacts.

### `pytest` from the repo root collects nothing

That's correct.  RM has no Python tests; the repo's `tests/` is
C++-only (driven by ctest) and `harness/` is benchmark/research
scripts.  See [`test_corpora.md`](test_corpora.md) for what to run.

### NFsim regen requires `NFSIM_BIN`

Reference data (`tests/reference/nfsim/ensemble/*.tsv`) is
checked in; you don't need NFsim to run RM or its parity benchmarks.
Only **regenerating** references requires a local NFsim build, set
via the `NFSIM_BIN` env var.  See
[`tests/reference/nfsim/PROVENANCE.md`](../tests/reference/nfsim/PROVENANCE.md)
for what a regeneration would entail (the regen scripts are not
currently in this repo's tree).

## Performance

### My model runs slower than NFsim

[`timing_comparison.md`](timing_comparison.md) breaks the 173-model
RM-vs-NFsim comparison into six speedup buckets and explains where
each engine wins.  The short version: RM wins on
count-relation Species observables (`Species X R()=N` shapes) and
short total-wall runs;  NFsim wins on heavy enzyme-kinetics models
(`e1`–`e9`) and very-long stochastic runs.  Pattern canonical
labelling (a candidate optimisation flagged in `CHANGELOG.md`) would
narrow the latter.
