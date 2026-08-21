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

### Fixed

- **A rate law whose derivative keeps a `max()` or a `min()` no longer loses its
  analytic Jacobian (issue #460).** The interpreted emitter wrote sympy's
  spelling, `Max(k1, k2)`, and the engine's expression parser is case sensitive,
  so it answered "Undefined symbol: 'Max'" and the model fell back to the
  finite-difference Jacobian without saying so.

  `_SYMPY_FUNC_TO_EXPRTK` did map `Min` and `Max` to the engine's names, and
  those two entries had never once been read. sympy's `Min` and `Max` are not
  `Function` subclasses, so they never reached the printer's `_print_Function`,
  and sympy's own `StrPrinter._print_LatticeOp` handled them instead. That
  method prints the class name. The C emitter was never affected, because it has
  carried its own `_print_Min` and `_print_Max` all along.

  The same defect cost a second thing. The zero-logarithm guard from issue #333
  rewrites a rate law by parsing it to sympy and printing it back through this
  same printer, so a law carrying both a guardable logarithm and a `max()` came
  back spelled `Max(...)`, the engine refused to install it, and the guard was
  dropped in silence. Such a law then answers `nan` at zero concentration where
  it should answer zero, which is the outcome issue #333 exists to prevent.

  Two models in the parity corpus are affected, `ATG_model_v12` and
  `ATG_model_v16`, and both now attach the analytic Jacobian they were meant to
  have. Nothing about their runs moves: every solver statistic is identical and
  the trajectories agree to 2e-12, which is where an exact Jacobian and a
  difference-quotient one differ inside the corrector. Both models are small, so
  there is no measurable speed difference either. No model in a 1703 model sweep
  loses the logarithm guard this way.

  A `max()` over a variable being differentiated is a separate matter and stays
  refused on purpose, since its derivative is a step.

  Two guards come with it, for the blind spot rather than the instance. Anything
  else reaching `_print_LatticeOp` is now refused instead of printed under its
  class name. And the boolean nodes sympy can build but no printer here has a
  spelling for are refused too. That last one is not a defect anything reaches
  today, but its cost is different in kind: sympy prints `Xor` infix as `^`,
  which is legal ExprTk and means exponentiation there, so it would be a wrong
  number rather than a refusal.

### Added

- **A model that calls `mratio()` in a rate law now gets an analytic forward
  sensitivity instead of CVODES' difference quotient (issue #457).** The
  differentiation layer had never heard of `mratio`, so any derivative through
  it came back unevaluated and the whole model fell back to the difference
  quotient. That answer is correct but slower, and the derivative it was falling
  back from is available in closed form.

  `mratio(a,b,z)` is `M(a+1,b+1,z)` over `M(a,b,z)`, and Kummer's identity is
  `dM/dz = (a/b)*M(a+1,b+1,z)`. Through the quotient rule, writing `R` for
  `mratio`,

      dR/dz = R(a,b,z) * [ (a+1)/(b+1)*R(a+1,b+1,z) - (a/b)*R(a,b,z) ]

  so the derivative in the third argument is two more `mratio` calls. No new
  special function and no new numerics. This is the derivative worth having:
  BNG builds `a` and `b` from molecule counts and puts the rate constant in `z`,
  as `z = -1/Keq`, so fitting a rate constant goes through exactly this. The
  first two arguments have no comparable closed form and get none. A model
  differentiating through one of them declines and keeps the difference
  quotient, as it did before.

  The second call is the awkward part. Issue #453 gave `mratio` a region it
  trusts and made it refuse outside it, and `(a+1, b+1, z)` is not automatically
  inside: 97.6% of the arguments `mratio` answers for have the shifted call
  answered too, and the whole of the rest has `a` above -1, where `a+1` turns
  positive. Every model BNG generates has `a` at or below -1. Refusing the rest
  would turn a model that runs today into one that fails, since the compiled
  path has no retry, so the emitted helper falls back to a second exact
  expression for the same derivative. Kummer's equation
  `z*M'' + (b-z)*M' - a*M = 0` gives the contiguous relation

      (a+1)/(b+1)*R(a+1,b+1,z)*R(a,b,z) = ( b - (b-z)*R(a,b,z) ) / z

  so the whole derivative follows from the one call the value already makes.
  That subtraction cancels, which is why it is the second choice and not the
  first: against a 60 digit reference over 1293 arguments the shifted-call form
  is right to 9e-16 at the median and this one only to 5e-13, and for a small
  `|z|` it loses everything. A small `|z|` is where `mratio` trusts the shifted
  call unconditionally, so the fallback only ever runs at a large `|z|`, where
  its worst measured error is 1.7e-8.

  The interpreted engine has `mratio` but nothing for its derivative, so the
  interpreted Jacobian declines it and uses finite differences, which is what it
  did before. Only the generated C has the helper.

- **`mratio()` answers for 40% more of the arguments it used to refuse (issue
  #456).** Issue #453 made `mratio(a, b, z)` refuse the arguments its continued
  fraction cannot be trusted with, which removed a class of silently wrong
  answers. The cost was that it also refused arguments the fraction would have
  got right, since in the uncertain region there is no way to tell which ones
  those are. This narrows that region with a second method that can vouch for
  its own answer.

  For a large `|z|` the ratio has an asymptotic expansion in which the Gamma
  factors cancel between numerator and denominator. That cancellation is the
  point: the individual Kummer functions overflow a double long before their
  ratio does, which is the whole reason `mratio` exists.

      z to -infinity:  R(a,b,z) = (b/(-z)) * 2F0(a+1, a-b+1; ; -1/z)
                                           / 2F0(a,   a-b+1; ; -1/z)
      z to +infinity:  R(a,b,z) = (b/a)    * 2F0(b-a, -a;    ;  1/z)
                                           / 2F0(b-a, 1-a;   ;  1/z)

  `2F0` is a divergent asymptotic series, so each is summed to its smallest
  term, which is where such a series comes closest to the function it stands
  for, and that term estimates how close it got. Certifying its own accuracy is
  what makes the route safe to add at all: it is consulted only for arguments
  that were going to be refused, and only its own estimate lets it answer, so it
  cannot change a value `mratio` already returns and cannot turn a refusal into
  a wrong number. Where the estimate is not small enough the refusal stands.

  Two things it gets wrong if they are not guarded, both found by breaking it.
  An expansion whose leading coefficient is `1/Gamma(b-a)` has no leading term
  at all when `b - a` is a non-positive integer, so what is left is the
  exponentially small piece the expansion drops; unguarded it returned 5e-06 for
  `M(a,a,z)` over itself, which is exactly 1. And a series that stopped early
  because a term hit exactly zero has demonstrated no decay, so its smallest
  term says nothing; unguarded it returned 14.33 where the ratio is 0.9998.
  Neither needs a test of its own in the end. `lgamma` is `+inf` at exactly the
  first set of arguments, so estimating the dropped branch in logs refuses them
  by arithmetic, and taking the estimate to be the last term actually added
  rather than the zero that ended the series covers the second.

  Measured over 5955 argument triples against a 40 to 60 digit reference, the
  refused fraction of the grid falls from 46.4% to 28.5% and the worst error
  among the newly returned values is 2.7e-11. The estimate itself is honest to
  about a factor of ten, so the 1e-10 it is held to is worth roughly 1e-9 in the
  worst case.

  As with the fraction, `src/expression.cpp` is the single source of truth and
  the generated C carries a copy held to it. The two agree bit for bit at every
  one of the 5955 triples.

- **A rate law that switches on a repeating schedule now has a forward-sensitivity
  gradient (issue #436).** `if(time() - 24*floor(time()/24) >= 7, on, off)` is "on
  for the last 17 hours of every 24 hour day", and with a start time and a fitted
  period it is how a model writes repeated dosing, a light and dark cycle, or a
  train of stimulus pulses. bngsim refused every one of them. A census of the
  crossing gate over this repository's corpus — 585 `.net` and 1317 SBML models —
  found 22 models with a rate-law crossing nothing compensated, and 19 of the 22
  were this one shape.

  Every crossing recogniser before this one names a fixed number of crossing
  times, because its residual is a polynomial in the clock. A schedule has one in
  every period for as long as the run lasts, so the recogniser answers with the
  *pattern* — a period, an offset and a duty — and the run-time detector, which is
  the only part of this that knows how long the run is, enumerates the edges from
  it: `offset + k*period + duty` where the condition turns over inside a period,
  and `offset + (k+1)*period` where the `floor` steps and the remainder drops back
  to nothing. Each is then an ordinary issue #48 record, and `∂t*/∂p` follows from
  differentiating that expression, so the period moves the k-th edge k times as far
  as the first one. Nothing roots on anything and nothing searches: the edges are
  arithmetic.

  The recogniser reads the schedule rather than the spelling (issue #355), which
  matters because the corpus writes the same 24-hour cycle at least five ways —
  the threshold on either side, a start time folded in, a remainder taken in
  seconds and divided back to hours, a period hidden behind a derived parameter,
  and `ceil` in place of `floor`. What it will not read is a remainder *of* a
  remainder, or a residual that does not repeat period to period; both stay
  refused.

  `floor()` had a second, separate reason to refuse these rate laws: it is not
  differentiable, so the emitter rejected it wherever it appeared. Inside an
  `if()` condition it is never differentiated — sympy copies a Piecewise's
  conditions through untouched — so it is waived there and nowhere else, and the
  crossing gate decides what happens next.

  Corpus census, two arms over the same 1908 models: **11 SBML models gained the
  analytic sensitivity RHS, none lost it, and no model gained an error.** Six more
  moved only in the wording of their refusal. Three of the newly-admitted models
  were checked against a finite difference of two trajectories with the parameter
  edited in the SBML document rather than written through the loader:
  BIOMD0000000693 agrees to 4e-7 on its stimulus period, BIOMD0000000678 to 3e-6
  on a period reached through a derived parameter, and BIOMD0000000450 — whose
  schedule is written with literals, so what it gains is the in-branch derivative
  rather than a jump — to 3e-9.

  How many edges a window holds is the one thing about a schedule that is not a
  property of the model, so a long enough run at a short enough period can ask for
  an unbounded number of stop times. There is a budget, and past it bngsim refuses
  the run and says so: compensating the edges that fit and not the ones after them
  would give a gradient right at the start of a run and silently wrong at the end.
  The budget is set well above what the schedules people write need — a hundred
  days of hourly dosing fits, and the largest any corpus model asks for over its
  own reported time course is 200 edges.

- **A Simulator now says whether this model's gradient is analytic, and why not
  when it is not (issue #438).** Whether forward sensitivity runs on bngsim's
  analytic `∂f/∂p` or on CVODES' internal difference quotient is a property of the
  (build, model) pair, not of the build, so no `capabilities()` key can answer it:
  two models on one bngsim get different answers, and the same model can flip on a
  derivation-budget timeout. `CVodeSensInit1` takes one sensitivity-RHS callback
  for every column, so a single rate law that cannot be differentiated declines the
  analytic derivative for the whole model, and the difference quotient that
  replaces it costs an extra right-hand-side evaluation per column per step —
  roughly N times the sensitivity cost on an N-parameter fit. bngsim knew the
  answer and published neither half of it. `Simulator.has_analytic_sens_rhs` is the
  verdict, read off the compiled artifact exactly as the private method it promotes
  did, and `Simulator.sens_rhs_decline_reason` is the reason, or `None`.

- **The decline reason now survives the codegen cache.** It was derived once, while
  source was being generated, and thrown away. Since issue #174 the cache key is
  structural, so a warm cache resolves the `.so` without generating any source —
  and the first construction of a declining model reported the reason while every
  construction after it said nothing, both on the same fallback. The cache is on
  disk, so the run that heard nothing was typically the second run: the one made
  after the first came back empty. The reason is now written into a small note
  beside the artifact it describes and replayed from there on a cache hit, in the
  same words a cold build uses, so a consumer listening on the `bngsim` logger sees
  no difference between the two. Whether the difference quotient is merely slower
  or answers a different question — a branch crossing whose time moves, where every
  column is wrong at and after it — is carried through the note as well, since
  "wrong" and "slow" are different statements. `bngsim-cache` knows the note as one
  of bngsim's own files: it is counted under its own kind, `clear` removes it, and
  `prune` removes it exactly when it removes the artifact it describes.

### Fixed

- **A second build with different options silently changed what
  `scripts/rebuild_editable.py` produced (issue #459).** Every build for one
  interpreter and platform uses the same CMake build directory —
  `build-dir = "build/{wheel_tag}"`, and the wheel tag records the Python version
  and the platform, not the virtual environment being installed into and not the
  options passed. So an install into a *separate* venv rewrites the cache the
  first one's rebuilds read.

  That script re-specified some options on its configure line and inherited the
  rest. `BNGSIM_ENABLE_KLU` and `BNGSIM_REQUIRE_KLU` were never passed, so they
  came entirely from whatever the last build left behind; `BNGSIM_ENABLE_MIR` was
  passed only when its environment variable was set, so a cached `ON` stayed
  `ON`. One MIR-configured install into a second venv was enough to turn
  `print(HAS_KLU, HAS_MIR)` from `True False` into `False True` at the next
  rebuild of the original environment. Nothing said so: the staleness guard
  compares source timestamps against the binary, and the binary really was
  fresh — it was just configured differently.

  The configure line now names every option that decides what gets compiled in,
  resolved from the CMakeLists.txt `option()` defaults, then `pyproject.toml`,
  then a same-named environment variable — never from the tree. `CMAKE_BUILD_TYPE`
  is pinned the same way, which was the same bug one step quieter: a tree another
  install configured `Debug` rebuilt as `Debug`, with only the compile lines to
  say so. A test holds the table to CMakeLists.txt and fails if an option is
  added there without being classified, because an unclassified option is one the
  configure line does not pass, which is one the cache decides.

  A cache that disagrees with what the rebuild asks for now stops it, naming each
  option, both values, and the recovery: `rm -rf` on the tree plus a reinstall,
  because reconfiguring an already-built tree leaves link settings from the
  configuration it was built with — in the reported case an extension that would
  not load at all (`Library not loaded: @rpath/libklu.2.dylib`). A difference
  this invocation asked for by environment variable is not a disagreement, so
  `BNGSIM_ENABLE_MIR=1 python scripts/rebuild_editable.py` still means what
  `scripts/MIR_VENDORING.md` says it means, and the same spelling now works for
  every option in the table.

  The committed `_bngsim_core.pyi` stub is where the drift was noticed — a
  modified tracked file flipping `HAS_KLU` and `HAS_MIR`, easy to commit by
  accident. Those flags describe one build, exactly like the `__build_commit__`
  and `__version__` stamps already normalized out of that file, so they normalize
  to `...` too. That also covers the deliberate case, where someone building with
  `-DBNGSIM_ENABLE_MIR=ON` is left holding a diff no other build agrees with.

- **`mratio()` had 60 more silently wrong answers than issue #453 measured
  (issue #456).** The rule that issue #453 shipped admits arguments when
  `b*b >= 64*|z*a|`, which bounds the *odd* partial numerators of the continued
  fraction, the ones carrying `z*(a+k)`. The even ones carry `z*(a-b-k)`, so for
  a large `b` they are about `z/b` however small `z*a` is, and nothing looked at
  them. The grid behind issue #453 had no arguments in that corner, so it
  reported no leaks there.

  On a denser grid it has 60, out of 3191 answers, every one with a positive `z`
  and a large `b`. `mratio(1, 901, 3000)` returned -0.4287 where the ratio is
  630.7. `mratio(0.5, 101, 300)` returned 1.0987 where it is 134.16. These are
  the same class of defect issue #453 set out to remove, in a corner it did not
  sample.

  Where the fraction turns is sharp and it is at `z = b`: over 597 triples it is
  clean through `z/b = 1.02` and 12 of 47 are wrong at 1.05. So the clause now
  also asks for `2*z <= b` when `z` is positive, a factor of two below a
  threshold whose position is understood rather than merely observed. A negative
  `z` neither needs the companion bound nor gets one, measured clean out to
  `|z|/b = 300`, because there the partial numerators alternate in sign and the
  fraction stays a contraction.

  Tightening a boundary gives up answers that happened to be right, since what
  makes the corner unsafe is that the right ones and the wrong ones cannot be
  told apart without a reference. Of the 3131 triples the old rule answered
  correctly, 3080 come back bit for bit identical, 51 are now refused, and none
  changed value. Sixty wrong answers for 51 unverifiable right ones is the trade,
  and the asymptotic route above more than covers it.

  Taken together, `mratio` now answers 4259 of those 5955 triples against 3191
  before, and none of them wrongly.

- **`mratio()` no longer returns a badly wrong value without saying so (issue
  #453).** `mratio(a, b, z)` is the ratio of contiguous Kummer functions,
  `M(a+1,b+1,z) / M(a,b,z)`, computed from Gauss's continued fraction by the
  modified Lentz method. Outside a certain range of arguments that fraction
  converges to something that is not the ratio. The error reached a factor of a
  thousand, and in places the sign was wrong. Nothing failed and nothing hung:
  the value was simply returned, and no caller could tell it from a good one.

  Three things had to be established before the fix, and each rules out a
  cheaper one. The approach to the false limit is indistinguishable from the
  approach to the true one, since the iteration error decays smoothly and
  geometrically into both, so no stopping test can separate them. Iterating
  longer does not help either: at 120 digits the same recurrence does reach the
  true value, but only after about 1600 steps where double precision settles at
  about 84, by which point rounding has frozen the iterates. So the decision has
  to be taken from the arguments, before the fraction runs.

  `mratio` now computes an answer where it can be trusted to and refuses
  elsewhere. It is trusted when `a` is a non-positive integer, when `a` is at or
  below zero and `z` is too, when `|z|` is at most 20, or when `b*b >= 64*|z*a|`.
  The first of those is a fact about the algorithm rather than a measurement: the
  odd partial numerator carries `z*(a+k)` and reaches exactly zero at `k = -a`,
  so the fraction terminates and computes a finite exact sum. The rest were
  measured against a 40 to 60 digit reference. The `|z|` clause is the widest:
  over 3400 triples chosen to be hostile, with small `b` and `a` on both sides of
  zero, the fraction was right at every one with `|z|` up to 50, and the first
  wrong answer anywhere is at 60 and is marginal.

  No model BNG generates is affected. A model builds `a = -min(AT, BT)` and
  `z = -1/Keq`, which is inside the second clause, and stays inside it when a fit
  moves the counts off the integers. That last point is why the rule is written
  the way it is: a rule stated only as a bound on `|z*a|` against `b*b` passes
  every grid check and still refuses the fit path of the very model `mratio`
  exists for.

  Over 1409 argument triples checked against a 40 digit reference, the number
  returned wrongly goes from 70 to zero, with a quarter of that grid refused.
  Most of those refusals are arguments the fraction would have got right, because
  in the uncertain region it is right most of the time and there is no way to
  tell which times. A refusal that names the region is worth more than a number
  that is usually right.

  Both paths say why. The interpreter raises and names the region and the issue.
  The generated C cannot raise, so its copy returns a NaN, which the caller
  already turns into a failed step — and that failure now carries the same
  sentence, because describing a non-finite witness re-evaluates the model at the
  offending state and the refusal comes back out of it.

  That last part needed a fix of its own, in `describe_nonfinite_witness`. It
  wrapped the whole description in a `catch (...)` that returned an empty string,
  which was harmless while every rate law answered with a number. A law that
  refuses throws instead, and the description vanished along with it, leaving a
  bare `CV_FIRST_RHSFUNC_ERR` and nothing about `mratio` — a worse thing to
  diagnose than the wrong number it replaced. A refusal is now passed on rather
  than swallowed, for any function that refuses, not only this one.

  What is refused could be narrowed later. The ratio has an asymptotic expansion
  for large `|z|` in either direction whose truncation term certifies its own
  accuracy, and routing to it recovers about a quarter of the refusals. It is
  left out here because it computes a new value rather than gating an existing
  one, so a mistake in it would put back the class of defect this removes.

- **A rate law using `mratio()` now compiles (issue #451).** `mratio` is the last
  of the engine's reserved functions that C has no name for. Issue #448 dealt
  with the other five by rewriting each to a C expression. This one is a loop, so
  the generated source now carries the loop: a port of `expr_compat::mratio` from
  `src/expression.cpp`, emitted into every generated file alongside the
  portability macros, with `mratio` in a model's text rewritten to call it.
  `src/expression.cpp` stays the single source of truth, and the port is held to
  that by tests that ask both for the same numbers over a swept argument and
  require them to be equal rather than close. They are: over seven models that
  ride `z`, `a` and `b` through the species, including the large-argument case
  the C++ comment singles out, the two paths agree bit for bit.

  The C++ throws when its iteration cap is reached. There is no exception to
  throw in C, so the port returns NaN there, which the caller already turns into
  a recoverable step failure naming the time and the state that produced it. No
  new mechanism, and the cap is far out of reach in any case: the arguments a
  BNG model produces converge in about 25 iterations.

  Adding a special function of this kind now takes three steps and only three,
  written down next to the code that does it: register it on the interpreter, add
  its C to one tuple, and add one line to the name table.

  One line of the C++ changed with it. The parity bookkeeping used to swap `odd`
  with a second flag that nothing ever reads, and is now the toggle that says,
  which is the same thing written shorter. It had to change because c2mir, the
  frontend of the MIR JIT backend, miscompiles that swap: it accepted the code
  and then returned a wrong number for every argument, on all four MIR platforms.
  The two spellings are kept identical so that this function stays the single
  source of truth for both. The change was held to bit-for-bit equality over 300
  argument triples, and none of them moved.

  The derivative is unchanged: a model calling `mratio` in a rate law still
  declines the analytic sensitivity right-hand side and uses CVODES' difference
  quotient, because nothing here tells the differentiation layer what the
  function means. Worth recording for whoever picks that up, since it is not
  obvious: with `R(a,b,z) = mratio(a,b,z)`, Kummer's identity
  `dM/dz = (a/b) M(a+1,b+1,z)` gives

      dR/dz = R(a,b,z) * [ (a+1)/(b+1) * R(a+1,b+1,z) - (a/b) * R(a,b,z) ]

  so the derivative with respect to the third argument is two calls to `mratio`
  itself, needing no new function and no new numerics. It was checked against
  `scipy.special.hyp1f1` and against finite differences of both scipy's ratio and
  bngsim's own, agreeing to 1e-10, and it works at `a = -1000, b = 9001,
  z = -10000` where scipy overflows and cannot form the ratio at all. That
  derivative is the one a fit needs: both models in this repository that use
  `mratio` do so with `z = -1/Keq`. The derivatives with respect to the first two
  arguments have no such closed form.

  Measuring the port turned up something separate and worse, now filed as issue
  #453: the algorithm itself returns a badly wrong value, with no warning, when
  its first argument is positive and its third is a large negative number. Over
  576 grid points checked against a 60 digit reference, every one of the 288 with
  a non-positive first argument is right to about 1e-15, and 15 of the rest are
  wrong, the worst by a factor of a thousand. The regime BNG models produce is
  entirely inside the good region. The port copies that behaviour rather than
  diverging from it, and a test pins the two copies to each other so that fixing
  one without the other fails immediately.

- **A rate law using `sign()`, `sgn()`, `clamp()`, `avg()` or `sum()` now compiles
  (issue #448).** All five are in the engine's reserved function list, so a model
  is allowed to call them and the interpreter evaluates them, but C has none of
  them under those names. The name went into the generated source unchanged and
  the compile failed with "call to undeclared function", which took down an
  explicit `codegen=True` run and every forward sensitivity run of the same
  model. A plain interpreted run was fine throughout, so the failure only
  appeared once a user asked for speed or for gradients.

  Each of the five now becomes an ordinary C expression before the generated
  source is written. The forms transcribe the engine's own implementation rather
  than the textbook definition, because for `clamp` the two differ: the engine
  returns the low bound when the value is below it and the high bound when the
  value is above it, tested in that order, so with the bounds crossed neither
  `fmax(lo, fmin(x, hi))` nor `fmin(hi, fmax(x, lo))` gives what the interpreter
  gives. Crossed bounds are a nonsense model, but the bounds can be fitted
  parameters, and a fit that walks them past each other should not change which
  answer a user gets.

  The compiled and interpreted paths were compared over whole trajectories for
  every one of the five, plus nested and conditional combinations of them, and
  agree to the last bit. The derivative is a separate question and is unchanged:
  none of these functions has a derivative the emitter can write, so a
  sensitivity run still declines the analytic right-hand side and uses CVODES'
  difference quotient, which is what it was always meant to do. Those
  sensitivities were checked against a finite difference taken by editing the
  model text and reloading.

  Corpus census: no model in this repository uses any of the five, in either
  format, so this changes no existing result. Measured rather than assumed: the
  new rewriting pass runs over every expression of every model, and across 66707
  expressions from 1821 `.net` files and 390 SBML models it left every one of
  them exactly as it was.

  One function in the same list is still affected. `mratio` is bngsim's own
  confluent hypergeometric ratio, computed by a continued fraction rather than by
  a one-line expression, so giving it a C spelling is a separate piece of work
  and is tracked in issue #451.

- **An ordinary ODE run no longer prints SUNDIALS error lines about forward
  sensitivity (issue #447).** A plain `Simulator(model, method="ode").run(...)`
  put lines like

      [ERROR][rank 0][.../cvodes_io.c:2322][CVodeGetSensNumNonlinSolvConvFails]
      Forward sensitivity analysis not activated.

  on standard error even though the run was healthy. bngsim reads CVODE's
  counters once per solver segment, and two of the counters it asked for belong
  to the forward sensitivity solve. CVODES answers a request for either of those
  on a run that has no sensitivities by printing one of those lines and then
  returning a flag that says so. Leaving the two counters at zero is the right
  answer for such a run, so the flag was ignored, but the printed line went to
  the user regardless: two lines for a plain run, and two more for each event
  fire, because every fire re-initializes CVODE and closes a segment. A 240 unit
  run of a model dosed every 12 units printed 42 of them.

  bngsim now skips the two requests on a run that has no forward sensitivities,
  rather than switching off the SUNDIALS error handler, so a real solver error
  still reaches the user. Nothing about the reported counters changes: a run that
  does compute sensitivities still asks for both and still reports them.

- **A derived expression that is a single `floor()`, `ceil()` or `sign()` call no
  longer crashes a sensitivity run (issue #441).** A model whose initial condition
  is built from a step function, `A0 = floor(P)`, died with
  `SimulationError: Simulation failed: maximum recursion depth exceeded` as soon as
  a forward sensitivity was asked for. Sympy answers `d/dP floor(P)` with an
  unevaluated `Derivative` object rather than with a number, and evaluating that
  object recurses until Python gives up. `RecursionError` is not one of the
  exceptions this path expects, so it escaped the codegen build and came out of
  `Simulator.run`.

  Whether it crashed depended on the shape of the expression, which is why it was
  easy to miss: `floor(P)` crashed, while `floor(P)*7` came back as a clean
  refusal, because a product containing an unevaluated `Derivative` raises
  `TypeError` before the recursion starts.

  The answer for all of them is to decline. A value that steps as a parameter
  moves has no useful derivative with respect to that parameter, so the caller now
  gets the same empty result it already gets for any expression that cannot be
  differentiated, and the warning that follows names the expression and the
  parameter whose chain rule was dropped. The test is structural — is any part of
  what sympy returned still an unevaluated `Derivative` — so a function nobody
  thought to list, including one sympy has never heard of, is covered by the same
  line.

  Issue #436 closed one route to this crash by refusing any crossing threshold
  whose text contains `floor(`, `ceil(`, `sign(` or four other names, which left
  every other caller exposed: the initial-condition seeds above, and the derived
  parameter chain rule. That name list is gone now that the partials answer for
  themselves, and one case it was wrong about goes with it: `floor(5)` is 5, a
  crossing nothing moves, so a rate law that switches at one keeps its analytic
  sensitivity right-hand side instead of losing it to a text match. Its columns
  were checked against the closed-form solution of that model (agreement to 2e-11)
  and against a finite difference taken by editing the model text and reloading
  (agreement to 9e-7, which is that reference's own noise floor).

  Corpus census, two arms over the same 1908 models: **no model changed any
  answer** — not the crossing gate, not the analytic right-hand side, not a
  switch-time record, not an initial-condition sensitivity seed, and no model
  gained or lost an error. Five `.net` models seed an initial condition from
  `rint()` of a fitted parameter; all five declined that chain rule before and
  decline it now, in the same place, with only the wording of the warning
  different. Because the census found nothing, it was run against two controls to
  show it would have seen something: the model from the issue report, which moves
  from `RecursionError` to a clean decline, and a rate law switching at `floor(5)`,
  which moves from refused to compensated.

- **A BNGL rate law that switches on simulation time is no longer integrated
  straight over the switch (issue #440).** A pure accumulator that fills at 0.1
  for 40 of its 240 time units reported `0.0` where the answer is `4.0`, and
  nothing warned. Inside each branch of `if(time() >= 100, k, 0)` the right-hand
  side is a constant, so CVODE's local error estimate over a step spanning the
  whole branch is near zero and nothing stops the step from growing until it
  swallows the window. Tightening `rtol` does not help, because there is no error
  to see. A repeating schedule is worse, because there are many windows to miss:
  `if(time() - 24*floor(time()/24) >= 7, k, 0)` reported 23.3 against an answer of
  17, and a three-unit period with a half-unit window reported `0.0` against 4.

  The same model written in SBML was already right. Its loader walks the document
  at load time and registers every `time` comparison as a CVODE root, and issue
  #305 resolves each root to a crossing time and ends the step exactly on it. A
  `.net`/BNGL model is built entirely in C++, so its loader has no build-time seam
  to register anything at, and it registered nothing at all.

  The conditions are now recovered from the built model's own function bodies —
  the same scan the forward-sensitivity path already ran over the same text — and
  handed to the same stop-time machinery, which lands the step on the crossing and
  reinitialises there. A repeating schedule gets one stop per edge, enumerated
  from the pattern recogniser issue #436 added. Admission is narrow: a comparison
  against simulation time, against values that hold still for the whole run.
  A threshold over live state crosses at a time nobody knows in advance and is
  left to issue #150, and an equality is left alone as well, matching what the
  SBML scan admits.

  One C++ line goes with it. The warm CVODE fast path has no stop-time handling of
  its own, and until this issue nothing could reach it carrying stops, because
  every model that had them had registered the roots that produced them. A `.net`
  model has the stops without the roots, so the exclusion is now made on the stops
  themselves.

  **The schedule half of this also repairs SBML**, which the root registration
  did not. A GH #72 root is evaluated on the *boolean*, and the boolean of a
  repeating schedule reads the same on both sides of a step spanning a whole
  period, so there is no sign change for the root finder to see. The accumulator
  above written as an SBML `piecewise` on `time - 24*floor(time/24) >= 7`
  reported 10.6 against the same answer of 17, with its one root registered and
  none of its twenty edges stopped at.

  One neighbouring shape is measured and left alone. A BNGL model more often
  measures time with a counter species than with `time()`, and a rate law that
  thresholds the counter has the same defect: that is issue #443, and admitting
  it moves the stepping of 37 corpus models and needs a BioNetGen parity re-run,
  which matters more there because BioNetGen's own integrator has the same
  blindness and a reference trajectory may itself have stepped over the switch.

  Corpus census, two arms over the same 1908 models (585 `.net` and 1323 SBML):
  **no model gained or lost an error, and nine models moved, all of them SBML
  models with a repeating schedule.** This repository's `.net` corpus contains no
  model that switches on simulation time at all — 80 of the 585 carry a
  conditional rate law and none of those mentions time — so the BioNetGen parity
  suite and the `.net` timing benchmarks measure models the change cannot touch.
  Confirmed rather than assumed: all 592 ODE jobs were re-run against BioNetGen
  2.9.3 in both arms and **not one number moved**, with the same 590 passing, the
  same 2 differing (a Lorenz attractor and a proliferation model, both already
  differing before this), and every error against the reference identical to the
  last digit.

  Of the nine, seven move by 2e-5 or less. `BIOMD0000000808` sits at its
  reference's own convergence in both arms (1.70e-7 against 1.68e-7 from a run
  bounded at a fortieth of the pulse width). `MODEL0406793751` — a stimulus on for
  one part in a thousand of its period — moves a long way and moves the right way:
  against a reference bounded below the pulse width and converged to 5e-6, the
  distance drops from 1.14e5 to 1.37e4 in the mean-square, from 6.07e3 to 4.97e2
  in the mean, and from 9.20e5 to 1.02e5 at the peak. RoadRunner parity over all
  nine is unchanged: eight PASS at `max_rel_err = 0` in both arms, and
  `MODEL0406793751` is the ninth, which RoadRunner declines to load.

  Cost, measured serially: the per-run crossing resolution over all 339 SBML
  models that carry a registered condition is 1098 ms against main's 1085 ms, and
  the one-time recovery over all 80 conditional `.net` models is 2 ms. Both
  numbers took a memo to reach. Recognizing a schedule and then checking it
  against the model's own residual is seven sympy round trips, paid on every
  `run()` and so on every evaluation of a fit, which put the first of those
  numbers at 1183 ms; it is now keyed on the condition text, the window and the
  parameters the condition reads, exactly as the issue #305 crossing memo already
  was. The second number is 43 ms on the largest `.net` model alone without the
  pre-check on the raw rate-law text, which is why that pre-check asks about
  `time` and not only about `if()`.

- **A BNGL rate law that switches on a counter species is no longer integrated
  straight over the switch either (issue #443).** Issue #440 fixed the same defect
  written against `time()`. This is the spelling BNGL models actually use: a
  species fed by a zeroth-order reaction at rate 1, read back through a group and
  conventionally called `t`. Of the 585 `.net` models in this repository's corpus
  **37 threshold such a counter and none thresholds `time()`**, so until now the
  fix reached no model here at all. The issue's own accumulator, which fills at
  0.1 for 40 of its 240 time units, reported `0.0` where the answer is `4.0`.

  A counter obeys `dc/dt = 1`, so its value is simulation time plus a constant
  offset for the whole of the run and one substitution turns a condition on the
  counter into the condition on time it already is: `c` becomes
  `time() + (c(t_start) - t_start)`. Both resolvers issue #440 built then read it
  unchanged, so a single threshold and a repeating schedule are both placed, and
  neither of them, nor anything below them, needs a second code path. The
  conversion is the one `compute_switch_time_sens` has always used to put an issue
  #48 switch time on a counter clock, and `_unit_rate_clock_species` is the same
  detector deciding what counts as a counter, so the stepping and the gradient
  cannot disagree about which species is a clock.

  A stop on a counter is not a stop on a time, and the difference is issue #82. A
  counter is *integrated*, so at the stop it reads a couple of parts in `1e14`
  BELOW the threshold it is defined as reaching there, the condition is still
  false, and the run restarts on the branch that just ended and meets the
  discontinuity inside the first step after a restart with no history to fall back
  on. Issue #82 already repairs this on the forward-sensitivity path, because a
  sensitivity record carries the clock index and the threshold beside the time.
  The crossing stop list was a plain vector of doubles and carried neither, so it
  is now a record too, and both paths land the clock through one shared rule. On
  the issue's own witness the counter reads `99.99999999999993` at the stop
  without the repair.

  The repair stands down on a run that has registered roots, and that is not
  caution. CVODE finds a root by a sign change across a step it accepts; moving
  the clock past the threshold during the restart presents no such step, and the
  root then never fires at all. An SBML rate rule `dk/dt = 1` makes `k` a counter,
  and on a model whose events trigger on `k > 4.5` the whole event was lost, with
  the pre-event stoichiometry carried to the end of the run. A `.net` model has
  stops precisely because it could register no root, which is the whole population
  this issue is about. The stop itself is still placed either way, and is still
  what makes the root reachable. Where a root IS registered on the crossing it
  reinitialises there itself, which is the whole of what the repair exists to do,
  so nothing is given up by leaving it to do it.

  The question is asked of the roots rather than assumed, and that matters:
  "stand down whenever the run has any root" would be simple and wrong. 21 of
  the 37 counter-clock models root on a state threshold — `V > 0` and the like —
  that has nothing to do with the clock, and those roots are registered on a
  fitted run, so such a rule would take the issue #82 repair away from exactly
  the runs that need it. Instead the clock is moved, the root function is asked
  again, and the move is taken back only where some root changed sign.

  **The issue #48 sensitivity jump had the same hole and is fixed with it**,
  because that is where the repair has lived since issue #82 and it applied
  unconditionally. On a model whose rate law thresholds a counter at a *fitted*
  parameter — the one shape that reaches the jump with a counter clock — and
  whose event triggers on the same value, the event fired on the plain run and
  did not fire on the sensitivity run, so the trajectory a fit saw disagreed with
  the trajectory a plain call saw about whether the event happened. It is
  reachable only through that combination, which no model in either corpus
  writes: the one SBML model with a counter-clock condition thresholds it at
  literals, so no parameter moves it and no switch-time record is emitted for it
  at all. Both paths ask the same question of the roots, so neither can drift.

  Corpus census over the 37, each run over its own reported horizon against a
  reference bounded at a thirty-two-thousandth of that horizon and integrated at
  `rtol = atol = 1e-12`, with the reference's own reproducibility measured
  alongside it (a second reference whose step bound differs by one part in
  32000; the worst of the 35 agrees with its twin to 1.2e-9 in the mean).
  **The largest mean error in either arm is 1.1e-4, and 29 of the 35 are at
  1.1e-7 or below in both arms.** `model_step2_v1` is the clear mover: a
  reference-independent 7.8e-4 before, 1.2e-4 after, on the pointwise metric.
  Nine models sit further from the reference afterwards at their own tolerance,
  at most 1.1e-4 in the mean, and a tolerance sweep over all nine says why: both
  arms shrink together as `rtol` tightens, with no floor in either, and at
  `rtol = 1e-12` every one of them is at 1.1e-8 or below. A stop in the wrong
  place would leave an error no tolerance could remove, and there is none.

  Two of the 37 have no converged reference to be read against, because neither
  can be integrated at `rtol = atol = 1e-12` in the first place: `m15` fails at
  t = 35, and `ItalyModel_v7` runs for tens of minutes before failing as well.
  Both are small either way: against BioNetGen `m15` moves from 7.3e-7 to 2.4e-5
  and `ItalyModel_v7` from 1.1e-8 to 1.6e-5.

  All 592 BioNetGen ODE parity jobs were re-run against BioNetGen 2.9.3 in both
  arms: **the same 590 passing and the same 2 differing, with no model changing
  its verdict.** Comparing the 37 against `run_network`'s own trajectories is the
  one place a number gets worse, and the reason is the one the issue predicted:
  22 of the 37 move away from BioNetGen, five of them from about `1e-13` to about
  `1e-7`, because BioNetGen's integrator has exactly the same blindness and
  bngsim used to reproduce its error step for step. Against a converged reference
  those same five move by `2e-8` or less.

  On the SBML side **exactly one model of 1323 is reached**, MODEL1508170000,
  whose rate rules make two parameters counters and whose rate laws threshold
  them at 480 to 483. Its parity job stops at an invented horizon of 100, so
  nothing in the suite touches those crossings; run out to 600 it moves by 8.2e-4
  at the peak and 1.6e-9 in the mean, and both arms sit the same distance from
  RoadRunner.

  Issue #54's stall fixture is a counter-clock model, so it now integrates rather
  than collapsing its step size at `sigma`, which is the better outcome and is
  asserted as such. The three tests that need the stall reach it by standing the
  crossing stop down, which is what a model whose crossing time bngsim cannot
  resolve looks like anyway.

  Cost: the one-time recovery over the whole 585-model `.net` corpus goes from
  19 ms to 211 ms, because 37 models now build a context they used to skip. The
  right-hand-side probe that decides whether a model has a counter at all costs
  0.6 ms across the 77 conditional models that never mention time, which is what
  keeps the rest of that work off them. Per-run crossing resolution over the 37 is
  26 ms once the issue #440 memo is warm.

- **A crossing time that steps rather than moves is declined instead of crashing
  the codegen pass (issue #436).** `if(time() >= sign(P), ...)` took bngsim down
  with a `RecursionError`: sympy answers `d/dP sign(P)` with an unevaluated
  `Derivative`, and evaluating that recurses until Python gives up, which is not an
  exception anything on the codegen path handles. `sign` is not one of the
  constructs the emitter pre-scan rejects, so this was reachable before this
  release; `floor` and `ceil` join it now that a `floor()` inside a condition is
  waived. All three now decline the model with a reason, which is also the honest
  answer — a crossing time that steps as a parameter moves has no chain rule to
  the model's primary parameters.

- **Switch-time crossing detection is no longer quadratic, and no longer
  re-derives one condition once per rate law that spells it.** Both cost the same
  thing — sympy work repeated for an answer already held — and both are paid on
  every `run()`, so a fit paid them on every evaluation.

  Each newly found crossing was compared against every crossing found so far, to
  decide whether it is one already recorded and written a second way (issue #375).
  Nothing in the corpus made that visible until a repeating schedule could
  contribute a thousand crossings to one run: at 1600 crossings the comparisons
  took 440 ms against 10 ms for everything else the detector does. The comparison
  now runs only against crossings sharing a clock and a threshold value, which is
  what the sameness test asks for on its first line anyway. 1600 crossings now take
  10 ms.

  Separately, the scan ran the threshold recognisers once per (rate law, atom)
  pair rather than once per distinct atom. A meal-timing model writes six
  conditions across twenty rate laws, so 120 recogniser passes produced 6 answers.
  Everything the scan does reads the atom text and the model scope and nothing
  else, so the extra passes could only re-derive what was already found.
  BIOMD0000000450 spent 87 ms per detection pass before and spends 18 ms now,
  BIOMD0000000268 78 ms and 16 ms — and those are the measurements for a build
  that also does the issue #436 schedule work the old one did not.

- **The analytic-`∂f/∂p` verdict no longer answers for an artifact the run has
  replaced.** `compute_all_sensitivities` and `steady_state` take
  `sensitivity_params` as a method argument and rebuild the codegen artifact for
  themselves, turning a plain build (which carries no sensitivity RHS at all since
  issues #209/#217) into one that does. The verdict is memoized on first read, and
  a read taken before that rebuild stayed False afterwards. It is now dropped
  wherever the artifact is replaced, in both directions. Found while publishing the
  reader, which is what made a read at that moment something a caller would do.

- **A rate law or switch threshold that spells one of the seven built-in
  physical constants is differentiated instead of declined.** `_pi`, `_e`,
  `_kB`, `_NA`, `_R`, `_h` and `_F` are bound by the expression evaluator on
  every model, and the engine reserves the names so nothing else can hold them.
  The Python differentiation layer knew them piecemeal: two of the seven were
  special-cased in one rate-law emitter, none of the seven were known to the
  derived-expression preparation that resolves a switch threshold, and the
  species-derivative emitter knew none either. The costs were real and silent
  about their cause. A single `_pi` in one rate law declined the analytic
  sensitivity RHS for a whole model, with a message about a derivative that could
  not be emitted as C. `if(time() < A*_pi, ...)` on BIOMD0000000616 was refused
  for forward sensitivity outright, reported as a threshold that does not reduce
  to a constant, when it is an ordinary clock crossing at A times pi.

  Both sympy entry points now bind the constants to their values, so every site
  downstream sees a number rather than a free symbol and needs no entry of its
  own. `_pi` and `_e` bind to sympy's own constants, so a rate law carrying one
  still prints `M_PI` or `M_E` in the generated C; the other five have no sympy
  counterpart and print as full-precision literals. The names and the values are
  both pinned against the engine itself in `test_builtin_constants.py`, so this
  second copy of the table cannot drift from the C++ one.

### Added

- **A rate-law switch condition quadratic in the clock is now compensated at
  both of its crossings, so its forward-sensitivity run is no longer refused
  (issue #421).** `if((time()-5)*(time()-5) >= thresh, ...)` is how a model
  writes a *window*: true early, false through the middle, true again late. Every
  recogniser before this one could name at most one crossing, so issue #414
  refused the shape outright. The quadratic formula writes both crossings in
  closed form, and differentiating a root expression is the implicit function
  theorem for that residual, so each crossing is an ordinary issue #48 record —
  evaluate it for `t*`, differentiate it for `∂t*/∂p`. `(time()-5)^2 >= thresh`
  at `thresh = 9` now stops at t=2 and t=8 with `∂t*/∂thresh` of ∓1/6, and the
  sensitivity column matches a central finite difference of two trajectories to
  5e-6 away from the two nodes.

  The recogniser therefore answers with a *list* of thresholds, and an atom is
  compensated only when every crossing in that list is. Compensating one edge of
  a window while the other flips the branch unjumped would be a silently wrong
  gradient, so the gate and the detector read one shared per-crossing rule and
  refuse the whole atom together.

  A clock threshold cubic or higher stays refused, and not because sympy declines
  to write its roots down: a cubic with three real roots has none expressible in
  real radicals, so the closed forms route through complex intermediates that
  would be read as crossings that never happen — dropping real jumps silently.
  Those want a numeric root find over the run window, which is different
  machinery.

- **A clock crossing whose time comes out non-real is read as a crossing that
  does not happen, instead of an unreadable threshold (issue #421).**
  `time()*time() >= thresh` at a negative `thresh` is true for the entire run:
  the branch never flips and `∂f/∂thresh` is a correct clean zero. bngsim refused
  the run, because the solved crossing time `sqrt(thresh)` did not evaluate to a
  real number and that was indistinguishable from a threshold it could not read
  at all. It now tells the two apart and runs. This mattered little while only
  single powers were solved; with the quadratic formula the discriminant of
  `(time()-5)^2 >= thresh` goes negative as soon as `thresh` does, so a whole
  region of parameter space is in this case and a fit could walk into it.

- **`capabilities()` answers behavioural questions, and says which build it is
  (issue #431).** The report described compiled backends and build options,
  which is the right answer to "was NFsim linked in?" and the wrong answer to
  the question a fitting frontend has to settle before it commits to hours of
  gradient work: does this build compute the thing correctly? Four new keys in
  `features` answer that one, and a new top-level `build` block says which build
  is answering.

  A version string could not stand in. bngsim bumps `__version__` at the
  **start** of a release cycle, so the string identifies a cycle rather than a
  build, and every from-source build made between the bump and a given fix
  declares the same number as the release that finally carries it. Nor could a
  `hasattr` probe: these fixes change what a build *computes*, not what it
  *exposes*, so nothing in the namespace appears or disappears at any of them.
  Downstream, PyBNF was reading `features["effective_ic_sensitivity"]` as a
  **witness** for the event fixes — a key about initial conditions, usable only
  because issue #155 landed a few commits after them (lanl/PyBNF#605). That is a
  fact about commit ordering, not about semantics, and it stops being evidence
  the moment the two are decoupled, silently.

  What makes this expensive rather than untidy is that the two wrong answers are
  not symmetric. A build without one of these fixes does not *refuse* the case
  it cannot handle — it returns a finite tensor with a term missing. A consumer
  that guesses "absent" loses a gradient fit; one that guesses "present" runs to
  completion, converges, and reports a number with nothing wrong on its face. On
  0.12.1 a state-reading event assignment reported `-10.96` where the model's own
  central difference says `-311.20`.

  The keys, all four published on every build so that a `False` is an answer
  rather than a silence (an absent key means only "too old to have been asked"):

  | key | claims | issue |
  |---|---|---|
  | `event_sensitivities` | forward sensitivities survive a discrete event, carrying a state-reading assignment's `∂h/∂x·s⁻` and the sensitivity history across a root that fires nothing | #144, #146 |
  | `cross_compartment_sensitivities` | a reaction whose species live in compartments of different size keeps the analytic `∂f/∂p` instead of putting every column of the model on difference quotients | #160 |
  | `per_species_atol` | `Simulator.run(atol=...)` takes a vector | #196 |
  | `tracking_atol` | `Simulator.run(atol=TrackingAtol(...))` is honoured | #213 |

  Each probe reads the half of the install that can actually be wrong. Three ask
  the loaded extension for a binding the fix added, because two of these fixes
  are half C++ and in a source checkout the extension is built separately and
  does not rebuild on import (issue #23); the fourth reads
  `BNGSIM_NO_FUNCTIONAL_SENS_RHS`, the A/B hatch that is the only way to turn its
  behaviour off. `python/tests/test_behaviour_capability_keys.py` measures what
  each key claims — a closed form across the event, the emitted source for the
  cross-compartment model, the analytical solution for both tolerances — and
  asserts the key against the measurement, so a key cannot go on being published
  after the behaviour it names has gone.

  `capabilities()["build"]` is `{"commit": ..., "stale": ...}`: the commit CMake
  baked into the extension (`None` when it was built outside a git checkout), and
  whether that extension is older than the C++ next to it. Two installs
  declaring one `version` are distinguishable by the commit and by nothing else
  in the public API. The staleness bit is here because an install reporting
  `0.12.2` was found whose compiled core predated its own `.cpp` by three days:
  every version-, metadata- and feature-based check passes there, because nothing
  in the Python layer moved. bngsim already warns about it at import, which for a
  consumer package is while that package is still loading, before its logging is
  configured — so the same signal is now readable at a moment of the consumer's
  choosing, without importing a private module. Both come from
  `bngsim._build_provenance.summary()`, which is new and public within that
  module, and both honour `BNGSIM_NO_BUILD_CHECK` like every other reader there.

  Also fixed while making the contract true in both directions: **`missing`
  never explained `mir`**. It is `False` on every default build (an off-by-default
  prototype), so a caller doing the documented thing — read `features[name]`, and
  on `False` print `missing[name]` — got a `KeyError` instead of a sentence. It
  now names `-DBNGSIM_ENABLE_MIR=ON` and says that nothing needs it, and a test
  asserts the two directions symmetrically: every unavailable feature is
  explained, and every explanation belongs to an unavailable feature.

### Changed

- **The manuscript's eight named BNGL models are generated from their curated
  `BNGL-Models` records, once, and every suite that runs them reads the same
  artifact (issue #423).** `suites/ssa_table5` (Table 5, exact Gillespie SSA),
  `suites/psa` (Table 7, partial-scaling approximation) and `suites/ssa`
  (cross-engine correctness) each vendored their own pre-generated `.net` files.
  `tcr_signaling` existed as **three** copies, and the three sets predated
  `wshlavacek/bngsim-paper#6`'s re-pointing of the manuscript's named models at
  the house-curated collection. A benchmark re-run would have faithfully
  re-measured superseded networks and left the manuscript citing files nothing
  had simulated.

  `benchmarks/models/bngl/curated/` now holds the eight records verbatim,
  `benchmarks/models/net/curated/` the networks generated from them, and
  `benchmarks/models/regenerate_curated_nets.py` generates and verifies both
  against `curated_nets.json`, which pins every upstream and artifact sha256.
  `--check` regenerates into a temp directory and diffs, so a stale artifact
  fails rather than being re-measured. Generation strips every action *except*
  `generate_network`, so a record's own protocol never runs — the horizons stay
  the manuscript's, which several records disagree with — and keeps
  `generate_network`'s options, which is what decides the network. All eight
  are read by `suites/ssa_table5`, five of them by `suites/ssa`, three by
  `suites/psa`; the byte-identical `models/{bngl,net}/{psa,ssa}` duplicates are
  removed, and `models/bngl/ssa/` is down to the seven models with no curated
  record.

  Three networks moved, and the manuscript re-measures those rows:

  | model | was | now | why |
  |---|---|---|---|
  | `prion_aggregation` | 104/2809 | **121/3843** | the record raises `max_iter=>150` over BNG's 100-iteration default, so chains reach its own `max_stoich=>{PrP=>120}` cap; the 17 added species are zero-population chain tails, so the event count holds (~605 k at `t_end=10`) and only per-event cost rises (~20 %) |
  | `samoilov_futile_cycle` | 6/6 | **7/10** | the record is the primary file, external noise driver included |
  | `gene_expr_3stage` | 6/6 | **4/6** | the superseded copy carried a `Src()` marker and a `$Null()` sink the record does not; dynamics unchanged |

  `tcr_signaling` keeps its 37/97 network but starts from the paper's primed
  state (~3 % more events); `erk_activation` is a pure relabelling, identical to
  the event; `gene_expression` and `mckane_predator_prey` are unchanged.

  `gene_bursts` keeps its `Protein=467` seed, but *derived* rather than
  hand-baked: a model may declare a `relax` step, which generation runs against
  the record before writing the artifact. It is the one thing generation adds
  beyond `generate_network`, and two rules keep it from becoming a place to hide
  hand-tuning. The horizon must be **converged**, so the seed is a steady state
  — a property of the model — not a point on a transient; `gene_bursts` gives
  the identical state at 3.6e5, 3.6e6 and 3.6e7 s, where the record's own
  3.6e4 s stops at `Protein=111.7`, still climbing. And the seeds are **rounded
  to whole molecules**, because a fractional molecule count is ill-posed for a
  discrete solver and the engines disagree about it: bngsim and `run_network`
  round, but RoadRunner's gillespie takes 0.389 literally and walks the species
  negative — the signature that got Smith2013/474 dropped from the corpus.

  The row measures identically either way (median 923 / mean 930 events at
  `t_end=3600` over 200 seeds, for this `.net` and the superseded one alike), so
  the manuscript's B07 number does not move. Without the relaxation it would
  have: `t_end=3600` is one cell cycle, and from the bare `0/0` seeds the model
  sits in its basal regime — median 95 events, with **25 of 200 replicates
  firing nothing**. B07's horizon and its initial state were a matched pair in
  the superseded actions block (`simulate ode t_end=360000` immediately followed
  by `simulate ssa t_end=3600`), and the manuscript kept the horizon; keeping
  half the pair is what would have made the row arbitrary.

  One coverage cell moved with them: the curated Samoilov record's driver step
  `N + N -> E+ + N` is a repeated reactant, whose converted SBML law `k*N*N` is
  not the exact propensity `k*N*(N-1)`, so **`samoilov_futile_cycle`/RoadRunner
  is now N/A** (COPASI derives the combinatorial propensity itself and its cell
  stands). `convert_all.py` now checks every conversion verdict it computes
  against the coverage table the orchestrator obeys and exits non-zero if they
  disagree, so that class of drift cannot go unnoticed again.

- **Table 5's `samoilov_futile_cycle` row runs the curated record's own horizon,
  `t_end=10` (issue #425).** Re-pointing the row above changed its model and left
  its horizon alone, so it ran a 7/10 model at `t_end=0.0018` — a value chosen for
  the 6/6 artifact that had just been replaced. That value does not come from
  Samoilov et al. (2005) at all: it is the `@SIM` annotation of
  `benchmarks/models/antimony/ssys/Samoilov2005.ant`, the file the superseded
  artifact was converted from, which is a deterministic ODE encoding kept in a
  different corpus as a stiffness pathology case. The record's own protocol runs
  to `t_end=10` sampled every 0.005 s, reproducing Fig. 3A, so the model and the
  horizon now come from the same file.

  Two measurements decided it. The model has not started at 0.0018 s: the
  trajectory is still in the burn-in from the paper's initial condition, with `X*`
  down only from 2000 to about 1600 molecules against the record's own operating
  band of 110–286, which it first enters at a median 0.0167 s over 30 seeds. And
  the cell was timing setup rather than simulation: interleaved against a run of
  the same model that fires zero events, 91 % of the bngsim wall and 89 % of the
  `run_network` wall was per-run fixed overhead, which makes it a poor row in a
  cost table however the modelling question is settled.

  The short horizon was never a cost concession. At `t_end=10` a replicate fires a
  median 1.36e7 events for 0.46 s of bngsim wall and 1.70 s of `run_network` wall,
  both far inside the harness's 120 s per-run cap and in the range of the other
  rows. **This one does move a published number** — B10's cost rises by about four
  orders of magnitude — where the `gene_bursts` fix above moved none. Coverage is
  unaffected: the RoadRunner cell is N/A because of the repeated reactant at any
  horizon, and the record's `_unordered_pair` variant does not rescue it, since it
  writes the same reaction with the rate constant halved.

  So that a horizon cannot outlive its model again, every BNGL row in
  `corpus.json` now carries `record_horizon`, the `t_end` of the record's own
  exact-SSA action. One test reads that value back out of the record and fails if
  the corpus disagrees; another requires a row running a different horizon to name
  the record's value in its caveats. Writing the guard turned up two divergences
  nobody had written down: `gene_expr_3stage` runs 2e8 where its record runs 2.1e8
  (harmless — the record discards its first 1e7 s as burn-in, so both measure the
  same window), and `prion_aggregation` runs 300 where its record runs 10, with
  both files crediting Lin et al. (2019) for their value. The paper settles that
  one (issue #429): Fig. 7 runs the prion model from 0 to 300 days, so `t_end=300`
  is the published benchmark horizon, the model's time unit is days, and the
  record's ten-day run is its own choice, which its protocol note used to credit to
  Lin. `wshlavacek/BNGL-Models#45` corrected that note and the vendored copy here
  is the corrected file, so the collection pin moves to `a158912`. **B14's cost
  does not move**, and neither does the generated network: the upstream fix was
  comments only, so every `net_sha256` is unchanged and only `source_sha256`
  moved. Both divergences now say so in the corpus.

- **`ssa_table5`'s `corpus.json` is the corpus SSOT.** Artifacts, horizons and
  output-point counts were typed out in both `corpus.json` and `_ssa_config.py`;
  the corpus is not on the timing path, so a horizon edited in one and not the
  other would have stayed invisible until the manuscript quoted it.
  `_ssa_config.MODELS` is derived from the corpus and keeps only runner policy
  (warm-N, cheap→expensive order, coverage). The psa suite's `Nc` sweep is
  likewise declared once, in `run.POPLEVELS` — its README and emitter documented
  a 4-value sweep while the runner swept 5.

### Fixed

- **`test_neither_suite_vendors_a_net_of_its_own` no longer fails on a machine
  that has run the other benchmark suites.** It searched every suite for a
  vendored `.net`, so it tripped over the generated networks under
  `suites/ode_fullnet/nets/` and `suites/ode_engines_s3/sbml/`, which are
  gitignored build products. It passed only in CI, where those files do not
  exist. It is now scoped to `ssa_table5` and `psa`, the two suites its own
  docstring is about.

- **A `TotalRate` rule with a symmetric reaction center no longer runs at a
  fraction of the rate the model asks for (issue #426).** `TotalRate` means the
  rate law gives the whole propensity of a rule. The reaction center symmetry
  factor exists to correct a counting problem — a reactant pattern with a
  non-trivial automorphism matches the same reaction more than once — so where
  the rate is stated outright there is no count to correct and the factor must
  not be applied. NFsim applied it anyway, halving the propensity on a homodimer.
  On the issue's reproducer (a homodimer stating `k=0.02` over `t=1000` from 400
  free monomers, so 20 firings consuming two monomers each leave 360), bngsim
  ended at 383.4 where released NFsim v1.14.3 ends at 360.3; it now ends at
  361.6. bngsim's own RuleMonkey backend was already correct at 359.8, so the
  two network-free engines had been disagreeing by 2x on this shape.

  The trigger is the intersection of two features, which is why it survived.
  BNG2.pl forces every `TotalRate` rate law into a `Function` even when it is a
  bare constant (`RateLaw.pm`: `my $force_fcn = $totalRate ? 1 : 0;`) and rejects
  `TotalRate` on Sat/MM/Hill, Arrhenius, and local functions — so a
  BNG-generated `TotalRate` rule always lands in `FunctionalRxnClass` and never
  in the `Ele`/`setBaseRate()` path. That is also the class issue #195 taught to
  scale by `baseRate`, which for it carries the symmetry factor and nothing
  else. Nothing in the corpus sits in that cell: of the 18 `parity_checks` models
  using `TotalRate`, 172 rules carry `symmetry_factor="1"` and 4 carry `"0.5"`,
  and not one of the four is a `TotalRate` rule. The issue #195 fixture is
  entirely `totalrate="0"`. And BNG2.pl does not implement `TotalRate` for network
  simulations (`RxnRule.pm:27`: "TODO: implement TotalRate feature for Network
  simulations") — it is the generator the parity harness runs, so no ODE or SSA
  reference trajectory exists for a parity check to compare against. (The newer
  C++ BioNetGen does honor the rule-level modifier, in `PsaSimulator`, so this is
  a limitation of the Perl reference rather than of BioNetGen as a whole.)

  The guard sits at propensity-evaluation time rather than where the factor is
  folded into `baseRate`: `NFinput` calls `setTotalRateFlag()` after both the
  `ReactionClass` constructor and `setBaseRate()`, so `totalRateFlag` is still
  false at both of those points and a guard placed there would never fire.
  `BasicRxnClass` is deliberately left alone — it reads the folded `baseRate` in
  its own `TotalRate` branch, but BNG cannot emit `Ele` with `totalrate="1"`, and
  for hand-written XML that did, matching released NFsim v1.14.3 is worth more
  than the unreachable correction. `MMRxnClass` and `DORRxnClass` ignore
  `totalRateFlag` entirely on their non-RuleMonkey paths, which is a separate
  upstream gap that BNG's own rejection of those combinations keeps unreachable.

  Carried as `bngsim/carry-total-rate-skips-symmetry-factor` (queue 13 → 14).
  This one is upstream-bound rather than a permanent local divergence: the
  defect reached upstream `master` through RuleWorld/nfsim #89, which absorbed
  the issue #195 carry, so upstream NFsim has it too.

  `tests/data/nfsim/symmetry_factor_total_rate.{bngl,xml}` and
  `TestTotalRateIgnoresTheSymmetryFactor` guard the `TotalRate` row of the 2x2;
  `symmetry_factor_rate_laws.xml` is left untouched so that a "fix" which simply
  stopped applying the factor everywhere fails there rather than passing here.
  The new fixture puts a symmetric and an asymmetric `TotalRate` pool on the
  same expected survivor count (2000 of 4000): before the fix the symmetric pool
  ended at 2999.0 and the asymmetric control at 2004.5, after it 1995.3 and
  1999.6. Both of the pre-existing controls — an asymmetric `TotalRate` rule and
  a symmetric elementary-rate rule — are byte-identical across the change
  (380.73 and 286.10 over 60 seeds), so nothing outside the `TotalRate` path
  moved.

- **The `ssa_table5` suite no longer needs one particular developer's machine
  (issue #423).** `_ssa_config.RUN_NETWORK_BIN` was
  `/Users/wish/Simulations/.../run_network`, an absolute path into a home
  directory no other machine has; it now resolves `$RUN_NETWORK`, then
  `$BNGPATH/bin/run_network`, then the canonical
  `~/Simulations/BioNetGen-2.9.3` — the same convention as every other suite.
  `TIMING_HARNESS.md` documented an absolute `VENV=` for the same reason.

- **`emit_ssa_table.py` renders a partial result set.** It walks all 14×4 cells
  and indexed `r["model"]` on the `{}` it gets for a cell the results file has no
  record of, so it raised `KeyError` after any `--only` / `--engines` run or
  interrupted sweep — exactly the runs whose output you want to look at. Those
  cells now render as `missing`.

## [0.14.0] - 2026-08-18

### Added

- **A rate-law switch condition that is a single power of the clock is now
  compensated, so its forward-sensitivity run is no longer refused (issue #418).**
  `if(time()*time() >= thresh, ...)` — the shape issue #414 refused, since neither
  issue #48's affine solver nor issue #150's state root brackets it — has a
  crossing in closed form: `c·clock^n` is strictly monotonic on `clock ≥ 0`, so it
  crosses exactly once, at `clock = (thresh/c)^(1/n)`. `_clock_monomial_threshold`
  solves that (`time = sqrt(thresh)` for the quadratic), and because the result is
  the clock **value** at the crossing — the same `(clock, threshold_expr)` contract
  `_clock_affine_threshold` (issue #355) already returns — the whole issue #48
  machinery jumps it unchanged: `∂t*/∂thresh = 1/(2·sqrt(thresh))`, the crossing
  registered as a stop time, the in-branch `∂f/∂p` a clean Piecewise zero. The
  detector and the issue #68 gate share the one recognizer, so a model that was
  declined for this shape is now admitted on both. This is the first step of the
  machinery half of issue #414; a clock threshold that is not a bare power
  (`(time-5)^2`, two crossings; `time^2 + time`, mixed) has no single crossing to
  name from the text alone and stays refused, and the difference-quotient
  reproducer that #415 exercises under the MIR JIT now runs the analytic path with
  its self-multiply in the *sensitivity* RHS as well as the state RHS.

- **A codegen artifact says which codegen key built it, so a cache sweep can tell
  live from orphaned (issue #363).** Artifacts are now named
  `rhs_<key>_<hash><suffix>` — e.g. `rhs_28+317a5b34d5dc9959_9e1f…` — where `<key>`
  is `_CODEGEN_CACHE_KEY`, the `_CODEGEN_VERSION` constant plus a digest of the
  emitters' own source (#51). The key is still mixed *into* `<hash>`, so nothing
  about invalidation changes; carrying it beside the hash is what makes the dead
  corpus **countable**.

  That matters because #51's key is deliberately conservative: any edit to
  `_codegen.py` / `_jacobian.py` / `_saturable_jacobian.py` / `_switch_sensitivity.py`
  — a comment included — orphans every artifact on the machine at once. So on a
  machine that tracks bngsim development, "everything from before my last emitter
  edit is garbage" is the common case, and until now it was unexpressible: two
  opaque hex strings, with time as the only signal a filename carried.

  - `bngsim-cache info` reports **live** and **orphaned** as two numbers, plus a
    per-key table — the audit of a shared or pre-warmed artifact directory, which
    says which bngsim's artifacts are in it and how much each holds.
  - `bngsim-cache prune --orphaned` sweeps exactly the artifacts this install can
    never load again. Much better targeted than `--older-than`, which keeps orphans
    that happen to be recent and evicts live artifacts that are merely idle. The
    `--min-age` floor still applies: a *fresh* orphan is a sibling process's compile
    under its own key, about to be `dlopen`ed.
  - `--keep-key KEY` (repeatable) spares another install's artifacts, because a venv
    per project is ordinary and each has its own key — "orphaned" from one venv's
    point of view is live from another's, and sweeping on that would make a shared
    cache thrash. `prune_codegen_cache(orphaned=True, keep_keys=[…])` is the API
    form; passing `keep_keys` without `orphaned` raises rather than reading as a
    protection the age and size bounds do not honor.
  - `CacheEntry.codegen_key`, `CacheInfo.live` / `.orphaned` / `.by_key`, and
    `bngsim.cache.artifact_key()` expose the same thing in process.

  **This is a one-time full invalidation.** Every artifact already on disk is under
  the old scheme and unreachable under the new one, so every model recompiles once.
  `info` counts those pre-#363 names as orphaned (nothing will ever look one up
  again) and lists them under `-`; `prune --orphaned --keep-key -` spares them, for
  a directory shared with an install too old to write a key. No `_CODEGEN_VERSION`
  bump: this edits `_codegen.py`, so #51's source digest invalidates every cache on
  every machine anyway — which is the argument for landing the rename together with
  it, since the cost is paid once either way.

- **The codegen artifact cache is inspectable and prunable: `bngsim-cache`
  (issue #205).** `~/.cache/bngsim/codegen` grew without bound and there was no
  supported way to look at it — 2.0 GB across 14,377 entries after six weeks of
  ordinary work on one developer machine, with `rm -rf` on a path you had to know
  by heart as the only remedy. New console script `bngsim-cache` (also
  `python -m bngsim.cache`, which needs no reinstall) with four verbs:

  - `info` — path, entry count, total size, build/last-used dates, and a
    breakdown by artifact kind (model RHS, SSA propensity, the `src_` fallback
    key of #174, plus the leaked partials below).
  - `clean` — remove *only* the debris of interrupted compiles: `bngsim_shard_*`
    scratch directories, the stray `rhs_<key>_<hash>.<pid>_<n>.c` beside them, and
    temp/sidecar libraries. Nothing else cleans these up, and no compiled
    artifact is touched, so no cache hit is lost.
  - `prune --older-than 30d` / `--max-size 2G` — evict least-recently-used
    artifacts to fit an age and/or size bound. The partial sweep runs first, so
    the size cap is a bound on the whole directory.
  - `clear` — everything, behind a confirmation prompt (`--yes` for scripts;
    required, not assumed, when stdin is not a terminal).

  `--dry-run` on every mutating verb, `--json` on `info`, and `-C/--cache-dir` to
  point any of them at a directory other than the configured cache — e.g. to
  audit a pre-warmed artifact directory from a login node.
  `bngsim.codegen_cache_info()`, `clean_codegen_cache()`,
  `prune_codegen_cache()` and `clear_codegen_cache()` are the same four in
  process, for a notebook or a fitting harness.

  **Nothing prunes automatically, deliberately.** No size cap at `compile_rhs`
  time, no sweep on import or on `Simulator` construction. A library that deletes
  files as a side effect of being used is a surprise, and the failure mode —
  evicting the artifact another process is about to `dlopen` — is the exact class
  of bug the cache exists to avoid.

  Two safety properties hold across every verb. **Only files bngsim wrote are
  ever removed**: `BNGSIM_CODEGEN_CACHE_DIR` is a user-supplied path that people
  point at shared scratch, so anything unrecognized is classified `foreign`,
  reported, and left alone — by `clear` as much as by `clean`. And **nothing
  recently touched is removed**: a compile in flight writes its scratch files
  into this very directory, so every verb holds off on entries used or written
  within `--min-age` (default `1h`, over the 600 s default `BNGSIM_CODEGEN_TIMEOUT`) and
  on POSIX additionally holds a partial whose compile is still running.

  LRU order is `max(atime, mtime)`, which was measured rather than assumed: a
  plain `read()` on macOS APFS leaves `atime` untouched while `dlopen` — the only
  way this cache is ever used — advances it, and a `noatime` mount never moves it
  at all. Taking the newer of the two gives true recency wherever the filesystem
  records it and degrades to build order where it does not, and `info` reports
  which one you are getting.

  Not addressed here, and filed rather than guessed at: an entry carried no
  record of the codegen key that built it, so nothing could report how much of the
  cache was *orphaned* (#363, above, which put the key in the filename); and MSVC
  leaked a `.lib`/`.exp` pair per successful Windows compile (#362, whose fix rode
  along with an invalidation this release was paying anyway — `clean` collects any
  debris left from before it).

- **BNGL models load, behind a `bngl` extra (issue #162).**
  `Model.from_bngl("m.bngl")` and `Model.load("m.bngl")` now work;
  `_LOAD_DISPATCH` gained `.bngl` and the "expand it yourself first" refusal is
  gone. bngsim still has no BNGL parser and does not want one — BNGL describes
  *rules*, and turning them into a network is `BNG2.pl generate_network`'s job —
  so the loader shells out and reads the emitted `.net` through
  `Model.from_net`. `pip install 'bngsim[bngl]'` supplies BNG2.pl via
  PyBioNetGen; `$BNG2_PL` / `$BNGPATH` / a `BNG2.pl` on `PATH` all outrank it,
  and `perl` has to be there too (so BNGL loading reports itself unavailable on
  stock Windows rather than failing at load time).

  **The file's experiment is not executed — and only the experiment is
  stripped.** A `.bngl` in the wild ends in `simulate({...})` or
  `parameter_scan({...})`, and BNG2.pl runs whatever it is handed, so loading a
  model would have meant running the author's whole experiment to obtain a
  network `generate_network` alone produces in seconds. But dropping *every*
  action is the opposite mistake, and the first cut here made it: an action-scope
  `setOption("NumberPerQuantityUnit",6.0221e23)` above the model block
  configures the generation, and without it BNG2.pl emitted the same topology
  with every bimolecular rate constant off by that factor — `1e12` where the
  reference network says `1.66e-12`, caught by diffing
  `benchmarks/models/bngl/ode/catalysis.bngl` against a direct BNG2.pl run.
  What is dropped is now exactly the verbs that would run the experiment, write
  an artifact (a stray `writeNET` would land on the network being read), or stop
  BNG2.pl early — plus everything after the source's own `generate_network`,
  which is where the protocol begins. That call's `max_iter` / `max_agg` /
  `max_stoich` are forwarded: they are what make an unbounded rule set finite, so
  regenerating with BNG2.pl's defaults would silently give a *different model* (or
  never return). `from_bngl(..., protocol=True)` returns `(model, ProtocolSpec)`
  so the experiment is recovered rather than discarded.

  Verified by diffing against direct `BNG2.pl` runs over fifteen corpus models:
  thirteen are byte-identical. The other two differ only because the *reference*
  is contaminated by its own protocol — `toy-jim.bngl` equilibrates with
  `simulate({...steady_state=>1})`, and BNG2.pl rewrites the `.net` as it goes,
  leaving evaluated numbers where the model declared `R_tot`. Loading through
  `from_bngl` keeps the declared symbolic seeds, so `set_param("R_tot", ...)`
  still reaches the initial condition. `ode/fceri_gamma` and
  `ode/prion_aggregation` reproduce the species/reaction counts recorded in their
  own header comments, which is what pins the option forwarding.

  **Generated networks are cached**, under `~/.cache/bngsim/networks`
  (`$BNGSIM_BNGL_CACHE_DIR`), keyed on a digest of the flattened model text
  *and* the BNG2.pl that produced it — so an edit or a BioNetGen upgrade
  regenerates, and an actions-only edit correctly reuses. That is not only
  speed: `Model.from_net` records the path in `_net_path`, and codegen prefers
  that file precisely because a BNG2.pl network carries derived rate-constant
  parameters (`_rateLaw{N}`) whose chain rules the model-based path does not
  reconstruct (issue #15). A scratch directory deleted on the way out would have
  left every `from_bngl` model with a dangling path, failing at simulate time on
  the models that need the `.net` route most. An unwritable cache degrades to a
  per-process directory rather than failing the load.

  Compartmental (cBNGL) input loads — BNG2.pl bakes the volumes into the
  generated rate constants exactly as for a hand-generated `.net`, which is why
  `compartment_sizes=` is refused for `.bngl` as it already was for `.net`.
  Verified across `benchmarks/models/bngl/`; the models that do not load are the
  network-free ones, which time out with a message naming the remedy.

  `bngsim.HAS_BNGL` and `capabilities()["features"]["bngl"]` report it. Unlike
  its neighbours `HAS_BNGL` is a **runtime probe, not an import check** — BNG2.pl
  can arrive from an env var, and `bionetgen`-importable-but-no-`perl` cannot
  load BNGL at all — and it is a lazy module attribute, so `import bngsim` never
  pulls in `bionetgen` (a 12.8 MB package that brings libroadrunner with it).
  `capabilities()["missing"]["bngl"]` carries the resolver's full trail.

  The extra is deliberately **not** a base dependency and **not** folded into
  `dev`: promoting `bionetgen` would put libroadrunner, seaborn and networkx
  into every install and add a Perl runtime requirement, and `dev` carrying it
  would hand the resolver both a version range and the `parity` group's git pin
  to reconcile. A test asserts the base install stays clear of both.

### Changed

- **Four NFsim carries are gone: RuleWorld/nfsim#89 merged and the vendored tree
  now takes the fixes from upstream (issue #410).** The carry queue is 13 topics,
  down from 17, and `third_party/nfsim` is rebuilt on upstream `master`
  `5962ea9`. The four that left are the ones bngsim contributed upstream:
  product molecularity tested against all the bonds a rule deletes at once,
  complex tracking turned on by a Species observable independently of `-bscb`,
  the reaction center symmetry factor applied on every rate law rather than only
  `Ele`, and a pure context reactant counted once per complex. `git am` of the
  remaining 13 patches replays onto the new base without a conflict, and
  `python/tests/test_nfsim_symmetry_factor.py`,
  `python/tests/test_nfsim_molecularity.py` and
  `python/tests/test_nfsim_context_symmetry.py` — all written against the
  carries — pass unchanged against the carry-free tree, which is what shows the
  upstream versions behave the same.

  Two things came back different from what left. Upstream's pure-context commit
  is a superset of the carry it replaces: it also fixes `exactRuleMonkey_a()`,
  which the carry skipped because the vendor export trims `src/NFsim.cpp` and
  the RuleMonkey-exact path is unreachable here, and it drops a `ReactantList`
  version counter the carry declared and bumped but never read. And the refresh
  picks up upstream #88, which rewrites the free-substrate root in the MM rate
  law to stop it cancelling to zero when `Km` is small and the enzyme is in
  excess — unrelated to the prune, but it lands in the same file the symmetry
  fix does, since upstream wrote the symmetry commit on top of it.

- **The BNG2.pl resolver is now shipped, as `bngsim._bngpath` (issue #162).**
  It was `parity_checks/_core/bngpath.py`, which is developer-only and not
  packaged — and `bngsim.convert._bng2.find_bng2` had already grown a seventh,
  weaker copy of the lookup the module exists to prevent duplicating: `$BNGPATH`
  and `$PATH` only, no `$BNG2_PL`, no bundled PyBioNetGen, no record of what was
  tried. So a machine carrying either of those looked BNG-less to the cBNGL
  round-trip gate while the parity suite found BNG2.pl fine. Both now resolve
  through one module, `_core.bngpath` re-exports from it (a test asserts object
  identity, not equal behavior), and the promotion is a strict superset:
  `BNG2.pl` on `$PATH` joined the precedence order between the env vars and the
  bundled copy. `require_bng`, which `sys.exit`s, stays in `parity_checks/` —
  a sweep-entrypoint concern with no business in a library.

- **The sensitivity tolerance shape stays as it is, and the measurement that
  settles it is now on the record (issue #354).** No behavior change:
  `atolS[iS][i] = atol*scale[i]/pbar[iS]` is unchanged, both #352 hatches keep
  their shipped defaults, and this is comments plus tests.

  The alternative — drop the `/pbar` divisor, which is what AMICI's flat `atol`
  amounts to — was swept over the full 1323-model corpus x 2 corrector methods,
  one variable, both arms from one binary via `BNGSIM_SENS_PBAR=unit`. It lost
  on every axis: **rescued 2 rows and broke 8** (every moved row re-verified
  serially, which mattered — a third apparent rescue was a worker that died in
  the arm and did not reproduce); **identical `max_rel_err` against the same
  AMICI oracle on all 1946 rows that passed both ways**, so it buys no accuracy
  anywhere; and a load-independent step count over 452 models of median 1.032x,
  p90 1.407x.

  The mechanism is that **`pbar` has two consumers and the hatch moves both**.
  Besides dividing `atolS` it is handed to `CVodeSetSensParams`, where CVODES
  uses it as the perturbation scale of its internal difference quotient — live
  on the 117 of 1323 corpus models with no analytic sensitivity RHS
  (structurally: `floor()`, `abs()`, `ceil()`, a non-emittable functional
  species-derivative). Six of the eight broken rows are those models. Pinned to
  1.0, a probe of `sqrt(eps)*|p|` becomes `sqrt(eps)`: measured against the
  closed form on a declined one-reaction model, **124x worse at `k=1e6` and
  9176x at `k=1e8`**, and exact parity at `k=1` where `pbar` is 1.0 anyway. Three
  new tests in `test_sensitivity_tolerance_hatches.py` pin that, including a
  positive control that the model under test really is on the
  difference-quotient path.

  The framing that motivated the change does not survive either.
  `CVodeSensEEtolerances` divides by `pbar` itself — in SUNDIALS' words, "the
  scaled sensitivity `pbar_i*yS_i` has the same error weight vector calculation
  as the solution vector" — so the `/pbar` half *is* CVODES' own rule and
  `scale[i]` (#214) is what bngsim adds. AMICI reaches the same place from the
  other side, putting parameters on a log scale and applying the chain-rule
  factor `p` itself (`amici/src/model.cpp:1005`); the parity harness has to pin
  that off to make the two tensors commensurable, which is what creates the
  apparent asymmetry. "Match AMICI's shape" would mean adopting a setting AMICI
  only runs when something else carries the scaling.

- **`amici_parity` budgets the coupled system rather than capping parameters
  (issue #331).** Forward sensitivity integrates `n_species*(Np+1)` states, so a
  job's cost is set by that product, not by `Np` alone. The old flat cap of 20
  spent the same columns on a 3-species toy and a 1604-species model, leaving
  **534 of 1115 models (48%) compared on a 20-column sample**. `--param-budget`
  (default 20,000 coupled states) derives each model's parameter count from its
  species count instead; `--param-cap` remains as an optional extra ceiling, and
  `--param-cap 20` reproduces the old behavior.

  Measured over the corpus: **441 models get more parameters, 525 are unchanged,
  and exactly one gets fewer** (`MODEL1009150002`, 20 → 12), for 3.66x the solve
  work — which is ~+5% of wall, since the solve is only 1.8% of a sweep dominated
  by timeouts (37%) and AMICI compiles (37%).

  Uncapped is deliberately not the default. `MODEL1009150002` currently passes
  with a 14.4 s warm `simultaneous` solve at `Np=20`; at its full 7304 parameters
  that extrapolates to roughly **87 minutes**, so removing the bound would convert
  three passing models into TIMEOUTs. Memory is not the constraint (~0.6 GB worst
  case) — wall-clock is.

- **`amici_parity` records degeneracy witnesses per row (issue #328).** A job
  could be reported `PASS` when the whole forward-sensitivity tensor lay below
  the magnitude either solver can resolve — nothing was meaningfully compared,
  but the row was indistinguishable from a real pass. Each row now carries
  `max_abs_sx`, `n_resolvable_params` (parameter columns whose peak exceeds
  **their own** floor) and `state_span`, and the comment says `DEGENERATE` when
  no column is resolvable, so the census is a query over the report rather than
  a corpus re-run.

  The obvious proxy does not work and is deliberately not used:
  `n_below_noise_floor / n_cells` (issue #316's renaming of what shipped as
  `n_noise_forgiven`) reaches 100% for *excellent agreement* too, because the
  mask keys on `|bn − am|` rather than on magnitude — `MODEL7909395757` has
  3636/3636 cells inside the floor at `max|sx| = 0.6`. Magnitude has to be
  recorded directly.

  The floor is applied **per parameter column**. Reducing it with `max` over the
  parameter axis lets one tiny-valued parameter inflate the threshold for the
  entire tensor and marks a live model degenerate — `BIOMD0000000002` has real
  dynamics and 11 of 20 resolvable columns, but a global floor declared it
  wholly unresolvable. That is the same global-reduction mistake as issue #322's
  transversality floor.

- **`bngsim.SensitivityUnsupportedError` — a clean forward-sensitivity refusal is
  now distinguishable by type.** bngsim has two constructs it *declares* it
  cannot differentiate and raises on rather than answer wrongly: an event whose
  crossing time moves with a requested parameter in a way `dt*/dp` cannot be
  computed for (GH #205), and a rate law codegen cannot differentiate to closed
  form — a non-smooth `min`/`max`/`abs`/`floor`, `rateOf()` in a rate law, an
  unparseable expression (GH #214). Both raised a bare `ValueError`, which made a
  documented capability gap indistinguishable from a bug without matching on
  message text.

  Both now raise `SensitivityUnsupportedError`, the forward-sensitivity peer of
  `SsaValidationError`. A fitting driver can catch it to fall back on a
  derivative-free optimizer; the `amici_parity` sensitivity sweep buckets it as
  `UNSUPPORTED` (non-scoring) instead of `EXCEPTION` (an actionable bngsim bug),
  matching it by type rather than by message prefix so rewording a refusal
  cannot silently sink it back. `BIOMD0000000342` is the corpus model that was
  diluting `EXCEPTION` this way.

  **Not a break.** The class inherits both `BngsimError` and `ValueError`, so
  every existing `except ValueError` handler keeps working. Environment failures
  are deliberately excluded: "no C compiler and no JIT available" is a fixable
  local problem, not a property of the model, and stays a plain `RuntimeError`.

### Fixed

- **A rate law that squares a value no longer smashes the stack under the MIR JIT
  (issue #413).** A forward-sensitivity `run()` over `beta*I*time()*time()` on a
  `-DBNGSIM_ENABLE_MIR=ON` build died with glibc's `*** stack smashing detected
  ***` and SIGABRT on Linux — no `except` can see it, so the pytest session went
  with it — while the same commit passed on macOS x86_64, macOS arm64 and Windows.
  The defect is in MIR, not in bngsim's use of it. `MIR_gen`'s register allocator
  (`try_spilled_reg_mem`, `third_party/mir/mir-gen.c`) rewrites every operand that
  names one spilled register into a memory operand and records the rewritten
  indices in a **two**-element `op_nums[MAX_INSN_RELOAD_MEM_OPS]`, bounded upstream
  by `gen_assert` alone — and `gen_assert` is `assert`, which `-DNDEBUG` deletes
  from every release build. An insn *can* name the register three times: `x = x*x`
  reaches the allocator as the three-operand `dmul r, r, r`, so the loop matches
  three times and the third store lands past the array. The three passing platforms
  were never evidence of safety; they lay that frame out without a canary in the
  way. Fixed by a carry in `mir-gen.c` that raises the table to 3 and replaces the
  vanishing assert with a real runtime check which undoes the rewrites and
  declines. Both halves are load-bearing: 3 is what the three-operand case needs,
  and the check has to stay because 3 is not a *provable* bound — the call site is
  reached for `MIR_USE`, whose `nops` is unbounded. The same assertion is reported
  upstream as `vnmakarov/mir#410`, open since 2024-07-08 and written up as a
  debug-build assertion failure, so the release-build overrun it becomes is not
  recorded there; the fix is proposed upstream as `vnmakarov/mir#468`, whose code
  is byte-identical to the carry. Two corrections to the diagnosis this shipped
  with: **sensitivity is
  not the trigger** — it is what forces codegen on, since a four-species model is
  far below the auto-codegen threshold and a plain ODE run therefore JITs nothing;
  `codegen=True` reproduces it with no sensitivities at all. And it is **not the
  `if()` / the threshold / the crossing** — the self-multiply alone is enough,
  wherever the emitted C puts it. Only `-O2`/`-O3` `MIR_gen` is affected, which is
  every source under 512 KB. `python/tests/test_codegen_jit_self_multiply.py` is
  the regression.

- **`vendor_mir.py --check` now verifies a local carry is still applied, not just
  that checksums match (issue #413).** The `mir-gen.c` fix above is the vendored
  MIR tree's first local carry, and the SHA256 anchors cannot police one: a
  refresh rewrites the file from upstream *and* re-anchors its checksum, so a
  dropped carry leaves a tree matching its own metadata perfectly. Each
  `VENDOR.json` → `local_carries` entry now names a `marker` comment the patch
  carries; `--check` and `test_mir_vendoring.py` grep for it, and `vendor_mir.py`
  exits nonzero after a refresh that loses one. The old assertion was
  `local_carries == []`.

- **A forward-sensitivity run over a rate-law crossing nothing can locate is
  refused rather than answered from the difference quotient (issue #414).** When a
  rate law branches on a condition whose crossing time moves with the trajectory —
  an equality, a comparison buried in a call argument, a clock threshold that
  reduces to neither a constant nor a single clock power — codegen declines the
  analytic sensitivity RHS. `CVodeSensInit1` takes ONE callback for every column,
  so that decline puts the **whole model** on CVODES' internal difference quotient,
  which integrates the variational equation smoothly *through* the crossing and
  drops the saltation jump `(f⁻−f⁺)·dt*/dθ` the analytic path was declined for. On
  AMICI's `nested_events` every column comes back a factor of two low after the
  crossing (issue #146). bngsim returned that gradient with only a warning; it now
  raises `SensitivityUnsupportedError`, the way it already refuses an event whose
  crossing time it cannot differentiate (GH #205) and a rate law it cannot
  differentiate at all (GH #214).

  **The refusal takes two facts together**, so neither alone can misfire. The built
  artifact must carry no `bngsim_codegen_sens_rhs` — the ground truth the C++
  resolves at run time — *and* `model_uncompensated_crossing_reason` must find a
  crossing nothing brackets. Absence alone is not a dropped jump: a *compensated*
  crossing left on the difference quotient (a `t>=sigma` clock forced there by
  `BNGSIM_NO_FUNCTIONAL_SENS_RHS`, or an `I>=thresh` state threshold) still gets
  its jump from `_apply_switch_time_sens` / `_apply_state_switch_sens` at run time,
  and an underivable-but-smooth rate law with no crossing (`erf(I)*beta*I`)
  declines the analytic RHS but drops nothing — both keep their correct difference
  quotient. A crossing alone is not enough either: if the artifact still carries
  the analytic RHS, issue #48/#150 compensated it. Only the conjunction — no
  analytic RHS **and** a crossing nothing brackets — is a gradient wrong at the
  crossing. Checking the artifact first is also what keeps the scan off the
  analytic-path majority, and what stops a spurious re-derivation from ever
  refusing a model the build actually admitted.

  The gate keys on the same `uncompensated_condition_reason` recognizer codegen
  declines with, so the build and the run-time check cannot disagree about which
  crossings are compensated. It scans **reaction rate expressions only**, inlined
  exactly as codegen inlines them: `∂f/∂p` is declined only over a rate law, so a
  condition living in an observable or expression function that no reaction uses as
  its rate law is not a rate-law crossing and does not refuse the run — its own
  output-sensitivity request is refused on its own terms (GH #198).

  **This settles the policy half of issue #414 only.** Compensating the saltation
  jump for a moving *state* crossing, the way issue #150 did for the
  single-rootable-comparison case, is the elaborate-design piece and stays open;
  the machinery half took its first step with the clock monomial of issue #418
  above, which is why `if(time()*time() >= thresh, ...)` is now admitted rather
  than refused. Detection is deliberately best-effort: if the scan itself fails,
  the pre-#414 behaviour stands — the codegen warning has already fired — rather
  than refuse a run that cannot be justified. Validate against a trajectory finite
  difference if you need an approximate gradient for a model this refuses.

- **An equality in a rate-law switch condition no longer declines a model's
  analytic sensitivity RHS (issue #381).** `MODEL2003190004`'s forward-sensitivity
  solve stalled outright — `CVODE made no progress while integrating to the next
  output point`, at t ≈ 2.82673 — for 33 of its 43 shared parameters. The plain
  ODE run is fine and AMICI produces an oracle, and the 33/10 split tracked the
  `APC` trajectory rather than any one column: none of the 10 that succeeded
  appears in `APC`'s rateRule.

  **The crossing was found; the gate was what refused it.** The model gates a
  synthesis rate on `APC <= 0.2`, but spells it as an `<or/>` of `<eq/>` and
  `<lt/>` over one pair of operands. `_split_logical_atoms` hands a disjunction
  over as its two atoms and the issue #68 gate requires *every* atom's crossing to
  be compensated. `APC<0.2` was — issue #150 roots on `APC − 0.2` and jumps the
  saltation term there — but `APC==0.2` was not, because
  `NetworkModel::state_switch` refused an equality on the grounds that a
  continuous trajectory satisfies one only on a measure-zero set. Right about the
  geometry, wrong about the consequence: it declined the analytic sensitivity RHS
  for the **whole model**, handed every column to CVODES' difference quotient —
  whose probe evaluates `f` at `y + σ·s`, which just past a crossing lands on the
  other branch — and the stall was at exactly the crossing the `<` half had
  already earned a root for.

  `state_switch` now builds `(lhs)-(rhs)` for `==`, `!=` and ExprTk's single `=`
  as well. The measure-zero argument answers a question the switch path is not
  asking: what it needs is the surface its branch can *change across*, and for
  `x == c` that is the `x − c = 0` that `x < c` names. That residual identity is
  what proves the two atoms of an `<or/>` are one crossing rather than two
  coincident ones — the pair issue #153 refuses.

  **The event path still refuses it, and that is the whole of the difference.**
  An event trigger (issue #144) fires on a rising edge, which `x == c` does not
  have — `trigger_residual_source` now takes a `ResidualUse` saying which question
  is being asked, so the one splitter can answer both without either caller
  inheriting the other's premise. Pinned from both sides, on one model.

  **A lone equality is admitted at the gate and must NOT earn a root of its own —
  a root would MAKE the branch it is admitted for not having.** This is the half
  that is easy to get backwards, and getting it backwards is a wrong trajectory
  rather than a slow one. A CVODE root does not step *over* its surface, it stops
  the integrator *on* it. On `I − thresh = 0` the equality is then true, the branch
  the exact solution never takes is live, and for `if(I==thresh, 0, beta*I)` its
  rate is zero — so `I` never leaves. The measure-zero set stops being measure-zero
  because the solver was told to land in it.

  Measured, not reasoned: with the equality registered, `ubuntu-latest` returned
  `I(t)` climbing normally and then holding exactly 4.0 = `thresh` for the rest of
  the run, against 7.389 unconditional, while macOS landed a few ulps off the
  surface and did not latch. A platform-split trajectory is the worst form this
  could take.

  So the two halves are separated. `uncompensated_condition_reason` admits an
  equality on a ground of its own — there is no branch *interval*, so there is
  nothing to compensate, which is issue #382's ground reached from the other side
  — and `state_switch_conditions` skips it, which is the one place a root is
  registered. The redundant spelling still gets its root, because the `<` half
  earns it: `MODEL2003190004` registers `APC<0.2`, exactly what it registered
  before. Both halves are pinned by tests, the structural one (`no root is
  registered`) because it holds on every platform and the numeric one (the
  trajectory matches the unconditional model, and its `beta` column matches a
  central finite difference) because it is what a caller would notice.

  **An operand that is itself a comparison is refused, and that half is new
  caution rather than new reach.** The splitter finds its operator at depth 0, so
  `(x>0) != (y>0)` — the `<xor/>` idiom — yields the residual `(x>0) - (y>0)`, a
  difference of two BOOLEANS. That is a step: its gradient is zero wherever it is
  defined, so there is nothing for CVODE to bracket and nothing for `dt*/dθ`'s
  denominator `∂g/∂y·f` to be. Admitting it would be the silent zero the #68 gate
  exists to stop, since the gate's admission *is* the promise that #150 located
  the crossing. The `<` spelling of the same shape was never safe either — it was
  saved only by sympy declining to parse `Lt(Gt(…), Gt(…))` downstream, while the
  root was registered regardless. Both are refused at the recognizer now, which
  is the one place that answers for both callers.

  The test is the operand's OWN outermost operator, never "names a comparison
  anywhere": seven corpus residuals are arithmetic over an inner `if()` whose head
  is a comparison — `(… if(PO2AMB>80, 80, PO2AMB) …) < 0` — and those are surfaces
  the trajectory really does cross. All seven still register, `MODEL0911270005`
  (issue #382's witness) among them.

  Measured over the manifest: **3 of 214 condition-carrying corpus models change
  gate verdict**, and only in the refused → admitted direction.

  | model | before | after |
  |---|---|---|
  | `MODEL2003190004` | EXCEPTION (stall) ×2 | **PASS, `max_rel_err = 0`, Np=43** ×2 |
  | `BIOMD0000000301` | PASS ×2, difference quotient | PASS, `max_rel_err = 0`, Np=33 ×2, analytic RHS |
  | `BIOMD0000000446` | declined at the condition gate | declined for an unrelated reason |

  Both corrector methods, manifest horizon, `rtol=1e-9 atol=1e-12`.
  `BIOMD0000000446` is the row worth reading carefully: its condition gate now
  passes, but `_functional_dfdp_terms` still declines the model over reaction 37,
  whose derivative w.r.t. `CReP` is not representable in C. So it stays on the
  difference quotient and only *registers* two roots it did not have. Run
  column-by-column with those roots and with them dropped, it is bit-identical
  over the full manifest horizon — one residual is bounded away from zero for the
  whole run (`min = 0.05`), and the other is zero only at `t = 0`, where SUNDIALS
  deactivates the root and reactivates it without reporting a crossing.

  ODE parity is unchanged for all of them, as it must be: state-switch roots are
  registered only on a run that asks for sensitivities.

  No `_CODEGEN_VERSION` bump. The verdict this moves is carried into the codegen
  key by `switch_gate_cache_digest`, whose per-atom row holds
  `bool(state_switch_residual(...))` — so an artifact built under the old answer
  keys differently from one built under the new.

- **A Hill ratio's *state* derivative no longer evaluates to `inf/inf` either
  (issue #402).** #393 stopped `x^n/(K^n + x^n)` overflowing when it is
  differentiated w.r.t. its exponent. It did not reach the state direction of the
  plainest Hill ratio there is, and the reason was one rewrite upstream:
  `_remove_removable_power_denominators` (#96/#351) folds the quotient rule's
  `x^(2n)/x` into `x^(2n-1)` — it has to, that quotient is `0/0` at `x = 0` — and
  `2n − 1` is no multiple of the `n` in the denominator's sum, so #393's match by
  base-and-integer-exponent-ratio does not fire. Both `x^(2n-1)` and
  `(K^n + x^n)^2` are `inf` from `x^n = 1e154` up, the same band, and this shape
  is the analytical Jacobian's own diagonal rather than a corner of it. A Hill
  exponent of 10 reaches it at `x > 1e16`, which a concentration divided by a
  small compartment volume gets to.

  One square root lower down it is **worse than a NaN**. At `x = 1e16` the
  numerator is still finite and only the squared denominator has overflowed, so
  the second term silently reads `0`, drops out of the subtraction, and the
  emitted derivative comes back `1e-15` where the truth is `1e-172` — a wrong
  number with nothing to mark it.

  The numerator is now matched a numeric offset wider, so `x^(2n-1)` is
  recognised as `(x^n)^2·x^-1`, and the leftover is placed by expanding the
  divided-through denominator into a sum of single powers:

      f^m·b^c/(a + f)^m  ==  1/Σ_j C(m,j)·a^j·b^(−c − j·e)     for f = b^e

  **Where the leftover goes is the whole design**, and the two other placements
  were measured and rejected: left standing beside the term, `x^-1·(f/(a+f))^m`
  is `inf·0` at `x = 0` — the removable `0/0` #96's fold exists to remove, handed
  straight back; spread as an `m`-th root through the factors it is `sqrt(x)`,
  `NaN` at a negative state where the `pow(x, 2n-1)` it replaces is an ordinary
  number, and a species dipping below zero mid-solve is routine. Over one grid of
  9236 evaluations the three placements broke 251, 6 and **1** previously-finite
  points respectively. Nothing was reordered: `offset == 0` rows keep the exact
  text #393 gave them, because the expansion forms an `a^m` the factored spelling
  does not.

  Carried in **both** emitters. `bngsim._saturable_jacobian` prints its own C and
  never sees a rewrite living in `sympy_to_c`; it reaches the same shortfall by
  another route, writing `n·S^(n-1)` directly rather than `n·S^n/S`. Which way
  the ratio faces decides whether a model survives to notice: an activating
  `S^n/(K^n + S^n)` has no reachable case on that path — `S^(n-1)` can only
  overflow once `S^n` has, by which time the rate law's own value is `inf/inf`
  too — while an *inhibitory* `1/(1 + (S/K)^n)` evaluates to a clean `0` at
  exactly the state where its derivative is `NaN`. Without the fix that model
  does not return a NaN: CVODES refuses the run at the first call of the
  sensitivity RHS (#395), `CV_FIRST_SRHSFUNC_ERR` at `t = 0`, on both paths.

  **Corpus A/B.** All 1291 loadable `rr_parity` models, every rate law
  differentiated w.r.t. every free symbol, emitted through `sympy_to_c` *and*
  `sympy_to_exprtk` *and* the native emitter, both arms in one process with the
  pre-#402 matcher patched back in for the `before` arm — verified against
  `HEAD`'s own source over 49308 rows before anything was concluded from it.
  272957 SymPy-path rows and 63024 native-path rows; **2238 and 800 moved**,
  across 256 models. Every moved SymPy row was then evaluated *as emitted text*
  at 24 random points: over 51864 samples, **5087 are non-finite before and
  finite after, and 0 are the reverse**. Of the 889 that are finite in both and
  differ, scored against a 200-digit evaluation, the new form is closer at 768
  and the old at 101. Median emitted-C length change over the moved rows: **−10
  characters**.

  No `_CODEGEN_VERSION` bump: `_jacobian` and `_saturable_jacobian` are both in
  `_CODEGEN_SOURCE_MODULES`, so #51's source digest invalidates every cached
  artifact anyway.

- **A switch condition that cannot cross no longer declines a model's analytic
  sensitivity RHS (issue #382).** `MODEL0911270005` failed its forward-sensitivity
  solve outright — `CVODE integration failed at t=1.000000 with flag=-4
  (CV_CONV_FAILURE)` — with exactly one of its 31 shared parameters, `CRRFLX`,
  responsible. Three more witnesses in the same document family behaved the same
  way: `MODEL0911272039` (`ANPKNS`), `MODEL0911342562` (`ANGKNS`),
  `MODEL0911376350` (`ALDKNS`).

  **The failing column was never the defect.** These are reduced Guyton
  circulation models: most of the loop is cut away, and what remains refers to the
  removed parts through frozen `<parameter>` declarations. So they carry rate-law
  conditions like `CRRFLX>1e-07` with `CRRFLX = 0`, and `PO2ART<80.0` with
  `PO2ART = 97.0439` — comparisons that are false at the first step and false at
  the last. The issue #68 gate admitted a condition on three grounds (a clock
  threshold #48 stops at, a state comparison #150 roots on, or a comparison naming
  no symbol at all, `0>0`), and a comparison between run-constants — the same
  compile-time constant as `0>0`, only spelled with names — fell through all
  three. That declined the analytic sensitivity RHS **for the whole model**,
  handed the entire sensitivity solve to CVODES' difference quotient, and the
  `CV_CONV_FAILURE` was three steps downstream of it.

  Ground 3 is now `condition_cannot_cross`: an atom every one of whose names is a
  run-constant holds one truth value for the whole run, so there is no crossing in
  the window for anything to compensate and the in-branch derivative is the whole
  story. The test is structural, never numeric — it reads which names an atom
  carries and what kind each is — so moving a rate constant cannot flip it and
  `switch_gate_cache_digest` does not have to carry it.

  **Deliberately narrow about what counts as constant.** A model *function* also
  names a parameter slot (#227/#266) and `evaluate_functions()` rewrites that slot
  before every derivative evaluation, so those are excluded: their value moves
  with the trajectory even though the address is a parameter's. So are clock
  symbols and `time` in either spelling, species, observables, `rate_of__`
  accessors, a derived parameter that survived inlining, and any call to a model
  function — whose body is not at the call site and may read state the scan cannot
  see. Missing a crossing is the direction that ships silent zeros, which is what
  this gate exists to prevent.

  Measured over the manifest: 28 of the 182 condition-carrying corpus models
  change verdict, all in one direction (refused → admitted).
  `MODEL0911270005` moves from EXCEPTION to **PASS at `max_rel_err = 0` across all
  31 shared parameters**, on both corrector methods. AMICI cannot reference the
  other three (it fails them itself with `Inf` in `sxdot` at t=0), so those are
  checked against a finite difference of bngsim's own trajectory instead.

  **A parameter sitting exactly on its own threshold gets the one-sided
  derivative, and that is the honest answer rather than a quiet one.** `ANPKNS = 0`
  under `ANPKNS > 0` is a discontinuity in *parameter* space: perturbing it down
  moves the trajectory by exactly nothing, and perturbing it up moves it by a
  fixed ~1.0 that does not shrink with the step — a step, not a derivative. No
  saltation jump applies, because nothing crosses in *time*. bngsim reports the
  branch that is taken, which is what AMICI does with the same construct and what
  any AD system does with a `Piecewise`. What changed is that it now reports it
  from the analytic RHS instead of refusing every column in the model over it.

- **`MODEL0910846879`'s parity rows are attributed to AMICI, where the defect is
  (issue #382).** Surfaced by the fix above — admitting the model's run-constant
  conditions moved it off CVODES' difference quotient and so put a comparison in
  front of a row that had none. The DIFF itself is older and is identical before
  and after (`max_rel` 0.918 ode / 1.0 sens either way), so nothing here is a
  regression; what changed is that it is now visible and attributable.

  The model has no reactions and one state, so it reduces exactly: every
  assignment rule is over constants, giving `TVZ = 9.341479411e-4`, and the lone
  rateRule integrates to `TVD(t) = TVZ + (TVD0 − TVZ)·exp(−t/30)`. bngsim matches
  that to 1.9e-9. AMICI returns `3.499040825e-5` at t=100, which is
  `TVD0·exp(−100/30)` to 10 significant figures — the trajectory for `TVZ = 0`
  exactly. AMICI's own expression vector confirms it from the inside: it computes
  `TVZ1 = 9.3414794e-4` correctly at every time point and then reports
  `TVZ = piecewise(0, TVZ1 < 0, TVZ1)` as 0 at every time point. The control is in
  the same model — `AHTH = piecewise(0, AHTH1 < 0, AHTH1)` is structurally
  identical and AMICI gets it right — so it is not the piecewise shape as such.
  What separates them is that `TVZ`'s condition reads `TVZ1`, which is itself a
  sum over another piecewise.

  Reported upstream as
  [AMICI-dev/AMICI#3233](https://github.com/AMICI-dev/AMICI/issues/3233),
  minimized to four assignment rules and one state: a piecewise inside another
  piecewise's *condition* makes the outer one take the wrong branch. Changing the
  inner piecewise's first-piece value flips the outer result even though it
  cannot change the inner value, which is what pins the mechanism to the outer
  condition.

  Recorded as `INVALID_REFERENCE` entries on both regimes (the sens row cannot
  agree while the trajectory it is differentiated about is 92% apart), so all
  three rows read `REFERENCE_FAILED`/`invalid_result` rather than scoring as
  bngsim DIFFs.

  The evidence is a test, not a claim: `test_amici_dispositions` recomputes the
  closed form from the SBML values and checks bngsim against it, so the entry
  cannot outlive the fact it rests on. Two disposition contracts were widened to
  admit it, both preserving their intent — the third-oracle guard now accepts a
  closed form (strictly stronger than a third engine: no implementation left to
  be wrong) alongside RoadRunner, with a new test proving an entry supported only
  by bngsim is still rejected; and the regime-scoping tests now distinguish "a
  disposition must not leak into a regime it was not authored for" from "a model
  may be authored on both", which this is the first entry to do.


- **The codegen setup-time test now compiles something before it times it (issue
  #397).** `TestCodegenSetupTime::test_cc_codegen_records_cold_compile_then_cache_hit`
  failed one `macos-14` leg of a PR that touches neither codegen nor caching, on
  `assert 0.00032 < 0.00018` — 321 µs against 177 µs. Neither number is a compile.
  The "cold" leg was already a cache hit, so the assertion compared two draws from
  one near-zero distribution, and whichever way the noise fell decided the run.

  **The eviction was the bug, and `codegen_cache_hit` says so outright.** After the
  three lines the test used to open with —

      so = cg.get_cached_so(cg.compute_model_hash(net))
      if so is not None:
          so.unlink()
      cg._PREPARE_CODEGEN_MEMO.clear()

  — the first simulator reports `codegen_cache_hit is True`, in 199 µs.
  `simple_decay` has a complete analytical Jacobian and two observables and is not
  on a sensitivity run, so its .net codegen keys under
  `:codegen_jac:codegen_outputs:no_sens_rhs` (GH #162/#163, issues #209/#217),
  which hashes to a filename the base model hash never names. Nothing was ever
  unlinked, and every "cold" construction in the file reused the .so that
  `TestCodegenBackend` compiles one screen up. Nothing else in the suite evicts
  this way — `get_cached_so(compute_model_hash(...))` appeared at this one call
  site — and the sibling `TestCodegenCacheHit` both enumerates the suffixes and
  asserts the flag, so it cannot pass while warm. The caching is not implicated.

  The test now points `_codegen.CACHE_DIR` at a `tmp_path` and swaps the in-process
  memo for an empty dict, which says "nothing is cached" outright rather than
  naming keys that fall out of date as suffixes are added — the enumeration in the
  sibling class already predates `:codegen_output_sens`, `:sens_term_scale`, and
  `:chunk=`. The cold/warm ordering is read off `codegen_cache_hit`, `False` then
  `True`: the flag the pipeline records at the `get_cached_so` branch, which the
  sibling class documents as the definitive signal and which this was the last test
  still inferring from wall time.

  **T0.3's own claim survives as a floor, not an ordering:** `cold > 1e-3`. A cc
  invocation is a subprocess spawn plus a compile, a link, and a dlopen — 150 ms
  for this model on an M-series laptop, against ~200 µs to resolve a cache hit — so
  a millisecond sits two orders below any real compile on any runner and five times
  above the 177 µs the run accepted as one. `cold > 0.0` is what should have caught
  this; it passes for any nonzero measurement, so it passed, and the next line
  flaked instead. The cold leg now costs a real compile (~0.15 s) where it used to
  cost nothing.

- **A Hill ratio's derivative no longer evaluates to `inf/inf` when its own base
  is large (issue #393).** `x^n/(K^n + x^n)` saturates at `1`, so once `x^n` is
  large the fraction is flat and every derivative of it is a very small number.
  The quotient rule writes `x^n` in the numerator a second time, though, and
  `x^n·x^n` overflows one square root before the fraction does — from `x^n =
  1e154` up, not `1e308`. `BIOMD0000000829` sits in that band: its `mass_s`
  assignment rule reads `(1/mTOR_R)^n_1/(K_m^n_1 + (1/mTOR_R)^n_1)` with
  `mTOR_R = 4.58e-21`, so `(1/mTOR_R)^n_1 = 2.45e203` is an ordinary double and
  its square is not. The trajectory never noticed — the fraction is a clean `1` —
  and the `n_1` sensitivity column was NaN at the first output point.

  Issue #388's rewrite, one family over. `_rewrite_saturating_ratio` (was
  `_rewrite_saturating_exp_ratio`) divides through by the factor that runs away:

      f^m/(a + f)^k  ==  (1/(1 + a/f))^m · (a + f)^(m−k)

  an identity wherever `f` is finite and nonzero, with no intermediate that can
  overflow into a ratio of infinities — one factor saturates as `f → ∞` and the
  other as `f → 0`. For the sigmoid `f` is `exp(u)` (#388/#391); here it is
  `x^n`. `m > 1` is the state direction of the same ratio, where sympy folds the
  quotient rule's `x^n·x^n` into a single `x^(2n)`, so the numerator is matched
  by base and integer exponent ratio rather than by identity. It is carried in
  **both** emitters — `bngsim._saturable_jacobian` prints its own C and would
  otherwise let the identical NaN back in through `J·yS`.

  `BIOMD0000000829` now returns a wholly finite sensitivity tensor — including
  the `mass_s` assignment-rule row, which the run used to decline — and every
  column that carries signal matches central finite differences of bngsim's own
  trajectories to 1.2e-6 or better.

  **The zero-base logarithm guard now runs first.** #310/#317's guard and this
  rewrite want the same `x^n`, at opposite ends of its range: the guard replaces
  `x^n·ln x` with its limit at `x = 0`, this one divides through at `x^n → ∞`.
  Dividing first takes the guard's power away and turns a term that read a clean
  `0` at `x = 0` into a NaN. Running second, the rewrite finds that power already
  inside a `Piecewise` and leaves it there — and still reaches every power the
  guard did not claim, which is where the overflow lives.

  **Corpus A/B.** All 1274 loadable `rr_parity` models, every rate law
  differentiated w.r.t. every free symbol, emitted through `sympy_to_c` *and*
  through the native emitter, both arms in one process with the pre-#393
  pipeline patched back in for the `before` arm. 194052 SymPy-path rows and 6889
  native-path rows; **6669 and 271 moved**, across 301 of 1242 models. 6661 of
  the SymPy-path moves are the power numerator and the reordering; 301 rows are
  touched by the `f^m` case.

  Every moved SymPy-path row was then re-derived in both arms and evaluated *as
  emitted text* at 24 random points each — `lambdify` is not a model of the
  emitters, since it spells `a/x^n` as `a·x^(−n)`, and a separately-computed
  reciprocal overflows where the quotient does not. Over 5903 rows and 118032
  samples: **9710 points are non-finite in the old form and finite in the new
  one**, and 34 are the reverse. Those 34 are reassociation, not the identity:
  dividing `f` out of the numerator takes a small factor away from a sibling that
  was overflowing behind it, and sympy's printer emits every numerator factor
  before the division. They occur only where the sampler puts a Hill exponent at
  160 and its base at `1e-2`; both arms are already non-finite at 28138 further
  points in that regime. Where both were finite but disagreed by more than
  `1e-9` relative (1713 points), each was scored against an exact evaluation at
  200 digits: the new form is closer at 1078 and the old at 610.

  A partial `x^(2n−1)` case is deliberately left out: #96/#351's removable-power
  fold runs first and turns `x^(2n)/x` into an exponent no longer commensurate
  with the sum's, and undoing that trades the overflow for the `0/0` at `x = 0`
  that fold exists to remove. Measured and split out as issue #402.

  No `_CODEGEN_VERSION` bump: `_jacobian` and `_saturable_jacobian` are both in
  `_CODEGEN_SOURCE_MODULES`, so #51's source digest invalidates every cached
  artifact anyway.

- **A parameter reaching the dynamics through a constant assignment rule now
  carries its sensitivity chain (issue #385).** COPASI exports write a knob
  through an indirection — `<initialAssignment symbol="ModelValue_0"> Theta`,
  `<assignmentRule variable="Alpha"> ModelValue_0/(24*3.344)`,
  `<initialAssignment symbol="ModelValue_1"> Alpha`. The lift took the first hop
  and refused the third, so `ModelValue_1` was declared a **primary** parameter —
  an independent knob — when it is `Theta/80.256`. Values were right either way;
  what was lost was every sensitivity term routed through it.

  On `BIOMD0000000587` that is most of `Theta`'s column and half of `rho_f`'s,
  enough to flip two of three signs: `Theta` returned +1.06159 where RoadRunner
  and AMICI agree on −33.2326, `f` returned −0.008265 against −37.80348, and
  `rho_f` 180.43 against 334.1004. All three now match a central difference of
  the trajectory, as do `BIOMD0000000586`'s three columns and
  `BIOMD0000000852`'s `cxx` (+2.97e10 → −8.6601e14) — the issue suggested that
  third model might be a separate defect, and it is not.

  The refusal was deliberate and its reasoning was sound for the case it was
  written against: an assignment rule's *slot* is function-backed, so a lifted
  expression reading it re-derives from the rule's value at the last integrated
  point rather than the `t = 0` fold the initialAssignment means
  (`BIOMD0000000570`'s `ModelValue_60 = O2c_bar` would go 5.68 → 7.87 on the
  next write after a run). The lift now substitutes the rule's **body**, so the
  emitted expression never mentions the slot, and only when that body reaches
  nothing but constant parameters — the condition under which the two readings
  coincide anyway, because such a rule cannot move. `O2c_bar` is excluded by
  exactly that test, its rule reading species.

  Across the 1644-model corpus this moves `primary_param_names` on **19 models**
  and nothing else: no model newly fails or newly loads, trajectories are
  unchanged, and the identity write `set_params(dict(zip(param_names, vec)))`
  round-trips exactly on every one of the 19. Four PBPK models
  (`BIOMD0000001027`/`1028`/`1029`/`1039`) drop ~16 `Compartment_*` entries,
  which are COPASI initial-size parameters like `Compartment_1 := Liver :=
  ModelValue_1 * 0.0549` — organ volumes as fractions of body weight, whose
  independent coordinate is the body weight (issue #203).

- **A Hill exponent's sensitivity column no longer NaNs because the predictor put
  its base a few ulps below zero (issue #392).** Differentiating a power law
  w.r.t. its exponent produces `base^n·ln(base)`, and `_guard_exponent_log_at_zero`
  (issues #310/#317) replaces that product with its limit — but the condition it
  emits is `base == 0`, and the solver does not hand it a base of exactly zero.
  `BIOMD0000000833`'s `S35` is `0.0` at `t = 0` and non-negative at every one of
  2001 output points, and CVODES' predictor puts it at `-3.75e-36` between two of
  them, twenty-four orders below the run's own `atol`. `pow(x, 4.0)` is finite
  there so the trajectory never notices; `log(x)` is not, and `x == 0.0` is false,
  so the branch never fires.

  The compiled sensitivity RHS now retries at a state whose sub-`atol` negative
  components are snapped to zero, which is the issue #135 conditional clamp the
  value RHS and the analytical Jacobian have both had for years, applied where the
  run's own tolerance is known. Nothing in the emitted C changes, so no cached
  artifact is invalidated, and the retry is reached only from an already
  non-finite `ySdot` — on a corpus A/B of 166 models, both arms, **nothing moved**.

  Telling "numerically zero" from "negative" needs a scale the emitter does not
  have at build time, which is why this is a retry rather than a wider condition:
  `base <= 0` in the emitter would answer a confident `0` for a base that is
  *genuinely* negative — `BIOMD0000000374` carries a species `V_membrane = -61` —
  where `∂/∂n base^n` does not exist at all. A component negative by more than
  `atol` is left alone and its NaN still reaches the issue #384/#386 refusal.

  Issue #395's recoverable return already rescued the two witnesses by cutting `h`
  until the predictor stopped overshooting, so this is mostly a matter of how much
  work that cost: on `MODEL0911120000` the sensitivity corrector failed **1464**
  times and now fails none, Jacobian evaluations drop 2730 → 356 and steps
  4689 → 2692. The tensors agree to 2.9e-07 across every column that carries
  signal. `BNGSIM_SENS_CLAMP_NUMERIC_ZERO=0` restores the previous behaviour.

- **Three derivative emitters returned a non-finite value where the derivative is
  an ordinary number (issue #388).** Each cost a forward-sensitivity column on the
  corpus, and each is a distinct shape:

  - **An overflowing sigmoid.** `exp(u)/(a + exp(u))^n` is `inf/inf` as soon as
    `|u| > 709`, at points where the ratio itself is perfectly ordinary. It is now
    divided through by `exp(u)` — issue #393's rewrite, one family over. Carried in
    **both** emitters: the native saturable differentiator bypasses the sympy
    chokepoint, so the sensitivity RHS's `J·yS` half would otherwise reintroduce
    the NaN one term later.
  - **One logarithm carrying two bases.** `ln(k*(t − T))` guards both `k^s` and
    `(t − T)^(s−1)`, and the zero-base guard handed the expression to the first
    power it met and left the second unguarded.
  - **A quotient that still mentions the symbol.** `(x/(x + K))^n / x` was refused
    because `1/(x + K)` is not free of `x`, leaving `0/0` in the emitted derivative.

  These are the failures that reach the issue #384 refusal below at the **first**
  output point — a derivative that was never defined, rather than a blow-up the
  issue #177 tolerance floor admits partway through a run — which is why neither a
  tighter `atol` nor `BNGSIM_SENS_ERROR_FLOOR=0` ever moved them.

- **A non-finite forward-sensitivity tensor is refused instead of returned
  (issue #384).** CVODES can return `CV_SUCCESS` with NaN in the sensitivity
  vectors: its error test is a comparison, and every comparison against NaN is
  false, so the machinery whose job is to reject the value cannot see it. The run
  reported a clean solve and handed back a poisoned gradient. It now raises,
  naming the axis (parameter column or initial-condition column), the first
  affected output point, and every column implicated there — the time localizes the
  blow-up for a bisection, the column names what to drop to get a usable run out of
  the same model.

  **`inf` and `nan` are counted apart (issue #394)**, because they mean different
  things and take different advice. An `inf` is a derivative that diverges — the
  model sitting on a branch point of its own — and there is no finite number to
  return, so the `atol` and `BNGSIM_SENS_ERROR_FLOOR=0` knobs are offered only for
  a `nan`; neither moves a derivative that is unbounded. The scan runs on past the
  first affected point when that point carried no `inf`, so "a derivative diverges
  later in this run" is not buried (28 ms over a 1001×41×75 tensor, on a run that
  is already failing).

  A `nan` is an arithmetic accident — `0*inf`, `inf/inf`, `0/0`, a domain error —
  and has been a bngsim defect every time so far (GH #310, #317, #333, #351, #391).
  **It is not proof of one, and the measurement is why the message says so.** Issue
  #394's two examples — `BIOMD0000000632`'s `d(sqrt(Gy))/dGy` at `Gy = 0` and
  `MODEL2403070001`'s `(t − Tmeal)^-0.6` at `t = Tmeal` — are both genuinely
  unbounded and were expected to arrive as `inf`. Measured, they do not: what
  reaches the tensor is `nan`, the `inf` annihilated on the way. So the census
  reports which arithmetic arrived and then hands over the check that actually
  discriminates — whether the named column's parameter sits at a singular value of
  its own.

  **A value is caught where it appears, not where CVODES notices it (issue #395).**
  CVODES reduces the per-column sensitivity norms to their maximum by seeding the
  accumulator with column 0 and keeping `cvals[is] > nrm`; every comparison against
  NaN being false, that reduction propagates a NaN in column 0 and discards one in
  any later column. The corrector's convergence test and the sensitivity error test
  both read that number, so an identical NaN either stalled the solve or was
  invisible to it — decided by nothing but the parameter's position in
  `sensitivity_params`. Two routes in, so two places: the sensitivity RHS returns
  the recoverable code on a non-finite `ySdot`, so CVODES cuts `h` and retries at
  the point of production (a NaN the predictor caused by overshooting a domain
  boundary for one step is rescued; one that is really the model fails, naming the
  time), and the initial seed `∂x(0)/∂θ` is checked before `CVodeSensInit1` and
  refused by column and row — no step can repair it, and it is not the dynamics.
  `BNGSIM_SENS_NONFINITE_RECOVER=0` restores the previous behaviour exactly, and
  the tensor scan remains the backstop for every column CVODES
  difference-quotients itself, where bngsim's callback is not in the loop.

  **`Result.solver_stats` gained the sensitivity solve's own counters** —
  `n_sens_err_test_fails` and `n_sens_nonlin_conv_fails`, from
  `CVodeGetSensNumErrTestFails` / `CVodeGetSensNumNonlinSolvConvFails`. The
  existing `n_err_test_fails` and `n_nonlin_conv_fails` count the **state** solve,
  so a run could reject steps on its sensitivity error test that the state solve
  never saw and still report `n_err_test_fails == 0`. That is not a smaller number
  than the truth, it is a different quantity, and reading it as "the solve was
  clean" is how a poisoned column got called healthy. Both are `0` on a run with
  no sensitivities.

  **The blow-up itself is not fixed here**, and the tolerance floor is one cause
  rather than the cause. Issue #177's floor admits it on some models, but sweeping
  the floor moves the failure rather than removing it — `BIOMD0000000480` has 0
  non-finite cells at `tau` 9e-4, 1517 at 1e-3, 0 at 1.05e-3, 33661 at 1.2e-3 and 0
  at 1.5e-3 — and issue #388 above measured 14 corpus models that stay non-finite
  with the floor off. There is no tweak to make, only a silence to end.

- **An `<initialAssignment>`-mediated parameter now seeds the initial-condition
  term it was owed (issue #379).** Issue #313 below unfroze these parameters in the
  dynamics; the `t = 0` seed `∂x(0)/∂θ` was still withheld from them by three
  separate filters, so a column could be right for the whole run and wrong at its
  first point. **24 of `BIOMD0000001102`'s 27 read zero at `t = 0` where both other
  engines report a number.**

  - An SBML `constant="false"` declaration says a symbol *may* vary, not that
    anything varies it. The IC-lowering predicate treated it as disqualifying, but
    the only thing that makes a symbol unsafe there is being promoted to a species
    — which this loader does for rate-rule and event-assignment targets only, both
    already subtracted. The extra filter bought no safety and withheld the seed
    from every `<initialAssignment>` reading such a parameter.
  - An `<initialAssignment>` may read another **species**, meaning that species'
    initial value, and rejecting those withheld the seed from every column of the
    expression. A state reference now resolves to whatever expresses that state's
    own IC — its parameter, its synthetic derived parameter, or its constant value
    — lowered in dependency order so the chain rule composes through the existing
    derived-parameter DAG.
  - A species whose own IC is a declared constant therefore contributes a constant,
    so the lift keeps the parameters the expression also reads instead of freezing
    all of them behind the fold.

  **Two substitution guards keep issue #164's compartment-size refusal honest.**
  Section 0 binds a `hasOnlySubstanceUnits` symbol to its amount, so substituting
  the number would bake a live compartment size into the lifted expression as a
  literal — the fold #164 refuses a size over, one layer down and invisible to the
  refusal. And a species declared in the unit its symbol does not mean carries an
  `amount/V` conversion: emitting the conversion rather than the converted number
  keeps the size symbolic, so a later `set_param` still reaches the initial
  condition. `rateOf` is excluded at both sites — it needs the species symbol, not
  its value. Two size refusals are **earned away rather than relaxed**
  (`MODEL1710030000` among them): all of their initial conditions lower now, so a
  write lands exactly where a rebuild lands, which is the property the refusal
  protected. What still cannot lift is the `hasOnlySubstanceUnits` shape, and the
  param-lift freeze warning is repointed at it.

- **Running the test suite no longer fills the developer's own bngsim caches
  (issue #372).** A pytest session redirects both content-addressed caches —
  compiled `.so` artifacts and BNG2.pl-generated networks — from `~/.cache/bngsim`
  to a directory the suite owns: `.pytest_cache/d/bngsim/`, or a per-run temp
  directory when the cache provider is off (`-p no:cacheprovider`, which every CI
  leg passes). Both the module attribute and the env var are set, so subprocesses
  that import bngsim land there too.

  Nothing had redirected `CACHE_DIR` at session scope, so every test that built a
  `codegen=True` simulator or loaded a `.bngl` wrote into the real cache and left
  it there. On the box this was filed from, two suite runs in one afternoon
  accounted for 303 live artifacts, behind an orphan pile of 146 MB; two BNGL test
  files alone left seven `.net` files in `~/.cache/bngsim/networks`. Since #363 put
  the codegen key in the artifact *name*, the worst of it became visible: two rows
  in `bngsim-cache info` under a key **no install has ever had**, left by a test
  that monkeypatched the key without the directory. `test_prepare_codegen_memo.py`
  and `test_codegen_sensitivity.py` — the two that invented keys — now patch
  `CACHE_DIR` as well, and assert their artifacts stayed contained.

  The redirect is persistent rather than per-run, so runs stay warm (a cold full
  suite measured 19m19s against roughly 14m). `pytest --cache-clear` wipes it for
  a deliberate cold run, and `BNGSIM_TEST_CACHE_DIR` relocates it. This is test
  infrastructure only — no library behavior changes, and `BNGSIM_CODEGEN_CACHE_DIR`
  / `BNGSIM_BNGL_CACHE_DIR` mean exactly what they meant for anything but pytest.

- **Two switch times set to the same number no longer charge each other's jump
  (issue #375).** A switch-time parameter's whole gradient is the jump
  `s⁺ = s⁻ + (f⁻−f⁺)·∂t*/∂p` at its crossing (issue #48), and the core reads
  `f⁻`/`f⁺` by nudging the *clock* a few ulp either side of the threshold. That
  reads the whole right-hand side, so every condition thresholding that clock at
  that value flips together and the difference is their **sum**. The detector
  keyed crossings on the threshold's *value*, so two conditions holding the same
  number merged into one record whose `∂t*/∂p` was the union of both, and each
  parameter came back with the other's jump added.

  On two independent ramps — `if(time()>=tA, 0, kA)` and `if(time()>=tB, 0, kB)`,
  sharing no state and neither rate law naming the other's parameter — the exact
  matrix `[[kA, 0], [0, kB]]` came back as `[[kA, kA], [kB, kB]]` at `tA = tB`,
  and exactly right the moment they differed. On the corpus it reaches
  `BIOMD0000000075` and `BIOMD0000000161`, which each ship three stimulus onsets
  at `tau = 0.05`: `∂PI_PM/∂tau0_PLCact` read `2.7e+03` where the true value is
  unresolvably small, and `∂PIP2_PM/∂tau0_PLCact` came back `-2.6e+03` against a
  true `+23.7` — wrong sign, two orders of magnitude, nothing logged. The
  spurious entries largely cancel along the conservation chain, so a sum over
  species looks right while the individual columns do not.

  **Two things had to change, and the first alone would not have been enough.**
  Crossings are now keyed on `∂threshold/∂primary` rather than on the threshold's
  value, so one threshold gating six rate laws still collapses to one crossing —
  that merge is load-bearing, or its jump is applied six times — while two
  thresholds that merely share a number stay apart. But separate records would
  still each read the same clock-nudged `f⁻ − f⁺` and still get the combined
  jump. So a crossing that shares its instant is now separated by moving its
  *threshold* instead: `SwitchTimeSens` carries a parameter bump the core applies
  only while it reads `f⁻`, sized so this crossing's threshold alone rises off
  the instant. With the clock held on the after side, that condition alone falls
  back to its before-branch, the difference is its own jump, and the core's
  existing per-instant sum is correct again.

  The parameter raised is one no coinciding threshold reads, and the step is
  capped at a quarter of the distance to the nearest neighbouring threshold so it
  cannot flip a condition this crossing does not own. Where no such parameter
  exists the crossings are genuinely inseparable this way and bngsim raises
  `SensitivityUnsupportedError` rather than return the merged jump. Models whose
  switch times are distinct — nearly all of them — carry no bump and take the
  pre-#375 path unchanged.

  Crossings that no *requested* parameter moves are now detected and kept as
  well. They emit no column, but they flip at that instant and contaminate `f⁻`
  just the same, which is why #375 reproduced even when a single parameter was
  requested.

- **`run(sample_times=…)` is documented for what it actually does (issue #368).**
  The docstring asked for "at least 3 values" where the code takes 2, and read as
  though the argument were an output grid laid over `t_span`. It overrides
  `t_span` outright, so its *first* entry is the integration start:
  `sample_times=[5.0, 10.0]` integrates `[5, 10]` from the model's initial
  conditions and never sees `[0, 5]`. That matters most to the one job this
  argument is reached for — placing output points around a feature so a finite
  difference can be read off them — because a grid that opens past `t = 0`
  silently compares two runs of a different problem. Docstring only; no
  behaviour changes.

- **An `==` or `!=` in a rate law no longer drops the condition it belongs to
  (issue #335).** `_exprtk_to_sympy` handed the preprocessed rate-law string to
  sympy's `parse_expr`, which evaluates it with **Python** semantics. `==` and `!=`
  are the two relationals where that is wrong: `Symbol('x') == 0` is Python's
  structural equality, so it returns the bool `False` rather than `Eq(x, 0)`, and
  the comparison is gone before sympy ever sees a condition. `if(x==0,1,2)` parsed
  to a bare `2` and `if(x!=0,1,2)` to a bare `1` — **the wrong branch, taken
  unconditionally, with nothing to mark it.**

  Both are now rewritten to the `Eq` / `Ne` call form in the shared logical
  rewriter, the exact analogue of the existing `&&`/`||` → `And`/`Or` treatment and
  at the same tighter-than-logical precedence; the four ordering relationals
  already built sympy relationals and are untouched. The downstream was already
  equality-ready — `_print_Relational` (#310) spells `Eq`/`Ne` infix for both the
  ExprTk and C emitters, and `_is_emittable` skips Piecewise conditions — so a
  corrected condition round-trips with no other change, and since a Piecewise
  condition is never differentiated (conditions copy through), this only makes the
  derivative respect the right branch.

  **Corpus sweep over the 52 loadable BioModels carrying `==`/`!=`** — 214 affected
  functional rate laws — shows no new declines and no new errors (off→on: emit→emit
  198, decline→decline 8, error→error 8), and the corrected derivatives match
  finite differences. On the derived-parameter chain-rule and IC-seed paths the old
  behavior was **a wrong number rather than lost coverage**: `if(sel==1, kA, kB)`
  collapsed to `kB`, so the seed took the else branch at every value of `sel`.

  Two consequences of making equality reachable ride along. sympy folds an
  `Eq`-over-Piecewise condition inside an `and` — the BNGL boolean-coercion idiom
  `if((if(c,1,0)==1) and rest, t, f)` — into an `ITE` node, which is a `Boolean`
  rather than a `Function`, so `_is_emittable`'s `atoms(Function)` scan does not see
  it (the same blind spot `Min`/`Max` have) and both printers fell through to a
  literal `ITE(...)` the ExprTk/C engine cannot parse. The C++ reliability gate then
  rejected the derivative and the whole model silently dropped to the
  finite-difference Jacobian — a capability regression surfaced by
  `MODEL1708310001`'s periodic chemo schedule. `_normalize_booleans` now rewrites
  `ITE(c, t, f)` to the boolean identity `(c & t) | (~c & f)` at the top of both
  `sympy_to_exprtk` and `sympy_to_c`, guarded on `has(ITE)` so every ITE-free
  expression is returned unchanged and its emitted text is byte-for-byte identical.
  And a non-finite (`zoo`/`oo`/`nan`) derivative is now declined rather than emitted
  as broken code.

- **A power law whose base reaches zero no longer NaNs its own derivative
  (issue #351).** SymPy differentiates `u^n` as `n·u^n·u'/u` and leaves the two
  `Pow`s uncombined — with a symbolic exponent and a base of unknown sign,
  `u^a·u^b = u^(a+b)` crosses a branch cut it will not assume. Wherever `u`
  reaches zero the emitted C is `pow(u,n)/u` = `0/0` = NaN, at a point where the
  law's own value is finite and the true derivative is an ordinary number. One
  NaN poisons that parameter's whole sensitivity column, or defeats the corrector
  outright when it is the only column: `BIOMD0000000703` failed `CV_CONV_FAILURE`
  at `t=0` through `(A4 − A4_star)^nA4` with `A4(0) = A4_star = 1.0`, while its
  plain ODE run succeeded at the same tolerances.

  The rewrite that removes it (`_remove_removable_power_denominators`, extended
  from GH #96) now cancels `base^n / d` whenever `d` **is the base itself**, for
  any base whatever — a symbol, a difference, a whole rational sub-expression.
  GH #96 could only ask "is `base` a linear multiple of the symbol `d`?", which
  required `d` to be a bare `Symbol` and so reached `x^n/x` by accident while
  missing `(A4 − A4_star)^n/(A4 − A4_star)` entirely.

  **This needs no case split on the exponent, and that is the point.**
  `pow(u, n-1)` in IEEE arithmetic is `0` for `n > 1`, `1` for `n = 1`
  (`pow(0,0)` is 1 by C99) and `+inf` for `n < 1` — the true value of `n·u^(n-1)`
  in every regime, *including* the infinite one. Unlike the log /
  fractional-power family (#310/#317/#333/#336), where the state is outside the
  law's domain and no finite answer exists, here the answer existed and the
  emitter was throwing it away by not cancelling. Only a **symbolic** exponent
  can reach the bug at all: sympy evaluates `diff(u**3, x)` to `3*u**2*u'`
  itself, so no division is ever emitted for a literal one.

  Both emitters share the one chokepoint, so the interpreted RHS Jacobian and
  the compiled `∂f/∂p` get the same arithmetic.

  **Corpus-wide before/after over the 1,319 rr_parity models** (each variant in
  its own process with its own cold cache, because `_CODEGEN_CACHE_KEY` digests
  the emitter sources and a shared cache would have reported "no change" for
  everything): 22 models' emitted source changes, 1,258 are byte-identical, 39
  do not load for unrelated reasons. Of the 22, exactly **two** models' numbers
  move, and finite differences back the new value in both:

  - `BIOMD0000000703` — 440 of 2,772 sensitivity cells were non-finite; now none
    are, and the column matches a central difference of the trajectory to 1e-7.
  - `BIOMD0000000617` — a *silently wrong finite* answer, which is the worse half
    of this bug. `∂v/∂Kxx1` at t=10 was `0.9436790062` against FD's
    `0.9764789789`; it is now `0.9764789783`, agreeing to 1.8e-12. 20 of 252
    cells moved, all toward FD. Nothing flagged this model before — it produced
    no NaN, raised nothing, and had simply been wrong by 1e-4 relative.

  The other 20 are unchanged to 1e-9 or better.

- **MSVC no longer leaks a `.lib`/`.exp` pair into the codegen cache on every
  successful Windows compile (issue #362).** `cl /LD /Fe:<out>` writes an import
  library and an export file beside the DLL, and `compile_rhs`'s cleanup named
  only the DLL — which on the success path has already been `os.replace`d away,
  so its `unlink` was a no-op and the pair stayed, one per compile, forever.
  Unlike the shard-directory and stray-`.c` leaks `bngsim-cache clean` collects
  (#205), this one did not need an interrupted compile. It rides along here
  because #351 already invalidates every cached artifact (#51), so the fix costs
  nothing extra rather than spending a global invalidation on its own.

- **A non-finite compartment size no longer loads an amount-only species as
  `nan` (issue #353).** `MODEL2002070001` declares `size="NaN"` on both of its
  compartments and makes every species `hasOnlySubstanceUnits="true"` — the
  quantities are amounts, and no rate law reads a size. RoadRunner and AMICI both
  integrate it in amount units. bngsim stores a *concentration* (`amount / V`), so
  the `NaN` size divided a well-specified `initialAmount=10` into `nan`: six of
  seven species loaded non-finite and the **plain ODE run** — sensitivities off —
  failed at `t=0` with `flag=-9`, naming the rate laws as if they were the cause.

  A non-finite size is not a legitimate divisor. The loader now substitutes a
  unit volume for it (and warns, once per compartment), so an amount-only species
  loads as its declared amount and the model integrates in amounts, matching both
  reference engines to solver tolerance. A model with a *finite* size is untouched
  — its amount is still stored as `amount / V`, byte-for-byte — so the
  substitution can only ever turn a `nan` state into a usable one. Refusing the
  model instead, the way issue #170 refuses an ambiguous compartment-size *write*,
  would have made bngsim strictly worse than the engines it checks against; #170's
  writability guard already keeps a later `set_param` on the substituted size
  honest.

- **The non-finite diagnostic names the corrupt state, not just the innocent
  law (issue #353).** `describe_nonfinite_witness` reported the species it thought
  responsible with a below-zero test — and `nan < 0.0` is false, so a `nan` state
  was invisible to it. What the user got was the rate laws that answered `nan`
  (because their inputs already had) and a closing sentence about rate-law domains
  that sent them to fix a law that was fine. The witness scan now names non-finite
  state components first and separately, and when the state itself is non-finite
  the closing guidance points at the initial condition (an under-specified
  species, or a size that is not finite and positive) rather than at the laws that
  merely inherited it.

- **A parameter that both sets an `if()` switch time and scales its own branch is
  answered on the analytic path rather than refused outright (issue #358).**
  `if(t>=sigma, sigma*k, 0)` with `sigma` requested was refused for forward
  sensitivity everywhere. Its gradient is the interior variational term plus the
  crossing jump, and bngsim already computes both: `bngsim_dfdp` emits the clean
  in-branch `∂f/∂p` (the Piecewise derivative, no boundary delta) and
  `compute_switch_time_sens` produces the saltation jump. What could not combine
  them was CVODES' difference-quotient probe, where pinning the switch time holds
  that in-branch term at a wrong `0`. On a model carrying an analytic sensitivity
  RHS there is no probe: the two terms sum to the correct total and the pin is
  inert.

  So the refusal narrows to the difference-quotient path instead of standing
  everywhere — `Simulator` reads the artifact's `bngsim_codegen_sens_rhs` symbol to
  decide which path a run is on, the same ground truth issue #414's gate keys on,
  and the typed refusal of issue #320 below is what a caller sees when the narrowed
  case does apply. Validated on a closed-form fixture (exact, no drift across
  `rtol`) and against a central finite difference on `BIOMD0000001007/1009/1010`,
  whose ODE-species columns now match to ~1e-6.

  Two switch-time detector fixes fall out of getting those three right, both in
  `_switch_sensitivity.py`. A clock threshold can **hide behind a function** —
  `heav_x = if(x<0, ...)` over `x = time()-ModelValue_27` — so the raw atom `x<0`
  named no clock, its crossing was compensated by neither detector, and the jump
  was dropped; references are now inlined before scanning, as the issue #150 state
  detector already did, and `_condition_only_params` inlines the same way and scans
  the reaction rate laws rather than helper functions in isolation, so a
  parameter's branch-vs-condition role is read where it actually occurs. A census
  over 443 condition-carrying `rr_parity` models shows the change only resolves
  detector/gate disagreements: crossings hidden behind constants or `floor()` that
  were spuriously attributed to helper symbols are now correctly ignored or
  declined. And a switch time that *also* reads a condition nothing compensates — a
  `floor()`-periodic dose schedule — is refused: pinning it holds that crossing's
  dependence at a wrong `0` (`MODEL1708310001`'s `cycle_int` measured `0` against a
  finite difference peaking at ~19) and un-pinning reintroduces the issue #48
  stall, so neither is safe.

- **A clock threshold is recognized by the crossing it has, not by how it was
  typed (issue #355).** `_clock_threshold_split` decided what counted as a
  recognized clock threshold by *where the clock sat*: exactly one side had to be
  the clock symbol itself. So `time()<T` was admitted and `time()-T<0` — the same
  threshold, constant moved across the comparison — was declined, as was anything
  affine with a parameter-dependent slope.

  That was not cosmetic. The gate is per model, so one such condition took the
  **whole model** off the analytic sensitivity RHS and onto CVODES' difference
  quotient, which integrates straight through the crossing and drops the
  `(f⁻−f⁺)·∂t*/∂p` jump — the failure issue #232 measured at 53% wrong. And the
  decline warning said the condition "reads model state" when what it reads is
  the clock.

  The recognizer now solves the residual for the clock: `a·t + b = 0` gives the
  crossing `t* = −b/a` as a symbolic expression, which is what `∂t*/∂p` must be
  differentiated from. The spelling test still answers **first and unchanged**,
  so no threshold recognized before this change moves path or threshold text;
  only an atom it declines reaches the solve. Reading the clock on *both* sides
  (`t<2*t`) is still not a clock threshold — that one stays the state path's.

  Blast radius measured over all 1323 `rr_parity` models: **7 carry an atom that
  newly resolves**, and re-sweeping all of them against AMICI moved 8 of 14 rows
  with **zero regressions** — `MODEL2307130001` goes TIMEOUT → PASS on both
  corrector methods, and `BIOMD0000001007/1009/1010` (three of issue #326's five
  open models, all stalling at `t ≈ 49.99841887`, which is exactly where
  `time()-Tdam` crosses with `Tdam = 50`) convert from an unexplained
  `CVODE made no progress` into a precise declared refusal naming `Dam0`,
  `Tdam` and `krepair` as parameters that set a switch time *and* appear inside a
  branch. Those three are explained rather than fixed, and #326's recorded ruling
  that they carry no time discontinuity does not hold.

  Also corrects the decline warning, which pointed readers at issue #150 —
  closed 2026-08-03 — for a residue #150 never covered.

- **An event whose trigger residual starts exactly ON its threshold no longer
  fires spuriously (issue #340).** `BIOMD0000000285` declares `PIdeath > 0` on a
  species whose initial amount is 0. The trigger is false at `t_start` (`0 > 0`)
  but its residual sits exactly on zero, and the aggregation cascade that feeds
  `PIdeath` starts at zero too, so `d(PIdeath)/dt` there is zero as well. The
  species still creeps positive once the cascade turns over, the boolean trigger
  flips, and bngsim read that as a rising edge: `kalive := 0` at **t = 2.7e-27**,
  freezing every species at its initial value for the whole run. Both of the
  model's death events did it at the same instant, which is how forward
  sensitivity saw it — an ambiguous `dt*/dp` at a crossing whose time is set by
  the step controller rather than by the model. It fires on a plain ODE run too;
  the sensitivity path is only where it became loud.

  What separates that from a trigger that genuinely crosses at `t_start` is
  `dg/dt` along the flow, read there and nowhere else — after one step the
  residual is off zero and the coincidence has left no trace:

  - `dg/dt > 0` — the trajectory **leaves** the threshold into the trigger's true
    side. The crossing is real and sits at `t_start`. Unchanged.
  - `dg/dt <= 0` — it does not leave, so a root reported at `t_start` is the
    initial condition being re-read rather than a transition, and does not fire.
    The mark is one-shot and scoped to a window of 100 ulps of the run's time
    scale, so a later genuine crossing of the same trigger (a residual that dips
    to the false side and comes back) is untouched.

  This is the rule SUNDIALS applies to its own root functions — a `g_i`
  identically zero at `t0` is deactivated until it has moved away — which bngsim
  cannot inherit, because the root it registers is the **boolean** trigger minus
  0.5 and that is never zero. AMICI roots on the residual, gets the rule from
  SUNDIALS, and does not fire here; bngsim now agrees with it on this model to 8
  significant figures.

  **The `dg/dt` test is what makes this safe, and it is not optional.** A
  structural scan of the 1323-model `rr_parity` corpus finds 19 event triggers
  whose residual is exactly zero at `t_start` (27 further triggers do not resolve
  statically and are not in that count). **Nine of them are `time > 0`** —
  `BIOMD0000000318/338/339/474/632/736`, `MODEL1108260014/1412200000/1708210000`,
  which is how those models spell "at the start". Their residual is exactly zero
  too; only `dg/dt = 1` tells them apart from `PIdeath > 0`, and suppressing on
  the zero residual alone would have silently disarmed all nine. Of the
  remaining ten, five read *true* at `t_start` (`>=`/`<=` sitting on the
  threshold, governed by the existing `initialValue` rule), three declare
  `initialValue="true"` and were already suppressed, and two are
  `BIOMD0000000285`'s.

  **Measured blast radius: one model.** Every corpus model that declares an event
  — 194 of them — was run at its manifest horizon and tolerances before and
  after. 190 completed on both sides and **exactly one trajectory changed:
  `BIOMD0000000285`.** The other 189 agree to within 1e-9 relative; the four
  non-completions (`BIOMD0000000137` fast-reaction refusal, `BIOMD0000000404`
  timeout, `MODEL2006080001` RHS failure, `MODEL2205030001` undefined-symbol
  refusal) are identical messages on both sides.

- **The forward-sensitivity job raises its integrator step budget, symmetrically
  (issue #339).** Both engines default to 10,000 steps per solve — bngsim's
  `Simulator.run(max_steps=)`, AMICI's `Solver.set_max_steps` — and the harness
  set neither, so the budgets were already equal. What #331 changed is what they
  are spent on: raising `Np` from a flat 20 to a coupled-state budget put up to
  306 sensitivity columns in the error test, and on the first sweep carrying it
  **8 models went `PASS` → `REFERENCE_FAILED`** with AMICI out of steps. Losing
  the oracle to a solver setting neither engine's user chose is not a result.

  `--max-steps` (default **100,000**) is now applied to both engines from one
  read of the job spec, so no future edit can raise it for one and not the other,
  and each row records the budget it was solved under.

  **Measured, not scaled.** #339 proposed scaling the budget to the coupled
  system; the data does not support that. All 8 were probed at 10k / 100k / 1M:

  | model | Np | at 10k | at 100k |
  |---|---|---|---|
  | `BIOMD0000000832` | 56 | `AMICI_ERROR` | ok, 0.6 s |
  | `BIOMD0000000061` | 69 | `TOO_MUCH_WORK` | ok, 0.9 s |
  | `BIOMD0000000667` | 83 | `TOO_MUCH_WORK` | ok, 4.4 s |
  | `BIOMD0000000474` | 150 | `TOO_MUCH_WORK` | ok, 15.9 s |
  | `MODEL2401050001` | 161 | `TOO_MUCH_WORK` | ok, 11.9 s |
  | `MODEL2202020001` | 188 | `TOO_MUCH_WORK` | ok, 5.0 s |
  | `MODEL0911120000` | 33 | `TOO_MUCH_WORK` | **fails at 1M too** |
  | `MODEL1701170001` | 135 | `FIRST_SRHSFUNC_ERR` | **fails at 1M too** |

  Coupled size does not predict the need: the *smallest* system of the eight (9
  species x 34) is the one no budget rescues, while 37 x 189 clears 100,000 in 5
  seconds. Step count tracks the stiffness a model happens to have, not its
  width. 1,000,000 is not chosen because a higher ceiling is paid for by the
  models it cannot help — a recovering model costs ~nothing extra, while
  `MODEL0911120000` goes 0.2 s → 3.1 s → 32.6 s across the three budgets. The
  per-job `--timeout` remains the real bound on a runaway.

  Re-running the eight end to end: **6 PASS, 2 `REFERENCE_FAILED`**. The last two
  are now an *established* verdict rather than one inherited from a default —
  AMICI's own generated sensitivity RHS returns `NaN` (`sxdot[7]` at t=27.57;
  `sxdot[0]` on the first call), so no budget helps. Recorded in
  `AMICI_KNOWN_ISSUES.md` as Class 3, with the point for triage that a
  `TOO_MUCH_WORK` status alone does not distinguish "budget too small" from
  "cannot converge at all".

- **The sensitivity noise floor now reports what it *rescued*, not what fell
  inside it (issue #316).** #312 added the solver-resolution floor and, with it,
  a per-row `n_noise_forgiven` documented as "the count of cells the floor
  silenced ... so a run can never quietly forgive its way to a PASS". It counted
  `mask.sum()` — every cell *inside* the floor, a strict superset of the cells
  the floor rescued, dominated by cells the two engines already agreed on and in
  particular by cells where both return exactly `0.0`. A run with **zero
  disagreement anywhere** reported every one of its cells as forgiven, and a
  matrix comment reading `noise 4820` said nothing about whether that row would
  have passed without the floor.

  It could not be fixed where it lived: `sens_verdict` built the mask *before*
  calling `differ` and never saw `fail_mask`, so it had nothing to intersect
  against. `differ.deterministic_verdict` now returns two audit keys whenever a
  `forgive_mask` is supplied — `n_forgive_rescued` (cells removed from
  `effective_fail` by that mask **and by nothing else**: a one-side-non-finite
  cell is re-added unconditionally, and a cell the near-zero backstop or the
  dynamic-range gate already forgave owes the mask nothing) and
  `passed_without_forgive` (the verdict recomputed with the mask empty, through
  the same code path rather than a re-derivation of it). Both are *absent* when
  no mask was passed, so "no mask" cannot be misread as "the mask rescued
  nothing".

  A row now carries three numbers, because the audit question needs three:
  `n_below_noise_floor` (how much of the tensor is under solver resolution — the
  old number, under a name that describes it), `n_noise_rescued` (how much of the
  verdict rested on the floor), and `noise_decisive` (whether the floor is what
  made this row a PASS). On the issue's own two demonstrations: exact agreement
  goes 12 → `rescued 0, decisive False`; one genuinely rescued cell among 99
  trivially-agreeing ones goes 100 → `rescued 1, decisive True`.

  `n_noise_forgiven` is **not** kept as an alias. Its meaning would change under a
  name already written into every row of the shipped reports, and a census cannot
  tell which definition a row was written under.

  `noise_decisive` is currently equivalent to `passed and n_noise_rescued > 0` —
  a rescued *soft* cell can only live in a column the significance gate calls
  real, and a column is real only when some cell exceeds `HARD_REL_CEILING`
  (0.05, 500x `REL_TOL`), which is itself a hard fail no budget absorbs. It is
  computed from the recomputed verdict anyway, since that equivalence is a
  property of two constants and a gate ordering rather than of the field; a
  2,000-case fuzz pins it, so a future change to either is a visible event.

  **`parity_checks/tests` now runs in CI**, found while fixing the above. No
  workflow executed it, so the shared oracle every suite's verdict comes out of
  had no gate at all — and a silent change to `differ` does not produce a failing
  sweep, it produces a wrong one. These are the harness's unit tests, not the
  sweeps: ~3 s, no new provisioning, and every dependency they lack they already
  skip for (measured with amici and roadrunner both unimportable: 420 passed, 23
  skipped, 2.9 s). Carries the same two-denominator false-green floor as the main
  suite; `COLLECTED` is 443 whether or not amici, roadrunner and BNG2.pl are
  present, which is what makes that floor meaningful.

- **`amici_parity` exception capture keeps the end of the message, and every row
  carries a stable failure key (issue #324).** A report row's `exception` is
  capped at 400 characters — the report is the durable artifact and 2,646
  unbounded tracebacks are not readable — but the cap was a plain head cut, and
  that is the wrong end to keep.

  Several bngsim refusals enumerate model symbols *before* naming the fault. The
  under-specified-model refusal (#323) reads `Parameters 'A', 'B', ... have no
  value attribute and no initialAssignment, but are referenced by ...`, so on a
  model with a long parameter list all 400 characters were names and the
  diagnostic never appeared. `MODEL0848342500`, `MODEL7980735163` and
  `MODEL9808533471` (892–1173 characters of message each) could not be classified
  from the report at all and had to be read against their source — on a sweep that
  costs ~4 hours at 4 workers to redo.

  The cap now drops the **middle**, marking how many characters went, at the same
  400-character budget. All three models now close with `... Set the parameter
  value or add an initialAssignment. To restore the legacy lenient default-to-0
  behavior, set BNGSIM_ALLOW_UNSET_PARAMS=1.`

  Every row also carries **`exception_class`** — `"<phase>:<ExceptionType>"`, e.g.
  `bngsim-params:UnderSpecifiedModelError`, joined with ` || ` when both engines
  raised. That key is stable across models however long their symbol lists are,
  which is the by-hand grouping the issue describes doing. Deliberately the
  exception *type* rather than a parsed leading clause: a clause is a guess about
  wording, the type is a fact, and it is already the axis `is_declared_refusal`
  matches on. Optional on `JobResult`, so other suites and older reports leave it
  null.

  One consequence worth naming: `reference_refusal` (`feature_gap` / `compile` /
  `integrator` / `other`) is now decided in the worker against the **full** AMICI
  message. It used to be classified in the parent against the capped text, so a
  keyword past the head budget set the subclass only by accident of message
  length — and middle-elision makes the head smaller still.

- **A forward-sensitivity column for an `<assignmentRule>` target is refused
  rather than answered `0.0` (issue #329).** Naming a parameter that a model
  *function* owns — an SBML `<parameter>` an `<assignmentRule>` defines, or a
  `functions` block entry — in `Simulator(sensitivity_params=...)`,
  `compute_all_sensitivities(params=[...])`, or
  `steady_state(sensitivity_params=...)` now raises `ValueError`.

  Such a slot is not a coordinate: the engine rewrites it from the function's own
  expression before every derivative evaluation, so `set_param` refuses a
  value-changing write to it (#227/#266) and — unlike a derived parameter —
  `force_override` does not lift that refusal, because there is no pin the next
  evaluation would respect. Issue #203 had already dropped these from
  `compute_all_sensitivities(params=None)` with a warning saying the column
  "would be identically zero"; naming one explicitly was the remaining route that
  handed that zero back.

  **What the zero cost.** The #328 degeneracy census sampled 20 parameters per
  model straight out of `Model.param_names` and found three assignment-rule-driven
  corpus models with a moving state and an exactly-zero sensitivity tensor.
  `Model.param_names` on such a model is mostly rule targets — 38 of 46 in
  `BIOMD0000000126`, 35 of 38 in `BIOMD0000000266`, 17 of 36 in
  `MODEL1006230116` — so the sample was almost entirely non-knobs, and the
  all-zero result read as a missing chain rule through `<assignmentRule>`, #313's
  `<initialAssignment>` bug one construct over.

  **It was not.** The chain rule is there — a parameter whose only route to the
  right-hand side is an `<assignmentRule>` differentiates to its closed form, one
  rule deep and two — and the three models' real knobs genuinely have zero
  influence, which a finite difference through bngsim's own trajectory settles
  with no reference engine. `MODEL1006230116` is the clearest: its one rate rule
  is `d(Ca_sr)/dt = 1`, a constant, so every derivative is zero while the state
  still spans 100. The `amici_parity` sweep never compared those columns either
  (AMICI lowers the same construct to an expression and drops it from its free
  ids, so the intersection was already 7/1/18 real parameters, not 20); the sweep
  now excludes them on the bngsim side too, so the two engines' eligibility rules
  agree by construction rather than by coincidence.

  The refusal is narrow. `_V0_<comp>` is internal too and stays answerable — it
  really is in the emitted right-hand side, and #203 shipped a test asserting the
  explicit ask returns it — as does a derived parameter, whose column is the
  "derivative on its own terms" a `force_override` pin makes real and what
  `bngsim.jax.differentiable_solve(flat=True)` differentiates over. Only the class
  where no write of any kind moves the coordinate is refused.

- **The zero-base logarithm guard reaches the Jacobian, and a compiled solve
  that fails on a non-finite value now names the rate law (issue #336).** Two
  findings, and the second is the reason the first went unseen for two issues.

  **The guard leak.** #310 and #317 take the limit of `base^exp·ln(base)` at
  `base == 0`, and both apply it inside the SymPy emitters (`sympy_to_exprtk` /
  `sympy_to_c`). #151's native SymPy-free differentiator recognizes `^` and `ln`
  and emits closed forms directly, reaching neither. So `vmax·S^n·ln(S)`
  differentiated natively to `(n·S^(n-1))·ln(S) + S^n·(1/S)` — both halves `0·∞`
  at `S = 0`, where the derivative's own limit is `0`. That expression *is* the
  Jacobian entry and the `J·yS` half of the analytic sensitivity RHS, which is
  why #336's reproducer failed on its first call, at the exact zero, for **every
  parameter** — including `kdeg`, which appears only in a rate law with no
  logarithm. A logarithm of a state-dependent argument now defers to SymPy,
  where the guard applies; a logarithm of a constant (`ln(KM)`) stays native and
  costs no derivation budget. The reproducer's `d(species)/dn` is now
  `[5.0e-25, 0, 194.65]` against the `A(0)=1e-30` control's `194.65`, and #317's
  end-to-end test drops its strict `xfail`.

  This also corrects #338's diagnosis of the same reproducer. The evidence there
  was that `ln(abs(S))` makes the run complete — true, but not because `abs`
  finitizes a negative argument: `abs` is not in the native differentiator's
  whitelist, so spelling it that way pushed the law onto SymPy and picked up the
  guard.

  **The missing diagnostic.** A rate law that genuinely leaves its domain still
  answers `nan` (#310's contract), and on the compiled path the user learned
  nothing: the emitted C calls libm's `log` directly, with none of issue #42's
  instrumentation the interpreted ExprTk adapters carry — and a
  forward-sensitivity run *forces* codegen. `CVODE integration failed at t=... with
  flag=-4` now reads:

  > CVODE integration failed at t=1.000000 with flag=-4 (CV_CONV_FAILURE). The
  > compiled RHS returned a non-finite value at t=1. Non-finite there:
  > `logterm() = ln(1 - Atot)` -> nan. Species below zero there: `C() = -1`. …

  No finiteness test was added to the generated C. The callbacks already scan
  their output for non-finiteness (#135's nonnegative-clamp retries), so the
  first scan that trips with no clamp left to try keeps `(t, y)` as a witness,
  and only a run that *fails* replays that state through the interpreted
  evaluator — which does carry the instrumentation, so the model's own
  `'ln(-1e-09)' returned nan` appears alongside. A run CVODE recovers from stays
  silent. The one addition to a healthy hot path is an `isfinite` scan of the
  compiled sensitivity RHS's output, which has no clamp retry of its own;
  measured on `fceri_fyn` (1281 species, 4 sensitivity columns, 5 runs) at
  6.33 s median with it and 6.32 s without — below the noise floor. Numerics are
  unchanged: no callback's return value moved.

- **A grazing-but-transversal event crossing is now computed instead of refused
  (issue #322).** The transversality guard protecting `dt*/dp = -num/flow` had
  two arms. The first — near-total cancellation of `flow`'s own terms — is the
  real definition of non-transversality and is kept. The second was an absolute
  floor, `eps * sum|dg/dx| * ||f||inf`: a *valid* roundoff bound, but loose by
  `||f||inf / |f_support|` because it ranged over every state including ones the
  trigger never touches. One fast species therefore poisoned the test for events
  on every slow species.

  On `BIOMD0000000711` — a non-negativity clamp on a viral load that decays to
  ~0 — the true roundoff is ~3e-29 while that bound computed 2.22e-12, off by a
  factor of ~1e16, and it refused a crossing whose derivative is perfectly
  computable. With the arm removed the model runs and the answer is right:
  analytic **3.17424e6** against a finite difference of **3.17411e6**, agreeing
  to 8.6e-4 over 700 points.

  Removing it is safe because `dt*/dp` is never consumed alone: every use in the
  jump multiplies it by a flow term (`(f⁻-f⁺)·tau`, `f⁺·tau`) that carries the
  same smallness, so a grazing crossing cancels back to a finite jump. Where it
  genuinely does not cancel, a new post-jump finiteness guard refuses — at the
  point where the overflow is observed, rather than predicted from a proxy on an
  intermediate.

  `BIOMD0000000285` still refuses, now for its actual cause rather than a
  mis-scaled floor: two events whose trigger species both start at exactly 0
  fire spuriously at t=0.

- **`bngsim.UnderSpecifiedModelError` — an under-specified model is a declared
  refusal, not a bug (issue #323).** When a `<parameter>` has no `value` and no
  `<initialAssignment>` yet is referenced by a kineticLaw / rule / event, bngsim
  refuses rather than default it to `0.0`. That refusal was a bare `ModelError`,
  which also covers `.net` parse failures and invalid model state, so a caller
  could not tell a documented refusal from an actionable bug. The `amici_parity`
  suites scored **12 corpus models** as `EXCEPTION` for it; they are now
  `UNSUPPORTED`, matched by type, in **both** the sensitivity and state-parity
  jobs (the refusal is raised at load, so it reaches both).

  Confirmed with `BNGSIM_ALLOW_UNSET_PARAMS=1`: with the hatch on, **none of the
  12 is an `EXCEPTION`** — 6 agree with AMICI at `max_rel_err=0`, 5 fail in both
  engines (3 with the identical first-RHS-evaluation error), and 1 runs in
  bngsim where AMICI cannot. The strictness is the only difference.

  The default stays strict, and `MODEL1006230032` is why: under the hatch it
  integrates to **all five species identically zero for the whole run**, with
  every cross-engine verdict green. A model computing nothing, reported as
  agreement.

  As with issue #320, typing it exposed re-wrapping: `Model.from_sbml`,
  `from_sbml_string`, `from_antimony` and `from_antimony_string` all rebuilt it
  as a generic `ModelError`, so the type a caller saw depended on which entry
  point they used. All four now pass it through. It still subclasses
  `ModelError`, so existing handlers are unaffected.

- **A declared switch-time refusal is now typed, and `run()` no longer launders
  it (issue #320).** `compute_switch_time_sens` refuses a parameter that both
  sets an `if()` switch time and acts inside a branch, because the crossing jump
  is then not the whole gradient (issue #48). That refusal was a bare
  `ValueError`, so the `amici_parity` sensitivity sweep scored it `EXCEPTION` —
  "AMICI ran and bngsim broke" — where it accounted for **26 of 77 rows across 13
  corpus models**. It now raises `SensitivityUnsupportedError`, and the sweep
  buckets it `UNSUPPORTED`, matched by type.

  Typing it exposed a second defect. `run()` wraps `RuntimeError` raised in the
  solve region into `SimulationError`; the refusal had been escaping only because
  a plain `ValueError` is not a `RuntimeError`. Adding the `BngsimError` base
  silently routed it *into* that wrapper, so the newly-typed refusal reached
  callers as a generic solver failure — destroying the distinction it had just
  been given. `run()` now passes `SensitivityUnsupportedError` through unchanged,
  as it already did for `SimulationTimeout`. Only the switch-time site raises
  from inside that `try` (the event and codegen refusals are raised during
  setup), which is why one of the three typed sites regressed and the other two
  did not.

- **`amici_parity`: AMICI's `amici_` id-collision prefix is now undone before
  species alignment (issue #321).** AMICI renames SBML ids that collide with its
  own generated C++ symbols — `x` *is* the state vector there — so
  `<species id="x">` comes back as `amici_x`. `align_common` intersected ids
  exactly, so the intersection came out empty and the job was reported as a
  structural loader divergence at `value=inf` — the loudest verdict the suite
  emits — on models where the two engines agree exactly.

  `BIOMD0000000114`, `115`, `346` and `919` now all `PASS` at `max_rel_err=0`, on
  **both** the sensitivity and the state-parity job: `align_common` is shared, so
  the ODE matrix carried the same false DIFFs.

  The de-prefixing lives in the AMICI adapter rather than the shared
  `_rr_common` helper, because it is AMICI's naming rule and RoadRunner never
  emits it. It follows the same discipline as bngsim's `_lp_` local-parameter
  prefix: positional (leading occurrence only), and an ambiguous strip is dropped
  rather than guessed at — mis-pairing two species would yield a
  confident-looking DIFF, which is worse than the loud non-comparison it
  replaces.

- **A codegen BUILD failure is no longer reported as a non-differentiable
  model.** The model-path codegen entry points (`prepare_model_codegen`,
  `prepare_model_codegen_source`) catch every exception and return `None`,
  because most callers want "no codegen, fall back to the interpreted RHS"
  rather than a traceback. The forward-sensitivity path read that `None` as a
  single condition and raised "its rate laws could not be differentiated to
  closed form" — a cause it had never checked.

  `BIOMD0000000608` is the case that found it. Its rate laws differentiate
  perfectly: codegen generated **66.6 MB** of C and then blew the 600 s compile
  budget on it. The user-facing advice was therefore exactly wrong — the fix is
  more cores, `BNGSIM_CODEGEN_JOBS`, or `BNGSIM_CODEGEN_TIMEOUT`, not rewriting a
  smooth rate law — and the original exception was discarded along with its
  traceback.

  The two entry points now record why they returned `None`
  (`bngsim._codegen.last_codegen_error()`, cleared on entry so it is never
  stale), and the sensitivity path asks before it refuses: a recorded cause
  raises the existing "Failed to build the analytical sensitivity RHS" envelope,
  chained with `raise ... from`, and only a `None` with no cause is a declared
  refusal. This matters more now that declared refusals are non-scoring in the
  parity taxonomy — misattributing a resource failure would have made it
  disappear from the tally rather than merely mislabel it.

- **A parameter that reaches the model only through an `<initialAssignment>` is
  no longer frozen (issue #313).** bngsim evaluates every `<initialAssignment>`
  once at load and hands the builder a number, so a parameter whose only route
  into the model is that expression became a dangling constant that nothing
  depended on. `set_param` on it succeeded and changed nothing, and its forward
  sensitivity was **identically zero** at every species and every time — not
  merely inaccurate, since the symbol was absent from the RHS the chain rule
  differentiates.

  `BIOMD0000000569` is the case that found it, via the new `amici_parity`
  sensitivity job. The document defines a chain — `BSk0 = BSk1*BSc^p`,
  `BSk1 = BSk2*BSc^p`, … — puts the `BSk*` constants in the rate laws, and
  mentions `BSc` nowhere else, so a 50% write to `BSc` moved none of the four
  derived constants. It is not one document's quirk: 113 of the 1324 vendored
  BioModels (8.5%, 678 parameters) have at least one such parameter, because it
  is the ordinary COPASI spelling for a derived rate constant.

  The fix is issue #170's lift, applied to every parameter `<initialAssignment>`
  rather than only the volume-dependent ones: the target becomes a *derived*
  parameter, so #43's chain rule re-derives it on a write and differentiates
  through it for the sensitivity. Two consequences are worth naming. Lifted
  targets are emitted in **dependency order** rather than document order —
  derived parameters are re-evaluated in one pass over the parameter list, and
  569 declares `BSk0` before the `BSk1` it reads, so lifting in place would leave
  `BSk0` reading a stale `BSk1` for one write. And a lifted target drops out of
  `primary_param_names`, exactly as `_rateLaw_<rid>` does: the document defines
  its value, so it is not a knob a caller can hold.

  An `<initialAssignment>` over an **assignmentRule target** is refused rather
  than lifted, on both sides of the assignment. The rule already disqualified
  such a parameter as a lift *target* — the fold is only its t=0 value and the
  rule, not the expression, is what a write has to survive — and the same fact
  disqualifies it as a lift *dependency*: its slot is function-backed, rewritten
  from the rule before every derivative evaluation, so a lifted expression
  reading it would re-derive from whatever that slot held at write time. After a
  run that is the rule's value at the last integrated point. `BIOMD0000000570`
  is the shape (`ModelValue_60 = O2c_bar`): lifted, it went 5.68 → 7.87 on the
  next write after a run, including the identity write
  `set_params(dict(zip(param_names, vec)))` round-trips through. 22 corpus
  models carry it. The cost is three compartment sizes — `compartment_2` and
  `compartment_3` of 570, `artery` of `BIOMD0000000627` — whose folds can no
  longer be put back on the size and which are therefore refused by name, which
  is #170's contract for exactly this residue.

  Nothing moves at the nominal point — the builder is still seeded with the
  folded number, and a derived parameter is not re-evaluated until a write
  arrives. Across the 1323-model corpus (101 models, 598 parameters lifted),
  load-time parameter values, species initial conditions and trajectories are
  bit-identical, with one exception: five parameters of `BIOMD0000000833` move by
  one ulp, because a lifted expression is evaluated by ExprTk rather than by the
  loader's numeric folder and the two associate `0.1*150*u/0.9` differently. That
  is 7.6e-6 relative at the end of its trajectory — under the parity suite's 1e-4
  gate, and the only trajectory in the corpus that moves at all.

  What still cannot be lifted — an `<initialAssignment>` reading a species, a
  reaction rate, `time`, or a symbol some rule or event promotes to a state — is
  no longer silent about it. The loader warns once, naming the parameters whose
  contribution through that assignment stays frozen at its load-time value.

- **Forward sensitivity w.r.t. a power/Hill exponent no longer NaNs when the
  base is zero (issue #310).** Differentiating `base^n` with respect to the
  exponent gives `base^n · ln(base)`. At `base = 0` — five of the six species in
  `BIOMD0000000012` start there, and an unset initial condition is the ordinary
  case at `t = 0` — that evaluates as `0 · (-inf)` = `NaN`, even though the limit
  exists and is `0` for every `n > 0`.

  One NaN there is not a local blemish. It enters `∂f/∂p` on the first step, and
  from there either poisons that parameter's whole sensitivity column while every
  other column reads fine, or, when the exponent is the only column, defeats the
  corrector and fails the solve outright (`flag=-4` at `t = 0`). AMICI returns
  the limit; central finite differences on bngsim's own state trajectories agree
  with AMICI to ~5 significant figures. The new `amici_parity` forward-sensitivity
  job is what surfaced it.

  `bngsim._jacobian` now rewrites the `base^exp · log(base)` form to its limit at
  a zero base. It is one symbolic pass applied by both emitters on their way out,
  so the interpreted ExprTk path and every codegen backend (cc / MIR compile the
  same C) get the same value from one place, rather than a guard replicated per
  backend. The guard is not a blanket `base == 0 → 0`: the limit is `0` only for
  a **positive** exponent, so a numeric non-positive exponent is left alone and a
  symbolic one is decided at run time against its current value — where the limit
  does not exist, the NaN is the honest answer and survives. A negative base is
  untouched for the same reason.

  This also resolves the one column GH #243 could only report on: BIOMD0000000044's
  `_lp_v7_n` is a Hill exponent, and it was unresolvable at every `chunk_size` for
  exactly this reason. All 23 of that model's columns now integrate, agree across
  chunk sizes, and match finite differences. The #243 machinery — retry a failed
  chunk column by column, refuse a non-finite column instead of returning it —
  is unchanged; it guards a property of chunking, not of that one model.

- **The committed type stub no longer snapshots the version it was built from.**
  `python/bngsim/_bngsim_core.pyi` is machine-written by `pybind11_stubgen`,
  which copies whatever the freshly built module reports — including the
  `__version__` CMake stamps from `pyproject.toml`. Committed as a literal, that
  line is a fifth version anchor that nothing derives and nothing updates: a
  release bumps `pyproject.toml`, no one rebuilds, and the stub goes on naming
  the previous release. 0.13.0 shipped with the stub still reading `'0.12.2'`,
  and the drift surfaced only as a phantom one-line diff in the working tree of
  the next person to run `scripts/rebuild_editable.py` — noise from the build
  script, not a change anyone made.

  `_normalize_stub_build_stamps` now pins `__version__` to `'unknown'` the same
  way it already pins `__build_commit__` and `__pybind11_version__` (PR #70, GH
  #288), so a rebuild is version-neutral and no release step has to remember the
  stub. Nothing reads the value — mypy checks the declared *type*, and a stale
  version is worse documentation than none. The invariant that matters is
  unchanged and still enforced where it can be true: `test_version_consistency`
  holds the *runtime* `_bngsim_core.__version__` to `pyproject.toml`, and a new
  test there fails if a version literal ever lands back in the committed stub.

## [0.13.0] - 2026-08-11

### Added

- **`bngsim.HAS_LAPACK_DENSE`, and a Linux CI leg that actually runs the BLAS
  dense solver (issue #269).** The optional `dgetrf` dense factor (GH #84) had
  no supported capability probe and, after #265, no Linux coverage at all.

  The probe first, because the CI guard needed it: the flag existed only as
  `bngsim._bngsim_core.HAS_LAPACK_DENSE`, so the one test file that gated on it
  reached into the private extension module with a `getattr` default. It is now
  `bngsim.HAS_LAPACK_DENSE`, in `__all__`, mirrored by
  `capabilities()["features"]["lapack_dense"]` with a `missing` entry that names
  a concrete rebuild path. `False` costs speed on large dense Jacobians and
  nothing else — the built-in LU is the default on every platform and the BLAS
  path is opt-in via `BNGSIM_LAPACK_DENSE=1` — so the flag answers "will that
  variable do anything here?". Of the published wheels only the macOS ones say
  yes; the manylinux and Windows legs resolve no LAPACK. The population this
  backend actually reaches is source installs on a LAPACK-equipped Linux box.

  That population had never run a CI job. `python-tests.yml` built the backend
  on Linux only as a side effect of `libsuitesparse-dev`, which #265 removed on
  purpose (it is what makes that leg the bare host an sdist install lands on),
  and `native-tests.yml` builds on a stock `ubuntu-latest` where
  `find_package(LAPACK QUIET)` finds nothing. Two new legs, neither touching the
  bare-host proof:

  - `native-tests.yml` gains a `blas: liblapack` matrix entry. It configures
    `-DBNGSIM_ENABLE_KLU=OFF`, so it never reaches the SuiteSparse autobuild and
    cannot weaken #178 by construction. The stock entry stays — that is the
    configuration the manylinux and Windows wheels ship.
  - `python-tests.yml` gains a third leg, `ubuntu-latest` + `liblapack-dev` and
    nothing else. Still no SuiteSparse, so it still exercises
    `BNGSIM_KLU_AUTOBUILD`. The bare leg is untouched.

  Both assert the backend they expect, in both directions: a leg that quietly
  loses its BLAS fails, and so does a bare leg that quietly gains one.

- **An absolute tolerance that follows the trajectory: `atol="tracking"` /
  `bngsim.TrackingAtol` (issue #213).** #196 gave the state axis a per-species
  `atol` and removed the *cross-species* compromise. It left the *within-species,
  over-time* half untouched: whatever number species `i` gets, it keeps for the
  whole run, so a species that starts at order one and decays to something tiny
  outgrows its own tolerance partway through and stops being error-controlled
  from there on. CVODE's construct for that is `CVodeWFtolerances`, an
  error-weight function evaluated at the state actually being integrated, and
  this is that third mode. The rule:

      atol_i(y) = clamp(rtol*|y_i|, ceiling_i * 10**-decades, ceiling_i)

  `ceiling` is the #196 vector (`"auto"` by default), so the mode is a strict
  extension of it: at `decades=0` it reduces to that vector exactly, and the
  clamp's upper end means tracking is never *looser* than the vector it was
  built from — only tighter, by at most `decades`.

  New `tests/data/deep_decay.net` is the reproducer, and it is built so the
  #196 vector is provably a no-op on it: both live species start at exactly
  1.0, so `atol="auto"` derives 1e-8 for every species — the number the scalar
  it replaces would have used. One then decays sixteen decades. Worst relative
  error in that species against the analytical `exp(-t)`:

  | mode                        | rel. err. | steps |
  |-----------------------------|----------:|------:|
  | scalar `atol=1e-8`          |   1.8e+04 |   162 |
  | `atol="auto"` (#196 vector) |   1.8e+04 |   162 |
  | `TrackingAtol(decades=3)`   |   3.6e+01 |   275 |
  | `TrackingAtol(decades=6)`   |   9.5e-02 |   366 |
  | `TrackingAtol()` (12)       |   2.6e-06 |   605 |
  | `TrackingAtol(decades=20)`  |   9.4e-06 |   477 |

  The default depth of 12 is where the accuracy stops improving; past it the
  limit is roundoff rather than the tolerance, which is why it gets slightly
  *worse* at 16 and 20. The cost is real and is not hidden: roughly 4x the
  steps on this model.

  Opt-in and orthogonal to #196 — a scalar `atol` stays on `CVodeSStolerances`
  and a vector on `CVodeSVtolerances`, unchanged. Accepted by `Simulator.run`,
  `run_batch`, `parameter_scan`, `compute_all_sensitivities`, `steady_state`,
  `steady_state_batch` and `set_tolerances`, and frozen once per batch/scan the
  same way `"auto"` is. `hasattr(bngsim, "TrackingAtol")` is the capability
  probe.

  The **sensitivity** axis is re-decided rather than left to inherit: `atolS`
  keeps reading the *ceiling*, so turning tracking on leaves the sensitivity
  tolerances exactly where the same vector would have put them. Re-deriving
  `atolS` from the live state would make that base *tighten* mid-run, which is
  the step-controller hazard the issue #183 high-water mark exists to avoid.
  Measured on the same fixture, that costs nothing: the sensitivity column's
  worst relative error against `-t exp(-t)` goes from 2.4e+02 to 1.8e-06 under
  tracking anyway, because the state axis is what drives the step size.

  **What it costs on real models**, swept over the first 400 `rr_parity` SBML
  models (391 of which integrate at the default tolerance over `t = 0..100`).
  37 of those 391 have a species that falls six or more decades below its own
  initial value, so the mechanism is not exotic:

  | arm | integrates | lost | gained | steps vs default |
  |---|---:|---:|---:|---:|
  | default (control) | 391 | — | — | 1.00 |
  | `decades=3` | 393 | 0 | 2 | 1.18 |
  | `decades=6` | 392 | 1 | 2 | 1.24 |
  | `decades=12` | 387 | 6 | 2 | 1.58 |

  Six models that integrate at the default tolerance do **not** at the default
  depth — a species whose value is a difference of large fluxes cannot be
  resolved below its own roundoff, and the step size collapses instead. Those
  failures are loud, not silent, and a solver failure under tracking now names
  the depth and suggests lowering it, because CVODE's own report ("made no
  progress", `flag=-3`) mentions nothing about the tolerance mode that caused
  it. Two models go the other way and integrate *only* under tracking.

  The control arm is the important row: the same 391 models, run with the
  default scalar `atol` before and after this change, produce **391/391
  byte-identical trajectories with identical step counts**. Nothing moves
  unless it is asked to.

  **On `Bruno_JExpBot2016`**, which #213 cites as the measured case, the
  premise does not reproduce. The issue reports the objective moving 1.3e-05
  between `atol=1e-8` and "a converged tail near 1e-16". There is no tail:
  reconstructing the shipped job's six conditions and sweeping both axes, the
  worst deviation of any observable from a `rtol=1e-12, atol=1e-16` reference
  is 7.3e-08 at `atol=1e-8` and 2.4e-08 at `atol=1e-16` — both at the `rtol=1e-8`
  truncation floor, which is where the sequence stops. Drop `rtol` to 1e-10 and
  the whole atol axis below 1e-10 collapses onto 5e-10. The reason is
  structural: every live species in that model stays between about 0.2 and 6.4
  for the whole run, so none of them ever goes under its own `atol`. Tracking
  moves it by 7.9e-08, i.e. into the same band — which is the right answer for a
  model that has nothing for it to fix. The mechanism is real; that model is not
  an instance of it, and `tests/data/deep_decay.net` is.

- **The per-species `atol` derivation a caller needs is now public (issue
  #212).** #196 shipped the capability and exported the wrong half of it.
  `Simulator.auto_atol()` — which derives from the model's **live** state — was
  public; the stateless `derive_atol(state, rtol, floor=...)`, which derives
  from a state *you* hand it, was in the private `bngsim._atol`, along with the
  `AUTO` token. That is backwards for the primary consumer. A parameter fit has
  to derive from the model's **nominal** state and hold the vector constant: a
  tolerance that moved with a fitted initial condition would put a step in the
  objective everywhere the derivation crossed a rounding boundary, and it fails
  invisibly — the objective still looks correct, the finite-difference gradient
  check still passes, and only the search behaves oddly. So the function a
  fitting frontend needs was the one it could not reach, and the one it could
  reach was the one that would break it. `bngsim.derive_atol`,
  `bngsim.normalize_atol_vector` and `bngsim.AUTO` are now exported from the
  package namespace and listed in `__all__`.

  `hasattr(bngsim, "AUTO")` is consequently a working capability probe for the
  whole feature. It previously returned `False` on a build that *has* the
  capability, which is the worst failure mode a probe has — it silently routes a
  capable install down the scalar fallback and the result still looks right. The
  version string is not a substitute: the checkout that first carried #196 still
  declared `0.12.2`, the version of the wheel 25 commits behind it.

  `normalize_atol_vector` went public rather than staying private on a narrower
  argument: every `atol=` entry point already runs a vector through it, so
  calling it yourself changes only *when* the error arrives — and for a caller
  that assembles its own vector (per-species clamping, a table, a nominal
  state), taking the length-and-position check once at setup beats taking it at
  the first `run()` of a fit. `is_scalar_atol` and the `AtolLike` alias stay
  internal.

  No behavior changed and nothing was renamed; these are re-exports of the same
  objects. The docs gained the distinction that motivated all of it: which state
  each derivation reads, and the recipe for holding one constant across a fit
  (`docs/user-guide/solvers.md`, "Which state the tolerance comes from").

- **`atol` takes one value per species, so a model spanning decades has a usable
  tolerance (issue #196).** `Simulator.run` took a scalar and the core set it
  with `CVodeSStolerances`. For a model whose species span ten decades that is
  one number asked to mean two incompatible things, and there is no value that
  satisfies both ends: the tolerance the smallest species needs makes the model
  unintegrable, and the tolerance the model can integrate at leaves the smallest
  species unresolved. `Brannmark_JBC2010` is the case the issue was filed from —
  `IRp` at 1.8e-09 wants an `atol` of 1.8e-17, `X` at 1.0e+01 wants 1.0e-07, and
  the shipped job completes at 3.3e-10 and times out at either 1e-16 or 1.8e-17.

  `atol` now also accepts a sequence of `n_species` values ordered like
  `Model.species_names`, routed to `CVodeSVtolerances`. A float still goes to
  `CVodeSStolerances` on the identical code path, so every existing call is
  bit-for-bit unchanged. The vector is positional and its length is *checked*:
  a mismatch raises rather than broadcasting or truncating, since the
  alternative hands species *i* the number written for species *j* and returns a
  plausible-looking trajectory. Accepted on `run`, `set_tolerances`, `run_batch`,
  `parameter_scan`, `bifurcate`, `run_until`, `compute_all_sensitivities`,
  `steady_state`, `steady_state_batch` and `EvaluationSpec`; a call that runs
  many points resolves it once, so the points of a batch or scan can be compared
  with one another.

  `atol="auto"` (and the public `Simulator.auto_atol()`, which returns the same
  array for inspection or adjustment) derives `rtol * max(|y_i|, floor)` from the
  model's own state — the heuristic every caller was otherwise reimplementing.
  `floor` defaults to the smallest strictly positive species value: a species
  sitting at zero has no magnitude of its own, so it is treated as living at the
  smallest scale the model exhibits.

  Two consumers of `atol` follow the vector rather than the scalar, both
  per-species by nature: the GH #214 sensitivity floor, so `atolS` for ∂x_i/∂θ
  is built from species *i*'s own tolerance instead of collapsing the vector back
  onto one number, and the GH #95 event chatter guard. `steady_state_tol` keeps
  reading the scalar — `||f(t,y)||₂ / n_species` is one norm over every species
  and has no per-species reading to take.

  The new `wide_dynamic_range.net` fixture is the smallest reproducer of the
  half that reproduces from a model alone (the issue is explicit that the
  *unintegrable* half needs a full pre-equilibration protocol): two decoupled
  decays nine decades apart, where the default `atol=1e-8` leaves the small
  species uncontrolled for the whole run and it comes back **negative** where
  the analytical answer is 6.7e-12. What this does not fix, and is not sold as
  fixing: a species that starts at order one and decays to something tiny is a
  within-species, over-time mismatch that no initial-value vector can see —
  that one wants `CVodeWFtolerances`.

  One caveat worth stating: a *constant* vector is not bit-identical to the same
  scalar. `cvEwtSetSS` scales then adds a constant while `cvEwtSetSV` takes one
  fused `N_VLinearSum`, and only the second is FMA-contractable, so the two agree
  to about one ulp in the error weights. Passing a float keeps the old path
  exactly; it is asking for the vector spelling of the same number that costs the
  ulp. SED-ML export refuses a per-species `atol` outright rather than writing
  one entry as `KISAO:0000211`, which would describe a different run.

- **A cross-platform Python test gate (issue #169).**
  `.github/workflows/python-tests.yml` runs the whole of `python/tests` on
  `ubuntu-latest` and `macos-14` in the **default** build configuration. Before
  it, GitHub CI ran 533 of the suite's ~3490 Python tests on any non-Windows
  host — and every one of them under `BNGSIM_CODEGEN_JIT=mir` on a
  `-DBNGSIM_ENABLE_MIR=ON` build, so the count exercised in the configuration a
  wheel actually ships was **zero** on Linux and macOS. A regression in SBML
  loading, `.net` parsing, events, steady state, SSA, conversion or coupling
  could be caught only on Windows, and only if the file happened to be named in
  one of two hand-maintained lists.

  The gap was structural, not a bug. Every job that runs pytest names a curated
  file list, so a *new* test file defaults to running nowhere, and the assumed
  backstop was the local pre-push hook — which covers whichever platform the
  developer happens to be on (here macOS arm64) and which `git push --no-verify`
  removes. `native-tests.yml` stated that assumption in its header as fact.

  The new job therefore carries no `paths:` filter (a selectively-firing gate
  reintroduces the per-file opt-in through the trigger instead of the run list),
  no file list (it runs the directory, the way `native-tests.yml` drives `ctest`
  rather than one named target), and no `-D` overrides (every other workflow
  disables something — KLU off in four of them, MIR *on* in `mir.yml` — which is
  how the shipped configuration ended up untested). Provisioning is `uv sync
  --extra dev` off `uv.lock` and the pytest call is the pre-push hook's own, so a
  green run means what a clean `git push` means locally, on a host the developer
  does not have.

  Two false-green guards, mirroring the ones `native-tests.yml` added for the C++
  suite: a floor on the passed count, because a module that stops importing skips
  at collection and shrinks the denominator without failing anything; and
  `BNGSIM_SKIP_AUDIT=strict`, because a test that quietly turns into a skip for
  an undeclared reason is the same invisibility in a different form. `HAS_KLU` is
  asserted after the build for the reason `mir.yml` asserts `HAS_MIR` — a build
  that silently lost KLU would skip the sparse-solver tests and still be green.

  Side effect worth naming: the macOS leg is the only place anywhere that
  exercises `BNGSIM_KLU_AUTOBUILD` (GH #209). The wheel legs all resolve a
  prebuilt SuiteSparse through `SUITESPARSE_ROOT` and every other job sets
  `ENABLE_KLU=OFF`, so the from-source KLU subset that an sdist install on a bare
  box falls back to had no CI at all. Wiring it up immediately showed why that
  matters: the same autobuild **cannot** complete on a bare `ubuntu-latest`,
  because SuiteSparse's own CMake calls `find_package(BLAS)` and the runner image
  ships none, so the configure dies in `SuiteSparse_config` before KLU is
  reached. So GH #209's self-sufficiency claim holds on macOS but not on a bare
  Linux host. The Linux leg therefore installs `libsuitesparse-dev`, the same
  system-package route cibuildwheel's Linux leg takes.

  And the gate earned itself on its first real run: **four tests fail on Linux
  that pass on macOS**, in two unrelated subsystems, both pre-existing on `main`
  and both exactly the class #169 said nothing could see. GH #176's
  finite-difference retry fires correctly on
  `ltype_calcium_discontinuous_jacobian.net` and then dies at a *second*
  threshold crossing (t≈34.6) the fixture's own header does not mention; and
  `nested_derived_rate_const.net`'s reduced Jacobian is exactly singular under
  Linux's reference LAPACK where Accelerate leaves it merely ill-conditioned, so
  it takes the refusal branch its sibling test exists to assert rather than the
  warning branch its own test asserts. Both are quarantined under
  `xfail(sys.platform.startswith("linux"), strict=True, raises=SimulationError)`
  and reported in lanl/bngsim#176 — `strict` so they retire themselves, and
  quarantined at all so the new gate lands green rather than permanently red,
  which is the distinction `native-tests.yml`'s header spells out.

- **`compartment_sizes=` at load, the supported way to change a volume (issue
  #164).** `Model.from_sbml`, `from_sbml_string`, `from_antimony`,
  `from_antimony_string`, and `Model.load` take
  `compartment_sizes={"Liver": 2.5}`, applied to the parsed document before
  bngsim interprets it — so the size reaches every constant it is folded into,
  and the result is bit-identical to loading a source that carries that `size=`
  outright. A volume scan or a fit over a volume is a loop over such loads, and
  its gradient is a finite difference over two of them. An `initialAssignment`
  on an overridden compartment is dropped (it would otherwise take precedence);
  a compartment whose size an *assignment rule* computes is refused, since the
  rule and not the attribute is its volume; `.net` is refused for having no
  compartment left to set.

- **`Model.compartment_size_params`** — the parameter names the two rules above
  apply to, so a fitting harness can build its vector from what is writable.

- **Analytic forward sensitivities for cross-compartment reactions (issue
  #160).** A reaction whose affected species live in compartments of different
  size (`per_species_volume_scaling`) used to decline the analytic sensitivity
  RHS for the **whole model** — `CVodeSensInit1` installs one callback for every
  column, so a single such reaction put every column on CVODES' internal
  difference quotient. The fallback was correct, so this is a cost fix, not a
  correctness one.

  The missing piece was never a derivative: a cross-compartment kinetic law
  evaluates to amount/time while each affected species stores amount/V_c with a
  V_c of its own, so every accumulation row divides by its own compartment
  volume — and the `∂f/∂p` scatter had no form for that divide. It has one now,
  taken from the same `_psvs_row_divisor` lookup the RHS scatter (which defines
  the divide) and the `J·yS` half already share, folded into the scatter
  coefficient for a static compartment and emitted as a runtime divide by the
  live volume for a variable one (issue #171). Both halves of
  `ySdot = J·yS + ∂f/∂p` are therefore analytic for these models.

  Measured over the two SBML corpora plus the `.net` corpus (1,380 models):
  **21 models gain an analytic sensitivity RHS** (`benchmarks/sbml_events`
  177 → 198), none lose one, and every other model's emitted RHS, analytical
  Jacobian and sensitivity RHS is byte-identical. Where the difference quotient
  used to converge the two agree to 8.3e-06 relative over 65 sensitivity columns
  on 14 real models; where it did not, the step counts are the story —
  `BIOMD0000000706` integrates 460 steps against the difference quotient's 23.0
  million.

  Elementary and Michaelis–Menten reactions carrying the flag still decline (and
  now say so): their `J·v` comes from a scatter that has no per-row divide, so
  half a divide would be worse than the fallback. No loader emits one.

### Changed

- **The GH #88 periodic step bound is now derived only where the discontinuity
  roots provably cannot reach (issue #274).** #262 measured that no corpus model's
  answer the bound changes; this acts on it. The bound and the GH #72/#231/#259
  roots address the same hazard — an integrator step spanning a schedule edge —
  and where a root already forces the stop, `max_step` is pure cost: it shortens
  every step over the whole horizon, not just the ones near an edge.

  The test is value-position reachability from the ODE RHS through the assignment
  rules, pruning every piecewise condition that IS a registered root
  (`_periodic_disc_escapes_roots`). A `floor`/`ceiling`/`modulo` influences the
  RHS two ways and only one of them is rooted: through a piecewise **condition**
  (`exposure = piecewise(D, frac < w, 0)` — a root on that condition forces the
  stop, so the bound adds nothing), or as a **value** (`dose = D * frac`, or
  `k = k0 * i` with `i = floor(time/24)` — the sawtooth jump is an RHS
  discontinuity in its own right, at an instant no relational brackets). Only the
  second keeps the bound.

  On the 25 corpus models that carried one: kept on 10, dropped on 15, cutting
  1,015,702 internal steps to 519,472 (**-49%**) with no answer moving —
  `MODEL0406553884` -60%, `MODEL0406793751` -44%, `MODEL0847869198` -40%,
  `MODEL1708310001` -29%. That last one is the check that matters: it is the model
  #259 gave the roots to, with an exact segmented oracle (953.07) confirming they
  resolve its schedule, and the predicate drops its bound without being told.

  **Not a root count.** The tempting predicate — "apply the bound only when the
  model registered no roots" — is unsound, and measurably so: three corpus models
  (`BIOMD0000000577`, `BIOMD0000000589`, `MODEL1006230027`) are rooted yet carry a
  periodic disc node reaching the RHS ungated. `BIOMD0000000589` is the clearest,
  with 15 of its 16 schedule conditions of the form `i == 0`, `i == 1`, … on
  `i := floor(time/24)`: **equalities**, which the inequality-only emitters cannot
  root, on a step function that genuinely moves with time.

  A new fixture makes that concrete, and is the first thing in the suite to
  witness the bound's *necessity* since #259 removed the old one — the reason this
  is a narrowing rather than a retirement. Its pulse is gated by an equality on a
  floor while an unrelated `time < 5` threshold is rooted, so the model has roots
  and still needs the bound: with it, `y(10) = 36.788` against an exact 36.788;
  without, 182.212 — 395% high, identically at rtol 1e-9 and 1e-11. Tol-stably
  wrong is the signature of a pulse never sampled at all. A root-count predicate
  returns 182.212 here.

  `Model._periodic_disc_max_step` is `None` for the 15 dropped models, so anything
  reading it now reads "the bound this model needs" rather than "a periodic
  schedule was detected". `_periodic_time_disc_max_step` keeps its pre-#274
  behaviour when called without the registered-condition set, so a caller that has
  not run the root scan cannot accidentally drop a bound it has no basis to drop.

- **The build-time derivation now declines before differentiating what it could
  never emit (issue #250).** `BIOMD0000000385` spent **138 s** against a 20 s
  budget deriving a Jacobian the emitter then refused — a 6.9x overshoot the
  budget could not bound, because the deadline is only testable between `sp.diff`
  calls and this was one call.

  The issue proposed subdividing the derivation to make the deadline reachable.
  Profiling says that would not have worked: recursing that rate law to 117
  deadline-checkable steps still leaves two `dAbs` at 62.4 s and 34.8 s, with
  every other step under 0.04 s. `Abs` is an atomic leaf. What the profile showed
  instead is that the 138 s bought a 5.7-million-op expression (a ~1500x blow-up)
  that `sympy_to_exprtk` then rejected — the verdict was decidable up front.

  Six functions the emitter accepts have no derivative it can print, derived by
  differentiating every name in `_SYMPY_FUNC_TO_EXPRTK` rather than listed by
  hand: `Abs`, `Max`, `Min` (which produce `re`/`im`/`Heaviside`) and `ceiling`,
  `floor`, `sign` (an unevaluated `Derivative`, per the fix above). That set is a
  fact about the emitter map rather than a judgement, so a test re-runs the
  derivation and compares — adding a function to the map without re-deriving
  would otherwise reintroduce this quietly for that function. A rate law
  with one of those over a **differentiation variable** now falls back
  immediately. Position matters and the check respects it — `Abs(k)*A`
  differentiates to `Abs(k)` and is fine, and a Piecewise *condition* is copied
  through undifferentiated, so `MODEL1006230034`'s
  `Piecewise(…, mincond_J_K < Abs(deltaPsi))` keeps its complete analytical
  Jacobian. A position-blind check took it away; that regression is what the
  corpus A/B caught.

  **Verified over the whole rr_parity corpus** (1319 models, unbudgeted
  derivation, `complete` flag diffed against the pre-change sweep): **0 models
  changed classification** in either direction, 0 error-status changes, 0 models
  measurably slower. Total unbudgeted derivation 1296 s → 1100 s (−15.1%);
  `BIOMD0000000385` 148.3 s → 0.9 s, `470`/`473`/`472` ~7-10 s → ~0.7 s. Worst
  overshoot left in the corpus is **1.0x** the budget, down from 6.9x.

- **The analytical-Jacobian budget has no correctness floor left, and the test
  that claimed one was red on `main` (issue #249).** Since #95 this suite held
  the shipping derivation budget above a floor justified by one model —
  `BIOMD0000000457` was said to be stiff enough that its finite-difference solve
  *fails* at the parity tolerance, so a smaller budget would strand it. #244 made
  that premise a test rather than a docstring sentence, which was the right move
  and immediately showed it to be false: the assertion has been failing since
  `bac36e7` merged. It went unnoticed because it is corpus-gated and CI has no
  corpus — the job log reads `ssss.ss` for that file under `model corpus absent
  from this checkout`, so a green tick says nothing about it, while every local
  push from a checkout that *has* the corpus was blocked.

  457's FD solve is not marginal: it returns in 0.07 s with 626 steps and a
  finite trajectory, and survives rtol 1e-6/1e-12 through 1e-12/1e-15 on both
  linear solvers and both RHS backends. Nothing resembling the documented "CVODE
  returns -3 at t~3.36 with h~1e-42" appears anywhere in that grid. Its
  derivation now costs 0.283 s, which puts it below the 0.5 s band #244 selected
  fixtures from, so that recipe would not pick it today either.

  Re-running the classification over the **whole** corpus rather than a slow
  band — forcing `jacobian="fd"`, which costs no derivation, so all of it is
  affordable — the floor turns out to have no subject at all. Of the 1,218 models
  the analytical Jacobian attaches to (1,323 probed, 1,291 load, 73 declines),
  **1,198 solve on FD; the 15 that fail fail identically with the analytical
  Jacobian**, and 5 cannot be simulated at all (`fast="true"` reactions). **Zero
  models need it.** Nor is there a weaker property to re-anchor on: across the 23
  models whose derivation is slow enough to be starved by any plausible budget,
  the worst FD-vs-analytical trajectory difference is 1.9e-6 relative — solver
  noise at rtol 1e-9.

  So the floor is retired rather than re-derived, which would have repeated the
  mistake #244 named: a tripwire standing on a measurement that has evaporated.
  The shipping default is **unchanged at 20 s** — dropping a requirement is not a
  licence to move the constant. What guards it now is an equality pin with an
  actionable message, because with no correctness requirement there is no
  meaningful inequality left to assert, and the premise test is inverted into a
  canary: 457's FD solve must keep succeeding *and* keep agreeing with the
  analytical one (measured 8.5e-6, asserted under 1e-3). If either flips, the
  instruction is to re-run the classification, not to edit the test.

- **The budget does have a floor — on cost, not correctness — and the 457 canary
  was pinning the one thing about that model that is not portable (issue #245).**
  #249 above is right that no corpus model *needs* the analytical Jacobian, and
  right to make the guard an equality pin rather than invent a correctness floor.
  It also concluded that with no correctness requirement there is no meaningful
  inequality left to assert. There is one, because its sweep asked whether the FD
  solve **works** and never what it **costs**.

  `BIOMD0000000608` derives for **4.76 s** and solves **4.16x faster** for it
  (0.065 s vs 0.015 s, 52 species). FD is perfectly correct there, just slow, so a
  viability screen is blind to it by construction. Nor is it alone:
  `MODEL1603150001` (3.0x), `MODEL1601050000` (2.7x), `MODEL1602080000` (1.7x) and
  `MODEL1504130000` (1.4x) derive in 2.2–3.0 s and pay too. 608 is the most
  expensive derivation that pays for itself; the cheapest that does not is
  `BIOMD0000000628` at 59.3 s, whose analytical solve is *slower* than its FD one.
  That window is 12.5x wide against #95's 3.4x — the gap did not close, it moved
  and opened — and 20 s sits 4.2x above the floor and 3.0x below the ceiling,
  above the 16.8 s geometric centre because too high spends build seconds once
  while too low buys a permanently slower solve. Between the bounds nothing needs
  getting right: `BIOMD0000000496` and `497` derive to completion on the default
  and measure 1.02x and 1.25x — real waste, ~22 s of it across 1286 models.

  So a floor comes back at **15.0 s** (4.76 s × the 3.3x machine spread), beside
  #249's equality pin rather than instead of it: the pin catches any move of the
  constant and asks for a justification, the floor is the bound a justification
  may not talk past. 608's premise is run, not asserted in prose.

  **Solve times must be medians over repeats on a warm codegen cache.**
  `fd_viability.jsonl` takes one cold sample per mode, analytical first, so that
  arm absorbs codegen warm-up: it reports 496 at 5.84x (really 1.02x) and
  `MODEL1603150001` at 0.33x (really 3.01x) — wrong by 5.8x one way and 9x the
  other. That artifact is why the paying population was invisible.

  **The 457 canary is now a tolerance ladder.** Its FD solve fails at exactly rtol
  1e-9/atol 1e-12 on x86_64 macOS — "CVODE -3 at t~3.36 with h~3.1e-42", #95's
  signature — and succeeds at 1e-6, 1e-8, 1e-10 and 1e-12, while on arm64 it
  succeeds at all five. Same commit, corpus present, core rebuilt from the tree.
  Both readings are real: this is a knife-edge stiff transient whose convergence
  is not portable, and #244 and #249 each pinned the single cell where the two
  architectures disagree — so that assertion has been red on one machine or the
  other continuously since #244, invisibly, because the file is corpus-gated and
  CI has no corpus. It **strengthens** #249's conclusion: a model that genuinely
  needed the analytical Jacobian would fail across a band and hardest at the
  tightest rung, where 457 instead agrees to 9.8e-10. The canary now walks the
  ladder, tolerates one isolated interior failure, requires the tightest rung to
  solve, and requires agreement wherever it solves. Mutation-checked both ways.

  The key stays wall-clock, which #245's other half proposed replacing with
  something that travels between machines (cost per reaction, inlined rate-law
  size, a #97-style step count). Measured over the same corpus, each predicts
  derivation cost far worse than the 3.3x machine spread it would replace:
  per-inlined-token cost runs 2.5–2351 µs/token over the 256 models with ≥ 1000
  tokens (**923x** end to end, 142x from the median up), `BIOMD0000000385` and
  `246` share a largest inlined rate law (~47k tokens) and derive 19x apart, and
  the best log-log correlation of any static key is 0.685. The loosest size-keyed budget that cuts nothing it cuts today
  hands `MODEL1006230049` 5838 s where wall-clock gives it 20 s. #187 and #97
  already ship the sound version and are unchanged: clock as the mechanism, a
  size that travels scaling the allowance upward.

  What the sweep does leave open, recorded next to the budget rather than fixed
  here: the deadline can only be tested *between* `sp.diff` calls, so a single
  pathological rate law overshoots it by however long one derivative takes.
  Against the 20 s default that is 1.0x on `MODEL1006230053` and
  `MODEL1006230090`, and **6.9x on `BIOMD0000000385`** — 138 s to reach the first
  check, after which it declines anyway. Filed as #250: bounding it means
  subdividing the derivation so the deadline is reachable, and a size gate is not
  the answer (`BIOMD0000000246`'s largest inlined rate law is 1% smaller and
  derives in 6.2 s).

  No shipped value changed, so no model's build or solve behaves differently;
  `main` goes green again on a corpus-bearing x86_64 checkout.

- **A plain ODE build no longer emits the sensitivity RHS at all — the
  Elementary half is gated too (issue #217).** #209/#214 stopped a plain
  `Simulator(model, method="ode")` from deriving the Functional/MM analytic
  `∂f/∂p`, but deliberately left the Elementary `bngsim_codegen_sens_rhs`
  unconditional: it is plain text emission with no sympy in it, so gating it
  would buy no *derivation* time, only source size, and leaving it alone kept
  every all-Elementary model's source byte-identical. Correct about the
  derivation, wrong about the size. On the 20 largest `.net` models it is
  **55.6% of a plain build's C source** (44.8 MB of 80.6 MB) — a symbol
  `CVodeSensInit1` is never called to install — and because `_resolve_opt_flag`
  picks its tier from total translation-unit size, that dead weight held five of
  them at a lower `-O` for the RHS the solve *does* call: `fceri_fyn` compiled
  its plain RHS at `-O0` where 5.2 MB gets `-O1`, and four models took `-O1`
  where `-O3` was available. Measured on the in-repo `.net` corpus after the
  change: plain source down **47.9%** (68.6 MB → 35.8 MB over 12 models), 4 of
  12 recovering a higher `-O`.

  The cost that held it back was already paid. Gating this was argued to make
  every model's plain artifact differ from its sensitivity artifact "where today
  only Functional/MM ones do", roughly doubling entries in an already 2 GB cache
  (issue #205) — but #177's `:sens_term_scale` has been in both keys since
  before #209, so plain and sensitivity have had separate entries, and separate
  sources, for every model since then. What changes is the *content* of the
  plain entry.

  Sensitivity runs are unaffected: a 16-model A/B over the `.net` corpus keeps
  `bngsim_codegen_sens_rhs` on every one and reproduces every sensitivity value
  bit-for-bit. `emit_functional_sens` is now `emit_sens_rhs` and
  `want_functional_sens_rhs()` is `want_sens_rhs()`, both private. The
  `.net` cache suffix `:no_functional_sens` now means only what GH #67 gave it
  (the `BNGSIM_NO_FUNCTIONAL_SENS_RHS` A/B hatch); "nobody asked" gets its own
  `:no_sens_rhs`, because for an Elementary model those two stopped emitting the
  same source and sharing one namespace across that difference is the issue #51
  inertness trap. The hatch is correspondingly no longer folded into
  `want_sens_rhs()` — with it set, a sensitivity run still emits the Elementary
  sensitivity RHS and loses only the Functional extension, which is the pre-#67
  behaviour it exists to restore.

- **`compute_all_sensitivities(params=None)` now returns independent columns —
  `Model.primary_param_names` rather than `Model.param_names` (issue #203).**
  The default meant "every parameter", and on a model with derived
  (expression-backed) parameters that list is not a set of coordinates. A
  derived parameter reaches the trajectory only through the primaries it is
  built from, and *their* columns are total derivatives **through** it — the
  chain rule #188 restored. So `alpha` and `_rateLaw_R16_fwd = alpha*konBT`
  reported the same physical effect twice, in exact proportion
  `d(derived)/d(primary)`, and `Result.gradient` contracts that whole axis into
  one `(n_params,)` vector its own docstring hands to `scipy.optimize.minimize`
  over a parameter vector of the same width. Nothing bounded the size of the
  error: on `BIOMD0000000701` the derived columns are ~6.7e-12 against `alpha`'s
  1.0e-3 and it does not matter, but the ratio is the model's, not a constant.

  `Result.fisher_information` is the sharper symptom, because there the damage
  does not depend on that ratio at all: `Sᵀ Σ⁻¹ S` over an axis holding two
  exactly proportional columns is rank-deficient **by construction**, so the old
  default handed back a matrix with a null direction on every model carrying
  derived parameters — and the identifiability reading the user guide recommends
  (smallest eigenvalues → least identifiable) takes that round-off eigenvalue for
  a finding about the model.

  Dropped the way issue #164 established in the same five lines for the
  compartment sizes `set_param` refuses: a warning naming them, and an explicit
  `params=[...]` that still returns the column for anyone who wants
  `∂x/∂_rateLaw` **on its own terms** — the axis
  `bngsim.jax.differentiable_solve(..., flat=True)` asks for, which goes through
  the explicit path and is unchanged. The default now agrees with that function's
  own `flat=False` default, which has always differentiated over
  `primary_param_names`.

  The synthesized `_V0_<comp>` goes the same way, for the older reason: it is
  bngsim's record of a compartment's size at load, which the rate constants in
  that compartment are normalised against, and `set_param` refuses a
  value-changing write to it (`test_the_load_time_volume_record_is_not_a_knob`
  has said so since #170 stage 1) — so a gradient entry for it is one an
  optimizer would fit against nothing. The compartment size itself is an
  ordinary writable, differentiable parameter and stays in the tensor.

  Measured on the 1,291 loadable rr_parity models: 279 carry derived parameters
  and 216 of those also carry a `_V0_`, so 279 models see a narrower default —
  10,048 columns dropped in total (9,524 derived, 524 internal), no model left
  with an empty column set, and the three skip classes disjoint (no parameter
  carries two of the flags). On the models A/B'd column for column, the columns
  that remain are unchanged.

- **A plain ODE run no longer derives, emits or compiles the analytic `∂f/∂p` it
  never installs (issue #209).** `generate_combined_from_model` called
  `generate_sens_from_model` unconditionally, so `Simulator(model, method="ode")`
  with no `sensitivity_params` ran the Functional sensitivity derivation through
  sympy, emitted it, and compiled it into the cached `.so` — for a solve that
  never calls `CVodeSensInit1`. On `BIOMD0000000496` (295 species, 333 functional
  reactions, cold codegen cache, the analytical Jacobian derived first so the GH
  #95 budget cannot decide the answer) that was **46 s of codegen against 29 s, and
  a 26.7 MB `.so` against 1.8 MB (14.6x)**, and the solve itself 0.45 s against
  0.29 s. The GH #198 output-sensitivity evaluator three lines below in the same
  function was already gated on `_want_output_sens` for exactly this reason.

  Two side effects on models this large. 12.6 MB of C source crosses the 8 MB
  `_CODEGEN_HUGE_SOURCE_BYTES` threshold and 3.7 MB does not, so the translation
  unit now compiles at `-O1` rather than `-O0` — which moves the trajectory in the
  last bits (7.5e-15 relative on this model; pinning `-O0` on the gated build
  reproduces the old trajectory *bit for bit*, which is how that was attributed).
  The faster solve is **not** that: `-O0` on the gated build is just as fast, so
  what is left is the 8.9 MB of never-called code sharing the image, and this
  change does not identify the mechanism further.

  Scoped to the Functional/Michaelis-Menten half. An Elementary model's
  sensitivity RHS is plain text emission with no sympy in it, so it stays
  unconditional and its source (and every `.so` cached for it) is byte-identical.

  The cost of the gate is entirely in *not* silently downgrading a sensitivity run
  to CVODES' difference quotient, so the resolved flag reaches both cache keys
  (sharing the existing `:no_functional_sens` namespace on the `.net` side — a
  build with the GH #67 hatch set and a build with nobody asking emit the same
  source), and `Simulator._prepare_output_sens_codegen` — which `steady_state()`
  and `compute_all_sensitivities()` share — regenerates a plain artifact instead
  of reusing it. That helper's own `n_functions > 0` condition is gone with it: a
  Michaelis-Menten model can have no functions at all, and #177's
  `bngsim_codegen_sens_term_scale` had already made the byte-identical-source
  claim behind that condition false.

- **The model-side `.so` cache key is structural, so a warm cache generates no C
  source (issue #174).** `prepare_model_codegen` derived its key by generating
  the source and hashing it, which meant a cache hit skipped only the `cc`
  compile: every `Simulator` construction still paid the RHS + `∂f/∂p` + Jacobian
  derivation, and none of that work depends on the parameter values a fit is
  moving. `compute_model_codegen_hash` — dead code whose docstring already
  claimed to hash "model structure" — now actually does: `codegen_data()` minus
  each parameter's *value*, the Jacobian scatter plan, the functional-Jacobian
  context, and the process-scoped emit decisions, all cheap C++ reads. This is
  what `prepare_codegen` has always done for the `.net` path. On
  `Smith_BMCSystBiol2013` construction against a warm cache goes from 1.31 s to
  0.03 s.

  Dropping parameter values is safe for one reason that had to be established
  rather than assumed: the generated C reads parameters from the runtime `p[]`
  array. The single exception is the issue #68 switch-condition gate, which
  probes the RHS to find clock species and evaluates a clock threshold
  numerically — so one `set_param` really can add or remove the whole analytic
  sensitivity RHS. `switch_gate_cache_digest` carries that *verdict* (the
  booleans, not the values), which keeps the key stable across a fit while still
  separating the two artifacts.

  **This invalidates every cached `.so` on the model path.** The next run per
  model recompiles once. The `.net` path's key form is unchanged apart from the
  chunking fix below.

- **An SBML compartment size is a writable parameter (issue #170).** A volume
  plays two roles in a loaded model: a *symbol* in kinetic laws, which is an
  ordinary `p[]` a write has always moved, and the *storage convention* — bngsim
  stores `amount/V_c`, so V decides the amount↔concentration conversion, an
  amount-declared initial condition, the mass-action scalar's `Π V^n / V_storage`
  and the SSA propensity volume. The second was folded at load and never
  re-derived, so issue #164 refused the write outright. Each fold is now put back
  on the parameter: the mass-action scalar carries the volume as a ratio on the
  reaction's rate parameter (`k · (C/V_load)`, exactly 1.0 at the nominal point),
  the Functional storage divide is emitted against the compartment symbol even
  when the load-time size is 1, and `set_param` re-derives `volume_factor` and an
  amount-declared IC. The load-time size the ratio normalises against is carried
  by a synthesized `_V0_<comp>` parameter rather than a printed literal — ExprTk's
  decimal literal parser is not correctly rounded, and a 1-ulp denominator moves
  the rate constant. `_V0_<comp>` is marked internal: `primary_param_names` omits
  it and `set_param` refuses a value-changing write, since moving it would
  rescale the rates in that compartment without moving the volume. `set_param("cell", v)` now reproduces *loading the model at
  `v`* bit for bit, and matches RoadRunner, on every shape issue #170 tabulated —
  including the pair that made its case, where the same law loaded at V=1 and at
  V=4 gave opposite answers.

  The **generated C** reads the volume from `p[]` too, so the write lands on the
  compiled backend as well as the interpreted one. Previously the emitted source
  baked it — the amount factor of an amount-valued (`hasOnlySubstanceUnits`)
  species's rate, its observable weights, its `∂/∂x` chain factor and `rateOf`
  scaling, plus a cross-compartment reaction's `inv_vf` reciprocal table and
  per-row Jacobian / `∂f/∂p` divisors — and those two shapes were refused rather
  than honored on one backend and half-applied on the other. The invariant that
  makes it safe: **the emitted source no longer depends on the load-time volume at
  all**, so one compiled `.so` is valid at every size and a write that arrives
  after the source was generated (a `parameter_scan`, any post-construction
  `set_param`) still lands. The per-species `∂func/∂x` chain coefficient went
  symbolic for the same reason, on both backends: it is the `J·yS` half of the
  forward-sensitivity RHS, so freezing V there was a wrong *sensitivity* (measured
  at 100% on a 400x volume write), not merely a slower solve.

  Lifting that refusal made 38 more corpus models writable, and a sweep of "does
  `set_param` reproduce `compartment_sizes=` at the same value?" over all 173 found
  three more places where a load at V=1 emitted something a load at V≠1 did not,
  so the write had nothing to move — the `_vd_<rid>_unified` Functional storage
  divide for a *multi-compartment* reaction, an event assignment writing an amount
  into an `hasOnlySubstanceUnits` species's slot, and the report-time
  amount→concentration divide for an assignment-rule target (the one conversion
  that lives entirely on the Python side, so no engine refresh could reach it).
  All three are fixed and all three are numerically free at the nominal point,
  since `x/1.0 == x` exactly.

  **177 of the corpus's 207 compartment-carrying models are now fully writable**
  (135 before the codegen half), 9 partly; the models with any refused size drop
  from 72 to 30 and the refused sizes from 230 to 132. Refused by name rather than
  in a blanket (`Model.unwritable_compartment_size_params`), and now for two
  reasons rather than three: an assignment-rule compartment (the rule recomputes
  its size every step), and one whose storage divide a single mass-action scalar
  shares across two equal-sized compartments (that scalar stops being exact the
  moment they differ). Not yet *differentiable*: forward sensitivity still refuses
  a `d/dV` column (issue #170 stage 3).

  Behaviour at the nominal point is unchanged: over the 214-model SBML corpus the
  RHS is bit-identical on all 214 and the trajectory on 206. Five of the eight
  that move do so because they *gained* an analytical Jacobian (202 → 207
  complete, none lost) — see below; the other three differ by ≤ 8.3e-16 relative.
  The codegen half moves nothing further: against the interpreted-and-writable
  build, the RHS fingerprint and the trajectory are bit-identical on every model,
  interpreted and `codegen=True` alike, even though the emitted C text changes for
  the whole SBML corpus (a new `.so` cache key, not a new answer).

- **`_DECLARED_SKIPS` is now checked against the reason strings the tests emit,
  in both directions, and carries a tier (issue #179).** The list is the
  codebase's mechanism for forcing a permanent skip to be justified in a diff,
  and nothing compared it to reality. It had drifted to **25 undeclared reasons
  across 47 files** — none of which fire in the default build, which is exactly
  why nobody saw them. They are *build-variant* reasons, and the variants they
  describe are the ones the other CI legs use, so the only run that could have
  surfaced them was the one nobody had turned the audit on for.

  A new AST-based check in `test_skip_audit.py` asserts every hand-written skip
  reason matches a declared pattern, and every declared pattern still matches
  something. Both directions are verified to fail on real drift rather than
  being vacuous. The two traps #179 flagged are handled and pinned by their own
  tests: `pytest.importorskip("sympy")` *generates* `could not import 'sympy'`
  at run time, so scanning its call sites invents failures (~3× the apparent
  problem size), and `xfail(reason=...)` is not a skip reason at all. Both are
  decided by what the AST node is, which no regex over `reason=` can do.

  Trap 1 has a sub-case the issue did not name, and the first version of this
  check had it wrong: `importorskip` takes an optional `reason=`, and when it is
  given the generated text is never produced — so the string in the source *is*
  what the audit sees. Ignoring those call sites wholesale hid two genuine
  undeclared reasons (`vivarium-core not installed`, `tomllib is 3.11+`). A
  strict run caught them minutes after the scan had pronounced the tree clean,
  which is the argument for both checks existing rather than either alone.

  Nine phrasings for two conditions — `NFsim not built`, `bngsim compiled
  without NFsim support`, `no NFsim support`, … — are consolidated onto the two
  strings the list already declared, across 29 sites. That was drift away from
  an existing convention, not the absence of one. `scipy` is removed: no
  hand-written reason contains it, so every scipy skip is an
  `importorskip`-generated `could not import 'scipy'` the neighbouring entry
  already matched.

  Declarations now carry a **tier**, because one flat list could not say the
  thing that matters. `KLU not compiled` and `no C compiler on PATH` are both
  fair skips on a laptop, but only the first is fair in CI: a leg that silently
  lost `cc` would skip ~22 files' worth of codegen tests and report success —
  a false green waved through by the list meant to catch it. `LOCAL_ONLY`
  reasons (the C-compiler family; `requires libsbml`, which is a *hard*
  dependency and so cannot be a build variant) print a `!!` row and end the run
  under `BNGSIM_SKIP_AUDIT=strict`. Strict is only ever set by a workflow, so
  the tier is enforced exactly where "this environment is incomplete" stops
  being an acceptable answer. None of these fires on any leg today, so it costs
  nothing now and buys the alarm later.

  `BNGSIM_SKIP_AUDIT=strict` is consequently on for `mir.yml` and
  `windows-tail.yml`, the two KLU-off legs — the first time the audit has had
  teeth outside the default build, which is where these reasons live. Note the
  issue proposed `native-tests.yml` as the cheapest KLU-off leg; that workflow
  is ctest-only and runs no pytest, so there is nothing there to turn on.

  One test changed rather than being declared: `test_conservation_laws.py`'s
  `test_dependent_block_is_identity` skipped when a model reported no
  conservation laws. All five of its fixtures are checked-in `.net` files that
  have them, so the branch had never fired — and had conservation-law detection
  regressed to zero, the test would have gone green by skipping. It asserts
  instead.

- **A parity run now records WHICH bngsim produced it, not just which version
  (issue #163).** The `bng_parity` harness records the engines behind a run so a
  golden can be reproduced. PyBioNetGen was recorded as a resolved git commit;
  bngsim was recorded as a bare `__version__` — which identified an artifact only
  while bngsim could not be installed from PyPI. It can now, and `__version__`
  bumps only at release, so a PyPI install, a `ship_wheel.py` wheel, and every
  commit between two releases all report the same string.

  `bngsim_backend.backend_status()` — and through it each sweep's `_summary.json`
  and `golden.json`'s `_meta` — gains **`bngsim_build_commit`** (the commit the
  loaded `_bngsim_core` was compiled from, baked in by CMake) and
  **`bngsim_install`** (PEP 610 origin: `index` / `wheel:<file>` / `editable` /
  `vcs:<sha>`). Both are needed: the release protocol builds the published wheel
  *from* the release commit, so a locally built wheel of that commit reports the
  identical build commit (measured — PyPI 0.12.2 and a `ship_wheel.py` wheel both
  report `1737003f0c81`); the install origin is what separates them.
  `bngsim_version` is now read from `bngsim.__version__` directly rather than
  through the bridge's `bionetgen.BNGSIM_VERSION` re-export, and the whole bngsim
  half is collected before bionetgen is imported, so an env with a broken or
  absent bridge still records which bngsim is installed in it. The bridge's own
  view is kept alongside as `bngsim_bridge_version`, where the two disagreeing is
  now visible instead of silently authoritative.

  `bootstrap_parity_env.py` gains **`--bngsim-pypi <version>`**, mutually
  exclusive with `--bngsim-wheel` / `--build-bngsim`: for a consumer *reproducing*
  a published golden, installing the released wheel is the faithful route, and it
  is the case the harness was written to assume impossible. Its no-source ABORT
  used to say "bngsim is not on PyPI" and foreclose exactly that option; it now
  names all three sources. No harness prose claims bngsim is absent from PyPI.

- **An SBML compartment size is no longer writable, and says so (issue #164).**
  A compartment volume had two representations in a loaded model and a write
  moved only one. The kinetic law reads `p[]`; the *storage convention* is
  folded at load into constants nothing re-derives — `Species::volume_factor`,
  an amount-declared `initial_conc` (= amount/V), the Elementary scalar rate's
  `Π V^n / V_storage`, `Reaction::ssa_volume_factor`, and the `inv_vf` table in
  the emitted C. `set_param` reached the first and none of the second.

  The result was not a stale value but an internally inconsistent model. On the
  issue's two-compartment model `set_param("C1", 3.0)` moved `A(5)` from 22.3 to
  1.11 — a factor of 20 — on a trajectory that is *exactly* `C1`-invariant; the
  other direction was a silent no-op (`set_param("C2", 7.0)` changed nothing),
  `parameter_scan` over a compartment returned one trajectory N times, and a
  forward-sensitivity column was wrong in **both** directions at once: `dA/dC1`
  reported 36.6 against a true 0, `dB/dC2` reported 0 against a true 2.30.

  **Wider than the issue scoped it.** #164 measured single-compartment models as
  safe. Against RoadRunner as an independent oracle, only the exact
  `compartment·k·A` convention with concentration ICs is V-invariant: a bare
  `k*A` law (the common BioModels form), any `initialAmount` species, and every
  `hasOnlySubstanceUnits="true"` species move with V under a rebuild and did not
  under a write. Which half of a write landed was not even uniform inside one
  model — a mass-action law folded the volume away entirely, a Functional law
  loaded at V ≠ 1 divided by the live compartment symbol, and the same law
  loaded at V = 1 had that divide normalized out — with nothing visible to the
  caller to tell them apart. Hence a refusal rather than a patched subset.

  So: `set_param` / `set_params` raise `ValueError` on a compartment-size
  *change*, `Simulator(sensitivity_params=[...])` and
  `steady_state(sensitivity_params=[...])` refuse the column, and
  `compute_all_sensitivities()` skips compartments from its "all parameters"
  default with a warning (an explicit `params=[...]` raises). Writing the value
  a compartment already holds stays legal, so round-tripping a full parameter
  vector through `set_params` still works, and the check runs in that method's
  validation phase so its all-or-nothing contract holds. `.net` models are
  unaffected — BNG2.pl folded their volumes into rate constants long before
  bngsim sees them — and a compartment the loader promotes to a species
  (rate-rule or event-resized) is genuine live state, not flagged.

  Inert for every model that does not write a compartment: no emitted source,
  cache key, or trajectory changes.

  Issue #170 tracks making the volume live everywhere, which retires the
  refusal and turns the sensitivity column into a real one.

### Fixed

- **The committed `_bngsim_core.pyi` had drifted from the bindings, and nothing
  checked that it hadn't.** The stub is machine-written and committed because it
  is the only description of the compiled extension a type checker or an editor
  can read. #305 added `SolverOptions.set_crossing_stop_times` without
  regenerating it, so main described an API missing a method it had: every
  rebuild dirtied the tree with the regenerated hunk, and mypy — which believes
  the stub — reports `"SolverOptions" has no attribute "set_crossing_stop_times"`
  for any caller reaching it through a typed reference. That drift survived
  review only because the one caller passes `opts` as an untyped parameter, so
  the attribute was never checked; the invariant was resting on call sites
  *staying* untyped.

  The stub is regenerated, and a test now asserts every pybind11-bound name
  appears in it (224 names today). Names rather than signatures on purpose:
  regenerating in CI and diffing would also catch a changed signature, but needs
  a built extension on the leg and `pybind11_stubgen` output moves with the tool
  version and the platform, so the strictness would be paid for in flakes. The
  realistic failure is a binding added without a regeneration, which a regex
  catches in milliseconds on every leg.

- **A registered time-discontinuity root could never be reached, so the run
  wedged one ulp below the crossing (issue #305).** GH #72 registers every
  `time` inequality in a `piecewise` as a CVODE root so the integrator stops at
  each pulse edge instead of stepping over it. That is only half of reaching the
  crossing: **CVODE tests for a root solely on a step it accepts**, and where
  the branch jump is large enough that the local error test rejects every step
  containing the crossing, the accepted steps land short. `h` shrinks, `t`
  creeps to the last representable double below `t*`, and from there every step
  that would carry it across is under one ulp — `t + h == t`, and the run dies
  with the #54 stall error having never once evaluated `g` past the crossing.
  The root never fires, not once.

  Measured on `Weber_BMC2015` (`piecewise(0, (time - PdBu_time) < 0, PdBu_dose)`
  at a fixed `PdBu_time = 24`): 6 of 100 points sampled from that fit's own
  parameter box die outright, every one wedged at `nextafter(24, -inf)`, with
  **zero root returns** and half of 20,000 steps rejected by the error test.

  Neither of the two things it looks like. Not a sensitivity defect — the plain
  state solve dies identically, and on this model the analytic sensitivity RHS
  is declined anyway. Not honest stiffness at an awkward parameter point — the
  same points at the same tolerances integrate the moment the step is made to
  land on the crossing, and in the failing runs the post-jump right-hand side is
  never reached at all.

  The fix resolves each *registered* condition to a crossing time and stops the
  step there (`CVodeSetStopTime`), which is the mechanism issue #48 already uses
  for a crossing a **fitted** parameter moves — applied here to the far more
  common crossing that nothing moves and that therefore has no `dt*/dp` to jump
  by. Three things had to change together, since any one alone leaves Weber
  wedged:

  - the crossing time is resolved from the registered condition text, by two
    probes of its residual in `time` plus a linearity check, rather than through
    `_clock_threshold_split` — which requires a **bare** clock symbol on one
    side and so declines `(time - p) < 0`, the spelling a PEtab export writes.
    This is #259's lesson (registration already admits either side being
    time-dependent) carried to the resolution path;
  - a crossing no requested sensitivity parameter moves is no longer skipped.
    Its `dt*/dp` is a correct zero; it still has to be *reached*;
  - none of it is gated on sensitivities, because the plain solve needs it too.

  Resolution runs per `run()` against **live** parameter values, which is
  load-bearing for pre-equilibration protocols: the same condition parameter can
  put the crossing inside the measured window and outside the equilibration
  window that precedes it, and a stop armed where a phase has no crossing
  measurably perturbs its steady-state march.

  On the minimal reproduction the crossing also stops being expensive: error
  test failures at the crossing fall from 33–58 to 0–4, and the step count
  roughly halves. `max_step` is *not* an alternative remedy — measured at 1.0,
  0.1 and 0.01 it still wedges — so the GH #88 periodic step bound would not
  have covered this either, and #274's "where a root already forces the stop,
  `max_step` is pure cost" holds only where the root can fire.

  Also worth knowing: of the two remedies the stall error message suggests,
  moving the discontinuity onto an **event** works (the RHS is then smooth in
  `t` and the jump is applied discretely at the root), and moving it onto a
  **sample time** does not — `CV_NORMAL` interpolates output points, so an
  output point at the crossing does not bound the step that spans it. Weber's
  own failing runs already had `t = 24` as a sample time.

- **NFsim counted a reactant pattern the rule does not transform once per
  matching *molecule* instead of once per matching *complex* (issue #281).**
  BioNetGen gives such a pattern one reaction instance per complex, however many
  molecules inside that complex match it: every embedding yields the identical
  reaction — same reactants, same products, same transformation — so there is
  only one reaction to count. NFsim enumerates matches per molecule, so a rule
  whose catalyst was a homodimer fired twice as fast as the same rule with a
  heterodimer, and a homotrimer ring three times as fast. That is a common
  shape: a dimeric enzyme or a receptor dimer used as catalytic context.

  This is distinct from #195, where BNG emitted a `symmetry_factor` and NFsim
  discarded it. Here BNG emits `symmetry_factor="1"` and there is nothing to
  discard.

  **It is not a symmetry effect**, although the symmetric cases are the ones
  that make it obvious. Checked against BNG's generated network, a whole
  homodimer, a *single subunit* of that homodimer, a heterodimer, and a scaffold
  holding two **distinguishable** copies all get a bare rate constant — and the
  scaffold case has no automorphism in either its pattern or its species. So any
  rule keyed on automorphisms, including the `embeddings / |Aut(pattern)|` the
  issue proposes, gets that case wrong. What BNG discriminates on is whether the
  rule *transforms* the pattern: a homodimer whose subunit the rule binds gets
  `2*k`, because there the two subunits are two real reactive sites.

  Measured on the new `tests/data/nfsim/context_symmetry.xml` against BNG's own
  network expansion simulated with bngsim's ODE engine, 10 seeds, `t=2000`:

  | pool | before | after | BNG network → ODE |
  |---|---|---|---|
  | homodimer catalyst, constant rate | 1485.7 | 2435.0 | 2426.1 |
  | homodimer catalyst, global function | 1475.3 | 2423.4 | 2426.1 |
  | homodimer catalyst, local function (DOR) | 1470.3 | 2425.0 | 2426.1 |
  | homotrimer ring catalyst | 892.7 | 2432.4 | 2426.1 |
  | symmetric enzyme, Michaelis-Menten | 1290.4 | 2333.2 | 2344.8 |
  | single-subunit pattern `Ux(d!+)` vs a homodimer | 1474.6 | 2424.8 | 2426.1 |
  | single-molecule pattern vs two distinguishable copies | 1474.9 | 2426.9 | 2426.1 |
  | single-subunit controls (4 pools) | correct | correct | — |

  `ReactionClass` flags such reactants at construction and the three reaction
  classes route their propensity-side counts through `countDistinctComplexes()`.
  A DOR reactant's propensity comes from its reactant tree's rate factor sum
  rather than from a count, so that path sums one representative term per
  complex instead; `ReactantTree` gains the flat `getMappingSetByIndex()`
  accessor `ReactantList` already had, which pairs with the identically indexed
  `getRateFactor()`.

  Which reactants are pure context cannot be read off the transformation types
  after the fact: `TransformationSet::finalize()` marks an untransformed
  reactant with an `EMPTY` transform, and `EMPTY` is also what
  `genBindingTransform2()` puts on the second partner of a binding — which is
  the `2*k` case above, and correcting it would cost that factor of two.
  `finalize()` therefore records the pure context reactants *before* appending
  the placeholder, treating a `LOCAL_FUNCTION_REFERENCE` as non-transforming so
  a DOR reactant can still qualify. That case is in the fixture as
  `Bind_sym`/`Bind_asym`.

  The correction requires complex bookkeeping — with it off every molecule
  reports complex id `-1`, so "two molecules in one complex" and "two complexes"
  are indistinguishable and the correction is skipped. `NFinput` already enables
  it whenever `blockSameComplexBinding` is set (bngsim's default) or the model
  declares a `Species` observable.

  Counting complexes rather than molecules is O(n) in the reactant list where
  reading `size()` was O(1), and `getCorrectedReactantCount()` sits on the
  propensity hot path, so a naive version costs a great deal: a transcription
  model (`DNA() -> DNA() + RNA()`, 500 templates, `RNA` growing to ~1000, both
  pure context) went from 0.14 s to 5.99 s, 43x. Two changes remove it. The
  scan reuses a thread-local buffer and sorts instead of building a `std::set`,
  and — the one that matters — a reactant container records whether any molecule
  mapped into it has ever belonged to a complex of more than one molecule. While
  that is false no complex can hold two matches, so the count *is* `size()` and
  the scan is skipped outright. That is the common case for a catalytic rule
  over a large monomer pool. The flag is conservative: a stale `true` only costs
  the scan. Measured back at 0.15 s against main's 0.14 s, with identical output.

  Checked for regressions by running 181 models from the RuleMonkey corpora
  (`corpus`, `feature_coverage`, `nfsim_basicmodels`) through NFsim at a fixed
  seed before and after: 169 ran and gave identical trajectories, 2 were inert,
  10 failed identically in both arms, and the only model whose output changed
  was the deliberate positive control.

  Carried as `bngsim/carry-pure-context-per-complex` and reported upstream as
  RuleWorld/nfsim#87. The same over-count is present in vendored RuleMonkey
  (`method="nf_exact"`), which is a separate engine and is not addressed here.

- **`find_package(pybind11)` resolved from whatever `.venv` sat in the checkout,
  not from the interpreter the build targets (issue #288).** CMake asked a list
  of candidate interpreters for `pybind11.get_cmake_dir()` and took the first
  answer. The list did not contain `Python_EXECUTABLE` — the interpreter
  scikit-build-core is actually building for — and its third entry was
  `${CMAKE_SOURCE_DIR}/.venv/bin/python`. So on any machine whose checkout has a
  `.venv`, that entry won every build that set neither `BNGSIM_PYTHON_EXECUTABLE`
  nor `VIRTUAL_ENV`: wheel builds for other interpreters, and isolated builds
  whose own `[build-system] requires` had already resolved and installed a
  different pybind11, all compiled against the dev venv's copy. Two wheels built
  from one commit while answering #275 recorded the same
  `<checkout>/.venv/.../pybind11` in their caches although `Python_EXECUTABLE`
  named two different venvs in them, and although the isolated one had resolved
  `pybind11>=2.13` to a newer release and installed it.

  `Python_EXECUTABLE` (with the `Python3_`/`PYTHON_` spellings) is now consulted
  right after the `$BNGSIM_PYTHON_EXECUTABLE` escape hatch and ahead of
  everything else. It is a *prepend*, not a replacement, because the rest of the
  list is what makes `scripts/rebuild_editable.py` work "after the
  build-isolation venv is gone" (#23, #229) — the target interpreter of a plain
  cmake rebuild routinely has no pybind11 in it, since pybind11 is a
  build-system requirement uv never installs into `.venv`. Both ways of failing
  to answer fall straight through: a *deleted* build env is dropped from the
  cache before the walk (the phantom guard #23 added, which is what keeps the new
  first entry from shadowing the fallbacks), and a live interpreter without
  pybind11 just declines. An interpreter that reports a directory holding no
  `pybind11Config.cmake` is now skipped too, rather than pinned — that trades a
  clear "could not find pybind11" for a stranger error about a bad
  `pybind11_DIR`, the same reasoning as #229.

  The logic moved to `cmake/BngsimResolvePybind11.cmake` so the ordering can be
  tested: the new suite runs the real module under real cmake against fake
  interpreters, and each rule above is one case. Building the whole project per
  case, against interpreters that genuinely carry different pybind11 versions,
  is the alternative — which is why this went untested and then wrong. The
  stale-binary guard (#125) now counts `cmake/*.cmake` as build-graph source, so
  an edit to that module invalidates the binary the way a `CMakeLists.txt` edit
  does.

  None of this was visible in an artifact, which is what let it run: nothing
  recorded which pybind11 compiled an extension. The binary now carries
  `_bngsim_core.__pybind11_version__`, `local_ci_smoke.py` reports it (a wheel
  that still says `unknown` fails the check), and the configure log names the
  interpreter that answered. The macOS arm64 matrix re-run is green for
  cp310–cp313, and each of the four wheels now resolves pybind11 from its own
  isolated build env — 3.1.0, what `pybind11>=2.13` resolves to, where the
  caches previously recorded the dev venv's 3.0.4 for every one of them. The
  Linux x86_64 leg (`scripts/local_ci_linux_docker.sh`) has not been re-run;
  it needs a Docker daemon this box did not have up.

- **`local_ci.py matrix` could not build a wheel on macOS at all, and said so in
  a report nobody read.** Found re-running the matrix for #288. `build_wheel`
  has passed `CMAKE_ARGS=-DBNGSIM_ENABLE_KLU=OFF` on Darwin since the initial
  release; `BNGSIM_REQUIRE_KLU = "ON"` went into pyproject's
  `[tool.scikit-build.cmake.define]` months later (#209), and it reaches every
  scikit-build-core build. CMake rejects the pair outright, so every macOS
  matrix run since then failed to configure for all four Pythons and wrote
  `build: FAIL` into `local_ci_report-darwin-arm64-matrix.md`. Before that it
  was validating a dense-only wheel no published macOS wheel resembles —
  `[tool.cibuildwheel.macos]` sets `ENABLE_KLU=ON` and `REQUIRE_KLU=ON` — which
  is #275's finding one layer down. The override is gone; KLU now comes from
  whatever the box has (a system SuiteSparse, or the pinned
  `BNGSIM_KLU_AUTOBUILD` source build), and the macOS matrix reports
  `klu: True` like the published wheels do. A test checks the two halves against
  each other for every `BNGSIM_REQUIRE_*` pyproject declares, not KLU alone.

- **`ship_wheel.py` refused to build for an interpreter pip could have built
  for, on a claim about the wheel matrix that was not true (issue #275).**
  `_build_command` had three outcomes — unisolated `pip wheel`, `uv build`, or a
  `RuntimeError` — and justified the refusal by calling the unisolated form
  "what the wheel matrix validates". Measured, it is not: `local_ci.py` builds
  each matrix wheel with pypa/build's **default isolation** in a throwaway venv
  holding `build`/`cmake`/`ninja` and no PEP 517 backend, and the Linux leg runs
  cibuildwheel, likewise isolated. The unisolated command is the dev-loop
  shortcut; isolation is what `scripts/LOCAL_CI.md` actually measures. So the
  refusal was defending the artifact against the one form the matrix does
  validate.

  It now falls back to plain `pip wheel .` after `uv build`, leaving "no pip
  *and* no uv" as the only unrecoverable combination — the same answer #272
  landed in `rebuild_editable.py`. Verified end to end on a `uv venv --seed`
  interpreter (pip, no `scikit-build-core`): the isolated build produces a wheel
  that installs and imports with `capabilities()["features"]["klu"] == True`.
  `uv build` stays ahead of isolated pip here, unlike in `rebuild_editable.py`,
  because this script ships the artifact elsewhere and `--python` pins the ABI
  tag explicitly. Nothing that built before changes branch: the documented dev
  venv is a `uv venv` with no pip at all, so it took, and still takes, `uv
  build`.

  Two wheels built from the same commit — one unisolated against a pinned
  backend, one isolated — carry the **same file set and the same extension
  size**, and differ only in `SUN_JOB_ID` (a build timestamp), the `.a` archives
  and `.so` that embed it, and the CMake-generated `sundials_export.h` /
  `SUNDIALSTargets.cmake`. Those last two are a CMake version difference (4.4.2
  from the `cmake` wheel vs the Homebrew 3.28.3 on `PATH`) — a `NOLINTNEXTLINE`
  comment and some version-guard boilerplate — not an isolation difference,
  which is the honest form of the answer, because the backend
  versions the two builds declare do not reach `find_package(pybind11)` at all
  on a machine with a `.venv` in the checkout. That is filed separately as #288.

  A test pins the fact the fallback rests on, so `local_ci.py` cannot quietly
  start building unisolated and leave the docstring wrong again.

- **`test_lapack_dense_linsol` reported four no-ops and two self-comparisons as
  `6/6 passed` (issue #269).** Four cases opened with `return 0` when no BLAS
  backend was linked, and `RUN_TEST` counts `rc == 0` as a pass, so a host
  without one printed a summary byte-identical to a host where all six really
  ran — in 0.03 s. The other two were `solve_with(false, …)` vs
  `solve_with(true, …)`; with no backend the second call falls back to the
  built-in factor, so both arms were the same code and the reported difference
  was exactly `0`. The only tell was a `lapack_dense_available = no` line that
  nothing gated on, and `ctest` suppresses stdout on success so it never reached
  the log.

  Skips are now a third status (ctest's `77` convention, per case): the summary
  reads `2/7 passed, 5 SKIPPED (no BLAS dense backend)` on a bare host and
  `7/7 passed` with one. The exit code still stays `0` when cases skip — a host
  with no BLAS is a supported configuration — so the count is the signal, and
  the CI leg that is supposed to have a backend asserts it is zero.

  The two parity tests now skip rather than compare the built-in solver against
  itself. What they exercised incidentally — that asking for the BLAS factor on
  a build without one still returns a working dense solver — is covered on
  purpose by a new `test_prefer_lapack_fallback_contract`, which runs on every host and
  also pins the counter accessors to `lapack_dense_available()`. So the no-BLAS
  leg keeps real coverage while no longer claiming the parity coverage it never
  had.

- **NFsim dropped the reaction center symmetry factor on every rate law except
  a constant one (issue #195).** BNG2.pl emits `symmetry_factor` on a
  `<ReactionRule>` whenever the reactant pattern has a non-trivial automorphism
  — `RxnRule.pm` computes `MultScale = 1/automorphisms/context-permutations`
  and applies it as the statistical factor to every generated reaction,
  independently of rate law type. NFsim computed the correction and then threw
  it away: `ReactionClass`'s constructor scaled its own `baseRate` *argument*,
  which shadows the member the argument had already been copied into. Only
  `Ele` recovered it, because `NFinput` follows the constructor with
  `setBaseRate()`, which applies the factor itself.

  So the issue's report — a symmetric rule with a *functional* rate firing at
  2x — was one of four. Every rate law that is constructed with `baseRate=1`
  and never routes through `setBaseRate` was affected: a global function
  (`FunctionalRxnClass`), a local function (`DORRxnClass`), a function product
  (`DOR2RxnClass`), and Michaelis-Menten (`MMRxnClass`). Measured on
  `tests/data/nfsim/symmetry_factor_rate_laws.xml`, where five pools of 4000
  dimers decay under five rate laws that all encode the same per-dimer rate, so
  the correct survivor count at `t=1000` is 1471.5 and the 2x-too-fast one is
  541.3:

  | rate law | before | after |
  |---|---|---|
  | symmetric, global function | 539.3 | 1463.3 |
  | symmetric, local function (DOR) | 548.3 | 1479.5 |
  | symmetric, Michaelis-Menten | 527.8 | 1451.3 |
  | symmetric, constant (control) | 1472.2 | 1479.0 |
  | asymmetric, global function (control) | 1459.5 | 1464.8 |
  | asymmetric, Michaelis-Menten (control) | 1469.3 | 1470.3 |

  Assigning through `this->` repairs the two DOR classes, which build the
  propensity as `a = baseRate * ...`; `setBaseRate()` assigns rather than
  multiplies, so `Ele` does not double-apply. `FunctionalRxnClass::update_a()`
  and `MMRxnClass::update_a()` never read `baseRate` at all — they override
  `BasicRxnClass::update_a()` and rebuild the propensity from scratch — so both
  now scale by it like every other rate law class. The RuleMonkey-exact entry
  points inherit the fix.

  Michaelis-Menten needs the factor in a different place from the others. What
  the factor corrects is a *match multiplicity*: `getCorrectedReactantCount(0)`
  counts pattern embeddings, and a symmetric substrate pattern matches each
  complex twice, so the law is handed `2N` substrate when only `N` complexes
  exist. Scaling the finished propensity is therefore exact only where MM is
  linear in that count; scaling the substrate count is exact at any saturation,
  and the two agree wherever the law is linear. Measured at `X0/Km = 0.4`,
  where the two placements separate (10 seeds, `t=2000`; the pairing is the
  oracle — the two rules differ only in whether the substrate dimer's halves
  are the same molecule type, so they must land together):

  | | symmetric | asymmetric |
  |---|---|---|
  | no factor at all | 153.0 | 754.8 |
  | factor on the propensity | 977.0 | 746.0 |
  | factor on the substrate count | **758.0** | **756.3** |

  Only the substrate needs it: an MM rule does not transform its enzyme, and
  BNG's `MultScale` counts reaction-center automorphisms, so an enzyme-side
  symmetry comes through as `symmetry_factor="1"` and this factor is always the
  substrate's. Guarded by `symmetry_factor_mm_saturated.xml`, which the
  companion linear-regime fixture cannot replace — below saturation the two
  placements are numerically identical.

  Separately, NFsim *does* over-count a symmetric reactant pattern that carries
  no reaction center, at N× for a homo-N-mer, and BNG attaches no
  `symmetry_factor` to that shape at all. That is a distinct defect from this
  one and is tracked in issue #281.

  Ships as vendored-NFsim carry `bngsim/carry-symmetry-factor-all-rate-laws`;
  candidate to push upstream, where the defect is ~14 years old and untouched
  on `RuleWorld/nfsim` master.

- **A parallel fan-out could still die with no diagnosis: clones shared one
  ExprTk parser, and a parser keeps a strong handle on the last symbol table
  compiled through it (issue #257).** #201 found two threads compiling through
  one `exprtk::parser` and serialized `compile()`. The regression test it added
  then failed about **10% of the time on `main`** — 2 of 20 consecutive runs,
  and it took a `git push` down through the pre-push hook once. The residue was
  not the `compile()` race: the macOS crash report puts the faulting thread in
  `NetworkModel::clone()` and another, concurrently, in
  `ExprTkEvaluator::evaluate()` under `CVode` with the GIL released — an
  evaluator being *read* against an evaluator being *constructed*, which a mutex
  around `compile()` cannot cover.

  The mechanism is one line of ExprTk. `parser::compile()` ends with

      symtab_store_.symtab_list_ = expr.get_symbol_table_list();

  and never clears it, so a parser retains a *strong handle* on the symbol table
  of the last expression compiled through it — and ExprTk refcounts symbol
  tables with a plain `std::size_t`. One parser behind two evaluators therefore
  means thread B's compile dropping thread A's symbol table while thread A is
  churning that same counter with no lock at all: in `register_symbol_table`,
  in the growth of its expression vector, in its own destructor. A lost update
  runs `clear()` on a symbol table whose variable addresses are already baked
  into A's live compiled nodes. The failure is a corrupt heap, so it surfaces
  wherever the *next* allocation looks — the reported signature is `SIGTRAP`
  from the macOS malloc freelist check, which is none of the three signals #201
  was calibrated against.

  No lock could have closed it, because the counter is touched on paths no
  evaluator API sees. So the sharing is gone instead: **an `ExprTkEvaluator` now
  owns all of its ExprTk state and shares none of it** — parser, symbol table,
  expression list, function adapters — and `clone_empty()` hands back an
  evaluator that inherits nothing. The mutex #201 added is gone with it.

  The performance argument for sharing turned out to be illusory. It existed to
  avoid re-constructing the ~100 KB `exprtk::parser<double>` (51 µs, measured)
  per model clone — but `NetworkModel copy;` default-constructs an evaluator that
  `clone()` replaces one line later, so that construction was being paid and
  thrown away on every clone anyway. The parser is now built lazily on first
  compile, which makes the discarded evaluator free, and the clone that needs a
  parser builds exactly the one it uses. `NetworkModel::clone()`, before → after:

  | model | before | after |
  |---|---:|---:|
  | `BIOMD0000000701.xml` (71 params, events) | 521 µs | 527 µs |
  | `func_composition.net` | 92 µs | 94 µs |
  | `simple_decay.net` (no expressions) | 67 µs | **34 µs** |

  Unchanged where the clone compiles anything, and halved where it compiles
  nothing — that model used to construct a parser only to discard it.

  Every `run_batch` / `parameter_scan` / `compute_all_sensitivities` fan-out over
  a model with events was on this path, and the failure was a silent process
  death rather than an exception — a long job lost a worker with nothing to read.
  Two new cases in `tests/test_bngsim.cpp` cover it, both run against the pre-fix
  design before being trusted: the invariant directly (a clone's parser is not
  its source's, via the new `ExprTkEvaluator::parser_identity()`), and the race
  it prevents (8 threads cloning, compiling and evaluating). Rebuilt on the old
  shared-parser evaluator, the test binary died in **8 of 8** runs, always inside
  that second case, always on the `SIGTRAP`; a standalone harness at the same
  parameters put the rate at 19 of 20 over a wider sample and turned up `SIGSEGV`
  and `SIGBUS` as well. On the fix, 20 of 20 clean.

  `python/tests/test_expression_parser_thread_safety.py` — the #201 test that was
  flaking, 18 of 20 on `main` — is **25 of 25** clean on the fix, and its failure
  message now says which of the two defects a given signal would mean.

- **A parameter that a same-named function shadows was still listed as a knob
  (issues #256 and #266 — the same defect reported from the SBML side and the
  `.net` side).** Every function gets a parameter slot to hold its evaluated
  value, and `evaluate_functions()` overwrites that slot before every derivative
  evaluation. #227 flagged the slots bngsim *synthesizes*; it left the other
  branch of the same binding loop, where the function's name matches a parameter
  the input already declared and the function binds to **that** slot. The engine
  overwrites it exactly the same way, so the number in the `parameters` block is
  the seed the slot holds until the function first evaluates — never a knob. But
  `primary_param_names` listed it, so an optimizer handed that list spent a
  coordinate on a column that is identically zero, and `set_param` accepted a
  write that `get_param` echoed back and the next RHS evaluation discarded.

  This is the SBML `<assignmentRule>` shape: the parameter keeps its initial
  value and the rule becomes the function. Measured before and after on two
  binaries, it is much more common than either issue's reproducer suggested:

  | corpus | models | carrying the shape |
  |---|---:|---:|
  | tracked `.net` | 140 | **0** |
  | tracked SBML | 327 | **107** |
  | generated `.net` under `benchmarks/` | 677 | 4 |

  `BIOMD0000000613` alone leaked 141 names; #256's own reproducer
  (`BIOMD0000000701`) leaked 12, and its default sensitivity column set goes
  36 → 24. The sweep that should have caught this
  (`test_primary_param_names.py`) globs `*.net`, and the shape reaches `.net`
  only through conversion — so the gap was never tracked-vs-untracked, it was
  one input format's sweep standing in for both. `tests/data/
  shadowed_function_param.net` now carries the shape in the format those sweeps
  read.

  A second, quieter half: a shadowed row written as arithmetic (`recycle 1/4`)
  was already kept out of the list, because the `.net` reader guesses
  `is_expression` from the value text — the right outcome for the wrong reason,
  and #203 reported it to the user as a *derived* parameter "not independent of
  its primaries". It is not derived; nothing recomputes it from primaries, a
  function overwrites it. The fact now rides `is_internal` and the derived flag
  comes off, which is also what keeps the two flags disjoint —
  `primary_param_names` is the residue of subtracting both, so a row carrying
  each would be reported under neither reason.

  That row also carried a live violation of #181's "one rule, two readers"
  invariant. The build had an explicit carve-out — a function-bound parameter
  was exempt from #261's "a constant written as arithmetic is a knob" demotion —
  so the loader called `recycle 1/4` derived while `_classify_parameter_kinds`,
  reading the same line, called it a constant. The two `.net` readers are
  supposed to partition the parameter block identically, and that is what makes
  the model-based and text-based codegen paths emit the same sensitivity RHS.
  No corpus model has the shape, so nothing could see it; the new fixture fails
  `test_the_loader_agrees_with_the_codegen_net_parser` against the old binary.

  Classification only: `is_internal` is read in exactly one place that affects
  behaviour (the `set_param` refusal). Across 1819 `.net` models the emitted C
  and the trajectories are byte-identical, and exactly 5 models change
  classification — the 4 known plus the new fixture. No `_CODEGEN_VERSION` bump
  is needed, which is a measurement and not an argument: codegen re-derives its
  `is_const` from the expression text rather than from this flag, so the emitted
  source did not move even for the model whose flag flipped.

  `set_param`'s refusal was reworded. It said the name "is not a parameter of
  the model but a function" — true of a synthesized slot, and exactly what a
  reader looking at their own `parameters` block would dispute.

- **`rebuild_editable.py` picked `--no-build-isolation` on the strength of `pip`
  alone, the inference `ship_wheel.py` exists to reject (issue #271).** An
  unisolated PEP 517 build needs pip *and* `[build-system] requires` importable
  in the same interpreter, and the two come apart: a `uv venv --seed`
  interpreter has pip and no `scikit-build-core`, because the backend lives only
  in `[build-system] requires` and uv puts that in a transient build env — the
  same fact behind #229. Measured on exactly such an interpreter, the command
  the script chose died with `BackendUnavailable: Cannot import
  'scikit_build_core.build'`, from inside pip's vendored `pyproject_hooks`,
  naming neither the missing dist nor the fix.

  Reachable through both callers — `_bootstrap_editable` (no build metadata for
  this interpreter) and `_refresh_editable_metadata` (a `pyproject.toml` version
  bump, since `cmake --install` refreshes the extension but not the dist-info).
  A uv venv has no pip at all and so never saw it; a pip-carrying venv whose
  editable install was built isolated hits it on the next version bump.

  `scripts/ship_wheel.py:_has_build_deps` already rejects this exact inference
  in a docstring that says so outright ("Having pip is not it"). One dependency
  question answered in two files, fixed in one.

  **The fix is not a refusal.** Dropping the flag lets pip supply the backend
  through its own build isolation, which is a path that works — the same
  `pip install --no-deps -e .` that failed unisolated on that interpreter
  succeeds with isolation and produces a loadable extension. So no environment
  that built before stops building, and the only unrecoverable combination (no
  pip *and* no uv) keeps the named error it already had. The cost is a
  from-scratch build on that branch, which is why it is the fallback.

  The module list is **read from `[build-system] requires`** rather than copied
  into a third constant: ship_wheel has to hardcode its copy because it probes
  *foreign* interpreters, while this script only ever targets the one it is
  running in and is always run from a checkout. An unparseable table is treated
  as "assume the deps are absent" — the direction that costs time rather than a
  build. A test pins the parse against `tomllib` and against ship_wheel's copy.

- **`scripts/rebuild_editable.py` could not configure in a uv venv, and it is the
  remedy the stale-binary guard names (issue #229).** The script drives `cmake`
  directly against the environment it runs in, so `find_package(pybind11)` has to
  resolve from there. pybind11 is declared only in `[build-system] requires`,
  which uv supplies in a transient isolated build env and never installs into
  `.venv` — no extra declared it either. Someone who hit the staleness guard and
  followed its instruction got a CMake error 900 lines into someone else's build
  system, and the only other thing the guard offers is
  `BNGSIM_ALLOW_STALE_CORE=1`: proceed against a binary that does not match the
  source, which is exactly what the guard exists to prevent. The failure pushed
  people toward the unsafe escape hatch, on the path designed to keep them off it.

  Latent rather than always-broken, and the reason it took this long to surface
  is worth stating precisely. `CMakeLists.txt` already probes a list of candidate
  interpreters for `pybind11.get_cmake_dir()` before `find_package`, so the
  script worked wherever *any* of them had pybind11 — a venv that kept a copy
  from an earlier `uv pip install`, a `python3` on `PATH` that happens to carry
  one, or a system-wide install (Homebrew ships `pybind11` under
  `/opt/homebrew/share/cmake`). It broke the moment none of them did, and
  CONTRIBUTING.md's own `uv sync --reinstall-package bngsim` line was enough to
  get there: without extras named, that command prunes the venv on its way to
  rebuilding. That is how the report started.

  Two halves. `pybind11>=2.13` is now declared in the `dev` extra — the same
  specifier as `[build-system] requires`, since it is one dependency reached by
  two routes, with a test that fails if they drift. And the script asks its own
  interpreter and pins `-Dpybind11_DIR`, which fixes more than the error: this
  machine's build directory had cached Homebrew's pybind11 **3.0.2** while uv's
  isolated build used **3.1.0**, so the two documented rebuild paths were
  compiling the same source against different pybind11 versions, silently. The
  pin ends that — the cmake rebuild now uses the pybind11 the project resolves.

  Measured on one box by hiding every pybind11 the machine had: the old script
  died with the reported `Could not find a package configuration file provided by
  "pybind11"` plus a `CalledProcessError` traceback; the new one prints
  `pybind11: not importable in this interpreter` *before* the configure and, when
  it fails, exits with a note naming `uv sync --extra dev` and the isolated-build
  alternative. With pybind11 present the same configure succeeds where the old
  one could not. A missing pybind11 is deliberately **not** a refusal — cmake can
  still find a system copy, and plenty of machines rebuild that way today.

  The guard message at `_build_provenance.py` gained a conditional line: when
  pybind11 is not importable it says so next to the remedy, so a remedy that may
  fail no longer sits alone beside `BNGSIM_ALLOW_STALE_CORE=1`. CONTRIBUTING.md's
  rebuild line now names its extras.

- **The `^` → `pow()` rewrite split a scientific-notation literal at its
  exponent sign, emitting C that does not compile (issue #240).** Both operand
  scans in `_codegen._replace_power_op` walk the character class `alnum . _`,
  which the `+`/`-` of a signed exponent is not in — so the scan stopped inside
  the number: `2.4279e-09^1.6123` became `2.4279e-pow(09, 1.6123)`, and on the
  other operand `x^1e-3` became `pow(x, 1e)-3`. An *unsigned* exponent
  (`1.5e3^2`) was always fine, which is why it took a signed one to surface.

  `clang` rejects the result twice over (`exponent has no digits`, and `09` is
  an invalid octal constant), so the model lost codegen entirely rather than
  computing a wrong number. The visible symptom was downstream and misleading:
  `MODEL1108260014` reported `Could not generate an analytical sensitivity RHS
  for this model`, which reads as a statement about rate-law
  differentiability rather than about the printer.

  Both scans now take a numeric literal as a unit. The gate is deliberately
  narrow: the base-side extension only fires when the run collected is all
  digits and the characters before it spell `<mantissa>[eE][+-]` with the
  mantissa not itself the tail of an identifier, so `k_2e-3^2` stays
  `k_2e-pow(3, 2)` — a subtraction, not a literal.

  Blast radius, measured: every string in `codegen_data()` for all 1,537 models
  of `parity_checks/rr_parity` + `benchmarks/sbml_events`, rewritten both ways.
  Two expressions change, both `MODEL1108260014` (it is in both corpora), both
  carrying the split-literal artifact before and none after. That model now
  builds its analytical sensitivity RHS, and its compiled and interpreted
  trajectories agree to 8.9e-12 — the ExprTk path always read the literal
  correctly, since only the compiled path goes through this rewrite.

- **`compute_all_sensitivities`: whether the solve succeeded depended on
  `chunk_size`, a documented performance knob (issue #243).** A chunk is
  `chunk_size` sensitivity columns sharing one CVODES error test, so the
  grouping decides whether a marginal column's step is accepted. On
  `BIOMD0000000044` the same 23 columns raised at `chunk_size` 1, 2 and 4 and
  returned at 3; `BIOMD0000000166` had its own version of the same. Nothing in
  the failure said the scope was the chunk — the bare `CVODE integration
  failed` message reads as a statement about the model — and one marginal
  column took down the other 22 with it.

  A failed chunk is now retried one column at a time before anything is
  reported. Every column integrating alone means the grouping was the problem:
  the tensor is completed from those single-column solves and the call
  succeeds, with a warning saying so. A column that fails alone is
  unresolvable, and the `SimulationError` names it, names the columns that were
  fine, states that the scope is the chunk and that another `chunk_size` may
  succeed, and prints the `params=[...]` that keeps every computable column —
  the bisection a caller previously had to do by hand. The retry costs one
  solve per column of a failing chunk, so past 16 columns it is named rather
  than run, and the message then does not claim to know which column is at
  fault.

  Also caught by the same work: a CVODES failure is not the only way a column
  can be unusable, and not the way that hurts. At `chunk_size=3`
  `BIOMD0000000044` returned *success* with `_lp_v7_n` NaN from the first output
  point on — the columns sharing its chunk passed the error test on its behalf.
  `Result.gradient` contracts the whole parameter axis, so one such column NaNs
  every entry of the gradient. A non-finite sensitivity block is now a chunk
  failure and takes the same retry (rows the #221 assignment-rule redirect
  refuses on purpose are excluded — `Result.ar_sensitivity_refused` is that
  set). Both models the issue names now reach the same verdict at every
  `chunk_size`, naming the same marginal column.

- **Nothing asserted whether the GH #88 periodic step bound is still necessary,
  and the module docstring still claimed it was (issue #262).** The bound
  derives a `max_step` ceiling for models whose dosing schedule is periodic
  `floor`/`modulo` arithmetic; 25 of the 1,323 `rr_parity` models get one. GH
  #259 gave `MODEL1708310001` five more discontinuity roots, they bracket the
  same pulse edges, and the test that used to witness necessity became a test
  that the roots resolve the schedule *without* the bound — leaving
  `test_sbml_periodic_floor_dosing.py`'s header asserting the opposite of what
  its tests do.

  Measured, as #262 asked: a two-arm sweep (`max_step` default vs
  `max_step=-1`) at rtol and rtol/100 over all 25, each arm scored against its
  own tolerance stability. No model's answer the bound changes. 23 agree to
  within their own tol-stability; the other two (`BIOMD0000000858`/`859`) are
  not tol-stable in either arm and are byte-identical between them, so they
  cannot adjudicate. The bound is inert on 15 of the 25 — identical step counts
  in both arms — and where it binds it costs up to 2.5x the steps
  (`MODEL0406553884`: 439,367 vs 176,919) for no change in the answer.

  That includes the class #262 flagged as the one roots provably cannot reach:
  seven of the 25 carry a bound and *zero* discontinuity roots, and disabling it
  changes nothing on any of them. Two tests now pin this — `BIOMD0000000312`
  (bound, no roots, and the bound does bind, so the agreement is not vacuous)
  and `MODEL0847869198` (rooted, bound binds hard: >20% more steps for a
  1e-9 difference) — and each asserts its own precondition, so neither can decay
  into a vacuous pass the way the necessity claim did. The docstring now records
  the measurement instead of the stale claim. Narrowing or retiring the bound on
  the strength of it is a separate call and is not made here.

- **`BNGSIM_KLU_AUTOBUILD` could not complete on a host with no BLAS, which made
  a source `pip install` on a bare Linux box a hard configure failure rather
  than a dense-only fallback (issue #178).** The autobuild exists so a box with
  no system SuiteSparse still gets the sparse solver, and `pyproject.toml` sets
  `BNGSIM_REQUIRE_KLU=ON` for every scikit-build-core build — so an autobuild
  that cannot finish does not degrade quietly, it errors out. That was every
  stock Linux host: HPC modules, conda-less clusters, `pip install --no-binary`,
  aarch64 and musllinux (which `cibuildwheel` skips), and any Python outside the
  `cp310-cp313` wheel range. Published wheels bundle KLU and were never affected.

  The blocker was never KLU. KLU is a sparse LU with a BTF/AMD/COLAMD
  preordering and calls no BLAS — nor do AMD, COLAMD or BTF; none of the four so
  much as mentions one in its CMake. `SuiteSparse_config`, which every component
  includes, probes for one unconditionally, and that module ends in
  `find_package(BLAS REQUIRED)`. On an image with no BLAS this killed the
  configure inside `SuiteSparse_config`, before KLU was reached at all. macOS
  survived only because Accelerate answers the probe.

  `SuiteSparseBLAS.cmake` has an opt-out the earlier work believed did not exist
  — `ci/build_suitesparse.ps1` said "no opt-out" and downloaded OpenBLAS on
  Windows purely to answer the probe. Defining `BLAS_LIBRARIES` takes its
  user-supplied-BLAS early return, which skips the probe outright; both
  `ci/build_suitesparse.sh` and `.ps1` now pass it empty, and the Windows
  OpenBLAS download is gone with it. Nothing is lost: the probe is bookkeeping
  for the packages this subset does not build (`SuiteSparse_config`'s own CMake
  says it "does not itself require the BLAS"), and the branch taken still
  includes `SuiteSparseBLAS32`, so `SuiteSparse_BLAS_integer` lands on `int32_t`
  either way. Measured probe-found vs probe-skipped on the same source tree:
  identical installed file sets, identical `SuiteSparse_config.h`, identical
  defined-symbol tables in all five libraries, and in both arms zero BLAS
  symbols and no BLAS linked.

  `python-tests.yml`'s Linux leg no longer installs `libsuitesparse-dev`. That
  step was the workaround for this bug, so dropping it is the standing proof:
  both legs now build the KLU subset from source exactly as an sdist install on
  a bare host does, and the job's existing `HAS_KLU` assertion fails if either
  stops getting it.

- **`not(...)` around a compound condition was declined where the `!` spelling
  of the same condition was admitted (issue #234).** `_split_logical_atoms`
  dropped a leading `!` and split what was under it, but kept a `not(...)` call
  whole. Negation is the one operator with two spellings here, and the whole one
  was the reading every real model got: `_ast_to_exprtk` renders `<not/>` as the
  call form (and `<implies/>` as `(not(a)) or (b)`), while this build's ExprTk
  rejects `!` outright (`ERR007`/`ERR248`) — so `!((a>1) && (b>2))` split into
  its two surfaces at the splitter and `not((a>1) and (b>2))`, the same window
  and the only one of the two a model can be written in, did not. (That corrects
  the issue's framing: no loadable `.net` model was getting the good reading
  either.) Kept
  whole, the atom is neither a clock threshold nor a rootable comparison, so the
  model's analytic sensitivity RHS was declined and it ran on CVODES' internal
  difference quotient. On #232's reproduction written as De Morgan's complement
  (`piecewise(k_base, not((X<8) and (X>3)), k_boost)`, closed form
  `dX(6)/dk_boost = -1.3120451477`) that is **18 % wrong** at `rtol=1e-8` and
  raises `SimulationError` at `1e-10` — the same difference-quotient-at-a-state-
  switch mechanism #232 measured.

  Negation is now **peeled rather than interpreted**, in both spellings and
  through one helper. That is sound for these callers precisely because they ask
  only *where* the branch flips: the core reads f⁻/f⁺ by evaluating the real RHS
  on each side of the located crossing, never by interpreting the condition, and
  De Morgan supplies the rest — ∂(¬(A∧B)) ⊆ ∂A ∪ ∂B, so the peeled reading names
  no surface the condition does not have and names exactly the pair the
  un-negated spelling already registers. That also answers #153's collision
  question: the negated spelling hands the solver no root the plain conjunction
  does not, so it cannot collide anywhere `(A and B)` does not already. The
  event-trigger path (#49) still **refuses** a negated trigger, and the two are
  not in tension — an event's ∂t\*/∂p is derived for a false→true edge, so that
  reduction orients each atom into a lower or an upper bound and takes
  `t* = max(lower)`; negation swaps those roles, and peeling there would return a
  confidently wrong number rather than a coarser one.

  #232's acceptance criterion is now met in full: all four spellings of its one
  window — `and`, nested `piecewise`, `or` over the complement, and
  `not(... and ...)` — return the same gradient, in the same number of steps, to
  the last digit. **Corpus: 0 models move.** No model in the 1,319-model
  BioModels corpus carries a `<not/>` or an `<implies/>` (126 carry `<and/>`, 18
  `<or/>`, 13 `<xor/>`), and no `.net`/BNGL model in this tree carries a `not(`
  outside a comment — so this lifts a decline that nothing committed exercises.
  Probed `main` against the branch over the 255 condition-bearing corpus files
  (243 SBML + 12 `.net`, 252 distinct models; 230 load, 22 refuse identically on
  both sides, 188 carry a condition, 23,209 condition atoms in total): the
  atoms, the crossings handed to the solver, the `.so` cache digests and the
  gate verdicts are identical on every one.

  Two smaller disagreements of the same shape, found in the same functions and
  fixed here: `!(t>=sigma)` came back from the splitter as `(t>=sigma)`, which
  `_relational_split_op` does not read as a relational atom at all (it stops
  looking at depth > 0), so the `!` spelling of a *clock* threshold was refused
  where `not(t>=sigma)` is now admitted; and `uncompensated_condition_reason`'s
  scan for a comparison outside an `if()` head watched `!` but not `not()`, so a
  rate law of `not(X)` — a step at `X=0`, the same boolean-as-a-number idiom as
  `(X>0)`, which that scan rejects — was admitted and sympy differentiated `~X`
  to a clean `1` with nothing warned.

- **`Model.primary_param_names` was wrong in both directions on a `.net` model:
  it omitted literal-valued constants and listed every function name (issue
  #227).** This is the accessor whose own docstring says to hand it to an
  external optimizer or sampler, and `bngsim.jax.differentiable_solve` takes it
  as the default differentiation set (`flat=False`), so both errors landed in a
  gradient. `benchmarks/models/net/ode/SIR.net` — 23 lines — showed both at once:
  `gamma`, the recovery rate, was absent, and `betaI`, a *function*, was present
  with a gradient of exactly `0.0`.

  **A constant written as arithmetic is a knob.** The loader decided "derived" by
  whether the value text parses as a float, so `gamma 1/7` was derived and the
  list dropped it. But `1/7` references nothing: there is no primary underneath
  it to fit instead, `set_param` moves the trajectory through it, and its
  sensitivity column is live (`max|S| = 8.7e7` on `SIR.net`). BNG2.pl draws the
  line the other way and annotates that line `# Constant`, reserving
  `# ConstantExpression` for a value that names another parameter — which is
  exactly the rule #181 gave the codegen `.net` parser, so since #226 the loader
  and the codegen had disagreed about the same lines. A parameter is now derived
  when, and only when, its expression **references another of the model's
  symbols**; a referenceless one is constant-folded in `ModelBuilder::build()`,
  *after* it is evaluated, which is also what keeps `gamma` at `1/7` rather than
  the `1.0` a partial `stod("1/7")` stops at. 76 parameter lines across 14 of the
  139 `.net` files in this tree move; **0 of the 1,614 SBML models that load do**,
  and the two `.net` readers (`net_file_loader.cpp` and `bngsim._net_reader`) now
  converge on one answer because the classification happens in the builder they
  share.

  **A function is not a knob.** Each function gets a parameter slot to hold its
  evaluated value, and the engine rewrites that slot from the function's own
  expression before every derivative evaluation. The slot set neither flag
  `primary_param_names` filtered on, so it was listed — universally, on `.net`:
  every function-carrying `.net` model leaked every one of its function names,
  because a BNG2.pl `.net` never gives a function the name of a declared
  parameter. `set_param("betaI", 3.0)` was **accepted**, `get_param` echoed `3.0`
  back, and the trajectory did not move. The slot now carries `is_internal`, the flag #170 gave `_V0_<comp>`, for
  the same reason and with the same two consequences: out of
  `primary_param_names`, and a value-changing write is refused with a message
  naming the function instead of silently discarded. On
  `BIOMD0000000701` that is 35 of the 71 default sensitivity columns, every one
  of them identically zero.

  **Breaking**, and deliberately so — a silently-absent parameter and a
  permanently-zero coordinate both survive every check a fitting frontend can
  make locally. The gradient vector `differentiable_solve(flat=False)` returns
  changes *length and order* on any model with a function or with a constant
  written as arithmetic, as does the default `compute_all_sensitivities` tensor;
  `result.sensitivity_params` and `Model.primary_param_names` remain the
  authority on both. `set_param` on a function name now raises `ValueError` where
  it used to return; an unchanged write is still legal, so a full parameter-vector
  round trip is untouched. `flat=True` is now `param_names` minus the synthesized
  slots rather than `param_names`: it writes every name it claims to
  differentiate, and those writes are refused — which also un-breaks `flat=True`
  on any SBML model carrying a `_V0_<comp>`, where it has raised since #170.

  `Model.param_is_internal` and `Model.function_names` are now public, so the
  flat vector and the two exclusions above are computable from the API rather
  than only observable through it.

  What is **not** fixed here is the other half of the same binding: a function
  whose name is a parameter the input itself declared, which is how an SBML
  `<assignmentRule>` arrives. Those are still listed, still writable, and still
  discard the write — 722 of the 1,614 loadable SBML models in this tree carry
  one, 10,954 names in total. Refusing a write to a name the user's own file
  declared wants its own measurement, so it is issue #256.

- **A time threshold with no csymbol, or with no constant side, registered no
  discontinuity root (issue #259).** The GH #72 scan called a relational a time
  threshold when *exactly one* side referenced the `time` csymbol. Two common
  schedules fail that test and were stepped over exactly as #72 and #231 were:

  - **An assignment-rule alias.** A model writes
    `<assignmentRule variable="model_time"> time` and then spends the rest of
    the file comparing against `model_time`. Neither side of
    `model_time >= 0.7` is a csymbol.
  - **No constant side.** `2*time >= time + 0.7` is the `t >= 0.7` edge, and
    `BIOMD0000000589`'s `time >= i*24` with `i := floor(time/24)` is a real
    sawtooth crossing. "Exactly one side" refuses both.

  Both return X ≡ 0.0 on #72's pulse fixture, identically at rtol 1e-6 through
  1e-12 — and note the second does so with its *closing* edge rooted: a window
  whose opening edge is missed is missed however well the other end is
  bracketed.

  The filter now admits a relational when **either** side moves with time, and
  "moves with time" includes the assignment-rule aliases. Admitting one that
  never actually flips costs one ExprTk evaluation per root-function call and
  nothing else: a discontinuity root is the boolean condition itself
  (`gout = evaluate(cond) - 0.5`), so its value is ±0.5 and it can neither
  vanish identically nor be bracketed where it does not change. A missed
  crossing is a wrong trajectory; a spurious candidate is an evaluation.

  Because "either side" is strictly weaker than "exactly one side" over a
  strictly larger set of time-dependent names, **no model can lose a root**, and
  none does. Over the 1,323 `rr_parity` SBML models, comparing registered
  condition strings: 1,312 byte-identical, 11 gain 42 conditions between them,
  **0 lose any**, 0 change load outcome. Keeping the old "exactly one side" test
  while admitting aliases — the naive widening — instead moves 12 models and
  costs `BIOMD0000000589` both of the roots it has.

  What the 11 gain is schedule arithmetic: `X1 := (time - tdose1)/24` dose
  onsets (`BIOMD0000000238`), `daytime` circadian windows (`BIOMD0000000268`,
  `450`, `674`), a `Light_Dark_Tracker` (`BIOMD0000000858/859`),
  `x := time - t0` (`BIOMD0000001007/1009/1010`), a cell-cycle phase
  `ltime < t_cycle/3` where `t_cycle` is itself time-switched
  (`MODEL1006230027`), and `rem_time - floor(rem_time)` chemo intervals
  (`MODEL1708310001`). The arms agree to 4e-10 … 7e-8 at a tolerance tightened
  1e-4x, 8 of the 11 become *more* tolerance-stable, and `rr_parity` verdicts
  are unchanged (12/12 PASS including `BIOMD0000000589` as the no-loss control).

  One consequence worth naming: `MODEL1708310001` no longer needs the GH #88
  periodic step bound. With the bound disabled it used to jump its pulses in 221
  steps and overshoot to y(100)≈1602.95 against an exact 953.07; it now reaches
  953.069 without it. It was the only test in the suite asserting that bound is
  necessary, so that test now asserts the roots resolve the schedule instead,
  and whether the bound still earns its place anywhere is issue #262.

- **A time threshold inside a called `<functionDefinition>` registered no
  discontinuity root, so the window was stepped over (issue #231).** The GH #72
  scan walked the *call site's* AST only. A schedule written one level down —
  `pulse(tt, t_on, t_off, amp) := piecewise(amp, tt >= t_on and tt <= t_off, 0)`,
  called with the `time` csymbol — put the relational in the callee's body,
  against the callee's formal, where the scan never looked. GH #194 closed this
  gap on the state side; this is its time twin, and it shares #194's frame walk
  rather than adding a second one.

  It is a wrong trajectory, not only a structural gap. On the #72 chemo-pulse
  fixture with the piecewise moved into a function definition, the 0.05-wide
  infusion is never delivered and `X(3)` comes back **exactly 0.0** against a
  closed form of 0.514 — identically at rtol 1e-6, 1e-8, 1e-10 and 1e-12. An
  error that does not move across six decades of tolerance is a missing root,
  not an accuracy shortfall. Post-fix that model is bit-identical to the inline
  one at every tolerance.

  Two pieces the descent needs and would be silently wrong without:

  - **The reaction's `<localParameter>` map**, threaded into the time scan the
    way #194 threads it into the state scan. `MODEL2105110001`'s four `t_switch`
    thresholds are kinetic-law parameters with no global of that name, so an
    unmangled condition is an undefined ExprTk symbol: the model does not load
    at all (`failed to compile discontinuity trigger '(time()<t_switch)'`).
  - **Assignment-rule aliases of the csymbol.** `BIOMD0000000255` calls
    `stepfunc(model_time, 1799.99, …)` where `model_time` is an assignment rule
    that *is* `time`, so nothing in the argument is a csymbol. A formal bound to
    such an alias counts as time-dependent.

  Corpus blast radius over all 1,323 `rr_parity` SBML models, comparing the
  registered condition *strings* rather than their count: **1,316 byte-identical,
  7 gain conditions, none lose any, and no model changes load outcome.** The
  seven agree with their pre-change selves to 1e-12 … 2e-8 at a tolerance
  tightened 1e-4x — the roots change how fast a model reaches its answer, not
  the answer — and total internal steps over them fall 3,945 → 3,924. Cross-engine
  `rr_parity` verdicts are unchanged on all seven (6 PASS, 1 REFERENCE_FAILED
  from RoadRunner's own `CV_TOO_MUCH_WORK`, both before and after).

  Deliberately **not** widened: a threshold written against a time alias at the
  *call site* (`model_time > 1800` directly in a rule) still registers nothing,
  as before. That reading subtracts as well as adds, because a side counts as
  the time side only when the other does not — `BIOMD0000000589`'s
  `time >= i*24` with `i := floor(time/24)` becomes time-vs-time and loses both
  roots it has today. Tracked separately as issue #259.

- **A rate law using `sign`, `floor` or `ceil` over a state variable emitted a
  broken Jacobian term instead of falling back (issue #250).** Those three
  differentiate to an unevaluated sympy `Derivative`, and `_is_emittable` scanned
  `atoms(sp.Function)` only — `Derivative` is not a `Function`, so it passed the
  gate and both printers printed it verbatim:

  ```
  sympy_to_exprtk -> 'Derivative(sign(x), x)'
  sympy_to_c      -> 'Derivative(((double)((0.0 < (x)) - ((x) < 0.0))), x)'
  ```

  `Derivative` is not an ExprTk builtin and not a declared C function, so what
  reached the emitter was not a usable derivative. `_is_emittable` now rejects any
  unevaluated `Derivative`, which covers `differentiate_rate_law`,
  `sympy_to_exprtk` and `sympy_to_c` in one place — the three call sites — and the
  affected model falls back to finite differences as it should. No corpus model
  currently reaches this (`BIOMD0000001072` carries `floor` but declines earlier),
  so it is a latent defect fixed rather than an observed one.

- **`CONTRIBUTING.md` named three of the four modules the codegen source digest
  covers, so it told you to bump `_CODEGEN_VERSION` for an edit the digest
  already catches (issue #267).** The "Changing generated code" section is how a
  contributor decides whether their change needs a hand-written version bump: the
  digest covers the modules on the list, and the constant is the escape hatch for
  everything else. #51 wrote that paragraph out longhand when the list held three
  names; #68 added `_switch_sensitivity` to `_CODEGEN_SOURCE_MODULES` — precisely
  because an edit there changes which models get an analytic sensitivity RHS at
  all — and did not touch the prose, which nothing tied to it. The function's own
  docstring carried a second copy of the stale count ("three file reads (~350
  KB)"; four files, and 648 KB by now, the modules having roughly doubled since).

  Erring toward a bump is the safe direction — over-invalidation costs one
  recompile, under-invalidation is a silently wrong gradient, which is #51's
  whole point — but it is not free: a bump discards every user's cache on the
  next release and adds an entry to a comment block that is a curated record of
  real reasons.

  Fixed by deleting the second copy rather than re-syncing it, since re-syncing
  leaves the same trap set for the next module anyone adds:
  `_CODEGEN_SOURCE_MODULES` is now the only list, `CONTRIBUTING.md` points at it,
  and the docstring states neither a count nor a byte total. Two tests hold that
  — one that the pointer is present, one that a restated list may not be a
  *subset* of the tuple, which is the exact shape that drifted. (The shipped #51
  changelog entry below still describes the list as it was then; it is a dated
  record and is left alone.)

- **`steady_state()` reported an SBML `<assignmentRule>` species at its frozen
  initial value, and that species' zero Jacobian row refused the whole model's
  sensitivity solve (issue #247).** #221 fixed the time-course path; the
  steady-state one ran neither of its two passes, and the consequences there were
  worse, because the **value** was wrong and not only the derivative.

  A rule-target species is emitted `fixed`, so it is not an unknown of `f(y) = 0`
  at all — its value is dictated by the rule and its Jacobian row is identically
  zero. `ss.concentrations` therefore held whatever the slot was seeded with at
  t=0: `2.0` on the issue's fixture where the steady value is `20.0`, while
  `run()` on the same model reports `20.0`. Two entry points, one quantity, a
  factor of ten, with the steady-state one presenting an initial condition as an
  equilibrium. Across the rr_parity corpus, **790 reported values were wrong in
  198 models, and in 146 of them by more than 50%**.

  The zero row also made `J` structurally singular, so `-J⁻¹·(∂f/∂p)` refused
  **the entire model** — including the perfectly well-posed gradient of every
  integrated species — under a message about a conservation-law continuum, which
  is a real but different cause. One assignment rule was enough. **91 corpus
  models now return a steady-state gradient that was refused outright**, with no
  model losing one.

  Both halves mirror machinery that already existed. The value comes from the
  rule's observable / function evaluated at the returned state — the steady-state
  analogue of the `_apply_ar_report_map` pass `run()` has always had, using the
  same `update_observables` + `evaluate_functions` pair the RHS uses. The species
  is folded out of the solved subspace exactly as issue #74 folds out a
  write-only accumulator: an accumulator contributes a structurally zero
  *column*, a rule target a zero *row*, and either one makes the system singular.
  Its `dY_ss/dp` row is then the chain rule through the assignment, which is what
  #221 fills the time-course tensor with. A caller's own `mask=` is intersected,
  never overridden.

  Excluding these species also takes them out of the residual average, which is
  the right reading rather than a side effect — a slot whose derivative is
  identically zero contributes nothing to "has the system settled" but does
  inflate the divisor, making `tol` easier to meet the more rules a model has. On
  the corpus that changes **no** convergence verdict, and no non-rule species'
  reported value moves anywhere.

  `steady_state_batch()` builds its results from a per-entry clone and needed the
  same two passes: without them a dose scan reported every entry at the seeded
  value. Its divisor is resolved against the clone, so a scan that moves a
  compartment volume moves the rescale with it.

- **An SBML `<assignmentRule>` species had an identically-zero sensitivity row,
  with no warning (issue #221).** A species an assignment rule defines has no ODE
  of its own: BNGsim emits its state slot `fixed` and overwrites the reported
  column each step from the rule's live value. The forward-sensitivity tensor was
  left as the integrator wrote it — the `yS` of that frozen slot, which is
  identically zero because its variational right-hand side is zero too. So
  `result.species[:, i]` and `result.sensitivities[:, i, :]` described different
  quantities, and the zero was silent. `Result.gradient` / `sse_gradient`
  contract a `dL/dY` built from `result.species` against that tensor
  **row-for-row**, so a fit scoring an assignment-rule species — `IRS_total`,
  `InR_active`, the *reported* quantities of a published model, which is what
  assignment rules are for — got a gradient that was zero in every direction and
  read it as a flat objective rather than as a missing term.

  The row is a chain rule away, and the run already computes it. GH #205 routed
  the `output_sensitivities("species:<name>")` *selector* through the rule's
  observable (linear-on-species) or expression; the tensor now carries the same
  row, so the two agree by construction and every consumer that never goes
  through a selector works. On `Smith_BMCSystBiol2013` — the model the issue was
  filed from — a central difference of the model's own trajectory at `rtol 1e-11`
  over three relative steps agreed with **0 of 52** resolvable assignment-rule
  entries before and **52 of 52** after, to 8.5e-6.

  Across the rr_parity corpus, 257 of the 1,291 loadable models carry
  assignment-rule species (955 of them). All 257 were probed; of the 639 such
  rows reachable in the 215 that run under sensitivities, **590 now carry the
  chain rule and 551 of those were exactly 0.0 before**. **No
  non-assignment-rule row moved, in any model** — the pass fills the derivative
  exactly where the value pass fills the value, and a species whose rule source
  is not reported keeps its frozen value, whose derivative the raw `yS` already
  is.

  The remaining 49 rows (31 models) are `NaN` rather than 0.0, which is the point
  of the issue: a structural zero is indistinguishable from a measured one.
  Almost all are rules codegen already declined to differentiate (a `piecewise`
  lowers to an `if()`, which #198 refuses rather than guess at) — the species row
  now agrees with the expression row it mirrors instead of contradicting it; the
  rest are species whose reported value carries a time-varying volume rescale the
  redirect does not model. The run warns naming them, `Result.output_sensitivities`
  raises with the specific reason, and the new `Result.ar_sensitivity_refused`
  reports them programmatically (it survives an HDF5 round trip).

  A `NaN` row does not cost you the rest of the gradient. IEEE makes `0 · NaN`
  NaN, so one unknown row would otherwise poison every parameter of a fit that
  never scored that species; `gradient`, `sse_gradient`, `chi2_gradient` and
  `neg_log_likelihood_gradient` now drop a refused row wherever `dL/dY` is
  exactly zero — which contributed exactly zero anyway, so a run with nothing
  refused takes the untouched `einsum` path. An entry the loss *does* weight
  keeps its `NaN`: that derivative is genuinely unknown and must not come back
  looking like a number.

  Fixed alongside, because it is the same defect one level down: the redirect map
  handed to `Result` kept the **load-time** compartment size while the value pass
  read it live, so on a model with a writable compartment size (#170) the
  reported value and `output_sensitivities("species:<ar>")` were out by exactly
  the write's factor — `set_param("C", 3.0)` on a V=1 load gave a derivative 3×
  the finite difference of the column it claims to differentiate. There is now
  one resolution site, and both passes read it.

- **The integration march could be captured by its own BDF history and sit still
  until the budget ran out (issue #235).** CVODE can reach a configuration where
  the state, the step size, the order and the accumulated history are mutually
  self-consistent at a point that is *not* a steady state, and then reproduce it
  indefinitely. On `ltype_calcium_discontinuous_jacobian.net` the march held
  `h = 8.091`, `q = 2`, every failure counter frozen and the residual constant at
  `1.7400e-07` for more than 100,000 steps, then reported unconverged — while
  holding a state 2e-9 (relative) away from the right answer.

  The trap is in the integrator's history, not in the problem: a fresh march
  started from that exact state converges in **four** steps. So the march now
  notices when the residual has stopped improving and re-initializes, keeping the
  state and discarding the step, the order and the history. Tolerances, user data
  and the linear solver survive `CVodeReInit`, so nothing else changes. Escapes
  are capped, because a model that will not settle is a different answer from an
  integrator that has stalled.

  The captured case goes from 124,189 steps and unconverged to **1,004 steps and
  converged**, and — the load-bearing check — asking it for `tol=1e-13` now
  yields a residual of `5.2e-15`, so it is sitting on the true root rather than
  having clipped a low point in transit. All six parking gaps from #176 × five
  budgets converge, against three gaps that failed at the default. Models that
  were never stuck are untouched: the rule needs 400 consecutive steps without
  improvement, and an ordinary march converges in far fewer.

  Three diagnoses were discarded against measurement, and all three are recorded
  in the tests because each is a plausible thing to re-propose. It is **not** the
  convergence criterion (the same march reaches 5e-15 once it is not captured);
  **not** a residual floor (stuck at 1.2e-7, escaped reaches 1e-13, same model,
  same tolerances); and **not** chatter at the discontinuity, which is what the
  issue was originally filed as. The symptom presented as an isolated-island
  lottery in two unrelated parameters — #176's parking gap and `max_time` —
  because neither creates the trap; they only decide whether a given trajectory
  falls into it, `max_time` because CVODE also derives the initial step from the
  `tout` it is handed. Tuning either one only re-rolls that dice.

- **The sensitivity tensor and `result.species` disagreed about what a species
  index means, so `Result.gradient`'s own documented workflow raised (issue
  #202).** `gradient(loss_fn)` hands `loss_fn` the `result.species` array and
  contracts what it returns against `result.sensitivities`. GH #71 projects
  `species` / `species_names` / `n_species` to the *reported* species — an SBML
  parameter or compartment that an event assigns to is promoted to full
  integrator state but is not a floating-species trajectory column — while the
  tensor kept the promoted rows. So the two arrays handed to the same callback
  were different widths, and the documented scipy objective
  (`lambda sp, t: 2 * (sp - data)`) could not run for *any* `data` shaped like
  `result.species`. **120 of the 212 tracked models under
  `benchmarks/sbml_events` carry that projection.**

  The species (row) axis of `sensitivity_data` and `sensitivity_ic_data` is now
  projected in the pybind layer, where `species_data` / `species_names` /
  `n_species` already project — one rule, one place, and an empty projection
  takes the byte-identical path, so a model that reports every species is
  untouched. Six surfaces were dead on a projected model and all six now work:
  `gradient`, `sse_gradient` / `chi2_gradient` /
  `neg_log_likelihood_gradient` (these three raised a raw NumPy `einsum`
  broadcast error), `fisher_information` with a per-species `sigma` vector,
  `output_sensitivities("species:…")` (which refused by design rather than
  return a mismatched slice), `to_xarray` (conflicting `state` dimension), and
  the JAX bridge, which pairs `species_data` with `sensitivity_data` directly.

  Dropping the promoted rows loses nothing: the loader gives every promoted
  symbol a same-named observable precisely so referencing expressions resolve
  its live value, so its derivative is
  `output_sensitivities("observable:<name>")` — a nonzero, event-stepped column,
  not a placeholder. Verified two ways beyond the unit tests: across the 175
  runnable models of that corpus every same-named observable's output
  sensitivity is a constant multiple of the species' tensor row (the constant is
  1, or the amount/concentration factor when the compartment is not unit-sized),
  and a finite-difference of `result.species` reproduces the projected tensor
  column on the projected models. Two caveats that cost time and are worth
  recording: a species column is a *concentration* while its same-named
  observable is an *amount*, so the two derivatives differ under
  `d/d(compartment size)` and coincide only at V = 1; and an AssignmentRule
  target's state slot is frozen (GH #205), so its raw tensor row is ~0 by
  design.

  Fixed in the same pass, being the same defect one axis over: the observable
  output-sensitivity blocks kept the loader's internal
  `__bngsim_net_rewrite_obs_*` scaffolding rows (issue #61) while
  `observable_names` dropped them, so `output_sensitivities("observable:…")` and
  a named-output FIM refused on every legacy `Sat`/`Hill` `.net` model.

- **The GH #176 fallback tests asserted that FD rescues their fixture, which is
  a rounding outcome rather than a property of the model (issue #176).** The
  fixture's `v_rec = if((-70+V)<-20, 0.5, 0.05)` steps at `V == 50`, and `V`
  obeys `dV/dt = k_v_stim - k_v_leak*V` with `k_v_stim/k_v_leak` = `50/1` — so
  the asymptote *was the threshold*. `V` never crossed the step; it parked on it,
  landing an ulp either side of 50 with the side decided by the last bit. FD's
  rescue exists only on the low side, where its `srur*|V|` = 7.4e-7 perturbation
  reaches across the step and returns the regularizing slope; on the high side
  the perturbation stays in the same branch and returns nothing. That is the
  whole of the reported Linux failure — the same source integrating on one host
  and dying at t≈34.6 on another — and it is the third defect of this shape after
  #228, where the branch taken was decided by which way the last pivot rounded.

  The asymptote now sits a defined `1e-11` below the threshold: 1407 ulps of 50,
  so no host's rounding can move `V` to the far side, and ~5 orders below the
  7.4e-7 FD perturbation, so that perturbation still straddles the step
  everywhere. Both margins are stated in the fixture header, which also records
  that the file is deliberately edited away from its upstream BNGL source.

  The three quarantined run-half tests come off `xfail(sys.platform ==
  "linux", strict=True)` and now assert the *policy* — `auto` is "try analytical,
  then FD", so it reproduces an explicit-FD run taken on the same host, bit for
  bit when FD integrates and identically when it does not — instead of a
  hard-coded "this model integrates". A new `test_auto_is_exactly_the_fd_run`
  carries that contract on every host, and the guard that skips the
  rescue-asserting tests where FD gives up is itself covered, because its branch
  is by construction never taken on the host that usually runs the suite.

  Not fixed here, and filed as issue #235: sweeping the parking gap over 23
  values finds isolated gaps — 2e-12, 9e-11, 1e-10 — where the steady-state march
  burns 68k–147k steps and reports unconverged while both neighbours converge in
  ~600. The fixture sits mid-way through the widest clean run and the four
  steady-state tests pass, but what the march should do when the RHS chatters at
  a state discontinuity needs its own design.

  Two claims in the surrounding prose were wrong and are corrected rather than
  softened. A threshold the state crosses *transversally* does not reproduce the
  failure at all — with the asymptote at 55, 60 or 100 the analytical Jacobian
  integrates the same model fine, because CVODE steps over a lone value jump
  whatever the Jacobian says; the effect needs the trajectory to park. And
  "FD … integrate[s] the model cleanly" was never true in general, so
  `_run_ode_with_jacobian_fallback`'s docstring now says the retry is a second
  attempt and not a guarantee.

- **Writing a `piecewise` condition with `<and/>` declined the whole model's
  analytic sensitivity RHS, and the difference quotient it fell back to was 53%
  wrong (issue #232).** `_functional_rate_law_partials` ends its pre-scan by
  looking for call heads it does not recognize, so a rate law reaching a table
  function or an un-inlined SBML helper declines with a message naming the call.
  That scan is *lexical* — it matches `identifier(` — and in `(X<hi) and (X>lo)`
  the `and` is an infix operator whose right operand merely happens to be
  parenthesized. It read as a call to an unknown function `and`, and since
  `CVodeSensInit1` takes one callback for every column, one such rate law
  declined the entire model. Invisible on the `.net` side, where BNGL writes
  `&&` (not an identifier, so the scan never saw it — 0 of 585 corpus `.net`
  models can reach this), and universal on the SBML side, where `_ast_to_exprtk`
  renders every `<and/>` as the word.

  The measurement that isolates it is one window written three ways —
  `piecewise(k_boost, (X<8) and (X>3), k_base)`, the same window as nested
  `piecewise`, and `or` over De Morgan's complement — against the closed form
  `dX(6)/dk_boost = -1.3120451477`. The `and` and `or` arms came back
  `-6.197678503e-01` (**53% wrong**) at `rtol=1e-8` in 273 steps and raised
  `SimulationError` at `1e-10` and below; the nested arm was right to `2.5e-10`
  in 179 steps. All three now return the same doubles in the same number of
  steps at every tolerance.

  Boolean connectives are now admitted as call heads behind the same
  `switch_scope is not None` gate `if` has always been behind, so a model whose
  conditions were *not* cleared still reports them as unsupported. The set is
  read out of `_LOGICAL_LEVELS` rather than re-listed, so a spelling added to
  the ExprTk→sympy rewrite cannot go missing here; `xor`/`nand`/`nor`/`xnor` are
  deliberately excluded from both, since no rewrite translates them.

  Over the 132 SBML corpus models carrying a MathML logical: 45 go from declined
  to admitted (41 of them gaining an emitted sensitivity RHS; the other 4 hit an
  unrelated decline further down), **0 lose one, and 0 model that was already
  admitted changed a single byte of emitted C**. Over the 80 condition-bearing
  `.net` models, nothing changes at all.

  The new admissions were checked against a central difference of each model's
  own trajectory, re-solved at `p(1±h)` through `set_param` — a valid oracle at a
  state switch, because it re-solves the moved crossing too. Over 20 of them, each
  judged on its most responsive parameter at its own SED-ML horizon, the worst
  analytic-vs-FD relative error is `1.1e-02`, on a model whose FD disagrees with
  itself by `1.2e-02` across two step sizes; 13 of 20 sit at or below the FD's own
  noise and 18 of 20 within 3x of it. No row disagrees at a level the oracle can
  resolve.

- **The decline warning called CVODES' difference quotient "correct, but
  slower"; at a state switch it is neither (issue #232).** For an underivable
  rate law or an exhausted derivation budget the claim is true — the problem is
  smooth and the difference quotient answers the same question. It is false
  whenever the declined model carries a branch crossing whose *time* moves: the
  fallback integrates the variational equation smoothly through the crossing,
  dropping the jump `(f⁻−f⁺)·dt*/dθ`, and for a state threshold its probe
  evaluates `f` at `y + σ·s`, which just past the surface lands on the other
  branch. That sentence is what made the 53% above silent — a reader had no
  reason to distrust the number.

  This half was established by measurement rather than inherited from the issue:
  forcing the *nested* spelling (correct to `2.5e-10`) onto the same fallback
  reproduces the `and` spelling exactly — `-6.197678503e-01`, 273 steps, the
  same `SimulationError` below `rtol=1e-9` — so the error is the fallback, not
  the connective.

  `model_moving_crossings` now answers, per model, which branch conditions can
  cross at a moving time, excluding the two shapes that cannot: a comparison
  naming no symbol, and a clock threshold against a literal (whose `∂t*/∂p` is
  exactly 0, the same ground `clock_crossing_compensated` admits it on — now one
  predicate, `fixed_clock_threshold`, with both readers on it). A decline on a
  model that has one is tagged `DeclinedAtMovingCrossingReason` and gets the
  honest message, ending with the remedy that actually applies: removing the
  named decline restores the analytic path, which does apply the jump. Every
  decline in `generate_sens_from_model` and `_functional_dfdp_terms` routes
  through one place so none can be the quiet one, and the derived-rate-constant
  warning was folded into the same function (byte-identical output) rather than
  left as a second site to drift. 18 SBML and 13 `.net` corpus models are
  relabelled; the wording is unchanged for a model with no moving crossing, and
  the issue #146 class keeps its own class and its own issue #150 pointer.

- **`_split_logical_atoms` never returned on a negated compound condition, which
  is the real mechanism behind issue #216.** It re-descended into any part
  carrying a logical, including one it had not reduced: in
  `not((X<hi) and (X>lo))` — what the SBML loader emits for `<not/>` around
  `<and/>` — the `and` is neither at depth 0 nor inside a strippable paren
  group, so the strip step returned the string unchanged and the function called
  itself on it forever. Any *call* whose argument carries a word-spelled logical
  (`max(a, b and c) > 1`) is the same shape.

  Issue #216 reported the symptom on `MODEL0911047946` — `compute_model_codegen_hash`
  raising `RecursionError` and silently falling back to the source-hash key —
  and attributed it to `_canon_update`'s nesting depth. It is not: the frames are
  `compute_model_codegen_hash → switch_gate_cache_digest → _iter_condition_atoms
  → _split_logical_atoms`, self-recursing. #216's own reproducer no longer emits
  the warning.

  The guard recurses only into a part this pass actually reduced, which is
  conservative by construction: every step above only shortens, so an unchanged
  part is one that could not have returned. Verified rather than argued: over 22
  hand-written condition shapes only the four that used to raise move, and an
  atom-level differential across all 1,908 corpus models (`.net` and SBML, 255 of
  which yield at least one condition atom) moves **exactly one** — MODEL0911047946,
  from `RecursionError` to a list. Every other model's atoms are identical, so
  nothing downstream of them can have moved either. Keeping the unreduced part whole also lands it in the
  class it belongs to (a crossing nobody can locate, declined with the warning
  above) rather than the `RecursionError` it used to be. Splitting the negated
  atoms out instead would lift a decline, so it is tracked as issue #234.

- **A `piecewise` gated on a species, in a rule or a kinetic law, registered no
  discontinuity root, so a narrow state-gated window was stepped over entirely
  (issue #194).** The state twin of GH #72's time roots. The loader registered a
  CVODE root for every inequality against the `time` csymbol in an
  assignment/rate rule or kinetic law, and for every relational atom of an
  *event trigger*, but nothing for a plain state threshold in the math that
  feeds the RHS. `piecewise(k_boost, X < hi and X > lo, k_base)` matched none of
  the three collectors, and a window narrow in *time* — narrow precisely because
  the boosted rate is large — fell inside a single adaptive step. The reference
  case came back at `10·exp(-0.2·6)` to the last bit: not the boost applied
  badly, the boost not applied at all. The tell that this was a missing root
  rather than an accuracy shortfall is that **the error did not move with
  `rtol`/`atol`** — four decades of tolerance left it at `3.97e-03`.

  The fix runs the existing per-atom edge routine
  (`_collect_relational_edge_conditions`, until now used for event triggers
  only) over assignment-rule, rate-rule and kinetic-law math, for the atoms one
  of whose sides reads integrated state — a non-constant species, a rate-rule
  target, or anything an assignment rule builds out of those. Splitting the
  Boolean into atoms is the whole point: over one wide step the conjunction
  `(X < hi) && (X > lo)` reads false at *both* ends and never changes sign,
  while each half does. The scan descends into the function definitions a rule
  or kinetic law calls, binding each callee's formals to the call site's
  argument text, because the threshold is just as often written one level down
  (`MAX(a,b) := piecewise(a, a >= b, b)`); a kinetic-law `<localParameter>` in
  the threshold is emitted with the same `_lp_<rid>_` mangling the RHS uses. A
  threshold neither side of which moves with the state is deliberately left
  unrooted — it can never change sign — and a pure-time threshold stays GH #72's
  single root.

  Reference case: `3.97e-03` → `1.5e-09`, and the residual now falls with the
  tolerance (`3.8e-08` at `rtol=1e-8` → `2.2e-11` at `rtol=1e-12`). Over the
  1,291-model SBML corpus 64 models gain a root and the rest load byte-identical.
  Against libRoadRunner on those 64 nothing changes verdict (52 PASS both arms);
  two rows improve their failing-cell count (BIOMD0000000628 993→434 of 135,135;
  BIOMD0000000923 342→293 of 4,004) and none worsens. At `1e-4×` the sweep
  tolerance the two arms agree to a median `5.7e-10`, so the roots change how
  fast a model reaches its answer, not what the answer is — the three models
  that still differ there fail their own self-convergence check by order 1 in
  *both* arms and are undetermined at any tolerance either can reach.

  The retrigger hazard a state root carries — it is not monotone, so it can be
  re-crossed or grazed — was settled by measurement rather than by adding a
  dormancy guard: total internal steps over the 64 fall from 24,942 to 16,026,
  57 of 63 are unchanged or cheaper (BIOMD0000000831 `891 → 19`,
  BIOMD0000000787 `2,459 → 80`), the worst case is `+13 %`, and no model gains a
  root that chatters. A sliding surface, a grazing one, and a damped oscillator
  crossing its threshold 37 times all get *cheaper* (`4,019 → 89` steps on the
  oscillator). A true Filippov chattering system is unsolvable identically
  before and after.

- **`solver_stats` counted only the segment after the last event fire, so
  `n_steps` could read 0 for a run that took thousands of steps (issue #182).**
  Every `CVodeGetNum*` counter counts from CVODE's last (re-)initialization, and
  this path re-initializes at each event fire, at a switch-time or state-switch
  crossing, and at a chatter re-arm. `record_solver_stats` sampled the counters
  once at the end, so what it reported was the tail after the final restart and
  nothing before it — an under-count on **any** model with events, by however
  much of the run preceded the last fire.

  The issue was found by moving `t_ins` across the end of the span on
  `Smith_BMCSystBiol2013`: 4 steps at 239, **0** at 240, 1065 at 241, with an
  identical trajectory in all three. Zero is the worst of those, because it
  reads as a cheap run rather than as a broken counter — and these are the
  numbers a benchmark or a performance diagnosis records. The smallest form of
  it is one species: exponential decay with a single `time() >= 10` bolus over
  `t_span=(0, 10)` reported `n_steps=0`, `n_rhs_evals=0` for the same 127-step
  integration the undosed model reports in full.

  Each segment's counters are now banked when a re-init closes it and added to
  the still-open segment at the end, so `n_steps`, `n_rhs_evals`, `n_jac_evals`,
  `n_err_test_fails`, `n_nonlin_iters` and `n_nonlin_conv_fails` all cover the
  run. Every mid-run `CVodeReInit` goes through a single accumulate-then-reinit
  helper, so a re-init added later cannot silently drop a segment on the floor.
  Nothing changes for a model that never re-initializes: the banked half stays
  empty and the reported numbers are the ones it reported before — including on
  the warm path, which takes no events. `n_dense_blas_factorizations` was
  already whole-run (it is read off the linear solver, which a re-init does not
  touch), and `SteadyStateResult`'s own `n_steps` / `n_rhs_evals` march on one
  `CVodeInit` with no re-init at all; neither is affected.

  A model that *does* re-initialize now reports a larger number than it did
  yesterday, and anything calibrated against the old one moves with it. One
  assertion in-tree was: `test_surface_reached_but_not_crossed_still_integrates`
  (issue #194) bounded a rooted arrival at `n_steps < 20`, which was the handful
  of steps after the root's re-init — the same run is 39 counted whole, nearly
  all of it spent walking down to the surface. Its bound is recalibrated; what
  it asserts (the root fires once and is not chased) is unchanged.

- **The steady-state conditioning-warning test asserted a pivot of 1.26e-17 —
  below machine epsilon — so which branch it exercised was decided by the LU
  implementation (issue #176, item 2).**
  `test_degenerate_steady_state_is_flagged` used
  `nested_derived_rate_const.net` to cover the "ill-conditioned, so warn" branch,
  with its sibling covering "exactly singular, so refuse". But that model's
  equilibrium set really is a line: `A→B→D` and `A→C` with no reverse steps
  leaves J rank 2 of 4 with one conservation law, so the reduced 3x3 is **exactly
  singular in exact arithmetic**. There is no conditioning to measure — only
  which way the last pivot rounds — and the two roundings fall on opposite sides
  of the branch, a clean `0.0` taking the non-finite refusal and a denormal-ish
  `1.26e-17` passing for a merely ill-conditioned answer.

  #176 recorded this as Linux-vs-macOS and suspected reference LAPACK vs
  Accelerate, noting the discriminating experiment had not been run: "a macOS
  build forced onto reference LAPACK would be the cheap way to confirm it." The
  measurement says the axis is neither. On one macOS/Accelerate x86_64 build the
  *same binary* reports `rcond = 0.00e+00` and refuses under SUNDIALS' built-in
  GETRF, and `rcond = 1.26e-17` and warns under LAPACK `dgetrf`
  (`BNGSIM_LAPACK_DENSE=1`) — same machine, same arithmetic width, opposite
  branches. It is not the platform and not the BLAS; below eps there is nothing
  to be right about.

  The warning branch now has a fixture that is honestly ill-conditioned and full
  rank: two decoupled reversible pairs ten orders apart in rate (`A ⇌ B` at 1,
  `C ⇌ D` at 1e10), each conserving its own total, so the reduced 2x2 has pivots
  `-(kf+kr)` and `-(kff+kfr)` and `rcond` is their ratio, `1e-10`, analytically
  and to the last digit — two orders below `_SS_SENS_RCOND_FLOOR` so the warning
  fires, six orders *above* eps so no rounding can tip it into the refusal
  branch. It is identical under both LU backends.

  The test is also strictly stronger than the one it replaces. The root is
  isolated and reached exactly (all four species at 0.5), and the flagged
  gradient is checked against the closed form `dA/dkf = -kr/(kf+kr)²`, which it
  matches to 1.3e-15 relative. So it now pins what the warning actually claims of
  itself — that the ratio "is wrong in both directions" and the caller may
  overrule it — rather than only that the warning fires. The
  `xfail(sys.platform.startswith("linux"), strict=True)` quarantine is removed;
  the refusal branch keeps its own structural fixture. Items 1 and 3–4 of #176
  (the three `test_jacobian_discontinuous_fallback.py` failures) are untouched
  and that issue stays open.

- **`write_omex` stamped every zip entry with the wall clock, so the archive
  `net_to_omex(created=...)` documents as byte-reproducible was reproducible only
  when two builds landed in the same second (issue #224).** The stated purpose of
  `created` is byte reproducibility, and it reached `metadata.rdf` — but never the
  archive. `ZipFile.writestr` given a *str* name synthesizes a `ZipInfo` stamped
  with `time.localtime(time.time())[:6]`, so every entry carried the build's clock
  to the second: two archives from identical inputs and the same `created`, 1.2 s
  apart, differed only in their entry headers (`(2026, 8, 8, 12, 27, 14)` vs
  `…, 16`) while every member's decompressed content was byte-identical. That also
  made `test_omex.py::test_net_to_omex_provenance` a coin flip on run duration — it
  failed on the `macos-14` leg of #223, a `conftest.py`-only change that cannot
  reach this code, while `ubuntu-latest` passed the same commit.

  `write_omex` now takes the same `created` argument and stamps it on every entry,
  and `net_to_omex` passes its own through — including under `provenance=False`,
  where the entry headers are the only place it can land. The stamp uses the
  timestamp's calendar fields verbatim: a zip entry header has no timezone field,
  so a UTC offset is dropped rather than applied, and `metadata.rdf` stays the
  carrier of the authoritative instant. `created=None` keeps the wall clock, so
  the default is unchanged — and still not reproducible, which the docstrings now
  say outright instead of implying the opposite. A `created` that is not ISO-8601,
  or that predates the 1980 DOS epoch a zip header cannot encode, is refused by
  name before any file is written; `net_to_omex` refuses it before running the
  conversion and the gate rather than after. `bngsim-omex pack` gained the
  matching `--created TIMESTAMP`: the knob was unreachable from the CLI, which is
  the interface a reproducible build script actually calls.

  Handing `writestr` an explicit `ZipInfo` also drops the two defaults it would
  otherwise synthesize — a bare `ZipInfo` is `ZIP_STORED` with null permission
  bits — so both are restored, and a test asserts the stamped and unstamped
  archives agree on `compress_type`, `external_attr` and `compress_size`: the only
  thing stamping changes is the timestamp. The reproducibility tests drive a fake
  clock that jumps a day per reading, so they fail on the old behavior every run
  rather than only when two builds straddle a second boundary.

- **A `.net` without BNG2.pl's kind-annotation comments returned an identically
  zero sensitivity column, with no warning (issue #181).** Two `.net` files
  differing *only* in the trailing `# Constant` / `# ConstantExpression` comments
  on the parameter lines loaded to models that were identical through every
  accessor and integrated to the same trajectory, but one reported `dX/dp = 0`
  everywhere and the other the correct answer. The codegen `.net` parser had been
  reading those comments as the *definition* of which parameters are derived, so
  without them `a  p*c1` was taken for a leaf, contributed no `∂a/∂p` to the
  sensitivity RHS, and left `∂f/∂p` structurally empty. Nothing else was
  affected — `primary_param_names` classified `a` as derived the whole time, and
  the `set_param` chain rule, which does not go through that source, moved `a`
  correctly — which is why the model looked right from Python while the gradient
  did not. Any hand-written `.net`, or one from a tool other than BNG2.pl, was
  exposed; a zero column reads to a gradient fit as "this parameter does not
  matter".

  A parameter is now derived exactly when its value expression **references
  another declared parameter**, which is the condition that makes the chain rule
  necessary in the first place. Over the 1,817 `.net` files in this tree (41,433
  annotated parameter lines) that rule reproduces BNG2.pl's own annotation on
  every line BNG2.pl emits, and an A/B of the generated C over all 1,817 moves
  exactly one file: the un-annotated reproduction from the issue. The narrower
  reading — decide the kind by whether the value text parses as a float — was
  measured and rejected: it reclassifies the 628 lines of the
  `pi = 2*asin(1)` / `Temp = 37+273.15` shape that BNG2.pl (correctly) calls
  `# Constant`, and rewrites the sensitivity RHS of 54 models that were never
  broken.

- **A forward-sensitivity run never returned after crossing a rate-law switch
  that turned out to be continuous, and whether it did depended on `n_points`
  (issue #187).** Issue #150 established that the integration must resume just
  *past* a located crossing rather than on it — resuming on the surface puts the
  discontinuity inside the first step after the restart, and the root fires again
  — but it wrote that restart under the saltation jump. A switch measured
  CONTINUOUS at its own threshold (the clamp idiom, the most common `piecewise`
  in the corpus) returned before reaching it and left the state exactly where the
  root finder put it; `cvRootfind` short-circuits on an exact zero, so that is
  routinely `g(x) == 0.0` bit-for-bit. On `Smith_BMCSystBiol2013`
  (`PI345P3 > pip3_basal`) the run then restarted at `h ≈ ε·|t_end| ≈ 3e-15`, took
  a step far too short to move a 1.2e13-scale species by one ulp, rooted on the
  same crossing and was re-initialized back to the same `h` — 19,297 times in
  10 s, advancing simulated time by ~3.5e-15 an iteration, while the scalar run
  of the same model took 0.02 s. `n_points` only decided whether a run reached
  that state (2, 3, 4 and 8 hung; 5, 16 and 50 did not), which is why it looked
  like output points changed solvability. The restart is now a property of having
  stopped at a crossing, not of having jumped at one. The Smith gradient the
  issue was found on now completes at every `n_points` in 0.87 s and agrees with
  a three-step central difference over 199 resolved entries to 5.5e-5. Corpus
  A/B: 145/145 byte-identical with sensitivities off (state-switch roots are
  registered only under sensitivities), 145 identical + 22 identical refusals +
  13 moved of 181 condition-carrying SBML models, and every mover moves by at
  most what a 1% change in `rtol` alone moves it on the same binary.

- **A sensitivity `Simulator` reused the plain `.so` an earlier `Simulator` had
  left on the same model, and silently lost `bngsim_codegen_output_sens`.** The
  constructor's artifact-reuse block took whatever `model._codegen_so_path` /
  `_codegen_c_source` held, and `_auto_codegen_for_sensitivity` no-ops the moment
  anything is attached — so `Simulator(m)` followed by
  `Simulator(m, sensitivity_params=[...])` handed the second one a `.so` built
  when `_want_output_sens` was False, which carries neither the GH #198
  expression evaluator nor #177's `bngsim_codegen_sens_term_scale`. No exception,
  no warning: `d(func)/dθ` just took the finite-difference fallback. Measured on
  `BIOMD0000000012` before the fix. Reuse is now conditional on the artifact
  having been built for a sensitivity run — `_want_output_sens` is the record of
  that, and `Model.copy()` already carried the two together. The converse stays
  allowed, since a sensitivity artifact is a superset. Found while gating the
  sensitivity RHS (issue #209), which would have added `bngsim_codegen_sens_rhs`
  to the list of symbols this quietly dropped.

- **The `∂func/∂θ` analysis memo did not follow a derived-parameter override, so
  a reused model kept emitting the pre-write partials.** `_analyze_output_sens`
  keyed its memo on four model counters and the derivation budget, on the stated
  grounds that `set_param` only writes values — which issue #188 falsified for a
  *derived* parameter: overriding one detaches it, the issue #15 chain rule
  through it disappears from `∂func/∂θ`, and the emitted C changes with every
  counter unmoved. Content addressing hid this (the `.so` was keyed on the stale
  source it was compiled from, so it matched itself, and a fresh process derived
  the right thing); under issue #174's structural key the stale source would be
  cached under the *post*-write key and served to every later process. The
  attachment vector is now part of the memo key.

- **A `.net` `.so` compiled without source chunking was served to a chunked run.**
  `prepare_codegen`'s key carries every process-scoped hatch that changes the
  emitted source — `BNGSIM_NO_CODEGEN_JAC`, the GH #67 functional-sensitivity
  hatch, the GH #90 budget — except `BNGSIM_CODEGEN_CHUNK`, which changes what
  `generate_rhs_c` emits (4,974 → 5,385 chars on `akt-signaling`). So an A/B of
  the chunking feature measured the same binary twice. The key gains a
  `:chunk=<threshold>x<size>` suffix, appended only when the policy is
  overridden, so the default key form is unchanged.

- **An ExprTk derivative over a Python-keyword-named parameter silently dropped
  the whole model to the FD Jacobian.** `_exprtk_to_sympy` aliases a parameter
  named `def` / `lambda` / `is` to `_BNG_KW_def` so `parse_expr` accepts it, and
  the C emitter (`sympy_to_c`) resolves every symbol through a callback keyed by
  alias — but the ExprTk emitter prints names straight through, so the derivative
  came back reading `_BNG_KW_def`, ExprTk rejected it as an undefined symbol, and
  `set_functional_jacobian` failed for the entire model. Visible only under
  `BNGSIM_JAC_DEBUG`. Five corpus models regain their analytical Jacobian.

- **A synthesized `_rateLaw_<rid>` could collide with an SBML parameter of the
  same name**, which `ModelBuilder::validate` rejected outright — reached by
  re-importing bngsim's own `.net` → SBML export, which writes `_rateLaw_<rid>`
  out as a real parameter. Synthesized names are now uniquified.

- **A sensitivity absolute tolerance set below the roundoff of the arithmetic
  that produces it, so CVODES micro-stepped forever (issue #177).** For column
  `iS` the variational equation is `ṡ = J·s + ∂f/∂p`, so row `i`'s derivative is
  *assembled* by summing terms — and on a model whose species span many orders
  those terms cancel. `∂f/∂p = 1e18 − 1e18` reads as ~0 while carrying ~ε·2e18
  of roundoff, and the reported value says nothing about the size of what
  cancelled. A row whose own `|s_i|` has decayed to zero contributes nothing to
  `rtol·|s_i|` either, leaving `atolS_i` as the only thing holding the error
  weight finite. Set below that roundoff, the error test cannot pass at any step
  size: `h` collapses and the run never returns. The state solve has no
  equivalent — a species at zero has a zero RHS, not a difference of huge
  fluxes — which is why this was a sensitivity-only defect.

  `atolS` is now floored, per (row × column), at the roundoff two independent
  measurements say the arithmetic actually carries, refreshed from the live state
  while the run holds control:

  * **The assembly floor**, `ε·τ·Σ|term|`. The emitter gained a companion to
    `bngsim_dfdp` — `bngsim_dfdp_term_scale`, reported through the exported
    `bngsim_codegen_sens_term_scale` — that accumulates the *magnitudes* of the
    very contributions the signed switch sums into each row, from the same
    traversal, so the two cannot come to describe different reaction sets. The
    `J·s` half is `Σ_j|J_ij||s_j|` from the analytical Jacobian. `τ` is a
    thousandth of the integration horizon: `Σ|term|` has the units of `ṡ` and a
    tolerance has the units of `s`, and `τ` is exactly the smallest step that RHS
    noise alone may force. A genuine accuracy requirement still shrinks `h` as
    far as it likes.
  * **The column's representation floor**, `ε‖s‖∞`. Each BDF step solves for the
    whole column at once, so every entry is assembled from quantities of size
    `‖s‖∞` and inherits ~`ε‖s‖∞` of absolute error whatever its own size.

  The two are complementary, not redundant, and the measurements say so: on the
  minimal reproduction (`tests/data/sens_scale_cancellation.net`, whose `∂f/∂p`
  is a difference of two 2e18-scale terms) the assembly floor alone takes the run
  from **183,219 steps to 546**, and the representation floor alone changes
  nothing; on `Smith_BMCSystBiol2013` at the reported dose it is the other way
  round, and 14 of the 16 sensitivity columns go from *not finishing* to 0.135 s.

  **This does not close #177.** Run all 16 columns together and the model still
  does not finish: `k7`/`kminus7a` fail with `CV_ERR_FAILURE` at `t=23.2961`
  (`h=8.4e-13`) and `k8`/`kminus8` still stall between `t=2.4` and `t=24`.
  Raising this floor by 1000x changes neither, so whatever remains is not the
  roundoff this addresses. It is also not a missing saltation jump: the model's
  one state-dependent rate-law switch (`PI345P3 > pip3_basal`) is detected and
  rooted, but `PI345P3` sits at ~1e13 against a threshold of 200 and never
  crosses it.

  Everything about this is a `max()` against the tolerances that shipped, and
  `Σ|term|` is ~`ε·(a few)` for a well-scaled model, so a model whose arithmetic
  is clean keeps exactly the tolerances — and the step sequence — it had. Over a
  244-model sweep of the rr_parity corpus, 232 came back bit-identical and none
  moved; over the 61 models whose initial state is large enough for the floor to
  bind, 13 moved, and each was refereed against a finite-difference oracle rather
  than a digest: no accuracy regression on any of them, one improvement, and the
  state trajectory inside `rtol` everywhere (max 5.2e-8). `x(t)` moving at all is
  inherent — `atolS` enters the error test, the error test picks `h` — but the
  floor never reaches the state tolerances.

  `BNGSIM_SENS_ERROR_FLOOR=0` restores the pre-#177 tolerances from the same
  binary and the same `.so`, which is what makes that sweep a one-variable
  experiment; `BNGSIM_SENS_FLOOR_PARTS` selects the two floors independently.

  The term-scale switch is `O(parameters × reactions)`, so it is emitted only for
  a sensitivity run — the same `_want_output_sens` signal the GH #198
  output-sensitivity block uses, carried into the `.net` cache key as
  `:sens_term_scale` so a `.so` built for a plain run is never reused for one that
  needs the symbol. Without that gate `BIOMD0000000496`'s `.so` went from 18 MB to
  29 MB and its cold build from 33.9 s to 39.6 s, for a symbol it can never call.

- **A LAPACK-dense skip that no `_DECLARED_SKIPS` entry matched (found while
  wiring issue #169).** Two files skip for the same build-variant condition and
  phrase it differently: `test_engine_choice_accessors.py` says `"LAPACK-dense
  not built in this configuration"`, `test_lapack_dense_solver.py` says `"build
  links no BLAS dense backend (Accelerate / LAPACK)"`. Only the first was
  declared, so the second read as an *undeclared* skip — the audit's signal for
  "a test stopped running and nobody decided it should".

  It could not be seen on macOS: `find_package` always resolves Accelerate there,
  so neither test skips at all. It shows up only where CMake finds no BLAS, and
  until #169 no CI job ran the full suite anywhere but a developer's macOS box.
  The second phrasing is declared now, so `BNGSIM_SKIP_AUDIT=strict` does not
  fail a Linux leg for a legitimate build-variant skip.

- **The #161 analytic sensitivity RHS was a net regression on the model it
  targeted, and the cost was in the build, not in the emitted code (issue
  #165).** `Smith_BMCSystBiol2013` (133 species, 16 sensitivity columns,
  `t_span=(0, 200)`) went 1.23 s → 2.02 s across #161, +64% — the +56% the issue
  reports. Splitting it by phase says the integration is not where it went: the
  run itself got **faster** (0.246 s → 0.180 s, 705 steps → 562, 146 Jacobian
  evaluations → 49). The whole regression, and more, is `Simulator(...)`
  construction: 0.99 s → 1.84 s.

  Construction pays for derivation because the compiled-`.so` cache key is a hash
  of the generated source, so **every** construction generates that source —
  every symbolic derivative in it included — even on a cache hit. #161 removed an
  early decline, so a cross-compartment model now reaches the derived-parameter
  chain rule the decline used to skip.

  That chain rule asked "which parameters does this expression name?" one
  parameter at a time, with a freshly interpolated
  `re.search(rf"\b{name}\b", expr)` per candidate. A real model's parameter count
  outruns `re`'s internal 512-entry pattern cache, so each of those searches
  *recompiled* its pattern — 82,058 compilations per pass on Smith's 922
  parameters × 89 derived expressions, ~0.9 s of the 1.84 s. It is one tokenizing
  pass over the expression now: a name made only of word characters matches
  `\bname\b` exactly when it is one of the expression's maximal `\w+` runs, so
  one pass answers the question for every such name at once. Names carrying a
  non-word character (no SBML or BNGL identifier does, but nothing guarantees it)
  keep the per-name search.

  Smith, same measurement: construction 1.84 s → 0.67 s and total 2.02 s →
  0.84 s, which is **32% below the pre-#161 baseline** instead of 64% above it.
  Its difference-quotient path gets the same saving (1.23 s → 0.65 s), because
  the scan is also reached from the #198 output-sensitivity analysis — so this is
  not specific to #161's decline, only to the models that reach either one.

  Which models those are is worth stating rather than implying, because 512 is a
  cliff and not a slope. Below it the per-name searches all hit `re`'s cache and
  cost ~12x the token pass but little in absolute terms; above it every search
  recompiles. Measured over 89 expressions: 500 names 0.052 s, 512 names 0.052 s,
  **520 names 0.318 s**, 922 names 0.563 s — against 0.004–0.007 s for the token
  pass throughout. So the win needs *both* >512 differentiation names and enough
  derived-parameter expressions to multiply them, which is Smith's shape (89
  loader-synthesized `_rateLaw_*` × 922) and no corpus model's: only 24 of 213
  `benchmarks/sbml_events` models carry a derived parameter at all, the largest
  product among them is 7,880 against Smith's 82,058, and the twelve
  highest-parameter models in that corpus measure 1.0x. Inert there, in other
  words — and provably so: the emitted C is byte-identical over 1,381 models
  (`benchmarks/sbml_events` 214, `suites/rr_bngl/sbml` 581,
  `suites/ode_fullnet/nets` 585, plus Smith), with no model gaining or losing an
  analytic sensitivity RHS.

  The issue's own hypothesis, that the per-species volume divide lands inside the
  inner scatter, was measured rather than assumed and is not the case. For a
  static compartment the divide is already folded into the coefficient at emit
  (Smith's whole sensitivity RHS contains zero runtime divides); only a
  *variable*-volume compartment emits one, which it must. Dropping the 191
  `+= (0) * contrib` dead lines from `bngsim_jac_vec` — 14% of its scatter — was
  measured at 0.1685 s → 0.1694 s, i.e. nothing, so they stay. And the issue's
  "same-volume analytic 0.026 s vs cross-compartment analytic 0.775 s on an
  otherwise identical model" is not two analytic paths: changing `size(C2)`
  changes the loader's *classification*, so the uniform model's reactions are
  Elementary (closed-form ∂f/∂p, no sympy at all) while the cross-compartment
  model's are Functional.

## [0.12.2] - 2026-08-03

### Added
- **`Model.effective_ic_sensitivity()` — the `∂x(0)/∂θ` the `parameter` axis
  already carries (issue #155).** `output_sensitivities(axis="parameter")` is a
  *total* derivative: it carries the right-hand-side path **and** the
  initial-condition seeding. `axis="ic"` is the companion basis `∂y(t)/∂x_k(0)`
  with the initial value held independent. The two are therefore **not**
  orthogonal —

      d_param[θ] = (right-hand-side path) + Σ_k (∂x_k(0)/∂θ)·d_ic[x_k]

  — so a consumer that routes a fitted parameter to every native column it
  reaches and sums them double-counts any seeded initial condition. That was
  always the contract; nothing about the numbers changes here. What changes is
  that it is now *stated*, and answerable in code rather than by measurement.

  The reader is paired with the existing writer `declare_ic_sensitivity`, and
  answers from **model structure alone** — parameter graph, live initial
  conditions, declarations — with no integration, so a fitting frontend builds
  its gradient routing once at setup instead of burning a throwaway run or
  re-deriving per evaluation. It reports the *effective* matrix: after the issue
  #113 retirement of a species moved off its declared initial condition, and
  after the issue #111 declaration overlay. Keys are the ids
  `sensitivity_params=` accepts — a compound `<initialAssignment>` lowered by
  issue #147 reports the **original** symbols, never the synthetic
  `_ic_<species>` carrier.

  **Present-and-zero is not absent.** A present entry valued `0.0` means
  "seeded, coefficient zero at this state" — `∂(a*R0)/∂R0` is `a`, which
  vanishes at `a = 0` without the seeding path ceasing to exist — while an
  absent entry means there is no seeding path at all and the caller's own
  initial-condition term is the missing piece. Only the second is a signal to
  add an `ic`-axis term. Making that distinction visible required separating
  *structural reachability* from *numeric value* in the seed derivation: the
  shared derived-parameter DAG walk drops numeric zeros (it is also the
  switch-time scan's "not a parameter threshold" signal), so
  `compute_ic_param_sens_seed` now recovers presence from the token closure and
  emits a zero-valued row. The solver still receives no zero rows — they seed
  nothing — so no gradient changes.

  `Result.ic_sensitivity_seed` carries the same matrix as the per-run record,
  for the cases the model-level reader structurally cannot answer: a batch or
  scan over a nonlinear derived initial condition (`Rtot = R0*scale`) has
  point-dependent coefficients, so each row stamps its own, `squeeze` keeps the
  matrix only if every row agreed, and `compute_all_sensitivities` takes the
  union over parameter chunks. It reports `None` — *not recorded*, distinct from
  `{}` — for a `carry_sensitivities=True` phase, where the engine seeds from the
  prior phase's `dx/dθ` and discards these rows entirely (#210/#81); reporting
  the computed-but-unused rows there would have been the same silent-wrong-answer
  class this whole property exists to end.

  Both readers and the solver seeding share one derivation
  (`Model._ic_sensitivity_triples`), so a reported matrix cannot drift from the
  seed it describes.

  Also: a `capabilities()["features"]["effective_ic_sensitivity"]` flag to gate
  on instead of a version string (a build without the reader cannot say what the
  seed carries, and every answer a consumer could guess is silently wrong, so
  refusing a gradient fit is the honest behaviour); the contract written into the
  `output_sensitivities` docstring, `docs/reference/api.md` and the PyBNF
  integration guide; and a note that `SteadyStateResult` raises no such question
  (its `parameter` axis is the implicit-function derivative `J·∂x*/∂p = −∂f/∂p`,
  with no seeding term, and `axis="ic"` raises).

### Fixed
- **Forward sensitivity refused two state-switch crossings at one instant, but
  on the corpus they are always one crossing written twice (issue #153).** Issue
  #150 registers one CVODE root per switch condition and deduplicates them by
  the residual's *text*, which merges `X<1` with `X<=1` and nothing else. Two
  spellings of ONE crossing therefore arrive as two roots that fire together,
  and the batch was refused outright — each jump reads `f` on the two branches
  of its own condition, and one step across a shared crossing cannot separate
  them. The reasoning is right for genuinely independent switches; it is just
  not what reaches it. Both models that hit the refusal write one crossing
  twice: `sp_fourier_synthesizer` roots on `ds1` and on `3·ds1 − 12·s1²·ds1`
  (five residuals in all, every one a multiple of `Cos1 − amp_offset`, all
  crossing at `t = π/2`), and `ml_hopfield` on `dS1/dt` and `dS3/dt`, which are
  identically equal along the trajectory because its own weight matrix leaves
  `S1 ≡ S3` invariant. One is visible in the text and one is not, which is why
  the decision belongs at the crossing rather than in the dedup.

  The batch is now carried into the jump together and the two halves of the
  saltation term are checked separately, at the crossing:

  * `f⁻ − f⁺` has to carry **every** branch change, which the one flow probe
    does exactly when it crosses every residual in the batch — so the `δt`
    ladder now grows until they all flip *together*, and a batch that never
    does is several conditions meeting by coincidence rather than one crossing.
  * `dt*/dθ` has to be **one** vector, and flipping together does *not*
    establish that. A common factor does: `h` cancels out of the
    implicit-function ratio, scaling numerator and denominator alike where the
    residual vanishes. An equality that holds only along the *trajectory* does
    not — `ml_hopfield`'s two residuals have non-parallel gradients (cosine
    −0.30 at the crossing) and their `dt*/dθ` are permutations of each other
    under the model's `W12 ↔ W23` symmetry, so a perturbation that breaks it
    splits the crossing in two. The vector is therefore formed from each
    residual in turn and compared, and a batch that disagrees is refused with
    the numbers.

  The second is the criterion, and it is deliberately weaker than "one
  surface": what the jump needs is a single `t*(θ)` to shift the flow along,
  which two *independent* crossings also have when the requested columns move
  them together. A fixture of two unrelated species decaying through a shared
  threshold merges exactly (its `∂W/∂c` is the closed-form −8 to eight digits)
  for the threshold column and is refused for the column that splits the pair.

  Order matters and is the cheap way round: the branch gap is measured before
  any `dt*/dθ`, and both corpus models are the BNGL signed-rate idiom
  (`if(r>0, r, 0)` against `if(r<0, −r, 0)`, continuous where `r = 0`), so they
  return with no jump at all — and never reach the second check. Their columns
  come from the in-branch sensitivity RHS and match a finite difference of the
  model's own trajectory.

  Corpus A/B on the `netcond` arm — the 80 of 585 `.net` models whose text
  carries an `if()`, under forward sensitivities, against the issue #150 binary:
  **76 identical, 2 newly answered, 1 wall-clock capped, 1 identical refusal**.
  Nothing moved: a single-switch crossing takes the same path it did, and the
  two newly answered rows are exactly `ml_hopfield` and `sp_fourier_synthesizer`.
  The remaining refusal is `ml_q_learning`, whose residual is not
  differentiable across its own surface (issue #154).

- **Forward sensitivity through a state-dependent switch in a rate law missed
  the saltation jump at its crossing (issue #150).** A condition that reads the
  state — `piecewise(0, Virus < 1, Virus*rho_V)` — flips a branch of `f` at a
  crossing whose time `t*(θ)` moves with **every** parameter through the
  trajectory. The in-branch derivative was never the problem: `sympy.diff` of
  the `Piecewise` is right on both sides and carries no boundary delta. What is
  discontinuous is `∂x/∂θ` itself, by the saltation term

      s(t*⁺) = s(t*⁻) + (f⁻ − f⁺)·dt*/dθ

  and *neither* way bngsim could produce a sensitivity RHS carried it — the
  analytic one differentiates each branch where it is live, CVODES' internal
  difference quotient integrates the variational equation straight across. GH
  #68 declined the analytic path here and issue #146 corrected the warning to
  say the fallback was wrong too; this is the fix.

  Three pieces, all of them the rate-law twin of something issue #144 already
  built for a state-dependent event *trigger*: split the condition into a
  residual `lhs − rhs` (`NetworkModel::state_switch`, cached per condition and
  re-derived by `clone()` — an expression id means something else in another
  evaluator); register that residual as a CVODE root, so the crossing is
  **located** rather than chased; and at the root, read `f` on both branches at
  `x(t*)`, form `dt*/dθ` by the implicit function theorem, and jump. The
  ∂t*/∂θ solve is now one function (`Impl::residual_dtstar`) reached from both
  the event trigger and the rate law, transversality floor included.

  The branch selection is the one genuinely new mechanic. A clock switch picks
  its branch by nudging the clock across its threshold; a state switch has no
  such variable, so the whole state is nudged **along the flow**, `x^± = x(t*) ±
  δt·f`, which for a unit-rate counter reduces to the clock nudge exactly. `δt`
  is measured, not assumed: CVODE locates a root only to ~`100·ε·(|t|+|h|)`, so
  the size starts just past that and grows until the residual actually changes
  sign across the pair. Failing to flip inside the ladder is refused rather than
  answered with a jump of zero — "we could not tell the branches apart" and "the
  branches agree" are otherwise indistinguishable. The verified `x + δt·f` is
  also what the run restarts from, so the discontinuity cannot land inside the
  first step after the restart: leaving it on the *before* side collapsed `h` to
  ~1e-17 and re-fired the root, applying the jump twice (issue #82's pit,
  reached from the rate-law side).

  On the issue's own reproduction — `dX/dt = if(X<1,0,rho)·X − delta·X`,
  crossing at `t* = ln(1000)/0.8`, saltation factor `f⁺/f⁻ = 2` — the `rho`
  column read a factor of exactly **2** low after the crossing and `delta`
  between 1.72 and 1.96 (it mixes the missing jump with a correct in-branch
  part, which is why "multiply the tail by `f⁺/f⁻`" was never a fix). Both now
  match a central finite difference of the model's own trajectory to 5 digits.
  On AMICI's `nested_events` fixture all four columns agree with their own
  finite difference across the whole run.

  **The GH #68 decline is lifted for exactly the conditions this compensates**,
  and that is required, not opportunistic: once the crossing is resolved to a
  root, the difference-quotient fallback becomes *worse*. Its probe evaluates
  `f` at `y + σ·s` with `σ ≈ √rtol`, and just past a crossing `σ·|s|` is easily
  wide enough to put the probe back on the other branch — on the reproduction
  that injected `rho·X/σ ≈ 2.7e4` into `ds/dt` for the sliver of time the state
  stayed within `σ·|s|` of the surface, and the column came out 28% high with
  the jump correctly applied. So a condition whose crossing is compensated now
  reaches the analytic RHS. What still declines — and still carries
  `UncompensatedCrossingReason`, because the difference quotient is still wrong
  for it — is the crossing no machinery can bracket: an equality (measure-zero
  on a continuous trajectory), a `not()` call, a comparison with no `if()` head,
  and a clock threshold that neither reduces to a constant nor reads state.

  Three recognizers now have to agree without overlapping, since a crossing
  claimed by *both* the issue #48 clock path and this one would have its jump
  applied twice — a BNGL counter clock is a species, so `t >= sigma` reads live
  state and the residual splitter would happily take it.
  `clock_crossing_compensated` is asked first everywhere, and the partition is
  asserted behaviourally over nine rate laws rather than by shared plumbing
  (`TestTheGateAndTheDetectorsAgree`), which is the property #56 lost.

  **A `piecewise` that is CONTINUOUS at its own switch gets no jump and no
  refusal**, which the corpus insisted on. The saltation term is
  `(f⁻ − f⁺)·dt*/dθ`, so it is identically zero when the branches meet — and
  that is the most common `piecewise` there is, because it is how a clamp is
  written. BIOMD0000000161's basal PIP synthesis is
  `piecewise(0.581·k·(exp((basal − PIP)/basal) − 1), PIP < basal, 0)`, whose live
  branch is exactly 0 where `PIP = basal`; the trajectory then *rides* that
  surface, so the flow probe never leaves one branch and the first cut of this
  feature refused a model that had always run. The branch gap is now measured
  before `dt*/dθ` is ever formed, on both paths — from the flow probe where it
  flips, and from a coordinate probe of the residual's own support where it does
  not (the gradient survives a tangential flow; orientation does not matter when
  only `|f_a − f_b|` is being read). What is left refused is the genuine case: a
  tangential crossing with a real discontinuity at it.

  **The detector reads the same text the gate does.** A condition can only
  *become* a state condition under inlining, and the gate judges the inlined rate
  law. BIOMD0000000837 writes `Lymphocyte_Term` as
  `piecewise(…, 1 − Total_Lymphocytes/K > 0, 0)` where `Total_Lymphocytes` is an
  assignment-rule parameter — a *parameter* address that reads no live state on
  its own. The gate sees it after substitution as
  `1 − (B+C_e+C_m+H_e+H_m+L)/K > 0` and admits; a detector scanning raw function
  bodies registered nothing, so the decline was lifted with no crossing behind
  it. That is the #68 silent zero reintroduced from the other side, it hit 3 of
  the 182 condition-carrying corpus models, and no test saw it — the corpus A/B
  found it as three rows in a "numbers moved, nothing registered" bucket.

  State-switch roots are registered **only for a run that asks for
  sensitivities**. That is the whole blast radius: every plain trajectory keeps
  exactly the stepping and exactly the numbers it had. Registering them
  unconditionally would help the integrator on its own — the issue makes that
  case — but it is a trajectory-accuracy change with its own corpus risk and
  wants its own measurement.

  Corpus A/B, three arms. **`sbmlcond`** — the 182 condition-carrying
  `rr_parity` models under forward sensitivities, the only arm that can execute
  the diff: 64 register at least one state switch, and of the 182, **144 are
  byte-identical, 28 moved, 8 refuse identically, 2 are wall-clock capped and 0
  newly refuse**. Every one of the 28 movers carries a registered switch —
  nothing moved that the change cannot reach. **`sbmlplain`** — the same 182 with
  sensitivities OFF, the empirical half of the blast-radius claim: **179
  identical, 2 identical refusals, 1 moved**, and that one (BIOMD0000000497,
  295 species) differs by 2.8e-16 of its peak with an identical `n_steps` and
  `n_rhs_evals`, i.e. a recompilation artifact rather than a behavioural change.
  **`netcond`** — the 80 of 585 `.net` models whose text carries an `if()`:
  **51 identical, 25 moved, 1 wall-clock capped, 3 newly refused**; the other
  505 `.net` models are not an arm because they cannot be one, the detector
  short-circuiting on "no `if()` in any function body" before it probes the
  model.

  The 3 new refusals are all in the rulehub "BNGL as a general-purpose
  computation" examples (a Hopfield net, a Q-learning agent, a Fourier
  synthesiser), whose trajectories ride their switching surfaces continuously:
  two hit the refusal for **two switches crossing at the same instant** (each
  jump reads f on the branches of its own condition, which one shared crossing
  cannot separate) and one a genuine tangency with a real discontinuity at it.
  All three previously returned numbers that issue #146's warning already
  described as wrong. Two of the three — the Hopfield net and the Fourier
  synthesiser — are lifted again by issue #153 above, which merges a batch that
  resolves to one crossing time instead of refusing it; only the Q-learning
  agent still refuses in this release.

- **A CVODE root that fired nothing left the forward sensitivities behind
  (issue #146).** Every `CV_ROOT_RETURN` reinitialises the state stepper —
  `CVodeReInit` at the root time — so the integrator restarts *at* the
  discontinuity instead of stepping over it. `CVodeReInit` rewinds only the
  state: CVODES' sensitivity history stays at the end of the internal step the
  root interrupted. Both existing jump paths (the GH #212 event jump and the
  issue #48 switch jump) already read `s⁻` before the re-init and resumed with
  `CVodeSensReInit`; a root that fires **no** event did neither, so the run
  resumed with `s(t_n)` attached to `x(t_ret)` and every column picked up a
  spurious `s'·(t_n − t_ret)` step, at every such root.

  Two ways to reach one, and neither is exotic: a GH #72 discontinuity root
  (registered for a `piecewise` threshold, and for every event trigger's
  relational subconditions — including those of an event this loader then skips
  for having no assignments), and an event root whose trigger crossed without
  rising.

  Found on AMICI's `events` fixture, whose two triggers carry no
  `eventAssignment` at all, so `n_events == 0` while their roots still fire.
  Against the model's own central finite difference bngsim read **5.2e-2 on
  every one of the four parameter columns**, where AMICI read 3.4e-5 against
  its own; the impulses land exactly where `x2` crosses `x3` (t ≈ 0.15), where
  `x1` crosses `x3` (t ≈ 1.33), and where `x1` falls back through `x3`
  (t ≈ 4.6). After the fix the same four columns read 1.7e-6, 4.5e-6, 9.7e-6
  and 1.0e-3 — the last being finite-difference smearing at the `time > p4`
  kink, where bngsim's analytic column matches AMICI's to seven digits — and
  cross-engine agreement on the whole tensor goes from 7.2e-2 to 5.7e-4.

  The trajectory was never affected, which is why no parity suite could have
  caught it: only the sensitivity vectors were left behind.

  Corpus A/B, event-carrying `rr_parity` subset under forward sensitivities (194
  models): **151 byte-identical, 32 refused identically, 3 wall-clock-capped, 8
  moved.** Every one of the 8 carries at least one event — so a `CV_ROOT_RETURN`
  is reachable — and 5 of them also carry GH #72 discontinuity roots. The
  `.net` corpus is not an arm here because it cannot be one: all 585 models load
  with `n_events == 0` **and** `n_discontinuity_triggers == 0`, so `n_roots == 0`
  and the changed code is unreachable on every one of them. What the corpus arm
  establishes is *which* rows move and that nothing else does; that the movement
  is a correction is established by the fixtures above and by the closed forms in
  the tests, not by it.

- **The GH #68 decline told you the difference-quotient fallback was correct
  when it is not (issue #146).** Declining the analytic sensitivity RHS is a
  fallback, not an error, and the warning has always ended "CVODES' internal
  difference quotient is used instead (correct, but slower)". That is true for
  an underivable rate law (#56/#66) or an exhausted derivation budget (#90) —
  the problem is smooth and CVODES answers the same question more slowly. It is
  **false** for the one class the GH #68 gate exists to catch: an *uncompensated
  crossing*. What the gate reports is a branch whose crossing time nobody
  compensates, and the difference quotient integrates the variational equation
  straight through that crossing, dropping the very jump the analytic path was
  declined for.

  On AMICI's `nested_events` — `piecewise(0, Virus < 1, Virus*rho_V)`, crossing
  at t = 10.6347 — the true `∂x/∂θ` jumps by the saltation factor
  `f⁺/f⁻ = −1.6/−0.8 = 2`, and every parameter column comes back **a factor of
  exactly two low** after the crossing, against both AMICI and bngsim's own
  central finite difference. Running the gate over the 1324-model `rr_parity`
  corpus: 107 models decline (42 on a state threshold, 70 on a fitted non-clock
  threshold, 1 on an unreduced clock threshold), and of the 67 that could be
  scored against their own difference quotient, **28 disagree by more than 5%**.
  The other 39 agree — a crossing that never happens inside the run window, or
  one whose branches meet with equal slope, leaves the tensor right, which is
  why this is a corrected warning rather than a refusal.

  The reason now carries its own class (`UncompensatedCrossingReason`) so the
  warning cannot promise correctness for it, and every site that adds context to
  a reason re-tags through one function — an f-string over a `str` subclass is a
  plain `str`, and that wrap is exactly where the distinction would have gone
  missing. **Issue #150** tracks the fix: the saltation jump, which is the
  rate-law twin of the state-dependent event trigger #144 already differentiates.

- **An `<initialAssignment>` on a rate-rule or event-assigned *parameter*
  carried no sensitivity seed, and `set_param` could not move that initial
  condition (issue #146).** An SBML `<species>` is not the only thing this
  loader turns into an integrator state: a parameter or compartment carrying a
  rate rule, or written by an event assignment, is promoted to a species by §8
  and §10. That is how Antimony's pure-ODE spelling arrives — `Virus' = …`
  emits `<parameter id="Virus" constant="false"/>` plus a `<rateRule>` — and it
  is what AMICI's `nested_events` fixture is written in. The initialAssignment
  scan keyed on the SBML species list alone, so `Virus = V_0` registered
  nothing.

  The sensitivity consequence is a silent zero of the same shape as the bare-
  `<ci>` gap above: on `nested_events` the entire `V_0` column read **exactly
  0.0** where AMICI read 1.0 and AMICI's own difference quotient confirmed
  0.9999999. The *trajectory* consequence is worse and was invisible for the
  same reason it was on the species path — the IC link is also what
  `NetworkModel::set_param` re-resolves (issue #79), so `set_param("V_0", …)`
  ran the identical trajectory at every point of a scan, and a trajectory
  finite difference measured zero in exactly the way the missing seed did.

  Fixed by widening the scan to the symbols this loader promotes and
  registering the link at both promotion sites through one shared function, so
  the three sites cannot drift about which states get a seed.

  Corpus A/B, load-only over the whole 1324-model `rr_parity` SBML corpus (the
  registration is the entire reach of this change, and load-only measures it
  without paying a codegen compile per model): **6 models gain an IC parameter
  ref, 0 lose one, and the resolved initial state is byte-identical on all six.**
  That last clause is the one that matters — the failure this predicate is
  deliberately strict about is a lowering that writes a wrong value *over* a real
  initial condition (BIOMD0000000856 in the entry below), so the initial state
  rides along in the A/B record and is checked separately from the ref set. No
  model moved its `x0` with an unchanged ref set either. The six span every shape
  the fix reaches: four bare `<ci>` on rate-rule-promoted parameters
  (BIOMD0000000234, BIOMD0000000327, BIOMD0000000349, MODEL1108260014), one on a
  promoted **compartment** (BIOMD0000000429, `compartment_1 ← parameter_46`), and
  one **compound** expression lowered to a synthetic derived parameter
  (BIOMD0000000695, `doseBL = 90*skinType`).

- **An SBML `<initialAssignment>` over parameters carried no sensitivity seed
  unless it was a bare `<ci>`.** A species whose initial condition is an
  expression over model parameters has a `∂x_i(0)/∂θ` the forward-sensitivity
  seed must carry, exactly as for the `R() R0` parameter-named IC of issue #43.
  The SBML loader only ever registered the trivial single-symbol case, so
  `u(0) = b*v0` produced **no seed and no warning**: the `b` column came back
  short by the whole initial-condition term.

  Found by running AMICI's `neuron` fixture — the model issue #144 was filed on
  — against AMICI itself on the identical document. bngsim's `b` column peaked
  at 1602.8 against AMICI's 7898.4, a 5x error, while the other three columns
  already agreed to 7e-8.

  **Two things had hidden it, and the second is the one worth remembering.**
  Nothing refused: the column was finite, smooth and wrong. And a trajectory
  finite difference *agreed with it* — `set_param` did not re-resolve an
  initialAssignment either, so the oracle held `x(0)` fixed in exactly the way
  the seed did. Engine and oracle shared one defect, which is why the existing
  `neuron` regression test passes with or without this fix, and why the new
  tests are closed forms.

  The fix lowers a compound parameter-only initialAssignment to a synthetic
  derived parameter, because `compute_ic_param_sens_seed` (issue #43) already
  differentiates a derived IC to its primaries with the sympy chain rule —
  nothing new has to know how to differentiate, the expression just has to
  reach that code. Registering the link also makes `set_param` re-resolve the
  IC, so a scan over a parameter the IC reads now lands where a fresh load with
  that value lands (verified on BIOMD0000000999: `kdeg_R1 x1.5` moved
  `TGFb_R1_surface(0)` 17.474 → 18.349, where it had been pinned at 17.474).

  The predicate is deliberately strict: every referenced symbol must be a
  genuinely *constant* parameter — not an assignment- or rate-rule target, not
  an event-assignment target, not a bare `constant="false"` declaration, since
  this loader promotes all of those to species. BIOMD0000000856
  (`WHISBF = 0.66*NSt`, `NSt` non-constant) is why: lowering it produced a
  derived parameter evaluating to 0 against a symbol that is a species in the
  built model, and the build-time IC resolution wrote that 0 over the species'
  real initial condition. A wrong IC is far worse than a missing seed.

  Corpus (198 `rr_parity` models declaring `<listOfInitialAssignments>`): 34
  gain a seed. A plain fresh-load run is unchanged on 191 of 196; the 5 that
  move do so by **1 ulp** (1e-16 to 1e-14 relative on the IC), because the
  initial condition is now re-derived from the same expression the sensitivity
  differentiates, through the engine's own evaluator, instead of the loader's
  separate Python fold — one defining site instead of two. A `set_param` scan
  moves on 15, which is the correction itself.

- **bngsim vs AMICI on the `neuron` fixture, after both fixes:** identical SBML
  document, `t_end = 98.92`, `N_p = 4`, AMICI's default tolerances — the full
  sensitivity tensor agrees to **2.2e-6**, the trajectory to 9.9e-7, at 11.2 ms
  against AMICI's 14.3 ms. (At `rtol = 1e-10` AMICI fails to integrate past
  t ≈ 92.5 on its own fixture; bngsim completes the window.)

### Added
- **Forward sensitivity through a state-dependent event trigger (issue #144).**
  A trigger that reads the state — AMICI's `neuron` fixture fires on `v > 30` —
  has a crossing time that moves with every parameter *through the trajectory*
  while naming none of them, so `∂t*/∂p` is non-zero and neither existing path
  can supply it: fixed-time events have `∂t*/∂p = 0`, and issue #49's detector
  resolves a *threshold* to primary parameters, which cannot serve `v > 30`
  (the threshold is the constant 30; the whole dependence is in the trajectory).
  Since #52 these models were refused. They are now answered.

  The crossing is differentiated where it happens, by the implicit function
  theorem on the trigger's residual `g`:

      dt*/dθ = − (∂g/∂x·S(t*⁻) + ∂g/∂p) / (∂g/∂t + ∂g/∂x·f(x⁻))

  feeding the four-term jump the solver already applies. Two consequences worth
  naming: the **initial-condition** columns get a crossing shift too (issue
  #49's `∂t*/∂p` covers parameter columns only, because an IC cannot move a
  clock — it plainly can move a threshold crossing), and the denominator is the
  transversality condition, so a crossing rate that is a near-total
  cancellation of its own terms, or is at the right-hand side's own noise
  floor, is refused *as a tangential crossing* rather than divided by. That
  replaces the blanket "state-dependent trigger" refusal with a statement about
  the actual obstruction.

  The residual is the trigger's `lhs − rhs`: the compiled trigger is a boolean,
  which is what lets CVODE *locate* the crossing but not differentiate it. Only
  a single relational comparison qualifies, and what does not now says so by
  name — a conjunction, disjunction or negation (its true-set boundary is
  assembled from several surfaces, and which one carries the rising edge can
  move with a parameter) or an equality (measure-zero on a continuous
  trajectory). Execution delays are still unsupported.

  Two more refusals are newly *reachable*, both because a state-dependent
  trigger is what it takes to reach them. A same-instant cascade (SBML "events
  triggering events") executes after the located batch, reading a state that
  already jumped, while the jump differentiates at the pre-batch `x⁻` — so the
  rows it assigns would keep the sensitivity of the value they held before it,
  and it refuses rather than go stale (GH #205). And an SBML L3 §3.4.5 t=0
  fire, which is *not* refused, is deliberately not differentiated either: its
  trigger was already satisfied when the run began, so it fires at `t_start`
  for every θ in a neighbourhood and `∂t*/∂θ` is 0 there.

  Validated against central finite differences of whole trajectories: on the
  Izhikevich `neuron` (all four parameters, 13 spikes over the window) the
  disagreement falls as `h²` to 1.4e-6, and against closed forms where one
  exists. On the event-carrying `rr_parity` subset (194 models, 3 sensitivity
  parameters each): **32 models newly answered**, 0 lost, 131 byte-identical;
  every remaining state-dependent refusal names its obstruction (17
  conjunctions/disjunctions, 3 equalities, 2 unresolvable crossing rates). The
  585-model `.net` control arm under sensitivities is byte-identical.

### Fixed
- **The event-assignment jump dropped `∂h/∂x` on every SBML model, and the
  derived-parameter chain in `∂h/∂p` on all of them (found while validating
  issue #144).** The GH #212 jump differenced an event assignment only with
  respect to the variables whose *bound addresses* it referenced. Both halves of
  that test have a hole, and both fail silent-zero rather than loud:

  * ModelBuilder registers a species as an ExprTk variable only when its name is
    free, and an SBML model gives each species a same-named observable — so the
    `S` in `S := 0.5·S` binds to the observable total, never to
    `&sp.concentration`. The difference perturbed nothing and reported
    `∂h/∂x = 0`, which restarts the sensitivity column from zero at the event.
    On `dS/dt = −kS` zero is a fixed point, so the column stayed there for the
    rest of the run. (This is issue #52's shadowing, on the assignment side.)
  * A derived parameter hides its primaries: `S := 2·Ktot` with
    `Ktot = k1 + k2` references neither `k1` nor `k2`, so a requested `k1`
    looked absent and its column came back flat. This was a documented
    "Phase-1 limitation"; it is now differenced through the chain.

  Both are closed by one shared support computation
  (`NetworkModel::expression_support`), which follows an observable to the
  species behind it and a derived or assignment-rule-bound parameter to the
  primaries behind it. The same function prunes issue #144's `∂g/∂x` and
  `∂g/∂p`, so all four differences agree about what "reads" means.

  Three corpus models move numerically as a result, all three with a fixed-time
  event whose assignment adds to the species it writes (`V := V + dose`). On
  BIOMD0000000816/817 the affected column was being reset to zero at each of
  three doses and now tracks a central finite difference to 1.5e-7; on
  BIOMD0000000480 the assigned rows are numerically zero either way and only
  their last bits move.
- **Two user-facing refusals cited "GH #212", which is not an issue in this
  repository** (lanl/bngsim's highest is #143; lanl/PyBNF#212 is an unrelated
  closed ticket). A user who hit either had nowhere to look. Both now point at
  issue #144, and the internal uses of the label — an unpublished phase plan —
  carry one anchor comment saying what the phases are and where each is tracked.

## [0.12.1] - 2026-08-02

### Fixed
- **`set_param` no longer overwrites a state a run advanced, when the species
  happened not to move (issue #141).** #79 made `set_param` re-resolve a species
  initial condition that names the written parameter, and decided whether the
  *live* value still belonged to that IC with `concentration == initial_conc`.
  That is a value test standing in for a provenance question, and the two part
  company whenever the dynamics leave a species exactly where it started: the
  test reads "still at baseline" and the rebuild discards integrated state.

  `kinetics_mb1n.bngl` is the case in the wild — free antigen is not consumed
  while its `m()` switch is 0, so its carried value is bit-identical to
  `min(Agtot,Agmax)`, and the next `setParameter("const",0)` in a 10-segment dose
  protocol zeroed the pool mid-run. It went PASS → DIFF (`max_rel_err` 1.0, the
  whole final segment) in `bng_parity` on the 0.12.0 release sweep. Any
  `run()` → `set_state()` → `set_param()` script is exposed; nothing warned.

  The live state now also requires that no run has advanced it — GH #210's
  `ic_state_dirty`, *read* here, never set (setting it from `set_param` is what
  the issue originally proposed, and it makes the next forward-sensitivity run
  raise instead). `parameter_scan`'s per-point reset clears the marker with the
  rest of its restore, since `reset_conc=True` is a declared fresh start; the
  pre-equilibration carry (#81) is `reset_conc=False` and keeps it.

  Deliberately unchanged: a caller who assigns a species *the same number its own
  declared IC produces*, on a model that has not run. There the assignment and the
  expression agree numerically and which was meant is genuinely ambiguous —
  issue #113's `_superseded_ic_rows` documents that same case and keeps the row,
  with `declare_ic_sensitivity` (#111) as the way to say either. Only the
  unambiguous case — a run advanced the state — is carved out, so the #79 rebuild
  and the #113 sensitivity seeding still agree.

## [0.12.0] - 2026-08-02

### Added
- **A build-time budget for the expression output-sensitivity derivation, and one
  analysis per model instead of two (issue #97).** #90 bounded every symbolic
  derivation on the sensitivity-RHS build. One sympy site on the *same* build was
  left unbounded: `_analyze_output_sens`, which parses every global function and
  every derived-parameter expression and takes one derivative per symbol each
  references, for the GH #198 chain rule `d func/dθ`. So the "build appears to
  hang" failure mode #90 removed was still reachable one emitter over.

  The budget reads the same `BNGSIM_SENS_DERIV_BUDGET_S` — one knob for one build
  — but resolves its **own** deadline, so a slow `∂f/∂p` cannot starve this phase
  (and it scales with **derivation steps**, not species count: `MODEL1112100000`
  carries 3633 global functions on 1265 species, and a species-scaled curve is
  loose exactly where this work is). An expiry does **not** decline anything:
  output sensitivities are per function, so every function derived before the
  deadline keeps working and the rest are marked `unsupported`, which the emitted
  C already expresses as a NaN sentinel and the `Result` raises — now naming the
  budget and the override — at selection time.

  Measured over the BioModels SBML corpus on the current emitters, the slope is
  ~9x the worst rate anything real derives at and the base ~14x the worst model
  below the knee, so no corpus model changes behaviour:

  | model | species | steps | analysis | ms/step | headroom |
  |---|---:|---:|---:|---:|---:|
  | `MODEL1603150001` | 6047 | 15568 | 85.7 s | 5.5 | 9.1x |
  | `MODEL1504130000` | 5063 | 14880 | 67.2 s | 4.5 | 11.1x |
  | `MODEL1112100000` | 1265 | 14532 | 13.3 s | 0.9 | 54.7x |
  | `BIOMD0000000497` | 295 | 3986 | 19.2 s | 4.8 | 10.4x |

  A flat budget was the other candidate the issue proposed, on the strength of a
  measurement that is now stale: `BIOMD0000000063` at 10.2 s on nine species is
  0.86 s on the current tree (#96's printer fix), and with the expression-driven
  outliers gone the remaining cost is size-driven. A flat 20 s would cut the first
  two models above, which today complete and emit.

  The analysis is now **memoized on the model**, which is a correctness
  requirement rather than a saving. It is genuinely evaluated twice per
  sensitivity workflow — once by the C emitter, once by the `Result`'s support map
  — and a wall-clock bound makes it no longer a pure function of the model, so two
  evaluations can cut at different functions and the emitted C would carry a NaN
  for a function the support map reports as supported. One evaluation, shared
  (and clones inherit it, so parallel fitting does not re-derive per worker).

  What the budget does not buy, stated because the issue is explicit about it: a
  deadline can only be checked *between* sympy calls, so a single pathological
  expression still overshoots by one uninterruptible `sp.diff`. This bounds the
  accumulating case; the outliers are derivation defects and are fixed where they
  live (#96, and #99 for `synthesis_v3`).

- **`steady_state(mask=…)`: solve `f(y) = 0` on a subspace, and say when a
  write-only accumulator is why a solve failed (issue #74).** Counting cumulative
  flux with a "degraded" / "produced" / "secreted" pool — a species some reaction
  produces and none consumes — is a common BNGL idiom, and it put every such model
  outside `steady_state()`'s reach. The pool's derivative is a non-zero constant
  for as long as its producing reactions fire, so `||f(y)||₂/n_species` has a floor
  above `tol` and the solve reported failure however long it integrated, with
  nothing on the result to distinguish that from ordinary numerical trouble. On
  `beta_catenin_destruction_complex_barua2013` (409 species, 2737 reactions, four
  pure sinks) the residual sits at **7.4990e-3 across `max_time` = 2.5e6 / 2.5e7 /
  2.5e8** while `max|y|` grows linearly through 7.49e6 → 7.49e8 — a constant
  derivative, not a slow tail — even though the other 405 species are settled to
  1e-10 by the first horizon. The documented workaround was to edit the model file
  and delete the accumulator's products, which is only provably safe *because* the
  species is a pure sink, exactly the property the library can check itself.

  Three pieces, mirroring AMICI's `Model.set_steadystate_mask`:

  - `Simulator.steady_state(mask=…)` and `steady_state_batch(mask=…)` take a
    boolean array over species (or the names to keep) selecting what enters the
    convergence norm. The divisor follows the mask (`‖f_included‖₂ / n_included`),
    so `tol` keeps its meaning as a per-species residual scale. Integer indices are
    rejected as ambiguous between a 0/1 mask and a list of indices.
  - `Model.pure_sink_species()` / `Model.is_pure_sink()` find the accumulators from
    the reaction list, so the mask needs no hand-listed species:
    `sim.steady_state(mask=~model.is_pure_sink())`. A species qualifies when it is
    a product of ≥1 reaction, a reactant of none, **read by no other species'
    derivative**, and not a `$`-fixed boundary condition. The third clause is not
    implied by the first two — an Elementary rate law reads only its reactants, but
    a Functional one reads observables — and is what makes excluding the species
    provably harmless to the rest of the system.
  - A failed solve now reports `ss.unconverged_pure_sinks` (and logs a WARNING)
    naming any accumulator that was in the test and is carrying flux. An empty list
    means the failure was *not* structural, so more `max_time` may still help.

  Barua 2013 with the mask: **converged, residual 9.10e-10, 405 of 409 species in
  the norm**, on both `method="integration"` and `method="newton"`.

  The mask also restricts the KINSOL polish's unknown set and the `dY_ss/dp` linear
  system, because it has to: an accumulator contributes a structurally zero
  Jacobian *column*, so leaving it in makes both systems singular at every seed —
  this is the unexplained half of GH #27 Bug 3 ("Barua 2013's 404×404 rank-deficient
  system"). Excluded species are held at the values integration left them at, which
  is exact precisely because nothing else's derivative reads them, and their
  `dY_ss/dp` rows come back **NaN**: a species with no steady value has no
  steady-state gradient, and `0.0` would be a confident wrong answer a fitter would
  read as "this parameter does not matter". `ss.n_residual_species` and
  `ss.excluded_species` report what the test covered.

  Unmasked behaviour is unchanged, and an all-true mask is routed back onto the
  unmasked path rather than through the restricted one, so `mask=ones(n)` cannot
  quietly mean something different from `mask=None`. Verified as a no-op across the
  585-model `ode_fullnet` corpus.

- **The initial condition an `on_point` hook assigns gets its own `∂x(0)/∂θ`
  (issue #111).** #81 carried the pre-equilibration's `dx/dθ` into a scan and
  treated a hook's `setConcentration` dose as a literal — `∂x_k(0)/∂θ = 0` for
  that species. That is right for the ordinary dose and wrong, silently, for a
  dose *computed from* a fitted parameter (nM converted to molecules through a
  fitted volume) or for an *increment* of the carried pool (`x_k + dose`, which
  should keep the carried row). The hook assigns the state its point integrates
  from, so that state's derivative is the hook's own — and bngsim now obtains it
  instead of assuming it.

  Row by row, the most specific thing available wins: a row the hook installed
  wholesale, then one **declared** with the new
  `Model.declare_ic_sensitivity({species: {param: value}})`, then one **measured
  through the hook**, and otherwise the carried row bit-exact (a known-exact
  number is never routed through a measurement). The measurement is the chain rule
  by difference quotient — the hook is a map `H: (x, θ) → x'`, so
  `dx'/dθ_i = ∂H/∂θ_i + (∂H/∂x)·s_i` with `s_i` the carried column, and each term
  is a central difference of the hook (in θ_i, and along `s_i`). That second term
  is what makes an increment come out right rather than being mistaken for a
  literal.

  Measured against a closed form on `preequil_prod_deg.net` (four hook shapes,
  three doses off one equilibration, `sensitivity_params=[k_prod, k_deg]`):

  | `on_point` assigns | exact `∂A(0)/∂θ` | max rel err vs closed form |
  | --- | --- | --- |
  | `A(0) = 3.0` (literal) | `(0, 0)` | 1.3e-10 |
  | `A(0) = dose·k_prod` | `(dose, 0)` | 1.8e-10 |
  | `A(0) = dose·√(k_prod·k_deg)` | `(dose√(k_deg/k_prod)/2, dose√(k_prod/k_deg)/2)` | 3.1e-08 |
  | `A(0) = A_ss + dose` (increment) | `(1/k_deg, −k_prod/k_deg²)` | 2.6e-10 |

  Rows 2 and 4 were 100% wrong before this change (reported `0`); row 1 is
  unchanged and row 3 carries the difference quotient's truncation, the only place
  the answer is not exact.

  On `IGF1R_model_v1` (589 species / 4198 reactions) with the *real* conversion —
  ligand amount `= dose_nM·1e-9·NA·Vecf`, `Vecf = 2.1e-9·f`, `f` fitted — the
  measured `d(pY980)/df` matches central FD over the **full** protocol at
  **9.4e-11 … 5.5e-10** across two step sizes, and equals the declared route to
  1.6e-11. The pre-#111 literal rule was off by **79% / 94% / 93%** at
  dose = 0.1 / 1 / 10 nM. Probing cost 26 hook calls per point at three
  parameters and no measurable wall clock (0.66 s either way) — the ODE solve
  dominates; declaring every written row reduces it to the one nominal call.

  Refusals rather than a plausible number: a hook whose assigned IC is not
  differentiable in a parameter (a dose rounded to whole molecules — the two step
  sizes disagree by the O(1/h) blow-up of a jump), a hook that raises at a
  perturbed input, and a hook that is not a deterministic function of
  `(model, value)` (checked by re-running it). Each names the species, the
  parameter and the fix, which is to declare that row.

  `declare_ic_sensitivity` is honoured on a plain `run()` too. That closes the same
  hole in the *hand-assigned* case: `set_concentration` replaces the `.net` IC
  expression that the parameter-graph seeding (issue #43) differentiates, so a
  hand-assigned θ-dependent IC previously seeded `0` with no way to say otherwise
  (measured on the toy model: reported `0` against a true `∂A(0)/∂k_prod = 3`;
  with the declaration, 1.2e-10 against the closed form).

- **Forward sensitivities carry from a pre-equilibration into a parameter scan
  (issue #81).** A dose-response experiment that pre-equilibrates and then scans
  could not be fit by a gradient method: `parameter_scan` / `bifurcate` refused
  every sensitivity-configured `Simulator` outright. The refusal was right —
  each point starts from the equilibrated snapshot, so re-seeding it as if it
  were the model's seed ICs discards the `dx/dθ` accumulated during the
  equilibration — but there was no *correct* option to offer instead. Now there
  is: the state and its θ-derivative travel together.

  `carry_sensitivities=True` (#210) already had the right semantics for a
  sequential two-phase run; what was missing was that every primitive which
  *restores* a state dropped the derivative. Three did, and each is fixed at the
  level where the state lives:

  * `Model.save_concentrations()` redefines the IC baseline to the current state.
    That state's `dx/dθ` did not change, so the new baseline now **inherits** it
    (stashed alongside `initial_conc`), and `reset()` restores both. A baseline
    saved with nothing carried is θ-independent literal ICs, i.e. the pre-#81
    fresh start — so `reset()` on an ordinary model still means "fresh start",
    and the every-action-reset backends are unaffected.
  * `Model.save_concentrations(label=…)` / `restore_concentrations(label)` capture
    and restore a named snapshot's `dx/dθ` the same way.
  * `Simulator.parameter_scan` / `bifurcate` restore the reset target's state
    **and** its `dx/dθ` per point, integrate each point with
    `carry_sensitivities=True`, and leave the model — state, scanned parameter,
    carried derivative, dirty flag — exactly as they found it. A continuation
    scan (`reset_conc=False`) instead chains each point's `dx/dθ` from the
    previous point, making the whole sweep one differentiable protocol.

  The `NetworkModel` seed accessor gained its write half
  (`set_pending_sensitivity_seed(seed, param_names)`), which is what lets a
  protocol restore a state together with its derivative;
  `has_baseline_sensitivity_seed` introspects the baseline's own.

  Measured against a closed form (`preequil_prod_deg.net`: `dA/dt = k_prod −
  (k_deg + dose)·A` equilibrated at `dose=0`, so both `dA/dθ` columns are exact
  for every dose), scanning four doses off one equilibration:

  | dose | carried (this change) | re-seeded per point, `t=0` | re-seeded, `t=3` (`k_prod` / `k_deg`) |
  | --- | --- | --- | --- |
  | 0.5 | 1.2e-10 | **100%** | 9.5% / 15.3% |
  | 1.0 | 1.7e-10 | **100%** | 3.3% / 8.4% |
  | 2.0 | 2.3e-10 | **100%** | 0.3% / 1.3% |
  | 4.0 | 1.2e-09 | **100%** | 0.0% / 0.0% |

  The trajectories are identical to 1e-7 in every row — the seed is the only
  difference — and the same scan scored at each point's own steady state
  (`steady_state=True`) matches `dA_ss/dθ = (1/λ, −k_prod/λ²)` to 1e-12. Note the
  shape of the wrong column: re-seeding is 100% wrong at `t=0` and its error
  *decays* with dose, so it is worst at the low-dose end of a dose-response —
  the informative part of the fit — and looks fine at saturating dose. That is
  the failure mode a fit cannot detect on its own.

  On the issue's own model family (`IGF1R_model_v1`, 589 species / 4198
  reactions, equilibrated at a basal ligand dose so 1107 of 1178 carried seed
  entries are live, then scanned over three doses applying each with `on_point` +
  `set_concentration`): `d(pY980)/d{kp, kdp}` matches central FD of the measured
  observable over the **full** protocol at **1e-8 to 1e-9**, stable across
  `h/p = 1e-3, 1e-4, 1e-5` (so it is signal, not FD noise), and the whole scan
  runs off ONE equilibration in 2.0 s. The scan is also bit-identical to the
  hand-rolled equivalent — restore the snapshot, install the seed, `run(
  carry_sensitivities=True)` — on every point.

  This also closes a silent version of the same defect that needed no scan at
  all: `equilibrate → save_concentrations() → run(sensitivities)` used to drop the
  carry *and* clear the "state is carried-over" flag, so it re-seeded a
  θ-dependent initial condition as a fresh start and returned wrong derivatives
  with no warning (100% off at `t=0` on the model above). The BNG action ordering
  `simulate(steady_state) → saveConcentrations() → parameter_scan` is exactly
  that path, and now carries correctly.

  Refusals (never a silently re-seeded gradient) when the reset target carries no
  matching `dx/dθ`, when the scanned parameter is itself a `sensitivity_params`
  entry (each point overwrites it, so the derivative carried into the point was
  taken at a different value of the same symbol), when `sensitivity_ic` is
  requested across the boundary, or when an `on_point` hook moves a differentiated
  parameter. An `on_point` hook *may* apply the usual coupled `setConcentration`
  dose override; how that override's own `∂x_k(0)/∂θ` is obtained is described in
  the issue #111 entry above (it was a literal-⇒-zero rule as this shipped).
  `run_batch` remains the right primitive for a sweep whose points start from the
  model's own seed ICs.

  Side effect of the same restore work: a plain (non-sensitivity) scan no longer
  leaves the model marked as carried-over dynamics — it rewinds the state it
  advanced, so the flag it rewinds to is the one it found.

- **Forward sensitivity w.r.t. an onset time encoded as an SBML *event* (issue
  #49).** A switch time written as `piecewise(kin, time >= T0, 0)` has been
  differentiable since #48; the *same* switch, same dynamics, same gradient,
  written as an event (`time >= T0` → `on := 1`, rate `kin*on`) was refused.
  Which encoding a model uses is usually decided by whichever tool exported it,
  so the asymmetry was arbitrary from the modeller's side. A survey of
  `parity_checks/rr_parity` found 24 models / 88 events whose trigger thresholds
  a fitted constant, with names like `treatment_start`, `ton`/`toff`, `tstim`
  and `Lockdown_start` — exactly the things one fits.

  The general jump across an event at a parameter-dependent time is

      s⁺ = ∂h/∂x·(s⁻ + f⁻·∂t*/∂p) + ∂h/∂p − f⁺·∂t*/∂p

  — shift `s⁻` along the pre-event flow by how far the event time moves, apply
  the event Jacobian, shift back along the post-event flow. Both corners were
  already implemented: `∂t*/∂p = 0` is the GH #212 state jump, `h = identity` is
  the #48 switch jump. This adds the cross terms and the `∂t*/∂p` that feeds
  them (`bngsim._switch_sensitivity.compute_event_time_sens`, which reduces the
  trigger through the *same* `_clock_threshold_split` recognizer the rate-law
  path uses, so the two cannot drift about what a locatable crossing is).

  Unlike #48 this needs neither `CVodeSetStopTime` nor parameter pinning: an
  event trigger is not part of `f`, so `f` is smooth right up to the root
  (CVODE's root finder already stops exactly at `t*`) and `∂f/∂T0` is a genuine
  zero without help. It does need #48's nominal-parameter restore — `f⁻`/`f⁺`
  multiply `∂t*/∂p`, so reading them wherever CVODES' last finite-difference
  probe left the parameters would scale the whole jump by `1 ∓ √rtol`, i.e. an
  answer that moves when `rtol` does.

  Two refusals were also retired as *vacuous* (the issue's sub-finding). A delay
  of literal `0` is not a delay — `process_firing_batch` already takes the
  immediate path for it, so there is no trigger-time-to-execution-time window;
  and `persistent=false` can only cancel a fire inside that window (SBML L3v2
  §4.11.3), so without a delay the flag has nothing to act on. Ghanbari2020 and
  Zongo2020 were blocked by exactly this.

  Corpus result on the 225 event-bearing `rr_parity` models, requesting **every**
  parameter: **13 models newly allowed** (BIOMD1, 117, 152, 153, 244, 301, 327,
  340, 422, 494, 650, 820, MODEL2310250001), 114 unchanged, 0 lost. Validated
  against a central difference of the trajectory itself, normalized to the
  observable's own scale and filtered to samples where the difference quotient
  is self-consistent across two step sizes: the onset column's worst error sits
  at or below the same model's *control* (non-trigger) parameters — 3.5e-4 vs
  3.2e-4 on Owen1998 (BIOMD650), 2.1e-7 vs 2.7e-4 on BIOMD301, 8.3e-6 vs 3.3e-4
  on BIOMD820. The two models where the onset column starts worse (BIOMD301
  `pulse1_length` 1.0e-1, BIOMD327 `ton` 1.3e-2 at the default `rtol=1e-8`)
  converge with the integration (1.0e-5 and 1.2e-3 at `rtol=1e-10`), which is
  the CVODES difference-quotient sensitivity RHS's own accuracy on those models
  rather than the jump.

  One correctness trap worth recording, because it is the failure mode this
  module exists to avoid rather than an omission. A threshold that is an SBML
  `<assignmentRule>` parameter is *not* a constant, even though
  `param_is_expression` is false for it and reading its current value looks like
  reading a literal — the loader turns such a rule into a model function bound
  to the same-named parameter. BIOMD0000000301 writes its pulse schedule as
  `pulse2_start = pulse1_start + pulse1_length + pulse_interval`; attributing
  the whole `∂t*/∂p` to `pulse2_start` put the gradient on a column no fitter
  moves and left `pulse1_start`'s at zero. Rule-bound parameters now join the
  inlining map when their body reduces to arithmetic over parameters, and every
  identifier that survives the flattening must be a primary — anything else
  (`floor()`-based dose schedules, a rule that reads state) is refused with a
  reason instead of evaluated at its current value.

- **A build-time derivation budget for the sensitivity `∂f/∂p` path (issue #90),
  the last unaddressed "still has to bail" item of #55.** #95/#187 bound the
  symbolic derivation of the analytical *Jacobian*, so a model that does not
  derive in time falls back to the finite-difference Jacobian instead of hanging
  the load. The `∂f/∂p` path added by #55 (#65/#66/#67/#68) does the same kind of
  sympy work and had no budget at all: `python/bngsim/_codegen.py` contained zero
  references to `deadline`, and its three `sp.diff` sites were unguarded. The one
  that scales badly runs **one `sp.diff` per (distinct rate law, parameter it
  reads) pair** — the product of rate-law size and parameter count, exactly the
  axis #95 found super-linear — so a genome-scale Functional model with a long
  `sensitivity_params` list would present as a build that *hangs* rather than one
  that declines and says why (and to a `rr_parity`-style harness, which times
  build and solve together, as an ODE timeout).

  The derivation is now bounded by `BNGSIM_SENS_DERIV_BUDGET_S`. It shares the
  Jacobian budget's base, per-species slope and override grammar — one
  implementation, so the two policies cannot drift — but is resolved
  independently, so `BNGSIM_JAC_DERIV_BUDGET_S=inf` (the documented genome-scale
  workaround for keeping an analytical Jacobian) does not silently uncap it.
  The deadline is threaded down to every `sp.diff` on the path — the Functional
  rate laws and both flavours of derived-parameter chain rule, on the model path
  and the `.net` text path alike — and checked on entry to each rate law as well
  as before each partial, so overshoot is bounded to one (law, parameter) pair
  rather than one whole model. Expiry declines through the existing
  `_warn_functional_sens_rhs_refused`, landing beside every other decline reason
  with a warning naming how far the derivation got and the override to raise.

  It also covers the *other* derivation on the same build, which the issue's
  three-site table does not list: `ySdot = J·yS + ∂f/∂p`, and the `J·yS` half
  re-derives ∂f/∂x through `_functional_jacobian_groups`. That is the same math
  `attach_functional_jacobian` already ran at load under the #95 budget, but a
  re-derivation with no clock of its own — and on a model whose load-time attach
  was itself cut off there is no earlier bound to inherit, so bounding only ∂f/∂p
  would have left the identical hang reachable one call later. The #151 native
  saturable emitters run no sympy and are unaffected; only their fallback takes
  the deadline.

  **One difference from the Jacobian budget is deliberate: this one never becomes
  unbounded by size.** `_FD_NONVIABLE_SPECIES` exists because past that scale an
  FD Jacobian does not converge — there is nothing to fall back *to*, so the
  analytical Jacobian is mandatory. Declining a sensitivity RHS only hands the
  columns to CVODES' internal difference quotient, which is what every Functional
  model used before #55 and which is correct at every scale. Correct is not cheap
  (measured 9–37x per column, and on a stiff model at a tight `rtol` the DQ's
  `~sqrt(rtol)` accuracy collapses the step size outright), which is why the
  decline is a warning that names the knob rather than a silent downgrade.

  Corpus A/B over all 585 `.net` models, default budget vs unbounded: **544 models
  emit an analytic sensitivity RHS in both arms, 0 source hashes differ**, total
  derivation 7.6 s with the slowest single model at 0.44 s — 46x headroom under
  the 20 s base, so nothing real is near the cut-off. An explicit override also
  joins the `.net` path's in-process memo key and its on-disk `model_hash`
  (that path keys on the model's *content*, not on the generated C, so without it
  a build made under a tight budget would be served back to one made without it —
  the same trap #67's A/B hatch had to sidestep).

- **Per-model composition counts in the ODE Jacobian characterization report
  (issue #42).** `jacobian_characterization.py` recorded rich Jacobian-side
  structure (`N`, `rank`, `n_reactions`, `density`, `stiffness_ratio_*`) but
  nothing about model composition, so the paper's representative-BNGL-models
  table had to carry its IC-not-zero and parameter-count columns as hand-curated
  constants — the one column pair in that table not derived from data. Each
  result row now also carries `n_seed_nonzero`, `n_parameters`,
  `n_independent_parameters`, `excluded_parameters` and
  `excluded_parameter_reasons`.

  `n_seed_nonzero` is read off the loaded network rather than by counting `seed
  species` lines, so parameter-valued initial conditions are already evaluated:
  `Lang_2024`'s 73 seed lines resolve to 65 nonzero species, and a line whose
  parameter is `0` is correctly not counted. `n_independent_parameters` applies
  the table caption's rule — numeric-valued independent parameters, with
  unit-conversion constants and parameters derived from others excluded — which no
  single parse rule reproduces on its own, so every excluded name is emitted with
  its reason (`derived` or `unit_conversion`) and
  `n_independent_parameters == n_parameters - len(excluded_parameters)` holds by
  construction. BNG's synthetic `_rateLaw{N}` symbols are dropped before the
  census and appear in neither, and the unit-conversion name sets are recorded in
  the report's `_meta.params`. Avogadro's number is matched case-sensitively and
  guarded by magnitude, so `race.bngl`'s `nA 5` — an Erlang step count — stays a
  model parameter while `Na 6.022e23` and the RuleHub tutorials' `NaV 6.02e8`
  (Avogadro rescaled to /um^3) are excluded. Across the 278-model `bngl_models`
  slice the only names the policy excludes are `NA`, `Na` and `Vref`. The six
  exemplars reproduce the curated values exactly: `Lang2024` 65/205,
  `Kocieniewski2012` 4/10, `Barua2007` 2/22, `Blinov2006` 6/43, `Barua2013` 6/25,
  `fceri_fyn` 5/31.

- **`Simulator(..., force_sparse_linear_solver=True)` for `method="ode"`, the
  missing counterpart to `force_dense_linear_solver` (issue #29).** The ODE
  linear-solver kind is auto-selected — sparse KLU when a model is both large
  (`n_species >= 50`) and sparse (Jacobian density `< 0.10`), dense otherwise —
  and until now the only override pushed toward dense. That made the rule
  unfalsifiable on the half of a corpus it sends to dense: forced-dense shows
  what KLU buys on large sparse networks, but nothing showed KLU's setup and
  indexing overhead on the small dense ones, which is the evidence that the rule
  does real work rather than being replaceable by "always KLU". Under the auto
  rule the 585-model BNGL ODE corpus splits 541 dense / 44 KLU; the forced-sparse
  arm now covers the other 541.

  The flag bypasses the size and density gates and nothing else — KLU still
  needs a real sparsity pattern and a non-JAX Jacobian, so it stays a no-op in a
  build without KLU, exactly as `force_dense_linear_solver` documents itself.
  Passing both force flags raises `ValueError` (and the C++ `SolverOptions` path
  rejects the pair too) rather than letting one quietly win, which would hand a
  benchmark auto-selected numbers under a "forced" label. Flipping the flag
  between `run()` calls on one `Simulator` invalidates the warm CVODE cache, so
  the solver is rebuilt rather than reused.

  Forced-sparse reaches KLU on all 585 models. Getting the last 7 there took
  making the Curtis-Powell-Reid coloring lazy (below); a run that still cannot
  supply a sparse Jacobian is refused with a message naming the flag, rather
  than failing inside CVODE with "no Jacobian constructor available".

  `benchmarks/suites/ode_fullnet/run_forced.py --mode sparse` is unblocked.

- **One BNG2.pl resolver for the whole suite (`parity_checks/_core/bngpath.py`),
  plus a `parity` dependency group that provisions it.** BNG2.pl was located by
  six near-duplicate helpers across eleven files that disagreed about precedence
  (see Fixed). They are replaced by `resolve_bng()` / `require_bng()` /
  `skip_reason()`, which try `$BNG2_PL` → `$BNGPATH` → PyBioNetGen's bundled copy
  and record every attempt, so a failure names each location searched and what it
  yielded instead of reporting a bare absence. Two consequences beyond the fix:
  the sweep entrypoints (`parity_golden.py`, `bng_ode_run.py`, `bng_stoch_run.py`)
  now find the bundled BNG2.pl on their own — no `export BNGPATH` needed to
  regenerate a golden — and a stale env var falls through to a working install
  rather than poisoning the lookup. `uv sync --extra dev --group parity` installs
  the pinned PyBioNetGen (which bundles BNG2.pl, `bin/run_network` and
  `bin/NFsim`), taking `parity_checks/tests/` from 264 passed / 14 skipped to
  **288 passed / 0 skipped** with no environment variables set. The group lives
  in PEP 735 `[dependency-groups]`, not `[project.optional-dependencies]`, because
  its `git+https://` direct reference would otherwise land in published metadata
  and make the distribution unpublishable to PyPI; `test_pin_agreement.py` fails
  if its commit drifts from `requirements-pybionetgen.txt`, the source of truth.

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

### Changed
- **The two forward-sensitivity jump handlers move out of `CvodeSimulator::run`
  into named `Impl::` methods (issue #135, follow-up to #109).** #109 stopped at
  the sequential setup and left six function-scope lambdas behind, fenced as
  "the event / switch-crossing logic". They split cleanly in two, on one
  question: does the lambda touch the mutable event bookkeeping the stepping
  loop also writes (`trigger_was_true`, `pending_events`, `event_dormant`,
  `event_rng`)? `process_firing_batch` and `cascade_triggered_events` do —
  moving those needs a shared run-state struct, where a parameter accidentally
  taken *by value* would compile, run, and silently break event edge detection.
  The two *sensitivity jumps* touch none of it, which is why they can move and
  those two stay.

  `restore_nominal_params`, `capture_event_sens`, `apply_event_sensitivity_jump`
  (the GH #212 event jump plus issue #49's four-term event-time composition) and
  `apply_switch_sensitivity_jump` (issue #48's crossing jump, carrying issue
  #82's threshold-landing correction) are now `Impl::` methods, each with the
  reasoning that block carried. `run()` drops from **957 NLOC / CCN 253 to
  709 / 189** (`lizard`).

  The case is **navigability at the point of highest churn**, not size: #48, #49
  and #82 all landed inside these two bodies, and until now they were anonymous
  lambdas ~1,100 lines into `run()`. It does *not* reduce coupling — the event
  jump needed 13 pieces of run state before and needs the same 13 now; what
  changes is that the dependency is a parameter list instead of an implicit
  `[&]`. `SensitivityState` (added by #109) already carried six of them, so each
  signature gains three or four arguments rather than the eleven to thirteen a
  free-function extraction would have taken. The switch jump's scratch buffers
  become a `SwitchJumpScratch` that `run()` still pre-sizes before the loop, so
  a crossing allocates nothing — and it is **non-copyable on purpose**, because
  passing it by value would compile and the only symptom would be the
  per-crossing allocation the pre-sizing exists to prevent. (`SensitivityState`
  gets that guarantee for free: its `N_Vector` guard is non-copyable.)

  Two things fall out. `run()` loses three aliases it kept only for these
  lambdas (`sens_method`, `sens_param_indices`, `sens_p`). And the guard
  `if (!wants_sensitivity || n_sens == 0)` was testing one condition twice —
  `wants_sensitivity` is `!param_names.empty() || !ic_species_names.empty()`,
  and `SensitivityState::n_total` is set once to the sum of those two sizes and
  never changed — so it is now written once.

  **Extraction only — no behavior change, no API change, no performance
  change.** The bodies moved verbatim, comments included; the stepping loop, the
  two firing lambdas and the jump maths are untouched.

  Verified as **byte identity**, the standard #109 established: each arm digests
  the full float64 bytes of times, species, observables, expressions and all
  four sensitivity blocks plus the solver-statistics dict, so one changed bit or
  one extra CVODE step moves the hash. **3,224 runs hash identically and 121
  refuse with identical messages; 0 divergences**, over six sweeps — the 585
  `ode_fullnet` `.net` models forced cold, the same 585 on the default
  (warm-eligible) dispatch, the same 585 under three-parameter forward
  sensitivities, and the 1,324-model `rr_parity` SBML corpus. Five models hit
  the harness's wall-clock cap in one arm or the other and are counted apart, as
  inconclusive rather than as divergences.

  Two of those six arms are new, because a `.net`-only sweep proves nothing here:
  `.net` models carry no events and no fitted switch time, and #109's SBML arm
  ran without sensitivities, so **neither** of #109's corpus arms executed a
  single line of these two bodies. The additions are the 194 event-declaring
  rr_parity models run *with* sensitivities (127 produce a trajectory; the rest
  are refused identically by the Python event-sensitivity guard), and a
  72-run parameter sweep over the exemplars #48/#49/#82 were validated on —
  including the three issue #82 knife-edge crossings, a t=0 event fire, and the
  onset model whose fitted parameter appears only in the trigger.

- **`steady_state()`'s march routes to sparse KLU by the same rule `run()` uses,
  and the two force flags reach it (issue #128).** `SteadyStateMarcher` built a
  `SUNDenseMatrix` unconditionally — `ss_make_dense_linsol`'s own comment said
  the steady-state paths had no KLU option — while `run()` on the same model
  routed to KLU whenever the Jacobian was large and sparse. `SteadyStateOptions`
  did not carry `force_sparse_linear_solver` / `force_dense_linear_solver` at
  all, so a Simulator built with either got a dense factorization out of
  `steady_state()` and `steady_state_batch()` without saying so.

  The decision itself now has one implementation
  (`route_to_sparse_linear_solver` in the new `bngsim/sparse_jacobian.hpp`),
  which `CvodeSimulator::Impl::choose_use_sparse` and the march both call: same
  `SPARSE_THRESHOLD` (50), same `SPARSE_DENSITY_MAX` (10%), same force flags,
  same JAX exclusion. `ss.linear_solver` reports the outcome — `"klu"`,
  `"dense"` or `"lapack-dense"` — which is the only outward sign of a change
  that moves the cost and not the answer.

  **What it buys**, `method="integration"` with the interpreted RHS at
  `tol=1e-9`: `BaruaBCR_2012` (1122 species, 2.5%) **5.59 s → 1.78 s** (3.1×),
  `fceri_fyn` (1281, 2.3%) **13.25 s → 6.92 s** (1.9×), `egfr_ground` (356,
  6.7%) 0.19 s → 0.18 s. The density ceiling earns its place at the bottom of
  that range, which is why the rule is not simply "always KLU".

  **What it changes about the answer: nothing measurable.** Over the 585-model
  `.net` corpus at `max_time=1e4`, both methods, before and after: the 536
  models the routing leaves dense are **byte-identical** (max state difference
  exactly 0.0), as they must be. Of the 44 it routes, `converged` flips on none
  under either method, and over every converged model the state moves by at most
  **2.0e-8** of the model's own scale (median exactly 0). The `n_steps` ratio on
  routed models is a median of 1.000 (0.93–1.07): KLU changes the factorization,
  and with it the step sequence, not the equations. The two models that move by
  more than 1e-7 are both *unconverged* — they return where the trajectory got
  to at `max_time`, at the same residual to four digits on both sides.

  `method="newton"` flips `method_used` on **three** corpus entries, which are
  one 354-species network under three names (`fceri_ji`, `fceri_ji_4`,
  `test_network_gen`): the KLU-routed march crosses the parity criterion
  *during* rung 0, so the ladder returns the burst — which it documents as
  "integration itself reached the parity tolerance — done" — instead of handing
  the seed to KINSOL. Converged either way, same root to 1.5e-8 of scale, at
  residual 9.5e-10 rather than the polish's 2.8e-14. Which side of `tol` a march
  stops on is set by the step sequence, so any change to it can move this.

  `jacobian="fd"` stays on the sparse route rather than reverting to dense,
  because on a sparse matrix a Jacobian callback is not optional: CVODE's
  built-in difference quotient supports dense and banded matrices only. The
  march fills the CSC values with the Curtis-Powell-Reid *colored* difference
  quotient — the same `colored_fd_jacobian` `run()` uses, now shared rather than
  copied, and differencing the march's own (compiled, when there is one) RHS.
  Only `jacobian="jax"` forces the dense route. One case is deliberately not
  routed where `run()` refuses instead: a Jacobian with no structural nonzero
  stays dense, because the steady-state auto rule can reach it unprompted
  (density 0 < 10%) and `f(y) ≡ 0` makes such a model a steady state that used
  to solve immediately.

  **The KINSOL polish and the `dY_ss/dp` solve are left dense on purpose.** Both
  factor the *reduced* system — the model's Jacobian projected through the
  conservation-law reconstruction — and that projection fills in entries the
  model's sparsity pattern does not have, so the reduced pattern is a different
  object that would have to be derived. Both also factor once per solve rather
  than once per integration step.

  `CvodeSimulator` gives up its private copies of the routing rule, the CSC
  structure reinstall (three sites) and the difference formula to the shared
  header; that half is behaviour-neutral, checked as byte-identical trajectories
  and solver stats over 882 (model, jacobian, route) cases on 147 corpus models,
  0 divergences. Pinned by a new native case
  (`test_steady_state_linear_solver_routing`, whose KLU-less half asserts the
  pre-#128 dense behavior on the CI build that has no KLU), a steady-state class
  in `test_force_sparse_linear_solver.py`, and
  `test_steady_state_linear_solver.py` for the published networks and for the
  compiled Jacobian in either layout — the codegen emits exactly one of the two
  shapes per model (GH #162), so a forced route now converts rather than
  silently dropping to the interpreted fill.

- **Both steady-state tiers now install the Jacobian the model already has
  (issue #127).** `SteadyStateMarcher`'s constructor calls `CVodeSetJacFn` and
  `solve_by_newton` calls `KINSetJacFn` whenever the model carries a closed form
  and `jacobian=` asks for it. Neither ever did: CVODE and KINSOL each built
  their own difference-quotient Jacobian — **one RHS evaluation per unknown per
  setup** — on models whose analytical Jacobian was assembled, compiled and
  `dlopen`'d in the same object. `jacobian=` therefore now reaches the *solvers*,
  not only the two consumers that need the matrix itself (`dY_ss/dp` and the #78
  stability certificate); `"fd"` selects exactly the old behavior, and
  `ss.solver_jacobian_source` reports which matrix was factored.

  The polish's Jacobian is the interesting half. It solves on
  `ss_unknown_species` — the conservation-law independents, narrowed by any
  `mask=` — so its matrix is the *reduced* one. KINSOL's difference quotient
  differentiates the reduced residual directly and gets that projection for free;
  a closed-form fill is of the full `ns × ns` system and is projected by hand
  through the same `ss_reduce_jacobian` chain rule `dY_ss/dp` and the certificate
  already use.

  **What it buys.** Seeded at its steady state, `SHP2_base_model`'s single polish
  now makes **4** model-level RHS calls where it made 151 (147 unknowns). Over
  the 585-model `.net` corpus — 560 of which have a closed form to install — a
  solve makes a median 1.09× fewer RHS evaluations (`method="newton"` 1.10×, up
  to 33×). Wall clock depends on what share of the solve the Jacobian setup was:
  interpreted, 1.08–1.29× on 149–356 species; compiled, 1.06× at 356 species and
  1.19× on `fceri_fyn` (1281), but **0.87–0.89× at 149 species** — a compiled RHS
  makes a difference-quotient column cheap, and a profile there puts ~80% of
  CVODE's time in `SUNDlsMat_denseGETRF` and none in the fill. That dense
  factorization (the march cannot route to KLU) is issue #128, filed separately
  so the two are not conflated.

  **What it changes about convergence**, measured before/after on the corpus at
  `max_time=1e4`, both methods. `method="newton"`: **one** model flips
  `converged` and it flips *to* True (`Ras_WT_in_vitro`, residual 4.6e-1 →
  3.7e-10); 16 flip `method_used`, 8 gaining the polish (residual 1.2e-10 →
  5.1e-17) and 8 losing it (8.3e-12 → 5.6e-10, still under `tol`), all 16
  converged either way. `method="integration"`: 5 flip `converged`, 3 to False
  and 2 to True, and every one of the five returns the *same state* — they moved
  by 1e-18 to 1.5e-9 of the model's own scale, with the residual straddling
  `tol`. These are models whose parity residual is a cancellation of large fluxes
  and therefore at its own roundoff floor; a different step sequence lands one
  ulp either side of the criterion. The 20 corpus models with no closed form are
  byte-identical before and after, as they must be.

  **One model needed the analytical Jacobian called off**, and it is the model
  GH #176's docstring already names: `l-type-calcium-channel-dynamics`, whose
  `v_rec = if((-70+V)<-20, 0.5, 0.05)` is discontinuous in a state variable the
  trajectory approaches asymptotically. The exact derivative omits the jump, so
  the corrector meets an unanticipated step, the error test fails repeatedly and
  the march collapses to `hmin` at t≈24 — the identical failure `run()` has on
  the identical model. `steady_state()` had no equivalent of `run()`'s retry;
  under `jacobian="auto"` a **hard integrator failure** (not mere
  non-convergence, which retrying cannot help) now retries the solve once on
  difference quotients, restoring that model's pre-#127 answer exactly.
  `ss.solver_jacobian_retried` says so, the Simulator remembers it so a scan does
  not re-pay the doomed march at every point, and an explicit
  `jacobian="analytical"` still surfaces the failure rather than being
  second-guessed. Three corpus models take that path, and the model itself was
  already a tracked fixture — `test_jacobian_discontinuous_fallback.py` gains the
  steady-state mirror of its four `run()` cases.

  Pinned by a new native case (`test_steady_state_solver_jacobian`, which checks
  the reduced projection against `pure_sink_conserved.net`'s closed-form root
  under both strategies) and by
  `python/tests/test_steady_state_polish_jacobian.py`, rewritten from the
  behavior it pinned for #126. One knife-edge assertion elsewhere had to be
  restated: `test_pre_fix_value_is_far_from_the_answer` compared
  `max|got − dropped|` against `0.5·max|got|`, which is algebraically
  `|got| > |exact|` and so turned on whether the march stopped one roundoff above
  or below the answer; it now states the factor of two with a tolerance.

- **`CvodeSimulator::run` puts its setup in named `Impl::` helpers (issue
  #109).** `run()` spanned 2,346 lines of one function body — 1,375 NLOC at
  cyclomatic complexity 368. The codebase is otherwise small (median function 12
  lines at CCN 3), and `run()` was the one place where high complexity, high
  churn (11 commits to
  `src/cvode_simulator.cpp` in 12 months, nearly all sensitivity work: #48, #49,
  #54, #63, #82) and a monolithic single-function body all coincided. Every one
  of those changes had to re-read the whole function to find where its case
  belonged.

  Eleven blocks of *sequential configuration* now live in named helpers, in the
  style of the existing `setup_codegen_rhs` / `setup_linsol_and_jac`: the
  `ns == 0` algebraic-only path (GH #229), the dense/sparse decision and its two
  force flags (#29/#102), the SUNContext / state vector / CVODE-memory creation
  and codegen-RHS wiring, the `opts.jacobian` validation, the whole CVODES
  forward-sensitivity initialization (`Impl::setup_forward_sensitivities`, the
  single largest block), the `Result` allocation, the event/discontinuity root
  function and its registration, the GH #197 observable-sensitivity coefficient
  table, the solver statistics, and the write-back of the final state plus the
  GH #210 carry-over seed. `run()` now reads as that sequence of calls followed
  by the integration loop, and drops from **1,375 NLOC / CCN 368 to 957 / 253**
  (`lizard`, same file). `run_warm` shrinks too: its solver-statistics block was
  a byte-identical copy of `run()`'s and now calls the same helper, so the two
  cannot drift.

  **Extraction only — no behavior change, no API change, no performance change.**
  The stepping loop and the event / switch-crossing handlers are untouched (they
  carry the #82 threshold-landing fix and the #49 event-time jump), the
  warm/cold eligibility rule is unchanged, and every explanatory comment moved
  with the block it explains — the length of this function is largely accreted
  correctness, and losing that reasoning would cost more than the length saves.

  Verified as **byte identity**, not agreement to tolerance. A before/after A/B
  digests the full float64 bytes of every array a run produces — times, species,
  observables, expressions and all four sensitivity blocks — plus the
  solver-statistics dict, so a single changed bit in any column, or one extra
  CVODE step, moves the hash. Across four sweeps — the 585 `ode_fullnet` `.net`
  models on the forced-cold path, the same 585 on the default (warm-eligible)
  dispatch, the same 585 again under three-parameter forward sensitivities, and
  the 1,323-model `rr_parity` SBML corpus (events, assignment rules, `piecewise`
  discontinuity triggers, no-species algebraic models) — **3,025 runs hash
  identically and 50 refuse with identical messages; 0 divergences.** The three
  remaining models hit the harness's wall-clock cap in both arms. Byte-identical
  sensitivity columns make the FD-oracle agreement identical by construction,
  and the parity-suite verdicts likewise.

- **Steady-state `d(func)/dp` runs through the compiled output-sensitivity
  evaluator instead of finite differences (issue #75).**
  `steady_state(sensitivity_params=[…])` projects the species `dY_ss/dp` onto the
  model's observables and global functions. The observable half was always exact
  (a linear group map); the function half built *both* of its terms — the state
  chain `Σ_i (∂func/∂x_i)·dY_ss_i/dp` and the explicit `∂func/∂p` — from one-sided
  √eps difference quotients. It now calls `bngsim_codegen_output_sens`, the GH #198
  chain rule a CVODES forward-sensitivity `run()` already uses, handed the solved
  `dY_ss/dp` columns. A steady-state gradient and a converged-long-run gradient
  therefore come from the same evaluator rather than from two independent
  derivations, and `#63`'s "prefer closed form, record which ran" shape now covers
  all three factors of the output sensitivity instead of two.

  On `ss_expr_sens_derived.net`, whose `flux() = _rateLaw1·A_tot` with
  `_rateLaw1 = chi·kon` has a full closed form, max relative error against it goes
  **8.6e-8 → 1.1e-9 (75x)** — and the remaining 1.1e-9 is the steady-state solve's
  own tolerance, not the chain rule. The state-chain term also cost `n_species`
  full observable+function re-evaluations, which the compiled evaluator does in one
  call for all parameters at once.

  On the corpus, 290 of the 585 `ode_fullnet` `.net` models carry global
  functions; 138 reach a converged steady state under a 3-parameter probe. The
  compiled evaluator answers **every** function on 90 of them, **some** on 33
  more (`"mixed"` — a table function, a non-smooth builtin, an auto-`_rateLawN`
  outside the user closure) and none on 15 (a whole-model decline). Against the
  previous answers the median change is **5.6e-9** — FD noise — but **22 models
  move by more than 1e-6**, and adjudicating those row-by-row against an
  independent oracle (the last point of a converged CVODES forward-sensitivity
  run) the compiled path is closer on **82 rows**, tied on 44, and nominally
  behind on 6 — all 6 where the two paths agree with *each other* to 5+ digits
  and both differ from the oracle, i.e. the steady-state solve disagrees, not the
  output block (two of those models carry `sens_jacobian_rcond ≈ 5e-7`). The
  large movers are the FD path having been confidently wrong:

  | model | `d(function)/dp` | old (FD) | new (codegen) | CVODES-run oracle |
  |---|---|---|---|---|
  | `model1` | `d(controllability_PTEN)/dV` | `+4.94e10` | `-8.88e11` | `-8.88e11` |
  | `inhibitors_1` | `d(f_PIKK_ATP_formula)/dVecf` | `-5.20e8` | `0.0` | `0.0` |

  — a sign flip and an 18x magnitude error in the first, and a fabricated
  gradient on a parameter the function does not depend on in the second. On
  `inhibitors_1` the compiled block matches the oracle to exactly `0.0` on every
  row.

  Getting the symbol there was the actual work, and it was not "resolve one more
  symbol". It is emitted only when the model carries `_want_output_sens`, which
  `Simulator.__init__` sets from its **constructor** `sensitivity_params`, while
  `steady_state()` takes its own as a **method** argument — so the documented usage
  `Simulator(m, method="ode").steady_state(sensitivity_params=[…])` arrived at the
  solver with no such symbol, on a `.so` that otherwise reported
  `rhs_backend == "codegen-so"`. `steady_state()` now runs the same GH #205 re-prep
  `compute_all_sensitivities` does (set the flag, drop a plain-RHS artifact that
  would shadow the sensitivity one, regenerate, restore if regeneration yields
  nothing), extracted as one shared helper so the two entry points cannot drift.

  **Finite differences stay as the fallback, per function.** The compiled emitter
  writes a NaN sentinel for a function it declines (a table function, `min`/`max`/
  `abs`/`floor`, anything transitively depending on one) and leaves a function
  outside the user-selectable closure untouched; the buffer is pre-filled with NaN
  so both read as "no compiled answer" and route to the FD block, which is skipped
  entirely when every row was answered. A whole-model decline (no compiled
  artifact, `rateOf`, an embedded table-function wrapper) falls back wholesale, as
  before. New `ss.sens_output_source` reports `"codegen"`, `"mixed"`, or
  `"finite-difference"`; the observable block is exact either way and is not
  covered by it.

  One cost worth naming: a `steady_state(sensitivity_params=…)` call on a model
  with global functions now pays the GH #198 build-time derivation, which it did
  not before. That is the same derivation `compute_all_sensitivities` and a
  `sensitivity_params`-built `Simulator` already pay, including its known
  unbounded cases (issues #97 and #99); the `.so` cache makes repeats — a fitting
  loop — free.

- **An assignment retires the parameter-graph IC sensitivity row it superseded
  (issue #113).** `∂(IC)/∂p` (issue #43) differentiates the initial condition the
  model *declares* — `species[].initial_conc`, what `reset()` returns to. A
  `set_concentration` replaces that initial condition with a literal, but the seed
  kept being derived from the `.net` expression, so the engine reported a gradient
  through an initial condition the model no longer had. On `ic_direct.net`
  (`R() R0`, with `R0` in **no rate law**) pinning `R` to 7.0 reported
  `dR/dR0 = 1.0, 0.779, 0.607, …` — the seed propagated as `e^{−kf t}` — where the
  truth and a rebuild finite difference are both exactly `0`. Not approximate: a
  gradient with respect to a quantity the trajectory does not depend on, with no
  warning.

  The engine can tell, and now does: an expression-derived row applies only while
  the species is still *at* that baseline, so a row whose live concentration has
  moved off `initial_conc` is dropped. A `Model.declare_ic_sensitivity` row
  (issue #111) is the more specific statement and still wins — including a
  deliberate nonzero one for an assignment computed from a fitted parameter. The
  C++ fallback identity loop (used when Python injects nothing) applies the same
  rule, so both seeding paths agree.

  Scope of the behaviour change: nothing happens unless a caller has assigned
  concentrations before a sensitivity run. Every model in the 585-model `.net`
  corpus and 120 SBML event models loads with its live state exactly equal to its
  baseline, so no freshly loaded model is affected, and `run_batch` /
  `steady_state_batch` (which clone then `reset()`) are unaffected by construction.
  A caller who re-asserts a species' *own* IC value keeps its row — the assignment
  and the expression then agree numerically and which was meant is genuinely
  ambiguous; declare to settle it either way.

  `NetworkModel.get_initial_state()` exposes the baseline (bulk, ordered like
  `species_names()`), the counterpart of `get_state()` this comparison needs.

- **Forward sensitivities now use the analytic RHS for Functional rate laws whose
  expressions are smooth algebra (issue #67, closing stage 3 of #55).** A single
  Functional reaction used to put a whole model on CVODES' internal difference
  quotient, because `CVodeSensInit1` installs one callback for every sensitivity
  column and `generate_sens_rhs_c` declined on the first non-Elementary rate law.
  On the 585-model corpus that was **187 models (32%)**, blocked by **1.1% of
  reactions**. #66 supplied the analytic `∂f/∂p` behind a keyword; this change
  supplies the other half of `ySdot = J·yS + ∂f/∂p` and turns the keyword on.

  `J·yS` is not a second derivation. It is the *same* per-species chain rule and
  per-observable product rule `generate_jacobian_from_model` already emits and
  validates; both callers now share one reconstruction
  (`_functional_jacobian_groups`) and differ only in where a contribution lands.
  The sensitivity RHS passes a scatter that fuses the matvec —
  `Jv_out[i] += coeff·dj·v[j]` instead of `jac[j*n+i] += coeff·dj` — so no `n×n`
  scratch buffer is formed inside the CVODES callback (267 KiB on the widest
  Functional corpus model, memset once per column per step), the work is O(nnz)
  rather than O(n²), and `CodegenSensUserData` does not have to widen. The
  analytical Jacobian's emitted C is byte-identical on all 585 models.

  Corpus result: **535/585 models get the analytic sensitivity RHS, up from 398**
  (+137), 0 models lost, and all 137 gained sources compile and export
  `bngsim_codegen_sens_rhs`. The 50 that still decline are Michaelis–Menten and
  the Functional laws carrying a condition (`if()`, a comparison, a logical —
  issue #68 owns those, and lifting them needs the switch-time guard, because
  sympy differentiates a `Piecewise` w.r.t. a condition-only parameter to a clean
  `0` and drops the jump) or a non-smooth builtin
  (`abs`/`min`/`max`/`floor`/`ceil`/`round`). Every decline warns and names what
  blocked it, so none of them is the quiet one.

  Sampled end-to-end on 46 gained models (every one above 100 reactions, plus a
  random 24): **46/46 agree** with the difference-quotient path, median relative
  disagreement 6.1e-08, worst 6.0e-05. Cost per analytic sensitivity column,
  measured warm in plain-solve equivalents at 16 parameters: **0.59–1.04**,
  against the difference quotient's **9.0–36.7**. The difference quotient needed
  a median **295×** more steps and up to 177 060×, and on 2 of the 6 benchmarked
  models it never converged at all — it exhausted `mxstep` and returned a
  gradient of exactly zero, which reads downstream as a converged answer. Set
  `BNGSIM_NO_FUNCTIONAL_SENS_RHS=1` to force the previous behaviour for an A/B
  (it is part of the .net codegen cache key, so a hatched run cannot collide with
  a .so compiled without it).

  Two downstream consumers pick this up without asking. `steady_state.cpp` reads
  the same `bngsim_codegen_sens_rhs` at `yS = 0` for the `∂f/∂p` factor of
  `dY_ss/dp = -J⁻¹·(∂f/∂p)`, so `ss.sens_dfdp_source` on a Functional model moves
  from `"finite-difference"` to `"codegen"` — retiring the √eps step floor issue
  #76 describes for those models — and the `.net`-loaded path reaches the
  model-based emitter through `generate_combined_c`, which is how a corpus model
  gets here at all.

- **The Curtis-Powell-Reid Jacobian coloring is computed on first use instead of
  at model load, and no longer skipped on dense patterns (issue #29).** `build()`
  used to color only when Jacobian density was `< 0.5`, on the reasoning that
  coloring a near-dense matrix saves nothing. That held while sparse KLU was
  reachable only through the auto rule, which requires density `< 0.10` and so
  always had a coloring to hand. `force_sparse_linear_solver` broke the
  invariant: a model at density `>= 0.5` with no complete analytical Jacobian had
  nothing to fill the CSC matrix KLU factorizes, and CVODE has no
  difference-quotient fallback for `SUNMATRIX_SPARSE`, so the run was refused. On
  the 585-model corpus that hit 7 models — all `N <= 5` with functional rate laws,
  which conservatively mark every species as a Jacobian dependency and so land at
  density 0.52–1.00.

  The density ceiling is gone: every pattern with a structural nonzero is now
  colored, however dense. A fully dense one degenerates to one column per color,
  i.e. colored FD becomes plain FD — correct, just not a speedup, which is the
  right answer for a flag whose purpose is measuring KLU's overhead on small
  dense models. Forced-sparse now reaches KLU on 585/585 rather than 578, and
  each of the 7 matches its forced-dense trajectory to within 5e-9 relative.

  Removing the ceiling would otherwise have charged every model load for a flag
  almost nobody sets, so the coloring moved off the build path entirely:
  `NetworkModel::ensure_jacobian_coloring()` materializes it once on first use,
  thread-safe and shared across clones, following `ensure_conservation_laws()`
  (#102). Only the sparse colored-FD Jacobian callback consumes it, so models
  that never take that path — including every dense-solver run — no longer pay
  for a coloring at all. Auto-selection is unchanged: the same corpus still
  splits 541 dense / 44 KLU.

- **`Simulator.steady_state()` and `steady_state_batch()` now default to
  `method="integration"` instead of `method="newton"` (issue #28).** Once #27
  reordered the two-tier solver to integrate *first*, its KINSOL polish stopped
  being an alternative to the integration path and became extra work layered on
  top of it — so wherever the burst already reaches the parity criterion, the
  polish is time spent buying a tighter root than `tol` asked for. The
  `steady_state` suite measures that cost across six published dose-response
  models (20 doses each), and `newton` loses on every one:

  | Model | species | `newton` | `integration` | ratio |
  |-------|--------:|---------:|--------------:|------:|
  | kinetic_proofreading | 9 | 19.00 ms | 5.05 ms | 3.8× |
  | genetic_switch | 2 | 10.22 ms | 3.14 ms | 3.3× |
  | lac_operon | 3 | 11.17 ms | 3.20 ms | 3.5× |
  | Kocieniewski_2012 | 85 | 183.18 ms | 108.25 ms | 1.7× |
  | Barua_2007 | 149 | 269.73 ms | 194.31 ms | 1.4× |
  | Barua_2013 | 409 | 19467 ms | 8929 ms | 2.2× |

  Geometric mean 2.5×. Both methods return the same steady state on all six
  (the suite's correctness gate cross-checks every engine against the physical
  integration reference at `rtol=1e-2`), so this trades no accuracy for the
  speedup. A wider sweep over the 41 vendored ODE models under 600 species
  (`benchmarks/models/net/ode/`, excluding known non-equilibrating systems)
  reproduced the gap — geometric mean 2.7×, `integration` ahead on 34 of 41 —
  and turned up no model at the default `max_time=1e6` where `integration`
  fails to converge but `newton` succeeds.

  `method="newton"` / `"kinsol"` remain available unchanged, and are still
  worth requesting when either applies: the polish resolves the root to a
  residual around `1e-13` where integration stops the moment it crosses `tol`
  (~`1e-9`); and because Newton reaches `tol` from a looser burst than
  integration needs alone, it can still converge under a `max_time` cut well
  below the default (no corpus model shows this at `1e6`; several do at `1e3`).

  **Migration:** pass `method="newton"` explicitly to keep the old behavior.
  Code that asserts on `ss.method_used` should expect `"integration"`.

- **`method="newton"` no longer rebuilds its integrator once per ladder rung,
  nor re-probes a KINSOL solve that cannot be factored (issue #28).** The
  two-tier solver's header has always claimed its burst ladder is "a single
  march to the tightest rung, not a restart per rung", but only the *state*
  carried across rungs: each rung called `solve_by_integration`, which built a
  fresh `SUNContext`, CVODE memory, dense `SUNMatrix` and linear solver, and
  restarted BDF at order 1 with a fresh initial step — discarding the step-size
  and order history the previous rung had paid to build up. The CVODE session
  now lives in a `SteadyStateMarcher` held across the whole ladder, and a rung
  only tightens the stop criterion, so the claim is true of the integrator as
  well as the state. Each march still gets `max_time` of *additional* simulated
  time, preserving the per-rung budget of the restart-per-rung code.

  Separately, a model whose reduced Jacobian is structurally singular failed
  every one of the `MAX_NEWTON_ATTEMPTS = 6` KINSOL probes at ~equal cost, since
  singularity is a property of the sparsity pattern and not of the seed.
  `solve_by_newton` now reports an unrecoverable linear-solver failure
  (`KIN_LSETUP_FAIL` and siblings) distinctly from ordinary non-convergence, and
  the ladder stops probing after the first one. Barua 2013 (409 species, GH #27
  Bug 3) returns `KIN_LSETUP_FAIL` on attempt 1 and now builds one doomed
  404×404 factorization instead of six.

  Re-measuring the six-model suite above, against the same `integration`
  baseline:

  | Model | species | `newton` before | `newton` after | vs `integration` |
  |-------|--------:|----------------:|---------------:|-----------------:|
  | kinetic_proofreading | 9 | 19.00 ms | 16.87 ms | 3.8× → 3.1× |
  | genetic_switch | 2 | 10.22 ms | 9.10 ms | 3.3× → 2.9× |
  | lac_operon | 3 | 11.17 ms | 12.10 ms | 3.5× → 3.6× |
  | Kocieniewski_2012 | 85 | 183.18 ms | 136.10 ms | 1.7× → 1.3× |
  | Barua_2007 | 149 | 269.73 ms | 231.15 ms | 1.4× → 1.2× |
  | Barua_2013 | 409 | 19467 ms | 17917 ms | 2.2× → 2.0× |

  Geometric mean of the gap to `integration` narrows from 2.5× to 2.2×. The
  three larger models carry the signal and reproduce across repeat runs
  (Kocieniewski_2012 136–143 ms, Barua_2007 227–231 ms, with the `integration`
  control stable to within 3%); at 2–9 species there is almost no setup cost to
  hoist and the per-dose work is small enough that run-to-run spread on the
  measuring host exceeds the effect, which is why `lac_operon` reads as a slight
  loss. `integration` remains the default and remains faster on every model —
  this narrows the gap for callers who opt into `newton` for its ~`1e-13`
  residual, it does not close it.

- **Re-pinned PyBioNetGen for the parity/benchmark suite, `5109a46` →
  `43b09a5` (issue #4).** The old pin carried RuleWorld/PyBioNetGen#109: under
  `simulator='bngsim'`, a model whose actions PyBioNetGen's Python parser could
  not inspect was routed **silently** to the legacy BNG2.pl subprocess instead
  of raising — so a caller who explicitly demanded bngsim could receive legacy
  output with no error and no warning. RuleWorld/PyBioNetGen#111 fixed it
  (strict `simulator='bngsim'` now returns `ROUTE_ERROR` carrying the underlying
  parse reason; `simulator='auto'` keeps the subprocess fallback and adds a
  one-time warning) and also made the list-arg grammar Perl-faithful (#110).
  `43b09a5` is RuleWorld/PyBioNetGen@main at that fix plus the merges after it,
  with upstream CI green. The fix is verified live rather than assumed from the
  merge: on issue #109's own repro (a `simulate` carrying a typo'd `atoll`
  argument, which `bngmodel()` rejects and BNG2.pl tolerates) the bootstrapped
  env routes `simulator='bngsim'` to `error` and `simulator='auto'` to
  `subprocess`. Note this could never have altered the suite's *numbers*: since
  GH #175 the sweep drives bngsim in-process via `run_bngsim_job` rather than
  through `bionetgen.run(simulator='bngsim')`, so the bridge's routing pass is
  provenance, not the execution path.

- **Regenerated the `bng_parity` golden references on the new pin** (all 895
  manifest jobs `ok`; `_meta.bionetgen_commit` `5109a46e58ec` →
  `43b09a534640`). 885 of 895 records reproduce the previous golden's checksum
  **byte-for-byte** across a bngsim 0.9.59 → 0.11.35 gap, which is what makes
  the 10 that moved readable: **6 are exactly the `CURATED_SIX`**
  (`Lang_2024`, `Kocieniewski_2012`, `Barua_2007`, `Blinov_2006`,
  `Barua_2013__PATCHED`, `fceri_fyn`) — re-sourced to house-curated, bug-fixed
  bodies after the golden was last generated, so those records had gone stale
  and now fingerprint different observables and horizons (`fceri_fyn` also moves
  off the D1 `.cdat` fallback, since its re-sourced body has observables). Of
  the remaining four, `ml_q_learning` is ODE drift at `max_rel` 1.0e-06 and the
  three Lin2019 stochastic models (`ERK_model`, `prion_model`, `TCR_model`) are
  single fixed-`seed=1` trajectories, which are chaotic and not stable across a
  bngsim version change by construction (per the golden contract's D2, the byte
  checksum — not the fingerprint — is their meaningful check).

- **`bootstrap_parity_env.py` now builds its venv on the interpreter that
  matches the bngsim wheel.** bngsim ships as an ABI-tagged wheel, so a venv on
  whatever `uv venv` picked by default failed at the *last* step — after the
  whole PyBioNetGen build — with "no wheels with a matching Python version tag".
  The venv now defaults to the version of the interpreter running the script
  (the one `--build-bngsim` targets), with `--python` as an explicit override.
  Re-running the bootstrap is also idempotent again (`uv venv --allow-existing`);
  current uv aborts on an existing venv rather than reusing it, which made a
  re-bootstrap after a pin bump fail on step one.

### Fixed
- **A pre-init `set_param` on the RuleMonkey backend is a parameter update
  again, not a model rebuild (issue #115).** `RuleMonkeySimulator` answered every
  pending override by re-evaluating the XML's whole `<Parameter expr=>` graph,
  writing two temp XMLs, and reconstructing the upstream engine from the second
  one — on *every* `run()` and every session `initialize()`. That existed to work
  around an upstream defect (GH #44): RuleMonkey parsed only the collapsed
  `<Parameter value=>`, so its own override cascade re-resolved a number to
  itself and could not reach a derived seed amount such as
  `LT = ((dose_nM*1e-9)*NA)*V_sim`. RuleMonkey **v3.7.0**
  ([`0ec6148`](https://github.com/richardposner/RuleMonkey/commit/0ec6148)) fixed
  that — the cascade re-derives from `expr=`, gated on the value actually moving
  — so the adapter now forwards `set_param` and lets the engine propagate. The
  vendored pin moves `fbdde54` → `8f87968` (v3.7.0 plus its docs follow-up).

  Two behavior bugs go with it, both consequences of maintaining a parameter
  evaluator in parallel with the engine's:

  * **An override no longer perturbs anything outside its dependency cone.**
    The re-bake re-rounded *every* derived parameter to `expr=` precision
    whenever any override was pending. Where BNG2 writes `NA` as
    `value="6.0221408e+23"` against `expr="6.02214076e23"`, that moved every
    bimolecular rate constant dividing by Avogadro's number by ~4e-9 relative —
    at every scan point, for parameters the scan never touched. A no-op
    `set_param(p, p)` was enough to trigger it.
  * **`clear_param_overrides()` after a run takes effect.** The rebuild ran only
    while an override was *pending*, so once a run had baked a dose into the
    model, clearing it left the baked seed amount and rate constants in place and
    the next run silently kept using the cleared dose. (Upstream's companion fix
    closes the same hole on its side.)

  The GH #51 half-up seed-count policy is unchanged in effect and re-expressed in
  mechanism: instead of rewriting `<Species concentration=>` in a temp XML, the
  adapter reads the amount the engine has already resolved
  (`initial_species()`) and pins the rounded integer (`set_initial_amount()`),
  re-derived after every `set_param` / `clear_param_overrides`. That also removes
  the overlay's blind spot — it could only resolve a bare parameter or a literal,
  where the engine resolves whatever the model wrote.

  Per-scan-point cost, median of 12 points, zero-horizon `run()` so the setup is
  not hidden behind SSA work (RuleMonkey's own model corpus):

  | model | XML lines | before | after | no-override control |
  |---|---|---|---|---|
  | `r21` (ribosome) | 51,919 | 29.8 ms | **0.41 ms** | 0.37 ms |
  | `tcr` | 10,457 | 121.7 ms | **113.5 ms** | 112.9 ms |
  | `ensemble` | 15,133 | 249.8 ms | **242.7 ms** | 243.0 ms |
  | `egfr_net` | 2,996 | 24.6 ms | **20.7 ms** | 20.7 ms |
  | `blbr_posner2004` | 2,085 | 10.1 ms | **7.0 ms** | 6.9 ms |

  An overridden point now costs what an un-overridden one costs, which is the
  actual claim; the spread across models is just how much of a point is XML
  parsing versus instantiating the pool. Construction gets cheaper too (`r21`
  23.9 → 19.4 ms) since the cold path no longer writes a seed-rounded temp XML.

  Found on the way and fixed with it: **no workflow was triggered by a change to
  the RuleMonkey adapter or its vendored tree.** `native-tests.yml` builds
  `-DBNGSIM_BUILD_RULEMONKEY=OFF`, so `src/rulemonkey_simulator.cpp` is not in
  that build; the jobs that do compile it only did so because
  `BNGSIM_BUILD_RULEMONKEY` defaults ON, and none listed it in `paths:`. A
  vendor bump could have failed to compile with nothing red.
  `windows-tail.yml` — which already builds RuleMonkey and already runs
  `test_seed_count_rounding.py`'s RuleMonkey class — now triggers on
  `src/rulemonkey_simulator.cpp` and `third_party/rulemonkey/**` and runs
  `test_rulemonkey.py`, at no extra build cost. `scripts/vendor_rulemonkey.py`'s
  clean-destination guard is fixed alongside: it ran `git status` from the
  *parent* of the checkout, which exits 128 in a standalone clone, so the guard
  aborted the refresh instead of checking anything.

- **The derived-parameter chain rule walks the parameter DAG instead of
  flattening it before differentiating (issue #99).** Issue #41 taught
  `_compute_derived_param_jacobian` to reach a *nested* derived
  (ConstantExpression) parameter by substituting it — and its whole dependency
  graph — textually before handing one expression to sympy. That substitution is
  exponential in the depth of the graph. On `ode/synthesis_v3` — **five
  species**, 28 derived parameters — a 43-character parameter flattens to 20 KB
  and its dependent to 40 KB, and because the nesting lands in an *exponent*
  (`n = ln(...)/ln(ratio)`, `Fh = (…^n…)^(1/n)`) the `sp.diff` on the result
  never returns. It was the one model in the 585-model `.net` corpus whose
  `_analyze_output_sens` never completed, and #97's budget could not save it:
  the cost is a **single uninterruptible sympy call**, and a wall-clock deadline
  can only be checked *between* them.

  The chain rule now composes over the graph —
  `∂p_d/∂θ = (∂p_d/∂θ)_direct + Σ_k (∂p_d/∂s_k)·(∂s_k/∂θ)` — differentiating each
  parameter as written and memoising per node, so a diamond is derived once and
  a cycle in an ill-formed `.net` is reported rather than recursed into.
  `∂p_d/∂s_k` prints the nested parameter as its `p[idx]` slot, which the runtime
  already holds. This is the shape `_functional_dfdp_terms` has always used for a
  rate law naming a derived parameter; this was the one place that still
  flattened. `synthesis_v3` now derives in **1.2 s**, and `∂n/∂primary` emits
  12 KB of C where the flattening emitted 202 KB.

  **The same flattening was on the initial-condition path, so this was never
  only a codegen-build problem.** Species `F` starts at `F0`, the 40 KB one, so
  *every* parameter-sensitivity run on that model hung in
  `compute_ic_param_sens_seed` before any C was generated. Both halves — the one
  that emits C and the one that substitutes values — now walk the same DAG
  through the same shared parse, which is what keeps them from drifting apart
  again.

  Two consequences beyond the model that motivated it. Nested `min()`/`max()`
  flattened into one expression produced a derivative `sp.ccode` refused with
  *"Invalid NaN comparison"*, losing the entire chain rule for `phi`, `alpha` and
  `gamma` on three corpus models — which under #56's rule declines the analytic
  sensitivity RHS. Walked, each level prints on its own: those three models now
  emit, and nine functions across three more models gain output sensitivities
  they were refused. (The `min`/`max` derivative itself was never the silent-zero
  class the issue suspected: `Max` is *continuous*, so unlike an `if()` there is
  no jump at the kink and no delta term to lose.) And #97's derivation-step count
  — one per expression parsed, one per symbol it names — is now the work the
  phase actually does rather than a number with no fixed relation to it.

  Validated over the 585-model `.net` corpus: 226,408 `(derived parameter,
  primary)` pairs finite-differenced against `set_param`-perturbed parameter
  values, no disagreement outside the FD noise floor; 13,451 partials evaluated
  against the pre-fix arm in the same process, agreeing to roundoff with **no
  chain-rule term lost anywhere**; 378 of 584 models byte-identical, every
  emitter change in the declines-to-emits direction, and total emitted C +2.6%.
  End to end on `model_step1_v1`, whose `fa__FREE` and `fr__FREE` reach the
  dynamics only through the chain that could not be derived, the newly analytic
  sensitivities match a re-solved-trajectory finite difference to 1.4e-6 and
  5.8e-7 relative.

- **Concurrent C expression emission now keeps symbol resolvers isolated per thread
  (issue #117).** The cached SymPy C printer is now thread-local, preventing one
  emission from replacing or clearing another emission's resolver and causing a
  supported derivative to be reported as unavailable.

- **`set_param` did not rebuild a species initial condition that references the
  parameter (issue #79).** `A() Stot` in a `.net` species block — or an SBML
  `initialAssignment` that is a bare `<ci>` — declares that species' initial
  condition to *be* the parameter. `set_param("Stot", 1e6)` moved the parameter
  and nothing else: `get_state()` returned the network-generation amount,
  `reset()` restored it, and `get_param` confirmed the write, so **a dose scan
  over a total amount silently ran every dose at the same initial condition**
  with no error and no warning. **406 of the 585 `ode_fullnet` corpus models
  (1,725 species) have such an initial condition**; before the fix `set_param`
  moved none of them, after it, all of them. It was also a cross-engine trap —
  AMICI reads the same system through SBML `initialAssignment` and recomputes
  `x0`, so a comparison scanning such a parameter disagreed for a reason that
  had nothing to do with the solver under test.

  The dependency and the machinery were both already there — `species_ic_param_refs`
  names the (species, parameter) pair — but nothing consulted them after load.
  The issue proposed routing the invalidation through `ic_state_dirty`; that flag
  is the GH #210 pre-equilibration carry-over marker ("this state is advanced
  dynamics, not an initial condition"), and setting it from `set_param` would have
  made the next forward-sensitivity run *raise* instead. `set_param` now
  re-resolves the initial conditions directly, after the derived-parameter
  re-evaluation so a species IC named by a `ConstantExpression` (`R() Rtot` with
  `Rtot = 0.5*R0`) picks up its new value too, and over every ref rather than only
  the refs naming the parameter just written.

  Two fields, two rules. The **declared** initial condition (`initial_conc`)
  always follows the parameter — which is what makes `reset()` rebuild from
  current parameter values rather than a load-time snapshot, and what makes the
  `set_params(); reset()` sequence inside `run_batch()` and
  `steady_state_batch()` correct without their knowing anything about IC
  parameters. The **live** concentration follows only while the species is still
  sitting on that baseline, so a species the dynamics advanced (or a caller
  assigned) keeps its value and picks the new IC up at the next `reset()` —
  the same `concentration == initial_conc` test the issue #113 sensitivity
  seeding uses, so the two agree. `save_concentrations()` (unlabeled) redefines
  the baseline to a captured state that the declared IC no longer describes, and
  latches the new read-only `NetworkModel.ic_baseline_saved`, retiring the
  rebuild rather than discarding a pre-equilibration; dose such a protocol with
  `set_concentration`. `clone()` carries the flag.

  `Simulator.parameter_scan` needed one more turn of the crank: it restores each
  point's live concentrations from the invocation snapshot, but the IC baseline
  is model state too, so from point 1 on the species sat off a baseline the
  *previous* point had moved and the scan would have applied only its first dose.
  It now rewinds the scanned parameter before restoring the state.

  Fixed alongside it, because it is the same three lines and the re-resolve had
  to pick a rule: the load-time resolve wrote the raw parameter value over the
  `amount / V` the SBML loader had computed for a
  `hasOnlySubstanceUnits="true"` species, so such a species loaded **V times too
  large** when its amount came from a single-`<ci>` `initialAssignment` — while
  an identical model spelling the same amount as `initialAmount=` loaded
  correctly. Both sites now share one conversion (`resolve_ic_from_param`). This
  is the identity for every `.net` model and every `hOSU=false` species; a sweep
  of all 5,482 SBML files in the repo finds **one** affected model,
  `MODEL2002070001`, whose compartment sizes are `NaN` and which is already
  recorded as a BAD_TEST in the RoadRunner parity overrides.

  Regression coverage in `python/tests/test_set_param_ic_rebuild.py` (25 cases,
  20 of which fail without the fix) and one clone-contract case in
  `test_model_clone.py`. The 585-model `.net` corpus loads unchanged.

- **The docs said the KINSOL polish uses an analytical Jacobian. It never has.**
  `solve_by_newton` does not call `KINSetJacFn`, so KINSOL installs its own
  difference-quotient Jacobian and each setup costs **one RHS evaluation per
  unknown** — measured directly: seeded at its steady state, `SHP2_base_model`
  (147 unknowns) runs one polish that KINSOL reports as 1 iteration / 2 function
  evaluations while the model sees **151**. Corrected in the `steady_state()`
  docstring, the steady-state guide, `steady_state.hpp`, `SteadyStateOptions`,
  and the stale comment in `SteadyStateMarcher`'s constructor that claimed the
  march skipped the closed form because "the analytical Jacobian is used by
  KINSOL (Tier 2) instead" (the march does not install one either).
  `jacobian=` reaches the consumers that need the matrix *itself* — `dY_ss/dp`
  and the issue #78 stability certificate — not either solver tier, so it cannot
  move the polished root. Both facts are now pinned by
  `python/tests/test_steady_state_polish_jacobian.py`, which fails if a Jacobian
  function is installed — the cue to update the prose along with it. Installing
  one is a real per-setup saving on codegen-backed models (the artifact already
  carries `bngsim_codegen_jac`) but changes how both tiers converge, so it wants
  its own before/after rather than a drive-by.

- **`steady_state(method="newton")` no longer returns the saddle on a bistable
  model (issue #78).** On the Gardner 2000 toggle at `alpha_2 = 53.526315789`,
  one dose of the model's own 20-point scan, it returned `[28.245, 1.830]` with
  `converged=True` and a residual of `2.82e-10`. That state is a genuine root of
  `f(y) = 0` — and the **saddle** between the two branches, an equilibrium no
  trajectory can rest on: perturb it by one part in `1e6` and it runs away to a
  *different* attractor depending on the sign. Its Jacobian's eigenvalues are
  `+0.40643` and `-2.40643`. `method="integration"` was right at that dose and at
  the other 19.

  The seed-stability guard could not catch it, and the reason is structural
  rather than a tuning problem: it asks whether *refining the seed* moves the
  root, which is not the question. Near a separatrix the trajectory slows to a
  crawl — this one comes within 3% of the saddle by `t = 2` and stays within 10%
  for about 4.5 time units — so two successively tighter bursts hand KINSOL
  near-identical seeds, both polish to the saddle, and they agree. The guard is
  satisfied precisely where it is needed.

  A polished root is now certified before it is accepted: the eigenvalues of the
  Jacobian **restricted to the species the polish solved for** (conservation-law
  independents, narrowed by any `mask=`, minus `$`-fixed species — the same
  unknown set KINSOL and `dY_ss/dp` use, now a shared helper rather than a third
  copy) must all lie in the closed left half-plane. A root with
  `max Re(λ) > 1e-6·max|λ|` is discarded and the ladder keeps integrating.
  Rejecting the saddle costs nothing: the burst leaves its neighborhood on its
  own and a later rung polishes the attractor, so that dose now returns the
  correct branch at a residual of **7.5e-14** against integration's `3.2e-10`,
  still reporting `method_used="newton"`.

  Two new `SteadyStateResult` fields make the verdict readable:
  `ss.root_stability` is `"stable"` / `"undetermined"` / `"unstable"` for a
  Newton root and `""` for the integration path (a trajectory cannot come to rest
  on an unstable equilibrium, so there is nothing to certify);
  `ss.n_unstable_roots_rejected` counts the discards, which is what explains a
  `method_used` of `"integration"` from a `method="newton"` call. `"unstable"` is
  returned only when the *caller's own initial condition* was already that root —
  integration would return the same state, so the verdict is reported rather than
  acted on, which is an answer the library could not previously give at all.

  The spectrum comes from a self-contained balance → Householder-Hessenberg →
  Francis-QR eigensolver (`include/bngsim/dense_eigenvalues.hpp`), not LAPACK:
  `dgeev` is only linked when CMake finds a BLAS backend, and a guard that
  silently does not run on one platform is worse than no guard. Cross-checked
  against LAPACK on ~1,000 corpus Jacobians, its `max Re(λ)` agrees to **1.1e-8**
  of the spectral radius in the worst case (a defective eigenvalue, where
  `eps^(1/m)` accuracy is all any method has) — a hundredfold under the `1e-6`
  threshold, which itself sits five decades under the saddle's `+0.169`.

  Measured over the `ode_fullnet` corpus — 449 models solved with
  `method="newton"` both before and after (404 at the default horizon, 45 of the
  slow tail at `max_time=1e4`), and 570 of the 585 after; the remainder are
  oscillators and accumulators that do not settle within any horizon the sweep
  could afford. **Exactly one model changed**, and it changed for the same reason
  the Gardner toggle did. `ml/ml_q_learning` — a Q-learning model
  whose rate laws are a stack of `if()` conditions — used to return
  `converged=True` at `[10.0, 1.803e7, 1.803e7, 1.803e6, 50]` with a residual of
  **exactly 0.0**: a real root, on which the Jacobian has an eigenvalue at
  `+0.19·max|λ|` and from which a **one-part-in-1e6** nudge of either Q value
  leaves by **19× the root's own magnitude within `t` = 500** (the perturb-and-
  integrate test, run as an independent oracle). It now reports `converged=False`,
  which is the honest answer for a model whose Q values accumulate without bound.
  No other model moved: `method_used`, `converged` and the residual are unchanged
  on the other 448 A/B'd models, and no root the solver accepts has
  `max Re(λ)/max|λ|` above `1.7e-16` (the zero eigenvalues of a conserved system,
  at roundoff).

  The certificate costs one O(n³) eigen-decomposition per accepted root — 0.05 s
  at 256 unknowns, 0.26 s at 624 — and **declines above 512 unknowns**, reporting
  `"undetermined"` and accepting the root as before, because past that the
  spectrum costs more than the solve it is checking (2.1 s against a 15.2 s solve
  at 1281 species). Nine corpus models report `"undetermined"`: eight above the
  size limit and one whose reduced Jacobian is identically zero.

  Two cheaper tests were measured and rejected. A **Cayley-transform power
  iteration** (`(I − sJ)⁻¹(I + sJ)` has spectral radius > 1 exactly when
  `Re λ > 0`, for any `s > 0`) needs one LU instead of a full spectrum, but a
  non-normal Jacobian's transient growth is indistinguishable from an unstable
  mode in a finite number of iterations: it called **57 of 250** corpus roots
  unstable, one at a growth factor of 1.028. The **determinant-parity screen**
  (`sign(det) ≠ (−1)ⁿ` proves an odd number of eigenvalues with positive real
  part, at `n³/3`) fails for the opposite reason — on the reduced Jacobians where
  it would be the only available screen it flagged **6** corpus models whose true
  `max Re(λ)/max|λ|` is `1e-16` or smaller, because `|det|` there is ~`1e-21` and
  the sign is roundoff. An LU cannot tell that case apart, which is the same thing
  `sens_jacobian_rcond` was measured to be unable to do in #63.

- **A steady-state `∂f/∂p` component whose response is roundoff takes the wide
  probe instead of fabricating a gradient (issue #123).** #76 made the
  finite-difference parameter probe relative to the parameter, which is right
  where the old absolute floor was wrong and wrong where it was right: when a
  parameter's own term is a small fraction of the derivative it sits in,
  `eps·|p|` moves that derivative by **less than its own roundoff**, and the
  difference quotient is noise — often an exact zero, which a fitter reads as
  "this parameter does not matter". `tests/data/cancelled_parameter_term.net` is
  the minimal case: `dA/dt = ksyn + ktrace − kdeg·A` with `ksyn` = 100 and
  `ktrace` = 1e-9 has a closed-form `dA*/dktrace = 1/kdeg = 1`, and the relative
  step alone returns **exactly 0.0**.

  Each parameter probe now takes **two** steps — the relative one and the
  pre-#76 absolute one — and every component keeps the quotient that carried a
  response:

  * a component's response counts as signal when it clears that component's own
    roundoff floor by 100x. The floor is `uround · max(|f_i|, Σ_j |J_ij·y_j|)`:
    `|f_i|` alone is the wrong scale at a steady state, where `f_i` is a
    cancellation of large rate terms and its roundoff is set by the **terms**,
    not by the near-zero sum. The Jacobian row sum recovers that term scale (a
    rate term of degree *d* in the species contributes *d* times the term) and
    `J` is already assembled one step earlier, so it costs an O(n²) pass over
    memory already in hand. The function half of `compute_ss_output_sensitivity`
    gets the same treatment from `Σ_i |∂func_m/∂x_i · x_i|`, which its
    state-chain sweep already computes.
  * everything else keeps the relative step, so #76's fixes stand: on the
    585-model corpus **every** `|p| >= 1` column is bit-identical to #76 (there
    is no second step to choose there) and so are 92% of the rest.

  A first attempt used the response to a deliberately tiny (4-ulp) probe as the
  noise floor, which is the obvious estimator and **fails in exactly the case
  that matters**: when the probe moves `f_i` by less than one ulp the measured
  "noise" is exactly zero, no response can fail to clear it, and nothing ever
  widens. It recovered 18 of 159 mis-stepped corpus columns where the term-scale
  floor recovers 121. A geometric ladder of intermediate steps recovered **0** —
  clearing a noise floor is not the same as being accurate at that width.

  Measured end-to-end over the 585-model corpus — every model solved twice at
  the same root, once with the compiled `∂f/∂p` and once forced onto the
  fallback — on the 850 columns where the step rule can change anything
  (`|p| < 1`, and a sensitivity system that is not itself degenerate):

  | | pre-#76 | #76 | now |
  |---|---|---|---|
  | columns wrong by > 1e-3 | 103 | 57 | **52** |
  | columns wrong by > 1e-1 | 70 | 42 | **41** |

  Five columns cross 1e-3 the right way against #76 and **none** the wrong way,
  and against pre-#76 the score is 51 fixed / **0 broken** — the five regressions
  #76 disclosed are gone. 759 of the 850 are bit-identical to #76, as are all 278
  control columns at `|p| >= 1`. The choice of 100x is flat across two decades
  either side; the PR has the sensitivity table.

- **The steady-state finite-difference probes are relative to what they perturb
  (issue #76).** `dY_ss/dp = -J⁻¹·(∂f/∂p)` differences either factor when no
  closed form is available, and both probes sized their step as
  `eps·max(|x|, 1)` with `eps = sqrt(machine eps)` = 1.49e-8. The floor is there
  to survive `x == 0`, but it also overrides the relative step for everything
  smaller than 1, and then the probe is no longer small compared with what it
  perturbs. On `ode/before_bunching`, whose `KD` is 1e-9 and whose forward rate
  constant is derived as `kon/KD`, the probe was **1500% of the parameter** — it
  drags `kf` from 1.0 to 0.063, so the difference quotient is a secant across a
  decade and a half of the rate law's curvature rather than a derivative:

  | `max abs dY_ss/dKD` on `before_bunching` | |
  |---|---|
  | formula, shipped step | 6.754783e+10 |
  | formula, relative step | 1.074089e+12 |
  | truth, re-solved central difference at `h/p` = 1e-3 / 1e-4 / 1e-5 | 1.074089e+12 / 1.074089e+12 / 1.074091e+12 |

  **15.9x low**, against a reference stable to 6 significant figures across three
  step sizes. Each probe is now relative:

  - **parameters** — `eps·|p|`, falling back to the old absolute step only for a
    parameter that is exactly zero (no scale of its own) or subnormal (no
    relative step survives the addition). Parameters have no common unit, so
    `|p|` is the only scale the model offers, which is also why nothing better
    than 1.0 is available for the zero case.
  - **species** — `eps·max(|y_j|, max|y|)`, i.e. relative to the species but
    floored at the state's own magnitude rather than at 1.0. Unlike parameters,
    every species is a concentration in one unit, so the state *has* a typical
    scale to probe a zero species against. The absolute floor was wrong in both
    directions here: a nanomolar model was probed at 1 molar, and a model in
    molecule counts (~1e6) was probed at 1e-14 of its own state, which is
    cancellation noise rather than a derivative. The scale skips species a
    `mask=` excluded (issue #74) — a write-only accumulator holds whatever
    integration left it at and would otherwise set the scale for everyone else.

  Both quotients now also divide by the step the write actually realized,
  `(x + h) - x`, rather than the one requested.

  **Blast radius.** The finite-difference `∂f/∂p` is reachable from
  `Simulator.steady_state(sensitivity_params=…)` only when codegen emits no
  `bngsim_codegen_sens_rhs` for the model — since #67/#89 that is Michaelis-Menten
  and Functional laws carrying a condition or a non-smooth builtin, **41 of the
  585 `ode_fullnet` models**. **15** of those 41 also carry a parameter the floor
  probes at more than 0.1% of its value, with ratios up to 1.27e30
  (`ode/AVdyn6`'s `epsilon` = 1.18e-38). Driving `find_steady_state` directly
  without a codegen artifact — which is how the repo's own tests reach the
  fallback, and how `before_bunching` was found — exposes any of the **267**
  corpus models carrying such a parameter. The species probe is reachable
  wherever the model has no complete analytical Jacobian, and through
  `jacobian="fd"` everywhere.

  **Corpus A/B.** Every corpus model was solved twice at the same root, once
  with the compiled `∂f/∂p` and once forced onto the fallback, over the four
  smallest parameters plus the largest as a control — 1,201 columns over 358
  models that converge and have an analytical `∂f/∂p` to be scored against. Of
  the 850 columns where the step rule changes at all (`|p| < 1`) and the
  sensitivity system is not itself degenerate (`sens_jacobian_rcond > 1e-8`),
  scored against the *model's* sensitivity scale rather than the column's own:

  | | before | after |
  |---|---|---|
  | columns wrong by > 1e-3 | 103 | **57** |
  | columns wrong by > 1e-1 | 70 | **42** |
  | crossing 1e-3 | — | 51 fixed, 5 broken |
  | crossing 1e-1 | — | 39 fixed, 11 broken |

  The 278 `|p| >= 1` control columns move by at most 2.2e-9 (the realized-step
  rounding), which is what confirms the change is confined to where the step
  differs. The largest single class of fixes is compartment volumes: 37 columns
  over 35 models — `Vcyt`, `V`, `Vecf` at 1e-12 — were **100% wrong** (the probe
  is 15,000x the parameter) and now land within 1e-5.

  The regressions are the opposite failure mode, and they are real: where a
  parameter's own term is a tiny fraction of the RHS component it sits in, a
  probe relative to the parameter can move `f` by less than its own roundoff,
  where the old wide step did not. The five columns that cross 1e-3 the wrong way
  are of that kind (`FceRI_viz`'s `kp1 = 1.7e-6`, `4.2e-07 -> 5.0e-01`, is the
  worst). Fixing that needs a step chosen from the *response* rather than from
  the parameter alone, which is a separate piece of work — filed as its own
  issue rather than folded in here.

- **Steady-state observable sensitivities carry the amount-valued volume factor
  (issue #119).** `compute_ss_output_sensitivity` projected `dY_ss/dp` onto the
  observables with the bare `GroupEntry::factor`. Every other site that touches
  that quantity multiplies in the species' `volume_factor` when it is
  `amount_valued` — SBML `hasOnlySubstanceUnits="true"`, where the symbol denotes
  an *amount* rather than the stored concentration — including
  `update_observables`, which is what defines the observable's **value**. So the
  steady-state block returned the derivative of something the same result does not
  report, off by the compartment volume, while `Result.output_sensitivities(...,
  axis="parameter")` at a converged run was right. Within one `SteadyStateResult`
  the `expression:` rows were correct (both of their paths carry the factor) and
  the `observable:` rows were not.

  On a birth-death hOSU model with a closed form
  (`d(amount*)/d[k_prod, k_deg] = [2, -16]`, independent of the compartment size):

  | compartment size | before | after / run / closed form |
  |---|---|---|
  | `V = 1` | `[2, -16]` | `[2, -16]` |
  | `V = 2` | `[1, -8]` | `[2, -16]` |
  | `V = 3` | `[0.667, -5.33]` | `[2, -16]` |

  The true answer does not depend on the compartment volume; the old one was
  inversely proportional to it, which is why `V = 1` hid this. `.net` models are
  unaffected (`amount_valued` is set only by the SBML loader), as are `V_c = 1`
  and every `hasOnlySubstanceUnits="false"` species — the fix is a no-op wherever
  the projection was already right. The species block is storage-based and
  deliberately keeps its unscaled weight.

- **A switch-time crossing now resumes on the branch it just crossed into, so
  forward sensitivities stop dying at fitted `if(t>=p, …)` onsets (issue #82).**
  The issue #48 stop time lands `t` exactly on the crossing `t*`, but the `if()`
  condition is read off the *counter species*, and that counter is integrated:
  it came back **1–2e-14 below** the threshold whose value defines `t*`. So the
  `CVodeReInit` at the crossing re-entered on the **before** branch, and the
  discontinuity fell inside the first step after the restart — the one thing the
  stop time exists to prevent. CVODES sized `h` from the pre-switch RHS
  (identically zero on the motivating model: no transmission, no distancing),
  every corrector answered with the post-switch RHS, and the error test failed at
  every step size down to ~1e-10: seven failures, no step completed,
  `CV_ERR_FAILURE` **at the crossing**.

  The clock is now set to the threshold (one ulp above, so a strict `>` lands on
  the after-branch too) before the restart. `t*` is *defined* as the time a
  unit-rate counter reaches that threshold, so this corrects accumulated
  integration error rather than perturbing the state; a discrepancy too large to
  be roundoff is left alone, since that would mean the crossing was detected in
  the wrong place and moving state would only hide it.

  Which side of the threshold the last bits fell on was deterministic but
  effectively arbitrary, which is why issue #82 presented as isolated spikes in
  parameter space, moved non-monotonically with `rtol` (1e-7 fine, 1e-8 fatal,
  1e-9 fine) and ignored `max_steps` entirely. On `Lin-2021/nyc_multiphase` at
  the published MAP, sensitivity w.r.t. `t0` failed while the plain solve and
  sensitivity w.r.t. `beta`/`fD` succeeded. Over 200 parameter vectors drawn from
  that job's own `uniform_var` box (5 seeds), the switch-time sensitivity solve
  went from **losing 25% of points the plain solve integrated** to **200/200**,
  with the gradient unchanged wherever it previously survived.

  Diagnosis note for the issue's own hypothesis: this is **not** the CVODES
  difference quotient. The analytic Functional sensitivity RHS from the #55 chain
  is engaged on this model and the failure was unchanged by it, so #82 was not
  closed by #66/#67/#68 landing.

- **The Michaelis–Menten free substrate is no longer clamped to zero, so the
  interpreted and compiled backends stop disagreeing and the emitted Jacobian
  stops contradicting the emitted RHS beside it (issue #93).** `compute_rxn_rate`
  floored a negative `sFree` to 0; `_mm_rate_lines` never did. Meanwhile
  `mm_tqssa_derivatives`, `_mm_jacobian_groups` and `_mm_v_lines` all guarded on
  `sFree > 0` and returned 0 — the clamp seen from the derivative side. So for
  `S < 0` the one `.so` handed CVODE a Jacobian asserting no dependence on `E`
  over a region where its own RHS varied with `E`, and the two backends returned
  different numbers for the same model at the same state.

  That is not a corner: `sFree < 0` requires only `S < 0`, which is the
  substrate-exhaustion endgame, where the integrator routinely oversteps zero.
  Both shipped MM fixtures reach it on an ordinary run (`mm_tqssa.net` to
  `S = -1.0e-8`), and the disagreement reached the **trajectory**, not just a
  probe: pre-fix, `mm_tqssa.net` finished at `S = 1.53e-9` compiled against
  `-1.22e-8` interpreted, and `mm_tqssa_stiff.net` at `9.15e-14` against
  `-8.32e-8`. Both backends now agree bit-for-bit (max relative gap exactly 0.0,
  from 3.6e-10 and 4.4e-9).

  The clamp is gone from every site rather than added to the ones that lacked it,
  because the negative branch is a genuine smooth continuation and the clamp was
  the artifact. `delta² + 4·Km·S` factors as `(S + Km − E)² + 4·Km·E`, so the
  square root is real for *every* `S`; the continuation is restoring (as
  `S → −∞` the rate tends to `kcat·S`, pushing `S` back toward 0) where the clamp
  left a flat dead zone; and at `S = 0` the unclamped rate is differentiable,
  with both one-sided derivatives equal to `kcat·E/(Km + E)`. The old guard
  reported `∂rate/∂S = 0` at exactly that state — which every species with a zero
  initial condition starts in.

  What replaces the clamp is a guard on the rate's own denominator, `Km + sFree`,
  used identically by the RHS and by all three derivative emitters (it is emitted
  once, by the shared `_mm_sfree_c_lines`, so they cannot drift apart again).
  `Km + sFree` vanishes only where `Km·E == 0`, and it also repairs a
  pre-existing NaN at `Km = 0, S < E`, which used to evaluate `0/0`. Guarding the
  denominator rather than `Km` or `E` keeps the correct `kcat·E` at `Km = 0,
  S > E`. On the degenerate set the guard is a genuine jump, not a smooth patch —
  approaching `E = 0` from above with `S < −Km` the rate tends to
  `kcat·(S + Km)`, not 0 — which is accepted because at `E == 0` the alternative
  is `0/0` and a reaction with no enzyme has rate 0 by definition.

  A downstream consequence worth calling out: the clamp was manufacturing a
  **false continuum** in steady-state sensitivities. `mm_tqssa.net` settles at
  `S ≈ -7e-8`, and inside the clamped flat region `∂f_S/∂S` was 0, so the reduced
  Jacobian was exactly singular and `steady_state(sensitivity_params=...)` refused
  the model with "dY_ss/dp does not exist … the steady state is a continuum".
  It is now solved with `rcond = 1.0`, and the gradient it returns (≈0, since the
  root `E=10, S=0, P=100` does not depend on `kcat` or `Km`) agrees with a
  re-solve at `p ± h`.

  The SBML export moves with the engine, as it must — `_mm_formula` dropped
  `max(sFree, 0)` and carries the denominator guard as a `piecewise`, keeping
  `max_rhs_delta` at exactly 0.0 against the engine on both fixtures. (An export
  that drifts is not a loud failure; it silently drops the model from parity.)
  `_CODEGEN_VERSION` is bumped to 27 so a cached v26 `.so` cannot keep serving the
  contradictory pair. The two corpus MM models (`test_MM`, `mCaMKII_Ca_Spike`) do
  not move at all — trajectories byte-identical, same step counts.

- **A Functional model built with `sensitivity_params` could not run on the MIR
  JIT backend at all — c2mir refused the generated source on every platform
  (issue #85).** `MirJit` does not hand c2mir the codegen's C unchanged: c2mir
  cannot parse the platform SDK's `<math.h>` / `<stdlib.h>` / `<string.h>`, so
  `make_jit_source` strips every `#include <…>` line and prepends a prelude that
  re-declares the libc/libm *functions* the RHS can call. Stripping a header also
  strips its **types and macros**, and the prelude re-declared none of those.
  `bngsim_codegen_output_sens` (GH #198) casts its column index with `(size_t)`;
  with `size_t` unknown as a typedef name that stops being a cast, the
  declaration it sits in fails to parse, and c2mir reports the next token — the
  `syntax error on double (expected '<statement>')` the issue opens with. `cc`
  compiles the identical source, so this was never invalid C, and it was
  pre-existing on `main` for as long as #198 had been emitted.

  The prelude now includes c2mir's own bundled `<stddef.h>` for `size_t`, and
  defines `NULL` and `NAN`. Those last two are the same hole one step behind
  `size_t`, reachable and not theoretical: a model whose functions read no
  observable passes `NULL` for the function blocks' `obs` argument, and #198
  writes `NAN` as the sentinel for a function it cannot differentiate. Neither
  can come from the bundled header — c2mir's `<stddef.h>` omits `NULL` on
  macOS-x86_64 and on Windows, and no bundled header defines `NAN` at all.
  Taking `size_t` from that header rather than a hand-written typedef also makes
  the `memset`/`memcpy` size argument match what c2mir's `sizeof` yields by
  construction, replacing the LP64/LLP64 branch GH #3 had to maintain from the
  outside.

  The 11 `xfail(strict)` quarantine markers this bug earned across
  `test_codegen_functional_sens_rhs.py`, `test_codegen_switch_condition_sens.py`
  and `test_codegen_sens_budget.py` are gone; the MIR job's list is 328 passed,
  0 xfailed under `BNGSIM_CODEGEN_JIT=mir`, and unchanged under `cc`. The new
  `test_codegen_jit_prelude.py` guards the class rather than the instance: it
  diffs the libc names appearing in emitted C against what `jit_prelude()`
  supplies, so the next such name the codegen starts emitting fails in a text
  comparison — on every backend, and in the pre-push hook — instead of in c2mir
  on the one CI job that compiles generated C with anything but `cc`.

- **The Michaelis–Menten (tQSSA) free substrate lost about two significant digits
  per decade of `|δ|/√(4·Km·S)`, in the rate itself and not only in a derivative
  (issue #89).** `sFree` was computed as

  ```
  delta = S - Km - E;  D = sqrt(delta*delta + 4*Km*S);  sFree = 0.5*(delta + D);
  ```

  When `delta < 0` — a large enzyme excess, or a small `Km·S` — that last line
  subtracts two nearly-equal positive numbers. At `|δ|/√(4·Km·S) = 1e4` the result
  has 8 correct digits, at 1e8 it has **none**. `sFree` is the positive root of
  `x² − δ·x − Km·S = 0`, so the `delta < 0` branch now uses the conjugate form
  `2·Km·S/(D − delta)`, which multiplies out to the same value with no subtraction
  and is exact to 0–1 ulp at every ratio measured. `delta ≥ 0` keeps the textbook
  expression bit-for-bit.

  Fixing the root was necessary but not sufficient. The derivatives were written
  as the chain rule through `sFree` — `∂sFree/∂E = ½(−1 − δ/D)`,
  `∂sFree/∂S = ½(1 + (δ + 2·Km)/D)`, `∂sFree/∂Km = ½(−1 + (2S − δ)/D)` — and those
  cancel in exactly the same regime, so `∂rate/∂E` stayed at a relative error of
  **1e+10** with a correct `sFree` in hand. On the regression fixture the
  cancellation went past magnitude and **flipped the sign**: `∂rate/∂E` came out
  as `-3.83e-05` where the true value is `+4.96e-15`. A negative `∂rate/∂E` says
  adding enzyme slows the reaction, and this entry feeds CVODE's Newton
  iteration — a sign error there steps confidently the wrong way, which is worse
  than converging slowly. Differentiating the symmetric form of
  the rate instead (the tQSSA complex is `c = ½(A − D)` with `A = E + S + Km`, and
  that same `D` is `√(A² − 4·E·S)`) collapses each partial to one subtraction-free
  quotient:

  ```
  ∂rate/∂E = kcat·stat·sFree/D
  ∂rate/∂S = kcat·stat·E·Km/((Km + sFree)·D)
  ∂rate/∂Km = −rate/D
  ```

  Each is identical to `sympy.diff` of the shipped rate. Measured against mpmath
  at 60 digits over the four regime sweeps from the issue — worst relative error
  in float64, before → after:

  | sweep | `sFree` | `rate` | `∂/∂E` | `∂/∂S` | `∂/∂Km` |
  |---|---|---|---|---|---|
  | uniform, O(1) | 4.5e-14 → 4.1e-16 | 4.3e-14 → 4.9e-16 | 5.6e-13 → 6.1e-16 | 2.6e-15 → 7.3e-16 | 1.4e-12 → 7.6e-16 |
  | log-spread 1e-4..1e3 | 6.2e-04 → 3.5e-15 | 6.2e-04 → 5.8e-16 | 3.0e+03 → 1.8e-15 | 2.5e-10 → 6.1e-15 | 2.6e+03 → 2.4e-15 |
  | Km ≫ S,E | 2.6e-09 → 5.4e-16 | 2.6e-09 → 6.5e-16 | 2.6e-09 → 7.6e-16 | 7.6e-16 → 4.7e-16 | 9.4e-09 → 7.9e-16 |
  | Km ≪ S,E (deep saturation) | 2.1e-02 → 2.4e-13 | 2.1e-02 → 5.1e-16 | 2.3e+09 → 4.3e-14 | 5.3e-05 → 4.7e-13 | 8.7e+09 → 2.3e-13 |

  This is a **trajectory** change, not only a gradient one, and it lands in every
  place the three lines were open-coded: `compute_rxn_rate`'s MM branch and the
  SSA propensity emitter (`src/model.cpp`), the closed-form analytical Jacobian
  shared by the dense and sparse CVODE paths (`include/bngsim/mm_jacobian.hpp`),
  the two compiled-RHS emitters, the analytical Jacobian / fused `J·v`, and the
  analytic `∂f/∂p` from #55 (`python/bngsim/_codegen.py`), plus the JAX RHS
  (`python/bngsim/_jax_rhs.py`). The expression was duplicated at each site; it is
  now emitted from one helper per language. `_CODEGEN_VERSION` → 26, since a
  cached v25 `.so` would keep serving pre-fix numbers.

  Both corpus MM models (`test_MM`, `mCaMKII_Ca_Spike`) run at
  `|δ|/√(4·Km·S) ≤ 5`, where the two forms differ by 1–3 ulp — measured largest
  `sFree` shift 1.9e-16 and 6.0e-16 — so the corpus shows no trajectory change and
  the emitted C is byte-identical on the 120 non-MM models sampled. The
  regression fixture is therefore a deliberately stiff one
  (`tests/data/mm_tqssa_stiff.net`, ratio ~1e7), where the old code was 2% out on
  the rate and returned the sign-flipped `∂rate/∂E` above. The C++ suite asserts
  that sign directly, so the sharpest symptom is the first thing to fail.

- **The SBML exporter kept the cancelling tQSSA root after #89 fixed the engine,
  so `net2sbml` emitted a model its own validator rejects (follow-up to issue
  #89).** `_mm_formula` in `python/bngsim/convert/_sbml_writer.py` was the one
  site deliberately left out of #89, on the reading that an interchange artifact
  should stay a faithful transcription of the canonical literature formula. That
  reading does not survive contact with the measurement: SBML MathML is evaluated
  by the *consumer*, in float64, so the exported law performs the same
  `0.5*(delta + D)` subtraction the engine had just stopped performing.

  With #89 in place the two sides disagree, and the gate says so. On
  `tests/data/mm_tqssa_stiff.net` (ratio ~1e7), before → after:

  | | textbook export | stable export |
  |---|---|---|
  | RHS vs source model | 2.17e-02 | **0.0** (bit-for-bit) |
  | conversion gate | L2 **fail**, L3 **fail**, `ok=False` | L0–L3 pass, `ok=True` |
  | libRoadRunner on the artifact | `CV_ERR_FAILURE`, no trajectory | agrees with bngsim to 2.9e-11 |
  | rate at t=0 (vs mpmath, 50 dps) | 2.09e-02 | 1.90e-16 |

  So the faithful-transcription reading was shipping a model that bngsim itself
  marks not-ok and that RoadRunner cannot integrate. The export now carries the
  same branch as `_mm_sfree_c_lines`, spelled as the L3v2 `piecewise` the writer
  already targets:

  ```
  sFree = piecewise(0.5*(delta + D), delta >= 0, 2*Km*S/(D - delta))
  ```

  It omits only that helper's `(D - delta) > 0` degenerate guard, unreachable
  once `delta < 0` and backstopped by the enclosing `max(..., 0)`.

  The `piecewise` costs less than it looks. It is not a new construct for this
  writer — the target is L3v2 chosen for exactly this MathML, `piecewise` is
  already in `_EXPRTK_SUPPORTED_CALLS`, and every BNGL `if()` already exports as
  one. It does not move L4 either: that level is *already* `inconclusive` on MM
  (pinned by `test_full_gate_l4_is_non_gating`) because the law already contains
  `max`, and `_validate.py` lists `Max` and `Piecewise` in the same `_NONSMOOTH`
  tuple. And unlike that `max` — a genuine kink — this branch is *removable*:
  both arms agree to the last ulp at `delta == 0` (measured across
  `delta = ±1e-3 … ±1e-12`, worst 1.11e-16), so no consumer's integrator gains a
  discontinuity to resolve. The cost that is real is size: the emitted document
  grows 6519 → 12121 bytes on the stiff fixture, because `sFree` appears twice in
  the law and SBML has no `let` binding.

  Where the corpus lives nothing moves at all: at ratio ~1 and on the `delta >= 0`
  branch the emitted law is **bit-identical** to the textbook one, so both corpus
  MM models (`test_MM`, `mCaMKII_Ca_Spike`) export unchanged. Pinned by
  `test_mm_tqssa_stiff_rhs_exact` and `test_mm_tqssa_stiff_full_gate_passes`; the
  reasoning is recorded in the `_mm_formula` docstring so the scope question does
  not get re-opened from first principles.

- **The conservation-law reduction chose dependent species it could not solve
  for, so the reduced system was singular on models that are perfectly well posed
  (follow-up to issue #63).** `detect_conservation_laws` found every law — the
  count matches `ns_free - rank(S_free)` on all 408 solvable corpus models, so
  nothing was missing — but then picked each law's *dependent* species
  independently: the largest-|coefficient| species no earlier law had claimed.

  Every consumer eliminates one species per law in a single pass.
  `reconstruct_full` solves law `k` for `y[dep_k]` from the current value of every
  other species, and `compute_ss_sensitivity` forward-substitutes the same walk to
  build `D = ∂y_dep/∂y_ind`. Both are exact only when `L[:, dependent]` is the
  identity. The greedy rule guaranteed neither triangularity nor even
  invertibility, and on 52 of 374 corpus models with laws it chose a set for which
  `L[:, dependent]` is outright **singular** — an elimination with no solution.
  The smallest case is four species (now `tests/data/conservation_singular_dep.net`):
  the two laws are `Xtot = X0 + Xp + XY` and `Ytot = Y + XY`, and choosing `X0` and
  `Xp` as the dependents gives `L[:, dep] = [[-1,-1],[1,1]]`, rank 1.

  The failure was silent in three places at once. The reduction violated the
  constraints it was enforcing (`|L_ind + L_dep·D|` reached 2.0 instead of 0); the
  reduced Jacobian inherited a null space of the reduction's own making, so
  `dY_ss/dp` was whatever the LU made of a singular system; and the reduced-space
  KINSOL solve failed and quietly downgraded to `method_used="integration"`.

  Dependents are now the pivot columns of the row-reduced law matrix, chosen by
  full pivoting, so `L[:, dependent] = I` by construction and the invariant both
  consumers assume is a property of the data rather than a hope.

  Measured over the 585-model `ode_fullnet` corpus (408 reach a steady state with
  a complete analytical Jacobian), before → after:

  | | before | after |
  |---|---|---|
  | `L[:, dependent]` is the identity | 156/374 | **374/374** |
  | `L[:, dependent]` singular | 52 | **0** |
  | worst constraint violation `\|L_ind + L_dep·D\|` | 2.0 | **0** |
  | rank-deficient reduced Jacobian | 93/408 | **48/408** |
  | `dY_ss/dp` wrong vs. a finite difference of the root | 104/332 | **32/332** |

  The three models issue #63 reported as having continuum steady states were all
  this bug, not their dynamics:

  | model | rank before | `min\|U\|/max\|U\|` before | rank after | after |
  |---|---|---|---|---|
  | `IGF1R_model_v1` | 578/579 | 9.8e-10 | **579/579** | **1.5e-3** |
  | `Reduced_IGF1R_hela` | 546/549 | 4.8e-12 | **549/549** | **4.7e-3** |
  | `fceri_fyn` | 1274/1276 | 5.9e-13 | **1276/1276** | **1.6e-4** |

  On all three, `steady_state(method="newton")` now converges as Newton instead of
  falling back to integration (residual 9e-9 → 3e-14, 8e-9 → 4e-13, 9e-9 → 1e-13),
  and `dY_ss/dp` matches a central difference of the steady state itself to ~1e-5
  where it previously did not agree at all.

- **A steady-state sensitivity solve that returns NaN now raises instead of
  handing back the NaN (follow-up to issue #63).** When the reduced LU hits an
  exact zero pivot the dense SUNDIALS solver has no least-squares fallback, so
  `ss.sensitivity` came back non-finite behind nothing louder than a log warning —
  the shape a fitter silently turns into a non-update. `steady_state()` now raises
  `SimulationError`, naming the species whose gradient is non-finite. Driving the
  real entry point over the corpus: 395 models return a gradient, 31 are refused,
  and no NaN reaches the caller. `steady_state_batch()` is unchanged, as the
  warning never covered it either.

  The `min|U|/max|U| < 1e-8` **warning was deliberately not promoted to a
  refusal**, which is what the follow-up set out to do. The full 585-model sweep
  says no threshold on that ratio can carry one. Ground truth: solve for
  `dY_ss/dp` as `compute_ss_sensitivity` does, then check it against a central
  difference of the steady state itself (re-solve at `p ± h` from the same initial
  conditions), keeping only probes that converge in the step size. Of the 308
  models where the reduced solve returns a finite answer, 286 are right and 22 are
  wrong — and the populations interleave:

  * correct gradients go arbitrarily low — `ode/simplifications_v1` measures
    1.5e-42 and is accurate to 7e-7; `RBM_covid_v2` (n=112) measures 1.1e-13 and
    is accurate to 1.2e-6;
  * wrong gradients go arbitrarily high — 6 of the 22 have a *perfectly*
    conditioned reduced Jacobian, two of them at exactly 1.0, wrong by >100%.
    Those are not conditioning failures, so no conditioning number can see them.

  The best available cut on the ratio (4.3e-9) still misclassifies 10; the shipped
  1e-8 would discard 6 correct results and let 6 wrong ones through. `1/κ₁` and
  `σ_min/σ_max` do no better (9 and 10 errors at their best cuts). The warning
  therefore stays a warning, and its text no longer asserts the steady state is a
  continuum — about half the models it fires on return a correct gradient — but
  says what was measured and tells the reader to check against a finite
  difference.

- **The committed type stub carried whatever commit the last local rebuild came
  from, `+dirty` marker and all.** `scripts/rebuild_editable.py` regenerates
  `python/bngsim/_bngsim_core.pyi` from the freshly built module, and
  pybind11-stubgen faithfully copies the module's `__build_commit__` — which
  CMake stamps with the current git commit, suffixed `+dirty` when the tree has
  uncommitted changes. Since every C++ change runs that script, the stub picked
  up a new value on each rebuild: a spurious one-line diff every time, and a
  standing invitation to commit one developer's build stamp. PR #70 merged
  `'e61f83d57358+dirty'` exactly that way.

  The regeneration now rewrites that line to `'unknown'` — CMake's own default
  when git provenance is unavailable, which is precisely a type stub's
  situation. Nothing downstream reads it (mypy checks the declared *type*, and
  the provenance guard from #125 reads the compiled module's runtime attribute,
  never the stub), so pinning it costs nothing and makes the stub reproducible
  across machines and working-tree states. A test asserts the committed stub
  carries no build stamp, so a dropped normalization fails loudly instead of
  landing in the next merge.

- **A zero-arg observable call inside a function made the codegen emitters emit C
  that does not compile, so forward sensitivity refused the model (issue #28, the
  codegen half).** BNGL accepts an Observable written as a zero-arg call —
  `divide()` — anywhere the bareword is valid, and BNG2.pl preserves whichever
  form the user wrote when it emits the `.net`. `ExprTkEvaluator::compile` already
  strips `name()` → `name` for every name registered as a scalar variable
  (`strip_empty_parens`, `src/expression.cpp`), so the interpreted engine has run
  these models since #28. The two codegen identifier tables did not: they set
  `eats_empty_parens` only for *functions*, so an observable was rewritten to its
  C scalar with the empty call still attached — `func[0] = (1.0-obs[1]())*1.0;`
  and `double func__rateLaw1 = (1.0-obs_divide())*1.0;` — and the compile failed
  with *error: called object type 'double' is not a function or function pointer*.
  Every model name resolves to a scalar in the emitted C (`obs[j]`, `p[k]`,
  `y[i]`, `func[m]`, `current_derivs[i]`, `M_PI`), so all of them now eat the
  parens; `eats_empty_parens=False` is left only for the built-ins that really are
  C functions or operators (`fabs`/`log`/`round`/`fmax`/`fmin`, `&&`/`||`/`!`),
  where an empty argument list is not valid ExprTk to begin with.

  The same missing strip reached the sympy-facing differentiator, where it was
  worse than a build failure. `parse_expr` reads `divide()` as an *applied
  undefined function* that shares no symbol with the bareword, so `∂/∂divide` of
  `100*divide()` came back empty: a silently zero analytical-Jacobian entry when
  such a body is a rate law, and a silently zero `d func/dθ` in the #198
  expression output sensitivities — both reported as ordinary results, neither
  refused. A body like `scale*Atot()` instead differentiated to something the C
  printer cannot render, i.e. a spurious "not representable" decline.
  `_preprocess_exprtk` now strips the parens for the sympy path too, after the
  `time()` rewrite that needs them.

  `ode/proliferation.bngl` — whose functions read `(1-divide())*1`, `100*divide()`
  and `10*divide()` against the observable `divide` — is the one refusal the #69
  entry below left open, and the only model in the 585-model `ode_fullnet` corpus
  that calls an observable or parameter as `name()` inside a function body. Its
  sensitivity Simulator now builds, taking the corpus to 585/585, and its compiled
  trajectory matches the interpreted RHS to 4.5e-11 over `t ∈ [0, 20]`. A new
  fixture (`tests/data/obs_zero_arg_call_sens.net`, distilled from that model)
  pins the emitted C against a call on any C scalar, and checks `d func/dθ` for
  the call form against the model's closed forms on both the constant-coefficient
  shape (which used to come back as exactly 0.0) and the parameter-coefficient
  shape (which used to decline).

- **Steady-state expression output sensitivities dropped the derived-parameter
  chain rule.** `compute_ss_output_sensitivity`'s explicit-parameter term
  `∂func/∂p` is a finite difference taken by writing the perturbed value straight
  into the model's `Parameter` vector — and neither `update_observables` nor
  `evaluate_functions` re-derives ConstantExpression parameters; only `set_param`
  does. So when BNG2.pl encodes a rate law or function coefficient as a derived
  parameter (`_rateLaw1 = chi*kon`), perturbing the primary `kon` left `_rateLaw1`
  at its nominal value and `∂func/∂_rateLaw1 · ∂_rateLaw1/∂kon` silently vanished
  from `ss.output_sensitivities(["expression:..."])`. The same defect class as
  issues #2 / #41, and the same hole `compute_ss_sensitivity`'s `∂f/∂p` had until
  #63 — that one was repaired by routing parameter writes through
  `SteadyStateRhs::sync_params(held)`; this one was left behind because it
  computes a different derivative.

  The probe now takes the same route: `sync_params(pi)` after the perturb *and*
  after the restore, re-deriving every expression parameter except the one being
  probed (matching `set_param`'s detach-then-refresh rule, so a probe of a
  derived parameter is not immediately undone). The restore call matters
  independently — without it the derived parameters keep the values the last
  probe gave them and corrupt every later column and the caller's model.

  What came back before was not a small error. On a fixture in the shape of
  `tests/data/derived_rate_const.net` with `flux() = _rateLaw1*A_tot` over
  A ⇌ B (a = chi·kon, b = koff, so flux = a·b/(a+b)), the returned value was the
  bare state-chain term `a·dA_ss/dp` — the wrong sign and 21× too large:

  | | d flux/d kon | d flux/d chi | d flux/d koff |
  |---|---:|---:|---:|
  | before | −0.4535 | −0.04535 | 0.9070 |
  | after / closed form | 0.02268 | 0.002268 | 0.9070 |

  `koff` was always right (it reaches the function only through the state), and
  so was a probe of `_rateLaw1` itself (a direct write needs no chain) — only the
  primaries feeding a derived parameter were wrong. The CVODES codegen
  output-sensitivity chain rule (GH #198) agrees with the closed form to 1e-4,
  and is used as an independent oracle in the regression test.

- **`steady_state()` never received the compiled RHS, and finite-differenced all
  of `dY_ss/dp` — including the Jacobian the model already had analytically
  (issue #63).** Two defects, both invisible from Python.

  *The codegen artifact never reached the solver.* `Simulator.steady_state()` and
  `steady_state_batch()` built a `SteadyStateOptions` and set `tol`, `max_time`,
  `method`, `rtol`, `atol`, `max_steps` and `jacobian` — but never
  `codegen_so_path`, and `steady_state.cpp` never read it either (the string
  "codegen" appeared in that file exactly once, in a comment). There was no
  `codegen_c_source` field at all, so the MIR JIT backend was not even
  *representable* for a steady-state solve. A Simulator whose `codegen_backend`
  reported `"cc"` therefore solved on the interpreted ExprTk RHS,
  indistinguishably from one built with `codegen=False`: on fceri_fyn (1281
  species) the two took **13686.5 ms and 13686.2 ms** and reported the same 685
  steps and 931 RHS evaluations. Every RHS evaluation in the file — the CVODE
  march, the KINSOL polish, the residual check, the sensitivity assembly — now
  goes through one backend-dispatching object, and `ss.rhs_backend` reports which
  ran.

  *Both factors of `dY_ss/dp` were finite differences.* `compute_ss_sensitivity`
  built `J` by one interpreted RHS evaluation per species (~1300 of them on a
  1281-species model) at a fixed `sqrt(eps)` step, never consulting the complete
  analytical Jacobian that `jacobian="auto"` selects everywhere else and that the
  `newton` path's KINSOL polish in the same file already uses; and `∂f/∂p` by
  perturbing each parameter in place. `J` now prefers the compiled analytical
  Jacobian, then the interpreted one, then finite differences — the same rule as
  everywhere else, with `jacobian="fd"` still pinning the difference quotient —
  and `∂f/∂p` uses the analytical column the codegen sensitivity RHS emits
  (evaluated at `yS = 0`, which zeroes its `J·yS` term exactly). Like `run()` and
  `compute_all_sensitivities()` since GH #214, `sensitivity_params` now *requires*
  codegen and refuses rather than degrading. `ss.sens_jacobian_source` and
  `ss.sens_dfdp_source` report each factor's provenance; a model whose rate laws
  are not all Elementary has no analytical `∂f/∂p` to emit (issue #55) and still
  differences that factor, now with a warning.

  Measured on `benchmarks/suites/ode_fullnet/nets` (best of 2, `tol=1e-9`,
  4 sensitivity parameters). "assembly" is the sensitivity solve minus the
  identical solve without it, so the Jacobian and `∂f/∂p` work is not buried under
  the march:

  | Model | species | solve before | solve after | assembly before | assembly after |
  |-------|--------:|-------------:|------------:|----------------:|---------------:|
  | egfr_net | 356 | 234.9 ms | 174.3 ms | 27.9 ms | 7.9 ms (3.5×) |
  | IGF1R_model_v1 | 589 | 285.7 ms | 242.3 ms | 75.4 ms | 14.8 ms (5.1×) |
  | before_bunching | 593 | 474.3 ms | 398.6 ms | 86.1 ms | 15.1 ms (5.7×) |
  | Models_n | 624 | 519.7 ms | 457.2 ms | 95.5 ms | 25.2 ms (3.8×) |
  | fceri_fyn | 1281 | 13157 ms | 15489 ms | 903.2 ms | 375.1 ms (2.4×) |

  Two honest caveats in that table. The compiled RHS is *slower* on fceri_fyn: it
  is arithmetically the same function but rounds differently, which walks CVODE
  down a 19% longer step path (812 steps / 1185 RHS evals versus 685 / 931), and
  this model's cost is dominated by 1281×1281 dense LU rather than RHS
  evaluation, so cheaper steps do not pay for more of them. Both backends land on
  the same root (1.1e-8 relative). And below the codegen crossover the usual rule
  applies — a 20-species model with an explicit `codegen=True` spends more on the
  compiled call than it saves (0.8 ms → 2.0 ms), exactly as `run()` does, which is
  why the auto-attach threshold is 256 species.

  Part of the assembly speedup is independent of codegen: the linear solve
  factorized the *same* Jacobian once per sensitivity parameter. It now factors
  once and solves `np` right-hand sides.

  Two further corrections fell out of making the two paths agree:

  * The finite-difference `∂f/∂p` wrote perturbed values straight into the
    Parameter vector, which nothing re-derives — only `set_param()` refreshes
    constant-expression parameters. So on a model where BNG2.pl encodes a rate law
    as `_rateLaw1 = chi*kon`, perturbing `kon` left `_rateLaw1` at its nominal
    value and the chain-rule term silently vanished (the issue #2 / #41 defect,
    surviving here). Parameter writes now route through a sync that re-derives
    every expression parameter except the one being probed — `set_param`'s own
    detach-then-refresh rule — so the FD fallback and the analytical path agree
    with the closed form on `derived_rate_const.net`, including the `_rateLaw1`
    column that used to come back zero.
  * `dY_ss/dp` only exists when the Jacobian at the root has full rank, and on
    real models it often does not: of eight large corpus models, three came back
    rank-deficient by 1–3 (IGF1R_model_v1, Reduced_IGF1R_hela, fceri_fyn) — a
    steady state that is a continuum rather than an isolated point. The old
    finite-difference Jacobian's `sqrt(eps)` noise perturbed the singular
    direction just enough for the LU to return a finite, modest-looking, entirely
    meaningless answer; an exact Jacobian does not launder that. The result now
    carries `ss.sens_jacobian_rcond` (`min|U|/max|U|` off the LU: 1e-4 to 1e-1 for
    the five well-posed models, 1e-12 to 1e-9 for the three singular ones) and
    warns below `1e-8`. It warns rather than refusing because eight models is not
    enough to set a refusal threshold.

- **A parameter named `p` or named after a C keyword made the sensitivity RHS
  emit C that does not compile, so forward sensitivity refused the model.**
  `_derived_param_jacobian_checked` differentiates a derived rate constant with
  sympy and then maps `sp.ccode`'s output back to `p[idx]` *by parameter name*.
  Two names break that round trip, and both are real in the 585-model
  `ode_fullnet` corpus:

  A parameter literally named `p` (`ode/localfunc_2.bngl`) collided with the
  parameter array the rewrite writes. The rewrites ran one `re.sub` per name over
  a string they were themselves editing, so for `_rateLaw = k*p` the `k` → `p[0]`
  pass went first and `\bp\b` then matched the `p` it had just written:
  `v = (p[1][0]) * y[1];` — *error: subscripted value is not an array, pointer,
  or vector*. The rewrites are now a single alternation pass, so text a
  substitution injects is never rescanned.

  A parameter named `const` (`ode/pulses_demo_fixed.bngl`) was renamed by sympy
  on the way out: the C printer appends `reserved_word_suffix` to any symbol
  whose name is a C reserved word, so `Symbol("const")` printed as `const_` — a
  name no rewrite matched and nothing declares: *error: use of undeclared
  identifier 'const_'*. C reserved words now take the same alias path
  `lambda` and the other Python keywords already took (issue #27); the alias is
  not reserved, so it survives `ccode` verbatim. Two parameters that would land
  on the same alias are refused with a reason rather than silently merged into
  one chain rule.

  Both models are function-free, so forward sensitivity requires the analytic
  RHS and a failed build is a hard `RuntimeError`, not a fallback to CVODES'
  difference quotient. Across the corpus this moves the sensitivity Simulator
  from 578/585 to 584/585 — the two above plus `4var_model`,
  `4var_model_with_FDC`, `simple_1` (the `p` shape) and `kinetics_mb1n` (the
  `const` shape). Every recovered parameter's sensitivities match a
  finite-difference reference taken over the `.net` source to ~1e-6 relative.
  The one model still refused, `ode/proliferation.bngl`, fails for an unrelated
  reason: a zero-arg observable call `divide()` inside a function is emitted as
  `obs[1]()`.

  `localfunc_2` and `pulses_demo_fixed` are the two function-free models the
  issue #62 entry below records as refusing the single-shot path over "a separate
  defect, not a differentiability limit". That defect is this one, so both entry
  points now answer them.

- **`compute_all_sensitivities` skipped sensitivity codegen on models with no
  functions and ran the interpreted RHS, up to 49x slower than the identical
  coupled solve (issue #62).** The chunked entry point attached the analytical
  sensitivity RHS only when `model.n_functions > 0`. That gate belongs to the
  #198 *expression* output-sensitivity evaluator, which genuinely has nothing to
  emit for a function-free model — but it also decided whether the chunks got a
  sensitivity RHS at all, so those models ran every chunk interpreted, with
  CVODES finite-differencing the whole `∂f/∂y·s + ∂f/∂p`. `Simulator(model,
  sensitivity_params=...)` has no such gate: since #214 it requires the
  analytical RHS unconditionally, because the finite-difference one silently
  fails at tight tolerances. Two entry points computing the same tensor from the
  same work therefore disagreed, and `n_functions` separated the two groups
  exactly — models with functions at parity with the coupled solve, every
  function-free one degraded, and the degradation growing with the network.

  Measured at `rtol = atol = 1e-8`, `t_span = (0, 1000)`, `n_points = 101`, all
  primary parameters, `chunk_size = Np`, `n_workers = 1` — one chunk over exactly
  the parameter list the coupled call differentiates, i.e. identical work:

  | model | species | reactions | funcs | Np | coupled | chunked (before) | chunked (after) |
  |-------|--------:|----------:|------:|---:|--------:|-----------------:|----------------:|
  | SHP2_base_model_2 | 149 | 1032 | 0 | 24 | 32 ms | 78 ms (2.5×) | 33 ms (1.0×) |
  | egfr_net_6 | 356 | 3749 | 0 | 43 | 1.44 s | 27.9 s (19.3×) | 1.43 s (1.0×) |
  | IGF1R_model_v1 | 589 | 4198 | 0 | 10 | 212 ms | 262 ms (1.2×) | 210 ms (1.0×) |
  | fceri_fyn | 1281 | 15328 | 0 | 34 | 15.7 s | 768 s (48.8×) | 15.8 s (1.0×) |
  | MTORC1_assembly_v3 | 330 | 2519 | 13 | 41 | 560 ms | n/a — gate passed | 508 ms (0.9×) |

  The extra cost is not only the finite-difference evaluations. Their ~sqrt(eps)
  noise also degrades step-size control, so the gap compounds with the
  integration horizon: on `egfr_net_6` the old chunked path took 8750 internal
  steps against the coupled solve's 658, and on `fceri_fyn` 40202 against 656.
  The new path matches the coupled step count exactly on every function-free
  model in the table, which is the sharper statement of the fix — the chunk is
  now doing the coupled solve's work, not a noisier version of it.

  The attach is now unconditional, matching the constructor. What is genuinely
  expression-specific stays gated: only the output-sensitivity *rebuild* — the
  clear-and-regenerate that upgrades an already-attached plain-RHS codegen to one
  carrying the #198 evaluator — still tests `n_functions`, because
  `_codegen_emit_flags` emits that evaluator only for models that have functions.
  For a function-free model the generated source is byte-identical either way, so
  there is nothing to rebuild and an inherited plain-RHS codegen is already the
  right artifact.

  Parameter sharding is built on `compute_all_sensitivities`, so this turns it
  from a pessimization back into a speedup on exactly the large function-free
  networks it exists for, and removes the inflation in any speedup measured
  against an `n_shards = 1` baseline that was itself degraded.

  **Behavior change:** #214's refusal now reaches function-free models too.
  `compute_all_sensitivities` with `codegen=False`, `BNGSIM_NO_CODEGEN`, no
  codegen backend, or rate laws that do not differentiate to closed form raises
  instead of silently returning finite-difference sensitivities — the same answer
  `Simulator(..., sensitivity_params=...)` has given since #214. On the 585-model
  corpus that reaches 2 of the 293 function-free models, and both already refuse
  the single-shot path today (their codegen emits invalid C — a separate defect,
  not a differentiability limit), so no model that gets a sensitivity tensor from
  one entry point now fails at the other.

- **A collapsed step size at a rate-law discontinuity made `Simulator.run` never
  return, and no step bound stopped it (issue #54).** At an `if(t >= sigma)` rate
  jump CVODE drives the step size to ~1e-15 until `t + h == t` and returns
  `CV_TOO_MUCH_WORK`. That return is ordinarily recoverable — `max_steps` is a
  batch size *per output point*, not a ceiling on the run, so the integrator's
  state is intact and calling `CVode` again simply continues. The retry loop had
  no exit for a batch that bought no progress, so the run never ended:
  `max_steps=1_000_000` changed nothing, `max_step=0.5` changed nothing, and only
  the wall-clock `timeout` ever stopped it. In a PyBNF fit with
  `wall_time_sim = 60` every such trial burned a full minute before being scored
  `inf`.

  The retry now stops the moment a batch fails to advance the integrator's
  internal time and raises a `SimulationError` naming the `t` and `h` it wedged
  at, plus the likely cause. On the reported reproducer that turns a run that
  never returned into a failure in **0.14 s**, pointing at `t = 68.3718` — which
  is exactly `sigma` — with `h = 6.6e-15`.

  Bounding on *progress* rather than on a step count is what keeps this free of
  false positives. A model that legitimately needs many steps advances every
  batch, however slowly, so it is untouched; the same reproducer at
  `rtol = atol = 1e-7` still integrates normally, as the issue's own table
  records, where a cumulative step ceiling would have refused it.

- **The event-time sensitivity guard missed state-dependent triggers reached
  through SBML, answering those models instead of refusing them (issue #52).**
  The guard refuses forward sensitivities when an event's crossing time depends
  on a requested parameter, and it decides that from the trigger's *bound
  addresses*. It compared against species concentrations only. But ModelBuilder
  registers a species as an ExprTk variable only when its name is still free, and
  SBML models routinely give each species an observable of the same name — so the
  species registration is skipped and a trigger token binds to the observable
  total, never to `&sp.concentration`. Every SBML state-dependent trigger
  therefore slipped the guard and was answered, with the event contributions
  missing entirely. On AMICI's `neuron` fixture (Izhikevich, trigger `v > 30`,
  which names no parameter but whose crossing time depends on `a` and `b` through
  the trajectory) the returned sensitivities were 6x–135x off, uniformly in one
  direction, across all four parameters.

  The guard now tests against every address that carries live state: species
  concentrations, observable totals, and rateOf accessors. An observable total is
  a linear functional of the state and a rateOf accessor is dx/dt, so a trigger
  reading either has a non-zero `dt*/dp` exactly as a concentration read does.
  Refusing is unchanged as a policy — this is only the coverage of what counts as
  state — and the message now says which of the three it saw and why that implies
  a moving crossing time.

- **The codegen cache did not invalidate on a codegen change, so a fix could be
  silently inert on a warm cache (issue #51).** The `.net` path keys its compiled
  `.so` on the model content plus the hand-maintained `_CODEGEN_VERSION` constant
  rather than on the generated C — hashing the C would mean a full source-gen on
  every cache probe. That made the constant load-bearing: a change altering the
  emitted forward-sensitivity RHS *without* bumping it was invisible to any
  machine with a warm `~/.cache/bngsim/codegen`, which kept loading the stale
  library and returning the pre-change numbers. #41 and #43 both shipped that
  way. On the reported model the difference is stark: warm cache gives
  `max|dy/dp| == 0` for all six fitted parameters, an empty cache gives values up
  to `1.0e6` agreeing with AMICI to `1.5e-5` — same wheel, same model.

  The cache key is now `_CODEGEN_CACHE_KEY`: the constant *plus* a digest of the
  source of every module that determines the emitted C (`_codegen.py`,
  `_jacobian.py`, `_saturable_jacobian.py`). Editing an emitter changes the key
  whether or not anyone remembers the constant, which makes the omission
  harmless rather than silent. The digest is computed once at import from three
  file reads (~350 KB, well under a millisecond) and costs nothing per probe, so
  the memo fast path stays fast. It applies to every codegen artifact keyed this
  way, including the SSA propensity `.so` and the `prepare_codegen` memo.

  Deliberately conservative in two directions: it hashes source text, so a
  comment-only edit also invalidates — over-invalidation costs one recompile,
  under-invalidation is a silently wrong gradient — and it covers the Python
  emitters only, so `_CODEGEN_VERSION` remains the escape hatch for a C++ change
  that alters `codegen_data()`, and for deliberately invalidating a release's
  caches. On a `.pyc`-only or zipped install the sources cannot be read and the
  key degrades to the constant alone, the pre-fix behavior. `CONTRIBUTING.md`
  now documents when a manual bump is still required.

- **A derived parameter with a compound condition silently zeroed its
  forward-sensitivity chain rule (issue #56).** A `ConstantExpression` parameter
  defined by `if((sel>=1)&&(sel<10), kA, kB)` produced the same trajectory as the
  equivalent simple condition but a gradient component of exactly `0.0` — not an
  approximation, a wrong number. The two derived-parameter differentiators in
  `_codegen.py` (`_compute_derived_param_jacobian` for rate constants,
  `_derived_expr_partials_numeric` for initial-condition seeds and switch-time
  thresholds) rewrote only `if()` before `parse_expr`, never the logical
  operators; `parse_expr` then raised, both functions caught it, and their
  callers read the missing partial as `∂p_d/∂primary = 0` — indistinguishable
  from a primary that genuinely does not appear. In a fitting workflow that
  surfaced as an optimizer that simply never moved one parameter. This is the
  same gap #53 closed in `_jacobian._preprocess_exprtk`, where the consequence
  was only a fall back to a finite-difference Jacobian: slower, but right.

  Both functions now run the same ExprTk-to-sympy pipeline the rate-law
  differentiator uses. `_rewrite_logicals` and its helpers moved from `_jacobian`
  to `_codegen`, next to the `if()`→`Piecewise` rewriter — one implementation,
  and the import keeps going one way only. That also picks up two neighboring
  silent zeroes in the same expressions: `^` (BNGL exponentiation, which Python
  reads as XOR) and `not(x)`. Four of the five call sites were affected — the
  `.net` and model-path sensitivity RHS (#15), the derived-IC sensitivity seed
  (#43), and #48's switch-time `∂t*/∂p`; `_analyze_output_sens` already failed
  loudly.

  Failures that remain are no longer silent, and where there is a correct
  alternative they are no longer wrong. `_derived_param_jacobian_checked`
  separates "this expression references no primary, so zero is the right answer"
  from "a real contribution was lost", which `None` alone could not express. The
  two sensitivity-RHS generators act on that: if a derived parameter that is
  actually some reaction's **rate constant** cannot be differentiated, the whole
  analytic sensitivity RHS is declined with a warning, so the run falls back to
  CVODES' internal difference quotient — slower, but right — instead of emitting
  a gradient component of exact zeros. This is the same trade #53 made. Only rate
  constants are considered, so a derived parameter used purely for reporting (an
  observable or a function) no longer costs a model its analytic sensitivities,
  and no longer warns about a chain rule that never fed the RHS. On the
  585-model `ode_fullnet` corpus this changes 7 models: 5 that previously
  *crashed* codegen now get an analytic sensitivity RHS, and 2 whose derived rate
  constant is genuinely undifferentiable now decline cleanly with a warning; no
  model lost an analytic RHS it previously had, and the corpus produces 2
  warnings in total.

  The initial-condition seeding path has no such fallback — the seed is either
  computed or left at zero — so there a lost partial is reported as a warning
  naming the expression, the reason, and the parameters whose sensitivities will
  read as zero. It warns rather than raises because the seeding scan visits every
  parameter-referenced initial condition in the model, most unrelated to the
  requested sensitivity parameters.

  Two adjacent hazards are now refused outright: a primary parameter whose name
  shadows a sympy class (`And`, `Or`, `Piecewise`), which would have been
  captured by the class and differentiated to zero; and a derivative sympy cannot
  render as C, which used to escape as `PrintMethodNotImplementedError` and abort
  the entire codegen build, against the differentiator's documented
  `None`-or-dict contract.

  `_CODEGEN_VERSION` is bumped to `23`. Both directions of this change alter the
  emitted `.net` sensitivity RHS, and that path's cache key is content+version
  rather than the generated source, so without the bump a warm
  `~/.cache/bngsim/codegen` would keep serving the pre-fix `.so` and the fix
  would be silently inert — the failure mode issue #51 documents for #41 and #43.

- **`set_param` was not propagated on two network-free session paths, both
  silently (issue #44).** Two independent gaps let a network-free session run
  with plausible-but-wrong state. (1) `NfsimSession.set_param` after
  `initialize()` updated the parameter value (`get_parameter` reflected it) but
  did not refresh the reaction rate — a rule whose rate loaded at exactly zero
  had been *dropped* by NFsim's parser (`NFinput` only registers a rule when its
  base rate is `> 0`), so no later write could resurrect it, and the common
  "equilibrate, then switch a rate on, continue" protocol was a no-op. NFsim now
  keeps zero-base-rate rules when the host asks (new opt-in
  `System::keepZeroRateReactions`, threaded through `initializeFromXML` and
  carried as vendor patch 0015); their propensity is zero so trajectories are
  unchanged, but a post-init `set_param` can now activate them. (2)
  `RuleMonkeySession.set_param` before `initialize()` did not re-derive a
  seed-species amount given by a derived expression (`Ntot = 100*scale`): upstream
  RuleMonkey records only each parameter's precomputed `value=` and never
  cascades an override through dependent parameters, so the count stayed at the
  XML-time value while `NfsimSession` scaled correctly — the two engines silently
  started from different initial conditions. The RuleMonkey wrapper now bakes the
  override-resolved parameter namespace into the XML before the engine parses
  (and re-rounds fractional seed amounts), mirroring the NFsim path, so both
  engines start identically. The `<Parameter>`-table + override-baking machinery
  is now shared between the two backends in `src/param_override_xml.hpp`.

- **An explicit `$BNGPATH` was silently ignored whenever PyBioNetGen was
  installed.** The test-side BNG2.pl helpers asked `bionetgen.main.get_conf()`
  for its bundled copy first and read `$BNGPATH` / `$BNG2_PL` only from an
  `except` branch — so with bionetgen importable, `export
  BNGPATH=/path/to/BioNetGen-2.9.3` changed nothing and said nothing, and an
  install whose `get_conf()` returned no `bngpath` resolved to `None` with the
  env var sitting there unread. Precedence is now explicit-beats-implicit
  (`_core.bngpath`, see Added), and `test_bngpath_resolver.py` pins it so the
  inversion cannot return. The symptom was a machine holding three separate
  BioNetGen installs reporting "needs BNG2.pl".

- **`parity_checks/tests/test_corpus_manifest_schema.py` had never run, in any
  environment.** It guards on `pytest.importorskip("jsonschema")`, but
  `jsonschema` was declared nowhere — not in `pyproject.toml`, zero occurrences
  in `uv.lock` — so the import always failed and all 13 of its checks always
  skipped. Hand-installing it did not stick either: `uv sync` prunes anything
  absent from the lock, so the package vanished again on the next sync. Now
  declared in the `test` extra; the 13 checks (manifest-vs-schema conformance,
  record-id uniqueness, `patched` iff repairs, and BNGL license/source
  agreement, each across the `biomodels` / `bng_parity` / `dsmts` /
  `rr_parity_sedml` corpora) run and pass. Same class as the GH #27 guard fix
  above: a test that skips everywhere is not a test.

- **`scripts/ship_wheel.py` could not build from the project venv at all.** Its
  canonical `python -m pip wheel . --no-build-isolation` assumes pip is present,
  but `uv venv` — the venv CONTRIBUTING tells you to create (`uv sync --extra
  test`) — ships no pip, so the script died with "No module named pip" before
  building anything. It now falls back to `uv build` when the interpreter has no
  pip, keeping build isolation (an env without pip generally has no
  scikit-build-core either) and passing `--python` so the wheel carries this
  interpreter's ABI tag. Errors clearly when neither pip nor uv is available.

- **`scripts/ship_wheel.py` produced an uninstallable wheel when run on Apple
  Silicon.** It forced `MACOSX_DEPLOYMENT_TARGET=10.15` unconditionally to match
  the `wheelhouse-local` convention — correct on the x86_64 build box, but on
  arm64 it yields a `macosx_10_15_arm64` tag, and since Apple Silicon starts at
  macOS 11.0 neither pip nor uv ever generates a `macosx_10_*_arm64`
  compatibility tag. Such a wheel installs **nowhere**, including on the machine
  that built it (`uv`: "wheel is compatible with macOS (`macosx_10_15_arm64`),
  but you're on macOS (`macosx_26_0_arm64`)"; `pip`: "not a supported wheel on
  this platform"). The target is now chosen per build architecture — 10.15 on
  x86_64, 11.0 on arm64 — so one script is correct on both boxes;
  `platform.machine()` also reports `x86_64` under Rosetta, which is the right
  answer there. Surfaced by `bootstrap_parity_env.py`, whose contract is to
  install bngsim from this repo's own wheel.

- **The GH #27 steady-state regression guards were silently skipping in any
  fresh clone or git worktree.** `python/tests/test_steady_state_gh27.py` read
  its four published models from `benchmarks/suites/ode_fullnet/nets/`, which is
  a *build artifact* of the `ode_fullnet` suite — untracked, and present only in
  a checkout that has run it. Everywhere else `_net()` hit its
  "published net not available" `pytest.skip`, so all four tests reported as
  skipped and the wrong/NaN-root contract went unverified. They now read the
  byte-identical copies vendored (and tracked) under `benchmarks/models/net/ode/`,
  and a net missing from that tracked corpus is an assertion failure rather than
  a skip; only the absence of the whole benchmark tree (testing an installed
  wheel) still skips.

- **Steady-state solver (`method="newton"` / `Simulator.steady_state`) returned
  wrong or NaN roots for several published dose-response models (issue #27).**
  The default seeded KINSOL at the raw initial condition and only fell back to
  integration on non-*convergence*, so it silently returned a spurious root of
  `f(y)=0` (or `NaN`) the dynamics never reach. Three defects are fixed:
  - **Bug 1 — a `NaN` result was reported `converged=True`.** `solve_by_newton`
    marked convergence with `if (residual >= tol) converged = false`; when Newton
    walks a species negative, Hill/power laws yield `NaN`, and `NaN >= tol` is
    false, so `NaN` residuals passed as converged (Gardner 2000 genetic toggle
    returned `conc=[nan, nan]`). The guard is now the positive test
    `!(residual < tol)` (true for `NaN`) plus a finite/non-negative concentration
    check, so an unphysical Newton root never passes.
  - **Bug 2 — Newton converged to a spurious `f(y)=0` root.** The default now
    runs the manuscript's two-tier method in the intended order: **integrate
    first** (a CVODE burst carries the state into the physical root's basin),
    **then** KINSOL polishes. The burst tolerance is adaptive — a KINSOL root is
    accepted only once it is *seed-stable* (two Newton solves from successively
    tighter bursts land on the same state); otherwise integration continues. This
    is correct on multi-root models (e.g. Hlavacek 2001 kinetic proofreading,
    previously ~52% off; Kocieniewski 2012) while still surfacing the root-finding
    speedup on unique-root models (e.g. Barua 2007).
  - **Bug 3 — the dense KINSOL linear-solver setup failed at ~400 species.** The
    reduced steady-state Jacobian is structurally singular for Barua 2013 (409
    sp), so KINSOL cannot factor it at any seed. The two-tier solver falls back to
    integration (a correct answer), stops probing KINSOL after a bounded number of
    failed attempts, and routes the KINSOL context's error log to the null sink so
    the expected failure no longer spams stderr.

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
