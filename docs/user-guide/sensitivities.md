# Sensitivity analysis & gradients

## Forward sensitivity analysis (CVODES)

BNGsim integrates CVODES forward sensitivity analysis to compute ∂Y/∂p —
how species trajectories change with respect to parameters. This enables
parameter identifiability analysis, Fisher information, and gradient-based
optimization.

```python
# Compute sensitivities for specific parameters
sim = bngsim.Simulator(
    model, method="ode",
    sensitivity_params=["kf", "kr"],
)
result = sim.run(t_span=(0, 100), n_points=101)

# Sensitivity tensor: (n_times, n_species, n_params)
print(result.sensitivities.shape)  # (101, 5, 2)
print(result.sensitivity_params)   # ["kf", "kr"]
print(result.has_sensitivities)    # True

# dA/dkf at the last time point
print(result.sensitivities[-1, 0, 0])
```

### Parameters that set an initial condition

When a species' initial condition names a parameter — `R() R0` in the `.net`, or
`R() Rtot` with `Rtot = R0` — that parameter reaches the trajectory through
`x_R(0)`, so the seed `∂x_R(0)/∂R0` is part of the answer. BNGsim differentiates
the IC expression (through nested derived parameters) and seeds it automatically.

That seed describes the initial condition the *model declares*, which is what
`reset()` returns the state to. If you **assign** the species instead —
`set_concentration("R()", 7.0)`, a bulk `set_state`, an externally injected state —
the parameter no longer reaches its initial condition, and the row is dropped: a
literal assignment has `∂x_R(0)/∂θ = 0`, which is also what `set_concentration`
documents about itself. Nothing changes for a model you have not assigned into
(across the 585-model `.net` corpus and 120 SBML models, every one loads with its
live state equal to its baseline).

If the value you assign *does* depend on a fitted parameter — a dose in molecules
computed through a fitted volume — say so, because only you know:

```python
v = dose_nM * 1e-9 * NA * model.get_param("Vecf")
model.set_concentration("L(r)", v)
model.declare_ic_sensitivity({"L(r)": {"Vecf": v / model.get_param("Vecf")}})
```

A declaration is the most specific statement available and wins over both the
expression-derived row and the drop; declaring `{}` pins a row to zero explicitly.
Inside a `parameter_scan` `on_point` hook the derivative is *measured* through the
hook instead, so a dose there needs no declaration — see
[Steady state](steady-state.md#the-dose-an-on_point-hook-applies-issue-111).

### SBML species set by an `<assignmentRule>`

A species an assignment rule defines — `IRS_total`, `InR_active`, the *reported*
quantities of a published SBML model — has no ODE of its own. BNGsim emits its
state slot `fixed` and overwrites the reported column each step from the rule's
live value: an observable when the rule is linear in species, a function
otherwise. Its **sensitivity row is the chain rule through that assignment**, and
`result.sensitivities[:, i, :]` carries it, so it lines up with
`result.species[:, i]` the way every other row does and `Result.gradient` can
score it (issue #221). `output_sensitivities("species:<name>")` returns the same
numbers.

Where the chain rule is not available the row is `NaN`, never `0.0` — a
structural zero is indistinguishable from a measured one, and an optimizer reads
it as a flat objective rather than as a missing term. The run warns naming the
species, and `result.ar_sensitivity_refused` lists them:

```python
result.ar_sensitivity_refused          # frozenset({"stimulus"})
result.output_sensitivities("species:stimulus")
# ValueError: ... has no output sensitivity — uses unsupported construct: if() conditional
```

Two things put a species there: codegen declined the rule's own output
sensitivity (a `piecewise` lowers to an `if()`, which #198 refuses rather than
guess at), or the reported value carries a time-varying volume rescale the
redirect does not model. A `NaN` row does **not** cost you the rest of the
gradient: `gradient`, `sse_gradient`, `chi2_gradient` and
`neg_log_likelihood_gradient` drop these rows wherever your `dL/dY` is exactly
zero, so a fit that never scores that species still gets a number. Weight one and
the gradient is `NaN`, which is the honest answer.

## Parameters that set *when*, not *how fast*

Some fitted parameters never appear in a rate. They set the **time at which the
dynamics change** — the onset of an intervention, the start of a dose, the
moment a promoter flips. BNGL and `.net` models spell this as an `if()` over a
clock:

```
begin functions
    1 rate_X()  if((t>=sigma), k, 0)
end functions
```

where `t` is an observable on a unit-rate counter species (`dT/dt = 1`), which
is how BNG models read simulation time inside a rate law.

`sigma` is reached only through the *condition*, so `∂f/∂sigma` is a clean `0`
inside each branch — sympy drops the boundary delta when a parameter appears
nowhere else. A forward sensitivity that used only the variational source term
would therefore report a flat zero for a parameter the trajectory obviously
depends on. The whole gradient is a finite jump at the crossing `t*`:

```
s(t*⁺) = s(t*⁻) + (f⁻ − f⁺) · ∂t*/∂p
```

BNGsim locates the crossings, computes `∂t*/∂p`, stops the integration *on*
`t*`, and applies the jump. Nothing needs declaring: `.net` and BNGL models
register no discontinuity triggers — only the SBML loader does — so the
conditions are recovered from the function bodies themselves.

```python
res = sim.run(t_span=(0, 10), n_points=101, sensitivity_params=["sigma", "k"])
res.sensitivities[:, x_idx, 0]   # ∂X/∂sigma: exactly 0 before t*, −k after
```

A model with `if()` conditions but **no fitted switch time** yields no records
at all, and its stepping stays bit-for-bit identical to a run without
sensitivities.

### Clock forms that are recognised

The crossing has to be solved for from the condition. These forms are handled:

| Condition shape | Example | Notes |
|---|---|---|
| Affine in the clock | `t >= sigma` | The common case. `∂t*/∂p = ∂(threshold)/∂p`. |
| A monomial in the clock | `t^3 >= thresh` | Any degree. `c·clock^n` is monotonic over a non-negative clock, so it crosses once. |
| A quadratic in the clock | `a*t^2 + b*t >= c` | The first shape with genuinely two crossings, and both are compensated (issue #421). A *general* polynomial of degree 3 or more is declined: a cubic with three real roots has no usable radical form, and from degree 5 there is none at all. |
| Repeating schedule | `rem(t, P) >= d` | A `floor()`-periodic dosing schedule (issue #436). |
| libSBML's `rem()` expansion | `if(sign(a)!=sign(b), …)` | libSBML does not emit `rem()`; it expands it into a sign test over two remainders. Read back to the same schedule (issue #465), so a model gets the same gradient whichever tool wrote its SBML. |
| A guard spelled as a comparison | `(t < 0) == 0` | Resolved the way the equivalent boolean is (issue #473). |
| A threshold selected by clock guards | `t > if(t<365, d_2021, if(t<730, 365+d_2022, …))` | A season written through a year selection. Each branch is a crossing where its threshold falls inside its own branch's range of the clock, so `730+d_2023` crosses at day 862 and nowhere else (issue #545). |

A crossing time that comes out non-real is treated as a crossing that never
happens, rather than crashing the run.

### Derived switch times

A threshold built from other parameters is chain-ruled to the fitted primaries.
With `sigma = t0 + t_delta` and `tau1 = sigma + t_delta2`, a `t0` jump is placed
at **both** the `sigma` and `tau1` crossings, while `t_delta2` gets a jump only
at `tau1`. A threshold the run never reaches contributes an exactly zero column.

### Two switches at the same instant

When two crossings share an instant, each must be charged only its own jump. The
before-branch is read with the other switch's threshold bumped away, so a
coincident pair does not merge into a single doubled jump (issue #375). Models
with distinct switch times are unaffected and read the plain `f⁻ − f⁺`.

Two crossings share an instant when they are within 64 ulp of each other, which
is how far the clock is nudged to read the two branches. Crossings further apart
than that, however close, are stopped at one after the other, fitted or fixed. The same holds for
a crossing just after an output time, an event or the start of the run: its jump
is taken at its own instant, not at the stop before it (issue #737).

An event on the same fitted time as a switch, a dose at `tau` beside
`piecewise(k*X, time >= tau, 0)`, is supported: the event's jump and the
switch's compose (issue #767). The switch may be on `time` or on a counter.

If a requested parameter moves one of the two and not the other, they come
apart under it. Where the event changes what the switched rate law contributes,
the result depends on which comes first and the sensitivity does not exist: a
reset of `X` at `tau` beside `piecewise(k*X, time >= 3, 0)` with `tau = 3`.
bngsim measures whether the event and the switch commute, and refuses the run
where they do not. A pair that commutes runs: a bolus beside an infusion that
starts at the same time.

Several switches can share the event's instant. If all of them move with the
event, two rate laws gated on the dose time, that is one crossing and it runs.
If some do not, the event has to commute with what the instant does as a whole
and with each fitted switch on it alone. Back-to-back infusions with a bolus on
the boundary run; a rate law switched off at `tau` and on again at the literal
3, with a reset at `tau = 3`, is refused. A switch within a few hundred ulp of
an event, fixed or fitted by another parameter, is asked too, so a pair that
does not commute is refused a little before the two times coincide, where the
derivative still exists.

What is not detected returns a number:

- a `floor` step on a fitted switch (issue #944);
- a state-dependent switch that crosses within the integration tolerance of the
  event (issue #945), or on the instant of a time switch (issue #946);
- two time switches of one rate law on one instant, with no event (issue #951);
- a fixed condition whose rounding puts it more than a few hundred ulp from the
  time it is written at, `time + 3000 >= 3003`, which is read where it is
  written;
- a parameter that moves the event by under 1e-9 of its time per unit relative
  change, which counts as not moving it.

### Landing on the crossing

A discontinuity root alone cannot catch these. CVODE tests for a root only on a
step it *accepts*, so where the jump is large enough that the error test rejects
every step containing the crossing, `t` creeps to the last double below `t*` and
wedges at `t + h == t` without the root ever firing (issue #305). BNGsim
therefore registers the crossing times as explicit stops, so the integrator
lands on each one and restarts on the far side. This applies to a rate law
switching on simulation time (issue #445) and to one switching on a counter
species (issue #443), where the counter is landed exactly on its threshold so
the restart reads the after-branch.

These stops are added for any model carrying such a switch, sensitivities or
not, because stepping over the discontinuity was never correct. A crossing whose
time the state decides, such as `time() >= 4*S`, has no stop that can be placed
before the run. Instead, each comparison over the state in a rate law's `if()`
condition is a CVODE root in every run (issue #897). A run without sensitivities
applies no jump there, and it restarts only at a genuine crossing. The solver's
own trajectory must go past the threshold by more than the requested tolerance
can blur, and the flow arriving at the threshold, read on the near side, must
stay finite as the threshold is approached and carry the state across it. A
trajectory that only approaches the threshold, parked beside it or relaxing onto
it, can cross it on the solver's interpolant alone, within its own error, so the
run steps on there as it did before these roots existed. A window narrower than
the tolerance can resolve is therefore not guaranteed, as it is not in main.

Stepping on can leave a run pinned on the threshold: a step long enough to move
the state across by one ulp carries the rate law's jump into an error test it
fails, and a shorter one leaves the state where it is. A slow approach does it
with one condition: a species that rises by 1e-8 of itself a unit of time. Where
the solver has spent a whole batch of steps that way, and the flow reaches the
threshold and would have crossed in the time the state has been seen there,
the state is put the few ulp across, each species the threshold reads that the
flow moves by a few ulp of its own, and the run restarts there (issue #928). This is
for `.net` models: a `piecewise` condition on a species in an SBML model is not
a state switch of a plain run, and such a run still stalls.

How late the crossing is depends on how slow the approach is and on
`max_steps`. The crossing time is known no better than the threshold species
is, its tolerance over its rate, and a pinned run spends a batch of
`max_steps` steps before it is put across. A state that slides along the
threshold, with both branches pointing into it, is not moved, and neither is
one that comes to rest just short of it.

Issue #904 lists what this does not yet cover:
- a comparison used as a number outside `if()`, such as `k*(X > 1)`;
- a single comparison whose residual turns back within one step, such as
  `abs(X - 5) < 0.01`;
- `steady_state()`.

### A rate that rises from zero at the crossing

A switch does not have to jump. A pulse that rises continuously from its onset,
such as `k0 + k1*s^(a-1)*(1-s)` over a window `s = (t - on)/D` that is 0
outside it, is finite and continuous for any `a > 1`, so there is no jump to
apply. Its derivative with respect to the onset is another matter. It goes as
`s^(a-2)`, which for `1 < a < 2` is infinite at the onset and unbounded just
past it. The sensitivity itself stays finite, because that singularity is
integrable, but no polynomial step resolves the forcing, and most of its
integral lies within a few ulp of the onset when `a` is close to 1. Integrated
as it stands, the onset column fails at the onset, stalls just past it, or
finishes measurably wrong (issue #545; 2% at `a = 1.2` on one model).

So past such a crossing BNGsim integrates a different column. If the crossing
moves at `c = ∂t*/∂on`, then `V = S + c·f` obeys `V' = J·V + β`, where `β` is the
derivative of `f` along `on` and the clock *together*. That shift leaves `s`
unchanged, so the singular terms of `∂f/∂on` and `c·∂f/∂t` cancel in `β`, and
`V` is as smooth as the state. The code generator emits `β` for each parameter
whose crossing makes such a power singular: a base that can reach 0, raised to an
exponent between 0 and 1 or to one that is not a number, such as `a-1`. At that
crossing the solver switches the column to `V`, and at the next restart it
switches back to `S`. `S = V − c·f` is what every output reports. A crossing
that is *approached* through such a power, the closing edge of a window
`s*(1-s)^(a-1)`, needs the column in `V` before it: the solver switches at a
stop it takes shortly before that crossing, and switches a column in `V` back
to `S` shortly before any crossing that is not the column's own (issue #760).
Two cases are refused: a restart within about 7e-10 of the time of such an
edge, where no stop can stand off from it, and another rate-law condition that
crosses on the edge itself. A model that has an event keeps every column in
`S`, and is 0.4% off at such an edge (issue #958).
A parameter that moves no such crossing keeps its plain column. A model with no such power
emits the code it always did, and that includes a logistic onset
`1/(1+exp(-k*(t-on)))`, whose base is never 0.

`β` carries the clock's share for every rate that reads a counter clock, not
only for a rate law that names it: mass action with the clock species as a
reactant, a rate law multiplied by it, a rate law reading an observable that
sums it with other species, and a Michaelis–Menten rate with the clock as its
enzyme or its substrate (issue #749). A derived onset, `on = lam*1.0`, gets a comoving column of
its own when it is requested, as each parameter it is defined from does
(issue #750).

The comoving column is used when:

- the run has the analytic sensitivity RHS (`sim.has_analytic_sens_rhs`);
- the model has no events (a state-dependent switch or an SBML discontinuity
  trigger is fine);
- the crossing moves at the shift the generator derived from the power's own
  base. `t - on` shifts at `c = 1`, and so does a season's `t - (730 + d_2023)`
  written through a year selection.

Elsewhere the plain column meets the singular forcing as before, and the
failure says so. It names the sensitivity column and `∂f/∂on` when the RHS goes
non-finite at the onset, or the restart where the step gave out. An exponent of
2 or more (`a >= 2`) keeps the derivative finite, and leaving the onset
parameter out of `sensitivity_params` avoids it. `BNGSIM_SENS_COMOVING=0` keeps
every column plain, for comparison.

### What is declined

A condition whose crossing time moves with the *state*, such as `if(X < 1, …)`
or a window `(X > lo) && (X < hi)`, is not declined. Each comparison's residual
is registered as a root, and the saltation jump `(f⁻ − f⁺)·dt*/dθ` is applied
where it crosses (issue #150). A crossing where the rate law is continuous, such
as a clamp whose live branch is 0 at its threshold, needs no jump. That is
decided from the reactions whose rate law reads the condition, so a fast flux
elsewhere in the model cannot make a real jump read as continuous, nor roundoff
there read as a jump (issue #763). Two independent thresholds that the
trajectory crosses within about 1e-13 of each other in time are one stop for
the solver. If both jump and the requested columns move them apart, the run is
refused, because one jump would otherwise be moved with the other's `dt*/dθ`.
A state that slides along the surface is refused too (issue #926): with
`if(S < 1, amp, -amp)` both branches point into `S = 1`, the state stays on it,
and the sensitivity there is that of neither branch.
A state that stays within a few tolerances of such a surface without reaching
it, turning back five tolerances short, is refused as well: the run cannot
tell it from one that touches. So is a run that ends within about a tolerance
of the surface on its way there.
So is a state held just short of a surface it approaches too slowly for any
step to cross (issue #952): `if(A > thr, kb, 0)` with `A` rising at 1e-10 a unit
of time. The step that moves `A` by one ulp fails its error test on the jump,
so no crossing is located and there is no crossing time for the sensitivities
to move with. A run without sensitivities is carried across.
A conjunction or a negation is split into its comparisons first. What the
analytic path declines is a crossing nothing can locate: a comparison outside any
`if()`, such as `k*(X > 1)`, and one whose sides
are themselves comparisons, such as `(X > 1) == (Y > 1)`. A parameter that both
sets a switch time and acts inside a branch is answered on the analytic path,
which adds the in-branch term to the jump (issue #358), and rejected on the
fallback, which cannot.

An event that assigns a value which is not smooth at the point the event reads
it is refused. The jump needs the value's derivative in each parameter, in the
state, and in the fire time, and takes each as a central difference over a
millionth of what it moves. Across a step, a bend, or a value that turns inside
that span, the difference is not a derivative to a part in a thousand: `u := piecewise(5, time >= T0 + 1, 0)`
assigned at `time >= T0 + 1` gave dB/dT0 in the millions, for 0 (issue #915). The
error names the event, the species, and what the value is not smooth in. Move
the step away from the event, or leave the parameters that reach it out of
`sensitivity_params`. A step or a bend that moves exactly with the event does
have a derivative, and is refused all the same. The test is of the differences
themselves, to a part in a thousand: a bend that changes the value's slope in
what is moved by less than about 0.8% is not seen, and is differenced across as
before.

A decline is never silent. Ask the Simulator directly:

```python
sim.has_analytic_sens_rhs      # False when the run falls back
sim.sens_rhs_decline_reason    # why, in words, or None
```

The fallback is CVODES' own difference quotient. It is slower, and it is right
where every rate law is continuous in the state along the run. It is not right
across a jump: the quotient reads the rate law at the state moved along each
sensitivity, which just short of a surface the state crosses is on the other
branch, so a column takes part of the jump before the crossing, by more the
looser the tolerance.

A time course on the fallback is therefore refused, before it starts, for a
model with (issues #938, #932):

- a rate-law condition that reads the state, `if(X < thr, kb, 0)`, whatever the
  law does where the condition flips;
- a sign or a step written by dividing by an `abs`, `max` or `min` where it is 0,
  or by what it flips on: `(thr - X)/abs(thr - X)`, `max(X - thr, 0)/(X - thr)`;
- a condition on a counter species that a requested column moves;
- a step call on the state, `floor(X)`, or a table function read as a step and
  indexed by an observable or by a function;
- a step call on time, or on a counter nothing moves, whose argument reads a
  requested parameter, `floor(time()/P)` with `P` requested;
- a comparison over parameters alone that a requested one is close to flipping:
  `if(n > 1, kb, 0)` with `n` requested and within a quarter of itself of 1.
  The quotient moves a parameter as it moves the state, by up to its size times
  the root of the relative tolerance. Likewise an equality on a requested
  parameter that holds, and a step call or a step table on one.

The refusal goes by what the rate laws' text says and by the sign of each
parameter. Nothing is evaluated, so it costs the same before every run, and it
does not tell a law that jumps where its condition flips from one that only
bends there: `if(X < thr, kb*(thr - X), 0)` is refused, though the quotient is
right across a bend. Write a bend with `max` or `min`, which are continuous by
what they are and run: `kb*max(thr - X, 0)`, `max(0, min(X, n))`,
`v/max(X, 0.01)`.

A condition on literal time runs, and so does a steady-state solve. What the
scan does not see is a jump written with no condition and not as one of the
quotients above: `sqrt(X*X)/X`, `tanh(1e9*(X - thr))`, and a pole cut off on
both sides, `min(max(k/(X - thr), -5), 5)`.
See the [PyBNF guide](pybnf.md#ask-each-model-whether-its-gradient-is-analytic)
for using this to triage a fit.

## Parallel sensitivity computation

For models with many parameters (Np), computing all sensitivities serially
is expensive (O(Np) overhead per CVODE step). `compute_all_sensitivities()`
splits parameters into chunks and runs them in parallel via thread pool:

```python
sim = bngsim.Simulator(model, method="ode")

# Compute full sensitivity tensor using parallel chunks
result = sim.compute_all_sensitivities(
    t_span=(0, 100),
    n_points=101,
    chunk_size=2,     # 2 params per CVODES job (optimal)
    n_workers=8,      # parallel threads
)

# Full tensor: (n_times, n_species, n_params)
print(result.sensitivities.shape)  # (101, 149, 40)
print(result.sensitivity_params)   # == model.primary_param_names
```

Each chunk clones the model (thread-safe deep copy) and runs an independent
CVODES instance. The GIL is released during C++ CVODE integration, so threads
achieve real parallelism. Near-linear speedup from 1→2→4→8 workers.

### What the default column set is, and why it is narrower than `param_names`

`params=None` means *every independent knob*, which is
`model.primary_param_names` — not `model.param_names` (issue #203).
`result.sensitivity_params` is always the authority on what the columns are, and
anything dropped is named in a warning. Two classes come out:

- **Derived (expression-backed) parameters** — `_rateLaw1 = chi*kon` from a
  compound BNGL rate law, `_rateLaw_R16_fwd = alpha*konBT` from an SBML kinetic
  law. Such a parameter reaches the trajectory only through the primaries it is
  built from, and *their* columns are total derivatives through it. So the two
  columns are the same physical effect twice, in exact proportion
  `d(derived)/d(primary)`, and `Result.gradient` contracts the whole parameter
  axis into one vector an optimizer then steps along in every coordinate at
  once. Roughly one SBML model in five carries some (279 of the 1,291 loadable
  rr_parity models, 9,524 parameters in total).
- **Synthesized slots** (`model.param_is_internal`) — parameter slots bngsim
  created for its own bookkeeping rather than ones the model declared.
  `set_param` refuses a value-changing write to either kind, so neither is a
  coordinate that can move on its own. `_V0_<comp>` is bngsim's record of a
  compartment's size at load, which the rate constants in that compartment are
  normalised against — differentiate the compartment size itself, an ordinary
  writable parameter. And every **function** has one, holding the value it last
  evaluated to; the engine rewrites it from the function's own expression before
  every derivative evaluation, so that column is identically zero (issue #227) —
  differentiate the parameters the function's expression reads.

Naming a derived parameter — or `_V0_<comp>` — in `params=[...]` still returns
its column: an explicit ask is a statement that you want that derivative *on its
own terms*, treating the parameter as a free axis. That is exactly what
`bngsim.jax.differentiable_solve(..., flat=True)` asks for, and why the default
here (`flat=False`'s list) and that opt-in now agree end to end. (`flat=True`
leaves out the synthesized slots too, for the same reason: its vector is one
`set_param` per name, and those writes are refused.)

**A function's slot is the exception, and it raises** (issue #329). "On its own
terms" needs terms to exist: a derived parameter has them, because
`set_param(..., force_override=True)` pins it and the pin survives. A function's
slot has none — the next `evaluate_functions()` overwrites it whatever you do,
which is why `set_param` refuses it outright and `force_override` does not help.
So the column can only ever be identically zero, and asking for it by name in
`sensitivity_params=`, `params=[...]`, or `steady_state(sensitivity_params=...)`
is refused with a `ValueError` naming the expression to differentiate instead.

This matters most on **assignment-rule-driven SBML**, where the rule targets can
outnumber the real knobs several to one — 38 of 46 parameters in
`BIOMD0000000126`, 35 of 38 in `BIOMD0000000266`. Passing `model.param_names`
wholesale used to return a tensor that was mostly structural zeros with nothing
marking which columns those were. Pass `model.primary_param_names`, or let
`params=None` pick it.

The chain rule *through* an `<assignmentRule>` is unaffected: a parameter whose
only route to the right-hand side is a rule target differentiates normally, since
bngsim lowers the rule to a function the emitted sensitivity RHS differentiates
like any other expression. It is the rule's **target** that is not a column, not
the parameters underneath it.

What is *not* dropped is a constant written as arithmetic — `gamma 1/7`,
`pi 2*asin(1)`, `c6 ln(2)/120`. Those name nothing, so there is no primary
underneath them carrying their effect, and they are ordinary knobs (issue #227).

## Fisher Information Matrix

The Fisher Information Matrix (FIM) quantifies how much information observed
species trajectories carry about each parameter — the foundation for
parameter identifiability analysis and experimental design.

```python
# Compute FIM from sensitivity data
fim = result.fisher_information(sigma=0.1)  # scalar noise σ
print(fim.shape)  # (n_params, n_params)

# Per-species noise
sigma_per_species = np.array([0.1, 0.5, 1.0, ...])
fim = result.fisher_information(sigma=sigma_per_species)

# Identifiability diagnostics
print(np.linalg.cond(fim))        # condition number
eigvals = np.linalg.eigvalsh(fim)
print(eigvals[:3])                 # smallest eigenvalues → least identifiable
```

The FIM is the Cramér–Rao lower bound on parameter covariance:
Cov(p̂) ≥ FIM⁻¹. Large diagonal entries indicate identifiable parameters;
near-zero eigenvalues indicate practical non-identifiability.

That last reading is only sound if the columns are independent, which is what
the default column set above guarantees. `Sᵀ Σ⁻¹ S` over a parameter axis that
contains both a derived parameter and a primary underneath it is rank-deficient
**by construction** — the two columns are exactly proportional, so there is a
null direction that says nothing about the model or the data. Before #203 that
was the default on any model with derived parameters.

## Parameter gradients for optimization

`Result.gradient()` computes ∇_p L from the sensitivity tensor and a
user-supplied loss function, enabling gradient-based optimization:

```python
import numpy as np
from scipy.optimize import minimize

data = np.load("experimental_data.npy")  # (n_times, n_species)

# The fitted vector is the default column set, so `grad` lines up with `p_vec`.
names = model.primary_param_names

def objective(p_vec):
    # Set parameters and simulate with sensitivities
    model.set_params(dict(zip(names, p_vec)))
    model.reset()
    result = sim.compute_all_sensitivities(
        t_span=(0, 100), n_points=101,
        n_workers=8,
    )
    assert result.sensitivity_params == names

    # Compute loss and gradient
    loss = np.sum((result.species - data) ** 2)
    grad = result.gradient(
        lambda species, time: 2 * (species - data)
    )
    return loss, grad

# L-BFGS-B optimization with analytical gradients
opt = minimize(objective, x0=[model.get_param(n) for n in names],
               method='L-BFGS-B', jac=True)
```

`gradient()` sums over time and species but *not* over parameters — the
double-counting hazard is downstream, in the optimizer, which steps along every
coordinate of the returned vector at once. That is only meaningful if the
coordinates are independent, which is what the default column set above
guarantees and a hand-written `sensitivity_params=` list does not: if such a
list names both `_rateLaw1` and the `kon` underneath it, do not hand the
resulting vector to an optimizer over both.

The gradient computation is O(n_times × n_species × n_params) — a single
matrix multiply per time point. Combined with parallel
`compute_all_sensitivities()`, the total cost of loss + gradient is
dominated by the CVODES solve, not the gradient algebra.

**SBML compartment sizes are writable and differentiable** (issues #164, #170).
On an SBML model `model.param_names` includes the compartments. A
compartment size is now an ordinary writable parameter — `set_param("Liver", v)`
re-derives everything the volume decides (the amount↔concentration conversion,
an amount-declared initial condition, the mass-action scalar, the SSA propensity
volume, and any `<initialAssignment>` that reads the size — the PBPK idiom of
copying each compartment into its own parameter) and reproduces *reloading the
model at that size*, bit for bit. A volume scan or a gradient-free fit needs
nothing special. The generated C reads the
volume from `p[]` rather than baking it, so the emitted source does not depend on
the load-time size and the write lands on `codegen=True` and on an
already-compiled `.so` too — including a write that arrives mid-scan, after the
source was generated.

The **gradient** followed in stage 3: `d/dV` now carries the storage half as well
as the kinetic-law half — including the initial-condition seed, which is
`-amount/V²` for an amount-declared species rather than zero — so `Liver` is an
ordinary column of `compute_all_sensitivities()` and `sensitivity_params=["Liver"]`
is accepted. `_V0_Liver` is not: that is bngsim's record of the load-time size
rather than the volume, `set_param` refuses to move it, and it is one of the
things the `params=None` default drops (see
[the default column set](#what-the-default-column-set-is-and-why-it-is-narrower-than-param_names)).

A handful of compartments still cannot be written, and those keep the original
refusal — `compute_all_sensitivities()` skips the column with a warning and
`sensitivity_params=["Liver"]` raises, because a column is exactly as trustworthy
as the write is. `model.unwritable_compartment_size_params` lists them and the
error names the reason per size. Reload at the size instead:

```python
m = bngsim.Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v})
```

...or difference over the rebuild, which is exact:

```python
def dloss_dV(v, h):
    up = bngsim.Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v + h})
    dn = bngsim.Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v - h})
    return (loss(up) - loss(dn)) / (2 * h)
```

## Differentiable ODE solving with JAX

BNGsim provides a JAX-traceable ODE solver via `bngsim.jax.differentiable_solve`.
This registers CVODE as a JAX custom primitive with a `custom_jvp` rule that
dispatches to CVODES forward sensitivities — combining SUNDIALS-quality stiff ODE
solving (0.1ms) with JAX's composable automatic differentiation (`jax.grad`,
`jax.value_and_grad`, `jax.jacfwd`).

```python
import jax
import jax.numpy as jnp
from bngsim.jax import differentiable_solve

model = bngsim.Model.from_net("model.net")

# Differentiate over primary parameters only (default). Derived
# ConstantExpression parameters such as BNG2.pl-emitted ``_rateLaw{N}``
# (for compound BNGL rate laws like ``chi*kon``) are recomputed from
# their primaries automatically, so ``jax.grad`` returns gradients
# with respect to ``model.primary_param_names`` with the chain rule
# through derived expressions correctly applied.
p0 = jnp.array(
    [model.get_param(n) for n in model.primary_param_names]
)

# Forward solve (no differentiation)
Y = differentiable_solve(model, p0, (0, 100), 101)

# Gradient of a loss function w.r.t. primary parameters
data = jnp.load("experimental_data.npy")

def loss(p):
    Y = differentiable_solve(model, p, (0, 100), 101)
    return jnp.sum((Y - data) ** 2)

grad = jax.grad(loss)(p0)                    # parameter gradient
val, grad = jax.value_and_grad(loss)(p0)     # loss + gradient

# Full sensitivity matrix via jacfwd
def solve_flat(p):
    return differentiable_solve(model, p, (0, 100), 101).ravel()

J = jax.jacfwd(solve_flat)(p0)  # (n_times*n_species, n_primary_params)

# Legacy / advanced: treat every parameter (including derived
# ``_rateLaw{N}``) as an independent coordinate. Use only when you
# really want to vary derived parameters independently of their
# defining expression. The vector is ``param_names`` minus the
# synthesized slots ``set_param`` refuses to write (issues #170, #227).
internal = [n for n, f in zip(model.param_names, model.param_is_internal) if f]
p_flat = jnp.array(
    [model.get_param(n) for n in model.param_names if n not in internal]
)
Y_flat = differentiable_solve(model, p_flat, (0, 100), 101, flat=True)
```

Requires: `pip install 'bngsim[jax]'`

**How it works**: The `@jax.custom_jvp` rule runs CVODES once per JVP call,
computing the primal solution and forward sensitivities simultaneously (single
solve, not two). The sensitivity tensor is contracted with the tangent vector
via `jnp.einsum('tsp,p->ts', sens, dp)`.

**Performance**: ~1.2× overhead vs plain ODE solve for large models — 23,000×
faster than Diffrax in internal benchmarking. Each call clones the model internally for
thread safety. Solver options (`rtol`, `atol`, `max_steps`) are passed through
as keyword arguments.

**When to use**: For JAX ecosystem integration (`optax`, `numpyro`, `blackjax`,
`scipy.optimize`). For non-JAX gradient computation, use `Result.gradient()`
which is lower-overhead and doesn't require JAX.

## Built-in objective gradients

BNGsim provides built-in gradient methods for the most common parameter
estimation objectives, eliminating the need to manually derive `dL/dY`:

```python
result = sim.compute_all_sensitivities(
    t_span=(0, 100), n_points=101, chunk_size=2, n_workers=8,
)

# Sum of squared errors (most common)
loss, grad = result.sse_gradient(data)

# Chi-squared (weighted by measurement noise)
loss, grad = result.chi2_gradient(data, sigma=0.1)
loss, grad = result.chi2_gradient(data, sigma=per_species_sigma)

# Negative Gaussian log-likelihood (includes constant term)
nll, grad = result.neg_log_likelihood_gradient(data, sigma=0.1)

# Partial observation (only fit species 0 and 2)
loss, grad = result.sse_gradient(
    data_subset, species_indices=[0, 2]
)

# Direct use with scipy L-BFGS-B
from scipy.optimize import minimize
def objective(p_vec):
    model.set_params(dict(zip(param_names, p_vec)))
    model.reset()
    result = sim.compute_all_sensitivities(...)
    return result.sse_gradient(data)  # returns (loss, grad)
opt = minimize(objective, x0, method='L-BFGS-B', jac=True)
```

All methods return `(loss_value, gradient_vector)` — the format expected by
`scipy.optimize.minimize(..., jac=True)`. For custom objectives not covered
by the built-ins, use `Result.gradient(loss_fn)` with a user-supplied
`dL/dY` function, or the JAX bridge for automatic differentiation.

### Adding a new built-in objective (Developer Guide)

The pattern for adding a new objective gradient method to `Result` is:

1. **Derive `dL/dY`** — the partial derivative of your loss function with
   respect to each species value at each time point. This is a
   `(n_times, n_species)` array.

2. **Add a method** to the `Result` class in `bngsim/python/bngsim/_result.py`.

3. **Contract with sensitivity tensor** — the parameter gradient is
   `∇_p L = Σ_t (dY/dp)^T · (dL/dY)_t`, computed as a loop over time points.

**Worked example: negative binomial log-likelihood** (for count data in
epidemiological models where `Y` is the expected count and `D` is observed):

```python
def negbinom_gradient(
    self,
    data: NDArray[np.float64],
    r: Union[float, NDArray[np.float64]],
    *,
    species_indices: Optional[list[int]] = None,
) -> tuple[float, NDArray[np.float64]]:
    """Negative binomial NLL and parameter gradient.

    NLL = -Σ_{t,i} [D*log(p) + r*log(1-p)]  (up to constants)
    where p = Y/(Y+r), Y = model prediction, D = observed count.

    dL/dY = (D - Y*r/(Y+r)) * (-r/(Y+r)^2)
          = r*(D - Y) / (Y*(Y+r))
    """
    if not self.has_sensitivities:
        raise ValueError("No sensitivity data.")

    data = np.asarray(data, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    Y = self._species
    sens = self._sensitivities

    if species_indices is not None:
        Y = Y[:, species_indices]
        sens = sens[:, species_indices, :]

    # p = Y / (Y + r)
    p = Y / (Y + r_arr)
    p = np.clip(p, 1e-15, 1 - 1e-15)  # numerical safety

    # NLL (negative log-likelihood, dropping constant terms)
    nll = -float(np.sum(
        data * np.log(p) + r_arr * np.log(1 - p)
    ))

    # dL/dY = r * (Y - D) / (Y * (Y + r))
    dL_dY = r_arr * (Y - data) / (Y * (Y + r_arr) + 1e-30)

    # Contract with sensitivity tensor
    nt = sens.shape[0]
    np_ = sens.shape[2]
    grad = np.zeros(np_, dtype=np.float64)
    for t in range(nt):
        grad += sens[t].T @ dL_dY[t]

    return nll, grad
```

**Key rules:**
- The method must check `self.has_sensitivities` and raise `ValueError` if missing.
- Support `species_indices` for partial observation.
- Return `(loss, gradient)` tuple — both are always computed together.
- The gradient contraction loop `for t in range(nt): grad += sens[t].T @ dL_dY[t]`
  is the same for ALL objectives — only `dL_dY` changes.
- Add tests in `test_objective_gradients.py` that verify:
  (a) shape, (b) zero-residual gradient is zero, (c) consistency with
  `Result.gradient()` using the same `dL/dY` manually, (d) error handling.
