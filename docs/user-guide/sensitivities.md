---
summary: "Forward sensitivities of an ODE run to parameters and initial conditions, all parameters at once, Fisher information, gradients for fitting, the JAX bridge, and how BNGsim handles rate laws that switch."
---

# Sensitivity analysis & gradients

BNGsim computes forward sensitivities with CVODES: how each species changes
with each parameter, `∂x_i(t)/∂p_k`, along an ODE run. This page covers how to
ask for them, how to use them for Fisher information and fitting, and what
BNGsim does when a rate law switches during the run.

## The main path

Name the parameters when you build the Simulator, then run:

```python
import numpy as np
import bngsim

model = bngsim.examples.load("binding")  # A + B <-> C, at rates kf and kr

sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kf", "kr"])
result = sim.run(t_span=(0, 10), n_points=11)

print(result.sensitivities.shape)  # (11, 3, 2): (n_times, n_species, n_params)
print(result.sensitivity_params)   # ['kf', 'kr']
print(result.has_sensitivities)    # True

# dA/dkf at the last time point
print(result.sensitivities[-1, 0, 0])
```

`result.sensitivities_observables` holds the same derivatives for the
observables.

What to watch out for:

- **ODE only.** Sensitivities need `method="ode"`.
- **A C compiler.** BNGsim compiles the model's derivatives. Without a
  compiler, building the Simulator with `sensitivity_params` raises
  `bngsim.CodegenToolchainError`. With `codegen=False`, or with
  `BNGSIM_NO_CODEGEN` set, it raises `ValueError`.
- **Differentiate the primary parameters.** `model.primary_param_names` lists
  the parameters you can move on their own. A parameter that a function owns
  (a BNGL `functions` entry, or an SBML parameter set by an
  `<assignmentRule>`) is rewritten from its expression before every step, so
  its column could only be zero. Naming one raises `ValueError`, which names
  the expression to differentiate instead.
- **A refusal is an answer.** Where BNGsim cannot compute a column correctly,
  it raises `bngsim.SensitivityUnsupportedError` rather than return a wrong
  number. The error names the construct.
- **Check the path.** `sim.has_analytic_sens_rhs` says whether the run uses
  the compiled analytic derivatives, and `sim.sens_rhs_decline_reason` says why
  not, in words. See [The fallback](#the-fallback-difference-quotient).

## Initial-condition sensitivities

To differentiate with respect to a species' starting value, name the species:

```python
model.reset()  # sensitivities need a fresh start, and the run above moved the state
sim = bngsim.Simulator(model, sensitivity_ic=["A(b)"])
result = sim.run(t_span=(0, 10), n_points=11)
result.sensitivities_ic  # (n_times, n_species, n_ic): ∂x_i(t)/∂x_k(0)
```

This helps when you fit a starting value through your own
reparameterization, such as `model.set_concentration("Epo", 10**theta)`.

### Parameters that set an initial condition

When a species' initial condition names a parameter (`R() R0` in a `.net`
file, or `R() Rtot` with `Rtot = R0`), that parameter's column includes its
effect through the initial condition. BNGsim differentiates the
initial-condition expression, through any derived parameters, for you.

That holds for the initial condition the model declares, which is what
`reset()` returns to. If you assign the species instead, with
`set_concentration`, `set_state` or a state from elsewhere, the parameter no
longer sets it, and that part of the column is 0.

If the value you assign does depend on a fitted parameter, say so:

```python
model.reset()                       # a fresh start again
dose = 2.5                          # a dose of A, in units of A0
v = dose * model.get_param("A0")
model.set_concentration("A(b)", v)
model.declare_ic_sensitivity({"A(b)": {"A0": dose}})   # dv/dA0, since v = dose * A0
```

A declaration wins over both the derived row and the zero. Declaring `{}` sets
a row to zero on purpose. Inside a `parameter_scan` `on_point` hook, BNGsim
measures the derivative through the hook, so a dose set there needs no
declaration. See
[Steady state](steady-state.md#the-dose-an-on_point-hook-applies).

## All parameters at once

`compute_all_sensitivities()` splits the parameters into chunks and runs them
in parallel threads. Each chunk runs on its own clone of the model.

```python
model.reset()  # a fresh start, from the initial conditions
sim = bngsim.Simulator(model, method="ode")
result = sim.compute_all_sensitivities(
    t_span=(0, 100),
    n_points=101,
    chunk_size=2,  # parameters per CVODES run (the default)
    n_workers=8,   # threads
)
print(result.sensitivity_params)  # == model.primary_param_names
```

### Which columns you get by default

With no `params=` list, the columns are `model.primary_param_names`, not
`model.param_names`. `result.sensitivity_params` always says what the columns
are, and a warning names anything left out. Two kinds are left out:

- **Derived parameters**, such as `_rateLaw1 = chi*kon` from a compound BNGL
  rate law. Their effect is already in the columns of the primaries they are
  built from, so a column for both would count it twice. Writing one with
  `set_param` pins it to that value and cuts it off from its inputs (see
  [below](#a-derived-parameter-set-after-the-simulator-is-built)).
- **Internal slots** (`model.param_is_internal`), which BNGsim creates for its
  own use. `set_param` refuses to change them, so they are not coordinates you
  can move. `_V0_<comp>` records a compartment's size at load; differentiate
  the compartment size itself instead. A function's slot is always zero;
  differentiate the parameters its expression reads.

This matters most for SBML models driven by assignment rules, where the rules'
targets, which are function slots, can outnumber the real parameters several
to one.

You can still name a derived parameter, or `_V0_<comp>`, in `params=[...]`.
You then get its column, treating it as a free axis. A function's slot is the
exception: naming it in `sensitivity_params=`, `params=` or
`steady_state(sensitivity_params=...)` raises `ValueError`.

The chain rule through an `<assignmentRule>` still works: a parameter that
reaches the model only through a rule differentiates normally. It is the
rule's target that is not a column, not the parameters it reads.

A constant written as arithmetic, such as `gamma 1/7` or `c6 ln(2)/120`, is
not derived from anything. It is an ordinary parameter and keeps its column.

## Fisher information

```python
fim = result.fisher_information(sigma=0.1)   # one noise level
sigma_per_species = np.array([0.1, 0.1, 0.5])
fim = result.fisher_information(sigma=sigma_per_species)
print(fim.shape)  # (n_params, n_params)

print(np.linalg.cond(fim))
print(np.linalg.eigvalsh(fim)[:3])  # smallest: least identifiable directions
```

The inverse of the Fisher information is the Cramér–Rao lower bound on the
parameter covariance. Large diagonal entries mean identifiable parameters.
Near-zero eigenvalues mean practical non-identifiability, but only if the
columns are independent. The default column set makes sure of that. A list
that names both a derived parameter and a primary it is built from gives a
singular matrix that says nothing about the model or the data.

## Gradients for fitting

The built-in objectives return `(loss, gradient)`, which is what
`scipy.optimize.minimize(..., jac=True)` expects:

```python
result = sim.compute_all_sensitivities(t_span=(0, 100), n_points=101)

# Data to fit, one row per time point. Here it is the run itself with noise
# added; yours would be measured.
rng = np.random.default_rng(1)
data = result.species + rng.normal(0, 0.1, result.species.shape)
data_subset = data[:, [0, 2]]          # species 0 and 2 only
obs_data = result.observables + rng.normal(0, 0.1, result.observables.shape)

loss, grad = result.sse_gradient(data)                       # sum of squares
loss, grad = result.chi2_gradient(data, sigma=0.1)           # weighted by noise
nll, grad = result.neg_log_likelihood_gradient(data, sigma=0.1)

# Fit only some species
loss, grad = result.sse_gradient(data_subset, species_indices=[0, 2])

# Data measured on observables
loss, grad = result.sse_gradient(obs_data, target="observables")
loss, grad = result.chi2_gradient(
    obs_data[:, [1]], sigma=0.1, target="observables", observable_indices=[1]
)
```

`target` says what the data is compared with: `"species"` (the default) or
`"observables"`. The shape of the data does not decide it. On a model with as
many observables as species, observable data passed without
`target="observables"` is scored against the species, with no error.

For any other loss, pass `dL/dY` to `result.gradient()`:

```python
from scipy.optimize import minimize

names = model.primary_param_names  # the default columns, so grad lines up

def objective(p_vec):
    model.set_params(dict(zip(names, p_vec)))
    model.reset()
    result = sim.compute_all_sensitivities(t_span=(0, 100), n_points=101)
    loss = np.sum((result.species - data) ** 2)
    grad = result.gradient(lambda species, time: 2 * (species - data))
    return loss, grad

opt = minimize(objective, x0=[model.get_param(n) for n in names],
               method="L-BFGS-B", jac=True,
               bounds=[(0, None)] * len(names))  # rate constants and amounts stay >= 0
```

`gradient()` sums over time and species, not over parameters. An optimizer
steps along every coordinate of the vector at once, so the coordinates must be
independent. Don't fit a vector from a list that names both `_rateLaw1` and
the `kon` it is built from.

A gradient ignores rows that your `dL/dY` weights by exactly zero. So a `NaN`
row (see [below](#species-set-by-an-assignmentrule)) does not spoil a fit
that never scores that species. Weight it, and the gradient is `NaN`.

## Differentiable solving with JAX

`bngsim.jax.differentiable_solve` makes a CVODE solve a JAX primitive. Its
derivative rule runs CVODES forward sensitivities, so `jax.grad`,
`jax.value_and_grad` and `jax.jacfwd` work through it. It needs
`pip install "bngsim[jax]"`.

```python
import jax
import jax.numpy as jnp
from bngsim.jax import differentiable_solve

model = bngsim.examples.load("binding")  # a fresh copy, at its initial conditions

# The vector is ordered as model.primary_param_names. Derived parameters
# follow their primaries, so the gradient includes the chain rule through them.
p0 = jnp.array([model.get_param(n) for n in model.primary_param_names])

Y = differentiable_solve(model, p0, (0, 100), 101)  # (n_times, n_species)

data = jnp.asarray(data)  # the data fitted above, (n_times, n_species)

def loss(p):
    Y = differentiable_solve(model, p, (0, 100), 101)
    return jnp.sum((Y - data) ** 2)

grad = jax.grad(loss)(p0)
val, grad = jax.value_and_grad(loss)(p0)

def solve_flat(p):
    return differentiable_solve(model, p, (0, 100), 101).ravel()

J = jax.jacfwd(solve_flat)(p0)  # (n_times*n_species, n_primary_params)

# flat=True treats every parameter, derived ones included, as its own axis.
# The vector is param_names without the internal slots set_param refuses.
internal = [n for n, f in zip(model.param_names, model.param_is_internal) if f]
p_flat = jnp.array(
    [model.get_param(n) for n in model.param_names if n not in internal]
)
Y_flat = differentiable_solve(model, p_flat, (0, 100), 101, flat=True)
```

Things to know:

- **Each call starts from the model's initial conditions**, after the
  parameters are written, not from its current state. It warns where the two
  differ. To start from the current state, call `model.save_concentrations()`
  first.
- A state you set on a fresh model, or after `reset()`, is then a constant: an
  initial-condition parameter no longer moves it, and its derivative is 0.
- A state that a `run()` advanced depends on the parameters in a way nothing
  tracked, so a solve from it is refused. Call `model.reset()` first.
- `rtol`, `atol` and `max_steps` pass through as keywords. Each call clones the
  model, so calls are thread-safe.
- Outside JAX, `Result.gradient()` and the built-in objectives do the same job
  with less overhead.

## SBML models

### Species set by an `<assignmentRule>`

A species that an assignment rule defines has no ODE of its own. Its
sensitivity row is the chain rule through the rule, so
`result.sensitivities[:, i, :]` lines up with `result.species[:, i]` like
every other row. `output_sensitivities("species:<name>")` returns the same
numbers.

Where the chain rule is not available, the row is `NaN`, never `0.0`: a zero
would look like a real answer. The run warns naming the species, and
`result.ar_sensitivity_refused` lists them:

```python
from pathlib import Path

# pulse.xml: the species stimulus is set by an assignment rule that holds a
# piecewise, a step from 0 to 1 at t_on, and X is made at rate k*stimulus.
Path("pulse.xml").write_text(
    """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="pulse">
  <listOfCompartments>
    <compartment id="cell" size="1" constant="true"/>
  </listOfCompartments>
  <listOfSpecies>
    <species id="stimulus" compartment="cell" initialConcentration="0"
             hasOnlySubstanceUnits="false" boundaryCondition="true" constant="false"/>
    <species id="X" compartment="cell" initialConcentration="0"
             hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
    <parameter id="k" value="1" constant="true"/>
    <parameter id="t_on" value="5" constant="true"/>
  </listOfParameters>
  <listOfRules>
    <assignmentRule variable="stimulus">
      <math xmlns="http://www.w3.org/1998/Math/MathML">
        <piecewise>
          <piece>
            <cn> 1 </cn>
            <apply> <geq/>
              <csymbol definitionURL="http://www.sbml.org/sbml/symbols/time"> t </csymbol>
              <ci> t_on </ci>
            </apply>
          </piece>
          <otherwise> <cn> 0 </cn> </otherwise>
        </piecewise>
      </math>
    </assignmentRule>
  </listOfRules>
  <listOfReactions>
    <reaction id="make_X" reversible="false">
      <listOfProducts>
        <speciesReference species="X" stoichiometry="1" constant="true"/>
      </listOfProducts>
      <listOfModifiers>
        <modifierSpeciesReference species="stimulus"/>
      </listOfModifiers>
      <kineticLaw>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply> <times/> <ci> k </ci> <ci> stimulus </ci> </apply>
        </math>
      </kineticLaw>
    </reaction>
  </listOfReactions>
</model>
</sbml>
""",
    encoding="utf-8",
)
m = bngsim.Model.from_sbml("pulse.xml")
result = bngsim.Simulator(m, sensitivity_params=["k"]).run(t_span=(0, 10), n_points=11)

result.ar_sensitivity_refused          # frozenset({'stimulus'})
try:
    result.output_sensitivities("species:stimulus")
except ValueError as e:
    print(e)  # ... has no output sensitivity — uses unsupported construct: if() conditional.
```

Two things cause this. The rule holds a `piecewise` (which becomes an `if()`,
and BNGsim does not differentiate through it). Or the species is an amount in
a compartment that changes size, and the size's own rule has such a
construct.

### Amounts in a compartment that changes size

A species with `hasOnlySubstanceUnits` in a compartment that a rate rule, an
assignment rule or an event resizes is held as an amount. `run()` reports it
as `amount / V(t)`, using the live size. Its sensitivity row is the derivative
of that reported value, by the quotient rule, so a parameter that only moves
the volume still has a column. A species that is set by an assignment rule and
sits in such a compartment gets the same treatment, through the rule's row.
Where the size's rule has a `piecewise`, the row is `NaN` and the species is
in `result.ar_sensitivity_refused`.

A function that reads `rateOf` has no expression output sensitivity, and
neither do the functions that depend on it. The rest of the model keeps its
own.

Where the starting amount was converted at load using the size the rule had
then, the parameters the rule reads are
[folded in at load](#parameters-folded-in-at-load), and their columns are
refused. `steady_state()` reports the stored amount, and its rows are that
amount's.

With `sensitivity_ic=["V"]` on a compartment a rate rule moves, the column is
the derivative of a run after `set_concentration("V", ...)`: the live size
moves and the stored amount does not.

### Compartment sizes

On an SBML model, `model.param_names` includes the compartments, and a
compartment size is an ordinary writable, differentiable parameter.
`set_param("Liver", v)` updates everything the volume decides, and gives the
same model as loading it at that size. `sensitivity_params=["Liver"]` works,
and the column includes the initial-condition term (`-amount/V²` for an
amount-declared species).

`_V0_Liver` is not the volume. It is BNGsim's record of the size at load,
`set_param` refuses to move it, and the default column set leaves it out.

A few compartments cannot be written. `model.unwritable_compartment_size_params`
lists them, and the error gives the reason for each.
`compute_all_sensitivities()` skips their columns with a warning, and naming
one in `sensitivity_params` raises. Reload at the size instead:

```python
# pbpk.xml: blood and liver, both of size 1, and uptake from one to the
# other at a mass-action rate. A reaction between two compartments of the
# same size is one of the cases where a size cannot be written.
Path("pbpk.xml").write_text(
    """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="pbpk">
  <listOfCompartments>
    <compartment id="Blood" size="1" constant="true"/>
    <compartment id="Liver" size="1" constant="true"/>
  </listOfCompartments>
  <listOfSpecies>
    <species id="D_blood" compartment="Blood" initialConcentration="10"
             hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    <species id="D_liver" compartment="Liver" initialConcentration="0"
             hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
    <parameter id="k_in" value="0.3" constant="true"/>
  </listOfParameters>
  <listOfReactions>
    <reaction id="uptake" reversible="false">
      <listOfReactants>
        <speciesReference species="D_blood" stoichiometry="1" constant="true"/>
      </listOfReactants>
      <listOfProducts>
        <speciesReference species="D_liver" stoichiometry="1" constant="true"/>
      </listOfProducts>
      <kineticLaw>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply> <times/> <ci> Blood </ci> <ci> k_in </ci> <ci> D_blood </ci> </apply>
        </math>
      </kineticLaw>
    </reaction>
  </listOfReactions>
</model>
</sbml>
""",
    encoding="utf-8",
)
bngsim.Model.from_sbml("pbpk.xml").unwritable_compartment_size_params  # ['Blood', 'Liver']

v = 0.5
m = bngsim.Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v})
```

or take a central difference over reloads, which is exact:

```python
def dloss_dV(v, h):  # loss is your objective, a function of a Model
    up = bngsim.Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v + h})
    dn = bngsim.Model.from_sbml("pbpk.xml", compartment_sizes={"Liver": v - h})
    return (loss(up) - loss(dn)) / (2 * h)
```

### Parameters folded in at load

Some values an SBML model computes from its parameters are worked out once,
at load, and kept as numbers:

- the size of a compartment that an `initialAssignment` sets (`c = 2*p`), or
  that an assignment rule sets from parameters alone (`c := 2*p`), when the
  compartment holds a species. A rate law that names such a compartment reads
  the rule itself;
- the starting value of a species declared in the unit it is not held in, in a
  compartment that a rate rule or an event resizes and an `initialAssignment`
  sizes, or that an assignment rule sizes from time or the state
  (`c := v0 + g*time`);
- a stoichiometry: an L2 `<stoichiometryMath>`, an `initialAssignment` to a
  `speciesReference` id, or the id itself;
- a `conversionFactor`;
- a starting value whose `initialAssignment` reads something that is not a
  parameter, such as a reaction rate or the time.

BNGsim follows what each value reads all the way down: through other
`initialAssignment`s and rules, from a reaction id to its kinetic law, and from
a species to its starting value and compartment size. `c = J0`, `c = q` with
`q := 3*p`, and `c = S0` with `S0 = 2*p` each fold in `p`.

Changing such a parameter would not change what it was folded into, so BNGsim
refuses, by name:

```python
# model.xml: compartment c holds a species, and an initialAssignment sets
# its size to 2*p.
Path("model.xml").write_text(
    """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="folded">
  <listOfCompartments>
    <compartment id="c" size="2" constant="true"/>
  </listOfCompartments>
  <listOfSpecies>
    <species id="S" compartment="c" initialConcentration="5"
             hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
    <parameter id="p" value="1" constant="true"/>
  </listOfParameters>
  <listOfInitialAssignments>
    <initialAssignment symbol="c">
      <math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply> <times/> <cn> 2 </cn> <ci> p </ci> </apply>
      </math>
    </initialAssignment>
  </listOfInitialAssignments>
</model>
</sbml>
""",
    encoding="utf-8",
)
m = bngsim.Model.from_sbml("model.xml")
m.frozen_params                     # ['p']
try:
    m.set_param("p", 2.0)
except bngsim.ParameterError as e:
    print(e)                        # ... for the size of compartment 'c', which an initialAssignment sets ...
try:
    bngsim.Simulator(m, sensitivity_params=["p"])
except bngsim.SensitivityUnsupportedError as e:
    print(e)                        # Forward sensitivity is not supported for 'p', ...
```

Writing exactly the value it already holds is allowed, so a whole parameter
vector read off the model can be written back. A value that differs in the
last digit is a change, and is refused. `compute_all_sensitivities()` leaves
these columns out and says so. `differentiable_solve` solves at the values the
model holds and refuses another value for one of them, and `jax.grad` through
it is refused for such a model. To move such a parameter, change the SBML file
and load it again. The compartment size itself (`c` above) stays writable.

**Compartmental BNGL models have these too.** BNG2.pl writes each reaction's
volume factor into its rate constant as a number (`0.1*kb` for `1/Ve` at
`Ve = 10`) and keeps the expression only in a comment. So every parameter such
a comment names is in `frozen_params`, along with what a derived parameter or
function among them reads. This holds for `Model.from_bngl`, for a `.net` file
BNG2.pl wrote with its expressions kept (the default), and for
`build_model_from_parsed(parse_net_file(path))`. To move a volume, change the
BNGL file and generate the network again.

Two cases are covered less fully. A network written with
`writeNetwork({evaluate_expressions=>1})` still has the comments, so the
parameters they name are in `frozen_params`, but its derived parameters are
numbers, so what a derived volume reads (`r` in `vol = 4*r^3`) is not known.
A model written by BNGsim's converters (`write_net`, `net_to_sbml`) and loaded
again has no comments at all, so its volume is an ordinary parameter that
nothing reads.

## A derived parameter set after the Simulator is built

A derived parameter, `k2 = 2*k1`, follows its expression until
`set_param("k2", 5.0)` pins it, and then `k1` does not reach it. Writing back
the value its expression gives attaches it again. The compiled sensitivity
code differs between the two, so a Simulator checks when a sensitivity run
starts and rebuilds the code if needed. You can keep the Simulator, the model
and its clones across such a write.

The code also depends on whether each clock condition in a rate law resolves
to a switch time at the current values. If a write moves a parameter to where
one does not (`time >= sqrt(E)` at `E = 0`), the next run rebuilds and is
refused as a freshly loaded model would be. After a write back, the
Simulator builds the analytic code again at its next run.
`has_analytic_sens_rhs` describes the code in hand, so it changes only at the
next run. A Simulator first built where a condition does not resolve stays on
the [fallback](#the-fallback-difference-quotient), as does any later Simulator
on that model or a clone. A Simulator on a freshly loaded model gets the
analytic code.

In `run_batch`, a row that pins or re-attaches a derived parameter a requested
column runs through gets its own code, built before any row starts. A row at
whose values a switch time does not resolve is refused with
`SensitivityUnsupportedError`. Make such a write with `set_param` before the
batch, or run that row with `set_params` and `run()`.

## Rate laws that switch at a time

Some parameters set *when* the dynamics change, not how fast: the start of a
dose, the onset of an intervention. BNGL writes this as an `if()` over a clock:

```
begin functions
    1 rate_X()  if((t>=sigma), k, 0)
end functions
```

where `t` is an observable on a counter species with `dT/dt = 1`.

`sigma` appears only in the condition, so inside each branch the rate does not
depend on it. Its whole effect is a jump in the sensitivity at the crossing
time `t*`:

```
s(t*⁺) = s(t*⁻) + (f⁻ − f⁺) · ∂t*/∂p
```

BNGsim finds the crossings, computes `∂t*/∂p`, stops the integration exactly
at `t*`, and applies the jump. You don't need to declare anything.

```python
# The rate law above in a .net file: a clock T made at rate 1, so that
# dT/dt = 1, read as the observable t, and X made at rate rate_X().
Path("switch.net").write_text(
    """\
begin parameters
    1 sigma  4
    2 k      2
    3 one    1
end parameters
begin functions
    1 rate_X() if((t>=sigma),k,0)
end functions
begin species
    1 T() 0
    2 X() 0
end species
begin reactions
    1 0 1 one
    2 0 2 rate_X
end reactions
begin groups
    1 t 1
end groups
""",
    encoding="utf-8",
)
model = bngsim.Model.from_net("switch.net")
x_idx = model.species_names.index("X()")

sim = bngsim.Simulator(model, sensitivity_params=["sigma", "k"])
res = sim.run(t_span=(0, 10), n_points=101)
res.sensitivities[:, x_idx, 0]   # ∂X/∂sigma: exactly 0 before t* = 4, −k = −2 after
```

A model with `if()` conditions but no fitted switch time is not affected: it
steps exactly as it would without sensitivities.

### Conditions BNGsim can solve for a crossing

| Condition | Example | Notes |
|---|---|---|
| Linear in the clock | `t >= sigma` | The common case. |
| A power of the clock | `t^3 >= thresh` | Any degree. It crosses once. |
| A quadratic in the clock | `a*t^2 + b*t >= c` | Both crossings are handled. A general polynomial of degree 3 or more is declined. |
| A repeating schedule | `rem(t, P) >= d` | A dosing schedule. SBML's `rem` truncates, as MathML defines it. Over an offset clock, `rem(t - t0, P)` is negative before `t0` and is not read as a schedule; a sensitivity run of it is refused by name. |
| libSBML's expansion of `rem()` | `if(sign(a)!=sign(b), …)` | Read as the same schedule, so the gradient does not depend on which tool wrote the SBML. |
| A comparison used as a guard | `(t < 0) == 0` | Read as the equivalent boolean. |
| A threshold chosen by clock guards | `t > if(t<365, d_2021, if(t<730, 365+d_2022, …))` | Each branch crosses only where its threshold falls inside its own range of the clock. |

A crossing time that comes out complex is treated as one that never happens.

### Derived switch times

A threshold built from other parameters is chain-ruled to the primaries. With
`sigma = t0 + t_delta` and `tau1 = sigma + t_delta2`, `t0` gets a jump at both
crossings and `t_delta2` only at `tau1`. A threshold the run never reaches
gives a zero column.

### Two switches at the same instant

Each crossing is charged only its own jump, so two crossings at the same
instant do not merge into one doubled jump. Crossings within 64 ulp of each
other count as one instant, because that is how far BNGsim nudges the clock to
read the two branches. Crossings further apart, however close, are
handled one after the other. The same holds for a crossing just after an
output time, an event or the start of the run.

BNGsim tells two crossings on one instant apart by a parameter that only one
of them reads. One crossing may have no such parameter. With
`sigma = t0 + t_delta` and `t_delta = 0` (where a bounded optimizer puts a
parameter that wants to be smaller), the `t0` column takes the instant's whole
jump and the `t_delta` column takes `sigma`'s jump alone. Two or more
crossings with no parameter of their own cannot be told apart, and the run is
refused.

When a rate law combines the two crossings, such as `U()*Lambda()` with one
gate at `t0` and the other at `sigma`, the run sits on a kink in `t_delta`.
BNGsim then gives the one-sided derivative in the direction that delays
`sigma` past `t0`, raises `bngsim.OneSidedSensitivityWarning` naming the
parameter and the direction, and lists it in `result.one_sided_sensitivities`.
BNGsim cannot see your bounds: with `sigma = t0 - t_delta` that direction is
`t_delta` decreasing, which is infeasible at `t_delta >= 0`. The derivative in
the other direction is not computed.

This applies only to a pair of crossings on one clock, one of which has no
parameter of its own, within half that nudge of each other:
`32·eps·max(|t|, 1)`. These are refused:

- two combined crossings that each have a parameter of their own and are equal
  in value;
- a third crossing on the instant of such a pair;
- such a pair on two clocks;
- such a pair that is further apart than half the nudge but agrees to about
  twelve digits (`t_delta = 1e-12`). Its merged jump would leave one crossing
  out, or the run is near a kink rather than on it;
- a crossing read after the other whose rate goes to zero at a window's edge,
  as a power does;
- an instant with a left-over crossing, in a model that also has a rate law
  that opens as a power singular at its switch, such as `k*(t - sigma)^0.2`.

A species near steady state whose production reads the bumped parameter can
also be refused here, even though its reading is a true jump.

### An event at the same time as a switch

An event at the same fitted time as a switch, such as a dose at `tau` beside
`piecewise(k*X, time >= tau, 0)`, works: the two jumps combine. The switch can
be on `time` or on a counter.

If a requested parameter moves one and not the other, the order matters. Where
the event changes what the switched rate law contributes, the derivative does
not exist. BNGsim checks whether the event and the switch commute, and
refuses the run where they do not. A bolus beside an infusion that starts at
the same time commutes and runs. A reset of `X` at `tau = 3` beside
`piecewise(k*X, time >= 3, 0)` does not.

Several switches can share the event's instant. If they all move with the
event, that is one crossing and it runs. If not, the event must commute with
the instant as a whole and with each fitted switch alone. A switch within a
few hundred ulp of an event is checked too, so a pair that does not commute is
refused a little before the two times meet.

These cases are not detected and return a number:

- a `floor` step on a fitted switch;
- a state-dependent switch that crosses within the integration tolerance of an
  event, or on the instant of a time switch;
- two time switches of one rate law on one instant, with no event;
- a fixed condition written so that rounding moves it more than a few hundred
  ulp, such as `time + 3000 >= 3003`;
- a parameter that moves an event by less than 1e-9 of its time per unit
  relative change, which counts as not moving it.

### Landing on the crossing

BNGsim registers each switch time as a stop, so the solver lands on it and
restarts on the far side. This is done for every model with such a switch,
with or without sensitivities, because stepping over a discontinuity is not
correct. It covers rate laws that switch on `time` and on a counter species.
A counter is landed exactly on its threshold.

A crossing that depends on the state, such as `time() >= 4*S`, cannot be
placed before the run. Instead, each state comparison in a rate law's `if()`
is a root the solver looks for in every run, and so is a comparison used as a
number, such as `k*(X > 1)`. A run without sensitivities restarts there but
applies no jump. The state must clearly cross the
threshold: a trajectory that only creeps up to it, or settles beside it, may
cross within the solver's error and is stepped over. A window narrower than
the tolerance can resolve is not guaranteed to be seen.

A slow approach can pin a run on the threshold. When that happens for a whole
batch of steps, and the flow would have crossed by then, BNGsim moves the
state a few ulp across and restarts. How late the crossing is depends on how
slow the approach is and on `max_steps`. A state that slides along the
threshold, with both branches pointing into it, is not moved, and neither is
one that stops just short. This applies to `.net` models. A `piecewise`
condition on a species in an SBML model is not a state switch in a plain run,
and such a run can still stall.

A window written as one comparison of a lone `abs()`, or of a lone square,
such as `abs(X - 5) < 0.01` or `(X - 5)^2 < 0.0001`, is rooted at its two
edges, `X - 5 = ±0.01`, so it is not missed. A sensitivity run applies the
jump for the square. `abs()` has no analytic derivative, so for it the run
uses the [fallback](#the-fallback-difference-quotient), and refuses the model
where the state or a requested parameter moves the run across an edge. An
SBML event trigger written this way, directly or through a function
definition such as `near(X, 5, 0.01)`, is rooted at its edges too. Not
covered: the same window written another way (`pow(X - 5, 2)` in a `.net`
model, `2*abs(X - 5) < 0.02`, or `(X - 5)^2 < w` with a non-numeric bound),
and `steady_state()`.

### A rate that rises from zero at the crossing

A switch need not jump. A pulse such as `k0 + k1*s^(a-1)*(1-s)`, over a window
`s = (t - on)/D`, rises continuously from its onset for any `a > 1`. But its
derivative with respect to `on` goes as `s^(a-2)`, which is unbounded at the
onset for `1 < a < 2`. The sensitivity stays finite, but the plain column
cannot be integrated accurately there: it fails, stalls or ends up wrong.

So past such a crossing BNGsim integrates a shifted column, `V = S + c·f`,
where `c = ∂t*/∂on`. The singular terms cancel in `V`, which is as smooth as
the state. Outputs always report `S = V − c·f`. For a crossing approached
through such a power, the closing edge of a window `s*(1-s)^(a-1)`, the
column switches to `V` at a stop shortly before it.

A power counts as singular when its base can reach 0 and its exponent is
between 0 and 1, or is not a number (such as `a-1`). This is judged at the
run's parameter values: with `a = 3` nothing is singular and every column
runs plainly. An exponent that reads a species, the time or a value an event
assigns cannot be judged and is taken to be singular. An exponent chosen by a
condition is checked value by value, up to 64 values. A logistic onset,
`1/(1+exp(-k*(t-on)))`, never reaches 0 and is not affected.

The shifted column is used when the run has the analytic sensitivity code
(`sim.has_analytic_sens_rhs`), the model has no events (a state switch or an
SBML discontinuity trigger does not count as one), and the crossing moves
at the shift BNGsim derived from the power's base (`t - on` moves at
`c = 1`). It covers every rate that reads a counter clock, and a derived onset
such as `on = lam*1.0` gets its own shifted column.

These are refused:

- a restart within about 7e-10 of such an edge;
- another rate-law condition that crosses on the edge itself;
- a model with an event, for every crossing where the column would have needed
  `V`;
- a column that moves the counter clock itself (its starting value, its rate
  constant or its initial-condition axis) in a model with a singular power of
  that counter and a condition in some rate law. This holds whether or not the
  run reaches the power's zero. It includes the counter's rate constant, which
  `compute_all_sensitivities()` asks for by default, so leave it out. A
  counter that no such power reads keeps its columns;
- a column whose parameter moves such an edge at a rate that is not constant,
  such as `wb` in an onset `on/(wb - wa)`, `r` in `s = (t - on)*r`, or the
  width under a power whose base is not linear in time, `(1 - s^2)^(a-1)`.
  Where the power's base reads a species, the column of every parameter, and
  of every `sensitivity_ic` species, is checked. A parameter in the base that
  does not move its zero, `kk` in `(kk*(1 - s))^(a-1)`, keeps its column;
- a column that needed `V` in a model where BNGsim could not derive it.

The counter's own columns are also refused for a negative exponent where the
base is 0 at a condition the same rate law has on that counter: `(1 - s)^(a - 1)`
with `a < 1` in a law that closes at `s < 1`. A condition that is not linear
in the counter, reads the state, is chosen through a selection, or names the
counter another way is assumed to be on the zero. A gate just short of a
strong pole is not refused, and its accuracy depends on the tolerance.

Without the analytic code, or at a crossing that moves at another rate, the
plain column meets the singular term. If the run fails at the onset, the error
names the column and `∂f/∂on`. To avoid it, use an exponent of 2 or more, or
leave the onset parameter out of `sensitivity_params`.
`BNGSIM_SENS_COMOVING=0` keeps every column plain, for comparison; a column
refused for lack of `V` is still refused.

## Rate laws that switch on the state

A condition whose crossing time depends on the state, such as `if(X < 1, …)`
or a window `(X > lo) && (X < hi)`, is supported. Each comparison is a root,
and the jump `(f⁻ − f⁺)·dt*/dθ` is applied where it crosses. A crossing where
the rate law is continuous, such as a clamp whose live branch is 0 at the
threshold, needs no jump. BNGsim decides that from the reactions whose rate
laws read the condition. Conjunctions and negations are split into their
comparisons.

On the analytic path, a parameter that both sets a switch time and acts inside
a branch gets both terms. The analytic path declines a comparison outside any
`if()`, such as `k*(X > 1)`, and one whose sides are themselves comparisons,
such as `(X > 1) == (Y > 1)`.

On the [fallback](#the-fallback-difference-quotient), those comparisons are
refused where a requested column moves them, and so is a parameter that both
sets a switch time and acts inside a branch.

These are refused:

- two thresholds crossed within about 1e-13 of each other in time, both with
  jumps, that the requested columns move apart;
- a state that slides along the surface, such as `if(S < 1, amp, -amp)`, where
  both branches point into `S = 1`;
- a state that comes within a few tolerances of a surface without reaching
  it, or a run that ends within about a tolerance of one on its way there;
- a state held just short of a surface it approaches too slowly for any step
  to cross. A run without sensitivities is carried across;
- a state that leaves an equality's surface where that changes a rate law,
  such as `if(Bobs == 0, kb, 0)` with `B` held at 0 for a stretch. Write the
  switch as the crossing that ends the stretch instead: `if(Aobs > q, kb, 0)`.
  `Bobs <= 0` is refused the same way. A stretch that outlasts the run, or that
  only an event ends, runs, as does one that changes no rate law.

### Events that assign a value that is not smooth

An event whose assigned value is not smooth where the event reads it is
refused. The jump needs that value's derivatives, which BNGsim takes by
central differences, and across a step or a bend they are not derivatives.
For example, `u := piecewise(5, time >= T0 + 1, 0)` assigned at
`time >= T0 + 1`. The error names the event, the species and what the value
is not smooth in. Move the step away from the event, or leave the parameters
that reach it out of `sensitivity_params`. A step or bend that moves exactly
with the event is refused too, although it has a derivative. A bend that
changes the slope by less than about 0.8% is not detected.

### Narrow pulses

A narrow pulse on a counter or a state, such as a triangle
`kx*(1/w)*max(0, 1 - abs(Tobs - c)/(w/2))` or a rectangle written with `if()`,
has edges and corners that are roots. BNGsim applies the jump at each edge and
integrates the columns again over the step past each corner. Some cases can
still be off:

- a corner whose root falls at the very end of its step, or a step too stiff
  to integrate again. The error then depends on the output grid;
- a crossing that the clock drives much faster than the species do, such as
  `3*time() - b*Tobs` with `b` below about 0.09. These can stall, or return
  wrong columns with no error;
- a rate law that reads `time()` far from 0, where the clock's rounding error
  is amplified.

A sawtooth of the clock-driven kind at `rtol=1e-11` can fail even without
sensitivities.

To check such a run, repeat it on a different output grid or at a tighter
`rtol`. A column that moves by more than the tolerance is one of these. For
the clock-driven case, compare with a finite difference of runs without
sensitivities.

## The fallback difference quotient

Where BNGsim cannot use its compiled analytic derivatives, it uses CVODES' own
difference quotient. That is slower. It is right where every rate law is
continuous in the state along the run, but not across a jump: it reads the
rate law at a slightly moved state, which near a surface is on the other
branch. Ask the Simulator which path it took:

```python
sim.has_analytic_sens_rhs      # False when the run falls back
sim.sens_rhs_decline_reason    # why, in words, or None
```

On the fallback, a time course is refused before it starts if the model has:

- a rate-law condition on the state, `if(X < thr, kb, 0)`, unless the law is
  shown to bend there rather than jump (below);
- a sign or step made by dividing by an `abs`, `max` or `min` where it is 0,
  or by what it flips on, such as `(thr - X)/abs(thr - X)`;
- a condition on a counter species that a requested column moves;
- a step call on the state, `floor(X)`, or a step table function indexed by an
  observable or a function;
- a step call on time, or on a counter nothing moves, whose argument reads a
  requested parameter, such as `floor(time()/P)` with `P` requested;
- a comparison over parameters alone that a requested parameter is close to
  flipping, within a quarter of its value, such as `if(n > 1, kb, 0)` near
  `n = 1`; an equality on a requested parameter that holds; a step call or
  step table on one. Derived parameters are checked too;
- a power under 1, a root or a logarithm of an `abs`, `max` or `min` that may
  be 0, such as `max(thr - X, 0)^0.1`, or an inverse sine or cosine of one;
- a comparison that is not an `if()` condition, such as `kb*(X > 1)`, or
  ExprTk's single `=`, on what a column moves, in a rate law or in a derived
  parameter (`kd = 3*(n > 1)` with `n` requested and close to 1);
- a call BNGsim does not know, such as `atan2`, on what a column moves;
- a rate law that could not be read.

The check reads the rate laws' text and the parameter values when the run
starts. It is repeated when a parameter is set and for each row of a batch.

A condition is allowed where the law is shown to bend rather than jump:

- one branch is 0 and the other is a product with the condition's own
  difference as a factor: `if(v > 0, v, 0)`, `if(X < thr, kb*(thr - X), 0)`;
- or the two branches are the two sides of the comparison: `if(a < b, a, b)`;
- and every division in the law is by something known to be nonzero from the
  signs of numbers and parameters, such as `max(X, 0.01)` or `Km + abs(X)`
  with `Km > 0`. `Km + X` is not known to be nonzero.

Any other bend is refused, such as `if(X < thr, kb*sqrt(thr - X), 0)`, a
power that is a parameter, or `if(S > 0, Vm*Q/S, 0)`. `max` and `min` are continuous and run:
`kb*max(thr - X, 0)`, `v/max(X, 0.01)`.

A condition on literal time runs. A steady-state solve is not checked. It
differences the rate in the parameter by a step of about 1.5e-8 times the
parameter, so it is wrong for a parameter that close to a threshold or a kink
of its own, such as `if(n > 1, kb, 0)` at `n = 1`.

The check does not see a jump written with no condition, such as
`sqrt(X*X)/X`, `(thr - X)/(abs(thr - X) + 1e-9)`, `tanh(1e9*(X - thr))` or
`min(max(k/(X - thr), -5), 5)`; a sharp bend with no `abs`, `max` or `min`,
such as `((thr - X)^2)^0.05`; or two crossings of one condition at one instant
on a counter, which is refused on `time()`.

See the [PyBNF guide](pybnf.md#ask-each-model-whether-its-gradient-is-analytic)
for using this to sort models before a fit.

## Adding a built-in objective

To add a method like `sse_gradient` to `Result` (`python/bngsim/_result.py`),
derive `dL/dY` as an `(n_times, n_species)` array and contract it with the
sensitivity tensor: `grad += sens[t].T @ dL_dY[t]` for each time. That loop is
the same for every objective. Raise `ValueError` when `has_sensitivities` is
false, support `species_indices`, and return `(loss, gradient)`. Test it in
`test_objective_gradients.py`: the shape, a zero gradient at zero residual,
agreement with `Result.gradient()` given the same `dL/dY`, and the errors.
