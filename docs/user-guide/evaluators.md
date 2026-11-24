---
summary: "Evaluating a model outside a run: right-hand side, Jacobian, stoichiometry matrix and propensities at a given state."
---

# Model evaluators: RHS, Jacobian, stoichiometry, propensities

A `Model` can evaluate the functions its solvers use, at any state you give it,
as NumPy arrays. Use them for analysis outside BNGsim: a continuation code, a
check of a root another solver found, a moment-closure or
finite-state-projection generator. Each one is the function the matching engine
runs, so it gives what a solve sees.

```python
import numpy as np
import bngsim

# Schlogl's bistable switch, one of the models that ship with BNGsim
m = bngsim.Model.from_sbml(bngsim.examples.path("schlogl", "sbml"))  # or your own model
y = m.get_state()                    # or any (n_species,) array of your own

S = m.stoichiometry_matrix()         # (n_species, n_reactions), net coefficients
f = m.rhs(y, t=0.0)                  # dy/dt at the current parameters
J = m.jacobian(y, t=0.0)             # ∂f/∂y; J.source is "analytical" or "finite-difference"
a = m.propensities(y, t=0.0)         # SSA propensity per reaction
```

Watch out:

- The state is in `species_names` order. A state of the wrong length raises
  `ValueError`.
- None of them changes the model's stored state: `get_state()` is the same
  afterwards.
- They read the model's current parameters, so call `set_param` between calls
  to scan a parameter.
- A row of `Result.species` is not always a valid state for them. Use
  `Result.state` (see [Evaluating along a trajectory](#evaluating-along-a-trajectory)).

## `rhs(y, t=0.0)`

The right-hand side the ODE solver evaluates. For a model with functions, it
first recomputes the observables and function-valued rate parameters at `y`.
`t` reaches `time()`, `rateOf` and table functions. A `$`-fixed boundary
species has a zero derivative, as in a solve.

## `jacobian(y, t=0.0, *, sparse=False)`

`J[i, j] = ∂f_i/∂x_j`. It is the matrix a solve would use at that point under
`jacobian="auto"`:

- from the model's closed-form derivatives when they cover every reaction;
- otherwise, from a one-sided difference quotient of `rhs`, stepped as the
  steady-state solver steps it.

`J.source` says which: `"analytical"` or `"finite-difference"`, the same values
`SteadyStateResult.solver_jacobian_source` uses. If the closed form covers only
some reactions (a rate law the symbolic differentiator cannot take, such as a
table function), the whole matrix is differenced and labeled so.

A difference quotient is checked before it is returned, as `steady_state`
checks it. The columns of species at 0, or far below the state's scale, are
differenced again twice: with a step sized to the species itself, and with
half the first step. Where an entry
moves by more than 1e-4 of itself (and at least 1e-10 of its column's largest),
`jacobian` raises `SimulationError` naming the entry and its two values. For
example, next to a species at 1e8, one near 0 is stepped by 1.5, so a
second-order term in it gives a secant, not a slope. A model whose rate laws all
have a closed form is never differenced, so never refused.

`sparse=True` returns a `scipy.sparse.csc_array` over the model's structural
sparsity pattern. The pattern is the same for either source, so you can
allocate for it once. On a model with functional rate laws, the first
`jacobian` call derives the closed-form terms (`prepare_analytical_jacobian`),
as an ODE solve does at setup.

The dense result is a `bngsim.JacobianMatrix`: a float64 `ndarray` with a
`source` attribute, so `np.linalg.eigvals(J)` and `J @ v` work as usual.

## `stoichiometry_matrix(*, sparse=False)`

`S[i, r]` is the net change in species `i` when reaction `r` fires: products
positive, reactants negative. `A + A -> B` puts `-2` in A's row.

- Every [conservation law](conservation-laws.md) satisfies `L @ S == 0`. The
  exception is an SBML model with a reaction between compartments of different
  sizes, whose laws are over amounts and satisfy `L @ rhs(y) == 0`.
- For a mass-action `.net` model, `rhs(y) == S @ v(y)`, with `v` the ODE
  reaction rates.
- A `$`-fixed species has a zero row, whatever the reaction says: neither the
  ODE nor an SSA firing changes it.
- Use `sparse=True` (a CSC array) on a large network, where the dense matrix
  does not fit in memory.

## `propensities(y, t=0.0)`

One entry per reaction, in reaction order: the per-event rate an SSA step
samples from. It differs from the ODE rate of the same reaction in two ways:

- A repeated reactant of a mass-action reaction takes the falling factorial of
  its molecule count. `A + A -> B` fires at `k·n(n−1)/2`, where the ODE rate is
  `k·n²/2`. The count is `n = x·V` for a concentration in a compartment of size
  `V`, and `n = x` in a `.net` model.
- A reaction in a compartment of volume `V` is multiplied by `V`, turning the
  ODE's rate per volume into a rate per event.

An SBML rate law that BNGsim does not read as mass action (a boundary reactant,
a law across compartments of different sizes, a compartment set by an
assignment rule) is evaluated as written. If it is a product of its species,
a species it reads 2 or more times still takes the falling factorial. A law
that writes its own combinatorics (`k*X*(X-1)/2`), or reads a species other
than as a factor, is left exactly as written.

For a `.net` model with volume 1 and no repeated reactant,
`S @ propensities(y) == rhs(y)`. Observables and function-valued parameters are
recomputed at `y` first, as in the SSA. Values are read from `y` as given, so
pass molecule counts where the model expects counts.

## Evaluating along a trajectory

The state a solver integrates is not always what a run reports. In an SBML
model, `Result.species` leaves out the parameters and compartments that events
turn into state. A species in a compartment whose size changes, or one set by
an assignment rule, is reported after a conversion. So a row of
`Result.species` on such a model is too short for `rhs` or `jacobian`, or is
not the point the solver was at.

`Result.state` is the trajectory the solver held, every entry, unconverted, in
`species_names` order. `Result.state_names` names its columns.

```python
res = bngsim.Simulator(m, method="ode").run(t_span=(0.0, 10.0), n_points=101)
X, T = res.state, res.time            # X.shape == (101, m.n_species)
J = [m.jacobian(X[i], t=T[i]) for i in range(len(T))]
```

On a `.net` model, and on an SBML model with nothing converted, `state` and
`species` are the same. Only a result a solve returned has `state`. A result
loaded from disk or stacked from a batch raises `ValueError` for it.

## Checking a root found elsewhere

```python
from scipy.optimize import fsolve

y_guess = [380.0]                    # near the upper of its stable states, X = 400
root = fsolve(lambda y: m.rhs(y), y_guess)
ev = np.linalg.eigvals(m.jacobian(root))
stable = np.all(ev.real < 0)
```

A model with conservation laws has one zero eigenvalue per law. To restrict the
spectrum as the steady-state solver's stability check does, or to read the
spectrum it already computed, see
[`SteadyStateResult.eigenvalues`](steady-state.md#unstable-roots-saddles).

## A CME or moment generator

`stoichiometry_matrix` and `propensities` are all such a generator needs. Each
reaction's state change is a column of `S`, and its rate at a state is the
matching entry of `propensities(y)`.
