# Model evaluators: RHS, Jacobian, stoichiometry, propensities

A `Model` can evaluate the functions its integrators integrate, at any state
you hand it, as NumPy arrays. This is the provider an analysis built outside
bngsim needs — a continuation code stepping along an equilibrium branch, a
classifier of roots found by another solver, a moment-closure or
finite-state-projection generator — and each evaluator is the one the
corresponding engine runs, so they cannot drift from what a solve sees.

```python
import numpy as np
import bngsim

m = bngsim.Model.from_net("model.net")
y = m.get_state()                    # or any (n_species,) array of your own

S = m.stoichiometry_matrix()         # (n_species, n_reactions), net coefficients
f = m.rhs(y, t=0.0)                  # dy/dt, live parameters
J = m.jacobian(y, t=0.0)             # ∂f/∂y; J.source is "analytical" or "finite-difference"
a = m.propensities(y, t=0.0)         # SSA propensity per reaction, SSA volume convention
```

All four take the state in `species_names` order and raise `ValueError` for a
state of the wrong length. None of them reads or writes the model's stored
concentrations: `y` is the state evaluated, `get_state()` is unchanged
afterwards. They read the model's *live* parameters, so `set_param` between
calls is how a parameter scan drives them.

## `rhs(y, t=0.0)`

The interpreted right-hand side the CVODE callback evaluates, with the same
observable-and-function refresh: for a model with functions, observable totals
and function-bound rate parameters are recomputed at `y` before the rate loop;
for a pure mass-action model that pass is skipped as dead work (the
`rhs_evaluates_observables` gate). `t` reaches `time()`, `rateOf` and table
functions. A `$`-fixed boundary species reports a zero derivative, as it does
in a solve.

## `jacobian(y, t=0.0, *, sparse=False)`

`J[i, j] = ∂f_i/∂x_j`. The matrix comes from the model's closed-form terms
when they cover every reaction, and from a one-sided difference quotient of
`rhs` otherwise — the rule the ODE integrator and the steady-state solver apply
under `jacobian="auto"`, stepped by the steady-state solver's own rule, so what
you get is the matrix a solve would have factored at that point. The result
says which: `J.source` is `"analytical"` or `"finite-difference"`, the same
spellings `SteadyStateResult.solver_jacobian_source` uses. A model whose
analytical Jacobian is *partial* — a rate law the symbolic differentiator
declined, such as a table function — is never handed the partial matrix under
the first label; it is differenced, and reported as such.

`sparse=True` returns a `scipy.sparse.csc_array` over the model's structural
sparsity pattern instead of a dense array. The pattern is the same for either
source, so a consumer can allocate for it once. The first `jacobian` call on a
model with functional rate laws derives the closed-form terms
(`prepare_analytical_jacobian`), which an ODE solve would do at its own setup.

The dense return is a `bngsim.JacobianMatrix`: an ordinary float64 `ndarray`
with the `source` attribute added, so `np.linalg.eigvals(J)` and `J @ v` work
as usual.

## `stoichiometry_matrix(*, sparse=False)`

`S[i, r]` is the net change in species `i` when reaction `r` fires — products
positive, reactants negative, so `A + A -> B` puts `-2` in A's row. This is the
matrix [conservation-law detection](conservation-laws.md) was row-reduced
from, so every law satisfies `L @ S == 0`, and for a mass-action `.net` model
`rhs(y) == S @ v(y)` for the vector of ODE reaction rates. A `$`-fixed species
has an all-zero row: the RHS zeroes its derivative and an SSA firing never
updates it, whatever the reaction line says. `sparse=True` returns a CSC array;
ask for it on a large network, where the dense matrix does not fit.

## `propensities(y, t=0.0)`

One entry per reaction, in reaction order: the amount/time propensity a
stochastic step samples from, in the **SSA volume convention** rather than the
ODE one. Two things differ from the ODE rate of the same reaction:

- a repeated reactant takes the falling factorial — `A + A -> B` fires at
  `k·x(x−1)/2`, where the ODE rate is `k·x²/2`;
- a reaction in a compartment of volume `V` is multiplied by `V`, converting
  the ODE's storage-units rate to a per-event rate.

For a `.net` model with volume 1 and no repeated reactant the two conventions
agree, and `S @ propensities(y) == rhs(y)`. Observables and function-bound
parameters are refreshed at `y` first, as the SSA loop refreshes them before
its propensity pass. Values are read from `y` as given — pass molecule counts
for a count-valued answer.

## Evaluating along a trajectory

The state a solve integrates is not always the state it reports.
`Result.species` omits the entries an SBML event promotes to state (a
parameter or a compartment the event assigns, GH #71), and a species column
in an event-resized or rate-ruled compartment, or one an assignment rule
targets, is remapped to its reported value after the solve (GH #85, #131). A
row of `Result.species` on such a model is therefore not a state `rhs` or
`jacobian` accepts: it is too short, or it is not the point the integrator
was at. `Result.state` is the trajectory the integrator actually held —
every entry, unprojected and unremapped, in `species_names` order — and
`Result.state_names` names its columns.

```python
res = bngsim.Simulator(m, method="ode").run(t_span=(0.0, 10.0), n_points=101)
X, T = res.state, res.time            # X.shape == (101, m.n_species)
J = [m.jacobian(X[i], t=T[i]) for i in range(len(T))]
```

On a `.net` model, and on an SBML model without promoted or remapped
entries, `state` and `species` are the same array. Only a result returned by
a solve carries `state`; a result loaded from disk or stacked from a batch
holds the reported block alone and raises `ValueError` for it.

## Classifying a root found elsewhere

```python
from scipy.optimize import fsolve

root = fsolve(lambda y: m.rhs(y), y_guess)
ev = np.linalg.eigvals(m.jacobian(root))
stable = np.all(ev.real < 0)
```

A model with conservation laws has one zero eigenvalue per law in the full
spectrum. To reproduce the restriction the steady-state solver's stability
certificate applies, or to read the spectrum it already computed, see
[`SteadyStateResult.eigenvalues`](steady-state.md#unstable-roots-saddles).

## A CME or moment generator

`stoichiometry_matrix` and `propensities` are all such a generator needs from
bngsim: the state change of each reaction is a column of `S`, and its rate at
a state is the matching entry of `propensities(y)`. Nothing else in the model
has to be reconstructed.
