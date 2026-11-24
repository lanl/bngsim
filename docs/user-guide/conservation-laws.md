---
summary: "The conservation laws BNGsim finds in a network, how steady-state solves use them, and BNGL parameter_scan through PyBNF."
---

# Conservation laws & parameter scans

## Conservation laws

BNGsim finds a model's conservation laws when it loads it. A conservation law
is a linear combination of species that stays constant in time:

```
Σ L[k,i] · y[i] = constant_k    for all time
```

They come from conserved molecules (total receptor, total ligand). BNGsim finds
them for every input format: `.net`, SBML, Antimony and `ModelBuilder`.

```python
import bngsim

# The bundled A + B <-> C model (species A(b), B(a), A(b!1).B(a!1)); or your own .net
model = bngsim.Model.from_net(bngsim.examples.path("binding"))

laws = model.conservation_laws
print(laws["n_laws"])           # 2: the number of conservation laws, total A and total B
print(laws["dependent"])        # [0, 1]: dependent species indices (0-based)
print(laws["independent"])      # [2]: independent species indices
print(laws["constants"])        # [100.0, 80.0]: the conserved totals, from the initial conditions
print(laws["coefficients"])     # [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]: n_laws × n_species
print(laws["n_species"])        # 3: the species count the coefficient rows span
```

`dependent` lists one species per law, the one the steady-state solver
eliminates. They are the pivot columns of the row-reduced coefficient matrix,
so `coefficients[k][dependent[j]]` is `1` when `k == j` and `0` otherwise.
Each law can therefore be solved for its own dependent species.

### Compartments

A law is over the state the model stores: what `model.get_state()` returns and
`model.rhs(y)` takes. So `coefficients @ model.rhs(y)` is zero at any state `y`.

In an SBML or Antimony model, a reaction between compartments of different
sizes conserves an amount, so each species' coefficient carries its
compartment's size. `A` in a compartment of size 1 exchanged with `B` in one of
size 2 gives `A + 2*B`, not `A + B`.

- Where a compartment size is a parameter, the laws follow it, per model.
  After `model.set_param("c2", 4)` the law is `A + 4*B` for that model and its
  clones. Each entry of `steady_state_batch` that sets a size is solved with
  its own.
- A concentration species in a compartment whose size changes (by a rate rule
  or an event) is in no law.
- A `substanceOnly` species there is stored as its amount over the size at
  load, and its law is over that. A run reports it over the current size.

### Steady states

A model with conservation laws has a singular Jacobian, and a plain Newton
solver fails on it. BNGsim's steady-state solver works on the independent
species only and fills in the dependent ones from the laws.
`sim.steady_state()` works the same whether or not the model has laws.

Before a steady-state solve, BNGsim evaluates the rates at two states to check
that every total it reports is kept. A law that is not kept raises
`SimulationError` rather than solving with a wrong total. The check compares
against the fluxes through the law's species, so a total that moves by less
than 1e-10 of them is not caught.

Steady-state sensitivities use the same reduction. BNGsim returns them where
the steady state is an isolated root of the reduced system, and refuses them
where it is one of a continuum. See [Steady state](steady-state.md).

## `parameter_scan` in BNGL models through PyBNF

PyBNF's `BngsimModel` runs a BNGL `parameter_scan` action through BNGsim's
batch steady-state or time-course calls. For example:

```
parameter_scan({method=>"ode", parameter=>"L_0", \
    par_min=>0.01, par_max=>100, n_scan_pts=>50, \
    log_scale=>1, steady_state=>1, t_end=>1e6})
```

- `steady_state=>1` (the default path): each point runs
  `sim.run(steady_state=True)`. This matches BNG2.pl's `run_network -c`, which
  integrates until `||f||_2/n` is small.
- `ss_method=>"newton"` (or its alias `ss_method=>"kinsol"`): each point uses
  `sim.steady_state_batch()`, which integrates and then polishes the root with
  Newton. It resolves the root more tightly than BNG2.pl's criterion. It is not
  faster: the polish is extra work on top of the same integration.
- `steady_state=>0`: a time course to `t_end`.

`ss_method=>"newton"` is not used for `bifurcate` scans. PyBNF warns and uses
the BNG2.pl path instead. A `bifurcate` scan carries the state from point to
point to show hysteresis, and an independent Newton solve per point can jump
to another branch and hide it.

The output is a `.scan` file with a column for the scanned parameter and one
for each observable and expression, in BNG2.pl's `parameter_scan` format.
