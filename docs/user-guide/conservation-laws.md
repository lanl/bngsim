# Conservation laws & parameter scans

## Conservation laws

BNGsim automatically detects conservation laws at model load time via
Gaussian elimination on the stoichiometry matrix. A conservation law is
a linear combination of species that remains constant during the dynamics:

```
Σ L[k,i] · y[i] = constant_k    for all time
```

Conservation laws arise from molecular conservation (total receptor,
total ligand, etc.) and are detected for ALL input formats (`.net`,
Antimony, SBML, programmatic `ModelBuilder`).

```python
model = bngsim.Model.from_net("model.net")

# Inspect conservation laws
laws = model.conservation_laws
print(laws["n_laws"])           # number of conservation laws
print(laws["dependent"])        # dependent species indices (0-based)
print(laws["independent"])      # independent species indices
print(laws["constants"])        # conservation constants from ICs
print(laws["coefficients"])     # n_laws × n_species coefficient matrix
print(laws["n_species"])        # species count the coefficient rows span
```

**Impact on the steady-state solver**: Models with conservation laws have
a rank-deficient Jacobian, which causes standard Newton solvers to fail.
BNGsim's reduced-space Newton solver automatically handles this:

1. **Identifies** N - n_laws independent species from the conservation structure
2. **Builds** a reduced residual function on the independent subspace
3. **Solves** the non-singular reduced system via KINSOL
4. **Reconstructs** dependent species from the conservation constraints

This is transparent to the user — `sim.steady_state()` just works, regardless
of whether the model has conservation laws.

## `parameter_scan` integration

BNGsim supports BioNetGen's `parameter_scan` action, commonly used in BNGL
models for dose-response analysis. When PyBNF encounters a `parameter_scan`
action in a BNGL model, it automatically routes through BNGsim's batch
steady-state or time-course infrastructure.

For BNGL models that include:
```
parameter_scan({method=>"ode", parameter=>"L_0", \
    par_min=>0.01, par_max=>100, n_scan_pts=>50, \
    log_scale=>1, steady_state=>1, t_end=>1e6})
```

PyBNF's `BngsimModel` parses this action and dispatches to:
- **strict BNG2.pl parity (default)** when `steady_state=>1`: each scan point
  runs `sim.run(steady_state=True)` — the same `run_network -c`
  integrate-to-`||f||_2/n` early-stop BNG2.pl uses.
- `sim.steady_state_batch()` when the model opts in with `ss_method=>"newton"`
  (or its alias `ss_method=>"kinsol"`) — the two-tier integrate-then-polish
  solver, which resolves the root far tighter than the parity criterion asks
  for. It is *not* a speed accelerator: since the burst is the integration path
  itself, the polish is net overhead (issue #28).
- Time-course simulation to `t_end` when `steady_state=>0`.

`ss_method=>"newton"` is **rejected for `bifurcate` continuation scans**
(warn + downgrade to the parity path): bifurcate carries state between points
to detect hysteresis/multistability, and independent-per-point Newton finds
*a* root — it can jump branches and destroy the hysteresis signal.

The output is a `.scan` file with columns for the scanned parameter and
all observables/expressions, matching BNG2.pl's `parameter_scan` output
format. This enables transparent acceleration of existing BNGL workflows
that use `parameter_scan` for dose-response curve generation.
