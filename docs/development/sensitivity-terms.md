# Forward-sensitivity terms

This page writes down, once, every term that goes into a forward-sensitivity
column: the seed, the variational right-hand side, and the jumps at events,
rate-law switches, counter clocks and comoving frames. For each term it names
where the term is implemented and which oracle test checks it. It is the
derivation issue #788 asked for, and the reference to hold a fix in that family
against.

Every term has a test in `python/tests/test_codegen_sens_fd_oracle.py`. Each test
builds a model on which that term is the only one in play and compares its column
with a central difference of plain runs. Plain runs have no sensitivities, so the
reference never enters the code it checks. `python/tests/_fd_sens.py` computes
that difference. It picks the step from the measured noise floor rather than a
fixed `1e-6·|p|`, and it refuses to compare a sample that sits on a kink in the
parameter. Issue #368 (`test_time_switch_sens_fd_reference.py`) shows the two ways
a careless difference misleads: a step below the noise floor, and a sample on the
kink. A term with an open defect is a strict `xfail` naming its issue.

## Notation

```
x' = f(t, x, p)          state, with x(t₀) = x₀(p)
s  = ∂x/∂p               one column per requested parameter p
                         (or per sensitivity_ic species, with p = x_i(t₀))
f⁻, f⁺                   the right-hand side just before / after an instant
t*                       the instant of an event or a crossing
τ  = ∂t*/∂p              how far that instant moves with p
```

A **derived** parameter is one defined by an expression (`Rt = 3*R0`, an SBML
initial assignment, BNG2.pl's `_rateLawN`). Its own column treats it as a free
axis: the derivative you get once `set_param` pins it to a value. A primary's
column reaches through every derived parameter that reads it.

## 1. The seed: `s(t₀) = ∂x₀/∂p`

```
s_i(t₀) = ∂x₀,i/∂p                       parameter column
s_i(t₀) = δ_ij                           sensitivity_ic column of species j
```

`∂x₀/∂p` is the chain rule through the parameter graph. A primary column also
reaches through a derived parameter (`B() Rt`, `Rt = 3*R0`: `∂B₀/∂R0 = 3`). A
derived parameter's own column is the identity where a species names it.

- Python: `compute_ic_param_sens_seed` (`_codegen.py`), called by
  `Simulator._apply_ic_param_sens_seed`. A carried-over seed from
  pre-equilibration (GH #210) replaces the seed.
- C++: the `yS(0)` seeding block ahead of `CVodeSensInit1` in `CvodeSimulator::run`.
- Oracle: `ic-seed-param`, `ic-seed-axis`, `ic-seed-derived-via-primary`.
- **Open:** a derived parameter's own column gets no seed, so it reads all zero
  (#715, `ic-seed-derived-own-column`).

## 2. The variational right-hand side

```
s' = f_x·s + f_p,      f_p = ∂f/∂p + Σ_d (∂f/∂d)·(∂d/∂p)   over derived d
```

- Analytic: `_emit_sens_rhs_body`, `_functional_dfdp_terms`, and
  `_derived_param_jacobian_dag` for the derived chain (`_codegen.py`).
- CVODES' own difference quotient takes over when the analytic RHS is declined
  (for example `abs()` or `floor()`). The parameter sync for each probe is
  `sync_sens_params` (`cvode_simulator.cpp`).
- Oracle: `rhs-dfdp`, `rhs-derived-chain`, `rhs-derived-own-column`, `dq-primary`.
- **Open:** on the difference-quotient path, the sync re-derives a derived
  parameter that is being probed, so its column is exactly zero (#707,
  `dq-derived-own-column`). A compiled sensitivity RHS that is already attached
  is reused after `set_param` changes a derived parameter's attachment, so it
  keeps the old chain rule (#708, `stale-artifact-after-override`).

## 3. An event: the jump at `t*`

An event assigns `x⁺ = h(x⁻, p, t*)` at a trigger time `t*(p)`. The total
derivative of the post-event state, less the flow after the event, is:

```
assigned row k:     s⁺_k = h_x·(s⁻ + f⁻·τ) + h_p + h_t·τ − f⁺_k·τ
unassigned row k:   s⁺_k = s⁻_k + (f⁻_k − f⁺_k)·τ
```

`τ` depends on the trigger:

```
time trigger   t ≥ T(p)        τ = ∂T/∂p                                 (#49)
state trigger  g(x, p, t) = 0  τ = −(g_p + g_x·s⁻) / (g_x·f⁻ + g_t)      (#144)
fixed time                     τ = 0                                     (GH #212)
```

If `g_x·f⁻ + g_t` cancels to roundoff, the crossing is tangential and `τ` is
unbounded, so the run is refused.

- `τ` for a time trigger: `compute_event_time_sens` (`_switch_sensitivity.py`),
  plumbed by `Simulator._apply_event_time_sens`. For a state trigger:
  `residual_dtstar` (`cvode_simulator.cpp`).
- The jump: `apply_event_sensitivity_jump`, which reads `s⁻` through
  `capture_event_sens`.
- Oracle: `event-fixed-time`, `event-time-trigger`, `event-state-trigger`,
  `event-assignment-reads-state`, `event-initial-value-true`.

**`h_t` and `rateOf`.** `h_t` is the assignment's explicit time dependence:
`Tlast = time`, `B = time + 1`, or a rule or function that reads `time`. An
assignment that reads `rateOf(x_j)` reads `f_j`. Its `h_x` and `h_p` therefore
include `∂f_j/∂x` and `∂f_j/∂p`, and the finite-difference sync that forms them
has to refresh the `rateOf` buffer.
- **Open:** the `h_t·τ` term is missing (#735, `event-assignment-reads-time`,
  `event-assignment-reads-time-clock`). The `rateOf` buffer is never refreshed,
  and `rateOf` gets no parameter support, so the row is zero (#764,
  `event-assignment-reads-rateof`).

**At `t_start`.** An SBML event with `initialValue=false` whose trigger is already
true at `t_start` fires there. Then `τ = 0`, and `s⁻` is the seed that is already
in `yS`. There is nothing to interpolate from, because CVODES has not taken a step.
- **Open:** `s⁻` is interpolated anyway, which gives NaN at `t_start = 0` and
  `CV_BAD_T` elsewhere, so the run is refused (#717, `event-fires-at-t-start`).

## 4. A batch of events at one instant

Events that fire together run one at a time: highest priority first, with random
tie-breaking. Each fire's assignment reads either the pre-batch state
(`useValuesFromTriggerTime=true`) or the state that the earlier fires left
(`false`). A non-persistent instance whose trigger an earlier fire made false is
cancelled. The jump composes in *execution* order:

```
D₀ = s⁻ + f⁻·τ
D_k = D_{k−1}, except on the rows fire k assigns:
      D_k = h_{k,y}·D_read + h_{k,p} + h_{k,t}·τ     D_read = D₀ (UVFTT=true) or D_{k−1} (false)
      (a cancelled instance contributes nothing)
s⁺ = D_n − f⁺·τ                                     f⁺ after the whole batch
```

- `process_firing_batch` (the run loop in `cvode_simulator.cpp`) executes the
  batch, and `apply_event_sensitivity_jump` differentiates it.
- Oracle: `event-batch-survivor-declared-last`.
- **Open:** the jump walks the fired events in declaration order and
  differentiates every one of them at `x⁻` (#722, `event-batch-priority-order`).

## 5. A rate-law switch

The state is continuous across a switch, and `f` jumps. `f⁻` and `f⁺` are the two
branches evaluated at the same `x(t*)`:

```
s⁺ = s⁻ + (f⁻ − f⁺)·τ
```

When the condition compares time against parameters (`if(time()>=tau, …)`), the
crossing is known in advance. `τ` comes from the threshold's partials, including
those of a derived threshold (#475). The solver stops exactly at `t*` and reads
each branch by nudging the clock a few ulp either side of the threshold. When the
condition reads state, `τ` comes from the implicit-function formula of §3, with
`g` the condition's residual (#150). A switch whose branches meet (`f⁻ = f⁺`) needs
no jump.

- Time crossings: `compute_switch_time_sens` and `_absorb_schedule_crossings`
  (`_switch_sensitivity.py`) compute them, and `apply_switch_sensitivity_jump`
  (`cvode_simulator.cpp`) applies them. The run loop takes a switch as reached
  within `switch_t_eps = 1e-9·max(1, horizon)`.
- State crossings: `state_switch_conditions` (`_switch_sensitivity.py`) finds
  them, and `apply_state_switch_sensitivity_jump` applies them.
- Oracle: `time-switch`, `state-switch`.
- **Open:** a switch that lies within that window after an output time or other
  stop is jumped at the wrong instant (#737,
  `time-switch-just-after-an-output`). The branches-meet test divides by
  `max|f|` over *every* species, so one large, unrelated flux makes a real jump
  read as continuous (#763, `state-switch-beside-a-large-pool`).

**An event and a switch at the same instant.** `f⁻` is the before-branch at
`x⁻`, and `f⁺` is the after-branch at `x⁺`:

```
assigned row:     s⁺ = h_x·(s⁻ + f_before(x⁻)·τ) + h_p + h_t·τ − f_after(x⁺)·τ
unassigned row:   s⁺ = s⁻ + (f_before(x⁻) − f_after(x⁺))·τ
```

- Oracle: `event-beside-clock-switch`.
- **Open:** the event jump reads `f⁻` on the after-branch, and the switch jump
  then adds its gap at `x⁺` (#767, `event-coincident-with-clock-switch`).

## 6. A counter clock

BNGL reads time through a *counter* species: `0 -> C() rc`, exposed through a
group conventionally named `t`. A threshold on the clock crosses where
`C(t*) = θ(p)`. Differentiating that crossing condition gives:

```
τ = (θ_p − s_C(t*⁻)) / C'(t*)
```

The clock's own sensitivity `s_C` is nonzero when the column is the clock's seed
parameter, its sensitivity_ic axis, or its rate. The jump itself is §5's.

- `_unit_rate_clock_indices` (`_switch_sensitivity.py`) recognizes a clock, and
  `compute_switch_time_sens` treats it as time plus an offset.
  `land_clock_on_threshold` (`cvode_simulator.cpp`) sets the counter to its exact
  threshold value at the stop.
- Oracle: `counter-clock-threshold`, `not-a-clock`.
- **Open:** the `s_C` term is dropped, and sensitivity_ic columns are never jumped
  (#725: `counter-clock-seed`, `counter-clock-rate`, `counter-clock-ic-axis`). The
  recognizer takes any species whose RHS is 1 at two probe points for a clock for
  the whole run (#733, `not-a-clock-read-as-one`).

## 7. The comoving frame

A pulse that rises from zero as `s^(a−1)`, with `s = (t − on)/D` and `1 < a < 2`,
has `∂f/∂on ~ s^(a−2)`. That is unbounded just past the onset, so the plain column
cannot be integrated there. Past a crossing that `p` moves at rate `c = τ`, the
solver integrates `V = S + c·f` instead:

```
V' = f_x·V + β,      β = f_p + c·Σ_clock (∂f/∂x_clock)·x_clock'    (or f_p + c·∂f/∂t for time())
S  = V − c·f                                                     wherever S is read
```

`f_p` and `c·∂f/∂clock` carry the singular part with opposite signs, so `β` stays
bounded. With a counter clock, `V`'s clock rows are held at `S` (zero). `J·V`
skips the clock column, so every rate that reads the clock has to put its
`c·∂f/∂x_clock` into `β`.

- Python: `_comoving_coefficients` and `_functional_comoving_plan` emit `c` and
  `β`, and `_guard_clock_columns` masks the clock column (`_codegen.py`).
- C++: `comoving_enter` (at the crossing) and `comoving_leave` (at the next
  restart) in `cvode_simulator.cpp`.
- Oracle: `comoving-onset`, `comoving-onset-loose`, `comoving-closing-edge-regular`.
- **Open:**
  - `β` keeps `c·∂f/∂clock` only for laws whose text names the clock. Mass action,
    a species factor or an observable that reads the clock species loses it
    (#749, `comoving-onset-clock-fed-rate`).
  - A column for a *derived* onset never gets its comoving case (#750,
    `comoving-onset-derived`).
  - A window closing as `(1−s)^(a−1)` is singular on the *approach* to its
    crossing, which the frame, entered at the crossing, never covers (#760,
    `comoving-closing-edge`).

## Order at one instant

When an event root and a switch land on the same instant, the run loop applies
them in this order:

1. The event jump (§3, §4) at `x⁻`.
2. The switch jump (§5), at the post-event state.

The two steps together must equal §5's coincident-event formula; #767 is the case
where they do not.

A comoving frame (§7) is entered inside the switch jump at its crossing and left
at the next restart. It is never entered on a run that has events.
