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
  `event-assignment-reads-state`, `event-fires-after-t-start`.

**`h_t` and `rateOf`.** `h_t` is the assignment's explicit time dependence:
`Tlast = time`, `B = time + 1`, or a rule or function that reads `time`. An
assignment that reads `rateOf(x_j)` reads `f_j`. Its `h_x` and `h_p` therefore
include `∂f_j/∂x` and `∂f_j/∂p`, and the finite-difference sync that forms them
has to refresh the `rateOf` buffer.
- **Covered:** `h_t` is a central difference of the assignment in time at
  `(x⁻, p₀)`, with the functions and the `rateOf` buffer re-evaluated at each
  offset, so time read through a rule counts (#735,
  `event-assignment-reads-time`, `event-assignment-reads-time-clock`). The sync
  refreshes the `rateOf` buffer and re-evaluates the functions after it, and a
  `rateOf` read counts every parameter as support (#764,
  `event-assignment-reads-rateof`).
- **Refused:** an assignment whose value is not smooth where the event reads it:
  a step in time at the fire instant (`u := piecewise(5, time >= T0 + 1, 0)`,
  `B = u`, fired at `time >= T0 + 1`), a step in a parameter at the parameter's
  own value, a bend, or a value that turns inside a millionth of what it reads.
  `∂h/∂p`, `∂h/∂x` and `∂h/∂t` are central differences, and across any of these
  a central difference is not a derivative. Across a smooth value the
  difference over half the step is half as large and the second difference
  about the point a quarter as large; the run is refused where either is out by
  more than a part in 1e3 of what the value moves by across the step, and by
  more than what may be rounding: 16 ulp of the largest thing the value reads,
  and no more than 2e-9 of the value's own size. A value that is not finite at
  the point or beside it is refused too. A difference retaken over a wider
  step (#767) is across a value that is straight there, and is not asked. A
  species is refused only where a column carries something through it, and the
  time only where a column moves the fire time (#915,
  `test_event_value_time_step_sensitivity.py`). A value that reads the same
  at both ends of the step and at the point is asked between them, off any
  simple fraction of the step, so a sawtooth whose period divides the step is
  refused. A value that is flat at the point is not refused, in two cases.
  One that is even about the point and leaves it as a power of the distance
  of order 1.75 or more (`X⁴/(K⁴ + X⁴)` at X = 0): its derivative there is 0
  and so is the central difference, whatever its size. And one that leaves the
  point as such a power on each side and does not move to speak of: its slope
  across the step, times what is moved (or 1, if that is larger), is under a
  millionth of the value (or of 1, if the value is larger than that, or
  smaller than what it moves by), as `X³/(8 + X³)` at X = 0. A power from the
  point that is a
  slope across the step is refused: `1e6·max(X − 1, 0)^1.81` at X = 1, a ramp
  squared that is done inside the step, a Hill function of X at 0 whose
  half-saturation is within about a hundred thousand steps. A step or a bend
  that moves
  with the event has a derivative, and is refused with the rest.
- **Not caught**, and returned as before:
  - a bend under a value that reads something a million times its own size,
    where the change of slope times what is moved is under about 0.8% of the
    value: `kcat·E0·X/(Km + X) + max(X − 3, 0)` at X = 3 with kcat at 1e9 and
    E0 at 1e-6 returns 0.61 for 0.11 or 1.11, and is refused with kcat at 1e3
    and E0 at 1;
  - a bend that changes the partial it is in by under about 0.8%, where the
    column's terms cancel to less than that: `1000·(time − T0) + max(time −
    2.3, 0)` fired at T0 + 1 = 2.3 returns 0.5 for 0 or 1;
  - a smooth value whose difference rounds, in line, by more than it resolves:
    `(1 − exp(−k·time))/k` at k = 1e-10 returns 1.11 for 1;
  - a feature centred on the point, narrower than 0.6 of the step, that
    returns to the point's value at both ends of it:
    `piecewise(X − 3, abs(X − 3) < 3e-7, 0)` at X = 3 returns 0 for 1;
  - a sawtooth whose period divides the step, riding on a slope:
    `(2·X + 0.25) − floor(2·X + 0.25) + 1e-3·X` at X = 1e6 returns 0.001 for
    2.001;
  - a kink under an even term steep enough to hide it:
    `abs(X − 3) + 1e13·abs(X − 3)³` at X = 3 returns 0 for −1 or 1.

**At `t_start`.** An SBML event with `initialValue=false` whose trigger is already
true at `t_start` fires there. Then `τ = 0`, and `s⁻` is the seed that is already
in `yS`. There is nothing to interpolate from, because CVODES has not taken a step.
- **Open:** `s⁻` is interpolated anyway, which gives NaN at `t_start = 0` and
  `CV_BAD_T` elsewhere, so the run is refused (#717, `event-fires-at-t-start`).

## 4. A batch of events at one instant

Events that fire together run one at a time: highest priority first, with random
tie-breaking. A fire can arm another event at the same instant (a cascade), which
joins the batch. Each instance's assignment reads the state at its own trigger
time when `useValuesFromTriggerTime=true`: that is the pre-batch state for an
instance armed before the batch, and the state after the arming fire for a
cascaded one. With `false`, it reads the state the earlier fires left. A
non-persistent instance whose trigger an earlier fire made false is cancelled.
The jump composes in *execution* order:

```
D₀ = s⁻ + f⁻·τ
D_k = D_{k−1}, except on the rows fire k assigns:
      D_k = h_{k,y}·D_read + h_{k,p} + h_{k,t}·τ
      D_read = D₀       UVFTT=true, armed before the batch
             = D_j      UVFTT=true, armed by fire j of this batch
             = D_{k−1}  UVFTT=false
      (a cancelled instance contributes nothing)
s⁺ = D_n − f⁺·τ                                     f⁺ after the whole batch
```

Sensitivities refuse a batch with a cascaded instance today, so only the first
and third cases of `D_read` are reachable.

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
- Oracle: `time-switch`, `time-switch-just-after-an-output`, `state-switch`,
  `state-switch-beside-a-large-pool`.

Whether a state switch's branches meet is read from the flux of the reactions
that read the switch alone (#763), at two probes on each side of the surface
along the flow. Both branches are extended to the root, and the difference of
the extensions is the branch change. Under the rounding of its four readings,
16·ε of them, it is no jump. Past `1e-6` of the rate that drives the crossing it
is one. That tolerance is for a flux that vanishes on both branches and differs
across a pair of probes by its slope times the crossing's speed; it is far too
wide for a step under a fast threshold species, where a jump of 3 under a pool
moving at 5e6 was dropped (#917).

Between the two, the branches are read at one state: the state on the surface,
with only the species the residual reads moved a few ulp to either side of it,
16 ulp of the residual and more where that does not flip its sign. Every term
that does not switch is the same in both readings, whatever it rounds by, so
their difference is the step, and it is a jump unless it is within

- the rounding of the two readings, 16·ε of them;
- eight times what the flux does on its own over as far again on each side: a
  term that reads the residual's species moves with them;
- eight times a tread, where the flux rounds as a staircase: `s() + off − thr`
  with `s() = B − off` moves in steps of an ulp of `off`. A side that reads the
  same further out is followed to where it moves, and it has to move a second
  time to count, so that another switch of the same rate law out there is not
  taken for a tread.

Where the two sides cannot be reached by moving those species, the reading
stands as the drive tolerance has it.

The same reading takes back a jump past the drive tolerance that is not the
switch's. A term beside a continuous switch that rounds as a staircase and does
not read the switch steps between the probes, and its tread went into the column
as a jump: −2029.6 for 4.42. The crossing is continuous where the two branches
at one state are the same to the rounding of the two readings, and within the
drive tolerance with each side carried to the surface along its own slope. To
their rounding and no more: what the flux does further out is no measure here,
since a second switch of the same rate law a few ulp away is in that reading,
and a staircase that reads the threshold species puts a tread between the two
sides.

It also sizes a jump that such a term stepped beside. The jump applied is the
whole right-hand side's change between the probes, extended to the surface, and
a tread of 15 beside a jump of 3 gave −1826 for 207.9. Where one switch jumps
and its branch change read at one state differs from the one the probes give by
more than that reading allows, the jump is the one read at one state.

**An event and a switch at the same instant.** `f⁻` is the before-branch at
`x⁻`, and `f⁺` is the after-branch at `x⁺`:

```
assigned row:     s⁺ = h_x·(s⁻ + f_before(x⁻)·τ) + h_p + h_t·τ − f_after(x⁺)·τ
unassigned row:   s⁺ = s⁻ + (f_before(x⁻) − f_after(x⁺))·τ
```

The event jump reads both of its flows on the switch's before-branch, a nudge
before the switch's own time or with a counter a hair short of its threshold,
and the switch jump that follows adds `(f_before(x⁺) − f_after(x⁺))·τ` (#767).

A switch the event comes apart from under some column, a fixed one or one
fitted by another parameter, leaves a kink unless the two commute:
`H·Δ(x⁻) = Δ(x⁺)`, with `H` the batch's Jacobian and `Δ = f_before − f_after`.
`Δ` is the limit of `f` from before the instant less its limit from after, each
carried to the switch from reads one, two and four nudges out. On an instant
that several crossings of one clock share, where not all of them move with the
event, the instant is asked as a whole and each record on it for its own jump,
read by its isolation bump (#375). The run is refused where either disagrees.

- Oracle: `event-beside-clock-switch`, `event-coincident-with-clock-switch`.
- **Open:** a `floor` step on a fitted switch (#944), a state-dependent switch
  within the integration tolerance of an event (#945), a state-dependent
  switch and a time switch on one instant (#946), and two time switches of one
  rate law on one instant with no event (#951).

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
- C++: `comoving_enter` (at the crossing), `comoving_enter_ahead` (before it) and
  `comoving_leave` (at the next restart) in `cvode_simulator.cpp`.
- Oracle: `comoving-onset`, `comoving-onset-loose`, `comoving-onset-clock-fed-rate`,
  `comoving-onset-derived`, `comoving-closing-edge-regular`, `comoving-closing-edge`.

A window closing as `(1−s)^(a−1)` is singular on the *approach* to its crossing:
the forcing of the column that moves the edge goes as `(1−s)^(a−2)` before it. A
frame entered at the crossing has already integrated that. The generator marks
such a case (`bngsim_codegen_comoving_approach`, asked at the run's parameter
values: an exponent `a − 1` is singular below 1 and not at 0, so for `a < 2`),
and a plain column whose next switch time moves at that case's `c` enters ahead
of it (#760). A case with nothing but closing powers, none of them singular at
the run's values, is not entered at all: the column is plain, as it is without
the case. `V = S + c·f` holds for any constant `c` anywhere, so entering early
changes only which column is integrated.

A column is in its frame only next to a crossing it moves. In its frame it
carries `c·∂f/∂t` for everything `f` does, so a frame that reaches the closing
edge of a window the column does not move meets that window's singular forcing.
After a restart the run takes a stop a sixteenth of the stretch short of the
next crossing (`kComovingEntryFraction`):

- frames are left there, unless every column in one moves the next switch time
  at its frame's own `c` and no other crossing lies before it, or no crossing is
  left ahead at all;
- a column whose next switch time is approached through a singular power enters
  there, where no other crossing lies between.

Not sooner after the restart: a window that also opens as a power has an `f`
whose slope is unbounded just past the opening, and the plain column of the
parameter that moves it is no better there. A stretch too short to stand off
from such an edge is refused (`kComovingMinStretch`, 8,192 of the root finder's
read-backs): a restart that soon after the crossing a frame was entered at, or
that soon before the switch time a column has to enter ahead of.

A frame is left against `f` just before the stop. At one of the run's own stops
that is `f` where it is. At a registered root it is `f` kept from the stop the
frame entered at, when the root is within the root finder's reach of that stop,
and otherwise two reads just before the root extended to it.

A crossing that shares its instant with another and is itself the edge of such
a window is refused (#949): its jump is read by its isolation bump (#375), a
hair from the power's zero. Only on the edge a frame was entered ahead of,
which is one whose power is singular at the run's values. A power that opens
on the instant is not yet on when it is read that way.

- **Open:** a closing edge whose crossing is not a switch time of the run (a
  state-dependent one) is still approached in the plain column.
- **Open:** a carried phase that starts exactly on a closing edge raises (the
  clock lands on the threshold, where the split power is 0^(a−2)). Main returns
  a value there.
- **Open:** a column in its frame reads back as `V − c·f`, to the tolerance of
  `c·f` and not of `S`. Beside a background flux 5000 times the pulse, dX/dD is
  9e-3 off at rtol 1e-6 (3e-3 before #760) and 3e-4 at rtol 1e-8 (4e-3 before).
- **Open:** a window whose width is written negative, `s = (t − on)/(−E)`, is
  not split, and its closing edge is 0.4% off as before #760.
- **Open:** a sensitivity to a counter clock's own initial value or rate has no
  frame (#948).
- **Open:** a run that has an event keeps every column plain, so a closing edge
  is 0.4% off there and a crossing on it is not refused (#958).
- **Open:** the approach is asked per parameter and `c`, not per crossing, so
  the onset column of a window that closes as a singular power enters ahead of
  the window's opening too. Harmless, except that a stretch too short before
  that opening, or a crossing on it, is refused where nothing is singular.
- **Open:** a width written as a rate, `s = (t − on)·r`, has no case for `r`:
  0.4% off, as before #760.
- **Open:** a window narrower than about 1e-6 of the time: dX/dD is 3% off at a
  width of 1e-6 at t = 3 and 33% at 1e-9, as before #760.

## Order at one instant

When an event root and a switch land on the same instant, the run loop applies
them in this order:

1. The event jump (§3, §4) at `x⁻`.
2. The switch jump (§5), at the post-event state.

The two steps together equal §5's coincident-event formula, with the event's
flows read on the switch's before-branch (#767).

A comoving frame (§7) is entered inside the switch jump at its crossing, or
ahead of a crossing approached through a singular power, and left at the next
restart or at the stop the run takes short of the next crossing. It is never
entered on a run that has events.
