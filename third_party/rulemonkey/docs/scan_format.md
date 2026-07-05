# `.scan` output format and parameter sweeps

`rm_scan` sweeps one model parameter across a range of values and reports
the endpoint response at each value, in `.scan` format on `stdout`.  CPU
time goes to `stderr`.  This document is the authoritative reference for
the format and the sweep semantics.

`rm_scan` is the RuleMonkey equivalent of BioNetGen's `parameter_scan`
and `bifurcate` actions.  The sweep math (linear / geometric spacing,
endpoint extraction, the forward/backward continuation) follows
BioNetGen; see [BNG correspondence](#bng-correspondence) for the precise
mapping and the two deliberate differences.

## Two sweep modes

- **`parameter_scan`** (default) — for each parameter value, run the
  model over a per-point time window and record the *endpoint*: the
  observable (and, opt-in, global-function) values at `t_end`.  This is
  the standard dose-response / steady-state-characterization sweep.
- **`bifurcate`** (`--bifurcate`) — run the sweep forward (`min → max`)
  and then backward (`max → min`) as one continuous trajectory: the
  molecular state carries over from each point to the next *and* across
  the forward/backward turn.  A bistable model then traces out a
  hysteresis loop, with the forward and backward branches landing on
  different endpoint values over the bistable parameter range.

## `parameter_scan` file shape

One header line, then one data line per swept parameter value.  Every
line is tab-separated.

```
# <param><TAB>obs_1<TAB>...<TAB>obs_K[<TAB>fn_1<TAB>...<TAB>fn_F]
v_0<TAB>e_1,0<TAB>...<TAB>e_K,0[<TAB>g_1,0<TAB>...<TAB>g_F,0]
v_1<TAB>e_1,1<TAB>...<TAB>e_K,1[<TAB>g_1,1<TAB>...<TAB>g_F,1]
...
```

- The header begins with a literal `#` and a space, then the swept
  parameter name, then the observable names in BNGL `begin observables`
  declaration order.
- Each data row is the parameter value `v_k` followed by the endpoint
  observable values `e_*,k` at that parameter value.
- The bracketed `fn_*` / `g_*` columns — the model's *global functions*
  (the `begin functions` entries with no local, per-molecule arguments)
  — are present only when `--print-functions` is passed.  This opt-in
  mirrors BNGL's `print_functions=>1` and matches `rm_driver`'s
  `--print-functions` flag (see [issue #7][i7] and
  [`gdat_format.md`](gdat_format.md)).

## `bifurcate` file shape

`--bifurcate` emits both branches as one stream.  Each observable (and
each function, with `--print-functions`) gets a `_fwd` and a `_bwd`
column:

```
# <param><TAB>obs_1_fwd<TAB>obs_1_bwd<TAB>...<TAB>obs_K_fwd<TAB>obs_K_bwd
v_0<TAB>f_1,0<TAB>b_1,0<TAB>...
v_1<TAB>f_1,1<TAB>b_1,1<TAB>...
...
```

Both branches are reported on the **same ascending parameter axis**:
row `k` carries the forward-branch endpoint and the backward-branch
endpoint *at the same parameter value* `v_k`.  The backward sweep
physically runs `max → min`, but its rows are re-ordered into ascending
order so a `_fwd` / `_bwd` pair on one row is directly comparable.  For
a model with no hysteresis the two columns agree to within stochastic
noise; a bistable model shows them diverging over the bistable range.

## Numeric format

- All columns are tab-separated.
- Doubles are printed at 17 significant digits — the IEEE-754 binary64
  round-trip precision, identical to `rm_driver`'s `.gdat` output — so a
  `.scan` file reloads without precision loss.
- Observable and function names are emitted verbatim from the XML.

## Sweep specification

The swept values come from either an explicit list or a generated range:

- `--values v1,v2,...` — an explicit comma-separated value list, used
  verbatim and in the given order.
- `--min M --max X --n-points N` — a generated range of `N` values from
  `M` to `X`.  Linear spacing by default; `--log` gives geometric
  (log-uniform) spacing, which requires `M > 0` and `X > 0`.

`--values` takes precedence: if it is given, `--min` / `--max` /
`--n-points` are ignored.

Range validation, matching BioNetGen:

- `N >= 1` is required.
- `N >= 2` is required when `min != max` (a non-degenerate range needs
  at least two points).
- `N == 1`, or any `N` with `min == max`, yields a flat sweep — the
  single value repeated.

Each point is simulated over the window `[--t-start, --t-end]`
(`--t-start` defaults to 0).  `--n-steps` sets the per-point sampling
resolution; it does not change the recorded value, since the endpoint
is always the sample at `t_end`.  `--n-steps` defaults to 1.

Every point uses the same `--seed`.  As in BioNetGen, the seed is a
run-level setting, not per-point: the points share one random stream and
differ only by the swept parameter, which keeps the sweep reproducible
and removes seed-to-seed noise from the comparison between points.

## `reset_conc`: fresh vs. carried-over state

`--reset-conc` controls whether molecular state is reset between points
(it is forced off for `--bifurcate`, where carry-over is intrinsic):

- `--reset-conc 1` *(default)* — every point starts fresh from the
  model's seed species.  Points are independent runs; this is the
  dose-response case.  A swept parameter that feeds a seed-species
  concentration (a ligand dose) re-resolves that concentration at every
  point, matching BNG2/NFsim.
- `--reset-conc 0` — every point continues from the previous point's
  final molecular state (BNG `reset_conc=>0`).  The sweep is then one
  continuous-time trajectory.

## BNG correspondence

| BioNetGen                         | `rm_scan`                          |
|-----------------------------------|------------------------------------|
| `parameter_scan({parameter=>…})`  | default mode                       |
| `bifurcate({parameter=>…})`       | `--bifurcate`                      |
| `par_min` / `par_max` / `n_scan_pts` | `--min` / `--max` / `--n-points` |
| `par_scan_vals`                   | `--values`                         |
| `log_scale=>1`                    | `--log`                            |
| `reset_conc=>0`                   | `--reset-conc 0`                   |
| last `.gdat` row of each run      | the endpoint row                   |

Two deliberate differences from BioNetGen:

1. **Single combined output stream.** BNG's `parameter_scan` writes a
   working directory of per-point `.gdat` files plus one `.scan` file,
   and `bifurcate` writes one `.scan` file *per observable*
   (`<prefix>_bifurcation_<obs>.scan`).  `rm_scan` instead computes
   endpoints in-process and emits a single combined `.scan` stream on
   `stdout` — no working directory, no per-observable file split.  The
   row content (parameter value + endpoint observables) is equivalent.
2. **Continuous clock under carry-over.** With `--reset-conc 0` (and
   always for `--bifurcate`), `rm_scan` runs one continuous-time
   trajectory: the simulation clock advances cumulatively across scan
   points rather than resetting to 0 at each point as BNG does.  For the
   overwhelmingly common case of a model with no explicit time
   dependence this is observationally identical, since SSA propensities
   do not depend on absolute time.  A model whose rate laws reference
   `time` directly (e.g. via a `TFUN` table function keyed on time) will
   differ between the two, because each carried-over point sees a later
   clock value.

[i7]: https://github.com/richardposner/RuleMonkey/issues/7
