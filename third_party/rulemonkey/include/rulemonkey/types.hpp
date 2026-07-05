#pragma once

// Interface-only contract snapshot for the RuleMonkey <-> BNGsim boundary.
// This header is intentionally free of engine code.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace rulemonkey {

// Cooperative-cancellation hook for long-running run() / simulate() / step_to()
// calls.  The engine polls this callback periodically from the SSA event loop
// (granularity is roughly every 1024 SSA events, fine enough for wall-clock
// timeouts but coarse enough to keep per-event overhead negligible).  Return
// `true` to let the simulation keep running, `false` to cancel; cancellation
// causes the engine to throw a `Cancelled` exception out of the current call.
// A default-constructed (empty) std::function disables the check entirely —
// callers who don't need cancellation pay no per-event overhead.
using CancelCallback = std::function<bool()>;

// Thrown out of run() / simulate() / step_to() when the supplied
// `CancelCallback` returns false.  Inherits from std::runtime_error so generic
// catch handlers still see it, but a dedicated type lets embedders distinguish
// a cooperative cancellation (a clean stop requested from outside) from a
// genuine engine error.
class Cancelled : public std::runtime_error {
public:
  Cancelled() : std::runtime_error("RuleMonkey simulation cancelled") {}
  explicit Cancelled(const std::string& msg) : std::runtime_error(msg) {}
};

struct TimeSpec {
  double t_start = 0.0;
  double t_end = 0.0;
  int n_points = 0;

  // Explicit, possibly non-uniform output instants.  When non-empty this
  // OVERRIDES the uniform `n_points` grid: the engine records output at
  // exactly these times instead of at `t_start + i*(t_end - t_start)/n_points`.
  // The list is the network-free analogue of BNG2.pl's `simulate_nf`
  // sample_times branch — the typical use is recording at an experimental
  // dataset's independent-variable points so an embedder (e.g. PyBNF via
  // BNGsim) can fit against them directly.
  //
  // Contract:
  //   * Must be sorted ascending — the SSA recorder advances a single cursor
  //     through this vector and never looks back, so an out-of-order list
  //     throws std::runtime_error rather than silently mis-sampling.  Equal
  //     adjacent times are allowed (they record the same instant twice).
  //   * Times are absolute, on the same clock as `t_start` / `t_end`.
  //   * `t_end` still bounds the run: the SSA loop stops at `t_end`, so set
  //     `t_end` to at least the largest sample time you care about.  Times at
  //     or beyond `t_end` (and any below `t_start`) are still emitted, recorded
  //     at the final / initial state respectively, so the output row count
  //     always equals `sample_times.size()`.
  //
  // Sampling is non-invasive: output times never draw from the RNG or perturb
  // reaction selection, so a run with explicit `sample_times` is bit-identical
  // to the uniform-grid run at any instants the two schedules share, and
  // `Result::event_count` is unaffected.
  std::vector<double> sample_times;
};

// Plain data carrier for one simulation run's output.
//
// `observable_data` is column-major: `observable_data[obs_idx][t_idx]`
// gives the value of the obs_idx-th observable at the t_idx-th sample
// time.  Both inner and outer vectors have the same length on every
// successful run (n_observables and n_times respectively).
//
// `function_data` mirrors `observable_data` for BNGL global functions —
// the entries of a `begin functions ... end functions` block that have
// no local (per-molecule/per-species) arguments.  These are the derived
// quantities models routinely use as their measured/fitted outputs
// (e.g. `Clusters() = monomer + dimer + ...`).  The engine already
// evaluates them for rate laws; this is an exposure of values it
// already computes.  `function_data[fn_idx][t_idx]` gives the value of
// the fn_idx-th global function at the t_idx-th sample time, sampled at
// the same time points as the observables.  Local functions are
// excluded — they evaluate per-molecule and have no single global
// value — so `function_names` may be shorter than the model's full
// `begin functions` block.
//
// Construction and population is the engine's responsibility — host
// code reads but does not write to these fields.
struct Result {
  std::vector<double> time;
  std::vector<std::string> observable_names;
  std::vector<std::vector<double>> observable_data;
  std::vector<std::string> function_names;
  std::vector<std::vector<double>> function_data;
  int64_t event_count = 0; // SSA events fired during this run.

  std::size_t n_times() const noexcept { return time.size(); }
  std::size_t n_observables() const noexcept { return observable_names.size(); }
  std::size_t n_functions() const noexcept { return function_names.size(); }
};

// Specification for a parameter sweep (parameter_scan / bifurcate).
//
// The swept parameter takes either an explicit value list (`values`) or a
// generated range (`par_min` / `par_max` / `n_points`, with optional
// `log_scale` geometric spacing).  `values` takes precedence when non-empty;
// this mirrors BNG's `par_scan_vals` vs `par_min`/`par_max`/`n_scan_pts`.
//
// `per_point` is the time window simulated at each point; the recorded
// response is the *endpoint* value (the observables / global functions at
// `t_end`), matching BNG's parameter_scan, which extracts the last `.gdat`
// row of each per-point run.
struct ScanSpec {
  std::string parameter;      // declared parameter name to sweep
  std::vector<double> values; // explicit value list; precedence over range
  double par_min = 0.0;       // range form (used when `values` is empty):
  double par_max = 0.0;
  int n_points = 0;
  bool log_scale = false; // geometric (log) spacing across the range
  // When true, every point runs fresh from the model's seed species (an
  // independent dose-response point).  When false, each point continues
  // from the previous point's final molecular state (BNG `reset_conc=>0`).
  // bifurcate() always carries state over and ignores this field.
  bool reset_conc = true;
  TimeSpec per_point; // time window simulated at each point
};

// One parameter_scan result: the endpoint observable / global-function
// values at each swept parameter value.  `*_endpoints` are row-major —
// `observable_endpoints[point_idx][obs_idx]` — one row per parameter value,
// columns parallel to `observable_names` / `function_names`.
struct ScanResult {
  std::string parameter;
  std::vector<double> param_values;
  std::vector<std::string> observable_names;
  std::vector<std::string> function_names;
  std::vector<std::vector<double>> observable_endpoints;
  std::vector<std::vector<double>> function_endpoints;

  std::size_t n_points() const noexcept { return param_values.size(); }
};

// bifurcate result: a forward sweep (par_min -> par_max) and a backward
// sweep (par_max -> par_min) run as one continuous trajectory, so a
// bistable model surfaces hysteresis between the two branches.
//
// Both ScanResults carry the *same ascending* `param_values`: the backward
// sweep's endpoints are re-ordered out of run order into ascending
// parameter order, so column `i` of `forward` and `backward` always refer
// to the same parameter value.
struct BifurcateResult {
  ScanResult forward;
  ScanResult backward;
};

// One row of an enumerated species census: a canonical BNGL species
// pattern string and the number of complex instances of that species
// currently live in the pool.  Two complexes contribute to the same
// row iff they are graph-isomorphic (same species); `species` is the
// true canonical label, so it doubles as the dedup key and as a valid
// BNG-format pattern line.  Produced by
// `RuleMonkeySimulator::enumerate_species()` and written one per line
// to a BNG-format `.species` file (see `write_species_file()`).
struct SpeciesRow {
  std::string species; // canonical BNGL pattern string
  long count;          // number of complex instances of this species
};

enum class Severity { Warn, Error };

struct UnsupportedFeature {
  Severity severity;
  std::string element; // XML element or attribute name
  std::string feature; // human-readable BNGL feature description
};

} // namespace rulemonkey
