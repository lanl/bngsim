#pragma once

#include "model.hpp"
#include "rulemonkey/types.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace rulemonkey {

class Engine {
public:
  Engine(const Model& model, uint64_t seed, int molecule_limit = -1);
  ~Engine();

  // Engine owns a unique_ptr<Impl> with PIMPL semantics — copying it
  // would alias the per-engine RNG and pool state, which is never what
  // a caller wants.  Move is also disabled because callers always take
  // an Engine by reference (the live session is held inside the
  // surrounding Simulator).
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;

  // Initialize agent pool from seed species; compute initial propensities.
  void initialize();

  // Run SSA from current time, recording observable values at sample points.
  // `should_continue`, if non-empty, is polled inside the SSA loop; returning
  // false throws `rulemonkey::Cancelled` out of run() at a safe between-event
  // point (no partial event state is left exposed).
  Result run(const TimeSpec& ts, const CancelCallback& should_continue = {});

  double current_time() const;
  std::vector<double> get_observable_values();

  // Global-function values for the current pool state, in the same order
  // as `function_names()`.  Non-const for the same reason as
  // get_observable_values(): the values are re-evaluated on demand.
  std::vector<double> get_function_values();

  // Names of the model's global (non-local) functions, in XML
  // declaration order.  Parallel to a Result's `function_names`.
  std::vector<std::string> function_names() const;

  int get_molecule_count(const std::string& type_name) const;
  void add_molecules(const std::string& type_name, int count);

  // --- Pattern-keyed species methods (issue #9 §1) -----------------------
  //
  // Each takes a `Pattern` produced by parse_species_pattern: an exact,
  // fully-specified, connected species.  These are paused-session
  // calls — none touches the SSA event loop.

  // Live instance count of the species denoted by `pat` — `pat`
  // canonicalized, then matched byte-equal against the pool's
  // canonical labels (the §2 species_count path).
  long get_species_count(const Pattern& pat) const;

  // Instantiate `count` fresh copies of the species `pat` into the
  // pool, then resync rule propensities and observable tracking.
  // Throws if `count <= 0`.
  void add_species(const Pattern& pat, int count);

  // Remove `count` live copies of the species `pat` from the pool,
  // then resync.  Throws if `count <= 0`, or if fewer than `count`
  // live copies exist.
  void remove_species(const Pattern& pat, int count);

  // Drive the live count of species `pat` to exactly `count` by adding
  // or removing the difference.  Throws if `count < 0`.
  void set_species_count(const Pattern& pat, int count);

  // Evaluate an arbitrary BNGL expression string against the current
  // session state (issue #9 §1).  Resolvable symbols: every model
  // parameter, the bare clock `t` and the `time()` builtin, every
  // observable, and every global function — all settled against the
  // current pool.  `extra` name=value bindings are layered last, so a
  // caller-supplied name shadows a model symbol on a clash.  Non-const:
  // observables / functions are recomputed on demand, as in
  // get_observable_values().  Throws if `expr` fails to compile.
  double evaluate_expression(const std::string& expr,
                             const std::unordered_map<std::string, double>& extra = {});

  // Save full simulation state to file; load restores pool and derived state.
  void save_state(const std::string& path) const;
  void load_state(const std::string& path);

  // Enumerate the live species in the current pool: walk every complex,
  // canonicalize it, and group graph-isomorphic complexes into one row
  // with a summed instance count.  Rows are sorted by the canonical
  // species string.  A one-shot pool walk (issue #9 §2 batch mode); an
  // un-initialized engine yields an empty list.
  std::vector<SpeciesRow> enumerate_species() const;

  // Write enumerate_species() to `path` as a BNG-format `.species` file
  // (BNG2.pl `readNFspecies`-compatible: `#` comment header, then one
  // `<pattern>  <count>` line per species).  Throws if `path` cannot be
  // opened for writing.
  void write_species_file(const std::string& path) const;

  // Count of live complexes whose canonical species string equals
  // `canonical` — expected to be a string previously emitted by
  // enumerate_species().  No pattern parsing: a string that is not
  // byte-identical to a canonical label RM would emit yields 0.  A
  // batch query — internally a from-scratch pool walk, the same cost
  // as enumerate_species().  Empty pool / un-initialized engine: 0.
  long species_count(const std::string& canonical) const;

  // Number of live complexes in the pool — the network-free analogue
  // of the total complex/particle population.  Equals the sum of every
  // enumerate_species() row count, but is computed without
  // canonicalization.  Empty pool / un-initialized engine: 0.
  long total_complex_count() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rulemonkey
