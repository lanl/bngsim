#pragma once

// =============================================================================
// Dev profiler / invariant infrastructure for engine.cpp.
//
// Carries the gate constants, sampling cadences, profile-struct
// definitions, and end-of-run reporting bodies that the engine uses
// to instrument hot paths during optimization sprints.  Profile-struct
// instances live as per-Engine members on `Engine::Impl` / `AgentPool`
// in engine.cpp; nothing in this header has process-wide mutable
// state.  None of it is part of the runtime contract or the public
// API — pure dev-time scaffolding.
//
// In default builds the master macro `RM_DEV_PROFILES` is undefined,
// every gate is `false`, and every `if constexpr (k*Profile)` block in
// engine.cpp is dead-stripped by the compiler.  Opt in with:
//
//   cmake --preset release -DRULEMONKEY_ENABLE_DEV_PROFILES=ON
//
// Then flip individual gates here back to `false` to disable just that
// profiler.  Invariant gates (the second block below) stay `false`
// even when profilers are on; they run an O(N) reference-vs-fast-path
// equality check on every call, intended for correctness verification
// during refactors of the matching/sampling fast paths.
//
// Adding a new profile.  Five mechanical steps:
//
//   1. Declare a gate + (optional) sample cadence in the gate block:
//        inline constexpr bool kMyPhaseProfile = kDevProfilesEnabled;
//        inline constexpr int kMyPhaseProfileSampleEvery = 16;
//   2. Define a struct in the "Profile struct definitions" block whose
//      fields match what you increment.  Counters are `uint64_t`; chrono
//      sums use `_ns` suffix.  Field order must match increment order in
//      engine.cpp (the report function below traverses by name, not by
//      order, but the convention keeps grep useful).
//   3. Add a `MyPhaseProfile my_profile_;` member to the owning class
//      in engine.cpp (Engine::Impl or AgentPool — pick by call-site
//      ownership; per-Engine, never global).
//   4. Wrap increments in `if constexpr (kMyPhaseProfile) { ... }` at
//      every call site.  For chrono, gate on the sample counter:
//        if constexpr (kMyPhaseProfile) {
//          if (++my_profile_.calls % kMyPhaseProfileSampleEvery == 0) {
//            auto t0 = std::chrono::steady_clock::now();
//            ...
//            my_profile_.total_ns += elapsed_ns(t0);
//            my_profile_.sampled_calls++;
//          }
//        }
//   5. Add a `report_my_profile()` body to the report block at the
//      bottom of this header and call it from Engine::Impl::~Impl()
//      next to the existing `report_*()` calls.
//
// All five steps are local; nothing in the public API changes.  When
// the master gate is OFF, every `if constexpr` block dead-strips and
// the new struct's per-Engine storage costs sizeof(MyPhaseProfile)
// bytes per Engine instance — no runtime overhead.
// =============================================================================

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rulemonkey {

// ---------------------------------------------------------------------------
// Master gate.  Driven by the RULEMONKEY_ENABLE_DEV_PROFILES cmake option.
// ---------------------------------------------------------------------------
#ifdef RM_DEV_PROFILES
inline constexpr bool kDevProfilesEnabled = true;
#else
inline constexpr bool kDevProfilesEnabled = false;
#endif

// ---------------------------------------------------------------------------
// Profile gates and sampling cadences.  Off by default (master gate is
// off); flip an individual gate to `false` here to mute one profiler
// when the master gate is on.
// ---------------------------------------------------------------------------

// AgentPool::split_complex_if_needed — call counters + sampled chrono.
inline constexpr bool kRemoveBondProfile = kDevProfilesEnabled;
inline constexpr int kRemoveBondProfileSampleEvery = 8;

// count_multi_mol_fast dispatcher and generic body.
inline constexpr bool kCountMultiProfile = kDevProfilesEnabled;
inline constexpr int kCountMultiProfileSampleEvery = 8;

// count_2mol_1bond_fc — engaged-set hot-spot specialized matcher.
// SampleEvery × phase rotor (5) keeps chrono overhead bounded against
// the 215-269 ns/call baseline.
inline constexpr bool kCmmFcProfile = kDevProfilesEnabled;
inline constexpr int kCmmFcProfileSampleEvery = 32;

// select_reactants — five inner paths classified, K-sampled whole-call
// chrono per path, always-on counters for sampler sub-decisions.
inline constexpr bool kSelectReactantsProfile = kDevProfilesEnabled;
inline constexpr int kSelectReactantsProfileSampleEvery = 16;

// fire_rule — per-op-type call counts plus K-sampled sub-phase chrono.
inline constexpr bool kFireRuleProfile = kDevProfilesEnabled;
inline constexpr int kFireRuleProfileSampleEvery = 8;

// incremental_update — two-level sampling: K-sampled outer call, then
// every Mth per-mid entry inner-sampled within those K-sampled calls.
inline constexpr bool kIncrUpdateProfile = kDevProfilesEnabled;
inline constexpr int kIncrUpdateProfileSampleEvery = 8;
inline constexpr int kIncrUpdateProfileInnerSample = 32;

// record_at — full-sample (K=1, fires only n_steps+1 times).
inline constexpr bool kRecordAtProfile = kDevProfilesEnabled;

// incremental_update_observables / flush_species_incr_observables.
inline constexpr bool kObsIncrProfile = kDevProfilesEnabled;
inline constexpr int kObsIncrProfileSampleEvery = 16;

// expr::evaluate hot path — evaluate_rate / evaluate_local_rate /
// update_eval_vars.  Drives issue #6: is function-based rate-law
// evaluation a meaningful fraction of total wall time?  A single
// expr::evaluate call is short (a recursive walk of a small AST), so
// per-call chrono would be swamped by clock overhead — instead we
// keep always-on invocation counters and K-sample whole-call chrono
// on the enclosing functions, then divide.
inline constexpr bool kExprEvalProfile = kDevProfilesEnabled;
inline constexpr int kExprEvalProfileSampleEvery = 16;

// ---------------------------------------------------------------------------
// Invariant gates.  Slow correctness checks that run a reference path
// alongside a fast path and abort on divergence.  OFF by default even
// when the master profiler gate is on — flip on while refactoring the
// fast path, run the suite, flip off.
// ---------------------------------------------------------------------------

// sample_molecule_weighted Fenwick.find vs cumulative-weight check.
inline constexpr bool kFenwickInvariant = false;

// FastMatchSlot specialization vs generic embedding path.  With this
// gate on, count_multi does strictly more work — perf is NOT a valid
// A/B signal under this configuration.
inline constexpr bool kFastMatchInvariant = false;

// Cycle-bond-counter product-molecularity short-circuit vs BFS.
inline constexpr bool kProductMolInvariant = false;

// FastMatchSlot select_reactants fast path vs generic select_multi_mol_unimolecular.
inline constexpr bool kFastSelectInvariant = false;

// Species-observable incremental tracker vs full-walk reference.
// Feature flag itself is `kSpeciesIncrObs` in engine.cpp.
inline constexpr bool kSpeciesIncrObsInvariant = false;

// 2-mol-1-bond-fc obs-tracking specialization vs generic.  Feature flag
// itself is `kObsFastMatch` in engine.cpp.
inline constexpr bool kObsFastMatchInvariant = false;

// Canonical-label cache self-check (issue #9 §2 step 5, plan decision #6).
// The cached-incremental canonical-label layer has no production consumer
// yet (partial scaling, plan §7.2, is future work), so its correctness is
// proven by an invariant: every cached-label read must equal a from-scratch
// recanonicalization.  Unlike the gates above — manually flipped while
// refactoring — this one is driven by the build type: the
// RULEMONKEY_CANONICAL_CACHE_SELFCHECK compile definition is set by
// CMakeLists.txt for Debug and ASan builds and left unset for Release, so
// ctest and the corpus guard tier exercise the invariant while the release
// `.species` sweep stays a plain from-scratch walk with no cache traffic.
// `assert` cannot carry this: the asan preset is RelWithDebInfo, which
// defines NDEBUG and so disables assert.
#ifdef RULEMONKEY_CANONICAL_CACHE_SELFCHECK
inline constexpr bool kCanonicalCacheSelfCheck = true;
#else
inline constexpr bool kCanonicalCacheSelfCheck = false;
#endif

// Local-observable tracker self-check (issue #10).  evaluate_observable_on's
// obs_mol_contrib fast path must equal a from-scratch embedding recompute.
// When on, evaluate_observable_on computes both and std::abort()s on
// mismatch.  Like kCanonicalCacheSelfCheck this is build-type driven: the
// RULEMONKEY_LOCAL_OBS_SELFCHECK compile definition is set by CMakeLists.txt
// for Debug and ASan and left unset for Release, so ctest and the guard
// tier exercise the invariant while Release takes the bare table read.
#ifdef RULEMONKEY_LOCAL_OBS_SELFCHECK
inline constexpr bool kLocalObsTrackInvariant = true;
#else
inline constexpr bool kLocalObsTrackInvariant = false;
#endif

// ===========================================================================
// Profile struct definitions
// ---------------------------------------------------------------------------
// All structs are POD-ish counters consumed by the report_*() functions
// below.  Ordering of fields matches engine.cpp's increment sites; do
// not reorder without updating both places.
//
// All eight profile structs are per-Engine: six live as Engine::Impl /
// AgentPool members directly; the two whose call sites are static free
// functions (CountMultiProfile, CmmFcProfile) are owned by Engine::Impl
// and threaded into count_multi_mol_fast / count_2mol_1bond_fc by
// pointer.  No process-wide profile state — concurrent Engines on
// different threads (BNGsim's ThreadPoolExecutor pattern, the
// PyBNF integration target) get clean per-Engine reports with no
// cross-talk.
// ===========================================================================

struct CountMultiProfile {
  uint64_t calls = 0;                   // dispatcher entries
  uint64_t fm_hits = 0;                 // FastMatchSlot specialization taken
  uint64_t generic_calls = 0;           // fell through to generic body
  uint64_t sampled_calls = 0;           // generic calls bracketed with chrono
  uint64_t singleton_pattern_calls = 0; // pat_end - pat_start <= 1
  uint64_t zero_seed_calls = 0;         // seed_embs.empty() early return
  uint64_t disjoint_calls = 0;          // entered the unassigned (disjoint) path
  uint64_t seed_emb_sum = 0;
  uint64_t seed_emb_max = 0;
  std::array<uint64_t, 7> seed_emb_hist = {0, 0, 0, 0, 0, 0, 0};
  uint64_t bfs_visited_sum = 0;
  uint64_t bfs_visited_max = 0;
  std::array<uint64_t, 7> bfs_visited_hist = {0, 0, 0, 0, 0, 0, 0};
  uint64_t n_pat_mols_sum = 0;
  std::array<uint64_t, 7> n_pat_mols_hist = {0, 0, 0, 0, 0, 0, 0};
  // Sampled (×K at report)
  uint64_t total_ns = 0;
  uint64_t seed_emb_ns = 0;
  uint64_t bfs_ns = 0;
  uint64_t disjoint_ns = 0;
};

struct CmmFcProfile {
  uint64_t fc_calls = 0;
  uint64_t fc_inactive_seed = 0;
  uint64_t fc_seed_type_mismatches = 0;
  uint64_t fc_seed_non_bond_rejects = 0;
  uint64_t fc_candidate_iters = 0;
  // Reject taxonomy (sum + fc_total_matches == fc_candidate_iters)
  uint64_t fc_candidate_oob = 0;
  uint64_t fc_candidate_state_rejects = 0;
  uint64_t fc_candidate_bond_rejects = 0;
  uint64_t fc_candidate_self_bond = 0;
  uint64_t fc_partner_inactive = 0;
  uint64_t fc_partner_type_mismatches = 0;
  uint64_t fc_plocal_not_found = 0;
  uint64_t fc_plocal_not_ok = 0;
  uint64_t fc_partner_state_rejects = 0;
  uint64_t fc_partner_non_bond_rejects = 0;
  uint64_t fc_total_matches = 0;
  // Size sums (/ fc_calls -> per-call mean)
  uint64_t fc_seed_bond_candidates_sum = 0;
  uint64_t fc_partner_bond_candidates_sum = 0;
  uint64_t fc_partner_non_bond_checks_sum = 0;
  uint64_t fc_seed_non_bond_checks_sum = 0;
  uint64_t fc_seed_bond_candidates_max = 0;
  uint64_t fc_partner_bond_candidates_max = 0;
  uint64_t fc_partner_non_bond_checks_max = 0;
  std::array<uint64_t, 7> fc_partner_non_bond_checks_hist = {0, 0, 0, 0, 0, 0, 0};
  // Sampled per-phase (×K×rotor at report; rotor width = 5)
  // Phases: 0=seed_checks, 1=partner_trace, 2=partner_local_scan,
  //         3=plocal_ok_scan, 4=partner_non_bond_checks.
  uint64_t sampled_calls = 0;
  std::array<uint64_t, 5> phase_ns = {0, 0, 0, 0, 0};
  std::array<uint64_t, 5> phase_hits = {0, 0, 0, 0, 0};
};

struct SrProfile {
  // Path identity (classification is exclusive; sum == total calls).
  static constexpr int kPathZero = 0;
  static constexpr int kPathUniSingle = 1;
  static constexpr int kPathUniMultiFm = 2;
  static constexpr int kPathUniMultiGen = 3;
  static constexpr int kPathBimol = 4;
  static constexpr int kNPaths = 5;

  uint64_t calls = 0;
  std::array<uint64_t, kNPaths> path_calls = {0, 0, 0, 0, 0};
  std::array<uint64_t, kNPaths> path_null_no_seed = {0, 0, 0, 0, 0};
  std::array<uint64_t, kNPaths> path_null_post = {0, 0, 0, 0, 0};
  std::array<uint64_t, kNPaths> path_success = {0, 0, 0, 0, 0};

  // Sampler sub-decisions (global, not per-path).
  uint64_t sampler_calls = 0;
  uint64_t sampler_fenwick_uses = 0;
  uint64_t sampler_fenwick_drifts = 0;
  uint64_t sampler_drift_invalid_mid = 0;
  uint64_t sampler_drift_inactive_mol = 0;
  uint64_t sampler_drift_type_mismatch = 0;
  uint64_t sampler_drift_target_eq_sum = 0;
  uint64_t sampler_drift_target_lt_sum = 0;
  double sampler_drift_excess_sum = 0.0;
  double sampler_drift_total_sum = 0.0;
  uint64_t sampler_linear_calls = 0;
  uint64_t sampler_empty_pool = 0;
  uint64_t sampler_local_prop_calls = 0;

  // Per-path work widths.
  uint64_t uni_single_embs_sum = 0;
  uint64_t uni_single_embs_empty = 0;
  uint64_t uni_mm_fm_success = 0;
  uint64_t uni_mm_fm_null = 0;
  uint64_t uni_mm_gen_seed_embs_sum = 0;
  uint64_t uni_mm_gen_seed_embs_empty = 0;
  uint64_t uni_mm_gen_success = 0;
  uint64_t uni_mm_gen_null = 0;
  uint64_t bimol_embs_a_sum = 0;
  uint64_t bimol_embs_b_sum = 0;
  uint64_t bimol_same_mol_rejects = 0;
  uint64_t bimol_same_cx_rejects = 0;
  uint64_t bimol_embs_empty = 0;
  uint64_t bimol_resolve_calls = 0;
  uint64_t bimol_resolve_failures = 0;

  // K-sampled whole-call chrono, bucketed per path.
  uint64_t sampled_calls = 0;
  std::array<uint64_t, kNPaths> path_ns = {0, 0, 0, 0, 0};
  std::array<uint64_t, kNPaths> path_hits = {0, 0, 0, 0, 0};
};

struct RemoveBondProfile {
  uint64_t remove_bond_calls = 0;
  uint64_t split_calls = 0;
  uint64_t singleton_short_circuit = 0;
  uint64_t bfs_calls = 0;
  uint64_t cycle_bond_removals = 0;
  uint64_t tree_bond_splits = 0;
  uint64_t sampled_calls = 0;
  uint64_t total_ns = 0;
  uint64_t bfs_ns = 0;
  uint64_t partition_ns = 0;
  // Histogram buckets: 1 / 2 / 3-4 / 5-8 / 9-16 / 17-32 / 33+
  std::array<uint64_t, 7> cx_size_all = {0, 0, 0, 0, 0, 0, 0};
  std::array<uint64_t, 7> cx_size_bfs = {0, 0, 0, 0, 0, 0, 0};
  std::array<uint64_t, 7> cx_size_split = {0, 0, 0, 0, 0, 0, 0};
  std::array<uint64_t, 7> half_edges_hist = {0, 0, 0, 0, 0, 0, 0};
  uint64_t cx_size_sum_bfs = 0;
  uint64_t cx_size_max_bfs = 0;
  uint64_t half_edges_sum = 0;
  uint64_t half_edges_max = 0;
};

struct FireRuleProfile {
  // OpType order matches model.hpp: 0 AddBond, 1 DeleteBond, 2 StateChange,
  // 3 AddMolecule, 4 DeleteMolecule.
  uint64_t calls = 0;
  uint64_t sampled_calls = 0;
  std::array<uint64_t, 5> op_calls = {0, 0, 0, 0, 0};
  std::array<uint64_t, 5> op_ns = {0, 0, 0, 0, 0};
  uint64_t mark_mol_calls = 0;
  uint64_t mark_comp_calls = 0;
  uint64_t ensure_mask_resizes = 0;
  uint64_t bfs_fires = 0;
  uint64_t bfs_ns = 0;
  uint64_t cleanup_ns = 0;
  uint64_t switch_ns = 0;
  uint64_t total_ns = 0;
  uint64_t cx_exp_fires = 0;
  uint64_t cx_exp_sampled = 0;
  uint64_t cx_exp_ns = 0;
  uint64_t affected_final_sum = 0;
  uint64_t affected_final_max = 0;
};

struct IncrUpdateProfile {
  // Always-on counters.
  uint64_t calls = 0;
  uint64_t sampled_calls = 0;
  uint64_t mids_visited = 0;
  uint64_t rules_visited = 0;
  uint64_t rules_skipped_needed = 0;
  uint64_t have_local_calls = 0;
  uint64_t mols_for_local_sum = 0;
  uint64_t rule_local_rate_cache_clears = 0;
  uint64_t per_mid_entries = 0;
  uint64_t cache_uncacheable = 0;
  uint64_t cache_hits_epoch = 0;
  uint64_t cache_hits_mask = 0;
  uint64_t cache_misses = 0;
  uint64_t count_a_multi_calls = 0;
  uint64_t count_a_single_calls = 0;
  uint64_t count_b_multi_calls = 0;
  uint64_t count_b_single_calls = 0;
  uint64_t shared_comp_calls = 0;
  uint64_t local_rate_path_calls = 0;
  uint64_t fenwick_a_updates = 0;
  uint64_t fenwick_b_updates = 0;
  uint64_t propensity_recomputes = 0;
  std::array<uint64_t, 7> rules_visited_hist = {0, 0, 0, 0, 0, 0, 0};
  // Persistent across calls so the inner sample doesn't always land on
  // the originally-affected mol.
  uint64_t inner_counter = 0;
  // Outer chrono (×K at report).
  uint64_t total_ns = 0;
  uint64_t expand_ns = 0;
  uint64_t dispatch_ns = 0;
  uint64_t rule_loop_ns = 0;
  // Inner chrono (×K×M at report).
  uint64_t per_mid_sampled = 0;
  uint64_t cache_hit_ns = 0;
  uint64_t subtract_ns = 0;
  uint64_t count_a_ns = 0;
  uint64_t count_b_ns = 0;
  uint64_t shared_ns = 0;
  uint64_t local_rate_ns = 0;
  uint64_t add_ns = 0;
  uint64_t fenwick_ns = 0;
  uint64_t store_ns = 0;
  // Per-rule call counters (resize-on-demand; reported as top-N).
  std::vector<uint64_t> per_rule_entries;
  std::vector<uint64_t> per_rule_recomputes;
  std::vector<uint64_t> per_rule_count_a_multi;
};

struct RecordAtProfile {
  uint64_t calls = 0;
  uint64_t compute_obs_ns = 0;
  uint64_t record_ns = 0;
  struct PerObs {
    std::string name;
    std::string type;
    int pat_count = 0;
    bool has_quantity = false;
    std::string quantity_relation;
    int quantity_value = -1;
    uint64_t evaluate_calls = 0;
    uint64_t evaluate_ns = 0;
    uint64_t species_branch_calls = 0;
    uint64_t molecules_branch_calls = 0;
    uint64_t n_cx_visited = 0;
    uint64_t n_counted_cx_hits = 0;
    uint64_t n_embed_calls = 0;
    uint64_t n_cx_members_walked = 0;
  };
  std::vector<PerObs> obs;
  int active_oi = -1;
  // Init-path compute_observables calls (initialize, load_state,
  // add_molecules) leave this false so the sum-per-obs vs compute_obs
  // cross-check stays meaningful.
  bool inside_record_at = false;
};

struct ObsIncrProfile {
  uint64_t update_calls = 0;
  uint64_t update_sampled = 0;
  uint64_t affected_sum = 0;
  uint64_t flush_calls = 0;
  uint64_t flush_sampled = 0;
  uint64_t flush_dirty_cx_sum = 0;
  uint64_t flush_dead_cx_sum = 0;

  uint64_t update_total_ns = 0;
  uint64_t obs_loop_ns = 0;
  uint64_t prev_cx_loop_ns = 0;
  uint64_t flush_total_ns = 0;

  struct PerObs {
    std::string name;
    std::string type;
    int pat_count = 0;
    int seed_type_index = -1;
    std::string pat_signature;
    uint64_t per_mid_calls = 0;
    uint64_t per_mid_ns = 0;
    uint64_t dirty_inserts = 0;
    uint64_t contrib_deltas = 0;
  };
  std::vector<PerObs> obs;
};

struct ExprEvalProfile {
  // Always-on counters — a bare increment, negligible overhead.
  uint64_t evaluate_rate_calls = 0;       // evaluate_rate() entries (all rules)
  uint64_t evaluate_rate_dynamic = 0;     // ... with a dynamic rate law (real work)
  uint64_t evaluate_local_rate_calls = 0; // evaluate_local_rate() entries
  uint64_t update_eval_vars_calls = 0;    // update_eval_vars() entries
  uint64_t update_eval_vars_rebuilds = 0; // ... that missed the generation cache
  uint64_t global_fn_ast_evals = 0;       // global function ASTs settled in rebuilds
  uint64_t tfun_evals = 0;                // TableFunction evals during rebuilds
  uint64_t local_fn_ast_evals = 0;        // local function ASTs settled per local-rate call
  // K-sampled whole-call chrono on the enclosing functions.  evaluate_rate
  // and evaluate_local_rate are disjoint paths (a rule is one or the
  // other); each estimate INCLUDES the nested update_eval_vars cost when a
  // rebuild lands inside it — usually a no-op via the generation cache.
  uint64_t sampled_rate_calls = 0;
  uint64_t evaluate_rate_ns = 0;
  uint64_t sampled_local_rate_calls = 0;
  uint64_t evaluate_local_rate_ns = 0;
  // Sub-spans of evaluate_local_rate, accumulated on the same sampled
  // calls as evaluate_local_rate_ns.  obs = per-molecule observable
  // re-evaluation (evaluate_observable_on); expr = expr::evaluate of the
  // local-function ASTs.  Only the expr slice is ExprTk-addressable; the
  // remainder (local-rate est minus these two) is update_eval_vars plus
  // the eval-var save/restore bookkeeping.
  uint64_t local_obs_eval_ns = 0;
  uint64_t local_expr_eval_ns = 0;
  uint64_t sampled_uev_calls = 0; // bracketed update_eval_vars rebuilds
  uint64_t update_eval_vars_ns = 0;
  // --- issue #10 spike: evaluate_observable_on redundancy/width ---
  // Always-on counters sizing the per-molecule observable re-eval cost.
  // A "key" is (observable, scope-id) where scope-id is the molecule for
  // per-molecule scope or the complex for complex-wide scope; within one
  // event the underlying graph is static, so two evaluate_observable_on
  // calls on the same key MUST yield the same value — that gap is what
  // memoization / incremental tracking would reclaim.
  uint64_t eoo_calls = 0;           // evaluate_observable_on entries
  uint64_t eoo_complex_wide = 0;    // ... in complex-wide scope
  uint64_t eoo_embed_counts = 0;    // count_embeddings_* invocations (work width)
  uint64_t eoo_events = 0;          // incremental_update entries (events)
  uint64_t eoo_unchanged_total = 0; // result == last result for this key (run-wide)
  uint64_t eoo_unchanged_intra = 0; // ... and key already computed this same event
  uint64_t eoo_redundant_embed = 0; // count work attributable to unchanged-total calls
};

// ===========================================================================
// End-of-run reporting bodies
// ---------------------------------------------------------------------------
// Each `report_*()` is a no-op when its gate is `false` (the call sites
// in engine.cpp wrap them in `if constexpr (k*Profile)` so the compiler
// drops the calls entirely in default builds).  The bodies live here so
// engine.cpp doesn't carry several hundred lines of `fprintf` formatting
// that nothing else in the file references.
// ===========================================================================

inline void report_fire_rule(const FireRuleProfile& p, double timing_fire) {
  const int K = kFireRuleProfileSampleEvery;
  static const char* const op_names[5] = {"AddBond", "DeleteBond", "StateChange", "AddMolecule",
                                          "DeleteMolecule"};
  auto sample_to_sec = [&](uint64_t ns, uint64_t sampled) -> double {
    if (sampled == 0)
      return 0.0;
    return static_cast<double>(ns) / 1e9 * static_cast<double>(K);
  };
  double const total_est = sample_to_sec(p.total_ns, p.sampled_calls);
  double const switch_est = sample_to_sec(p.switch_ns, p.sampled_calls);
  double const cleanup_est = sample_to_sec(p.cleanup_ns, p.sampled_calls);
  double const bfs_est = sample_to_sec(p.bfs_ns, p.sampled_calls);
  double const cx_exp_est = (p.cx_exp_sampled == 0)
                                ? 0.0
                                : static_cast<double>(p.cx_exp_ns) / 1e9 * static_cast<double>(K);
  double const denom_fr = total_est > 0 ? total_est : 1.0;
  std::fprintf(stderr, "[fire_rule profile] K=%d  calls=%llu  sampled=%llu\n", K,
               static_cast<unsigned long long>(p.calls),
               static_cast<unsigned long long>(p.sampled_calls));
  std::fprintf(stderr,
               "  total_est=%.3fs  vs timing_fire=%.3fs  (chrono/fire ratio "
               "%.2f)\n",
               total_est, timing_fire, timing_fire > 0 ? total_est / timing_fire : 0.0);
  std::fprintf(stderr,
               "  switch_loop=%.3fs (%.1f%% of total_est)  "
               "cleanup=%.3fs (%.1f%%)  in_fire_bfs=%.3fs (%.1f%%)\n",
               switch_est, 100.0 * switch_est / denom_fr, cleanup_est,
               100.0 * cleanup_est / denom_fr, bfs_est, 100.0 * bfs_est / denom_fr);
  std::fprintf(stderr,
               "  post_fire_cx_exp=%.3fs  (fires=%llu sampled=%llu)  "
               "[billed to timing_fire today]\n",
               cx_exp_est, static_cast<unsigned long long>(p.cx_exp_fires),
               static_cast<unsigned long long>(p.cx_exp_sampled));
  std::fprintf(stderr, "  per-op calls:\n");
  for (int i = 0; i < 5; ++i) {
    double const op_est = sample_to_sec(p.op_ns[i], p.sampled_calls);
    std::fprintf(stderr,
                 "    %-15s  calls=%llu  est_time=%.3fs (%.1f%% of "
                 "total_est)\n",
                 op_names[i], static_cast<unsigned long long>(p.op_calls[i]), op_est,
                 100.0 * op_est / denom_fr);
  }
  std::fprintf(stderr,
               "  mask_bookkeeping: mark_mol=%llu  mark_comp=%llu  "
               "ensure_mask_resizes=%llu\n",
               static_cast<unsigned long long>(p.mark_mol_calls),
               static_cast<unsigned long long>(p.mark_comp_calls),
               static_cast<unsigned long long>(p.ensure_mask_resizes));
  std::fprintf(
      stderr,
      "  in_fire_bfs: fires=%llu  affected_final: sum=%llu max=%llu "
      "avg=%.1f\n",
      static_cast<unsigned long long>(p.bfs_fires),
      static_cast<unsigned long long>(p.affected_final_sum),
      static_cast<unsigned long long>(p.affected_final_max),
      p.calls > 0 ? static_cast<double>(p.affected_final_sum) / static_cast<double>(p.calls) : 0.0);
}

inline void report_remove_bond(const RemoveBondProfile& p) {
  const int K = kRemoveBondProfileSampleEvery;
  auto est = [&](uint64_t ns) -> double {
    if (p.sampled_calls == 0)
      return 0.0;
    return static_cast<double>(ns) / 1e9 * static_cast<double>(K);
  };
  double const total_est = est(p.total_ns);
  double const bfs_est = est(p.bfs_ns);
  double const partition_est = est(p.partition_ns);
  std::fprintf(stderr,
               "[remove_bond profile] K=%d  remove_bond_calls=%llu  "
               "split_calls=%llu  sampled=%llu\n",
               K, static_cast<unsigned long long>(p.remove_bond_calls),
               static_cast<unsigned long long>(p.split_calls),
               static_cast<unsigned long long>(p.sampled_calls));
  std::fprintf(stderr,
               "  total_est=%.3fs  bfs=%.3fs (%.1f%%)  "
               "partition=%.3fs (%.1f%%)\n",
               total_est, bfs_est, total_est > 0 ? 100.0 * bfs_est / total_est : 0.0, partition_est,
               total_est > 0 ? 100.0 * partition_est / total_est : 0.0);
  std::fprintf(stderr,
               "  path counts: singleton=%llu (%.1f%%)  cycle_bond=%llu (%.1f%%)"
               "  tree_split=%llu (%.1f%%)\n",
               static_cast<unsigned long long>(p.singleton_short_circuit),
               p.split_calls > 0 ? 100.0 * static_cast<double>(p.singleton_short_circuit) /
                                       static_cast<double>(p.split_calls)
                                 : 0.0,
               static_cast<unsigned long long>(p.cycle_bond_removals),
               p.split_calls > 0 ? 100.0 * static_cast<double>(p.cycle_bond_removals) /
                                       static_cast<double>(p.split_calls)
                                 : 0.0,
               static_cast<unsigned long long>(p.tree_bond_splits),
               p.split_calls > 0 ? 100.0 * static_cast<double>(p.tree_bond_splits) /
                                       static_cast<double>(p.split_calls)
                                 : 0.0);
  auto print_hist = [&](const char* name, const std::array<uint64_t, 7>& h) {
    uint64_t tot = 0;
    for (auto v : h)
      tot += v;
    std::fprintf(stderr, "  %s: total=%llu  buckets", name, static_cast<unsigned long long>(tot));
    static const char* const labels[7] = {"1", "2", "3-4", "5-8", "9-16", "17-32", "33+"};
    for (int i = 0; i < 7; ++i) {
      std::fprintf(stderr, "  %s=%llu", labels[i], static_cast<unsigned long long>(h[i]));
    }
    std::fprintf(stderr, "\n");
  };
  print_hist("cx_size_all  ", p.cx_size_all);
  print_hist("cx_size_bfs  ", p.cx_size_bfs);
  print_hist("cx_size_split", p.cx_size_split);
  print_hist("half_edges   ", p.half_edges_hist);
  double const cx_avg =
      p.bfs_calls > 0 ? static_cast<double>(p.cx_size_sum_bfs) / static_cast<double>(p.bfs_calls)
                      : 0.0;
  double const he_avg =
      p.bfs_calls > 0 ? static_cast<double>(p.half_edges_sum) / static_cast<double>(p.bfs_calls)
                      : 0.0;
  std::fprintf(stderr,
               "  bfs_cx_size: avg=%.2f max=%llu   "
               "half_edges_per_call: avg=%.2f max=%llu\n",
               cx_avg, static_cast<unsigned long long>(p.cx_size_max_bfs), he_avg,
               static_cast<unsigned long long>(p.half_edges_max));
}

inline void report_incr_update(const IncrUpdateProfile& p, double timing_update) {
  const int K = kIncrUpdateProfileSampleEvery;
  const int M = kIncrUpdateProfileInnerSample;
  auto outer_sec = [&](uint64_t ns) -> double {
    if (p.sampled_calls == 0)
      return 0.0;
    return static_cast<double>(ns) / 1e9 * static_cast<double>(K);
  };
  auto inner_sec = [&](uint64_t ns) -> double {
    if (p.per_mid_sampled == 0)
      return 0.0;
    return static_cast<double>(ns) / 1e9 * static_cast<double>(K) * static_cast<double>(M);
  };
  double const total_est = outer_sec(p.total_ns);
  double const expand_est = outer_sec(p.expand_ns);
  double const dispatch_est = outer_sec(p.dispatch_ns);
  double const rule_loop_est = outer_sec(p.rule_loop_ns);
  double const cache_hit_est = inner_sec(p.cache_hit_ns);
  double const subtract_est = inner_sec(p.subtract_ns);
  double const count_a_est = inner_sec(p.count_a_ns);
  double const count_b_est = inner_sec(p.count_b_ns);
  double const shared_est = inner_sec(p.shared_ns);
  double const local_rate_est = inner_sec(p.local_rate_ns);
  double const add_est = inner_sec(p.add_ns);
  double const fenwick_est = inner_sec(p.fenwick_ns);
  double const store_est = inner_sec(p.store_ns);
  double const denom = total_est > 0 ? total_est : 1.0;
  std::fprintf(stderr,
               "[incr_update profile] K=%d  M=%d  calls=%llu  sampled=%llu"
               "  per_mid_sampled=%llu\n",
               K, M, static_cast<unsigned long long>(p.calls),
               static_cast<unsigned long long>(p.sampled_calls),
               static_cast<unsigned long long>(p.per_mid_sampled));
  std::fprintf(stderr,
               "  total_est=%.3fs  vs timing_update=%.3fs  "
               "(chrono/update ratio %.2f)\n",
               total_est, timing_update, timing_update > 0 ? total_est / timing_update : 0.0);
  std::fprintf(stderr,
               "  outer: expand=%.3fs (%.1f%%)  dispatch=%.3fs (%.1f%%)  "
               "rule_loop=%.3fs (%.1f%%)  (rule_loop includes per-rule "
               "propensity recompute)\n",
               expand_est, 100.0 * expand_est / denom, dispatch_est, 100.0 * dispatch_est / denom,
               rule_loop_est, 100.0 * rule_loop_est / denom);
  double const inner_sum = cache_hit_est + subtract_est + count_a_est + count_b_est + shared_est +
                           local_rate_est + add_est + fenwick_est + store_est;
  std::fprintf(stderr,
               "  inner (est of full rule_loop via ×K×M; sum=%.3fs): "
               "cache_hit=%.3fs (%.1f%%)  subtract=%.3fs (%.1f%%)  "
               "count_a=%.3fs (%.1f%%)  count_b=%.3fs (%.1f%%)\n",
               inner_sum, cache_hit_est, 100.0 * cache_hit_est / denom, subtract_est,
               100.0 * subtract_est / denom, count_a_est, 100.0 * count_a_est / denom, count_b_est,
               100.0 * count_b_est / denom);
  std::fprintf(stderr,
               "                 shared=%.3fs (%.1f%%)  "
               "local_rate=%.3fs (%.1f%%)  add=%.3fs (%.1f%%)  "
               "fenwick=%.3fs (%.1f%%)  store=%.3fs (%.1f%%)\n",
               shared_est, 100.0 * shared_est / denom, local_rate_est,
               100.0 * local_rate_est / denom, add_est, 100.0 * add_est / denom, fenwick_est,
               100.0 * fenwick_est / denom, store_est, 100.0 * store_est / denom);
  std::fprintf(stderr,
               "  counters: mids_visited=%llu  rules_visited=%llu  "
               "rules_skipped_synth=%llu  have_local_calls=%llu  "
               "mols_for_local_sum=%llu  rule_cache_clears=%llu\n",
               static_cast<unsigned long long>(p.mids_visited),
               static_cast<unsigned long long>(p.rules_visited),
               static_cast<unsigned long long>(p.rules_skipped_needed),
               static_cast<unsigned long long>(p.have_local_calls),
               static_cast<unsigned long long>(p.mols_for_local_sum),
               static_cast<unsigned long long>(p.rule_local_rate_cache_clears));
  std::fprintf(stderr,
               "  per_mid: entries=%llu  cache_hits_epoch=%llu  "
               "cache_hits_mask=%llu  cache_misses=%llu  "
               "cache_uncacheable=%llu\n",
               static_cast<unsigned long long>(p.per_mid_entries),
               static_cast<unsigned long long>(p.cache_hits_epoch),
               static_cast<unsigned long long>(p.cache_hits_mask),
               static_cast<unsigned long long>(p.cache_misses),
               static_cast<unsigned long long>(p.cache_uncacheable));
  std::fprintf(stderr,
               "  recompute_calls: count_a_multi=%llu  count_a_single=%llu"
               "  count_b_multi=%llu  count_b_single=%llu  "
               "shared=%llu  local_rate=%llu  fenwick_a=%llu  fenwick_b=%llu"
               "  propensity=%llu\n",
               static_cast<unsigned long long>(p.count_a_multi_calls),
               static_cast<unsigned long long>(p.count_a_single_calls),
               static_cast<unsigned long long>(p.count_b_multi_calls),
               static_cast<unsigned long long>(p.count_b_single_calls),
               static_cast<unsigned long long>(p.shared_comp_calls),
               static_cast<unsigned long long>(p.local_rate_path_calls),
               static_cast<unsigned long long>(p.fenwick_a_updates),
               static_cast<unsigned long long>(p.fenwick_b_updates),
               static_cast<unsigned long long>(p.propensity_recomputes));
  {
    uint64_t tot = 0;
    for (auto v : p.rules_visited_hist)
      tot += v;
    static const char* const labels[7] = {"1", "2", "3-4", "5-8", "9-16", "17-32", "33+"};
    std::fprintf(stderr, "  rules_visited_hist: total=%llu  buckets",
                 static_cast<unsigned long long>(tot));
    for (int i = 0; i < 7; ++i) {
      std::fprintf(stderr, "  %s=%llu", labels[i],
                   static_cast<unsigned long long>(p.rules_visited_hist[i]));
    }
    std::fprintf(stderr, "\n");
  }
  // Per-rule top-N attribution (call-count based, no chrono).
  auto dump_topn = [&](const char* label, const std::vector<uint64_t>& v, int topn) {
    uint64_t tot = 0;
    for (auto x : v)
      tot += x;
    std::vector<std::pair<uint64_t, int>> idx;
    idx.reserve(v.size());
    for (size_t i = 0; i < v.size(); ++i)
      if (v[i] > 0)
        idx.emplace_back(v[i], static_cast<int>(i));
    std::sort(idx.begin(), idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::fprintf(stderr, "  per_rule_%s (top %d of %zu, total=%llu):\n", label, topn, idx.size(),
                 static_cast<unsigned long long>(tot));
    uint64_t cum = 0;
    int const shown = std::min<int>(topn, static_cast<int>(idx.size()));
    for (int i = 0; i < shown; ++i) {
      cum += idx[i].first;
      double const pct = tot > 0 ? 100.0 * idx[i].first / tot : 0.0;
      double const cum_pct = tot > 0 ? 100.0 * cum / tot : 0.0;
      std::fprintf(stderr, "    RR%d: %llu (%.1f%%)  cum=%.1f%%\n", idx[i].second + 1,
                   static_cast<unsigned long long>(idx[i].first), pct, cum_pct);
    }
  };
  dump_topn("recomputes", p.per_rule_recomputes, 15);
  dump_topn("count_a_multi", p.per_rule_count_a_multi, 10);
  dump_topn("entries", p.per_rule_entries, 10);
}

inline void report_record_at(const RecordAtProfile& p, double timing_record,
                             bool use_incremental_obs) {
  double const compute_obs_est = static_cast<double>(p.compute_obs_ns) / 1e9;
  double const record_est = static_cast<double>(p.record_ns) / 1e9;
  double const sum_est = compute_obs_est + record_est;
  double const denom = timing_record > 0 ? timing_record : 1.0;
  std::fprintf(stderr,
               "[record_at profile] K=1  calls=%llu  "
               "use_incremental_obs=%d\n",
               static_cast<unsigned long long>(p.calls), static_cast<int>(use_incremental_obs));
  std::fprintf(stderr,
               "  compute_obs=%.3fs (%.1f%% of timing_record)  "
               "record_time_point=%.3fs (%.1f%%)  "
               "chrono_sum/timing_record=%.3f\n",
               compute_obs_est, 100.0 * compute_obs_est / denom, record_est,
               100.0 * record_est / denom, timing_record > 0 ? sum_est / timing_record : 0.0);

  double const co_denom = compute_obs_est > 0 ? compute_obs_est : 1.0;
  std::fprintf(stderr, "  per-observable breakdown:\n");
  for (size_t i = 0; i < p.obs.size(); ++i) {
    const auto& po = p.obs[i];
    double const ev_s = static_cast<double>(po.evaluate_ns) / 1e9;
    std::fprintf(stderr,
                 "    [%zu] %s (type=%s pats=%d)  calls=%llu  wall=%.4fs"
                 " (%.1f%% of compute_obs)\n",
                 i, po.name.c_str(), po.type.c_str(), po.pat_count,
                 static_cast<unsigned long long>(po.evaluate_calls), ev_s, 100.0 * ev_s / co_denom);
    std::fprintf(stderr, "        species_branch=%llu  molecules_branch=%llu\n",
                 static_cast<unsigned long long>(po.species_branch_calls),
                 static_cast<unsigned long long>(po.molecules_branch_calls));
    std::fprintf(stderr,
                 "        cx_visited=%llu  counted_cx_hits=%llu  "
                 "embed_calls=%llu  cx_members_walked=%llu\n",
                 static_cast<unsigned long long>(po.n_cx_visited),
                 static_cast<unsigned long long>(po.n_counted_cx_hits),
                 static_cast<unsigned long long>(po.n_embed_calls),
                 static_cast<unsigned long long>(po.n_cx_members_walked));
    if (po.has_quantity) {
      std::fprintf(stderr, "        quantity_relation=\"%s\"  quantity_value=%d\n",
                   po.quantity_relation.c_str(), po.quantity_value);
    }
  }

  // Cross-check: sum of per-obs evaluate_ns ≈ compute_obs_ns.
  uint64_t sum_obs_ns = 0;
  for (const auto& po : p.obs)
    sum_obs_ns += po.evaluate_ns;
  double const sum_obs_s = static_cast<double>(sum_obs_ns) / 1e9;
  std::fprintf(stderr,
               "  cross-check: Sum_per_obs_wall=%.4fs  compute_obs=%.4fs"
               "  ratio=%.3f\n",
               sum_obs_s, compute_obs_est, compute_obs_est > 0 ? sum_obs_s / compute_obs_est : 0.0);
}

inline void report_obs_incr(const ObsIncrProfile& p, double timing_obs,
                            const std::vector<int>& incr_tracked_obs_indices) {
  const int K = kObsIncrProfileSampleEvery;
  double const update_est = static_cast<double>(p.update_total_ns) * K / 1e9;
  double const obs_loop_est = static_cast<double>(p.obs_loop_ns) * K / 1e9;
  double const prev_cx_loop_est = static_cast<double>(p.prev_cx_loop_ns) * K / 1e9;
  double const flush_est = static_cast<double>(p.flush_total_ns) / 1e9; // flush is full-sampled
  double const denom_obs = timing_obs > 0 ? timing_obs : 1.0;
  std::fprintf(stderr,
               "[obs_incr profile] K=%d  update_calls=%llu  sampled=%llu  "
               "flush_calls=%llu\n",
               K, static_cast<unsigned long long>(p.update_calls),
               static_cast<unsigned long long>(p.update_sampled),
               static_cast<unsigned long long>(p.flush_calls));
  std::fprintf(stderr,
               "  avg_affected=%.2f  flush_dirty_cx_avg=%.2f  "
               "flush_dead_cx_avg=%.2f\n",
               p.update_calls > 0
                   ? static_cast<double>(p.affected_sum) / static_cast<double>(p.update_calls)
                   : 0.0,
               p.flush_calls > 0
                   ? static_cast<double>(p.flush_dirty_cx_sum) / static_cast<double>(p.flush_calls)
                   : 0.0,
               p.flush_calls > 0
                   ? static_cast<double>(p.flush_dead_cx_sum) / static_cast<double>(p.flush_calls)
                   : 0.0);
  std::fprintf(stderr,
               "  incremental_update=%.3fs (%.1f%% of timing_obs)  "
               "flush=%.3fs (%.1f%%)\n",
               update_est, 100.0 * update_est / denom_obs, flush_est,
               100.0 * flush_est / denom_obs);
  std::fprintf(stderr, "  update sub-phases:  obs_loop=%.3fs  prev_cx_loop=%.3fs\n", obs_loop_est,
               prev_cx_loop_est);

  std::fprintf(stderr, "  per-tracked-obs breakdown:\n");
  std::vector<int> order;
  order.reserve(incr_tracked_obs_indices.size());
  for (int const oi : incr_tracked_obs_indices)
    order.push_back(oi);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return p.obs[a].per_mid_ns > p.obs[b].per_mid_ns; });
  for (int const oi : order) {
    const auto& po = p.obs[oi];
    double const ev_s = static_cast<double>(po.per_mid_ns) * K / 1e9;
    std::fprintf(stderr,
                 "    [%d] %s (type=%s pats=%d seed_t=%d)  "
                 "per_mid_calls=%llu  wall=%.4fs  deltas=%llu  "
                 "dirty_inserts=%llu\n",
                 oi, po.name.c_str(), po.type.c_str(), po.pat_count, po.seed_type_index,
                 static_cast<unsigned long long>(po.per_mid_calls), ev_s,
                 static_cast<unsigned long long>(po.contrib_deltas),
                 static_cast<unsigned long long>(po.dirty_inserts));
  }

  // Dedup audit: group tracked obs by pattern signature.
  std::unordered_map<std::string, std::vector<int>> by_sig;
  for (int const oi : incr_tracked_obs_indices) {
    by_sig[p.obs[oi].pat_signature].push_back(oi);
  }
  int singleton_groups = 0, multi_groups = 0, obs_in_multi = 0;
  for (auto& kv : by_sig) {
    if (kv.second.size() == 1)
      singleton_groups++;
    else {
      multi_groups++;
      obs_in_multi += kv.second.size();
    }
  }
  std::fprintf(stderr,
               "  dedup audit: total_tracked=%zu  unique_signatures=%zu  "
               "singleton_groups=%d  multi_groups=%d  obs_in_multi=%d\n",
               incr_tracked_obs_indices.size(), by_sig.size(), singleton_groups, multi_groups,
               obs_in_multi);
  if (multi_groups > 0) {
    std::fprintf(stderr, "  multi-obs groups (top 5 by size):\n");
    std::vector<std::pair<std::string, std::vector<int>>> groups(by_sig.begin(), by_sig.end());
    std::sort(groups.begin(), groups.end(),
              [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });
    int printed = 0;
    for (auto& kv : groups) {
      if (kv.second.size() <= 1)
        continue;
      if (printed++ >= 5)
        break;
      std::fprintf(stderr, "    size=%zu  obs=", kv.second.size());
      for (size_t i = 0; i < std::min<size_t>(4, kv.second.size()); ++i) {
        std::fprintf(stderr, "%s%s", i ? "," : "", p.obs[kv.second[i]].name.c_str());
      }
      if (kv.second.size() > 4)
        std::fprintf(stderr, ",...");
      std::fprintf(stderr, "\n      sig=%s\n", kv.first.c_str());
    }
  }
}

inline void report_count_multi(const CountMultiProfile& p) {
  const int K = kCountMultiProfileSampleEvery;
  auto ksec = [&](uint64_t ns) -> double {
    if (p.sampled_calls == 0)
      return 0.0;
    return static_cast<double>(ns) / 1e9 * static_cast<double>(K);
  };
  double const total_est = ksec(p.total_ns);
  double const seed_emb_est = ksec(p.seed_emb_ns);
  double const bfs_est = ksec(p.bfs_ns);
  double const disjoint_est = ksec(p.disjoint_ns);
  double const denom = total_est > 0 ? total_est : 1.0;
  std::fprintf(
      stderr,
      "[count_multi profile] K=%d  calls=%llu  fm_hits=%llu (%.1f%%)"
      "  generic=%llu (%.1f%%)  sampled=%llu\n",
      K, static_cast<unsigned long long>(p.calls), static_cast<unsigned long long>(p.fm_hits),
      p.calls > 0 ? 100.0 * static_cast<double>(p.fm_hits) / static_cast<double>(p.calls) : 0.0,
      static_cast<unsigned long long>(p.generic_calls),
      p.calls > 0 ? 100.0 * static_cast<double>(p.generic_calls) / static_cast<double>(p.calls)
                  : 0.0,
      static_cast<unsigned long long>(p.sampled_calls));
  std::fprintf(stderr,
               "  generic paths: singleton_pattern=%llu (%.1f%%)"
               "  zero_seed=%llu (%.1f%%)  disjoint=%llu (%.1f%%)\n",
               static_cast<unsigned long long>(p.singleton_pattern_calls),
               p.generic_calls > 0 ? 100.0 * static_cast<double>(p.singleton_pattern_calls) /
                                         static_cast<double>(p.generic_calls)
                                   : 0.0,
               static_cast<unsigned long long>(p.zero_seed_calls),
               p.generic_calls > 0 ? 100.0 * static_cast<double>(p.zero_seed_calls) /
                                         static_cast<double>(p.generic_calls)
                                   : 0.0,
               static_cast<unsigned long long>(p.disjoint_calls),
               p.generic_calls > 0 ? 100.0 * static_cast<double>(p.disjoint_calls) /
                                         static_cast<double>(p.generic_calls)
                                   : 0.0);
  std::fprintf(stderr,
               "  total_est=%.3fs  seed_emb=%.3fs (%.1f%%)  bfs=%.3fs (%.1f%%)"
               "  disjoint=%.3fs (%.1f%%)\n",
               total_est, seed_emb_est, 100.0 * seed_emb_est / denom, bfs_est,
               100.0 * bfs_est / denom, disjoint_est, 100.0 * disjoint_est / denom);
  double const full_sum = seed_emb_est + bfs_est + disjoint_est;
  std::fprintf(stderr,
               "  sub-phase sum=%.3fs  residual vs total_est=%.3fs "
               "(should be ≈ singleton/zero-seed early returns)\n",
               full_sum, total_est - full_sum);
  double const npm_avg = p.generic_calls > p.singleton_pattern_calls
                             ? static_cast<double>(p.n_pat_mols_sum) /
                                   static_cast<double>(p.generic_calls - p.singleton_pattern_calls)
                             : 0.0;
  uint64_t const gen_non_singleton =
      p.generic_calls > p.singleton_pattern_calls ? p.generic_calls - p.singleton_pattern_calls : 0;
  double const seed_avg = gen_non_singleton > 0 ? static_cast<double>(p.seed_emb_sum) /
                                                      static_cast<double>(gen_non_singleton)
                                                : 0.0;
  double const bfs_avg = gen_non_singleton > 0 ? static_cast<double>(p.bfs_visited_sum) /
                                                     static_cast<double>(gen_non_singleton)
                                               : 0.0;
  std::fprintf(stderr,
               "  per-call avg (non-singleton generic): n_pat_mols=%.2f (max=*)"
               "  seed_embs=%.3f (sum=%llu max=%llu)"
               "  bfs_visited=%.3f (sum=%llu max=%llu)\n",
               npm_avg, seed_avg, static_cast<unsigned long long>(p.seed_emb_sum),
               static_cast<unsigned long long>(p.seed_emb_max), bfs_avg,
               static_cast<unsigned long long>(p.bfs_visited_sum),
               static_cast<unsigned long long>(p.bfs_visited_max));
  auto print_hist = [&](const char* name, const std::array<uint64_t, 7>& h) {
    uint64_t tot = 0;
    for (auto v : h)
      tot += v;
    std::fprintf(stderr, "  %s hist: total=%llu  buckets", name,
                 static_cast<unsigned long long>(tot));
    static const char* const labels[7] = {"0-1", "2", "3-4", "5-8", "9-16", "17-32", "33+"};
    for (int i = 0; i < 7; ++i) {
      std::fprintf(stderr, "  %s=%llu", labels[i], static_cast<unsigned long long>(h[i]));
    }
    std::fprintf(stderr, "\n");
  };
  print_hist("n_pat_mols ", p.n_pat_mols_hist);
  print_hist("seed_embs  ", p.seed_emb_hist);
  print_hist("bfs_visited", p.bfs_visited_hist);
}

inline void report_cmm_fc(const CmmFcProfile& q, const CountMultiProfile& cm) {
  const int K = kCmmFcProfileSampleEvery;
  const int R = 5; // phase rotor width
  auto phase_sec = [&](int p) -> double {
    return static_cast<double>(q.phase_ns[p]) / 1e9 * static_cast<double>(K) *
           static_cast<double>(R);
  };
  double const p_seed = phase_sec(0);
  double const p_ptrc = phase_sec(1);
  double const p_plsc = phase_sec(2);
  double const p_pok = phase_sec(3);
  double const p_pnbc = phase_sec(4);
  double const p_total_est = p_seed + p_ptrc + p_plsc + p_pok + p_pnbc;
  double const denom = p_total_est > 0 ? p_total_est : 1.0;
  std::fprintf(stderr,
               "[cmm_fc profile] K=%d  rotor=%d  fc_calls=%llu  sampled=%llu"
               "  inactive_seed=%llu  seed_type_mismatches=%llu"
               "  seed_non_bond_rejects=%llu\n",
               K, R, static_cast<unsigned long long>(q.fc_calls),
               static_cast<unsigned long long>(q.sampled_calls),
               static_cast<unsigned long long>(q.fc_inactive_seed),
               static_cast<unsigned long long>(q.fc_seed_type_mismatches),
               static_cast<unsigned long long>(q.fc_seed_non_bond_rejects));
  std::fprintf(stderr,
               "  phase wall (×K×rotor): seed=%.4fs (%.1f%%)"
               "  partner_trace=%.4fs (%.1f%%)"
               "  partner_local_scan=%.4fs (%.1f%%)"
               "  plocal_ok=%.4fs (%.1f%%)"
               "  partner_non_bond=%.4fs (%.1f%%)"
               "  sum=%.4fs\n",
               p_seed, 100.0 * p_seed / denom, p_ptrc, 100.0 * p_ptrc / denom, p_plsc,
               100.0 * p_plsc / denom, p_pok, 100.0 * p_pok / denom, p_pnbc, 100.0 * p_pnbc / denom,
               p_total_est);
  std::fprintf(stderr,
               "  phase_hits: seed=%llu  partner_trace=%llu"
               "  partner_local_scan=%llu  plocal_ok=%llu  partner_non_bond=%llu\n",
               static_cast<unsigned long long>(q.phase_hits[0]),
               static_cast<unsigned long long>(q.phase_hits[1]),
               static_cast<unsigned long long>(q.phase_hits[2]),
               static_cast<unsigned long long>(q.phase_hits[3]),
               static_cast<unsigned long long>(q.phase_hits[4]));
  std::fprintf(stderr,
               "  candidate_iters=%llu  total_matches=%llu\n"
               "    oob=%llu  state_rej=%llu  bond_rej=%llu  self_bond=%llu"
               "  partner_inactive=%llu  partner_type_mm=%llu\n"
               "    plocal_not_found=%llu  plocal_not_ok=%llu  partner_state_rej=%llu"
               "  partner_non_bond_rej=%llu\n",
               static_cast<unsigned long long>(q.fc_candidate_iters),
               static_cast<unsigned long long>(q.fc_total_matches),
               static_cast<unsigned long long>(q.fc_candidate_oob),
               static_cast<unsigned long long>(q.fc_candidate_state_rejects),
               static_cast<unsigned long long>(q.fc_candidate_bond_rejects),
               static_cast<unsigned long long>(q.fc_candidate_self_bond),
               static_cast<unsigned long long>(q.fc_partner_inactive),
               static_cast<unsigned long long>(q.fc_partner_type_mismatches),
               static_cast<unsigned long long>(q.fc_plocal_not_found),
               static_cast<unsigned long long>(q.fc_plocal_not_ok),
               static_cast<unsigned long long>(q.fc_partner_state_rejects),
               static_cast<unsigned long long>(q.fc_partner_non_bond_rejects));
  uint64_t const rej_sum = q.fc_candidate_oob + q.fc_candidate_state_rejects +
                           q.fc_candidate_bond_rejects + q.fc_candidate_self_bond +
                           q.fc_partner_inactive + q.fc_partner_type_mismatches +
                           q.fc_plocal_not_found + q.fc_plocal_not_ok + q.fc_partner_state_rejects +
                           q.fc_partner_non_bond_rejects;
  std::fprintf(stderr,
               "  cross-check: rej_sum+matches=%llu  iters=%llu  (must be equal)"
               "  fm_hits=%llu (from count_multi)\n",
               static_cast<unsigned long long>(rej_sum + q.fc_total_matches),
               static_cast<unsigned long long>(q.fc_candidate_iters),
               static_cast<unsigned long long>(cm.fm_hits));
  double const fc = q.fc_calls > 0 ? static_cast<double>(q.fc_calls) : 1.0;
  std::fprintf(stderr,
               "  per-call means: seed_bond_cand=%.2f (max=%llu)"
               "  partner_bond_cand=%.2f (max=%llu)"
               "  partner_non_bond_checks=%.2f (max=%llu)"
               "  seed_non_bond_checks=%.2f\n",
               static_cast<double>(q.fc_seed_bond_candidates_sum) / fc,
               static_cast<unsigned long long>(q.fc_seed_bond_candidates_max),
               static_cast<double>(q.fc_partner_bond_candidates_sum) / fc,
               static_cast<unsigned long long>(q.fc_partner_bond_candidates_max),
               static_cast<double>(q.fc_partner_non_bond_checks_sum) / fc,
               static_cast<unsigned long long>(q.fc_partner_non_bond_checks_max),
               static_cast<double>(q.fc_seed_non_bond_checks_sum) / fc);
  {
    const auto& h = q.fc_partner_non_bond_checks_hist;
    uint64_t tot = 0;
    for (auto v : h)
      tot += v;
    std::fprintf(stderr,
                 "  partner_non_bond_checks hist (per fc call): total=%llu"
                 "  0=%llu  1=%llu  2=%llu  3=%llu  4=%llu  5=%llu  6+=%llu\n",
                 static_cast<unsigned long long>(tot), static_cast<unsigned long long>(h[0]),
                 static_cast<unsigned long long>(h[1]), static_cast<unsigned long long>(h[2]),
                 static_cast<unsigned long long>(h[3]), static_cast<unsigned long long>(h[4]),
                 static_cast<unsigned long long>(h[5]), static_cast<unsigned long long>(h[6]));
  }
}

inline void report_select_reactants(const SrProfile& q, double timing_sample) {
  const int K = kSelectReactantsProfileSampleEvery;
  static const char* const path_names[SrProfile::kNPaths] = {"zero", "uni_single", "uni_multi_fm",
                                                             "uni_multi_gen", "bimol"};
  auto path_sec = [&](int p) -> double {
    return static_cast<double>(q.path_ns[p]) / 1e9 * static_cast<double>(K);
  };
  double total_est = 0.0;
  for (int i = 0; i < SrProfile::kNPaths; ++i)
    total_est += path_sec(i);
  double const denom = total_est > 0 ? total_est : 1.0;
  double const ts_denom = timing_sample > 0 ? timing_sample : 1.0;

  std::fprintf(stderr,
               "[sr profile] K=%d  calls=%llu  sampled=%llu"
               "  total_est=%.3fs  vs timing_sample=%.3fs  (chrono/sel ratio "
               "%.2f)\n",
               K, static_cast<unsigned long long>(q.calls),
               static_cast<unsigned long long>(q.sampled_calls), total_est, timing_sample,
               timing_sample > 0 ? total_est / timing_sample : 0.0);
  std::fprintf(stderr, "  per-path (calls  wall  %%sel  %%est  ns/call"
                       "  success  null_no_seed  null_post):\n");
  for (int i = 0; i < SrProfile::kNPaths; ++i) {
    double const sec = path_sec(i);
    double const ns_per_call = q.path_calls[i] > 0 ? static_cast<double>(q.path_ns[i]) * K /
                                                         static_cast<double>(q.path_calls[i])
                                                   : 0.0;
    std::fprintf(stderr,
                 "    %-14s  %10llu  %7.3fs  %5.1f%%  %5.1f%%  %7.1fns"
                 "  %10llu  %10llu  %10llu\n",
                 path_names[i], static_cast<unsigned long long>(q.path_calls[i]), sec,
                 100.0 * sec / ts_denom, 100.0 * sec / denom, ns_per_call,
                 static_cast<unsigned long long>(q.path_success[i]),
                 static_cast<unsigned long long>(q.path_null_no_seed[i]),
                 static_cast<unsigned long long>(q.path_null_post[i]));
  }

  std::fprintf(stderr,
               "  sampler: calls=%llu"
               "  fenwick_uses=%llu (%.1f%%)  fenwick_drifts=%llu (%.1f%%)"
               "  linear=%llu (%.1f%%)  empty=%llu  local_prop=%llu\n",
               static_cast<unsigned long long>(q.sampler_calls),
               static_cast<unsigned long long>(q.sampler_fenwick_uses),
               q.sampler_calls > 0 ? 100.0 * static_cast<double>(q.sampler_fenwick_uses) /
                                         static_cast<double>(q.sampler_calls)
                                   : 0.0,
               static_cast<unsigned long long>(q.sampler_fenwick_drifts),
               q.sampler_calls > 0 ? 100.0 * static_cast<double>(q.sampler_fenwick_drifts) /
                                         static_cast<double>(q.sampler_calls)
                                   : 0.0,
               static_cast<unsigned long long>(q.sampler_linear_calls),
               q.sampler_calls > 0 ? 100.0 * static_cast<double>(q.sampler_linear_calls) /
                                         static_cast<double>(q.sampler_calls)
                                   : 0.0,
               static_cast<unsigned long long>(q.sampler_empty_pool),
               static_cast<unsigned long long>(q.sampler_local_prop_calls));
  {
    uint64_t const d = q.sampler_fenwick_drifts;
    double const dd = d > 0 ? static_cast<double>(d) : 1.0;
    std::fprintf(stderr,
                 "  drift breakdown: invalid_mid=%llu (%.1f%%)"
                 "  inactive_mol=%llu (%.1f%%)"
                 "  type_mismatch=%llu (%.1f%%)\n",
                 static_cast<unsigned long long>(q.sampler_drift_invalid_mid),
                 100.0 * static_cast<double>(q.sampler_drift_invalid_mid) / dd,
                 static_cast<unsigned long long>(q.sampler_drift_inactive_mol),
                 100.0 * static_cast<double>(q.sampler_drift_inactive_mol) / dd,
                 static_cast<unsigned long long>(q.sampler_drift_type_mismatch),
                 100.0 * static_cast<double>(q.sampler_drift_type_mismatch) / dd);
    {
      double const imd =
          q.sampler_drift_invalid_mid > 0 ? static_cast<double>(q.sampler_drift_invalid_mid) : 1.0;
      double const avg_excess_per_event =
          q.sampler_drift_target_eq_sum > 0
              ? q.sampler_drift_excess_sum / static_cast<double>(q.sampler_drift_target_eq_sum)
              : 0.0;
      double const avg_total_per_event =
          q.sampler_drift_target_eq_sum > 0
              ? q.sampler_drift_total_sum / static_cast<double>(q.sampler_drift_target_eq_sum)
              : 0.0;
      double const frac_loss =
          avg_total_per_event > 0 ? avg_excess_per_event / avg_total_per_event : 0.0;
      std::fprintf(stderr,
                   "  invalid_mid breakdown:"
                   " target_ge_sum=%llu (%.1f%%)  target_lt_sum=%llu (%.1f%%)\n"
                   "  weight loss when target>=sum: avg_excess=%.6g"
                   "  avg_total=%.6g  avg_loss_frac=%.4f%%\n",
                   static_cast<unsigned long long>(q.sampler_drift_target_eq_sum),
                   100.0 * static_cast<double>(q.sampler_drift_target_eq_sum) / imd,
                   static_cast<unsigned long long>(q.sampler_drift_target_lt_sum),
                   100.0 * static_cast<double>(q.sampler_drift_target_lt_sum) / imd,
                   avg_excess_per_event, avg_total_per_event, 100.0 * frac_loss);
    }
  }

  // Per-path work widths.
  double const uni_single_avg_embs =
      q.path_calls[SrProfile::kPathUniSingle] > 0
          ? static_cast<double>(q.uni_single_embs_sum) /
                static_cast<double>(q.path_calls[SrProfile::kPathUniSingle])
          : 0.0;
  double const bimol_a_avg = q.path_calls[SrProfile::kPathBimol] > 0
                                 ? static_cast<double>(q.bimol_embs_a_sum) /
                                       static_cast<double>(q.path_calls[SrProfile::kPathBimol])
                                 : 0.0;
  double const bimol_b_avg = q.path_calls[SrProfile::kPathBimol] > 0
                                 ? static_cast<double>(q.bimol_embs_b_sum) /
                                       static_cast<double>(q.path_calls[SrProfile::kPathBimol])
                                 : 0.0;
  std::fprintf(stderr,
               "  uni_single: avg_embs=%.3f  embs_empty=%llu\n"
               "  uni_mm_fm:  success=%llu  null=%llu\n"
               "  uni_mm_gen: success=%llu  null=%llu\n"
               "  bimol: avg_embs_a=%.3f  avg_embs_b=%.3f"
               "  same_mol=%llu  same_cx=%llu  embs_empty=%llu"
               "  resolve_calls=%llu  resolve_failures=%llu\n",
               uni_single_avg_embs, static_cast<unsigned long long>(q.uni_single_embs_empty),
               static_cast<unsigned long long>(q.uni_mm_fm_success),
               static_cast<unsigned long long>(q.uni_mm_fm_null),
               static_cast<unsigned long long>(q.uni_mm_gen_success),
               static_cast<unsigned long long>(q.uni_mm_gen_null), bimol_a_avg, bimol_b_avg,
               static_cast<unsigned long long>(q.bimol_same_mol_rejects),
               static_cast<unsigned long long>(q.bimol_same_cx_rejects),
               static_cast<unsigned long long>(q.bimol_embs_empty),
               static_cast<unsigned long long>(q.bimol_resolve_calls),
               static_cast<unsigned long long>(q.bimol_resolve_failures));
}

inline void report_expr_eval(const ExprEvalProfile& p, double timing_wall) {
  const int K = kExprEvalProfileSampleEvery;
  auto est = [&](uint64_t ns, uint64_t sampled) -> double {
    if (sampled == 0)
      return 0.0;
    return static_cast<double>(ns) / 1e9 * static_cast<double>(K);
  };
  auto ull = [](uint64_t v) { return static_cast<unsigned long long>(v); };
  double const rate_est = est(p.evaluate_rate_ns, p.sampled_rate_calls);
  double const local_est = est(p.evaluate_local_rate_ns, p.sampled_local_rate_calls);
  double const local_obs_est = est(p.local_obs_eval_ns, p.sampled_local_rate_calls);
  double const local_expr_est = est(p.local_expr_eval_ns, p.sampled_local_rate_calls);
  double const uev_est = est(p.update_eval_vars_ns, p.sampled_uev_calls);
  // Headline: evaluate_rate and evaluate_local_rate are disjoint paths,
  // so their estimates add.  update_eval_vars cost is already folded into
  // both — reported separately only as a sub-component, not added here.
  double const expr_total = rate_est + local_est;
  double const denom = timing_wall > 0 ? timing_wall : 1.0;
  std::fprintf(stderr, "[expr_eval profile] K=%d  wall=%.3fs\n", K, timing_wall);
  std::fprintf(stderr,
               "  evaluate_rate:        calls=%llu  dynamic=%llu  sampled=%llu  "
               "est=%.4fs (%.2f%% of wall)\n",
               ull(p.evaluate_rate_calls), ull(p.evaluate_rate_dynamic), ull(p.sampled_rate_calls),
               rate_est, 100.0 * rate_est / denom);
  std::fprintf(stderr,
               "  evaluate_local_rate:  calls=%llu  sampled=%llu  "
               "est=%.4fs (%.2f%% of wall)\n",
               ull(p.evaluate_local_rate_calls), ull(p.sampled_local_rate_calls), local_est,
               100.0 * local_est / denom);
  if (p.sampled_local_rate_calls > 0) {
    double const ldenom = local_est > 0 ? local_est : 1.0;
    double const local_rest = local_est - local_obs_est - local_expr_est;
    std::fprintf(stderr,
                 "      of which: observable re-eval=%.4fs (%.1f%% of local)  "
                 "expr::evaluate=%.4fs (%.1f%% of local, ExprTk-addressable)  "
                 "rest=%.4fs (%.1f%%)\n",
                 local_obs_est, 100.0 * local_obs_est / ldenom, local_expr_est,
                 100.0 * local_expr_est / ldenom, local_rest, 100.0 * local_rest / ldenom);
  }
  std::fprintf(stderr,
               "  update_eval_vars:     calls=%llu  rebuilds=%llu  sampled=%llu  "
               "est=%.4fs (%.2f%% of wall)  [sub-component of the two above]\n",
               ull(p.update_eval_vars_calls), ull(p.update_eval_vars_rebuilds),
               ull(p.sampled_uev_calls), uev_est, 100.0 * uev_est / denom);
  std::fprintf(stderr, "  >> expression-eval total est = %.4fs (%.2f%% of wall)\n", expr_total,
               100.0 * expr_total / denom);
  std::fprintf(stderr, "  function settles: global_ast=%llu  tfun=%llu  local_ast=%llu\n",
               ull(p.global_fn_ast_evals), ull(p.tfun_evals), ull(p.local_fn_ast_evals));
  if (p.sampled_rate_calls > 0)
    std::fprintf(stderr, "  mean ns/sampled evaluate_rate call:       %.1f\n",
                 static_cast<double>(p.evaluate_rate_ns) /
                     static_cast<double>(p.sampled_rate_calls));
  if (p.sampled_local_rate_calls > 0)
    std::fprintf(stderr, "  mean ns/sampled evaluate_local_rate call: %.1f\n",
                 static_cast<double>(p.evaluate_local_rate_ns) /
                     static_cast<double>(p.sampled_local_rate_calls));
  // issue #10 spike: evaluate_observable_on redundancy.
  if (p.eoo_calls > 0) {
    auto const calls = static_cast<double>(p.eoo_calls);
    double const embed = p.eoo_embed_counts > 0 ? static_cast<double>(p.eoo_embed_counts) : 1.0;
    std::fprintf(stderr,
                 "  [#10] evaluate_observable_on: calls=%llu  complex_wide=%llu (%.1f%%)  "
                 "embed_counts=%llu  events=%llu  calls/event=%.1f\n",
                 ull(p.eoo_calls), ull(p.eoo_complex_wide), 100.0 * p.eoo_complex_wide / calls,
                 ull(p.eoo_embed_counts), ull(p.eoo_events),
                 p.eoo_events > 0 ? calls / static_cast<double>(p.eoo_events) : 0.0);
    std::fprintf(
        stderr,
        "  [#10] result-unchanged-vs-last: %llu calls (%.1f%%)  "
        "of which intra-event %llu (%.1f%%)  |  redundant embed work %llu (%.1f%% of embed)\n",
        ull(p.eoo_unchanged_total), 100.0 * p.eoo_unchanged_total / calls,
        ull(p.eoo_unchanged_intra), 100.0 * p.eoo_unchanged_intra / calls,
        ull(p.eoo_redundant_embed), 100.0 * p.eoo_redundant_embed / embed);
  }
}

} // namespace rulemonkey
