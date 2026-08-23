#include "engine.hpp"

#include "canonical.hpp"
#include "engine_profile.hpp"
#include "model.hpp"
#include "table_function.hpp"

#include "bngsim/expression.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// Dev-time profiler infrastructure (gates, profile struct definitions,
// end-of-run report bodies) lives in engine_profile.hpp; the eight
// profile struct *instances* are per-Engine members on Engine::Impl /
// AgentPool below.  Every `if constexpr (k*Profile)` site dead-strips
// in default builds (master gate off); turn it on with cmake
// -DRULEMONKEY_ENABLE_DEV_PROFILES=ON.

namespace rulemonkey {

// All TU-local helpers, structs, classes, and constants live in this
// anonymous namespace.  Anything outside it (Engine::Impl and the
// out-of-class Engine member definitions) is part of the engine's
// implementation surface; everything inside is private to this TU and
// has internal linkage by default.

namespace {

// ---------------------------------------------------------------------------
// Schema fingerprint — gates load_state on XML mismatch.
//
// The pool serialization (write_state / read_state) stores molecule
// instances and components by integer indices into Model::molecule_types
// and MoleculeType::components.  A state file written against one XML
// can be read against a different XML without runtime errors, but every
// index then refers to a different schema slot — silently corrupt
// trajectories.
//
// We compute an FNV-1a 64-bit hash over the canonical schema text
// (molecule type names, ordered component names, ordered allowed
// states) at save_state and re-compute on load_state; mismatch throws
// loudly with the saved-vs-current digests so the caller can see they
// passed the wrong XML.  Parameter values, rate constants, and seed
// species do NOT participate — those are run-time inputs that legally
// vary between save and load (e.g., resuming a checkpoint with
// modified set_param overrides).
// ---------------------------------------------------------------------------
uint64_t compute_schema_fingerprint(const Model& model) {
  // FNV-1a 64-bit.
  constexpr uint64_t kOffset = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t h = kOffset;
  auto absorb = [&](std::string_view s) {
    for (unsigned char const c : s) {
      h ^= c;
      h *= kPrime;
    }
    h ^= '\0';
    h *= kPrime; // record-separator
  };
  for (const auto& mt : model.molecule_types) {
    absorb(mt.name);
    for (const auto& comp : mt.components) {
      absorb(comp.name);
      for (const auto& st : comp.allowed_states)
        absorb(st);
      absorb("/c"); // component terminator
    }
    absorb("/m"); // molecule-type terminator
  }
  return h;
}

// ---------------------------------------------------------------------------
// Helper: check that all assigned pattern molecules map to distinct actual
// molecules.  BNGL semantics require each molecule node in a pattern to
// correspond to a different actual molecule; without this check, a self-bonded
// molecule (e.g. A(x!1,y!1)) could satisfy a two-molecule ring pattern
// (e.g. A(x!1,y!2).A(x!2,y!1)) by mapping both pattern molecules to
// the same actual molecule.
// ---------------------------------------------------------------------------
bool all_distinct_molecules(const std::vector<int>& mol_assignments, int start, int end) {
  for (int i = start; i < end; ++i) {
    if (mol_assignments[i] < 0)
      continue;
    for (int j = i + 1; j < end; ++j) {
      if (mol_assignments[j] == mol_assignments[i])
        return false;
    }
  }
  return true;
}

// ===========================================================================
// AgentPool — particle-level molecule tracking
// ===========================================================================

struct ComponentInstance {
  int state_index = -1;  // -1 = no internal state; else index into allowed_states
  int bond_partner = -1; // global component ID of bonded partner, -1 = free
};

struct MoleculeInstance {
  int type_index = -1;
  std::vector<int> comp_ids; // global component indices
  int complex_id = -1;
  bool active = false;
};

class AgentPool {
public:
  explicit AgentPool(const Model& model) : model_(model) {
    type_mol_index_.resize(model.molecule_types.size());
  }

  // Capacity hint for a bulk build: `n_molecules` molecules carrying
  // `n_components` components between them, `per_type[t]` of them of
  // molecule type `t`.  Every arena below otherwise grows by doubling
  // through the whole walk, so the pool is reallocated and copied a
  // logarithmic number of times — and `molecules_` copies a vector per
  // element, so that is the most expensive of them (issue #67).
  //
  // Pure capacity: nothing here changes what the pool holds, and a count
  // that is short only costs the growth it failed to pre-empt.  The
  // totals are `long long` because they are products of a seed count and
  // a per-species molecule count, either of which can be large; the pool
  // addresses molecules and components with `int`, so a total that does
  // not fit one is not a reservation to honour — skip it and let the
  // walk fail where it always would.
  void reserve_for_seed(long long n_molecules, long long n_components, long long n_complexes,
                        const std::vector<long long>& per_type) {
    constexpr long long kMax = std::numeric_limits<int>::max();
    if (n_molecules <= 0 || n_molecules > kMax || n_components < 0 || n_components > kMax)
      return;
    auto const mols = static_cast<size_t>(n_molecules);
    auto const comps = static_cast<size_t>(n_components);

    molecules_.reserve(molecules_.size() + mols);
    type_mol_pos_.reserve(molecules_.size() + mols);
    cx_moves_.reserve(cx_moves_.size() + mols);
    // Sized by the complexes the seed LEAVES, not the one-per-molecule the
    // walk creates before its bonds merge them: a hash map never gives a
    // bucket array back, so reserving the transient peak would hold it for
    // the rest of the run.  A seed bond that closes a cycle rather than
    // merging makes this an under-estimate by one, which costs a rehash
    // near the end and nothing else.
    if (n_complexes > 0 && n_complexes <= kMax)
      complex_members_.reserve(complex_members_.size() + static_cast<size_t>(n_complexes));
    components_.reserve(components_.size() + comps);
    comp_to_mol_.reserve(comp_to_mol_.size() + comps);
    for (size_t t = 0; t < per_type.size() && t < type_mol_index_.size(); ++t) {
      if (per_type[t] > 0 && per_type[t] <= kMax)
        type_mol_index_[t].reserve(type_mol_index_[t].size() + static_cast<size_t>(per_type[t]));
    }
  }

  // Create a new molecule of the given type. Returns molecule ID.
  int add_molecule(int type_index) {
    auto& mtype = model_.molecule_types[type_index];
    int mol_id;
    if (!free_mol_ids_.empty()) {
      mol_id = free_mol_ids_.back();
      free_mol_ids_.pop_back();
      molecules_[mol_id] = MoleculeInstance{};
    } else {
      mol_id = static_cast<int>(molecules_.size());
      molecules_.emplace_back();
    }

    auto& mol = molecules_[mol_id];
    mol.type_index = type_index;
    mol.active = true;

    // Allocate components
    mol.comp_ids.reserve(mtype.components.size());
    for (const auto& component : mtype.components) {
      int comp_id;
      if (!free_comp_ids_.empty()) {
        comp_id = free_comp_ids_.back();
        free_comp_ids_.pop_back();
        components_[comp_id] = ComponentInstance{};
        comp_to_mol_[comp_id] = mol_id;
      } else {
        comp_id = static_cast<int>(components_.size());
        components_.emplace_back();
        comp_to_mol_.push_back(mol_id);
      }
      // Default state: 0 if states exist, -1 if no states
      components_[comp_id].state_index = component.allowed_states.empty() ? -1 : 0;
      mol.comp_ids.push_back(comp_id);
    }

    // New molecule gets its own complex
    int const cx = next_complex_id_++;
    mol.complex_id = cx;
    complex_members_[cx] = {mol_id};
    // Deliberately not `mark_cx_dirty(cx)`.  `cx` was just minted from a
    // counter that never recycles, so no entry for it can exist in any
    // label cache, and `cached_label_of` already treats an absent id as
    // dirty (see `cxs_dirty_`).  The set insert would be information-free
    // — and at seed time it is one per molecule, which is what left the
    // dirty set holding every complex in the model (issue #67).
    //
    // The edit log is a different question: its reader has to SEE the
    // birth rather than ask whether an edit is outstanding, so it still
    // gets the id.
    if (track_cx_edits_)
      cx_edits_.push_back(cx);
    cx_moves_.push_back(mol_id);

    if (mol_id >= static_cast<int>(type_mol_pos_.size()))
      type_mol_pos_.resize(molecules_.size(), -1);
    type_mol_pos_[mol_id] = static_cast<int>(type_mol_index_[type_index].size());
    type_mol_index_[type_index].push_back(mol_id);
    ++active_mol_count_;
    return mol_id;
  }

  // Delete a molecule (handles unbinding all bonds first).
  void delete_molecule(int mol_id) {
    auto& mol = molecules_[mol_id];
    if (!mol.active)
      return;

    // Unbind all bonds
    for (int const cid : mol.comp_ids) {
      if (components_[cid].bond_partner >= 0)
        remove_bond(cid);
    }

    // Remove from complex
    auto cxit = complex_members_.find(mol.complex_id);
    if (cxit != complex_members_.end()) {
      auto& members = cxit->second;
      members.erase(std::remove(members.begin(), members.end(), mol_id), members.end());
      if (members.empty()) {
        cxs_died_.push_back(cxit->first);
        cxs_dirty_.erase(cxit->first); // dead complex: drop any pending dirty mark
        complex_members_.erase(cxit);
      } else {
        mark_cx_dirty(cxit->first); // surviving complex lost a member -> relabel
      }
    }

    // Remove from type index.  Swap-with-back, not an order-preserving
    // erase: a linear `std::remove` here is O(population of the type),
    // paid on every deletion, and that alone made per-event cost grow
    // with system size (issue #62).  Nothing downstream depends on the
    // list's order — `molecules_of_type` consumers either scan it whole
    // or sample from it by weight, and the Fenwick samplers are keyed by
    // molecule id, not by position — so the cheap removal is free of
    // semantic consequence.  `type_mol_pos_` keeps each live molecule's
    // slot so the swap is O(1); `kPoolIndexSelfCheck` (Debug and ASan
    // builds) proves that slot against the list itself on every removal.
    auto& tlist = type_mol_index_[mol.type_index];
    int const pos = type_mol_pos_[mol_id];
    if constexpr (kPoolIndexSelfCheck) {
      if (pos < 0 || pos >= static_cast<int>(tlist.size()) || tlist[pos] != mol_id) {
        fprintf(stderr,
                "[pool_index_selfcheck] type_mol_pos_ out of step: mol=%d type=%d "
                "pos=%d list_size=%d list_at_pos=%d\n",
                mol_id, mol.type_index, pos, static_cast<int>(tlist.size()),
                (pos >= 0 && pos < static_cast<int>(tlist.size())) ? tlist[pos] : -1);
        std::abort();
      }
    }
    int const moved = tlist.back();
    tlist[pos] = moved;
    type_mol_pos_[moved] = pos;
    tlist.pop_back();
    type_mol_pos_[mol_id] = -1;

    // Free component slots
    for (int const cid : mol.comp_ids) {
      components_[cid] = ComponentInstance{};
      comp_to_mol_[cid] = -1;
      free_comp_ids_.push_back(cid);
    }

    mol.active = false;
    mol.comp_ids.clear();
    free_mol_ids_.push_back(mol_id);
    cx_moves_.push_back(mol_id);
    --active_mol_count_;
    if constexpr (kPoolIndexSelfCheck) {
      // Independent reading: every inactive slot is on the free list and
      // nothing else is, so the arena minus the free list is the live count.
      auto const independent = static_cast<int>(molecules_.size() - free_mol_ids_.size());
      if (active_mol_count_ != independent) {
        fprintf(stderr, "[pool_index_selfcheck] active_mol_count_ drift: tally=%d arena-free=%d\n",
                active_mol_count_, independent);
        std::abort();
      }
    }
  }

  void set_state(int comp_id, int new_state) {
    components_[comp_id].state_index = new_state;
    // No membership change, but the complex's canonical label depends on
    // component states, so its cached label is now stale.
    mark_cx_dirty(molecules_[comp_to_mol_[comp_id]].complex_id);
  }

  void add_bond(int comp_a, int comp_b) {
    components_[comp_a].bond_partner = comp_b;
    components_[comp_b].bond_partner = comp_a;

    int const mol_a = comp_to_mol_[comp_a];
    int const mol_b = comp_to_mol_[comp_b];
    int const cx_a = molecules_[mol_a].complex_id;
    int const cx_b = molecules_[mol_b].complex_id;

    if (cx_a != cx_b) {
      merge_complexes(cx_a, cx_b); // dirties cx_a (keep), evicts cx_b (merged away)
    } else {
      // P7: both endpoints already in the same complex — the new bond
      // closes a cycle (|edges| increases by 1, |vertices| unchanged).
      // The complex id is unchanged but its graph (and label) changed.
      ++cycle_bond_count_[cx_a];
      mark_cx_dirty(cx_a);
    }
  }

  // Remove bond at comp_a (and its partner). May split complex.
  void remove_bond(int comp_a) {
    if constexpr (kRemoveBondProfile)
      remove_profile_.remove_bond_calls++;
    int const comp_b = components_[comp_a].bond_partner;
    if (comp_b < 0)
      return;

    components_[comp_a].bond_partner = -1;
    components_[comp_b].bond_partner = -1;

    int const mol_a = comp_to_mol_[comp_a];
    int const mol_b = comp_to_mol_[comp_b];
    int const old_cx = molecules_[mol_a].complex_id;

    // A bond was removed from old_cx, so its graph (and label) changed.
    // old_cx always survives remove_bond — mol_a stays in it whether or
    // not a split occurs — so dirtying it here is unconditional; the
    // split case additionally dirties the new complex inside
    // split_complex_if_needed.
    mark_cx_dirty(old_cx);

    // Check if complex needs splitting
    split_complex_if_needed(mol_a, mol_b, old_cx);
  }

  // Bucket a size into one of 7 histogram slots:
  //   0: 1   1: 2   2: 3-4   3: 5-8   4: 9-16   5: 17-32   6: 33+
  static int profile_size_bucket(size_t n) {
    if (n <= 1)
      return 0;
    if (n == 2)
      return 1;
    if (n <= 4)
      return 2;
    if (n <= 8)
      return 3;
    if (n <= 16)
      return 4;
    if (n <= 32)
      return 5;
    return 6;
  }

  // Accessors
  const MoleculeInstance& molecule(int mol_id) const { return molecules_[mol_id]; }
  const ComponentInstance& component(int comp_id) const { return components_[comp_id]; }
  int mol_of_comp(int comp_id) const { return comp_to_mol_[comp_id]; }
  const std::vector<int>& molecules_of_type(int type_index) const {
    return type_mol_index_[type_index];
  }
  int complex_of(int mol_id) const { return molecules_[mol_id].complex_id; }
  const std::vector<int>& molecules_in_complex(int cx_id) const {
    static const std::vector<int> empty;
    auto it = complex_members_.find(cx_id);
    return (it != complex_members_.end()) ? it->second : empty;
  }

  // All live complexes, keyed by complex id -> the molecule ids it
  // contains.  The species enumerator (issue #9 §2) walks this to visit
  // the pool one connected complex at a time.  Empty member lists are
  // pruned eagerly (see delete_molecule / merge_complexes), so every
  // entry is a non-empty live complex.
  const std::unordered_map<int, std::vector<int>>& complexes() const { return complex_members_; }

  // P7: number of cycle bonds in complex `cx_id` (|edges| - |vertices| + 1
  // for the connected subgraph).  Returns 0 for tree complexes or unknown
  // complex ids.  Updated incrementally by add_bond / remove_bond /
  // merge_complexes / split_complex_if_needed.  Consumers use this to
  // short-circuit graph connectivity checks that would be no-ops on trees.
  int cycle_bond_count(int cx_id) const {
    auto it = cycle_bond_count_.find(cx_id);
    return (it != cycle_bond_count_.end()) ? it->second : 0;
  }

  // O(1).  The SSA loop consults this every event when a molecule limit
  // is set, so a scan of the pool here turned `-gml` into a per-event
  // O(population) tax even when the cap could never bind (issue #62).
  // Maintained by add_molecule / delete_molecule and recomputed on
  // load_state.
  int active_molecule_count() const { return active_mol_count_; }

  int molecule_count() const { return static_cast<int>(molecules_.size()); }

  // High-water mark of the component arena, INCLUDING slots currently on
  // the free list.  A caller that wants an O(1) side table keyed by
  // global component id sizes it by this, not by the live count.
  int component_slot_count() const { return static_cast<int>(components_.size()); }

  // Increment / decrement state
  void increment_state(int comp_id) {
    auto& c = components_[comp_id];
    int const mol_id = comp_to_mol_[comp_id];
    auto& mtype = model_.molecule_types[molecules_[mol_id].type_index];
    // Find which component this is
    auto& mol = molecules_[mol_id];
    for (int i = 0; i < static_cast<int>(mol.comp_ids.size()); ++i) {
      if (mol.comp_ids[i] == comp_id) {
        int const max_state = static_cast<int>(mtype.components[i].allowed_states.size()) - 1;
        if (c.state_index < max_state)
          c.state_index++;
        mark_cx_dirty(mol.complex_id); // state feeds the canonical label
        return;
      }
    }
  }

  void decrement_state(int comp_id) {
    auto& c = components_[comp_id];
    if (c.state_index > 0)
      c.state_index--;
    mark_cx_dirty(molecules_[comp_to_mol_[comp_id]].complex_id);
  }

  // --- State serialization ---

  void write_state(std::ostream& os) const {
    os << molecules_.size() << "\n";
    for (auto& m : molecules_) {
      os << m.type_index << " " << m.complex_id << " " << (m.active ? 1 : 0) << " "
         << m.comp_ids.size();
      for (int const cid : m.comp_ids)
        os << " " << cid;
      os << "\n";
    }
    os << components_.size() << "\n";
    for (auto& c : components_)
      os << c.state_index << " " << c.bond_partner << "\n";
    os << comp_to_mol_.size() << "\n";
    for (int const v : comp_to_mol_)
      os << v << " ";
    os << "\n";
    os << complex_members_.size() << "\n";
    for (auto& [cx, members] : complex_members_) {
      os << cx << " " << members.size();
      for (int const mid : members)
        os << " " << mid;
      os << "\n";
    }
    os << next_complex_id_ << "\n";
    os << free_mol_ids_.size() << "\n";
    for (int const v : free_mol_ids_)
      os << v << " ";
    os << "\n";
    os << free_comp_ids_.size() << "\n";
    for (int const v : free_comp_ids_)
      os << v << " ";
    os << "\n";
  }

  void read_state(std::istream& is) {
    int n;
    is >> n;
    molecules_.resize(n);
    for (int i = 0; i < n; ++i) {
      auto& m = molecules_[i];
      int ncomps, active_int;
      is >> m.type_index >> m.complex_id >> active_int >> ncomps;
      m.active = (active_int != 0);
      m.comp_ids.resize(ncomps);
      for (int j = 0; j < ncomps; ++j)
        is >> m.comp_ids[j];
    }
    is >> n;
    components_.resize(n);
    for (int i = 0; i < n; ++i)
      is >> components_[i].state_index >> components_[i].bond_partner;
    is >> n;
    comp_to_mol_.resize(n);
    for (int i = 0; i < n; ++i)
      is >> comp_to_mol_[i];
    is >> n;
    complex_members_.clear();
    // read_state replaces the whole pool: every prior dirty mark refers to
    // a complex id that no longer exists.  Clearing the set here pairs with
    // the wholesale cx_label_cache_ clear in Engine::Impl::load_state_from.
    cxs_dirty_.clear();
    cx_edits_.clear();
    for (int i = 0; i < n; ++i) {
      int cx, nm;
      is >> cx >> nm;
      auto& members = complex_members_[cx];
      members.resize(nm);
      for (int j = 0; j < nm; ++j)
        is >> members[j];
    }
    is >> next_complex_id_;
    is >> n;
    free_mol_ids_.resize(n);
    for (int i = 0; i < n; ++i)
      is >> free_mol_ids_[i];
    is >> n;
    free_comp_ids_.resize(n);
    for (int i = 0; i < n; ++i)
      is >> free_comp_ids_[i];

    // Rebuild type_mol_index (and its position side table, plus the live
    // molecule tally) from loaded molecules
    type_mol_index_.clear();
    type_mol_index_.resize(model_.molecule_types.size());
    type_mol_pos_.assign(molecules_.size(), -1);
    active_mol_count_ = 0;
    for (int i = 0; i < static_cast<int>(molecules_.size()); ++i) {
      if (molecules_[i].active) {
        auto& tlist = type_mol_index_[molecules_[i].type_index];
        type_mol_pos_[i] = static_cast<int>(tlist.size());
        tlist.push_back(i);
        ++active_mol_count_;
      }
    }

    // P7: recompute cycle-bond counts per complex from the restored graph,
    // using |edges| - |vertices| + 1 on each connected component.  Bonds
    // are double-counted via endpoints, so halve at the end.
    cycle_bond_count_.clear();
    std::unordered_map<int, int> half_edges; // cx_id -> 2 * edges
    std::unordered_map<int, int> vertices;   // cx_id -> |mols|
    for (const auto& m : molecules_) {
      if (!m.active)
        continue;
      ++vertices[m.complex_id];
      for (int const cid : m.comp_ids) {
        if (components_[cid].bond_partner >= 0)
          ++half_edges[m.complex_id];
      }
    }
    for (const auto& [cx, half] : half_edges) {
      int const edges = half / 2;
      int const verts = vertices[cx];
      int const cycles = edges - (verts - 1);
      if (cycles > 0)
        cycle_bond_count_[cx] = cycles;
    }
  }

private:
  void merge_complexes(int cx_keep, int cx_merge) {
    auto& keep = complex_members_[cx_keep];
    auto& merge = complex_members_[cx_merge];
    for (int const mid : merge) {
      molecules_[mid].complex_id = cx_keep;
      keep.push_back(mid);
      cx_moves_.push_back(mid);
    }
    complex_members_.erase(cx_merge);
    cxs_died_.push_back(cx_merge);

    // cx_keep absorbed cx_merge's molecules, so its cached label is stale;
    // cx_merge is gone, so drop any dirty mark it carried.
    mark_cx_dirty(cx_keep);
    cxs_dirty_.erase(cx_merge);

    // P7: combined complex inherits the sum of both counters.  The bridging
    // bond that triggered this merge is a tree edge in the merged complex
    // (it was the unique path between the two sub-complexes before merge),
    // so no new cycle is created by the merge itself.
    auto it = cycle_bond_count_.find(cx_merge);
    if (it != cycle_bond_count_.end()) {
      cycle_bond_count_[cx_keep] += it->second;
      cycle_bond_count_.erase(it);
    }
  }

  void split_complex_if_needed(int mol_a, int mol_b, int old_cx) {
    // BFS from mol_a within old complex; if mol_b is not reached, split.
    // P7: also count bonds within mol_a's reachable piece — needed so that
    // on a split we can redistribute the cycle count across the two pieces
    // without a second BFS over the other side.
    bool prof_sampled = false;
    std::chrono::steady_clock::time_point prof_t_total_start;
    if constexpr (kRemoveBondProfile) {
      remove_profile_.split_calls++;
      prof_sampled = (remove_profile_.split_calls % kRemoveBondProfileSampleEvery) == 0;
      if (prof_sampled) {
        remove_profile_.sampled_calls++;
        prof_t_total_start = std::chrono::steady_clock::now();
      }
    }

    auto& members = complex_members_[old_cx];
    if constexpr (kRemoveBondProfile) {
      int const b = profile_size_bucket(members.size());
      remove_profile_.cx_size_all[b]++;
    }

    if (members.size() <= 1) {
      // Singleton: if we got here via remove_bond, the removed bond was a
      // self-bond on one molecule — that bond was a cycle bond, so the
      // complex's cycle count must drop by 1.
      auto it = cycle_bond_count_.find(old_cx);
      if (it != cycle_bond_count_.end()) {
        if (--it->second <= 0)
          cycle_bond_count_.erase(it);
      }
      if constexpr (kRemoveBondProfile) {
        remove_profile_.singleton_short_circuit++;
        if (prof_sampled) {
          auto t1 = std::chrono::steady_clock::now();
          remove_profile_.total_ns +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - prof_t_total_start).count();
        }
      }
      return;
    }

    if constexpr (kRemoveBondProfile) {
      remove_profile_.bfs_calls++;
      int const b = profile_size_bucket(members.size());
      remove_profile_.cx_size_bfs[b]++;
      auto sz = static_cast<uint64_t>(members.size());
      remove_profile_.cx_size_sum_bfs += sz;
      remove_profile_.cx_size_max_bfs = std::max(sz, remove_profile_.cx_size_max_bfs);
    }

    std::chrono::steady_clock::time_point prof_t_bfs_start;
    if constexpr (kRemoveBondProfile) {
      if (prof_sampled)
        prof_t_bfs_start = std::chrono::steady_clock::now();
    }

    std::unordered_set<int> visited;
    std::queue<int> queue;
    visited.insert(mol_a);
    queue.push(mol_a);
    int piece_a_half_edges = 0; // each bond counted twice (once per endpoint)

    while (!queue.empty()) {
      int const cur = queue.front();
      queue.pop();
      auto& m = molecules_[cur];
      if (!m.active)
        continue;
      for (int const cid : m.comp_ids) {
        int const partner = components_[cid].bond_partner;
        if (partner < 0)
          continue;
        int const neighbor = comp_to_mol_[partner];
        ++piece_a_half_edges;
        if (visited.insert(neighbor).second)
          queue.push(neighbor);
      }
    }

    std::chrono::steady_clock::time_point prof_t_bfs_end;
    if constexpr (kRemoveBondProfile) {
      if (prof_sampled) {
        prof_t_bfs_end = std::chrono::steady_clock::now();
        remove_profile_.bfs_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(prof_t_bfs_end - prof_t_bfs_start)
                .count();
      }
      auto he = static_cast<uint64_t>(piece_a_half_edges);
      remove_profile_.half_edges_sum += he;
      remove_profile_.half_edges_max = std::max(he, remove_profile_.half_edges_max);
      remove_profile_.half_edges_hist[profile_size_bucket(he)]++;
    }

    int const old_cycle = cycle_bond_count(old_cx);

    if (visited.count(mol_b)) {
      // Still connected — removed bond was a cycle bond in the surviving
      // complex, so decrement (|edges| dropped by 1, |vertices| unchanged).
      int const new_cycle = old_cycle - 1;
      if (new_cycle > 0)
        cycle_bond_count_[old_cx] = new_cycle;
      else
        cycle_bond_count_.erase(old_cx);
      if constexpr (kRemoveBondProfile) {
        remove_profile_.cycle_bond_removals++;
        if (prof_sampled) {
          auto t1 = std::chrono::steady_clock::now();
          remove_profile_.total_ns +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - prof_t_total_start).count();
        }
      }
      return;
    }

    if constexpr (kRemoveBondProfile) {
      remove_profile_.tree_bond_splits++;
      remove_profile_.cx_size_split[profile_size_bucket(members.size())]++;
    }

    std::chrono::steady_clock::time_point prof_t_part_start;
    if constexpr (kRemoveBondProfile) {
      if (prof_sampled)
        prof_t_part_start = std::chrono::steady_clock::now();
    }

    // Split: mol_b and its connected component get a new complex.  The
    // removed bond was a tree edge across the cut, so total cycles across
    // the two resulting pieces equals the pre-split cycle count.  Piece A's
    // cycles come from its (edges - vertices + 1); Piece B gets the remainder.
    int const new_cx = next_complex_id_++;
    mark_cx_dirty(new_cx); // the split-off piece is a brand new complex
    std::vector<int> new_members;
    std::vector<int> old_members;

    for (int const mid : members) {
      if (visited.count(mid)) {
        old_members.push_back(mid);
      } else {
        molecules_[mid].complex_id = new_cx;
        new_members.push_back(mid);
        cx_moves_.push_back(mid);
      }
    }

    int const piece_a_edges = piece_a_half_edges / 2;
    int const piece_a_mols = static_cast<int>(visited.size());
    int piece_a_cycle = piece_a_edges - (piece_a_mols - 1);
    piece_a_cycle = std::max(piece_a_cycle, 0);
    int piece_b_cycle = old_cycle - piece_a_cycle;
    piece_b_cycle = std::max(piece_b_cycle, 0);

    complex_members_[old_cx] = std::move(old_members);
    complex_members_[new_cx] = std::move(new_members);

    if (piece_a_cycle > 0)
      cycle_bond_count_[old_cx] = piece_a_cycle;
    else
      cycle_bond_count_.erase(old_cx);
    if (piece_b_cycle > 0)
      cycle_bond_count_[new_cx] = piece_b_cycle;

    if constexpr (kRemoveBondProfile) {
      if (prof_sampled) {
        auto t1 = std::chrono::steady_clock::now();
        remove_profile_.partition_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - prof_t_part_start).count();
        remove_profile_.total_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - prof_t_total_start).count();
      }
    }
  }

  // model_ is a const reference because Engine never owns or rebinds the
  // model — it's passed in from the surrounding Simulator and outlives
  // the Engine for the duration of any session.  Rule-of-zero with a
  // value or unique_ptr would force a copy or move and break that
  // contract.  NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
  const Model& model_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::vector<MoleculeInstance> molecules_;
  std::vector<ComponentInstance> components_;
  std::vector<int> comp_to_mol_;
  std::unordered_map<int, std::vector<int>> complex_members_;
  int next_complex_id_ = 0;
  std::vector<std::vector<int>> type_mol_index_;
  // mol_id -> its slot in type_mol_index_[type], or -1 when the molecule
  // is not live.  Sized by the molecule arena's high-water mark, so a
  // reused mol_id slot is already addressable.
  std::vector<int> type_mol_pos_;
  int active_mol_count_ = 0;
  std::vector<int> free_mol_ids_;
  std::vector<int> free_comp_ids_;

public:
  // Cx IDs that died (via merge or last-member delete) since the
  // observer (species-obs tracker) last consumed this list.  Cheap
  // side-channel so flush can iterate only the handful of dead cxs
  // instead of the full stored cx_match_count map.
  std::vector<int> cxs_died_;
  void consume_dead_cxs(std::vector<int>& out) {
    out.swap(cxs_died_);
    cxs_died_.clear();
  }

  // Molecules whose complex membership changed since the observer last drained
  // this — a merge moving one side's members, a split moving the severed
  // piece, a birth, or a death.  Issue #33's per-complex reactant tallies are
  // keyed by complex, so they have to learn about these: a molecule far from
  // the edited bond has its complex reassigned without ever entering
  // fire_rule's affected-molecule BFS, which is exactly what a tally keyed off
  // `affected` would miss.
  //
  // Ids only — the reader re-reads the destination from the pool, so this list
  // never has to be replayed in order and a molecule that moved twice in one
  // event settles correctly either way.  The mutators that append here already
  // walk the molecules they are moving, so this costs a push_back and no
  // asymptotics.  Kept unconditional (rather than gated on "some rule needs
  // it") so the pool has no dependency on rule classification; the engine
  // clears it in one branch when no rule counts per complex.
  std::vector<int> cx_moves_;
  void consume_cx_moves(std::vector<int>& out) {
    out.swap(cx_moves_);
    cx_moves_.clear();
  }

  // Cx IDs whose cached canonical label is stale — a structural mutator
  // edited the complex since its label was last (re)computed (issue #9
  // §2 step 5, plan decision #6).  The structural mutators above own the
  // marking: it is pure cx-id bookkeeping with no canonical dependency,
  // so placing it on AgentPool catches every mutation site uniformly
  // (fire_rule's op loop, the add_molecules API, and read_state) instead
  // of relying on each caller to remember.  The label cache itself and
  // the lazy recompute live on Engine::Impl (it owns extract_complex /
  // canonical_label); Engine::Impl::cached_label_of consults this set on
  // read and erases an id once it recomputes.
  //
  // Bounded by the live complex count, not the run length: a mutator
  // that kills a complex (merge / last-member delete) erases its id
  // here, so the set never holds a dead id.  An absent id is treated as
  // dirty too (a just-born complex has no cache entry), so missing a
  // mark for a brand-new complex is harmless; missing one for an *edited
  // existing* complex is the bug the decision-#6 self-check catches.
  std::unordered_set<int> cxs_dirty_;

  // The same marks as `cxs_dirty_`, as a drainable list rather than a
  // set — for an observer that has to SEE each edit rather than ask
  // whether one is outstanding.  Two consumers cannot share the set: it
  // is cleared by whoever reads it first, so the second never learns of
  // the edit.
  //
  // Issue #52's canonical per-complex representative is the consumer: a
  // state change inside a complex reorders its canonical form without
  // moving any molecule between complexes, so `cx_moves_` above never
  // mentions it, yet the molecule that should price the complex has
  // changed.  Gated because it is pure overhead for every other model —
  // the flag is set at rule setup only when some rule needs a canonical
  // representative, which needs a pure-context reactant pattern carrying
  // a per-molecule local function tag.  Duplicates are fine; the reader
  // de-duplicates.
  std::vector<int> cx_edits_;
  bool track_cx_edits_ = false;
  void consume_cx_edits(std::vector<int>& out) {
    out.swap(cx_edits_);
    cx_edits_.clear();
  }

private:
  // O(1) cache-invalidation hook for the structural mutators above: a
  // single set insert per edited complex.  This is the entire event-loop
  // cost the cached-incremental label mode adds (plan §5); canonical
  // labeling itself never runs here.
  void mark_cx_dirty(int cx_id) {
    cxs_dirty_.insert(cx_id);
    if (track_cx_edits_)
      cx_edits_.push_back(cx_id);
  }

  // P7: |edges| - |vertices| + 1 per complex, maintained incrementally.
  // Absent entries mean 0 (tree or unknown complex).
  std::unordered_map<int, int> cycle_bond_count_;

public:
  const RemoveBondProfile& remove_profile() const { return remove_profile_; }

private:
  RemoveBondProfile remove_profile_;
};

// ===========================================================================
// Pattern matching — embedding counting
// ===========================================================================

// Count the number of valid embeddings of a PatternMolecule into a molecule
// instance. Optionally enumerate all valid component assignments.
//
// When `reacting_local` is non-null, the returned count (and the list of
// assignments, if requested) is deduplicated by reacting-component targets:
// two injective mappings that send the SAME local pattern component indices
// listed in `reacting_local` to the SAME molecule components are collapsed
// to a single embedding.  This matches NFsim's convention of treating
// multiple ways to fill in non-reacting pattern components on the same
// molecule as one physical reaction (see NFsim's checkForEquality in
// NFreactions/reactions/reaction.cpp and MappingSet::checkForEquality in
// NFreactions/mappings/mappingSet.cpp).  Without dedup, a 2-component
// shorthand pattern like L(s!+,s) on a 3-site L with one free site over-
// counts each ligand by the number of ways to assign the non-reacting
// `!+` component to the ligand's bonded sites.
int count_embeddings_single(const AgentPool& pool, int mol_id, const PatternMolecule& pm,
                            const Model& model,
                            std::vector<std::vector<int>>* all_assignments = nullptr,
                            const std::vector<int>* reacting_local = nullptr) {

  auto& mol = pool.molecule(mol_id);
  if (!mol.active || mol.type_index != pm.type_index)
    return 0;

  auto& mtype = model.molecule_types[pm.type_index];
  int n_pat = static_cast<int>(pm.components.size());
  if (n_pat == 0) {
    // Pattern with no component constraints: 1 embedding
    if (all_assignments)
      all_assignments->emplace_back();
    return 1;
  }

  // For each pattern component, find candidate actual components
  std::vector<std::vector<int>> candidates(n_pat);
  for (int pi = 0; pi < n_pat; ++pi) {
    auto& pc = pm.components[pi];
    for (int ci = 0; ci < static_cast<int>(mol.comp_ids.size()); ++ci) {
      // Check component type match
      // The actual component at position ci corresponds to mtype.components[ci]
      // The pattern component has comp_type_index (or we match by name)
      std::string const actual_name = mtype.components[ci].name;
      bool name_match = false;
      if (pc.comp_type_index >= 0) {
        name_match = (ci == pc.comp_type_index) ||
                     (actual_name == mtype.components[pc.comp_type_index].name);
      }
      if (!name_match) {
        // Try matching by name directly (handles symmetric components)
        name_match = (actual_name == pc.name);
        if (!name_match) {
          // Strip trailing digits from pattern name
          std::string base = pc.name;
          while (!base.empty() && std::isdigit(static_cast<unsigned char>(base.back())))
            base.pop_back();
          name_match = (actual_name == base);
        }
      }
      if (!name_match)
        continue;

      int const comp_id = mol.comp_ids[ci];
      auto& comp = pool.component(comp_id);

      // Check state constraint
      if (pc.required_state_index >= 0 && comp.state_index != pc.required_state_index)
        continue;

      // Check bond constraint.  Bound and BoundTo share the
      // "bond_partner must exist" gate, but BoundTo also has a
      // partner-identity check that runs in a later phase, so the
      // cases stay textually parallel rather than collapsed into
      // fall-through.  NOLINT(bugprone-branch-clone)
      switch (pc.bond_constraint) {
      case BondConstraint::Free:
        if (comp.bond_partner >= 0)
          continue;
        break;
      // NOLINTNEXTLINE(bugprone-branch-clone)
      case BondConstraint::Bound:
        if (comp.bond_partner < 0)
          continue;
        break;
      case BondConstraint::BoundTo:
        if (comp.bond_partner < 0)
          continue;
        // BoundTo validation deferred to bond-pair check
        break;
      case BondConstraint::Wildcard:
        break;
      }

      candidates[pi].push_back(ci);
    }
  }

  // Check all candidates non-empty
  for (int pi = 0; pi < n_pat; ++pi)
    if (candidates[pi].empty())
      return 0;

  // Enumerate valid assignments (backtracking, no duplicate components)
  int count = 0;
  std::vector<int> assignment(n_pat, -1);
  std::vector<bool> used(mol.comp_ids.size(), false);

  // Lambda recursion via explicit self-pass — avoids the std::function vtable
  // dispatch + heap alloc on every recursive call.  This routine is on the
  // hot path of propensity update and observable evaluation.
  auto enumerate = [&](auto& self, int pi) -> void {
    if (pi == n_pat) {
      // Valid assignment found
      if (all_assignments)
        all_assignments->push_back(assignment);
      ++count;
      return;
    }
    for (int const ci : candidates[pi]) {
      if (used[ci])
        continue;
      assignment[pi] = ci;
      used[ci] = true;
      self(self, pi + 1);
      used[ci] = false;
    }
  };

  enumerate(enumerate, 0);

  // Deduplicate by reacting-component targets if requested.
  // Two embeddings that send all `reacting_local` pattern components to the
  // same molecule-side component indices are the same physical reaction.
  if (reacting_local && !reacting_local->empty() && all_assignments && count > 1) {
    std::set<std::vector<int>> seen;
    std::vector<std::vector<int>> deduped;
    deduped.reserve(all_assignments->size());
    for (auto& emb : *all_assignments) {
      std::vector<int> key;
      key.reserve(reacting_local->size());
      for (int const rci : *reacting_local) {
        if (rci >= 0 && rci < static_cast<int>(emb.size()))
          key.push_back(emb[rci]);
      }
      if (seen.insert(key).second)
        deduped.push_back(std::move(emb));
    }
    *all_assignments = std::move(deduped);
    count = static_cast<int>(all_assignments->size());
  } else if (reacting_local && !reacting_local->empty() && count > 1) {
    // Count-only path: recompute count via explicit enumeration with dedup.
    std::set<std::vector<int>> seen;
    std::vector<int> tmp(n_pat, -1);
    std::vector<bool> tmp_used(mol.comp_ids.size(), false);
    int deduped_count = 0;
    auto enumerate2 = [&](auto& self, int pi) -> void {
      if (pi == n_pat) {
        std::vector<int> key;
        key.reserve(reacting_local->size());
        for (int const rci : *reacting_local) {
          if (rci >= 0 && rci < n_pat)
            key.push_back(tmp[rci]);
        }
        if (seen.insert(key).second)
          ++deduped_count;
        return;
      }
      for (int const ci : candidates[pi]) {
        if (tmp_used[ci])
          continue;
        tmp[pi] = ci;
        tmp_used[ci] = true;
        self(self, pi + 1);
        tmp_used[ci] = false;
      }
    };
    enumerate2(enumerate2, 0);
    count = deduped_count;
  }

  return count;
}

// For multi-molecule pattern matching with bond constraints:
// Given a set of molecule-to-instance assignments, verify that bond pairs
// in the pattern are satisfied.
bool check_pattern_bonds(
    const AgentPool& pool, const Pattern& pat,
    const std::vector<int>& mol_assignments, // pattern mol idx -> actual mol_id
    const std::vector<std::vector<int>>&
        comp_maps) { // per-mol: pattern comp idx -> actual comp local idx

  for (auto& bond : pat.bonds) {
    // Find which pattern molecule and local comp each flat index belongs to
    int const flat_a = bond.comp_flat_a, flat_b = bond.comp_flat_b;
    int base = 0;
    int mol_idx_a = -1, local_a = -1, mol_idx_b = -1, local_b = -1;
    for (int mi = 0; mi < static_cast<int>(pat.molecules.size()); ++mi) {
      int const nc = static_cast<int>(pat.molecules[mi].components.size());
      if (flat_a >= base && flat_a < base + nc) {
        mol_idx_a = mi;
        local_a = flat_a - base;
      }
      if (flat_b >= base && flat_b < base + nc) {
        mol_idx_b = mi;
        local_b = flat_b - base;
      }
      base += nc;
    }

    if (mol_idx_a < 0 || mol_idx_b < 0 || local_a < 0 || local_b < 0)
      return false;

    int const actual_mol_a = mol_assignments[mol_idx_a];
    int const actual_mol_b = mol_assignments[mol_idx_b];
    if (actual_mol_a < 0 || actual_mol_b < 0)
      return false;

    if (mol_idx_a >= static_cast<int>(comp_maps.size()) ||
        mol_idx_b >= static_cast<int>(comp_maps.size()))
      return false;
    if (local_a >= static_cast<int>(comp_maps[mol_idx_a].size()) ||
        local_b >= static_cast<int>(comp_maps[mol_idx_b].size()))
      return false;

    int const actual_comp_local_a = comp_maps[mol_idx_a][local_a];
    int const actual_comp_local_b = comp_maps[mol_idx_b][local_b];

    int const actual_comp_id_a = pool.molecule(actual_mol_a).comp_ids[actual_comp_local_a];
    int const actual_comp_id_b = pool.molecule(actual_mol_b).comp_ids[actual_comp_local_b];

    if (pool.component(actual_comp_id_a).bond_partner != actual_comp_id_b)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Multi-molecule pattern matching for observables
// ---------------------------------------------------------------------------

// Count the number of complete multi-molecule pattern embeddings that start
// from a given seed molecule (pattern molecule index 0).
//
// Algorithm: for each embedding of the seed pattern molecule into the actual
// molecule, follow the pattern's bond constraints to discover which actual
// molecules must fill the remaining pattern molecule slots. Recursively
// verify that each discovered actual molecule matches its pattern molecule
// and that all bonds are satisfied.
//
// For "Molecules" type observables, this counts total embeddings.
// For "Species" type, the caller de-duplicates by complex.

int count_multi_molecule_embeddings(const AgentPool& pool, int seed_mol_id, const Pattern& pat,
                                    const Model& model) {

  int n_pat_mols = static_cast<int>(pat.molecules.size());
  if (n_pat_mols == 0)
    return 0;
  if (n_pat_mols == 1) {
    // Single-molecule pattern: use the fast path
    return count_embeddings_single(pool, seed_mol_id, pat.molecules[0], model);
  }

  // Precompute: for each pattern bond, which pattern molecules and local
  // component indices are involved.
  struct BondInfo {
    int mol_a, local_a, mol_b, local_b;
  };
  std::vector<BondInfo> bond_infos;
  for (auto& bond : pat.bonds) {
    BondInfo bi{-1, -1, -1, -1};
    int base = 0;
    for (int mi = 0; mi < n_pat_mols; ++mi) {
      int const nc = static_cast<int>(pat.molecules[mi].components.size());
      if (bond.comp_flat_a >= base && bond.comp_flat_a < base + nc) {
        bi.mol_a = mi;
        bi.local_a = bond.comp_flat_a - base;
      }
      if (bond.comp_flat_b >= base && bond.comp_flat_b < base + nc) {
        bi.mol_b = mi;
        bi.local_b = bond.comp_flat_b - base;
      }
      base += nc;
    }
    bond_infos.push_back(bi);
  }

  // Build adjacency: for each pattern molecule, which bonds connect it
  // to which other pattern molecules.
  // adj[mi] = list of (bond_idx, other_pat_mol, my_local_comp, other_local_comp)
  struct AdjEntry {
    int bond_idx, other_mol, my_local, other_local;
  };
  std::vector<std::vector<AdjEntry>> adj(n_pat_mols);
  for (int bi_idx = 0; bi_idx < static_cast<int>(bond_infos.size()); ++bi_idx) {
    auto& bi = bond_infos[bi_idx];
    if (bi.mol_a >= 0 && bi.mol_b >= 0) {
      adj[bi.mol_a].push_back({bi_idx, bi.mol_b, bi.local_a, bi.local_b});
      adj[bi.mol_b].push_back({bi_idx, bi.mol_a, bi.local_b, bi.local_a});
    }
  }

  // Get all embeddings of the seed pattern molecule (index 0) into seed_mol_id
  std::vector<std::vector<int>> seed_embs;
  count_embeddings_single(pool, seed_mol_id, pat.molecules[0], model, &seed_embs);

  if (seed_embs.empty())
    return 0;

  int total_count = 0;

  // For each seed embedding, recursively enumerate all complete embeddings.
  //
  // The enumerator picks the next unassigned pattern molecule reachable from
  // an already-assigned one via a bond, walks that bond into the actual
  // partner molecule, and BRANCHES over all partner pattern-embeddings whose
  // bond endpoint matches the walked bond. Branching is required when a
  // partner molecule type has identical (symmetric) components only one of
  // which the pattern names: every such embedding is a distinct mapping that
  // contributes to a `Molecules` count. Once no bond-reachable pattern
  // molecules remain, control falls through to the disconnected-pattern
  // backtracker (preserving the original behavior for patterns like
  // `egfr().egfr()`). Bonds whose endpoints are both already assigned are
  // verified once at the leaves via `check_pattern_bonds`.
  for (auto& seed_comp_map : seed_embs) {
    std::vector<int> mol_assignments(n_pat_mols, -1);
    std::vector<std::vector<int>> comp_maps(n_pat_mols);
    mol_assignments[0] = seed_mol_id;
    comp_maps[0] = seed_comp_map;

    std::function<int()> enumerate_extensions = [&]() -> int {
      // Find an unassigned pattern molecule reachable from an assigned one
      // via a bond. Deterministic order: scan assigned pats in index order
      // and pick the first bond leading to an unassigned partner.
      int via_pat = -1;
      const AdjEntry* via_ae = nullptr;
      for (int p = 0; p < n_pat_mols; ++p) {
        if (mol_assignments[p] < 0)
          continue;
        for (auto& ae : adj[p]) {
          if (mol_assignments[ae.other_mol] < 0) {
            via_pat = p;
            via_ae = &ae;
            break;
          }
        }
        if (via_pat >= 0)
          break;
      }

      if (via_pat < 0) {
        // No bond-reachable extension. Either everything's assigned, or the
        // pattern has disconnected components.
        std::vector<int> unassigned;
        for (int mi = 0; mi < n_pat_mols; ++mi) {
          if (mol_assignments[mi] < 0)
            unassigned.push_back(mi);
        }

        if (unassigned.empty()) {
          if (!check_pattern_bonds(pool, pat, mol_assignments, comp_maps))
            return 0;
          if (!all_distinct_molecules(mol_assignments, 0, n_pat_mols))
            return 0;
          return 1;
        }

        // Disconnected components: enumerate over molecules in the seed's
        // complex (matches the original assign_unassigned behavior).
        int const cx = pool.complex_of(seed_mol_id);
        auto cx_members = pool.molecules_in_complex(cx);

        int sub = 0;
        std::function<void(int)> assign_unassigned = [&](int ui) {
          if (ui == static_cast<int>(unassigned.size())) {
            if (check_pattern_bonds(pool, pat, mol_assignments, comp_maps) &&
                all_distinct_molecules(mol_assignments, 0, n_pat_mols))
              ++sub;
            return;
          }
          int const pat_mi = unassigned[ui];
          auto& target_pm = pat.molecules[pat_mi];
          for (int const cand : cx_members) {
            if (!pool.molecule(cand).active)
              continue;
            if (pool.molecule(cand).type_index != target_pm.type_index)
              continue;
            bool already_used = false;
            for (int mi2 = 0; mi2 < n_pat_mols; ++mi2) {
              if (mi2 != pat_mi && mol_assignments[mi2] == cand) {
                already_used = true;
                break;
              }
            }
            if (already_used)
              continue;

            std::vector<std::vector<int>> cand_embs;
            count_embeddings_single(pool, cand, target_pm, model, &cand_embs);
            for (auto& emb : cand_embs) {
              mol_assignments[pat_mi] = cand;
              comp_maps[pat_mi] = emb;
              assign_unassigned(ui + 1);
            }
            mol_assignments[pat_mi] = -1;
            comp_maps[pat_mi].clear();
          }
        };
        assign_unassigned(0);
        return sub;
      }

      // Walk the bond from via_pat to find the actual partner molecule.
      int const via_actual = mol_assignments[via_pat];
      auto& via_cm = comp_maps[via_pat];
      if (via_ae->my_local >= static_cast<int>(via_cm.size()))
        return 0;
      int const my_actual_local = via_cm[via_ae->my_local];
      int const my_actual_comp_id = pool.molecule(via_actual).comp_ids[my_actual_local];
      int const partner_comp_id = pool.component(my_actual_comp_id).bond_partner;
      if (partner_comp_id < 0)
        return 0;
      int const partner_mol_id = pool.mol_of_comp(partner_comp_id);

      int const next_pat = via_ae->other_mol;
      auto& other_pm = pat.molecules[next_pat];
      auto& partner_mol = pool.molecule(partner_mol_id);
      if (partner_mol.type_index != other_pm.type_index)
        return 0;

      // Locate the partner-side component bonded to my_actual_comp_id.
      int partner_local = -1;
      for (int ci = 0; ci < static_cast<int>(partner_mol.comp_ids.size()); ++ci) {
        if (partner_mol.comp_ids[ci] == partner_comp_id) {
          partner_local = ci;
          break;
        }
      }

      std::vector<std::vector<int>> other_embs;
      count_embeddings_single(pool, partner_mol_id, other_pm, model, &other_embs);

      int sub = 0;
      for (auto& emb : other_embs) {
        if (via_ae->other_local >= static_cast<int>(emb.size()))
          continue;
        if (emb[via_ae->other_local] != partner_local)
          continue;
        // BUG FIX (vs. prior BFS): branch over EVERY embedding consistent
        // with the bond. The old code committed to the first match and
        // dropped sym-equivalent alternatives, under-counting `Molecules`
        // observables whose pattern reaches a sym-component partner via a
        // non-sym bond endpoint.
        mol_assignments[next_pat] = partner_mol_id;
        comp_maps[next_pat] = emb;
        sub += enumerate_extensions();
      }
      mol_assignments[next_pat] = -1;
      comp_maps[next_pat].clear();
      return sub;
    };

    total_count += enumerate_extensions();
  }

  return total_count;
}

// Pre-computed bond adjacency for multi-molecule pattern matching.
struct PatternAdj {
  struct Entry {
    int other_mol, my_local, other_local;
  };
  std::vector<std::vector<Entry>> adj; // indexed by pattern molecule
};
// Build pre-computed adjacency for a sub-range of a pattern.
PatternAdj build_pattern_adjacency(const Pattern& pat, int pat_start, int pat_end) {
  int const n_pat_mols = static_cast<int>(pat.molecules.size());
  PatternAdj result;
  result.adj.resize(n_pat_mols);

  for (auto& bond : pat.bonds) {
    int mol_a = -1, local_a = -1, mol_b = -1, local_b = -1;
    int base = 0;
    for (int mi = 0; mi < n_pat_mols; ++mi) {
      int const nc = static_cast<int>(pat.molecules[mi].components.size());
      if (bond.comp_flat_a >= base && bond.comp_flat_a < base + nc) {
        mol_a = mi;
        local_a = bond.comp_flat_a - base;
      }
      if (bond.comp_flat_b >= base && bond.comp_flat_b < base + nc) {
        mol_b = mi;
        local_b = bond.comp_flat_b - base;
      }
      base += nc;
    }
    if (mol_a >= pat_start && mol_a < pat_end && mol_b >= pat_start && mol_b < pat_end) {
      result.adj[mol_a].push_back({mol_b, local_a, local_b});
      result.adj[mol_b].push_back({mol_a, local_b, local_a});
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// 2-mol-1-bond fully-constrained fast path (Candidate B).
//
// Specializes count_multi_mol_fast for the shape common to every cmm-using
// rule on the engaged multisite/homopoly workloads (16 multisite catalysis/
// reverse-binding rules + 1 homopoly reverse rule).  A FastMatchSlot is
// populated at model load for each
// eligible (rule, pattern-side) pair; at cmm entry, when enabled, we dispatch
// here and skip the generic BFS/embedding-enumeration body.
//
// Eligibility (all must hold):
//   - pat[pat_start..pat_end) has exactly 2 molecules and exactly 1 bond
//     within range, connecting them.
//   - Non-symmetric side: pattern components each target distinct mol-type
//     components (no shorthand aliasing like `E(s!1,s)`).  Any K pat comps
//     allowed.
//   - Symmetric side (type has repeated component names, e.g. P(s,s)):
//     pattern must have exactly one pat component (K=1) — the bond endpoint —
//     so the number of embeddings collapses to a simple enumeration over the
//     name-matching mol-type-local indices.
//   - No BoundTo labels other than the single pattern bond endpoint.
//
// Under those conditions the generic count_multi_mol_fast returns an integer
// equal to the number of valid (seed-emb, partner-emb) pairs whose bond
// endpoints line up, and the specialized path returns the same value.  The
// invariant gate (kFastMatchInvariant) runs both paths and aborts on mismatch
// during development.
struct FastMatchSlot {
  bool enabled = false;
  int seed_type = -1;
  int partner_type = -1;

  // Pattern molecule indices of the seed and its partner (P6: needed so the
  // select path can populate ReactionMatch.mol_ids at the correct slots).
  int seed_pat_idx = -1;
  int partner_pat_idx = -1;

  // Pattern component index of the bond endpoint, within each pat molecule.
  // For symmetric sides (K=1), these equal 0 trivially; for non-symmetric
  // K>1 they identify which pat comp is the bond-carrying one.
  int seed_bond_pi = -1;
  int partner_bond_pi = -1;

  // Bond-endpoint candidates: mol-type-local indices whose name matches the
  // bond-endpoint pat component.  Size 1 for non-symmetric sides, N for a
  // symmetric side (with K=1).
  std::vector<int> seed_bond_candidates;
  std::vector<int> partner_bond_candidates;

  // Bond-endpoint dynamic checks (applied per candidate at match time):
  int seed_bond_state_req = -1; // -1 = don't care
  int seed_bond_bond_req = 0;   // 0=don't care, 1=Free, 2=Bound
  int partner_bond_state_req = -1;
  int partner_bond_bond_req = 0;

  // Non-endpoint pat components on each side.  Only populated for
  // non-symmetric sides with K>1; each CompCheck targets a unique
  // mol-type-local index.
  struct CompCheck {
    int local_idx;
    int state_req;
    int bond_req;
  };
  std::vector<CompCheck> seed_non_bond_checks;
  std::vector<CompCheck> partner_non_bond_checks;

  // Per-pat-comp mol-type-local mapping, populated at load for ReactionMatch
  // construction (P6).  Size == K_seed / K_partner.  The bond-endpoint entry
  // (index seed_bond_pi / partner_bond_pi) is a sentinel -1 on symmetric
  // sides — resolved to the chosen candidate at select time.  All other
  // entries are resolved mol-type-local indices.
  std::vector<int> seed_pat_ci_locals;
  std::vector<int> partner_pat_ci_locals;
};

// Attempt to populate `out` with a fast-path descriptor for the sub-pattern
// pat[pat_start..pat_end) seeded at seed_pat_idx.  Returns true on success
// (also sets out.enabled=true); returns false and leaves out.enabled=false
// when the sub-pattern does not match the eligibility template.
bool build_fastmatch_slot(const Pattern& pat, int pat_start, int pat_end, int seed_pat_idx,
                          const PatternAdj& pa, const Model& model, FastMatchSlot& out,
                          std::string* reason_out = nullptr) {
  out = FastMatchSlot{};
  auto set_reason = [&](const std::string& r) {
    if (reason_out)
      *reason_out = r;
  };

  if (pat_end - pat_start != 2) {
    set_reason("not_2_mols");
    return false;
  }
  if (seed_pat_idx < pat_start || seed_pat_idx >= pat_end) {
    set_reason("seed_oob");
    return false;
  }
  if (seed_pat_idx >= static_cast<int>(pa.adj.size())) {
    set_reason("seed_adj_oob");
    return false;
  }
  if (pa.adj[seed_pat_idx].size() != 1) {
    set_reason("seed_deg_ne_1");
    return false;
  }

  auto& ae = pa.adj[seed_pat_idx][0];
  int const other_pat = ae.other_mol;
  if (other_pat < pat_start || other_pat >= pat_end) {
    set_reason("partner_oob");
    return false;
  }
  if (other_pat == seed_pat_idx) {
    set_reason("self_bond");
    return false;
  }
  if (pa.adj[other_pat].size() != 1) {
    set_reason("partner_deg_ne_1");
    return false;
  }

  const auto& pm_seed = pat.molecules[seed_pat_idx];
  const auto& pm_partner = pat.molecules[other_pat];
  if (pm_seed.type_index < 0 || pm_partner.type_index < 0) {
    set_reason("type_unresolved");
    return false;
  }
  if (pm_seed.type_index >= static_cast<int>(model.molecule_types.size())) {
    set_reason("seed_type_oob");
    return false;
  }
  if (pm_partner.type_index >= static_cast<int>(model.molecule_types.size())) {
    set_reason("partner_type_oob");
    return false;
  }

  const auto& mt_seed = model.molecule_types[pm_seed.type_index];
  const auto& mt_partner = model.molecule_types[pm_partner.type_index];
  const bool seed_sym = mt_seed.has_symmetric_components();
  const bool partner_sym = mt_partner.has_symmetric_components();

  // Build per-side eligibility + descriptor.  On a symmetric side, the
  // pattern must have exactly one component (K=1) — the bond endpoint — so
  // we enumerate name-matching mol-type-locals.  On a non-symmetric side,
  // any K is allowed; the bond endpoint's candidates set is a singleton.
  // Bound and BoundTo share the bond requirement code (2) but BoundTo
  // additionally constrains the partner identity via a later check, so
  // the cases are kept parallel for readability.
  // NOLINT(bugprone-branch-clone)
  auto bond_req_of = [](BondConstraint bc) -> int {
    switch (bc) {
    case BondConstraint::Free:
      return 1;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case BondConstraint::Bound:
      return 2;
    case BondConstraint::BoundTo:
      return 2;
    case BondConstraint::Wildcard:
      return 0;
    }
    return 0;
  };

  auto name_matches_local = [](const std::string& pat_name,
                               const std::string& actual_name) -> bool {
    if (actual_name == pat_name)
      return true;
    std::string base = pat_name;
    while (!base.empty() && std::isdigit(static_cast<unsigned char>(base.back())))
      base.pop_back();
    return actual_name == base;
  };

  auto build_side = [&](const PatternMolecule& pm, const MoleculeType& mt, int bond_pi, bool sym,
                        const char* side_tag, std::vector<int>& out_bond_candidates,
                        int& out_bond_state_req, int& out_bond_bond_req,
                        std::vector<FastMatchSlot::CompCheck>& out_non_bond,
                        std::vector<int>& out_pat_ci_locals) -> bool {
    auto sym_label = [&](const char* s) -> std::string { return std::string(side_tag) + "_" + s; };
    int const n_type_comps = static_cast<int>(mt.components.size());
    out_bond_candidates.clear();
    out_bond_state_req = -1;
    out_bond_bond_req = 0;

    int const K = static_cast<int>(pm.components.size());
    if (sym && K != 1) {
      set_reason(sym_label("sym_K_gt_1"));
      return false;
    }
    if (bond_pi < 0 || bond_pi >= K) {
      set_reason(sym_label("bond_pi_oob"));
      return false;
    }

    // pat-ci → mol-type-local mapping used by the select path.  The bond
    // endpoint's slot is a sentinel (-1) here — filled in at select time
    // from the chosen seed/partner candidate.  Non-endpoint slots carry
    // resolved locals (for symmetric K=1 sides, there are no non-endpoints).
    out_pat_ci_locals.assign(K, -1);

    // Collect the bond-endpoint pat component's candidates and dynamic req.
    {
      const auto& pc = pm.components[bond_pi];
      if (pc.bond_constraint == BondConstraint::BoundTo) {
        // bond endpoint with BoundTo is canonical — ok.
      }
      out_bond_state_req = pc.required_state_index;
      out_bond_bond_req = bond_req_of(pc.bond_constraint);

      if (pc.comp_type_index >= 0 && pc.comp_type_index < n_type_comps && !sym) {
        out_bond_candidates.push_back(pc.comp_type_index);
      } else {
        // Enumerate every mol-type-local whose name matches the pattern
        // component's name.  Matches count_embeddings_single's name logic.
        for (int ci = 0; ci < n_type_comps; ++ci) {
          if (name_matches_local(pc.name, mt.components[ci].name))
            out_bond_candidates.push_back(ci);
        }
      }
      if (out_bond_candidates.empty()) {
        set_reason(sym_label("bond_cand_empty"));
        return false;
      }
      // On non-symmetric sides the bond endpoint must resolve uniquely.
      if (!sym && out_bond_candidates.size() != 1) {
        set_reason(sym_label("bond_cand_ambiguous"));
        return false;
      }
    }

    // Non-endpoint pat components: only on non-symmetric sides.  Each must
    // resolve to a unique mol-type-local distinct from the bond endpoint.
    if (!sym && K > 1) {
      std::vector<bool> used_local(n_type_comps, false);
      used_local[out_bond_candidates.front()] = true;

      for (int pi = 0; pi < K; ++pi) {
        if (pi == bond_pi)
          continue;
        const auto& pc = pm.components[pi];

        // BoundTo on a non-endpoint pat component would imply a chain of
        // bonds — not fast-path-eligible.
        if (pc.bond_constraint == BondConstraint::BoundTo) {
          set_reason(sym_label("nonbond_boundto"));
          return false;
        }

        int local = -1;
        if (pc.comp_type_index >= 0 && pc.comp_type_index < n_type_comps) {
          local = pc.comp_type_index;
        } else {
          for (int ci = 0; ci < n_type_comps; ++ci) {
            if (name_matches_local(pc.name, mt.components[ci].name)) {
              local = ci;
              break;
            }
          }
        }
        if (local < 0) {
          set_reason(sym_label("nonbond_name_unresolved"));
          return false;
        }
        if (used_local[local]) {
          set_reason(sym_label("nonbond_local_collision"));
          return false;
        }
        used_local[local] = true;

        out_non_bond.push_back({local, pc.required_state_index, bond_req_of(pc.bond_constraint)});
        out_pat_ci_locals[pi] = local;
      }
    }
    return true;
  };

  out.seed_type = pm_seed.type_index;
  out.partner_type = pm_partner.type_index;
  out.seed_pat_idx = seed_pat_idx;
  out.partner_pat_idx = other_pat;
  out.seed_bond_pi = ae.my_local;
  out.partner_bond_pi = ae.other_local;
  if (!build_side(pm_seed, mt_seed, ae.my_local, seed_sym, "seed", out.seed_bond_candidates,
                  out.seed_bond_state_req, out.seed_bond_bond_req, out.seed_non_bond_checks,
                  out.seed_pat_ci_locals))
    return false;
  if (!build_side(pm_partner, mt_partner, ae.other_local, partner_sym, "partner",
                  out.partner_bond_candidates, out.partner_bond_state_req,
                  out.partner_bond_bond_req, out.partner_non_bond_checks,
                  out.partner_pat_ci_locals))
    return false;

  out.enabled = true;
  return true;
}

// Specialized 2-mol-1-bond-fully-constrained counter.  Returns the number of
// pattern embeddings seeded at seed_mol_id — equal to the generic cmm
// function's return value.  Pre-condition: fm.enabled == true.
inline int count_2mol_1bond_fc(const AgentPool& pool, int seed_mol_id, const FastMatchSlot& fm,
                               CmmFcProfile* fc_prof) {
  // -- profile scaffolding (gated, see kCmmFcProfile) --
  using fc_clock = std::chrono::steady_clock;
  bool fc_sampled = false;
  int fc_phase = -1;
  fc_clock::time_point fc_t0{};
  auto fc_ns = [](fc_clock::time_point a, fc_clock::time_point b) -> uint64_t {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  };
  if constexpr (kCmmFcProfile) {
    fc_prof->fc_calls++;
    fc_prof->fc_seed_bond_candidates_sum += static_cast<uint64_t>(fm.seed_bond_candidates.size());
    fc_prof->fc_partner_bond_candidates_sum +=
        static_cast<uint64_t>(fm.partner_bond_candidates.size());
    auto pnbc = static_cast<uint64_t>(fm.partner_non_bond_checks.size());
    fc_prof->fc_partner_non_bond_checks_sum += pnbc;
    fc_prof->fc_seed_non_bond_checks_sum += static_cast<uint64_t>(fm.seed_non_bond_checks.size());
    fc_prof->fc_seed_bond_candidates_max =
        std::max<uint64_t>(fm.seed_bond_candidates.size(), fc_prof->fc_seed_bond_candidates_max);
    fc_prof->fc_partner_bond_candidates_max = std::max<uint64_t>(
        fm.partner_bond_candidates.size(), fc_prof->fc_partner_bond_candidates_max);
    fc_prof->fc_partner_non_bond_checks_max =
        std::max(pnbc, fc_prof->fc_partner_non_bond_checks_max);
    int const pnbc_bucket = pnbc >= 6 ? 6 : static_cast<int>(pnbc);
    fc_prof->fc_partner_non_bond_checks_hist[pnbc_bucket]++;
    if ((fc_prof->fc_calls % kCmmFcProfileSampleEvery) == 0) {
      fc_sampled = true;
      fc_phase = static_cast<int>(fc_prof->sampled_calls % 5);
      fc_prof->sampled_calls++;
      fc_prof->phase_hits[fc_phase]++;
    }
  }
  auto phase_start = [&](int p) {
    if constexpr (kCmmFcProfile) {
      if (fc_sampled && fc_phase == p)
        fc_t0 = fc_clock::now();
    }
  };
  auto phase_stop = [&](int p) {
    if constexpr (kCmmFcProfile) {
      if (fc_sampled && fc_phase == p) {
        fc_prof->phase_ns[p] += fc_ns(fc_t0, fc_clock::now());
      }
    }
  };

  // ---- Phase 0: seed checks (type + non-bond).  Returns 0 on any reject. ----
  phase_start(0);

  const auto& seed = pool.molecule(seed_mol_id);
  if (!seed.active) {
    if constexpr (kCmmFcProfile)
      fc_prof->fc_inactive_seed++;
    phase_stop(0);
    return 0;
  }
  if (seed.type_index != fm.seed_type) {
    if constexpr (kCmmFcProfile)
      fc_prof->fc_seed_type_mismatches++;
    phase_stop(0);
    return 0;
  }
  int const n_seed = static_cast<int>(seed.comp_ids.size());

  // Non-endpoint checks on seed: each targets a unique mol-type-local (only
  // populated for non-symmetric sides with K>1).
  for (const auto& chk : fm.seed_non_bond_checks) {
    bool reject = false;
    if (chk.local_idx >= n_seed)
      reject = true;
    else {
      const auto& c = pool.component(seed.comp_ids[chk.local_idx]);
      // Three independent reject conditions (state mismatch / bond
      // present-but-required-absent / bond absent-but-required-present)
      // share the same `reject = true` body deliberately.
      // NOLINTBEGIN(bugprone-branch-clone)
      if (chk.state_req >= 0 && c.state_index != chk.state_req)
        reject = true;
      else if (chk.bond_req == 1 && c.bond_partner >= 0)
        reject = true;
      else if (chk.bond_req == 2 && c.bond_partner < 0)
        reject = true;
      // NOLINTEND(bugprone-branch-clone)
    }
    if (reject) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_seed_non_bond_rejects++;
      phase_stop(0);
      return 0;
    }
  }
  phase_stop(0);

  int total = 0;
  for (int const seed_ci : fm.seed_bond_candidates) {
    if constexpr (kCmmFcProfile)
      fc_prof->fc_candidate_iters++;

    // ---- Phase 1: partner trace (seed bond endpoint checks → partner type) ----
    phase_start(1);
    if (seed_ci >= n_seed) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_candidate_oob++;
      phase_stop(1);
      continue;
    }
    int const seed_bond_cid = seed.comp_ids[seed_ci];
    const auto& sc = pool.component(seed_bond_cid);
    if (fm.seed_bond_state_req >= 0 && sc.state_index != fm.seed_bond_state_req) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_candidate_state_rejects++;
      phase_stop(1);
      continue;
    }
    if (fm.seed_bond_bond_req == 1 && sc.bond_partner >= 0) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_candidate_bond_rejects++;
      phase_stop(1);
      continue;
    }
    if (fm.seed_bond_bond_req == 2 && sc.bond_partner < 0) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_candidate_bond_rejects++;
      phase_stop(1);
      continue;
    }
    // Bond tracing requires a partner regardless of pat bond_req.
    int const partner_cid = sc.bond_partner;
    if (partner_cid < 0) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_candidate_bond_rejects++;
      phase_stop(1);
      continue;
    }

    int const partner_mol_id = pool.mol_of_comp(partner_cid);
    if (partner_mol_id == seed_mol_id) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_candidate_self_bond++;
      phase_stop(1);
      continue;
    }
    const auto& partner = pool.molecule(partner_mol_id);
    if (!partner.active) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_partner_inactive++;
      phase_stop(1);
      continue;
    }
    if (partner.type_index != fm.partner_type) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_partner_type_mismatches++;
      phase_stop(1);
      continue;
    }
    phase_stop(1);

    // ---- Phase 2: partner local scan (find partner_cid's local idx) ----
    phase_start(2);
    int plocal = -1;
    int const n_partner = static_cast<int>(partner.comp_ids.size());
    for (int k = 0; k < n_partner; ++k) {
      if (partner.comp_ids[k] == partner_cid) {
        plocal = k;
        break;
      }
    }
    if (plocal < 0) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_plocal_not_found++;
      phase_stop(2);
      continue;
    }
    phase_stop(2);

    // ---- Phase 3: plocal_ok scan + partner bond-endpoint state check ----
    phase_start(3);
    bool plocal_ok = false;
    for (int const x : fm.partner_bond_candidates)
      if (x == plocal) {
        plocal_ok = true;
        break;
      }
    if (!plocal_ok) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_plocal_not_ok++;
      phase_stop(3);
      continue;
    }
    // Partner bond-endpoint dynamic state check.  (bond_req == 2 auto-ok
    // since partner_cid is on a bonded component.)
    if (fm.partner_bond_state_req >= 0 &&
        pool.component(partner_cid).state_index != fm.partner_bond_state_req) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_partner_state_rejects++;
      phase_stop(3);
      continue;
    }
    phase_stop(3);

    // ---- Phase 4: partner non-bond checks ----
    phase_start(4);
    bool ok = true;
    for (const auto& chk : fm.partner_non_bond_checks) {
      if (chk.local_idx >= n_partner) {
        ok = false;
        break;
      }
      if (chk.local_idx == plocal)
        continue; // bond endpoint already handled
      const auto& pc = pool.component(partner.comp_ids[chk.local_idx]);
      if (chk.state_req >= 0 && pc.state_index != chk.state_req) {
        ok = false;
        break;
      }
      if (chk.bond_req == 1 && pc.bond_partner >= 0) {
        ok = false;
        break;
      }
      if (chk.bond_req == 2 && pc.bond_partner < 0) {
        ok = false;
        break;
      }
    }
    phase_stop(4);
    if (!ok) {
      if constexpr (kCmmFcProfile)
        fc_prof->fc_partner_non_bond_rejects++;
      continue;
    }

    if constexpr (kCmmFcProfile)
      fc_prof->fc_total_matches++;
    ++total;
  }

  return total;
}

// Pattern signature for the obs-incr profiler's dedup audit.  Not a
// true canonical form — sufficient to flag obs that share substructure.
[[maybe_unused]] std::string pattern_signature(const Pattern& pat) {
  std::ostringstream os;
  for (size_t mi = 0; mi < pat.molecules.size(); ++mi) {
    if (mi)
      os << '.';
    auto& pm = pat.molecules[mi];
    os << pm.type_name << '(';
    for (size_t ci = 0; ci < pm.components.size(); ++ci) {
      if (ci)
        os << ',';
      auto& pc = pm.components[ci];
      os << pc.name << ':' << pc.required_state_index << ':' << static_cast<int>(pc.bond_constraint)
         << ':' << pc.bond_label;
    }
    os << ')';
  }
  os << '|';
  for (auto& b : pat.bonds)
    os << b.comp_flat_a << '-' << b.comp_flat_b << ';';
  return os.str();
}

// Species-observable incremental tracker.  Feature flag, not a profiler:
// extends the Molecules-only obs_mol_contrib path to Species observables
// via per-mol contribution + dirty-cx flush.
constexpr bool kSpeciesIncrObs = true;

// 2-mol-1-bond-fc obs-tracking specialization.  Feature flag, not a
// profiler: per-event obs recompute dispatches eligible patterns to
// count_2mol_1bond_fc instead of the generic BFS.
constexpr bool kObsFastMatch = true;

// Fast multi-molecule embedding count using pre-computed adjacency.
// When `fm` is non-null and fm->enabled is true, dispatches to the
// 2-mol-1-bond-fully-constrained specialized matcher.
int count_multi_mol_fast(const AgentPool& pool, int seed_mol_id, const Pattern& pat,
                         const Model& model, int pat_start, int pat_end, int seed_pat_idx,
                         const PatternAdj& pa, CountMultiProfile* cm_prof, CmmFcProfile* fc_prof,
                         const FastMatchSlot* fm = nullptr,
                         const std::vector<int>* reacting_local = nullptr);

// Generic body extracted so the dispatcher can call it for the invariant-gate
// comparison without recursing through the dispatch.
//
// `cm_prof` is the per-Engine CountMultiProfile destination — must be
// non-null; the call sites pass `&Engine::Impl::cm_profile`.  Always
// passed even when `kCountMultiProfile` is `false`, because the gate is
// `if constexpr` and the increment sites compile to nothing in default
// builds, so the parameter has no runtime cost there.
//
// `reacting_local`, when supplied, is forwarded to the seed-side
// count_embeddings_single call so that injective embeddings sending all
// reacting pattern components to the same molecule components collapse to
// one. This dedup is what differentiates a 2-mol unimolecular RULE rate
// (where embeddings differing only in non-reacting sym slots produce the
// same physical reaction) from a Molecules OBSERVABLE count (which keeps
// every embedding). Mirrors the dedup the single-mol path applies via
// count_embeddings_single's reacting_local argument.
int count_multi_mol_fast_generic(const AgentPool& pool, int seed_mol_id, const Pattern& pat,
                                 const Model& model, int pat_start, int pat_end, int seed_pat_idx,
                                 const PatternAdj& pa, CountMultiProfile* cm_prof,
                                 const std::vector<int>* reacting_local = nullptr) {

  // -- profiling scaffolding (gated) --
  using cm_clock = std::chrono::steady_clock;
  bool cm_sampled = false;
  cm_clock::time_point cm_t_total_start, cm_t_section;
  auto cm_dns = [](cm_clock::time_point a, cm_clock::time_point b) -> uint64_t {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  };
  auto cm_bucket = [](uint64_t v) -> int {
    if (v <= 1)
      return 0;
    if (v == 2)
      return 1;
    if (v <= 4)
      return 2;
    if (v <= 8)
      return 3;
    if (v <= 16)
      return 4;
    if (v <= 32)
      return 5;
    return 6;
  };
  if constexpr (kCountMultiProfile) {
    cm_prof->generic_calls++;
    cm_sampled = (cm_prof->generic_calls % kCountMultiProfileSampleEvery) == 0;
    if (cm_sampled) {
      cm_prof->sampled_calls++;
      cm_t_total_start = cm_clock::now();
      cm_t_section = cm_t_total_start;
    }
  }

  if (pat_end - pat_start <= 1) {
    if constexpr (kCountMultiProfile) {
      cm_prof->singleton_pattern_calls++;
      if (cm_sampled) {
        auto now = cm_clock::now();
        cm_prof->total_ns += cm_dns(cm_t_total_start, now);
      }
    }
    return count_embeddings_single(pool, seed_mol_id, pat.molecules[seed_pat_idx], model);
  }

  if constexpr (kCountMultiProfile) {
    int const npm = pat_end - pat_start;
    cm_prof->n_pat_mols_sum += static_cast<uint64_t>(npm);
    cm_prof->n_pat_mols_hist[cm_bucket(static_cast<uint64_t>(npm))]++;
  }

  std::vector<std::vector<int>> seed_embs;
  count_embeddings_single(pool, seed_mol_id, pat.molecules[seed_pat_idx], model, &seed_embs,
                          reacting_local);
  if constexpr (kCountMultiProfile) {
    if (cm_sampled) {
      auto now = cm_clock::now();
      cm_prof->seed_emb_ns += cm_dns(cm_t_section, now);
      cm_t_section = now;
    }
    auto nse = static_cast<uint64_t>(seed_embs.size());
    cm_prof->seed_emb_sum += nse;
    cm_prof->seed_emb_max = std::max(nse, cm_prof->seed_emb_max);
    cm_prof->seed_emb_hist[cm_bucket(nse)]++;
  }
  if (seed_embs.empty()) {
    if constexpr (kCountMultiProfile) {
      cm_prof->zero_seed_calls++;
      if (cm_sampled) {
        auto now = cm_clock::now();
        cm_prof->total_ns += cm_dns(cm_t_total_start, now);
      }
    }
    return 0;
  }

  int const n_pat_mols = static_cast<int>(pat.molecules.size());
  int total_count = 0;
  uint64_t bfs_visited_this_call = 0;
  bool entered_disjoint = false;

  for (auto& seed_comp_map : seed_embs) {
    std::vector<int> mol_assignments(n_pat_mols, -1);
    std::vector<std::vector<int>> comp_maps(n_pat_mols);
    mol_assignments[seed_pat_idx] = seed_mol_id;
    comp_maps[seed_pat_idx] = seed_comp_map;

    std::queue<int> bfs_queue;
    bfs_queue.push(seed_pat_idx);
    bool valid = true;

    while (!bfs_queue.empty() && valid) {
      int const cur_pat = bfs_queue.front();
      bfs_queue.pop();
      if constexpr (kCountMultiProfile)
        bfs_visited_this_call++;
      int const cur_actual = mol_assignments[cur_pat];
      auto& cur_comp_map = comp_maps[cur_pat];

      for (auto& ae : pa.adj[cur_pat]) {
        int const other_pat = ae.other_mol;

        // Follow the bond from the current molecule
        if (ae.my_local >= static_cast<int>(cur_comp_map.size())) {
          valid = false;
          break;
        }
        int const my_actual_comp_id = pool.molecule(cur_actual).comp_ids[cur_comp_map[ae.my_local]];
        int const partner_comp_id = pool.component(my_actual_comp_id).bond_partner;
        if (partner_comp_id < 0) {
          valid = false;
          break;
        }
        int const partner_mol_id = pool.mol_of_comp(partner_comp_id);

        if (mol_assignments[other_pat] >= 0) {
          // Already assigned — validate molecule and component consistency
          if (mol_assignments[other_pat] != partner_mol_id) {
            valid = false;
            break;
          }
          auto& other_comp_map = comp_maps[other_pat];
          if (ae.other_local >= static_cast<int>(other_comp_map.size())) {
            valid = false;
            break;
          }
          int const expected_local = other_comp_map[ae.other_local];
          int const expected_comp_id = pool.molecule(partner_mol_id).comp_ids[expected_local];
          if (expected_comp_id != partner_comp_id) {
            valid = false;
            break;
          }
          continue;
        }

        auto& other_pm = pat.molecules[other_pat];
        if (pool.molecule(partner_mol_id).type_index != other_pm.type_index) {
          valid = false;
          break;
        }

        std::vector<std::vector<int>> other_embs;
        count_embeddings_single(pool, partner_mol_id, other_pm, model, &other_embs);

        int partner_local = -1;
        for (int ci = 0; ci < static_cast<int>(pool.molecule(partner_mol_id).comp_ids.size());
             ++ci) {
          if (pool.molecule(partner_mol_id).comp_ids[ci] == partner_comp_id) {
            partner_local = ci;
            break;
          }
        }

        bool found = false;
        for (auto& emb : other_embs) {
          if (ae.other_local < static_cast<int>(emb.size()) &&
              emb[ae.other_local] == partner_local) {
            mol_assignments[other_pat] = partner_mol_id;
            comp_maps[other_pat] = emb;
            bfs_queue.push(other_pat);
            found = true;
            break;
          }
        }
        if (!found) {
          valid = false;
          break;
        }
      }
    }

    // bfs_ns span closes at end of BFS (before the disjoint path).  The
    // disjoint path has its own span.  Both are inside the seed_embs for-loop
    // so their sums over iterations read as "total BFS / total disjoint" for
    // this call.
    if constexpr (kCountMultiProfile) {
      if (cm_sampled) {
        auto now = cm_clock::now();
        cm_prof->bfs_ns += cm_dns(cm_t_section, now);
        cm_t_section = now;
      }
    }

    // Handle unassigned (disjoint) pattern molecules not reachable via bonds.
    if (valid) {
      std::vector<int> unassigned;
      for (int mi = pat_start; mi < pat_end; ++mi) {
        if (mol_assignments[mi] < 0)
          unassigned.push_back(mi);
      }

      if (!unassigned.empty()) {
        if constexpr (kCountMultiProfile)
          entered_disjoint = true;
        int const cx = pool.complex_of(seed_mol_id);
        auto cx_members = pool.molecules_in_complex(cx);

        std::function<void(int)> assign_unassigned = [&](int ui) {
          if (ui == static_cast<int>(unassigned.size())) {
            if (all_distinct_molecules(mol_assignments, pat_start, pat_end) &&
                check_pattern_bonds(pool, pat, mol_assignments, comp_maps))
              ++total_count;
            return;
          }
          int const pat_mi = unassigned[ui];
          auto& target_pm = pat.molecules[pat_mi];
          for (int const cand : cx_members) {
            if (!pool.molecule(cand).active)
              continue;
            if (pool.molecule(cand).type_index != target_pm.type_index)
              continue;
            bool already_used = false;
            for (int mi2 = pat_start; mi2 < pat_end; ++mi2) {
              if (mi2 != pat_mi && mol_assignments[mi2] == cand) {
                already_used = true;
                break;
              }
            }
            if (already_used)
              continue;

            std::vector<std::vector<int>> cand_embs;
            count_embeddings_single(pool, cand, target_pm, model, &cand_embs);
            for (auto& emb : cand_embs) {
              mol_assignments[pat_mi] = cand;
              comp_maps[pat_mi] = emb;
              assign_unassigned(ui + 1);
            }
            mol_assignments[pat_mi] = -1;
            comp_maps[pat_mi].clear();
          }
        };
        assign_unassigned(0);
        if constexpr (kCountMultiProfile) {
          if (cm_sampled) {
            auto now = cm_clock::now();
            cm_prof->disjoint_ns += cm_dns(cm_t_section, now);
            cm_t_section = now;
          }
        }
        continue;
      }
    }

    if (valid && all_distinct_molecules(mol_assignments, pat_start, pat_end))
      ++total_count;
  }
  if constexpr (kCountMultiProfile) {
    cm_prof->bfs_visited_sum += bfs_visited_this_call;
    cm_prof->bfs_visited_max = std::max(bfs_visited_this_call, cm_prof->bfs_visited_max);
    cm_prof->bfs_visited_hist[cm_bucket(bfs_visited_this_call)]++;
    if (entered_disjoint)
      cm_prof->disjoint_calls++;
    if (cm_sampled) {
      auto now = cm_clock::now();
      cm_prof->total_ns += cm_dns(cm_t_total_start, now);
    }
  }
  return total_count;
}

// Dispatcher: if a FastMatchSlot is enabled for this (rule, pattern-side),
// run the specialized 2-mol-1-bond-fc path; otherwise fall through to the
// generic BFS.  When kFastMatchInvariant is true, both paths run and are
// compared — any mismatch aborts with diagnostics so we catch
// specialization bugs before they silently corrupt propensities.
int count_multi_mol_fast(const AgentPool& pool, int seed_mol_id, const Pattern& pat,
                         const Model& model, int pat_start, int pat_end, int seed_pat_idx,
                         const PatternAdj& pa, CountMultiProfile* cm_prof, CmmFcProfile* fc_prof,
                         const FastMatchSlot* fm, const std::vector<int>* reacting_local) {
  if constexpr (kCountMultiProfile)
    cm_prof->calls++;
  if (fm && fm->enabled) {
    if constexpr (kCountMultiProfile)
      cm_prof->fm_hits++;
    int const specialized = count_2mol_1bond_fc(pool, seed_mol_id, *fm, fc_prof);
    if (kFastMatchInvariant) {
      int const generic =
          count_multi_mol_fast_generic(pool, seed_mol_id, pat, model, pat_start, pat_end,
                                       seed_pat_idx, pa, cm_prof, reacting_local);
      if (generic != specialized) {
        std::fprintf(stderr,
                     "[FastMatch mismatch] seed_mol=%d seed_type=%d partner_type=%d "
                     "generic=%d specialized=%d pat_start=%d pat_end=%d seed_pat=%d\n",
                     seed_mol_id, fm->seed_type, fm->partner_type, generic, specialized, pat_start,
                     pat_end, seed_pat_idx);
        std::abort();
      }
    }
    return specialized;
  }
  return count_multi_mol_fast_generic(pool, seed_mol_id, pat, model, pat_start, pat_end,
                                      seed_pat_idx, pa, cm_prof, reacting_local);
}

// Compute the diameter (max shortest-path distance between any two molecules)
// of a pattern's molecule bond graph.  Used to bound the BFS expansion depth
// in fire_rule().
int pattern_diameter(const Pattern& pat) {
  int const n = static_cast<int>(pat.molecules.size());
  if (n <= 1)
    return 0;

  // Build adjacency list from pattern bonds
  std::vector<std::vector<int>> adj(n);
  for (auto& bond : pat.bonds) {
    int mol_a = -1, mol_b = -1;
    int base = 0;
    for (int mi = 0; mi < n; ++mi) {
      int const nc = static_cast<int>(pat.molecules[mi].components.size());
      if (bond.comp_flat_a >= base && bond.comp_flat_a < base + nc)
        mol_a = mi;
      if (bond.comp_flat_b >= base && bond.comp_flat_b < base + nc)
        mol_b = mi;
      base += nc;
    }
    if (mol_a >= 0 && mol_b >= 0 && mol_a != mol_b) {
      adj[mol_a].push_back(mol_b);
      adj[mol_b].push_back(mol_a);
    }
  }

  // All-pairs BFS to find diameter
  int diameter = 0;
  for (int start = 0; start < n; ++start) {
    std::vector<int> dist(n, -1);
    dist[start] = 0;
    std::queue<int> q;
    q.push(start);
    while (!q.empty()) {
      int const cur = q.front();
      q.pop();
      for (int const nb : adj[cur]) {
        if (dist[nb] < 0) {
          dist[nb] = dist[cur] + 1;
          diameter = std::max(dist[nb], diameter);
          q.push(nb);
        }
      }
    }
  }
  return diameter;
}

// ===========================================================================
// Fenwick tree (Binary Indexed Tree) for O(log N) weighted sampling
// ===========================================================================

constexpr int FENWICK_THRESHOLD = 300;

struct FenwickTree {
  std::vector<double> tree; // 1-indexed
  int n = 0;

  FenwickTree() = default;

  void init(int size) {
    n = size;
    tree.assign(size + 1, 0.0);
  }

  void clear() {
    n = 0;
    tree.clear();
  }

  // Ensure capacity for index i (0-based)
  void ensure(int i) {
    if (i >= n) {
      int const new_n = std::max(i + 1, n * 2);
      tree.resize(new_n + 1, 0.0);
      n = new_n;
    }
  }

  void update(int i, double delta) {
    for (++i; i <= n; i += i & (-i))
      tree[i] += delta;
  }

  // Sum of all weights (full prefix sum at index n).
  double sum() const {
    double s = 0;
    for (int i = n; i > 0; i -= i & (-i))
      s += tree[i];
    return s;
  }

  // Largest power of 2 <= x.  Requires x > 0.  Portable bit_floor
  // (std::bit_floor is C++20 and we target C++17).
  static int highest_pow2(int x) {
#if defined(__GNUC__) || defined(__clang__)
    return 1 << (31 - __builtin_clz(static_cast<unsigned int>(x)));
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse(&idx, static_cast<unsigned long>(x));
    return 1 << idx;
#else
    int pw = 1;
    while ((pw << 1) > 0 && (pw << 1) <= x)
      pw <<= 1;
    return pw;
#endif
  }

  // Find smallest 0-based index where prefix sum > target.
  // Uses O(log N) bit-descent.
  int find(double target) const {
    if (n == 0)
      return 0;
    int pos = 0;
    for (int pw = highest_pow2(n); pw > 0; pw >>= 1) {
      if (pos + pw <= n && tree[pos + pw] <= target) {
        pos += pw;
        target -= tree[pos];
      }
    }
    return pos; // 0-indexed mol_id
  }
};

// ===========================================================================
// Rate computation structures
// ===========================================================================

// The per-(rule, molecule) fields EVERY rule's table carries, kept apart
// from the ones only some rule shapes ever touch (PerMolRuleAux below).
// The split is what a rule's table costs: a full rescan sizes the table by
// the molecule arena, so the row width is paid once per molecule in the
// pool per rule, in zeroing at every session build and in residency for
// the whole run.  A unimolecular, non-local rule with no pure-context slot
// reads two of the eleven fields, and used to be charged for all of them —
// 80 bytes a row against these 12 (issue #71).
struct PerMolRuleData {
  int count_a = 0; // embeddings of reactant pattern A seed molecule
  int count_b = 0; // embeddings of reactant pattern B seed molecule
  // P1 cache (step 2): set true after the first full recompute of this
  // (rule, molecule) entry, so incremental_update can tell "valid cached
  // value" from "default-initialized sentinel".  Bumped back to true on
  // every recompute; never reset to false during a run.
  bool cache_init = false;
};

// The per-(rule, molecule) fields a rule carries only if its shape uses
// them, in a table parallel to `RuleState::mol_data` that is left empty
// otherwise (`RuleState::needs_mol_aux`).  Three groups, and the shape
// predicate is the union of what writes each:
//   - the shared-component split, written only where the rule has two
//     reactant slots;
//   - the four local-rate fields, written only under a local rate law
//     (including the FunctionProduct B side);
//   - the pure-context complex ids, written only for a slot counted per
//     complex.
// A rule that needs any of them allocates all of them: the groups are
// small next to the count fields, and the shapes that want one commonly
// want another.
struct PerMolRuleAux {
  double a_only = 0;             // binding sites matching only A
  double b_only = 0;             // binding sites matching only B
  double ab_both = 0;            // binding sites matching both A and B
  double local_rate = 0.0;       // per-molecule rate (for local function rules)
  double local_propensity = 0.0; // count_a * local_rate
  // FunctionProduct (DOR2) B-side factor.  For a molecule matching reactant
  // pattern B, local_rate_b = f2(mol) and local_propensity_b = count_b * f2.
  // Unused (left 0) for ordinary local-rate and non-local rules.
  double local_rate_b = 0.0;
  double local_propensity_b = 0.0;
  // Complex this molecule is currently counted in by the pure-context tally
  // of reactant slot A / B (issue #33), or -1 when it is not in that tally at
  // all.  This is the tally's own record, deliberately NOT re-derived from
  // `pool.complex_of(mid)`: a molecule can change complex without this rule
  // ever recomputing its count, and the removal has to be applied to the
  // complex the hit was actually credited to.  Unused (left -1) for slots the
  // rule transforms.
  int percx_cx_a = -1;
  int percx_cx_b = -1;
};

// ---------------------------------------------------------------------------
// N-ary reactant support (issues #24, #26)
//
// The two-slot machinery above (a_*/b_* fields, compute_shared_components,
// the bimolecular sampler) is the hot path and stays exactly as it was.  A
// rule with three or more reactant patterns takes this separate, general
// path instead.
//
// Slots.  Slot i is reactant pattern i, spanning the pattern molecules
// `[reactant_pattern_starts[i], reactant_pattern_starts[i+1])`.  A slot may
// be a single molecule (`A(s)`) or a `.`-joined complex
// (`A(s,d!1).D(d!1)`), exactly as a bimolecular slot may (issue #26); the
// slot's *seed* is its first pattern molecule and `c_i(m)` below counts the
// embeddings of the whole sub-pattern whose seed lands on molecule `m`.
//
// Propensity.  With `n` reactant patterns and per-molecule embedding counts
// `c_i(m)` for slot i, mass action over *distinct* seed molecules is
//
//     D = Σ_{(m_0..m_{n-1}) all distinct} Π_i c_i(m_i)
//
// which expands over the partition lattice of {0..n-1} as
//
//     D = Σ_P μ(P) · Π_{B ∈ P} T_B ,   T_B = Σ_m Π_{i ∈ B} c_i(m)
//     μ(P) = Π_{B ∈ P} (−1)^(|B|−1) · (|B|−1)!
//
// For n = 2 this is the familiar `T_0·T_1 − T_{01}`; for n = 3 it is
// `T_0T_1T_2 − T_01T_2 − T_02T_1 − T_12T_0 + 2·T_012`.  With all slots
// identical and one embedding per molecule it collapses to the falling
// factorial N(N−1)…(N−n+1), so `k·symmetry_factor·D` reproduces the
// textbook k·N(N−1)(N−2)/6 for a trimolecular rule (BNG2 emits
// symmetry_factor = 1/n! for n identical patterns).
//
// The `T_B` are maintained incrementally over all 2^n − 1 non-empty
// subsets, so a per-molecule count change costs 2^n multiply-adds rather
// than a rescan.  `kMaxNarySlots` caps n so both that and the Bell(n)
// partition sum stay small (Bell(6) = 203).
//
// Sampling.  Each slot is drawn independently ∝ c_i(·) and the tuple is
// rejected unless all n *seed* molecules are distinct — which yields exactly
// the distribution ∝ Π c_i(m_i) conditioned on seed distinctness, the same
// measure `D` integrates.  For a rule of single-molecule slots that is the
// whole story: no null events are spent on coincidence and the SSA is exact.
//
// Distinctness with multi-molecule slots (issue #26).  Seed distinctness is
// no longer enough — a molecule pulled into slot i's complex may be the seed
// of slot j, or sit inside slot j's complex too, and firing on such a draw
// would consume one molecule twice.  `D` counts those tuples, so it is an
// upper bound on the true count `D_inj` (the same sum restricted to draws
// whose *whole* pattern-molecule assignment is injective).  The draw is
// therefore rejected as a null event unless the full assignment is
// injective, which realizes
//
//     D · k · sf · (D_inj / D)  =  D_inj · k · sf
//
// exactly — the standard inflated-propensity/null-event trick the
// bimolecular sampler already uses for same-molecule and same-complex
// draws.  The cost is wasted SSA cycles in proportion to how often the
// patterns actually overlap, which is rare (and rarer still under
// `-bscb`, which rejects same-complex seed pairs before this point).
// Deflating `D` to `D_inj` exactly would mean expanding over which
// *pattern molecules* coincide rather than which slots — combinatorially
// far worse, for a correction that changes nothing statistically.
// ---------------------------------------------------------------------------

constexpr int kMaxNarySlots = 6;

// One partition of the slot index set, pre-expanded at rule setup.
struct NaryPartition {
  double mu = 1.0;              // Möbius coefficient Π (−1)^(|B|−1)(|B|−1)!
  std::vector<uint32_t> blocks; // block membership bitmasks
};

// Per-complex tally for one pure-context reactant slot (issue #33).
//
// BioNetGen gives a reactant pattern the rule does not transform ONE reaction
// instance per matching complex, however many molecules inside that complex
// match: every embedding yields the identical reaction, so there is only one.
// Recomputing that count by walking the matched molecules and de-duplicating
// their complex ids is O(N) on the propensity hot path, and measurably so —
// 32x on `r08` (`DNA -> DNA + RNA`, `RNA -> RNA + Protein`), where the context
// pattern's population is the one that grows.  So it is maintained instead:
// `list.size()` is the count, delta-updated as molecules start and stop
// matching and as complexes merge and split.
//
// `rep` is the instance that actually fires.  The count and the draw have to
// agree: a bimolecular rule rejects draws that put both reactants in the same
// complex, and that rejection only offsets the propensity when the reactant is
// drawn the same way it is counted.  Counting per complex while drawing per
// molecule leaves a systematic bias wherever complexes hold unequal numbers of
// context matches (tests/bng_oracle/models/context_sampler.bngl pins it).
// Which instance is chosen is immaterial WHEN THE INSTANCES ALL PRICE THE
// SAME — the rule changes nothing about them — so `rep` is by default simply
// the lowest live matching molecule id, which makes it a function of the pool
// state rather than of event history and costs one integer compare to
// maintain.
//
// `rate_sum` is the DOR case: a pure-context pattern may still carry a local
// function tag, and then the collapsed instances DO price differently and
// `rep` also decides the rate.  Lowest-id is not good enough there.  A
// molecule id is a fact about pool history, so two chemically identical
// complexes assembled in different orders price differently and the same
// model answers differently depending on how its complexes were built —
// measured at 5.5x against BNG2 and 5.7x against RuleMonkey itself (issue
// #52).  BNG2 does arbitrate this, contrary to what issue #33 assumed: it
// expands the rule to a ConstantExpression obtained by evaluating the tagged
// observable at the species' canonically-FIRST matching molecule, which is a
// property of the species, so every copy of a species prices identically.
//
// So a slot that prices per molecule takes `rep` from the complex's canonical
// form instead (percx_canon_a/_b select it, percx_find_rep implements it).
// The cost is one canonicalization per REP CHANGE on those slots only; a slot
// whose instances all price the same — every rule with no local function, and
// every complex-wide (`local_arg_is_molecule == false`) tag, where the
// observable is evaluated over the whole complex and so reads the same at
// every molecule in it — keeps the integer compare and never reaches
// canonical.cpp.  That gate is what keeps the models this machinery exists
// for (`r08`, `machine`, `ensemble`) off the canonicalizer entirely.
//
// An n-ary rule leaves `rate_sum` at zero: nary_shape_supported refuses a
// local function outright, so its collapsed instances all price the same and
// its rep stays lowest-id.
struct PerCxTally {
  struct Entry {
    int hits = 0;          // matching molecules of this slot inside the complex
    int rep = -1;          // the matching molecule that prices and fires this
                           // complex: lowest live id, or the canonically-first
                           // when the slot prices per molecule
    int pos = -1;          // index into `list`
    double rep_rate = 0.0; // rep's local rate, as currently summed into rate_sum
  };
  std::unordered_map<int, Entry> entries;
  std::vector<int> list; // complexes with hits > 0, dense, for O(1) uniform draw
  double rate_sum = 0.0; // Σ rep_rate over `list` (local-rate rules only)

  int size() const { return static_cast<int>(list.size()); }
  void clear() {
    entries.clear();
    list.clear();
    rate_sum = 0.0;
  }
};

// Per-rule state for a rule with >= 3 reactant patterns.  Left disabled
// (and unallocated) for every uni- and bimolecular rule.
struct NaryState {
  bool enabled = false;
  int n_slots = 0;

  // Per-slot embedding counts, [slot][mol_id].  Slot i is reactant pattern
  // i, spanning pattern molecules [slot_start[i], slot_end[i]); its seed is
  // `slot_start[i]` and counts[i][m] is the number of embeddings of the
  // whole sub-pattern that place that seed on molecule m.
  std::vector<std::vector<int>> counts;
  std::vector<int> slot_start;
  std::vector<int> slot_end;
  std::vector<char> slot_multi;                 // span > 1: a `.`-joined complex
  std::vector<PatternAdj> slot_adj;             // bond adjacency, multi-molecule slots
  bool any_multi = false;                       // any slot spans > 1 molecule
  std::vector<int> slot_type_index;             // molecule type of the seed, per slot
  std::vector<std::vector<int>> reacting_local; // per-slot reacting components

  // Pure-context slots, counted once per matching COMPLEX (issue #42).
  //
  // The two-slot path substitutes the complex count for the per-molecule
  // total in compute_propensity (issue #33).  There is nothing to substitute
  // into here: the propensity is the distinct-tuple sum
  //
  //   D = Σ_{(m_0..m_{n-1}) all distinct} Π_i c_i(m_i)
  //     = Σ_P μ(P) · Π_{B ∈ P} T_B ,   T_B = Σ_m Π_{i ∈ B} c_i(m)
  //
  // whose every term is defined over molecules — T_B couples slots AT the
  // same molecule, and the distinctness the expansion enforces is
  // distinctness of seed molecules.
  //
  // So the count moves instead of the sum.  Per-complex counting IS
  // per-molecule counting with the slot's count function replaced by the
  // indicator of one designated representative molecule per matching complex:
  //
  //   c_j(m) = 1 if m represents its complex for slot j, else 0
  //
  // Then Σ_m c_j(m) is the number of matching complexes, which is the count
  // BNG's network expansion gives that slot, while D stays a molecule-level
  // sum and the partition-lattice expansion, the seed-distinctness retry, the
  // full-assignment injectivity check and the per-slot sampler all keep
  // working unchanged.  The sampler in particular needs no per-complex case
  // at all: drawing ∝ c_j(m) already draws a matching complex uniformly and
  // fires its representative, which is exactly the count/draw agreement
  // issue #33 had to add by hand on the two-slot path.
  //
  // `raw` is what the tally is built from — the slot's actual embedding count
  // per molecule, which decides membership — and `counts` holds the indicator
  // derived from it.  They are separate vectors because the tally has to be
  // able to say "this molecule still matches, it just is not the
  // representative any more".
  std::vector<char> slot_context;      // 1 = the rule transforms nothing in this slot
  bool any_context = false;            // any slot_context set
  std::vector<PerCxTally> percx;       // per context slot; unused entries stay empty
  std::vector<std::vector<int>> raw;   // per context slot, embeddings per molecule
  std::vector<std::vector<int>> cx_of; // per context slot, complex credited, or -1

  // T_B for every subset mask in [0, 1<<n_slots).  Index 0 is unused.
  std::vector<double> subset_totals;
  std::vector<NaryPartition> partitions;

  // O(log N) samplers, one per slot, for slot types above FENWICK_THRESHOLD.
  std::vector<FenwickTree> fenwick;
  std::vector<char> use_fenwick;   // vector<bool> is a bitfield; char is plainer
  std::vector<double> slot_totals; // Σ_m c_i(m), == subset_totals[1<<i]

  int count_for(int slot, int mid) const {
    const auto& cs = counts[slot];
    return (mid >= 0 && mid < static_cast<int>(cs.size())) ? cs[mid] : 0;
  }
};

// Population count of a block mask.  std::popcount is C++20 and we target
// C++17.  Unlike highest_pow2 below — a Fenwick-descent inner step, where
// the per-compiler intrinsics earn their keep — this runs once per partition
// at rule setup over a mask of at most kMaxNarySlots bits, so the portable
// Kernighan loop costs nothing and sidesteps the intrinsic spellings
// entirely (__builtin_popcount is GCC/Clang-only, and MSVC's __popcnt is
// x86-only and needs a POPCNT-capable target).
inline int popcount_mask(uint32_t v) {
  int c = 0;
  for (; v != 0; v &= v - 1)
    ++c;
  return c;
}

// Enumerate the partitions of {0..n-1} with their Möbius coefficients.
// Restricted-growth-string enumeration: element 0 always opens block 0, and
// element i may join any existing block or open the next one.
std::vector<NaryPartition> build_nary_partitions(int n) {
  std::vector<NaryPartition> out;
  std::vector<int> block_of(n, 0);

  std::function<void(int, int)> rec = [&](int idx, int n_blocks) {
    if (idx == n) {
      NaryPartition p;
      p.blocks.assign(n_blocks, 0);
      for (int i = 0; i < n; ++i)
        p.blocks[block_of[i]] |= (uint32_t{1} << i);
      p.mu = 1.0;
      for (uint32_t const b : p.blocks) {
        int const size = popcount_mask(b);
        // (−1)^(size−1) · (size−1)!
        double term = ((size - 1) % 2 == 0) ? 1.0 : -1.0;
        for (int f = 2; f <= size - 1; ++f)
          term *= f;
        p.mu *= term;
      }
      out.push_back(std::move(p));
      return;
    }
    for (int b = 0; b < n_blocks; ++b) {
      block_of[idx] = b;
      rec(idx + 1, n_blocks);
    }
    block_of[idx] = n_blocks;
    rec(idx + 1, n_blocks + 1);
  };

  rec(1, 1); // element 0 is pinned to block 0
  return out;
}

// Is every molecule of the pattern sub-range [start, end) reachable from
// `start` along the pattern's own bonds?  A `.`-joined reactant whose
// molecules are *not* bonded to each other — `A(x).B(y)` with no shared bond
// label — means "these molecules, anywhere in the same complex".  The n-ary
// sampler places a slot's non-seed molecules by walking bonds out of the
// seed (resolve_pattern_context), so it cannot realize such a draw, while
// the count would still be non-zero: the rule would hold propensity and
// null-event every firing.  That is the silent inertness of #24 again, so
// the shape is refused rather than run.
bool nary_slot_connected(const Pattern& pat, int start, int end) {
  if (end - start <= 1)
    return true;
  PatternAdj const pa = build_pattern_adjacency(pat, start, end);
  std::vector<char> seen(pat.molecules.size(), 0);
  std::queue<int> q;
  seen[start] = 1;
  q.push(start);
  while (!q.empty()) {
    int const cur = q.front();
    q.pop();
    for (const auto& e : pa.adj[cur]) {
      if (e.other_mol >= start && e.other_mol < end && seen[e.other_mol] == 0) {
        seen[e.other_mol] = 1;
        q.push(e.other_mol);
      }
    }
  }
  for (int mi = start; mi < end; ++mi)
    if (seen[mi] == 0)
      return false;
  return true;
}

// Is this >= 3-pattern rule a shape the n-ary path implements?  Kept as one
// predicate so the engine gate and the load-time refusal in
// simulator.cpp::scan_unsupported cannot drift apart: anything this rejects
// must be refused there, or it would silently never fire again.
bool nary_shape_supported(const Rule& rule) {
  int const n = rule.molecularity;
  if (n < 3 || n > kMaxNarySlots)
    return false;
  if (static_cast<int>(rule.reactant_pattern_starts.size()) != n)
    return false;

  // A reactant pattern may be a single molecule or a bonded complex
  // (issue #26), but it has to be one connected piece — see above.
  int const n_mols = static_cast<int>(rule.reactant_pattern.molecules.size());
  for (int slot = 0; slot < n; ++slot) {
    int const start = rule.reactant_pattern_starts[slot];
    int const end = (slot + 1 < n) ? rule.reactant_pattern_starts[slot + 1] : n_mols;
    if (start < 0 || start >= n_mols || end <= start || end > n_mols)
      return false;
    if (!nary_slot_connected(rule.reactant_pattern, start, end))
      return false;
  }

  // Rate laws defined only on 1-2 reactant slots.
  if (rule.rate_law.type != RateLawType::Ele)
    return false;
  if (rule.rate_law.is_local)
    return false;

  return true;
}

struct RuleState {
  std::vector<PerMolRuleData> mol_data; // indexed by mol_id
  // The wide half of the per-molecule row, for the rules whose shape reads
  // it (see PerMolRuleAux).  Either empty, or exactly as long as `mol_data`
  // and indexed by the same mol_id — every site that grows one grows the
  // other.  `needs_mol_aux` is the shape predicate, settled by
  // rescan_all_molecules_for_rule before it sizes either table.
  std::vector<PerMolRuleAux> mol_aux;
  bool needs_mol_aux = false;
  NaryState nary; // >= 3 reactant patterns (issue #24)
  double a_total = 0;
  double b_total = 0;
  double a_only_total = 0;
  double b_only_total = 0;
  double ab_both_total = 0;
  // Σ ab_both(m)² over molecules.  Tracked only for `same_components` rules
  // so the homodimer propensity formula can deflate `ab²/2` to the analytic
  // `(ab² − ab_sq)/2`, removing the null-event self-pair waste that the
  // sampler previously paid via unconditional `mol_a == mol_b` rejection.
  // Other rules leave it at 0 and skip the bookkeeping.
  double ab_both_sq_total = 0;
  double propensity = 0;
  bool has_local_rates = false;        // rule uses local functions
  double local_propensity_total = 0.0; // sum of per-mol local_propensity (S1, A side)
  // FunctionProduct (DOR2): sum of per-mol local_propensity_b over reactant-B
  // matches (S2, B side).  Realized propensity is S1·S2.  0 for other rules.
  double local_propensity_b_total = 0.0;
  double embedding_correction_a = 1.0; // overcounting correction for pattern A
  double embedding_correction_b = 1.0; // overcounting correction for pattern B
  // Reactant pattern A / B is pure context — no operation in this rule touches
  // it — so it is counted once per matching complex rather than once per
  // matching molecule (issue #33).  See PerCxTally.
  bool per_complex_a = false;
  bool per_complex_b = false;
  // …and that collapsed instance also has to be PRICED, at one of the
  // molecules the slot collapsed.  Set when the choice is observable — the
  // slot carries a per-molecule local function tag, so its matching molecules
  // inside one complex can price differently — and then the representative
  // comes from the complex's canonical form rather than from id order (issue
  // #52).  False everywhere else, which is where the cheap path stays.
  bool percx_canon_a = false;
  bool percx_canon_b = false;
  PerCxTally percx_a;
  PerCxTally percx_b;
  FenwickTree fenwick_a;      // O(log N) sampler for slot A (if active)
  FenwickTree fenwick_b;      // O(log N) sampler for slot B (if active)
  bool use_fenwick_a = false; // true if type population > FENWICK_THRESHOLD
  bool use_fenwick_b = false;
  bool use_multi_mol_count = false;     // true for multi-mol reactant A has >1 molecule
  bool use_multi_mol_count_b = false;   // true for multi-mol bimolecular reactant B
  bool needs_complex_expansion = false; // true if multi-mol pattern has disjoint molecules

  PatternAdj pat_adj_a; // pre-computed adjacency for reactant A's sub-pattern
  PatternAdj pat_adj_b; // pre-computed adjacency for reactant B's sub-pattern

  // 2-mol-1-bond-fully-constrained fast path (Candidate B).  Populated at
  // model load when the reactant pattern matches the template; otherwise
  // enabled=false and cmm falls through to the generic path.
  FastMatchSlot fm_a;
  FastMatchSlot fm_b;
  // Local (within-seed-molecule) indices of pattern components that are
  // operated on by the rule.  Used to deduplicate injective embeddings that
  // differ only in which non-reacting pattern components target which
  // molecule components.  See count_embeddings_single's reacting_local param.
  std::vector<int> reacting_local_a;
  std::vector<int> reacting_local_b;

  // Relevant-component bitmask (P1 embedding cache, step 1).  Bit i is set
  // iff component i of the A-seed (or B-seed) molecule type is referenced
  // by this rule's reactant pattern.  A component is "referenced" if it
  // appears in the pattern with any state or bond constraint, or if it is
  // a bond endpoint in a multi-mol pattern.  A molecule-side state or bond
  // change that touches only components outside this mask cannot alter
  // count_a (resp. count_b) for this rule on that molecule.  Populated at
  // model-load; read by incremental_update's cache-hit check (step 2+).
  // a_mask_complete/b_mask_complete is false iff the seed type has >64
  // components (bitmask overflow) — in that case the cache must treat every
  // change as relevant.
  uint64_t a_relevant_mask = 0;
  uint64_t b_relevant_mask = 0;
  bool a_mask_complete = false;

  // First-occurrence latch for the negative-propensity clamp diagnostic
  // in set_rule_propensity.  Once a rule's rate has evaluated negative
  // and been clamped to 0 once, further negative evaluations are silent
  // (otherwise an oscillator with a state-dependent rate would spam the
  // log).  Default-constructed false; never reset, so the warning fires
  // at most once per rule per simulator lifetime.
  bool clamp_warned = false;
  bool b_mask_complete = false;
};

// ---------------------------------------------------------------------------
// Per-rule side-table residency (issue #71).
//
// Every per-molecule table a rule owns is sized by the molecule ARENA and
// not by the population the rule can actually see, so what a model holds in
// them is a (rule count x arena) product rather than a function of what the
// rules match.  Four kinds, all of them indexed by mol_id:
//
//   * `mol_data`, carried by every rule with a reactant seed molecule;
//   * `mol_aux`, carried by the shapes that read one (see PerMolRuleAux);
//   * an n-ary rule's per-slot `counts`, plus `raw` / `cx_of` on its
//     context slots;
//   * a Fenwick tree per slot whose type population is over threshold,
//     `init(pool.molecule_count())` on a sampler only ever updated at the
//     ids of one type.
//
// #72 narrowed the row (80 bytes to 12, with the wide half allocated only
// where it is read); the SIZING is what is left of #71, and the case for
// changing it is residency on real models rather than on a synthetic
// scale-up.  This is the accounting that measures it -- reported per rule
// and per model by the `[RM tables]` block at the end of a run (gated on
// RM_PRINT_TABLES), which is what `harness/rule_table_footprint.py` sweeps
// the corpora with.
//
// `size()`, not `capacity()`.  Every row in [0, size) has been written by
// the `assign` that sized the table or the `resize` that grew it, so it is
// resident; the tail up to `capacity()` is address space the geometric
// growth of `resize(mid + 1)` reserved and nothing has touched, which is
// not.  Counting capacity put several corpus models over 100% of their own
// peak RSS, which is the tell.  The consequence is that this UNDERSTATES
// what the allocator holds -- a table that grew has slack, and a `clear()`
// on a Fenwick whose type fell back below FENWICK_THRESHOLD returns
// nothing -- so it is a floor on the residency, which is the right side to
// err on for an argument that the tables are too big.
//
// The per-complex tallies (`PerCxTally`) are deliberately not counted: they
// are sized by matching complexes, so they are not part of the arena
// product this exists to measure, and their hash-map storage has no
// equivalent of `size() * sizeof(row)` to report.
// ---------------------------------------------------------------------------
struct RuleTableBytes {
  std::size_t mol = 0;     // per-molecule rows (mol_data, mol_aux, n-ary slots)
  std::size_t sampler = 0; // Fenwick trees
  std::size_t rows = 0;    // longest per-molecule table, in entries
  std::size_t total() const { return mol + sampler; }
};

// `cap_rows`, when non-zero, caps every table at that many rows instead of
// counting the rows it has: what the same rule would hold if its tables
// were sized by something other than the arena.  A table that does not
// exist stays at zero either way -- the cap shortens tables, it does not
// conjure them.
RuleTableBytes rule_table_bytes(const RuleState& rs, std::size_t cap_rows = 0) {
  RuleTableBytes b;
  auto capped = [cap_rows](std::size_t n) { return (cap_rows != 0) ? std::min(n, cap_rows) : n; };
  auto add_rows = [&b, &capped](std::size_t n, std::size_t width) {
    b.mol += capped(n) * width;
    b.rows = std::max(b.rows, capped(n));
  };
  auto add_tree = [&b, &capped](std::size_t n) {
    // A Fenwick over k rows is k + 1 doubles; it is 1-indexed.
    b.sampler += (n == 0) ? 0 : std::min(n, capped(n - 1) + 1) * sizeof(double);
  };
  add_rows(rs.mol_data.size(), sizeof(PerMolRuleData));
  add_rows(rs.mol_aux.size(), sizeof(PerMolRuleAux));
  for (const auto& c : rs.nary.counts)
    add_rows(c.size(), sizeof(int));
  for (const auto& c : rs.nary.raw)
    add_rows(c.size(), sizeof(int));
  for (const auto& c : rs.nary.cx_of)
    add_rows(c.size(), sizeof(int));
  add_tree(rs.fenwick_a.tree.size());
  add_tree(rs.fenwick_b.tree.size());
  for (const auto& f : rs.nary.fenwick)
    add_tree(f.tree.size());
  return b;
}

// One past the highest live molecule id this rule's tables can be indexed
// at.  The arena is what they are SIZED by; this is what they could be
// sized by instead without any new pool-side machinery, since every read of
// every one of them either bounds-checks against the table's length or
// grows it first -- which is the audit #68 relied on to leave a seedless
// rule's table empty outright.
//
// It is also the shortcut #71 names and warns about, and reporting it
// beside the row count is how that warning gets a number: a rule whose seed
// type happens to sit at the top of the arena reaches all of it and saves
// nothing, and after #62 made type-index removal a swap-with-back, a
// long-running model's ids of any one type have no reason to stay low.
// Whether real models leave a rule short of the arena is a measurement.
//
// Derived here rather than recorded during the rescan, so that it is a
// snapshot of the same instant as the row counts it prints next to.
int rule_table_reach(const AgentPool& pool, const Rule& rule, const RuleState& rs) {
  auto type_reach = [&pool](int type_index) {
    int r = 0;
    if (type_index < 0)
      return r;
    for (int const mid : pool.molecules_of_type(type_index))
      if (pool.molecule(mid).active && mid + 1 > r)
        r = mid + 1;
    return r;
  };
  if (rs.nary.enabled) {
    int r = 0;
    for (int const t : rs.nary.slot_type_index)
      r = std::max(r, type_reach(t));
    return r;
  }
  const auto& mols = rule.reactant_pattern.molecules;
  int const seed_a = rule.reactant_pattern_starts.empty() ? 0 : rule.reactant_pattern_starts[0];
  if (seed_a >= static_cast<int>(mols.size()))
    return 0; // no seed molecule, so no table at all (#68)
  int r = type_reach(mols[seed_a].type_index);
  if (rule.molecularity >= 2 && rule.reactant_pattern_starts.size() > 1) {
    int const seed_b = rule.reactant_pattern_starts[1];
    if (seed_b >= 0 && seed_b < static_cast<int>(mols.size()))
      r = std::max(r, type_reach(mols[seed_b].type_index));
  }
  return r;
}

// Distinct-tuple sum D over the n reactant slots (issue #24).  See the
// NaryState comment for the partition-lattice expansion this evaluates.
double nary_distinct_sum(const NaryState& ns) {
  double d = 0;
  for (const auto& p : ns.partitions) {
    double term = p.mu;
    for (uint32_t const b : p.blocks) {
      term *= ns.subset_totals[b];
      if (term == 0)
        break;
    }
    d += term;
  }
  // Cancellation between partition terms can leave a tiny negative residue
  // where the true value is zero (e.g. two molecules, three slots).  A
  // negative propensity is meaningless and would poison the SSA draw.
  return (d > 0) ? d : 0.0;
}

// Reaction-center symmetry factor for the local-rate propensity paths.
//
// `compute_propensity` multiplies BNG2's `symmetry_factor` into every rate
// law it handles, but the local-function and FunctionProduct paths compute
// their propensity directly from Σ w·f(mol) and used to skip it.  A
// symmetric rule then ran at `1/symmetry_factor` times its intended rate —
// 2x for a homodimer.
//
// Stock NFsim has the identical defect, and for the identical reason: every
// rate law except the ones routing through `setBaseRate()` is constructed
// with `baseRate=1` and never recovers the factor.  bngsim's vendored NFsim
// patches it in `ReactionClass::ReactionClass` (lanl/bngsim#195, fixed in
// #278) — the patch comment names "global function, local function (DOR),
// function product, MM" as the affected set.  BNG2.pl's network expansion
// has always applied it.  RM follows BNG2.
//
// The homodimer branch of compute_propensity deliberately does NOT apply it
// (its `/2` already is the 0.5, and multiplying would double-count).  The
// local paths have no such implicit halving: for a DOR pair the propensity
// is the *ordered* distinct-pair sum, and 0.5 converts it to unordered.  The
// MM branch is the third case: it applies the factor to the count of the
// reactant the rule transforms, inside the law rather than to the finished
// propensity, because the law is not linear in that count (see there).
double dor_symmetry(const Rule& rule) { return rule.symmetry_factor; }

// Compute propensity for a rule given its accumulated state.
// embedding_correction_a/b correct for overcounting due to permutations
// of identical non-reacting components in the pattern.
double compute_propensity(const RuleState& rs, const Rule& rule, double rate, double a_cx = -1.0,
                          double b_cx = -1.0) {
  double const ca = rs.embedding_correction_a;
  double const cb = rs.embedding_correction_b;
  // Pure-context reactant patterns count complexes, not molecules (issue #33).
  double const a_used = (a_cx >= 0.0) ? a_cx : (rs.a_total / ca);
  double const b_used = (b_cx >= 0.0) ? b_cx : (rs.b_total / cb);

  if (rule.molecularity == 0) {
    // Zero-order synthesis: propensity is just the rate
    return rate;
  }

  // Three or more reactant patterns (issue #24).  Mass action over distinct
  // molecule tuples; symmetry_factor carries BNG2's 1/n! for n identical
  // patterns, exactly as it carries the 1/2 in the homodimer case below.
  if (rs.nary.enabled) {
    if (rule.rate_law.is_total_rate) {
      // The rate function already returns the whole propensity; it still
      // has to vanish when any reactant is absent.
      for (double const t : rs.nary.slot_totals)
        if (t <= 0)
          return 0;
      return rate;
    }
    return nary_distinct_sum(rs.nary) * rate * rule.symmetry_factor;
  }

  // MM(kcat, Km) — Michaelis-Menten with quasi-steady-state for free
  // substrate.  Mirrors NFsim's MMRxnClass::update_a (reaction.cpp):
  //   S = corrected substrate count   (reactant 0)
  //   E = corrected enzyme count      (reactant 1)
  //   sFree = 0.5 * ((S-Km-E) + sqrt((S-Km-E)^2 + 4*Km*S))
  //   a     = kcat * sFree * E / (Km + sFree)
  // NFsim requires exactly 2 reactants; ditto here.
  //
  // symmetry_factor scales one of the two COUNTS, inside the law — not the
  // finished propensity (issue #37).  What it corrects is a match
  // multiplicity: a pattern with a non-trivial reaction-center automorphism
  // matches each complex more than once (2x for the symmetric dimer
  // S(d!1).S(d!1)), so the law would otherwise be handed more of that
  // reactant than exists.  Michaelis-Menten is not linear in either count, so
  // scaling the finished propensity is exact only below saturation, where the
  // two agree; scaling the count is exact everywhere and reproduces BNG2's
  // network expansion, which builds the reaction from species counts and
  // folds the factor into the reaction multiplicity.
  //
  // Which count: the one the rule TRANSFORMS.  BNG2 refuses an MM rule whose
  // two reactant patterns are isomorphic ("requires non-identical
  // reactants"), so no automorphism can carry one pattern onto the other and
  // the reactant automorphism group is the direct product of the two
  // patterns' own.  A pure-context pattern contributes 1 — all of its
  // automorphisms stabilize the (empty) reaction center — and is already
  // counted per complex by the issue-#33 tally, so applying the factor to it
  // would deflate a count that never was inflated.  So the factor lands on
  // the transformed slot, and after it the law reads complex counts on both
  // sides, which is exactly what BNG2's ODE integrates.
  //
  // For the canonical MM shape (enzyme is a catalyst, so pure context) that
  // slot is the substrate.  A rule that transforms its enzyme instead is
  // off-spec for MM — the QSS derivation assumes a conserved enzyme — but
  // BNG2 writes the XML for it and puts the factor there, so honour it.  If
  // BOTH are transformed the scalar is a product of the two patterns' factors
  // and cannot be split; the substrate is the better guess and is what the
  // canonical shape needs.  (cf. lanl/bngsim#281, about NFsim over-counting a
  // symmetric CONTEXT match — a separate defect RM does not share since #33.)
  if (rule.rate_law.type == RateLawType::MM) {
    if (rule.molecularity != 2)
      return 0;
    bool const sym_on_enzyme = rs.per_complex_a && !rs.per_complex_b;
    double const S = a_used * (sym_on_enzyme ? 1.0 : rule.symmetry_factor);
    double const E = b_used * (sym_on_enzyme ? rule.symmetry_factor : 1.0);
    if (S <= 0 || E <= 0)
      return 0;
    double const kcat = rule.rate_law.mm_kcat;
    double const Km = rule.rate_law.mm_Km;

    // Km <= 0 is outside the law as written (issue #46).
    //
    // Km == 0 is a removable singularity, not a zero.  With Km = 0,
    // sFree = max(S-E, 0), so for S > E the law already reads kcat*E, while
    // for S <= E it reads 0/0 — and the limit there is NOT zero:
    //
    //   sFree -> Km*S/(E-S)  as Km -> 0+   =>   a -> kcat*S
    //
    // i.e. a = kcat*min(S,E) on the whole Km = 0 line, which is the physical
    // reading (binding infinitely tight: every substrate molecule is bound,
    // so turnover is substrate-limited when the enzyme is in excess and
    // enzyme-limited otherwise).  Returning 0 instead left the rule silently
    // inert, and froze a Km = 0 model that started at S > E the moment its
    // substrate decayed to S == E.
    //
    // Km < 0 has no reading at all — the discriminant can go negative, and
    // where it does not the expression yields a finite but meaningless rate.
    // Answer NaN so set_rule_propensity clamps it to zero and says so once;
    // a statically negative Km is additionally refused at load.
    if (Km <= 0) {
      if (Km < 0)
        return std::numeric_limits<double>::quiet_NaN();
      return kcat * std::min(S, E);
    }

    // sFree is the positive root of sFree^2 + (Km + E - S)*sFree - Km*S = 0.
    // Taking it as 0.5*(diff + sqrt(diff^2 + 4*Km*S)) cancels catastrophically
    // when diff < 0 — the regime Km*S << (E - S)^2, i.e. exactly the approach
    // to the Km = 0 limit above.  Measured on S=100, E=200, kcat=1, where the
    // limit is 100: the naive form reads 100.093 at Km=1e-12, 117.392 at
    // 1e-14, and 0 from ~1e-15 down.  The conjugate form
    // 2*Km*S/(q - diff) is algebraically identical (their product of factors
    // is q^2 - diff^2 = 4*Km*S) and adds like-signed quantities, so it holds
    // the limit exactly.  diff >= 0 keeps the direct form, which is the
    // well-conditioned branch there and stays finite as Km -> 0.
    double const diff = S - Km - E;
    double const q = std::sqrt((diff * diff) + (4.0 * Km * S));
    double const sFree = (diff >= 0) ? (0.5 * (diff + q)) : ((2.0 * Km * S) / (q - diff));
    if (Km + sFree <= 0)
      return 0;
    return kcat * sFree * E / (Km + sFree);
  }

  // totalrate="1": the rate function already returns the total propensity
  // (e.g., kf * Observable), so don't multiply by population counts.
  // However, if any reactant population is zero, propensity must be zero
  // (matching NFsim's FunctionalRxnClass / BasicRxnClass behavior).
  if (rule.rate_law.is_total_rate) {
    if (rule.molecularity <= 1) {
      if (a_used <= 0)
        return 0;
    } else {
      if (a_used <= 0 || b_used <= 0)
        return 0;
    }
    return rate;
  }

  if (rule.molecularity <= 1) {
    // Unimolecular: multiply by symmetry_factor to correct for identical-
    // molecule permutations in multi-molecule patterns (e.g., M(a!1).M(a!1)
    // has 2 seed embeddings per dimer; sf=0.5 gives propensity = rate per dimer).
    return a_used * rate * rule.symmetry_factor;
  }

  // Bimolecular — apply corrections to the per-pattern totals.
  double const a_eff = a_used;
  double const b_eff = b_used;
  if (!rule.same_components) {
    // Heterodimer (A+B different types, or A+A with different bond-target
    // component names).  symmetry_factor is 1 in BNGL.  After the sampler
    // rejects mol_a==mol_b cases (no-op for true heterodimer; ~1/N for
    // same-type asymmetric self-binding), effective rate = N(N-1)*k.
    return (a_eff * b_eff) * rate * rule.symmetry_factor;
  } // Homodimer (A+A, same type and same bond-target component).  The
  // analytic propensity is `Σ_{m1<m2} ab(m1)·ab(m2) · k`, which equals
  // `(ab² − Σ ab(m)²) / 2 · k` — i.e., the ordered self-pair contribution
  // `Σ ab(m)²` is subtracted before halving.  We track `ab_both_sq_total`
  // (the sum of squares) incrementally so this deflated form is exact and
  // the bimolecular sampler does NOT need to null-event self-pairs; it
  // retries until `mol_a != mol_b` instead, saving SSA cycles at small N.
  // Symmetry_factor=0.5 emitted by BNG2 is already implicit in the /2 —
  // don't multiply it in (would be a double 0.5).
  double const ao = rs.a_only_total / ca;
  double const bo = rs.b_only_total / cb;
  double const ab = rs.ab_both_total / ca; // ab_both counted from A perspective
  double const ab_sq = rs.ab_both_sq_total / (ca * ca);
  return ((ao * b_eff) + (ab * bo) + (((ab * ab) - ab_sq) / 2.0)) * rate;
}

// ===========================================================================
// Shared component analysis for bimolecular overcounting
// ===========================================================================

// For a bimolecular rule, determine which binding sites on a molecule
// match reactant pattern A only, B only, or both.
// The "binding site" is the component that gets bound by the AddBond operation.
void compute_shared_components(const AgentPool& pool, int mol_id, const Rule& rule,
                               const Model& model, int seed_mol_a,
                               int seed_mol_b, // pattern molecule indices for each reactant
                               int bind_comp_local_a,
                               int bind_comp_local_b, // local comp idx in the BIND molecule
                               int bind_mol_a,
                               int bind_mol_b, // pattern molecule index where AddBond occurs
                               double& out_a_only, double& out_b_only, double& out_ab_both) {

  // Get all embeddings of seed_mol_a in this molecule
  auto& pm_a = rule.reactant_pattern.molecules[seed_mol_a];
  std::vector<std::vector<int>> embs_a;
  count_embeddings_single(pool, mol_id, pm_a, model, &embs_a);

  // If the binding site is on a non-seed molecule in a multi-molecule
  // reactant pattern, the bind_comp_local index refers to a different
  // molecule's component layout.  In that case, each multi-mol embedding
  // uniquely determines one binding site on the partner molecule.  We can't
  // resolve it via the seed embedding alone, but we know that each valid
  // seed match maps 1:1 to a binding site.  Use synthetic site indices.
  bool const bind_on_nonseed_a = (bind_mol_a != seed_mol_a);
  bool const bind_on_nonseed_b = (bind_mol_b != seed_mol_b);

  // Collect binding sites from A embeddings
  std::unordered_set<int> sites_a;
  if (bind_on_nonseed_a) {
    // Each embedding contributes a unique binding site (use embedding index
    // shifted by a large offset to avoid collisions with B's real indices)
    for (int i = 0; i < static_cast<int>(embs_a.size()); ++i)
      sites_a.insert(i + 1000000);
  } else {
    for (auto& emb : embs_a) {
      if (bind_comp_local_a >= 0 && bind_comp_local_a < static_cast<int>(emb.size()))
        sites_a.insert(emb[bind_comp_local_a]);
    }
  }

  // Get all embeddings of seed_mol_b in this molecule
  auto& pm_b = rule.reactant_pattern.molecules[seed_mol_b];
  std::vector<std::vector<int>> embs_b;
  count_embeddings_single(pool, mol_id, pm_b, model, &embs_b);

  // Collect binding sites from B embeddings
  std::unordered_set<int> sites_b;
  if (bind_on_nonseed_b) {
    for (int i = 0; i < static_cast<int>(embs_b.size()); ++i)
      sites_b.insert(i + 1000000);
  } else {
    for (auto& emb : embs_b) {
      if (bind_comp_local_b >= 0 && bind_comp_local_b < static_cast<int>(emb.size()))
        sites_b.insert(emb[bind_comp_local_b]);
    }
  }

  // Compute intersection
  double both = 0;
  if (bind_on_nonseed_a && bind_on_nonseed_b) {
    // Both binding sites on non-seed molecules.  For symmetric patterns
    // (same structure), every embedding of A maps to the same partner as B.
    // All overlap: both = min(|sites_a|, |sites_b|).
    both = std::min(static_cast<double>(sites_a.size()), static_cast<double>(sites_b.size()));
  } else {
    for (int const s : sites_a)
      if (sites_b.count(s))
        ++both;
  }

  out_a_only = static_cast<double>(sites_a.size()) - both;
  out_b_only = static_cast<double>(sites_b.size()) - both;
  out_ab_both = both;
}

// ===========================================================================
// Engine implementation
// ===========================================================================

// Determine the binding site local component indices for a bimolecular rule.
// Returns (seed_mol_a, bind_local_a, seed_mol_b, bind_local_b).
struct BindInfo {
  int seed_mol_a = -1;   // pattern molecule index for reactant A
  int bind_local_a = -1; // local component index of binding site in A's pattern molecule
  int seed_mol_b = -1;   // pattern molecule index for reactant B
  int bind_local_b = -1; // local component index of binding site in B's pattern molecule
};

BindInfo find_bind_info(const Rule& rule) {
  BindInfo bi;
  if (rule.molecularity < 2)
    return bi;
  if (rule.reactant_pattern_starts.size() < 2)
    return bi;

  int const rp1_start = rule.reactant_pattern_starts[0];
  int const rp2_start = rule.reactant_pattern_starts[1];

  for (auto& op : rule.operations) {
    if (op.type != OpType::AddBond)
      continue;
    if (op.comp_flat_a < 0 || op.comp_flat_b < 0)
      continue;

    // Map flat indices to (mol_idx, local_comp_idx)
    int const flat_a = op.comp_flat_a, flat_b = op.comp_flat_b;
    int base = 0;
    for (int mi = 0; mi < static_cast<int>(rule.reactant_pattern.molecules.size()); ++mi) {
      int const nc = static_cast<int>(rule.reactant_pattern.molecules[mi].components.size());
      if (flat_a >= base && flat_a < base + nc) {
        // Determine which reactant pattern this molecule belongs to
        if (mi >= rp2_start) {
          bi.seed_mol_b = mi;
          bi.bind_local_b = flat_a - base;
        } else {
          bi.seed_mol_a = mi;
          bi.bind_local_a = flat_a - base;
        }
      }
      if (flat_b >= base && flat_b < base + nc) {
        if (mi >= rp2_start) {
          bi.seed_mol_b = mi;
          bi.bind_local_b = flat_b - base;
        } else {
          bi.seed_mol_a = mi;
          bi.bind_local_a = flat_b - base;
        }
      }
      base += nc;
    }
    break; // use first AddBond
  }

  // Default: seed molecules are first molecule of each reactant pattern
  if (bi.seed_mol_a < 0)
    bi.seed_mol_a = rp1_start;
  if (bi.seed_mol_b < 0)
    bi.seed_mol_b = rp2_start;

  return bi;
}

// Match info for a fired reaction
struct ReactionMatch {
  std::vector<int> mol_ids;  // pattern mol idx -> actual mol_id
  std::vector<int> comp_ids; // flat pattern comp idx -> actual comp_id
};

// The sole engine -> canonical-labeler bridge (plan §4); defined far
// below, after the pattern-matching machinery.  Forward-declared here so
// Engine::Impl::cached_label_of (the cached-incremental label layer) can
// call it.
canonical::ComplexGraph extract_complex(const AgentPool& pool, const Model& model,
                                        const std::vector<int>& mol_ids);

} // namespace

struct Engine::Impl {
  // Owned snapshot of the parsed model. Copied at Engine construction so
  // a running session is insulated from subsequent mutations on the
  // simulator side (parameter overrides, bscb toggles, etc.) — those
  // take effect on the next initialize()/run(), not retroactively on
  // the active session.
  Model model;
  AgentPool pool;
  std::vector<RuleState> rule_states;
  std::vector<BindInfo> bind_infos; // per rule
  std::vector<double> obs_values;
  double total_propensity = 0;
  double current_time = 0;
  int64_t event_count = 0;
  int64_t null_event_count = 0;
  int molecule_limit;
  std::mt19937_64 rng;
  bool initialized = false;

  // Cached-incremental canonical-label layer (issue #9 §2 step 5, plan
  // decision #6).  Maps a live complex id to its last-computed canonical
  // BNGL label.  Lazy: an entry is (re)computed by cached_label_of only
  // when read while its id is dirty (pool.cxs_dirty_) or absent — the SSA
  // event loop never canonicalizes, it only sets dirty bits.  Cleared
  // wholesale on load_state.  No production consumer yet; today it is
  // populated and validated only by the decision-#6 self-check inside
  // enumerate_species (gated on kCanonicalCacheSelfCheck).
  std::unordered_map<int, std::string> cx_label_cache_;

  // Lazy canonical label for one live complex: returns the cached string
  // when valid, otherwise recanonicalizes from scratch, stores, and
  // clears the dirty bit.  This is the cached-incremental read primitive
  // (plan §3.2 step 5); partial scaling's species census (plan §7.2)
  // will be its first real consumer.  An absent id is treated as dirty
  // (a just-born complex has no entry yet).
  const std::string& cached_label_of(int cx_id) {
    // erase() both clears the dirty mark and reports whether it was set.
    bool const was_dirty = pool.cxs_dirty_.erase(cx_id) != 0;
    auto it = cx_label_cache_.find(cx_id);
    if (was_dirty || it == cx_label_cache_.end()) {
      std::string label = canonical::canonical_label(
          extract_complex(pool, model, pool.molecules_in_complex(cx_id)));
      if (it == cx_label_cache_.end())
        it = cx_label_cache_.emplace(cx_id, std::move(label)).first;
      else
        it->second = std::move(label);
    }
    return it->second;
  }

  // Type→rule index: for each molecule type, which rules reference it as
  // reactant A or B.  Built once in init_rule_states(), used by
  // incremental_update() to skip irrelevant rules.
  std::vector<std::vector<int>> type_to_rules; // type_index → rule indices
  std::vector<int> dynamic_synthesis_rules;    // synthesis rules with dynamic rates
  std::vector<int> dynamic_rate_rules;         // non-synthesis rules with dynamic rates (func/expr)
  // Local-rate rules whose function chain reads a moving global (issue
  // #38): rescanned in full when no affected molecule marks the change.
  // Empty for every model whose local rates are built from tagged
  // observables and constants alone.
  //
  // The rescan is gated on the dependency values actually moving (issue
  // #40).  "Reads a global" is a load-time property and stays true for the
  // whole run; "that global just moved" is the per-event question, and on
  // the common shape — a bare observable standing for a volume, a total or
  // a clock — the answer is no at every event of the run.  Each entry
  // caches the values read at its last rescan; when they all match, the
  // rule keeps the ordinary delta path, which is what a local rate over a
  // genuinely constant global would have taken all along.
  struct GlobalDepWatch {
    int rule_index = -1;
    bool always_dirty = false;  // reads `time`, or a dependency §8b could not name
    std::vector<int> obs_slots; // observable slots read at system scope
    std::vector<double> last;   // their values at the last rescan
    bool primed = false;        // false until that first rescan has happened
  };
  std::vector<GlobalDepWatch> global_dep_local_rules;
  std::vector<char> rule_needed_buf; // pre-allocated, reused each update
  int max_pattern_depth = 0;         // max pattern diameter across all rules

  // Per-rule event counting for diagnostics
  std::vector<uint64_t> rule_fire_counts;

  // SSA loop timing instrumentation (cumulative seconds).
  //   timing_sample : select_reactants (pick a rule's reactants)
  //   timing_select : currently unused
  //   timing_fire   : fire_rule body + pre-fire constraint checks on
  //                   events that reach fire_rule
  //   timing_obs    : incremental observable update after each event
  //   timing_update : incremental_update (propensity recompute)
  //   timing_record : compute_observables + record at sample points
  //                   (wall clock that would otherwise be unattributed)
  //   timing_wall   : true wall clock for the whole SSA loop, for
  //                   an accounting check against the sum of the
  //                   per-phase timers
  double timing_sample = 0, timing_select = 0, timing_fire = 0;
  double timing_obs = 0, timing_update = 0;
  double timing_record = 0, timing_wall = 0;

  // fire_rule / incremental_update / record_at profile counters.  Struct
  // definitions live in engine_profile.hpp (along with their report_*()
  // bodies); the instances here are populated by the gated `if constexpr`
  // increment sites scattered through this file.
  FireRuleProfile fire_profile_;
  IncrUpdateProfile incr_profile_;
  RecordAtProfile rap_profile_;

  // Precomputed: indices into model.observables of every observable that
  // some local function evaluates at its tag — the union across functions.
  // These are the slots evaluate_local_factor overrides with a per-molecule
  // value.  Bare (global) observables of a local function are NOT here:
  // they keep their system-wide value (issue #38).
  std::vector<int> local_obs_indices;
  // Set only when the union above is ambiguous — some local function reads
  // an observable at global scope that another localizes.  A single slot
  // then cannot serve both, so evaluate_local_factor switches from one
  // override for the whole chain to a per-function re-scope.  False for
  // every model that does not mix the two scopings, which keeps the common
  // path exactly as it was.
  bool local_obs_scope_conflict_ = false;
  // Per function (indexed like model.functions), a mask over
  // local_obs_indices: 1 where that function localizes the observable, 0
  // where it reads it globally.  Populated only when
  // local_obs_scope_conflict_.
  std::vector<std::vector<char>> fn_local_obs_mask_;
  // Precomputed: whether any rule uses local rates
  bool have_local_rules_ = false;
  // Precomputed: whether any rule has disjoint multi-mol patterns
  bool any_needs_complex_expansion_ = false;
  // Reusable cache for local rate values (persists bucket storage across steps)
  std::unordered_map<int, double> local_rate_cache;
  // Companion cache for the FunctionProduct B-side factor (f2), keyed by
  // complex id.  Kept separate from local_rate_cache because the A- and
  // B-side factors are distinct functions over the same complex ids.
  std::unordered_map<int, double> local_rate_cache_b;

  // P1 cache (step 2): per-event record of which components of which
  // molecules were mutated by the current firing.  Populated by fire_rule
  // for each StateChange / AddBond / DeleteBond op; valid for one
  // fire_rule invocation and disambiguated across events by an epoch
  // counter that advances per firing.  Consumed by incremental_update to
  // decide whether cached PerMolRuleData entries are still valid for a
  // given (rule, molecule) pair.  A mol whose mask_epoch[mid] != current
  // has an effective change_mask of zero ("not directly mutated this
  // event") — typically a BFS-expanded neighbor — and is an automatic
  // cache hit for cacheable rules.
  //
  // Vector + epoch beats unordered_map here: each cache hit check in the
  // inner loop is a pair of vector reads plus an int compare, avoiding
  // the hash map's pointer chase that dominated overhead on small
  // populations (ft_push_pull, ~12k events).
  std::vector<uint64_t> event_mol_change_mask;
  std::vector<uint64_t> event_mol_change_epoch;
  uint64_t event_epoch = 0;

  // Incremental observable tracking.  Originally molecules-only, extended
  // to Species observables in the record_at Step 2 sprint.
  //
  // For every tracked observable (Molecules or Species, single-mol
  // patterns), obs_mol_contrib[oi][mid] holds the per-mol contribution.
  // Molecules observables aggregate the contribs by sum into obs_values;
  // Species observables aggregate by complex and apply the pattern's
  // quantity predicate, flushing dirty complexes at sample time.
  std::vector<std::vector<double>> obs_mol_contrib;
  std::vector<int> rate_dep_obs_indices; // rate-dependent obs only (kept for
                                         // compute_rate_dependent_observables)
  bool use_incremental_obs = false;

  // Species-path state (populated only for Species-tracked obs).
  // incr_tracked_obs_indices is the union of Molecules-tracked and
  // Species-tracked observables; incr_obs_is_species[oi] distinguishes
  // them.  Molecules-tracked obs also live in rate_dep_obs_indices when
  // they are rate-dependent, for the BFS delta path.
  std::vector<int> incr_tracked_obs_indices;
  std::vector<char> incr_obs_is_species; // size = model.observables.size()
  std::vector<char> incr_obs_is_tracked; // 1 if either path; size = model.observables.size()
  // Trivial-pattern flag: obs with exactly one pattern of exactly one
  // molecule with no components — `R()`, `L()`, etc.  For these, the
  // per-mid contribution is 1 for any active type-matching mol and
  // doesn't depend on state or bonds, so the per-event recompute can
  // skip the count_multi_molecule_embeddings call.  Dirty-cx marking
  // still fires for Species trivials so bond-op cx changes are seen.
  std::vector<char> incr_obs_trivial;

  // Per Species-tracked obs: per-complex aggregate + passes cache + dirty
  // set.  Indexed by obs index (sparse — non-tracked slots hold empty
  // containers).
  std::vector<std::unordered_map<int, int>> obs_cx_match_count;
  std::vector<std::unordered_map<int, char>> obs_cx_passed;
  std::vector<std::unordered_set<int>> obs_dirty_cx;

  // Per-mid snapshot of the last-seen complex id across all Species-
  // tracked obs.  Populated/updated whenever a mid is touched by an
  // event.  Used to dirty both the old and new cx when a bond op moves
  // mid between complexes.  -1 means "no snapshot yet" (mid has not been
  // touched since init); at init we seed this with pool.complex_of(mid)
  // for every mid of a type referenced by a tracked Species obs.
  std::vector<int> species_mid_prev_cx;
  bool species_incr_any_tracked = false;

  // Scratch for the Species full-walk in evaluate_observable: a
  // generation stamp per molecule id marking "the complex this molecule
  // belongs to has already been counted for the pattern in flight".
  // Molecule ids are dense, so this replaces a per-call
  // unordered_set<int> of complex ids — see the comment at the use site.
  std::vector<uint64_t> obs_walk_cx_stamp;
  uint64_t obs_walk_cx_gen = 0;
  // One-element stand-in for the member list of a singleton complex, so
  // the walk's inner loop stays a single code path.
  std::vector<int> obs_walk_singleton_scratch{-1};

  // Max pattern diameter across all tracked obs.  Determines the BFS
  // depth required for incremental contrib updates: a bond/state
  // change on mol X can alter contrib(Y) only when Y is within
  // obs_max_depth bond-hops of X.  Rule BFS (fire_rule) uses its own
  // rule_max_pattern_depth — when obs_max_depth exceeds that, we need
  // to expand `affected` further before recomputing per-mid contribs.
  int obs_max_pattern_depth = 0;

  // Per-(tracked-obs, pattern) FastMatchSlot for the 2-mol-1-bond-fc
  // specialization on the obs-tracking path.  obs_pat_fm[oi][pi].enabled
  // is true iff patterns[pi] matches the 2-mol-1-bond eligibility
  // template (same template as the rule-side P6 fast path).  Indexed by
  // observable index; inner vector size matches obs.patterns.size() when
  // the obs is tracked, else empty.
  std::vector<std::vector<FastMatchSlot>> obs_pat_fm;

  // Observables-bucket profile counters (incremental_update_observables +
  // flush_species_incr_observables) and select_reactants per-path
  // counters.  Both struct definitions live in engine_profile.hpp.
  ObsIncrProfile obs_incr_profile_;
  SrProfile sr_profile_;
  ExprEvalProfile expr_eval_profile_;
  // issue #10 spike state.  Per (observable, scope-id) key: the last
  // computed value and the event epoch it was computed in.  Lets
  // evaluate_observable_on classify each call as a fresh compute, an
  // intra-event repeat (per-event memoization would catch it), or a
  // cross-event repeat whose value happens to be unchanged (only a
  // persistent incremental tracker would catch it).  Touched only under
  // `if constexpr (kExprEvalProfile)`, so it stays empty in normal builds.
  struct EooSpikeRec {
    double value;
    uint64_t epoch;
  };
  std::unordered_map<uint64_t, EooSpikeRec> eoo_spike_prev_;
  uint64_t eoo_spike_epoch_ = 0;
  // Per-Engine homes for the two profile structs whose call sites are
  // static free functions (count_multi_mol_fast / count_2mol_1bond_fc).
  // Threaded in by pointer so concurrent Engines on different threads
  // don't trample each other and per-Engine reports are exact.
  CountMultiProfile cm_profile_;
  CmmFcProfile cmm_fc_profile_;

  // Dynamic rate evaluation state.  `eval_vars_flat` holds variable
  // values in a fixed layout — params, then time, then observables,
  // then function values (with an extra slot per tfun function for
  // the `__tfun_NAME__` magic name).  Every slot is bound by address
  // into `expr_eval_` once at engine construction (init_eval_layout):
  // `eval_vars_flat` is sized exactly once and never reallocates, so
  // `&eval_vars_flat[slot]` is stable for the engine's lifetime.
  // Per-event evaluation just writes the live values into the slots
  // and calls `expr_eval_.evaluate(id)`.
  std::vector<double> eval_vars_flat;
  std::unordered_map<std::string, int> eval_layout_index;
  int eval_time_slot = -1;
  std::vector<int> eval_obs_slot;     // [obs_idx] -> slot
  std::vector<int> eval_gf_main_slot; // [func_idx] -> slot
  std::vector<int> eval_gf_tfun_slot; // [func_idx] -> slot, -1 if not tfun

  // ExprTk evaluator for every rate-law / global-function expression
  // in the model (issue #6 — replaces the hand-rolled AST evaluator).
  // `gf_expr_id_[func_idx]` is the compiled-expression id for that
  // global function, or -1 for a pure-TFUN function with no wrapper
  // expression (evaluate the table directly).  Rate laws are always
  // emitted by BNG2 as named global functions, so the engine never
  // compiles a per-rule inline expression.
  bngsim::ExprTkEvaluator expr_eval_;
  std::vector<int> gf_expr_id_;

  // Global (non-local) functions whose value is exposed to embedders via
  // Result::function_data and get_function_values().  Local functions are
  // excluded: they evaluate per-molecule and have no single global value.
  // `reactant_N()` placeholders are excluded for the same reason one step
  // further out — they have no value even at one molecule, only inside a
  // rule — and NFsim does not create a function for one at all, so a column
  // named after one would be RM inventing an output its reference does not
  // have.  `output_function_indices` indexes into model.functions; the names
  // are cached in declaration order to parallel Result::function_names.
  std::vector<int> output_function_indices;
  std::vector<std::string> output_function_names;

  // Stale-flag short-circuit for update_eval_vars.  `eval_vars_gen` is
  // bumped whenever the inputs to the rate-law layout change
  // (current_time advances, obs_values is mutated by compute_observables
  // / incremental_update_observables / compute_rate_dependent_observables).
  // `eval_vars_seen_gen` records the last gen the cached `eval_vars_flat`
  // was refreshed against; update_eval_vars short-circuits when the two
  // match.  Within one event's incremental_update there are typically
  // many evaluate_rate calls — pre-#17 they each rebuilt the full table
  // from scratch.  Parameters are hoisted out of the refresh loop
  // (init_eval_layout writes them once) so they don't participate.
  uint64_t eval_vars_gen = 1;
  uint64_t eval_vars_seen_gen = 0;

  // Eval-layout slots of the `reactant_N()` placeholders the model
  // declares, as (N, slot) pairs in declaration order (issue #59).  The
  // placeholder is an ordinary named slot — nothing else writes it, since
  // it has no expression to evaluate — so binding a rule's reactant counts
  // is a handful of stores into `eval_vars_flat` before the rate is
  // evaluated.  Empty for the overwhelming majority of models, which never
  // mention the construct.
  std::vector<std::pair<int, int>> eval_reactant_slots;

  // Which rule's counts a function should be evaluated against when the
  // whole function table is settled (rebuild_eval_vars).  A rate law that
  // reads a placeholder has no value outside the rule that owns it, so the
  // settle pass binds that rule's counts first; -1 means the function does
  // not read one.  Two rules sharing one rate function is not a shape BNG2
  // writes — it names a fresh `_rateLawN` per rule — but if it happened the
  // first owner wins and the settled value is that rule's.  Only the
  // reported value is affected: evaluate_rate always binds the counts of
  // the rule it was called for.
  std::vector<int> gf_reactant_owner_rule_;

  // Slot index for each LOCAL function, precomputed once so
  // evaluate_local_rate can save/restore its overrides without
  // walking model.functions on every per-mol call.
  std::vector<int> eval_local_fn_slots;

  // Save buffers used by evaluate_local_rate to restore the global
  // eval_vars_flat view after per-mol overrides.  Member-scoped so
  // we don't allocate on every per-mid call (size is stable once
  // local_obs_indices and eval_local_fn_slots are populated).
  std::vector<double> eval_local_obs_save_;
  std::vector<double> eval_local_fn_save_;
  // Companion buffer: the localized value of each local_obs_indices entry
  // at the molecule being evaluated.  Only read under
  // local_obs_scope_conflict_, where the slots are re-scoped per function
  // instead of once for the whole chain.
  std::vector<double> eval_local_obs_value_;

  // pool is initialized from `model` (the just-copied snapshot), not
  // from `m` — declaration order guarantees `model` is constructed
  // before `pool`, so AgentPool::model_ references the engine's own
  // copy and outlives the input reference.
  Impl(Model m, uint64_t seed, int mol_limit)
      : model(std::move(m)), pool(model), molecule_limit(mol_limit), rng(seed) {
    init_eval_layout();
  }

  // Build the flat variable layout for rate-law evaluation: assign every
  // parameter / time / observable / function / tfun-slot a slot index in
  // `eval_vars_flat`, bind each slot by address into `expr_eval_`, and
  // compile every global-function expression.  Runs once at engine
  // construction.  The simulator owns a separate ExprTk evaluator for the
  // parameter cascade (see resolve_cached in simulator.cpp).
  void init_eval_layout() {
    eval_layout_index.clear();
    int next = 0;

    for (auto& name : model.parameter_names_ordered)
      eval_layout_index[name] = next++;

    eval_time_slot = next++;
    eval_layout_index["time"] = eval_time_slot;
    eval_layout_index["t"] = eval_time_slot;

    eval_obs_slot.assign(model.observables.size(), -1);
    for (int i = 0; i < static_cast<int>(model.observables.size()); ++i) {
      eval_obs_slot[i] = next;
      eval_layout_index[model.observables[i].name] = next++;
    }

    eval_gf_main_slot.assign(model.functions.size(), -1);
    eval_gf_tfun_slot.assign(model.functions.size(), -1);
    for (int i = 0; i < static_cast<int>(model.functions.size()); ++i) {
      auto& gf = model.functions[i];
      eval_gf_main_slot[i] = next;
      eval_layout_index[gf.name] = next++;
      if (gf.is_tfun) {
        eval_gf_tfun_slot[i] = next;
        eval_layout_index["__tfun_" + gf.name + "__"] = next++;
      }
    }

    eval_vars_flat.assign(next, 0.0);

    // Parameters are constant for the lifetime of the engine
    // (set_param is rejected during an active session and rebuilds the
    // engine on the next run), so write them into eval_vars_flat once
    // here instead of re-reading model.parameters on every rate eval.
    for (int i = 0; i < static_cast<int>(model.parameter_names_ordered.size()); ++i) {
      auto& name = model.parameter_names_ordered[i];
      auto it = model.parameters.find(name);
      eval_vars_flat[i] = (it != model.parameters.end()) ? it->second : 0.0;
    }

    // Bind every layout slot into the ExprTk evaluator by address.
    // `eval_vars_flat` is now sized for good and never reallocates, so
    // `&eval_vars_flat[slot]` is stable for the engine's lifetime.
    // `time` is NOT bound as a variable — ExprTk registers `time()` as
    // a zero-arg function, wired straight to the live `current_time`
    // member.  Bare `t` IS bound (RM aliases it to the time slot,
    // which rebuild_eval_vars keeps equal to current_time).
    expr_eval_.set_time_ptr(&current_time);
    for (auto& [name, slot] : eval_layout_index) {
      if (name == "time")
        continue;
      expr_eval_.define_variable(name, &eval_vars_flat[slot]);
    }

    // Compile every global-function / rate-law expression.  A pure-TFUN
    // function carries no wrapper expression → id -1, evaluated straight
    // off the table.  A non-TFUN function whose expression fails to
    // compile is a hard model error and propagates out of construction.
    gf_expr_id_.assign(model.functions.size(), -1);
    for (int i = 0; i < static_cast<int>(model.functions.size()); ++i) {
      auto& gf = model.functions[i];
      if (gf.expression_text.empty())
        continue;
      std::string const src = strip_layout_call_args(gf.expression_text);
      try {
        gf_expr_id_[i] = expr_eval_.compile(src);
      } catch (const std::exception&) {
        if (!gf.is_tfun)
          throw;
        gf_expr_id_[i] = -1; // pure-TFUN: fall back to the raw table value
      }
    }

    // Cache the global-function output list once: it depends only on the
    // model (which function entries have local arguments), never on run
    // state, so embedders see a stable name vector across runs.
    output_function_indices.clear();
    output_function_names.clear();
    for (int i = 0; i < static_cast<int>(model.functions.size()); ++i) {
      if (!model.functions[i].is_local() && model.functions[i].reactant_count_index == 0) {
        output_function_indices.push_back(i);
        output_function_names.push_back(model.functions[i].name);
      }
    }

    eval_local_fn_slots.clear();
    for (int i = 0; i < static_cast<int>(model.functions.size()); ++i) {
      if (model.functions[i].is_local() && gf_expr_id_[i] >= 0)
        eval_local_fn_slots.push_back(i);
    }

    // `reactant_N()` placeholders and the rate laws that read them (#59).
    eval_reactant_slots.clear();
    gf_reactant_owner_rule_.assign(model.functions.size(), -1);
    for (int i = 0; i < static_cast<int>(model.functions.size()); ++i) {
      int const n = model.functions[i].reactant_count_index;
      if (n > 0)
        eval_reactant_slots.emplace_back(n, eval_gf_main_slot[i]);
    }
    if (!eval_reactant_slots.empty()) {
      for (int ri = 0; ri < static_cast<int>(model.rules.size()); ++ri) {
        const auto& rl = model.rules[ri].rate_law;
        if (rl.max_reactant_count_index == 0 || rl.function_name.empty())
          continue;
        auto fit = model.function_index.find(rl.function_name);
        if (fit != model.function_index.end() && gf_reactant_owner_rule_[fit->second] < 0)
          gf_reactant_owner_rule_[fit->second] = ri;
      }
    }
  }

  // Rewrite `name(...)` -> `name` for every identifier that names an
  // eval-layout slot (parameter / observable / global function / tfun
  // magic name).  RM's legacy AST evaluator treated a reference like
  // `M_on(z)` or `frac_P()` as "the current value of the slot named
  // M_on / frac_P" — the argument list was parsed but never evaluated
  // (BNG2 emits observable-at-local-scope and function references this
  // way).  ExprTk has no such notion: `M_on` is a scalar variable, so
  // `M_on(z)` would be misparsed as the implicit product `M_on * (z)`.
  // Stripping the balanced argument list reproduces the slot-value
  // semantics exactly.  Builtin calls (`exp`, `if`, `time`, ...) are not
  // eval-layout names, so they pass through untouched.
  std::string strip_layout_call_args(const std::string& expr) const {
    std::string out;
    out.reserve(expr.size());
    size_t i = 0;
    const size_t n = expr.size();
    while (i < n) {
      const bool at_boundary =
          (i == 0) ||
          (!std::isalnum(static_cast<unsigned char>(expr[i - 1])) && expr[i - 1] != '_');
      const bool ident_start = std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_';
      if (at_boundary && ident_start) {
        size_t const start = i;
        ++i;
        while (i < n && (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_'))
          ++i;
        std::string const ident = expr.substr(start, i - start);
        out += ident;
        if (i < n && expr[i] == '(' && eval_layout_index.count(ident)) {
          int depth = 0;
          for (; i < n; ++i) {
            if (expr[i] == '(') {
              ++depth;
            } else if (expr[i] == ')') {
              --depth;
              if (depth == 0) {
                ++i;
                break;
              }
            }
          }
        }
        continue;
      }
      out += expr[i];
      ++i;
    }
    return out;
  }

  // Uniform [0, 1)
  double uniform() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
  }

  // --- Total propensity (delta-updated) ---
  //
  // total_propensity is maintained as a running sum: every write to a
  // rule's propensity goes through set_rule_propensity, which credits the
  // delta to total_propensity.  This replaces the per-event O(R) sum at
  // the top of the SSA loop.  Periodic re-baseline (every K events, plus
  // a defensive re-sum on suspected absorbing-state fall-through) bounds
  // accumulated FP drift.
  static constexpr int64_t kPropensityRebaselineInterval = 4096;
  int64_t events_since_propensity_rebaseline = 0;

  void set_rule_propensity(RuleState& rs, double new_value) {
    // `!(v >= 0)` rather than `v < 0` so NaN takes the clamp too (issue #46).
    // A NaN comes from a rate law evaluated outside its domain — a negative
    // MM Km is the reachable one, since Km is a parameter and an override or
    // a fit can move it there after load.  NaN compares false against
    // everything, so the old `v < 0` test passed it straight through and
    // std::max returned it unchanged: total_propensity became NaN and the
    // sampler kept drawing events off it.  Measured on an MM rule with
    // Km = -1: 80 events fired against a NaN propensity, exit code 0, and an
    // ordinary-looking .gdat.
    if (!(new_value >= 0) && !rs.clamp_warned) {
      rs.clamp_warned = true;
      // rs is always inside rule_states (every call site indexes through
      // rule_states[ri] or iterates the vector), so pointer arithmetic to
      // recover the index is well-defined for vector elements.
      auto const ri = static_cast<size_t>(&rs - rule_states.data());
      auto const& rule = model.rules[ri];
      const char* fn = rule.rate_law.function_name.c_str();
      // Name the cause when we know it: an MM rule reporting NaN with a
      // negative Km is that Km, not an arithmetic accident downstream.
      bool const mm_domain =
          std::isnan(new_value) && rule.rate_law.type == RateLawType::MM && rule.rate_law.mm_Km < 0;
      if (mm_domain) {
        fprintf(stderr,
                "WARN: rule '%s' (%s) propensity clamped to 0 — MM Km=%g is outside "
                "the rate law's domain (Km must be > 0) at t=%g; further clamps on "
                "this rule are silent\n",
                rule.id.c_str(), rule.name.c_str(), rule.rate_law.mm_Km, current_time);
      } else {
        fprintf(stderr,
                "WARN: rule '%s' (%s) propensity clamped to 0 — rate%s%s%s "
                "evaluated to %g at t=%g; further clamps on this rule are silent\n",
                rule.id.c_str(), rule.name.c_str(), (fn[0] != 0 ? " function '" : ""), fn,
                (fn[0] != 0 ? "'" : ""), new_value, current_time);
      }
    }
    if (!(new_value >= 0))
      new_value = 0;
    total_propensity += new_value - rs.propensity;
    rs.propensity = new_value;
  }
  void set_rule_propensity(int ri, double new_value) {
    set_rule_propensity(rule_states[ri], new_value);
  }

  void recompute_total_propensity() {
    double s = 0;
    for (auto& rs : rule_states)
      s += rs.propensity;
    total_propensity = s;
    events_since_propensity_rebaseline = 0;
  }

  // --- State save / restore ---

  void save_state_to(const std::string& path) const {
    std::ofstream os(path);
    if (!os)
      throw std::runtime_error("Cannot open state file for writing: " + path);
    os << std::setprecision(17);
    // V2 (2026-04-30): added schema fingerprint between marker and time.
    os << "RM_STATE_V2\n";
    os << std::hex << compute_schema_fingerprint(model) << std::dec << "\n";
    os << current_time << "\n";
    os << event_count << "\n";
    os << null_event_count << "\n";
    pool.write_state(os);
    os << rng << "\n";
    os << "END\n";
    if (!os)
      throw std::runtime_error("Error writing state file: " + path);
  }

  void load_state_from(const std::string& path) {
    std::ifstream is(path);
    if (!is)
      throw std::runtime_error("Cannot open state file for reading: " + path);
    std::string marker;
    is >> marker;
    if (marker == "RM_STATE_V1") {
      // V1 has no schema fingerprint; an XML mismatch silently corrupts
      // the loaded run.  Reject loudly: callers should re-save against
      // V2 (any save_state from this build emits V2).
      throw std::runtime_error("State file is RM_STATE_V1 (pre-fingerprint format); re-save with "
                               "this RuleMonkey build to get an XML-mismatch-checked V2 file");
    }
    if (marker != "RM_STATE_V2")
      throw std::runtime_error("Invalid state file header: " + marker);

    // Schema fingerprint check.  Mismatch means the XML used at
    // save_state is structurally different from the one this simulator
    // was constructed from, so molecule/component indices in the pool
    // serialization don't refer to the same schema slots.  Refuse
    // rather than silently producing corrupt trajectories.
    uint64_t saved_fp = 0;
    is >> std::hex >> saved_fp >> std::dec;
    uint64_t const current_fp = compute_schema_fingerprint(model);
    if (saved_fp != current_fp) {
      std::ostringstream msg;
      msg << "State file schema fingerprint mismatch: saved=" << std::hex << saved_fp
          << " current=" << current_fp << std::dec
          << " — the XML loaded into this simulator does not match the one "
             "used at save_state (molecule types, components, or allowed "
             "states differ)";
      throw std::runtime_error(msg.str());
    }

    is >> current_time;
    is >> event_count;
    is >> null_event_count;
    pool.read_state(is);
    // The restored pool has all-new complex ids; every cached label is
    // stale.  pool.read_state already cleared pool.cxs_dirty_; drop the
    // label cache to match (plan §3.2 step 5 — the load_state trap).
    cx_label_cache_.clear();
    is >> rng;
    is >> marker;
    if (marker != "END")
      throw std::runtime_error("State file missing END marker, got: " + marker);

    // Rebuild derived state from the restored pool
    compute_observables();
    init_rule_states();
    init_incremental_observables();
    initialized = true;
  }

  // --- Initialization ---

  // Build a mapping from species-XML component index to MoleculeType component
  // index for a single molecule.  Handles duplicate component names by matching
  // the N-th occurrence in the species to the N-th occurrence in the type.
  static std::vector<int> build_comp_mapping(const SpeciesInitMol& sim, const MoleculeType& mtype) {
    std::vector<int> mapping(sim.comp_states.size(), -1);
    std::unordered_map<std::string, int> name_count;

    for (int ci = 0; ci < static_cast<int>(sim.comp_states.size()); ++ci) {
      auto& cname = sim.comp_states[ci].first;
      int const occurrence = name_count[cname]++;

      int found = 0;
      for (int mi = 0; mi < static_cast<int>(mtype.components.size()); ++mi) {
        if (mtype.components[mi].name == cname) {
          if (found == occurrence) {
            mapping[ci] = mi;
            break;
          }
          ++found;
        }
      }
    }
    return mapping;
  }

  // One seed species with every per-copy decision already made: which
  // molecule types to create, which component slots to set to which
  // state index, and which component slots to bond.  Everything here is
  // a function of the SpeciesInit and the model alone, so it is the same
  // for all `count` copies — resolving it per copy is what made
  // `build_comp_mapping` (an unordered_map keyed by component name,
  // built and thrown away per molecule) about a sixth of session build
  // on a million-molecule pool (issue #67).
  struct SeedTemplate {
    struct Mol {
      int type_index = -1;
      // (component index in the molecule type, state index) — only the
      // slots the species actually pins.  A component the species leaves
      // unstated keeps the default add_molecule gives it.
      std::vector<std::pair<int, int>> states;
    };
    struct Bond {
      int mol_a = -1;
      int local_a = -1; // component index within mol_a's molecule type
      int mol_b = -1;
      int local_b = -1;
    };
    std::vector<Mol> molecules;
    std::vector<Bond> bonds;
    int count = 0; // copies to instantiate
  };

  // Resolve one seed species against the model.  The guards match the
  // per-copy code this replaced one for one: an unmapped component, a
  // state name the type does not declare, a bond endpoint outside the
  // species, and a component index outside the molecule type are all
  // dropped rather than diagnosed, exactly as before.
  SeedTemplate resolve_seed_template(const SpeciesInit& si, int count) const {
    SeedTemplate tmpl;
    tmpl.count = count;
    tmpl.molecules.reserve(si.molecules.size());
    // Species-XML component index -> molecule-type component index, per
    // molecule.  Kept only until the bonds below are resolved.
    std::vector<std::vector<int>> comp_mappings;
    comp_mappings.reserve(si.molecules.size());

    for (const auto& sim : si.molecules) {
      const auto& mtype = model.molecule_types[sim.type_index];
      auto const n_type_comps = static_cast<int>(mtype.components.size());
      auto cmap = build_comp_mapping(sim, mtype);

      SeedTemplate::Mol m;
      m.type_index = sim.type_index;
      for (int ci = 0; ci < static_cast<int>(sim.comp_states.size()); ++ci) {
        const auto& [cname, cstate] = sim.comp_states[ci];
        if (cstate.empty())
          continue;
        int const actual_ci = cmap[ci];
        if (actual_ci < 0)
          continue;
        int const state_idx = mtype.state_index(actual_ci, cstate);
        if (state_idx >= 0 && actual_ci < n_type_comps)
          m.states.emplace_back(actual_ci, state_idx);
      }
      tmpl.molecules.push_back(std::move(m));
      comp_mappings.push_back(std::move(cmap));
    }

    auto const n_mols = static_cast<int>(si.molecules.size());
    for (const auto& bond : si.bonds) {
      if (bond.mol_a >= n_mols || bond.mol_b >= n_mols || bond.comp_a < 0 || bond.comp_b < 0)
        continue;
      // A bond endpoint beyond the mapping is taken as already being a
      // molecule-type component index.
      int const local_a = (bond.comp_a < static_cast<int>(comp_mappings[bond.mol_a].size()))
                              ? comp_mappings[bond.mol_a][bond.comp_a]
                              : bond.comp_a;
      int const local_b = (bond.comp_b < static_cast<int>(comp_mappings[bond.mol_b].size()))
                              ? comp_mappings[bond.mol_b][bond.comp_b]
                              : bond.comp_b;
      auto const n_a = static_cast<int>(
          model.molecule_types[si.molecules[bond.mol_a].type_index].components.size());
      auto const n_b = static_cast<int>(
          model.molecule_types[si.molecules[bond.mol_b].type_index].components.size());
      if (local_a < 0 || local_b < 0 || local_a >= n_a || local_b >= n_b)
        continue;
      tmpl.bonds.push_back({bond.mol_a, local_a, bond.mol_b, local_b});
    }
    return tmpl;
  }

  void init_species() {
    std::vector<SeedTemplate> templates;
    templates.reserve(model.initial_species.size());
    // Seed totals, for the pool's capacity hint.  `long long` because
    // each is a count times a per-species molecule count.
    long long n_molecules = 0;
    long long n_components = 0;
    long long n_complexes = 0;
    std::vector<long long> per_type(model.molecule_types.size(), 0);

    for (auto& si : model.initial_species) {
      // Truncate-toward-zero (NFsim parity).  NFsim's NFinput.cpp:774
      // explicitly does `(int) convertToDouble(specCount)` with a
      // comment "always round the number down to the nearest whole
      // integer" — switching to `std::lround` here breaks parity on
      // any model with a non-integer concentration that lands above
      // .5 (e.g. basicTLBR's `NL = (L_conc*1e-9)*(NA*V) = 421.5498`
      // becomes 422 under round but 421 under truncate; the reference
      // ensemble was generated with NFsim's 421, and the new #29
      // hard-fail catches the 1-molecule degenerate-observable
      // mismatch).  Keep the cast explicit and the truncation intent
      // documented rather than masking the FP edge case with rounding.
      int const count = static_cast<int>(si.concentration); // truncate toward zero (NFsim parity)
      if (count <= 0)
        continue;
      templates.push_back(resolve_seed_template(si, count));
      {
        // Complexes a copy leaves behind: one per molecule, less one for
        // each bond that merges two of them.
        const auto& t = templates.back();
        auto const per_copy =
            static_cast<long long>(t.molecules.size()) - static_cast<long long>(t.bonds.size());
        n_complexes += count * (per_copy > 1 ? per_copy : 1);
      }
      for (const auto& m : templates.back().molecules) {
        if (m.type_index < 0 || m.type_index >= static_cast<int>(per_type.size()))
          continue; // capacity accounting only — the walk itself is unchanged
        n_molecules += count;
        n_components +=
            count * static_cast<long long>(model.molecule_types[m.type_index].components.size());
        per_type[m.type_index] += count;
      }
    }

    pool.reserve_for_seed(n_molecules, n_components, n_complexes, per_type);

    std::vector<int> mol_ids; // one instance's molecules; reused across copies
    for (const auto& tmpl : templates) {
      mol_ids.reserve(tmpl.molecules.size());
      for (int n = 0; n < tmpl.count; ++n) {
        mol_ids.clear();
        for (const auto& m : tmpl.molecules) {
          int const mid = pool.add_molecule(m.type_index);
          // Reference into the molecule arena: nothing below adds a
          // molecule before it is done with, so it cannot dangle.
          const auto& mol = pool.molecule(mid);
          for (const auto& [local_ci, state_idx] : m.states)
            pool.set_state(mol.comp_ids[local_ci], state_idx);
          mol_ids.push_back(mid);
        }
        for (const auto& b : tmpl.bonds) {
          int const cid_a = pool.molecule(mol_ids[b.mol_a]).comp_ids[b.local_a];
          int const cid_b = pool.molecule(mol_ids[b.mol_b]).comp_ids[b.local_b];
          pool.add_bond(cid_a, cid_b);
        }
      }
    }
  }

  // Instantiate one concrete copy of an exact species `pat` into the
  // pool (issue #9 §1 add_species).  `pat` comes from
  // parse_species_pattern: every molecule lists every component (in
  // listed order) with a concrete state, and pat.bonds gives the edges
  // by flat component index.  This is the parse-side analogue of
  // init_species's per-instance inner loop.
  void instantiate_pattern_complex(const Pattern& pat) {
    // flat component index -> the pool component id just created for it
    std::vector<int> flat_to_cid;
    flat_to_cid.reserve(pat.flat_comp_count());
    for (const auto& pm : pat.molecules) {
      int const mid = pool.add_molecule(pm.type_index);
      const auto& mol = pool.molecule(mid);
      for (const auto& pc : pm.components) {
        // comp_type_index addresses the physical component; comp_ids is
        // indexed by molecule-type declaration order, exactly as
        // pool.add_molecule allocates it.
        int const cid = mol.comp_ids[pc.comp_type_index];
        if (pc.required_state_index >= 0)
          pool.set_state(cid, pc.required_state_index);
        flat_to_cid.push_back(cid);
      }
    }
    for (const auto& b : pat.bonds)
      pool.add_bond(flat_to_cid[b.comp_flat_a], flat_to_cid[b.comp_flat_b]);
  }

  // Resync all derived state after a structural pool mutation that is
  // not localized to one molecule type (add_species / remove_species,
  // issue #9 §1).  Unlike add_molecules' targeted rescan, every rule is
  // rescanned: a multi-molecule species touches several types at once,
  // and these are rare paused-session calls, so a full recompute is
  // both correct and simple.
  void resync_all_after_structural_change() {
    compute_observables();
    for (int ri = 0; ri < static_cast<int>(model.rules.size()); ++ri)
      rescan_all_molecules_for_rule(ri);
    recompute_total_propensity();
    init_incremental_observables();
    // Every pure-context tally was just rebuilt from the live pool, so the
    // moves this mutation logged have nothing left to say.  Draining them at
    // the next event would be harmless (percx_sync is idempotent) but would
    // let the list grow across a paused session that adds many species.  The
    // edit log is in the same position: the rebuild re-elected every
    // canonical representative from the pool as it now stands.
    pool.cx_moves_.clear();
    pool.cx_edits_.clear();
  }

  // Embedding overcounting is now corrected at count time inside
  // count_embeddings_single via its `reacting_local` parameter (passed
  // from rs.reacting_local_a/_b for both single-mol and multi-mol
  // paths).  The previous compile-time pattern-automorphism correction
  // (compute_embedding_correction[_multimol]) duplicated that work and
  // halved the rate when both fired, so it has been removed.
  // embedding_correction_a / _b are kept at 1.0 throughout.

  void init_rule_states() {
    int const n_rules = static_cast<int>(model.rules.size());
    rule_states.resize(n_rules);
    bind_infos.resize(n_rules);
    rule_fire_counts.assign(n_rules, 0);

    // Precompute local_obs_indices: the observable slots that any local
    // function evaluates at its tag, resolved via observable_index.  MUST
    // run before the per-rule rescan loop below —
    // rescan_all_molecules_for_rule evaluates local-rate / FunctionProduct
    // propensities through evaluate_local_factor, which localizes exactly
    // these observable slots.  If this set were empty at rescan time, those
    // observables would be read globally and the initial propensity would
    // be wildly wrong (e.g. a free monomer's local count read as the
    // whole-system count).
    //
    // The loader classifies each of a local function's observables as
    // tagged or bare (issue #38); only the tagged ones belong here.
    {
      local_obs_scope_conflict_ = false;
      fn_local_obs_mask_.clear();
      std::unordered_set<int> idx_set;
      auto resolve = [&](const std::string& obs_name) {
        auto it = model.observable_index.find(obs_name);
        return it != model.observable_index.end() ? it->second : -1;
      };
      for (auto& gf : model.functions) {
        if (!gf.is_local())
          continue;
        for (auto& obs_name : gf.local_observable_names) {
          int const oi = resolve(obs_name);
          if (oi >= 0)
            idx_set.insert(oi);
        }
      }
      local_obs_indices.assign(idx_set.begin(), idx_set.end());

      // One slot per observable means the union can only stand in for the
      // per-function sets while they agree.  If some local function reads
      // an observable bare that another localizes, the override would
      // reach into the function that wants the global value; note it and
      // record the per-function sets so evaluate_local_factor can re-scope
      // around each function in the chain.
      for (auto& gf : model.functions) {
        if (!gf.is_local())
          continue;
        for (auto& obs_name : gf.global_observable_names) {
          int const oi = resolve(obs_name);
          if (oi >= 0 && idx_set.count(oi))
            local_obs_scope_conflict_ = true;
        }
      }
      if (local_obs_scope_conflict_) {
        fn_local_obs_mask_.assign(model.functions.size(),
                                  std::vector<char>(local_obs_indices.size(), 0));
        for (int i = 0; i < static_cast<int>(model.functions.size()); ++i) {
          auto& gf = model.functions[i];
          if (!gf.is_local())
            continue;
          for (auto& obs_name : gf.local_observable_names) {
            int const oi = resolve(obs_name);
            for (size_t k = 0; k < local_obs_indices.size(); ++k) {
              if (local_obs_indices[k] == oi)
                fn_local_obs_mask_[i][k] = 1;
            }
          }
        }
      }
    }

    for (int ri = 0; ri < n_rules; ++ri) {
      auto& rule = model.rules[ri];
      bind_infos[ri] = find_bind_info(rule);

      // Compute embedding overcounting corrections per reactant pattern
      auto& rs = rule_states[ri];
      int const start_a =
          (!rule.reactant_pattern_starts.empty()) ? rule.reactant_pattern_starts[0] : 0;
      int const end_a = (rule.reactant_pattern_starts.size() > 1)
                            ? rule.reactant_pattern_starts[1]
                            : static_cast<int>(rule.reactant_pattern.molecules.size());

      // Collect reacting-component LOCAL indices within the seed molecules of
      // each reactant pattern.  These drive embedding deduplication in
      // count_embeddings_single so that multiple injective mappings which
      // send the same reacting pattern components to the same molecule
      // components collapse to one physical reaction.  See the comment on
      // count_embeddings_single for the motivation (NFsim's checkForEquality).
      rs.reacting_local_a.clear();
      rs.reacting_local_b.clear();
      {
        std::unordered_set<int> flat_reacting;
        for (auto& op : rule.operations) {
          if (op.comp_flat >= 0)
            flat_reacting.insert(op.comp_flat);
          if (op.comp_flat_a >= 0)
            flat_reacting.insert(op.comp_flat_a);
          if (op.comp_flat_b >= 0)
            flat_reacting.insert(op.comp_flat_b);
        }
        int flat_base = 0;
        for (int mi = 0; mi < static_cast<int>(rule.reactant_pattern.molecules.size()); ++mi) {
          int const nc = static_cast<int>(rule.reactant_pattern.molecules[mi].components.size());
          if (mi == start_a) {
            for (int ci = 0; ci < nc; ++ci)
              if (flat_reacting.count(flat_base + ci))
                rs.reacting_local_a.push_back(ci);
          }
          if (rule.molecularity >= 2 && rule.reactant_pattern_starts.size() > 1 &&
              mi == rule.reactant_pattern_starts[1]) {
            for (int ci = 0; ci < nc; ++ci)
              if (flat_reacting.count(flat_base + ci))
                rs.reacting_local_b.push_back(ci);
          }
          flat_base += nc;
        }
      }

      // Pure-context reactant patterns (issue #33).  A reactant pattern that
      // NO operation in this rule touches is context: every embedding of it
      // yields the identical reaction, so BNG's network expansion emits one
      // reaction instance per matching COMPLEX rather than one per matching
      // molecule.  The derivation is shared with the loader, which reads the
      // same fact to decide whether an MM rule's symmetry_factor can be
      // attributed to a slot (issue #37) — see reactant_pattern_transforms in
      // model.hpp, including why an unresolved endpoint disables it.
      {
        ReactantTransforms const rt = reactant_pattern_transforms(rule);
        rs.per_complex_a = rt.is_context(0);
        rs.per_complex_b = rt.is_context(1);
        // These two flags are the TWO-SLOT path's, and only its: they gate
        // bookkeeping (mol_data, percx_a/percx_b) that a rule with three or
        // more reactant patterns never builds, because
        // rescan_all_molecules_for_rule hands such a rule to nary_rescan
        // first.  An n-ary rule's context slots are classified from the same
        // ReactantTransforms in nary_setup and carried on NaryState, one per
        // slot (issue #42).  Clearing here is what keeps a pure-context n-ary
        // slot from half-entering the two-slot path, where an empty tally read
        // as a count would take the rule to zero propensity.
        if (rule.reactant_pattern_starts.size() > 2 || rule.molecularity > 2) {
          rs.per_complex_a = false;
          rs.per_complex_b = false;
        }
        // Does the collapse also have to decide a RATE (issue #52)?  Only
        // when the slot's local factor is evaluated per molecule and is not
        // the constant 1 the DOR1 normalization puts on the untagged side: a
        // complex-wide tag reads the same observable at every molecule of the
        // complex, so every instance it collapsed prices identically and the
        // cheap lowest-id representative is as good as any.
        rs.percx_canon_a = rs.per_complex_a && rule.rate_law.is_local &&
                           !rule.rate_law.unity_factor_a && rule.rate_law.local_arg_is_molecule;
        rs.percx_canon_b = rs.per_complex_b && rule.rate_law.type == RateLawType::FunctionProduct &&
                           rule.rate_law.is_local_b && !rule.rate_law.unity_factor_b &&
                           rule.rate_law.local_arg_is_molecule_b;
      }

      // Relevant-component bitmask for the A-seed (P1 cache, step 1).
      // A component index ci is relevant iff the A-seed pattern molecule
      // has a PatternComponent that can match the type's component ci —
      // using the same matching rules as count_embeddings_single.  For
      // symmetric components (multiple same-named comps), all matching
      // bits are set, since the pattern can match any of them.
      {
        rs.a_relevant_mask = 0;
        rs.a_mask_complete = false;
        if (start_a < static_cast<int>(rule.reactant_pattern.molecules.size())) {
          auto& pm = rule.reactant_pattern.molecules[start_a];
          int const type_idx = pm.type_index;
          if (type_idx >= 0 && type_idx < static_cast<int>(model.molecule_types.size())) {
            int const n_type_comps =
                static_cast<int>(model.molecule_types[type_idx].components.size());
            if (n_type_comps <= 64) {
              rs.a_mask_complete = true;
              auto& mtype = model.molecule_types[type_idx];
              for (auto& pc : pm.components) {
                bool any_match = false;
                std::string base = pc.name;
                while (!base.empty() && std::isdigit(static_cast<unsigned char>(base.back())))
                  base.pop_back();
                for (int ci = 0; ci < n_type_comps; ++ci) {
                  const std::string& actual_name = mtype.components[ci].name;
                  bool name_match = false;
                  if (pc.comp_type_index >= 0 && pc.comp_type_index < n_type_comps) {
                    name_match = (ci == pc.comp_type_index) ||
                                 (actual_name == mtype.components[pc.comp_type_index].name);
                  }
                  if (!name_match)
                    name_match = (actual_name == pc.name);
                  if (!name_match)
                    name_match = (actual_name == base);
                  if (name_match) {
                    rs.a_relevant_mask |= (uint64_t{1} << ci);
                    any_match = true;
                  }
                }
                if (!any_match)
                  rs.a_mask_complete = false;
              }
            }
          }
        }
      }

      // For multi-molecule patterns, use full-match counting (scoped to
      // each reactant).  Embedding deduplication is done at count time
      // via count_embeddings_single's reacting_local parameter, applied
      // to the seed-side enumeration in count_multi_mol_fast_generic.
      // This subsumes the old pattern-automorphism correction for sym
      // non-reacting components — applying both would halve the rate.
      // Symmetric with the single-mol path below.
      rs.fm_a = FastMatchSlot{};
      rs.fm_b = FastMatchSlot{};
      rs.use_multi_mol_count = (end_a - start_a > 1);
      if (rs.use_multi_mol_count) {
        rs.embedding_correction_a = 1.0;
        rs.pat_adj_a = build_pattern_adjacency(rule.reactant_pattern, start_a, end_a);
        // Candidate-B eligibility: only for the 2-mol-1-bond-fc template and
        // only for rules that don't need complex-expansion.  Disjoint patterns
        // are explicitly excluded.
        {
          std::string fm_reason;
          build_fastmatch_slot(rule.reactant_pattern, start_a, end_a, start_a, rs.pat_adj_a, model,
                               rs.fm_a, kSelectReactantsProfile ? &fm_reason : nullptr);
          if constexpr (kSelectReactantsProfile) {
            if (!rs.fm_a.enabled) {
              int const n_mols = end_a - start_a;
              int n_bonds = 0;
              for (int mi = start_a; mi < end_a; ++mi) {
                if (mi < static_cast<int>(rs.pat_adj_a.adj.size()))
                  n_bonds += static_cast<int>(rs.pat_adj_a.adj[mi].size());
              }
              n_bonds /= 2;
              fprintf(stderr, "[fm_reject A] rule=%zu name=%s mols=%d bonds=%d reason=%s\n",
                      static_cast<size_t>(&rule - model.rules.data()), rule.name.c_str(), n_mols,
                      n_bonds, fm_reason.empty() ? "unknown" : fm_reason.c_str());
            }
          }
        }

        // Detect disjoint patterns: BFS from seed through pattern adjacency.
        // If any pattern molecule in [start_a, end_a) is unreachable, the
        // pattern has disjoint molecules whose count_a depends on full complex
        // contents — incremental_update must expand affected_mols for this rule.
        {
          std::vector<bool> reached(end_a, false);
          reached[start_a] = true;
          std::queue<int> q;
          q.push(start_a);
          while (!q.empty()) {
            int const cur = q.front();
            q.pop();
            for (auto& e : rs.pat_adj_a.adj[cur]) {
              if (e.other_mol >= start_a && e.other_mol < end_a && !reached[e.other_mol]) {
                reached[e.other_mol] = true;
                q.push(e.other_mol);
              }
            }
          }
          rs.needs_complex_expansion = false;
          for (int mi = start_a; mi < end_a; ++mi) {
            if (!reached[mi]) {
              rs.needs_complex_expansion = true;
              break;
            }
          }
        }
      } else {
        // Single-mol patterns: embedding deduplication is done at count time
        // via count_embeddings_single's reacting_local parameter, which
        // subsumes the old pattern-automorphism correction and also handles
        // shorthand patterns like L(s!+,s) on molecules with more components
        // than the pattern specifies.
        rs.embedding_correction_a = 1.0;
      }

      if (rule.molecularity >= 2 && rule.reactant_pattern_starts.size() > 1) {
        int const start_b = rule.reactant_pattern_starts[1];
        int const end_b = static_cast<int>(rule.reactant_pattern.molecules.size());
        rs.use_multi_mol_count_b = (end_b - start_b > 1);
        if (rs.use_multi_mol_count_b) {
          rs.embedding_correction_b = 1.0;
          rs.pat_adj_b = build_pattern_adjacency(rule.reactant_pattern, start_b, end_b);
          {
            std::string fm_reason;
            build_fastmatch_slot(rule.reactant_pattern, start_b, end_b, start_b, rs.pat_adj_b,
                                 model, rs.fm_b, kSelectReactantsProfile ? &fm_reason : nullptr);
            if constexpr (kSelectReactantsProfile) {
              if (!rs.fm_b.enabled) {
                int const n_mols = end_b - start_b;
                int n_bonds = 0;
                for (int mi = start_b; mi < end_b; ++mi) {
                  if (mi < static_cast<int>(rs.pat_adj_b.adj.size()))
                    n_bonds += static_cast<int>(rs.pat_adj_b.adj[mi].size());
                }
                n_bonds /= 2;
                fprintf(stderr, "[fm_reject B] rule=%zu name=%s mols=%d bonds=%d reason=%s\n",
                        static_cast<size_t>(&rule - model.rules.data()), rule.name.c_str(), n_mols,
                        n_bonds, fm_reason.empty() ? "unknown" : fm_reason.c_str());
              }
            }
          }
        } else {
          rs.embedding_correction_b = 1.0;
        }

        // Relevant-component bitmask for the B-seed (mirrors A-side above).
        rs.b_relevant_mask = 0;
        rs.b_mask_complete = false;
        if (start_b < static_cast<int>(rule.reactant_pattern.molecules.size())) {
          auto& pm = rule.reactant_pattern.molecules[start_b];
          int const type_idx = pm.type_index;
          if (type_idx >= 0 && type_idx < static_cast<int>(model.molecule_types.size())) {
            int const n_type_comps =
                static_cast<int>(model.molecule_types[type_idx].components.size());
            if (n_type_comps <= 64) {
              rs.b_mask_complete = true;
              auto& mtype = model.molecule_types[type_idx];
              for (auto& pc : pm.components) {
                bool any_match = false;
                std::string base = pc.name;
                while (!base.empty() && std::isdigit(static_cast<unsigned char>(base.back())))
                  base.pop_back();
                for (int ci = 0; ci < n_type_comps; ++ci) {
                  const std::string& actual_name = mtype.components[ci].name;
                  bool name_match = false;
                  if (pc.comp_type_index >= 0 && pc.comp_type_index < n_type_comps) {
                    name_match = (ci == pc.comp_type_index) ||
                                 (actual_name == mtype.components[pc.comp_type_index].name);
                  }
                  if (!name_match)
                    name_match = (actual_name == pc.name);
                  if (!name_match)
                    name_match = (actual_name == base);
                  if (name_match) {
                    rs.b_relevant_mask |= (uint64_t{1} << ci);
                    any_match = true;
                  }
                }
                if (!any_match)
                  rs.b_mask_complete = false;
              }
            }
          }
        }
      }

      nary_setup(ri);

      // The rescan below already elects representatives, so the election's
      // rank tables have to exist before it runs (issue #53).  Built at the
      // first rule that elects, which means a model without the construct
      // never builds them at all.
      if ((rs.percx_canon_a || rs.percx_canon_b) && canon_type_rank_.empty())
        build_canon_ranks();

      rescan_all_molecules_for_rule(ri);
    }

    // Build type→rule index
    int const n_types = static_cast<int>(model.molecule_types.size());
    type_to_rules.assign(n_types, {});
    dynamic_synthesis_rules.clear();
    dynamic_rate_rules.clear();
    global_dep_local_rules.clear();
    for (int ri = 0; ri < n_rules; ++ri) {
      auto& rule = model.rules[ri];
      if (rule.molecularity == 0) {
        if (rule.rate_law.is_dynamic)
          dynamic_synthesis_rules.push_back(ri);
        continue;
      }
      // Non-synthesis rules with dynamic (function/expression) rates need
      // propensity recomputation whenever rate-dependent observables change,
      // regardless of which molecule types were directly affected.
      if (rule.rate_law.is_dynamic && !rule.rate_law.is_local)
        dynamic_rate_rules.push_back(ri);
      // Local-rate counterpart (issue #38).  A local rate that reads a
      // moving global — bare observable, time, a global function over
      // either — changes for EVERY instance when that global moves, and no
      // molecule is marked affected by it.  Marking the rule needed is not
      // enough: incremental_update only revisits molecules in the affected
      // set, so each untouched instance would keep its stale per-molecule
      // rate.  These rules take a full rescan instead — on the events where
      // one of those globals really did move (issue #40); the watch starts
      // unprimed, so the first event after any init_rule_states rescans and
      // establishes the baseline.
      if (rule.rate_law.local_rate_tracks_global) {
        GlobalDepWatch w;
        w.rule_index = ri;
        w.always_dirty = rule.rate_law.global_dep_time || rule.rate_law.global_dep_opaque;
        w.obs_slots = rule.rate_law.global_dep_observables;
        w.last.assign(w.obs_slots.size(), 0.0);
        global_dep_local_rules.push_back(std::move(w));
      }
      int const seed_a =
          (!rule.reactant_pattern_starts.empty()) ? rule.reactant_pattern_starts[0] : 0;
      if (seed_a >= static_cast<int>(rule.reactant_pattern.molecules.size()))
        continue;
      int const type_a = rule.reactant_pattern.molecules[seed_a].type_index;
      if (type_a >= 0 && type_a < n_types)
        type_to_rules[type_a].push_back(ri);

      if (rule.molecularity >= 2 && rule.reactant_pattern_starts.size() > 1) {
        int const seed_b = rule.reactant_pattern_starts[1];
        if (seed_b < static_cast<int>(rule.reactant_pattern.molecules.size())) {
          int const type_b = rule.reactant_pattern.molecules[seed_b].type_index;
          if (type_b != type_a && type_b >= 0 && type_b < n_types)
            type_to_rules[type_b].push_back(ri);
        }
      }

      // Slots 2..n-1 of an n-ary rule (issue #24).  Without this a change to
      // a molecule matching only the third pattern would never reach the
      // rule, leaving its propensity stale.
      const auto& ns = rule_states[ri].nary;
      if (ns.enabled) {
        for (int slot = 2; slot < ns.n_slots; ++slot) {
          int const t = ns.slot_type_index[slot];
          if (t < 0 || t >= n_types)
            continue;
          auto& lst = type_to_rules[t];
          if (std::find(lst.begin(), lst.end(), ri) == lst.end())
            lst.push_back(ri);
        }
      }
    }
    rule_needed_buf.resize(n_rules, 0);

    // Compute max pattern diameter across all rules for BFS expansion limit
    max_pattern_depth = 0;
    for (int ri = 0; ri < n_rules; ++ri) {
      int const d = pattern_diameter(model.rules[ri].reactant_pattern);
      max_pattern_depth = std::max(d, max_pattern_depth);
    }

    // Bucket the pure-context slots by seed molecule type (issue #33), so the
    // per-event complex-move drain can skip a molecule whose type no rule
    // counts per complex — which in a model with no pure-context reactant at
    // all is every molecule, making the whole mechanism free there.
    percx_slots_by_type.assign(model.molecule_types.size(), {});
    any_percx = false;
    percx_canon_slots.clear();
    for (int ri = 0; ri < n_rules; ++ri) {
      auto& rule = model.rules[ri];
      auto& rs = rule_states[ri];
      // An n-ary rule carries its context slots on NaryState, and any number
      // of them (issue #42); the two-slot path carries at most the two.  The
      // drain reads the same bucket for both and dispatches on nary.enabled.
      if (rs.nary.enabled) {
        for (int slot = 0; slot < rs.nary.n_slots; ++slot) {
          if (rs.nary.slot_context[slot] == 0)
            continue;
          int const ti = rs.nary.slot_type_index[slot];
          if (ti < 0 || ti >= static_cast<int>(percx_slots_by_type.size()))
            continue;
          percx_slots_by_type[ti].emplace_back(ri, slot);
          any_percx = true;
        }
        continue;
      }
      for (int slot = 0; slot < 2; ++slot) {
        if (!(slot == 0 ? rs.per_complex_a : rs.per_complex_b))
          continue;
        int const seed =
            (slot == 0)
                ? ((!rule.reactant_pattern_starts.empty()) ? rule.reactant_pattern_starts[0] : 0)
                : ((rule.reactant_pattern_starts.size() > 1) ? rule.reactant_pattern_starts[1]
                                                             : -1);
        if (seed < 0 || seed >= static_cast<int>(rule.reactant_pattern.molecules.size()))
          continue;
        int const ti = rule.reactant_pattern.molecules[seed].type_index;
        if (ti < 0 || ti >= static_cast<int>(percx_slots_by_type.size()))
          continue;
        percx_slots_by_type[ti].emplace_back(ri, slot);
        any_percx = true;
        if (slot == 0 ? rs.percx_canon_a : rs.percx_canon_b)
          percx_canon_slots.emplace_back(ri, slot);
      }
    }
    // Only now, with the canonical slots known, does the pool start logging
    // per-complex edits — the list has no reader in any other model, and the
    // mark it hangs off is on every structural mutator.
    pool.track_cx_edits_ = !percx_canon_slots.empty();
    // The pool appends to its move side-channel unconditionally; if nothing
    // reads it, drop whatever accumulated during seeding.
    pool.cx_moves_.clear();
    pool.cx_edits_.clear();

    // Precompute have_local_rules_ flag (Fix 3)
    have_local_rules_ = false;
    for (auto& rs : rule_states) {
      if (rs.has_local_rates) {
        have_local_rules_ = true;
        break;
      }
    }

    // Precompute any_needs_complex_expansion_ flag
    any_needs_complex_expansion_ = false;
    for (auto& rs : rule_states) {
      if (rs.needs_complex_expansion) {
        any_needs_complex_expansion_ = true;
        break;
      }
    }

    // Establish a clean baseline for the delta-updated total_propensity.
    // The per-rule rescans above already credited each rs.propensity to
    // total_propensity via set_rule_propensity, but a fresh sum here makes
    // the post-init invariant unambiguous (and absorbs any prior stale
    // total left over from a re-init).
    recompute_total_propensity();
  }

  // --- N-ary reactant path (issues #24, #26) -------------------------------

  // Decide whether rule `rule_idx` takes the n-ary path, and if so build its
  // NaryState.  Uni- and bimolecular rules are left untouched on the
  // existing two-slot machinery.
  //
  // The path covers rules whose every reactant pattern is one connected
  // piece — `A + A + A`, `A + B + C`, `A(s,d!1).D(d!1) + A(s) + A(s)`, and so
  // on.  Shapes left out are refused at load time by scan_unsupported rather
  // than silently mis-simulated:
  //
  //   * a reactant pattern whose `.`-joined molecules are not bonded to each
  //     other (`A().B() + C + D`), which the sampler cannot place — see
  //     nary_slot_connected;
  //   * MM (NFsim itself requires exactly 2 reactants), local-function and
  //     FunctionProduct rate laws, all of which are defined on 1-2 slots.
  void nary_setup(int rule_idx) {
    auto& rule = model.rules[rule_idx];
    auto& ns = rule_states[rule_idx].nary;
    ns = NaryState{};

    int const n = rule.molecularity;
    if (n < 3)
      return;

    // Safety net for issue #24's actual failure mode.  A rule of >= 3
    // reactant patterns that does NOT get an n-ary state falls through to
    // the two-slot machinery, where slot B swallows patterns 2..n into one
    // bond-free pattern that scores zero embeddings — the rule holds zero
    // propensity and never fires, with mass still conserved, so the
    // trajectory looks valid.  That is precisely the silence this issue was
    // about.
    //
    // scan_unsupported() is expected to have refused such a model at load,
    // but it reaches that verdict from the raw XML while the decision here
    // is made on the parsed Rule.  Rather than trust the two to agree
    // forever, say so on stderr unconditionally: if the load-time check
    // ever misses a shape, the run is still loud instead of quietly wrong.
    // (A throw would be wrong — it would break the documented
    // --ignore-unsupported contract of running with the rule inert.)
    if (!nary_shape_supported(rule)) {
      fprintf(stderr,
              "WARN: rule '%s' (%s) has %d reactant patterns in a shape the "
              "engine does not implement — it will never fire. The rest of "
              "the model still simulates, and mass stays conserved, so the "
              "trajectory will look valid. See issue #24.\n",
              rule.id.c_str(), rule.name.c_str(), n);
      return;
    }

    int const n_mols = static_cast<int>(rule.reactant_pattern.molecules.size());

    ns.enabled = true;
    ns.n_slots = n;
    ns.counts.assign(n, {});
    ns.slot_start.assign(n, 0);
    ns.slot_end.assign(n, 0);
    ns.slot_multi.assign(n, 0);
    ns.slot_adj.assign(n, PatternAdj{});
    ns.slot_type_index.assign(n, -1);
    ns.reacting_local.assign(n, {});
    ns.subset_totals.assign(std::size_t{1} << n, 0.0);
    ns.slot_context.assign(n, 0);
    ns.percx.assign(n, PerCxTally{});
    ns.raw.assign(n, {});
    ns.cx_of.assign(n, {});
    ns.slot_totals.assign(n, 0.0);
    ns.partitions = build_nary_partitions(n);
    ns.fenwick.resize(n);
    ns.use_fenwick.assign(n, 0);

    for (int slot = 0; slot < n; ++slot) {
      ns.slot_start[slot] = rule.reactant_pattern_starts[slot];
      ns.slot_end[slot] = (slot + 1 < n) ? rule.reactant_pattern_starts[slot + 1] : n_mols;
      if (ns.slot_end[slot] - ns.slot_start[slot] > 1) {
        ns.slot_multi[slot] = 1;
        ns.any_multi = true;
        // Bond adjacency for the slot's own sub-range, as the two-slot path
        // builds pat_adj_a/_b.  Drives both count_multi_mol_fast and the
        // sampler's resolve_pattern_context.  (The 2-mol-1-bond FastMatchSlot
        // specialization is deliberately not wired up here: n-ary rules are
        // not a hot path, and the generic counter is the reference the
        // specialization is checked against.)
        ns.slot_adj[slot] =
            build_pattern_adjacency(rule.reactant_pattern, ns.slot_start[slot], ns.slot_end[slot]);
      }
    }

    // Reacting-component local indices per slot, mirroring the
    // reacting_local_a/_b computation for the two-slot path: they drive
    // embedding deduplication in count_embeddings_single so that mappings
    // differing only in which non-reacting component is targeted collapse
    // to one physical reaction.  Taken on the slot's seed molecule, which is
    // where both counters enumerate embeddings.
    std::unordered_set<int> flat_reacting;
    for (auto& op : rule.operations) {
      if (op.comp_flat >= 0)
        flat_reacting.insert(op.comp_flat);
      if (op.comp_flat_a >= 0)
        flat_reacting.insert(op.comp_flat_a);
      if (op.comp_flat_b >= 0)
        flat_reacting.insert(op.comp_flat_b);
    }
    for (int slot = 0; slot < n; ++slot) {
      int const mi = ns.slot_start[slot];
      auto& pm = rule.reactant_pattern.molecules[mi];
      ns.slot_type_index[slot] = pm.type_index;
      int flat_base = 0;
      for (int j = 0; j < mi; ++j)
        flat_base += static_cast<int>(rule.reactant_pattern.molecules[j].components.size());
      for (int ci = 0; ci < static_cast<int>(pm.components.size()); ++ci)
        if (flat_reacting.count(flat_base + ci))
          ns.reacting_local[slot].push_back(ci);
    }

    // Which slots are pure context (issue #42).  Read from the same
    // derivation the two-slot path uses, which already answers for every
    // reactant pattern rather than just the first two — engine.cpp simply
    // refused to act on slots 2..n-1 until this state existed to hold the
    // answer.  `is_context` is false for a rule whose operations could not
    // all be resolved to a reactant pattern, so an unresolved endpoint keeps
    // the conservative per-molecule count rather than halving a real rate.
    {
      ReactantTransforms const rt = reactant_pattern_transforms(rule);
      for (int slot = 0; slot < n; ++slot) {
        if (!rt.is_context(static_cast<size_t>(slot)))
          continue;
        ns.slot_context[slot] = 1;
        ns.any_context = true;
      }
    }
  }

  // Does the match a multi-molecule reactant pattern resolved to actually
  // stand up?  resolve_pattern_context walks bonds greedily out of the seed
  // and does not re-check a molecule it has already placed, so a pattern
  // with a ring can come back with a bond unsatisfied; nor does it require
  // the pattern's molecules to be distinct from one another.
  // count_multi_mol_fast checks both while counting, so checking them here
  // too keeps the set of acceptable seed embeddings the same size as the
  // count that weighted the draw — which is what makes the realized rate
  // exact.  Used by both the bimolecular and the n-ary sampler; `start` /
  // `end` bound the pattern's own molecules.
  bool slot_match_valid(const Rule& rule, int start, int end, const ReactionMatch& m) const {
    for (int i = start; i < end; ++i) {
      if (m.mol_ids[i] < 0)
        return false;
      for (int j = i + 1; j < end; ++j)
        if (m.mol_ids[i] == m.mol_ids[j])
          return false;
    }
    // Every pattern bond inside the slot must be realized between the
    // components the match picked.  The skip-then-test loop reads more
    // clearly than std::all_of over a lambda with two range guards.
    int const flat_start = rule.reactant_pattern.flat_index(start, 0);
    int const flat_end = rule.reactant_pattern.flat_index(end, 0);
    for (const auto& bond : rule.reactant_pattern.bonds) { // NOLINT(readability-use-anyofallof)
      if (bond.comp_flat_a < flat_start || bond.comp_flat_a >= flat_end)
        continue;
      if (bond.comp_flat_b < flat_start || bond.comp_flat_b >= flat_end)
        continue;
      int const ca = m.comp_ids[bond.comp_flat_a];
      int const cb = m.comp_ids[bond.comp_flat_b];
      if (ca < 0 || cb < 0 || pool.component(ca).bond_partner != cb)
        return false;
    }
    return true;
  }

  // Embeddings of slot `slot`'s reactant pattern seeded at molecule `mid`.
  // Single-molecule slots take the plain per-molecule counter; a `.`-joined
  // complex takes the same multi-molecule counter the bimolecular path uses
  // for its own slots (issue #26).
  int nary_slot_count(const Rule& rule, const NaryState& ns, int slot, int mid) {
    int const start = ns.slot_start[slot];
    if (ns.slot_multi[slot] == 0)
      return count_embeddings_single(pool, mid, rule.reactant_pattern.molecules[start], model,
                                     nullptr, &ns.reacting_local[slot]);
    return count_multi_mol_fast(pool, mid, rule.reactant_pattern, model, start, ns.slot_end[slot],
                                start, ns.slot_adj[slot], &cm_profile_, &cmm_fc_profile_,
                                /*fm=*/nullptr, &ns.reacting_local[slot]);
  }

  // --- Per-complex tallies for an n-ary rule's context slots (issue #42) ---
  //
  // The same tally and the same representative rule as the two-slot path,
  // against different storage: an n-ary rule never builds `mol_data` —
  // rescan_all_molecules_for_rule hands it straight to nary_rescan, before
  // that bookkeeping starts — so a slot's match count and its membership
  // record live in NaryState's own per-slot vectors instead.  `rate_sum` has
  // no counterpart at all: nary_shape_supported refuses a local function, so
  // there is no DOR case here to arbitrate.

  static void nary_percx_grow(NaryState& ns, int slot, int mid) {
    if (mid >= static_cast<int>(ns.raw[slot].size()))
      ns.raw[slot].resize(mid + 1, 0);
    if (mid >= static_cast<int>(ns.cx_of[slot].size()))
      ns.cx_of[slot].resize(mid + 1, -1);
  }

  // Is `mid` the representative of the complex this slot credits it to?  This
  // is c_j(mid) for a context slot — see the NaryState comment for why the
  // per-complex count is carried as an indicator on one molecule.
  static int nary_percx_indicator(const NaryState& ns, int slot, int mid) {
    if (mid < 0 || mid >= static_cast<int>(ns.cx_of[slot].size()))
      return 0;
    int const cx = ns.cx_of[slot][mid];
    if (cx < 0)
      return 0;
    auto it = ns.percx[slot].entries.find(cx);
    return (it != ns.percx[slot].entries.end() && it->second.rep == mid) ? 1 : 0;
  }

  // Lowest molecule id this slot counts inside `cx`.  Membership is read off
  // the tally's own record, exactly as percx_find_rep reads it, so a molecule
  // that has already been retired from the slot cannot be elected.
  int nary_percx_find_rep(const NaryState& ns, int slot, int cx) const {
    int best = -1;
    for (int const m : pool.molecules_in_complex(cx)) {
      if (m >= static_cast<int>(ns.cx_of[slot].size()))
        continue;
      if (ns.cx_of[slot][m] != cx)
        continue;
      if (best < 0 || m < best)
        best = m;
    }
    return best;
  }

  // Bring one context slot's tally in line with the pool for `mid`, appending
  // every molecule whose representative status changed to `moved` — the
  // caller owes those molecules a re-fold, since the indicator IS the count.
  // The raw match count is read, not recomputed: the complex-move drain calls
  // this for molecules a merge or split relocated without touching a bond of
  // theirs, and those still match exactly what they matched before.
  void nary_percx_sync(NaryState& ns, int slot, int mid, std::vector<int>& moved) const {
    nary_percx_grow(ns, slot, mid);
    PerCxTally& t = ns.percx[slot];
    int& cur = ns.cx_of[slot][mid];
    bool const live = mid < pool.molecule_count() && pool.molecule(mid).active;
    int const want = (live && ns.raw[slot][mid] > 0) ? pool.complex_of(mid) : -1;
    if (cur == want)
      return;

    if (cur >= 0) {
      int const from = cur;
      cur = -1; // before find_rep, which reads it — see percx_sync
      auto it = t.entries.find(from);
      if (it != t.entries.end()) {
        int const old_rep = it->second.rep;
        if (--it->second.hits > 0) {
          if (old_rep == mid) {
            it->second.rep = nary_percx_find_rep(ns, slot, from);
            moved.push_back(mid);
            if (it->second.rep >= 0)
              moved.push_back(it->second.rep);
          }
        } else {
          // Complex holds no match any more: swap-erase it out of the dense
          // list so the count stays list.size().
          int const pos = it->second.pos;
          int const back = t.list.back();
          t.list[pos] = back;
          t.list.pop_back();
          auto bit = t.entries.find(back);
          if (bit != t.entries.end())
            bit->second.pos = pos;
          t.entries.erase(from);
          if (old_rep >= 0)
            moved.push_back(old_rep);
        }
      }
    }

    cur = want;
    if (want < 0)
      return;

    PerCxTally::Entry& e = t.entries[want];
    if (e.hits == 0) {
      e.pos = static_cast<int>(t.list.size());
      t.list.push_back(want);
      e.rep = -1;
    }
    ++e.hits;
    if (e.rep < 0 || mid < e.rep) {
      if (e.rep >= 0)
        moved.push_back(e.rep);
      e.rep = mid;
      moved.push_back(mid);
    }
  }

  // Subtract / add one molecule's contribution to every subset total.  Every
  // count change is bracketed by these two, one molecule at a time, so a
  // representative moving between two molecules is two separate brackets and
  // never a half-applied one.
  static void nary_totals_sub(NaryState& ns, int mid) {
    uint32_t const n_masks = uint32_t{1} << ns.n_slots;
    for (uint32_t mask = 1; mask < n_masks; ++mask)
      ns.subset_totals[mask] -= nary_mol_product(ns, mid, mask);
  }
  static void nary_totals_add(NaryState& ns, int mid) {
    uint32_t const n_masks = uint32_t{1} << ns.n_slots;
    for (uint32_t mask = 1; mask < n_masks; ++mask)
      ns.subset_totals[mask] += nary_mol_product(ns, mid, mask);
  }

  // Write one slot count, keeping that slot's Fenwick sampler in step.
  void nary_set_count(NaryState& ns, int slot, int mid, int new_count) const {
    int const old_count = ns.counts[slot][mid];
    if (new_count == old_count)
      return;
    ns.counts[slot][mid] = new_count;
    if (ns.use_fenwick[slot] == 0)
      return;
    // Growing the tree in place would leave new ancestor nodes at zero,
    // silently truncating prefix sums — rebuild, as the a/b path does.
    if (mid >= ns.fenwick[slot].n) {
      int const new_n = std::max(mid + 1, ns.fenwick[slot].n * 2);
      ns.fenwick[slot].init(new_n);
      for (int const m : pool.molecules_of_type(ns.slot_type_index[slot])) {
        if (m < 0 || m >= pool.molecule_count() || !pool.molecule(m).active)
          continue;
        if (ns.count_for(slot, m) > 0)
          ns.fenwick[slot].update(m, ns.count_for(slot, m));
      }
      return;
    }
    ns.fenwick[slot].update(mid, new_count - old_count);
  }

  // Re-derive every context slot's indicator at one molecule and fold the
  // change into the subset totals.  Used for the molecules a representative
  // change moved, which are not otherwise part of the event at all.
  void nary_refold_context(NaryState& ns, int mid) const {
    if (mid < 0)
      return;
    for (int slot = 0; slot < ns.n_slots; ++slot)
      if (mid >= static_cast<int>(ns.counts[slot].size()))
        ns.counts[slot].resize(mid + 1, 0);
    nary_totals_sub(ns, mid);
    for (int slot = 0; slot < ns.n_slots; ++slot)
      if (ns.slot_context[slot] != 0)
        nary_set_count(ns, slot, mid, nary_percx_indicator(ns, slot, mid));
    nary_totals_add(ns, mid);
    for (int slot = 0; slot < ns.n_slots; ++slot)
      ns.slot_totals[slot] = ns.subset_totals[uint32_t{1} << slot];
  }

  // A molecule the pool moved between complexes, for every context slot its
  // type can seed on this rule.  The n-ary half of percx_sync_all_slots.
  void nary_percx_moved(int rule_idx, int slot, int mid) {
    auto& ns = rule_states[rule_idx].nary;
    nary_moved_scratch_.clear();
    nary_percx_sync(ns, slot, mid, nary_moved_scratch_);
    for (int const m : nary_moved_scratch_)
      nary_refold_context(ns, m);
    nary_moved_scratch_.clear();
  }

  std::vector<int> nary_moved_scratch_;

  // Full rescan of every slot's per-molecule counts, rebuilding the subset
  // totals from scratch.  Called from rescan_all_molecules_for_rule.
  void nary_rescan(int rule_idx) {
    auto& rule = model.rules[rule_idx];
    auto& rs = rule_states[rule_idx];
    auto& ns = rs.nary;

    int const pool_size = pool.molecule_count();
    for (int slot = 0; slot < ns.n_slots; ++slot)
      ns.counts[slot].assign(pool_size, 0);
    std::fill(ns.subset_totals.begin(), ns.subset_totals.end(), 0.0);
    std::fill(ns.slot_totals.begin(), ns.slot_totals.end(), 0.0);
    if (ns.any_context) {
      for (int slot = 0; slot < ns.n_slots; ++slot) {
        if (ns.slot_context[slot] == 0)
          continue;
        ns.percx[slot].clear();
        ns.raw[slot].assign(pool_size, 0);
        ns.cx_of[slot].assign(pool_size, -1);
      }
    }

    for (int slot = 0; slot < ns.n_slots; ++slot) {
      for (int const mid : pool.molecules_of_type(ns.slot_type_index[slot])) {
        if (mid < 0 || mid >= pool_size || !pool.molecule(mid).active)
          continue;
        int const c = nary_slot_count(rule, ns, slot, mid);
        if (ns.slot_context[slot] == 0) {
          ns.counts[slot][mid] = c;
          continue;
        }
        // Context slot: the embedding count decides tally membership, and
        // membership decides who represents the complex.  Counts are written
        // once the whole slot has been walked, since the representative is
        // not settled until then.
        ns.raw[slot][mid] = c;
        nary_percx_sync(ns, slot, mid, nary_moved_scratch_);
      }
      nary_moved_scratch_.clear();
      if (ns.slot_context[slot] == 0)
        continue;
      for (int const cx : ns.percx[slot].list) {
        auto it = ns.percx[slot].entries.find(cx);
        if (it != ns.percx[slot].entries.end() && it->second.rep >= 0)
          ns.counts[slot][it->second.rep] = 1;
      }
    }

    // T_B = Σ_m Π_{i∈B} c_i(m) over every non-empty subset.
    uint32_t const n_masks = uint32_t{1} << ns.n_slots;
    for (int mid = 0; mid < pool_size; ++mid) {
      for (uint32_t mask = 1; mask < n_masks; ++mask)
        ns.subset_totals[mask] += nary_mol_product(ns, mid, mask);
    }
    for (int slot = 0; slot < ns.n_slots; ++slot)
      ns.slot_totals[slot] = ns.subset_totals[uint32_t{1} << slot];

    // Per-slot Fenwick samplers for large populations, matching the
    // FENWICK_THRESHOLD policy the two-slot path uses.
    for (int slot = 0; slot < ns.n_slots; ++slot) {
      ns.fenwick[slot].clear();
      ns.use_fenwick[slot] = 0;
      if (static_cast<int>(pool.molecules_of_type(ns.slot_type_index[slot]).size()) <=
          FENWICK_THRESHOLD)
        continue;
      ns.fenwick[slot].init(std::max(pool_size, 1));
      ns.use_fenwick[slot] = 1;
      for (int const mid : pool.molecules_of_type(ns.slot_type_index[slot])) {
        if (mid < 0 || mid >= pool_size || !pool.molecule(mid).active)
          continue;
        if (ns.counts[slot][mid] > 0)
          ns.fenwick[slot].update(mid, ns.counts[slot][mid]);
      }
    }
  }

  // Π_{i ∈ mask} c_i(mid).
  static double nary_mol_product(const NaryState& ns, int mid, uint32_t mask) {
    double p = 1.0;
    for (int slot = 0; slot < ns.n_slots; ++slot) {
      if ((mask & (uint32_t{1} << slot)) == 0)
        continue;
      int const c = ns.count_for(slot, mid);
      if (c == 0)
        return 0.0;
      p *= c;
    }
    return p;
  }

  // Recompute every slot count for one molecule and fold the delta into the
  // subset totals.  This is the incremental counterpart of nary_rescan; cost
  // is 2^n multiply-adds rather than a pool scan.
  void nary_update_mol(int rule_idx, int mid) {
    auto& rule = model.rules[rule_idx];
    auto& rs = rule_states[rule_idx];
    auto& ns = rs.nary;
    if (mid < 0)
      return;

    for (int slot = 0; slot < ns.n_slots; ++slot) {
      if (mid >= static_cast<int>(ns.counts[slot].size()))
        ns.counts[slot].resize(mid + 1, 0);
    }

    bool const live = (mid < pool.molecule_count()) && pool.molecule(mid).active;

    // Context slots settle their tallies first (issue #42).  The count this
    // molecule carries for such a slot is not its embedding count but whether
    // it represents its complex, and that is not knowable until membership is
    // in: a molecule that stops matching hands the representation to another
    // molecule, which is not part of this event and would otherwise keep a
    // stale zero.  Collect those here, apply them below.
    nary_moved_scratch_.clear();
    if (ns.any_context) {
      for (int slot = 0; slot < ns.n_slots; ++slot) {
        if (ns.slot_context[slot] == 0)
          continue;
        nary_percx_grow(ns, slot, mid);
        ns.raw[slot][mid] = (live && pool.molecule(mid).type_index == ns.slot_type_index[slot])
                                ? nary_slot_count(rule, ns, slot, mid)
                                : 0;
        nary_percx_sync(ns, slot, mid, nary_moved_scratch_);
      }
    }

    nary_totals_sub(ns, mid);
    for (int slot = 0; slot < ns.n_slots; ++slot) {
      int new_count = 0;
      if (ns.any_context && ns.slot_context[slot] != 0)
        new_count = nary_percx_indicator(ns, slot, mid);
      else if (live && pool.molecule(mid).type_index == ns.slot_type_index[slot])
        new_count = nary_slot_count(rule, ns, slot, mid);
      nary_set_count(ns, slot, mid, new_count);
    }
    nary_totals_add(ns, mid);

    for (int const m : nary_moved_scratch_)
      if (m != mid)
        nary_refold_context(ns, m);
    nary_moved_scratch_.clear();

    for (int slot = 0; slot < ns.n_slots; ++slot)
      ns.slot_totals[slot] = ns.subset_totals[uint32_t{1} << slot];
  }

  // Draw one molecule for `slot`, weighted by its embedding count.
  int nary_sample_slot(const NaryState& ns, int slot) {
    double const total = ns.slot_totals[slot];
    if (total <= 0)
      return -1;

    if (ns.use_fenwick[slot] != 0) {
      double const r = uniform() * total;
      int const mid = ns.fenwick[slot].find(r);
      if (mid >= 0 && mid < pool.molecule_count() && pool.molecule(mid).active &&
          ns.count_for(slot, mid) > 0)
        return mid;
      // Fenwick can land off the end on rounding; fall through to the scan.
    }

    double r = uniform() * total;
    for (int const mid : pool.molecules_of_type(ns.slot_type_index[slot])) {
      if (mid < 0 || mid >= pool.molecule_count() || !pool.molecule(mid).active)
        continue;
      double const w = ns.count_for(slot, mid);
      if (w <= 0)
        continue;
      r -= w;
      if (r <= 0)
        return mid;
    }
    // Floating-point drift past the end: return the last eligible molecule.
    for (auto it = pool.molecules_of_type(ns.slot_type_index[slot]).rbegin();
         it != pool.molecules_of_type(ns.slot_type_index[slot]).rend(); ++it) {
      int const mid = *it;
      if (mid >= 0 && mid < pool.molecule_count() && pool.molecule(mid).active &&
          ns.count_for(slot, mid) > 0)
        return mid;
    }
    return -1;
  }

  // --- Pure-context per-complex tallies (issue #33) -------------------------
  //
  // Every mutation of a pure-context slot's membership funnels through
  // percx_sync(), which compares the tally's own record of where a molecule is
  // counted (PerMolRuleData::percx_cx_*) against where it should be counted
  // now.  Both callers — a recomputed match count, and a molecule whose
  // complex changed underneath it — are the same operation from the tally's
  // point of view, so there is one path and not two.

  // Which (rule, slot) pairs count per complex, bucketed by the slot's seed
  // molecule type.  Empty for every type in a model whose rules all transform
  // their reactants, which is what keeps the drain below free there.
  std::vector<std::vector<std::pair<int, int>>> percx_slots_by_type;
  bool any_percx = false;
  std::vector<int> percx_moved_scratch;

  // The (rule, slot) pairs whose representative comes from the complex's
  // canonical form (issue #52).  Empty for every model that does not put a
  // per-molecule local function tag on a pure-context reactant pattern, and
  // that emptiness is what keeps the canonicalizer off the event loop there.
  std::vector<std::pair<int, int>> percx_canon_slots;
  std::vector<int> percx_edit_scratch;

  // The canonicalizer's integer entry point and everything it needs
  // (issue #53).  Electing a representative asks the canonicalizer for one
  // permutation and throws the rest away; going through ComplexGraph to get
  // it meant building a std::string per molecule and per bonded component
  // and interning them through a std::map, on every edit of a tagged
  // complex.  These describe the same complex in ranks instead, so the
  // election costs no allocation at all once the buffers are warm.
  //
  // The ranks are NOT the engine's own indices: they have to sort the way
  // the color strings do or the two entry points would elect different
  // molecules.  canonical::rank_* is what derives them, and it is deliberately
  // the only place that correspondence is written down.  Built by
  // build_canon_ranks() when the first canonical slot is registered, and
  // read-only afterwards — molecule types cannot change mid-session.
  std::vector<int> canon_type_rank_;  // model type_index -> rank
  std::vector<int> canon_comp_first_; // model type_index -> base into the two below
  std::vector<int> canon_comp_name_rank_;
  std::vector<int> canon_state_first_;    // flat comp index -> base into ranks
  std::vector<int> canon_state_rank_;     // flat (comp, state_index) -> rank
  int canon_stateless_rank_ = 0;          // the rank of "", for a stateless component
  mutable canonical::Workspace canon_ws_; // reused refinement buffers
  mutable canonical::RankedComplex canon_rc_;
  mutable std::vector<int> canon_order_;
  mutable std::vector<int> canon_comp_slot_; // global comp id -> index within canon_rc_

  // The rebuild below elects representatives the cheap way and lets
  // percx_resum_rates canonicalize each complex once at the end, rather than
  // once per molecule that joins the tally on the way there.
  bool percx_rebuilding = false;

  static PerCxTally& tally_of(RuleState& rs, int slot) {
    return (slot != 0) ? rs.percx_b : rs.percx_a;
  }
  static int& percx_cx_of(PerMolRuleAux& ax, int slot) {
    return (slot != 0) ? ax.percx_cx_b : ax.percx_cx_a;
  }
  static bool percx_canon_of(const RuleState& rs, int slot) {
    return (slot != 0) ? rs.percx_canon_b : rs.percx_canon_a;
  }
  static int count_of(const PerMolRuleData& md, int slot) {
    return (slot != 0) ? md.count_b : md.count_a;
  }
  // The local rate priced for this molecule on this slot.  Zero for a rule
  // with no local function, which keeps rate_sum trivially 0 there.
  static double percx_rate_of(const RuleState& rs, const PerMolRuleAux& ax, int slot) {
    if (!rs.has_local_rates)
      return 0.0;
    return (slot != 0) ? ax.local_rate_b : ax.local_rate;
  }

  // Re-sum `cx`'s representative rate into rate_sum.  Called on every rep
  // change and whenever the sitting rep's local rate is recomputed.
  static void percx_refresh_rate(RuleState& rs, int slot, int cx) {
    PerCxTally& t = tally_of(rs, slot);
    auto it = t.entries.find(cx);
    if (it == t.entries.end())
      return;
    double const nv = (it->second.rep >= 0 && it->second.rep < static_cast<int>(rs.mol_aux.size()))
                          ? percx_rate_of(rs, rs.mol_aux[it->second.rep], slot)
                          : 0.0;
    t.rate_sum += nv - it->second.rep_rate;
    it->second.rep_rate = nv;
  }

  // Is molecule `m` one of the matches this slot currently credits to `cx`?
  // Membership is read off the tally's own record rather than off the match
  // count, so a molecule that has moved but not yet been re-synced cannot be
  // promoted into a complex it is no longer credited to.
  static bool percx_holds(RuleState& rs, int slot, int cx, int m) {
    return m >= 0 && m < static_cast<int>(rs.mol_aux.size()) &&
           percx_cx_of(rs.mol_aux[m], slot) == cx;
  }

  // Rank every name in the model the way the canonical color strings order
  // them, so the election can describe a complex in integers (issue #53).
  // Model-wide rather than per-complex: only the RELATIVE order of the
  // ranks matters, and a table that covers every name covers every subset
  // of them.  Run once, off the event loop.
  void build_canon_ranks() {
    int const n_types = static_cast<int>(model.molecule_types.size());
    std::vector<std::string> type_names;
    type_names.reserve(n_types);
    for (const auto& mt : model.molecule_types)
      type_names.push_back(mt.name);
    canon_type_rank_ = canonical::rank_molecule_type_names(type_names);

    // Component names and state strings both flatten across the whole model:
    // two molecule types sharing a component name compare their states
    // against each other inside one color string, so the state universe has
    // to be global too.  "" goes in last and is the stateless rank.
    canon_comp_first_.assign(n_types + 1, 0);
    std::vector<std::string> comp_names;
    std::vector<std::string> state_names;
    std::vector<int> state_first;
    for (int ti = 0; ti < n_types; ++ti) {
      canon_comp_first_[ti] = static_cast<int>(comp_names.size());
      for (const auto& ct : model.molecule_types[ti].components) {
        comp_names.push_back(ct.name);
        state_first.push_back(static_cast<int>(state_names.size()));
        for (const auto& s : ct.allowed_states)
          state_names.push_back(s);
      }
    }
    canon_comp_first_[n_types] = static_cast<int>(comp_names.size());
    state_first.push_back(static_cast<int>(state_names.size()));
    state_names.emplace_back(); // the stateless "" — ranks first, by construction

    canon_comp_name_rank_ = canonical::rank_component_names(comp_names);
    canon_state_rank_ = canonical::rank_state_names(state_names);
    canon_stateless_rank_ = canon_state_rank_.back();
    canon_state_first_ = std::move(state_first);
  }

  // Restate `mol_ids` — one connected complex — as a RankedComplex, in
  // buffers the next call reuses.  The integer twin of extract_complex; the
  // two must describe isomorphic graphs, or the election would answer for a
  // different complex than the label does.
  void build_ranked_complex(const std::vector<int>& mol_ids) const {
    canon_rc_.clear();
    // Grown, never refilled: the slot of a component outside this complex is
    // never read, so stale entries cost nothing, and growing leaves the
    // vector's geometric reallocation to absorb a pool that keeps expanding.
    if (static_cast<int>(canon_comp_slot_.size()) < pool.component_slot_count())
      canon_comp_slot_.resize(pool.component_slot_count(), -1);
    for (int const mid : mol_ids) {
      const auto& mol = pool.molecule(mid);
      int const base = canon_comp_first_[mol.type_index];
      canon_rc_.molecules.push_back({canon_type_rank_[mol.type_index], canon_rc_.component_count(),
                                     static_cast<int>(mol.comp_ids.size())});
      for (int li = 0; li < static_cast<int>(mol.comp_ids.size()); ++li) {
        int const cid = mol.comp_ids[li];
        int const si = pool.component(cid).state_index;
        int const sbase = canon_state_first_[base + li];
        int const state_rank = (si >= 0 && sbase + si < canon_state_first_[base + li + 1])
                                   ? canon_state_rank_[sbase + si]
                                   : canon_stateless_rank_;
        canon_comp_slot_[cid] = canon_rc_.component_count();
        canon_rc_.components.push_back({canon_comp_name_rank_[base + li], state_rank, -1});
      }
    }
    // Every component of the complex now has a slot, so the bond pass can
    // resolve both endpoints without a lookup table — a complex is closed
    // under its own bonds.  Each bond is written once, from its lower-id end.
    for (int const mid : mol_ids) {
      for (int const cid : pool.molecule(mid).comp_ids) {
        int const partner = pool.component(cid).bond_partner;
        if (partner > cid) {
          int const a = canon_comp_slot_[cid];
          int const b = canon_comp_slot_[partner];
          canon_rc_.components[a].partner = b;
          canon_rc_.components[b].partner = a;
        }
      }
    }
  }

  // Lowest live molecule id this slot counts inside `cx`.
  int percx_lowest_rep(RuleState& rs, int slot, int cx) const {
    int best = -1;
    for (int const m : pool.molecules_in_complex(cx)) {
      if (!percx_holds(rs, slot, cx, m))
        continue;
      if (best < 0 || m < best)
        best = m;
    }
    return best;
  }

  // The molecule that represents `cx` for this slot.  Only runs when the
  // sitting rep drops out or the complex is edited, which is rare next to the
  // count updates.
  //
  // Lowest live matching id, unless the slot prices per molecule — then which
  // one is observable, and the answer has to be a property of the species
  // rather than of pool history, so it comes from the complex's canonical
  // ordering (issue #52).  `mol_order` walks the complex in the order the
  // canonical label writes it, so the first member it names that this slot
  // still holds is the canonically-first match; that is the molecule BNG2's
  // network expansion evaluates the tagged observable at.
  //
  // `hits` is the caller's count of matches inside `cx`.  At one match there
  // is no election to hold — the two rules name the same molecule — so the
  // canonicalizer is not asked, which is what keeps a single-subunit context
  // pattern (the common shape) on the integer path even when the rule prices
  // per molecule.
  int percx_find_rep(RuleState& rs, int slot, int cx, int hits) const {
    if (hits <= 1 || !percx_canon_of(rs, slot))
      return percx_lowest_rep(rs, slot, cx);
    const auto& members = pool.molecules_in_complex(cx);
    // The integer entry point answers whenever refinement alone separates
    // the complex's molecules, which is all but the genuinely symmetric ones
    // (issue #53).  Where it declines — a ring, a homo-oligomer — the
    // canonical order is the one belonging to the lexicographically minimal
    // render, so only the strings can pick it and the ComplexGraph path
    // stands; the two answer identically either way.
    build_ranked_complex(members);
    if (!canonical::canonical_order_fast(canon_rc_, canon_ws_, canon_order_))
      canon_order_ = canonical::canonical_order(extract_complex(pool, model, members));
    for (int const pos : canon_order_) {
      if (pos < 0 || pos >= static_cast<int>(members.size()))
        continue;
      if (percx_holds(rs, slot, cx, members[pos]))
        return members[pos];
    }
    return -1;
  }

  void percx_add(RuleState& rs, int slot, int cx, int mid) const {
    PerCxTally& t = tally_of(rs, slot);
    PerCxTally::Entry& e = t.entries[cx];
    if (e.hits == 0) {
      e.pos = static_cast<int>(t.list.size());
      t.list.push_back(cx);
      e.rep = -1;
      e.rep_rate = 0.0;
    }
    ++e.hits;
    if (percx_canon_of(rs, slot) && !percx_rebuilding && e.hits > 1) {
      // Canonical order is not id order, so a joiner can displace the sitting
      // rep from either side and there is no cheap compare to make: re-derive.
      int const r = percx_find_rep(rs, slot, cx, e.hits);
      if (r != e.rep) {
        e.rep = r;
        percx_refresh_rate(rs, slot, cx);
      }
      return;
    }
    if (e.rep < 0 || mid < e.rep) {
      e.rep = mid;
      percx_refresh_rate(rs, slot, cx);
    }
  }

  void percx_remove(RuleState& rs, int slot, int cx, int mid) const {
    PerCxTally& t = tally_of(rs, slot);
    auto it = t.entries.find(cx);
    if (it == t.entries.end())
      return;
    if (--it->second.hits > 0) {
      if (it->second.rep == mid) {
        // The representative dropped out.  Re-derive it from the complex's
        // remaining members; this is the only O(complex) step in the tally,
        // and it runs on rep loss rather than on every count change.
        it->second.rep = percx_find_rep(rs, slot, cx, it->second.hits);
        percx_refresh_rate(rs, slot, cx);
      }
      return;
    }
    // Complex no longer holds a match: drop it out of the dense list with a
    // swap-erase so the uniform draw stays O(1).  Values are copied out first
    // — the map write below may rehash and invalidate `it`.
    int const pos = it->second.pos;
    double const rate = it->second.rep_rate;
    int const back = t.list.back();
    t.list[pos] = back;
    t.list.pop_back();
    auto bit = t.entries.find(back);
    if (bit != t.entries.end())
      bit->second.pos = pos;
    t.rate_sum -= rate;
    t.entries.erase(cx);
  }

  // Bring the tally's record for (rule slot, molecule) in line with the pool.
  void percx_sync(int ri, int slot, int mid) {
    auto& rs = rule_states[ri];
    if (mid < 0 || mid >= static_cast<int>(rs.mol_data.size()) ||
        mid >= static_cast<int>(rs.mol_aux.size()))
      return;
    auto& md = rs.mol_data[mid];
    int& cur = percx_cx_of(rs.mol_aux[mid], slot);
    bool const live = mid < pool.molecule_count() && pool.molecule(mid).active;
    int const want = (live && count_of(md, slot) > 0) ? pool.complex_of(mid) : -1;
    if (cur == want)
      return;
    if (cur >= 0) {
      // Retire the molecule's own membership record BEFORE the tally
      // re-derives the representative.  percx_find_rep decides who is
      // eligible by reading exactly this record, so leaving it standing lets
      // the molecule on its way out be re-elected representative of the
      // complex it is leaving.  Nothing then corrects it: the count is still
      // right (the complex does hold another match), so the slot keeps its
      // propensity, but every draw returns a molecule that no longer matches
      // and null-events, and that complex silently stops reacting until a
      // full rescan.  Measured at 1.8x too slow on a catalytic rule whose
      // dimeric enzyme loses its lower-id subunit to a state change.
      int const from = cur;
      cur = -1;
      percx_remove(rs, slot, from, mid);
    }
    cur = want;
    if (want >= 0)
      percx_add(rs, slot, want, mid);
  }

  // A molecule whose complex membership changed, for every pure-context slot
  // its type can seed.
  void percx_sync_all_slots(int mid) {
    if (mid < 0 || mid >= pool.molecule_count())
      return;
    // A dead molecule keeps its last type_index, which is what we want: its
    // stale hit still has to come out of the slots that were counting it.
    int const ti = pool.molecule(mid).type_index;
    if (ti < 0 || ti >= static_cast<int>(percx_slots_by_type.size()))
      return;
    for (auto& [ri, slot] : percx_slots_by_type[ti]) {
      if (rule_states[ri].nary.enabled)
        nary_percx_moved(ri, slot, mid);
      else
        percx_sync(ri, slot, mid);
    }
  }

  // Drain the pool's complex-move side-channel.  Runs once per event, before
  // any propensity is read.
  void percx_drain_moves() {
    if (!any_percx) {
      pool.cx_moves_.clear();
      pool.cx_edits_.clear();
      return;
    }
    pool.consume_cx_moves(percx_moved_scratch);
    for (int const mid : percx_moved_scratch)
      percx_sync_all_slots(mid);
    percx_moved_scratch.clear();
    percx_reelect_edited();
  }

  // Re-elect the representative of every complex this event edited, on the
  // slots whose representative is canonical (issue #52).
  //
  // The membership drain above cannot stand in for this.  A state change
  // inside a complex — the `m~0 -> m~1` of the issue's reproducer — moves no
  // molecule between complexes and need not change which molecules match, yet
  // it reorders the canonical form and so hands the complex to a different
  // representative.  Nothing else in the event loop would notice: the count
  // is unchanged, so the tally is never asked a question.
  //
  // Costs one canonicalization per (edited complex, canonical slot) pair per
  // event, and runs at all only for a model that has such a slot.  That is
  // the same order as the work `have_local_rules_` already does for such a
  // model, which re-evaluates a local function at every member of every
  // edited complex.
  void percx_reelect_edited() {
    if (percx_canon_slots.empty()) {
      pool.cx_edits_.clear();
      return;
    }
    pool.consume_cx_edits(percx_edit_scratch);
    std::sort(percx_edit_scratch.begin(), percx_edit_scratch.end());
    percx_edit_scratch.erase(std::unique(percx_edit_scratch.begin(), percx_edit_scratch.end()),
                             percx_edit_scratch.end());
    for (auto& [ri, slot] : percx_canon_slots) {
      auto& rs = rule_states[ri];
      PerCxTally& t = tally_of(rs, slot);
      for (int const cx : percx_edit_scratch) {
        auto it = t.entries.find(cx);
        if (it == t.entries.end())
          continue; // no match of this slot lives there
        if (it->second.hits <= 1)
          continue; // one candidate: the edit cannot have changed the answer
        int const r = percx_find_rep(rs, slot, cx, it->second.hits);
        if (r == it->second.rep)
          continue;
        it->second.rep = r;
        percx_refresh_rate(rs, slot, cx);
      }
    }
    percx_edit_scratch.clear();
  }

  // Rebuild one slot's tally from the current pool and match counts.
  void percx_rebuild(int ri, int slot) {
    auto& rs = rule_states[ri];
    PerCxTally& t = tally_of(rs, slot);
    t.clear();
    for (auto& ax : rs.mol_aux)
      percx_cx_of(ax, slot) = -1;
    auto& rule = model.rules[ri];
    int const seed =
        (slot != 0)
            ? ((rule.reactant_pattern_starts.size() > 1) ? rule.reactant_pattern_starts[1] : -1)
            : ((!rule.reactant_pattern_starts.empty()) ? rule.reactant_pattern_starts[0] : 0);
    if (seed < 0 || seed >= static_cast<int>(rule.reactant_pattern.molecules.size()))
      return;
    // Seed membership with the cheap representative and let percx_resum_rates
    // canonicalize each complex once at the end.  Electing canonically here
    // instead would canonicalize once per molecule that joins — the same
    // complex, once per subunit — for the identical answer.
    percx_rebuilding = true;
    for (int const mid : pool.molecules_of_type(rule.reactant_pattern.molecules[seed].type_index))
      percx_sync(ri, slot, mid);
    percx_rebuilding = false;
  }

  // Re-sum every complex's representative rate.  The from-scratch rescan
  // computes per-molecule local rates AFTER it rebuilds membership, so
  // rate_sum has to be settled once those are in hand; the incremental path
  // keeps it current through percx_refresh_rate instead.
  //
  // This is also where a canonical slot's representatives are elected after a
  // rebuild — one canonicalization per matching complex, which is what the
  // rebuild deferred to here.  Rebuild and resum are always called as a pair
  // on the canonical slots: percx_canon_a/_b imply a local-rate law, and
  // every branch of rescan_all_molecules_for_rule that reads a local rate
  // resums the slots it rebuilt.
  void percx_resum_rates(int ri, int slot) {
    auto& rs = rule_states[ri];
    PerCxTally& t = tally_of(rs, slot);
    bool const canon = percx_canon_of(rs, slot);
    t.rate_sum = 0.0;
    for (int const cx : t.list) {
      auto& e = t.entries[cx];
      if (canon && e.hits > 1)
        e.rep = percx_find_rep(rs, slot, cx, e.hits);
      e.rep_rate = (e.rep >= 0 && e.rep < static_cast<int>(rs.mol_aux.size()))
                       ? percx_rate_of(rs, rs.mol_aux[e.rep], slot)
                       : 0.0;
      t.rate_sum += e.rep_rate;
    }
  }

  // Effective reactant counts for compute_propensity: the complex count for a
  // pure-context slot, -1 (meaning "use the per-molecule total") otherwise.
  double percx_a(int ri) const {
    return rule_states[ri].per_complex_a ? static_cast<double>(rule_states[ri].percx_a.size())
                                         : -1.0;
  }
  double percx_b(int ri) const {
    return rule_states[ri].per_complex_b ? static_cast<double>(rule_states[ri].percx_b.size())
                                         : -1.0;
  }

  // Match count of rule `ri`'s reactant pattern `slot` (0-based) — the same
  // number compute_propensity multiplies the rate by, so a rate law that
  // reads `reactant_N()` and one that does not are priced off one count
  // (issue #59).  Out-of-range slots read 0; the loader refuses a rule whose
  // rate law asks for more reactants than it has, so that is a floor and not
  // a reachable answer.
  double reactant_slot_count(int ri, int slot) const {
    const auto& rs = rule_states[ri];
    if (slot < 0 || slot >= model.rules[ri].molecularity)
      return 0.0;
    if (rs.nary.enabled)
      return (slot < static_cast<int>(rs.nary.slot_totals.size())) ? rs.nary.slot_totals[slot]
                                                                   : 0.0;
    if (slot == 0) {
      double const cx = percx_a(ri);
      return (cx >= 0.0) ? cx : (rs.a_total / rs.embedding_correction_a);
    }
    double const cx = percx_b(ri);
    return (cx >= 0.0) ? cx : (rs.b_total / rs.embedding_correction_b);
  }

  // Write rule `ri`'s reactant counts into the `reactant_N()` slots, so the
  // next expression evaluation resolves them.  Passing ri = -1 clears them,
  // which is what the settle pass leaves behind: outside a rule the
  // placeholder stands for nothing.
  void bind_reactant_counts(int ri) {
    for (auto const& [n, slot] : eval_reactant_slots)
      eval_vars_flat[slot] = (ri >= 0) ? reactant_slot_count(ri, n - 1) : 0.0;
  }

  // Does this rule's shape ever touch a PerMolRuleAux field?  The union of
  // what writes each of the three groups:
  //   - `molecularity >= 2` is the only thing that writes the
  //     shared-component split (a_only / b_only / ab_both), in the rescan
  //     and in incremental_update;
  //   - a local rate law writes local_rate / local_propensity, and the
  //     FunctionProduct (DOR2) law writes the _b pair as well;
  //   - a pure-context slot writes percx_cx_a / percx_cx_b.
  // Read once per rescan, before either table is sized.
  static bool rule_needs_mol_aux(const Rule& rule, const RuleState& rs) {
    return rule.molecularity >= 2 || rule.rate_law.is_local ||
           rule.rate_law.type == RateLawType::FunctionProduct || rs.per_complex_a ||
           rs.per_complex_b;
  }

  // The value a full rescan fills `mol_data` with: all counts zero, and the
  // P1 cache flag already set because the rescan is what validates the row.
  static PerMolRuleData rescanned_row() {
    PerMolRuleData md;
    md.cache_init = true;
    return md;
  }

  // Grow both per-molecule tables to cover `mid`, keeping `mol_aux` exactly
  // as long as `mol_data` for a rule that has one.  `proto` is the row the
  // new `mol_data` entries take — `rescanned_row()` from inside a full
  // rescan, a default row from the on-demand paths, which is the difference
  // between "computed and zero" and "not computed yet".
  static void grow_mol_tables(RuleState& rs, int mid, const PerMolRuleData& proto) {
    rs.mol_data.resize(mid + 1, proto);
    if (rs.needs_mol_aux)
      rs.mol_aux.resize(mid + 1, PerMolRuleAux{});
  }

  void rescan_all_molecules_for_rule(int rule_idx) {
    auto& rule = model.rules[rule_idx];
    auto& rs = rule_states[rule_idx];
    auto& bi = bind_infos[rule_idx];

    // >= 3 reactant patterns (issue #24): the two-slot bookkeeping below
    // cannot represent this rule — slot B would swallow patterns 2..n — so
    // the n-ary path owns its counts and propensity outright.
    if (rs.nary.enabled) {
      nary_rescan(rule_idx);
      double const rate = evaluate_rate(rule);
      set_rule_propensity(rs, compute_propensity(rs, rule, rate));
      return;
    }

    rs.a_total = 0;
    rs.b_total = 0;
    rs.a_only_total = 0;
    rs.b_only_total = 0;
    rs.ab_both_total = 0;
    rs.ab_both_sq_total = 0;

    // Seed molecule for reactant pattern A
    int const seed_a =
        (!rule.reactant_pattern_starts.empty()) ? rule.reactant_pattern_starts[0] : 0;
    int const seed_b =
        (rule.reactant_pattern_starts.size() > 1) ? rule.reactant_pattern_starts[1] : -1;

    if (seed_a >= static_cast<int>(rule.reactant_pattern.molecules.size())) {
      // No reactant pattern molecule to seed on, so this rule has no
      // per-molecule state at all — leave mol_data empty rather than
      // sizing and zeroing one 80-byte entry per molecule in the pool
      // for a table nothing will read (373 MB on a 4.7e6-molecule pool,
      // per rule of this shape, at every session build).  The n-ary
      // early-out above already returns without building mol_data on
      // the same reasoning.  Every indexed read of mol_data elsewhere
      // either bounds-checks against its size or resizes first, so an
      // empty table is the same "no entry for this molecule" answer a
      // zeroed one gives.
      rs.mol_data.clear();
      rs.mol_aux.clear();
      rs.needs_mol_aux = false;
      // Synthesis rule (molecularity=0): propensity is just the rate
      if (rule.molecularity == 0) {
        double const rate = evaluate_rate(rule);
        set_rule_propensity(rs, rate);
      } else {
        set_rule_propensity(rs, 0);
      }
      return;
    }

    // Settle the shape predicate before either table is sized: it decides
    // whether the wide half is allocated at all, and every later site that
    // grows one table reads it to decide whether to grow the other.
    rs.needs_mol_aux = rule_needs_mol_aux(rule, rs);

    // Filled with the P1 cache flag already set.  A full rescan leaves every
    // entry a valid snapshot of the current pool — including the zero-valued
    // defaults for molecules of types this rule cannot seed on — so the flag
    // is part of the value the table is filled with rather than a second
    // pass over it afterwards, which on a 4.7e6-molecule pool was a whole
    // extra walk writing one byte per row (issue #71).  `PerMolRuleData{}`'s
    // own default stays false: the on-demand `resize(mid + 1)` in
    // incremental_update adds an entry for a molecule born since the rescan,
    // which genuinely has not been computed.
    rs.mol_data.assign(pool.molecule_count(), rescanned_row());
    if (rs.needs_mol_aux)
      rs.mol_aux.assign(pool.molecule_count(), PerMolRuleAux{});
    else
      rs.mol_aux.clear();

    auto& pm_a = rule.reactant_pattern.molecules[seed_a];

    // Scan all molecules of the matching type
    for (int const mid : pool.molecules_of_type(pm_a.type_index)) {
      if (!pool.molecule(mid).active)
        continue;
      if (mid >= static_cast<int>(rs.mol_data.size()))
        grow_mol_tables(rs, mid, rescanned_row());

      int count_a;
      if (rs.use_multi_mol_count) {
        int const end_a = (rule.reactant_pattern_starts.size() > 1)
                              ? rule.reactant_pattern_starts[1]
                              : static_cast<int>(rule.reactant_pattern.molecules.size());
        count_a = count_multi_mol_fast(pool, mid, rule.reactant_pattern, model, seed_a, end_a,
                                       seed_a, rs.pat_adj_a, &cm_profile_, &cmm_fc_profile_,
                                       &rs.fm_a, &rs.reacting_local_a);
      } else {
        count_a = count_embeddings_single(pool, mid, pm_a, model, nullptr, &rs.reacting_local_a);
      }
      rs.mol_data[mid].count_a = count_a;
      rs.a_total += count_a;
    }

    // For bimolecular rules, scan second reactant pattern
    if (rule.molecularity >= 2 && seed_b >= 0 &&
        seed_b < static_cast<int>(rule.reactant_pattern.molecules.size())) {
      auto& pm_b = rule.reactant_pattern.molecules[seed_b];

      for (int const mid : pool.molecules_of_type(pm_b.type_index)) {
        if (!pool.molecule(mid).active)
          continue;
        if (mid >= static_cast<int>(rs.mol_data.size()))
          grow_mol_tables(rs, mid, rescanned_row());

        int count_b;
        if (rs.use_multi_mol_count_b) {
          int const end_b = static_cast<int>(rule.reactant_pattern.molecules.size());
          count_b = count_multi_mol_fast(pool, mid, rule.reactant_pattern, model, seed_b, end_b,
                                         seed_b, rs.pat_adj_b, &cm_profile_, &cmm_fc_profile_,
                                         &rs.fm_b, &rs.reacting_local_b);
        } else {
          count_b = count_embeddings_single(pool, mid, pm_b, model, nullptr, &rs.reacting_local_b);
        }
        rs.mol_data[mid].count_b = count_b;
        rs.b_total += count_b;
      }

      // Compute shared components for overcounting
      // Need to scan all molecule types that could match either pattern
      std::unordered_set<int> scanned;
      for (int const mid : pool.molecules_of_type(pm_a.type_index)) {
        if (!pool.molecule(mid).active)
          continue;
        auto& md = rs.mol_data[mid];
        auto& ax = rs.mol_aux[mid];
        if (md.count_a > 0 && md.count_b > 0) {
          compute_shared_components(pool, mid, rule, model, seed_a, seed_b, bi.bind_local_a,
                                    bi.bind_local_b, bi.seed_mol_a, bi.seed_mol_b, ax.a_only,
                                    ax.b_only, ax.ab_both);
        } else {
          ax.a_only = md.count_a;
          ax.b_only = md.count_b;
          ax.ab_both = 0;
        }
        rs.a_only_total += ax.a_only;
        rs.b_only_total += ax.b_only;
        rs.ab_both_total += ax.ab_both;
        if (rule.same_components)
          rs.ab_both_sq_total += ax.ab_both * ax.ab_both;
        scanned.insert(mid);
      }
      // Also scan type B molecules not already scanned
      if (pm_a.type_index != pm_b.type_index) {
        for (int const mid : pool.molecules_of_type(pm_b.type_index)) {
          if (scanned.count(mid) || !pool.molecule(mid).active)
            continue;
          auto& md = rs.mol_data[mid];
          auto& ax = rs.mol_aux[mid];
          ax.a_only = md.count_a;
          ax.b_only = md.count_b;
          ax.ab_both = 0;
          rs.a_only_total += ax.a_only;
          rs.b_only_total += ax.b_only;
        }
      }
    }

    // Initialize Fenwick trees for large-population types
    rs.use_fenwick_a = false;
    rs.use_fenwick_b = false;
    rs.fenwick_a.clear();
    rs.fenwick_b.clear();
    if (static_cast<int>(pool.molecules_of_type(pm_a.type_index).size()) > FENWICK_THRESHOLD) {
      int const cap = pool.molecule_count();
      rs.fenwick_a.init(cap);
      rs.use_fenwick_a = true;
      for (int const mid : pool.molecules_of_type(pm_a.type_index)) {
        if (!pool.molecule(mid).active)
          continue;
        if (mid < static_cast<int>(rs.mol_data.size()) && rs.mol_data[mid].count_a > 0)
          rs.fenwick_a.update(mid, rs.mol_data[mid].count_a);
      }
    }
    if (rule.molecularity >= 2 && seed_b >= 0 &&
        seed_b < static_cast<int>(rule.reactant_pattern.molecules.size())) {
      auto& pm_b_fw = rule.reactant_pattern.molecules[seed_b];
      if (static_cast<int>(pool.molecules_of_type(pm_b_fw.type_index).size()) > FENWICK_THRESHOLD) {
        int const cap = pool.molecule_count();
        rs.fenwick_b.init(cap);
        rs.use_fenwick_b = true;
        for (int const mid : pool.molecules_of_type(pm_b_fw.type_index)) {
          if (!pool.molecule(mid).active)
            continue;
          if (mid < static_cast<int>(rs.mol_data.size()) && rs.mol_data[mid].count_b > 0)
            rs.fenwick_b.update(mid, rs.mol_data[mid].count_b);
        }
      }
    }

    // Compute propensity
    rs.has_local_rates = rule.rate_law.is_local;
    // Pure-context slots price complexes, so their tallies have to be standing
    // before the propensity is read.  Rebuilt rather than delta-updated here:
    // this whole routine is already a from-scratch O(N) rescan.
    if (rs.per_complex_a)
      percx_rebuild(rule_idx, 0);
    if (rs.per_complex_b)
      percx_rebuild(rule_idx, 1);
    double new_propensity;
    if (rule.rate_law.type == RateLawType::FunctionProduct) {
      // FunctionProduct (DOR2): propensity = S1·S2 where S1 = Σ_a w_a·f1(a)
      // over reactant-A matches and S2 = Σ_b w_b·f2(b) over reactant-B
      // matches.  This is the local-rate analogue of an ordinary bimolecular
      // rule's `a_eff·b_eff·k`; same-molecule pairs are rejected by the
      // sampler as null events (exact for distinct-type reactants).
      rs.local_propensity_total = 0;   // S1 (A side)
      rs.local_propensity_b_total = 0; // S2 (B side)
      auto& pm_a_loc = rule.reactant_pattern.molecules[seed_a];
      for (int const mid : pool.molecules_of_type(pm_a_loc.type_index)) {
        if (!pool.molecule(mid).active)
          continue;
        if (mid >= static_cast<int>(rs.mol_data.size()))
          continue;
        auto& md = rs.mol_data[mid];
        auto& ax = rs.mol_aux[mid];
        if (md.count_a > 0) {
          ax.local_rate = std::max<double>(evaluate_local_rate(rule, mid), 0);
          ax.local_propensity = (md.count_a / rs.embedding_correction_a) * ax.local_rate;
        } else {
          ax.local_rate = 0;
          ax.local_propensity = 0;
        }
        rs.local_propensity_total += ax.local_propensity;
      }
      if (seed_b >= 0 && seed_b < static_cast<int>(rule.reactant_pattern.molecules.size())) {
        auto& pm_b_loc = rule.reactant_pattern.molecules[seed_b];
        for (int const mid : pool.molecules_of_type(pm_b_loc.type_index)) {
          if (!pool.molecule(mid).active)
            continue;
          if (mid >= static_cast<int>(rs.mol_data.size()))
            continue;
          auto& md = rs.mol_data[mid];
          auto& ax = rs.mol_aux[mid];
          if (md.count_b > 0) {
            ax.local_rate_b = std::max<double>(evaluate_local_rate_b(rule, mid), 0);
            ax.local_propensity_b = (md.count_b / rs.embedding_correction_b) * ax.local_rate_b;
          } else {
            ax.local_rate_b = 0;
            ax.local_propensity_b = 0;
          }
          rs.local_propensity_b_total += ax.local_propensity_b;
        }
      }
      {
        if (rs.per_complex_a)
          percx_resum_rates(rule_idx, 0);
        if (rs.per_complex_b)
          percx_resum_rates(rule_idx, 1);
        double const s1 = rs.per_complex_a ? rs.percx_a.rate_sum : rs.local_propensity_total;
        double const s2 = rs.per_complex_b ? rs.percx_b.rate_sum : rs.local_propensity_b_total;
        new_propensity = s1 * s2 * dor_symmetry(rule);
      }
    } else if (rs.has_local_rates && rule.molecularity <= 1) {
      // Local-rate rule: propensity = sum of per-molecule (count_a * local_rate)
      rs.local_propensity_total = 0;
      int const seed_a_loc =
          (!rule.reactant_pattern_starts.empty()) ? rule.reactant_pattern_starts[0] : 0;
      auto& pm_a_loc = rule.reactant_pattern.molecules[seed_a_loc];
      for (int const mid : pool.molecules_of_type(pm_a_loc.type_index)) {
        if (!pool.molecule(mid).active)
          continue;
        if (mid >= static_cast<int>(rs.mol_data.size()))
          continue;
        auto& md = rs.mol_data[mid];
        auto& ax = rs.mol_aux[mid];
        if (md.count_a > 0) {
          ax.local_rate = evaluate_local_rate(rule, mid);
          ax.local_rate = std::max<double>(ax.local_rate, 0);
          ax.local_propensity = (md.count_a / rs.embedding_correction_a) * ax.local_rate;
        } else {
          ax.local_rate = 0;
          ax.local_propensity = 0;
        }
        rs.local_propensity_total += ax.local_propensity;
      }
      if (rs.per_complex_a) {
        percx_resum_rates(rule_idx, 0);
        new_propensity = rs.percx_a.rate_sum * dor_symmetry(rule);
      } else {
        new_propensity = rs.local_propensity_total * dor_symmetry(rule);
      }
    } else {
      double const rate = evaluate_rate(rule);
      new_propensity = compute_propensity(rs, rule, rate, percx_a(rule_idx), percx_b(rule_idx));
    }
    set_rule_propensity(rs, new_propensity);
  }

  // --- Rate evaluation ---

  double evaluate_rate(const Rule& rule) {
    if constexpr (kExprEvalProfile) {
      expr_eval_profile_.evaluate_rate_calls++;
      if (rule.rate_law.is_dynamic) {
        expr_eval_profile_.evaluate_rate_dynamic++;
        if (expr_eval_profile_.evaluate_rate_dynamic % kExprEvalProfileSampleEvery == 0) {
          expr_eval_profile_.sampled_rate_calls++;
          auto const t0 = std::chrono::steady_clock::now();
          double const r = evaluate_rate_impl(rule);
          expr_eval_profile_.evaluate_rate_ns +=
              static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - t0)
                                        .count());
          return r;
        }
      }
    }
    return evaluate_rate_impl(rule);
  }

  double evaluate_rate_impl(const Rule& rule) {
    if (!rule.rate_law.is_dynamic)
      return rule.rate_law.rate_value;

    update_eval_vars();

    // `reactant_N()` stands for this rule's own reactant counts, so bind
    // them before the expression is evaluated (issue #59).  `rule` is always
    // an element of model.rules — every call site indexes through it — so
    // recovering its index by pointer arithmetic is well-defined here, the
    // same trick set_rule_propensity uses on rule_states.
    if (rule.rate_law.max_reactant_count_index > 0)
      bind_reactant_counts(static_cast<int>(&rule - model.rules.data()));

    if (!rule.rate_law.function_name.empty()) {
      auto fit = model.function_index.find(rule.rate_law.function_name);
      if (fit != model.function_index.end()) {
        int const gi = fit->second;
        auto& gf = model.functions[gi];
        if (gf.is_tfun && gf.tfun) {
          double const ctr_val = get_tfun_counter_value(gf);
          double const tfun_val = gf.tfun->evaluate(ctr_val);
          if (gf_expr_id_[gi] >= 0) {
            // Patch the magic __tfun_NAME__ slot for this rule's
            // own tfun result (update_eval_vars wrote it from the
            // function's own counter, but BNG2 emits the rate-law
            // wrapper expecting it pre-patched on each call).
            eval_vars_flat[eval_gf_tfun_slot[gi]] = tfun_val;
            return expr_eval_.evaluate(gf_expr_id_[gi]);
          }
          return tfun_val;
        }
        if (gf_expr_id_[gi] >= 0)
          return expr_eval_.evaluate(gf_expr_id_[gi]);
      }
    }

    return rule.rate_law.rate_value;
  }

  double get_tfun_counter_value(const GlobalFunction& gf) {
    switch (gf.tfun_counter_source) {
    case TfunCounterSource::Time:
      return current_time;
    case TfunCounterSource::Parameter: {
      auto it = model.parameters.find(gf.tfun_counter_name);
      return (it != model.parameters.end()) ? it->second : 0.0;
    }
    case TfunCounterSource::Observable: {
      for (int i = 0; i < static_cast<int>(model.observables.size()); ++i) {
        if (model.observables[i].name == gf.tfun_counter_name)
          return (i < static_cast<int>(obs_values.size())) ? obs_values[i] : 0.0;
      }
      return 0.0;
    }
    case TfunCounterSource::Function: {
      auto fit = model.function_index.find(gf.tfun_counter_name);
      if (fit != model.function_index.end() && gf_expr_id_[fit->second] >= 0) {
        update_eval_vars();
        return expr_eval_.evaluate(gf_expr_id_[fit->second]);
      }
      return 0.0;
    }
    default:
      return 0.0;
    }
  }

  // Refresh the flat variable layout against the current model state.
  // Short-circuits when the cached layout is already up to date for
  // the current generation — within a single event's
  // incremental_update there are typically many evaluate_rate calls
  // and the global state doesn't change between them, so all but the
  // first are no-ops.  Parameters are NOT refreshed here: they're
  // constant for the lifetime of the engine and were written into
  // eval_vars_flat once by init_eval_layout.
  //
  // Function evaluation order matches the engine's internal
  // declaration order — BNG2 emits in dependency order, so earlier
  // functions are settled before later ones reference them.  The
  // seen-gen update is deferred to AFTER the loop so a recursive
  // call (TFUN counter source = Function) still does a full rebuild
  // and sees later function values written, matching prior behavior.
  void update_eval_vars() {
    if constexpr (kExprEvalProfile)
      expr_eval_profile_.update_eval_vars_calls++;
    if (eval_vars_seen_gen == eval_vars_gen)
      return;
    if constexpr (kExprEvalProfile) {
      expr_eval_profile_.update_eval_vars_rebuilds++;
      if (expr_eval_profile_.update_eval_vars_rebuilds % kExprEvalProfileSampleEvery == 0) {
        expr_eval_profile_.sampled_uev_calls++;
        auto const t0 = std::chrono::steady_clock::now();
        rebuild_eval_vars();
        expr_eval_profile_.update_eval_vars_ns +=
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count());
        return;
      }
    }
    rebuild_eval_vars();
  }

  // The rebuild body of update_eval_vars: runs only on a generation-cache
  // miss.  Split out so the K-sampled chrono in update_eval_vars brackets
  // the real work and not the (common) short-circuit return.
  void rebuild_eval_vars() {
    eval_vars_flat[eval_time_slot] = current_time;

    for (int i = 0; i < static_cast<int>(model.observables.size()); ++i) {
      eval_vars_flat[eval_obs_slot[i]] =
          (i < static_cast<int>(obs_values.size())) ? obs_values[i] : 0.0;
    }

    for (int i = 0; i < static_cast<int>(model.functions.size()); ++i) {
      auto& gf = model.functions[i];
      if (gf.is_tfun && gf.tfun) {
        if constexpr (kExprEvalProfile)
          expr_eval_profile_.tfun_evals++;
        double const ctr = get_tfun_counter_value(gf);
        double const val = gf.tfun->evaluate(ctr);
        eval_vars_flat[eval_gf_main_slot[i]] = val;
        eval_vars_flat[eval_gf_tfun_slot[i]] = val;
      } else if (gf_expr_id_[i] >= 0) {
        if constexpr (kExprEvalProfile)
          expr_eval_profile_.global_fn_ast_evals++;
        // A rate law that reads `reactant_N()` is only meaningful against
        // the rule that owns it, so settle it against that rule's counts
        // rather than against whatever was last bound (issue #59).
        if (!eval_reactant_slots.empty() && gf_reactant_owner_rule_[i] >= 0 &&
            gf_reactant_owner_rule_[i] < static_cast<int>(rule_states.size()))
          bind_reactant_counts(gf_reactant_owner_rule_[i]);
        try {
          eval_vars_flat[eval_gf_main_slot[i]] = expr_eval_.evaluate(gf_expr_id_[i]);
        } catch (...) {
          eval_vars_flat[eval_gf_main_slot[i]] = 0.0;
        }
      }
    }

    // The placeholders themselves have no value outside a rule; leave them
    // at zero so a settled function table reads the same however the rules
    // above happen to be ordered.
    if (!eval_reactant_slots.empty())
      bind_reactant_counts(-1);

    eval_vars_seen_gen = eval_vars_gen;
  }

  // Settle every function against the current observable values and
  // return the subset exposed to embedders (global, i.e. non-local,
  // functions in declaration order).  Bumps eval_vars_gen so the settle
  // runs even when the cached layout looks current: the caller — a
  // sample point or an explicit get_function_values() query — needs
  // values reflecting the just-refreshed obs_values, and the debug
  // invariant path of refresh_observables_for_sample does not bump the
  // generation itself.  The settle order matches update_eval_vars, so
  // functions that reference earlier functions see settled inputs.
  std::vector<double> compute_function_values() {
    if (output_function_indices.empty())
      return {};
    ++eval_vars_gen;
    update_eval_vars();
    std::vector<double> vals;
    vals.reserve(output_function_indices.size());
    for (int const fi : output_function_indices)
      vals.push_back(eval_vars_flat[eval_gf_main_slot[fi]]);
    return vals;
  }

  // Evaluate an arbitrary BNGL expression against the current session
  // state (issue #9 §1).  A fresh ExprTkEvaluator is built per call:
  // this is a paused-session query so the compile cost is acceptable,
  // and `extra` may carry names outside the model's symbol namespace
  // that the engine's shared `expr_eval_` does not bind.
  double evaluate_expression(const std::string& expr,
                             const std::unordered_map<std::string, double>& extra) {
    // Refresh derived state so observables / functions reflect the
    // current (possibly between-event) pool — same staleness reasoning
    // as get_observable_values() / get_function_values().
    compute_observables();
    ++eval_vars_gen;
    update_eval_vars();

    // Collect every resolvable name, model symbols first, `extra` last
    // (so an explicit override shadows a model symbol on a name clash).
    std::unordered_map<std::string, double> values;
    for (const auto& name : model.parameter_names_ordered) {
      auto it = model.parameters.find(name);
      values[name] = (it != model.parameters.end()) ? it->second : 0.0;
    }
    values["t"] = current_time;
    for (int i = 0; i < static_cast<int>(model.observables.size()); ++i)
      values[model.observables[i].name] = eval_vars_flat[eval_obs_slot[i]];
    for (int const fi : output_function_indices)
      values[model.functions[fi].name] = eval_vars_flat[eval_gf_main_slot[fi]];
    for (const auto& [name, val] : extra)
      values[name] = val;

    // Bind each name by address into a fresh evaluator.  `slots` is
    // sized up front and never grows, so the addresses stay valid for
    // the compile / evaluate below.
    bngsim::ExprTkEvaluator ev;
    ev.set_time_ptr(&current_time);
    std::vector<double> slots;
    std::vector<std::string> names;
    slots.reserve(values.size());
    names.reserve(values.size());
    for (const auto& [name, val] : values) {
      names.push_back(name);
      slots.push_back(val);
    }
    for (size_t i = 0; i < names.size(); ++i)
      ev.define_variable(names[i], &slots[i]);

    int id = -1;
    try {
      id = ev.compile(expr);
    } catch (const std::exception& e) {
      throw std::runtime_error("evaluate_expression: cannot compile '" + expr + "': " + e.what());
    }
    return ev.evaluate(id);
  }

  // --- Observable evaluation ---

  // Compute all observables (for output sampling and initialization).
  // Compute observable values.  `skip_tracked` is used by the sample
  // path: tracked obs (Species dirty-cx + Molecules per-mid) are kept
  // in sync incrementally and must not be overwritten here.  Untracked
  // obs fall through to the existing full-walk evaluator.
  void compute_observables(bool skip_tracked = false) {
    ++eval_vars_gen; // obs_values about to change → invalidate cached eval_vars_flat
    obs_values.resize(model.observables.size(), 0.0);
    if constexpr (kRecordAtProfile) {
      if (rap_profile_.obs.size() != model.observables.size()) {
        rap_profile_.obs.assign(model.observables.size(), RecordAtProfile::PerObs{});
        for (int oi = 0; oi < static_cast<int>(model.observables.size()); ++oi) {
          auto& po = rap_profile_.obs[oi];
          const auto& ob = model.observables[oi];
          po.name = ob.name;
          po.type = ob.type;
          po.pat_count = static_cast<int>(ob.patterns.size());
          if (!ob.patterns.empty() && !ob.patterns[0].relation.empty() &&
              ob.patterns[0].quantity >= 0) {
            po.has_quantity = true;
            po.quantity_relation = ob.patterns[0].relation;
            po.quantity_value = ob.patterns[0].quantity;
          }
        }
      }
    }
    for (int oi = 0; oi < static_cast<int>(model.observables.size()); ++oi) {
      if (skip_tracked && oi < static_cast<int>(incr_obs_is_tracked.size()) &&
          incr_obs_is_tracked[oi]) {
        continue;
      }
      if constexpr (kRecordAtProfile) {
        if (rap_profile_.inside_record_at) {
          rap_profile_.active_oi = oi;
          rap_profile_.obs[oi].evaluate_calls++;
          auto eo_t0 = std::chrono::steady_clock::now();
          obs_values[oi] = evaluate_observable(model.observables[oi]);
          auto eo_t1 = std::chrono::steady_clock::now();
          rap_profile_.obs[oi].evaluate_ns +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(eo_t1 - eo_t0).count();
        } else {
          obs_values[oi] = evaluate_observable(model.observables[oi]);
        }
      } else {
        obs_values[oi] = evaluate_observable(model.observables[oi]);
      }
    }
    if constexpr (kRecordAtProfile)
      rap_profile_.active_oi = -1;
  }

  // Sample-time observable refresh.  Flushes Species-tracked dirty
  // complexes, then full-walks only the untracked observables.  When
  // the invariant gate is on, verifies tracked obs_values by
  // re-computing full-walk into a reference vector and aborts on
  // mismatch (P6/P7-style gate).
  void refresh_observables_for_sample() {
    flush_species_incr_observables();
    if constexpr (kSpeciesIncrObsInvariant) {
      std::vector<double> ref(obs_values); // starts with incremental values
      for (int oi = 0; oi < static_cast<int>(model.observables.size()); ++oi) {
        ref[oi] = evaluate_observable(model.observables[oi]);
      }
      for (int oi = 0; oi < static_cast<int>(model.observables.size()); ++oi) {
        if (oi >= static_cast<int>(incr_obs_is_tracked.size()))
          break;
        if (!incr_obs_is_tracked[oi])
          continue;
        if (obs_values[oi] != ref[oi]) {
          auto& ob = model.observables[oi];
          fprintf(stderr,
                  "[species_incr_invariant] mismatch at t=%.17g oi=%d "
                  "name=%s type=%s incr=%.17g full_walk=%.17g\n",
                  current_time, oi, ob.name.c_str(), ob.type.c_str(), obs_values[oi], ref[oi]);
          if (incr_obs_is_species[oi]) {
            fprintf(stderr, "[species_incr_invariant] cx_passed dump:\n");
            auto& pass_map = obs_cx_passed[oi];
            auto& mc_map = obs_cx_match_count[oi];
            // Print all stored cx entries
            for (auto& kv : pass_map) {
              int const cx = kv.first;
              int const stored_mc = mc_map.count(cx) ? mc_map[cx] : -1;
              auto members = pool.molecules_in_complex(cx);
              int live_total = 0;
              auto& pat = ob.patterns[0];
              int const pm_type = pat.molecules[0].type_index;
              for (int const m : members) {
                if (!pool.molecule(m).active)
                  continue;
                if (pool.molecule(m).type_index != pm_type)
                  continue;
                int const c = count_multi_molecule_embeddings(pool, m, pat, model);
                live_total += c;
              }
              bool const live_pass = species_quantity_passes(pat, live_total);
              if ((kv.second != 0) != live_pass) {
                fprintf(stderr,
                        "  cx=%d stored_pass=%d stored_mc=%d "
                        "live_total=%d live_pass=%d members=%zu\n",
                        cx, static_cast<int>(kv.second), stored_mc, live_total,
                        static_cast<int>(live_pass), members.size());
              }
            }
          }
          std::abort();
        }
      }
      // Non-tracked obs need their full-walk values copied in.
      for (int oi = 0; oi < static_cast<int>(model.observables.size()); ++oi) {
        if (oi < static_cast<int>(incr_obs_is_tracked.size()) && incr_obs_is_tracked[oi])
          continue;
        obs_values[oi] = ref[oi];
      }
    } else {
      compute_observables(/*skip_tracked=*/use_incremental_obs);
    }
  }

  // Compute only rate-dependent observables (after each SSA event).
  void compute_rate_dependent_observables() {
    ++eval_vars_gen;
    obs_values.resize(model.observables.size(), 0.0);
    for (int oi = 0; oi < static_cast<int>(model.observables.size()); ++oi) {
      if (model.observables[oi].rate_dependent)
        obs_values[oi] = evaluate_observable(model.observables[oi]);
    }
  }

  // Apply a Species pattern's quantity predicate to a complex-wide
  // match count.  Mirrors the full-walk logic at evaluate_observable:
  // an empty or zero-count complex never passes, and an unconstrained
  // pattern passes whenever match_count > 0.
  static bool species_quantity_passes(const Pattern& pat, int match_count) {
    if (match_count <= 0)
      return false;
    if (pat.relation.empty() || pat.quantity < 0)
      return true;
    if (pat.relation == "==")
      return match_count == pat.quantity;
    if (pat.relation == ">=")
      return match_count >= pat.quantity;
    if (pat.relation == "<=")
      return match_count <= pat.quantity;
    if (pat.relation == ">")
      return match_count > pat.quantity;
    if (pat.relation == "<")
      return match_count < pat.quantity;
    return false;
  }

  // Initialize incremental observable tracking.  Every observable that
  // uses single-mol patterns is eligible; Molecules obs take the
  // per-mid delta path and Species obs take the per-cx dirty-set path.
  // Rate-dependent and sample-output obs are both tracked.
  void init_incremental_observables() {
    int const n_obs = static_cast<int>(model.observables.size());
    int const n_mols = pool.molecule_count();

    // Reset all tracker state.
    rate_dep_obs_indices.clear();
    incr_tracked_obs_indices.clear();
    incr_obs_is_species.assign(n_obs, 0);
    incr_obs_is_tracked.assign(n_obs, 0);
    incr_obs_trivial.assign(n_obs, 0);
    obs_mol_contrib.clear();
    obs_mol_contrib.resize(n_obs);
    obs_cx_match_count.clear();
    obs_cx_match_count.resize(n_obs);
    obs_cx_passed.clear();
    obs_cx_passed.resize(n_obs);
    obs_dirty_cx.clear();
    obs_dirty_cx.resize(n_obs);
    species_mid_prev_cx.clear();
    species_incr_any_tracked = false;
    use_incremental_obs = false;
    obs_max_pattern_depth = 0;
    obs_pat_fm.clear();
    obs_pat_fm.resize(n_obs);

    // Classify each observable.
    //   Molecules path: all patterns single-mol (seed-sum aggregation).
    //   Species path: exactly one pattern, any mol count (per-cx
    //                  aggregation via pattern-seeded contribs).
    // Multi-pattern Species obs and rate-dependent Species obs fall
    // through to the full-walk / rate-dep fallback.
    for (int oi = 0; oi < n_obs; ++oi) {
      auto& obs = model.observables[oi];
      if (obs.patterns.empty())
        continue;

      if (obs.type == "Molecules") {
        // Tracked via the per-mid delta path; obs_values stays fresh
        // after every event.  Any number of patterns, each multi-mol
        // or single-mol — Molecules aggregates per-mid via a straight
        // sum, so the incremental update is exact regardless of
        // pattern shape.
        incr_obs_is_species[oi] = 0;
        incr_obs_is_tracked[oi] = 1;
        incr_tracked_obs_indices.push_back(oi);
      } else if (obs.type == "Species" && kSpeciesIncrObs && !obs.rate_dependent &&
                 obs.patterns.size() == 1) {
        // Rate-dependent Species obs fall through to the full-walk
        // post-event path (compute_rate_dependent_observables); the
        // dirty-cx flush only fires at sample time and so can't
        // keep obs_values fresh enough for rate-law evaluation on
        // every event.
        //
        // Trivial Species obs (pattern is a bare `T()` with no
        // components or bonds) are also skipped: rm_tlbr_rings
        // declares 300 `Size_N R()=N` histogram observables with
        // identical structure, and per-event tracking across all 300
        // costs more than the per-sample full-walk (< 0.3 s for 100
        // samples at Step 1's baseline).  Full-walk is the cheaper
        // path for this class of obs.
        auto& p0 = obs.patterns[0];
        bool const trivial_pat =
            p0.molecules.size() == 1 && p0.molecules[0].components.empty() && p0.bonds.empty();
        if (trivial_pat) {
          continue; // fall back to sample-time full-walk
        }
        incr_obs_is_species[oi] = 1;
        incr_obs_is_tracked[oi] = 1;
        incr_tracked_obs_indices.push_back(oi);
        species_incr_any_tracked = true;
      } else if (obs.rate_dependent) {
        // Rate-dependent obs that doesn't fit either path — keep the
        // existing rate_dep full-walk fallback.
        rate_dep_obs_indices.push_back(oi);
      }

      // Trivial-pattern detection (kept for tracked obs only).  Avoids
      // a per-event per-mid count_multi_molecule_embeddings call when
      // the pattern has no structural constraints (R(), L(), ...).
      if (incr_obs_is_tracked[oi] && obs.patterns.size() == 1 &&
          obs.patterns[0].molecules.size() == 1 &&
          obs.patterns[0].molecules[0].components.empty() && obs.patterns[0].bonds.empty()) {
        incr_obs_trivial[oi] = 1;
      }
    }

    if (incr_tracked_obs_indices.empty() && rate_dep_obs_indices.empty()) {
      return;
    }
    use_incremental_obs = true;

    if constexpr (kObsIncrProfile) {
      obs_incr_profile_ = ObsIncrProfile{}; // reset
      obs_incr_profile_.obs.assign(n_obs, ObsIncrProfile::PerObs{});
      for (int const oi : incr_tracked_obs_indices) {
        auto& po = obs_incr_profile_.obs[oi];
        auto& ob = model.observables[oi];
        po.name = ob.name;
        po.type = ob.type;
        po.pat_count = static_cast<int>(ob.patterns.size());
        if (!ob.patterns.empty() && !ob.patterns[0].molecules.empty()) {
          po.seed_type_index = ob.patterns[0].molecules[0].type_index;
        }
        std::ostringstream sig;
        for (size_t pi = 0; pi < ob.patterns.size(); ++pi) {
          if (pi)
            sig << "||";
          sig << pattern_signature(ob.patterns[pi]);
        }
        po.pat_signature = sig.str();
      }
    }

    // Compute the max pattern diameter across tracked obs.  This
    // tells incremental_update_observables how far to BFS from each
    // directly-affected mid before per-mid contribs stabilise.
    for (int const oi : incr_tracked_obs_indices) {
      for (auto& pat : model.observables[oi].patterns) {
        int const d = pattern_diameter(pat);
        obs_max_pattern_depth = std::max(d, obs_max_pattern_depth);
      }
    }

    // Per-pattern FastMatchSlot for the 2-mol-1-bond-fc specialization
    // on the obs-tracking path.  Eligibility matches the rule-side P6
    // template; ineligible patterns leave fm.enabled=false and fall
    // through to the generic BFS at dispatch time.
    if constexpr (kObsFastMatch) {
      for (int const oi : incr_tracked_obs_indices) {
        auto& obs = model.observables[oi];
        int const n_pat = static_cast<int>(obs.patterns.size());
        obs_pat_fm[oi].assign(n_pat, FastMatchSlot{});
        for (int pi = 0; pi < n_pat; ++pi) {
          auto& pat = obs.patterns[pi];
          int const n_pm = static_cast<int>(pat.molecules.size());
          if (n_pm != 2)
            continue;
          PatternAdj const pa = build_pattern_adjacency(pat, 0, n_pm);
          build_fastmatch_slot(pat, 0, n_pm, /*seed_pat_idx=*/0, pa, model, obs_pat_fm[oi][pi]);
        }
      }
    }

    // Seed per-mol contribs for every tracked obs, through the same
    // `tracked_obs_contrib` the per-event delta path uses — so seeding
    // gets the trivial-pattern and 2-mol-1-bond shortcuts too, and the
    // two can never answer differently.
    //
    // The walk is over the distinct seed types of the observable's
    // patterns rather than per pattern: `tracked_obs_contrib` already
    // sums a molecule's contribution across every pattern it can seed,
    // so visiting a molecule once is both necessary and sufficient.
    std::vector<int> seed_types;
    for (int const oi : incr_tracked_obs_indices) {
      obs_mol_contrib[oi].assign(n_mols, 0.0);
      auto& obs = model.observables[oi];
      seed_types.clear();
      for (auto& pat : obs.patterns) {
        int const t = pat.molecules[0].type_index;
        if (std::find(seed_types.begin(), seed_types.end(), t) == seed_types.end())
          seed_types.push_back(t);
      }
      for (int const t : seed_types) {
        for (int const mid : pool.molecules_of_type(t)) {
          if (!pool.molecule(mid).active)
            continue;
          obs_mol_contrib[oi][mid] = tracked_obs_contrib(oi, mid);
        }
      }
    }

    // For Species-tracked obs, seed per-cx totals and pass flags from
    // the per-mol contribs above.  These pass flags are also what
    // seed_tracked_obs_values() counts to settle the observable's
    // initial value, so the two are consistent by construction rather
    // than by two independent walks agreeing.
    if (species_incr_any_tracked) {
      for (int const oi : incr_tracked_obs_indices) {
        if (!incr_obs_is_species[oi])
          continue;
        auto& obs = model.observables[oi];
        auto& mc_map = obs_cx_match_count[oi];
        auto& pass_map = obs_cx_passed[oi];
        for (auto& pat : obs.patterns) {
          auto& pm = pat.molecules[0];
          // Walk each live complex that contains a type-matching mol.
          // Visited complexes are marked by stamping their member
          // molecules, the same allocation-free dedupe the Species full
          // walk in evaluate_observable uses (issue #62) — a fresh
          // unordered_set of complex ids here cost one hash node per
          // complex at every session build.
          ++obs_walk_cx_gen;
          if (obs_walk_cx_stamp.size() < static_cast<size_t>(pool.molecule_count()))
            obs_walk_cx_stamp.resize(pool.molecule_count(), 0);
          for (int const mid : pool.molecules_of_type(pm.type_index)) {
            if (!pool.molecule(mid).active)
              continue;
            if (obs_walk_cx_stamp[mid] == obs_walk_cx_gen)
              continue;
            int const cx = pool.complex_of(mid);
            obs_walk_cx_stamp[mid] = obs_walk_cx_gen;
            int total = 0;
            for (int const m : pool.molecules_in_complex(cx)) {
              if (m >= 0 && m < static_cast<int>(obs_walk_cx_stamp.size()))
                obs_walk_cx_stamp[m] = obs_walk_cx_gen;
              if (!pool.molecule(m).active)
                continue;
              if (pool.molecule(m).type_index != pm.type_index)
                continue;
              total += static_cast<int>(obs_mol_contrib[oi][m]);
            }
            mc_map[cx] += total; // patterns with overlapping types sum
          }
        }
        for (auto& kv : mc_map) {
          // A multi-pattern obs may see the same cx through different
          // patterns; species_quantity_passes is per-pattern.  v1
          // assumes Species obs have one pattern (true for all three
          // target models).  When obs has multiple patterns, fall back
          // to evaluating with the first pattern's quantity predicate;
          // multi-pattern Species obs will be caught by the invariant
          // gate if this is ever wrong.
          bool const pass = species_quantity_passes(obs.patterns[0], kv.second);
          pass_map[kv.first] = pass ? 1 : 0;
        }
      }

      // Seed per-mid cx snapshot for every active mol.  Tracking
      // applies to all types (not just obs-seed types): a mid of an
      // unrelated type moving between complexes can still strand
      // type-matching peers whose cx membership thereby changes.
      species_mid_prev_cx.assign(n_mols, -1);
      for (int mid = 0; mid < n_mols; ++mid) {
        if (!pool.molecule(mid).active)
          continue;
        species_mid_prev_cx[mid] = pool.complex_of(mid);
      }
    }

    seed_tracked_obs_values();
  }

  // Settle obs_values for the tracked observables out of the tables
  // init_incremental_observables has just built, so Engine::initialize
  // does not have to full-walk them a second time.  The tables already
  // hold the walk's own terms: obs_mol_contrib is its per-molecule
  // embedding count, and obs_cx_passed is its per-complex verdict.  On a
  // million-molecule pool the duplicate walk was about half of session
  // build (issue #65).
  //
  // The equality this rests on is checked rather than asserted:
  // `kLocalObsTrackInvariant` (Debug and ASan builds) re-derives every
  // tracked observable from scratch here and aborts on a mismatch.
  void seed_tracked_obs_values() {
    if (incr_tracked_obs_indices.empty())
      return;
    ++eval_vars_gen;
    obs_values.resize(model.observables.size(), 0.0);
    for (int const oi : incr_tracked_obs_indices) {
      double value = 0.0;
      if (incr_obs_is_species[oi] != 0) {
        // Species: one per complex whose match count passed the
        // pattern's quantity predicate.
        for (const auto& kv : obs_cx_passed[oi])
          if (kv.second != 0)
            value += 1.0;
      } else {
        // Molecules: the straight sum over per-molecule contribs, which
        // is what the full walk accumulates, pattern by pattern.
        for (double const contrib : obs_mol_contrib[oi])
          value += contrib;
      }
      obs_values[oi] = value;

      if constexpr (kLocalObsTrackInvariant) {
        double const from_scratch = evaluate_observable(model.observables[oi]);
        if (value != from_scratch) {
          fprintf(stderr,
                  "[local_obs_selfcheck] seeded obs_values disagrees with the "
                  "full walk at init: oi=%d name=%s type=%s seeded=%.17g "
                  "full_walk=%.17g\n",
                  oi, model.observables[oi].name.c_str(), model.observables[oi].type.c_str(), value,
                  from_scratch);
          std::abort();
        }
      }
    }
  }

  // One molecule's contribution to a tracked observable: the number of
  // embeddings of the observable's patterns seeded at `mid`, summed over
  // patterns.
  //
  // Both the init seeding walk and the per-event delta path want exactly
  // this number, so they share the routine rather than each writing out
  // the dispatch.  The dispatch is the part worth not duplicating: a
  // structurally unconstrained pattern (`T()`) contributes one per
  // molecule of its type with no matching at all, a 2-molecule/1-bond
  // pattern goes through the `count_2mol_1bond_fc` specialization, and
  // only what is left falls to the generic BFS.  Seeding used to call
  // the generic BFS unconditionally while the per-event path took the
  // shortcuts — half of what session build had left after the duplicate
  // observable walk went away (issue #65).
  double tracked_obs_contrib(int oi, int mid) {
    auto& mol = pool.molecule(mid);
    auto& obs = model.observables[oi];

    if (incr_obs_trivial[oi] != 0) {
      int const trivial_type = obs.patterns[0].molecules[0].type_index;
      return (mol.active && mol.type_index == trivial_type) ? 1.0 : 0.0;
    }
    if (!mol.active)
      return 0.0;

    double total = 0.0;
    int const n_pat = static_cast<int>(obs.patterns.size());
    for (int pi = 0; pi < n_pat; ++pi) {
      auto& pat = obs.patterns[pi];
      auto& pm = pat.molecules[0];
      if (mol.type_index != pm.type_index)
        continue;
      if constexpr (kObsIncrProfile) {
        obs_incr_profile_.obs[oi].per_mid_calls++;
      }
      const FastMatchSlot* fm = nullptr;
      if constexpr (kObsFastMatch) {
        if (oi < static_cast<int>(obs_pat_fm.size()) &&
            pi < static_cast<int>(obs_pat_fm[oi].size()) && obs_pat_fm[oi][pi].enabled) {
          fm = &obs_pat_fm[oi][pi];
        }
      }
      int c;
      if (fm != nullptr) {
        c = count_2mol_1bond_fc(pool, mid, *fm, &cmm_fc_profile_);
        if constexpr (kObsFastMatchInvariant) {
          int const ref = count_multi_molecule_embeddings(pool, mid, pat, model);
          if (c != ref) {
            std::fprintf(stderr,
                         "[obs_fastmatch mismatch] oi=%d pi=%d mid=%d "
                         "seed_type=%d partner_type=%d specialized=%d generic=%d\n",
                         oi, pi, mid, fm->seed_type, fm->partner_type, c, ref);
            std::abort();
          }
        }
      } else {
        c = count_multi_molecule_embeddings(pool, mid, pat, model);
      }
      total += static_cast<double>(c);
    }
    return total;
  }

  // Incrementally update tracked-observable values after affected
  // molecules have changed.
  //
  // Molecules path (existing): O(affected × tracked_molecules_obs) per
  // event — obs_values[oi] is updated in-place.
  //
  // Species path (new in Step 2): per affected mid, recompute contrib
  // and mark the mid's previous + current complexes dirty.  Flushed in
  // record_at before the next sample output.
  void incremental_update_observables(const std::unordered_set<int>& affected) {
    if (incr_tracked_obs_indices.empty())
      return;
    ++eval_vars_gen;

    using oip_clock = std::chrono::steady_clock;
    bool oip_sampled = false;
    oip_clock::time_point oip_t_total_start, oip_t_section;
    auto oip_dns = [](oip_clock::time_point a, oip_clock::time_point b) -> uint64_t {
      return static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };
    if constexpr (kObsIncrProfile) {
      obs_incr_profile_.update_calls++;
      obs_incr_profile_.affected_sum += affected.size();
      oip_sampled = (obs_incr_profile_.update_calls % kObsIncrProfileSampleEvery) == 0;
      if (oip_sampled) {
        obs_incr_profile_.update_sampled++;
        oip_t_total_start = oip_clock::now();
        oip_t_section = oip_t_total_start;
      }
    }

    int const n_mols = pool.molecule_count();
    if (species_incr_any_tracked && static_cast<int>(species_mid_prev_cx.size()) < n_mols) {
      species_mid_prev_cx.resize(n_mols, -1);
    }

    // Obs-depth BFS: rule BFS (in fire_rule) expands `affected` by
    // rule max_pattern_depth hops.  If obs_max_pattern_depth exceeds
    // that, per-mid contribs further away can still be stale (e.g. a
    // neighbor's state change invalidates a 2-mol obs pattern seeded
    // at our mid).  Re-expand here to cover obs reach.
    const std::unordered_set<int>* affected_ptr = &affected;
    std::unordered_set<int> obs_expanded;
    if (obs_max_pattern_depth > max_pattern_depth) {
      obs_expanded = affected;
      std::queue<std::pair<int, int>> bfs_q;
      for (int const mid : affected) {
        if (mid >= 0 && mid < n_mols && pool.molecule(mid).active)
          bfs_q.emplace(mid, 0);
      }
      int const extra = obs_max_pattern_depth; // BFS up to this depth from each seed
      while (!bfs_q.empty()) {
        auto front = bfs_q.front();
        bfs_q.pop();
        int const mid = front.first;
        int const depth = front.second;
        if (depth >= extra)
          continue;
        auto& mol = pool.molecule(mid);
        for (int const cid : mol.comp_ids) {
          int const partner = pool.component(cid).bond_partner;
          if (partner < 0)
            continue;
          int const neighbor = pool.mol_of_comp(partner);
          if (pool.molecule(neighbor).active && !obs_expanded.count(neighbor)) {
            obs_expanded.insert(neighbor);
            bfs_q.emplace(neighbor, depth + 1);
          }
        }
      }
      affected_ptr = &obs_expanded;
    }
    const auto& eff_affected = *affected_ptr;

    for (int const oi : incr_tracked_obs_indices) {
      if (static_cast<int>(obs_mol_contrib[oi].size()) < n_mols)
        obs_mol_contrib[oi].resize(n_mols, 0.0);

      bool const is_species = incr_obs_is_species[oi];

      oip_clock::time_point oip_t_obs_start;
      if constexpr (kObsIncrProfile) {
        if (oip_sampled)
          oip_t_obs_start = oip_clock::now();
      }

      // Molecules: contrib is summed across all patterns into
      // obs_values.  Species: contrib is summed into obs_mol_contrib
      // and aggregated at flush time per cx.  Multi-pattern Species
      // obs fall through to full-walk at init, so here obs.patterns
      // has exactly one entry for Species and 1+ entries for Molecules.
      for (int const mid : eff_affected) {
        if (mid < 0 || mid >= n_mols)
          continue;
        auto& mol = pool.molecule(mid);
        double const old_c = obs_mol_contrib[oi][mid];
        double const new_c = tracked_obs_contrib(oi, mid);
        if (new_c != old_c) {
          obs_mol_contrib[oi][mid] = new_c;
          if (!is_species)
            obs_values[oi] += (new_c - old_c);
          if constexpr (kObsIncrProfile) {
            obs_incr_profile_.obs[oi].contrib_deltas++;
          }
        }
        if (is_species) {
          int const cx_now = mol.active ? pool.complex_of(mid) : -1;
          if (cx_now >= 0) {
            obs_dirty_cx[oi].insert(cx_now);
            if constexpr (kObsIncrProfile) {
              obs_incr_profile_.obs[oi].dirty_inserts++;
            }
          }
        }
      }

      if constexpr (kObsIncrProfile) {
        if (oip_sampled) {
          auto now = oip_clock::now();
          obs_incr_profile_.obs[oi].per_mid_ns += oip_dns(oip_t_obs_start, now);
        }
      }
    }

    if constexpr (kObsIncrProfile) {
      if (oip_sampled) {
        auto now = oip_clock::now();
        obs_incr_profile_.obs_loop_ns += oip_dns(oip_t_section, now);
        oip_t_section = now;
      }
    }

    // Dirty the previous cx for any mid whose complex changed since
    // the last event touched it.  The dirty marker is applied to
    // every tracked Species obs, not just the ones seeded at the
    // moving mid's type: a B mid moving between complexes can strand
    // T peers whose cx membership changed even though the T peers
    // themselves aren't in `affected`.  Flushing the prev cx ensures
    // its stale pass flag is reconciled from live membership.
    if (species_incr_any_tracked) {
      for (int const mid : affected) {
        if (mid < 0 || mid >= n_mols)
          continue;
        auto& mol = pool.molecule(mid);
        int const cx_now = mol.active ? pool.complex_of(mid) : -1;
        int const cx_prev = species_mid_prev_cx[mid];
        if (cx_prev != -1 && cx_prev != cx_now) {
          for (int const oi : incr_tracked_obs_indices) {
            if (!incr_obs_is_species[oi])
              continue;
            obs_dirty_cx[oi].insert(cx_prev);
            if (cx_now >= 0)
              obs_dirty_cx[oi].insert(cx_now);
          }
        }
        species_mid_prev_cx[mid] = cx_now;
      }
    }

    if constexpr (kObsIncrProfile) {
      if (oip_sampled) {
        auto now = oip_clock::now();
        obs_incr_profile_.prev_cx_loop_ns += oip_dns(oip_t_section, now);
        obs_incr_profile_.update_total_ns += oip_dns(oip_t_total_start, now);
      }
    }
  }

  // Flush dirty complexes for every Species-tracked obs and sync
  // obs_values.  Called from record_at before record_time_point so the
  // sample output sees up-to-date values.
  void flush_species_incr_observables() {
    if (!species_incr_any_tracked)
      return;

    using oip_clock = std::chrono::steady_clock;
    bool oip_flush_sampled = false;
    oip_clock::time_point oip_flush_start;
    if constexpr (kObsIncrProfile) {
      obs_incr_profile_.flush_calls++;
      // Flush fires once per sample (O(n_steps+1)), so it is cheap to
      // full-sample regardless of K.  Sample every call.
      oip_flush_sampled = true;
      if (oip_flush_sampled) {
        obs_incr_profile_.flush_sampled++;
        oip_flush_start = oip_clock::now();
      }
    }

    // Consume the pool's side-channel list of cxs that died via
    // merge/delete since the last flush.  Any stored cx_match_count
    // entry for such a cx is stale and must be re-flushed to drop its
    // obsolete pass flag (a cx can die without any of its former
    // members being bond-endpoints in the event that killed it).
    std::vector<int> dead_cxs;
    pool.consume_dead_cxs(dead_cxs);
    if constexpr (kObsIncrProfile) {
      obs_incr_profile_.flush_dead_cx_sum += dead_cxs.size();
    }
    for (int const oi : incr_tracked_obs_indices) {
      if (!incr_obs_is_species[oi])
        continue;
      auto& obs = model.observables[oi];
      auto& pat = obs.patterns[0];
      int const pm_type = pat.molecules[0].type_index;
      auto& mc_map = obs_cx_match_count[oi];
      auto& pass_map = obs_cx_passed[oi];
      auto& contribs = obs_mol_contrib[oi];
      for (int const cx : dead_cxs) {
        if (mc_map.count(cx))
          obs_dirty_cx[oi].insert(cx);
      }
      if constexpr (kObsIncrProfile) {
        obs_incr_profile_.flush_dirty_cx_sum += obs_dirty_cx[oi].size();
      }
      for (int const cx : obs_dirty_cx[oi]) {
        int total = 0;
        for (int const m : pool.molecules_in_complex(cx)) {
          if (!pool.molecule(m).active)
            continue;
          if (pool.molecule(m).type_index != pm_type)
            continue;
          if (m >= 0 && m < static_cast<int>(contribs.size()))
            total += static_cast<int>(contribs[m]);
        }
        bool const new_pass = species_quantity_passes(pat, total);
        auto pit = pass_map.find(cx);
        bool const old_pass = (pit != pass_map.end()) && (pit->second != 0);
        if (new_pass != old_pass) {
          obs_values[oi] += (new_pass ? 1.0 : -1.0);
        }
        if (total == 0 && !new_pass) {
          // Dead or empty cx: drop both maps to bound memory.
          mc_map.erase(cx);
          pass_map.erase(cx);
        } else {
          mc_map[cx] = total;
          pass_map[cx] = new_pass ? 1 : 0;
        }
      }
      obs_dirty_cx[oi].clear();
    }

    if constexpr (kObsIncrProfile) {
      if (oip_flush_sampled) {
        auto now = oip_clock::now();
        obs_incr_profile_.flush_total_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - oip_flush_start).count();
      }
    }
  }

  double evaluate_observable(const Observable& obs) {
    double total = 0;
    RecordAtProfile::PerObs* rap_po = nullptr;
    if constexpr (kRecordAtProfile) {
      if (rap_profile_.active_oi >= 0 &&
          rap_profile_.active_oi < static_cast<int>(rap_profile_.obs.size())) {
        rap_po = &rap_profile_.obs[rap_profile_.active_oi];
      }
    }
    for (auto& pat : obs.patterns) {
      if (pat.molecules.empty())
        continue;
      auto& pm = pat.molecules[0];
      bool const multi = pat.molecules.size() > 1;

      if (obs.type == "Molecules") {
        if constexpr (kRecordAtProfile) {
          if (rap_po)
            rap_po->molecules_branch_calls++;
        }
        // Count embeddings across all molecules of matching type
        for (int const mid : pool.molecules_of_type(pm.type_index)) {
          if (!pool.molecule(mid).active)
            continue;
          if constexpr (kRecordAtProfile) {
            if (rap_po)
              rap_po->n_embed_calls++;
          }
          if (multi) {
            total += count_multi_molecule_embeddings(pool, mid, pat, model);
          } else {
            total += count_embeddings_single(pool, mid, pm, model);
          }
        }
      } else {
        if constexpr (kRecordAtProfile) {
          if (rap_po)
            rap_po->species_branch_calls++;
        }
        // Species: count complexes containing a match.
        //
        // Dedupe by stamping the molecules of each complex already
        // visited, not by inserting its complex id into a set.  The set
        // was rebuilt from empty on every call and grew one node per
        // complex, so a sample of a million-complex pool spent more time
        // hashing than counting matches — and every untracked Species
        // observable pays this walk at every sample time (issue #62).
        // Molecule ids are dense, so a generation-stamped side table is
        // an allocation-free equivalent.
        ++obs_walk_cx_gen;
        if (obs_walk_cx_stamp.size() < static_cast<size_t>(pool.molecule_count()))
          obs_walk_cx_stamp.resize(pool.molecule_count(), 0);
        for (int const mid : pool.molecules_of_type(pm.type_index)) {
          if (!pool.molecule(mid).active)
            continue;
          if (obs_walk_cx_stamp[mid] == obs_walk_cx_gen) {
            if constexpr (kRecordAtProfile) {
              if (rap_po)
                rap_po->n_counted_cx_hits++;
            }
            continue;
          }
          int const cx = pool.complex_of(mid);
          obs_walk_cx_stamp[mid] = obs_walk_cx_gen;
          if constexpr (kRecordAtProfile) {
            if (rap_po)
              rap_po->n_cx_visited++;
          }

          // Count total pattern matches across the entire complex.
          // For quantity constraints like "species with exactly N matches",
          // we need the complex-wide total, not just one molecule's count.
          // A molecule with no bonded component is alone in its complex,
          // so resolving `cx` through the complex map would only hand
          // back {mid}.  Free monomers are the bulk of most pools, and
          // that map lookup was the last per-molecule hash in this walk.
          bool cx_is_singleton = true;
          for (int const cid : pool.molecule(mid).comp_ids) {
            if (pool.component(cid).bond_partner >= 0) {
              cx_is_singleton = false;
              break;
            }
          }
          obs_walk_singleton_scratch[0] = mid;
          const std::vector<int>& cx_members =
              cx_is_singleton ? obs_walk_singleton_scratch : pool.molecules_in_complex(cx);

          int match_count = 0;
          for (int const m : cx_members) {
            if constexpr (kRecordAtProfile) {
              if (rap_po)
                rap_po->n_cx_members_walked++;
            }
            if (m >= 0 && m < static_cast<int>(obs_walk_cx_stamp.size()))
              obs_walk_cx_stamp[m] = obs_walk_cx_gen;
            if (!pool.molecule(m).active)
              continue;
            if (pool.molecule(m).type_index != pm.type_index)
              continue;
            if constexpr (kRecordAtProfile) {
              if (rap_po)
                rap_po->n_embed_calls++;
            }
            if (multi) {
              match_count += count_multi_molecule_embeddings(pool, m, pat, model);
            } else {
              match_count += count_embeddings_single(pool, m, pm, model);
            }
          }

          if (match_count > 0) {
            // Apply quantity constraint if present
            if (!pat.relation.empty() && pat.quantity >= 0) {
              bool passes = false;
              if (pat.relation == "==")
                passes = (match_count == pat.quantity);
              else if (pat.relation == ">=")
                passes = (match_count >= pat.quantity);
              else if (pat.relation == "<=")
                passes = (match_count <= pat.quantity);
              else if (pat.relation == ">")
                passes = (match_count > pat.quantity);
              else if (pat.relation == "<")
                passes = (match_count < pat.quantity);
              if (!passes)
                continue;
            }
            total += 1;
          }
        }
      }
    }
    return total;
  }

  // --- Local function evaluation ---

  // Evaluate an observable over a set of candidate molecules.
  // When complex_wide is true, candidates are all molecules in mol_id's complex;
  // otherwise only mol_id itself is considered.
  // `obs_idx` is the observable's index in model.observables — used only
  // by the issue #10 spike instrumentation to key the redundancy tracker.
  // Not `const`: the spike mutates expr_eval_profile_ / eoo_spike_prev_.
  // From-scratch recompute of an observable's value seeded at mol_id.
  // `width` accumulates count_embeddings_* invocations — the real work
  // unit, read by the issue #10 spike counters and zero on the fast path.
  double evaluate_observable_on_recompute(const Observable& obs, int mol_id, bool complex_wide,
                                          uint64_t& width) const {
    double total = 0;
    if (complex_wide) {
      int const cx = pool.complex_of(mol_id);
      for (auto& pat : obs.patterns) {
        if (pat.molecules.empty())
          continue;
        auto& pm = pat.molecules[0];
        bool const multi = pat.molecules.size() > 1;
        for (int const mid : pool.molecules_in_complex(cx)) {
          auto& mol = pool.molecule(mid);
          if (!mol.active || mol.type_index != pm.type_index)
            continue;
          ++width;
          total += multi ? count_multi_molecule_embeddings(pool, mid, pat, model)
                         : count_embeddings_single(pool, mid, pm, model);
        }
      }
    } else {
      for (auto& pat : obs.patterns) {
        if (pat.molecules.empty())
          continue;
        auto& pm = pat.molecules[0];
        if (pool.molecule(mol_id).type_index != pm.type_index)
          continue;
        bool const multi = pat.molecules.size() > 1;
        ++width;
        total += multi ? count_multi_molecule_embeddings(pool, mol_id, pat, model)
                       : count_embeddings_single(pool, mol_id, pm, model);
      }
    }
    return total;
  }

  // issue #10 spike: classify a call against the last computation of the
  // same (observable, scope-id) key.  Within one event the graph is
  // static, so an unchanged result last seen this same epoch is
  // reclaimable by per-event memoization; an unchanged result last seen in
  // an earlier epoch needs a persistent incremental tracker.  Fast-path
  // calls record width=0, so eoo_embed_counts now tracks only the work the
  // fallback recompute still performs — the spike confirms the fix.
  void eoo_record_spike(int obs_idx, int mol_id, bool complex_wide, double total, uint64_t width) {
    if constexpr (kExprEvalProfile) {
      auto& p = expr_eval_profile_;
      p.eoo_calls++;
      p.eoo_embed_counts += width;
      uint32_t const scope_id = complex_wide ? static_cast<uint32_t>(pool.complex_of(mol_id))
                                             : static_cast<uint32_t>(mol_id);
      uint64_t const key = (static_cast<uint64_t>(static_cast<uint32_t>(obs_idx)) << 33) |
                           (static_cast<uint64_t>(complex_wide) << 32) |
                           static_cast<uint64_t>(scope_id);
      if (complex_wide)
        p.eoo_complex_wide++;
      auto it = eoo_spike_prev_.find(key);
      if (it != eoo_spike_prev_.end() && it->second.value == total) {
        p.eoo_unchanged_total++;
        p.eoo_redundant_embed += width;
        if (it->second.epoch == eoo_spike_epoch_)
          p.eoo_unchanged_intra++;
      }
      if (it == eoo_spike_prev_.end())
        eoo_spike_prev_.emplace(key, EooSpikeRec{total, eoo_spike_epoch_});
      else
        it->second = EooSpikeRec{total, eoo_spike_epoch_};
    }
  }

  double evaluate_observable_on(const Observable& obs, int mol_id, bool complex_wide, int obs_idx) {
    if (!pool.molecule(mol_id).active)
      return 0.0;

    // issue #10 fast path: a Molecules-type tracked observable is
    // delta-maintained per-mid in obs_mol_contrib by
    // incremental_update_observables, which runs before the propensity
    // recompute each event — so obs_mol_contrib is post-event fresh here.
    // Per-molecule local eval is then a table read; complex-wide eval is a
    // sum of the per-mid contribs over the complex.  Either way no
    // embedding counts.  Species-type and rate-dependent observables, and
    // !use_incremental_obs models, are not in this table and take the
    // recompute fallback below.
    if (use_incremental_obs && obs_idx >= 0 &&
        obs_idx < static_cast<int>(incr_obs_is_tracked.size()) && incr_obs_is_tracked[obs_idx] &&
        !incr_obs_is_species[obs_idx]) {
      auto& contrib = obs_mol_contrib[obs_idx];
      double fast = 0.0;
      bool fast_ok = false;
      // obs_mol_contrib grows on molecule-count growth; if it has not
      // caught up to a mid we need, fall back to recompute.
      if (complex_wide) {
        int const cx = pool.complex_of(mol_id);
        fast_ok = true;
        for (int const mid : pool.molecules_in_complex(cx)) {
          if (mid >= static_cast<int>(contrib.size())) {
            fast_ok = false;
            fast = 0.0;
            break;
          }
          fast += contrib[mid];
        }
      } else if (mol_id < static_cast<int>(contrib.size())) {
        fast = contrib[mol_id];
        fast_ok = true;
      }
      if (fast_ok) {
        if constexpr (kLocalObsTrackInvariant) {
          uint64_t chk_width = 0;
          double const ref = evaluate_observable_on_recompute(obs, mol_id, complex_wide, chk_width);
          if (ref != fast) {
            std::fprintf(stderr,
                         "kLocalObsTrackInvariant: obs=%d mol=%d complex_wide=%d "
                         "table=%.17g recompute=%.17g\n",
                         obs_idx, mol_id, static_cast<int>(complex_wide), fast, ref);
            std::abort();
          }
        }
        eoo_record_spike(obs_idx, mol_id, complex_wide, fast, /*width=*/0);
        return fast;
      }
    }

    uint64_t width = 0;
    double const total = evaluate_observable_on_recompute(obs, mol_id, complex_wide, width);
    eoo_record_spike(obs_idx, mol_id, complex_wide, total, width);
    return total;
  }

  // Evaluate the rate of a local-function rule for a specific molecule.
  // For molecule-level argument binding: evaluates observables on the single
  // molecule (per-molecule scope).
  // For pattern-level argument binding: evaluates observables across the
  // molecule's entire complex (complex-wide scope).
  double evaluate_local_rate(const Rule& rule, int mol_id) {
    // DOR1 (issue #34): this side carries no per-instance factor, so it
    // weighs in at its plain embedding count.  Returning 1 here — rather than
    // branching at every call site — keeps local_propensity = count_a/corr_a.
    if (rule.rate_law.unity_factor_a)
      return 1.0;
    return evaluate_local_factor(rule.rate_law.function_name, rule.rate_law.local_arg_is_molecule,
                                 mol_id);
  }

  // FunctionProduct B-side factor f2 evaluated at a reactant-B molecule.
  double evaluate_local_rate_b(const Rule& rule, int mol_id) {
    if (rule.rate_law.unity_factor_b)
      return 1.0;
    return evaluate_local_factor(rule.rate_law.function_name_b,
                                 rule.rate_law.local_arg_is_molecule_b, mol_id);
  }

  // Evaluate one named local-function factor at a molecule.  Backs both the
  // single-local-function rate law (evaluate_local_rate) and each of the two
  // FunctionProduct factors.  `per_molecule` selects per-molecule vs
  // complex-wide observable scope; `fn_name` selects which local function's
  // value to return after re-evaluating the local dependency chain.
  double evaluate_local_factor(const std::string& fn_name, bool per_molecule, int mol_id) {
    bool elr_sampled = false;
    std::chrono::steady_clock::time_point elr_t0;
    if constexpr (kExprEvalProfile) {
      expr_eval_profile_.evaluate_local_rate_calls++;
      elr_sampled =
          (expr_eval_profile_.evaluate_local_rate_calls % kExprEvalProfileSampleEvery == 0);
      if (elr_sampled) {
        expr_eval_profile_.sampled_local_rate_calls++;
        elr_t0 = std::chrono::steady_clock::now();
      }
    }
    update_eval_vars();

    // Save global values for the slots we are about to perturb.  A
    // follow-up rate eval (next rule in incremental_update, or this
    // rule's own next per-mol iteration) would otherwise see this
    // mol's per-mol overrides — and update_eval_vars's stale-flag
    // short-circuit means it can't re-read the global state from
    // model state.  Restoring keeps `eval_vars_flat` consistent with
    // `eval_vars_gen` so the cache survives the local-rate excursion.
    //
    // Buffer sizes are stable across calls (local_obs_indices and
    // eval_local_fn_slots are precomputed at engine init), so the
    // resize() is a no-op after the first call.
    eval_local_obs_save_.resize(local_obs_indices.size());
    for (size_t k = 0; k < local_obs_indices.size(); ++k)
      eval_local_obs_save_[k] = eval_vars_flat[eval_obs_slot[local_obs_indices[k]]];
    eval_local_fn_save_.resize(eval_local_fn_slots.size());
    for (size_t k = 0; k < eval_local_fn_slots.size(); ++k)
      eval_local_fn_save_[k] = eval_vars_flat[eval_gf_main_slot[eval_local_fn_slots[k]]];

    // Override observable slots with local evaluations at mol_id,
    // using precomputed local_obs_indices (avoids set rebuild + linear scan).
    // elr_sub is a rolling marker: on sampled calls it brackets first the
    // observable re-eval span, then the local-function expr::evaluate span,
    // so issue #6 can separate the ExprTk-addressable cost from the rest.
    std::chrono::steady_clock::time_point elr_sub;
    if constexpr (kExprEvalProfile) {
      if (elr_sampled)
        elr_sub = std::chrono::steady_clock::now();
    }
    eval_local_obs_value_.resize(local_obs_indices.size());
    for (size_t k = 0; k < local_obs_indices.size(); ++k) {
      int const oi = local_obs_indices[k];
      auto& obs = model.observables[oi];
      double const v = evaluate_observable_on(obs, mol_id, !per_molecule, oi);
      eval_local_obs_value_[k] = v;
      if (!local_obs_scope_conflict_)
        eval_vars_flat[eval_obs_slot[oi]] = v;
    }
    if constexpr (kExprEvalProfile) {
      if (elr_sampled) {
        auto const now = std::chrono::steady_clock::now();
        expr_eval_profile_.local_obs_eval_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - elr_sub).count());
        elr_sub = now;
      }
    }

    // Re-evaluate local functions only.  Functions are stored in
    // dependency order so sequential evaluation resolves the chain
    // correctly.  Global functions were already settled by the
    // top-of-call update_eval_vars and DO NOT depend on local
    // observables (BNG2 emits local arg references only inside
    // is_local() functions).
    //
    // With a scope conflict in play the observable slots cannot be set
    // once for the whole chain: each function is given its own view —
    // localized where it tags the observable, the saved global value
    // where it reads it bare — immediately before it is evaluated.  Its
    // result lands in its own slot, so a caller later in the chain still
    // sees the value that function computed under its own scoping.
    for (int const i : eval_local_fn_slots) {
      if constexpr (kExprEvalProfile)
        expr_eval_profile_.local_fn_ast_evals++;
      if (local_obs_scope_conflict_) {
        auto const& mask = fn_local_obs_mask_[i];
        for (size_t k = 0; k < local_obs_indices.size(); ++k)
          eval_vars_flat[eval_obs_slot[local_obs_indices[k]]] =
              mask[k] != 0 ? eval_local_obs_value_[k] : eval_local_obs_save_[k];
      }
      try {
        eval_vars_flat[eval_gf_main_slot[i]] = expr_eval_.evaluate(gf_expr_id_[i]);
      } catch (...) {
        eval_vars_flat[eval_gf_main_slot[i]] = 0.0;
      }
    }
    if constexpr (kExprEvalProfile) {
      if (elr_sampled)
        expr_eval_profile_.local_expr_eval_ns +=
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - elr_sub)
                                      .count());
    }

    auto fit = model.function_index.find(fn_name);
    double const result =
        (fit != model.function_index.end()) ? eval_vars_flat[eval_gf_main_slot[fit->second]] : 0.0;

    // Restore the global view we cached on entry.
    for (size_t k = 0; k < local_obs_indices.size(); ++k)
      eval_vars_flat[eval_obs_slot[local_obs_indices[k]]] = eval_local_obs_save_[k];
    for (size_t k = 0; k < eval_local_fn_slots.size(); ++k)
      eval_vars_flat[eval_gf_main_slot[eval_local_fn_slots[k]]] = eval_local_fn_save_[k];

    if constexpr (kExprEvalProfile) {
      if (elr_sampled)
        expr_eval_profile_.evaluate_local_rate_ns +=
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - elr_t0)
                                      .count());
    }
    return result;
  }

  // --- Incremental update after reaction firing ---

  void incremental_update(const std::unordered_set<int>& affected_mols) {
    // Pure-context per-complex tallies first (issue #33).  fire_rule may have
    // merged or split complexes, which moves molecules the affected-molecule
    // BFS below never visits — a molecule at the far end of a polymer keeps
    // its every component and still lands in a different complex.  Draining
    // before any count is touched means the rest of this routine only ever
    // sees a tally that already agrees with the pool about who is where.
    percx_drain_moves();

    // issue #10 spike: each incremental_update is one event.  Advance the
    // epoch so evaluate_observable_on can tell intra-event repeats apart
    // from cross-event ones.
    if constexpr (kExprEvalProfile) {
      eoo_spike_epoch_++;
      expr_eval_profile_.eoo_events++;
    }
    // Measurement gate (kIncrUpdateProfile).  On sampled (1-in-K) calls we
    // bracket outer sub-phases with chrono; within those, every Mth per-mid
    // inner-loop entry records recompute-body sub-phase spans.  Logic below
    // is unchanged — every profile action is behind `if constexpr`.
    using iup_clock = std::chrono::steady_clock;
    bool prof_sampled = false;
    bool prof_inner_sample = false;
    iup_clock::time_point prof_t_total_start, prof_t_section_start;
    auto iup_dns = [](iup_clock::time_point a, iup_clock::time_point b) -> uint64_t {
      return static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };
    if constexpr (kIncrUpdateProfile) {
      incr_profile_.calls++;
      prof_sampled = (incr_profile_.calls % kIncrUpdateProfileSampleEvery) == 0;
      incr_profile_.mids_visited += affected_mols.size();
      if (prof_sampled) {
        incr_profile_.sampled_calls++;
        prof_t_total_start = iup_clock::now();
        prof_t_section_start = prof_t_total_start;
      }
    }

    // For local-rate rules, local observables are evaluated over the entire
    // complex.  A state change on one molecule alters observable values for
    // every molecule in the same complex, so we must expand to all complex
    // members — not just bonded neighbors.
    std::unordered_set<int> expanded;
    if (have_local_rules_) {
      if constexpr (kIncrUpdateProfile)
        incr_profile_.have_local_calls++;
      std::unordered_set<int> seen_cx;
      for (int const mid : affected_mols) {
        if (mid < 0 || mid >= pool.molecule_count() || !pool.molecule(mid).active)
          continue;
        int const cx = pool.complex_of(mid);
        if (!seen_cx.insert(cx).second)
          continue;
        for (int const m : pool.molecules_in_complex(cx))
          if (pool.molecule(m).active)
            expanded.insert(m);
      }
      if constexpr (kIncrUpdateProfile)
        incr_profile_.mols_for_local_sum += expanded.size();
    }
    const auto& mols_for_local = have_local_rules_ ? expanded : affected_mols;

    if constexpr (kIncrUpdateProfile) {
      if (prof_sampled) {
        auto now = iup_clock::now();
        incr_profile_.expand_ns += iup_dns(prof_t_section_start, now);
        prof_t_section_start = now;
      }
    }

    // --- Type→rule index: only process rules relevant to affected types ---
    int const n_rules = static_cast<int>(model.rules.size());
    int const n_types = static_cast<int>(type_to_rules.size());
    std::memset(rule_needed_buf.data(), 0, rule_needed_buf.size());

    // Collect unique types from affected molecules, then mark relevant rules
    {
      std::vector<char> type_seen(n_types, 0);
      for (int const mid : affected_mols) {
        if (mid >= 0 && mid < pool.molecule_count()) {
          int const t = pool.molecule(mid).type_index;
          if (t >= 0 && t < n_types && !type_seen[t]) {
            type_seen[t] = 1;
            for (int const ri : type_to_rules[t])
              rule_needed_buf[ri] = 1;
          }
        }
      }

      // For local-rate rules, also check expanded neighbor types
      if (have_local_rules_) {
        for (int const mid : mols_for_local) {
          if (mid >= 0 && mid < pool.molecule_count()) {
            int const t = pool.molecule(mid).type_index;
            if (t >= 0 && t < n_types && !type_seen[t]) {
              type_seen[t] = 1;
              for (int const ri : type_to_rules[t]) {
                if (rule_states[ri].has_local_rates)
                  rule_needed_buf[ri] = 1;
              }
            }
          }
        }
      }
    }

    // Dynamic rate rules always need propensity recomputation because
    // their rates depend on observables that may have changed.
    for (int const ri : dynamic_synthesis_rules)
      rule_needed_buf[ri] = 1;
    for (int const ri : dynamic_rate_rules)
      rule_needed_buf[ri] = 1;

    if constexpr (kIncrUpdateProfile) {
      if (prof_sampled) {
        auto now = iup_clock::now();
        incr_profile_.dispatch_ns += iup_dns(prof_t_section_start, now);
        prof_t_section_start = now;
      }
      // Histogram rules visited per call (always-on counter — cheap).
      int rv = 0;
      for (int ri = 0; ri < n_rules; ++ri)
        if (rule_needed_buf[ri])
          ++rv;
      incr_profile_.rules_visited += rv;
      int rb;
      if (rv <= 1)
        rb = 0;
      else if (rv == 2)
        rb = 1;
      else if (rv <= 4)
        rb = 2;
      else if (rv <= 8)
        rb = 3;
      else if (rv <= 16)
        rb = 4;
      else if (rv <= 32)
        rb = 5;
      else
        rb = 6;
      incr_profile_.rules_visited_hist[rb]++;
    }

    // Local rates over a moving global (issue #38): when one of those
    // globals moves, every instance's rate moves with it and no molecule
    // is marked affected, so the delta update below cannot see it and a
    // full rescan replaces it.  The observables were refreshed before this
    // call, so the rescan reads current values.
    //
    // Gated on the values, not on the rule's shape (issue #40).  If
    // nothing the rate reads globally has changed since the last rescan,
    // no per-instance rate can have changed either — the skip is exact,
    // not an approximation — and the rule drops through to the ordinary
    // delta path, which still owns whatever this event itself touched.
    // `time` and an unenumerable dependency have no value to compare and
    // keep the unconditional rescan.
    for (auto& w : global_dep_local_rules) {
      bool moved = w.always_dirty || !w.primed;
      for (size_t k = 0; k < w.obs_slots.size(); ++k) {
        int const oi = w.obs_slots[k];
        double const v =
            (oi >= 0 && oi < static_cast<int>(obs_values.size())) ? obs_values[oi] : 0.0;
        if (v != w.last[k]) {
          w.last[k] = v;
          moved = true;
        }
      }
      if (!moved)
        continue;
      w.primed = true;
      rescan_all_molecules_for_rule(w.rule_index);
      rule_needed_buf[w.rule_index] = 0;
    }

    for (int ri = 0; ri < n_rules; ++ri) {
      if (!rule_needed_buf[ri])
        continue;

      auto& rule = model.rules[ri];
      auto& rs = rule_states[ri];
      auto& bi = bind_infos[ri];
      // Per-rule, NOT per-event: local_rate_cache is keyed by complex id
      // alone and caches the rule-specific local_rate.  Hoisting this
      // clear to per-event scope would let one rule's complex-wide rate
      // be reused for a different rule that touches the same complex —
      // wrong (it made AN/ANx fail the corpus tier; issue #10).
      local_rate_cache.clear();
      local_rate_cache_b.clear();
      if constexpr (kIncrUpdateProfile)
        incr_profile_.rule_local_rate_cache_clears++;

      int const seed_a =
          (!rule.reactant_pattern_starts.empty()) ? rule.reactant_pattern_starts[0] : 0;
      int const seed_b =
          (rule.reactant_pattern_starts.size() > 1) ? rule.reactant_pattern_starts[1] : -1;

      if (seed_a >= static_cast<int>(rule.reactant_pattern.molecules.size())) {
        // Synthesis rule: recompute propensity if rate is dynamic
        if (rule.molecularity == 0 && rule.rate_law.is_dynamic) {
          double const rate = evaluate_rate(rule);
          set_rule_propensity(rs, rate);
        }
        if constexpr (kIncrUpdateProfile)
          incr_profile_.rules_skipped_needed++;
        continue;
      }

      // >= 3 reactant patterns (issue #24): refresh every slot's count for
      // each touched molecule, then let compute_propensity re-evaluate the
      // distinct-tuple sum from the updated subset totals.
      if (rs.nary.enabled) {
        for (int const mid : affected_mols)
          nary_update_mol(ri, mid);
        double const rate = evaluate_rate(rule);
        set_rule_propensity(rs, compute_propensity(rs, rule, rate));
        continue;
      }

      auto& pm_a = rule.reactant_pattern.molecules[seed_a];

      // For local-rate rules, use the expanded set; otherwise use the original
      const auto& update_set = rs.has_local_rates ? mols_for_local : affected_mols;

      // Is this rule "simple" enough for the bitmask cache?  Step 2 limits
      // caching to rules whose count_a / count_b depend only on the
      // directly-iterated molecule (no multi-mol patterns, no local rates,
      // no disjoint-pattern complex expansion).  Later steps will extend
      // caching to multi-mol rules via a per-complex version tag.
      const bool rule_cacheable = !rs.use_multi_mol_count && !rs.use_multi_mol_count_b &&
                                  !rs.has_local_rates && !rs.needs_complex_expansion &&
                                  rs.a_mask_complete &&
                                  (rule.molecularity < 2 || rs.b_mask_complete);
      const uint64_t rule_relevant_mask =
          rule_cacheable
              ? (rs.a_relevant_mask | (rule.molecularity >= 2 ? rs.b_relevant_mask : uint64_t{0}))
              : ~uint64_t{0};

      for (int const mid : update_set) {
        // Ensure mol_data is large enough.  The default row's
        // cache_init=false is the point here: this entry is for a molecule
        // born since the last rescan, so nothing has computed it yet.
        if (mid >= static_cast<int>(rs.mol_data.size()))
          grow_mol_tables(rs, mid, PerMolRuleData{});

        auto& old = rs.mol_data[mid];

        iup_clock::time_point prof_t_mid_start, prof_t_sub;
        if constexpr (kIncrUpdateProfile) {
          incr_profile_.per_mid_entries++;
          if (static_cast<int>(incr_profile_.per_rule_entries.size()) <= ri)
            incr_profile_.per_rule_entries.resize(ri + 1, 0);
          incr_profile_.per_rule_entries[ri]++;
          if (prof_sampled) {
            prof_inner_sample = (incr_profile_.inner_counter % kIncrUpdateProfileInnerSample) == 0;
            incr_profile_.inner_counter++;
            if (prof_inner_sample) {
              incr_profile_.per_mid_sampled++;
              prof_t_mid_start = iup_clock::now();
              prof_t_sub = prof_t_mid_start;
            }
          } else {
            prof_inner_sample = false;
          }
        }

        // P1 cache hit check.  A valid cache entry (cache_init=true) on a
        // live molecule is still correct this event iff no change to mid
        // this event touches any component in the rule's relevant bitmask.
        // If mid has no matching epoch entry, it wasn't directly mutated
        // (BFS-expanded neighbor) — automatic hit.
        //
        // The bitmask refinement is safe only when relevant_mask != 0.
        // An empty mask means the rule's pattern has no component
        // constraints (e.g., Src() with no components, or A() in a
        // synthesis rule), and (chg & 0) == 0 would false-hit on slot
        // reuse after AddMolecule — the sentinel mask=~0 can't shift a
        // zero relevant_mask.  For those rules we hit only when the mol
        // was untouched this event.
        if (rule_cacheable && old.cache_init && mid < pool.molecule_count() &&
            pool.molecule(mid).active) {
          bool const had_change = mid < static_cast<int>(event_mol_change_epoch.size()) &&
                                  event_mol_change_epoch[mid] == event_epoch;
          if (!had_change) {
            if constexpr (kIncrUpdateProfile) {
              incr_profile_.cache_hits_epoch++;
              if (prof_inner_sample) {
                auto now = iup_clock::now();
                incr_profile_.cache_hit_ns += iup_dns(prof_t_mid_start, now);
              }
            }
            continue; // unchanged this event
          }
          if (rule_relevant_mask != 0 && (event_mol_change_mask[mid] & rule_relevant_mask) == 0) {
            if constexpr (kIncrUpdateProfile) {
              incr_profile_.cache_hits_mask++;
              if (prof_inner_sample) {
                auto now = iup_clock::now();
                incr_profile_.cache_hit_ns += iup_dns(prof_t_mid_start, now);
              }
            }
            continue; // change is outside the rule's relevant components
          }
          // Otherwise fall through to the subtract/recompute/add path.
        } else if constexpr (kIncrUpdateProfile) {
          if (!rule_cacheable)
            incr_profile_.cache_uncacheable++;
        }

        if constexpr (kIncrUpdateProfile) {
          incr_profile_.cache_misses++;
          if (static_cast<int>(incr_profile_.per_rule_recomputes.size()) <= ri)
            incr_profile_.per_rule_recomputes.resize(ri + 1, 0);
          incr_profile_.per_rule_recomputes[ri]++;
          if (prof_inner_sample)
            prof_t_sub = iup_clock::now();
        }

        // Subtract old values.  The aux aggregates move only for the rule
        // shapes that keep an aux row; the other shapes never read them
        // (a_only_total and friends are read only on the bimolecular branch
        // of compute_propensity), and leaving them at the zero the rescan
        // set is what a rule with no aux table has to say.
        rs.a_total -= old.count_a;
        rs.b_total -= old.count_b;
        if (rs.needs_mol_aux) {
          auto const& oldx = rs.mol_aux[mid];
          rs.a_only_total -= oldx.a_only;
          rs.b_only_total -= oldx.b_only;
          rs.ab_both_total -= oldx.ab_both;
          if (rule.same_components)
            rs.ab_both_sq_total -= oldx.ab_both * oldx.ab_both;
          if (rs.has_local_rates)
            rs.local_propensity_total -= oldx.local_propensity;
          if (rule.rate_law.type == RateLawType::FunctionProduct)
            rs.local_propensity_b_total -= oldx.local_propensity_b;
        }

        if constexpr (kIncrUpdateProfile) {
          if (prof_inner_sample) {
            auto now = iup_clock::now();
            incr_profile_.subtract_ns += iup_dns(prof_t_sub, now);
            prof_t_sub = now;
          }
        }

        // Recompute
        PerMolRuleData nd{};
        PerMolRuleAux ndx{};

        if (mid < pool.molecule_count() && pool.molecule(mid).active) {
          // Count A embeddings
          if (pool.molecule(mid).type_index == pm_a.type_index) {
            if (rs.use_multi_mol_count) {
              if constexpr (kIncrUpdateProfile) {
                incr_profile_.count_a_multi_calls++;
                if (static_cast<int>(incr_profile_.per_rule_count_a_multi.size()) <= ri)
                  incr_profile_.per_rule_count_a_multi.resize(ri + 1, 0);
                incr_profile_.per_rule_count_a_multi[ri]++;
              }
              int const end_a_inc = (rule.reactant_pattern_starts.size() > 1)
                                        ? rule.reactant_pattern_starts[1]
                                        : static_cast<int>(rule.reactant_pattern.molecules.size());
              nd.count_a = count_multi_mol_fast(pool, mid, rule.reactant_pattern, model, seed_a,
                                                end_a_inc, seed_a, rs.pat_adj_a, &cm_profile_,
                                                &cmm_fc_profile_, &rs.fm_a, &rs.reacting_local_a);
            } else {
              if constexpr (kIncrUpdateProfile)
                incr_profile_.count_a_single_calls++;
              nd.count_a =
                  count_embeddings_single(pool, mid, pm_a, model, nullptr, &rs.reacting_local_a);
            }
          }

          if constexpr (kIncrUpdateProfile) {
            if (prof_inner_sample) {
              auto now = iup_clock::now();
              incr_profile_.count_a_ns += iup_dns(prof_t_sub, now);
              prof_t_sub = now;
            }
          }

          // Count B embeddings
          if (rule.molecularity >= 2 && seed_b >= 0 &&
              seed_b < static_cast<int>(rule.reactant_pattern.molecules.size())) {
            auto& pm_b = rule.reactant_pattern.molecules[seed_b];
            if (pool.molecule(mid).type_index == pm_b.type_index) {
              if (rs.use_multi_mol_count_b) {
                if constexpr (kIncrUpdateProfile)
                  incr_profile_.count_b_multi_calls++;
                int const end_b_inc = static_cast<int>(rule.reactant_pattern.molecules.size());
                nd.count_b = count_multi_mol_fast(pool, mid, rule.reactant_pattern, model, seed_b,
                                                  end_b_inc, seed_b, rs.pat_adj_b, &cm_profile_,
                                                  &cmm_fc_profile_, &rs.fm_b, &rs.reacting_local_b);
              } else {
                if constexpr (kIncrUpdateProfile)
                  incr_profile_.count_b_single_calls++;
                nd.count_b =
                    count_embeddings_single(pool, mid, pm_b, model, nullptr, &rs.reacting_local_b);
              }
            }
          }

          if constexpr (kIncrUpdateProfile) {
            if (prof_inner_sample) {
              auto now = iup_clock::now();
              incr_profile_.count_b_ns += iup_dns(prof_t_sub, now);
              prof_t_sub = now;
            }
          }

          // Shared component analysis
          if (rs.needs_mol_aux) {
            if (rule.molecularity >= 2 && nd.count_a > 0 && nd.count_b > 0) {
              if constexpr (kIncrUpdateProfile)
                incr_profile_.shared_comp_calls++;
              compute_shared_components(pool, mid, rule, model, seed_a, seed_b, bi.bind_local_a,
                                        bi.bind_local_b, bi.seed_mol_a, bi.seed_mol_b, ndx.a_only,
                                        ndx.b_only, ndx.ab_both);
            } else {
              ndx.a_only = nd.count_a;
              ndx.b_only = nd.count_b;
            }
          }

          if constexpr (kIncrUpdateProfile) {
            if (prof_inner_sample) {
              auto now = iup_clock::now();
              incr_profile_.shared_ns += iup_dns(prof_t_sub, now);
              prof_t_sub = now;
            }
          }

          // Local rate computation.  For pattern-level argument binding,
          // local observables are complex-wide so the rate can be cached per
          // complex.  For molecule-level binding, each molecule may have a
          // different rate.
          if (rs.has_local_rates && nd.count_a > 0) {
            if constexpr (kIncrUpdateProfile)
              incr_profile_.local_rate_path_calls++;
            if (!rule.rate_law.local_arg_is_molecule) {
              // Pattern-level: cache per complex
              int const cx = pool.complex_of(mid);
              auto cit = local_rate_cache.find(cx);
              if (cit != local_rate_cache.end()) {
                ndx.local_rate = cit->second;
              } else {
                ndx.local_rate = std::max(evaluate_local_rate(rule, mid), 0.0);
                local_rate_cache[cx] = ndx.local_rate;
              }
            } else {
              // Molecule-level: evaluate per molecule
              ndx.local_rate = std::max(evaluate_local_rate(rule, mid), 0.0);
            }
            ndx.local_propensity = (nd.count_a / rs.embedding_correction_a) * ndx.local_rate;
          }

          // FunctionProduct B-side factor f2, mirroring the A-side path above
          // but keyed off count_b and the dedicated complex cache.
          if (rule.rate_law.type == RateLawType::FunctionProduct && nd.count_b > 0) {
            if (!rule.rate_law.local_arg_is_molecule_b) {
              int const cx = pool.complex_of(mid);
              auto cit = local_rate_cache_b.find(cx);
              if (cit != local_rate_cache_b.end()) {
                ndx.local_rate_b = cit->second;
              } else {
                ndx.local_rate_b = std::max(evaluate_local_rate_b(rule, mid), 0.0);
                local_rate_cache_b[cx] = ndx.local_rate_b;
              }
            } else {
              ndx.local_rate_b = std::max(evaluate_local_rate_b(rule, mid), 0.0);
            }
            ndx.local_propensity_b = (nd.count_b / rs.embedding_correction_b) * ndx.local_rate_b;
          }

          if constexpr (kIncrUpdateProfile) {
            if (prof_inner_sample) {
              auto now = iup_clock::now();
              incr_profile_.local_rate_ns += iup_dns(prof_t_sub, now);
              prof_t_sub = now;
            }
          }
        }

        // Add new values
        rs.a_total += nd.count_a;
        rs.b_total += nd.count_b;
        if (rs.needs_mol_aux) {
          rs.a_only_total += ndx.a_only;
          rs.b_only_total += ndx.b_only;
          rs.ab_both_total += ndx.ab_both;
          if (rule.same_components)
            rs.ab_both_sq_total += ndx.ab_both * ndx.ab_both;
          if (rs.has_local_rates)
            rs.local_propensity_total += ndx.local_propensity;
          if (rule.rate_law.type == RateLawType::FunctionProduct)
            rs.local_propensity_b_total += ndx.local_propensity_b;
        }

        if constexpr (kIncrUpdateProfile) {
          if (prof_inner_sample) {
            auto now = iup_clock::now();
            incr_profile_.add_ns += iup_dns(prof_t_sub, now);
            prof_t_sub = now;
          }
        }

        // Update Fenwick trees for O(log N) sampling.  When mid exceeds the
        // tree's current size, we cannot use FenwickTree::ensure() alone:
        // resizing the underlying vector leaves the new ancestor nodes at 0,
        // so prior weights stop contributing to prefix sums at indices past
        // the old size — `find()` then descends through zero-valued nodes
        // and returns out-of-range positions.  Rebuild the tree from
        // rs.mol_data on growth instead.  Amortized cost is O(N log N) per
        // doubling — i.e. O(log N) per insertion.
        if (rs.use_fenwick_a && nd.count_a != old.count_a) {
          if constexpr (kIncrUpdateProfile)
            incr_profile_.fenwick_a_updates++;
          if (mid >= rs.fenwick_a.n) {
            int const new_n = std::max(mid + 1, rs.fenwick_a.n * 2);
            rs.fenwick_a.init(new_n);
            for (int const m : pool.molecules_of_type(pm_a.type_index)) {
              if (m < 0 || m >= pool.molecule_count() || !pool.molecule(m).active)
                continue;
              if (m < static_cast<int>(rs.mol_data.size()) && rs.mol_data[m].count_a > 0)
                rs.fenwick_a.update(m, rs.mol_data[m].count_a);
            }
          }
          rs.fenwick_a.update(mid, nd.count_a - old.count_a);
        }
        if (rs.use_fenwick_b && nd.count_b != old.count_b) {
          if constexpr (kIncrUpdateProfile)
            incr_profile_.fenwick_b_updates++;
          if (mid >= rs.fenwick_b.n) {
            int const new_n = std::max(mid + 1, rs.fenwick_b.n * 2);
            rs.fenwick_b.init(new_n);
            if (seed_b >= 0 && seed_b < static_cast<int>(rule.reactant_pattern.molecules.size())) {
              auto& pm_b_local = rule.reactant_pattern.molecules[seed_b];
              for (int const m : pool.molecules_of_type(pm_b_local.type_index)) {
                if (m < 0 || m >= pool.molecule_count() || !pool.molecule(m).active)
                  continue;
                if (m < static_cast<int>(rs.mol_data.size()) && rs.mol_data[m].count_b > 0)
                  rs.fenwick_b.update(m, rs.mol_data[m].count_b);
              }
            }
          }
          rs.fenwick_b.update(mid, nd.count_b - old.count_b);
        }

        if constexpr (kIncrUpdateProfile) {
          if (prof_inner_sample) {
            auto now = iup_clock::now();
            incr_profile_.fenwick_ns += iup_dns(prof_t_sub, now);
            prof_t_sub = now;
          }
        }

        // P1 cache: mark this (rule, molecule) entry as a valid snapshot.
        nd.cache_init = true;
        rs.mol_data[mid] = nd;
        if (rs.needs_mol_aux) {
          // `ndx` is a fresh record, so it carries no memory of which complex
          // the pure-context tally is currently counting this molecule in.
          // Carry that across the overwrite: percx_sync is what moves the
          // hit, and it has to know where to take it from.
          auto& ax = rs.mol_aux[mid];
          ndx.percx_cx_a = ax.percx_cx_a;
          ndx.percx_cx_b = ax.percx_cx_b;
          ax = ndx;
        }
        if (rs.per_complex_a) {
          percx_sync(ri, 0, mid);
          if (rs.has_local_rates && rs.mol_aux[mid].percx_cx_a >= 0)
            percx_refresh_rate(rs, 0, rs.mol_aux[mid].percx_cx_a);
        }
        if (rs.per_complex_b) {
          percx_sync(ri, 1, mid);
          if (rs.has_local_rates && rs.mol_aux[mid].percx_cx_b >= 0)
            percx_refresh_rate(rs, 1, rs.mol_aux[mid].percx_cx_b);
        }

        if constexpr (kIncrUpdateProfile) {
          if (prof_inner_sample) {
            auto now = iup_clock::now();
            incr_profile_.store_ns += iup_dns(prof_t_sub, now);
          }
        }
      }

      // Recompute propensity (counted, not chrono-bracketed per rule —
      // bracketing was 8-10 extra now() calls per sampled call and
      // inflated total_est 50%).
      if constexpr (kIncrUpdateProfile)
        incr_profile_.propensity_recomputes++;
      double new_propensity;
      if (rule.rate_law.type == RateLawType::FunctionProduct) {
        {
          double const s1 = rs.per_complex_a ? rs.percx_a.rate_sum : rs.local_propensity_total;
          double const s2 = rs.per_complex_b ? rs.percx_b.rate_sum : rs.local_propensity_b_total;
          new_propensity = s1 * s2 * dor_symmetry(rule);
        }
      } else if (rs.has_local_rates) {
        new_propensity = rs.local_propensity_total * dor_symmetry(rule);
        if (rs.per_complex_a)
          new_propensity = rs.percx_a.rate_sum * dor_symmetry(rule);
      } else {
        double const rate = evaluate_rate(rule);
        new_propensity = compute_propensity(rs, rule, rate, percx_a(ri), percx_b(ri));
      }
      set_rule_propensity(rs, new_propensity);
    }

    if constexpr (kIncrUpdateProfile) {
      if (prof_sampled) {
        auto now = iup_clock::now();
        // rule_loop_ns = total time from end-of-dispatch to end-of-function
        incr_profile_.rule_loop_ns += iup_dns(prof_t_section_start, now);
        incr_profile_.total_ns += iup_dns(prof_t_total_start, now);
      }
    }
  }

  // --- Reactant selection ---

  ReactionMatch select_reactants(int rule_idx) {
    auto& rule = model.rules[rule_idx];
    auto& rs = rule_states[rule_idx];
    ReactionMatch match;

    int const seed_a =
        (!rule.reactant_pattern_starts.empty()) ? rule.reactant_pattern_starts[0] : 0;

    // -- profile scaffolding (gated, see kSelectReactantsProfile) --
    using sr_clock = std::chrono::steady_clock;
    sr_clock::time_point sr_t0{};
    bool sr_sampled = false;
    if constexpr (kSelectReactantsProfile) {
      sr_profile_.calls++;
      sr_sampled = (sr_profile_.calls % kSelectReactantsProfileSampleEvery) == 0;
      if (sr_sampled)
        sr_t0 = sr_clock::now();
    }
    // `finish` is invoked at every exit.  outcome: 0=null_no_seed,
    // 1=null_post_sample, 2=success (including zero-order, which has no
    // reactants but is not a null event).
    auto sr_finish = [&](int path, int outcome) {
      if constexpr (kSelectReactantsProfile) {
        auto& p = sr_profile_;
        p.path_calls[path]++;
        if (outcome == 0)
          p.path_null_no_seed[path]++;
        else if (outcome == 1)
          p.path_null_post[path]++;
        else
          p.path_success[path]++;
        if (sr_sampled) {
          auto ns =
              std::chrono::duration_cast<std::chrono::nanoseconds>(sr_clock::now() - sr_t0).count();
          p.path_ns[path] += ns;
          p.path_hits[path]++;
          p.sampled_calls++;
        }
      }
    };

    if (rule.molecularity == 0) {
      // Zero-order synthesis: no reactants to select, just return empty match
      // (fire_rule will process Add operations via product_mol_to_actual)
      sr_finish(SrProfile::kPathZero, /*outcome=*/2);
      return match;
    }

    if (rs.nary.enabled) {
      // >= 3 reactant patterns (issue #24).  Draw each slot independently
      // ∝ its embedding count and retry until the n molecules are distinct;
      // the accepted distribution is then ∝ Π c_i(m_i) over distinct tuples,
      // which is exactly the measure compute_propensity integrates.  The
      // retry cap protects against pathological weight concentration (say
      // three slots over three molecules of wildly unequal count); hitting
      // it yields a null event, as the homodimer path does at its own cap.
      auto& ns = rs.nary;
      constexpr int kMaxNaryRetries = 256;
      std::vector<int> chosen(ns.n_slots, -1);
      bool distinct = false;

      for (int retry = 0; retry < kMaxNaryRetries && !distinct; ++retry) {
        bool empty_slot = false;
        for (int slot = 0; slot < ns.n_slots; ++slot) {
          chosen[slot] = nary_sample_slot(ns, slot);
          if (chosen[slot] < 0) {
            empty_slot = true;
            break;
          }
        }
        if (empty_slot) {
          sr_finish(SrProfile::kPathNary, /*outcome=*/0);
          return match;
        }
        distinct = true;
        for (int i = 0; i < ns.n_slots && distinct; ++i)
          for (int j = i + 1; j < ns.n_slots; ++j)
            if (chosen[i] == chosen[j]) {
              distinct = false;
              break;
            }
      }
      if (!distinct) {
        sr_finish(SrProfile::kPathNary, /*outcome=*/1);
        return match;
      }

      // -bscb: bimolecular rules only fire across complexes.  Applied
      // pairwise here, as a null event, matching the two-slot behaviour.
      if (model.block_same_complex_binding) {
        for (int i = 0; i < ns.n_slots; ++i)
          for (int j = i + 1; j < ns.n_slots; ++j)
            if (pool.complex_of(chosen[i]) == pool.complex_of(chosen[j])) {
              sr_finish(SrProfile::kPathNary, /*outcome=*/1);
              return match;
            }
      }

      // Pick one embedding per slot and lay the components out flat.  Slot i
      // is seeded at pattern molecule ns.slot_start[i]; a slot that spans
      // more than one pattern molecule (issue #26) has its remaining
      // molecules resolved by walking bonds out of that seed, exactly as the
      // bimolecular sampler resolves its own multi-molecule patterns.
      int const n_flat = rule.reactant_pattern.flat_comp_count();
      match.mol_ids.assign(rule.reactant_pattern.molecules.size(), -1);
      match.comp_ids.assign(n_flat, -1);

      for (int slot = 0; slot < ns.n_slots; ++slot) {
        int const pat_mi = ns.slot_start[slot];
        auto& pm = rule.reactant_pattern.molecules[pat_mi];
        std::vector<std::vector<int>> embs;
        count_embeddings_single(pool, chosen[slot], pm, model, &embs, &ns.reacting_local[slot]);
        if (embs.empty()) {
          match.mol_ids.clear();
          sr_finish(SrProfile::kPathNary, /*outcome=*/1);
          return match;
        }

        // Write seed embedding `ei` into `m`, then, for a multi-molecule
        // slot, walk the slot's bonds out of the seed to place the rest.
        auto apply_emb = [&](int ei, ReactionMatch& m) {
          m.mol_ids[pat_mi] = chosen[slot];
          int flat_base = 0;
          for (int j = 0; j < pat_mi; ++j)
            flat_base += static_cast<int>(rule.reactant_pattern.molecules[j].components.size());
          int const nc = static_cast<int>(pm.components.size());
          for (int ci = 0; ci < nc && ci < static_cast<int>(embs[ei].size()); ++ci)
            m.comp_ids[flat_base + ci] = pool.molecule(chosen[slot]).comp_ids[embs[ei][ci]];
          if (ns.slot_multi[slot] == 0)
            return true;
          return resolve_pattern_context(rule.reactant_pattern, ns.slot_adj[slot], pat_mi,
                                         ns.slot_end[slot], pat_mi, chosen[slot], embs[ei], m) &&
                 slot_match_valid(rule, pat_mi, ns.slot_end[slot], m);
        };

        if (ns.slot_multi[slot] == 0) {
          // Single-molecule slot: every seed embedding is a whole match, so
          // one uniform draw settles it.  Untouched from the shape #24
          // shipped, so those rules keep their exact RNG stream.
          int ei = static_cast<int>(uniform() * embs.size());
          if (ei >= static_cast<int>(embs.size()))
            ei = static_cast<int>(embs.size()) - 1;
          apply_emb(ei, match);
          continue;
        }

        // Multi-molecule slot: only the seed embeddings that extend to a
        // whole match are physical reactions, and they are exactly what
        // c_i(m) counted for the propensity.  Filter first and draw among
        // the survivors — drawing over all seed embeddings and calling a
        // dead end a null event would dilute the rate by the fraction that
        // dead-ends (an A(d,d) bonded to a D and an E offers two seed
        // embeddings of `A(d!1).D(d!1)`, only one of which reaches the D).
        std::vector<int> extending;
        extending.reserve(embs.size());
        for (int ei = 0; ei < static_cast<int>(embs.size()); ++ei) {
          ReactionMatch trial = match;
          if (apply_emb(ei, trial))
            extending.push_back(ei);
        }
        if (extending.empty()) {
          match.mol_ids.clear();
          sr_finish(SrProfile::kPathNary, /*outcome=*/1);
          return match;
        }
        int pick = static_cast<int>(uniform() * extending.size());
        if (pick >= static_cast<int>(extending.size()))
          pick = static_cast<int>(extending.size()) - 1;
        apply_emb(extending[pick], match);
      }

      // Full-assignment injectivity (issue #26).  The distinctness enforced
      // above is over the n seeds only.  A molecule drawn into one slot's
      // complex may also be another slot's seed, or sit inside another
      // slot's complex — firing on such a draw would consume one molecule
      // twice, and a DeleteMolecules rule would try to delete it twice.  The
      // propensity counts those draws, so rejecting them here as null events
      // realizes the injective count exactly rather than approximately (see
      // the NaryState comment).  A rule of single-molecule slots can never
      // trip this; the loop is skipped outright for one.
      if (ns.any_multi) {
        int const n_pat_mols = static_cast<int>(match.mol_ids.size());
        for (int i = 0; i < n_pat_mols; ++i) {
          for (int j = i + 1; j < n_pat_mols; ++j) {
            if (match.mol_ids[i] >= 0 && match.mol_ids[i] == match.mol_ids[j]) {
              match.mol_ids.clear();
              sr_finish(SrProfile::kPathNary, /*outcome=*/1);
              return match;
            }
          }
        }
      }

      sr_finish(SrProfile::kPathNary, /*outcome=*/2);
      return match;
    }

    if (rule.molecularity <= 1) {
      // Unimolecular: sample molecule weighted by count_a (or local_propensity)
      auto& pm_a = rule.reactant_pattern.molecules[seed_a];
      bool const is_multi_pat = rule.reactant_pattern.molecules.size() > 1;
      int const uni_path = is_multi_pat ? (rs.fm_a.enabled ? SrProfile::kPathUniMultiFm
                                                           : SrProfile::kPathUniMultiGen)
                                        : SrProfile::kPathUniSingle;
      int const mol_id = rs.has_local_rates
                             ? sample_molecule_by_local_propensity(pm_a.type_index, rs)
                             : sample_molecule_weighted(pm_a.type_index, rs, true);
      if (mol_id < 0) {
        sr_finish(uni_path, /*outcome=*/0);
        return match;
      }

      bool const multi_mol_pattern = is_multi_pat;

      if (!multi_mol_pattern) {
        // Simple single-molecule unimolecular rule
        std::vector<std::vector<int>> embs;
        count_embeddings_single(pool, mol_id, pm_a, model, &embs, &rs.reacting_local_a);
        if constexpr (kSelectReactantsProfile) {
          sr_profile_.uni_single_embs_sum += embs.size();
          if (embs.empty())
            sr_profile_.uni_single_embs_empty++;
        }
        if (embs.empty()) {
          sr_finish(uni_path, /*outcome=*/1);
          return match;
        }
        int ei = static_cast<int>(uniform() * embs.size());
        if (ei >= static_cast<int>(embs.size()))
          ei = static_cast<int>(embs.size()) - 1;

        match.mol_ids.resize(rule.reactant_pattern.molecules.size(), -1);
        match.mol_ids[seed_a] = mol_id;

        int const n_flat = rule.reactant_pattern.flat_comp_count();
        match.comp_ids.resize(n_flat, -1);
        int base = 0;
        for (int mi = 0; mi < static_cast<int>(rule.reactant_pattern.molecules.size()); ++mi) {
          int const nc = static_cast<int>(rule.reactant_pattern.molecules[mi].components.size());
          if (mi == seed_a) {
            for (int ci = 0; ci < nc && ci < static_cast<int>(embs[ei].size()); ++ci) {
              int const actual_local = embs[ei][ci];
              match.comp_ids[base + ci] = pool.molecule(mol_id).comp_ids[actual_local];
            }
          }
          base += nc;
        }
      } else {
        // Multi-molecule unimolecular rule (e.g., E(s!1).S(p1~U!1) → ...)
        // Must resolve all pattern molecules via bond traversal from the seed.
        //
        // P6: when a FastMatchSlot is populated for this rule's reactant A,
        // dispatch to the 2-mol-1-bond-fc fast select; otherwise fall
        // through to the generic BFS.  With kFastSelectInvariant, run both
        // and assert mol_ids/comp_ids/RNG state all agree before returning.
        ReactionMatch full_match;
        if (rs.fm_a.enabled) {
          if (kFastSelectInvariant) {
            auto rng_before = rng;
            auto fast = select_2mol_1bond_fc_match(rule, rs, mol_id);
            auto rng_after_fast = rng;

            rng = rng_before;
            auto gen = select_multi_mol_unimolecular(rule, rs.pat_adj_a, mol_id, seed_a);
            auto rng_after_gen = rng;

            bool const rng_ok = (rng_after_fast == rng_after_gen);
            bool const match_ok = (fast.mol_ids == gen.mol_ids && fast.comp_ids == gen.comp_ids);
            if (!rng_ok || !match_ok) {
              std::fprintf(stderr,
                           "[FastSelect mismatch] rule=%d seed_mol=%d "
                           "rng_ok=%d match_ok=%d\n"
                           "  fast.mol_ids.size=%zu gen.mol_ids.size=%zu "
                           "fast.comp_ids.size=%zu gen.comp_ids.size=%zu\n",
                           static_cast<int>(&rule - model.rules.data()), mol_id,
                           static_cast<int>(rng_ok), static_cast<int>(match_ok),
                           fast.mol_ids.size(), gen.mol_ids.size(), fast.comp_ids.size(),
                           gen.comp_ids.size());
              if (fast.mol_ids.size() == gen.mol_ids.size()) {
                for (size_t k = 0; k < fast.mol_ids.size(); ++k)
                  std::fprintf(stderr, "  mol_ids[%zu]: fast=%d gen=%d\n", k, fast.mol_ids[k],
                               gen.mol_ids[k]);
              }
              if (fast.comp_ids.size() == gen.comp_ids.size()) {
                for (size_t k = 0; k < fast.comp_ids.size(); ++k)
                  if (fast.comp_ids[k] != gen.comp_ids[k])
                    std::fprintf(stderr, "  comp_ids[%zu]: fast=%d gen=%d\n", k, fast.comp_ids[k],
                                 gen.comp_ids[k]);
              }
              std::abort();
            }
            full_match = std::move(gen);
          } else {
            full_match = select_2mol_1bond_fc_match(rule, rs, mol_id);
          }
        } else {
          full_match = select_multi_mol_unimolecular(rule, rs.pat_adj_a, mol_id, seed_a);
        }
        if constexpr (kSelectReactantsProfile) {
          if (uni_path == SrProfile::kPathUniMultiFm) {
            if (full_match.mol_ids.empty())
              sr_profile_.uni_mm_fm_null++;
            else
              sr_profile_.uni_mm_fm_success++;
          } else {
            if (full_match.mol_ids.empty())
              sr_profile_.uni_mm_gen_null++;
            else
              sr_profile_.uni_mm_gen_success++;
          }
        }
        if (full_match.mol_ids.empty()) {
          sr_finish(uni_path, /*outcome=*/1);
          return match;
        }
        match = std::move(full_match);
      }

    } else {
      // Bimolecular
      int const seed_b = rule.reactant_pattern_starts[1];
      auto& pm_a = rule.reactant_pattern.molecules[seed_a];
      auto& pm_b = rule.reactant_pattern.molecules[seed_b];

      // For `same_components` rules (homodimer A+A on the same component),
      // the propensity formula is deflated by Σ ab(m)² so the realized rate
      // equals the analytic `Σ_{m1<m2} ab(m1)·ab(m2) · k`.  That requires the
      // sampler's accepted (mol_a, mol_b) distribution to be conditional on
      // `mol_a != mol_b` — equivalently, retry the full pair draw until
      // distinct.  A retry cap protects against pathological weight imbalance
      // (e.g., one mol carries ≫99% of the count) where retries could otherwise
      // dominate; on hitting the cap we fall back to a null event, which is
      // statistically indistinguishable from the cap-extending limit.
      //
      // For non-same_components rules the historical `mol_a == mol_b` null
      // event stays — its propensity is the inflated form and rejection rate
      // matches it exactly.
      // FunctionProduct (DOR2): draw each reactant weighted by its own local
      // factor (local_propensity for A, local_propensity_b for B) instead of
      // by raw embedding count.  The propensity S1·S2 includes same-molecule
      // pairs, which the same-molecule null-event rejection below removes —
      // exact for distinct-type reactants, statistically exact otherwise.
      bool const is_function_product = (rule.rate_law.type == RateLawType::FunctionProduct);
      int mol_a = -1, mol_b = -1;
      if (rule.same_components && !is_function_product) {
        constexpr int kMaxRetries = 64;
        bool got_distinct = false;
        for (int retry = 0; retry < kMaxRetries; ++retry) {
          mol_a = sample_molecule_weighted(pm_a.type_index, rs, true);
          mol_b = sample_molecule_weighted(pm_b.type_index, rs, false);
          if (mol_a < 0 || mol_b < 0)
            break;
          if (mol_a != mol_b) {
            got_distinct = true;
            break;
          }
          if constexpr (kSelectReactantsProfile)
            sr_profile_.bimol_same_mol_rejects++;
        }
        if (mol_a < 0 || mol_b < 0) {
          sr_finish(SrProfile::kPathBimol, /*outcome=*/0);
          return match;
        }
        if (!got_distinct) {
          sr_finish(SrProfile::kPathBimol, /*outcome=*/1);
          return match;
        }
      } else {
        if (is_function_product) {
          // A unity side's local_propensity is count/correction — proportional
          // to the raw embedding count — so draw it with the count-weighted
          // sampler, which is O(log N) via the Fenwick tree instead of the
          // local-propensity sampler's O(N) scan.  Same distribution.
          mol_a = rule.rate_law.unity_factor_a
                      ? sample_molecule_weighted(pm_a.type_index, rs, /*use_a=*/true)
                      : sample_molecule_by_local_propensity(pm_a.type_index, rs, /*use_b=*/false);
          mol_b = rule.rate_law.unity_factor_b
                      ? sample_molecule_weighted(pm_b.type_index, rs, /*use_a=*/false)
                      : sample_molecule_by_local_propensity(pm_b.type_index, rs, /*use_b=*/true);
        } else {
          mol_a = sample_molecule_weighted(pm_a.type_index, rs, true);
          mol_b = sample_molecule_weighted(pm_b.type_index, rs, false);
        }
        if (mol_a < 0 || mol_b < 0) {
          sr_finish(SrProfile::kPathBimol, /*outcome=*/0);
          return match;
        }
        // Reject if both reactants are the same molecule (always invalid).
        if (mol_a == mol_b) {
          if constexpr (kSelectReactantsProfile)
            sr_profile_.bimol_same_mol_rejects++;
          sr_finish(SrProfile::kPathBimol, /*outcome=*/1);
          return match;
        }
      }

      // When block_same_complex_binding is set (NFsim's -bscb flag),
      // bimolecular rules only fire between molecules in DIFFERENT
      // complexes.  Otherwise intra-complex binding is allowed (NFsim
      // default: rings can form via bimolecular rules).
      if (model.block_same_complex_binding && pool.complex_of(mol_a) == pool.complex_of(mol_b)) {
        if constexpr (kSelectReactantsProfile)
          sr_profile_.bimol_same_cx_rejects++;
        sr_finish(SrProfile::kPathBimol, /*outcome=*/1);
        return match;
      }

      // Sample embeddings (deduped by reacting-component targets so that
      // the uniform pick below selects among physically distinct reactions).
      std::vector<std::vector<int>> embs_a, embs_b;
      count_embeddings_single(pool, mol_a, pm_a, model, &embs_a, &rs.reacting_local_a);
      count_embeddings_single(pool, mol_b, pm_b, model, &embs_b, &rs.reacting_local_b);
      if constexpr (kSelectReactantsProfile) {
        sr_profile_.bimol_embs_a_sum += embs_a.size();
        sr_profile_.bimol_embs_b_sum += embs_b.size();
      }
      if (embs_a.empty() || embs_b.empty()) {
        if constexpr (kSelectReactantsProfile)
          sr_profile_.bimol_embs_empty++;
        sr_finish(SrProfile::kPathBimol, /*outcome=*/1);
        return match;
      }

      int const n_pat_mols = static_cast<int>(rule.reactant_pattern.molecules.size());
      int const n_flat = rule.reactant_pattern.flat_comp_count();
      int const end_a = seed_b; // pattern A spans [seed_a, seed_b)
      int const end_b = n_pat_mols;
      bool const multi_a = (end_a - seed_a) > 1;
      bool const multi_b = (end_b - seed_b) > 1;

      // Lay seed embedding `emb` down at `pat_seed`, then, for a
      // multi-molecule pattern, walk its bonds out of the seed to place the
      // rest.  False means this seed embedding does not reach a whole match.
      auto apply_slot = [&](int pat_seed, int pat_end, const PatternAdj& adj, int mol,
                            const std::vector<int>& emb, ReactionMatch& m) {
        m.mol_ids[pat_seed] = mol;
        int const flat_base = rule.reactant_pattern.flat_index(pat_seed, 0);
        int const nc =
            static_cast<int>(rule.reactant_pattern.molecules[pat_seed].components.size());
        for (int ci = 0; ci < nc && ci < static_cast<int>(emb.size()); ++ci)
          m.comp_ids[flat_base + ci] = pool.molecule(mol).comp_ids[emb[ci]];
        if (pat_end - pat_seed <= 1)
          return true;
        if constexpr (kSelectReactantsProfile)
          sr_profile_.bimol_resolve_calls++;
        return resolve_pattern_context(rule.reactant_pattern, adj, pat_seed, pat_end, pat_seed, mol,
                                       emb, m) &&
               slot_match_valid(rule, pat_seed, pat_end, m);
      };

      // A multi-molecule pattern is drawn over the seed embeddings that reach
      // a whole match — exactly the ones count_multi_mol_fast counted into the
      // weight this molecule was drawn by.  Drawing over all of them and
      // calling a dead end a null event instead would run the rule slow by
      // the fraction that dead-ends: an `A(d,d)` bonded to a D and an E has
      // two embeddings of `A(d!1)`, only one of which reaches the D, so
      // `A(d!1).D(d!1) + X()` fired at half its rate.  A single-molecule
      // pattern needs no filtering (every seed embedding is already a whole
      // match) and skips the work outright, which also leaves its draw
      // stream exactly as it was.
      auto extending_embs = [&](int pat_seed, int pat_end, const PatternAdj& adj, int mol,
                                const std::vector<std::vector<int>>& embs) {
        std::vector<int> keep;
        keep.reserve(embs.size());
        ReactionMatch trial;
        for (int i = 0; i < static_cast<int>(embs.size()); ++i) {
          trial.mol_ids.assign(n_pat_mols, -1);
          trial.comp_ids.assign(n_flat, -1);
          if (apply_slot(pat_seed, pat_end, adj, mol, embs[i], trial))
            keep.push_back(i);
        }
        return keep;
      };

      // With one seed embedding there is nothing to choose between, so the
      // filter is skipped and a dead end stays a null event — c/S is 0 or 1
      // there and no rate is lost.  That keeps the cost off the common
      // multi-molecule rules (one bond, one endpoint) entirely.
      //
      // Both filters run before either draw, so a rule that loses a whole
      // side to dead ends still spends no uniform() on the other — the same
      // ordering the embs_a/embs_b emptiness check above keeps.
      bool const filter_a = multi_a && embs_a.size() > 1;
      bool const filter_b = multi_b && embs_b.size() > 1;
      std::vector<int> ext_a, ext_b;
      if (filter_a) {
        ext_a = extending_embs(seed_a, end_a, rs.pat_adj_a, mol_a, embs_a);
        if (ext_a.empty()) {
          if constexpr (kSelectReactantsProfile)
            sr_profile_.bimol_resolve_failures++;
          sr_finish(SrProfile::kPathBimol, /*outcome=*/1);
          return match;
        }
      }
      if (filter_b) {
        ext_b = extending_embs(seed_b, end_b, rs.pat_adj_b, mol_b, embs_b);
        if (ext_b.empty()) {
          if constexpr (kSelectReactantsProfile)
            sr_profile_.bimol_resolve_failures++;
          sr_finish(SrProfile::kPathBimol, /*outcome=*/1);
          return match;
        }
      }

      auto draw = [&](std::size_t n) {
        auto i = static_cast<std::size_t>(uniform() * static_cast<double>(n));
        return (i >= n) ? (n - 1) : i;
      };
      std::size_t const pick_a = draw(filter_a ? ext_a.size() : embs_a.size());
      std::size_t const pick_b = draw(filter_b ? ext_b.size() : embs_b.size());
      int const ei_a = filter_a ? ext_a[pick_a] : static_cast<int>(pick_a);
      int const ei_b = filter_b ? ext_b[pick_b] : static_cast<int>(pick_b);

      match.mol_ids.resize(n_pat_mols, -1);
      match.comp_ids.resize(n_flat, -1);

      // Both sides were checked above, so these cannot fail; the branch is
      // kept so a future divergence between filter and apply surfaces as a
      // null event rather than a half-built match.
      if (!apply_slot(seed_a, end_a, rs.pat_adj_a, mol_a, embs_a[ei_a], match) ||
          !apply_slot(seed_b, end_b, rs.pat_adj_b, mol_b, embs_b[ei_b], match)) {
        match.mol_ids.clear(); // null event — context mismatch
        if constexpr (kSelectReactantsProfile)
          sr_profile_.bimol_resolve_failures++;
        sr_finish(SrProfile::kPathBimol, /*outcome=*/1);
        return match;
      }

      // Injectivity across the two patterns.  `mol_a != mol_b` above only
      // separates the seeds; a molecule pulled into one pattern's complex
      // can be the other pattern's match as well — `A(s,d!1).A(s,d!1) +
      // A(s)` can draw the dimer's second A for slot B — and firing on that
      // consumes one molecule twice, which a DeleteMolecules rule turns into
      // a double delete and a broken mass balance.  The propensity counts
      // those draws, so rejecting them here as null events realizes the
      // injective count exactly, the same way the n-ary path does.  Only
      // multi-molecule patterns can trip it; under `-bscb` the same-complex
      // check has already rejected the draw.  (Ordering note: this consumes
      // no uniform(), so a rule that cannot trip it keeps its draw stream.)
      if (multi_a || multi_b) {
        for (int i = 0; i < n_pat_mols; ++i) {
          for (int j = i + 1; j < n_pat_mols; ++j) {
            if (match.mol_ids[i] >= 0 && match.mol_ids[i] == match.mol_ids[j]) {
              match.mol_ids.clear();
              if constexpr (kSelectReactantsProfile)
                sr_profile_.bimol_same_mol_rejects++;
              sr_finish(SrProfile::kPathBimol, /*outcome=*/1);
              return match;
            }
          }
        }
      }

      sr_finish(SrProfile::kPathBimol, /*outcome=*/2);
      return match;
    }

    // Unimolecular success fall-through (molecularity <= 1 branch above).
    {
      bool const is_multi_pat = rule.reactant_pattern.molecules.size() > 1;
      int const uni_path = is_multi_pat ? (rs.fm_a.enabled ? SrProfile::kPathUniMultiFm
                                                           : SrProfile::kPathUniMultiGen)
                                        : SrProfile::kPathUniSingle;
      sr_finish(uni_path, /*outcome=*/2);
    }
    return match;
  }

  // Check exclude/include reactant/product constraints.
  // Returns true if all constraints are satisfied (reaction should proceed).
  bool check_constraints(const Rule& rule, const ReactionMatch& match) const {
    if (rule.constraints.empty())
      return true;

    for (auto& c : rule.constraints) {
      if (c.is_product)
        continue; // handled separately below

      // Reactant constraint: find the molecule matched for reactant pattern c.pattern_idx
      if (c.pattern_idx >= static_cast<int>(rule.reactant_pattern_starts.size()))
        continue;
      int const seed_mol_idx = rule.reactant_pattern_starts[c.pattern_idx];
      if (seed_mol_idx >= static_cast<int>(match.mol_ids.size()))
        continue;
      int const mol_id = match.mol_ids[seed_mol_idx];
      if (mol_id < 0)
        continue;

      // Check each molecule in the constraint pattern against the actual molecule.
      // Constraint patterns are typically single-molecule.
      bool matches = false;
      for (auto& cpm : c.pattern.molecules) {
        int const emb = count_embeddings_single(pool, mol_id, cpm, model);
        if (emb > 0) {
          matches = true;
          break;
        }
      }

      if (c.is_exclude && matches)
        return false; // excluded pattern matched → reject
      if (!c.is_exclude && !matches)
        return false; // include pattern not matched → reject
    }

    // Product constraints: find the reactant molecule that maps to the product
    // pattern, and check the constraint against it.  This works because product
    // constraints typically check molecular context orthogonal to the rule's
    // transformations (e.g., checking phosphorylation on an unbinding rule).
    for (auto& c : rule.constraints) {
      if (!c.is_product)
        continue;

      if (c.pattern_idx >= static_cast<int>(rule.product_pattern_starts.size()))
        continue;
      int const prod_mol_idx = rule.product_pattern_starts[c.pattern_idx];

      // Find the reactant molecule that maps to this product molecule
      // by reverse-looking through reactant_to_product_map.
      // The product molecule's first component flat index:
      int prod_flat_base = 0;
      for (int mi = 0; mi < prod_mol_idx; ++mi)
        prod_flat_base += static_cast<int>(rule.product_pattern.molecules[mi].components.size());

      // Find a reactant flat comp that maps to this product flat comp
      int reactant_mol_idx = -1;
      for (int ri = 0; ri < static_cast<int>(rule.reactant_to_product_map.size()); ++ri) {
        if (rule.reactant_to_product_map[ri] == prod_flat_base) {
          // Find which reactant molecule this flat index belongs to
          int base = 0;
          for (int mi = 0; mi < static_cast<int>(rule.reactant_pattern.molecules.size()); ++mi) {
            int const nc = static_cast<int>(rule.reactant_pattern.molecules[mi].components.size());
            if (ri >= base && ri < base + nc) {
              reactant_mol_idx = mi;
              break;
            }
            base += nc;
          }
          break;
        }
      }

      if (reactant_mol_idx < 0 || reactant_mol_idx >= static_cast<int>(match.mol_ids.size()))
        continue;
      int const mol_id = match.mol_ids[reactant_mol_idx];
      if (mol_id < 0)
        continue;

      bool matches = false;
      for (auto& cpm : c.pattern.molecules) {
        int const emb = count_embeddings_single(pool, mol_id, cpm, model);
        if (emb > 0) {
          matches = true;
          break;
        }
      }

      if (c.is_exclude && matches)
        return false;
      if (!c.is_exclude && !matches)
        return false;
    }

    return true;
  }

  int sample_molecule_weighted(int type_index, const RuleState& rs, bool use_a) {
    if constexpr (kSelectReactantsProfile)
      sr_profile_.sampler_calls++;
    // A pure-context slot is PRICED per complex, so it must be DRAWN per
    // complex (issue #33).  Drawing per molecule while pricing per complex
    // leaves the same-complex rejection mis-weighted whenever complexes hold
    // unequal numbers of context matches, which is a systematic bias and not
    // a sampling detail — see PerCxTally and context_sampler.bngl.  Which
    // instance inside the complex is returned does not matter to the
    // reaction, only that it is the one the propensity priced.
    if (use_a ? rs.per_complex_a : rs.per_complex_b) {
      const PerCxTally& t = use_a ? rs.percx_a : rs.percx_b;
      if (t.list.empty())
        return -1;
      int const cx =
          t.list[std::min(static_cast<std::size_t>(uniform() * t.list.size()), t.list.size() - 1)];
      auto it = t.entries.find(cx);
      return (it != t.entries.end()) ? it->second.rep : -1;
    }
    auto& mols = pool.molecules_of_type(type_index);
    if (mols.empty()) {
      if constexpr (kSelectReactantsProfile)
        sr_profile_.sampler_empty_pool++;
      return -1;
    }

    // Use pre-computed totals from incremental bookkeeping (avoids O(N) scan)
    double const total = use_a ? rs.a_total : rs.b_total;
    if (total <= 0) {
      if constexpr (kSelectReactantsProfile)
        sr_profile_.sampler_empty_pool++;
      return -1;
    }

    // O(log N) Fenwick tree sampling for large populations
    bool const use_fenwick = use_a ? rs.use_fenwick_a : rs.use_fenwick_b;
    if (use_fenwick) {
      auto& ft = use_a ? rs.fenwick_a : rs.fenwick_b;
      double const r = uniform() * total;
      int const mid = ft.find(r);
      if constexpr (kFenwickInvariant) {
        // Verify mid's mol-id-ordered cumulative range contains r:
        //   prefix_before_mid <= r < prefix_through_mid
        // (Linear-scan in creation order would land on a different mid for
        // the same r — both are statistically valid; this checks Fenwick's
        // bit-descent answer matches its own prefix-sum semantics.)
        double prefix_through_mid = 0.0;
        int const last_m = std::min(mid, pool.molecule_count() - 1);
        for (int m = 0; m <= last_m; ++m) {
          if (!pool.molecule(m).active)
            continue;
          if (pool.molecule(m).type_index != type_index)
            continue;
          if (m < static_cast<int>(rs.mol_data.size()))
            prefix_through_mid += use_a ? rs.mol_data[m].count_a : rs.mol_data[m].count_b;
        }
        double const mid_weight =
            (mid >= 0 && mid < static_cast<int>(rs.mol_data.size()))
                ? (use_a ? rs.mol_data[mid].count_a : rs.mol_data[mid].count_b)
                : 0.0;
        double const prefix_before_mid = prefix_through_mid - mid_weight;
        if (prefix_before_mid > r || r >= prefix_through_mid) {
          fprintf(stderr,
                  "[fw INVARIANT FAIL] type=%d  use_a=%d  r=%.6f  total=%.6f"
                  "  mid=%d  prefix_before=%.6f  prefix_through=%.6f"
                  "  tree_sum=%.6f\n",
                  type_index, static_cast<int>(use_a), r, total, mid, prefix_before_mid,
                  prefix_through_mid, ft.sum());
          std::abort();
        }
      }
      // Validate: must be active and of correct type
      bool const valid_mid = (mid >= 0 && mid < pool.molecule_count());
      bool const active = valid_mid && pool.molecule(mid).active;
      bool const type_ok = active && pool.molecule(mid).type_index == type_index;
      if (type_ok) {
        if constexpr (kSelectReactantsProfile)
          sr_profile_.sampler_fenwick_uses++;
        return mid;
      }
      if constexpr (kSelectReactantsProfile) {
        sr_profile_.sampler_fenwick_drifts++;
        if (!valid_mid) {
          sr_profile_.sampler_drift_invalid_mid++;
          double const tree_sum = ft.sum();
          if (r >= tree_sum) {
            sr_profile_.sampler_drift_target_eq_sum++;
            sr_profile_.sampler_drift_excess_sum += (r - tree_sum);
            sr_profile_.sampler_drift_total_sum += total;
          } else {
            sr_profile_.sampler_drift_target_lt_sum++;
          }
        } else if (!active)
          sr_profile_.sampler_drift_inactive_mol++;
        else
          sr_profile_.sampler_drift_type_mismatch++;
      }
      // Fenwick tree drift — fall through to linear scan
    }

    if constexpr (kSelectReactantsProfile)
      sr_profile_.sampler_linear_calls++;

    // O(N) linear scan fallback
    double const r = uniform() * total;
    double cum = 0;
    for (int const mid : mols) {
      if (!pool.molecule(mid).active)
        continue;
      if (mid < static_cast<int>(rs.mol_data.size())) {
        cum += use_a ? rs.mol_data[mid].count_a : rs.mol_data[mid].count_b;
        if (r < cum)
          return mid;
      }
    }
    return mols.back(); // rounding fallback
  }

  // Sample a molecule weighted by per-molecule local propensity (for local-rate
  // rules).  `use_b` selects the FunctionProduct B-side weights
  // (local_propensity_b / local_propensity_b_total) instead of the A-side.
  int sample_molecule_by_local_propensity(int type_index, const RuleState& rs, bool use_b = false) {
    if constexpr (kSelectReactantsProfile)
      sr_profile_.sampler_local_prop_calls++;
    // Pure-context slot: the propensity is Σ over complexes of the
    // representative's rate, so draw a complex with that same weight and fire
    // that representative (issue #33).
    if (use_b ? rs.per_complex_b : rs.per_complex_a) {
      const PerCxTally& t = use_b ? rs.percx_b : rs.percx_a;
      if (t.list.empty() || t.rate_sum <= 0)
        return -1;
      double const r = uniform() * t.rate_sum;
      double cum = 0;
      for (int const cx : t.list) {
        auto it = t.entries.find(cx);
        if (it == t.entries.end())
          continue;
        cum += it->second.rep_rate;
        if (r < cum)
          return it->second.rep;
      }
      auto last = t.entries.find(t.list.back());
      return (last != t.entries.end()) ? last->second.rep : -1;
    }
    auto& mols = pool.molecules_of_type(type_index);
    if (mols.empty())
      return -1;

    double const total = use_b ? rs.local_propensity_b_total : rs.local_propensity_total;
    if (total <= 0)
      return -1;

    double const r = uniform() * total;
    double cum = 0;
    for (int const mid : mols) {
      if (!pool.molecule(mid).active)
        continue;
      if (mid < static_cast<int>(rs.mol_aux.size())) {
        cum += use_b ? rs.mol_aux[mid].local_propensity_b : rs.mol_aux[mid].local_propensity;
        if (r < cum)
          return mid;
      }
    }
    return mols.back(); // rounding fallback
  }

  // Resolve non-seed molecules in a reactant pattern via BFS through bonds.
  // Given a seed molecule already placed at seed_pat_idx, follows pattern
  // bonds within [pat_start, pat_end) to discover the other molecules.
  // Fills match.mol_ids and match.comp_ids for the resolved molecules.
  // Returns false if the BFS fails (molecule not found or type mismatch).
  bool resolve_pattern_context(const Pattern& pat, const PatternAdj& pa, int pat_start, int pat_end,
                               int seed_pat_idx, int seed_mol_id,
                               const std::vector<int>& seed_comp_map, ReactionMatch& match) const {

    int const n_pat_mols = static_cast<int>(pat.molecules.size());

    // Bond adjacency restricted to [pat_start, pat_end) is pre-computed in
    // pa (rs.pat_adj_a/_b), built at init via build_pattern_adjacency with
    // the same range.  Reuse it instead of rebuilding per call.
    auto& adj = pa.adj;

    // BFS from seed
    std::vector<int> mol_assignments(n_pat_mols, -1);
    std::vector<std::vector<int>> comp_maps(n_pat_mols);
    mol_assignments[seed_pat_idx] = seed_mol_id;
    comp_maps[seed_pat_idx] = seed_comp_map;

    std::queue<int> bfs_queue;
    bfs_queue.push(seed_pat_idx);

    while (!bfs_queue.empty()) {
      int const cur_pat = bfs_queue.front();
      bfs_queue.pop();
      int const cur_actual = mol_assignments[cur_pat];
      auto& cur_comp_map = comp_maps[cur_pat];

      for (auto& ae : adj[cur_pat]) {
        int const other_pat = ae.other_mol;
        if (mol_assignments[other_pat] >= 0)
          continue; // already resolved

        if (ae.my_local >= static_cast<int>(cur_comp_map.size()))
          return false;
        int const my_actual_local = cur_comp_map[ae.my_local];
        int const my_actual_comp_id = pool.molecule(cur_actual).comp_ids[my_actual_local];
        int const partner_comp_id = pool.component(my_actual_comp_id).bond_partner;
        if (partner_comp_id < 0)
          return false;

        int const partner_mol_id = pool.mol_of_comp(partner_comp_id);
        auto& other_pm = pat.molecules[other_pat];
        if (pool.molecule(partner_mol_id).type_index != other_pm.type_index)
          return false;

        // Find compatible embedding
        std::vector<std::vector<int>> other_embs;
        count_embeddings_single(pool, partner_mol_id, other_pm, model, &other_embs);

        int partner_local = -1;
        for (int ci = 0; ci < static_cast<int>(pool.molecule(partner_mol_id).comp_ids.size());
             ++ci) {
          if (pool.molecule(partner_mol_id).comp_ids[ci] == partner_comp_id) {
            partner_local = ci;
            break;
          }
        }

        bool found = false;
        for (auto& emb : other_embs) {
          if (ae.other_local < static_cast<int>(emb.size()) &&
              emb[ae.other_local] == partner_local) {
            mol_assignments[other_pat] = partner_mol_id;
            comp_maps[other_pat] = emb;
            bfs_queue.push(other_pat);
            found = true;
            break;
          }
        }
        if (!found)
          return false;
      }
    }

    // Verify all molecules in range are resolved
    for (int mi = pat_start; mi < pat_end; ++mi) {
      if (mol_assignments[mi] < 0)
        return false;
    }

    // Fill match
    int base = 0;
    for (int mi = 0; mi < n_pat_mols; ++mi) {
      int const nc = static_cast<int>(pat.molecules[mi].components.size());
      if (mi >= pat_start && mi < pat_end && mi != seed_pat_idx) {
        match.mol_ids[mi] = mol_assignments[mi];
        auto& cmap = comp_maps[mi];
        for (int ci = 0; ci < nc && ci < static_cast<int>(cmap.size()); ++ci)
          match.comp_ids[base + ci] = pool.molecule(mol_assignments[mi]).comp_ids[cmap[ci]];
      }
      base += nc;
    }
    return true;
  }

  // P6 fast path — constructs a ReactionMatch directly from a FastMatchSlot
  // descriptor for the 2-mol-1-bond-fully-constrained template, without
  // enumerating all seed pattern embeddings or running the generic BFS.
  //
  // Trajectory invariant (see kFastSelectInvariant): this path must consume
  // exactly the same uniform() draws as select_multi_mol_unimolecular would on
  // the same seed_mol_id, so the RNG stream stays bit-exact.  The generic
  // path's draws all come from a single Fisher-Yates shuffle over seed_embs
  // with `n_se - 1` uniform() calls (see select_multi_mol_unimolecular,
  // line ~3708 in pre-P6 HEAD).  We reproduce that here: enumerate valid
  // seed bond-endpoint candidates in the same order count_embeddings_single
  // would, Fisher-Yates them with identical draws, then pick the first whose
  // partner satisfies the partner-side constraints — exactly the generic
  // "shuffle + first-extending-seed-emb" behavior.
  ReactionMatch select_2mol_1bond_fc_match(const Rule& rule, const RuleState& rs, int seed_mol_id) {
    const auto& pat = rule.reactant_pattern;
    const auto& fm = rs.fm_a;
    ReactionMatch match;

    const auto& seed = pool.molecule(seed_mol_id);
    if (!seed.active)
      return match;
    if (seed.type_index != fm.seed_type)
      return match;
    int const n_seed = static_cast<int>(seed.comp_ids.size());

    // Mol-level non-endpoint checks on seed (only populated on non-symmetric
    // K>1 sides).  If any fails, count_embeddings_single would return 0 and
    // the generic would bail with zero draws — match accordingly.
    for (const auto& chk : fm.seed_non_bond_checks) {
      if (chk.local_idx >= n_seed)
        return match;
      const auto& c = pool.component(seed.comp_ids[chk.local_idx]);
      if (chk.state_req >= 0 && c.state_index != chk.state_req)
        return match;
      if (chk.bond_req == 1 && c.bond_partner >= 0)
        return match;
      if (chk.bond_req == 2 && c.bond_partner < 0)
        return match;
    }

    // Enumerate valid seed bond-endpoint locals in the same order
    // count_embeddings_single does (ci ascending).  Each surviving entry
    // corresponds 1:1 with an entry in seed_embs.
    std::vector<int> seed_locals;
    seed_locals.reserve(fm.seed_bond_candidates.size());
    for (int const c : fm.seed_bond_candidates) {
      if (c >= n_seed)
        continue;
      const auto& sc = pool.component(seed.comp_ids[c]);
      if (fm.seed_bond_state_req >= 0 && sc.state_index != fm.seed_bond_state_req)
        continue;
      if (fm.seed_bond_bond_req == 1 && sc.bond_partner >= 0)
        continue;
      if (fm.seed_bond_bond_req == 2 && sc.bond_partner < 0)
        continue;
      // Bond tracing requires a partner regardless of pat bond_req.  The
      // generic BFS fails such a seed with a null event; we filter it
      // before the shuffle so the RNG-draw count matches seed_embs.size()
      // from count_embeddings_single (which for Bound/BoundTo templates
      // only emits embs where the endpoint has a partner).
      if (sc.bond_partner < 0)
        continue;
      seed_locals.push_back(c);
    }
    if (seed_locals.empty())
      return match;

    // Fisher-Yates shuffle — matches select_multi_mol_unimolecular's loop:
    //   for (int i = n_se - 1; i > 0; --i) {
    //     int j = (int)(uniform() * (i + 1));
    //     if (j > i) j = i;
    //     std::swap(seed_embs[i], seed_embs[j]);
    //   }
    int const n_se = static_cast<int>(seed_locals.size());
    for (int i = n_se - 1; i > 0; --i) {
      int j = static_cast<int>(uniform() * (i + 1));
      j = std::min(j, i);
      std::swap(seed_locals[i], seed_locals[j]);
    }

    // Walk shuffled candidates; take the first whose partner satisfies the
    // full partner-side template.  In the generic code, this mirrors BFS
    // extension on a single-bond pattern: find the partner mol, check type,
    // check partner_local lies in partner_bond_candidates, check dynamic
    // state/bond constraints on both endpoint and non-endpoint pat comps.
    int chosen_seed_local = -1;
    int partner_mol_id = -1;
    int partner_local = -1;
    for (int const seed_ci : seed_locals) {
      int const seed_cid = seed.comp_ids[seed_ci];
      int const partner_cid = pool.component(seed_cid).bond_partner;
      if (partner_cid < 0)
        continue;

      int const pmol = pool.mol_of_comp(partner_cid);
      if (pmol == seed_mol_id)
        continue; // distinctness from seed
      const auto& pm = pool.molecule(pmol);
      if (!pm.active)
        continue;
      if (pm.type_index != fm.partner_type)
        continue;

      int const n_partner = static_cast<int>(pm.comp_ids.size());
      int plocal = -1;
      for (int k = 0; k < n_partner; ++k) {
        if (pm.comp_ids[k] == partner_cid) {
          plocal = k;
          break;
        }
      }
      if (plocal < 0)
        continue;

      bool plocal_ok = false;
      for (int const x : fm.partner_bond_candidates)
        if (x == plocal) {
          plocal_ok = true;
          break;
        }
      if (!plocal_ok)
        continue;

      if (fm.partner_bond_state_req >= 0 &&
          pool.component(partner_cid).state_index != fm.partner_bond_state_req)
        continue;

      bool ok = true;
      for (const auto& chk : fm.partner_non_bond_checks) {
        if (chk.local_idx >= n_partner) {
          ok = false;
          break;
        }
        if (chk.local_idx == plocal)
          continue;
        const auto& pc = pool.component(pm.comp_ids[chk.local_idx]);
        if (chk.state_req >= 0 && pc.state_index != chk.state_req) {
          ok = false;
          break;
        }
        if (chk.bond_req == 1 && pc.bond_partner >= 0) {
          ok = false;
          break;
        }
        if (chk.bond_req == 2 && pc.bond_partner < 0) {
          ok = false;
          break;
        }
      }
      if (!ok)
        continue;

      chosen_seed_local = seed_ci;
      partner_mol_id = pmol;
      partner_local = plocal;
      break;
    }

    if (chosen_seed_local < 0)
      return match; // null event (matches generic)

    // Build ReactionMatch with the same shape as select_multi_mol_unimolecular
    // (mol_ids sized to n_pat_mols, comp_ids sized to flat count).
    int const n_pat_mols = static_cast<int>(pat.molecules.size());
    match.mol_ids.assign(n_pat_mols, -1);
    match.mol_ids[fm.seed_pat_idx] = seed_mol_id;
    match.mol_ids[fm.partner_pat_idx] = partner_mol_id;

    int const n_flat = pat.flat_comp_count();
    match.comp_ids.assign(n_flat, -1);

    auto fill_side = [&](int pat_mi, int bond_pi, int chosen_local,
                         const std::vector<int>& ci_locals, const MoleculeInstance& mol) {
      int base = 0;
      for (int mi = 0; mi < pat_mi; ++mi)
        base += static_cast<int>(pat.molecules[mi].components.size());
      int const K = static_cast<int>(pat.molecules[pat_mi].components.size());
      int const n_mol = static_cast<int>(mol.comp_ids.size());
      for (int ci = 0; ci < K; ++ci) {
        int const local = (ci == bond_pi)
                              ? chosen_local
                              : (ci < static_cast<int>(ci_locals.size()) ? ci_locals[ci] : -1);
        if (local >= 0 && local < n_mol)
          match.comp_ids[base + ci] = mol.comp_ids[local];
      }
    };
    fill_side(fm.seed_pat_idx, fm.seed_bond_pi, chosen_seed_local, fm.seed_pat_ci_locals, seed);
    fill_side(fm.partner_pat_idx, fm.partner_bond_pi, partner_local, fm.partner_pat_ci_locals,
              pool.molecule(partner_mol_id));

    return match;
  }

  // Resolve a full multi-molecule match for a unimolecular rule.
  // Given the seed molecule (seed_mol_id at pattern position seed_a),
  // follow bonds in the pattern to discover the other molecules.
  // Reuses the same BFS logic as count_multi_molecule_embeddings.
  ReactionMatch select_multi_mol_unimolecular(const Rule& rule, const PatternAdj& pa,
                                              int seed_mol_id, int seed_a) {
    auto& pat = rule.reactant_pattern;
    int n_pat_mols = static_cast<int>(pat.molecules.size());
    ReactionMatch match;

    // Bond adjacency for the (entire-pattern) reactant A is pre-computed in
    // pa (= rs.pat_adj_a, built at init via build_pattern_adjacency over the
    // full unimolecular pattern range).  Reuse it instead of rebuilding per
    // call.
    auto& adj = pa.adj;

    // Get embeddings of seed pattern molecule into seed_mol_id
    std::vector<std::vector<int>> seed_embs;
    count_embeddings_single(pool, seed_mol_id, pat.molecules[seed_a], model, &seed_embs);
    if (seed_embs.empty())
      return match;

    // Shuffle seed embeddings to avoid bias, then try each until one extends
    // to a complete multi-molecule match.  A seed embedding may fail to extend
    // when the bond partner doesn't satisfy downstream pattern constraints
    // (e.g., R(r!1).L(l!1,l) where the specific r leads to a fully-bound L).
    // Without retrying, this produces spurious null events that systematically
    // reduce the effective unbinding rate.
    {
      int const n_se = static_cast<int>(seed_embs.size());
      // Fisher-Yates shuffle
      for (int i = n_se - 1; i > 0; --i) {
        int j = static_cast<int>(uniform() * (i + 1));
        j = std::min(j, i);
        std::swap(seed_embs[i], seed_embs[j]);
      }
    }

    std::vector<int> mol_assignments(n_pat_mols, -1);
    std::vector<std::vector<int>> comp_maps(n_pat_mols);
    bool found_valid_seed = false;

    for (auto& seed_comp_map : seed_embs) {
      // BFS from seed to resolve all pattern molecules
      std::fill(mol_assignments.begin(), mol_assignments.end(), -1);
      for (auto& cm : comp_maps)
        cm.clear();
      mol_assignments[seed_a] = seed_mol_id;
      comp_maps[seed_a] = seed_comp_map;

      std::queue<int> bfs_queue;
      bfs_queue.push(seed_a);
      bool valid = true;

      while (!bfs_queue.empty() && valid) {
        int const cur_pat = bfs_queue.front();
        bfs_queue.pop();
        int const cur_actual = mol_assignments[cur_pat];
        auto& cur_comp_map = comp_maps[cur_pat];

        for (auto& ae : adj[cur_pat]) {
          int const other_pat = ae.other_mol;
          if (ae.my_local >= static_cast<int>(cur_comp_map.size())) {
            valid = false;
            break;
          }
          int const my_actual_local = cur_comp_map[ae.my_local];
          int const my_actual_comp_id = pool.molecule(cur_actual).comp_ids[my_actual_local];
          int const partner_comp_id = pool.component(my_actual_comp_id).bond_partner;
          if (partner_comp_id < 0) {
            valid = false;
            break;
          }

          int const partner_mol_id = pool.mol_of_comp(partner_comp_id);

          if (mol_assignments[other_pat] >= 0) {
            if (mol_assignments[other_pat] != partner_mol_id) {
              valid = false;
              break;
            }
            auto& other_comp_map = comp_maps[other_pat];
            if (ae.other_local >= static_cast<int>(other_comp_map.size())) {
              valid = false;
              break;
            }
            int const expected_local = other_comp_map[ae.other_local];
            int const expected_comp_id = pool.molecule(partner_mol_id).comp_ids[expected_local];
            if (expected_comp_id != partner_comp_id) {
              valid = false;
              break;
            }
            continue;
          }

          auto& other_pm = pat.molecules[other_pat];
          if (pool.molecule(partner_mol_id).type_index != other_pm.type_index) {
            valid = false;
            break;
          }

          std::vector<std::vector<int>> other_embs;
          count_embeddings_single(pool, partner_mol_id, other_pm, model, &other_embs);

          int partner_local = -1;
          for (int ci = 0; ci < static_cast<int>(pool.molecule(partner_mol_id).comp_ids.size());
               ++ci) {
            if (pool.molecule(partner_mol_id).comp_ids[ci] == partner_comp_id) {
              partner_local = ci;
              break;
            }
          }

          bool found = false;
          for (auto& emb : other_embs) {
            if (ae.other_local < static_cast<int>(emb.size()) &&
                emb[ae.other_local] == partner_local) {
              mol_assignments[other_pat] = partner_mol_id;
              comp_maps[other_pat] = emb;
              bfs_queue.push(other_pat);
              found = true;
              break;
            }
          }
          if (!found) {
            valid = false;
            break;
          }
        }
      }

      if (!valid)
        continue; // try next seed embedding

      found_valid_seed = true;
      break;
    }

    if (!found_valid_seed)
      return match;

    // Handle unassigned (disjoint) pattern molecules not reachable via bonds.
    {
      std::vector<int> unassigned;
      for (int mi = 0; mi < n_pat_mols; ++mi) {
        if (mol_assignments[mi] < 0)
          unassigned.push_back(mi);
      }

      if (!unassigned.empty()) {
        int const cx = pool.complex_of(seed_mol_id);
        auto cx_members = pool.molecules_in_complex(cx);

        // Reservoir sampling (k=1) instead of materializing every valid
        // complete assignment and picking one at the end.  Maintains a
        // single chosen snapshot; on the n-th valid assignment, replace
        // chosen with probability 1/n.  Equivalent to the prior uniform
        // pick over the full enumeration, without the O(N) memory and
        // per-assignment vector copies.
        //
        // The std::function-wrapped recursion is rewritten as a
        // self-passing auto lambda so the compiler can inline through the
        // recursive call (same pattern as #14's count_embeddings_single
        // rewrite — std::function went via vtable + heap alloc per call).
        int n_valid = 0;
        std::vector<int> chosen_mols;
        std::vector<std::vector<int>> chosen_cmaps;

        auto assign_unassigned = [&](auto& self, int ui) -> void {
          if (ui == static_cast<int>(unassigned.size())) {
            if (all_distinct_molecules(mol_assignments, 0, n_pat_mols) &&
                check_pattern_bonds(pool, pat, mol_assignments, comp_maps)) {
              ++n_valid;
              if (n_valid == 1 || uniform() * n_valid < 1.0) {
                chosen_mols = mol_assignments;
                chosen_cmaps = comp_maps;
              }
            }
            return;
          }
          int const pat_mi = unassigned[ui];
          auto& target_pm = pat.molecules[pat_mi];
          for (int const cand : cx_members) {
            if (!pool.molecule(cand).active)
              continue;
            if (pool.molecule(cand).type_index != target_pm.type_index)
              continue;
            bool already_used = false;
            for (int mi2 = 0; mi2 < n_pat_mols; ++mi2) {
              if (mi2 != pat_mi && mol_assignments[mi2] == cand) {
                already_used = true;
                break;
              }
            }
            if (already_used)
              continue;

            std::vector<std::vector<int>> cand_embs;
            count_embeddings_single(pool, cand, target_pm, model, &cand_embs);
            for (auto& emb : cand_embs) {
              mol_assignments[pat_mi] = cand;
              comp_maps[pat_mi] = emb;
              self(self, ui + 1);
            }
            mol_assignments[pat_mi] = -1;
            comp_maps[pat_mi].clear();
          }
        };
        assign_unassigned(assign_unassigned, 0);

        if (n_valid == 0)
          return match;

        mol_assignments = std::move(chosen_mols);
        comp_maps = std::move(chosen_cmaps);
      } else {
        // All assigned via BFS — check distinctness
        if (!all_distinct_molecules(mol_assignments, 0, n_pat_mols))
          return match;
      }
    }

    // Build the match
    match.mol_ids = mol_assignments;
    int const n_flat = pat.flat_comp_count();
    match.comp_ids.resize(n_flat, -1);
    int base = 0;
    for (int mi = 0; mi < n_pat_mols; ++mi) {
      int const nc = static_cast<int>(pat.molecules[mi].components.size());
      int const actual_mid = mol_assignments[mi];
      if (actual_mid >= 0) {
        auto& cmap = comp_maps[mi];
        for (int ci = 0; ci < nc && ci < static_cast<int>(cmap.size()); ++ci) {
          match.comp_ids[base + ci] = pool.molecule(actual_mid).comp_ids[cmap[ci]];
        }
      }
      base += nc;
    }

    return match;
  }

  // --- Graph rewriting ---

  struct FireResult {
    std::unordered_set<int> affected;
    bool bond_changed; // true if any AddBond or DeleteBond fired
  };

  FireResult fire_rule(int rule_idx, const ReactionMatch& match) {
    auto& rule = model.rules[rule_idx];
    std::unordered_set<int> affected;
    std::vector<int> deleted_mols; // track deletions for count bookkeeping
    bool bond_changed = false;

    // Profile sampling: every Kth call is timed at sub-phase resolution.
    bool prof_sampled = false;
    std::chrono::steady_clock::time_point prof_t_total_start;
    if constexpr (kFireRuleProfile) {
      fire_profile_.calls++;
      prof_sampled = (fire_profile_.calls % kFireRuleProfileSampleEvery) == 0;
      if (prof_sampled) {
        fire_profile_.sampled_calls++;
        prof_t_total_start = std::chrono::steady_clock::now();
      }
    }

    // P1 cache: advance the per-event epoch.  Any mol whose stored epoch
    // doesn't match `event_epoch` is treated as unchanged this event.
    ++event_epoch;

    auto ensure_mask_capacity = [&](int mid) {
      if (mid >= static_cast<int>(event_mol_change_mask.size())) {
        if constexpr (kFireRuleProfile)
          fire_profile_.ensure_mask_resizes++;
        int const new_n = std::max(mid + 1, 2 * static_cast<int>(event_mol_change_mask.size()));
        event_mol_change_mask.resize(new_n, 0);
        event_mol_change_epoch.resize(new_n, 0);
      }
    };

    auto mark_mol = [&](int mid, uint64_t bits) {
      if (mid < 0)
        return;
      if constexpr (kFireRuleProfile)
        fire_profile_.mark_mol_calls++;
      ensure_mask_capacity(mid);
      if (event_mol_change_epoch[mid] != event_epoch) {
        event_mol_change_epoch[mid] = event_epoch;
        event_mol_change_mask[mid] = 0;
      }
      event_mol_change_mask[mid] |= bits;
    };

    // Mark a single component as mutated this event.  Resolves comp_id to
    // (mol, local_ci) and sets the corresponding bit.
    auto mark_comp = [&](int comp_id) {
      if (comp_id < 0)
        return;
      if constexpr (kFireRuleProfile)
        fire_profile_.mark_comp_calls++;
      int const mid = pool.mol_of_comp(comp_id);
      if (mid < 0)
        return;
      auto& mol = pool.molecule(mid);
      for (int i = 0; i < static_cast<int>(mol.comp_ids.size()); ++i) {
        if (mol.comp_ids[i] == comp_id) {
          if (i < 64)
            mark_mol(mid, uint64_t{1} << i);
          return;
        }
      }
    };

    // Collect initial affected molecules
    for (int const mid : match.mol_ids)
      if (mid >= 0)
        affected.insert(mid);

    // Track newly added molecules for AddBond operations that reference products.
    // Maps product-pattern molecule index → actual molecule ID.
    std::unordered_map<int, int> product_mol_to_actual;

    std::chrono::steady_clock::time_point prof_t_switch_start;
    if constexpr (kFireRuleProfile) {
      if (prof_sampled)
        prof_t_switch_start = std::chrono::steady_clock::now();
    }

    for (auto& op : rule.operations) {
      std::chrono::steady_clock::time_point prof_t_op_start;
      if constexpr (kFireRuleProfile) {
        int const idx = static_cast<int>(op.type);
        if (idx >= 0 && idx < 5)
          fire_profile_.op_calls[idx]++;
        if (prof_sampled)
          prof_t_op_start = std::chrono::steady_clock::now();
      }
      switch (op.type) {
      case OpType::StateChange: {
        if (op.comp_flat < 0 || op.comp_flat >= static_cast<int>(match.comp_ids.size()))
          break;
        int const comp_id = match.comp_ids[op.comp_flat];
        if (comp_id < 0)
          break;
        if (op.is_increment) {
          pool.increment_state(comp_id);
        } else if (op.is_decrement) {
          pool.decrement_state(comp_id);
        } else {
          pool.set_state(comp_id, op.new_state_index);
        }
        affected.insert(pool.mol_of_comp(comp_id));
        mark_comp(comp_id);
        break;
      }

      case OpType::DeleteBond: {
        if (op.comp_flat_a < 0 || op.comp_flat_a >= static_cast<int>(match.comp_ids.size()))
          break;
        int const comp_a = match.comp_ids[op.comp_flat_a];
        if (comp_a < 0)
          break;
        int const partner = pool.component(comp_a).bond_partner;
        if (partner >= 0) {
          affected.insert(pool.mol_of_comp(partner));
          mark_comp(partner);
        }
        affected.insert(pool.mol_of_comp(comp_a));
        mark_comp(comp_a);
        pool.remove_bond(comp_a);
        bond_changed = true;
        break;
      }

      case OpType::AddBond: {
        int comp_a = -1, comp_b = -1;

        if (op.comp_flat_a >= 0 && op.comp_flat_a < static_cast<int>(match.comp_ids.size()))
          comp_a = match.comp_ids[op.comp_flat_a];
        if (op.comp_flat_b >= 0 && op.comp_flat_b < static_cast<int>(match.comp_ids.size()))
          comp_b = match.comp_ids[op.comp_flat_b];

        // Resolve references to newly added product molecules
        if (comp_a < 0 && op.product_mol_a >= 0 && op.product_comp_a >= 0) {
          auto it = product_mol_to_actual.find(op.product_mol_a);
          if (it != product_mol_to_actual.end()) {
            auto& mol = pool.molecule(it->second);
            if (op.product_comp_a < static_cast<int>(mol.comp_ids.size()))
              comp_a = mol.comp_ids[op.product_comp_a];
          }
        }
        if (comp_b < 0 && op.product_mol_b >= 0 && op.product_comp_b >= 0) {
          auto it = product_mol_to_actual.find(op.product_mol_b);
          if (it != product_mol_to_actual.end()) {
            auto& mol = pool.molecule(it->second);
            if (op.product_comp_b < static_cast<int>(mol.comp_ids.size()))
              comp_b = mol.comp_ids[op.product_comp_b];
          }
        }

        if (comp_a >= 0 && comp_b >= 0) {
          pool.add_bond(comp_a, comp_b);
          affected.insert(pool.mol_of_comp(comp_a));
          affected.insert(pool.mol_of_comp(comp_b));
          mark_comp(comp_a);
          mark_comp(comp_b);
          bond_changed = true;
        }
        break;
      }

      case OpType::AddMolecule: {
        int const new_mid = pool.add_molecule(op.add_spec.type_index);
        for (auto& [ci, si] : op.add_spec.comp_states) {
          auto& new_mol = pool.molecule(new_mid);
          if (ci < static_cast<int>(new_mol.comp_ids.size()))
            pool.set_state(new_mol.comp_ids[ci], si);
        }
        // Force a cache miss on the new molecule.  pool.add_molecule may
        // reuse a freed mol_id slot, and PerMolRuleData[new_mid] can
        // carry a stale cache_init=true flag from the previous tenant.
        // Marking every bit changed guarantees every rule's relevant
        // mask intersects, so incremental_update recomputes from scratch.
        mark_mol(new_mid, ~uint64_t{0});
        affected.insert(new_mid);
        if (op.add_product_mol_idx >= 0)
          product_mol_to_actual[op.add_product_mol_idx] = new_mid;
        break;
      }

      case OpType::DeleteMolecule: {
        if (op.delete_pattern_mol_idx < 0 ||
            op.delete_pattern_mol_idx >= static_cast<int>(match.mol_ids.size()))
          break;
        int const mid = match.mol_ids[op.delete_pattern_mol_idx];
        if (mid < 0)
          break;

        if (op.delete_connected) {
          // Delete entire complex
          int const cx = pool.complex_of(mid);
          auto members = pool.molecules_in_complex(cx); // copy
          for (int const m : members) {
            deleted_mols.push_back(m);
            affected.erase(m);
            pool.delete_molecule(m);
          }
        } else {
          // Collect neighbors before deletion.  Each neighbor's bonded
          // component loses its bond — mark that component mutated so
          // the cache invalidates correctly on the neighbor.
          auto& mol = pool.molecule(mid);
          for (int const cid : mol.comp_ids) {
            int const p = pool.component(cid).bond_partner;
            if (p >= 0) {
              affected.insert(pool.mol_of_comp(p));
              mark_comp(p);
            }
          }
          deleted_mols.push_back(mid);
          affected.erase(mid);
          pool.delete_molecule(mid);
        }
        break;
      }
      }
      if constexpr (kFireRuleProfile) {
        if (prof_sampled) {
          auto prof_t_op_end = std::chrono::steady_clock::now();
          int const idx = static_cast<int>(op.type);
          if (idx >= 0 && idx < 5)
            fire_profile_.op_ns[idx] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            prof_t_op_end - prof_t_op_start)
                                            .count();
        }
      }
    }

    std::chrono::steady_clock::time_point prof_t_cleanup_start;
    if constexpr (kFireRuleProfile) {
      if (prof_sampled) {
        prof_t_cleanup_start = std::chrono::steady_clock::now();
        fire_profile_.switch_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       prof_t_cleanup_start - prof_t_switch_start)
                                       .count();
      }
    }

    // Remove deleted molecules from affected set
    std::vector<int> to_remove;
    for (int const mid : affected) {
      if (mid < 0 || mid >= pool.molecule_count() || !pool.molecule(mid).active)
        to_remove.push_back(mid);
    }
    for (int const mid : to_remove)
      affected.erase(mid);

    // Re-include deleted molecules so incremental_update can zero out their
    // stale embedding counts from rule state totals (a_total, b_total, etc.)
    for (int const mid : deleted_mols)
      if (mid >= 0 && mid < pool.molecule_count())
        affected.insert(mid);

    std::chrono::steady_clock::time_point prof_t_bfs_start;
    if constexpr (kFireRuleProfile) {
      if (prof_sampled) {
        prof_t_bfs_start = std::chrono::steady_clock::now();
        fire_profile_.cleanup_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        prof_t_bfs_start - prof_t_cleanup_start)
                                        .count();
      }
      if (max_pattern_depth > 0)
        fire_profile_.bfs_fires++;
    }

    // Expand affected set to include molecules within max_pattern_depth
    // bonds of any directly-affected molecule.  This replaces the old full-
    // complex expansion: only molecules whose multi-mol embedding counts
    // could have changed need reprocessing.
    std::unordered_set<int> expanded = affected;
    if (max_pattern_depth > 0) {
      std::queue<std::pair<int, int>> bfs_q;
      for (int const mid : affected) {
        if (mid >= 0 && mid < pool.molecule_count() && pool.molecule(mid).active)
          bfs_q.emplace(mid, 0);
      }
      while (!bfs_q.empty()) {
        auto [mid, depth] = bfs_q.front();
        bfs_q.pop();
        if (depth >= max_pattern_depth)
          continue;
        auto& mol = pool.molecule(mid);
        for (int const cid : mol.comp_ids) {
          int const partner = pool.component(cid).bond_partner;
          if (partner < 0)
            continue;
          int const neighbor = pool.mol_of_comp(partner);
          if (pool.molecule(neighbor).active && !expanded.count(neighbor)) {
            expanded.insert(neighbor);
            bfs_q.emplace(neighbor, depth + 1);
          }
        }
      }
    }

    if constexpr (kFireRuleProfile) {
      if (prof_sampled) {
        auto prof_t_total_end = std::chrono::steady_clock::now();
        fire_profile_.bfs_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    prof_t_total_end - prof_t_bfs_start)
                                    .count();
        fire_profile_.total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      prof_t_total_end - prof_t_total_start)
                                      .count();
      }
      auto sz = static_cast<uint64_t>(expanded.size());
      fire_profile_.affected_final_sum += sz;
      fire_profile_.affected_final_max = std::max(sz, fire_profile_.affected_final_max);
    }

    return {std::move(expanded), bond_changed};
  }

  // --- Fixed-species clamping ---

  // Restore each declared Fixed species' population to its target count
  // by synthesizing or deleting singleton-complex molecules of the
  // matching type+state.  Called after every fire_rule() and before
  // incremental_update(), so the event's effect on a Fixed-species
  // population is undone within the same event.
  //
  // v1 matching rule (enforced at load time in simulator.cpp):
  //   - mol is of type FixedSpecies::mol_type_idx
  //   - mol is alone in its complex (no bonds)
  //   - each component in required_comp_state with value >= 0 matches
  //
  // Creates/deletes thread through the existing `affected` set so the
  // subsequent incremental_update pass picks them up.  New mols have
  // their cache bits fully invalidated (same treatment as
  // OpType::AddMolecule) so every rule's relevant mask intersects and
  // the rule-recompute path rebuilds from scratch.  Deletions wipe
  // `affected` membership and call pool.delete_molecule but do NOT
  // invalidate cache bits on the deleted mid itself — this matches
  // OpType::DeleteMolecule's convention (see ~line 6317).  The freed
  // slot's next tenant always re-invalidates at add time via
  // mark_mol_all_bits (here) or mark_mol (fire_rule AddMolecule op),
  // so stale bits cannot leak into a future event.  Fixed v1 matches
  // only singletons, so there are also no neighbors to mark.
  void replenish_fixed_species(std::unordered_set<int>& affected) {
    if (model.fixed_species.empty())
      return;

    auto mark_mol_all_bits = [&](int mid) {
      if (mid < 0)
        return;
      if (mid >= static_cast<int>(event_mol_change_mask.size())) {
        int const new_n = std::max(mid + 1, 2 * static_cast<int>(event_mol_change_mask.size()));
        event_mol_change_mask.resize(new_n, 0);
        event_mol_change_epoch.resize(new_n, 0);
      }
      event_mol_change_epoch[mid] = event_epoch;
      event_mol_change_mask[mid] = ~uint64_t{0};
    };

    for (const auto& fs : model.fixed_species) {
      std::vector<int> matching;
      for (int const mid : pool.molecules_of_type(fs.mol_type_idx)) {
        if (mid < 0 || mid >= pool.molecule_count())
          continue;
        const auto& mol = pool.molecule(mid);
        if (!mol.active)
          continue;
        if (pool.molecules_in_complex(mol.complex_id).size() != 1)
          continue;
        bool state_ok = true;
        for (int ci = 0; ci < static_cast<int>(fs.required_comp_state.size()); ++ci) {
          int const req = fs.required_comp_state[ci];
          if (req < 0)
            continue;
          if (ci >= static_cast<int>(mol.comp_ids.size())) {
            state_ok = false;
            break;
          }
          int const actual = pool.component(mol.comp_ids[ci]).state_index;
          if (actual != req) {
            state_ok = false;
            break;
          }
        }
        if (!state_ok)
          continue;
        matching.push_back(mid);
      }

      int const delta = fs.target_count - static_cast<int>(matching.size());
      if (delta > 0) {
        for (int k = 0; k < delta; ++k) {
          int const new_mid = pool.add_molecule(fs.mol_type_idx);
          auto& new_mol = pool.molecule(new_mid);
          for (int ci = 0; ci < static_cast<int>(fs.required_comp_state.size()); ++ci) {
            int const req = fs.required_comp_state[ci];
            if (req < 0)
              continue;
            if (ci < static_cast<int>(new_mol.comp_ids.size()))
              pool.set_state(new_mol.comp_ids[ci], req);
          }
          mark_mol_all_bits(new_mid);
          affected.insert(new_mid);
        }
      } else if (delta < 0) {
        // Delete uniform-random members of `matching` (was: tail-delete by
        // index).  The previous tail-delete was harmless because Fixed
        // species are scoped to singletons so all matches are interchangeable
        // (they participate in no other rule's reactant pattern), but the
        // intent of "remove excess copies" is "uniformly at random" — pin
        // that down explicitly so a future Fixed-species extension that
        // distinguishes among the matches doesn't silently pick the
        // last-added molecule.  Partial Fisher–Yates: shuffle the first
        // `n_delete` entries to the front, then delete those.
        int const n_delete = std::min(-delta, static_cast<int>(matching.size()));
        for (int k = 0; k < n_delete; ++k) {
          std::uniform_int_distribution<int> dist(k, static_cast<int>(matching.size()) - 1);
          int const j = dist(rng);
          if (j != k)
            std::swap(matching[k], matching[j]);
        }
        for (int k = 0; k < n_delete; ++k) {
          int const mid = matching[k];
          affected.erase(mid);
          pool.delete_molecule(mid);
        }
      }
    }
  }

  // --- SSA loop ---

  Result run_ssa(const TimeSpec& ts, const CancelCallback& should_continue = {}) {
    Result result;
    result.observable_names = model.observable_names_ordered;
    result.observable_data.reserve(result.observable_names.size());
    result.function_names = output_function_names;
    result.function_data.reserve(result.function_names.size());

    // Cooperative cancellation: poll `should_continue` every
    // kCancelCheckStride SSA events.  Stride is a power of two so the test
    // compiles to a single AND + branch — see BNGsim's own SSA wrapper for
    // the same idiom.  Hoist the validity check out of the loop so the
    // common no-callback case pays only one extra branch per stride.
    constexpr int64_t kCancelCheckStride = 1024;
    const bool has_cancel_cb = static_cast<bool>(should_continue);

    auto record_time_point = [&result](double t, const std::vector<double>& values) {
      assert(values.size() == result.observable_names.size());
      if (result.observable_data.empty())
        result.observable_data.resize(values.size());
      result.time.push_back(t);
      for (std::size_t i = 0; i < values.size(); ++i)
        result.observable_data[i].push_back(values[i]);
    };

    // Append the current global-function values as one more column-major
    // row.  Called from record_at immediately after record_time_point so
    // it settles functions against the same refreshed obs_values that the
    // observable row was built from; it deliberately does NOT touch
    // result.time (record_time_point owns that).
    auto record_function_point = [&result, this]() {
      if (output_function_indices.empty())
        return; // no global functions → nothing to record (common case)
      std::vector<double> const fvals = compute_function_values();
      if (result.function_data.empty())
        result.function_data.resize(fvals.size());
      for (std::size_t i = 0; i < fvals.size(); ++i)
        result.function_data[i].push_back(fvals[i]);
    };

    // Compute sample times.  An explicit `ts.sample_times` list (issue #16)
    // overrides the uniform grid entirely; the recorder below walks a generic
    // sorted vector either way, so only the construction of `sample_times`
    // changes here.
    std::vector<double> sample_times;
    if (!ts.sample_times.empty()) {
      // Honor the caller's exact output instants (e.g. an experimental
      // dataset's time points).  The recorder advances a single cursor and
      // never looks back, so the list must be sorted ascending — validate and
      // throw rather than silently dropping or mis-ordering samples.
      for (std::size_t i = 1; i < ts.sample_times.size(); ++i) {
        if (ts.sample_times[i] < ts.sample_times[i - 1])
          throw std::runtime_error("TimeSpec::sample_times must be sorted in "
                                   "ascending order");
      }
      sample_times = ts.sample_times;
    } else if (ts.n_points > 0) {
      double const dt = (ts.t_end - ts.t_start) / ts.n_points;
      for (int i = 0; i <= ts.n_points; ++i)
        sample_times.push_back(ts.t_start + (i * dt));
    } else {
      sample_times.push_back(ts.t_start);
      sample_times.push_back(ts.t_end);
    }

    int next_sample = 0;

    // Tolerance for "is this sample at the current logical time?" The
    // historical absolute 1e-15 was wrong-scale at extreme `t_end`:
    // a t_end of 1e6 has FP grid spacing ≈ 1.16e-10 between adjacent
    // doubles, so 1e-15 is below the resolution of `current_time` and
    // a sample exactly at t_end could miss its bucket.  Use a relative
    // tolerance (max(1e-15, 1e-12 * |t_end|)) so the check tracks the
    // representable spacing of the time grid at any scale.  Computed
    // once outside the SSA loop because t_end doesn't change mid-run.
    const double t_eps = std::max(1e-15, 1e-12 * std::fabs(ts.t_end));

    auto wall_t0 = std::chrono::steady_clock::now();

    // Record initial state if needed
    auto record_at = [&](double t) {
      auto rec_t0 = std::chrono::steady_clock::now();
      if constexpr (kRecordAtProfile) {
        rap_profile_.calls++;
        rap_profile_.inside_record_at = true;
        auto co_t0 = std::chrono::steady_clock::now();
        refresh_observables_for_sample();
        auto co_t1 = std::chrono::steady_clock::now();
        rap_profile_.inside_record_at = false;
        rap_profile_.compute_obs_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(co_t1 - co_t0).count();
        record_time_point(t, obs_values);
        auto rec_t1 = std::chrono::steady_clock::now();
        rap_profile_.record_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(rec_t1 - co_t1).count();
      } else {
        refresh_observables_for_sample();
        record_time_point(t, obs_values);
      }
      record_function_point();
      timing_record +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - rec_t0).count();
    };

    // Record samples up to current time
    while (next_sample < static_cast<int>(sample_times.size()) &&
           sample_times[next_sample] <= current_time + t_eps) {
      record_at(sample_times[next_sample]);
      ++next_sample;
    }

    // Main SSA loop
    while (current_time < ts.t_end) {
      // Cooperative cancellation: between-event safe point.  Throwing
      // before any state advances this iteration means the session is
      // left at the last completed event's time/pool state — safe to
      // discard via destroy_session() or to resume via another
      // run/simulate call.  The stride mask keeps overhead negligible
      // (~1024 events between checks); event_count starts at 0 so the
      // first iteration honors a callback that's already requesting
      // cancellation.
      if (has_cancel_cb && (event_count & (kCancelCheckStride - 1)) == 0 && !should_continue())
        throw Cancelled();

      // total_propensity is delta-updated by set_rule_propensity at every
      // rs.propensity write; periodically re-baseline to flush accumulated
      // FP drift (per-event delta sums + and - cancelations slowly bias the
      // running total relative to a fresh sum).
      if (++events_since_propensity_rebaseline >= kPropensityRebaselineInterval)
        recompute_total_propensity();

      if (total_propensity <= 0) {
        // Defensive re-baseline: a long sequence of +/- updates can drive
        // the running total to zero through FP cancellation while real
        // propensity remains positive.  Re-sum and re-check before
        // declaring the absorbing state.
        recompute_total_propensity();
        if (total_propensity <= 0) {
          // Absorbing state — record remaining samples at current values
          while (next_sample < static_cast<int>(sample_times.size())) {
            record_at(sample_times[next_sample]);
            ++next_sample;
          }
          break;
        }
      }

      // Save RNG state before drawing dt so that state save/restore
      // at t_end produces bit-exact continuation (the overshoot draw
      // is rolled back, and the restored engine re-draws the same dt).
      auto rng_before_dt = rng;

      // Sample time to next reaction
      double u1 = uniform();
      while (u1 == 0.0)
        u1 = uniform(); // avoid log(0)
      double const dt = -std::log(u1) / total_propensity;
      double const next_time = current_time + dt;

      // Record samples before this reaction
      while (next_sample < static_cast<int>(sample_times.size()) &&
             sample_times[next_sample] <= next_time) {
        record_at(sample_times[next_sample]);
        ++next_sample;
      }

      if (next_time > ts.t_end) {
        rng = rng_before_dt; // rollback unused dt draw
        current_time = ts.t_end;
        break;
      }

      current_time = next_time;
      ++eval_vars_gen; // time advanced → invalidate cached eval_vars_flat

      // Select reaction
      double const u2 = uniform() * total_propensity;
      double cum = 0;
      int selected = -1;
      for (int ri = 0; ri < static_cast<int>(rule_states.size()); ++ri) {
        cum += rule_states[ri].propensity;
        if (u2 < cum) {
          selected = ri;
          break;
        }
      }
      if (selected < 0) {
        // Fallthrough: the cumulative scan sum drifted below
        // total_propensity (delta-updated) by FP roundoff and the draw
        // landed in the gap.  The previous fallback "pick the last rule"
        // could land on a propensity-zero rule and waste an event (or
        // worse, fire a degenerate rule with zero propensity if its
        // selection logic happened to return a valid match).  Walk the
        // rule list backwards and pick the last rule with positive
        // propensity — that is the rule the small-FP draw would have
        // selected with infinite-precision arithmetic.
        for (int ri = static_cast<int>(rule_states.size()) - 1; ri >= 0; --ri) {
          if (rule_states[ri].propensity > 0) {
            selected = ri;
            break;
          }
        }
        // No active rule at all — the earlier total_propensity <= 0
        // guard above should have caught this, so reaching here means
        // the running total has positive bookkeeping but every per-rule
        // propensity is 0.  Force a re-baseline + abort the event loop
        // on the next iteration rather than dereferencing an invalid
        // selection.
        if (selected < 0) {
          recompute_total_propensity();
          continue;
        }
      }

      // Select reactants and fire
      auto t0 = std::chrono::steady_clock::now();
      auto match = select_reactants(selected);
      auto t1 = std::chrono::steady_clock::now();
      if (match.mol_ids.empty() && model.rules[selected].molecularity > 0) {
        timing_sample += std::chrono::duration<double>(t1 - t0).count();
        ++null_event_count;
        continue; // null event for non-synthesis rule
      }

      // Exclude/include reactant/product constraints
      if (!model.rules[selected].constraints.empty() &&
          !check_constraints(model.rules[selected], match)) {
        timing_sample += std::chrono::duration<double>(t1 - t0).count();
        ++null_event_count;
        continue;
      }

      // Product molecularity check: for a unimolecular rule with multiple
      // product patterns and a DeleteBond, verify that breaking the bond
      // actually separates the molecules into different connected components.
      // If they remain connected through another path (e.g., a ring bond),
      // the `+` between products cannot be satisfied, so reject as null.
      //
      // This enforces the RHS-`+` half of strict BNGL semantics.  It is
      // gated on `block_same_complex_binding` so that both halves of the
      // `+` check (LHS binding and RHS unbinding) toggle together, matching
      // NFsim's `-bscb` (which implies `-cb` and enables both checks).
      if (model.block_same_complex_binding) {
        auto& rule = model.rules[selected];
        if (rule.molecularity <= 1 && rule.n_product_patterns > 1) {
          bool reject = false;
          for (auto& op : rule.operations) {
            if (op.type != OpType::DeleteBond)
              continue;
            if (op.comp_flat_a < 0 || op.comp_flat_a >= static_cast<int>(match.comp_ids.size()))
              continue;
            int const comp_a = match.comp_ids[op.comp_flat_a];
            if (comp_a < 0)
              continue;
            int const partner = pool.component(comp_a).bond_partner;
            if (partner < 0)
              continue;
            int const mol_a = pool.mol_of_comp(comp_a);
            int const mol_b = pool.mol_of_comp(partner);
            if (mol_a == mol_b) {
              reject = true;
              break;
            }
            // P7: when the reactant complex has zero cycle bonds (tree),
            // removing any single bond necessarily disconnects its two
            // endpoints — the BFS below is guaranteed to fall out with
            // found_b == false.  Skip it.  With kProductMolInvariant we
            // also run the BFS and assert agreement.
            int const cx = pool.complex_of(mol_a);
            bool const tree = (pool.cycle_bond_count(cx) == 0);
            if (tree && !kProductMolInvariant)
              continue;

            // BFS from mol_a avoiding the comp_a-partner edge; if we reach
            // mol_b, the molecules stay connected after breaking this bond.
            std::unordered_set<int> visited;
            std::queue<int> q;
            visited.insert(mol_a);
            q.push(mol_a);
            bool found_b = false;
            while (!q.empty() && !found_b) {
              int const cur = q.front();
              q.pop();
              auto& mol = pool.molecule(cur);
              for (int const cid : mol.comp_ids) {
                if (cid == comp_a || cid == partner)
                  continue; // skip broken bond
                int const p = pool.component(cid).bond_partner;
                if (p < 0)
                  continue;
                int const nb = pool.mol_of_comp(p);
                if (nb == mol_b) {
                  found_b = true;
                  break;
                }
                if (!visited.count(nb)) {
                  visited.insert(nb);
                  q.push(nb);
                }
              }
            }
            if (kProductMolInvariant && tree && found_b) {
              std::fprintf(stderr,
                           "[ProductMol mismatch] rule=%d cycle_bond_count=%d "
                           "but BFS reached mol_b (mol_a=%d mol_b=%d comp_a=%d "
                           "partner=%d cx=%d)\n",
                           selected, pool.cycle_bond_count(cx), mol_a, mol_b, comp_a, partner, cx);
              std::abort();
            }
            if (found_b) {
              reject = true;
              break;
            }
          }
          if (reject) {
            timing_sample += std::chrono::duration<double>(t1 - t0).count();
            ++null_event_count;
            continue;
          }
        }
      }

      // ensureConnected check: for DeleteBond operations with
      // ensure_connected=true, the bond break may only fire if the
      // two molecules remain connected through another path.
      // This encodes the `.` (same-complex) product constraint from BNGL.
      //
      // Must consider the NET effect of ALL bond operations: bonds removed
      // by DeleteBond AND bonds added by AddBond.  A rule like
      //   P(f!1).F(f!1,next!2).F(prev!2,f) -> P(f!3).F(f,next!2).F(prev!2,f!3)
      // deletes P-F0 but adds P-F1, keeping the complex connected.
      {
        bool has_ensure_connected = false;
        auto& rule = model.rules[selected];
        for (auto& op : rule.operations)
          if (op.type == OpType::DeleteBond && op.ensure_connected) {
            has_ensure_connected = true;
            break;
          }

        if (has_ensure_connected) {
          // Collect deleted bond edges (as comp_id pairs)
          std::set<std::pair<int, int>> deleted_edges;
          for (auto& op : rule.operations) {
            if (op.type != OpType::DeleteBond)
              continue;
            if (op.comp_flat_a < 0 || op.comp_flat_a >= static_cast<int>(match.comp_ids.size()))
              continue;
            int const ca = match.comp_ids[op.comp_flat_a];
            if (ca < 0)
              continue;
            int const cb = pool.component(ca).bond_partner;
            if (cb < 0)
              continue;
            deleted_edges.insert({std::min(ca, cb), std::max(ca, cb)});
          }

          // Collect added bond edges (as mol_id pairs, for existing molecules)
          std::set<std::pair<int, int>> added_mol_edges;
          for (auto& op : rule.operations) {
            if (op.type != OpType::AddBond)
              continue;
            // Only consider bonds between existing reactant molecules
            if (op.product_mol_a >= 0 || op.product_mol_b >= 0)
              continue;
            if (op.comp_flat_a < 0 || op.comp_flat_b < 0)
              continue;
            if (op.comp_flat_a >= static_cast<int>(match.comp_ids.size()) ||
                op.comp_flat_b >= static_cast<int>(match.comp_ids.size()))
              continue;
            int const ca = match.comp_ids[op.comp_flat_a];
            int const cb = match.comp_ids[op.comp_flat_b];
            if (ca < 0 || cb < 0)
              continue;
            int const ma = pool.mol_of_comp(ca);
            int const mb = pool.mol_of_comp(cb);
            if (ma != mb)
              added_mol_edges.insert({std::min(ma, mb), std::max(ma, mb)});
          }

          // For each ensureConnected DeleteBond, check connectivity considering
          // all deletions and additions.
          bool reject = false;
          for (auto& op : rule.operations) {
            if (op.type != OpType::DeleteBond || !op.ensure_connected)
              continue;
            if (op.comp_flat_a < 0 || op.comp_flat_a >= static_cast<int>(match.comp_ids.size()))
              continue;
            int const comp_a = match.comp_ids[op.comp_flat_a];
            if (comp_a < 0)
              continue;
            int const partner = pool.component(comp_a).bond_partner;
            if (partner < 0)
              continue;
            int const mol_a = pool.mol_of_comp(comp_a);
            int const mol_b = pool.mol_of_comp(partner);
            if (mol_a == mol_b)
              continue; // self-bond: always connected

            // BFS from mol_a, excluding all deleted bond edges and
            // including all added bond edges.
            std::unordered_set<int> visited;
            std::queue<int> q;
            visited.insert(mol_a);
            q.push(mol_a);
            bool found_b = false;
            while (!q.empty() && !found_b) {
              int const cur = q.front();
              q.pop();
              auto& mol = pool.molecule(cur);
              // Traverse existing bonds (minus deleted ones)
              for (int const cid : mol.comp_ids) {
                int const p = pool.component(cid).bond_partner;
                if (p < 0)
                  continue;
                // Skip if this bond is being deleted
                int const lo = std::min(cid, p), hi = std::max(cid, p);
                if (deleted_edges.count({lo, hi}))
                  continue;
                int const nb = pool.mol_of_comp(p);
                if (nb == mol_b) {
                  found_b = true;
                  break;
                }
                if (!visited.count(nb)) {
                  visited.insert(nb);
                  q.push(nb);
                }
              }
              if (found_b)
                break;
              // Traverse added bond edges
              for (auto& ae : added_mol_edges) {
                int nb = -1;
                if (ae.first == cur)
                  nb = ae.second;
                else if (ae.second == cur)
                  nb = ae.first;
                else
                  continue;
                if (nb == mol_b) {
                  found_b = true;
                  break;
                }
                if (!visited.count(nb)) {
                  visited.insert(nb);
                  q.push(nb);
                }
              }
            }
            if (!found_b) {
              reject = true;
              break;
            }
          }
          if (reject) {
            timing_sample += std::chrono::duration<double>(t1 - t0).count();
            ++null_event_count;
            continue;
          }
        }
      }

      auto fire_result = fire_rule(selected, match);
      auto& affected = fire_result.affected;

      // Undo this event's effect on any Fixed-species populations
      // (BNG2 ODE semantics: d/dt = 0 for Fixed species).  Must run
      // BEFORE complex-expansion + incremental_update so created/
      // destroyed singletons are included in this event's affected set.
      if (!model.fixed_species.empty())
        replenish_fixed_species(affected);

      // Expand affected_mols to full complexes for rules with disjoint
      // multi-mol patterns (e.g. L(r).R(l) ring closure).  Only needed
      // when complex membership changed (bond add/remove), not on pure
      // state changes.
      if (any_needs_complex_expansion_ && fire_result.bond_changed) {
        bool cx_prof_sampled = false;
        std::chrono::steady_clock::time_point cx_prof_t0;
        if constexpr (kFireRuleProfile) {
          fire_profile_.cx_exp_fires++;
          cx_prof_sampled = (fire_profile_.cx_exp_fires % kFireRuleProfileSampleEvery) == 0;
          if (cx_prof_sampled) {
            fire_profile_.cx_exp_sampled++;
            cx_prof_t0 = std::chrono::steady_clock::now();
          }
        }
        std::unordered_set<int> cx_expanded;
        std::unordered_set<int> seen_cx;
        for (int const mid : affected) {
          if (mid < 0 || mid >= pool.molecule_count() || !pool.molecule(mid).active)
            continue;
          int const cx = pool.complex_of(mid);
          if (!seen_cx.insert(cx).second)
            continue;
          for (int const m : pool.molecules_in_complex(cx))
            if (pool.molecule(m).active)
              cx_expanded.insert(m);
        }
        affected = std::move(cx_expanded);
        if constexpr (kFireRuleProfile) {
          if (cx_prof_sampled) {
            auto cx_prof_t1 = std::chrono::steady_clock::now();
            fire_profile_.cx_exp_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(cx_prof_t1 - cx_prof_t0)
                    .count();
          }
        }
      }

      ++rule_fire_counts[selected];
      auto t2 = std::chrono::steady_clock::now();

      // Update rate-dependent observables BEFORE recomputing propensities,
      // so dynamic rate laws see the post-event observable values.
      // Tracked obs (Molecules per-mid delta + Species dirty-cx) are
      // maintained by incremental_update_observables; Molecules
      // obs_values is kept fresh; Species obs_values stays stale
      // between events but its contribs drive any later flush.
      // Fallback rate-dep obs (multi-mol or Species rate-dep) are
      // full-walked here every event.
      if (use_incremental_obs)
        incremental_update_observables(affected);
      if (!rate_dep_obs_indices.empty())
        compute_rate_dependent_observables();
      auto t3 = std::chrono::steady_clock::now();

      // Molecule limit check.  After the observable refresh, not before:
      // the event that breached the limit is not rolled back, so leaving
      // it out of obs_values left the trailing sample rows one event
      // behind the pool they describe — and only for the incrementally
      // tracked observables, since the untracked ones full-walk the live
      // pool at sample time.  Recomputing propensities we are about to
      // throw away is the one event of wasted work this costs.
      if (molecule_limit > 0 && pool.active_molecule_count() > molecule_limit) {
        timing_sample += std::chrono::duration<double>(t1 - t0).count();
        timing_fire += std::chrono::duration<double>(t2 - t1).count();
        timing_obs += std::chrono::duration<double>(t3 - t2).count();
        ++event_count;
        break;
      }

      // Incremental update (recomputes propensities using fresh observables)
      incremental_update(affected);
      auto t4 = std::chrono::steady_clock::now();

      timing_sample += std::chrono::duration<double>(t1 - t0).count();
      timing_fire += std::chrono::duration<double>(t2 - t1).count();
      timing_obs += std::chrono::duration<double>(t3 - t2).count();
      timing_update += std::chrono::duration<double>(t4 - t3).count();

      ++event_count;
    }

    // Record remaining samples
    while (next_sample < static_cast<int>(sample_times.size())) {
      record_at(sample_times[next_sample]);
      ++next_sample;
    }

    timing_wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_t0).count();

    // Diagnostic stderr blocks (per-phase timing + per-rule fire counts) are
    // off by default — embedders and CLI users want a quiet stderr.  Set
    // RM_PRINT_TIMING=1 in the environment to re-enable; the harness
    // scripts that parse `events=N` / `total=Xs` from rm_driver's stderr
    // do this themselves.  RM_PRINT_TABLES is the second, separate gate,
    // below: it costs wall time and this one is read by the wall-time
    // harnesses.
    // getenv is read once via a function-local static; no concurrent
    // access surface even under multi-threaded callers.
    static const bool kPrintTiming = []() {
      const char* v = std::getenv("RM_PRINT_TIMING"); // NOLINT(concurrency-mt-unsafe)
      return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();

    if (kPrintTiming) {
      // `total` sums the five per-phase buckets; `wall` is the true wall
      // clock for the SSA loop; `unaccounted` fills the gap (outer-loop
      // overhead + anything not covered by a phase timer).
      double const total_time =
          timing_sample + timing_fire + timing_obs + timing_update + timing_record;
      if (total_time > 0 || timing_wall > 0) {
        double const denom = total_time > 0 ? total_time : timing_wall;
        fprintf(stderr, "[RM timing] events=%lld  null=%lld  total=%.3fs  wall=%.3fs\n",
                static_cast<long long>(event_count), static_cast<long long>(null_event_count),
                total_time, timing_wall);
        fprintf(stderr, "  select_reactants: %.3fs (%.1f%%)\n", timing_sample,
                100.0 * timing_sample / denom);
        fprintf(stderr, "  fire_rule:        %.3fs (%.1f%%)\n", timing_fire,
                100.0 * timing_fire / denom);
        fprintf(stderr, "  observables:      %.3fs (%.1f%%)\n", timing_obs,
                100.0 * timing_obs / denom);
        fprintf(stderr, "  incr_update:      %.3fs (%.1f%%)\n", timing_update,
                100.0 * timing_update / denom);
        fprintf(stderr, "  record_at:        %.3fs (%.1f%%)\n", timing_record,
                100.0 * timing_record / denom);
        double const unaccounted = timing_wall - total_time;
        if (unaccounted > 0.0005 * timing_wall) { // only show if > 0.05%
          fprintf(stderr, "  unaccounted:      %.3fs (%.1f%%)\n", unaccounted,
                  100.0 * unaccounted / timing_wall);
        }
      }

      // Per-rule fire counts and final propensities.
      int const n_rules = static_cast<int>(model.rules.size());
      fprintf(stderr, "[RM per-rule] fire_counts and final propensities:\n");
      for (int ri = 0; ri < n_rules; ++ri) {
        auto& rule = model.rules[ri];
        auto& rs = rule_states[ri];
        fprintf(stderr, "  %s (%s): fires=%llu  propensity=%.6g  a_total=%.6g\n", rule.id.c_str(),
                rule.name.c_str(), static_cast<unsigned long long>(rule_fire_counts[ri]),
                rs.propensity, rs.a_total);
      }
    }

    // Per-rule side-table residency (issue #71).  Its own gate, not
    // RM_PRINT_TIMING's: `rule_table_reach` walks each rule's seed-type
    // populations, which is a sizeable fraction of a second on a model with
    // a couple of hundred rules over a large arena, and RM_PRINT_TIMING is
    // what the timing harnesses set on every replicate.  A memory question
    // should not put its cost on a wall-clock measurement.
    static const bool kPrintTables = []() {
      const char* v = std::getenv("RM_PRINT_TABLES"); // NOLINT(concurrency-mt-unsafe)
      return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();

    if (kPrintTables) {
      // See RuleTableBytes for what is counted and why it is `size()`.  One
      // line per rule that holds a table -- a rule with no reactant seed
      // molecule holds none at all (#68) and would otherwise be a line of
      // zeroes -- and a model total, which is the number the sizing question
      // turns on.
      //
      // `reach_bytes` is the same total with every table cut to the highest
      // live id its rule can index, which is what sizing by the seed types'
      // high-water mark would cost.  The gap between it and `bytes` is what
      // that shortcut would buy on this model; see rule_table_reach for why
      // it is not obviously anything.
      //
      // Bytes are exact so a sweep can do arithmetic on them; the MB
      // rendering on the summary is for reading.
      int const n_rules = static_cast<int>(model.rules.size());
      struct Row {
        int rule_index;
        int reach;
        RuleTableBytes held;
      };
      std::vector<Row> rows;
      RuleTableBytes model_tables;
      RuleTableBytes model_reach;
      for (int ri = 0; ri < n_rules; ++ri) {
        RuleTableBytes const tb = rule_table_bytes(rule_states[ri]);
        if (tb.total() == 0)
          continue;
        int const reach = rule_table_reach(pool, model.rules[ri], rule_states[ri]);
        RuleTableBytes const rb =
            rule_table_bytes(rule_states[ri], static_cast<std::size_t>(reach));
        model_tables.mol += tb.mol;
        model_tables.sampler += tb.sampler;
        model_reach.mol += rb.mol;
        model_reach.sampler += rb.sampler;
        rows.push_back(Row{ri, reach, tb});
      }
      fprintf(stderr,
              "[RM tables] arena=%d rules=%d tabled=%d bytes=%llu (%.1f MB) mol=%llu "
              "sampler=%llu reach_bytes=%llu\n",
              pool.molecule_count(), n_rules, static_cast<int>(rows.size()),
              static_cast<unsigned long long>(model_tables.total()),
              static_cast<double>(model_tables.total()) / (1024.0 * 1024.0),
              static_cast<unsigned long long>(model_tables.mol),
              static_cast<unsigned long long>(model_tables.sampler),
              static_cast<unsigned long long>(model_reach.total()));
      for (const auto& row : rows) {
        fprintf(stderr, "  %s (%s): rows=%llu reach=%d mol=%llu sampler=%llu\n",
                model.rules[row.rule_index].id.c_str(), model.rules[row.rule_index].name.c_str(),
                static_cast<unsigned long long>(row.held.rows), row.reach,
                static_cast<unsigned long long>(row.held.mol),
                static_cast<unsigned long long>(row.held.sampler));
      }
    }

    // Dev-profile end-of-run reports (gated by build-time RM_DEV_PROFILES
    // master macro; dead-stripped in default builds).
    if constexpr (kFireRuleProfile)
      report_fire_rule(fire_profile_, timing_fire);

    if constexpr (kRemoveBondProfile)
      report_remove_bond(pool.remove_profile());

    if constexpr (kIncrUpdateProfile)
      report_incr_update(incr_profile_, timing_update);

    if constexpr (kRecordAtProfile)
      report_record_at(rap_profile_, timing_record, use_incremental_obs);

    if constexpr (kObsIncrProfile)
      report_obs_incr(obs_incr_profile_, timing_obs, incr_tracked_obs_indices);

    if constexpr (kCountMultiProfile)
      report_count_multi(cm_profile_);

    if constexpr (kCmmFcProfile)
      report_cmm_fc(cmm_fc_profile_, cm_profile_);

    if constexpr (kSelectReactantsProfile)
      report_select_reactants(sr_profile_, timing_sample);

    if constexpr (kExprEvalProfile)
      report_expr_eval(expr_eval_profile_, timing_wall);

    result.event_count = event_count;
    return result;
  }
};

// ===========================================================================
// Engine public interface
// ===========================================================================

Engine::Engine(const Model& model, uint64_t seed, int molecule_limit)
    : impl_(std::make_unique<Impl>(model, seed, molecule_limit)) {}

Engine::~Engine() = default;

// ===========================================================================
// Species enumeration (issue #9 §2 — `.species` output)
// ===========================================================================

namespace {

// extract_complex — the sole bridge from engine pool internals to the
// canonical labeler (plan §4).  Builds a canonical::ComplexGraph for one
// connected complex, given the molecule ids it contains.  Component
// states are rendered to their BNGL state-name strings; an unbonded or
// stateless component carries "" (no `~state`).
canonical::ComplexGraph extract_complex(const AgentPool& pool, const Model& model,
                                        const std::vector<int>& mol_ids) {
  canonical::ComplexGraph g;
  // global component id -> (graph molecule index, local component index)
  std::unordered_map<int, std::pair<int, int>> comp_loc;
  for (int const mid : mol_ids) {
    const auto& mol = pool.molecule(mid);
    const auto& mtype = model.molecule_types[mol.type_index];
    int const gmol = g.molecule_count();
    std::vector<std::pair<std::string, std::string>> comps;
    comps.reserve(mol.comp_ids.size());
    for (int li = 0; li < static_cast<int>(mol.comp_ids.size()); ++li) {
      int const cid = mol.comp_ids[li];
      const auto& ctype = mtype.components[li];
      int const si = pool.component(cid).state_index;
      std::string state;
      if (si >= 0 && si < static_cast<int>(ctype.allowed_states.size()))
        state = ctype.allowed_states[si];
      comps.emplace_back(ctype.name, state);
      comp_loc.emplace(cid, std::pair<int, int>{gmol, li});
    }
    g.add_molecule(mtype.name, comps);
  }
  // One graph edge per bond.  bond_partner is a global component id;
  // emit each bond once, from its lower-id endpoint.  Both endpoints are
  // in this complex, so comp_loc always resolves.
  for (int const mid : mol_ids) {
    for (int const cid : pool.molecule(mid).comp_ids) {
      int const partner = pool.component(cid).bond_partner;
      if (partner > cid) {
        const auto& a = comp_loc.at(cid);
        const auto& b = comp_loc.at(partner);
        g.add_bond(a.first, a.second, b.first, b.second);
      }
    }
  }
  return g;
}

// Build a ComplexGraph straight from a parsed species Pattern (issue #9
// §1) — the parse-side analogue of extract_complex.  `pat` is an exact,
// fully-specified species from parse_species_pattern, so each pattern
// molecule carries every component (once) with a concrete state, and
// pat.bonds gives the edges by flat component index.  The graph is
// isomorphic to the one extract_complex builds for the same physical
// species, so canonical_label yields the same string — that is what
// makes pattern-keyed species_count a byte-equal lookup.
//
// Components are placed in molecule-type DECLARATION order (by
// comp_type_index), exactly as extract_complex feeds the pool's
// components.  canonical_label preserves the input's component-name
// slot layout (canonical.cpp §3.3 "component-order contract"), so a
// pattern listed in a non-canonical component order — e.g. "X(p~0,y)"
// for the canonical "X(y,p~0)" — must be reordered here, or it yields a
// different label and silently matches nothing while the add path
// (instantiate_pattern_complex, which already keys by comp_type_index)
// does match.  That divergence was issue #13.
canonical::ComplexGraph pattern_to_complex_graph(const Pattern& pat) {
  canonical::ComplexGraph g;
  for (const auto& pm : pat.molecules) {
    // Slot i carries the molecule type's i-th declared component.  The
    // pattern is fully specified (every type component listed exactly
    // once), so comp_type_index is a bijection onto [0, n) and every
    // slot is filled.
    std::vector<std::pair<std::string, std::string>> comps(pm.components.size());
    for (const auto& pc : pm.components)
      comps[pc.comp_type_index] = {pc.name, pc.required_state}; // "" if stateless
    g.add_molecule(pm.type_name, comps);
  }
  // pat.bonds endpoints are flat indices in the pattern's *listed* order;
  // map each back to (molecule index, declaration-order local index) so
  // the bond lands on the reordered component above.
  const auto flat_to_loc = [&pat](int flat) -> std::pair<int, int> {
    int acc = 0;
    for (int mi = 0; mi < static_cast<int>(pat.molecules.size()); ++mi) {
      const auto& mol = pat.molecules[mi];
      const int n = static_cast<int>(mol.components.size());
      if (flat < acc + n)
        return {mi, mol.components[flat - acc].comp_type_index};
      acc += n;
    }
    throw std::runtime_error("pattern_to_complex_graph: flat-index overflow");
  };
  for (const auto& b : pat.bonds) {
    const auto a = flat_to_loc(b.comp_flat_a);
    const auto c = flat_to_loc(b.comp_flat_b);
    g.add_bond(a.first, a.second, c.first, c.second);
  }
  return g;
}

} // namespace

void Engine::initialize() {
  impl_->init_species();
  // Tracker first.  Building its tables costs exactly the per-molecule
  // embedding counts an observable full-walk costs, so seeding
  // obs_values out of them (seed_tracked_obs_values) lets the walk below
  // skip every tracked observable rather than repeat the work — half of
  // session build on a million-molecule pool (issue #65).  The tracker
  // reads only the model and the pool, so it does not care that the
  // other two steps have not run yet.
  impl_->init_incremental_observables();
  // Still before init_rule_states, which reads obs_values for Function
  // rate laws — but now only for the observables the tracker does not
  // keep.
  impl_->compute_observables(/*skip_tracked=*/impl_->use_incremental_obs);
  impl_->init_rule_states();
  impl_->initialized = true;
}

Result Engine::run(const TimeSpec& ts, const CancelCallback& should_continue) {
  if (!impl_->initialized)
    initialize();
  return impl_->run_ssa(ts, should_continue);
}

double Engine::current_time() const { return impl_->current_time; }

std::vector<double> Engine::get_observable_values() {
  // Recompute against current pool state — observable values are only
  // refreshed at SSA sample points during run_ssa, so a query in between
  // events would otherwise return stale data.
  impl_->compute_observables();
  return impl_->obs_values;
}

std::vector<double> Engine::get_function_values() {
  // Global functions are derived from observables, so refresh those
  // first (same staleness reasoning as get_observable_values), then
  // settle the functions against the fresh observable values.
  impl_->compute_observables();
  return impl_->compute_function_values();
}

std::vector<std::string> Engine::function_names() const { return impl_->output_function_names; }

double Engine::evaluate_expression(const std::string& expr,
                                   const std::unordered_map<std::string, double>& extra) {
  return impl_->evaluate_expression(expr, extra);
}

int Engine::get_molecule_count(const std::string& type_name) const {
  int const ti = impl_->model.mol_type_index(type_name);
  if (ti < 0)
    return 0;
  int count = 0;
  for (int const mid : impl_->pool.molecules_of_type(ti))
    if (impl_->pool.molecule(mid).active)
      ++count;
  return count;
}

void Engine::add_molecules(const std::string& type_name, int count) {
  int const ti = impl_->model.mol_type_index(type_name);
  if (ti < 0)
    throw std::runtime_error("Unknown molecule type '" + type_name + "'");
  for (int i = 0; i < count; ++i)
    impl_->pool.add_molecule(ti);

  // Refresh observables first so dynamic-rate rules see the post-add state
  // when their propensities are recomputed below.
  impl_->compute_observables();

  // Targeted rescan: only rules whose propensities can actually change.
  //   1. type_to_rules[ti]:        rules whose seed reactant pattern walks
  //                                a molecule of the just-added type.
  //   2. dynamic_rate_rules:       rules with Function-type rates whose
  //                                eval_vars (parameters / observables)
  //                                may have shifted.
  //   3. dynamic_synthesis_rules:  zero-reactant rules with dynamic rates;
  //                                same rationale.
  // Other rules' a_total is unaffected by adding free, unbound molecules
  // of type ti — their reactant patterns either don't reference ti, or
  // reference it only as a non-seed bonded position (which a free ti
  // molecule cannot satisfy without first being bonded by some other rule).
  const auto& impl = *impl_;
  std::vector<char> needed(impl.model.rules.size(), 0);
  if (ti >= 0 && ti < static_cast<int>(impl.type_to_rules.size())) {
    for (int const ri : impl.type_to_rules[ti])
      needed[ri] = 1;
  }
  for (int const ri : impl.dynamic_rate_rules)
    needed[ri] = 1;
  for (int const ri : impl.dynamic_synthesis_rules)
    needed[ri] = 1;
  for (int ri = 0; ri < static_cast<int>(impl.model.rules.size()); ++ri) {
    if (needed[ri])
      impl_->rescan_all_molecules_for_rule(ri);
  }

  // Re-baseline total_propensity after the partial rescan so the running
  // sum reflects the new per-rule propensities exactly (set_rule_propensity
  // credited deltas already, but a fresh sum is cheap and avoids drift on
  // long sequences of add_molecules calls).
  impl_->recompute_total_propensity();

  // Re-seed the incremental observable tracker.  Without this, the
  // kSpeciesIncrObs delta-update path keeps applying offsets relative
  // to its pre-add baseline and Species observable counts drift on
  // every subsequent SSA event (the post-add full compute_observables
  // above gives the right value at this instant, but the tracker's
  // dirty-cx / per-mol contribution caches don't know about the
  // new molecules).
  impl_->init_incremental_observables();
}

void Engine::save_state(const std::string& path) const { impl_->save_state_to(path); }

void Engine::load_state(const std::string& path) { impl_->load_state_from(path); }

std::vector<SpeciesRow> Engine::enumerate_species() const {
  // Canonicalize every complex; graph-isomorphic complexes share a
  // canonical label and so collapse to one map entry with a summed
  // count.  std::map keeps the rows sorted by the species string.
  //
  // The batch sweep stays a from-scratch recompute (plan §5): it does
  // NOT read the cached-incremental cache.  In Debug / ASan builds,
  // though, it doubles as the decision-#6 self-check — each complex's
  // fresh label is compared against cached_label_of, which returns the
  // cached string when the complex is not dirty.  A mismatch there means
  // a structural mutator failed to mark the complex dirty: the cache is
  // stale yet read as valid.  That is the load-bearing correctness proof
  // for the cached-incremental layer until partial scaling consumes it.
  std::map<std::string, long> counts;
  for (const auto& cx : impl_->pool.complexes()) {
    const std::vector<int>& members = cx.second;
    if (members.empty())
      continue;
    const auto graph = extract_complex(impl_->pool, impl_->model, members);
    std::string label = canonical::canonical_label(graph);
    if constexpr (kCanonicalCacheSelfCheck) {
      // cached_label_of returns the cached label for a non-dirty complex
      // and a fresh recompute for a dirty/absent one; either way it must
      // equal the independent from-scratch `label` computed just above.
      const std::string& cached = impl_->cached_label_of(cx.first);
      if (cached != label) {
        std::fprintf(stderr, "[canonical cache invariant violated] cx=%d cached='%s' fresh='%s'\n",
                     cx.first, cached.c_str(), label.c_str());
        std::abort();
      }
    }
    ++counts[std::move(label)];
  }
  if constexpr (kCanonicalCacheSelfCheck) {
    // Evict cache entries for complexes that have since died (merged
    // away or last-member deleted).  Complex ids are monotonic and never
    // reused, so a stale entry can never alias a live complex — this is
    // memory hygiene, not correctness.  The cache is only ever populated
    // here, so pruning here keeps it bounded by the live complex count.
    const auto& live = impl_->pool.complexes();
    for (auto it = impl_->cx_label_cache_.begin(); it != impl_->cx_label_cache_.end();) {
      if (live.find(it->first) == live.end())
        it = impl_->cx_label_cache_.erase(it);
      else
        ++it;
    }
  }
  std::vector<SpeciesRow> rows;
  rows.reserve(counts.size());
  for (const auto& [label, n] : counts)
    rows.push_back(SpeciesRow{label, n});
  return rows;
}

void Engine::write_species_file(const std::string& path) const {
  std::vector<SpeciesRow> const rows = enumerate_species();
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("Cannot open species file for writing: " + path);
  long total = 0;
  for (const auto& r : rows)
    total += r.count;
  // BNG-format `.species` file: `#` comment lines (stripped by BNG2.pl's
  // readNFspecies), then one `<canonical pattern>  <integer count>` line
  // per species.  Rows are already sorted by enumerate_species().
  out << "# RuleMonkey generated species list\n";
  out << "# " << rows.size() << " species, " << total << " complexes\n";
  for (const auto& r : rows)
    out << r.species << "  " << r.count << "\n";
}

long Engine::species_count(const std::string& canonical) const {
  // Batch-mode lookup (plan §5; step-6 decision (b) — the cached-
  // incremental cache stays unconsumed until partial scaling).  Delegate
  // to the from-scratch enumerate_species() sweep and return the
  // matching row's count.  enumerate_species() rows are sorted by the
  // species string, so a binary search pinpoints the row; a string RM
  // never emits falls between rows and yields 0.
  const std::vector<SpeciesRow> rows = enumerate_species();
  auto it =
      std::lower_bound(rows.begin(), rows.end(), canonical,
                       [](const SpeciesRow& r, const std::string& key) { return r.species < key; });
  if (it != rows.end() && it->species == canonical)
    return it->count;
  return 0;
}

long Engine::total_complex_count() const {
  // Live complexes only — enumerate_species() likewise skips empty
  // member lists, so this stays equal to the sum of its row counts.
  long n = 0;
  for (const auto& cx : impl_->pool.complexes())
    if (!cx.second.empty())
      ++n;
  return n;
}

// --- Pattern-keyed species methods (issue #9 §1) ---------------------------

long Engine::get_species_count(const Pattern& pat) const {
  // Canonicalize the parsed species and reuse the §2 batch lookup: the
  // pattern's canonical label is byte-equal to the label the pool's
  // copies of the same species carry, so species_count() matches them.
  const std::string label = canonical::canonical_label(pattern_to_complex_graph(pat));
  return species_count(label);
}

void Engine::add_species(const Pattern& pat, int count) {
  if (count <= 0)
    throw std::runtime_error("add_species: count must be positive");
  for (int i = 0; i < count; ++i)
    impl_->instantiate_pattern_complex(pat);
  impl_->resync_all_after_structural_change();
}

void Engine::remove_species(const Pattern& pat, int count) {
  if (count <= 0)
    throw std::runtime_error("remove_species: count must be positive");

  // Find every live complex whose canonical label matches the species.
  const std::string target = canonical::canonical_label(pattern_to_complex_graph(pat));
  std::vector<int> matching; // complex ids
  for (const auto& cx : impl_->pool.complexes()) {
    if (cx.second.empty())
      continue;
    const auto graph = extract_complex(impl_->pool, impl_->model, cx.second);
    if (canonical::canonical_label(graph) == target)
      matching.push_back(cx.first);
  }
  if (static_cast<int>(matching.size()) < count)
    throw std::runtime_error("remove_species: only " + std::to_string(matching.size()) +
                             " live cop-" + (matching.size() == 1 ? "y" : "ies") + " of '" +
                             target + "' (requested " + std::to_string(count) + ")");

  // Delete the first `count` matches.  molecules_in_complex returns a
  // reference into the live complex map, which delete_molecule mutates,
  // so snapshot the member list before deleting from it.
  for (int i = 0; i < count; ++i) {
    const std::vector<int> members = impl_->pool.molecules_in_complex(matching[i]);
    for (int const mid : members)
      impl_->pool.delete_molecule(mid);
  }
  impl_->resync_all_after_structural_change();
}

void Engine::set_species_count(const Pattern& pat, int count) {
  if (count < 0)
    throw std::runtime_error("set_species_count: count must be non-negative");
  const long current = get_species_count(pat);
  const long delta = static_cast<long>(count) - current;
  if (delta > 0)
    add_species(pat, static_cast<int>(delta));
  else if (delta < 0)
    remove_species(pat, static_cast<int>(-delta));
}

} // namespace rulemonkey
