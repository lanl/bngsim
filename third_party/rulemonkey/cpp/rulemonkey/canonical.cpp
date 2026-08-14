// Canonical complex labeling — implementation of the pure core.
//
// See canonical.hpp and dev/canonical_labeling_plan_2026_05_16.md.
//
// Algorithm (plan §3):
//
//   Encoding (§3.1) — a vertex-colored port graph:
//     * one vertex per molecule; its color folds in the molecule type
//       name and the sorted multiset of UNBONDED (name, state) pairs;
//     * one vertex per BONDED component; its color is (name, state);
//     * edges: molecule <-> each of its bonded-component vertices, and
//       one edge per bond between the two bonded-component vertices.
//     Folding unbonded components into the molecule color (rather than
//     giving every component a vertex, as NFsim does) keeps the graph
//     small: unbonded components carry no edges, so they cannot affect
//     graph structure.
//
//   Canonicalization (§3.2):
//     1. 1-WL color refinement — recolor each vertex by (own color,
//        sorted neighbor colors) until the partition stabilizes.
//     2. Fast path — when refinement gives every molecule vertex its
//        own color AND no molecule has two interchangeable bonded
//        components left tied, the refined colors fix a unique,
//        isomorphism-invariant ordering; render directly, no search.
//        This is the overwhelmingly common case (asymmetric complexes).
//     3. Individualization-refinement (plan §3.2 step 3) for residual
//        symmetric classes — rings, homo-oligomers.  Pick a target
//        cell, individualize each vertex in it in turn, re-refine,
//        recurse to a leaf (a coloring the fast-path test accepts);
//        the canonical label is the lexicographically minimal rendered
//        BNGL string over all leaves.  This makes the label a *true*
//        canonical form on every input — 1-WL alone is incomplete on
//        regular/symmetric graphs.
//
//   Rendering (§3.3): walk molecules in canonical order; within each
//   molecule keep the molecule type's component-name layout but order
//   interchangeable same-name components by their refined color; emit
//   BNGL `Type(comp~state!bond,...)` with bond labels assigned in
//   first-encounter order along the walk.  The string is both the dedup
//   key and the `.species` line, and the walk order is returned with it
//   as CanonForm::mol_order — see the header for why a caller may need
//   to name one molecule of a complex isomorphism-invariantly.
//
//   Component-order contract: the renderer preserves the *name layout*
//   of the input ComplexGraph (slot i keeps its declared name) and only
//   reorders components that share a name.  `extract_complex` must
//   therefore emit each molecule's components in molecule-type
//   declaration order — which it does, since the engine stores
//   MoleculeInstance::comp_ids that way.
//
//   Performance (plan §5, step 7): the per-search-node hot path —
//   refine / is_leaf / render, each run once per individualization
//   leaf — is allocation-light.  Component names and states are
//   interned to dense ints once per complex (Tables); refinement uses
//   a CSR adjacency and reuses its signature scratch across calls
//   (Refiner); is_leaf / render group components with reusable buffers
//   instead of rebuilding std::map<std::string,…> per call.  The
//   algorithm and the emitted label are unchanged — this is purely the
//   §5 "speed by design" cleanup, profiled and applied in step 7.
//
//   Reuse ACROSS calls (GH #53) is the other half of that story, and it
//   needs a caller that can hold state: `canonical_order_fast` takes a
//   RankedComplex plus a Workspace, and every buffer the refinement
//   touches lives in that workspace rather than on the stack of a pure
//   function.  The integer ranks stand in for the color strings, so the
//   refinement it runs is the same refinement `canonicalize` would have
//   run on the same complex; `ranked_initial_colors` is where the two
//   meet and `rank_*` is the correspondence that keeps them equal.
//
//   With the allocations gone, step 1 was what the election cost, and a
//   round that recolors every vertex is `O(rounds x V log V)` on a graph
//   whose rounds grow with its diameter.  Refinement is a worklist over
//   the cells a round can still split now (GH #56) — same rounds, same
//   partitions, same ranking, so every label and every elected
//   representative is unchanged; see Refiner for why the cells it skips
//   are provably whole.

#include "canonical.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace rulemonkey::canonical {

// ===========================================================================
// ComplexGraph builder
// ===========================================================================

int ComplexGraph::add_molecule(const std::string& type_name,
                               const std::vector<std::pair<std::string, std::string>>& comps) {
  Molecule m;
  m.type_name = type_name;
  m.first_comp = static_cast<int>(components.size());
  m.n_comp = static_cast<int>(comps.size());
  for (const auto& [name, state] : comps)
    components.push_back(Component{name, state, -1});
  molecules.push_back(std::move(m));
  return static_cast<int>(molecules.size()) - 1;
}

void ComplexGraph::add_bond(int mol_a, int local_a, int mol_b, int local_b) {
  int const ga = global_comp(mol_a, local_a);
  int const gb = global_comp(mol_b, local_b);
  components[ga].partner = gb;
  components[gb].partner = ga;
}

// ===========================================================================
// Canonicalization
// ===========================================================================

namespace {

// Leaf-count safety valve for individualization-refinement (plan §3.2
// step 3, and the step-3 brief's "recursion/leaf guard").  BNGL
// complexes are small and richly labeled, so the search tree is tiny
// and shallow in practice; this cap exists only so a pathological
// hand-built input cannot blow up.  It is NOT a tuning knob.  If the
// cap is ever hit the best leaf found so far is returned (still
// deterministic per input).
constexpr long kLeafBudget = 200000;

// A bonded component contributes a vertex whose initial color is this
// string.  Unbonded components instead fold into their molecule's color
// (see molecule_color).
std::string component_color(const ComplexGraph::Component& c) {
  return "C:" + c.name + "~" + c.state;
}

// A molecule's initial color: the type name plus the sorted multiset of
// its UNBONDED components' `(name, state)`.  Bonded components are
// excluded — they become their own vertices and carry the edges.  The
// `M:` / `C:` prefixes keep molecule and component colors in disjoint
// ranges of the interning table.
std::string molecule_color(const ComplexGraph& g, const ComplexGraph::Molecule& m) {
  std::vector<std::string> free_comps;
  for (int i = 0; i < m.n_comp; ++i) {
    const auto& c = g.components[m.first_comp + i];
    if (c.partner < 0)
      free_comps.push_back(c.name + "~" + c.state);
  }
  std::sort(free_comps.begin(), free_comps.end());
  std::string s = "M:" + m.type_name + "(";
  for (size_t i = 0; i < free_comps.size(); ++i) {
    if (i)
      s += ',';
    s += free_comps[i];
  }
  s += ')';
  return s;
}

// Interned per-complex name / state tables (plan §5 "colors as interned
// integers, not strings").  The hot path — is_leaf and render, run once
// per individualization leaf — groups and orders components by name and
// state.  Interning both to dense ints once per complex lets that path
// use int comparisons and allocation-light scratch instead of rebuilding
// std::map<std::string,…> on every call.
//
// State ids are assigned in sorted string order, so `state_id` ordering
// equals state-string ordering — comp_key depends on this (it orders a
// molecule's free same-name components by state).
struct Tables {
  std::vector<int> comp_name_id; // component global index -> dense name id
  std::vector<int> state_id;     // component global index -> dense state id
};

Tables build_tables(const ComplexGraph& g) {
  int const n = g.component_count();
  Tables t;
  t.comp_name_id.assign(n, 0);
  t.state_id.assign(n, 0);
  std::map<std::string, int> names;  // sorted -> ids follow string order
  std::map<std::string, int> states; // sorted -> ids follow string order
  for (int gc = 0; gc < n; ++gc) {
    names.emplace(g.components[gc].name, 0);
    states.emplace(g.components[gc].state, 0);
  }
  {
    int next = 0;
    for (auto& [str, id] : names)
      id = next++;
  }
  {
    int next = 0;
    for (auto& [str, id] : states)
      id = next++;
  }
  for (int gc = 0; gc < n; ++gc) {
    t.comp_name_id[gc] = names[g.components[gc].name];
    t.state_id[gc] = states[g.components[gc].state];
  }
  return t;
}

// Within-molecule canonical ordering key for one component.  Components
// that share a name are interchangeable BNGL sites; the renderer orders
// each such group by this key.  Free components (which have no graph
// vertex) sort ahead of bonded ones and break ties by state id; bonded
// components break ties by their refined color.
using CompKey = std::tuple<int, int, int>; // (bonded?, bonded_color, state_id)
CompKey comp_key(int bonded_color, int state_id) {
  if (bonded_color < 0)
    return {0, 0, state_id};
  return {1, bonded_color, 0};
}

// 1-WL color refinement, run as a worklist over the cells that can
// still split (GH #56).
//
// The rounds are the textbook ones: a round recolors each vertex by
// (own color, sorted neighbor colors) and ranks the signatures, so the
// output colors are a canonical 0..k-1 ranking derived purely from
// structure.  The own-color term means a round only ever SPLITS cells —
// it has converged once nothing splits.  Any integer colors are
// accepted on input (individualization passes a vertex an out-of-range
// marker color; the first round re-ranks everything), so this doubles
// as the post-individualization re-refine step.
//
// What the worklist changes is only WHICH cells a round looks at.
// Recoloring every vertex every round costs O(rounds x V log V), and a
// chain needs on the order of O(n) rounds for the colors to propagate
// in from its ends — 259 ms of a 20-subunit tagged catalyst's 264 ms
// election overhead before this (GH #56).  Two facts make most of that
// work provably empty:
//
//   * A round never merges two cells and never reorders them.  The own
//     color leads the signature, so a round splits each cell
//     INTERNALLY and orders the pieces among themselves; cells keep
//     their relative order, and so therefore do the colors.
//   * So a cell whose members all carry the same sorted neighbor-color
//     slice comes out whole, and it can only STOP carrying the same
//     slice when one of its members gains a neighbor in a different
//     cell — that is, when a cell holding one of their neighbors splits.
//
// A round therefore examines the cells marked by the previous round's
// splits (every cell, on the first round) and nothing else: an unmarked
// cell cannot split, so not examining it changes neither the partition
// nor the ranking.  The output is the same coloring the every-vertex
// version produced, which is what lets `.species` labels and the
// election's answer stay byte-for-byte what they were.
//
// A cell's color WHILE REFINING is its start index in `cell`.  That is
// order-preserving — a piece starts inside its parent's old span — and
// it leaves an untouched cell's color literally unchanged, so a round
// renumbers nothing it did not split.  One pass at the end turns the
// start indices into the dense 0..k-1 ranking callers see.
//
// The Refiner wrapper also exists for speed (plan §5, step 7): the
// adjacency is held CSR (flat arrays) and every buffer a round touches
// is reused across refine() calls.  Individualization-refinement calls
// refine() once per search node; on a large symmetric complex that was
// the dominant allocation cost, and reuse removes it.  build() extends
// that reuse across whole canonicalizations, for a caller that keeps
// its Refiner in a Workspace (GH #53).
//
// Ranking by signature also means refinement never REORDERS colors that
// already differ.  That is what lets an order-only caller stop as soon
// as the molecule colors are distinct — see mol_colors_distinct.
struct Refiner {
  std::vector<int> adj_off;  // CSR offsets, size n_vert + 1
  std::vector<int> adj_flat; // CSR neighbor lists, size = total degree
  std::vector<int> nbr;      // per-vertex sorted neighbor colors, CSR-laid
  std::vector<int> cursor;   // build(): per-vertex CSR fill position

  // The ordered partition and the worklist over it.  `cell` holds every
  // vertex grouped by cell, cells in color order; a cell is named by its
  // start index there, which is also its color while refining.
  std::vector<int> cell;
  std::vector<int> cell_end; // cell start -> end index, exclusive
  std::vector<int> queue;    // cell starts to examine this round
  std::vector<int> next_queue;
  std::vector<char> queued;               // cell start -> already in next_queue?
  std::vector<std::pair<int, int>> split; // [start, end) of this round's recolored vertices

  // (Re)point the refiner at an undirected graph given as an edge list.
  // Every buffer is resized rather than rebuilt, so a workspace that has
  // already seen a complex this size allocates nothing here.
  void build(int n_vert, const std::vector<std::pair<int, int>>& edges) {
    adj_off.assign(n_vert + 1, 0);
    for (const auto& [a, b] : edges) {
      ++adj_off[a + 1];
      ++adj_off[b + 1];
    }
    for (int v = 0; v < n_vert; ++v)
      adj_off[v + 1] += adj_off[v];
    adj_flat.resize(adj_off[n_vert]);
    cursor.assign(adj_off.begin(), adj_off.begin() + n_vert);
    for (const auto& [a, b] : edges) {
      adj_flat[cursor[a]++] = b;
      adj_flat[cursor[b]++] = a;
    }
    nbr.resize(adj_flat.size());
    cell.resize(n_vert);
    cell_end.resize(n_vert);
    queued.assign(n_vert, 0);
  }

  // Lexicographic compare of the sorted neighbor-color slices of a and
  // b, shorter-is-less on a tie.  `nbr` must already hold this round's
  // sorted neighbor colors.  Only ever called on two vertices of the
  // SAME cell, where the own-color half of the signature is equal by
  // construction — which is why it is not compared here.
  bool nbr_less(int a, int b) const {
    int const la = adj_off[a + 1] - adj_off[a];
    int const lb = adj_off[b + 1] - adj_off[b];
    int const l = std::min(la, lb);
    for (int k = 0; k < l; ++k)
      if (nbr[adj_off[a] + k] != nbr[adj_off[b] + k])
        return nbr[adj_off[a] + k] < nbr[adj_off[b] + k];
    return la < lb;
  }

  // True iff a and b, cellmates, carry an identical signature this round.
  bool nbr_same(int a, int b) const {
    int const la = adj_off[a + 1] - adj_off[a];
    int const lb = adj_off[b + 1] - adj_off[b];
    if (la != lb)
      return false;
    for (int k = 0; k < la; ++k)
      if (nbr[adj_off[a] + k] != nbr[adj_off[b] + k])
        return false;
    return true;
  }

  void refine(std::vector<int>& color) {
    int const n = static_cast<int>(color.size());
    if (n == 0)
      return;

    // --- the input coloring, restated as an ordered partition --------------
    //
    // Vertices grouped by input color, groups in ascending color order —
    // which is what a round with the own color leading its signature would
    // have produced.  From here on a vertex's color is its cell's start.
    for (int v = 0; v < n; ++v)
      cell[v] = v;
    std::sort(cell.begin(), cell.begin() + n, [&](int a, int b) { return color[a] < color[b]; });
    queue.clear();
    int n_cells = 0;
    for (int i = 0; i < n;) {
      int const c = color[cell[i]];
      int j = i + 1;
      while (j < n && color[cell[j]] == c)
        ++j;
      for (int k = i; k < j; ++k)
        color[cell[k]] = i;
      cell_end[i] = j;
      queue.push_back(i);
      ++n_cells;
      i = j;
    }

    // --- rounds -------------------------------------------------------------
    while (!queue.empty() && n_cells < n) {
      // Every signature this round reads the colors the last round left, so
      // all the slices are taken before any of this round's splits land.
      for (int const s : queue) {
        if (cell_end[s] - s < 2)
          continue; // a singleton cannot split, so nothing reads its slice
        for (int i = s; i < cell_end[s]; ++i) {
          int const v = cell[i];
          int const b = adj_off[v];
          int const e = adj_off[v + 1];
          for (int k = b; k < e; ++k)
            nbr[k] = color[adj_flat[k]];
          // Degree 2 is not a special case here, it is the common one: a
          // bonded-component vertex has exactly two edges by construction,
          // one to its molecule and one to its partner, and those vertices
          // outnumber the molecules.
          if (e - b == 2) {
            if (nbr[b] > nbr[b + 1])
              std::swap(nbr[b], nbr[b + 1]);
          } else if (e - b > 2) {
            std::sort(nbr.begin() + b, nbr.begin() + e);
          }
        }
      }

      split.clear();
      for (int const s : queue) {
        int const e = cell_end[s];
        if (e - s < 2)
          continue;
        // Likewise the two-member cell, which is what most of a refinement's
        // trailing rounds are cutting.
        if (e - s == 2) {
          if (nbr_less(cell[s + 1], cell[s]))
            std::swap(cell[s], cell[s + 1]);
        } else {
          std::sort(cell.begin() + s, cell.begin() + e,
                    [&](int a, int b) { return nbr_less(a, b); });
        }
        for (int i = s + 1, start = s; i <= e; ++i) {
          if (i < e && nbr_same(cell[i - 1], cell[i]))
            continue;
          cell_end[start] = i;
          if (start != s)
            ++n_cells;
          start = i;
        }
        if (cell_end[s] == e)
          continue; // one piece: the cell held together
        // Recolor only now.  The scan above compares slices alone, so
        // leaving the colors put keeps it independent of the order the
        // round happens to cut its cells in.
        for (int b = s; b < e; b = cell_end[b])
          for (int k = b; k < cell_end[b]; ++k)
            color[cell[k]] = b;
        // The leading piece inherits the cell's own start, so its members
        // keep the color they had and cannot have changed any neighbor's
        // signature.  Only the rest are news to anybody.
        split.emplace_back(cell_end[s], e);
      }

      // Mark next round's work: the cells holding a neighbor of a vertex
      // this round recolored.  Every other cell is provably whole.
      next_queue.clear();
      for (const auto& [s, e] : split) {
        for (int i = s; i < e; ++i) {
          int const v = cell[i];
          for (int k = adj_off[v]; k < adj_off[v + 1]; ++k) {
            int const c = color[adj_flat[k]];
            if (queued[c] == 0) {
              queued[c] = 1;
              next_queue.push_back(c);
            }
          }
        }
      }
      for (int const s : next_queue)
        queued[s] = 0; // leave the flags clear for the next round
      queue.swap(next_queue);
    }

    // --- the dense 0..k-1 ranking the callers see ---------------------------
    int rank = -1;
    for (int i = 0; i < n; i = cell_end[i]) {
      ++rank;
      for (int k = i; k < cell_end[i]; ++k)
        color[cell[k]] = rank;
    }
  }
};

// Reusable per-canonicalization scratch for the leaf-test and renderer,
// so a search that visits many leaves allocates these buffers once, not
// once per leaf.
struct Scratch {
  std::vector<int> mol_colors;            // is_leaf: molecule-color copy
  std::vector<std::pair<int, int>> pairs; // is_leaf: (name_id, color) per molecule
  std::vector<int> order;                 // render_leaf: molecule ordering
  std::vector<int> comp_label;            // render: bond-label assignment
};

// Everything one canonicalization needs that is not the input graph.
// canonicalize() holds one on the stack for the duration of a call;
// canonical_order_fast holds one in the caller's Workspace and reuses it
// call after call, which is what makes the election allocation-free
// once warm (GH #53).
struct Core {
  // Port graph (plan §3.1).
  std::vector<int> comp_vertex; // global comp index -> vertex id, -1 if free
  std::vector<int> vertex_comp; // component-vertex index -> global comp index
  std::vector<int> comp_to_mol; // global comp index -> molecule index
  std::vector<std::pair<int, int>> edges;

  // Initial colors, as the key slices ranked_initial_colors ranks.
  std::vector<int> key_data;
  std::vector<int> key_off; // vertex -> [key_off[v], key_off[v+1]) in key_data
  std::vector<int> key_order;
  std::vector<std::pair<int, int>> free_pairs; // one molecule's unbonded comps

  std::vector<int> color;
  Refiner refiner;
  Scratch scratch;
};

// Refined color of each bonded component's vertex (-1 if free) — the
// renderer's within-molecule ordering key.
std::vector<int> bonded_colors(const ComplexGraph& g, const std::vector<int>& color,
                               const std::vector<int>& comp_vertex) {
  std::vector<int> bc(g.components.size(), -1);
  for (int gc = 0; gc < static_cast<int>(g.components.size()); ++gc) {
    if (comp_vertex[gc] >= 0)
      bc[gc] = color[comp_vertex[gc]];
  }
  return bc;
}

// Half of the leaf test: did every molecule vertex land in its own color
// class?  That half is the WHOLE test for the molecule ordering, which is
// sort-by-refined-color over the molecule vertices and nothing else.
//
// Once it holds, no leaf below this coloring can order the molecules
// differently: individualization only ever splits classes, and refine()
// never reorders colors that already differ (see Refiner), so every leaf
// in the subtree sorts the molecules exactly as this coloring does.  That
// is why an order-only caller stops here while canonicalize, which still
// has to pick a RENDER, does not.
bool mol_colors_distinct(const std::vector<int>& color, int n_mol, Scratch& s) {
  s.mol_colors.assign(color.begin(), color.begin() + n_mol);
  std::sort(s.mol_colors.begin(), s.mol_colors.end());
  for (int i = 1; i < n_mol; ++i)
    if (s.mol_colors[i] == s.mol_colors[i - 1])
      return false;
  return true;
}

// Leaf test: is `color` discriminating enough that `render` produces a
// unique, isomorphism-invariant string?  True iff (a) every molecule
// vertex landed in its own color class and (b) no molecule has two
// interchangeable bonded components — same name, same refined color —
// left tied.  Either residual symmetry is what individualization
// resolves; this same test is the fast-path gate and the search's leaf
// condition.  (Component vertices on *different* molecules, or of
// different names, may still share a color — that never affects the
// render, so it does not block a leaf.)
bool is_leaf(const ComplexGraph& g, const Tables& tab, const std::vector<int>& color,
             const std::vector<int>& comp_vertex, int n_mol, Scratch& s) {
  if (!mol_colors_distinct(color, n_mol, s))
    return false;
  for (int m = 0; m < n_mol; ++m) {
    const auto& mol = g.molecules[m];
    s.pairs.clear();
    for (int i = 0; i < mol.n_comp; ++i) {
      int const gc = mol.first_comp + i;
      if (comp_vertex[gc] < 0)
        continue; // free component — no vertex, physically interchangeable
      s.pairs.emplace_back(tab.comp_name_id[gc], color[comp_vertex[gc]]);
    }
    std::sort(s.pairs.begin(), s.pairs.end());
    for (size_t i = 1; i < s.pairs.size(); ++i)
      if (s.pairs[i] == s.pairs[i - 1])
        return false; // two same-name bonded components, same color
  }
  return true;
}

// Render the canonical BNGL string.  `order` is the canonical molecule
// ordering; `bonded_color[gc]` is the refined color of bonded component
// `gc`'s vertex (-1 if `gc` is free).  Within each molecule the type's
// name layout is preserved and same-name components are placed in
// `comp_key` order; bond labels are assigned 1,2,3,... in first-
// encounter order along the resulting walk.
std::string render(const ComplexGraph& g, const Tables& tab, const std::vector<int>& order,
                   const std::vector<int>& bonded_color, Scratch& s) {
  s.comp_label.assign(g.components.size(), 0); // 0 = unassigned
  int next_label = 1;
  std::string out;
  out.reserve(48 * order.size());

  for (size_t oi = 0; oi < order.size(); ++oi) {
    if (oi)
      out += '.';
    const auto& m = g.molecules[order[oi]];
    out += m.type_name;
    out += '(';
    for (int i = 0; i < m.n_comp; ++i) {
      if (i)
        out += ',';
      // Slot i keeps its declared name.  The canonical component placed
      // here is, among this molecule's components of that name, the
      // r-th smallest by comp_key — where r is slot i's rank among
      // same-name slots in declared order.  At a leaf no two same-name
      // components of one molecule share a comp_key, so the r-th is
      // unambiguous.  Resolved by an O(n_comp^3) scan with no
      // allocation; n_comp is tiny (this replaces a per-molecule
      // std::map<std::string,…> grouping).
      int const slot_gc = m.first_comp + i;
      int const name = tab.comp_name_id[slot_gc];
      int r = 0;
      for (int j = 0; j < i; ++j)
        if (tab.comp_name_id[m.first_comp + j] == name)
          ++r;
      int chosen = slot_gc;
      for (int k = 0; k < m.n_comp; ++k) {
        int const gk = m.first_comp + k;
        if (tab.comp_name_id[gk] != name)
          continue;
        const auto kk = std::make_pair(comp_key(bonded_color[gk], tab.state_id[gk]), k);
        int less = 0;
        for (int p = 0; p < m.n_comp; ++p) {
          int const gp = m.first_comp + p;
          if (tab.comp_name_id[gp] != name)
            continue;
          if (std::make_pair(comp_key(bonded_color[gp], tab.state_id[gp]), p) < kk)
            ++less;
        }
        if (less == r) {
          chosen = gk;
          break;
        }
      }
      const auto& c = g.components[chosen];
      out += c.name;
      if (!c.state.empty()) {
        out += '~';
        out += c.state;
      }
      if (c.partner >= 0) {
        int& lbl = s.comp_label[chosen];
        if (lbl == 0) {
          lbl = next_label++;
          s.comp_label[c.partner] = lbl;
        }
        out += '!';
        out += std::to_string(lbl);
      }
    }
    out += ')';
  }
  return out;
}

// The canonical molecule order of a leaf coloring: sort by refined
// color.  At a leaf every molecule color is distinct, so this is a
// total order, and it is the order the render below walks in.
const std::vector<int>& leaf_order(const std::vector<int>& color, int n_mol, Scratch& s) {
  s.order.resize(n_mol);
  for (int m = 0; m < n_mol; ++m)
    s.order[m] = m;
  std::sort(s.order.begin(), s.order.end(), [&](int a, int b) { return color[a] < color[b]; });
  return s.order;
}

// Render a leaf coloring: take the canonical molecule order and the
// per-component bonded colors, then render.
std::string render_leaf(const ComplexGraph& g, const Tables& tab, const std::vector<int>& color,
                        const std::vector<int>& comp_vertex, int n_mol, Scratch& s) {
  leaf_order(color, n_mol, s);
  return render(g, tab, s.order, bonded_colors(g, color, comp_vertex), s);
}

// Pick the target cell for individualization: the smallest non-singleton
// color class, ties broken by (smallest) color value.  This rule is a
// pure function of the partition, hence isomorphism-invariant — which is
// what makes the set of leaves the search explores correspond under any
// graph isomorphism, and so the lexicographic-minimum leaf render a
// true canonical form.  Returns the chosen color, or -1 if the coloring
// is already discrete (never happens when called on a non-leaf).
int pick_cell(const std::vector<int>& color) {
  // refine() emits a dense 0..k-1 ranking, so a flat tally indexed by
  // color replaces the std::map this used.
  int mx = 0;
  for (int const c : color)
    mx = std::max(mx, c);
  std::vector<int> count(mx + 1, 0);
  for (int const c : color)
    ++count[c];
  int chosen = -1;
  int best_size = 0;
  for (int c = 0; c <= mx; ++c) { // ascending color order
    if (count[c] < 2)
      continue;
    if (chosen < 0 || count[c] < best_size) {
      chosen = c;
      best_size = count[c];
    }
  }
  return chosen;
}

// Individualization-refinement search state (plan §3.2 step 3).
// Inputs are held by pointer (not reference) so the struct stays a
// plain value type — clang-tidy gates reference data members.
struct SearchState {
  const ComplexGraph* g;
  const Tables* tab;
  Refiner* refiner;
  const std::vector<int>* comp_vertex;
  int n_mol;
  long leaf_budget;            // remaining leaves before the §5 guard trips
  bool best_set = false;       // false until the first leaf is rendered
  std::string best;            // lexicographically minimal leaf render so far
  std::vector<int> best_order; // the molecule ordering `best` was rendered in
  Scratch scratch;             // reused across every leaf this search visits
};

// Recurse: `color` is a WL-stable coloring.  At a leaf, render and keep
// the minimum.  Otherwise pick a target cell and, for each of its
// vertices, individualize that vertex (give it a fresh top color),
// re-refine, and recurse.  Individualization strictly splits the cell,
// so every path reaches a discrete (hence leaf) coloring in at most
// n_vert steps — termination needs no separate depth bound; the leaf
// budget guards only against a pathological branching factor.
void search(SearchState& st, const std::vector<int>& color) {
  if (is_leaf(*st.g, *st.tab, color, *st.comp_vertex, st.n_mol, st.scratch)) {
    std::string label = render_leaf(*st.g, *st.tab, color, *st.comp_vertex, st.n_mol, st.scratch);
    if (!st.best_set || label < st.best) {
      st.best = std::move(label);
      // render_leaf leaves the ordering it rendered in `scratch.order`;
      // keep the winner's alongside the winning string.
      st.best_order = st.scratch.order;
      st.best_set = true;
    }
    --st.leaf_budget;
    return;
  }
  if (st.leaf_budget <= 0)
    return; // §5 safety valve — keep the best leaf already found

  int const target = pick_cell(color);
  int mx = 0;
  for (int const c : color)
    mx = std::max(mx, c);

  int const n_vert = static_cast<int>(color.size());
  for (int v = 0; v < n_vert; ++v) {
    if (color[v] != target)
      continue;
    if (st.leaf_budget <= 0)
      return;
    // Individualize v: a fresh color above every existing one.  refine
    // re-ranks, leaving v alone in the top class; the choice is applied
    // identically to isomorphic graphs, so the search trees correspond.
    std::vector<int> branch = color;
    branch[v] = mx + 1;
    st.refiner->refine(branch);
    search(st, branch);
  }
}

// Build the port graph (plan §3.1) into `core` and return the vertex
// count.  Shared by both entry points, which differ only in where their
// molecule spans and bond partners come from:
//
//   Vertices: [0, n_mol)        molecule vertices
//             [n_mol, n_vert)   one per bonded component
//
// comp_vertex[gc] is the vertex id of bonded component `gc`, or -1 for an
// unbonded component (no vertex — it folds into its molecule's color).
// Edges are molecule <-> each of its bonded components, plus one per bond.
//
// `mol_span(m)` gives molecule m's (first_comp, n_comp) and
// `partner_of(gc)` component gc's bond partner (-1 = free).
template <class MolSpan, class PartnerOf>
int build_port_graph(int n_mol, int n_comp, MolSpan mol_span, PartnerOf partner_of, Core& core) {
  core.comp_vertex.assign(n_comp, -1);
  core.comp_to_mol.assign(n_comp, -1);
  core.vertex_comp.clear();
  for (int m = 0; m < n_mol; ++m) {
    const auto [first, count] = mol_span(m);
    for (int i = 0; i < count; ++i) {
      int const gc = first + i;
      core.comp_to_mol[gc] = m;
      if (partner_of(gc) >= 0) {
        core.comp_vertex[gc] = n_mol + static_cast<int>(core.vertex_comp.size());
        core.vertex_comp.push_back(gc);
      }
    }
  }
  int const n_cv = static_cast<int>(core.vertex_comp.size());
  core.edges.clear();
  for (int cv = 0; cv < n_cv; ++cv)
    core.edges.emplace_back(n_mol + cv, core.comp_to_mol[core.vertex_comp[cv]]);
  for (int cv = 0; cv < n_cv; ++cv) {
    int const gc = core.vertex_comp[cv];
    int const partner = partner_of(gc);
    if (gc < partner) // emit each bond edge once
      core.edges.emplace_back(n_mol + cv, core.comp_vertex[partner]);
  }
  return n_mol + n_cv;
}

// Initial vertex colors for a RankedComplex — the integer counterpart of
// molecule_color / component_color plus canonicalize_impl's intern map.
//
// Each vertex gets a KEY SLICE in core.key_data:
//
//   component vertex : [0, name_rank, state_rank]
//   molecule vertex  : [1, type_rank, then (name_rank, state_rank) for
//                       each UNBONDED component, ascending]
//
// and the colors are the dense rank of those slices in lexicographic
// order.  That reproduces the string ranking exactly:
//
//   * `C:` sorts before `M:` — the leading 0 / 1.
//   * A molecule color writes its free components as a `,`-joined list
//     inside `(...)`, and `(`, `,` and `)` are all outranked by every
//     character a BNGL identifier may hold.  So comparing the joined
//     lists is comparing the element sequences, and a shorter list that
//     is a prefix of a longer one sorts first — which is what comparing
//     a shorter key slice does.
//   * Element order, and name/state order within an element, are carried
//     by the ranks themselves: see rank_component_names / rank_state_names.
void ranked_initial_colors(const RankedComplex& c, int n_mol, Core& core) {
  int const n_cv = static_cast<int>(core.vertex_comp.size());
  int const n_vert = n_mol + n_cv;
  core.key_data.clear();
  core.key_off.assign(n_vert + 1, 0);

  for (int m = 0; m < n_mol; ++m) {
    const auto& mol = c.molecules[m];
    core.key_data.push_back(1);
    core.key_data.push_back(mol.type_rank);
    core.free_pairs.clear();
    for (int i = 0; i < mol.n_comp; ++i) {
      const auto& comp = c.components[mol.first_comp + i];
      if (comp.partner < 0)
        core.free_pairs.emplace_back(comp.name_rank, comp.state_rank);
    }
    std::sort(core.free_pairs.begin(), core.free_pairs.end());
    for (const auto& [name, state] : core.free_pairs) {
      core.key_data.push_back(name);
      core.key_data.push_back(state);
    }
    core.key_off[m + 1] = static_cast<int>(core.key_data.size());
  }
  for (int cv = 0; cv < n_cv; ++cv) {
    const auto& comp = c.components[core.vertex_comp[cv]];
    core.key_data.push_back(0);
    core.key_data.push_back(comp.name_rank);
    core.key_data.push_back(comp.state_rank);
    core.key_off[n_mol + cv + 1] = static_cast<int>(core.key_data.size());
  }

  const int* kd = core.key_data.data();
  const auto& off = core.key_off;
  auto slice_less = [&](int a, int b) {
    return std::lexicographical_compare(kd + off[a], kd + off[a + 1], kd + off[b], kd + off[b + 1]);
  };
  auto slice_equal = [&](int a, int b) {
    return off[a + 1] - off[a] == off[b + 1] - off[b] &&
           std::equal(kd + off[a], kd + off[a + 1], kd + off[b]);
  };

  core.key_order.resize(n_vert);
  std::iota(core.key_order.begin(), core.key_order.end(), 0);
  std::sort(core.key_order.begin(), core.key_order.end(), slice_less);
  core.color.assign(n_vert, 0);
  int rank = -1;
  for (int i = 0; i < n_vert; ++i) {
    if (i == 0 || !slice_equal(core.key_order[i - 1], core.key_order[i]))
      ++rank;
    core.color[core.key_order[i]] = rank;
  }
}

// Dense ranks of `keys` in lexicographic order: equal strings share a
// rank, and rank(a) < rank(b) iff a < b.  Runs once per model.
std::vector<int> dense_ranks(const std::vector<std::string>& keys) {
  std::vector<int> idx(keys.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return keys[a] < keys[b]; });
  std::vector<int> out(keys.size(), 0);
  int rank = -1;
  for (size_t i = 0; i < idx.size(); ++i) {
    if (i == 0 || keys[idx[i]] != keys[idx[i - 1]])
      ++rank;
    out[idx[i]] = rank;
  }
  return out;
}

// Shared body of canonicalize() and canonical_order().  `order_only`
// suppresses the render on every coloring that already fixes the molecule
// ordering — the one piece of work a caller that wants the ordering alone
// has no use for.  It changes nothing else: the ordering is
// sort-by-refined-color and does not consult the string, and a coloring
// that still ties two molecules renders regardless, since there the
// strings are what select the canonical leaf.
CanonForm canonicalize_impl(const ComplexGraph& g, bool order_only) {
  int const n_mol = g.molecule_count();
  if (n_mol == 0)
    return {"", true, {}};

  Tables const tab = build_tables(g);

  // --- Build the port graph ------------------------------------------------
  Core core;
  int const n_vert = build_port_graph(
      n_mol, g.component_count(),
      [&](int m) {
        return std::pair<int, int>{g.molecules[m].first_comp, g.molecules[m].n_comp};
      },
      [&](int gc) { return g.components[gc].partner; }, core);
  const std::vector<int>& comp_vertex = core.comp_vertex;

  // --- Initial colors ------------------------------------------------------
  //
  // Intern color strings through a sorted map, so the assigned integers
  // follow lexicographic string order — the integer colors are a
  // canonical, structure-derived ranking from round zero.
  //
  // ranked_initial_colors above is the integer twin of this block.  The
  // two must produce the same RANKING, or the two entry points would
  // answer different orders for the same complex.
  std::vector<std::string> init_str(n_vert);
  for (int m = 0; m < n_mol; ++m)
    init_str[m] = molecule_color(g, g.molecules[m]);
  for (int cv = 0; cv < static_cast<int>(core.vertex_comp.size()); ++cv)
    init_str[n_mol + cv] = component_color(g.components[core.vertex_comp[cv]]);

  std::map<std::string, int> intern;
  for (const auto& s : init_str)
    intern.emplace(s, 0);
  {
    int next = 0;
    for (auto& [str, id] : intern)
      id = next++;
  }
  std::vector<int>& color = core.color;
  color.resize(n_vert);
  for (int v = 0; v < n_vert; ++v)
    color[v] = intern[init_str[v]];

  // --- 1-WL color refinement ----------------------------------------------
  Refiner& refiner = core.refiner;
  refiner.build(n_vert, core.edges);
  refiner.refine(color);

  // --- Fast path -----------------------------------------------------------
  //
  // If refinement alone discriminated the complex (plan §3.2 step 2),
  // the refined colors fix a unique isomorphism-invariant ordering;
  // render directly.  This is the overwhelmingly common case.  A caller
  // that wants only the ordering stops one condition earlier — see
  // mol_colors_distinct.
  Scratch& scratch = core.scratch;
  if (order_only) {
    if (mol_colors_distinct(color, n_mol, scratch))
      return {std::string{}, /*fast_path=*/true, leaf_order(color, n_mol, scratch)};
  } else if (is_leaf(g, tab, color, comp_vertex, n_mol, scratch)) {
    // render_leaf writes the ordering into scratch.order, so read it out
    // after the call rather than before.
    std::string label = render_leaf(g, tab, color, comp_vertex, n_mol, scratch);
    return {std::move(label), /*fast_path=*/true, std::move(scratch.order)};
  }

  // --- Individualization-refinement (plan §3.2 step 3) ---------------------
  //
  // A genuinely symmetric complex (rings, homo-oligomers) survived
  // refinement.  Branch on a target cell, individualize each member,
  // re-refine, and recurse; the canonical label is the lexicographically
  // minimal leaf render.  This is a true canonical form — see search()
  // and pick_cell() for why the leaf set is isomorphism-invariant.
  SearchState st{&g,          &tab,  &refiner,      &comp_vertex,       n_mol,
                 kLeafBudget, false, std::string{}, std::vector<int>{}, Scratch{}};
  search(st, color);
  return {std::move(st.best), /*fast_path=*/false, std::move(st.best_order)};
}

} // namespace

CanonForm canonicalize(const ComplexGraph& g) { return canonicalize_impl(g, /*order_only=*/false); }

std::string canonical_label(const ComplexGraph& g) { return canonicalize(g).label; }

std::vector<int> canonical_order(const ComplexGraph& g) {
  return canonicalize_impl(g, /*order_only=*/true).mol_order;
}

// ===========================================================================
// The integer entry point (GH #53)
// ===========================================================================

// A molecule type name is written `M:Name(`, and `(` is outranked by every
// character a BNGL identifier may hold, so a name that is a prefix of
// another sorts FIRST.  Ranking `name + '('` says exactly that, whatever
// the characters turn out to be.
std::vector<int> rank_molecule_type_names(const std::vector<std::string>& names) {
  std::vector<std::string> keys;
  keys.reserve(names.size());
  for (const auto& n : names)
    keys.push_back(n + '(');
  return dense_ranks(keys);
}

// A component name is always written with its state behind a `~`, and `~`
// (0x7E) OUTRANKS every character a BNGL identifier may hold.  So `ab`
// sorts before `a` here — the opposite of the molecule-type rule above,
// and the reason these are two functions and not one.
std::vector<int> rank_component_names(const std::vector<std::string>& names) {
  std::vector<std::string> keys;
  keys.reserve(names.size());
  for (const auto& n : names)
    keys.push_back(n + '~');
  return dense_ranks(keys);
}

// A state is written last in its color string, and inside a molecule color
// it is followed by `,` or `)` — both outranked by every identifier
// character, exactly as end-of-string is.  So plain lexicographic order,
// and "" (a component with no internal state) ranks first.
std::vector<int> rank_state_names(const std::vector<std::string>& states) {
  return dense_ranks(states);
}

struct Workspace::Impl {
  Core core;
};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace&&) noexcept = default;
Workspace& Workspace::operator=(Workspace&&) noexcept = default;

bool canonical_order_fast(const RankedComplex& c, Workspace& ws, std::vector<int>& out) {
  Core& core = ws.impl_->core;
  int const n_mol = c.molecule_count();
  if (n_mol == 0) {
    out.clear();
    return true;
  }

  int const n_vert = build_port_graph(
      n_mol, c.component_count(),
      [&](int m) {
        return std::pair<int, int>{c.molecules[m].first_comp, c.molecules[m].n_comp};
      },
      [&](int gc) { return c.components[gc].partner; }, core);

  ranked_initial_colors(c, n_mol, core);
  core.refiner.build(n_vert, core.edges);
  core.refiner.refine(core.color);

  if (!mol_colors_distinct(core.color, n_mol, core.scratch))
    return false; // interchangeable molecules — only the render can choose
  const std::vector<int>& order = leaf_order(core.color, n_mol, core.scratch);
  out.assign(order.begin(), order.end());
  return true;
}

} // namespace rulemonkey::canonical
