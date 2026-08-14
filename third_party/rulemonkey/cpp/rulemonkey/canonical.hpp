#pragma once

// Canonical complex labeling — the pure species-identity primitive.
//
// Design & decisions: dev/canonical_labeling_plan_2026_05_16.md.
// Issue #9 §2 (`.species` output) is the first consumer; partial
// scaling (plan §7.2) is the consumer that shapes the design.
//
// This header is the PURE CORE only (plan decision #3): a value type
// `ComplexGraph` and the pure function `canonical_label`.  It has NO
// dependency on AgentPool / Engine internals, so it is callable from
// both the on-demand batch sweep and the (future) cached-incremental
// layer (decision #6), and is unit-testable with hand-built graphs and
// no simulator.
//
// `extract_complex` (engine -> ComplexGraph) is deliberately NOT
// declared here: AgentPool is TU-local to engine.cpp by design, so the
// engine->graph bridge lives with the engine.  See the plan handoff /
// the step-0 report.
//
// SCOPE as of plan §6 step 3: `canonical_label` is a complete
// canonicalizer — 1-WL color refinement, the all-distinct-colors fast
// path, and individualization-refinement search for genuinely
// symmetric complexes (rings, homo-oligomers).  The returned label is
// a *true* canonical form on every input: isomorphic complexes always
// produce byte-equal labels.  `canonicalize().fast_path` reports only
// which path was taken (informational), not a correctness boundary.
//
// A second entry point — `canonical_order_fast` over a `RankedComplex`,
// with a reusable `Workspace` — answers the molecule ORDER alone for a
// caller that asks once per simulator event rather than once per
// `.species` sweep (GH #53).  Same algorithm, same answer; it just
// never touches a string.  See the RankedComplex block below.

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rulemonkey::canonical {

// ---------------------------------------------------------------------------
// ComplexGraph — the vertex-colored port graph of one connected complex.
//
// Value type only: molecule / component / bond structure, no engine
// reference.  Hand-buildable for tests; produced by the engine-side
// `extract_complex` for live pools.
//
// Components are stored in one flat array, grouped by molecule and in
// each molecule's BNGL declaration order (the order the renderer
// emits).  A bond is recorded symmetrically as a `partner` global
// component index on both endpoints (-1 = free), mirroring the
// engine's ComponentInstance representation.
// ---------------------------------------------------------------------------
struct ComplexGraph {
  struct Component {
    std::string name;
    std::string state; // "" = stateless component (no `~state` rendered)
    int partner = -1;  // global component index of bonded partner; -1 = free
  };
  struct Molecule {
    std::string type_name;
    int first_comp = 0; // index of this molecule's first component
    int n_comp = 0;     // component count
  };

  std::vector<Molecule> molecules;
  std::vector<Component> components; // flat, grouped by molecule

  // --- Builder -------------------------------------------------------------

  // Append a molecule with its components in declaration order.
  // `comps` is a list of (name, state) pairs; pass "" for a stateless
  // component.  Returns the new molecule's index.
  int add_molecule(const std::string& type_name,
                   const std::vector<std::pair<std::string, std::string>>& comps);

  // Bond two components, each addressed as (molecule index, local
  // component index within that molecule).  A self-bond (same molecule,
  // two different components) is permitted.
  void add_bond(int mol_a, int local_a, int mol_b, int local_b);

  // Global component index of molecule `mol`'s local component `local`.
  int global_comp(int mol, int local) const { return molecules[mol].first_comp + local; }
  int molecule_count() const { return static_cast<int>(molecules.size()); }
  int component_count() const { return static_cast<int>(components.size()); }
};

// ---------------------------------------------------------------------------
// Canonicalization
// ---------------------------------------------------------------------------

// Result of canonicalizing one complex.
struct CanonForm {
  std::string label; // canonical BNGL species string (also the dedup key).
                     // ALWAYS a true canonical form: isomorphic complexes
                     // yield byte-equal labels.
  bool fast_path;    // INFORMATIONAL — which algorithm path produced
                     // `label`, not a correctness boundary.
                     // true  -> 1-WL refinement alone discriminated the
                     //          complex; no search ran (the common case).
                     // false -> the complex had genuine symmetry (a ring
                     //          or homo-oligomer) and `label` was found
                     //          by individualization-refinement search.

  // The molecule ordering `label` was rendered in: mol_order[k] is the
  // INPUT molecule index that `label` writes at position k.  Size
  // molecule_count(); a permutation of [0, molecule_count()).
  //
  // This is the label's own ordering, so it inherits the label's
  // guarantee: isomorphic complexes order corresponding molecules
  // identically.  That makes "the molecule at position 0" a property of
  // the species rather than of how the complex happened to be built,
  // which is what a caller needs when it has to pick ONE molecule of a
  // complex and have every copy of that species pick the same one —
  // BioNetGen prices a collapsed reaction instance at exactly this
  // molecule (GH #52).
  //
  // Within an orbit of the complex's automorphism group the choice is
  // arbitrary but deterministic: interchangeable molecules are
  // interchangeable, so which one lands first cannot be observed.
  std::vector<int> mol_order;
};

// Canonicalize a complex: 1-WL color refinement, the all-distinct fast
// path, and individualization-refinement search for symmetric residue
// (plan §3.2).  Pure function of the graph; `label` is a true canonical
// form on every input.  `fast_path` reports which path was taken, and
// `mol_order` the ordering `label` was rendered in.
CanonForm canonicalize(const ComplexGraph& g);

// Convenience wrapper: the canonical BNGL string only.  This is the
// signature pinned by the plan (§4); both calling modes use it.
std::string canonical_label(const ComplexGraph& g);

// CanonForm::mol_order only — for a caller that needs to name a molecule
// isomorphism-invariantly and has no use for the string.  Identical
// output to `canonicalize(g).mol_order`, but skips rendering whenever
// refinement alone has landed every molecule in its own color class,
// which is all the ORDER needs (see canonical_order_fast for why that
// is weaker than a leaf and still exact).  A complex with genuinely
// interchangeable molecules still renders: there the canonical order IS
// the one belonging to the lexicographically minimal render, so the
// strings are what pick it.
std::vector<int> canonical_order(const ComplexGraph& g);

// ---------------------------------------------------------------------------
// The integer entry point (GH #53)
//
// `canonicalize` interns its vertex colors through a std::map of color
// STRINGS.  That is the right shape for the `.species` batch sweep,
// which wants the strings anyway, and the wrong shape for a caller that
// asks once per simulator event for a single integer: building and
// interning those strings is roughly three quarters of the cost, and
// all of it is thrown away when the answer wanted is `mol_order`.
//
// A caller that already holds its names as dense integers — the engine
// does: molecule `type_index`, component `comp_type_index`,
// `state_index` — can describe the complex as a RankedComplex and skip
// the strings entirely.
// ---------------------------------------------------------------------------

// The same vertex-colored port graph as ComplexGraph, with every name
// replaced by a dense integer RANK.
//
// The ranks are not free integers.  `canonical_order_fast` has to answer
// exactly what `canonicalize().mol_order` answers, and the refined
// colors — hence the ordering — depend on the ORDER of the initial
// colors, not only on which vertices share one.  So the ranks have to
// sort the way the color strings do.  Do not hand-roll them: the
// `rank_*` functions below are the single definition of that
// correspondence.
struct RankedComplex {
  struct Component {
    int name_rank = 0;  // rank_component_names()
    int state_rank = 0; // rank_state_names(); the rank of "" for a component
                        // with no internal state
    int partner = -1;   // flat component index of bonded partner; -1 = free
  };
  struct Molecule {
    int type_rank = 0; // rank_molecule_type_names()
    int first_comp = 0;
    int n_comp = 0;
  };

  std::vector<Molecule> molecules;
  std::vector<Component> components; // flat, grouped by molecule, declaration order

  // Reuse across calls: drops the contents, keeps the capacity.
  void clear() {
    molecules.clear();
    components.clear();
  }
  int molecule_count() const { return static_cast<int>(molecules.size()); }
  int component_count() const { return static_cast<int>(components.size()); }
};

// Rank tables for RankedComplex.  Each takes names in the CALLER's own
// indexing — duplicates welcome, equal strings get equal ranks — and
// returns one rank per input position, so a model's tables can be
// ranked in place.
//
// Three functions rather than one because the three positions compare
// differently.  A color string writes a molecule type as `Type(`, a
// component name as `name~`, and a state last.  Every character a BNGL
// identifier may hold outranks `(`, `,` and `)` and is outranked by
// `~`, so a name that is a proper prefix of another sorts BEFORE it as
// a molecule type and AFTER it as a component name.  That asymmetry is
// exactly why the tables belong here rather than in the caller.
std::vector<int> rank_molecule_type_names(const std::vector<std::string>& names);
std::vector<int> rank_component_names(const std::vector<std::string>& names);
std::vector<int> rank_state_names(const std::vector<std::string>& states);

// Reusable scratch for canonical_order_fast: the port-graph adjacency,
// the refinement buffers, and the leaf-order workspace.  Hold one per
// caller and pass the same one to every call — that reuse, which a pure
// function taking a fresh graph cannot have, is half of what #53 is
// about.  Not thread-safe; one workspace per thread.
class Workspace {
public:
  Workspace();
  ~Workspace();
  Workspace(Workspace&&) noexcept;
  Workspace& operator=(Workspace&&) noexcept;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  struct Impl; // defined in canonical.cpp

private:
  std::unique_ptr<Impl> impl_;
  friend bool canonical_order_fast(const RankedComplex&, Workspace&, std::vector<int>&);
};

// CanonForm::mol_order for a ranked complex, without building a single
// string.  After the first call warms the workspace this allocates
// nothing.
//
// Answers only when 1-WL refinement alone lands every molecule vertex in
// its own color class.  That is strictly weaker than canonicalize()'s
// leaf test, which additionally requires no molecule to have two
// interchangeable bonded components left tied — and it is all the ORDER
// needs.  Individualization only ever splits color classes, and
// refinement preserves the relative order of colors that already
// differ, so once the molecule colors are distinct, no leaf the search
// could reach can reorder them; every leaf renders the molecules in the
// one order this returns.
//
// Returns true with `out` set to the permutation
// `canonicalize(<the same complex>).mol_order` returns.  Returns false,
// leaving `out` untouched, when the complex has genuinely
// interchangeable molecules (a ring, a homo-oligomer): there the
// canonical order IS the one belonging to the lexicographically minimal
// render, so the strings are what pick it and the caller must fall back
// to canonical_order(ComplexGraph).
bool canonical_order_fast(const RankedComplex& c, Workspace& ws, std::vector<int>& out);

} // namespace rulemonkey::canonical
