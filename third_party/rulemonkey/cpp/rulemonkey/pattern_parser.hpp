#pragma once

// Runtime BNGL species-pattern parser (issue #9 §1).
//
// The text frontend for the §1 session APIs: it turns a pattern string
// an embedder hands in at runtime — e.g. "A(b!1).B(a!1)" — into
// RuleMonkey's internal `Pattern`, resolved against the molecule types
// of an already-loaded `Model`.  It is the analogue of the XML-side
// `parse_pattern` in simulator.cpp, but for runtime pattern fragments
// rather than the BNG-XML the model-load contract is built on.
//
// This is NOT a `.bngl` model loader — the model-load contract stays
// BNG-XML.  It parses a single pattern fragment only.
//
// The function is pure (string + Model -> Pattern, no engine/pool
// state), so it is unit-testable in isolation.

#include "model.hpp"

#include <string>

namespace rulemonkey {

// Parse `text` as an exact, fully-specified, connected BNGL species
// pattern and return the resolved `Pattern` (issue #9 §1 design
// decision A — wildcard / partial patterns are out of scope).
//
// `text` must denote a single complex:
//   - every molecule name resolves to a declared MoleculeType;
//   - every component of each molecule type is listed exactly once
//     (same-named "symmetric" components are matched by occurrence);
//   - every stateful component carries a concrete `~state` drawn from
//     the type's allowed-state set; stateless components carry none;
//   - bonds are numeric labels `!N`, each `N` appearing on exactly two
//     distinct components; the bond wildcards `!+` / `!?` are rejected;
//   - the molecules form one connected complex.
//
// Throws std::runtime_error, naming the offending token, on any
// violation of the grammar or of the rules above.
Pattern parse_species_pattern(const std::string& text, const Model& model);

} // namespace rulemonkey
