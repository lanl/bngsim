"""Extract SBML model features for parity matrix annotations.

Provides functions to extract structural and dynamic features from SBML models
for display in the parity matrix HTML report.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def extract_sbml_features(xml_path: str | Path) -> dict:
    """Extract model features from SBML file.

    Returns a dict with:
        - n_species, n_reactions, n_parameters, n_compartments (counts)
        - features: list of notable SBML features (events, rules, etc.)
        - citation: Author et al. (YEAR) if available
        - file_size_kb: SBML file size in KB
    """
    try:
        xml_path = Path(xml_path)

        # Get file size in KB
        file_size_bytes = xml_path.stat().st_size
        file_size_kb = file_size_bytes / 1024.0

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract namespace from root tag if present
        ns = {}
        if "}" in root.tag:
            namespace_uri = root.tag.split("}")[0].strip("{")
            ns = {"sbml": namespace_uri}

        # Find model element - try with and without namespace
        model = None
        if ns:
            model = root.find("sbml:model", ns)
        if model is None:
            model = root.find("model")
        if model is None:
            # Last resort: find any element called model
            for elem in root.iter():
                if elem.tag.endswith("model") or elem.tag == "model":
                    model = elem
                    break

        if model is None:
            return _empty_features()

        # Count structural elements
        def count_elements(tag):
            """Count elements with given tag, trying different namespace patterns."""
            # Try without namespace first
            count = len(model.findall(f".//{tag}"))
            if count == 0 and ns:
                # Try with namespace
                count = len(model.findall(f".//sbml:{tag}", ns))
            return count

        n_species = count_elements("species")
        n_reactions = count_elements("reaction")
        n_parameters = count_elements("parameter")
        n_compartments = count_elements("compartment")

        # Detect features
        features = []

        # Events
        n_events = count_elements("event")
        if n_events > 0:
            features.append(f"events ({n_events})")

        # Rules
        n_rate_rules = count_elements("rateRule")
        n_assignment_rules = count_elements("assignmentRule")
        n_algebraic_rules = count_elements("algebraicRule")

        if n_rate_rules > 0:
            features.append("rate rules")
        if n_assignment_rules > 0:
            features.append("assignment rules")
        if n_algebraic_rules > 0:
            features.append("algebraic rules")

        # Initial assignments
        if count_elements("initialAssignment") > 0:
            features.append("initial assignments")

        # Constraints
        if count_elements("constraint") > 0:
            features.append("constraints")

        # Function definitions
        if count_elements("functionDefinition") > 0:
            features.append("function definitions")

        # Variable volume compartments (check for non-constant)
        compartments = model.findall(".//compartment")
        if not compartments and ns:
            compartments = model.findall(".//sbml:compartment", ns)

        has_variable_volume = any(c.get("constant") == "false" for c in compartments)
        if has_variable_volume:
            features.append("variable volume")

        # Compartmental (has compartments)
        if n_compartments > 0:
            features.append("compartmental")

        # Large model
        if n_species >= 1000 or n_reactions >= 3000:
            features.append("large model")

        # Try to extract citation from annotation
        citation = _extract_citation(model, ns)

        return {
            "n_species": n_species,
            "n_reactions": n_reactions,
            "n_parameters": n_parameters,
            "n_compartments": n_compartments,
            "features": features,
            "citation": citation,
            "file_size_kb": round(file_size_kb, 1),
        }

    except Exception:
        return _empty_features()


def _empty_features() -> dict:
    """Return a feature dict for a model whose SBML could NOT be parsed (empty
    file, COPASI/other non-SBML, or a parse error). The structural counts are
    OMITTED rather than set to 0, so a failed extraction renders "?" and sorts
    LAST (unknown complexity) instead of masquerading as a tiny 0-species model
    that sorts first. A genuinely species-free but valid SBML model (state carried
    by parameters + rate rules) still returns a real ``n_species: 0`` from the
    success path and sorts first, as intended."""
    return {
        "features": [],
        "citation": "Unknown",
        "file_size_kb": 0.0,
    }


def _extract_citation(model, ns) -> str:
    """Try to extract citation from SBML annotation.

    Returns "Author et al. (YEAR)" or "Unknown".
    """
    import re

    # Try to find annotation with citation info
    annotation = model.find(".//annotation")
    if annotation is None and ns:
        annotation = model.find(".//sbml:annotation", ns)
    if annotation is None:
        return "Unknown"

    # Get full annotation text
    annotation_text = ET.tostring(annotation, encoding="unicode", method="text")

    # Look for vCard:Family (author last name) - common in BioModels
    author = None
    for elem in annotation.iter():
        tag = elem.tag.lower()
        if "family" in tag and elem.text:
            author = elem.text.strip()
            break

    # Look for year in Journal or other fields
    year = None
    year_match = re.search(r"\b(19|20)\d{2}\b", annotation_text)
    if year_match:
        year = year_match.group(0)

    if author and year:
        return f"{author} et al. ({year})"

    # Fallback: look for Journal citation with year
    journal_match = re.search(r"([A-Z][a-z]+)\s+et\s+al\..*?(\d{4})", annotation_text)
    if journal_match:
        return f"{journal_match.group(1)} et al. ({journal_match.group(2)})"

    # Another fallback: extract from bibo:authorList
    author_list_match = re.search(r"bibo:authorList[^<]*([A-Z][a-z]+\s+[A-Z]+)", annotation_text)
    if author_list_match and year:
        author_name = author_list_match.group(1).split()[0]
        return f"{author_name} et al. ({year})"

    # Last resort: just year if we have it
    if year:
        return f"Unknown ({year})"

    return "Unknown"
