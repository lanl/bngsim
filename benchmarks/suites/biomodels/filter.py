#!/usr/bin/env python3
"""
Stage-1 BioModels SBML filter.

Deterministically partitions a directory of BioModels SBML files into
``keep`` (structurally simulable as an ODE system) and ``drop``
(malformed / un-simulable), and tags structural features. Emits
``manifest.csv`` -- the reproducible record of the good/bad split: a
reviewer who re-runs this script on the same corpus gets the identical
partition, and the per-file SHA-256 confirms byte-identical inputs.

Drop criteria are *un-simulability*, NOT SBML-validation severity
(validation severity is libSBML-version-sensitive; these structural
facts are not):

  no_model             -- libSBML produced no Model element (unreadable)
  missing_kinetic_law  -- a reaction has no kineticLaw (undefined ODE term)
  zero_dynamics        -- no reactions and no rate rules (nothing to integrate)

Everything else is kept. Structural features -- events, event delays,
algebraic rules, function definitions -- are *tagged*, never a drop
reason: BNGsim handles them, and downstream suites subset on the tags
(e.g. the events-tagged models feed the SBML-events figure).

Stage 2 -- whether BNGsim / libRoadRunner / AMICI actually load and
simulate a kept model -- is a *coverage result* recorded by the suite
runner, deliberately kept distinct from this structural partition.

Usage:
    python filter.py [--data DIR] [--out manifest.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import libsbml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "data" / "sbml_downloads"
DEFAULT_OUT = SCRIPT_DIR / "manifest.csv"

# Curated manual exclusions — un-simulable models that libSBML's structural checks
# do NOT flag (it validates them clean) but that BOTH bngsim AND RoadRunner reject:
# each references an identifier that exists in no namespace, so neither engine can
# build the model. Reference-engine-confirmed broken (not a bngsim bug). Surfaced by
# the rr_parity ODE EXCEPTION triage (2026-06-02); see
# parity_checks/rr_parity/dev/notes/rr_parity_triage.md for per-model evidence.
MANUAL_DROP = {
    "MODEL1208280001": "undefined_symbol",  # kinetic law refs 'rateLaw1' (defined nowhere; RR also rejects)
    "MODEL1101180000": "undefined_symbol",  # refs 'size_subVolume' (no such compartment/param; RR also rejects)
    "MODEL5974712823": "undefined_symbol",  # refs 'size_subVolume' (RR also rejects)
    # #89: 5 Nutsch2005 phototaxis models — the assignment rule
    # `CheYP_cw = min(max, 0, meanAct, ymax)` references `max` as a bare <ci>,
    # but `max` is declared nowhere (a corrupt export of `min(max(0,meanAct),ymax)`).
    # libSBML flags the MathML as invalid; RoadRunner 2.9.2 also rejects
    # ("symbol 'max' is not physically stored"). Not a bngsim min/max bug —
    # the 37 well-formed min/max corpus models load and pass parity.
    "MODEL0403888565": "undefined_symbol",  # min(max,…) — 'max' undefined; RR also rejects (#89)
    "MODEL0403928902": "undefined_symbol",  # min(max,…) — 'max' undefined; RR also rejects (#89)
    "MODEL0403954746": "undefined_symbol",  # min(max,…) — 'max' undefined; RR also rejects (#89)
    "MODEL0403988150": "undefined_symbol",  # min(max,…) — 'max' undefined; RR also rejects (#89)
    "MODEL0404023805": "undefined_symbol",  # min(max,…) — 'max' undefined; RR also rejects (#89)
    # #89: 2 models reference simulation time as a bare <ci>time</ci> (no time
    # csymbol, no declared `time` symbol). RoadRunner 2.9.2 also rejects
    # ("symbol 'time' is not physically stored"). Lenient bare-<ci>time</ci>
    # handling is tracked separately (real-world-loader robustness, gated).
    "MODEL1405070000": "undefined_symbol",  # Dolan2014 — bare <ci>time</ci>; RR also rejects (#89)
    "MODEL2402030002": "undefined_symbol",  # bare <ci>time</ci> in event trigger; RR also rejects (#89)
}

FIELDS = [
    "model_id",
    "verdict",
    "reason",
    "n_species",
    "n_reactions",
    "n_rate_rules",
    "n_events",
    "tags",
    "sha256",
]


def sha256(path: Path) -> str:
    """SHA-256 of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(path: Path) -> dict:
    """Return the manifest row for one SBML file."""
    row = dict.fromkeys(FIELDS)
    row["model_id"] = path.stem
    row["verdict"] = "keep"
    row["reason"] = ""
    row["n_species"] = row["n_reactions"] = row["n_rate_rules"] = row["n_events"] = 0
    row["tags"] = ""
    row["sha256"] = sha256(path)

    model = libsbml.readSBML(str(path)).getModel()
    if model is None:
        row["verdict"], row["reason"] = "drop", "no_model"
        return row

    n_rxn = model.getNumReactions()
    rules = [model.getRule(i) for i in range(model.getNumRules())]
    n_rate = sum(1 for r in rules if r.isRate())
    row["n_species"] = model.getNumSpecies()
    row["n_reactions"] = n_rxn
    row["n_rate_rules"] = n_rate
    row["n_events"] = model.getNumEvents()

    # --- drop criteria: un-simulability ---
    if any(not model.getReaction(i).isSetKineticLaw() for i in range(n_rxn)):
        row["verdict"], row["reason"] = "drop", "missing_kinetic_law"
    elif n_rxn == 0 and n_rate == 0:
        row["verdict"], row["reason"] = "drop", "zero_dynamics"

    # --- feature tags (never a drop reason) ---
    tags = []
    if model.getNumEvents() > 0:
        tags.append("events")
        if any(model.getEvent(i).isSetDelay() for i in range(model.getNumEvents())):
            tags.append("event_delay")
    if any(r.isAlgebraic() for r in rules):
        tags.append("algebraic_rule")
    if model.getNumFunctionDefinitions() > 0:
        tags.append("function_def")
    row["tags"] = ";".join(tags)

    # Curated manual exclusions override any auto-verdict (libSBML can't detect
    # these; both engines reject them). Keep the computed counts/tags for the record.
    if row["model_id"] in MANUAL_DROP:
        row["verdict"], row["reason"] = "drop", MANUAL_DROP[row["model_id"]]

    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage-1 BioModels SBML filter")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="directory of SBML files")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="manifest CSV path")
    args = ap.parse_args()

    files = sorted(args.data.glob("*.xml"))
    if not files:
        print(f"no SBML files found in {args.data}", file=sys.stderr)
        return 1

    rows = [classify(f) for f in files]

    with open(args.out, "w", newline="") as fh:
        fh.write(
            f"# stage-1 biomodels filter | libsbml {libsbml.getLibSBMLDottedVersion()}"
            f" | {len(rows)} models\n"
        )
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    kept = sum(1 for r in rows if r["verdict"] == "keep")
    reasons = Counter(r["reason"] for r in rows if r["verdict"] == "drop")
    print(f"{len(rows)} models -> {kept} keep, {len(rows) - kept} drop")
    for reason, n in sorted(reasons.items()):
        print(f"  drop[{reason}]: {n}")
    print(f"manifest -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
