#!/usr/bin/env python3
"""Filter fetched models to identify SSA benchmark candidates.

Uses libSBML to analyze SBML files directly. Applies heuristics
to classify models as suitable or unsuitable for stochastic
simulation benchmarking.

Adapted from ssys biomodels_batch/step2_filter.py.

Usage:
    python step2_filter.py

Output:
    results/candidates.csv    — Classified models with metadata
    data/sbml_candidates/     — Copies of qualifying SBML files
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import config
import pandas as pd
import utils
from tqdm import tqdm

if TYPE_CHECKING:
    import libsbml as _libsbml_types

logger = logging.getLogger(__name__)


def load_sbml(model_id: str) -> _libsbml_types.Model | None:
    """Load SBML model using libSBML."""
    try:
        import libsbml
    except ImportError as err:
        raise ImportError(
            "python-libsbml is required. Install with: pip install python-libsbml"
        ) from err

    sbml_path = Path(config.SBML_DOWNLOADS_DIR) / f"{model_id}.xml"
    if not sbml_path.exists():
        return None

    try:
        doc = libsbml.readSBML(str(sbml_path))

        if doc.getNumErrors() > 0:
            for i in range(doc.getNumErrors()):
                err = doc.getError(i)
                if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                    logger.debug(f"SBML error in {model_id}: {err.getMessage()}")

        return doc.getModel()
    except Exception as e:
        logger.error(f"Failed to read {model_id}: {e}")
        return None


def has_sbml_l3_packages(sbml_path: str) -> bool:
    """Check if SBML file uses L3 packages."""
    try:
        with open(sbml_path) as f:
            header = f.read(2000)

        l3_packages = [
            "layout:",
            "layout_L2",
            "fbc:",
            "comp:",
            "qual:",
            "multi:",
            "render:",
            "groups:",
            "distrib:",
        ]
        return any(pkg in header for pkg in l3_packages)
    except Exception:
        return False


def detect_sbml_features(model, sbml_path: str = None) -> dict:
    """Detect model features using libSBML."""
    import libsbml

    features = {
        "events": False,
        "delays": False,
        "algebraic_rules": False,
        "piecewise": False,
        "piecewise_heavy": False,
        "time_dependent": False,
        "sin_cos": False,
        "unsupported_trig": False,
        "exp": False,
        "log": False,
        "negative_species": False,
        "non_integer_concentrations": False,
        "not_substance_units": False,
        "sbml_l3_packages": False,
    }

    if model is None:
        return features

    features["events"] = model.getNumEvents() > 0

    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule.getTypeCode() == libsbml.SBML_ALGEBRAIC_RULE:
            features["algebraic_rules"] = True

    def check_formula(formula_str: str):
        """Check a formula string for special features."""
        if not formula_str:
            return

        fl = formula_str.lower()

        if "delay(" in fl:
            features["delays"] = True
        if "piecewise(" in fl:
            features["piecewise"] = True
            if fl.count("piecewise(") > 2:
                features["piecewise_heavy"] = True
        if "time" in fl or " t " in fl:
            features["time_dependent"] = True
        if "sin(" in fl or "cos(" in fl:
            features["sin_cos"] = True
        if "tan(" in fl or "tanh(" in fl:
            features["unsupported_trig"] = True
        if "exp(" in fl:
            features["exp"] = True
        if "log(" in fl or "ln(" in fl:
            features["log"] = True

    # Check all reaction kinetic laws
    for i in range(model.getNumReactions()):
        rxn = model.getReaction(i)
        kl = rxn.getKineticLaw()
        if kl and kl.getMath():
            formula_str = libsbml.formulaToString(kl.getMath())
            check_formula(formula_str)

    # Check all rules
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule.getMath():
            formula_str = libsbml.formulaToString(rule.getMath())
            check_formula(formula_str)

    # Check species initial values and SSA suitability
    # For SSA, species must be in substance units
    # (particle counts), not concentrations.
    has_conc_species = False
    has_non_hosu = False

    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        hosu = sp.getHasOnlySubstanceUnits()

        if not hosu:
            has_non_hosu = True

        if sp.isSetInitialAmount():
            val = sp.getInitialAmount()
            if val < 0:
                features["negative_species"] = True
            if val != math.floor(val):
                features["non_integer_concentrations"] = True
        elif sp.isSetInitialConcentration():
            has_conc_species = True
            val = sp.getInitialConcentration()
            if val < 0:
                features["negative_species"] = True
            if val != math.floor(val):
                features["non_integer_concentrations"] = True

    # Model is not SSA-ready if species use
    # concentrations (not substance/particle counts)
    if has_conc_species or has_non_hosu:
        features["not_substance_units"] = True

    # Check for L3 packages
    if sbml_path:
        features["sbml_l3_packages"] = has_sbml_l3_packages(sbml_path)

    return features


def classify_model(model_id: str, model, sbml_path: str = None) -> dict:
    """Classify a model for SSA benchmark suitability."""
    if model is None:
        return {
            "model_id": model_id,
            "n_species": 0,
            "n_reactions": 0,
            "n_parameters": 0,
            "can_attempt": False,
            "candidate": False,
            "blockers": "parse_error",
            "warnings": "",
            "has_events": False,
            "has_delays": False,
            "has_piecewise": False,
            "has_sin_cos": False,
            "has_unsupported_trig": False,
            "has_exp": False,
            "has_log": False,
            "has_non_integer_conc": False,
        }

    features = detect_sbml_features(model, sbml_path)

    # Count floating species only
    n_species = 0
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        if not sp.getBoundaryCondition():
            n_species += 1

    n_reactions = model.getNumReactions()
    n_params = model.getNumParameters()

    # Identify blockers
    blockers = []
    for feature in config.BLOCKING_FEATURES:
        if features.get(feature, False):
            blockers.append(feature)

    if n_species == 0:
        blockers.append("no_dynamics")

    # Identify warnings
    warnings = []
    for feature in config.WARNING_FEATURES:
        if features.get(feature, False):
            warnings.append(feature)

    can_attempt = len(blockers) == 0 and n_species > 0

    candidate = (
        can_attempt
        and n_species <= config.MAX_SPECIES
        and n_reactions <= config.MAX_REACTIONS
        and n_params <= config.MAX_PARAMETERS
    )

    return {
        "model_id": model_id,
        "n_species": n_species,
        "n_reactions": n_reactions,
        "n_parameters": n_params,
        "can_attempt": can_attempt,
        "candidate": candidate,
        "blockers": ",".join(blockers) if blockers else "",
        "warnings": ",".join(warnings) if warnings else "",
        "has_events": features.get("events", False),
        "has_delays": features.get("delays", False),
        "has_piecewise": features.get("piecewise", False),
        "has_sin_cos": features.get("sin_cos", False),
        "has_unsupported_trig": features.get("unsupported_trig", False),
        "has_exp": features.get("exp", False),
        "has_log": features.get("log", False),
        "has_non_integer_conc": features.get("non_integer_concentrations", False),
    }


def filter_all_models() -> pd.DataFrame:
    """Filter all fetched models using libSBML."""
    sbml_dir = Path(config.SBML_DOWNLOADS_DIR)
    if not sbml_dir.exists():
        logger.error(f"SBML directory not found: {sbml_dir}")
        return pd.DataFrame()

    sbml_files = list(sbml_dir.glob("*.xml"))
    model_ids = [f.stem for f in sbml_files]

    if not model_ids:
        logger.error("No models found to filter")
        return pd.DataFrame()

    logger.info(f"Filtering {len(model_ids)} models...")

    results = []
    for model_id in tqdm(model_ids, desc="Classifying"):
        sbml_path = str(sbml_dir / f"{model_id}.xml")
        model = load_sbml(model_id)
        classification = classify_model(model_id, model, sbml_path)
        results.append(classification)

    return pd.DataFrame(results)


def copy_candidates(df: pd.DataFrame) -> int:
    """Copy qualifying SBML files to candidates dir."""
    candidates = df[df["candidate"]]["model_id"].tolist()

    if not candidates:
        logger.warning("No candidate models to copy")
        return 0

    candidates_dir = Path(config.SBML_CANDIDATES_DIR)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for model_id in candidates:
        src = Path(config.SBML_DOWNLOADS_DIR) / f"{model_id}.xml"
        dst = candidates_dir / f"{model_id}.xml"

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1

    logger.info(f"Copied {copied} candidate SBML files to {candidates_dir}")
    return copied


def generate_summary(df: pd.DataFrame) -> str:
    """Generate summary of filtering results."""
    total = len(df)
    if total == 0:
        return "No models processed."

    all_blockers = []
    for blockers_str in df["blockers"]:
        if blockers_str:
            all_blockers.extend(blockers_str.split(","))
    blocker_counts = Counter(all_blockers)

    excluded = int(total - df["can_attempt"].sum())
    eligible = int(df["can_attempt"].sum())
    candidates = int(df["candidate"].sum())
    excluded_by_size = eligible - candidates

    eligible_df = df[df["can_attempt"]]

    has_non_int = int(df["has_non_integer_conc"].sum())

    lines = []
    lines.append("")
    lines.append("BioModels SSA Benchmark Filtering Summary")
    lines.append("=" * 42)
    lines.append("")
    lines.append(f"INPUT:  {total} models analyzed")
    lines.append("")

    lines.append("FILTERING FUNNEL:")
    lines.append(f"  {total:4d}  Total models")
    lines.append(f"  -{excluded:4d}  Excluded (blockers)")
    lines.append("  -----")
    lines.append(f"  {eligible:4d}  Eligible (no blockers)")
    lines.append(f"  -{excluded_by_size:4d}  Excluded (too large)")
    lines.append("  -----")
    lines.append(f"  {candidates:4d}  CANDIDATES for conversion")
    lines.append("")

    lines.append(f"BLOCKER DETAILS ({excluded} models excluded):")
    blocker_labels = {
        "no_dynamics": "No ODEs (0 floating species)",
        "events": "Contains discrete events",
        "parse_error": "SBML parse errors",
        "delays": "Contains delay functions",
        "algebraic_rules": "Algebraic constraints",
        "sbml_l3_packages": "Uses SBML L3 packages",
        "unsupported_trig": "Unsupported trig (tan/tanh)",
        "not_substance_units": "Not in particle counts",
    }

    for blocker, label in blocker_labels.items():
        if blocker in blocker_counts:
            count = blocker_counts[blocker]
            lines.append(f"    {count:4d}  {label}")

    lines.append("")

    if len(eligible_df) > 0:
        lines.append("COMPLEXITY STATISTICS (eligible models):")
        lines.append(
            f"    Species:    "
            f"median={eligible_df['n_species'].median():.0f}, "
            f"range=["
            f"{eligible_df['n_species'].min()}, "
            f"{eligible_df['n_species'].max()}]"
        )
        lines.append(
            f"    Reactions:  "
            f"median={eligible_df['n_reactions'].median():.0f}, "
            f"range=["
            f"{eligible_df['n_reactions'].min()}, "
            f"{eligible_df['n_reactions'].max()}]"
        )
        lines.append("")

    lines.append("SSA-RELEVANT NOTES:")
    lines.append(
        f"    {has_non_int:4d}  Non-integer initial concentrations (need rounding for SSA)"
    )
    lines.append("")

    return "\n".join(lines)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Filter models for SSA benchmarking")
    parser.add_argument(
        "--output",
        type=str,
        default=config.CANDIDATES_CSV,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy candidate SBML files",
    )

    args = parser.parse_args()

    utils.setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    logger.info("=" * 60)
    logger.info("BioModels Filter Script (SSA Benchmark)")
    logger.info("=" * 60)

    df = filter_all_models()

    if df.empty:
        logger.error("No models to filter. Run step1_fetch.py first.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    if not args.no_copy:
        copy_candidates(df)

    summary = generate_summary(df)
    print(summary)

    summary_path = output_path.parent / "filter_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info(f"Saved summary to {summary_path}")

    logger.info("\nFiltering complete!")


if __name__ == "__main__":
    main()
